import csv,gzip,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT
DATA=ROOT/'data/processed/Top5_Exact_Border_Bridge_Data_25mi.csv.gz'
JSON=ROOT/'results/generated/Top5_Exact_RD_Results.json'
CONTROLS=['age_2025','log_adt','structure_length_m','max_span_m','deck_width_m']
CATS=['structure_kind_code','structure_type_code','functional_class_code','design_load_code','highway_system_code']
OUTCOMES=[('lowest_rating','Lowest component rating'),('deck_rating','Deck rating'),('superstructure_rating','Superstructure rating'),('substructure_rating','Substructure rating')]
NUMS=['distance_miles_exact','signed_distance_miles_exact','deck_rating','superstructure_rating','substructure_rating','lowest_rating']+CONTROLS

def f(v):
 try:return float(v)
 except:return math.nan

def fin(x):return isinstance(x,(int,float,np.floating)) and math.isfinite(float(x))
def shared(rows,field,min_each=5):
 ca=Counter((r.get(field) or 'MISSING') for r in rows if r['side_b']==0); cb=Counter((r.get(field) or 'MISSING') for r in rows if r['side_b']==1)
 return {k for k in set(ca)|set(cb) if ca[k]>=min_each and cb[k]>=min_each}
def onehot(vals,keep,prefix):
 vals=[v or 'MISSING' for v in vals]; vals=[v if v in keep else 'OTHER' for v in vals]
 levs=sorted(set(vals)); cols=[]; names=[]
 for lev in levs[1:]: cols.append([1.0 if v==lev else 0.0 for v in vals]); names.append(prefix+'='+lev)
 return (np.column_stack(cols) if cols else np.zeros((len(vals),0))),names

def propensity_X(rows):
 nums=np.array([[r[c] for c in CONTROLS] for r in rows],float)
 nums=StandardScaler().fit_transform(nums)
 cols=[nums]; names=list(CONTROLS)
 for c in CATS:
  x,n=onehot([r.get(c,'') for r in rows],shared(rows,c,max(5,int(.01*len(rows)))),c); cols.append(x); names+=n
 x,n=onehot([r.get('nearest_border_grid_25km','') for r in rows],shared(rows,'nearest_border_grid_25km',2),'segment'); cols.append(x); names+=n
 return np.column_stack(cols),names

def outcome_X(rows):
 side=np.array([r['side_b'] for r in rows],float); dist=np.array([r['distance_miles_exact'] for r in rows],float)
 cols=[np.ones(len(rows)),side,dist,side*dist]; names=['Intercept','side_b','distance','interaction']
 nums=np.array([[r[c] for c in CONTROLS] for r in rows],float); nums=StandardScaler().fit_transform(nums)
 cols += [nums[:,j] for j in range(nums.shape[1])]; names+=CONTROLS
 for c in CATS:
  x,n=onehot([r.get(c,'') for r in rows],shared(rows,c,max(5,int(.01*len(rows)))),c)
  cols += [x[:,j] for j in range(x.shape[1])]; names+=n
 x,n=onehot([r.get('nearest_border_grid_25km','') for r in rows],shared(rows,'nearest_border_grid_25km',2),'segment')
 cols += [x[:,j] for j in range(x.shape[1])]; names+=n
 X=np.column_stack(cols)
 keep=[0]+[j for j in range(1,X.shape[1]) if np.std(X[:,j])>1e-12]
 return X[:,keep],[names[j] for j in keep]

def ess(w):
 w=np.asarray(w); return float(w.sum()**2/(w@w)) if (w@w)>0 else 0

rows_by=defaultdict(list)
with gzip.open(DATA,'rt',encoding='utf-8',newline='') as fobj:
 for r in csv.DictReader(fobj):
  for c in NUMS:r[c]=f(r.get(c,''))
  r['side_b']=1 if r['state']==r['state_b'] else 0
  rows_by[r['border_pair']].append(r)

out=[]
for pair,allrows in rows_by.items():
 for bw in (5,10,25):
  base=[r for r in allrows if r['distance_miles_exact']<=bw and all(fin(r[c]) for c in CONTROLS)]
  if len(base)<100:continue
  PX,_=propensity_X(base); yside=np.array([r['side_b'] for r in base])
  logit=LogisticRegression(max_iter=3000,C=1.0,solver='lbfgs')
  logit.fit(PX,yside); ps=np.clip(logit.predict_proba(PX)[:,1],.01,.99)
  auc=float(roc_auc_score(yside,ps)); weights=np.where(yside==1,1-ps,ps)
  for outcome,label in OUTCOMES:
   idx=[i for i,r in enumerate(base) if fin(r[outcome])]
   used=[base[i] for i in idx]; w=weights[idx]; y=np.array([r[outcome] for r in used],float)
   X,names=outcome_X(used); model=sm.WLS(y,X,weights=w)
   groups=np.array([r['nearest_border_grid_25km'] for r in used]); nc=len(set(groups))
   if nc>=10:res=model.fit(cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True});cov='Clustered by 25-km border segment'
   else:res=model.fit(cov_type='HC3');cov='HC3 robust'
   j=names.index('side_b'); ci=res.conf_int()[j]
   out.append({'border_pair':pair,'bandwidth':bw,'outcome':outcome,'outcome_label':label,'n':len(used),'auc':auc,
    'ess_a':ess(w[yside[idx]==0]),'ess_b':ess(w[yside[idx]==1]),'estimate_a_minus_b':-float(res.params[j]),'se':float(res.bse[j]),
    'ci_low':-float(ci[1]),'ci_high':-float(ci[0]),'p_value':float(res.pvalues[j]),'clusters':nc,'covariance':cov})

j=json.load(open(JSON));j['overlap_results']=out
json.dump(j,open(JSON,'w'),indent=2,allow_nan=True)
fields=list(out[0]);
with open(ROOT/'results/generated/top5_exact_outputs/overlap_results.csv','w',newline='',encoding='utf-8') as fobj:
 w=csv.DictWriter(fobj,fieldnames=fields);w.writeheader();w.writerows(out)
print('10-mile lowest')
for r in out:
 if r['bandwidth']==10 and r['outcome']=='lowest_rating':print(r)
