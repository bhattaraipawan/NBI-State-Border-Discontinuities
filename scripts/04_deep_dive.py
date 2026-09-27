import csv, gzip, json, math, os, zipfile
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import ttest_ind, ttest_rel, wilcoxon, chi2_contingency, t as student_t
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
BRIDGE_FILE = ROOT / 'data/processed/Top5_Exact_Border_Bridge_Data_25mi.csv.gz'
NBI_FILE = ROOT / 'data/raw/2025HwyBridgesDelimitedAllStates.txt'
OUT_JSON = ROOT / 'results/generated/KS_MO_Confounder_Matching_Results.json'
OUT_GZ = ROOT / 'data/processed/KS_MO_Enriched_Bridge_Data_25mi.csv.gz'
MATCH_CSV = ROOT / 'results/generated/KS_MO_Strict_Matched_Pairs.csv'
PAIR_RESULTS_CSV = ROOT / 'results/generated/KS_MO_Model_Results.csv'

PAIRS = ['KS-MO']
STATE_FIPS = {'KS':'20','MO':'29'}
CONT_BASE = ['age_2025','log_adt','structure_length_m','max_span_m','deck_width_m']
CONT_EXPANDED = ['age_2025','log_adt','truck_pct','log_aadtt','structure_length_m','max_span_m','deck_width_m','lanes_on']
CAT_BASE = ['structure_kind_code','structure_type_code','functional_class_code','design_load_code','highway_system_code']
CAT_EXPANDED = CAT_BASE + ['national_network','deck_protection_code']
OUTCOMES = [('lowest_rating','Lowest rating'),('deck_rating','Deck rating'),('superstructure_rating','Superstructure rating'),('substructure_rating','Substructure rating')]
NUMERIC_EXISTING = ['latitude','longitude','x_5070','y_5070','distance_miles_exact','signed_distance_miles_exact','deck_rating','superstructure_rating','substructure_rating','lowest_rating','year_built','age_2025','adt','log_adt','structure_length_m','max_span_m','deck_width_m','inspection_frequency_months','year_reconstructed']

EXTRA_FIELDS = [
    'PERCENT_ADT_TRUCK_109','TRAFFIC_LANES_ON_028A','OPEN_CLOSED_POSTED_041',
    'OPERATING_RATING_064','INVENTORY_RATING_066','POSTING_EVAL_070',
    'WORK_PROPOSED_075A','WORK_DONE_BY_075B','BRIDGE_IMP_COST_094','ROADWAY_IMP_COST_095',
    'TOTAL_IMP_COST_096','YEAR_OF_IMP_097','STRAHNET_HIGHWAY_100','NATIONAL_NETWORK_110',
    'DECK_STRUCTURE_TYPE_107','SURFACE_TYPE_108A','MEMBRANE_TYPE_108B','DECK_PROTECTION_108C',
    'FUTURE_ADT_114','YEAR_OF_FUTURE_ADT_115','DATE_OF_INSPECT_090'
]

def fnum(v):
    if v is None: return math.nan
    s=str(v).strip()
    if s=='': return math.nan
    try: return float(s)
    except: return math.nan

def finite(v):
    try: return math.isfinite(float(v))
    except: return False

def clean_code(v):
    if v is None: return ''
    return str(v).strip()

def pooled_smd(a,b):
    a=np.asarray([float(x) for x in a if finite(x)],float); b=np.asarray([float(x) for x in b if finite(x)],float)
    if len(a)<2 or len(b)<2: return math.nan
    sp=math.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    if sp<=1e-12: return 0.0
    return float((np.mean(a)-np.mean(b))/sp)

def weighted_mean(x,w):
    x=np.asarray(x,float); w=np.asarray(w,float)
    ok=np.isfinite(x)&np.isfinite(w)&(w>=0)
    if not ok.any() or w[ok].sum()==0: return math.nan
    return float(np.sum(x[ok]*w[ok])/np.sum(w[ok]))

def weighted_var(x,w):
    x=np.asarray(x,float); w=np.asarray(w,float)
    ok=np.isfinite(x)&np.isfinite(w)&(w>=0)
    x=x[ok];w=w[ok]
    if len(x)<2 or w.sum()==0: return math.nan
    m=np.sum(w*x)/w.sum()
    return float(np.sum(w*(x-m)**2)/w.sum())

def weighted_smd(a,wa,b,wb):
    ma=weighted_mean(a,wa); mb=weighted_mean(b,wb)
    va=weighted_var(a,wa); vb=weighted_var(b,wb)
    if not all(finite(x) for x in [ma,mb,va,vb]): return math.nan
    sp=math.sqrt((va+vb)/2)
    return (ma-mb)/sp if sp>1e-12 else 0.0

def cramers_v(vals,sides,weights=None):
    vals=[str(v or 'MISSING') for v in vals]; sides=list(sides)
    lev=sorted(set(vals)); sl=sorted(set(sides))
    if len(lev)<2 or len(sl)<2: return 0.0
    li={v:i for i,v in enumerate(lev)}; si={v:i for i,v in enumerate(sl)}
    tab=np.zeros((len(sl),len(lev)),float)
    if weights is None: weights=np.ones(len(vals))
    for v,s,w in zip(vals,sides,weights): tab[si[s],li[v]] += float(w)
    tab=tab[:,tab.sum(axis=0)>1e-12]
    if tab.shape[1]<2: return 0.0
    # Pearson chi-square calculated for weighted table as a descriptive association measure.
    n=tab.sum(); row=tab.sum(axis=1,keepdims=True); col=tab.sum(axis=0,keepdims=True)
    exp=row@col/n
    mask=exp>0
    chi=float(np.sum(((tab-exp)**2/exp)[mask]))
    denom=min(tab.shape[0]-1,tab.shape[1]-1)
    return math.sqrt((chi/n)/denom) if n>0 and denom>0 else 0.0

def shared_categories(rows, field, min_each=5):
    ca=Counter(clean_code(r.get(field)) or 'MISSING' for r in rows if r['side_b']==0)
    cb=Counter(clean_code(r.get(field)) or 'MISSING' for r in rows if r['side_b']==1)
    return {k for k in set(ca)|set(cb) if ca[k]>=min_each and cb[k]>=min_each}

def one_hot(values, keep=None, prefix='cat'):
    vals=[clean_code(v) or 'MISSING' for v in values]
    if keep is not None: vals=[v if v in keep else 'OTHER' for v in vals]
    levels=sorted(set(vals))
    if len(levels)<=1: return np.zeros((len(vals),0)),[]
    cols=[];names=[]
    for lev in levels[1:]:
        cols.append(np.array([1.0 if v==lev else 0.0 for v in vals]))
        names.append(f'{prefix}={lev}')
    return np.column_stack(cols),names

def make_design(rows, cont_controls, cat_controls, cutoff=0.0):
    n=len(rows)
    signed=np.array([r['signed_distance_miles_exact']-cutoff for r in rows],float)
    side=(signed>=0).astype(float)
    dist=np.abs(signed)
    cols=[np.ones(n),side,dist,side*dist]; names=['Intercept','side_b','distance_abs','side_x_distance']
    keep_seg=shared_categories(rows,'nearest_border_grid_25km',2)
    sx,sn=one_hot([r.get('nearest_border_grid_25km','') for r in rows],keep_seg,'segment')
    for j in range(sx.shape[1]): cols.append(sx[:,j])
    names += sn
    for field in cont_controls:
        x=np.array([r[field] for r in rows],float)
        mu=np.nanmean(x); sd=np.nanstd(x,ddof=1)
        x=(x-mu)/sd if finite(sd) and sd>1e-12 else np.zeros(n)
        cols.append(x); names.append(field+'_z')
    for field in cat_controls:
        keep=shared_categories(rows,field,max(5,int(.01*n)))
        cx,cn=one_hot([r.get(field,'') for r in rows],keep,field)
        for j in range(cx.shape[1]): cols.append(cx[:,j])
        names += cn
    X=np.column_stack(cols)
    keep=[0]+[j for j in range(1,X.shape[1]) if np.nanstd(X[:,j])>1e-12]
    return X[:,keep],[names[j] for j in keep]

def fit_rd(rows,outcome='lowest_rating',cont_controls=CONT_BASE,cat_controls=CAT_BASE,cutoff=0.0,weight_field=None,label=''):
    used=[];weights=[]
    required=[outcome]+list(cont_controls)
    for r in rows:
        if any(not finite(r.get(c)) for c in required): continue
        if weight_field is not None and (not finite(r.get(weight_field)) or r[weight_field]<=0): continue
        used.append(r); weights.append(float(r.get(weight_field,1.0)))
    if len(used)<80: return {'label':label,'n':len(used),'error':'insufficient sample'}
    X,names=make_design(used,cont_controls,cat_controls,cutoff)
    y=np.array([r[outcome] for r in used],float)
    groups=np.array([r.get('nearest_border_grid_25km','MISSING') for r in used])
    nc=len(set(groups))
    model=sm.WLS(y,X,weights=np.asarray(weights)) if weight_field else sm.OLS(y,X)
    try:
        if nc>=10:
            res=model.fit(cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True}); cov='cluster_25km'
        else:
            res=model.fit(cov_type='HC3'); cov='HC3'
        j=names.index('side_b'); ci=res.conf_int()[j]
        return {'label':label,'n':len(used),'clusters':nc,'k':X.shape[1],'covariance':cov,
                'estimate_a_minus_b':-float(res.params[j]),'se':float(res.bse[j]),
                'ci_low':-float(ci[1]),'ci_high':-float(ci[0]),'p_value':float(res.pvalues[j]),
                'r_squared':float(res.rsquared)}
    except Exception as e:
        return {'label':label,'n':len(used),'error':str(e)}

def propensity_overlap(rows,outcome='lowest_rating',cont_controls=CONT_EXPANDED,cat_controls=CAT_EXPANDED,label=''):
    used=[]
    for r in rows:
        if not finite(r.get(outcome)): continue
        if any(not finite(r.get(c)) for c in cont_controls): continue
        used.append(r)
    if len(used)<100: return {'label':label,'n':len(used),'error':'insufficient sample'},[]
    nums=np.array([[r[c] for c in cont_controls] for r in used],float)
    nums=StandardScaler().fit_transform(nums)
    blocks=[nums]
    for c in cat_controls:
        x,_=one_hot([r.get(c,'') for r in used],shared_categories(used,c,max(5,int(.01*len(used)))),c)
        if x.shape[1]: blocks.append(x)
    x,_=one_hot([r.get('nearest_border_grid_25km','') for r in used],shared_categories(used,'nearest_border_grid_25km',2),'segment')
    if x.shape[1]: blocks.append(x)
    PX=np.column_stack(blocks)
    side=np.array([r['side_b'] for r in used],int)
    logit=LogisticRegression(max_iter=4000,C=1.0,solver='lbfgs')
    logit.fit(PX,side)
    ps=np.clip(logit.predict_proba(PX)[:,1],.01,.99)
    auc=float(roc_auc_score(side,ps))
    w=np.where(side==1,1-ps,ps)
    for r,wi,pi in zip(used,w,ps):
        r['_overlap_weight']=float(wi); r['_propensity']=float(pi)
    res=fit_rd(used,outcome,cont_controls,cat_controls,weight_field='_overlap_weight',label=label)
    wa=w[side==0]; wb=w[side==1]
    res.update({'auc':auc,'ess_a':float(wa.sum()**2/(wa@wa)),'ess_b':float(wb.sum()**2/(wb@wb))})
    return res,used

def fit_continuity(rows,field,label=''):
    used=[r for r in rows if finite(r.get(field))]
    return fit_rd(used,outcome=field,cont_controls=[],cat_controls=[],label=label)

def common_age_range(rows):
    a=np.array([r['age_2025'] for r in rows if r['side_b']==0 and finite(r['age_2025'])]); b=np.array([r['age_2025'] for r in rows if r['side_b']==1 and finite(r['age_2025'])])
    lo=max(np.quantile(a,.05),np.quantile(b,.05)); hi=min(np.quantile(a,.95),np.quantile(b,.95))
    return float(lo),float(hi)

def broad_structure_type(v):
    s=clean_code(v)
    try:n=int(float(s))
    except:return s or 'M'
    # Legacy NBI Item 43B design codes (1995 Coding Guide).
    if n in (1,2,3,4): return 'slab/girder/tee'
    if n in (5,6,7): return 'box/frame'
    if n == 8: return 'orthotropic'
    if n in (9,10): return 'truss'
    if n in (11,12): return 'arch'
    if n in (13,14): return 'cable'
    if n in (15,16,17): return 'movable'
    if n == 18: return 'tunnel'
    if n == 19: return 'culvert'
    if n == 20: return 'mixed'
    if n == 21: return 'segmental_box'
    if n == 22: return 'channel_beam'
    return 'other'

def functional_group(v):
    s=clean_code(v)
    try:n=int(float(s))
    except:return s or 'M'
    # NBI functional classes: group principal/major, minor, local.
    if n in (1,2,11,12,14): return 'principal_interstate'
    if n in (6,7,8,9,16,17,19): return 'minor_collector_local'
    return 'other_arterial'

def greedy_match(rows,state_agency_only=False,age_caliper=5.0,distance_caliper=5.0,log_adt_caliper=None,log_aadtt_caliper=None,deck_width_caliper=None,truck_pct_caliper=None):
    pool=[r for r in rows if all(finite(r.get(c)) for c in ['age_2025','log_aadtt','structure_length_m','max_span_m','deck_width_m','lowest_rating'])]
    if state_agency_only:
        pool=[r for r in pool if r['state_agency_both']==1]
    a=[r for r in pool if r['side_b']==0]; b=[r for r in pool if r['side_b']==1]
    if not a or not b:return []
    # Exact strata on segment, material, broad design, roadway class, highway system.
    def strata(r):
        return (r['nearest_border_grid_25km'], clean_code(r['structure_kind_code']), broad_structure_type(r['structure_type_code']), functional_group(r['functional_class_code']), clean_code(r['highway_system_code']))
    ga=defaultdict(list);gb=defaultdict(list)
    for r in a:ga[strata(r)].append(r)
    for r in b:gb[strata(r)].append(r)
    pairs=[]; pid=0
    numeric=['age_2025','log_aadtt','structure_length_m','max_span_m','deck_width_m','distance_miles_exact']
    # pooled scaling
    arr=np.array([[r[c] for c in numeric] for r in pool],float)
    sd=np.nanstd(arr,axis=0,ddof=1);sd=np.where(sd>1e-9,sd,1.0)
    for key in sorted(set(ga)&set(gb)):
        left=ga[key]; right=gb[key]
        # iterate smaller side to reduce loss; always output a,b order
        candidates=[]
        for i,ra in enumerate(left):
            for j,rb in enumerate(right):
                if abs(ra['age_2025']-rb['age_2025'])>age_caliper:continue
                if abs(ra['distance_miles_exact']-rb['distance_miles_exact'])>distance_caliper:continue
                if log_adt_caliper is not None and abs(ra['log_adt']-rb['log_adt'])>log_adt_caliper:continue
                if log_aadtt_caliper is not None and abs(ra['log_aadtt']-rb['log_aadtt'])>log_aadtt_caliper:continue
                if deck_width_caliper is not None and abs(ra['deck_width_m']-rb['deck_width_m'])>deck_width_caliper:continue
                if truck_pct_caliper is not None and abs(ra['truck_pct']-rb['truck_pct'])>truck_pct_caliper:continue
                va=np.array([ra[c] for c in numeric],float);vb=np.array([rb[c] for c in numeric],float)
                d=float(np.sqrt(np.sum(((va-vb)/sd)**2)))
                candidates.append((d,i,j))
        useda=set();usedb=set()
        for d,i,j in sorted(candidates):
            if i in useda or j in usedb:continue
            useda.add(i);usedb.add(j);pid+=1
            pairs.append({'pair_id':pid,'match_distance':d,'a':left[i],'b':right[j],'state_agency_only':state_agency_only})
    return pairs

def matched_stats(pairs,outcome):
    if len(pairs)<5:return {'n_pairs':len(pairs)}
    dif=np.array([p['a'][outcome]-p['b'][outcome] for p in pairs],float)
    mean=float(np.mean(dif));se=float(np.std(dif,ddof=1)/math.sqrt(len(dif)))
    t=ttest_rel([p['a'][outcome] for p in pairs],[p['b'][outcome] for p in pairs])
    try:w=wilcoxon(dif).pvalue
    except:w=math.nan
    df=len(dif)-1
    crit=float(student_t.ppf(0.975,df))
    return {'n_pairs':len(dif),'mean_a_minus_b':mean,'se':se,'df':df,'inference_reference':f't({df})',
            'ci_low':mean-crit*se,'ci_high':mean+crit*se,'p_paired_t':float(t.pvalue),
            'p_wilcoxon':float(w),'median':float(np.median(dif))}

def matched_balance(pairs,fields):
    out=[]
    for c in fields:
        a=[p['a'][c] for p in pairs];b=[p['b'][c] for p in pairs]
        out.append({'variable':c,'mean_a':float(np.mean(a)) if a else math.nan,'mean_b':float(np.mean(b)) if b else math.nan,'smd':pooled_smd(a,b)})
    return out

# 1) load bridge sample for two borders
rows=[]
with gzip.open(BRIDGE_FILE,'rt',encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['border_pair'] not in PAIRS: continue
        for c in NUMERIC_EXISTING:r[c]=fnum(r.get(c))
        r['structure_number']=str(r['structure_number'])
        r['side_b']=1 if r['state']==r['state_b'] else 0
        rows.append(r)

# 2) pull needed extra variables from national NBI
wanted={(STATE_FIPS[r['state']],r['structure_number'].strip()):r for r in rows}
found=0
with open(NBI_FILE,'r',encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f)
    for nr in reader:
        key=(clean_code(nr.get('STATE_CODE_001')),clean_code(nr.get('STRUCTURE_NUMBER_008')))
        if key not in wanted:continue
        r=wanted[key]
        for field in EXTRA_FIELDS:r[field]=nr.get(field,'')
        found+=1

# 3) derive exposure/admin variables
for r in rows:
    # Item 109 may legitimately be blank (e.g., low-ADT roads); blank is missing, not 0% trucks.
    # Truck-expanded regressions therefore use complete cases for Item 109/AADTT rather than silently
    # recoding missing truck percentage to zero.
    r['truck_pct']=fnum(r.get('PERCENT_ADT_TRUCK_109'))
    r['truck_pct_observed']=1 if finite(r['truck_pct']) else 0
    r['aadtt']=r['adt']*r['truck_pct']/100.0 if finite(r['adt']) and finite(r['truck_pct']) else math.nan
    r['log_aadtt']=math.log1p(max(0,r['aadtt'])) if finite(r['aadtt']) else math.nan
    r['lanes_on']=fnum(r.get('TRAFFIC_LANES_ON_028A'))
    r['operating_rating']=fnum(r.get('OPERATING_RATING_064'))
    r['inventory_rating']=fnum(r.get('INVENTORY_RATING_066'))
    r['posting_eval']=fnum(r.get('POSTING_EVAL_070'))
    r['future_adt']=fnum(r.get('FUTURE_ADT_114'))
    r['national_network']=clean_code(r.get('NATIONAL_NETWORK_110')) or 'MISSING'
    r['strahnet']=clean_code(r.get('STRAHNET_HIGHWAY_100')) or 'MISSING'
    r['deck_protection_code']=clean_code(r.get('DECK_PROTECTION_108C')) or 'MISSING'
    r['surface_type_code']=clean_code(r.get('SURFACE_TYPE_108A')) or 'MISSING'
    r['open_status_code']=clean_code(r.get('OPEN_CLOSED_POSTED_041')) or 'MISSING'
    r['open_unrestricted']=1 if r['open_status_code']=='A' else 0
    r['state_owned']=1 if clean_code(r.get('owner_code'))=='01' else 0
    r['state_maintained']=1 if clean_code(r.get('maintenance_code'))=='01' else 0
    r['state_agency_both']=1 if r['state_owned'] and r['state_maintained'] else 0
    r['reconstructed']=1 if finite(r.get('year_reconstructed')) and r['year_reconstructed']>0 else 0
    r['year_of_improvement']=fnum(r.get('YEAR_OF_IMP_097'))
    r['recent_improvement']=1 if finite(r['year_of_improvement']) and r['year_of_improvement']>=2015 else 0
    r['inspection_frequency_months']=r.get('inspection_frequency_months',math.nan)

# output enriched data
out_fields=[
'border_pair','state_a','state_b','state','structure_number','latitude','longitude','distance_miles_exact','signed_distance_miles_exact','nearest_border_grid_25km','reported_state_geometry_match',
'deck_rating','superstructure_rating','substructure_rating','lowest_rating','year_built','age_2025','adt','log_adt','truck_pct','truck_pct_observed','aadtt','log_aadtt','lanes_on',
'structure_kind_code','structure_type_code','functional_class_code','design_load_code','owner_code','maintenance_code','highway_system_code','national_network','strahnet',
'structure_length_m','max_span_m','deck_width_m','inspection_frequency_months','year_reconstructed','reconstructed','year_of_improvement','recent_improvement',
'open_status_code','open_unrestricted','operating_rating','inventory_rating','posting_eval','deck_protection_code','surface_type_code','state_owned','state_maintained','state_agency_both'
]
with gzip.open(OUT_GZ,'wt',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out_fields);w.writeheader();w.writerows([{k:r.get(k,'') for k in out_fields} for r in rows])

results={'metadata':{'rows':len(rows),'national_rows_found':found,'pairs':PAIRS,'analysis_bandwidth':10,'expanded_continuous_controls':CONT_EXPANDED,'expanded_categorical_controls':CAT_EXPANDED,'truck_item_109_policy':'Blank Item 109 retained as missing; truck-expanded models and matching use complete cases for truck_pct/log_aadtt.'},'pair_results':{},'model_results':[],'balance':[],'matched_pairs':[],'climate_proxy':[]}
match_flat=[]
model_flat=[]

for pair in PAIRS:
    pr=[r for r in rows if r['border_pair']==pair and r['distance_miles_exact']<=10]
    a=[r for r in pr if r['side_b']==0];b=[r for r in pr if r['side_b']==1]
    pair_summary={'state_a':pr[0]['state_a'],'state_b':pr[0]['state_b'],'n_a':len(a),'n_b':len(b)}
    # Balance/diagnostic table
    cont_fields=['age_2025','adt','log_adt','truck_pct','aadtt','log_aadtt','lanes_on','structure_length_m','max_span_m','deck_width_m','inspection_frequency_months','operating_rating','inventory_rating','latitude','longitude']
    bal=[]
    for field in cont_fields:
        av=[r[field] for r in a if finite(r.get(field))];bv=[r[field] for r in b if finite(r.get(field))]
        p=float(ttest_ind(av,bv,equal_var=False).pvalue) if len(av)>1 and len(bv)>1 else math.nan
        bal.append({'border_pair':pair,'type':'continuous','variable':field,'n_a':len(av),'n_b':len(bv),'mean_a':float(np.mean(av)) if av else math.nan,'mean_b':float(np.mean(bv)) if bv else math.nan,'smd':pooled_smd(av,bv),'p_value':p})
    cat_fields=['structure_kind_code','structure_type_code','functional_class_code','design_load_code','owner_code','maintenance_code','highway_system_code','national_network','strahnet','deck_protection_code','open_status_code']
    for field in cat_fields:
        bal.append({'border_pair':pair,'type':'categorical','variable':field,'n_a':len(a),'n_b':len(b),'cramers_v':cramers_v([r.get(field) for r in pr],[r['state'] for r in pr])})
    results['balance'].extend(bal)

    # Main and sensitivity models
    specs=[]
    specs.append(('Original adjusted',pr,CONT_BASE,CAT_BASE))
    specs.append(('Truck-expanded adjusted',pr,CONT_EXPANDED,CAT_EXPANDED))
    lo,hi=common_age_range(pr)
    age_common=[r for r in pr if lo<=r['age_2025']<=hi]
    specs.append((f'Common-age support ({lo:.1f}-{hi:.1f} y)',age_common,CONT_EXPANDED,CAT_EXPANDED))
    state_agency=[r for r in pr if r['state_agency_both']==1]
    specs.append(('State-owned and state-maintained only',state_agency,CONT_EXPANDED,CAT_EXPANDED))
    open_only=[r for r in pr if r['open_unrestricted']==1]
    specs.append(('Open/unrestricted only',open_only,CONT_EXPANDED,CAT_EXPANDED))
    for label,sub,cont,cats in specs:
        for outcome,olabel in OUTCOMES:
            res=fit_rd(sub,outcome,cont,cats,label=label)
            row={'border_pair':pair,'model_family':'RD sensitivity','specification':label,'outcome':outcome,'outcome_label':olabel,**res}
            results['model_results'].append(row);model_flat.append(row)

    # Expanded overlap weighted
    for outcome,olabel in OUTCOMES:
        res,used=propensity_overlap(pr,outcome,CONT_EXPANDED,CAT_EXPANDED,label='Expanded overlap weighting')
        row={'border_pair':pair,'model_family':'Overlap weighting','specification':'Expanded overlap weighting','outcome':outcome,'outcome_label':olabel,**res}
        results['model_results'].append(row);model_flat.append(row)

    # Continuity tests for truck/admin/geographic proxies
    continuity_fields=['truck_pct','log_aadtt','lanes_on','inspection_frequency_months','operating_rating','inventory_rating','latitude','longitude']
    continuity=[]
    for field in continuity_fields:
        res=fit_continuity(pr,field,label='Continuity')
        continuity.append({'variable':field,**res})
    pair_summary['continuity']=continuity
    results['climate_proxy'].extend([{'border_pair':pair,**r} for r in continuity if r['variable'] in ('latitude','longitude')])

    # Broad and exposure-tight matching sensitivity analyses.
    matched_sets=[]
    match_specs=[
        ('Broad all-owner match',False,dict(age_caliper=5.0,distance_caliper=5.0)),
        ('Broad state-agency match',True,dict(age_caliper=5.0,distance_caliper=5.0)),
        ('Exposure-tight all-owner match',False,dict(age_caliper=5.0,distance_caliper=4.0,log_adt_caliper=.75,log_aadtt_caliper=.75,deck_width_caliper=3.0,truck_pct_caliper=3.0)),
        ('Exposure-tight state-agency match',True,dict(age_caliper=5.0,distance_caliper=4.0,log_adt_caliper=.75,log_aadtt_caliper=.75,deck_width_caliper=3.0,truck_pct_caliper=3.0)),
    ]
    for label,agency,kwargs in match_specs:
        mp=greedy_match(pr,state_agency_only=agency,**kwargs)
        mstats=[]
        for outcome,olabel in OUTCOMES:
            st=matched_stats(mp,outcome);mstats.append({'outcome':outcome,'outcome_label':olabel,**st})
        mb=matched_balance(mp,['age_2025','log_adt','truck_pct','log_aadtt','structure_length_m','max_span_m','deck_width_m','distance_miles_exact'])
        matched_sets.append({'label':label,'n_pairs':len(mp),'stats':mstats,'balance':mb})
        for p in mp:
            ra=p['a'];rb=p['b']
            flat={'border_pair':pair,'match_type':label,'pair_id':p['pair_id'],'match_distance':p['match_distance'],
                  'state_a':ra['state'],'structure_a':ra['structure_number'].strip(),'state_b':rb['state'],'structure_b':rb['structure_number'].strip()}
            for c in ['lowest_rating','deck_rating','superstructure_rating','substructure_rating','age_2025','adt','truck_pct','aadtt','structure_length_m','max_span_m','deck_width_m','distance_miles_exact']:
                flat[c+'_a']=ra[c];flat[c+'_b']=rb[c];flat[c+'_diff_a_minus_b']=ra[c]-rb[c]
            match_flat.append(flat)
    pair_summary['matched_sets']=matched_sets
    pair_summary['age_common_range']=[lo,hi]
    pair_summary['shares']={
        'state_agency_a':float(np.mean([r['state_agency_both'] for r in a])) if a else math.nan,
        'state_agency_b':float(np.mean([r['state_agency_both'] for r in b])) if b else math.nan,
        'open_unrestricted_a':float(np.mean([r['open_unrestricted'] for r in a])) if a else math.nan,
        'open_unrestricted_b':float(np.mean([r['open_unrestricted'] for r in b])) if b else math.nan,
        'reconstructed_a':float(np.mean([r['reconstructed'] for r in a])) if a else math.nan,
        'reconstructed_b':float(np.mean([r['reconstructed'] for r in b])) if b else math.nan,
        'recent_improvement_a':float(np.mean([r['recent_improvement'] for r in a])) if a else math.nan,
        'recent_improvement_b':float(np.mean([r['recent_improvement'] for r in b])) if b else math.nan,
        'truck_pct_observed_a':float(np.mean([r.get('truck_pct_observed',0) for r in a])) if a else math.nan,
        'truck_pct_observed_b':float(np.mean([r.get('truck_pct_observed',0) for r in b])) if b else math.nan,
    }
    results['pair_results'][pair]=pair_summary

# Write result files
with open(OUT_JSON,'w',encoding='utf-8') as f:json.dump(results,f,indent=2,allow_nan=True)
if match_flat:
    fields=[]
    for r in match_flat:
        for k in r:
            if k not in fields:fields.append(k)
    with open(MATCH_CSV,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(match_flat)
if model_flat:
    fields=[]
    for r in model_flat:
        for k in r:
            if k not in fields:fields.append(k)
    with open(PAIR_RESULTS_CSV,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(model_flat)

# Console summary
print(json.dumps({
    'rows':len(rows),'found':found,
    'summary':{
        pair:{
            'n':[results['pair_results'][pair]['n_a'],results['pair_results'][pair]['n_b']],
            'age_range':results['pair_results'][pair]['age_common_range'],
            'matches':[(x['label'],x['n_pairs'],next((s for s in x['stats'] if s['outcome']=='lowest_rating'),{}).get('mean_a_minus_b')) for x in results['pair_results'][pair]['matched_sets']],
            'expanded':next((r for r in results['model_results'] if r['border_pair']==pair and r['specification']=='Truck-expanded adjusted' and r['outcome']=='lowest_rating'),None),
            'state_agency':next((r for r in results['model_results'] if r['border_pair']==pair and r['specification']=='State-owned and state-maintained only' and r['outcome']=='lowest_rating'),None),
            'overlap':next((r for r in results['model_results'] if r['border_pair']==pair and r['model_family']=='Overlap weighting' and r['outcome']=='lowest_rating'),None),
        } for pair in PAIRS
    }
},indent=2,allow_nan=True))
