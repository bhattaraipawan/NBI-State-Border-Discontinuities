import csv, gzip, json, math
from pathlib import Path
from collections import Counter
import numpy as np
import statsmodels.api as sm
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
BRIDGE_FILE = ROOT / 'data/processed/KS_MO_Enriched_Bridge_Data_25mi.csv.gz'
NBI_FILE = ROOT / 'data/raw/2025HwyBridgesDelimitedAllStates.txt'
CLIMATE_FILE = ROOT / 'data/external/NOAA_nClimDiv_1991_2020_County_Climate_KS_MO.csv'
OUT_JSON = ROOT / 'results/generated/KS_MO_Climate_Sensitivity_Results.json'
OUT_CSV = ROOT / 'results/generated/KS_MO_Climate_Model_Results.csv'
OUT_BRIDGES = ROOT / 'data/processed/KS_MO_Climate_Enriched_Bridges_10mi.csv.gz'

PAIRS = ['KS-MO']
STATE_FIPS = {'KS':'20','MO':'29'}
FIPS_STATE = {v:k for k,v in STATE_FIPS.items()}
CONT_EXPANDED = ['age_2025','log_adt','truck_pct','log_aadtt','structure_length_m','max_span_m','deck_width_m','lanes_on']
CAT_EXPANDED = ['structure_kind_code','structure_type_code','functional_class_code','design_load_code','highway_system_code','national_network','deck_protection_code']
CLIMATE_PRIMARY = ['winter_djf_mean_temp_f','annual_precip_in']
CLIMATE_ANNUAL = ['annual_mean_temp_f','annual_precip_in']
OUTCOMES = [('lowest_rating','Lowest rating'),('deck_rating','Deck rating'),('superstructure_rating','Superstructure rating'),('substructure_rating','Substructure rating')]
NUMERIC = ['latitude','longitude','distance_miles_exact','signed_distance_miles_exact','deck_rating','superstructure_rating','substructure_rating','lowest_rating','year_built','age_2025','adt','log_adt','truck_pct','aadtt','log_aadtt','lanes_on','structure_length_m','max_span_m','deck_width_m','inspection_frequency_months','year_reconstructed','year_of_improvement','operating_rating','inventory_rating','posting_eval','reconstructed','recent_improvement','open_unrestricted','state_owned','state_maintained','state_agency_both','national_network','strahnet']

def fnum(v):
    try: return float(v)
    except: return math.nan

def finite(v):
    try: return math.isfinite(float(v))
    except: return False

def code(v): return '' if v is None else str(v).strip()

def shared_categories(rows, field, min_each=5):
    ca=Counter(code(r.get(field)) or 'MISSING' for r in rows if r['side_b']==0)
    cb=Counter(code(r.get(field)) or 'MISSING' for r in rows if r['side_b']==1)
    return {k for k in set(ca)|set(cb) if ca[k]>=min_each and cb[k]>=min_each}

def one_hot(values, keep=None, prefix='cat'):
    vals=[code(v) or 'MISSING' for v in values]
    if keep is not None: vals=[v if v in keep else 'OTHER' for v in vals]
    levs=sorted(set(vals))
    if len(levs)<=1: return np.zeros((len(vals),0)),[]
    cols=[]; names=[]
    for lev in levs[1:]:
        cols.append(np.array([1.0 if v==lev else 0.0 for v in vals]))
        names.append(f'{prefix}={lev}')
    return np.column_stack(cols),names

def make_design(rows, cont_controls, cat_controls):
    signed=np.array([r['signed_distance_miles_exact'] for r in rows],float)
    side=(signed>=0).astype(float); dist=np.abs(signed)
    cols=[np.ones(len(rows)),side,dist,side*dist]
    names=['Intercept','side_b','distance_abs','side_x_distance']
    sx,sn=one_hot([r.get('nearest_border_grid_25km','') for r in rows], shared_categories(rows,'nearest_border_grid_25km',2),'segment')
    for j in range(sx.shape[1]): cols.append(sx[:,j])
    names += sn
    for field in cont_controls:
        x=np.array([r[field] for r in rows],float)
        mu=np.nanmean(x); sd=np.nanstd(x,ddof=1)
        x=(x-mu)/sd if finite(sd) and sd>1e-12 else np.zeros(len(rows))
        cols.append(x); names.append(field+'_z')
    for field in cat_controls:
        cx,cn=one_hot([r.get(field,'') for r in rows], shared_categories(rows,field,max(5,int(.01*len(rows)))), field)
        for j in range(cx.shape[1]): cols.append(cx[:,j])
        names += cn
    X=np.column_stack(cols)
    keep=[0]+[j for j in range(1,X.shape[1]) if np.nanstd(X[:,j])>1e-12]
    return X[:,keep],[names[j] for j in keep]

def fit_rd(rows, outcome, cont_controls, cat_controls, label):
    used=[r for r in rows if finite(r.get(outcome)) and all(finite(r.get(c)) for c in cont_controls)]
    X,names=make_design(used,cont_controls,cat_controls)
    y=np.array([r[outcome] for r in used],float)
    groups=np.array([r['nearest_border_grid_25km'] for r in used])
    nc=len(set(groups))
    model=sm.OLS(y,X)
    if nc>=10:
        res=model.fit(cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True}); cov='cluster_25km'
        df=nc-1; reference=f't({df})'
        j=names.index('side_b'); est=-float(res.params[j]); se=float(res.bse[j])
        crit=student_t.ppf(.975,df); ci_low=est-crit*se; ci_high=est+crit*se
        p_value=float(2*student_t.sf(abs(est/se),df))
    else:
        res=model.fit(cov_type='HC3'); cov='HC3'; df=None; reference='normal robust'
        j=names.index('side_b'); est=-float(res.params[j]); se=float(res.bse[j])
        ci=res.conf_int(alpha=.05)[j]; ci_low=-float(ci[1]); ci_high=-float(ci[0])
        p_value=float(res.pvalues[j])
    return {
        'label':label,'n':len(used),'clusters':nc,'covariance':cov,'inference_reference':reference,
        'estimate_a_minus_b':est,'se':se,'ci_low':ci_low,'ci_high':ci_high,
        'p_value':p_value,'t_p_value':p_value,'r_squared':float(res.rsquared)
    }

def pooled_smd(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    sp=math.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return float((np.mean(a)-np.mean(b))/sp) if sp>1e-12 else 0.0

# Climate lookup
climate={}
with open(CLIMATE_FILE,'r',encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        climate[(r['state'],r['county_code'])]={
            'annual_mean_temp_f':float(r['annual_mean_temp_f']),
            'winter_djf_mean_temp_f':float(r['winter_djf_mean_temp_f']),
            'annual_precip_in':float(r['annual_precip_in'])
        }

# Read bridge rows first, then find NBI counties for exactly those structures.
with gzip.open(BRIDGE_FILE,'rt',encoding='utf-8-sig',newline='') as f:
    bridge_all=list(csv.DictReader(f))
keys={(r['state'],r['structure_number'].strip()) for r in bridge_all}
county={}
with open(NBI_FILE,'r',encoding='utf-8-sig',newline='',errors='replace') as f:
    rr=csv.DictReader(f,quotechar="'")
    for r in rr:
        st=FIPS_STATE.get(str(r.get('STATE_CODE_001','')).strip().zfill(2))
        if not st: continue
        key=(st,str(r.get('STRUCTURE_NUMBER_008','')).strip())
        if key in keys:
            county[key]=str(r.get('COUNTY_CODE_003','')).strip().zfill(3)

rows=[]
for r0 in bridge_all:
    if float(r0['distance_miles_exact'])>10: continue
    r=dict(r0)
    for c in NUMERIC: r[c]=fnum(r.get(c))
    r['side_b']=1 if r['state']==r['state_b'] else 0
    r['county_code']=county[(r['state'],r['structure_number'].strip())]
    cv=climate[(r['state'],r['county_code'])]
    r.update(cv)
    rows.append(r)

results={'metadata':{'records_10mi':len(rows),'counties':len(set((r['state'],r['county_code']) for r in rows)),
                     'climate_source':'NOAA nClimDiv county 1991-2020 long-term monthly averages; assigned by NBI county',
                     'primary_climate_controls':CLIMATE_PRIMARY,'expanded_bridge_traffic_controls':CONT_EXPANDED,'truck_item_109_policy':'Blank Item 109 retained as missing; climate models use complete cases for truck_pct/log_aadtt.'},
         'models':[],'climate_balance':[],'climate_continuity':[]}

for pair in PAIRS:
    pr10=[r for r in rows if r['border_pair']==pair]
    for bw in [5,10]:
        pr=[r for r in pr10 if r['distance_miles_exact']<=bw]
        for outcome,label in OUTCOMES:
            b=fit_rd(pr,outcome,CONT_EXPANDED,CAT_EXPANDED,f'{bw}-mi bridge/traffic adjusted')
            b.update({'border_pair':pair,'bandwidth_miles':bw,'outcome':outcome,'outcome_label':label,'specification':'Bridge/traffic adjusted'})
            results['models'].append(b)
            c=fit_rd(pr,outcome,CONT_EXPANDED+CLIMATE_PRIMARY,CAT_EXPANDED,f'{bw}-mi + winter climate')
            c.update({'border_pair':pair,'bandwidth_miles':bw,'outcome':outcome,'outcome_label':label,'specification':'Climate-adjusted (winter temperature + annual precipitation)'})
            results['models'].append(c)
        # Annual-temperature sensitivity for main outcome only.
        a=fit_rd(pr,'lowest_rating',CONT_EXPANDED+CLIMATE_ANNUAL,CAT_EXPANDED,f'{bw}-mi + annual climate')
        a.update({'border_pair':pair,'bandwidth_miles':bw,'outcome':'lowest_rating','outcome_label':'Lowest rating','specification':'Climate-adjusted sensitivity (annual temperature + annual precipitation)'})
        results['models'].append(a)
        # Climate balance and continuity.
        aa=[r for r in pr if r['side_b']==0]; bb=[r for r in pr if r['side_b']==1]
        for v,label in [('annual_mean_temp_f','Annual mean temperature (F)'),('winter_djf_mean_temp_f','DJF mean temperature (F)'),('annual_precip_in','Annual precipitation (in)')]:
            va=[r[v] for r in aa]; vb=[r[v] for r in bb]
            results['climate_balance'].append({'border_pair':pair,'bandwidth_miles':bw,'variable':v,'label':label,'n_a':len(va),'n_b':len(vb),'mean_a':float(np.mean(va)),'mean_b':float(np.mean(vb)),'difference_a_minus_b':float(np.mean(va)-np.mean(vb)),'smd':pooled_smd(va,vb)})
            cc=fit_rd(pr,v,[],[],f'{bw}-mi climate continuity')
            cc.update({'border_pair':pair,'bandwidth_miles':bw,'variable':v,'variable_label':label})
            results['climate_continuity'].append(cc)
    # 10-mile state-agency and common-age climate sensitivities.
    state_ag=[r for r in pr10 if int(r['state_agency_both'])==1]
    st=fit_rd(state_ag,'lowest_rating',CONT_EXPANDED+CLIMATE_PRIMARY,CAT_EXPANDED,'State-agency only + climate')
    st.update({'border_pair':pair,'bandwidth_miles':10,'outcome':'lowest_rating','outcome_label':'Lowest rating','specification':'State-owned/state-maintained + climate'})
    results['models'].append(st)
    a=np.array([r['age_2025'] for r in pr10 if r['side_b']==0]); b=np.array([r['age_2025'] for r in pr10 if r['side_b']==1])
    lo=max(np.quantile(a,.05),np.quantile(b,.05)); hi=min(np.quantile(a,.95),np.quantile(b,.95))
    common=[r for r in pr10 if lo<=r['age_2025']<=hi]
    ca=fit_rd(common,'lowest_rating',CONT_EXPANDED+CLIMATE_PRIMARY,CAT_EXPANDED,'Common-age + climate')
    ca.update({'border_pair':pair,'bandwidth_miles':10,'outcome':'lowest_rating','outcome_label':'Lowest rating','specification':f'Common-age ({lo:.1f}-{hi:.1f} y) + climate'})
    results['models'].append(ca)

with open(OUT_JSON,'w',encoding='utf-8') as f: json.dump(results,f,indent=2)
# Flat model CSV
fields=[]
for r in results['models']:
    for k in r:
        if k not in fields: fields.append(k)
with open(OUT_CSV,'w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results['models'])
# Climate-enriched bridge data
out_fields=list(rows[0].keys())
with gzip.open(OUT_BRIDGES,'wt',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out_fields); w.writeheader(); w.writerows(rows)

print(json.dumps({
    'metadata':results['metadata'],
    'main_10mi':[r for r in results['models'] if r['bandwidth_miles']==10 and r['outcome']=='lowest_rating' and r['specification'] in ['Bridge/traffic adjusted','Climate-adjusted (winter temperature + annual precipitation)','State-owned/state-maintained + climate']]
},indent=2))
