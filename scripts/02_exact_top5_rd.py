import csv, gzip, json, math, os, zipfile
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import geopandas as gpd
import shapely
from pyproj import Transformer
from scipy.stats import chi2_contingency, ttest_ind
import statsmodels.api as sm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
PROX = ROOT / 'data/processed/National_2025_NBI_Border_Proximate_Bridges_25mi.csv.gz'
STATE_ZIP = ROOT / 'data/raw/tl_2025_us_state.zip'
SCREENING_HIGH = ROOT / 'results/generated/National_2025_NBI_High_Priority_Borders.csv'

def load_screening_top5():
    if not SCREENING_HIGH.exists():
        raise SystemExit(f'Missing screening output: {SCREENING_HIGH}')
    with open(SCREENING_HIGH,'r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows) < 5:
        raise SystemExit(f'Need at least five high-priority borders; found {len(rows)}')
    rows.sort(key=lambda r: float(r.get('screening_score','nan')), reverse=True)
    labels=[r['border_pair'] for r in rows[:5]]
    pairs=[tuple(x.split('-',1)) for x in labels]
    return pairs, labels

TOP5, TOP5_ORDERED_LABELS = load_screening_top5()
ANALYSIS_LABELS = list(TOP5_ORDERED_LABELS)
ANALYSIS_PAIRS = list(TOP5)
ANALYSIS_LABEL_SET = set(TOP5_ORDERED_LABELS)
M_PER_MILE = 1609.344
OUTDIR = ROOT / 'results/generated/top5_exact_outputs'
OUTDIR.mkdir(parents=True, exist_ok=True)
(ROOT/'data/processed').mkdir(parents=True, exist_ok=True)
(ROOT/'results/generated').mkdir(parents=True, exist_ok=True)

NUMERIC_FIELDS = [
    'latitude','longitude','distance_miles_screening','signed_distance_miles_screening',
    'deck_rating','superstructure_rating','substructure_rating','lowest_rating',
    'year_built','age_2025','adt','log_adt','structure_length_m','max_span_m',
    'deck_width_m','inspection_frequency_months','year_reconstructed'
]
CONTROLS = ['age_2025','log_adt','structure_length_m','max_span_m','deck_width_m']
CAT_CONTROLS = ['structure_kind_code','structure_type_code','functional_class_code','design_load_code','highway_system_code']
OUTCOMES = [
    ('lowest_rating','Lowest component rating'),
    ('deck_rating','Deck rating'),
    ('superstructure_rating','Superstructure rating'),
    ('substructure_rating','Substructure rating'),
]

# ---------- utility ----------
def fnum(v):
    if v is None or v == '': return math.nan
    try: return float(v)
    except: return math.nan

def isfinite(v): return isinstance(v,(int,float,np.floating)) and math.isfinite(float(v))

def pooled_smd(a,b):
    a=np.asarray([x for x in a if isfinite(x)],float); b=np.asarray([x for x in b if isfinite(x)],float)
    if len(a)<2 or len(b)<2: return math.nan
    sp=math.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return (float(np.mean(a))-float(np.mean(b)))/sp if sp>0 else 0.0

def cramers_v(vals, sides):
    levels=sorted(set(vals)); side_levels=sorted(set(sides))
    if len(levels)<2 or len(side_levels)<2: return 0.0
    table=np.zeros((len(side_levels),len(levels)),dtype=int)
    si={v:i for i,v in enumerate(side_levels)}; li={v:i for i,v in enumerate(levels)}
    for v,s in zip(vals,sides): table[si[s],li[v]]+=1
    # remove empty columns
    table=table[:,table.sum(axis=0)>0]
    if table.shape[1]<2: return 0.0
    try: chi2,_,_,_=chi2_contingency(table, correction=False)
    except ValueError: return math.nan
    n=table.sum(); denom=min(table.shape[0]-1,table.shape[1]-1)
    return math.sqrt((chi2/n)/denom) if n>0 and denom>0 else 0.0

def shared_categories(rows, field, min_each=5, sides=None):
    """Categories represented on both analysis sides.

    For true-border models ``sides`` defaults to the stored state-side indicator.
    For shifted-cutoff placebos, callers must pass the pseudo-side indicator so
    fixed effects and categorical controls are defined relative to the fake
    cutoff rather than the original state border.
    """
    if sides is None:
        sides=[r['side_b'] for r in rows]
    ca=Counter((r.get(field,'MISSING') or 'MISSING') for r,side in zip(rows,sides) if int(side)==0)
    cb=Counter((r.get(field,'MISSING') or 'MISSING') for r,side in zip(rows,sides) if int(side)==1)
    return {k for k in set(ca)|set(cb) if ca[k]>=min_each and cb[k]>=min_each}

def one_hot(values, keep=None, prefix='cat'):
    vals=[(v if v not in (None,'') else 'MISSING') for v in values]
    if keep is not None:
        vals=[v if v in keep else 'OTHER' for v in vals]
    levels=sorted(set(vals))
    if len(levels)<=1: return np.zeros((len(vals),0)), []
    # drop first reference
    cols=[]; names=[]
    for lev in levels[1:]:
        cols.append(np.array([1.0 if v==lev else 0.0 for v in vals]))
        names.append(f'{prefix}={lev}')
    return np.column_stack(cols), names

def make_design(rows, adjusted=True, cutoff=0.0, window=None):
    # rows must already be filtered to desired window and complete numeric controls/outcome
    n=len(rows)
    signed=np.array([r['signed_distance_miles_exact']-cutoff for r in rows],float)
    side=(signed>=0).astype(float)
    dist=np.abs(signed)
    X=[np.ones(n),side,dist,side*dist]
    names=['Intercept','side_b','distance_abs','side_x_distance']
    # segment fixed effects; keep shared grid cells only, otherwise OTHER
    keep_seg=shared_categories(rows,'nearest_border_grid_25km',min_each=2,sides=side)
    segX,segN=one_hot([r.get('nearest_border_grid_25km','') for r in rows],keep_seg,'segment')
    if segX.shape[1]: X.extend([segX[:,j] for j in range(segX.shape[1])]); names.extend(segN)
    if adjusted:
        for field in CONTROLS:
            vals=np.array([r[field] for r in rows],float)
            mu=np.mean(vals); sd=np.std(vals,ddof=1)
            vals=(vals-mu)/sd if sd>0 else vals*0
            X.append(vals); names.append(field+'_z')
        for field in CAT_CONTROLS:
            keep=shared_categories(rows,field,min_each=max(5,int(0.01*n)),sides=side)
            cx,cn=one_hot([r.get(field,'') for r in rows],keep,field)
            if cx.shape[1]: X.extend([cx[:,j] for j in range(cx.shape[1])]); names.extend(cn)
    X=np.column_stack(X)
    # remove zero variance non-intercept columns
    keep_cols=[0]+[j for j in range(1,X.shape[1]) if np.nanstd(X[:,j])>1e-12]
    X=X[:,keep_cols]; names=[names[j] for j in keep_cols]
    return X,names,side,dist

def fit_jump(rows, outcome, adjusted=True, cutoff=0.0):
    used=[]
    for r in rows:
        if not isfinite(r.get(outcome,math.nan)): continue
        if adjusted and any(not isfinite(r.get(c,math.nan)) for c in CONTROLS): continue
        used.append(r)
    if len(used)<40 or len({r['state'] for r in used})<2 and cutoff==0:
        return None
    X,names,side,dist=make_design(used,adjusted=adjusted,cutoff=cutoff)
    y=np.array([r[outcome] for r in used],float)
    try:
        model=sm.OLS(y,X)
        groups=np.array([r.get('nearest_border_grid_25km','MISSING') for r in used])
        nclusters=len(set(groups))
        if nclusters>=10:
            res=model.fit(cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True})
            cov='Clustered by 25-km border segment'
        else:
            res=model.fit(cov_type='HC3')
            cov='HC3 robust'
        idx=names.index('side_b')
        est_b_minus_a=float(res.params[idx]); se=float(res.bse[idx]); p=float(res.pvalues[idx])
        ci=res.conf_int(alpha=0.05)[idx]
        # output A minus B, reverse sign and CI endpoints
        return {
            'n':len(used),'clusters':nclusters,'covariance':cov,'k':X.shape[1],
            'estimate_a_minus_b':-est_b_minus_a,'se':se,
            'ci_low':-float(ci[1]),'ci_high':-float(ci[0]),'p_value':p,
            'r_squared':float(res.rsquared)
        }
    except Exception as e:
        return {'error':str(e),'n':len(used)}

def balance(rows,bw):
    sub=[r for r in rows if r['distance_miles_exact']<=bw]
    a=[r for r in sub if r['side_b']==0]; b=[r for r in sub if r['side_b']==1]
    cont=[]
    for field in CONTROLS:
        av=[r[field] for r in a]; bv=[r[field] for r in b]
        avf=[x for x in av if isfinite(x)]; bvf=[x for x in bv if isfinite(x)]
        smd=pooled_smd(avf,bvf)
        p=ttest_ind(avf,bvf,equal_var=False,nan_policy='omit').pvalue if len(avf)>1 and len(bvf)>1 else math.nan
        cont.append({'border_pair':sub[0]['border_pair'] if sub else '', 'bandwidth':bw,'variable':field,
                     'n_a':len(avf),'n_b':len(bvf),'mean_a':float(np.mean(avf)) if avf else math.nan,
                     'mean_b':float(np.mean(bvf)) if bvf else math.nan,'smd':smd,'p_value':float(p)})
    cat=[]
    for field in CAT_CONTROLS:
        vals=[r.get(field,'MISSING') or 'MISSING' for r in sub]
        sides=[r['state'] for r in sub]
        cat.append({'border_pair':sub[0]['border_pair'] if sub else '', 'bandwidth':bw,'variable':field,
                    'n_a':len(a),'n_b':len(b),'cramers_v':cramers_v(vals,sides)})
    return sub,a,b,cont,cat

def binned_means(rows,bw=10,bin_width=1.0):
    bins=[]
    edges=np.arange(-bw,bw+bin_width,bin_width)
    for lo,hi in zip(edges[:-1],edges[1:]):
        vals=[r['lowest_rating'] for r in rows if lo<=r['signed_distance_miles_exact']<hi]
        if vals:
            bins.append({'bin_center':(lo+hi)/2,'mean_rating':float(np.mean(vals)),'n':len(vals)})
    return bins

# ---------- load bridge records ----------
records_by_pair=defaultdict(list)
with gzip.open(PROX,'rt',encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f)
    for row in reader:
        if row['border_pair'] not in ANALYSIS_LABEL_SET: continue
        for k in NUMERIC_FIELDS: row[k]=fnum(row.get(k,''))
        records_by_pair[row['border_pair']].append(row)

states=gpd.read_file('zip://'+str(STATE_ZIP)).to_crs(5070)
state_geom={r.STUSPS:r.geometry for _,r in states.iterrows()}
transformer=Transformer.from_crs(4326,5070,always_xy=True)

all_exact=[]; rd_rows=[]; balance_cont=[]; balance_cat=[]; covariate_rd=[]; placebo_rows=[]; sample_rows=[]; bin_rows=[]; border_meta=[]; spatial_validation=[]

for a,b in ANALYSIS_PAIRS:
    pair=f'{a}-{b}'
    rows=records_by_pair[pair]
    border=state_geom[a].boundary.intersection(state_geom[b].boundary)
    lons=np.array([r['longitude'] for r in rows]); lats=np.array([r['latitude'] for r in rows])
    xs,ys=transformer.transform(lons,lats)
    pts=shapely.points(xs,ys)
    dists=shapely.distance(pts,border)/M_PER_MILE
    exact=[]
    for r,x,y,d in zip(rows,xs,ys,dists):
        if not math.isfinite(float(d)) or d>25.0001: continue
        r=dict(r); r['x_5070']=float(x); r['y_5070']=float(y); r['distance_miles_exact']=float(d)
        r['side_b']=1 if r['state']==b else 0
        r['signed_distance_miles_exact']=float(d) if r['side_b']==1 else -float(d)
        # Diagnostic only: confirm the reported state agrees with the Census state polygon.
        # This is not used to define the primary sample; a sensitivity can exclude mismatches.
        pt=shapely.Point(float(x),float(y))
        r['reported_state_geometry_match']=bool(state_geom[r['state']].covers(pt))
        exact.append(r); all_exact.append(r)
    mismatch=sum(not r['reported_state_geometry_match'] for r in exact)
    border_meta.append({'border_pair':pair,'state_a':a,'state_b':b,'selection_role':'screening_top5','border_length_km':float(border.length/1000),'records_25mi':len(exact),'reported_state_geometry_mismatches':mismatch})
    spatial_validation.append({'border_pair':pair,'records_25mi':len(exact),'reported_state_geometry_matches':len(exact)-mismatch,'reported_state_geometry_mismatches':mismatch,'mismatch_share':mismatch/len(exact) if exact else math.nan})

    # maps
    fig,ax=plt.subplots(figsize=(8,6))
    gpd.GeoSeries([state_geom[a].boundary],crs=5070).plot(ax=ax,linewidth=0.8)
    gpd.GeoSeries([state_geom[b].boundary],crs=5070).plot(ax=ax,linewidth=0.8)
    # border as GeoSeries
    gpd.GeoSeries([border],crs=5070).plot(ax=ax,linewidth=2.0)
    sc=ax.scatter([r['x_5070'] for r in exact],[r['y_5070'] for r in exact],c=[r['lowest_rating'] for r in exact],s=8,alpha=0.65)
    fig.colorbar(sc,ax=ax,label='Lowest component rating')
    minx,miny,maxx,maxy=border.bounds; pad=35*M_PER_MILE
    ax.set_xlim(minx-pad,maxx+pad); ax.set_ylim(miny-pad,maxy+pad)
    ax.set_title(f'{pair}: Bridges within 25 miles of shared border')
    ax.set_xlabel('Projected easting (m)'); ax.set_ylabel('Projected northing (m)')
    ax.set_aspect('equal'); fig.tight_layout()
    fig.savefig(OUTDIR/f'{pair}_map.png',dpi=180); plt.close(fig)

    bins=binned_means(exact,10,1.0)
    for z in bins: z.update({'border_pair':pair}); bin_rows.append(z)
    fig,ax=plt.subplots(figsize=(8,5))
    left=[z for z in bins if z['bin_center']<0]; right=[z for z in bins if z['bin_center']>=0]
    if left: ax.plot([z['bin_center'] for z in left],[z['mean_rating'] for z in left],marker='o',label=a)
    if right: ax.plot([z['bin_center'] for z in right],[z['mean_rating'] for z in right],marker='o',label=b)
    ax.axvline(0,linestyle='--')
    ax.set_title(f'{pair}: Binned lowest ratings near border')
    ax.set_xlabel(f'Signed distance (miles; {a} negative, {b} positive)')
    ax.set_ylabel('Mean lowest component rating'); ax.legend(); fig.tight_layout()
    fig.savefig(OUTDIR/f'{pair}_rd_plot.png',dpi=180); plt.close(fig)

    for bw in (5,10,25):
        sub,ra,rb,bc,bk=balance(exact,bw)
        balance_cont.extend(bc); balance_cat.extend(bk)
        sample_rows.append({'border_pair':pair,'state_a':a,'state_b':b,'bandwidth':bw,'n_a':len(ra),'n_b':len(rb),'total_n':len(sub),
                            'mean_lowest_a':float(np.mean([r['lowest_rating'] for r in ra])) if ra else math.nan,
                            'mean_lowest_b':float(np.mean([r['lowest_rating'] for r in rb])) if rb else math.nan,
                            'raw_difference_a_minus_b':float(np.mean([r['lowest_rating'] for r in ra])-np.mean([r['lowest_rating'] for r in rb])) if ra and rb else math.nan})
        for out,label in OUTCOMES:
            base=fit_jump(sub,out,adjusted=False)
            adj=fit_jump(sub,out,adjusted=True)
            row={'border_pair':pair,'state_a':a,'state_b':b,'bandwidth':bw,'outcome':out,'outcome_label':label}
            for pref,res in [('base',base),('adjusted',adj)]:
                if res:
                    for k,v in res.items(): row[f'{pref}_{k}']=v
            rd_rows.append(row)

    # covariate continuity at 10 miles; outcome is the covariate, base model only
    sub10=[r for r in exact if r['distance_miles_exact']<=10]
    for field in CONTROLS:
        res=fit_jump(sub10,field,adjusted=False)
        row={'border_pair':pair,'variable':field}
        if res:
            row.update(res)
        covariate_rd.append(row)

    # Placebo cutoffs 10 miles inside each side, bandwidth ±5 miles
    for cutoff,label in [(-10,f'10 miles inside {a}'),(10,f'10 miles inside {b}')]:
        fake=[r for r in exact if abs(r['signed_distance_miles_exact']-cutoff)<=5]
        # fake model needs both sides of fake cutoff, same state is okay; fit function allows because cutoff !=0
        res=fit_jump(fake,'lowest_rating',adjusted=True,cutoff=cutoff)
        row={'border_pair':pair,'placebo_cutoff':cutoff,'placebo_label':label,'window_miles':5}
        if res: row.update(res)
        placebo_rows.append(row)

# ---------- output CSV/JSON ----------
exact_fields=[
    'border_pair','state_a','state_b','state','structure_number','latitude','longitude','x_5070','y_5070',
    'distance_miles_exact','signed_distance_miles_exact','nearest_border_grid_25km','reported_state_geometry_match',
    'deck_rating','superstructure_rating','substructure_rating','lowest_rating','year_built','age_2025','adt','log_adt',
    'structure_kind_code','structure_type_code','functional_class_code','design_load_code','owner_code','maintenance_code',
    'highway_system_code','structure_length_m','max_span_m','deck_width_m','inspection_frequency_months','year_reconstructed'
]
exact_gz=ROOT/'data/processed/Top5_Exact_Border_Bridge_Data_25mi.csv.gz'
with gzip.open(exact_gz,'wt',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=exact_fields); w.writeheader();
    for r in all_exact: w.writerow({k:r.get(k,'') for k in exact_fields})

def write_csv(path, rows):
    if not rows: return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

write_csv(OUTDIR/'sample_counts.csv',sample_rows)
write_csv(OUTDIR/'rd_results.csv',rd_rows)
write_csv(OUTDIR/'balance_continuous.csv',balance_cont)
write_csv(OUTDIR/'balance_categorical.csv',balance_cat)
write_csv(OUTDIR/'covariate_continuity.csv',covariate_rd)
write_csv(OUTDIR/'placebo_results.csv',placebo_rows)
write_csv(OUTDIR/'binned_means.csv',bin_rows)
write_csv(OUTDIR/'border_metadata.csv',border_meta)
write_csv(OUTDIR/'spatial_validation.csv',spatial_validation)

result={'selection':{'screening_top5':TOP5_ORDERED_LABELS,'analysis_pairs':ANALYSIS_LABELS},'metadata':border_meta,'sample_counts':sample_rows,'rd_results':rd_rows,'balance_continuous':balance_cont,
        'balance_categorical':balance_cat,'covariate_continuity':covariate_rd,'placebo_results':placebo_rows,
        'binned_means':bin_rows,'spatial_validation':spatial_validation,'bridge_records':len(all_exact)}
with open(ROOT/'results/generated/Top5_Exact_RD_Results.json','w',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=True)

# zip plots
plot_zip=ROOT/'results/generated/Top5_Border_Maps_and_RD_Plots.zip'
with zipfile.ZipFile(plot_zip,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUTDIR.glob('*.png')): z.write(p,p.name)

print(json.dumps({'selection':{'screening_top5':TOP5_ORDERED_LABELS,'analysis_pairs':ANALYSIS_LABELS},'bridge_records':len(all_exact),'rd_rows':len(rd_rows),'top10_results':[
    {k:r.get(k) for k in ['border_pair','bandwidth','outcome','adjusted_estimate_a_minus_b','adjusted_ci_low','adjusted_ci_high','adjusted_p_value','adjusted_n']}
    for r in rd_rows if r['bandwidth']==10 and r['outcome']=='lowest_rating'
], 'files':[str(exact_gz),str(plot_zip)]},indent=2))
