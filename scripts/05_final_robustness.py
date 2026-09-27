"""Final robustness checks for the Kansas-Missouri lead case.

This script consolidates the robustness code used for the manuscript:
small-cluster t inference, donut RD, shared-state exclusion, equivalence
sensitivity, leave-one-segment-out, wild-cluster sign bootstrap, cumulative
rating thresholds, and shifted-cutoff placebo tests.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.stats import t as student_t, norm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/KS_MO_Enriched_Bridge_Data_25mi.csv.gz"
NBI = ROOT / "data/raw/2025HwyBridgesDelimitedAllStates.txt"
OUT = ROOT / "results/generated/KS_MO_Final_Robustness_Results.json"

PAIRS = ["KS-MO"]
STATE_FIPS = {"KS":"20","MO":"29"}
CONT = ["age_2025","log_adt","truck_pct","log_aadtt","structure_length_m","max_span_m","deck_width_m","lanes_on"]
CATS = ["structure_kind_code","structure_type_code","functional_class_code","design_load_code","highway_system_code","national_network","deck_protection_code"]


def finite(x):
    try: return math.isfinite(float(x))
    except Exception: return False

def code(v): return str(v or "").strip()

def shared_categories(rows, field, min_each=5, sides=None):
    """Return levels represented on both current analysis sides.

    Passing ``sides`` is essential for shifted-cutoff placebo models because the
    pseudo cutoff lies inside one state and the original state-side indicator is
    constant within that window.
    """
    if sides is None:
        sides=[r["side_b"] for r in rows]
    ca=Counter(code(r.get(field)) or "MISSING" for r,side in zip(rows,sides) if int(side)==0)
    cb=Counter(code(r.get(field)) or "MISSING" for r,side in zip(rows,sides) if int(side)==1)
    return {k for k in set(ca)|set(cb) if ca[k]>=min_each and cb[k]>=min_each}

def one_hot(vals, keep, prefix):
    vals=[code(v) or "MISSING" for v in vals]
    vals=[v if v in keep else "OTHER" for v in vals]
    levs=sorted(set(vals)); cols=[]; names=[]
    for lev in levs[1:]:
        cols.append(np.array([1.0 if v==lev else 0.0 for v in vals])); names.append(f"{prefix}={lev}")
    return (np.column_stack(cols) if cols else np.zeros((len(vals),0))), names

def design(rows, cutoff=0.0, include_covariates=True):
    signed=np.array([r["signed_distance_miles_exact"]-cutoff for r in rows],float)
    side=(signed>=0).astype(float); dist=np.abs(signed)
    cols=[np.ones(len(rows)),side,dist,side*dist]; names=["Intercept","side_b","distance_abs","side_x_distance"]
    keep=shared_categories(rows,"nearest_border_grid_25km",2,sides=side)
    x,n=one_hot([r["nearest_border_grid_25km"] for r in rows],keep,"segment")
    for j in range(x.shape[1]): cols.append(x[:,j])
    names += n
    if include_covariates:
        for f in CONT:
            v=np.array([r[f] for r in rows],float); sd=np.std(v,ddof=1); v=(v-np.mean(v))/sd if sd>1e-12 else np.zeros(len(v))
            cols.append(v); names.append(f+"_z")
        for f in CATS:
            keep=shared_categories(rows,f,max(5,int(.01*len(rows))),sides=side)
            x,n=one_hot([r.get(f,"") for r in rows],keep,f)
            for j in range(x.shape[1]): cols.append(x[:,j])
            names += n
    X=np.column_stack(cols)
    keep_cols=[0]+[j for j in range(1,X.shape[1]) if np.std(X[:,j])>1e-12]
    return X[:,keep_cols],[names[j] for j in keep_cols]

def fit(rows, cutoff=0.0, outcome="lowest_rating", include_covariates=True):
    req=[outcome]+(CONT if include_covariates else [])
    used=[r for r in rows if all(finite(r.get(c)) for c in req)]
    if len(used)<40:
        raise ValueError(f"Insufficient complete cases for {outcome}: {len(used)}")
    X,names=design(used,cutoff,include_covariates)
    y=np.array([r[outcome] for r in used],float)
    groups=np.array([r["nearest_border_grid_25km"] for r in used])
    nc=len(set(groups)); j=names.index("side_b")
    model=sm.OLS(y,X)
    hc3=model.fit(cov_type="HC3")
    if nc>=10:
        res=model.fit(cov_type="cluster",cov_kwds={"groups":groups,"use_correction":True})
        covariance="cluster_25km"
        df=nc-1
        est=-float(res.params[j]); se=float(res.bse[j]); crit=student_t.ppf(.975,df)
        ci_low,ci_high=est-crit*se,est+crit*se
        p_value=float(2*student_t.sf(abs(est/se),df))
        reference=f"t({df})"
    else:
        res=hc3
        covariance="HC3"
        df=None
        est=-float(res.params[j]); se=float(res.bse[j])
        ci=res.conf_int(alpha=.05)[j]
        ci_low,ci_high=-float(ci[1]),-float(ci[0])
        p_value=float(res.pvalues[j])
        reference="normal robust"
    return {"n":len(used),"clusters":nc,"covariance":covariance,"inference_reference":reference,
            "estimate_a_minus_b":est,"se":se,"ci_low":ci_low,"ci_high":ci_high,"p_value":p_value,
            "t_p":p_value,"normal_p":float(res.pvalues[j]),"hc3_se":float(hc3.bse[j]),"df":df,
            "used":used,"X":X,"names":names,"y":y,"groups":groups,"model":model,"cluster_res":res}

def read_rows():
    rows=[]
    with gzip.open(DATA,"rt",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if r["border_pair"] not in PAIRS: continue
            for c in ["distance_miles_exact","signed_distance_miles_exact","lowest_rating","deck_rating","superstructure_rating","substructure_rating"]+CONT:
                try:r[c]=float(r[c])
                except:r[c]=math.nan
            for c in ["national_network","state_agency_both"]:
                try:r[c]=int(float(r[c]))
                except:r[c]=0
            r["side_b"]=1 if r["state"]==r["state_b"] else 0
            gm=code(r.get("reported_state_geometry_match")).lower()
            r["reported_state_geometry_match"]=gm in {"true","1","yes"} if gm else True
            rows.append(r)
    # Add NBI other-state flag.
    key={(STATE_FIPS[r["state"]],r["structure_number"].strip()):r for r in rows}
    with open(NBI,"r",encoding="utf-8-sig",newline="",errors="replace") as f:
        for x in csv.DictReader(f,quotechar="'"):
            k=(code(x.get("STATE_CODE_001")),code(x.get("STRUCTURE_NUMBER_008")))
            if k in key:
                val=code(x.get("OTHER_STATE_CODE_098A"))
                key[k]["shared_state_structure"]=bool(val and val not in {"0","00"})
    for r in rows: r.setdefault("shared_state_structure",False)
    return rows

def equivalence(est,se,df,pair):
    out=[]
    if df is None:
        crit90=norm.ppf(.95)
        sf=lambda z: float(norm.sf(z))
        cdf=lambda z: float(norm.cdf(z))
        reference="normal robust"
    else:
        crit90=student_t.ppf(.95,df)
        sf=lambda z: float(student_t.sf(z,df))
        cdf=lambda z: float(student_t.cdf(z,df))
        reference=f"t({df})"
    for margin in (.25,.50):
        lo=est-crit90*se; hi=est+crit90*se
        p_lower=sf((est+margin)/se)
        p_upper=cdf((est-margin)/se)
        out.append({"border_pair":pair,"margin":margin,"estimate":est,"ci90_low":lo,"ci90_high":hi,
                    "p_lower":p_lower,"p_upper":p_upper,"tost_p":max(p_lower,p_upper),"equivalent":max(p_lower,p_upper)<.05,
                    "inference_reference":reference})
    if est>=0:
        p=sf((est-.25)/se); direction="A higher by >0.25"
    else:
        p=cdf((est+.25)/se); direction="B higher by >0.25"
    out.append({"border_pair":pair,"margin":.25,"test":"magnitude_exceeds","direction":direction,"estimate":est,"one_sided_p":p,
                "inference_reference":reference})
    return out

def wild_cluster_p(base, draws=50000, seed=42):
    """Rademacher cluster sign bootstrap under H0: side coefficient = 0.

    With 10 or fewer clusters, enumerate every 2^G sign pattern exactly. With
    more clusters, use ``draws`` reproducible Monte Carlo sign patterns.
    """
    X=base["X"]; y=base["y"]; names=base["names"]; groups=base["groups"]
    j=names.index("side_b")
    X0=np.delete(X,j,axis=1)
    r0=sm.OLS(y,X0).fit(); fitted=r0.fittedvalues; resid=r0.resid
    uniq=np.array(sorted(set(groups)))
    observed=abs(base["estimate_a_minus_b"])
    if len(uniq)<=10:
        patterns=list(itertools.product((-1.0,1.0), repeat=len(uniq)))
        method="exhaustive_rademacher"
    else:
        rng=np.random.default_rng(seed)
        patterns=[tuple(rng.choice([-1.0,1.0],size=len(uniq))) for _ in range(draws)]
        method="monte_carlo_rademacher"
    boots=[]; signed=[]
    for pattern in patterns:
        smap=dict(zip(uniq,pattern))
        yb=fitted+np.array([smap[g] for g in groups])*resid
        rr=sm.OLS(yb,X).fit(cov_type="cluster",cov_kwds={"groups":groups,"use_correction":True})
        b=-float(rr.params[j])
        boots.append(abs(b)); signed.append(b)
    boots=np.asarray(boots); signed=np.asarray(signed)
    if method=="exhaustive_rademacher":
        p=float(np.mean(boots>=observed))
    else:
        p=float((1+np.sum(boots>=observed))/(len(boots)+1))
    q=np.quantile(signed,[.025,.975])
    return {"p_value":p,"null_q025":float(q[0]),"null_q975":float(q[1]),
            "method":method,"patterns":len(patterns),"seed":None if method=="exhaustive_rademacher" else seed}

def cumulative_gee(rows, threshold):
    used=[r for r in rows if all(finite(r.get(c)) for c in CONT+["lowest_rating"])]
    X,names=design(used,0,True); j=names.index("side_b")
    y=np.array([1 if r["lowest_rating"]>=threshold else 0 for r in used])
    groups=np.array([r["nearest_border_grid_25km"] for r in used])
    res=sm.GEE(y,X,groups=groups,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Independence()).fit()
    # side coefficient is B vs A; report OR A vs B by reversing sign.
    beta=-float(res.params[j]); se=float(res.bse[j]); crit=1.96
    return {"n":len(used),"odds_ratio_a_vs_b":math.exp(beta),"ci_low":math.exp(beta-crit*se),"ci_high":math.exp(beta+crit*se),"p_value":float(res.pvalues[j])}

def main():
    if not DATA.exists() or not NBI.exists(): raise SystemExit("Run scripts 01-04 first and place the raw NBI file in data/raw.")
    rows=read_rows(); result={"metadata":{
        "continuous_controls":CONT,"categorical_controls":CATS,
        "truck_item_109_policy":"Blank Item 109 retained as missing; final expanded models use complete cases for truck_pct/log_aadtt.",
        "covariance_rule":"cluster by 25-km border segment when clusters >= 10; otherwise HC3",
        "wild_cluster_rule":"exhaust all Rademacher sign patterns when clusters <= 10; otherwise 50,000 seeded Monte Carlo patterns",
        "placebo_rule":"pseudo-side and shared fixed-effect/category levels are defined relative to each shifted cutoff"},
        "baseline":[],"small_cluster":[],"donut":[],"spatial_validation":[],"equivalence":[],"leave_one_segment":[],"wild_cluster":[],"ordinal":[],"placebo":[]}
    for pair in PAIRS:
        pr=[r for r in rows if r["border_pair"]==pair and r["distance_miles_exact"]<=10]
        base=fit(pr)
        result["baseline"].append({"border_pair":pair,"n":base["n"],"clusters":base["clusters"],"covariance":base["covariance"],"estimate_a_minus_b":base["estimate_a_minus_b"],"se":base["se"],"p_value":base["p_value"],"hc3_se":base["hc3_se"]})
        result["small_cluster"].append({"border_pair":pair,"df":base["df"],"inference_reference":base["inference_reference"],"estimate_a_minus_b":base["estimate_a_minus_b"],"se":base["se"],"ci_low":base["ci_low"],"ci_high":base["ci_high"],"p_value":base["p_value"]})
        result["equivalence"].extend(equivalence(base["estimate_a_minus_b"],base["se"],base["df"],pair))
        for cut,label in [(.5,"Donut: exclude <0.5 mile"),(1.0,"Donut: exclude <1.0 mile")]:
            rr=fit([r for r in pr if r["distance_miles_exact"]>=cut]); result["donut"].append({"border_pair":pair,"specification":label,**{k:rr[k] for k in ["n","clusters","covariance","estimate_a_minus_b","se","ci_low","ci_high","p_value"]}})
        rr=fit([r for r in pr if not r["shared_state_structure"]]); result["donut"].append({"border_pair":pair,"specification":"Exclude shared-state structures",**{k:rr[k] for k in ["n","clusters","covariance","estimate_a_minus_b","se","ci_low","ci_high","p_value"]}})
        rr=fit([r for r in pr if int(r.get("state_agency_both",0))==1]); result["donut"].append({"border_pair":pair,"specification":"State-owned/state-maintained only",**{k:rr[k] for k in ["n","clusters","covariance","estimate_a_minus_b","se","ci_low","ci_high","p_value"]}})
        geo=[r for r in pr if r.get("reported_state_geometry_match",True)]
        if len(geo)<len(pr):
            rr=fit(geo)
            result["spatial_validation"].append({"border_pair":pair,"excluded_geometry_mismatches":len(pr)-len(geo),**{k:rr[k] for k in ["n","clusters","covariance","estimate_a_minus_b","se","ci_low","ci_high","p_value"]}})
        else:
            result["spatial_validation"].append({"border_pair":pair,"excluded_geometry_mismatches":0,"n":len(pr),"note":"No reported-state/Census-polygon mismatches in 10-mile sample"})
        # Leave one 25-km segment out.
        for seg in sorted(set(r["nearest_border_grid_25km"] for r in pr)):
            q=fit([r for r in pr if r["nearest_border_grid_25km"]!=seg])
            result["leave_one_segment"].append({"border_pair":pair,"omitted_segment":seg,"n":q["n"],"clusters":q["clusters"],"covariance":q["covariance"],"estimate":q["estimate_a_minus_b"],"se":q["se"],"p_value":q["p_value"]})
        wb=wild_cluster_p(base,draws=50000)
        result["wild_cluster"].append({"border_pair":pair,"clusters":base["clusters"],"estimate":base["estimate_a_minus_b"],"bootstrap_p":wb["p_value"],"method":wb["method"],"patterns":wb["patterns"],"seed":wb["seed"],"null_q025":wb["null_q025"],"null_q975":wb["null_q975"]})
        for th in (5,6,7):
            q=cumulative_gee(pr,th); result["ordinal"].append({"border_pair":pair,"threshold":th,**q,"model":"Cumulative binary GEE, independence working correlation, cluster-robust by 25-km segment"})
        # Shifted-cutoff placebo tests: same-state windows centered 5/10/15 miles from real border.
        for cutoff in (-15,-10,-5,5,10,15):
            fake=[r for r in rows if r["border_pair"]==pair and abs(r["signed_distance_miles_exact"]-cutoff)<=4]
            q=fit(fake,cutoff=cutoff)
            result["placebo"].append({"border_pair":pair,"cutoff":cutoff,"window":4,"n":q["n"],"clusters":q["clusters"],"covariance":q["covariance"],"inference_reference":q["inference_reference"],
                "pseudo_left_minus_right":q["estimate_a_minus_b"],"se":q["se"],"ci_low":q["ci_low"],"ci_high":q["ci_high"],"p_value":q["p_value"],"significant":q["p_value"]<.05})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with open(OUT,"w",encoding="utf-8") as f: json.dump(result,f,indent=2,allow_nan=True)
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
