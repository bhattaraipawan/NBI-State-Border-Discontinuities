"""National screening of contiguous U.S. state borders without using rating values.

Inputs
------
data/raw/2025HwyBridgesDelimitedAllStates.txt
data/raw/tl_2025_us_state.zip

Outputs
-------
data/processed/National_2025_NBI_Border_Proximate_Bridges_25mi.csv.gz
results/generated/National_2025_NBI_Border_Screening.csv
results/generated/National_2025_NBI_High_Priority_Borders.csv

The screening score and priority rules use bridge/roadway and geographic diagnostics, not condition-rating values. Numeric rating availability still defines the conventional-bridge analysis population used downstream.
"""
from __future__ import annotations

import csv
import gzip
import math
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NBI_FILE = ROOT / "data/raw/2025HwyBridgesDelimitedAllStates.txt"
STATE_ZIP = ROOT / "data/raw/tl_2025_us_state.zip"
OUT_BRIDGES = ROOT / "data/processed/National_2025_NBI_Border_Proximate_Bridges_25mi.csv.gz"
OUT_ALL = ROOT / "results/generated/National_2025_NBI_Border_Screening.csv"
OUT_HIGH = ROOT / "results/generated/National_2025_NBI_High_Priority_Borders.csv"

M_PER_MILE = 1609.344
SAMPLE_STEP_M = 250.0
BANDWIDTHS = (5, 10, 25)

# Contiguous states only. D.C. is intentionally excluded from the state-pair study.
FIPS_TO_STATE = {
    "01":"AL","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","12":"FL","13":"GA",
    "16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH",
    "34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA",
    "44":"RI","45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
    "54":"WV","55":"WI","56":"WY"
}
CONTROLS = ["age_2025", "log_adt", "structure_length_m", "max_span_m", "deck_width_m"]
CORE_CATS = ["structure_kind_code", "structure_type_code", "functional_class_code", "highway_system_code"]
# State-separability uses the broader set of pre-outcome composition variables
# used in the original screening workbook. These are diagnostics only; they are
# not automatically entered as baseline outcome-model controls.
SEPARABILITY_CATS = CORE_CATS + ["design_load_code", "owner_code", "maintenance_code"]


def num(v):
    try:
        return float(str(v).strip())
    except Exception:
        return math.nan


def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def dms_to_decimal(v, longitude=False):
    """Legacy NBI compact DMS: latitude DDMMSSss, longitude DDDMMSSss."""
    s = str(v or "").strip()
    if not s.isdigit():
        return math.nan
    width = 9 if longitude else 8
    s = s.zfill(width)
    deg_n = 3 if longitude else 2
    try:
        deg = int(s[:deg_n])
        minute = int(s[deg_n:deg_n+2])
        sec = int(s[deg_n+2:deg_n+4]) + int(s[deg_n+4:deg_n+6]) / 100.0
        val = deg + minute / 60.0 + sec / 3600.0
        return -val if longitude else val
    except Exception:
        return math.nan


def pooled_smd(a, b):
    a = np.asarray([x for x in a if finite(x)], float)
    b = np.asarray([x for x in b if finite(x)], float)
    if len(a) < 2 or len(b) < 2:
        return math.nan
    sp = math.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / (len(a)+len(b)-2))
    return float((np.mean(a)-np.mean(b))/sp) if sp > 1e-12 else 0.0


def cramers_v(vals, sides):
    vals = [str(v or "MISSING") for v in vals]
    levels = sorted(set(vals)); side_levels = sorted(set(sides))
    if len(levels) < 2 or len(side_levels) < 2:
        return 0.0
    tab = np.zeros((len(side_levels), len(levels)), dtype=int)
    li = {v:i for i,v in enumerate(levels)}; si = {v:i for i,v in enumerate(side_levels)}
    for v,s in zip(vals, sides):
        tab[si[s], li[v]] += 1
    tab = tab[:, tab.sum(axis=0) > 0]
    if tab.shape[1] < 2:
        return 0.0
    chi2, *_ = chi2_contingency(tab, correction=False)
    n = tab.sum(); denom = min(tab.shape[0]-1, tab.shape[1]-1)
    return math.sqrt((chi2/n)/denom) if n and denom else 0.0


def one_hot(vals, min_each=5, sides=None):
    vals = [str(v or "MISSING") for v in vals]
    if sides is not None:
        ca = Counter(v for v,s in zip(vals,sides) if s == 0)
        cb = Counter(v for v,s in zip(vals,sides) if s == 1)
        keep = {v for v in set(vals) if ca[v] >= min_each and cb[v] >= min_each}
        vals = [v if v in keep else "OTHER" for v in vals]
    levels = sorted(set(vals))
    if len(levels) <= 1:
        return np.zeros((len(vals), 0))
    return np.column_stack([[1.0 if v == lev else 0.0 for v in vals] for lev in levels[1:]])


def _fold_one_hot(train_vals, test_vals, min_count):
    """Fold-safe one-hot encoding using training-fold frequencies only.

    Unlike the old helper used for outcome-model controls, this function does
    NOT require a category to occur on both state sides. State-specific levels
    are precisely the kind of information the separability diagnostic is meant
    to detect.
    """
    train_vals = [str(v or "MISSING") for v in train_vals]
    test_vals = [str(v or "MISSING") for v in test_vals]
    counts = Counter(train_vals)
    keep = {v for v, n in counts.items() if n >= min_count}
    train_vals = [v if v in keep else "OTHER" for v in train_vals]
    test_vals = [v if v in keep else "OTHER" for v in test_vals]
    levels = sorted(set(train_vals))
    if len(levels) <= 1:
        return np.zeros((len(train_vals), 0)), np.zeros((len(test_vals), 0))
    # Drop the first training-fold level as the reference. Unknown test levels
    # become OTHER; if OTHER was absent in training they are represented by the
    # all-zero reference vector.
    cols = levels[1:]
    x_train = np.column_stack([[1.0 if v == lev else 0.0 for v in train_vals] for lev in cols])
    x_test = np.column_stack([[1.0 if v == lev else 0.0 for v in test_vals] for lev in cols])
    return x_train, x_test


def separability_auc(rows):
    """3-fold out-of-fold AUC for predicting state side from pre-outcome data.

    Numeric scaling and categorical encoding are fitted inside each training
    fold. The outcome ratings are never used.
    """
    if len(rows) < 200:
        return math.nan
    good = [r for r in rows if all(finite(r[c]) for c in CONTROLS)]
    if len(good) < 200:
        return math.nan
    y = np.array([int(r["side_b"]) for r in good], dtype=int)
    if len(np.bincount(y)) < 2 or min(np.bincount(y)) < 40:
        return math.nan

    numeric = np.array([[r[c] for c in CONTROLS] for r in good], dtype=float)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pred = np.full(len(good), np.nan, dtype=float)

    for train_idx, test_idx in cv.split(numeric, y):
        scaler = StandardScaler().fit(numeric[train_idx])
        train_blocks = [scaler.transform(numeric[train_idx])]
        test_blocks = [scaler.transform(numeric[test_idx])]
        min_count = max(5, int(0.01 * len(train_idx)))

        for c in SEPARABILITY_CATS:
            train_vals = [good[i].get(c, "") for i in train_idx]
            test_vals = [good[i].get(c, "") for i in test_idx]
            xtr, xte = _fold_one_hot(train_vals, test_vals, min_count)
            if xtr.shape[1]:
                train_blocks.append(xtr)
                test_blocks.append(xte)

        x_train = np.column_stack(train_blocks)
        x_test = np.column_stack(test_blocks)
        model = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
        model.fit(x_train, y[train_idx])
        pred[test_idx] = model.predict_proba(x_test)[:, 1]

    return float(roc_auc_score(y, pred))


def sample_border(line, pair):
    """Return (xy, pair labels, 25-km grid labels) sampled along a border."""
    parts = list(line.geoms) if hasattr(line, "geoms") else [line]
    xy=[]; labels=[]; grids=[]; offset=0.0
    for part in parts:
        if part.length <= 0: continue
        d = np.arange(0.0, part.length + SAMPLE_STEP_M, SAMPLE_STEP_M)
        pts = shapely.line_interpolate_point(part, d)
        coords = np.column_stack((shapely.get_x(pts), shapely.get_y(pts)))
        xy.append(coords)
        labels.extend([pair]*len(coords))
        grids.extend([f"{pair}:{int((offset+di)//25000)}" for di in d])
        offset += part.length
    return np.vstack(xy), labels, grids


def read_nbi():
    rows=[]
    with open(NBI_FILE, "r", encoding="utf-8-sig", newline="", errors="replace") as f:
        rr = csv.DictReader(f, quotechar="'")
        for x in rr:
            st = FIPS_TO_STATE.get(str(x.get("STATE_CODE_001", "")).strip().zfill(2))
            if not st: continue
            lat = dms_to_decimal(x.get("LAT_016"), False)
            lon = dms_to_decimal(x.get("LONG_017"), True)
            dr, sr, ur = num(x.get("DECK_COND_058")), num(x.get("SUPERSTRUCTURE_COND_059")), num(x.get("SUBSTRUCTURE_COND_060"))
            if not (finite(lat) and finite(lon) and 20 < lat < 55 and -130 < lon < -60): continue
            if not all(finite(v) and 0 <= v <= 9 for v in (dr,sr,ur)): continue
            yb = num(x.get("YEAR_BUILT_027")); adt = num(x.get("ADT_029"))
            r = {
                "state":st, "structure_number":str(x.get("STRUCTURE_NUMBER_008", "")).strip(),
                "latitude":lat, "longitude":lon,
                "deck_rating":dr, "superstructure_rating":sr, "substructure_rating":ur,
                "lowest_rating":min(dr,sr,ur), "year_built":yb,
                "age_2025":2025-yb if finite(yb) and 1800 <= yb <= 2025 else math.nan,
                "adt":adt, "log_adt":math.log1p(max(0.0,adt)) if finite(adt) else math.nan,
                "structure_length_m":num(x.get("STRUCTURE_LEN_MT_049")),
                "max_span_m":num(x.get("MAX_SPAN_LEN_MT_048")),
                "deck_width_m":num(x.get("DECK_WIDTH_MT_052")),
                "inspection_frequency_months":num(x.get("INSPECT_FREQ_MONTHS_091")),
                "year_reconstructed":num(x.get("YEAR_RECONSTRUCTED_106")),
                "structure_kind_code":str(x.get("STRUCTURE_KIND_043A","")).strip(),
                "structure_type_code":str(x.get("STRUCTURE_TYPE_043B","")).strip(),
                "functional_class_code":str(x.get("FUNCTIONAL_CLASS_026","")).strip(),
                "design_load_code":str(x.get("DESIGN_LOAD_031","")).strip(),
                "owner_code":str(x.get("OWNER_022","")).strip(),
                "maintenance_code":str(x.get("MAINTENANCE_021","")).strip(),
                "highway_system_code":str(x.get("HIGHWAY_SYSTEM_104","")).strip(),
            }
            rows.append(r)
    return rows


def main():
    for p in [NBI_FILE, STATE_ZIP]:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}. See data/README.md")
    (ROOT/"data/processed").mkdir(parents=True, exist_ok=True)
    (ROOT/"results/generated").mkdir(parents=True, exist_ok=True)

    bridges = read_nbi()
    states = gpd.read_file("zip://"+str(STATE_ZIP)).to_crs(5070)
    states = states[states["STUSPS"].isin(set(FIPS_TO_STATE.values()))]
    geom = {r.STUSPS:r.geometry for _,r in states.iterrows()}

    # Shared state borders.
    borders_dict={}
    stlist=sorted(geom)
    for i,a in enumerate(stlist):
        for b in stlist[i+1:]:
            inter = geom[a].boundary.intersection(geom[b].boundary)
            if not inter.is_empty and inter.length > 1000:
                borders_dict[f"{a}-{b}"]=(a,b,inter)

    # Sample border lines and create one nearest-border tree per state.
    by_state_samples=defaultdict(lambda: {"xy":[],"pair":[],"grid":[]})
    for pair,(a,b,line) in borders_dict.items():
        xy, labs, grids = sample_border(line, pair)
        for st in (a,b):
            by_state_samples[st]["xy"].append(xy)
            by_state_samples[st]["pair"].extend(labs)
            by_state_samples[st]["grid"].extend(grids)
    trees={}
    for st,d in by_state_samples.items():
        xy=np.vstack(d["xy"])
        trees[st]=(cKDTree(xy), xy, np.array(d["pair"],object), np.array(d["grid"],object))

    # Project bridge coordinates and assign each to its nearest border involving its state.
    tr=Transformer.from_crs(4326,5070,always_xy=True)
    by_state=defaultdict(list)
    for r in bridges: by_state[r["state"]].append(r)
    prox=[]
    for st,rs in by_state.items():
        if st not in trees: continue
        xs,ys=tr.transform([r["longitude"] for r in rs],[r["latitude"] for r in rs])
        pts=np.column_stack((xs,ys)); tree,_,pair_labels,grid_labels=trees[st]
        dist,idx=tree.query(pts,k=1)
        for r,x,y,d,j in zip(rs,xs,ys,dist,idx):
            miles=float(d/M_PER_MILE)
            if miles > 25.0: continue
            pair=str(pair_labels[j]); a,b,_=borders_dict[pair]
            q=dict(r)
            q.update({
                "border_pair":pair,"state_a":a,"state_b":b,"side_b":1 if st==b else 0,
                "x_5070":float(x),"y_5070":float(y),
                "distance_miles_screening":miles,
                "signed_distance_miles_screening":miles if st==b else -miles,
                "nearest_border_grid_25km":str(grid_labels[j]),
            })
            prox.append(q)

    # Screening metrics.
    pair_rows=[]
    for pair,(a,b,line) in sorted(borders_dict.items()):
        pr=[r for r in prox if r["border_pair"]==pair]
        for bw in BANDWIDTHS:
            sub=[r for r in pr if r["distance_miles_screening"]<=bw]
            aa=[r for r in sub if r["state"]==a]; bb=[r for r in sub if r["state"]==b]
            smds=[]
            for c in CONTROLS:
                smds.append(abs(pooled_smd([r[c] for r in aa],[r[c] for r in bb])))
            cvs=[cramers_v([r[c] for r in sub],[r["side_b"] for r in sub]) for c in CORE_CATS] if sub else [math.nan]*len(CORE_CATS)
            grids_a={r["nearest_border_grid_25km"] for r in aa}; grids_b={r["nearest_border_grid_25km"] for r in bb}
            denom=min(len(grids_a),len(grids_b))
            coverage=(len(grids_a & grids_b)/denom) if denom else 0.0
            valid_smds=[x for x in smds if finite(x)]
            valid_cvs=[x for x in cvs if finite(x)]
            row={
                "border_pair":pair,"state_a":a,"state_b":b,"bandwidth_miles":bw,
                "n_a":len(aa),"n_b":len(bb),"total_n":len(sub),"min_side_n":min(len(aa),len(bb)),
                "mean_abs_smd":float(np.mean(valid_smds)) if valid_smds else math.nan,
                "max_abs_smd":float(np.max(valid_smds)) if valid_smds else math.nan,
                "core_cat_mean_v":float(np.mean(valid_cvs)) if valid_cvs else math.nan,
                "core_cat_max_v":float(np.max(valid_cvs)) if valid_cvs else math.nan,
                "min_side_cell_coverage":coverage,
                "border_length_km":float(line.length/1000),
            }
            if bw==10:
                row["separability_auc_10mi"]=separability_auc(sub)
                auc=row["separability_auc_10mi"]
                # Transparent 0-100 screening score. Eligibility below is what governs inclusion.
                sample_score=min(1.0,row["min_side_n"]/1000.0)
                bal_score=max(0.0,1-min(row["mean_abs_smd"],0.6)/0.6) if finite(row["mean_abs_smd"]) else 0
                cat_score=max(0.0,1-min(row["core_cat_mean_v"],0.8)/0.8) if finite(row["core_cat_mean_v"]) else 0
                auc_score=max(0.0,1-max(0.0,(auc-0.5))/0.5) if finite(auc) else 0
                row["screening_score"]=100*(.20*sample_score+.25*bal_score+.20*cat_score+.20*auc_score+.15*coverage)
                high=(row["min_side_n"]>=200 and row["mean_abs_smd"]<=.25 and auc<=.90 and row["core_cat_mean_v"]<=.50 and coverage>=.70)
                secondary=(row["min_side_n"]>=100 and row["mean_abs_smd"]<=.35 and auc<=.93 and row["core_cat_mean_v"]<=.60 and coverage>=.50)
                if high: row["priority"]="High priority"
                elif secondary: row["priority"]="Secondary candidate"
                elif row["min_side_n"]<100: row["priority"]="Insufficient 10-mi sample"
                else: row["priority"]="Weak overlap/balance"
            pair_rows.append(row)

    # Carry 10-mi priority/score into 5/25 rows for convenience.
    priority={r["border_pair"]:(r.get("priority",""),r.get("screening_score",math.nan),r.get("separability_auc_10mi",math.nan)) for r in pair_rows if r["bandwidth_miles"]==10}
    for r in pair_rows:
        p,s,a=priority[r["border_pair"]]; r["priority"]=p; r["screening_score"]=s; r["separability_auc_10mi"]=a

    # Output bridge file.
    fields=list(prox[0].keys())
    with gzip.open(OUT_BRIDGES,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(prox)

    # Output all screening metrics.
    fields2=list(pair_rows[0].keys())
    with open(OUT_ALL,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields2); w.writeheader(); w.writerows(pair_rows)

    high=[r for r in pair_rows if r["bandwidth_miles"]==10 and r["priority"]=="High priority"]
    high.sort(key=lambda r:r["screening_score"], reverse=True)
    with open(OUT_HIGH,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields2); w.writeheader(); w.writerows(high)


    method_meta={
        "primary_bandwidth_miles":10,
        "numeric_balance_variables":CONTROLS,
        "core_categorical_balance_variables":CORE_CATS,
        "separability_variables":{"numeric":CONTROLS,"categorical":SEPARABILITY_CATS},
        "separability_cv":{"folds":3,"shuffle":True,"random_state":42,"foldwise_preprocessing":True},
        "priority_thresholds":{
            "high":{"min_side_n":200,"mean_abs_smd_max":0.25,"auc_max":0.90,"core_cat_mean_v_max":0.50,"coverage_min":0.70},
            "secondary":{"min_side_n":100,"mean_abs_smd_max":0.35,"auc_max":0.93,"core_cat_mean_v_max":0.60,"coverage_min":0.50}
        },
        "score_weights":{"sample":0.20,"numeric_balance":0.25,"categorical_similarity":0.20,"separability":0.20,"geographic_coverage":0.15},
        "outcome_used_in_screening":False
    }
    with open(ROOT/"results/generated/screening_method.json","w",encoding="utf-8") as f:
        import json
        json.dump(method_meta,f,indent=2)

    print(f"Conventional bridges retained: {len(bridges):,}")
    print(f"Shared borders: {len(borders_dict)}")
    print(f"Bridges within 25 mi of assigned border: {len(prox):,}")
    print(f"High-priority borders: {len(high)}")
    for r in high[:20]: print(r["border_pair"], round(r["screening_score"],1), r["n_a"], r["n_b"])

if __name__ == "__main__":
    main()
