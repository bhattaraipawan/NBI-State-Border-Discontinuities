"""Run lightweight consistency checks after a full analysis rerun."""
from pathlib import Path
import csv, json, math, hashlib

ROOT=Path(__file__).resolve().parents[1]
GEN=ROOT/'results/generated'
OUT=GEN/'reproducibility_report.json'
required=[
    GEN/'input_manifest.json',
    GEN/'National_2025_NBI_Border_Screening.csv',
    GEN/'National_2025_NBI_High_Priority_Borders.csv',
    GEN/'screening_method.json',
    GEN/'Top5_Exact_RD_Results.json',
    GEN/'KS_MO_Confounder_Matching_Results.json',
    GEN/'KS_MO_Strict_Matched_Pairs.csv',
    GEN/'KS_MO_Final_Robustness_Results.json',
    GEN/'KS_MO_Climate_Sensitivity_Results.json',
    GEN/'KS_MO_Climate_Model_Results.csv',
]
missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
checks=[]
checks.append({'check':'required_files_present','passed':not missing,'details':missing})

if (GEN/'KS_MO_Final_Robustness_Results.json').exists():
    r=json.loads((GEN/'KS_MO_Final_Robustness_Results.json').read_text())
    # Placebos must record the actually used covariance rule, not a hard-coded label.
    bad=[x for x in r.get('placebo',[]) if x.get('clusters',0)<10 and x.get('covariance')!='HC3']
    checks.append({'check':'placebo_hc3_below_10_clusters','passed':not bad,'details':bad})
    for wb in r.get('wild_cluster',[]):
        g=int(wb.get('clusters',0)); method=wb.get('method'); patterns=int(wb.get('patterns',0))
        if g<=10:
            ok=(method=='exhaustive_rademacher' and patterns==2**g)
        else:
            ok=(method=='monte_carlo_rademacher' and patterns==50000)
        checks.append({'check':f"wild_bootstrap_rule_{wb.get('border_pair','unknown')}",'passed':ok,'details':wb})

if (GEN/'National_2025_NBI_High_Priority_Borders.csv').exists():
    with (GEN/'National_2025_NBI_High_Priority_Borders.csv').open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r.get('screening_score','nan')), reverse=True)
    checks.append({'check':'at_least_five_high_priority_borders','passed':len(rows)>=5,'details':{'count':len(rows)}})
    screening_top5=[r['border_pair'] for r in rows[:5]]

    exact_path=GEN/'Top5_Exact_RD_Results.json'
    if exact_path.exists():
        exact=json.loads(exact_path.read_text())
        recorded=exact.get('selection',{}).get('screening_top5',[])
        checks.append({'check':'exact_top5_matches_screening_top5','passed':recorded==screening_top5,
                       'details':{'screening_top5':screening_top5,'exact_recorded_top5':recorded}})
        analysis_pairs=exact.get('selection',{}).get('analysis_pairs',[])
        checks.append({'check':'exact_analysis_contains_only_screening_top5',
                       'passed':analysis_pairs==screening_top5,
                       'details':{'screening_top5':screening_top5,'analysis_pairs':analysis_pairs}})
        checks.append({'check':'ksmo_available_for_lead_deep_dive',
                       'passed':'KS-MO' in screening_top5,
                       'details':{'screening_top5':screening_top5}})

report={'passed':all(c['passed'] for c in checks) and not missing,'checks':checks}
OUT.write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
print(json.dumps(report,indent=2,allow_nan=True))
if not report['passed']: raise SystemExit(2)
