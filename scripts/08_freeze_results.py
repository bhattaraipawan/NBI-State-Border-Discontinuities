"""Copy reviewed, small audit outputs into results/frozen and hash them."""
from pathlib import Path
import hashlib, shutil

ROOT=Path(__file__).resolve().parents[1]
GEN=ROOT/'results/generated'; FROZEN=ROOT/'results/frozen'
REPORT=GEN/'reproducibility_report.json'
if not REPORT.exists(): raise SystemExit('Run python scripts/07_validate_outputs.py first.')
import json
rep=json.loads(REPORT.read_text())
if not rep.get('passed'): raise SystemExit('Reproducibility validation did not pass; refusing to freeze.')
files=[
    'input_manifest.json','reproducibility_report.json',
    'National_2025_NBI_High_Priority_Borders.csv','National_2025_NBI_Border_Screening.csv','screening_method.json',
    'Top5_Exact_RD_Results.json','KS_MO_Confounder_Matching_Results.json',
    'KS_MO_Strict_Matched_Pairs.csv','KS_MO_Model_Results.csv',
    'KS_MO_Final_Robustness_Results.json','KS_MO_Climate_Sensitivity_Results.json',
    'KS_MO_Climate_Model_Results.csv'
]
FROZEN.mkdir(parents=True,exist_ok=True)
for p in FROZEN.iterdir():
    if p.is_file() and p.name!='README.md': p.unlink()
for name in files:
    src=GEN/name
    if src.exists(): shutil.copy2(src,FROZEN/name)

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
with (FROZEN/'SHA256SUMS.txt').open('w',encoding='utf-8') as f:
    for p in sorted(FROZEN.iterdir()):
        if p.is_file() and p.name not in {'SHA256SUMS.txt','README.md'}:
            f.write(f'{sha(p)}  {p.name}\n')
print(f'Frozen reviewed audit files to {FROZEN}')
