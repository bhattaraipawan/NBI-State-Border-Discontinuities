"""Write hashes and provenance for the exact raw inputs used in a rerun."""
from pathlib import Path
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/generated/input_manifest.json'
FILES=[
    ('FHWA 2025 NBI',ROOT/'data/raw/2025HwyBridgesDelimitedAllStates.txt','https://www.fhwa.dot.gov/bridge/nbi/ascii2025.cfm'),
    ('U.S. Census 2025 TIGER/Line States',ROOT/'data/raw/tl_2025_us_state.zip','https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html'),
    ('NOAA nClimDiv county climate subset',ROOT/'data/external/NOAA_nClimDiv_1991_2020_County_Climate_KS_MO.csv','https://doi.org/10.7289/V5M32STR'),
]

def sha256(path,chunk=1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

items=[]
for label,path,url in FILES:
    if not path.exists(): raise SystemExit(f'Missing input: {path.relative_to(ROOT)}')
    items.append({'label':label,'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha256(path),'source':url})
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps({'inputs':items},indent=2)+'\n')
print(f'Wrote {OUT}')
