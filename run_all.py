"""Run the complete manuscript analysis in order."""
from pathlib import Path
import shutil, subprocess, sys

ROOT = Path(__file__).resolve().parent
required = [
    ROOT / "data/raw/2025HwyBridgesDelimitedAllStates.txt",
    ROOT / "data/raw/tl_2025_us_state.zip",
    ROOT / "data/external/NOAA_nClimDiv_1991_2020_County_Climate_KS_MO.csv",
]
missing = [p for p in required if not p.exists()]
if missing:
    print("Missing required data files:")
    for p in missing: print(" -", p.relative_to(ROOT))
    print("See data/README.md")
    raise SystemExit(1)

# A full rerun starts from a clean generated-results directory. Frozen results
# are never touched here.
gen=ROOT/'results/generated'
if gen.exists(): shutil.rmtree(gen)
gen.mkdir(parents=True,exist_ok=True)

scripts = [
    "00_record_input_manifest.py",
    "01_screen_borders.py",
    "02_exact_top5_rd.py",
    "03_overlap_weighting.py",
    "04_deep_dive.py",
    "05_final_robustness.py",
    "06_climate_sensitivity.py",
    "07_validate_outputs.py",
]
for i, script in enumerate(scripts, 1):
    print(f"\n[{i}/{len(scripts)}] {script}")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)
print("\nAnalysis and consistency checks complete. Review results/generated/.\n"
      "Only after reconciling those values with the manuscript, run "
      "python scripts/08_freeze_results.py.")
