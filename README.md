# NBI State-Border Discontinuity

Reproducible Python workflow for the manuscript **Geographic Discontinuities in National Bridge Inventory Condition Ratings at U.S. State Borders**.

**Author:** Pawan Bhattarai

## Analysis design

The workflow screens 108 contiguous-state borders in the 2025 National Bridge Inventory using pre-outcome bridge, roadway, common-support, and geographic diagnostics. Thirteen borders satisfy the finalized high-priority criteria. The five highest-ranked borders are NJ-PA, KS-MO, IN-KY, ID-MT, and OR-WA.

Condition-rating outcomes are excluded from the national screening score and ranking. Exact analyses are run for the top five, and Kansas-Missouri is the sole detailed case for matching, robustness, placebo, few-cluster, and climate-sensitivity analyses.

## Repository structure

```text
NBI-State-Border-Discontinuities/
├── README.md
├── AUDIT_REPORT.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── run_all.py
├── data/
│   ├── README.md
│   ├── raw/
│   ├── external/
│   └── processed/
├── scripts/
│   ├── 00_record_input_manifest.py
│   ├── 01_screen_borders.py
│   ├── 02_exact_top5_rd.py
│   ├── 03_overlap_weighting.py
│   ├── 04_deep_dive.py
│   ├── 05_final_robustness.py
│   ├── 06_climate_sensitivity.py
│   ├── 07_validate_outputs.py
│   ├── 08_freeze_results.py
│   └── 09_make_manuscript_figures.py
└── results/
    ├── generated/
    └── frozen/
```

## Required data

Place these public files in `data/raw/` using the exact filenames below:

- `2025HwyBridgesDelimitedAllStates.txt` — FHWA 2025 NBI
- `tl_2025_us_state.zip` — U.S. Census Bureau 2025 TIGER/Line states

The Kansas-Missouri climate subset used in the paper is included at:

`data/external/NOAA_nClimDiv_1991_2020_County_Climate_KS_MO.csv`

Large raw and regenerated bridge-level processed files are intentionally not committed to GitHub.

## Run

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python run_all.py
```

On Windows, the final command may be run as:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

A successful run writes `"passed": true` to `results/generated/reproducibility_report.json`.

After reviewing generated values against the manuscript, freeze the manuscript-facing results with:

```bash
python scripts/08_freeze_results.py
```

Regenerate manuscript figures with:

```bash
python scripts/09_make_manuscript_figures.py
```

## Reproducibility

`results/frozen/` contains the reviewed manuscript-facing outputs and `SHA256SUMS.txt`. `input_manifest.json` records SHA-256 hashes of the three analysis inputs.

The estimates are state-associated discontinuities in reported NBI condition ratings and should not be interpreted automatically as inspector bias.

## License

Code is released under the MIT License. Public datasets retain their original terms and provenance.
