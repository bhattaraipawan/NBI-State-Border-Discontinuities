# Final Audit Report

**Status:** Final manuscript workflow validated on 2026-09-27.

- National screen: 108 contiguous-state borders; 13 high-priority borders.
- Final top five: NJ-PA, KS-MO, IN-KY, ID-MT, and OR-WA.
- Detailed case: Kansas-Missouri only.
- Validator: the frozen copy at `results/frozen/reproducibility_report.json` records `"passed": true`.
- Final KS-MO bridge/traffic estimate: +0.42498 rating points; 95% CI +0.33659 to +0.51336 using t(17) inference.
- Winter-climate-adjusted KS-MO estimate: +0.35468; 95% CI +0.21679 to +0.49257.
- Wild-cluster sign bootstrap: 50,000 draws, seed 42, p = 0.00004.
- Exposure-tight match: 67 pairs; +0.53731; 95% CI +0.31534 to +0.75928.

A final cross-check matched 47 manuscript numerical/table entries to the frozen outputs. All six manuscript figures were regenerated with `09_make_manuscript_figures.py` and matched the submitted figure PDFs exactly when rendered at 150 dpi. Input hashes are recorded in `results/frozen/input_manifest.json`, and frozen result hashes are listed in `results/frozen/SHA256SUMS.txt`.
