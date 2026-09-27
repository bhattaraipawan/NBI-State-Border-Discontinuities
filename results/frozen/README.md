# Frozen manuscript results

These are the reviewed v1.0.0 manuscript-facing outputs frozen after the final validation passed on 2026-09-27.

`SHA256SUMS.txt` records the hashes of the frozen result files. `input_manifest.json` records the hashes and provenance of the three analysis inputs.

Do not edit these files manually. To reproduce them, run `run_all.py`, verify `results/generated/reproducibility_report.json`, and then run `scripts/08_freeze_results.py`.
