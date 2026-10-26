# Followup Control Coverage Report

## Status: BLOCKED (awaiting production pipe repair)

## Smoke test (passed)
- File: analysis/autonomous_research/experiments/followup_control/smoke_test.py
- Result: cache/ar_followup_control_smoke.json
- Verify: `PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python analysis/autonomous_research/experiments/followup_control/smoke_test.py`
- PASS: classes [0,1,2], 226 features, P has all public/masked/ext keys, checkpoint saved.

## Failed experiments (recorded, not repeated)

### AR-C0: baseline226 screen (FAILED)
- File: analysis/autonomous_research/experiments/followup_control/screen.py
- Issue: KeyError 'public' at b.score() — P dict missing public/masked keys.
- Outcome: 3 folds trained (855k rows each) but models lost (no checkpoint saved).
- Status: FAILED, no results.

### AR-C1: phys228 features (NOT RUN)
- Hypothesis: +2 closing velocity features (relative ball-opponent speed/direction).
- Issue: initial attempt crashed on `has_tracking()` attribute missing.
- Status: NOT IMPLEMENTED. Queue pending.

### AR-C2: typed transition head (NOT RUN)
- Hypothesis: binary turnover gate learned from relative velocity features.
- Status: NOT IMPLEMENTED. Queue pending.

## E1 experiment (closed)
- Result: bench 0.658 / ext 0.613 / masked 0.703 vs baseline 0.661/0.616/0.706. Flat/slightly negative.
- Artifacts: cache/ar_e1_model_fold{0,1,2}.joblib
- Full record: analysis/autonomous_research/literature/sports/E1_FINAL.md

## Production v5 remains best. No new training until pipe repair.
