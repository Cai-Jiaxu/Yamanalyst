# Phys228 Transition Head Model Manifest

## Model paths
- cache/ar_phys228_fold0.joblib
- cache/ar_phys228_fold1.joblib
- cache/ar_phys228_fold2.joblib

## Script
- analysis/autonomous_research/experiments/followup_control/phys228_screen.py
- Function: `phys_features(c)` computes +2 extra features

## Feature formula (228 = 226 base + 2 extra)
Extra features from existing f dict (no raw player tracking):
1. `closing_speed_margin` = gradient(f["da"] - f["dh"]) — rate of change of nearest-opp margin (units: per frame at 25 fps)
2. `ball_motion_aligned` = f["vx"] * sign(f["da"] - f["dh"]) — ball velocity aligned with nearest-opp direction

## Input sources
- All from `c.f` dict: bx, by, vx, vy, bspeed, dh, da, nh2, na2, seen
- No raw player positions used

## Missing masking
- f["seen"] flag used; NaN where ball not seen
- No hidden raw ball data

## Training labels
- Target: H↔A turnover within 6 frames, excludes D (class 2)
- Binary classes: [0, 1] (no-turnover / turnover)
- Pos_weight: default (no class_weight); pos_rate ~1.2%

## Fold map (training matches)
- Fold 0: trains on idsse_J03WOH, idsse_J03WQQ, idsse_J03WOY, idsse_J03WR9, metrica_1, metrica_2
- Fold 1: trains on idsse_J03WPY, idsse_J03WN1, idsse_J03WMX, idsse_J03WR9, metrica_1, metrica_2
- Fold 2: trains on idsse_J03WPY, idsse_J03WN1, idsse_J03WMX, idsse_J03WOH, idsse_J03WQQ, idsse_J03WOY

## Model specs
- HistGradientBoostingClassifier, 100 trees
- Classes: [0, 1]
- Features: 228
- Random state: 0

## Scoring status
- Training: COMPLETE (3 folds)
- Scoring: UNSUPPORTED (binary 2-col P vs decoder 3-col H/A/D mismatch)
- Integration: pending (use cached P_R1 as base, apply cost delta)

## Cached predictions
- None saved. Head predictions must be generated from saved models.
