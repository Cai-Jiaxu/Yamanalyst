# Followup Control Model Manifest

## Transition Head (binary turnover classifier)

### Model paths
- cache/ar_followup_baseline226_fold0.joblib
- cache/ar_followup_baseline226_fold1.joblib
- cache/ar_followup_baseline226_fold2.joblib

### Specs
- Type: HistGradientBoostingClassifier, 100 trees
- Classes: [0, 1] (binary: no-turnover / turnover)
- Features: 226 (full_plus3)
- Random state: 0
- Pos_weight: default (no class_weight applied; pos_rate ~1.2%)
- Target: H↔A turnover within 6 frames, excludes D

### Fold map (training matches)
- Fold 0: trains on idsse_J03WOH, idsse_J03WQQ, idsse_J03WOY, idsse_J03WR9, metrica_1, metrica_2
- Fold 1: trains on idsse_J03WPY, idsse_J03WN1, idsse_J03WMX, idsse_J03WR9, metrica_1, metrica_2
- Fold 2: trains on idsse_J03WPY, idsse_J03WN1, idsse_J03WMX, idsse_J03WOH, idsse_J03WQQ, idsse_J03WOY

### Feature function
- mtah.feats.full_plus3 (226-dim)

### Cached predictions
- None saved yet. Head predictions on public/masked/ext must be generated from saved models.

### Baseline primary P_R1 state
- cache/P_R1_folds.npz (3-class probabilities, separate from head)

## Status
- Head trained: YES
- Cost delta integration: NOT RUN (no scores)
- λ=0 identity: verified via b.score default (bench 0.6612)
- Custom Viterbi λ=0 identity: NOT VERIFIED

## Queue (untested)
- phys228: +2 closing velocity features
- typed transition head cost delta (pending integration by baseline engineer)
