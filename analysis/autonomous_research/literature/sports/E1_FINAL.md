# E1 Convention Duel-Reweighting — Final Record

## Hypothesis
IDSSE training frames where the ball is geometrically contested (seen AND >=1 home AND >=1 away
player within 2 m) embody a brief-touch possession convention that disagrees with our stricter GT.
Down-weighting those frames should align training with our convention. This is our own geometric
hypothesis, NOT a rule reproduced from Socceraction (see integrity correction below).

## Screen (fast, 300 trees, no mirrors/noise)
| w_duel | bench | ext | masked | public |
|--------|-------|-----|--------|--------|
| 1.00   | 0.569 | 0.589 | 0.549 | 0.606 |
| 0.50   | 0.589 | 0.596 | 0.582 | 0.630 |
| 0.20   | 0.585 | 0.593 | 0.576 | 0.576 |

Screen suggested w=0.5 was optimal (+0.020 bench).

## Full recipe (600 trees, mirrors + 3 noise draws, stop_gate=100)
Official independent score: bench 0.658 / ext 0.613 / masked 0.703.
Baseline (frozen v5): bench 0.6612 / ext 0.616 / masked 0.706.

Delta: bench -0.003, ext -0.003, masked -0.003. The screen +0.020 did NOT transfer to the full
recipe. This matches the earlier R17 pattern (screen +0.040 shrank to +0.001). The reason is not
established: it could be development-label overfitting on repeatedly-evaluated labels, or an
interaction with the augmentation/mirror/noise pipeline. State: the gain did not transfer; the
root cause is not proven.

## Verdict
E1 does NOT pass the adoption gate (needs bench +>=0.005, no ext/masked regression). Result is
flat/slightly negative. Production v5 remains the best. No promotion.

## Artifacts (absolute paths)
- Models: cache/ar_e1_model_fold{0,1,2}.joblib (6.9 MB each, HGB 600, 226 features, classes [0,1,2])
- Results: cache/ar_e1_full_results.json
- Script: analysis/autonomous_research/experiments/convention/e1_full.py
- Screen: analysis/autonomous_research/experiments/convention/e1_screen.py

## Failures recorded (not hidden)
1. String-label permutation: first full run passed string labels to sklearn (alphabetical [A,D,H]
   instead of [H=0,A=1,D=2]), permuting probability columns -> bench 0.014. Models and checkpoint
   were deleted before discovery; this is an execution failure, not a negative method result.
2. X-label KeyError: label conversion crashed on 'X' frames before subsampling; fixed by converting
   only masked (non-X) frames, matching production _fit.
3. Key mismatch: extP used "ext#k" keys but bench.score() expects bare match names; fixed.
4. My quick rescore's ext/masked aggregation differs from the official score; official .658/.613/.703
   is authoritative.

## Integrity correction
Socceraction's `player_possession_time` is a same-player + previous-action-was-dribble event
counter with NO geometry. E1's geometric duel mask is our own hypothesis. See
analysis/autonomous_research/literature/integrity/FAMILY5_integrity_review.md.

## Holdout
MetricaGame3 (g3) is reserved for a single paired v5/E1 holdout run after parameters frozen.
No holdout tuning. The E1 models above are preserved for that comparison.
