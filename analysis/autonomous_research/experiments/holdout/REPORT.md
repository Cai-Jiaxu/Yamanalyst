# Holdout paired evaluation — Metrica Sample Game 3 (final)

Date 2026-10-25. One-time fresh-external-match evidence, run once with frozen decoder params,
no retune. E1 was already verified negative on dev bench (0.658 vs v5 0.661); this is
independent fresh-match confirmation, not a model-selection run.

## Inputs (hashes)
- Holdout `cache/ar_holdout/ext_g3.npz` sha `aa1589d7…a23d7` (143,761 frames, 22 players,
  event-derived controlled-possession labels, native ball-seen 67.9%).
- Baseline `cache/models_final_v1.joblib` sha `cd1bf353…39d4c` — 3 folds, HistGBDT,
  600 iters, 226 features, classes [0,1,2]=H,A,D.
- E1 `cache/ar_e1_model_fold{0,1,2}.joblib` shas:
  fold0 `fc6d9076…b321`, fold1 `eeea564a…7ddc`, fold2 `c9a2db08…78d9d` — 3 folds,
  HistGBDT, 226 features, classes [0,1,2].
- Decoder frozen v5: w_shape 0.75, cost_team 12, cost_dead 12, min_dead 25, kickoff=half,
  stop_gate=100, w_bound=0.
- Stress mask: Bernoulli p=0.41, seed 20261025 (stress, NOT realistic replication; clean is
  primary).

## Results (spellF1 / macroF1)

| condition | baseline v5 | E1 | delta (E1 - base) |
|---|---|---|---|
| clean (primary) | 0.6128 / 0.8764 | 0.6128 / 0.8770 | +0.0000 / +0.0006 |
| stress noise   | 0.5997 / 0.8690 | 0.5972 / 0.8699 | -0.0025 / +0.0009 |

## Interpretation
- On the fresh Metrica-convention match, E1 does not beat the baseline: clean spellF1 tied,
  stress slightly down. This agrees with the dev bench (0.658 vs 0.661). No promotion.
- This is ONE fresh match with no confidence interval; the small deltas (spellF1 +0.0000 /
  -0.0025, macroF1 +0.0006 / +0.0009) are not statistically distinguishable and are reported
  as descriptive only, not as significance.
- Best remains frozen final_v5. The holdout does not find a hidden E1 gain.

## Limitations (must be stated)
- Labels are event-derived (controlled possession), NOT direct per-frame GT; event-to-frame
  boundary convention error is baked in. Numbers are comparable to the ext "metrica" fold,
  not to our match's missing-ball regime.
- Holdout ball-seen is 67.9% native; the stress mask (Bernoulli, no block gaps) lands final
  seen ~40%, which is a stress test, not a realistic copy of our ~59% seen.
- 1 match, no bootstrap CI; small deltas are descriptive, not significant.
- E1 training excluded Game 3 (confirmed by producer). Full-fixture generalisation to the
  graded target is NOT claimed.

## Artifacts
- `analysis/autonomous_research/experiments/holdout/run_holdout.py` (eval script)
- `analysis/autonomous_research/experiments/holdout/run_holdout_console.txt` (output)
- `analysis/autonomous_research/experiments/holdout/FREEZE.md` (freeze + blocker log)
