# Holdout freeze — Metrica Sample Game 3 (ar_holdout)

Frozen 2026-10-25. This is an UNUSED external labelled match, held out before any model
prediction is run against it. No tuning, no model selection on it. It may later veto a
development candidate (showing it does not generalise to a fresh external-convention match),
but it must not be used to pick the next variant.

## Source (recorded, not redistributed)
- Repo: https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_3
- Files downloaded to `cache/ar_holdout/external/metrica_g3/`:
  - `events.json`      (1,709,486 bytes)
  - `metadata.xml`     (81,806 bytes)
  - `tracking.txt`     (53,443,115 bytes, 143,761 lines)
- License: repo ships no LICENSE file (404), no SPDX. Usage is permissive-by-custom,
  attribution requested, NO redistribution. Data stays local under `cache/` (gitignored).

## Why it is unused / independent
- The project's `scripts/download_external.sh` only fetches Sample Game 1 and Game 2.
  Game 3 has never been parsed into `cache/ext_*.npz`, never trained on, never evaluated.
- It is a different match (different fixture, anonymised Team A vs Team B; local=Team A).
- Caveat: labels follow the SAME Metrica event convention as the ext "metrica" fold
  (controlled possession from PASS/RECOVERY/SET PIECE/SHOT; CHALLENGE/BALL LOST do not
  switch; dead runs from BALL OUT / shot out / foul). So this tests generalisation to a
  fresh match under the Metrica convention, NOT transfer to our missing-ball regime.
  The masked yardstick remains the only missing-ball diagnostic.

## Frozen artifact
- `cache/ar_holdout/ext_g3.npz`  sha256 `aa1589d712bb2708dbbbe99017780f18929ec2bbe9d5e178281beb5e239a23d7`
- Shape: players (143761, 22, 3), ball (143761, 3), team (22,), labels (143761,),
  halves (2,3), pitch (2,). 25 fps, pitch 105x68.

## Team mapping / roster audit (aggregates only)
- Team A (FIFATMA, local) -> H (11 slots, team=0); Team B (FIFATMB, visiting) -> A (11 slots, team=1).
- Halves (0-based inclusive): (1, 0, 69660), (2, 69661, 143760) per metadata
  first_half 1..69661, second_half 69662..143761.
- Label distribution over all frames: H 36.8%, A 34.7%, D 28.3%, X 0.3%.
- Graded play frames (inside halves, not X): 143,401. Ball seen on graded: 67.9%.

## Deterministic noising / missing pattern (predeclared, STRESS test)
- Baseline ball seen on graded holdout frames: 67.9% (i.e. 32.1% already missing).
- Predeclared mask: Bernoulli p=0.41 of graded frames additionally set to NaN, seed 20261025.
  This is a STRESS-noise run, NOT a realistic replication of our target (~59% seen):
  adding 0.41 on top of the native 32.1% missing lands final seen around ~40%, and
  independent Bernoulli does not reproduce contiguous block gaps. Report it as stress
  robustness, not as a matched missing-ball diagnostic.
- The CLEAN (no-noise) holdout is the PRIMARY fresh-match assessment, paired v5 vs E1.
- Both runs frozen; no retune on either. NO target manual val, NO hidden GT used.

## Pairing recipe (when E1 is frozen)
- Baseline fold-ensemble probabilities come from `cache/P_R1.npz`, which was produced by the
  cached fold models in `cache/models_final_v1.joblib` (per `analysis/fold_probs.py` and
  `analysis/noisy_student.py`: each fold's teacher is that fold's final_v1 model).
- E1 must provide its own fold ensemble probabilities trained on the same external matches
  EXCEPT Game 3, and state the exact training size. Both sides are matched fold ensembles;
  full-fit v5 vs E1-fold would be unfair.
- Record counterpart training sizes: baseline 9 external matches (7 IDSSE + Metrica 1,2);
  E1 must confirm Game 3 excluded.

## Label caveat
- ext_g3 labels are algorithmically DERIVED from event timestamps (controlled possession),
  NOT direct per-frame ground-truth annotations. Same convention as the ext "metrica" fold,
  so numbers are comparable to that yardstick but carry event-to-frame boundary convention
  error. Independent integrity review noted.

## BLOCKER / WAIT (2026-10-25, updated)
- E1 bug: sklearn fit on string labels sorts classes alphabetically [A,D,H] instead of
  [H,A,D], permuting probability columns (full bench invalid, ~0.014). The buggy full model
  and its checkpoints were DELETED by the producer; no artifact salvage is possible. Original
  baseline unaffected. Integer-label X conversion then crashed on non-X masks and was fixed;
  a fresh full restart is running (pid 27869).
- The old `models_full_plus5.joblib` (240-dim, integer classes) is NOT the E1 salvage
  artifact and must not be scored on the holdout.
- Status = WAIT for 3 new E1 fold joblibs at `cache/ar_e1_model_fold{0,1,2}.joblib`.
- Before the one holdout run: read actual saved classes_ / n_features_in_ (must be 226) /
  fold map / sha from disk for all 3 folds, and build the explicit classes adapter even after
  the 012 safeguard. Do NOT score with an unadapted or dimensionally-mismatched candidate.
- The compute interruption is recorded as a failed run, not as solved. No duplicate fit.

## Baseline verification (structural, not filename-only)
- `cache/models_final_v1.joblib` sha256 `cd1bf353e0987d9bacd01745a699a4022c4b74bb44f28797e2f6334adfb39d4c`
- 3 folds, each HistGradientBoostingClassifier, max_iter=600, n_iter_=600,
  n_features_in_=226, loss=log_loss. These are actual fitted models.
- Feature dim confirmed 226 (full 149211x226, train 1.026Mx226). The earlier "246" was a
  guess, not a real discrepancy.

## What this does NOT change
- `mtah/external.py`, `run.py`, `mtah/final.py`, scorer untouched. Parser is isolated at
  `analysis/autonomous_research/experiments/holdout/parse_g3.py`. Game 3 stays in
  `cache/ar_holdout/` and is excluded from the 9-match training/bench.
