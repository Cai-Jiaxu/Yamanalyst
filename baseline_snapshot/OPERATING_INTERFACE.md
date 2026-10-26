# Baseline snapshot and operational interface

Date: 2026-10-26. Project: more-than-a-hack-2027 (MTAH 2027 Phase 1, ball possession).
This file is read-only evidence. It changes nothing in `mtah/`, `run.py`, or any
existing prediction CSV.

## 1. Task in one line

For every one of 199,319 tactical-video frames of FC Barcelona vs Athletic Club,
emit `H` (home Barcelona), `A` (away Athletic), `D` (dead ball) or `X` (abstain /
out-of-grading-window). Ranks on **spellF1** (mean possession-spell F1 over temporal
IoU 0.3/0.4/0.5/0.6/0.7, greedy one-to-one same-team matching). Secondary number is
**macroF1** over H/A/D on graded frames. 158,857 graded frames; public ground truth
covers only the first 5 minutes (frames 11185..18684, 7500 graded frames, 17
possessions).

## 2. Shipped baseline (do not overwrite)

| item | value |
|---|---|
| submitted file | `evaluation/results/final_v5.csv` |
| sha256 (full) | `4089da237ca8780a6041bad8cc78261f29e9c5f5ef9fe8d3bdd9d9a8e2951c8a` |
| public spellF1 (official) | **0.722581** |
| public macroF1 (official) | **0.909848** |
| predicted possessions on public | 14 (true 17) |
| label counts over full match | H 52668, A 33158, D 73031, X 40462 |
| deterministic reproduction | `uv run python run.py --name final_v5` (first run 30-60 min warm caches, later ~11 min) |

Byte-identical re-runs (`ar_baseline_check.csv`, `ar_verified_v5_a.csv` / `_b.csv`,
`snapshot_fresh_rebuild.csv`, all with the sha256 above) and the rejected DoG candidate
`ar_candidate_dog.csv` (sha `18fb0944...`, public spellF1 0.6857) were moved on 2026-10-26 to the
local, gitignored `_archive/evaluation_results/` so the scoreboard shows only real submissions
(docs/CLEANUP_REPORT.md). Nothing was deleted.

## 3. Official evaluator (the only number that matters)

```bash
# Board over every *.csv in evaluation/results/ against the PUBLIC 5-min GT.
.venv/bin/python evaluation/run_evaluation.py --no-detail
# Full detail per file, writes evaluation/reports/<name>.json
.venv/bin/python evaluation/run_evaluation.py
# Validate a single submission file (exact checks the scorer runs):
.venv/bin/python evaluation/validate_submission.py evaluation/results/final_v5.csv
```

- Scorer source: `evaluation/possession_metrics.py` (the spec; ~720 lines).
- Public GT: `evaluation/gt/BAR-ATH_possession_gt_public.csv` (199319 rows, real
  labels only on frames 11185..18684; `X` elsewhere, never scored).
- `--gt` takes `public` (default), `private` (organisers only), or a path to a
  `frame,label` CSV.
- A submission that fails validation is skipped, not fatal.
- On the public window alone, one missed possession moves spellF1 by ~0.06; the
  documented noise band on a full match is ~0.04 for spellF1 and ~0.01 for macroF1.
  The public 5-minute window is a sanity check, never a selection criterion.

## 4. Dev benchmark (the internal yardstick for feature work)

`mtah/bench.py` class `Bench`. Three yardsticks:

| yardstick | what it scores | sample size | role |
|---|---|---|---|
| `ext` | 9 external matches (7 IDSSE Bundesliga + Metrica 1/2), 3-fold by match, degraded to our tracker noise | ~3000 true possessions across 9 matches | **steadiest gate** |
| `masked` | our match, public 5-min window, ball hidden in 4-40 s stretches, 16 seeds | 17 true possessions, 16 ball-mask draws | proxy for the full match; sd is ball-placement noise |
| `public` | our match, public window as delivered | 17 possessions | sanity only |
| `val_h2` | 4 eye-labelled 5-min late-2nd-half windows in `work/labels/` | 57 live spells | independent in-domain check |

`bench = (ext_spellF1 + masked_spellF1) / 2`.

### Reproduced dev metrics (cached fold probabilities, no refit)

Command: `.venv/bin/python baseline_snapshot/reconstruct_baseline.py`
(writes `baseline_snapshot/baseline_metrics.json`, 40 s on warm caches).

| metric | value | documented |
|---|---|---|
| bench spellF1 | 0.6612 | 0.661 |
| ext spellF1 | 0.6163 | 0.616 |
| ext macroF1 | 0.8917 | - |
| idsse spellF1 | 0.6073 | 0.607 |
| metrica spellF1 | 0.6477 | 0.648 |
| masked spellF1 (mean over 16 seeds) | 0.7061 | 0.706 |
| masked spellF1 sd | 0.0236 | - |
| public spellF1 (fold ensemble) | 0.6968 | 0.697 |
| public macroF1 (fold ensemble) | 0.9098 | 0.910 |
| val_h2 spellF1 (fold ensemble) | 0.5627 | 0.563-0.579 |
| val_h2 macroF1 | 0.8631 | - |
| ext errors per 100 true possessions | split 2.0 / merge 10.6 / drift 19.4 / missed 22.3 / spurious 3.6 | - |
| predicted/true possession ratio (ext) | 0.707 | - |

Note: `val_h2` on the shipped full-fit CSV (`final_v5.csv`) is 0.593 (Yamanalyst
table); 0.563 above is the fold-ensemble number, which is what feature work sees.

## 5. Shipped model configuration (frozen)

Source: `mtah/final.py`.

- Feature function: `mtah.feats.full_plus3(match, f, shapeX) -> (n, k) ndarray`.
  `shapeX` is the cached ball-free team-shape matrix (`shape.compute(m, cam=False)`).
- Model: `sklearn.ensemble.HistGradientBoostingClassifier`, `max_iter=600`,
  `random_state=0`, defaults from `Bench.MODEL_DEFAULTS` (lr 0.05, max_leaf_nodes 31,
  min_samples_leaf 100, l2_regularization 1.0). Training frames thinned by
  `shape_model.STEP`.
- Training data: 9 external matches with TRUE per-frame labels, each degraded to
  our ATD noise profile (`mtah/noise.py`), plus left/right mirrors, home/away
  swaps, and 3 extra noise draws per match (augmented copies).
- Decoder (`mtah/decode.py` `decode()`), 3-state Viterbi per half, params:
  `w_shape=0.75, cost_dead=12.0, cost_team=12.0, min_dead=25, kickoff='half',
  use_rules=False, stop_gate=100, w_bound=0.0`.
- Rule labels (`mtah/rules.py`) are a FEATURE family, not a decoder vote.

## 6. How to run a feature experiment (read this before writing code)

```python
from mtah.bench import Bench
from mtah import feats

b = Bench()                       # ~24 s on warm caches; ~5 min cold
# Fast screen (ROADMAP rule 7): 9 base matches, no mirror/noise copies, 300 trees.
# Only drops clear losers; never adopts anything.
screen = b.screen("my_feature_screen", my_featfn)
# If screen does not lose by > 0.005 on bench, run the full fit:
res = b.evaluate("my_feature_full", my_featfn,
                 model_params={"max_iter": 600},
                 dec_params={"w_shape": 0.75, "cost_dead": 12.0,
                             "cost_team": 12.0, "min_dead": 25,
                             "kickoff": "half", "use_rules": False,
                             "stop_gate": 100},
                 use_rules=False)
# res["bench"], res["ext"], res["masked"], res["public"], res["val"],
# res["taxonomy"] (per-100 error counts).
```

Feature function signature (must match):
```python
def my_featfn(match, f, shapeX):
    # match: mtah.data.Match (has .n frames, .fps, .halves)
    # f: dict from mtah.features.compute(match):
    #    'seen' (bool ball visible), 'bspeed', 'sh'/'sa' (home/away speeds),
    #    on-ball distance columns, out-of-pitch flags, ...
    # shapeX: cached (n, k_s) ball-free team-shape matrix
    # returns: (n, k_new) ndarray of float features, one row per frame
    ...
```

Cache rule: `b.probs(featfn, model_params, key="my_key")` saves fold models to
`cache/models_my_key.joblib` and lets decoder experiments reuse them in seconds.
The key MUST change whenever featfn, model_params, extra augmented copies, or
relabel logic change.

## 7. Predefined gain criterion (locked before any new feature)

A new feature family is adopted into `mtah/final.py` ONLY IF ALL hold:

1. **Screen gate**: on `b.screen(...)`, bench is not worse than the current
   `full_plus3` screen by more than 0.005. Clear losers stop here.
2. **Full refit gate**: `res["bench"]` rises by **>= 0.005** versus the frozen
   0.6612, AND `res["ext"][0]` (0.6163) does NOT fall by more than 0.005, AND
   `res["masked"][0]` (0.7061) does NOT fall by more than 0.005.
3. **Public sanity**: `res["public"][0]` does not drop by more than 0.02 versus
   0.6968. A larger drop is a yellow flag requiring a `val_h2` check; it is not
   automatically fatal.
4. **No multiple-chances selection**: if several variants are tried and the best
   one wins by less than 0.01, treat it as noise and record it as "not adopted" in
   `docs/INVESTIGATION.md`. Borderline wins that needed several tries are "no".
5. **No silent changes**: do not change decoder params, thresholds, post-processing,
   or feature defaults under the same name. Any config change gets a new `key`.

Differences under 0.005 on bench are noise. The masked yardstick sd is 0.024, so
treat masked moves under 0.02 as ball-placement variation.

## 8. Eligibility and leak audit

### Allowed inputs
- Tactical ATD (players, ball, homography): `data/tactical/`.
- Broadcast ATD: `data/broadcast/` (not frame-aligned; prior integration was
  dropped because it fills only 6.5% of the tactical ball gaps).
- Smart Tagging: 638 auto phases (set pieces, build-up, transition). Prior review
  found team tags agree with true team on 77% of live frames (below model's 90%);
  set-piece tags start 3-12 s early. Use as a feature only, never as a label.
- Public 5-min GT: `evaluation/gt/BAR-ATH_possession_gt_public.csv`. Sanity only,
  never for tuning.
- External labelled matches: IDSSE (7 Bundesliga, CC BY 4.0) and Metrica sample
  games 1 and 2. Used for training. They are the ONLY source of true per-frame
  possession labels.
- Hand labels in `work/labels/` (4 val windows + 300-frame ball probe): allowed for
  VALIDATION only, never as training labels for the shipped model (R4 noisy-student
  was tried and rejected: -0.016 bench).

### Forbidden
- Private full-match ground truth (any derived form).
- Hand-labelling the submitted frames.
- Sending match data, videos, or any derived frame-level output to a hosted LLM
  (ChatGPT, Claude, Gemini, Copilot, etc.). Aggregate-only prints are fine.
- Committing `data/`, `cache/`, `external/`, `models/`, `work/`,
  `evaluation/results/*.csv`, `evaluation/reports/`, `evaluation/scoreboard.csv`.

### Temporal leak / purge rules
- External folds are by match (non-random): `FOLDS` in `mtah/bench.py`. No frame
  of a scored match ever appears in its own training fold.
- The current feature set uses symmetric ±1/3/5/10 s shape windows. Because the
  model is applied offline (not live), future-context features are acceptable as
  long as they are computed identically for training external matches and our
  match. The bench already degrades external tracking to our noise profile so the
  distribution matches.
- When adding a NEW feature that uses a lagged or future window:
  - Compute it on external matches EXACTLY as on our match (same window length,
    same missing-ball handling).
  - Do not use any future-frame label (e.g. a true possession change event) as a
    feature in training. Features must be functions of tracking only.
  - For any feature that depends on ball visibility, ensure the masked yardstick
    (`ball_mask` with 4-40 s gaps, `pace_weights`) still sees it degraded honestly;
    `Bench` recomputes `f` per seed via `compute(m, hide=...)`.
- Purging: the masked yardstick already simulates the 41% missing-ball rate with
  clumpy gaps. No additional purge window is required for new per-frame features,
  but if a feature uses a long history (>10 s) it must be re-tested with
  `masked_sd` reported explicitly.

## 9. Runtime, dependencies, seeds

- Python 3.11.15 (Clang 22.1.3).
- numpy 2.1.3, scipy 1.14.1, scikit-learn 1.5.2, joblib 1.6.0, pandas 2.2.3,
  kloppy 3.18.0. Pin set in `pyproject.toml`; install with `uv sync`.
- Random seed: model `random_state=0` everywhere. Augmentation noise is seeded
  per match name (`mtah/noise.py seed_for`). Ball-mask seeds 0..15 fixed.
- Bench construction: ~24 s on warm caches; first-ever run ~5 min.
- Full refit (`b.evaluate` with 600 trees + mirrors/noise): ~1 hour on the Mac;
  run with `caffeinate -i` and never run two refits concurrently.
- Shipped CSV regeneration: ~11 min warm, 30-60 min cold.

## 10. Existing prediction files (preserve all)

`evaluation/results/` (gitignored except the starter baselines):
- `final_v1.csv` ... `final_v5.csv` (v5 is the shipped one)
- `baseline_raw.csv`, `baseline_smooth25/50/125.csv` (published floor: smooth50
  public spellF1 0.3256, full-match 0.2433)
- `rules_v2.csv`, `fused_v2.csv`, `fused_ext_v2.csv` (older methods)

`_archive/evaluation_results/` (local only, since 2026-10-26):
- `duplicates_of_final_v5/`: `ar_baseline_check.csv`, `ar_verified_v5_a.csv`,
  `ar_verified_v5_b.csv`, `snapshot_fresh_rebuild.csv` (byte-identical to final_v5)
- `rejected_candidates/`: `ar_candidate_dog.csv` (0.6857), `ar_candidate_dog_fusion.csv`
  (broken schema, skipped by the evaluator), `research_audio_d_relabel.csv`,
  `research_hardD_A_v2.csv`, `research_hardD_B_v2.csv` (audio experiments, INVESTIGATION §26.3)

## 11. Early blockers / things to know before writing features

- The ball is missing on 41% of graded frames. Any feature that silently assumes a
  ball position exists will score well on the public window (ball seen 88-99%)
  and collapse on the masked yardstick and the private match.
- The model already reads possession from player shape when the ball is gone.
  Mid-possession team accuracy is ~90% on external data; the remaining errors are
  at possession changes (drift 19/100, missed 22/100).
- Video ball detection was tried and dropped: even a perfect ball restored in the
  gaps adds only +0.005 to bench. Do not start a video-ball track.
- CatBoost averaging was tried and dropped (-0.015 to -0.024 bench).
- Learned temporal refiners (GRU/TCN) over the trees' probabilities gain +0.011 to
  +0.017 on external folds but lose 0.02-0.03 on masked (they learn external label
  conventions). Do not repeat.
- Touch-sequence / kick features for short possessions: +0.040 on the fast screen
  but only +0.001 on the full recipe. Do not repeat as-is.
- The public window's 17 possessions are too few to select on. Use ext + masked +
  val_h2.

## 12. How to hand back a feature result

Write a short markdown note in `docs/INVESTIGATION.md` (or a new file under
`analysis/`) with: screen bench, full bench/ext/masked/public/val, error counts,
the exact `key=` used, and a clear "adopt" / "not adopt" verdict against the
criteria in section 7. Do not touch `mtah/final.py` until the verdict is adopt.
