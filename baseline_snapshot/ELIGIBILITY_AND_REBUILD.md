# Snapshot addendum: eligibility, GT provenance, rebuild distinction

Date: 2026-10-26. Companion to `OPERATING_INTERFACE.md`. All passages cited are
from the organiser's `challenge/` docs, not from this repo.

## 1. Fresh rebuild vs cached-P reconstruction (two different things)

| | cached-P bench reconstruction | fresh end-to-end rebuild |
|---|---|---|
| script | `baseline_snapshot/reconstruct_baseline.py` | `run.py --name snapshot_fresh_rebuild` |
| what it does | loads `cache/P_R1.npz` (already-fit fold probabilities), runs the v5 decoder, scores on ext/masked/public/val | refits HistGBDT on all 9 external matches + mirrors + noise copies, computes features on our match, decodes, writes a 199319-row CSV |
| touches training? | no | yes (deterministic, seed 0) |
| output | `baseline_snapshot/baseline_metrics.json` (numbers only) | `evaluation/results/snapshot_fresh_rebuild.csv` (since moved to `_archive/evaluation_results/duplicates_of_final_v5/`; result in `fresh_rebuild.log`) |
| wall time | 40 s warm caches | ~11 min warm caches, 30-60 min cold |
| expected CSV sha | n/a | `4089da237ca8780a...` (byte-identical to final_v5 if deterministic) |
| expected public score | n/a | spellF1 0.722581, macroF1 0.909848 |

The cached-P numbers (bench 0.661, ext 0.616, masked 0.706) are fold-ensemble
dev numbers. The fresh rebuild CSV, when scored by the official evaluator, must
land on the full-fit public number 0.722581. These are not the same number and
must not be conflated.

## 2. GT provenance for each yardstick

| yardstick | truth source | frames / possessions | is it independent? |
|---|---|---|---|
| `public` | `evaluation/gt/BAR-ATH_possession_gt_public.csv` (organiser public GT, derived mechanically from corrected tracking per CHALLENGE.md) | 7500 graded frames (11185..18684), 17 possessions | worked example, never tune on it |
| `masked` | same public GT as the truth; ball input hidden in 4-40 s clumpy gaps, 16 seeds 0..15 | same 17 possessions, 16 ball-mask draws | NOT independent: sd 0.024 is ball-placement noise, not possession variance |
| `ext` | true per-frame labels from IDSSE (7 Bundesliga matches, CC BY 4.0, Bassek et al.) and Metrica sample games 1 and 2 | ~3000 true possessions across 9 other fixtures | independent matches, but external label conventions differ |
| `val_h2` | hand-labelled by a person into `work/labels/val_h2_{122500,137500,167500,182500}.csv` (ROADMAP R3, allowed for validation only) | 30000 frames across 4 five-min late-2nd-half windows, 57 live spells | independent in-domain, but eye labels; timing convention was shown to reward rule boundaries (INVESTIGATION §20) |
| `private` (organiser only) | full-match corrected tracking, 158857 graded frames, ~280 possessions | unknown | the only true generalization test, invisible until submission |

Important: `masked` reuses the same 17 possessions as `public`. The masked yardstick
is a robustness check on ball visibility, not a second independent match. Treat masked
moves under 0.02 as noise.

## 3. Temporal / future-context eligibility (from primary docs, with caveats)

Primary-doc passages:
- CHALLENGE.md: "Given one full football match... produce one label for every frame."
- DATA.md: "A system built only on the ball track is as legitimate as one that reads
  the video. There is no intended solution."
- RULES.md: "Any method at all... nothing is off the table."
- RULES.md: "You may not use the private ground truth, or anything derived from it,
  beyond the 5 minutes published in this repository."
- RULES.md: "The labels must be produced by an algorithm." Hand-labelling submitted
  frames is forbidden. Small hand labels for training/validation are allowed.
- FAQ: 2nd half is graded from frame 110808 but kicks off at 113742; those 2934
  pre-kickoff frames are graded and all D. This is published structural info.

Inference (NOT an explicit grant):
- The docs describe offline full-match inference and impose no streaming or
  real-time constraint. Read literally, the whole match is available at inference.
- However, the docs do NOT explicitly authorise looking at future frames relative
  to the frame being labelled. The offline description is consistent with both
  causal and non-causal feature sets. Treat future-context features as an
  unresolved ambiguity, not as cleared.
- Inherited disclosure: the shipped `feats.full_plus3` already uses symmetric
  +-1/3/5/10 s shape windows (it looks at future frames). That is how the
  production model was built; it is a disclosure of current behaviour, not a
  green light for new future-looking features.

Guidance for new candidates:
- Prefer CAUSAL features (look backward only, or same-frame). If a future-looking
  feature is proposed, flag it explicitly in the experiment note and do not
  silently add it.
- A feature must not use any frame's label (public or private). Features must be
  functions of tracking, video, audio, or other inputs, not of ground truth.
- The public window must not be used for tuning. ext (other matches) is the
  cross-match gate with shifted label/source domain; masked is a ball-robustness
  proxy that reuses the same 17 public possessions; public is sanity only.

Audio eligibility (from primary docs):
- The tactical mp4 carries an AAC track. Local extraction and algorithmic use
  (whistle detection, crowd noise) is consistent with RULES.md "your own
  event-recognition models" and "any method". The T&Cs (FAQ line 99) only prohibit
  sending video/datasets to hosted/cloud LLMs; local processing is fine.
- Audio must be explained in the technical document. Do not redistribute audio or
  frame-level exports (RULES.md lines 77-81).

Audio training limitation (important):
- External training matches (IDSSE, Metrica) do not ship audio in this repo. If audio
  columns are added as tree features, they will be all-NaN for every external training
  frame. HistGBDT routes NaN natively but will never split on an all-NaN column.
  So audio cannot contribute as a tree feature. It can only be used as:
  (a) a decoder-side rule channel (like `mtah/rules.py` touch/dead-ball state
      machine), e.g. whistle peaks -> D boundaries,
  (b) post-processing on probabilities,
  (c) a standalone feature in a separate model trained only on our match with
      cross-validation by time (but that is high leak risk).
- If audio is added as a NaN-filled tree column, it is a silent no-op, not a leak.

## 4. Threshold caveat (predeclared, not a significance claim)

The +0.005 bench / -0.005 ext / -0.005 masked adoption gate is a development
practical criterion calibrated by the team over dozens of variants (ROADMAP rule 1).
It is NOT:
- a statistical significance test,
- a guarantee that a passing feature will improve the private full-match score.

What the yardsticks actually are:
- `ext` is a cross-match holdout (9 other fixtures), but with shifted label
  conventions (IDSSE vs Metrica vs our corrected-tracking GT) and shifted source
  domain (no tactical camera, ATD degraded to our noise). It measures transfer, not
  in-domain accuracy.
- `masked` reuses the same 17 public possessions with 16 ball-mask seeds; its
  sd 0.024 is ball-placement noise, not possession-count variance.
- `val_h2` is independent in-domain (57 spells across 20 min of the late 2nd half)
  but uses hand labels whose timing convention rewards rule boundaries
  (INVESTIGATION §20).
- The private full-match score is in-domain and final, but it is invisible until
  submission. It is one of several generalization surfaces, not the only one; ext
  is the cross-match surface, val_h2 is the in-domain hand-labelled surface.

A feature that passes the gate may still fail on the private match. The 14 Nov code
freeze and 17 Nov submission are the final evidence.

## 5. Version facts audit (preserved hashes)

Software:
- Python 3.11.15 (Clang 22.1.3)
- numpy 2.1.3, pandas 2.2.3, scipy 1.14.1, scikit-learn 1.5.2, kloppy 3.18.0,
  joblib 1.6.0, torch 2.5.1 (unused in shipped pipeline)
- pinned in `pyproject.toml`; install with `uv sync`

Model config:
- Feature fn: `mtah.feats.full_plus3`
- Classifier: `sklearn.ensemble.HistGradientBoostingClassifier`
- `MODEL_DEFAULTS` (bench): max_iter 300, learning_rate 0.05, max_leaf_nodes 31,
  min_samples_leaf 100, l2_regularization 1.0, early_stopping False, random_state 0
- Production override: `MODEL = {"max_iter": 600}` (all other defaults)
- Training frame thinning: `STEP = 5` (every 5th graded frame)
- External degradation version: `EXT_NOISE_VERSION = "v4"`
- Bench cache version: `BENCH_VERSION = "v3"`
- FOLDS (non-random, by match):
  - fold0: idsse_J03WPY, idsse_J03WN1, idsse_J03WMX
  - fold1: idsse_J03WOH, idsse_J03WQQ, idsse_J03WOY
  - fold2: idsse_J03WR9, metrica_1, metrica_2
- Mask seeds: 0..15 (16 draws)
- Public window: W0=11185, W1=18685 (slice end exclusive; real labels 11185..18684)
- Decoder: w_shape 0.75, cost_dead 12.0, cost_team 12.0, min_dead 25,
  kickoff="half", use_rules=False, stop_gate=100, w_bound=0.0

Cache hashes (sha256 first 16 hex):
- `cache/P_R1.npz` (cached fold probabilities used by reconstruction): `5c742d750d4b0e39`
- `cache/P_R1_folds.npz`: `fcdf0020149a8854`
- `evaluation/results/final_v5.csv`: `4089da237ca8780a...` (full hash in OPERATING_INTERFACE.md)
- External training feature caches: `cache/final_<match>_full_plus3_v4.npz` (9 files)
- External bench caches: `cache/bench_<match>_v3.npz` (9 files)

## 6. What the fresh rebuild must show when it completes

When `run.py --name snapshot_fresh_rebuild` finishes, capture:
- exit code
- wall time (the `time` output)
- sha256 of `evaluation/results/snapshot_fresh_rebuild.csv` (must equal
  `4089da237ca8780a...` for determinism)
- official evaluator score for that file (must equal final_v5: spellF1 0.722581,
  macroF1 0.909848)

If the sha differs from final_v5, the pipeline is not deterministic on this machine
and that is a blocker to report, not to hide.
