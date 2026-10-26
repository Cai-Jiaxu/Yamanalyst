# Repository / Data / Scorer Audit and Baseline Establishment

Audit date: 2026-10-25. Auditor: autonomous research sub-agent. Read-only on scorer,
baselines, `mtah/`, `run.py` and all existing files. New artifacts live only in
`analysis/autonomous_research/audit/` and the agent workspace. Nothing was pushed, pulled, or
committed.

Project root: `/Users/bytedance/Desktop/BARCA/more-than-a-hack-2027`

---

## 1. The task in one screen

- One match: FC Barcelona (H) vs Athletic Club (A), LaLiga 2024-25 matchday 2.
- Output: a CSV `frame,label`, exactly **199,319 rows**, frame 0..199318 in order. Label in
  `H` / `A` / `D` / `X`. `X` = abstain (free on ungraded frames; forward-filled under spellF1).
- Graded frames: video frames **11185..90651** (1st half) and **110808..190197** (2nd half) =
  **158,857 graded frames**. Ungraded (pre-match, halftime, post-whistle): write `X`, ignored.
  Note the 2nd half is graded from frame 110808 but kicks off at 113742: frames 110808..113741
  (2,934) are graded and are all `D`.
- **Ranking metric: `spellF1`** (possessions matched one-to-one by temporal IoU, F1 averaged
  over IoU thresholds 0.3/0.4/0.5/0.6/0.7). Tie-break: `macroF1` (unweighted per-class F1 over
  H/A/D). The scorer is the official result; no stricter private scorer exists.

## 2. Rules that route implementation work (read first)

1. **Labels must come from an algorithm.** Hand-labelling the submitted match is disqualified.
   Small hand labels for *training/validation* are allowed and already used (see `work/labels/`).
2. **No private (full-match) ground truth** beyond the published 5-minute window. It does not
   exist in this repo and must not be sought or reconstructed.
3. **The repo must regenerate the submission byte-for-byte** (`run.py` is deterministic).
4. **Data never leaves to a hosted LLM** (T&Cs): no match video, tracking, or frame-level
   exports to ChatGPT/Claude/Gemini/Copilot. Scripts print aggregates only. Keep all restricted
   outputs local and gitignored.
5. **Do not commit** `data/`, `cache/`, `external/`, `models/`, `colab_upload/`, `work/`,
   `evaluation/results/*.csv`. Repo becomes public at submission.
6. Any method is allowed if the organisers can inspect and run it; a private model/data you
   cannot share is the one thing ruled out.

## 3. Label availability (what truth we actually have)

| source | location | what it is | usable for |
|---|---|---|---|
| Public GT | `evaluation/gt/BAR-ATH_possession_gt_public.csv` | real labels, frames 11185..18684 (5 min), H 3454 / A 2577 / D 1469, **17 possessions**, mean 14.2 s. Rest X. | **pipeline sanity only, never tuning** |
| External true labels | `external/idsse/` (7 Bundesliga matches, CC BY 4.0) + `external/metrica/` (Sample Games 1-2) | per-frame possession labels for 9 matches | **training + the external half of the bench** |
| Person-labelled validation | `work/labels/val_h2_{122500,137500,167500,182500}.csv` (4 windows, ~30k frames, 57 live spells) + `ball_probe_300.csv` | by eye, GT convention; team/dead-ball calls, not exact timing | independent 2nd-half check (`val_h2`) |
| Private full-match GT | **absent** | 158,857 graded frames | the real score, only organisers run it |

Key consequence: the **only locally runnable scorer is against the public 5-minute window**. The
private full-match numbers are never observable locally; the team's proxy for them is the internal
**bench** (section 6), not the public window.

## 4. Baselines — reproduced with the official scorer

Ran the official scorer non-destructively:
`uv run python evaluation/run_evaluation.py --results <copy> --out <audit_out> --gt public --bootstrap 0 --no-detail`
(evidence: `analysis/autonomous_research/audit/reproduced_public_scoreboard.csv`, `console_board.txt`).

### 4.1 Starter / published baseline
`starter/baseline.py` = nearest tracked player to the ball, majority vote over a window.
Produces `baseline_raw / smooth25 / smooth50 / smooth125` (±0/25/50/125 frame windows).

| file (public window) | macroF1 | spellF1 | predicted poss (true 17) |
|---|---|---|---|
| baseline_raw | 0.5972 | 0.0378 | 131 |
| baseline_smooth25 | 0.6200 | 0.2868 | 36 |
| **baseline_smooth50 (published floor)** | **0.6207** | **0.3256** | 26 |
| baseline_smooth125 | 0.5812 | 0.4242 | 16 |

Published **full-match** `baseline_smooth50` = spellF1 **0.2433** (the floor to beat; per
`challenge/EVALUATION.md`). Reproduced public number 0.3256 matches the docs exactly.

### 4.2 Strongest existing credible result: `final_v5`
`uv run python run.py --name final_v5` -> `evaluation/results/final_v5.csv`.

| metric | value | how obtained |
|---|---|---|
| public spellF1 | **0.7226** | reproduced this audit (docs: 0.723) |
| public macroF1 | **0.9098** | reproduced (docs: 0.910) |
| public possessions | 14 (true 17) | scorer |
| full-match possessions | 269 | validator (669 raw spells counted; decoder-collapse to possessions) |
| **internal bench** | **0.661** | fold external 0.616 + masked public 0.706 (NOT local scorer) |
| `val_h2` (2nd-half eye windows) | 0.593 | independent validation |
| sha256 | `4089da237ca8780a6041bad8cc78261f29e9c5f5ef9fe8d3bdd9d9a8e2951c8a` | matches README/docs byte-for-byte |

Validation (`uv run python evaluation/validate_submission.py evaluation/results/final_v5.csv`):
199319 rows, frame 0..199318, H 52668 (26.4%) / A 33158 (16.6%) / D 73031 (36.6%) / X 40462 (20.3%, ungraded frames). **Valid.** (An earlier chained validator run misattributed the `baseline_smooth50` block to final_v5; corrected on 2026-10-25 reproduction, which also re-derived the label counts byte-for-byte.)

Method: hand-written touch + dead-ball rules (as features, not a decoder vote) -> gradient-boosted
trees (HistGradientBoosting, ~600 trees) trained on 9 external matches with true labels, noised
to our tracker, plus left/right and home/away mirrors and extra-noise copies -> 3-state Viterbi
decoder (H/A/D) with switch cost 12 nats, 1-s min dead ball, dead-ball-before-2nd-half-kickoff,
and a stoppage-blip gate. Ball is missing on 41% of graded frames, so the model reads possession
from ball-free team shape.

### 4.3 Other local results (public spellF1, for context)
final_v1/v2 0.700, final_v3/v4 0.653, rules_v2 0.461, fused_ext_v2 0.434, fused_v2 0.400.
`final_v3`/`v4` (rule vote in decoder) look better on eye labels but were **rejected** on real
ground truth — see contamination note 2.

## 5. Exact entry points for experiment agents

- Regenerate the submission (deterministic, ~11 min warm, 30-60 min first run):
  `uv run python run.py --name final_v5`
- Score any CSV against public GT (official):
  `uv run python evaluation/run_evaluation.py --gt public --no-detail`
  (add `--bootstrap 1000` for confidence intervals; `--results DIR` / `--out DIR` to stay
  non-destructive.)
- Validate a file before trusting it:
  `uv run python evaluation/validate_submission.py evaluation/results/<name>.csv`
- Internal benchmark (the real decision tool):
  ```python
  from mtah.bench import Bench
  from mtah import feats
  b = Bench()                                   # ~5 min first time, then cached
  b.evaluate("idea", feats.full_plus3,
             model_params={"max_iter": 600},
             dec_params={"w_shape":0.75,"cost_dead":12.0,"cost_team":12.0,"min_dead":25},
             use_rules=False)
  ```
  Reuse fitted probabilities for decoder-only work: `b.probs(featfn, key="X")` then
  `b.score(name, P, dec_params=...)`. Fast screen (drop clear losers only, never adopts):
  `b.screen(name, featfn, ...)`. See `analysis/yardsticks.py`, `analysis/decoder_sweep.py`,
  `analysis/round3.py`, `analysis/score_files.py`.

## 6. The internal "bench" and its noise floor (decision rules already set)

`bench` = mean of (a) spellF1 on 9 external matches, 3 match-folds, model trained on the other
two; and (b) our masked public-window spellF1, 16 seeds, ball hidden the way the full match loses
it (gaps biased to walking players, 30 s post-window hidden to block look-ahead).

- **Adopt only if bench rises by >= 0.005 AND neither ext nor masked falls by more than 0.005.**
  Smaller differences are noise. Full-match spellF1 noise band ~0.04; macroF1 ~0.01.
- Screen first: a candidate that loses by > 0.005 on the fast screen (9 base matches, no
  mirrors/noise, 300 trees) is dropped. The screen never adopts — mirrors (+0.039) are the
  biggest single gain and the screen omits them.
- Judge on bench, **never on the public window**.

## 7. Data fingerprints (sha256, for provenance)

| file | sha256 (prefix) |
|---|---|
| `evaluation/gt/BAR-ATH_possession_gt_public.csv` | e596a9af9626c623 |
| `evaluation/gt/..._public_meta.json` | cd8784f7c33127d9 |
| `data/tactical/BAR-ATH_tactical_atd.txt` | f130e350d2bfdf73 |
| `data/tactical/BAR-ATH_tactical_homography.csv` | 421c81abe16f0912 |
| `data/tactical/BAR-ATH_tactical_atd.xml` | 327e35e32e46a647 |
| `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | 0594d35e3df01b9c |
| `data/broadcast/BAR-ATH_broadcast_atd.txt` | a71c1c9391e56403 |
| `evaluation/results/final_v5.csv` | 4089da237ca8780a (matches docs) |

Sizes: tactical atd.txt 195 MB, homography 23 MB, tactical mp4 5.2 GB; broadcast atd.txt 142
MB, broadcast mp4 14 GB; `external/` 2.6 GB; `cache/` 24 GB (309 files); `work/labels/` 6.4 MB.
Env: Python 3.11.15, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.5.2 (via `uv sync`).

## 8. Resource inventory

- `mtah/` pipeline: `data.py` (Match loader), `features.py`, `rules.py` (touch + dead-ball state
  machine), `feats.py` (features incl. `full_plus3`, `flight_block`), `final.py` (GBT + external
  training), `decode.py` (Viterbi), `kickoff.py`, `bench.py`, `masking.py`, `noise.py`,
  `external.py`, `shape.py`/`shape_model.py`. Measured-but-unused: `video.py`, `video_ft.py`,
  `mbr.py`, `refiner.py`, `label_tool.py`.
- `cache/`: fitted fold models `models_*.joblib`, cached probabilities `P_R*.npz`,
  `ball_det_ft.npz` (video detector), `shape_proba.npy`.
- `models/`: yolo11{s,m,x}.pt (video ball detector weights).
- `external/`: `idsse/` (7 Bundesliga: J03W* meta/event/tracking) + `metrica/` (Sample Games 1-2).
- `analysis/`: 45 experiment scripts behind `docs/INVESTIGATION.md` (re-runnable, aggregates only).
- `evaluation/`: official scorer (`run_evaluation.py`, `possession_metrics.py`,
  `validate_submission.py`) — unchanged from the organiser starter.

## 9. Known evaluation contamination, traps, and uncertainty

1. **Public window is the best-tracked 5 minutes** (ball seen 88-99% there vs ~59% over the match,
   41% missing on graded frames). 17 possessions. Tuning on it has repeatedly picked the wrong
   version (the smoothing-window sweep is the organiser's worked example). Use it only to confirm
   the pipeline is aligned.
2. **Eye-labelled `val_h2` windows reward the rules' boundary timing.** On real ground truth the
   rules switch possession a median 1.6 s early (at the pass, not the controlled touch); on the
   eye labels changes land where the rules do (0-2 frames). This is why `final_v3`/`v4` (rule
   vote) scored best on `val_h2` but were **rejected** on the bench: `final_v5` = `final_v2`
   decoder + stoppage gate. Do not re-litigate this without a real-GT yardstick.
3. **Bench overfitting.** Dozens of variants have been run on one bench; picking the max over many
   noisy scores overstates the winner. Record every try, keep borderline calls as "no".
4. **Video ball is a closed avenue.** Fine-tuned YOLO finds the ball within 2 m on 79% of held-out
   frames, but 67% of genuinely-missing-ball frames are stoppages (D), and merging the track
   *lowers* the bench (0.656 -> 0.62-0.64) and `val_h2` (0.558 -> 0.536). Oracle: even a perfect
   ball restored at 30% missing adds only +0.005 bench. The detector was trained on public-window
   frames, so any masked video score is optimistic.
5. **Broadcast ATD fills only 6.5% of the tactical missing ball** (same gaps), not worth syncing.
   Smart tagging agrees with the true team on only ~77% of live frames (below the model's ~90%).
6. **Uncertainty we cannot resolve locally:** the true full-match spellF1/macroF1 for `final_v5`.
   The bench (0.661) and masked (0.706) are proxies, not the score. One possession on the public
   window moves spellF1 by ~0.06.

## 10. Already tried and rejected (do not redo without new evidence)

Video ball channel / better DP linker (closed); noisy-student self-training on our match
(-0.016); CatBoost or HGB+CatBoost average (-0.015 to -0.024); segmental MBR decoding (+0.001);
learned temporal refiner (GRU/TCN) over tree probs (ext +0.011-0.017 but masked -0.02 to -0.03,
learns external label conventions); min dead-ball length inside Viterbi (no change);
velocity-projected pitch control (no change full fit); R17 touch-sequence/kick features for short
possessions (screen +0.040 but full fit +0.001; <2 s recall stuck at 0.186).

## 11. Where the remaining error actually is (target for new work)

On external matches, per 100 true possessions: missed 22, boundary drift 19, merged 11,
spurious 3.6, split 2.1. The decoder is conservative (~0.71 predicted possessions per true one) —
it misses short possessions to avoid spurious ones. **The largest unsolved error is short
possessions**: ~20-30% of true spells are < 2 s, recall ~0.19. R17's touch/kick features did not
move this on a full refit. A genuine short-possession-recall gain that survives the full training
recipe (mirrors + noise copies) would be meaningful; expect the mirror/noise augmentation to
absorb most screen gains, as it has every time so far.
