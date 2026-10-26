# Family 5 read-only review: missing-data, weak supervision, uncertainty, metric-aligned learning, experimental integrity

Date: 2026-10-25. Scope: source family 5 only. Read-only on the repository. No match data,
tracking rows, video, or frame-level output was read out or sent to any hosted model. Only code,
organiser docs, public literature, and the already-printed aggregates were used. No shared code was
modified. No hand labels were created. Nothing was published.

Project root: `/Users/bytedance/Desktop/BARCA/more-than-a-hack-2027`.
Sibling artifacts: `analysis/autonomous_research/audit/AUDIT.md` (scorer + baseline reproduction),
`analysis/autonomous_research/literature/sequence/REVIEW_source_families_3_4.md` (temporal models).
This document does not repeat those.

---

## 1. What I independently verified from code (not from the team's claims)

These are facts I read directly in `evaluation/` and `mtah/`, independent of `docs/INVESTIGATION.md`.

1. **spellF1 definition** (`evaluation/possession_metrics.py`). Possessions are maximal runs of H
   or A inside a graded half. Predicted and true possessions are matched one-to-one by greedy
   highest-IoU overlap, same team required. F1 is averaged over IoU thresholds 0.3, 0.4, 0.5, 0.6,
   0.7. This matches `challenge/EVALUATION.md`. The scorer is deterministic.
2. **Abstention handling** (`fill_abstentions`). Before spell matching, every X is forward-filled
   with the previous non-X label inside the same half, back-filled at the half head. So on the
   ranking metric X means "repeat my last answer". At a hand-over that carries the old label across.
   Inside a possession it is free. This confirms the team's note that X on a graded frame cannot
   help spellF1.
3. **No cross-half leakage in the metric** (`valid_runs`). Temporal matching is computed separately
   inside each graded half. The 20,156-frame halftime gap cannot merge two dead stretches.
4. **Public ground truth** (`evaluation/gt/*_public_meta.json`). Exactly 17 possessions, 7,500
   graded frames, window 11185..18684, H 3454 / A 2577 / D 1469. Matches the docs.
5. **External folds are by match** (`mtah/bench.py`, `FOLDS`). Each external match is scored by a
   model trained on the other two folds. Mirrored and extra-noise copies of a match are added only
   to folds that train on that match. No match appears in both train and score.
6. **Look-ahead guard in masking** (`mtah/masking.py`). The ball is hidden for `TAIL = 750` frames
   past the public window end, so "frames until ball seen" features cannot peek at the well-tracked
   frames after the window.
7. **Hand labels are not in inference, for a structural reason, not because of the decoder flag.**
   The only reader of `work/labels/*.csv` is `mtah/bench.py:load_val()`, called inside
   `Bench.__init__`. The delivered `run.py` never imports or instantiates `Bench`. `use_rules=False`
   only means the rule string does not vote in the Viterbi decoder; rule-derived features from
   `mtah/rules.py` (an algorithmic touch and dead-ball state machine on tracking) still enter the
   trees. Those are algorithmic, not manual. Traced 2026-10-25, see
   `analysis/autonomous_research/experiments/integrity/REPORT.md` section 1.

The audit (`AUDIT.md`) already reproduced the public numbers: `final_v5` public spellF1 0.7226 and
macroF1 0.9098, and `baseline_smooth50` public 0.3256 / 0.6207. Those match the docs. I did not
re-run them.

---

## 2. Leakage and integrity risks, ranked

### R1 (medium): the masked bench shares the same 17 possessions as the public window
The masked yardstick hides the ball in the public window under 16 seeds, but the underlying true
possessions are the same 17. The 16 seeds vary only WHERE the ball is hidden. They do not resample
possessions. So masked has essentially zero possession-sampling variance. Its reported between-seed
sd of about 0.056 is ball-mask-placement noise, not the variance of scoring a new 5-minute block.
The official scorer's own bootstrap resamples whole possessions (`bootstrap_macro_f1` resamples
segments). The masked bench does not. Consequence: a +0.01 masked move is mostly seed luck, while
the private match is decided by roughly 270 possessions. The external half (about 3,000 possessions
over 9 matches) is the steadier number and should carry adoption.

### R2 (medium): bench and val_h2 are both selected-against, not independent validation
The team has chosen and then reversed versions on these same two yardsticks (v3/v4 chosen on val_h2,
reversed to v5; R17, R8, R5, R4 all screened and adopted on the bench). After many trials on two
small yardsticks, neither is an unbiased estimate of the private score. This is the classic
development-set leakage the team flags itself in ROADMAP rule 2. It is not a code bug. It is a
multiple-comparison bias baked into the decision procedure. The only true held-out signal is the
organiser's private full-match scorer, which is unobservable locally.

### R3 (low-medium): the stoppage gate was designed from two observed public-window spells
`stop_gate = 100` removes a live spell shorter than 4 s that sits between two dead spells while
players walk. The motivation was two specific spurious spells (1.84 s and 2.28 s) seen inside
ground-truth stoppages on the public window. It is the one structural decoder rule derived by
inspecting the 17-possession window. It held on external and on val_h2, and the condition
(short live spell between dead spells while players walk) is physically motivated, so the risk is
bounded. It should still be treated as the most public-window-shaped rule, not as a validated gain.

### R4 (low): masking bias `pace_weights` was set after inspecting the public window
Gaps start 3x more often when average player pace is below 1.3 m/s. This encodes the hypothesis
that ball loss clusters in stoppages. It is independently supported by the 300-frame probe
(67.0% of genuinely missing-ball frames are D), so it is not circular. It is mildly tuned to the
window but evidence-backed.

### R5 (central, acknowledged, not fixable locally): the private label convention is unknown
External training labels follow DFL / event conventions. IDSSE counts 4.3 brief touches per minute
against about 3.4 in our GT. A learned temporal stage trained on external labels adopts this
convention and loses on our match (INVESTIGATION section 23). The per-frame trees transfer better
precisely because they do not model label sequences. No local experiment can close this gap; it is
the dominant uncertainty and it is already documented.

### Verified non-issues
No raw tracking or video leaves the machine. The detector public-window contamination only affects
the closed video path (R6). The kickoff half-start rule uses published frame boundaries. Mirrored
copies are fold-local. The shape matrix for our match is ball-free and label-free.

---

## 3. The 0.005 promotion threshold is a development heuristic, not significance

ROADMAP rule 1 says adopt only if bench rises by at least 0.005 and ext and masked each move by no
more than 0.005. Treat that as a screening heuristic, not a statistical threshold.

- `challenge/EVALUATION.md` states the full-match spellF1 noise band is about 0.04. The 0.005 gate is
  eight times smaller than that band.
- The bench blends ext (3,000 possessions, stable) and masked (17 possessions, seed-noisy). A
  0.005 blend improvement can come entirely from a favourable ball-mask seed.
- With dozens of trials on one bench, the maximum over noisy scores is biased upward. ROADMAP rule
  2 already handles this by logging every try and calling borderline winners "no".

Recommendation for any future candidate, in order:
1. Reproduce the bench baseline first. The audit confirms the public scorer numbers but NOT the
   internal bench (ext 0.616, masked 0.706). Until that 0.661 is reproduced from a clean cache, any
   candidate delta is measured against a moving reference.
2. Require ext to move by at least the gate on its own, not just the blended bench. Ext has the most
   possessions and the least selection bias.
3. Require masked to not fall, and report the between-seed spread, not only the mean.
4. Require the gain to survive a second random seed or fold perturbation. One favourable seed is
   not a result.
5. Treat a borderline winner that needed several tries as "no". This is already the team's rule;
   restating it because it is the single best protection against bench overfitting.

---

## 4. Literature beyond the local search, with exact sources

The team's `docs/INVESTIGATION.md` section 0 already covers TCN possession, PathCRF, ballradar,
MS-TCN, ASRF, segmental MBR, HSMM, and the Kaggle precedents. Family 5 adds the methods below.
Evidence strength is tagged [verified-source] for papers I opened, [reported] for their abstracts.

### 4.1 Confident learning: audit noisy training labels (weak supervision, integrity)
- Source: Northcutt, Jiang, Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels",
  JAIR 2021. https://arxiv.org/abs/1911.00068. Code: https://github.com/cleanlab/cleanlab.
- Idea: estimate the joint distribution between given labels and true labels from out-of-fold
  predicted probabilities, then rank which training examples are likely mislabeled.
- Fit here: the 9 external matches carry labels from two sources with different conventions. The
  fold models already produce out-of-fold probabilities. We can rank every external frame by how
  strongly the model disagrees with its label, no refit needed to audit.
- Honest expectation: most disagreements will sit within a few frames of a possession change. That
  is the known IDSSE brief-touch convention, not an error. A residual mass of mid-possession
  disagreements would flag Metrica event-derived labels as candidates for real correction.

### 4.2 Data programming: combine weak labeling functions without hand labels
- Source: Ratner et al., "Data Programming: Creating Large Training Sets, Quickly", NeurIPS 2016.
  https://arxiv.org/abs/1605.07720. Snorkel system: Ratner et al., VLDB 2017.
- Idea: estimate labeling-function accuracies from their own agreement and disagreement on
  unlabeled data, then combine.
- Fit here: the hand rules are already labeling functions. The team already tried a supervised
  stacker of rules plus tree probabilities and it overfit (INVESTIGATION section 19, ext -0.12). An
  unsupervised combination is less likely to overfit, but the trees already dominate the rules, so
  the ceiling is low. Listed for completeness, not as a priority.

### 4.3 Black-box label shift: correct the prior between external and our match
- Source: Lipton, Wang, Smola, "Detecting and Correcting for Label Shift with Black Box Predictors",
  ICML 2018. https://arxiv.org/abs/1802.03916.
- Idea: estimate a label-prior shift from the model's output marginal on the unlabeled target match,
  using a confusion matrix estimated on validation data.
- Fit here: the team already tried a Saerens-style EM prior and it measured -0.017. The shift that
  hurts us is structural (how many brief touches count as a possession), not a marginal H/A/D
  frequency. BBSE corrects the marginal, not the structure. Expected gain: low. Keep as a monitor
  only.

### 4.4 Conformal and selective prediction
- Source: Xu and Xie, "Conformal Prediction Interval for Dynamic Time-Series" (EnbPI), ICML 2021.
  https://arxiv.org/abs/2010.09107. Selective classification: Geifman and El-Yaniv, 2017.
  https://arxiv.org/abs/1705.08500.
- Idea: distribution-free uncertainty sets for dependent time series; reject low-confidence inputs.
- Fit here: the scorer forward-fills X, so generic abstention does not buy spellF1 (already
  concluded in REVIEWS 17b and sixth-review C1). Uncertainty is useful only to gate the existing
  switch costs, which is what the trees' probabilities already drive. No new lever.

### 4.5 Metric-aligned (segment) training losses
- Source: e.g. "Diffusion Action Segmentation" boundary-alignment loss, https://arxiv.org/abs/2303.17959.
- Idea: train a temporal model with a boundary or segment-IoU loss instead of per-frame cross-entropy.
- Fit here: any learned temporal stage on external labels adopts the IDSSE convention and loses on
  our window (section 23). A segment loss does not fix the convention mismatch. Do not pursue before
  the convention question is resolved.

---

## 5. Ranked opportunities

Ranked by expected value per unit risk. "Cost" excludes a full model refit where possible.

### O1 (highest, zero model risk): reproduce the bench and add honest CIs
- What: run `Bench()` once to reproduce ext 0.616 and masked 0.706 from a clean cache. Then add a
  segment-bootstrap to the masked yardstick: resample the 17 public possessions (the scorer already
  resamples segments) and report a CI, alongside the 16 ball-mask seeds.
- Why: the internal bench is not yet independently reproduced (audit section 4.2). Until it is, no
  candidate delta has a stable reference. The masked mean currently hides that it resamples ball
  masks, not possessions.
- Assumptions: the cache is deterministic given the match name (verified in `noise.seed_for`).
- Evidence strength: high, derived directly from `mtah/bench.py` and `mtah/masking.py`.
- Smallest informative experiment: one `Bench()` run plus a 1,000-draw bootstrap over the 17 public
  possessions using the existing scorer bootstrap. Compare the segment-bootstrap sd to the
  between-seed sd. If they differ, masked is understating uncertainty.

### O2 (medium, read-only analysis): confident-learning audit of external labels
- What: use cached out-of-fold probabilities (for example `cache/P_R1*.npz`) to compute, per external
  frame, whether the fold model's argmax disagrees with the external label. Bin disagreements by
  distance to the nearest possession change.
- Why: separates expected convention drift (disagreements within a few frames of a boundary, which
  we must NOT relabel) from likely real label or tracker errors (disagreements mid-possession,
  especially in Metrica event-derived labels).
- Compatibility: no model refit to audit. No hand labels. No data leaves the machine.
- Evidence strength: medium. The known convention difference predicts most disagreements are benign.
- Smallest informative experiment: one pass over cached probabilities, print the histogram of
  disagreement distance-to-boundary per source. If over 80 percent sit within +/- 5 frames, the
  external labels are convention-consistent and no relabel helps (matches the measured -0.002). If a
  mid-possession mass exists, flag those matches for a person to spot-check (never an LLM).

### O3 (low, monitor only): prior-shift check
- What: estimate the H/A/D output marginal on our unlabeled match and compare it to the external
  training prior. Only if the gap is large, try BBSE-style reweighting.
- Why: EM prior already measured -0.017, and the real shift is structural not marginal.
- Evidence strength: low. Listed so it is not rediscovered later.

### O4 (closed, do not reopen)
Abstention or X on boundaries. Segmental MBR. GRU or TCN refiner. Video ball channel. CatBoost
average. Noisy-student self-training. Smart Tagging fusion. Global IDSSE relabel. Velocity pitch
control. R17 touch-sequence features. Each was measured in INVESTIGATION sections 9 through 25 and
in AUDIT section 10. Reopen only with a new mechanism, not a re-run.

### O5 (defer): fresh labelled matches in our convention
PFF requires personal sign-up. SkillCorner has no per-frame possession. No ready source exists
before the 14 November freeze.

---

## 6. Compliance checklist for this review

- No match data, video, or frame-level export was read out or sent to any hosted model.
- No hand labels were created. Existing `work/labels/` were treated as off-limits for tuning.
- No shared code was modified. Outputs live only in `analysis/autonomous_research/literature/
  integrity/`.
- Nothing was published, pushed, or committed.
- No broad model selection was run. The bench baseline reproduction (O1) is left as a recommended
  next step, not executed here.

## 7. Source verification addendum (2026-10-25)

Two sports-review claims were re-checked against the original source, not the secondary summary.

1. **Renner et al., arXiv:2610.11571, "GNN-LSTM match-phase classification".** Verified at
   https://arxiv.org/abs/2610.11571. The abstract confirms: rule-based labels over **203 matches**,
   seven-class phase per second, and the selected GNN-LSTM beats the strongest aggregated-feature
   baseline by **4.6 macro-F1 points** (the baseline scores 4.6 percent lower). The sports review's
   numbers match. Caveats to keep: it is a v1 preprint (8 Oct 2026), the labels are rule-generated,
   and the task is 7-class phase, not H/A/D possession. The "do not add a GNN" conclusion stands.

2. **Socceraction `player_possession_time`.** Verified against the source at
   https://raw.githubusercontent.com/ML-KULeuven/socceraction/master/socceraction/vaep/features.py.
   The docstring says possession time is "reset after a defensive interception attempt or a take-on".
   The implementation does NOT measure that. It accumulates seconds only when the SAME player performs
   consecutive actions and the previous action type was a dribble. Any other event type, a player
   change, or a period change resets the count. There is no geometry: no opponent distance, no
   contested-ball test. The docstring line is a narrative gloss for why the same-player-dribble chain
   breaks, not an implemented detector. Consequence: the sports review's E1 idea (apply this rule to
   our tracking to tag contested vs clean changes from geometry) rests on an over-reading. The
   function needs SPADL event types we do not have per frame and provides no geometry. It should not
   be cited as a geometric controlled-possession reference. The contested-moment test, if pursued, must
   be designed from our own tracking (nearest opponent at the change), not borrowed from this code.

**Public-fitting check.** This review proposes no duration fitting on the 17 public possessions and
no hyperparameter sweep on the public window. O1 adds a possession bootstrap only to report a CI; it
does not fit. The BOCPD-style hazard fit on public durations that the sequence family considered was
rejected and is not endorsed here.

## 8. Bottom line
The pipeline is disciplined and the scorer is trustworthy. The main integrity risk is not a data
leak. It is that two small yardsticks (a 17-possession window with 16 ball-mask seeds, and
eye-labelled windows chosen twice already) are being read as if they were independent validation.
The 0.005 gate is a development heuristic. Before any new candidate is trusted, reproduce the
0.661 bench from a clean cache, require ext to move on its own, and add a possession-level bootstrap
CI. Among family-5 methods, the only one with real upside is a read-only confident-learning audit of
the external training labels. Everything else is already measured closed.
