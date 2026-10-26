# Literature review: source families 3 and 4 for spellF1 possession

Scope: only two families, inspected from original sources and implementations.

- **Family 3** = temporal segmentation / structured sequence inference, academic.
- **Family 4** = Kaggle / competitive benchmark solutions for noisy, missing, sparse-label time series or event detection.

Goal: judge credible transfer to the Barca possession task (three states H/A/D, ranking
metric `spellF1`, only 5 minutes / 17 possessions of public labels, ball missing on 41% of
graded frames, external training data with a different possession convention).

Date: 2026-10-25. All web sources are public papers, write-ups or code repos. No match data
or derived frame-level output was sent anywhere; only aggregates and public literature were
consulted.

## 0. What is already covered (so this review does not repeat it)

From `docs/INVESTIGATION.md` section 0 and sections 21-25, the following are already inspected
and/or measured on our bench:

- MS-TCN / MS-TCN++ (section 0.2; single-stage dilated TCN refiner measured section 23, rejected).
- ASRF boundary-regression head (section 0.2; `w_bound` measured section 8, about zero).
- Segmental minimum-Bayes-risk decoding (Goel and Byrne 2004; measured section 21, +0.001).
- HSMM / fixed minimum-dead dwell (section 0.2 and section 11, no change).
- BiGRU temporal refiner over out-of-fold probabilities (section 22, rejected).
- Child Mind Institute sleep-state Kaggle solution (section 0.3), DFL Bundesliga shootout
  (section 0.3), NFL Player Contact Detection (section 0.3).
- PathCRF dynamic CRF with masked transitions (section 0.1).
- Borghesi et al. TCN for ball possession (section 0.1).
- Noisy-student self-training on our own match (section 14; failed in round 2, retest queued).
- Saerens-style unsupervised prior shift (mentioned in `docs/REVIEWS.md` 17a).

The single most important measured fact for transfer is section 23: **any learned temporal
stage trained on the 9 external matches learns the IDSSE/Metrica label convention (more
frequent switches, 4.3 vs 3.4 possessions per minute) and loses 0.02-0.06 spellF1 on our
window, even while gaining +0.011 on external leave-fold-out.** Every candidate below is
judged against this convention-drift risk.

## 1. Family 3: academic temporal segmentation and structured inference

### 1.1 Bayesian online changepoint detection (BOCPD)

- Source: Adams and MacKay, "Bayesian Online Changepoint Detection", 2007,
  arXiv:0710.3742, https://arxiv.org/abs/0710.3742.
- Implementations inspected: Meta `kats.detectors.bocpd`
  (https://facebookresearch.github.io/Kats/api/_modules/kats/detectors/bocpd.html),
  `hildensia/bayesian_changepoint_detection` (referenced by the `changepoint-doctor` PyPI
  package, https://pypi.org/project/changepoint-doctor/), and `fast-bocpd`
  (https://pypi.org/project/fast-bocpd/).
- Reported: [reported] exact online inference of the posterior over run length (time since
  the last changepoint) under a hazard prior and conjugate observation model; no threshold
  tuning, probabilistic change times.
- Data / eval assumptions: [reported] derived for i.i.d. regime segments with conjugate
  observations (Gaussian, Poisson). Categorical posteriors are non-conjugate and need a
  grid or particle approximation; this is a known extension, not the paper's default.
- Transfer to us: [hypothesis] This is the most different decoder on the list. Instead of
  Viterbi maximising path probability over our tree log-probabilities, BOCPD keeps a
  posterior over "how long have we been in the current possession" given per-frame
  observation likelihoods. It fits three things to our own data only: (a) the observation
  model from cached `P_R1`, (b) a hazard prior set from the duration distribution of the
  17 public possessions, and (c) nothing from external labels. Because no temporal stage is
  trained on external matches, it does not inherit the IDSSE touch-happy convention that
  killed the GRU/TCN refiners (section 23). Missing-ball frames are handled by skipping the
  observation update (the run-length posterior persists), which matches the 41% gap problem
  better than a frame-synchronous Viterbi that must emit a label every frame.
- Licensing / rule fit: [reported] arXiv paper is public for reading; `kats` is MIT,
  `fast-bocpd` is a small PyPI package. Reimplementing the forward recursion in numpy is
  trivial and adds no external weights. Compatible with rules (no hand labels, no external
  data beyond public code).

### 1.2 Timestamp-supervised action segmentation

- Source: Li, Abu Farha and Gall, "Temporal Action Segmentation from Timestamp Supervision",
  CVPR 2021, arXiv:2103.06669, https://arxiv.org/abs/2103.06669. Code:
  https://github.com/ZheLi2020/TimestampActionSeg (PyTorch, refined MS-TCN backbone).
- Follow-up: Abu Farha et al., "Robust Action Segmentation from Timestamp Supervision",
  BMVC 2022 (supplement at
  https://gall.cv-uni-bonn.de/download/jgall_robusttimestamp_bmvc22_supp.pdf).
- Different formulation: Hirsch et al., "Random Walks for Temporal Action Segmentation with
  Timestamp Supervision", WACV 2024,
  https://openaccess.thecvf.com/content/WACV2024/papers/Hirsch_Random_Walks_for_Temporal_Action_Segmentation_With_Timestamp_Supervision_WACV_2024_paper.pdf.
- Reported: [reported] one labelled frame per action instance is enough to reach close to
  fully-supervised segmentation on 50Salads / GTEA / Breakfast, using model output plus the
  timestamps to detect action changes, and a confidence loss forcing probabilities to
  decay monotonically with distance from the seed frame.
- Data / eval assumptions: [reported] timestamps are placed at representative (most
  distinctive) frames, i.e. away from ambiguous boundaries. Segment F1 at IoU tolerances,
  the same metric family as spellF1.
- Transfer to us: [hypothesis] We do not have per-instance seed frames in the unlabelled
  85 minutes, so the exact method does not apply. But the principle is a more conservative
  pseudo-labelling than the full noisy-student attempt (section 14): start only from the
  17 public possessions' boundary frames, propagate labels outward with a confidence that
  decays monotonically away from the seed, and treat the gap between two opposite-team seeds
  as a candidate change region rather than assigning it a soft label. This avoids feeding
  the model IDSSE-style transition density. The random-walk variant (Hirsch WACV 2024) is a
  transductive graph propagation over frame-frame similarity, which could in principle spread
  the 17 possessions across the match, but it is heavy and untested on our features.
- Licensing / rule fit: [reported] the GitHub repo does not state a LICENSE file in the
  inspected page; treat code as read-for-understanding only and reimplement. Reimplementing
  the confidence decay is a few loss lines. Compatible with rules (no hand labelling of
  the unlabelled 85 minutes; labels come from the algorithm).

### 1.3 NN-Viterbi: Viterbi loss for weakly supervised learning

- Source: Richard, Kuehne, Iqbal and Gall, "NeuralNetwork-Viterbi: A Framework for Weakly
  Supervised Video Learning", CVPR 2018, arXiv:1805.06875,
  https://arxiv.org/abs/1805.06875. Code linked from
  https://hildekuehne.github.io/.
- Reported: [reported] a Viterbi-based loss for online/incremental weakly supervised video
  learning; explicit context and length modeling give up to +10% over prior weakly
  supervised baselines on action segmentation benchmarks.
- Transfer to us: [hypothesis] The core idea is to train a frame classifier so that Viterbi
  alignment of a known sequence of labels recovers the true segmentation. Our external
  training already uses true frame labels, not ordered transcripts, so NN-Viterbi does not
  add supervision there. Its "explicit length modeling" is the part that overlaps with the
  duration questions in 1.4 below. Lower priority than 1.1 and 1.2.

### 1.4 Segmental RNN / semi-Markov CRF

- Source: Kong et al., "Segmental Recurrent Neural Networks", 2015, arXiv:1511.06018,
  https://arxiv.org/pdf/1511.06018. Ye and Ling, "Hybrid semi-Markov CRF for Neural
  Sequence Labeling", ACL 2018, arXiv:1805.03838, https://arxiv.org/html/1805.03838v1.
  Modern efficient head: `flash-semicrf` on PyPI, https://pypi.org/project/flash-semicrf/0.2.0/.
- Reported: [reported] segment-level (semi-Markov) models score whole spans instead of
  frames, with latent or observed boundaries; outperform BIO/CTC on handwriting and
  word segmentation.
- Transfer to us: [hypothesis] We already do a fixed minimum-dead dwell inside Viterbi
  (section 11, no effect). A learned duration distribution per state would be more
  expressive, but any duration learned from the 9 external matches learns IDSSE's shorter
  possessions. This is exactly the convention-drift trap of section 23. Only worth it if
  the duration prior is fit to the 17 public possessions, in which case it collapses to a
  fixed prior close to what we already tune. Net: low expected gain, high drift risk.

### 1.5 Posterior regularization (constrained EM)

- Source: Ganchev, Graca, Gillenwater and Taskar, "Posterior Regularization for Structured
  Latent Variable Models", JMLR 2010,
  https://jmlr.org/papers/volume11/ganchev10a/ganchev10a.pdf.
- Reported: [reported] soft linear constraints on posterior moments during the E-step of
  EM, keeping the model efficient while enforcing desired structural expectations (e.g.
  "every word aligned to at most one other word").
- Transfer to us: [hypothesis] The natural constraints are already the challenge's own
  possession rules: (a) contested / loose-ball phases keep the last team, (b) no micro-spell
  under some duration (the ceiling analysis in section 1 says a spurious 0.4 s blip per
  minute costs 0.227 spellF1). We currently approximate (a) and (b) with Viterbi switch
  costs and post-hoc dwell. Posterior regularization is the formal version: project the
  forward posterior to satisfy the constraints in expectation, and tune the slack on the 17
  public possessions. Expected gain is small because section 21 (MBR) showed Viterbi
  already sits near the expected-metric peak for these probabilities, but the cost is low
  and it gives a principled tuning knob rather than ad-hoc costs.

### 1.6 Streaming selective state-space boundary detection (new, 2026)

- Source: "Groupwise Selective State-Space Filtering for Accurate and Streaming Action
  Boundary Detection", 2026, arXiv:2609.33400, https://arxiv.org/html/2609.33400.
  Code: https://github.com/BOA-clk/Boundary_Detection.
- Reported: [reported in abstract] independent selective-scan chains over feature
  subspaces, stateful streaming decoding, boundary scores plus structured timestamp
  decoding.
- Transfer to us: [hypothesis] Very recent, single-team report, no independent replication.
  It is a temporal refiner, so it inherits the section 23 convention-drift problem if
  trained externally. Defer: only revisit if a cheaper decoder change does not move the
  bench and we have GPU budget.

## 2. Family 4: competitive benchmarks for noisy / missing / sparse-label series

### 2.1 NFL Big Data Bowl 2026 (player movement from tracking)

- Source: official page https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl.
  Write-ups inspected: 3rd place
  https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution,
  and a 33rd-place "ReLEmbed + Chiral Augmentation" write-up
  https://www.kaggle.com/c/nfl-big-data-bowl-2026-prediction/writeups/relembedding-architecture-and-chiral-augmentation.
- Reported: [reported] sorting players by distance to a reference point to make a slot
  invariant input tensor; a "chiral" flip augmentation enforcing left-right field symmetry;
  spatio-temporal transformers over tracking.
- Transfer to us: [hypothesis] The slot-invariant ordering and the field-symmetry flip are
  ideas we already implement: all features are team-level (permutation invariant), and home /
  away swap with team blocks swapped is already an augmentation (`docs/REVIEWS.md` 10).
  The write-up adds no new lever. Confirms we are doing the right thing, not a new action.

### 2.2 Sussex-Huawei Locomotion-Transportation (SHL) Recognition Challenge

- Source: challenge summary Wang et al. 2018, listed at
  http://www.shl-dataset.org/publications/. A missing-modality attention MIL approach:
  arXiv:2404.15323, https://arxiv.org/html/2404.15323v1. A 1D-sensor-DenseNet solution:
  Zhu et al. 2019 (PDF indexed in search results).
- Reported: [reported] per-modality subnets fused at the end; attention MIL that can cope
  with a missing modality by soft gating rather than a hard zero.
- Transfer to us: [hypothesis] The ball is a modality that is present 59% of the time and
  absent 41%. We currently feed a hard `seen` flag and blank ball columns when unseen. The
  SHL lesson is a soft missing-modality gate rather than a hard flag. This is a small
  feature change, not a new model. It is listed for completeness; expected gain overlaps
  with the existing ball-free shape model (section 3).

### 2.3 LANL Earthquake Prediction (2019)

- Source: "Laboratory earthquake forecasting: A machine learning competition" (Team Zoo
  first place, 238 entries); Tom Van de Wiele write-ups index
  https://www.kaggle.com/tvdwiele/writeups.
- Reported: [reported] predict time-to-failure from one 1-d acoustic series; winners used
  rolling-window aggregate features plus neural nets and LightGBM.
- Transfer to us: [hypothesis] The target is a regression (time to next event), not a
  segment F1. The rolling-feature recipe is already what our per-frame tree does. No new
  method to import.

### 2.4 Volcanic eruptions INGV-2022 (2022)

- Source: 1st-place write-up
  https://www.kaggle.com/competitions/predict-volcanic-eruptions-ingv-oe/writeups/shallow-learning-1st-place-mel-spectrogram-blended.
- Reported: [reported] mel-spectrogram images plus an ensemble of ResNets; each sensor is
  a channel with missing sensors.
- Transfer to us: [hypothesis] Sensor channels with missing values, fused late. Same lesson
  as 2.2. Spectrogram image conversion of our 1-d series would add a CNN over a signal that
  is already a 22-player geometric state, not an acoustic waveform. Low fit.

## 3. Fundamentally different formulations, ranked by credibility of transfer

| rank | formulation | why different from current Viterbi-plus-trees | convention-drift risk |
|---|---|---|---|
| A | BOCPD generative run-length filter (1.1) | online posterior over time-since-change instead of path-prob maximisation; observation-independent persistence during ball gaps | low (fit to our posteriors + public durations) |
| B | Timestamp-style conservative pseudo-labelling (1.2) | propagate only from seed frames with monotonic confidence decay, instead of soft labels on every unlabelled frame | low (no external label sequences) |
| C | Posterior-regularised EM (1.5) | project forward posterior to satisfy possession rules in expectation, instead of fixed switch costs | low |
| D | Transductive random-walk label spread (1.2, Hirsch) | graph diffusion of the 17 possessions across the match | medium (transductive, heavy) |
| E | Semi-Markov learned durations (1.4) | span-level scoring with learned duration | high (learns IDSSE durations) |
| F | Streaming SSM refiner (1.6) | recurrent selective-scan boundary net | high (same as rejected GRU/TCN) |

## 4. Ranked controlled experiments, with cost and decision criteria

All experiments reuse cached `P_R1` probabilities where possible (no external data, no new
weights, no frame annotation). The bench columns are the same as `docs/INVESTIGATION.md`
section 24: bench, ext (IDSSE + Metrica), masked, public (17 possessions), val_h2, valLo.
Adoption rule from prior rounds: bench must move at least +0.003 and masked / public must
not drop by more than 0.005; a gain only on ext while our window loses is rejected (section
23 lesson).

### E1. BOCPD decoder on cached `P_R1` (highest priority)

- What: implement the BOCPD forward recursion in numpy. Observation likelihood per frame is
  the 3-vector softmax of `P_R1` (H/A/D). Hazard prior `h = 1 / mean public possession
  duration`, estimated from the 17 public possessions (not external matches). Missing-ball
  frames skip the observation update so the run-length posterior persists. At the end, take
  the MAP state at each run, or forward-fill the run-mean state for graded frames.
- Cost: about half a day, pure numpy, deterministic, no GPU, no retraining.
- Decision criterion: bench >= 0.664 (+0.003 over current 0.661) AND masked >= 0.701 AND
  public not below 0.690. If bench moves but masked drops, reject.
- Why it can win where MBR (section 21) did not: MBR only re-picked among existing cost
  scales; BOCPD changes the smoothing prior itself and persists through ball gaps instead
  of forcing an emission every frame.

### E2. Seed-frame conservative pseudo-labelling for the 85 minutes

- What: take the 17 public possessions' boundary frames as seeds. For each seed, emit a
  soft label whose confidence decays linearly / exponentially with distance from the seed,
  and leave the inter-seed span as "change candidate" with no soft label. Refit the tree on
  external labels plus this conservative our-match signal, mirrors and noise copies.
- Cost: about one day; needs a refit, but no new data.
- Decision criterion: bench >= 0.664 AND masked >= 0.701 AND public >= 0.697. Gate hard:
  if ext improves while masked or public drops, reject (this is the exact section 23
  signature).
- Relation to section 14: noisy-student used full soft labels on all 85 minutes; this is a
  deliberately weaker signal, which the timestamp paper shows is enough on its own task.

### E3. Posterior-regularised switch cost

- What: replace the fixed Viterbi switch cost with an E-step projection that enforces the
  "no micro-spell < k frames" and "contested phases keep last team" expectations. Tune the
  slack parameter on the 17 public possessions only.
- Cost: half a day; reuses existing forward-backward code.
- Decision criterion: bench >= 0.664 AND no masked / public regression. Expected to be small
  given section 21; keep only if it clears the gate.

### E4. Soft missing-modality gate (SHL lesson)

- What: replace the hard `seen` flag and blanked ball columns with a learned per-modality
  gate that fuses a with-ball branch and a ball-free branch by reliability.
- Cost: half a day feature change; refit.
- Decision criterion: bench >= 0.664. Lower expected value because the ball-free shape
  model already exists (section 3).

### E5. Defer (do not run now)

- Semi-Markov learned durations (E): high drift risk, 17 possessions too few to fit a
  duration distribution per state credibly.
- Transductive random-walk label spread (D): heavy, transductive, no GPU budget before the
  14 November freeze.
- Streaming SSM refiner (F): a temporal refiner trained externally repeats section 23.
- LANL / volcanic spectrogram CNNs (2.3, 2.4): target and signal do not match.

## 5. Uncertainty and open questions

- Reported numbers for 1.2 and 1.3 are on 50Salads / GTEA / Breakfast, which have 50-100
  action classes and dense, clean features. They do not transfer numerically; only the
  mechanism transfers.
- BOCPD with a non-conjugate categorical observation model is not the paper's default; the
  grid size over run length (cap a few hundred seconds) and the numerical log-sum-exp are
  implementation details that could cost accuracy. This is the main uncertainty in E1.
- The 17 public possessions give a noisy duration prior (section 0.4: plus or minus
  0.04-0.07 spellF1 between masking seeds). E1 should sweep the hazard prior, not fix it.
- No code LICENSE was visible on the TimestampActionSeg repo page; treat its code as
  read-for-understanding only. Reimplementing is allowed and preferred.

## 6. Licensing and rule compatibility summary

- All consulted sources are public papers, public Kaggle write-ups or public code repos.
- No external weights, datasets or pre-trained models are proposed for adoption.
- No hand annotation of the 85 minutes is proposed; E2 uses algorithm-generated soft
  labels from the 17 public seeds, which keeps labels "produced by an algorithm" per
  `challenge/RULES.md`.
- Nothing in this review requires sending match data, video or derived frame-level output to
  any hosted service.
- The match data itself remains competition-only; this document cites no frame-level
  numbers from the match, only aggregates already published in `docs/INVESTIGATION.md`.

---

## 7. Decision addendum (after reading `analysis/autonomous_research/audit/AUDIT.md`)

The audit pins the real target, not the bench average: on external matches per 100 true
possessions we miss 22, drift on 19 boundaries, merge 11, with only 3.6 spurious and 2.1
splits. The decoder is conservative (0.71 predicted possessions per true one). It already
suppresses spurious blips (the costliest error per section 1); the open error is **short
possessions under 2 s, recall ~0.19**. Anything that trades spurious for missed-short is the
right direction, not the wrong one. The adoption gate is **bench +0.005** with ext and masked
each not dropping by more than 0.005 (the earlier +0.003 in section 4 is overridden).

This re-ranks the candidates against that one gap and against the audit's "already rejected"
list (noisy-student -0.016, R17 touch/kick +0.001 full fit, segmental MBR +0.001, GRU/TCN
refiner masked -0.02 to -0.03, min-dead no change).

### 7.1 The concrete new rationale (not "another decoder")

The current Viterbi couples two things that should be separate: (a) how likely a state
switch is at frame t, and (b) how long a possession is expected to last. Under Viterbi the
duration distribution is geometric, so a 1.5 s possession pays **two** transition costs
(enter and leave) within 2 s and is exponentially penalised. That is the mechanism behind
the 22 missed + 11 merged short spells. A semi-Markov (segmental) decoder separates them:
it scores each whole span with (i) the accumulated per-frame state likelihood from `P_R1`
and (ii) a duration-likelihood term drawn from a duration distribution.

The novelty constraint from the audit is satisfied because the duration distribution is fit
to the **17 public possessions** (mean 14.2 s, but the empirical histogram includes the short
ones), not to the 9 external matches. This avoids the exact convention-drift that killed the
GRU/TCN refiner (section 23 / audit item 10): we do not learn any label sequence from IDSSE,
we only borrow a span-scoring *form*.

### 7.2 Ranked experiments, final

**Protocol correction (frozen).** Fitting or sweeping a duration density on the 17 public
possessions is **disallowed**: the public window is sanity-only and using its durations in
inference leaks the public target. This review previously suggested that; it is withdrawn.
No duration prior may be estimated from public GT. The geometric-duration causal claim in
section 4 is a hypothesis only, not an explanation of the measured missed counts.

| rank | experiment | targets | cost | decision |
|---|---|---|---|---|
| tested | **BOCPD convention-neutral smoother on cached `P_R1`** (Adams and MacKay 2007, arXiv:0710.3742) | generic temporal persistence | 1 day | **NEGATIVE at every predeclared hazard** (see log below). Not adopted. |
| test next | label-free semi-Markov duration regularizer with a fixed weak prior | short-possession recall | ~1 day | only if it beats the duel Viterbi on bench +0.005 with no ext/masked regression |
| defer | geometry-conditioned duel Viterbi | belongs to the audit agent; do not overlap | — | — |
| drop | seed-frame pseudo-labelling (E2) | — | — | noisy-student already -0.016 (audit item 10); bench overfitting trap (item 9.3) |
| drop | streaming SSM refiner, external duration model | — | — | repeats section 23 convention drift |

**Executed result (2026-10-25, cached `P_R1`, official spellF1 via `analysis.yardsticks.Yard`).**

**Naming correction.** The recursion implemented, `q_t = normalize((1-h) q_{t-1} P_t + h prior P_t)`,
is **not** the Adams-MacKay BOCPD run-length posterior. True BOCPD maintains a posterior over run
length `R_t(r)` with run-conditional sufficient statistics; this recursion marginalizes run length
into a 3-state HMM with transition matrix `T[i,i]=1-h`, `T[i,j]=h/3`, i.e. a **fixed-hazard
geometric-duration HMM** (an exponential smoother with a changepoint prior). The negative result
below is for the geometric HMM, not for a true run-length BOCPD; section 1.1's description of
BOCPD stands.

Baseline `v2g` reproduced: bench 0.661, ext 0.616, masked 0.706, public 0.697 (14 spells).
The geometric-HMM smoother, fed back into the same v5 Viterbi decoder, scored:

| hazard h (expected run) | bench | ext | masked | public spells |
|---|---|---|---|---|
| baseline (no smoother) | 0.661 | 0.616 | 0.706 | 14 |
| 1/25 | 0.628 | 0.602 | 0.654 | 17 |
| 1/50 | 0.632 | 0.601 | 0.663 | 17 |
| 1/100 | 0.631 | 0.599 | 0.663 | 17 |
| 1/200 | 0.629 | 0.597 | 0.662 | 17 |
| 1/400 | 0.633 | 0.595 | 0.671 | 17 |

Every setting loses bench by 0.028-0.033 and masked by 0.04-0.05, even though public spell
count moves 14 to 17 (closer to the true 17). This is the same signature as the rejected
GRU/TCN refiner: more spells helps the public count but hurts the masked/ext yardsticks.
The existing Viterbi switch costs already encode the right persistence; a generic exponential
smoother cannot beat them because it ignores the separate H<->A and alive<->dead costs.

**Honest uncertainty.** One negative sweep over five hazard values on one bench is not a
significance test; it is a development heuristic. But the monotonic loss across the whole
grid makes a useful hazard unlikely. The next informative lever is a label-free duration
regularizer with a fixed weak prior, not another persistence smoother.

