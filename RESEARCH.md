# Research report: what limits the possession model, and what we changed

Written 2026-10-25 after a fresh review of the whole pipeline: the external literature, Kaggle
precedents, an audit of the current system and new experiments. Measured numbers come from
scripts in `analysis/` and print aggregates only (no match data left this machine or went to any
hosted LLM). INVESTIGATION.md §20–§23 has the measured detail; ROADMAP.md has the resulting
decisions.

Labels used below: **[measured]** = run here on our data; **[reported]** = a number published by
someone else, not reproduced; **[hypothesis]** = not tested.

---

## A. Literature and competition review

### A.1 Direct precedents: possession and game state from tracking data

| work | problem / method | data | reported result | what transfers to us |
|---|---|---|---|---|
| Borghesi, Costa, Morra, Lamberti, *Using Temporal Convolutional Networks to estimate ball possession in soccer games*, Expert Syst. Appl. 223 (2023) 119780, doi:10.1016/j.eswa.2023.119780 ([repo page](https://iris.polito.it/handle/11583/2976470)) | **Our exact 3-class task** (dead / home / away) from tracking. Per-object TCN embeddings, combined by permutation-invariant layers (sum, self-attention or 2-D conv); ternary vs two binary heads | professional tracking data (full text not reachable from here) | [reported] 86.2% possession accuracy (+7.3 over rule-based), 89.2% dead/alive (+33.2) | Confirms that learned models beat hand rules by a wide margin on dead/alive, which is also what we see (rules 0.36 ext vs model 0.62). Frame accuracy only: no segment metric, so it says nothing about spellF1-style over-segmentation |
| Kim et al., *PathCRF: ball-free soccer event detection via possession path inference* (arXiv 2602.12080, 2026) ([paper](https://arxiv.org/html/2602.12080), [code](https://github.com/hyunsungkim-ds/pathcrf)) | possession as a path over a player graph; set-attention + temporal attention encoder; **dynamic CRF with masked illegal transitions, Viterbi decoding** | IDSSE / Sportec Open DFL, 7 matches (5 train / 1 val / 1 test), 5 fps, in-play only | [reported] frame-level **team possession accuracy 92.4% with no ball** (1 test match); event F1 75.7%; Viterbi on a non-CRF model already gives 73.1 vs 65.3 without | Same training data as ours. Our ball-free shape model sits at 88–93% team accuracy without the ball on our public window, so a much heavier ball-free network is unlikely to move team accuracy much. Their ablation supports what we already do: structured decoding is worth more than the encoder |
| Kim et al., *Ball trajectory inference from multi-agent sports contexts using Set Transformer and hierarchical Bi-LSTM*, KDD 2023 ([arXiv 2306.08206](https://arxiv.org/abs/2306.08206), [code](https://github.com/pientist/ballradar)) | predict the possessing **player**, then regress the ball path conditioned on it | Metrica sample games + others | [reported] see paper; designed to fill missing ball | The possessor-first hierarchy is what our `ball_block` + shape features approximate. Imputing the ball would feed our rules a guessed ball; §9 / §12 showed even a *true* ball restored in gaps adds only +0.005 bench, so imputation is low value for us |
| Vidal-Codina, Evans, El Fakir, Billingham, *Automatic event detection in football using tracking data*, Sports Engineering 25:18 (2022) ([arXiv 2202.00804](https://arxiv.org/abs/2202.00804)) | deterministic decision tree: possession step, then events from possession changes and laws of the game | several providers and tournaments, clean tracking with ball | [reported] > 90% detection in most event categories | A good blueprint for touch/possession rules **on clean tracking**. Our rules follow the same idea and score 0.61 spellF1 on clean external tracking but 0.37 once degraded to our noise (§6): with 41% ball missing, rule systems alone are not enough |
| Link & Hoernig, *Individual ball possession in soccer*, PLOS ONE 12(7) e0179953 (2017); Hoernig et al. 2016 | Bayesian network / physics + ML for degrees of ball control | Bundesliga tracking | [reported] 80–92% precision/recall on clean data, ~4 points lower on uncorrected data | Supports CHALLENGE.md's "controlled possession" notion: a touch is not possession. Our GT measurements agree (§2) |
| Bassek et al., IDSSE / Sportec Open DFL dataset, Sci. Data (2025), doi:10.1038/s41597-025-04505-y | 7 Bundesliga matches with synchronized tracking, events, per-frame `BallPossession` / `BallStatus` | — | — | Our main training set (7 of 9 matches). Its possession convention counts more brief touches (4.3/min vs 3.4/min in our GT, §6) |

### A.2 Transferable methods: temporal segmentation with an over-segmentation penalty

| work | idea | evidence | relevance |
|---|---|---|---|
| Farha & Gall, *MS-TCN*, CVPR 2019 ([arXiv 1903.01945](https://arxiv.org/abs/1903.01945)); MS-TCN++ ([2006.09220](https://arxiv.org/abs/2006.09220)) | stacked dilated TCN stages, each refining the previous one's frame probabilities; truncated-MSE smoothing loss against flicker | [reported] state of the art on 50Salads / GTEA / Breakfast at the time, with segmental F1@{10,25,50}, a metric of the same family as spellF1 | Direct template for a **learned refiner on top of our tree probabilities** (tested here, §B.3 / INVESTIGATION.md §21) |
| Ishikawa et al., *ASRF*, WACV 2021 ([arXiv 2007.06866](https://arxiv.org/abs/2007.06866)) | frame classifier + boundary-regression branch; boundaries suppress over-segmentation | [reported] up to +16.1 segmental F1 | Our "boundary head" (`w_bound`) is this idea; measured ±0.001 on our bench (§8). Not repeated |
| Segmental minimum-Bayes-risk decoding, Goel & Byrne, IEEE TSAP (2004) | choose the output that maximises the **expected task metric** under the posterior, stretch by stretch | [reported] consistent WER gains in ASR | Our metric is a segment F1 and Viterbi optimises path probability, not spellF1. Tested here (§B.3 / INVESTIGATION.md §22) |
| HSMM / explicit-duration models | duration distributions instead of geometric | textbook | Our minimum-dead dwell inside Viterbi was measured as no change (§11); not repeated |

### A.3 Kaggle and competitive precedents

| competition | what won | what transfers | what does not |
|---|---|---|---|
| Child Mind Institute – Detect Sleep States (2023): per-step state segmentation of long accelerometer series, event AP with tolerances | [reported, secondary source: a competitor's summary deck] 1st place used a 1-D UNet + GRU; post-processing tuned to the metric added +0.02 private | Sequence heads over per-step features beat per-step GBDTs; **post-processing aligned with the metric matters as much as the model** | 100s of training series there vs 9 matches here; their labels were the same convention as the test |
| DFL – Bundesliga Data Shootout (2022): event spotting (play / challenge / throw-in) from video | write-ups found were 6th place and below; video CNNs + temporal heads | — | Video-only, different target; our video ball channel was already measured and closed (§9–12) |
| NFL Player Contact Detection (2023): contacts from tracking + video | [reported, secondary] GBDT on tracking features to filter, 3-D CNN on video, GBDT post-processing | Tracking-feature GBDTs are a strong base stage; a second stage over first-stage outputs is standard | Pairwise contact, not segment-level possession |

**Common lesson** across these: (1) a strong per-step model, (2) a temporal stage, (3) decoding and
post-processing tuned to the official metric on a validation set with the **same label convention
as the test**. Point (3) turned out to be where our system had a methodological problem (§B.1).

### A.4 What the literature cannot answer

No published work uses our setting: automatic uncorrected tracking with 41% ball loss, a
segment-level IoU metric, a GT whose convention we only see for 17 possessions. Three questions
need our own experiments:
1. Which yardstick represents the private GT? (§B.1)
2. Does a learned temporal stage beat a fixed-cost Viterbi on our features? (§B.3)
3. Does decoding for expected spellF1 beat decoding for path probability? (§B.3)

---

## B. Current-system diagnosis

### B.1 The 2nd-half validation labels reward the rules' timing, not the GT's [measured]

`final_v3` / `final_v4` were chosen over `final_v2` on `val_h2` (four eye-labelled 5-minute
windows of the 2nd half), at a cost on all three real-GT yardsticks. Two new measurements say
`val_h2`'s official spellF1 is the wrong judge for that decision:

**Where H↔A changes land** (`analysis/boundary_offsets.py`; offset = predicted − labelled change,
frames, nearest same-kind change within 5 s):

| labels | n team changes | model (final_v2 decoder) median [IQR] | rules median [IQR] |
|---|---|---|---|
| public window, **real GT** | 12 | **+4** [−10, +8] | **−39** [−82, −14] |
| 2nd-half windows, **eye labels** | 31 | +1 [−22, +18] | **+0** [+0, +2] |

On the real GT the rules switch possession a median **1.6 s early** (at the pass, before the new
team's controlled touch); the model is centred. On the eye labels the rules' changes coincide
with the labelled ones to 0–2 frames. A 1 s offset alone takes a perfect spellF1 to 0.65
(EVALUATION.md), so a yardstick that puts H↔A changes at the rules' timing rewards anything that
pulls boundaries towards the rules.

**Where team errors happen** (`analysis/err_locus.py`; live frames binned by distance to the
nearest labelled change):

| window | model error <1 s / 1–3 s / >3 s | rules error <1 s / 1–3 s / >3 s |
|---|---|---|
| public (real GT) | 0.21 / **0.05** / 0.01 | 0.37 / **0.21** / 0.04 |
| 2nd half, win 1 (eye) | 0.20 / 0.11 / 0.03 | 0.14 / 0.08 / 0.02 |
| win 2 | 0.23 / 0.13 / 0.03 | 0.32 / 0.21 / 0.08 |
| win 3 | 0.39 / 0.13 / 0.00 | 0.15 / 0.09 / 0.00 |
| win 4 | 0.33 / 0.25 / 0.15 | 0.10 / 0.08 / 0.16 |

Mid-possession (> 3 s from any change) the model is as good as or better than the rules
everywhere. The "model drifts in the 2nd half" finding (§18, R10) is almost entirely **near
changes**, i.e. timing, and against the real GT the model's timing is the better one.

**The decisive check** (`analysis/rev_v4.py`, cached fold probabilities):

| decoder | bench | ext | masked | public | val_h2 (eye) |
|---|---|---|---|---|---|
| rules only | 0.319 | 0.355 | 0.284 | 0.461 | **0.652** |
| `final_v2` | 0.656 | 0.617 | 0.695 | 0.675 | 0.573 |
| **`final_v2` + stop gate** | **0.661** | **0.616** | **0.706** | **0.697** | 0.579 |
| `final_v4` (submitted) | 0.636 | 0.589 | 0.683 | 0.667 | 0.642 |

The rules alone, the weakest system on every real-GT yardstick, score **highest of all** on
`val_h2`. So `val_h2` spellF1 measures agreement with touch timing; it cannot choose boundary
placement. It stays useful for what does not depend on exact timing: mid-possession team errors,
dead-ball detection and the number of possessions. **Decision: go back to the `final_v2` decoder
(no rule vote) and keep the stop gate** (+0.025 bench, +0.030 public vs `final_v4`).

### B.2 Pipeline and what else the audit checked [measured]

Raw ATD → `mtah/features.py` (ball cleaning, distances, pace) → `mtah/rules.py` (touch episodes,
dead-ball machine; output used as a feature) → `mtah/feats.py` `full_plus3` (multi-scale ball-free
shape, ball/touch block, rule internals, pitch control, set-piece geometry) → `mtah/final.py`
(HistGradientBoosting, 600 trees, 9 external matches degraded to our noise + mirrors + noise
copies) → `mtah/decode.py` (3-state Viterbi, 12/12 nats, 1 s minimum dead, half-start kickoff,
stoppage gate).

- **Baseline reproduced** exactly from cached fold probabilities (`analysis/rev_v4.py`: every
  §19 number), and the delivered files' public scores and full-match spell counts
  (`analysis/score_files.py`: `final_v2` 0.700 / 282, `final_v4` 0.653 / 278).
- **Cached fold models are the ones behind `P_R1`** (re-predicted to 1e-7, `analysis/fold_probs.py`).
- **Where the error is** (`analysis/miss_by_len.py`): true possessions under 2 s are 20–30% of all
  spells on external data and are found ~19% of the time; 2–5 s ~50%; over 10 s 76–90%. Our
  public window has 2 of 17 under 2 s, both missed.
- **Second independent review** (the team's integrated report, verdicts in
  RECOMMENDATIONS_REVIEW.md, sixth feedback): its negative decoder arms (abstention, dwell,
  two-stage) and its HGB-vs-linear check (+0.094) are consistent with ours; its "retain `v4`"
  conclusion is not supported by its own ext + masked yardsticks.

## C. Ranked research opportunities

| rank | direction | rationale / evidence | expected impact | uncertainty | effort | validation | status |
|---|---|---|---|---|---|---|---|
| 1 | **Fix model selection: judge timing on real GT only** | B.1 | +0.025 bench, +0.03 public | low | hours | all yardsticks | **done → `final_v5`** |
| 2 | Learned temporal refiner (MS-TCN / GRU) | A.2, A.3; ext misses / drift | +0.01–0.03 | medium | 1 day | leave-fold-out ext + masked | **tested, not adopted** (ext +0.011–0.017, masked −0.02–0.03) |
| 3 | Segmental MBR for expected spellF1 | A.2; metric ≠ path probability | +0.005–0.01 | medium | hours | all yardsticks | **tested, not adopted** (+0.001) |
| 4 | Short-possession recall without new spurious spells (R17) | B.2: the largest error mass left | up to ~+0.05 if half recovered | high | days | ext by duration bin + masked | open; needs a touch-level signal the trees lack, e.g. ball-flight / reception features |
| 5 | More labelled matches in our GT convention | every learned stage is limited by 9 matches and their conventions (§23) | unknown | high | data access | ext CV | blocked: no ready source (PFF sign-up; SkillCorner has no per-frame possession) |
| 6 | GT-convention calibration of the eye labels: a person labels the public window blind, then compare with the real GT | would turn `val_h2` back into a full yardstick | enables better selection, no direct gain | low | ~1 person-hour | eye vs GT spellF1 and offsets | recommended (team task; by a person, never an LLM) |
| — | Ball-free deep possessor models (PathCRF, Kim 2023) | reported 92.4% team accuracy without ball ≈ ours | small | — | high | — | deprioritised: our shape model is already at that level |
| — | Video ball, Smart Tagging, CatBoost, pseudo-labelling, pitch-control variants | measured earlier (INVESTIGATION.md §9–15) | ≤ 0 | — | — | — | closed |

## D. Experiments run in this study (all [measured], cached probabilities, aggregates only)

| # | experiment | script | bench | ext | masked | public | verdict |
|---|---|---|---|---|---|---|---|
| 0 | `final_v4` (submitted at start) | `rev_v4.py` | 0.636 | 0.589 | 0.683 | 0.667 | — |
| 1 | `final_v2` decoder + stop gate | `rev_v4.py` | **0.661** | 0.616 | **0.706** | **0.697** | **adopted (`final_v5`)** |
| 2 | segmental MBR, T 1 / 2 | `mbr_eval.py` | 0.662 / 0.661 | 0.620 / 0.621 | 0.705 / 0.702 | 0.697 | no |
| 3 | GRU refiner, best setting | `refine_eval.py` | 0.654 | **0.633** | 0.675 | 0.640 | no |
| 4 | TCN refiner, per-fold inputs, best setting | `refine_eval2.py` | 0.655 | 0.627 | 0.683 | 0.650 | no |

Failure analysis for 3–4: identical behaviour with averaged and per-fold inputs rules out the
input mismatch; the gain appears only where training and test share a label convention (external
CV), so a sequence stage trained on 9 external matches learns their conventions. MBR: with tuned
costs the expected metric is flat across cost scales, so per-stretch choice has nothing to pick.

Delivered full-fit file `final_v5.csv` (`analysis/score_files.py`): public spellF1 **0.723** /
macroF1 0.910, 14 spells (true 17); full match 269 possessions; mid-possession team error on the
2nd-half windows 4.4% (`final_v4` 4.3%). Byte-identical across two runs of `run.py`.

## E. Final implementation

`mtah/final.py` `DECODER`: `use_rules` False (rule vote removed), `stop_gate` 100 kept; nothing
else changed. `uv run python run.py --name final_v5` → `evaluation/results/final_v5.csv`
(sha256 `4089da237ca8780a…`), validated by `evaluation/validate_submission.py`. New tools kept but
unused by the submission: `mtah/mbr.py`, `mtah/refiner.py`, `analysis/yardsticks.py`.

## F. Final recommendations

- **What limits performance**: (1) short possessions (< 5 s, ~half of all spells) are mostly
  missed; (2) every learned component is limited by 9 external matches whose conventions differ
  from ours; (3) our only in-domain yardsticks are 17 real-GT possessions and eye labels that
  cannot judge timing.
- **What improved results**: selecting on real-GT timing (`final_v5`, +0.025 bench, +0.07 public on
  the delivered file vs `final_v4`).
- **What failed**: MBR decoding, GRU and TCN refiners (this study); video ball, CatBoost,
  pseudo-labelling, dwell, boundary head, Smart Tagging (earlier).
- **Next, in order**: (1) the blind eye-label calibration of the public window (cheap, settles what
  `val_h2` can judge); (2) R17 short-possession recall via ball-flight / reception features, judged
  per duration bin; (3) more labelled matches if a source with our convention appears. Code freeze
  14 Nov.

