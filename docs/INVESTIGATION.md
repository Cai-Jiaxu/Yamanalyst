# Investigation: the evidence behind every decision

**The single source of truth for measured numbers.** Every number below was produced by a script
in `analysis/` and can be re-run; only aggregates were printed, no tracking rows left this
machine. Tags: **[measured]** run on our data, **[reported]** published by others and not
reproduced, **[hypothesis]** untested. Open work and decisions: [ROADMAP.md](ROADMAP.md);
verdicts on outside advice: [REVIEWS.md](REVIEWS.md); the submitted write-up:
[../Yamanalyst.md](../Yamanalyst.md). Organiser documents are in [../challenge/](../challenge/).

## 0. Prior work and competition precedents (2026-10-25 review)

### 0.1 Direct precedents: possession and game state from tracking data

| work | problem / method | data | reported result | what transfers to us |
|---|---|---|---|---|
| Borghesi, Costa, Morra, Lamberti, *Using Temporal Convolutional Networks to estimate ball possession in soccer games*, Expert Syst. Appl. 223 (2023) 119780, doi:10.1016/j.eswa.2023.119780 ([repo page](https://iris.polito.it/handle/11583/2976470)) | **Our exact 3-class task** (dead / home / away) from tracking. Per-object TCN embeddings, combined by permutation-invariant layers (sum, self-attention or 2-D conv); ternary vs two binary heads | professional tracking data (full text not reachable from here) | [reported] 86.2% possession accuracy (+7.3 over rule-based), 89.2% dead/alive (+33.2) | Confirms that learned models beat hand rules by a wide margin on dead/alive, which is also what we see (rules 0.36 ext vs model 0.62). Frame accuracy only: no segment metric, so it says nothing about spellF1-style over-segmentation |
| Kim et al., *PathCRF: ball-free soccer event detection via possession path inference* (arXiv 2602.12080, 2026) ([paper](https://arxiv.org/html/2602.12080), [code](https://github.com/hyunsungkim-ds/pathcrf)) | possession as a path over a player graph; set-attention + temporal attention encoder; **dynamic CRF with masked illegal transitions, Viterbi decoding** | IDSSE / Sportec Open DFL, 7 matches (5 train / 1 val / 1 test), 5 fps, in-play only | [reported] frame-level **team possession accuracy 92.4% with no ball** (1 test match); event F1 75.7%; Viterbi on a non-CRF model already gives 73.1 vs 65.3 without | Same training data as ours. Our ball-free shape model sits at 88–93% team accuracy without the ball on our public window, so a much heavier ball-free network is unlikely to move team accuracy much. Their ablation supports what we already do: structured decoding is worth more than the encoder |
| Kim et al., *Ball trajectory inference from multi-agent sports contexts using Set Transformer and hierarchical Bi-LSTM*, KDD 2023 ([arXiv 2306.08206](https://arxiv.org/abs/2306.08206), [code](https://github.com/pientist/ballradar)) | predict the possessing **player**, then regress the ball path conditioned on it | Metrica sample games + others | [reported] see paper; designed to fill missing ball | The possessor-first hierarchy is what our `ball_block` + shape features approximate. Imputing the ball would feed our rules a guessed ball; §9 / §12 showed even a *true* ball restored in gaps adds only +0.005 bench, so imputation is low value for us |
| Vidal-Codina, Evans, El Fakir, Billingham, *Automatic event detection in football using tracking data*, Sports Engineering 25:18 (2022) ([arXiv 2202.00804](https://arxiv.org/abs/2202.00804)) | deterministic decision tree: possession step, then events from possession changes and laws of the game | several providers and tournaments, clean tracking with ball | [reported] > 90% detection in most event categories | A good blueprint for touch/possession rules **on clean tracking**. Our rules follow the same idea and score 0.61 spellF1 on clean external tracking but 0.37 once degraded to our noise (§6): with 41% ball missing, rule systems alone are not enough |
| Link & Hoernig, *Individual ball possession in soccer*, PLOS ONE 12(7) e0179953 (2017); Hoernig et al. 2016 | Bayesian network / physics + ML for degrees of ball control | Bundesliga tracking | [reported] 80–92% precision/recall on clean data, ~4 points lower on uncorrected data | Supports CHALLENGE.md's "controlled possession" notion: a touch is not possession. Our GT measurements agree (§2) |
| Bassek et al., IDSSE / Sportec Open DFL dataset, Sci. Data (2025), doi:10.1038/s41597-025-04505-y | 7 Bundesliga matches with synchronized tracking, events, per-frame `BallPossession` / `BallStatus` | — | — | Our main training set (7 of 9 matches). Its possession convention counts more brief touches (4.3/min vs 3.4/min in our GT, §6) |

### 0.2 Transferable methods: temporal segmentation with an over-segmentation penalty

| work | idea | evidence | relevance |
|---|---|---|---|
| Farha & Gall, *MS-TCN*, CVPR 2019 ([arXiv 1903.01945](https://arxiv.org/abs/1903.01945)); MS-TCN++ ([2006.09220](https://arxiv.org/abs/2006.09220)) | stacked dilated TCN stages, each refining the previous one's frame probabilities; truncated-MSE smoothing loss against flicker | [reported] state of the art on 50Salads / GTEA / Breakfast at the time, with segmental F1@{10,25,50}, a metric of the same family as spellF1 | Direct template for a **learned refiner on top of our tree probabilities** (tested, §22–23) |
| Ishikawa et al., *ASRF*, WACV 2021 ([arXiv 2007.06866](https://arxiv.org/abs/2007.06866)) | frame classifier + boundary-regression branch; boundaries suppress over-segmentation | [reported] up to +16.1 segmental F1 | Our "boundary head" (`w_bound`) is this idea; measured ±0.001 on our bench (§8). Not repeated |
| Segmental minimum-Bayes-risk decoding, Goel & Byrne, IEEE TSAP (2004) | choose the output that maximises the **expected task metric** under the posterior, stretch by stretch | [reported] consistent WER gains in ASR | Our metric is a segment F1 and Viterbi optimises path probability, not spellF1. Tested, §21 |
| HSMM / explicit-duration models | duration distributions instead of geometric | textbook | Our minimum-dead dwell inside Viterbi was measured as no change (§11); not repeated |

### 0.3 Kaggle and competitive precedents

| competition | what won | what transfers | what does not |
|---|---|---|---|
| Child Mind Institute – Detect Sleep States (2023): per-step state segmentation of long accelerometer series, event AP with tolerances | [reported, secondary source: a competitor's summary deck] 1st place used a 1-D UNet + GRU; post-processing tuned to the metric added +0.02 private | Sequence heads over per-step features beat per-step GBDTs; **post-processing aligned with the metric matters as much as the model** | 100s of training series there vs 9 matches here; their labels were the same convention as the test |
| DFL – Bundesliga Data Shootout (2022): event spotting (play / challenge / throw-in) from video | write-ups found were 6th place and below; video CNNs + temporal heads | — | Video-only, different target; our video ball channel was already measured and closed (§9–12) |
| NFL Player Contact Detection (2023): contacts from tracking + video | [reported, secondary] GBDT on tracking features to filter, 3-D CNN on video, GBDT post-processing | Tracking-feature GBDTs are a strong base stage; a second stage over first-stage outputs is standard | Pairwise contact, not segment-level possession |

**Common lesson** across these: (1) a strong per-step model, (2) a temporal stage, (3) decoding and
post-processing tuned to the official metric on a validation set with the **same label convention
as the test**. Point (3) turned out to be where our system had a methodological problem (§20).

### 0.4 What the literature cannot answer

No published work uses our setting: automatic uncorrected tracking with 41% ball loss, a
segment-level IoU metric, a GT whose convention we only see for 17 possessions. Three questions
need our own experiments:
1. Which yardstick represents the private GT? (§20)
2. Does a learned temporal stage beat a fixed-cost Viterbi on our features? (§21–23)
3. Does decoding for expected spellF1 beat decoding for path probability? (§21–23)

---

## 1. What the metric rewards (`analysis/ceiling.py`)

Each kind of error was applied to the true labels and the cost measured. Results were checked on
the public 5 minutes and on synthetic full matches resampled from them, about 350 possessions each.

| error | spellF1 (synthetic full match) |
|---|---|
| perfect | 1.000 |
| miss every possession < 1 s / < 2 s / < 3 s | 0.935 / 0.879 / 0.828 |
| every transition jittered σ = 0.25 / 0.5 / 1 / 2 s | 0.969 / 0.919 / 0.836 / 0.702 |
| … only alive↔dead jittered, σ = 1 s / 2 s | 0.977 / 0.950 |
| … only H↔A jittered, σ = 1 s / 2 s | 0.868 / 0.761 |
| **one spurious 0.4 s blip per minute** | **0.773** |
| two blips per minute | 0.623 |
| never predict D, but teams perfect | 0.831 |

**Conclusions**
1. **Spurious switches are the most expensive error.** One false flip per minute costs as much as
   missing every possession under 3 s. Precision on change of possession beats recall.
2. **H↔A timing matters, alive↔dead timing barely does.** Spend effort on placing changes of
   possession, not on the exact frame a stoppage starts.
3. **Dead-ball detection is worth about 0.17.** Without it, possessions on either side of a
   stoppage merge or extend.

## 2. What the ground truth means (`analysis/conventions.py`, `analysis/diagnose.py`)

| finding | evidence |
|---|---|
| **D → alive happens on the frame the ball starts moving** | ball speed crosses 3 m/s at offset **0 frames** in 4 of 4 restarts |
| **H↔A happens at the new team's first touch** | the new team is within 1.5 m of the ball at offset 0 / −1 / 0 / −5 / 0 in most switches |
| A touch without control does not switch possession | inside a 41 s H possession, an A player is on the ball for ~0.7 s and clears it: GT stays H |
| The raw tracking mislabels team at times | GT says H while an A slot is 0.7 m from the ball for 2 s. The corrected tracking fixed this and the raw data did not |
| Stoppages exist with the ball tracked and never out of bounds | a 20 s GT stoppage with the ball "seen" throughout. Both teams walking at 1.0–1.5 m/s is the signal |
| Nearest player to the ball agrees with GT only 78% of the time | 91% when someone is within 1 m, 60% at 3–5 m |
| The ATD ball speed column is empty 90% of the time | derive speed from positions |

So the target is: **the team of the first touch of a controlled possession, with stoppages
running from the ball going dead until the restart kick.**

## 3. The missing ball is the real battle

- 85% of missing-ball frames are in **gaps longer than 10 s** (79 such gaps). Short-gap
  interpolation recovers almost nothing.
- Ball visibility per 5-minute block falls from **88% at the start of the first half to 20–50%
  later**. **The public window is the best-tracked part of the match**, so public scores are
  optimistic. That is why the baselines drop from 0.33 public to 0.24 on the full match.
- Players move at 1.0 m/s on average when the ball is missing and 2.0 m/s when it is seen, so
  a large share of missing-ball time is stoppages.
- **Masked validation** (`mtah/masking.py`) hides the ball in 4–40 s stretches covering 45% of
  the public window, which imitates the full match. The rule system falls from **0.46 to 0.26**
  under masking. The private score is decided here.
- **Possession can be read without the ball.** A model on player shape and motion alone (team
  centroids, spread, defensive lines, forward motion, pressing distances, camera pan), trained
  only on pseudo-labels, gets the team right on **88% of live public frames without the
  ball**, and 81.5% three-class including D. The rules with the ball get 85.6%.
- The tactical camera's centre is a weak ball proxy: 12 m median x error, correlation 0.83.
  Useful as a feature, not as a ball position.

## 4. The theoretically best solution for this setup

From all of the above, the ideal system is:

1. **Touch-level event detection** wherever the ball is seen: first touch, control confirmation,
   restart kick. This gives touch-accurate H↔A boundaries, which is where spellF1 is most
   sensitive.
2. **A ball-free possession and stoppage model** for the long gaps, reading the players.
3. **A structured decoder** (HMM/HSMM) that fuses the two. It has explicit switching costs, because
   blips are the most expensive error, and it keeps touch-level boundaries where the ball is
   seen.
4. **Better ball coverage from video**: run a ball detector on the tactical video to shrink the
   41% gap. This is the single biggest remaining lever, because every downstream step works far
   better when the ball is seen. It is blocked on the video files. *(Since built and measured, §9:
   as a drop-in ball it lowers the bench. ROADMAP.md R2 measures whether a dedicated channel can
   pay.)*
5. **A validation set big enough to choose between versions.** The 17 public possessions give
   ±0.04–0.07 between masking seeds. Hand-labelled validation windows from the video (allowed by
   RULES.md) are the fix, and are also blocked on the video. *(Videos present since 2026-10-21;
   now ROADMAP.md R3, a team task.)*

Upper bound without video: on the public window the pipeline already recovers 16 of 17
possessions. The remaining loss is boundary timing (~1–3 s late on some switches) and the
missing-ball regime.

## 5. Workflows and status

| # | workflow | status | result |
|---|---|---|---|
| W1 | Environment, data cache, scorer harness, reproducible `run.py` | **done** | 6 s cache build, byte-identical reruns |
| W2 | Metric ceiling and GT-convention investigation | **done** | §1–3 |
| W3 | Rule system: touch episodes, control, dead-ball machine | **done (v2)** | public 0.461 / masked 0.26 |
| W4 | Ball-free shape model on pseudo-labels | **done** | 88% team accuracy with no ball |
| W5 | Viterbi fusion decoder | **done**, costs tuned on external labels | 12 / 6, mid-plateau |
| W6 | Ball+shape model, bagging, capacity | **done: hit the noise floor** | ball columns add nothing once the leak was fixed (they double-count the ball). Capacity beyond 300 trees is flat. Bagging kept for variance |
| W7 | Video ball detection: self-supervised YOLO on Colab | **done, not adopted** | §9: detector works (79% top-1 within 2 m), but the merged track lowers the bench (0.656 → 0.642/0.646); a DP linker too (0.629–0.635, §9.1) |
| W8 | External tracking data: IDSSE + Metrica sample (PFF needs a personal sign-up form, skipped) | **done** | §6: masked 0.35 → 0.47 |
| W9 | Technical document and public repo hygiene | **in progress** | draft `Yamanalyst.md`; ROADMAP R9, R23 |

## 6. W8: external tracking data

**Sources** (`scripts/download_external.sh`, parsed by `mtah/external.py`):

| source | matches | labels | licence |
|---|---|---|---|
| IDSSE (Bassek et al., *Sci Data* 2025, doi:10.1038/s41597-025-04505-y) | 7 Bundesliga 2022/23 | per-frame `BallPossession` and `BallStatus` from DFL's corrected tracking: the same construction as our GT | CC BY 4.0, fetched via the figshare API |
| Metrica Sports sample data, games 1–2 | 2 | built from events: team of each pass, recovery, set piece or shot. Challenges and "ball lost" do not switch. Dead runs from ball out, shot out or foul until the restart | open, attribution requested |
| PFF FC World Cup 2022 | — | — | access needs a personal sign-up form, so not used |

1.29 M labelled frames in total, about 3,000 possessions, against 17 in our public window.

**Do the conventions match ours?** On boundary timing, yes. Every source switches team at the
new team's touch and restarts on the frame the ball moves (median offset 0 frames; within 5
frames for 89–100% of boundaries on clean tracking). On *how many* touches count as a
possession they differ. Metrica's event labels give 3.4–3.7 possessions/min, close to our 3.4.
IDSSE gives 4.3/min (mean 7.7 s vs our 14.2 s), so it counts more brief touches. IDSSE's dead
share (41–52%) agrees with our pipeline's estimate for our match (42–45%).

**Noise simulation** (`mtah/noise.py`): clean external tracking is degraded to look like our
ATD, calibrated at runtime against our match's aggregates. Ball dropped to our 41% using our
actual gap-length distribution, with gaps starting 3× more often during dead ball. Players
dropped to our ~20.5 tracked. Team mislabels added at 0.5 per minute, plus 0.25 m position noise.
Seeds come from the match name, so the result does not depend on which other matches are present.

**Results**

| | spellF1 |
|---|---|
| rules on clean external tracking (3 matches) | 0.61 |
| rules on the same, degraded | 0.37 |
| rules + shape fusion on the same, degraded | 0.42 |

| shape model trained on … | team accuracy, our public window, no ball | our masked spellF1 / macroF1 |
|---|---|---|
| our own rule pseudo-labels (`fused_v2`) | 0.83 | 0.346 / 0.789 |
| **external true labels** (`fused_ext`) | **0.93** | **0.465 / 0.844** |
| average of both | — | 0.463 / 0.828 |
| one model on both | — | 0.463 / 0.840 |

**Rule parameters checked against real labels** (`analysis/rules_external.py`). Each parameter
was swept one at a time on all 9 degraded external matches. Every value stays within ±0.03 of
the defaults on both sources. The defaults are physically motivated, they hold on about 3,000
real possessions as well as on our 17, and they are unchanged. Under noise the rules find only
61% (IDSSE) and 71% (Metrica) of the true possessions, which is what an anti-blip design should
do. The decoder is where recall is recovered.

**Decoder penalties checked against real labels** (`analysis/decoder_tune.py`). Each external
match is scored with a model trained on the other eight:

| team / dead cost | IDSSE (7) | Metrica (2) | ours public | ours masked |
|---|---|---|---|---|
| 4 / 3 | 0.445 | 0.523 | 0.422 | 0.471 |
| 8 / 6 | 0.454 | 0.529 | 0.434 | 0.485 |
| **12 / 6** | **0.455** | **0.533** | **0.434** | **0.485** |
| 16 / 6 | 0.456 | 0.533 | 0.434 | 0.485 |
| 16 / 12 | 0.464 | 0.545 | 0.434 | 0.474 |
| 24 / 12 | 0.462 | 0.551 | 0.434 | 0.473 |

Team cost 8–16 with dead cost 6 forms a plateau on all four yardsticks. The decoder uses 12 / 6,
from the middle of it. Heavier dead-ball costs help the external sources slightly and hurt ours
slightly, so they are left alone.

The external-trained model is the clear win: +0.12 masked spellF1, against a between-seed
spread of ±0.056 over 16 seeds. Its three-class accuracy without the ball (0.864) beats the
rule system *with* the ball (0.856). The three external variants tie, so the pipeline uses the
simplest one: external true labels only.

## 7. Submission history

> **Superseded.** The current submission is `final_v5` (§24: public spellF1 0.723, bench 0.661).
> This section records the reasoning at the `fused_ext_v2` stage.

| | public spellF1 | public macroF1 | **masked spellF1** (full-match proxy) | full-match possessions |
|---|---|---|---|---|
| `rules_v2` | **0.461** | **0.852** | 0.26 | 305 |
| `fused_v2` (shape model on own pseudo-labels) | 0.400 | 0.808 | 0.35 | 334 |
| **`fused_ext_v2`** (default `run.py`: shape model on external true labels, decoder 12 / 6) | 0.434 | 0.841 | **≈0.485** | 322 |

`fused_ext_v2` is the submission. It is best under masking by a wide margin, and its decoder and
rule parameters are now checked against ~3,000 real external possessions rather than 17.
The masked figure is from the single-model sweep in `analysis/decoder_tune.py`; the submission
averages 5 models.

The earlier reasoning still holds for `rules_v2` against the fused versions. `rules_v2` wins on
the public window and the fused versions win under masking. The gap on the public
window is 1–2 possessions out of 17, which is noise. The gap under masking is 0.09 across 16
seeds, and the full match looks like the masked case: about 41% of the ball is missing, against
about 10% in the public window. This is the public-window trap
EVALUATION.md warns about, applied to our own versions.

Regenerate it with:

```
bash scripts/download_external.sh && uv run python -m mtah.external && uv run python run.py
```

The output is byte-identical across runs (sha256 `029547eb…`).

Baselines for reference: `baseline_smooth50` public 0.326 (full match 0.243), `baseline_smooth125`
public 0.424 (full match 0.224).

## 8. Rounds 1–2: learned fusion, augmentation, leak fixes (`mtah/bench.py`)

**Benchmark.** `bench` is the mean of two numbers:
- spellF1 on 9 external matches noised to look like ours. They are split into 3 folds by match,
  and each fold is scored by a model trained on the other two.
- our masked spellF1, from 16 seeds.

The masked test hides the ball in stretches placed preferentially when players are walking. It
also hides the ball for 30 s after the window, so look-ahead features cannot peek. Each run
prints an error breakdown: split, merge, boundary drift, missed and spurious possessions.

| step | bench | ext | masked | public |
|---|---|---|---|---|
| shape-only + rule fusion (`fused_ext_v2`) | 0.469 | 0.469 | 0.468 | 0.434 |
| + longer context (±5/10 s, lead–lag trends) | 0.483 | 0.496 | 0.470 | 0.485 |
| + ball/touch features + **rule label as a feature, no rule term in the decoder** | 0.530 | 0.542 | 0.518 | 0.511 |
| E6 = both, decoder retuned (emission weight 0.75, dead cost 12) | 0.575 | 0.574 | 0.576 | 0.625 |
| + rule internals, 600 trees (E7) | 0.590 | 0.586 | 0.595 | 0.648 |
| + k-nearest, approach speed, pitch control, slot-speed clipping (E8) | 0.595 | 0.589 | 0.600 | 0.634 |
| + left/right and home/away mirrors (E9) | 0.634 | 0.612 | 0.657 | 0.632 |
| leak fixes (half gaps, edge clamping), mirrors + noise copies (R0) | 0.649 | 0.621 | 0.677 | 0.662 |
| + minimum dead-ball length 1 s | 0.653 | 0.622 | 0.684 | 0.662 |
| **+ set-piece geometry (R1), the submission** | **0.656** | 0.617 | **0.695** | **0.675** |

**Measured and not adopted:**

| change | effect |
|---|---|
| boundary head modulating switch costs | ±0.001 |
| EM label-shift prior | −0.017 |
| IDSSE relabelled to our convention | −0.002 |
| lower switch cost | ±0.001 at 6–9; −0.013 at 18 |
| boundary snapping | 0 |
| kickoff evidence | −0.029 → −0.008 after two guards |

The kickoff detector finds both of our half-start kickoffs within 5 frames and the after-goal
kickoffs with the right team. The harm comes from external data, whose tracking starts at the
kick, so an after-goal kickoff gets mistaken for a half's opening one. Possible gain on our match
is about 5 possessions; the decoder does not use it.

**Where the remaining error is** (external, per 100 true possessions): missed 22, boundary
drift 19, merges 11, spurious 3.6, splits 2.1, predicted/true 0.71. The decoder is conservative
on purpose; a large part of "missed" is IDSSE's brief possessions, which our ground truth would
not count.

**Video (W7, unblocked; superseded by §9).** Off-the-shelf COCO YOLO with tiling finds the ball within 2 m on 59% of
frames where the tracker sees it. On genuine missing-ball frames its top candidate is plausible
only 18% of the time. `mtah/video_ft.py` fine-tunes a detector on this camera using ATD-projected
labels (the projection is unbiased: +0.5 / +0.8 px). It runs on Colab GPU via
`colab/ball_detector.ipynb`, with data in the team's Google Drive and Gemini off.

## 9. Video ball detection (W7): built, measured, not adopted

`mtah/video_ft.py` + `colab/ball_detector.ipynb`, run on a Colab A100 with the data in the team
Drive (Gemini off).

| | COCO YOLO (tiled) | fine-tuned on ATD-projected labels |
|---|---|---|
| held-out frames, top candidate within 2 m of the ATD ball (when present) | 51% | **79%** |
| genuine missing-ball frames, plausible top candidate | 18% | **26%** |

First attempt: ball-sized label boxes (~9 px) stalled training at val mAP50 ~0.03, because the
ATD position itself lands ~9 px from the ball. 1.2 m region labels fixed it (mAP50 0.27).
Detection on all 71,602 missing-ball frames took 85 min. After the plausibility filter
(`merge_track`) the track covers 9.7% (conf ≥ 0.25) or 14.1% (conf ≥ 0.15) of missing graded
frames.

**Effect on the submission model** (`analysis/video_score.py`: fold models fitted once, our
match re-predicted with the video track attached):

| | bench | masked | public spellF1 / macroF1 |
|---|---|---|---|
| `final_v1`, no video | **0.656** | **0.695** | 0.675 / 0.901 |
| + video, conf ≥ 0.25 | 0.642 | 0.666 | 0.675 / 0.902 |
| + video, conf ≥ 0.15 | 0.646 | 0.676 | 0.675 / 0.884 |

This is worse even though the masked test favours video (its hidden frames are ones the ATD saw).
The model was trained on external data where a ball position is exact or absent. Video
detections are neither: ~1 in 5 top picks is wrong and the rest are good to ~2 m. Feeding them
through the same channel as the ATD ball misleads the features. A fix would be a separate
"video ball" channel plus video-like noisy balls simulated in the external training data. That
is real work with an uncertain payoff, so `final_v1` stays.

### 9.1 Better linking does not rescue it (`analysis/video_audit.py`, `video_link.py`, `video_link_bench.py`)

Two external reviews blamed the greedy `merge_track` (one false positive blocks the true ball for
15 frames; thresholds set for COCO-scale confidences). We checked both claims before rebuilding
anything.

**What the detections look like** (public window, frames where the ATD sees the ball):

| | |
|---|---|
| some candidate within 2 m / top-1 within 2 m (given a candidate) | 70.8% / 81.6% |
| true candidate is ranked #1 when present | 97.3% |
| median confidence of the true candidate; share below 0.25 | 0.13; 79% |
| wrong top-1 picks outside the pitch | 3.4% |
| true candidate ≥ 4 m from every player | 22% |
| frames with two candidates < 0.8 m apart (tile / nested-box duplicates) | 79% |

So the 0.25 threshold throws away most correct detections (the 1.2 m region labels compress
YOLO's IoU-aware confidence), and re-ranking the top 5 can gain at most ~2 points.

**A DP tracklet linker** (`analysis/video_link.py`) over all deduplicated candidates replaces
the greedy filter. Scored on the public window with the ATD ball hidden (8 seeds; optimistic,
because those frames were in the detector's training pool and the ball is visible on them):

| track | hidden frames filled | within 2 m when filled | correct coverage |
|---|---|---|---|
| greedy, conf ≥ 0.25 (current) | 12.5% | 89.8% | 11.3% |
| greedy, conf ≥ 0.05 | 66.3% | 87.1% | 57.7% |
| DP linker, c0 = 0.08 | 75.2% | 84.5% | 63.6% |
| DP linker, c0 = 0.05 | 88.9% | 81.2% | 72.1% |

On genuine missing graded frames the linker fills 39% against 10% for the current filter.

**Effect on the submission model** (`analysis/video_link_bench.py`, same method as above):

| | missing filled | bench | masked | public spellF1 |
|---|---|---|---|---|
| `final_v1`, no video | — | **0.656** | **0.695** | **0.675** |
| greedy, conf ≥ 0.25 | 9.7% | 0.642 | 0.666 | 0.675 |
| DP linker, c0 = 0.12 | 25.3% | 0.635 | 0.653 | 0.662 |
| DP linker, c0 = 0.08 | 38.1% | 0.629 | 0.641 | 0.662 |
| greedy, conf ≥ 0.05 | 39.2% | 0.622 | 0.627 | 0.642 |

**The more video ball, the worse the score, public window included.** At equal coverage the
linker beats greedy (0.629 against 0.622), so its precision is real, but the sign does not change.
The cause is the train/test mismatch above: the model learned "exact or missing" and gets ~15%
wrong positions passed off as exact. Sparse video points inside a gap also reset `since` /
`until` / `seen`, which the external training data (solid NaN blocks from `mtah/noise.py`) never
shows.

Side check (`analysis/video_jitter.py`): low-threshold tracks fake > 3 m/s on 17–24% of
stationary-ball frames (4% after a 15-frame Savitzky–Golay smoother). Video lowers the rules'
dead-ball share from 29% to 14–17% (true 19.6%), trading recall (81% → 60–69%) for precision
(57% → 81–85%). That is a net gain for the rules, not the cause of the model's loss.

**What would have to change before video can help**, in order:
1. A separate video-ball channel (own columns and own `seen`), with video-like tracks simulated
   in the external training data: clumpy tracklets, ~85% within 2 m, **and** the ~15% false
   positives. Needs a full refit.
2. Retrain the detector with the public window (frames 11185–19435) out of its training pool;
   until then every masked score with video in it is optimistic.
3. A small hand-labelled sample of genuine missing-ball frames (by a person, never an LLM), the
   only way to measure precision where it matters.

Detector-side upgrades (centred label boxes, overlapping 4×2 tiles with cross-tile NMS, 3-frame
input, hard-example mining) are deferred until 1 is in place: a better detector cannot help a
model that is misled by its output.

## 10. Dead ball before the 2nd-half kickoff (ROADMAP R1, `analysis/kickoff_modes.py`)

FAQ.md: the 2nd half is graded from frame 110,808 but kicks off at 113,742, so those 2,934 frames
are all `D`. `final_v1` had **19 label spells** there (32% H/A): about 10 spurious possessions,
the costliest error type (§1), invisible to the bench because no external match and no public
frame comes before a half's kickoff.

Two fixes, both algorithmic:
1. **The kickoff detector found our 2nd-half kickoff 29 s early**, with the kick unseen: the
   pre-kick formation breaks for 1–2 s several times (a player over the line, slots dropping out),
   and the detector stopped at the first break. Bridging breaks under 2 s (`kickoff.BRIDGE`)
   finds it **5 frames before 113,742, with the kick seen**.
2. **A `kickoff='half'` decoder mode**: dead-ball evidence before a half's first kickoff only, and
   only for a waiting period of 10 s – 3 min. Without that bound, external "first" kickoffs that
   are really restarts after goals (up to 35 min into a half) got a dead ball imposed before
   them: IDSSE 0.608 → 0.587. No team evidence after kickoffs (the part that cost −0.008).

| decoder (cached `P_R1`) | bench | ext | masked | public |
|---|---|---|---|---|
| `kickoff=False` (final_v1) | 0.656 | 0.617 | 0.695 | 0.675 |
| `kickoff='half'`, unbounded wait | 0.648 | 0.601 | 0.695 | 0.675 |
| **`kickoff='half'`, wait 10 s – 3 min (final_v2)** | **0.656** | 0.617 | 0.695 | 0.675 |
| `kickoff='all'` (with bridging) | 0.648 | 0.601 | 0.695 | 0.675 |

**`final_v2`** = `final_v1` + bounded `'half'` mode. It changes exactly 942 frames, all inside the
pre-kickoff stretch, which goes from 19 spells to 2 (the last 5 frames before the detected kick
are not forced). Full-match possessions 292 → 282; public spellF1 unchanged at 0.700. Adopted
on "no harm on the bench + fixes a documented error", the rule ROADMAP.md set for R1.

## 11. Minimum dead-ball length inside Viterbi (ROADMAP R7, `analysis/dwell.py`)

`decode.viterbi_dwell` makes every dead spell at least k frames long inside the search (a chain
of k D sub-states), instead of absorbing short spells afterwards (`min_dead`). Fixed grid on
cached `P_R1`, no search:

| decoder | bench | ext | masked | public |
|---|---|---|---|---|
| `final_v2` (post-hoc `min_dead` 25) | 0.656 | 0.617 | 0.695 | 0.675 |
| dwell 25 in Viterbi | 0.656 | 0.617 | 0.695 | 0.675 |
| dwell 50 in Viterbi | 0.656 | 0.617 | 0.695 | 0.675 |
| dwell 25, dead cost 9 | 0.654 | 0.616 | 0.693 | 0.675 |
| dwell 25 + post-hoc 25 | 0.656 | 0.617 | 0.695 | 0.675 |

**No change** (IDSSE +0.001 at most). The post-hoc rule already removes the short dead spells
the in-search version prevents. Not adopted; the option stays in `decode.py` (`dwell`).

## 12. What is less missing ball worth? (ROADMAP R2, `analysis/ball_value.py`)

The decision gate for a dedicated video channel. The `final_v1` fold models (cached in
`cache/models_final_v1.joblib`) score external matches whose ball gaps are restored whole (true
ball + the usual 0.25 m noise), so the input stays "exact or missing", as in training. This is an
**upper bound**: a perfect detector that the model needs no retraining for.

| ball missing: ext / masked | bench | ext | masked | ext errors per 100: missed / drift / spurious |
|---|---|---|---|---|
| 46% / 45% (as now) | 0.656 | 0.617 | 0.695 | 22.1 / 19.4 / 3.6 |
| 30% / 25% | 0.661 | 0.641 | 0.681 | 20.6 / 18.7 / 3.3 |
| 17% / 10% | 0.667 | 0.664 | 0.671 | 19.2 / 18.1 / 3.1 |

- **On external matches the ball is worth about +0.0016 spellF1 per point of missing ball
  recovered**: +0.024 for 46% → 30%. That is roughly what the DP linker's coverage (38% of
  missing frames) would give *if every position were right and in-distribution*.
- **On our window it goes the other way**: hiding less ball scores lower (0.695 → 0.671), in line
  with public (10% missing, 0.675) sitting below masked. The masked mean has a standard error of
  about 0.014 (16 seeds, sd ±0.056), so this is suggestive, not proven; it is one more reason to
  get the R3 labels.
- **Decision: video channel (R6) closed.** The bench gain at the linker's coverage is +0.005
  even for a perfect ball, under the 0.01 bar ROADMAP set. A real channel would see ~15% wrong
  positions and need a full simulation and refit, for a fraction of that ceiling.

## 13. Round 3 fast screen: R8 velocity pitch control, R5 CatBoost (`analysis/round3.py`)

Screen tier (ROADMAP rule 7): 9 base matches, no mirror or noise copies, 300 trees. It only drops
clear losers (more than 0.005 below the same-tier baseline).

| screen | bench | ext | masked | public | verdict |
|---|---|---|---|---|---|
| `full_plus3` (baseline) | 0.569 | 0.589 | 0.549 | 0.606 | — |
| R8 `full_plus4` (+ velocity-projected pitch control) | **0.583** | 0.595 | 0.571 | 0.576 | full fit |
| R5 CatBoost (depth 6, lr 0.1, untuned) | 0.545 | 0.572 | 0.517 | 0.621 | dropped |
| R5 HGB + CatBoost, averaged | 0.554 | 0.588 | 0.519 | 0.600 | dropped |

CatBoost is more conservative (0.65 predicted per true possession against 0.70) and misses more
short possessions; averaging it in drags the trees down. Untuned, so this is "not as a drop-in",
not "never". Machine note: a browser tab held ~4 cores for the whole screen, which is why the
two HGB screens took ~53 min each.

## 14. Noisy student on our own match (ROADMAP R4, `analysis/noisy_student.py`)

Each fold's `final_v1` model labels our match (decoded; kept where it is confident, max
probability ≥ 0.8 and agreeing with the decoded label: 79–82% of graded frames), outside the
public window ± 30 s and the masked tail. The student is the full `final_v1` recipe plus that
match, fitted per fold.

| | bench | ext | masked | public |
|---|---|---|---|---|
| `final_v2` | **0.656** | 0.617 | **0.695** | **0.675** |
| + our match, pseudo-labelled | 0.640 | 0.618 | 0.663 | 0.634 |

**Not adopted: −0.016.** External is unchanged, and our own window, the one place it was meant
to help, gets worse (masked −0.032, public −0.041). The teacher's own mistakes on our match are
learned back as truth; one pseudo-labelled match among 54 training copies is enough to pull the
model towards them without adding information.

## 15. R8 full fit: velocity-projected pitch control (`analysis/r8_full.py`)

| | bench | ext | masked | public |
|---|---|---|---|---|
| `final_v2` | 0.656 | 0.617 | 0.695 | 0.675 |
| `full_plus4` (full recipe) | 0.656 | 0.619 | 0.693 | 0.675 |

**Not adopted: no change.** Its +0.014 on the fast screen (§13) disappears once the mirrored and
extra-noise copies are in, which is why the screen may only drop candidates, never adopt them.

**Round 3 summary.** Of R4, R5, R7 and R8, nothing beats `final_v2`; R1 (§10) was the only change
adopted, and R2 (§12) closed the video channel. The decoder and the feature set are on a
plateau. The remaining lever is validation (R3), not modelling.

## 16. Why "less hidden ball scores lower" on the public window, and R3 2nd-half validation (`analysis/r3_eval.py`)

In §12 (`analysis/ball_value.py`), hiding less ball raised external `spellF1` (`0.617 → 0.641 → 0.664`)
but lowered our public-window score (`0.695` at 45% hidden → `0.681` at 25% → `0.671` at 10% → `0.675`
at 0% extra hidden). Decomposing `spellF1 = 2 · TP / (17 + N_pred)` across masking levels explains why:

| extra ball hidden (public window) | spellF1 | mean TP / 17 (recall) | N_pred (precision) | spurious spells | teamAcc (live) | deadF1 | macroF1 |
|---|---|---|---|---|---|---|---|
| 0% (`public`, 10% missing) | 0.675 | 10.80 (0.635) | 15.00 (0.720) | 2.00 | 0.958 | 0.969 | 0.901 |
| 10% (`pace_weights`) | 0.671 | 10.84 (0.638) | 15.31 (0.708) | 2.25 | 0.958 | 0.969 | 0.903 |
| 25% (`pace_weights`) | 0.681 | 10.76 (0.633) | 14.62 (0.736) | 1.75 | 0.958 | 0.968 | 0.903 |
| 45% (`masked`, `pace_weights`) | **0.695** | 10.66 (0.627) | 13.69 (0.779) | 1.00 | 0.958 | 0.971 | 0.911 |
| 45% (uniform `weights=None`) | 0.685 | 10.38 (0.611) | 13.31 (0.780) | 0.81 | 0.957 | 0.968 | 0.906 |
| 100% (ball-free shape only) | 0.696 | 9.40 (0.553) | 10.00 (0.940) | 0.00 | 0.963 | 0.961 | 0.901 |

1. **Visible ball helps recall (`TP`), not hurts it.** True matched possessions rise from `9.40` (no
   ball) to `10.66` (45% hidden) to `10.80–10.84` (0–10% hidden). On live-play team calls
   (`teamAcc = 0.958`) and dead-ball F1 (`deadF1 = 0.968–0.971`), 0%–45% hidden are identical.
2. **The `0.675` vs `0.695` inversion comes from 2 spurious spells (`1.84 s` `H` and `2.28 s` `A`,
   103 frames total) inside ground-truth stoppages (`D = 100%`).** During those 103 frames, both
   teams are walking (`pace = 1.09` and `1.27 m/s`) and no player is within `5 m` of the ball
   (`min(dh, da) = 5.88 m` and `5.12 m`), but the ATD ball is tracked (`seen = 100%`) moving fast
   (`bspeed = 7.82` and `12.42 m/s`, e.g. ball returned to a restart spot). In external training,
   `D` frames have the ball seen only `40.0%` of the time (`50.6%` IDSSE, `3.0%` Metrica), whereas
   `public` has `D` ball seen `66.4%` of the time. The fast visible ball tricks the trees into two
   short live spells inside `D`, inflating `N_pred` from `13.69` to `15.00` (`-0.020` `spellF1` on a
   17-spell denominator).
3. **`pace_weights` (`3×` gap-start weight when `pace < 1.3 m/s`) selectively masks those slow
   stoppage frames.** Under `masked`, `D` ball visibility drops from `66.4%` to `26.1%`; when the
   ball is hidden on those 103 frames, the ball-free shape model (`100%` hidden) agrees with `gt = D`
   `100.0%` of the time, cutting `spurious` from `2.00` to `1.00` (`0.00` at 100% hidden). With
   uniform 45% masking (`weights=None`), half of the `masked - public` gap disappears (`0.685`,
   `se = 0.010`).

### R3 2nd-half validation windows (`work/labels/val_h2_*.csv`, `analysis/r3_eval.py`)

Four non-overlapping 5-minute (`7,500`-frame) windows in the 2nd half (`30,000` frames = 20 minutes
total, `57` graded live spells, `62.8%` mean ball visibility) were labelled in `work/labels/`:

| `--window` | file | frames (`tactical`) | ball seen | GT live spells | `final_v2` spells | `spellF1` | `lowIoU` (`0.1..0.3`) | `teamAcc` | `deadF1` | `collarMF1` ($\pm 1\text{ s}$) |
|---|---|---|---|---|---|---|---|---|---|---|
| `1` | `val_h2_122500.csv` | `122500..130000` | 72.5% | 13 | 18 | 0.594 | 0.753 | 0.933 | 0.935 | 0.819 |
| `2` | `val_h2_137500.csv` | `137500..145000` | 60.9% | 10 | 14 | 0.533 | 0.639 | 0.939 | 0.928 | 0.885 |
| `3` | `val_h2_167500.csv` | `167500..175000` | 49.3% | 13 | 11 | **0.717** | **0.833** | 0.909 | **0.959** | **0.929** |
| `4` | `val_h2_182500.csv` | `182500..190000` | 68.7% | 21 | 19 | 0.450 | 0.717 | 0.779 | 0.941 | 0.827 |

Across the combined 20-minute `val_h2` benchmark (`57` graded live spells):

| method (on combined `val_h2`) | `spellF1` | `lowIoU` | `macroF1` | `collarMF1` | `teamAcc` | `deadF1` | `TP / 57` | `N_pred` | `spurious` |
|---|---|---|---|---|---|---|---|---|---|
| `rules_v2` | 0.655 | 0.801 | 0.865 | 0.883 | 0.917 | 0.928 | 39.0 / 57 | 62 | 10 |
| `R0 final_v1` (`kickoff=None`) | 0.558 | 0.734 | 0.860 | 0.894 | 0.890 | 0.940 | 33.2 / 57 | 62 | 13 |
| `R1 final_v2` (`kickoff='half'`) | 0.558 | 0.734 | 0.860 | 0.894 | 0.890 | 0.940 | 33.2 / 57 | 62 | 13 |
| `R2` (50% unmasked copies) | 0.526 | 0.713 | 0.855 | 0.891 | 0.887 | 0.938 | 30.0 / 57 | 57 | 12 |
| `R4` (pseudo-label student) | 0.571 | 0.746 | 0.862 | 0.897 | 0.889 | 0.942 | 33.4 / 57 | 60 | 12 |
| `R6` DP video ball (`c0=0.08`) | 0.536 | 0.722 | 0.844 | 0.879 | 0.893 | 0.925 | 32.4 / 57 | 64 | 14 |
| `R6` DP video ball (no spots) | 0.536 | 0.722 | 0.844 | 0.879 | 0.893 | 0.925 | 32.4 / 57 | 64 | 14 |
| `R7` in-Viterbi `dwell=25` | 0.558 | 0.734 | 0.860 | 0.894 | 0.890 | 0.940 | 33.2 / 57 | 62 | 13 |
| `R8` velocity pitch control | 0.571 | 0.746 | 0.861 | 0.896 | 0.889 | 0.941 | 33.4 / 57 | 60 | 12 |
| `submission (final_v2.csv)` | 0.577 | 0.722 | 0.859 | 0.895 | 0.891 | 0.938 | 34.6 / 57 | 63 | 14 |

### Settling the "less hidden ball scores lower" question on `val_h2`

Running the `0% / 25% / 45% / 100%` additional ball-hiding sweep on `val_h2` (where natural ball
visibility is already `62.8%`, vs `89.7%` on the public window) settles the question:

| `val_h2` masking level | ball seen | `spellF1` | `lowIoU` | `TP / 57` | `N_pred` | `spurious` | `teamAcc` | `deadF1` | `collarMF1` |
|---|---|---|---|---|---|---|---|---|---|
| extra hide `0%` (natural `val_h2`) | 62.8% | **0.558** | 0.734 | **33.20** | 62.00 | 13.00 | **0.890** | 0.940 | 0.894 |
| extra hide `25%` (`pace_weights`) | 47.0% | 0.555 | **0.748** | 31.50 | 56.50 | 9.12 | 0.888 | **0.941** | 0.897 |
| extra hide `45%` (`pace_weights`) | 35.4% | 0.531 | 0.732 | 29.12 | 52.62 | 7.75 | 0.882 | **0.941** | **0.898** |
| extra hide `100%` (shape only) | 0.0% | 0.440 | 0.653 | 22.20 | 44.00 | 5.00 | 0.856 | 0.939 | 0.887 |

- **More visible ball strictly improves `spellF1` (`0.440 → 0.531 → 0.555 → 0.558`), true-possession
  recall (`TP: 22.20 → 29.12 → 31.50 → 33.20`), and live `teamAcc` (`0.856 → 0.882 → 0.888 → 0.890`)
  at natural 2nd-half missingness (`62.8%` seen).**
- At the same time, **the dead-ball spurious-spell mechanism holds in the 2nd half too:** in unmasked
  `val_h2` (`0%` extra hide), `final_v2` emits spurious live spells inside long stoppages whenever
  the ATD tracks a ball rolling along the touchline (windows 1 and 3) or being placed in the 6-yard
  box before a goal kick (window 2). Masking `25%` of slow frames via `pace_weights` eliminates `3.88`
  of those spurious spells (`13.00 → 9.12`) and raises relaxed-IoU `spellF1` (`0.734 → 0.748`) and
  `collarMF1` (`0.894 → 0.897`), while masking `45%` (`ball_seen = 35.4%`) over-starves live
  possessions (`TP` drops by `4.08` spells, lowering exact `spellF1` to `0.531`).

### 300-frame genuine missing-ball probe (`work/labels/ball_probe_300.csv`)

Visual inspection of the 300 stratified genuine missing-ball frames (`--probe`) shows:
- **`D` (Dead ball / stoppage):** `201 / 300` (**`67.0%`**) — throw-in waits, foul/free-kick waits,
  goal-kick/corner waits, cooling breaks, substitutions, and post-whistle frames.
- **`O` (Occluded / aerial / off-screen live play):** `52 / 300` (**`17.3%`**) — ball in player
  scrums or above the top crop of the tactical camera during high clearances/goal kicks.
- **`V` (Visible live ball on pitch):** `47 / 300` (**`15.7%`**, or `47.5%` of the 99 live missing-ball
  frames).
- **Why the un-retrained DP video ball (`R6`) hurts `final_v2` on `val_h2` (`0.558 → 0.536`,
  `deadF1 0.940 → 0.925`):** On the 300-frame probe, the DP linker (`c0 = 0.08`) links `115 / 300`
  (`38.3%`) missing-ball frames — recovering `74.5%` (`35 / 47`) of `V` frames and `0.0%` (`0 / 52`)
  of `O` frames, **but linking `39.8%` (`80 / 201`) of `D` frames** (`69.6%` of all linked frames are
  stoppages, including `21.7%` locked onto painted white penalty/center spots). Filling `seen = 1`
  during stoppages resets `since`/`until` and causes the un-retrained trees to over-predict live play
  inside stoppages.

## 17. Rules back in the decoder? The yardsticks disagree (`analysis/fusion_val.py`)

R3 showed `rules_v2` beating `final_v2` on the hand-labelled 2nd half (0.655 vs 0.558). Decoder-only
on cached `P_R1`, rule evidence added where the ball is seen nearby (`use_rules=True`):

| decoder | bench | ext | masked | public | **val_h2** | win 1 | win 2 | win 3 | win 4 |
|---|---|---|---|---|---|---|---|---|---|
| `final_v2` (rules as features only) | **0.656** | **0.617** | **0.695** | **0.675** | 0.558 | 0.594 | 0.533 | 0.717 | 0.450 |
| rules, w 0.25 | 0.632 | 0.589 | 0.675 | 0.645 | 0.630 | 0.648 | 0.589 | 0.800 | 0.530 |
| rules, w 0.5 | 0.567 | 0.545 | 0.589 | 0.480 | **0.680** | 0.624 | 0.491 | 0.864 | 0.702 |
| rules, w 1 | 0.531 | 0.492 | 0.570 | 0.447 | 0.652 | 0.708 | 0.448 | 0.829 | 0.623 |
| team-only rule evidence, w 0.25 | 0.632 | 0.590 | 0.674 | 0.645 | 0.630 | 0.627 | 0.509 | 0.800 | 0.530 |
| rules w 0.5, only where model max P < 0.8 | 0.610 | 0.582 | 0.638 | 0.613 | 0.592 | 0.640 | 0.360 | 0.696 | 0.652 |

- **A smooth trade-off, no free lunch.** At w 0.25 every 2nd-half window improves (val +0.072) while
  ext (−0.028), masked (−0.020) and public (−0.030) all fall; larger weights trade more.
- **Gating on model confidence does not help**: the val gain comes from frames where the model is
  confident *and* wrong. The model's live team accuracy is 0.958 on the public window but 0.890 on
  val_h2 (0.779 in window 4): it drifts later in the match, while the rules, which read the ball
  directly, do not.
- **Which yardstick represents the full match is the open question.** Ext and public use real
  ground truth but are other leagues / the match's best-tracked first 5 minutes; val_h2 is our
  match's 2nd half (57 possessions) but eye-labelled, a convention that may lean towards touches.

## 18. `final_v3`, and why the model drifts in the 2nd half (`analysis/drift.py`)

**`final_v3`** = `final_v2` with the rule labels voting in the decoder (`use_rules`, `w_rule` 0.25),
chosen on `val_h2` (§17). Delivered files: public spellF1 0.700 → 0.653 (macroF1 0.898 → 0.904),
`val_h2` 0.577 → **0.627** (macroF1 0.859 → 0.873), full-match possessions 282 → 286.

**Where the model's team calls go wrong** (live frames, `final_v2` decoder vs labels):

| window | model team acc | rules team acc | live frames with the ball seen | model error, ball seen | rules error, ball seen | ball seen on D frames |
|---|---|---|---|---|---|---|
| public | **0.958** | 0.889 | 94% | 4.5% | 11.8% | 66% |
| 2nd half, win 1 | 0.933 | **0.957** | 100% | 6.7% | 4.3% | 27% |
| win 2 | **0.939** | 0.877 | 91% | 6.7% | 13.5% | 18% |
| win 3 | 0.909 | **0.957** | 91% | 9.6% | 4.5% | 11% |
| win 4 | 0.779 | **0.879** | 98% | 21.2% | 12.2% | 29% |

- **Not tracking quality**: 10.5–11.0 players tracked per team in every window.
- **The model errs with the ball visible** (91–100% of live frames have it): it under-uses ball
  evidence that is good on our match. The errors are confident (top probability 0.77–0.81), which
  is why gating rules on model uncertainty failed (§17).
- **Neither source dominates**: the model wins in the public window and window 2, the rules in
  windows 1, 3 and 4. A fixed blend is a compromise; a per-frame choice could do better.
- **§16's "dead-ball frames have the ball seen 66%" is a public-window quirk**: in the 2nd half it is
  11–29%, close to training's 40%. Recalibrating the noise model's dead-ball bias to the public
  window would have been wrong.

## 19. R11: per-frame stacker (not adopted) and the stoppage gate (`final_v4`, `analysis/r11_stack.py`)

A multinomial logistic stacker (10 inputs: model log-probabilities, rule vote × ball near, ball
near, on-ball distance margin, ball seen, ball speed) fitted in-domain with leave-one-window-out CV
over the public window and the four val windows; ext scored by the stacker fitted on all 5, masked
by the one fitted on the 4 val windows only. Decoder-level on cached `P_R1`:

| config | bench | ext | masked | public | win 1 | win 2 | win 3 | win 4 | val avg |
|---|---|---|---|---|---|---|---|---|---|
| `final_v2` decoder | 0.656 | 0.617 | 0.695 | 0.675 | 0.594 | 0.533 | 0.717 | 0.450 | 0.573 |
| `final_v3` | 0.632 | 0.589 | 0.675 | 0.645 | 0.648 | 0.589 | 0.800 | 0.530 | 0.642 |
| **`final_v3` + stoppage gate (`final_v4`)** | 0.636 | 0.589 | 0.683 | **0.667** | 0.648 | 0.589 | 0.800 | 0.530 | 0.642 |
| stacker | 0.505 | 0.469 | 0.541 | 0.565 | 0.662 | 0.452 | 0.848 | 0.722 | 0.671 |
| stacker + gate | 0.506 | 0.468 | 0.543 | 0.565 | 0.662 | 0.473 | 0.883 | 0.722 | 0.685 |

- **Stacker: not adopted.** Held-out val +0.043, but public −0.080 and ext −0.120, and window 2
  falls: fitted mostly on eye labels, it leans on the rules further than the real GT supports.
- **Stoppage gate: adopted** (`decode.stop_gate`, `stop_gate=100`): a live spell of ≤ 4 s between
  two dead spells while the players walk becomes dead ball. Public +0.022 and masked +0.008 on the
  fold ensemble, val and ext unchanged. On the delivered file (full-fit model) public stays 0.653,
  `val_h2` 0.627 → **0.632**, full-match possessions 286 → 278. sha256 `0bc6e8c049b0cd22`.

## 20. The eye labels reward rule timing; back to the `final_v2` decoder (`analysis/boundary_offsets.py`, `err_locus.py`, `rev_v4.py`)

`final_v3`/`v4` were chosen on `val_h2` against all three real-GT yardsticks (§17–19). Three
measurements on cached `P_R1` say `val_h2`'s official spellF1 cannot make that call.

**Where H↔A changes land.** Offset = predicted − labelled change (frames), nearest same-kind
change within 5 s:

| labels | n | model (`final_v2` decoder) median [IQR] | rules median [IQR] |
|---|---|---|---|
| public, real GT | 12 | +4 [−10, +8] | **−39 [−82, −14]** |
| `val_h2`, eye labels | 31 | +1 [−22, +18] | **+0 [+0, +2]** |

On the real GT the rules switch a median 1.6 s early (at the pass, before the new team's
controlled touch); the model is centred. On the eye labels the rules' changes coincide with the
labelled ones to 0–2 frames. (Restarts: real GT model −22 / rules +25, eye +1 / +10; stops: real
GT model +41 / rules −1, eye +1 / 0; n = 4 each on the public window, so only indicative.)

**Where team errors are.** Live frames by distance to the nearest labelled change, error rate
`<1 s / 1–3 s / >3 s`: public (GT) model 0.21/0.05/0.01 vs rules 0.37/0.21/0.04; across the four
eye-labelled windows the model is 0.20–0.39 / 0.11–0.25 vs rules 0.10–0.32 / 0.08–0.21 near
changes, and mid-possession (> 3 s) the model is as good as or better than the rules in every
window (win 4: 0.15 vs 0.16, the only window where both are high). The 2nd-half "drift" of §18 is
near changes, i.e. timing.

**Every decoder on every yardstick** (`analysis/yardsticks.py`, `rev_v4.py`; valLo = val spellF1
at IoU 0.1–0.3, valMid = val team error > 3 s from a change):

| decoder | bench | ext | masked | public | val | valLo | valMid |
|---|---|---|---|---|---|---|---|
| rules only | 0.319 | 0.355 | 0.284 | 0.461 | **0.652** | 0.792 | 0.064 |
| `final_v2` | 0.656 | 0.617 | 0.695 | 0.675 | 0.573 | 0.735 | 0.046 |
| **`final_v2` + stop gate (`v2g`)** | **0.661** | 0.616 | **0.706** | **0.697** | 0.579 | 0.742 | 0.046 |
| `final_v3` | 0.632 | 0.589 | 0.675 | 0.645 | 0.642 | 0.780 | 0.047 |
| `final_v4` | 0.636 | 0.589 | 0.683 | 0.667 | 0.642 | 0.780 | 0.047 |

The rules alone, last on every real-GT yardstick, are **first on `val_h2`**. `val_h2` spellF1
therefore measures agreement with touch timing and must not choose boundary placement; it stays
valid for mid-possession team calls, dead-ball detection and spell counts. **Decision: `v2g`
(no rule vote, stop gate kept)**: +0.025 bench, +0.027 ext, +0.030 public over `final_v4`.

**Where `v2g` still loses** (`analysis/miss_by_len.py`, recall at IoU 0.5 by true duration
`<2 s | 2–5 | 5–10 | 10–20 | >20 s`): IDSSE share 0.30/0.31/0.19/0.12/0.09, recall
0.18/0.48/0.74/0.87/0.90; Metrica share 0.20/0.26/0.18/0.20/0.15, recall 0.20/0.51/0.74/0.83/0.76;
public share 0.12/0.24/0.24/0.12/0.29 (2 of 17 under 2 s, both missed). Short possessions are
the main loss.

## 21. Segmental MBR decoding for spellF1 (`mtah/mbr.py`, `analysis/mbr_eval.py`)

Viterbi maximises path probability; spellF1 counts matched possessions. MBR (Goel & Byrne 2004,
speech recognition) instead picks, stretch by stretch between dead balls on which every
candidate agrees, the candidate with the largest expected `2 TP − F1_ref · N` (the first-order
change of `2 TP / (G + N)`) against 24 paths sampled from the decoder's own posterior (forward
filtering, backward sampling). Candidates: the full `v2g` decode at switch-cost scales
0.5/0.7/1/1.4/2. Cached `P_R1`, no fitting:

| decoder | bench | ext | masked | public | val | valLo |
|---|---|---|---|---|---|---|
| `v2g` Viterbi | 0.661 | 0.616 | 0.706 | 0.697 | 0.579 | 0.742 |
| MBR, temperature 1 | 0.662 | 0.620 | 0.705 | 0.697 | 0.576 | 0.742 |
| MBR, temperature 2 | 0.661 | 0.621 | 0.702 | 0.697 | 0.575 | 0.734 |

**Not adopted** (+0.001 bench, under rule 1). With the tuned costs, Viterbi already sits where
the expected metric peaks for these probabilities; choosing among cost scales per stretch moves
ext by +0.004–0.005 and masked by −0.001–0.004. The option stays in `mtah/mbr.py`.

## 22. A learned temporal refiner on the trees' probabilities (`mtah/refiner.py`, `analysis/refine_eval.py`)

The temporal stage of MS-TCN / ASRF (action segmentation) and of the GRU/UNet heads that won
Kaggle's CMI sleep-state segmentation: a 2-layer bidirectional GRU at 5 fps over 13 per-frame
inputs (tree log-probabilities, boundary-head logit, ball seen / near, rule vote × near, distance
margin, ball speed, team pace, time since ball seen), trained on external matches with true labels
and home/away mirrors. Ext matches are refined by a GRU trained on the other two folds'
out-of-fold probabilities; our match by one trained on all 9 (3 seeds each). Output blended
geometrically with the trees (`a` = refiner share), then the `v2g` decoder.

| | bench | ext | idsse | metrica | masked | public | val | valLo |
|---|---|---|---|---|---|---|---|---|
| `v2g` | **0.661** | 0.616 | 0.607 | 0.648 | **0.706** | **0.697** | 0.579 | 0.742 |
| a 0.5, costs × 1 | 0.647 | 0.620 | 0.607 | 0.668 | 0.674 | 0.657 | 0.571 | 0.741 |
| a 0.5, costs × 0.5 | 0.654 | **0.633** | 0.620 | 0.677 | 0.675 | 0.640 | 0.572 | 0.743 |
| a 1, costs × 1 | 0.634 | 0.614 | 0.596 | 0.678 | 0.654 | 0.621 | 0.550 | 0.705 |
| a 1, costs × 0.5 | 0.641 | 0.630 | 0.614 | 0.685 | 0.652 | 0.621 | 0.548 | 0.714 |

The largest ext gain of any change since R1 (+0.017, both sources), but our window loses
(masked −0.031, public −0.057, fewer spells: 14 → 11–13). **Fails rule 1 as run.** A confound: the
refiner learned from single fold-model probabilities, while our match was given the 3-model
average, which is smoother. Round 2 (§23) removes it.

## 23. Refiner round 2: a dilated TCN, and the input mismatch removed (`analysis/fold_probs.py`, `refine_eval2.py`)

Single-stage MS-TCN (8 residual dilated conv layers, ±51 s at 5 fps), boundary-head input
dropped. Our match refined either from the 3-model average (as in §22) or once per fold model
(`cache/P_R1_folds.npz`; the three fold models reproduce `P_R1` to 1e-7) and averaged, so the
refiner sees the same single-model inputs it was trained on.

| our-match input | a | costs | bench | ext | idsse | metrica | masked | public | val |
|---|---|---|---|---|---|---|---|---|---|
| — (`v2g`) | — | × 1 | **0.661** | 0.616 | 0.607 | 0.648 | **0.706** | **0.697** | 0.579 |
| average | 0.5 | × 1 | 0.650 | 0.618 | 0.606 | 0.659 | 0.683 | 0.650 | 0.581 |
| average | 0.5 | × 0.5 | 0.655 | **0.627** | 0.616 | 0.668 | 0.682 | 0.650 | 0.582 |
| average | 1 | × 1 | 0.643 | 0.607 | 0.591 | 0.664 | 0.679 | 0.662 | 0.578 |
| average | 1 | × 0.5 | 0.649 | 0.620 | 0.607 | 0.667 | 0.677 | 0.642 | 0.580 |
| per fold | 0.5 | × 1 | 0.649 | 0.618 | 0.606 | 0.659 | 0.681 | 0.650 | 0.576 |
| per fold | 0.5 | × 0.5 | 0.655 | 0.627 | 0.616 | 0.668 | 0.683 | 0.650 | 0.582 |
| per fold | 1 | × 1 | 0.642 | 0.607 | 0.591 | 0.664 | 0.678 | 0.662 | 0.574 |
| per fold | 1 | × 0.5 | 0.649 | 0.620 | 0.607 | 0.667 | 0.678 | 0.637 | 0.598 |

- **The mismatch was not the cause**: per-fold and averaged inputs score the same on our window.
- **The refiner learns the external conventions, not possession in general.** Under leave-fold-out
  CV it gains on ext (+0.011, both sources, GRU or TCN), and it loses 0.02–0.03 masked and
  0.035–0.06 public on our match every time. Ext GT comes from DFL / Metrica-event conventions
  (IDSSE counts brief touches, §6), so a stage that models label *sequences* picks up exactly what
  differs between them and our GT; the per-frame trees transfer better.
- **Not adopted** (rule 1, two architectures, eight settings). `mtah/refiner.py` stays as a tool.

## 24. `final_v5`, the delivered file (`analysis/score_files.py`)

`mtah/final.py` `DECODER`: `use_rules` False, `stop_gate` 100, everything else as `final_v2`. Full
fit on all 9 external matches, as delivered by `run.py`:

| file | public spellF1 / macroF1 | public spells (true 17) | full-match possessions | val_h2 | valLo | valMid |
|---|---|---|---|---|---|---|
| `final_v2.csv` | 0.700 / 0.898 | 15 | 282 | 0.590 | 0.727 | 0.044 |
| `final_v4.csv` | 0.653 / 0.904 | 13 | 278 | 0.633 | 0.782 | 0.043 |
| **`final_v5.csv`** | **0.723 / 0.910** | 14 | 269 | 0.593 | 0.731 | 0.044 |

Fold models (the bench): 0.661, ext 0.616, masked 0.706 (§20). Two runs of `run.py` give the same
file byte for byte, sha256 `4089da237ca8780a…`; `evaluation/validate_submission.py` passes. The
fold-to-full-fit check asked for by the sixth review holds: the full fit's public gain over
`final_v4` (+0.070) is larger than the fold models' (+0.030).

## 25. R17: touch-sequence and kick features for short possessions (`feats.flight_block`, `analysis/r17_screen.py`, `r17_full.py`)

The other blocks see touches only through ±1–10 s smoothing, which averages a 1–2 s possession
away. `flight_block` (14 columns) gives each frame the touch sequence around it: team and length
of the last and next on-ball episode, the episode before the last and after the next (a second
touch confirms control), frames since / until an episode, episodes per team within ±2 s, and kick
events (ball speed jump > 4 m/s to > 6 m/s with a player within 2.5 m) with the kicker's team.
`full_plus5` = `full_plus3` + `flight_block`. `final_v5` decoder throughout.

**Fast screen** (rule 7: 9 base matches, no augmentation, 300 trees; ext recall at IoU 0.5 by true
duration `<2 | 2–5 | 5–10 | >10 s`, spurious spells per 100 true):

| screen | bench | ext | idsse | metrica | masked | public | valLo | ext recall by length | spurious |
|---|---|---|---|---|---|---|---|---|---|
| `full_plus3` | 0.573 | 0.590 | 0.578 | 0.629 | 0.557 | 0.625 | 0.739 | 0.156 / 0.456 / 0.709 / 0.845 | 4.6 |
| `full_plus5` | **0.613** | 0.602 | 0.593 | 0.635 | **0.624** | 0.653 | 0.784 | 0.174 / 0.470 / 0.720 / 0.842 | **3.8** |

+0.040 bench, the largest screen gain so far, with short-possession recall up **and** spurious
spells down. Passes the screen; the full refit decides (R8's +0.014 screen gain vanished there).

**Full fit** (`final_v1` recipe: 600 trees, mirrored + extra-noise copies; `cache/P_R17.npz`):

| | bench | ext | idsse | metrica | masked | public | valLo | ext recall by length | spurious |
|---|---|---|---|---|---|---|---|---|---|
| `final_v5` (`full_plus3`) | 0.661 | 0.616 | 0.607 | 0.648 | **0.706** | **0.697** | 0.742 | 0.186 / 0.482 / 0.740 / 0.861 | 3.6 |
| `full_plus5` | 0.662 | **0.625** | 0.619 | 0.645 | 0.699 | 0.684 | 0.755 | 0.186 / 0.492 / 0.755 / 0.866 | **2.9** |

**Not adopted** (rule 1: bench +0.001). Like R8 (§15), most of the screen gain is what the mirrored
and extra-noise copies already supply. What is left is real but small: ext +0.009 (IDSSE +0.012),
fewer spurious spells (3.6 → 2.9 per 100), recall up at 2–10 s, while our window loses a little
(masked −0.007, public −0.013, one possession either way on 17). Possessions under 2 s stay at
0.186: the touch sequence alone does not find them. `feats.flight_block` stays available.

## 26. After `final_v5`: three research tracks, nothing adopted (2026-10-25/26)

Three tracks ran after the freeze candidate `final_v5`, each in its own folder and each read-only on
`mtah/` and `run.py`. Full records are in the linked folders; this section keeps the numbers that
decide what not to repeat. Locked baseline throughout: `final_v5` fold models (`cache/P_R1.npz`),
bench **0.6612**, ext 0.6163, masked 0.7061, public (fold) 0.6968; full-fit public 0.7226 / 0.9098.
Reproduced by `baseline_snapshot/reconstruct_baseline.py` (see `baseline_snapshot/`).

### 26.1 Integrity review and decoder/training ideas (`analysis/autonomous_research/`)

Registry: `analysis/autonomous_research/REGISTRY.json`; log: `RESEARCH_LOG.md`.

| id | idea | bench | ext | masked | public (spell/macro) | verdict |
|---|---|---|---|---|---|---|
| AR-EARLY-DUEL | extra switch cost in duels | 0.660 | | | | flat |
| AR-EARLY-SMOOTH | geometric (BOCPD-style) smoother | 0.633 | | | | negative |
| AR-EARLY-SEMI | semi-Markov duration decoder (after fixing an unreachable-boundary bug) | 0.657 | 0.614 | 0.699 | | not adopted |
| AR-E1 | contest-duel sample reweighting (full recipe) | 0.658 | 0.613 | 0.703 | | negative (screen +0.020 vanished) |
| AR-G3 | Metrica sample game 3 as a one-time frozen holdout | | | | baseline 0.6128/0.8764, E1 0.6128/0.8770 | tied; **holdout now consumed** |
| AR-DOG0 | difference-of-Gaussians logit sharpening | 0.667 (dev) | | | 0.6857/0.9049 (full fit) | dev up, public down: rejected |
| AR-TI1 | typed transition head (226 feats, λ=2) | 0.6573 | 0.6220 | 0.6926 | 0.671/0.910 | rejected |
| AR-PI1 | physical transition head (228 feats, λ=2) | 0.6577 | 0.6220 | 0.6934 | 0.671/0.910 | rejected |
| AR-PL-v2 | hard physical control latch (velocity + carrier continuity) | 0.303 | 0.168 | 0.438 | 0.512/0.881 | strong negative |
| — | out-of-pitch emission boost ×2/4/8 | 0.6612 | 0.6163 | 0.7061 | | flat |
| — | broadcast-ball fusion, recovery diagnostic | | | | identical to baseline | 0 label changes |
| AR-PSEUDO | Smart Tagging as pseudo anchors | | | | | 638 events; kickoff D agreement 0.586; not ground truth |
| — | goal-conditioned kickoff prior | | | | | blocked: no goal events, team or clock in the tags |

Integrity findings worth keeping: the official scorer was read and confirmed (spellF1 over IoU
0.3–0.7, greedy same-team matching, X forward-filled within a half). `masked` reuses the public
window's 17 possessions across 16 ball-mask seeds, so its sd (0.024) is ball-placement noise, not
possession variance; `ext` (~3,000 possessions) is the steadiest gate, but it is 2 datasets, not 9
independent clusters. The out-of-fold disagreement on the external matches is 8.7% (IDSSE, mostly
H↔A) and 14.8% (Metrica, mostly alive called dead), consistent with label-convention differences.
No hand label reaches `final_v5`: `work/labels/` is read only by `Bench`, which `run.py` never builds.
The 0.005 adoption margin is a development heuristic, not a significance test.

### 26.2 Causal tracking features (`analysis/feat_track/`)

Five past-only, half-reset blocks on top of `full_plus3` (carrier tightness, centroid compression,
margin dwell, melee uncertainty, ball-free cluster). Screen: +0.016 to +0.031 bench each. Full
recipe: `c_margin_dwell` 0.6584, `c_carrier_tightness` 0.6586, both together 0.6495. **Not
adopted.** This is the fourth time (with R8 §15 and R17 §25) that a large screen gain vanished under
mirrors and noise copies: the screen drops losers but does not predict winners.
`c_ballfree_cluster`'s full fit was started and not finished; `c_melee_uncertainty` and
`c_centroid_compression` were screened only. Details: `analysis/feat_track/README.md`.

### 26.3 Audio and other sources (`analysis/audio_sources/`)

The tactical mp4 carries AAC 48 kHz audio on the same container timeline (edit lists 0; semantic
audio–video delay unmeasured). A tonal whistle detector (30 s CPU for the match) fires within
1.4 s of both half kickoffs and the 2nd-half end; 206 events at threshold 0.30, precision unknown
(no whistle labels). Uses tried on `final_v5`:

| config | public spellF1 / macroF1 | val_h2 spellF1 / macroF1 | verdict |
|---|---|---|---|
| `final_v5` | 0.7226 / 0.9098 | 0.5815 / 0.8593 | — |
| hard D-relabel at whistle events, thr 0.30 / 160 ms | identical | identical | null (3 events, all kickoffs) |
| hard D-relabel, thr 0.25 / 80 ms | 0.7000 / 0.9037 | 0.5581 / 0.8501 | hurts |
| soft D-probability nudge, W = 0.15 (fold P) | identical | identical | no effect |

PANNs Cnn14 runs locally (44 ms per 1 s chunk); its Whistle score sits at the output floor in the
one kickoff window tried. Main blocker: the external training matches have **no audio**, so audio
cannot be a tree feature; it can only enter as a decoder/post-processing channel or a model fitted
on our own match. Details: `analysis/audio_sources/docs/FINAL_SOURCE_SYNTHESIS.md`.

### 26.4 Reproducibility

`run.py` reproduced `final_v5` byte for byte on 2026-10-25 (twice) and 2026-10-26
(`baseline_snapshot/fresh_rebuild.log`, sha256 `4089da237ca8780a…`, public 0.722581 / 0.909848).

## 27. R20: stoppage-ball gate v2 → `final_v6` (`analysis/r20_gate.py`, 2026-11-17)

Post-decode only, on the cached fold probabilities `cache/P_R1.npz`. The script first reproduces
the locked baseline to 1e-9 (bench 0.6611595515) with a low-memory harness (no feature matrices,
< 1 GB), then scores each gate on every yardstick. Shipped gate for reference (§19): a live spell
of ≤ 100 frames with dead ball on **both** sides while the players walk (< 1.3 m/s) becomes dead
ball.

**What is left to remove** (`--diagnose`). Predicted live spells after the shipped gate; spurious =
no same-team overlap with any true possession:

| yardstick | predicted spells | spurious, inside true dead ball | spurious, other | of the spurious: ≤ 100 frames |
|---|---|---|---|---|
| ext (9 matches) | 2,639 | 49 | 86 | 114 of 135 |
| masked (16 seeds) | 211 | 8 | 0 | 8 of 8 |
| public | 14 | 1 | 0 | 1 of 1 |
| `val_h2` (eye labels) | 61 | 5 | 7 | 11 of 12 |

R20's premise does not hold for what remains: "players walk and no player within 5 m of the ball
on more than half of the spell's ball-seen frames" matches **0** of the spurious-in-dead-ball
spells on every yardstick. Of §16's two public-window blips the shipped gate removes one; the one
left (the same spell appears in 8 of the 16 masked seeds) is slow but has dead ball on one side
only, and players come within 5 m of the ball during it.

**Six configs, fixed before scoring, all reported** (rule 2). Each is the shipped gate plus one
more pass. Δ against `final_v5` fold models:

| config | bench | ext | masked | public | `val_h2` | spurious: ext /100 · masked /seed · public · val |
|---|---|---|---|---|---|---|
| `final_v5` (shipped gate) | 0.6612 | 0.6163 | 0.7061 | 0.6968 | 0.5627 | 3.62 · 0.50 · 1 · 12 |
| no gate | −0.0054 | +0.0004 | −0.0112 | −0.0218 | −0.0047 | 3.62 · 1.00 · 2 · 13 |
| G1 **R20 as specified**: 1+ side, ≤ 250, pace < 1.3, no player within 5 m | −0.0002 | −0.0004 | 0 | 0 | +0.0048 | 3.59 · 0.50 · 1 · 11 |
| G2 1+ side, ≤ 100, pace < 1.3 (the shipped constants, one-sided) | **+0.0052** | −0.0017 | +0.0120 | +0.0232 | +0.0097 | 3.37 · 0.00 · 0 · 10 |
| G3 1+ side, ≤ 100, pace < 1.0 | −0.0002 | −0.0003 | 0 | 0 | 0 | 3.62 · 0.50 · 1 · 12 |
| G4 2 sides, ≤ 250, pace < 1.3 | −0.0001 | −0.0002 | 0 | 0 | 0 | 3.59 · 0.50 · 1 · 12 |
| G5 2 sides, ≤ 100, pace < 1.6 | −0.0004 | −0.0008 | 0 | 0 | 0 | 3.45 · 0.50 · 1 · 12 |
| G6 2 sides, ≤ 100, no player within 5 m, any pace | −0.0001 | −0.0003 | 0 | 0 | +0.0048 | 3.62 · 0.50 · 1 · 11 |

**G2 on the delivered full-fit file** (the gate applied to `final_v5.csv`):

| file | public spellF1 / macroF1 | `val_h2` spellF1 / macroF1 | val spurious | live spells, full match |
|---|---|---|---|---|
| `final_v5` | 0.722581 / 0.909848 | 0.5815 / 0.8593 | 13 of 62 | 269 |
| `final_v5` + G2 | 0.746667 / 0.922535 | 0.5915 / 0.8611 | 11 of 60 | 264 |

G2 relabels **5 spells (299 frames, 0.19% of graded frames)** on the whole match: 1 in the public
window (spurious on the real GT), 2 in the val windows (both spurious on the eye labels, a
timing-free judgement, so `val_h2` may make it; dead-ball F1 0.939 and team accuracy 0.891 → 0.893
unchanged), 2 where we have no labels.

**Verdict: R20 as specified is null (G1). G2 is adopted → `final_v6` (§28).** The case against,
and why it did not decide:

- G2 is 1 of 6 configs and clears rule 1 by 0.0002 (bench +0.0052, ext −0.0017); rule 2 treats a
  borderline win after several tries as a "no" when the bench is all we have;
- its masked and public gains are **one spell** of the public window, counted once in `public` and
  in 8 of the 16 masked seeds. `--sensitivity` shows a step, not a slope: the gain appears whole
  between pace 1.2 and 1.3 and between 50 and 75 frames, and is flat from 75 to 150 frames;
- on ext it removes 9 spurious spells and 29 that overlap a true possession (IDSSE −0.0026,
  Metrica +0.0017), and the ext cost grows with the pace threshold (−0.0064 at 1.5 m/s). About a
  third of those 29 were matched possessions, judging by the size of the ext loss.

What decided it is that here the bench is not all we have: the change to the delivered file is
small enough to count. G2 relabels 5 spells of the match that is actually graded. Three of them
lie on frames we have labels for and all three are spurious, so removing them is a gain on the
private ground truth whatever the bench says (the public one certainly; the two val ones on a
timing-free judgement). Only the other two are a bet. With N_gt + N_pred ≈ 630 on the full match
(N_pred 264; N_gt ≈ 360 if the public window's 17 per 5 minutes holds, an estimate),
a spurious spell removed is worth about +0.001 spellF1 and a matched possession removed about
−0.002: +0.003 from the three, and between +0.002 (both spurious) and −0.004 (both matched) from
the two, about −0.0005 if they behave like the external matches. Expected gain ≈ +0.003, worst
case ≈ −0.001, and macroF1 (the tie-break) rises with every stoppage frame returned to `D`. No
constant was tuned: G2 uses the shipped gate's 100 frames and 1.3 m/s and changes only "both
sides" to "at least one".

This is a narrower claim than "G2 is a better decoder": on other matches the evidence is the ext
column, which is slightly negative. It is the better file for this match.

Also tried on top of G2, three values fixed in advance, none adopted: absorbing possessions shorter
than 25 / 50 / 75 frames flanked by one team (`min_poss`, `--min-poss`): bench −0.001 / −0.015 /
−0.079 against G2 (ext −0.001 / −0.024 / −0.062).

Variants tried in this section: 6 gate configs, 3 `min_poss` values, and a 25-cell sensitivity
grid that was not used to pick.

### 27.1 Checks on 2026-11-17 (no model change)

- `evaluation/results/final_v5.csv` sha256 `4089da237ca8780a…`, validator OK, official evaluator
  0.722581 / 0.909848; with `stop_gate_sides` at its default (2) `mtah/decode.py` gives the locked
  bench to 1e-9, so `final_v5` and every earlier number remain reproducible (`final.DECODER_V5`).
- `bash scripts/check.sh` (new): all tracked Python compiles; `tests/` (11 tests), the rebuild
  contract tests and the three handover test scripts pass. Run on Linux aarch64, Python 3.11.17,
  with the pinned numpy/pandas/scipy/scikit-learn — a third platform after macOS and Windows.
- A fresh clone of the public repository compiles, passes the data-free tests, `uv lock --check`
  passes, and `run.py` without data stops with the "data is gated" message. The full `run.py`
  replay from a fresh clone was **not** run here (it needs the data and ~1 h).
- Tracked files: no CSV other than the four organiser baselines and the public GT, no
  npz/joblib/audio/video; 0 broken relative links in our Markdown files.

## 28. `final_v6`, the delivered file (2026-11-17)

`final_v6` = `final_v5` + the one-sided pass of the stoppage gate (`mtah/final.py`:
`DECODER = dict(DECODER_V5, stop_gate_sides=1)`). Same features, same trees, same Viterbi settings.

| | public spellF1 / macroF1 | bench (ext / masked), fold models | `val_h2` spellF1 / macroF1 | live spells, full match |
|---|---|---|---|---|
| `final_v5` | 0.722581 / 0.909848 | 0.6612 (0.6163 / 0.7061) | 0.5815 / 0.8593 | 269 |
| **`final_v6`** | **0.746667 / 0.922535** | **0.6663** (0.6146 / 0.7180) | 0.5915 / 0.8611 | 264 |

`val_h2` here is scored over the four windows pooled; as the mean of the four windows (the 0.593
of §24) it goes 0.593 → 0.604.

The two files differ on 299 frames (5 spells), all changed from a team to `D`.

Reproduction: `run.py --name final_v6` (full refit, warm feature caches, ~9 min) on the original
Mac gives sha256 `349a7df54c8e0364e91e060b6279a2e7236bf5bfefc813e0bb63795b2e640ea3`, validator OK,
199,319 rows. The same hash had been predicted beforehand on a different platform (Linux aarch64)
by applying the gate to `final_v5.csv`, since the gate is the decoder's last step. A second full
refit (`--name final_v6_repro`) gives the same bytes. Logs: `cache/submission_final_v6/` (local).
The smallest distance of any eligible spell's mean pace from the 1.3 m/s threshold is 0.008 m/s, far
above floating-point differences between platforms. Not re-run on Windows.

The locked development baseline stays `final_v5` (`baseline_snapshot/`, `final.DECODER_V5`,
bench 0.6612): every earlier section's numbers are against it.
