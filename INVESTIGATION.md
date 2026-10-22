# What the best possible solution looks like

The investigation that drives the build. Every number below was produced by a script in `analysis/`
and can be re-run. Only aggregates were printed: no tracking rows left this machine.

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
   better when the ball is seen. It is blocked on the video files.
5. **A validation set big enough to choose between versions.** The 17 public possessions give
   ±0.04–0.07 between masking seeds. Hand-labelled validation windows from the video (allowed by
   RULES.md) are the fix, and are also blocked on the video.

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
| W7 | Video: ball detection (YOLO + homography), hand-labelled validation | **blocked: videos missing** | — |
| W8 | External tracking data: IDSSE + Metrica sample (PFF needs a personal sign-up form, skipped) | **done** | §6: masked 0.35 → 0.47 |
| W9 | Technical document and public repo hygiene | not started | — |

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

## 7. Current submission and why

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

**Video (W7, unblocked).** Off-the-shelf COCO YOLO with tiling finds the ball within 2 m on 59% of
frames where the tracker sees it. On genuine missing-ball frames its top candidate is plausible
only 18% of the time. `mtah/video_ft.py` fine-tunes a detector on this camera using ATD-projected
labels (the projection is unbiased: +0.5 / +0.8 px). It runs on Colab GPU via
`colab/ball_detector.ipynb`, with data in the team's Google Drive and Gemini off.
