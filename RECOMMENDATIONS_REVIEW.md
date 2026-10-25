# Review of `more_than_a_hack_2027_recommendations.md`

Each recommendation is checked against what this repo has already measured (`INVESTIGATION.md`,
`analysis/`, `mtah/bench.py`). The document predates the current pipeline: it quotes 0.469 as our
score, and the benchmark now stands at about 0.575 (E6 features with the decoder retuned).

**Two general cautions.**
- **Its expected-gain figures (+0.05 to +0.09 and so on) have no measurement behind them.** Treat
  them as guesses. Several things it calls high-impact have already been measured here as
  neutral (boundary snapping) or harmful (rule weights fed into the decoder).
- **Some citations could not be checked**, for example the arXiv 2607.* papers. Nothing below
  rests on them.

Verdicts: **Do** = implement and measure on the bench. **Test** = cheap experiment, keep only if
the bench improves. **Defer** = plausible but costly; only after cheaper wins. **No** = already
done, wrong for this scorer, or blocked.

## Section 1 / 2.5: decoding

| # | recommendation | verdict | why |
|---|---|---|---|
| P0 / 12 / 15 | Boundary-regression head that lowers Viterbi switch cost locally | **Do** | A learned P(boundary at t) is a principled replacement for fixed costs, and the CMI/ASRF precedent is real. However, hand-coded snapping (E5) measured ≈0 here, because the E6/E7 model already uses touch timing. Expect modest gains, not +0.05–0.09. |
| 16 | Minimum-duration filter / micro-spell absorption | **Test** | The ceiling analysis says blips are the costliest error, but Viterbi switch costs already suppress most of them. The current output still has 19 dead-ball spells under 1 s, which are almost never real. Cheap to test. |
| 17a | Class priors / temperature tuned with Optuna | **Test, differently** | Tuning priors on 17 public possessions overfits. Use unsupervised label-shift correction instead (EM on our match's predicted class mix, Saerens et al. 2002), which needs no labels. |
| 17b | Write `X` on high-entropy boundary frames | **No** | Wrong for this scorer. `possession_metrics.fill_abstentions` forward-fills X before spellF1, so an X at a turnover carries the old label across it. macroF1 counts X as a miss. X on a graded frame can only lose. |
| P0 | Smart Tagging as anchors for restarts and D | **No (measured)** | Checked against the public GT. The nearest `SET PIECES` tag starts 3–12 s *before* each true restart and lasts tens of seconds, so it cannot anchor a boundary to the frame. Attacking- and defending-phase team tags agree with the true team on only 77% of live frames, below the model's ~90%. The external training data has none, so a model cannot learn to weight them either. |

## 2.1: pre-processing

| # | recommendation | verdict | why |
|---|---|---|---|
| 1 | Fuse broadcast + tactical tracking | **No for now (measured)** | `analysis/broadcast_align.py` aligns the two ATDs by player configuration. Alignment works: 80% of sampled broadcast frames match a tactical frame, with a median error of 0.56 m. But on the half of graded play that confident alignments cover, the broadcast ball fills only **6.5%** of the frames where the tactical ball is missing. Both trackers lose the ball in the same situations. Even full coverage would cut missing ball by only ~5 points, so a full piecewise sync with replay detection is poor value. |
| 1b | Savitzky-Golay homography smoothing | **No** | We never project through the homography. The tactical ATD is already in pitch coordinates. |
| 2 | Tracklet split/stitch by velocity feasibility | **Partly** | Features are team-level sets, so slot identity barely matters. One place it does: per-slot forward velocity over ±1 s spikes on identity switches. Clipping implausible slot speeds (>11.5 m/s) to NaN is the cheap fix. Full stitching is not worth it. |
| 3 | Two-regime ball imputation | **Partly** | Short gaps are already interpolated (≤0.5 s), and 85% of missing frames sit in gaps over 10 s, so spline bridges recover little. Add `frames_until_ball_seen` (we only have "since") and a player-convergence ball estimate as a feature. Train a learned ball regressor on external data, where the true ball is known, only if those help. |
| 4 | Video: cut/replay, whistle audio, jersey colour at the ball | **Blocked** | Good ideas, especially jersey colour for team identity, but the videos are still missing. Revisit when they arrive. |

## 2.2: features

| # | recommendation | verdict | why |
|---|---|---|---|
| 5 | Permutation-invariant k-nearest features | **Mostly done; extend** | All features are already team-level, with no slot-indexed columns. Add k = 2, 3 distances, approach speed towards the ball and velocity–ball angle. |
| 6 | Pitch control at the ball and at the ball's position 1 s ahead | **Test** | A simplified time-to-intercept model at 2 points per frame is cheap, and it is a sensible stabiliser mid-pass. |
| 7 | Causal / anticausal windows, before–after contrast | **Done in part** | E1 already adds lead-minus-lag trends at ±1 s and ±2.5 s and long windows (+0.014). Extending to ±4 s and ±10 s is cheap; fold it into the feature batch. |
| 8 | Smart Tagging as time since/until features | **No as model features**, see above | Not present in the external training data. |
| 8b | Dead-ball kinematic signature | **Done** | Pace, ball-out flags and rule internals (E7). |

## 2.3: augmentation and validation

| # | recommendation | verdict | why |
|---|---|---|---|
| 9 | Transplant our exact ball-visibility mask | **Already done, better** | `mtah/noise.py` resamples our real gap lengths and biases gaps towards dead ball. Pasting raw slices of our mask would break that dead-ball correlation. Multi-level noise augmentation (E3) is queued. |
| 10 | Left/right flip; home/away swap with team blocks swapped | **Do** | Cheap. The swap also removes any home/away bias learned from Bundesliga teams, which means nothing for Barça. |
| 11 | Noisy-student on our unlabelled 85 minutes | **Test later** | It failed in round 2 with the weak model, but the current model is much stronger, so in-domain pseudo-labels may now help. Mixing in the public GT (×3 weight) would destroy our only in-domain validation, so do that only in the final model, if at all. |

## 2.4: models

| # | recommendation | verdict | why |
|---|---|---|---|
| 13 | MS-TCN++ / BiGRU refiner over out-of-fold probabilities | **Defer** | Plausible, but it adds torch, GPU non-determinism and a lot of effort. Try it after the cheap items if the bench stalls. |
| 14 | Multi-family ensemble (LightGBM, CatBoost, TCN) | **Defer** | LightGBM cannot load here (no libomp). CatBoost could add diversity later. |

## 3: workflow

| # | recommendation | verdict | why |
|---|---|---|---|
| 3.1 | Error taxonomy: split / merge / drift / confusion | **Do** | Cheap, and it tells us which knob to turn. |
| 3.2 | Don't tune on the public window | **Already practice** | The bench ranks on external + masked. Public is a sanity check. |
| 3.3 | Determinism, validator, public before deadline | **Already done** | Byte-identical reruns, validator in `run.py`. CI cannot run because the data cannot leave this machine. |
| 3.4 | Technical document with an architecture diagram | **Do, at the end** | W9. |

---

# Review of the second feedback (code audit)

Every claim was checked against the code and the data before acting.

| # | claim | verified? | verdict |
|---|---|---|---|
| 1a | Temporal filters bleed across the 1-frame gap between IDSSE halves (windows up to ±10 s, counters to 30 s) | **Yes.** `parse_idsse` leaves 1 empty frame between halves, and teams switch ends there | **Fix.** Pad the halves with a gap longer than every window (1,500 frames), then rebuild the external caches. Affects ~1% of training frames, but they sit at kickoffs, exactly where we are weak |
| 1b | `np.roll` in `long_context` wraps the match end onto frame 0 | **Yes**, up to 125 frames | **Fix**: edge-clamp the shift |
| 2a | Masked validation looks ahead past the public window's end | **Yes.** `ball_mask` stops at frame 18,685, and look-ahead features see the ball just after it | **Fix**: also hide the next 30 s |
| 2b | Masked gaps are placed uniformly, but real ball loss follows stoppages | **Yes** | **Fix**: weight gap starts by our own slow-pace signal (no labels used) |
| 3a | The 2,934 pre-kickoff frames of the 2nd half produce spurious possessions | **Yes**: `fused_ext_v2` has 4–5 H spells there; `rules_v2` has 5 | **Fix**: algorithmic kickoff detection (kickoff formation + ball leaving the centre spot), with D before each half's kickoff |
| 3b | The 1st-half kickoff is mislabelled | **Yes**: we predict D 20 frames, then H; truth is A from frame 1 | **Fix**: the kicking team (the side at the ball in the kickoff formation) owns the restart. Testable on all 18 external half-starts |
| 4a | The videos are present | **Yes, as of ~16:30–17:00 today.** Earlier notes saying "missing" are stale | **Unblocks W7.** Ball detection runs locally. Hand-labelling must be done by a person on the team; Claude cannot and must not view the video (T&Cs) |
| 4b | Smart Tagging kickoff tags are unused | **Yes** | Use the formation detector as primary. Tags were measured as loose (±3–12 s), so they are a fallback only |
| 5a | 80% of training rows carry IDSSE's touch-happy convention | **Yes** (4.3 vs 3.4 possessions/min) | **Test**: absorb same-team-flanked opposite-team blips under 1 s in IDSSE labels; judge on masked + public + Metrica |
| 5b | `final.py` fits one subsample offset, not 5 | **Yes** | **Low priority.** With 54 match-versions the data are already huge and the offsets are near-duplicate frames. 5× cost for little variance gain; revisit if time allows |
| 6a | Probabilities are uncalibrated | Partly | `w_shape` is a global temperature, and it is already tuned (0.75). Temperature scaling would add nothing. EM prior shift is already in testing |
| 6b | Static switch cost is a compromise | Yes | The learned boundary head (`w_bound`) is under test. It is the learned version of "cheap to switch at a visible touch". `min_dead` is also under test |
| 7a | Pitch control ignores player velocity | Yes | **Skip.** Per-slot velocities spike on identity switches, so the gain is uncertain |
| 7b | No set-piece geometry | Yes | **Add as features**: own-half symmetry index (kickoffs), players in each box, ball/camera distance to corner and touchline |
| 3 (gain claims) | +0.05–0.10 and so on | — | Unmeasured. Each item is kept or dropped on the bench |

# Review of the third feedback (video detector: label box, tiles, linking)

It promised ~90% on both video metrics. Checked against `cache/ball_det_ft.npz` and then on the
bench (INVESTIGATION.md §9.1). Its numbers had no measurement behind them.

| # | claim | verified? | verdict |
|---|---|---|---|
| 1 | `ball_box` puts the ATD point on the label box's bottom edge, so the ball often falls outside the 1.2 m region | **Yes, as a design flaw, not a train/detect mismatch** (detection also maps the box bottom back). With unbiased ~9 px ATD error, about half the labels have the ball centre below the box, where YOLO's assigner picks no positives | **Defer.** Centre the box and map the centre back (keep the COCO path on the box bottom). Better still: snap labels to round-1 detections and retrain with tight boxes. Only worth it once the model can use video at all |
| 2A | 3-frame temporal input separates the ball from socks and markings | Plausible (TrackNet-style), unmeasured | **Defer.** The camera pans, so neighbouring frames must be warped through the per-frame homographies first. A dataset and detect rewrite |
| 2B | `TILES` has zero horizontal overlap; overlapping tiles are not deduplicated | **Yes, both.** 79% of frames have two candidates < 0.8 m apart | **Defer** to the next detector run: 4×2 grid plus cross-tile NMS. Small effect, since the ball is near a seam on only a few % of frames |
| 2C | Mine hard frames: near players and near ATD dropouts | Plausible | **Defer.** Near-player frames yes. Near-dropout ATD labels are the least reliable ones |
| 2D | Re-rank the top 5 with pitch and player priors (spare balls off the pitch) | **No.** The true ball is ranked #1 on 97% of frames where it is present; only 3.4% of wrong top picks are off the pitch | **No.** At most ~2 points of top-1 |
| 3 | The 26% "plausible" probe is artificially low (conjunction with the previous sample, top-1 only, < 4 m from a player) | **Yes.** 22% of correct detections are ≥ 4 m from every player | Agreed, but it is only a proxy; it does not measure accuracy either way |
| 3 (fix) | Replace greedy `merge_track` with a gap-anchored linker over all 5 candidates: "0 GPU hours", "80–90% coverage" | **Built and measured.** Correct coverage on masked frames 11% → 64–72%; on real missing frames 10% → 39% filled | **No (measured).** Every variant lowers the bench: 0.656 → 0.629–0.635. More coverage scores worse |

# Review of the fourth feedback (video integration)

Written from the output of `analysis/video_score.py`. Its diagnosis is mostly right; its fix is
not.

| # | claim | verified? | verdict |
|---|---|---|---|
| Paradox | Video helps the rules but hurts `final_v1` (masked 0.695 → 0.666 / 0.676) | **Yes** for `final_v1`; the rules figure was not re-run | — |
| A | Region labels squash YOLO's IoU-aware confidence, so conf ≥ 0.25 drops most real detections | **Yes.** True-ball median conf 0.13, 79% below 0.25; coverage figures match ours | Agreed. "The detector sees the ball on nearly every live missing frame" does **not** follow: a candidate is not the ball |
| B | Box jitter fakes > 3 m/s and destroys dead-ball detection | **Half.** Its arithmetic uses the 0.65 m error vs the ATD as frame-to-frame jitter and ±2-frame differences (the code uses ±1 then a 5-frame mean); real median fake speed is 1.2–1.5 m/s. Fake > 3 m/s does happen on 17–24% of stationary frames | **Smooth if video is ever used** (Savitzky–Golay 15 → 4%). "Destroys D" is backwards: dead-ball precision rises 57% → 81–85%, F1 ≈ 0.67 → 0.74 |
| C | Sparse video points create a train/test shift: `seen`, `since`, `until` never look like this in the noised external data | **Yes**, from the code (`feats.py`, `rules.dead_mask`, `noise.py`) | **The cause.** Confirmed by the bench: more video, worse score, public included |
| Fix 1 | Linker at conf ≥ 0.07 plus smoothing makes video a net gain, no Colab needed | **Measured: no.** Best linker 0.635 against 0.656 | **No** |
| Fix 2 | Zero-retrain decoupling: `seen` / `since` / `until` from the ATD only, video fills `bx`, `by`, `dh`, `da` | `feats.py` blanks those columns wherever `seen` is false; changing that feeds `seen = 0` with a finite ball, a combination absent from training | **No.** It swaps one shift for another |
| Fix 3 | Simulate video recovery in `mtah/noise.py` so training matches test | Principled. Its version (35% of gap frames, 0.65 m Gaussian) leaves out the ~15% false positives that cause spurious switches | **Do, if video is pursued**: own columns and own `seen`, calibrated on measured coverage, clumpiness and error rate. Needs a full refit; payoff uncertain |
| 2 | Velocity-projected pitch control and `dv` touch features | `dv` exists; a simplified pitch control exists | **Test**, small. Only the velocity projection is new (see 7a above on velocity spikes) |
| 3 | Synthetic camera proxy on external data; late fusion of Smart Tagging | Tagging already measured (±3–12 s, 77% team agreement); the camera centre is a 12 m-median ball proxy | **No** for tagging (a fusion weight could only be tuned on 17 possessions). Camera proxy: **Defer** |
| 4 | Minimum dead-ball dwell inside Viterbi; `is_metrica` indicator | Post-hoc `min_dead = 25` already gave +0.004 | Dwell in Viterbi: **Test**, small. `is_metrica`: **No**, our match has no right value for it (IDSSE relabel measured −0.002) |
| 5 | TCN / BiGRU refiner | — | **Defer**, as before |
| Omitted | — | The detector's training pool includes the public window (`cand < 170000`) | **Fix before any further video scoring**: exclude frames 11185–19435 from training |

# Review of the fifth feedback (compute split and iteration process)

Its premise was "ignore the LLM compliance part". We cannot: the T&Cs forbid giving match data or
video to a hosted LLM, Claude included. Everything that depends on an LLM looking at frames is
rejected; the same work done by a person with local tools is fine.

| # | claim | verdict | why |
|---|---|---|---|
| 1 | Cache fold models and probabilities by default | **Do** (ROADMAP I1) | Three scripts refitted the same final_v1 models from scratch |
| 2 | Never run two refits at once on the Mac | **Do** (ROADMAP rule 5) | We did, and both slowed down |
| 3 | Fast screen (base matches, no augmentation, 300 trees) with a "+0.003 to pass" gate | **Do, as a loser filter only** (I2) | +0.003 is under the noise floor, and a screen without mirrors can rank features differently; it may drop clear losers, never adopt |
| 4 | Colab GPU for CatBoost / XGBoost / sequence models | **Test, for screening** (I4) | Plausible speed-up (unmeasured). GPU trees are not always bit-for-bit deterministic, and `run.py` must reproduce on CPU |
| 5 | Hand-labelled 2nd-half set as a third bench column | **Do** (R3) | Already planned |
| 6 | An LLM inspecting frames, transition tables, missing-ball frames; building R3 itself | **No** | T&Cs |
| 7 | A helper that pre-marks candidate boundaries from model output for the labeller | **No** | Labels steered by the model cannot validate it (PLAN.md) |
| 8 | "R3 in 45 minutes" | **No** | A few hours per window is realistic; eye labels also follow a different convention from the tracker-generated GT |
| 9 | `ball_value.py` passes stale rule labels and crashes | **No (checked)** | `full_plus3` recomputes the rules from the new features; stored rule labels are only read with `use_rules=True` |

# Review of the sixth feedback (integrated research and technical report, 2026-10-25)

A decoder-level study on cached `P_R1` plus a literature matrix. Its bottom line, "retain
`final_v4`", is not supported by its own yardsticks: it selects on ext + masked, and on those the
`final_v2` decoder + stop gate beats `v4` (bench 0.661 vs 0.636, ext 0.616 vs 0.589, masked 0.706
vs 0.683; INVESTIGATION.md §20). Its arms were all built on `v4`'s rule-voting decoder.

| # | claim | verdict | why |
|---|---|---|---|
| 1 | Retain `final_v4` weights and decoder | **No** | See above; the rule vote was chosen on eye labels that reward rule timing (§20) |
| 2 | C1 abstention (global / guarded X at 0.60): no robust gain | **Accept (their measurement)** | Matches EVALUATION.md: X is spellF1-neutral inside a possession and harmful at a hand-off; not rerun |
| 3 | C2 dwell filter 12 frames: null | **Accept** | Same as our in-Viterbi dwell (§11) |
| 4 | C3 two-stage alive/dead then team decoder: worse everywhere | **Accept** | Not rerun; decoder-level on the same probabilities |
| 5 | HGB beats a linear (SGD) model by +0.094 spellF1 on 3 held-out IDSSE matches | **Accept** | The model family is not the weak link |
| 6 | Fold-probability results were never retested on the full fit | **Do** | Applies to `final_v5` too: the delivered full-fit file is scored on public and val_h2 before adoption |
| 7 | Keep the delivered-file and fold-ensemble public numbers apart (0.653 vs 0.667 for `v4`) | **Do** | README and the technical document label which is which |
| 8 | Report external results per provider | **Done** | `analysis/yardsticks.py` prints IDSSE and Metrica separately |
| 9 | Metrica sample-data licence unresolved (no LICENSE upstream) | **Do** | One line in the technical document's assets section |
| 10 | O1 fresh independently labelled matches as top priority | **Defer** | No ready source in our convention (PFF needs a personal sign-up; SkillCorner has no per-frame possession); not before the 14 Nov freeze |
| 11 | O3 duration prior + weak labels | **No** | Dwell and minimum-possession variants already null (§11, item 3) |
| 12 | Cached runner with a drift guard (`research/possession_discovery_2026/`) | **Defer** | Good engineering, but pinned to `v4`'s hashes; `run.py` stays canonical |
