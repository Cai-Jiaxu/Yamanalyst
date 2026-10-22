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
