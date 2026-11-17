# Yamanalyst — MTAH 2027 Phase 1: ball possession from tracking

*Technical document (challenge/SUBMISSION.md §3) for submission `final_v6`, 17 November 2026.
Repository: <https://github.com/Cai-Jiaxu/Yamanalyst>. Every number here is copied from
docs/INVESTIGATION.md in that repository, which maps each one to the script that produced it.*

## 1. The approach in one paragraph

For each frame of FC Barcelona vs Athletic Club we output `H`, `A` or `D`. The metric scores
whole possessions, and one spurious 0.4 s flip per minute costs as much as missing every
possession shorter than 3 s, so we optimised for **precision on changes of possession**. The
pipeline is: hand-written touch and dead-ball rules → their outputs, plus ball-free team shape
and ball features, go into **gradient-boosted trees trained on 9 external matches with true
labels**, degraded to look like our tracker → a **3-state Viterbi decoder** with switching costs.
The ball is missing on 41% of graded frames, mostly in gaps longer than 10 s, so the model must
read possession from the players when the ball is not there.

## 2. Why this design and not another

| decision | evidence |
|---|---|
| Precision over recall on switches | Error injection on the true labels: one 0.4 s blip/min → spellF1 0.77; missing every possession < 3 s → 0.83 |
| Learn from **external** labelled matches, not from our 17 public possessions | The shape model trained on external true labels gets 93% team accuracy without the ball, against 83% from our own pseudo-labels; masked spellF1 +0.12 |
| Degrade the external data to our tracker | Clean → degraded external tracking drops the rules from 0.61 to 0.37 spellF1. We resample our real ball-gap lengths, bias gaps towards dead ball (3×), drop players to our ~20.5 tracked, add team mislabels and position noise |
| Rule outputs as **features**, not a vote in the decoder | +0.047 bench when the rule label became a feature and its decoder term was removed. Re-checked at `final_v5`: on the real ground truth the rules switch possession a median 1.6 s before it does (at the pass, not the controlled touch); a rule vote costs 0.025–0.03 on every real-GT yardstick |
| Judge boundary timing only on real ground truth | Our eye-labelled 2nd-half windows put possession changes where the rules do (0–2 frames), so the rules alone, last on every real-GT yardstick, score best there. We use those windows only for timing-free checks: mid-possession team calls, dead ball, spell counts |
| Mirrored (left/right, home/away) and extra-noise copies of every external match | +0.039, the largest single gain |
| Choose on a benchmark, never on the public window | The public window is the best-tracked part of the match (ball seen 88–99% of the time there, 59% over the match) and has 17 possessions. We rank on external matches (fold by match, ~3,000 possessions) plus our public window with the ball hidden the way the full match loses it (16 seeds) |

## 3. Assets: used, tried and dropped

| asset | used? | how / why not |
|---|---|---|
| Tactical ATD (players, ball, homography) | **yes** | everything; positions already in pitch metres |
| Public 5-minute GT | validation only | sanity check; never tuned on |
| IDSSE (7 Bundesliga matches, CC BY 4.0) and Metrica sample games 1–2 | **yes** | training data with true per-frame labels |
| Tactical video | tried, **dropped** | A YOLO fine-tuned on our camera with ATD-projected labels finds the ball within 2 m on 79% of held-out frames. A 300-frame visual probe of genuine missing-ball frames shows 67.0% are stoppages (`D`), 17.3% occluded/off-screen (`O`), and only 15.7% visible live play (`V`); 69.6% of DP-linked video detections occur during stoppages, lowering the benchmark (0.656 → 0.622–0.642) and `val_h2` (0.558 → 0.536) |
| Broadcast ATD | tried, **dropped** | aligned by player configuration (median 0.56 m), but it fills only 6.5% of the tactical tracker's missing ball: both lose it in the same situations |
| Smart Tagging | tried, **dropped** | set-piece tags start 3–12 s before the true restart; team-phase tags agree with the true team on 77% of live frames, below the model's ~90% |
| PFF World Cup data | not used | personal sign-up required |

The Metrica sample data carries no licence file upstream (its README asks for attribution and responsible use); we use it for training only and do not redistribute it. IDSSE is CC BY 4.0 (Bassek et al., *Sci Data* 2025).

Also measured and not adopted: velocity-projected pitch control (no change on bench, +0.013 on `val_h2`),
CatBoost alone or averaged with the trees (−0.015 to −0.024 on the screen), self-training on our own
unlabelled match (−0.016 on bench, +0.013 on `val_h2`), a minimum dead-ball length inside the Viterbi
search (no change), minimum-Bayes-risk decoding for expected spellF1 (+0.001), and a learned temporal
refiner (bidirectional GRU or dilated TCN) over the trees' probabilities: +0.011–0.017 on held-out external
matches but −0.02–0.03 on our own masked window, because it learns the external sources' label conventions. Touch-sequence and kick features aimed at
short possessions: +0.040 on the fast screen, +0.001 once the full training recipe is used.

After `final_v5` we measured four more groups of ideas against it; one small change came out of
them (the last bullet):

- **Decoder and training variants**: semi-Markov durations (0.657 against 0.661), extra switch cost
  in duels (0.660), learned transition heads (0.657–0.658), duel-weighted training (0.658), a hard
  physical "control latch" (0.303), logit sharpening (benchmark up to 0.667 but public down to
  0.686 on the full fit).
- **Five past-only tracking feature blocks** (carrier tightness, margin dwell and others): +0.016
  to +0.031 on the fast screen, 0.650–0.659 with the full recipe. Four times now a screen gain has
  vanished once mirrors and noise copies are in training.
- **Audio**: the video's sound track is aligned and a tonal whistle detector fires within 1.4 s of
  the three known whistles, but forcing dead ball at detected whistles does nothing or hurts
  (public 0.700), and the external training matches have no audio to learn from.
- **Stoppage-gate variants** (§5): gating on "players walk and nobody within 5 m of the ball"
  removes none of the remaining stoppage errors; longer or looser gates lose slightly. The one
  variant that helped is the last step of `final_v6`.

## 4. The possession rules we implemented

These follow how the ground truth is built (checked on the public window and on external data):

- **Change of possession** at the new team's **first touch of a controlled possession**: an
  episode with a player within 1.5 m of the ball and the other team at least 0.5 m farther,
  confirmed by holding the ball ~1 s or touching it again within 4 s before the opponent does.
- **Loose balls and challenges** stay with the last team. A touch without control (a block, a
  clearance the other side recovers) does not switch.
- **Dead ball** starts when the ball goes out, when it sits still for ~1.5 s with no one on it,
  when both teams walk for 4 s, or during a long missing-ball stretch with slow players. It ends
  on the frame the ball moves faster than 3 m/s (the restart kick). Dead spells last at least 1 s.
- **Stoppage blips**: a live spell of at most 4 s that touches a dead-ball spell while both teams
  walk (mean pace under 1.3 m/s) is the tracker following a ball being returned or placed, and
  becomes dead ball.
- **Half starts**: everything before a half's first kickoff is dead ball. The kickoff is detected
  from the formation (both teams in their own halves, standing) and the ball leaving the centre
  spot. In our match the 2nd half is graded 2,934 frames before its kickoff.

The trees learn when to trust each rule. The decoder pays 12 nats per H↔A switch and 12 per
alive↔dead switch.

## 5. Results

| version | public spellF1 | public macroF1 | benchmark | `val_h2` (20 min 2nd half, 57 spells) |
|---|---|---|---|---|
| official baseline (smooth50) | 0.326 | 0.621 | — | — |
| rules only (`rules_v2`) | 0.461 | 0.852 | — | 0.655 (macroF1 0.865, collarMF1 0.883) |
| `final_v2` | 0.700 | 0.898 | 0.656 | 0.577 (lowIoU 0.722, teamAcc 0.891, deadF1 0.938, collarMF1 0.895) |
| `final_v3`: + rule labels voting in the decoder | 0.653 | 0.904 | 0.632 | 0.627 (macroF1 0.873) |
| `final_v4`: + stoppage gate | 0.653 | 0.904 | 0.636 | 0.632 (macroF1 0.875) |
| `final_v5`: `final_v2` decoder + stoppage gate | 0.723 | 0.910 | 0.661 | 0.593 |
| **`final_v6` (submitted)**: + stoppage gate with dead ball on one side only | **0.747** | **0.923** | **0.666** | 0.604 |

Public columns are the delivered full-fit files; the benchmark uses the fold models (three models,
each scored on external matches it never saw). The benchmark is the mean of the external score and
the masked public-window score; we adopt a change only if it rises by at least 0.005 with neither
half falling by more than 0.005, and we record every attempt. `final_v3`/`v4`
were chosen on the hand-labelled 2nd half; `final_v5` reverses that because those labels reward the
rules' timing (§2): on the benchmark's external matches `final_v5` scores 0.616 against 0.589, on the
masked window 0.706 against 0.683. Mid-possession (> 3 s from a change) its team error on the 2nd-half
windows is 4.4%, the same as `final_v4`'s 4.3%.

`final_v6` differs from `final_v5` on 5 spells (299 frames, 0.19% of graded frames), all turned
into dead ball. `final_v5`'s gate removed a live spell of at most 4 s between two dead-ball spells
while the players walk (mean pace under 1.3 m/s); `final_v6` runs a second pass that accepts dead
ball on one side only. On the benchmark this is a marginal call: +0.005 overall, −0.002 on the
external matches, and it was the one success among six gate variants. We adopted it because the
change to the delivered file is small enough to inspect: of the 5 spells, the 3 that fall on
frames we have labels for (1 in the public window, 2 in our hand-labelled windows) all have no
true possession of that team under them. The other 2 we cannot check.

Remaining errors on external matches, per 100 true possessions: missed 22, boundary drift 19,
merged 11, spurious 3.6, split 2.1. A large part of "missed" is IDSSE's very brief possessions,
which our ground truth's convention would not count.

## 6. Limitations

- **The ball is missing on 41% of frames** and the model then reads possession from player shape
  alone; boundaries inside long gaps are the weakest part.
- **Stoppage-time ball movement**: when the tracker observes a spare or returned ball moving during
  a stoppage while players walk, the trees can emit short spurious live spells (confirmed on both
  the public window and the four 5-minute 2nd-half `val_h2` windows).
- **The decoder is conservative** (0.71 predicted possessions per true one on external data): it
  misses short possessions to avoid spurious ones.
- **Video**: 67.0% of genuine missing-ball frames are stoppages (`D`) and 17.3% are occluded/off-screen
  (`O`), so integrating video detections requires a dedicated channel trained with simulated
  stoppage false positives and clumpy tracklets.

- **Short possessions**: 20–30% of true possessions on the external matches last under 2 s and we
  recover 19% of them; touch-sequence features did not change that.
- **Our yardsticks are small or foreign**: 17 real possessions on our own match, ~3,000 on matches
  from other leagues and trackers with their own labelling habits. We never measured the
  benchmark's run-to-run noise, so the 0.005 adoption margin is a working rule, not a test.

With more time: a dedicated video-ball channel trained on simulated stoppage and tracklet noise;
labelled referee whistles to test audio as a dead-ball cue; a second labelled match from the same
tracker, which would do more than any model change above.

## 7. Reproduction

```bash
uv sync                                   # Python 3.11 via uv
# match data into data/ (DATA.md); external data:
bash scripts/download_external.sh && uv run python -m mtah.external
uv run python run.py --name final_v6      # -> evaluation/results/final_v6.csv
python evaluation/validate_submission.py evaluation/results/final_v6.csv
```

Deterministic: the output's sha256 is
`349a7df54c8e0364e91e060b6279a2e7236bf5bfefc813e0bb63795b2e640ea3`; the submitted CSV is this
file under our team name. Two independent full refits on macOS (Apple silicon) give these bytes.
`final_v5` (the same pipeline without the last gate pass, sha256
`4089da23…`) was reproduced byte for byte on macOS and in two independent full refits on Windows. First run
~30–60 min (caches), then ~11 min. `bash scripts/check.sh` runs the tests that need no data,
including ones that pin the decoder settings and the CSV bytes. Dependencies are pinned in
`pyproject.toml` and `uv.lock`. Every number above comes from a script in `analysis/`
(docs/INVESTIGATION.md maps each to its script).
