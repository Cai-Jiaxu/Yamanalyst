# Yamanalyst — MTAH 2027 Phase 1: ball possession from tracking

*Technical document (SUBMISSION.md §3). Draft of 2026-10-24, for submission `final_v3`; the
numbers are refreshed at the code freeze (ROADMAP.md). Export to `Yamanalyst.pdf` for the form.*

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
| Rule outputs as **features**, not a vote in the decoder | +0.047 bench when the rule label became a feature and its decoder term was removed |
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

Also measured and not adopted: velocity-projected pitch control (no change on bench, +0.013 on `val_h2`),
CatBoost alone or averaged with the trees (−0.015 to −0.024 on the screen), self-training on our own
unlabelled match (−0.016 on bench, +0.013 on `val_h2`), a minimum dead-ball length inside the Viterbi
search (no change).

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
| **`final_v3` (submitted)**: + rule labels voting in the decoder | 0.653 | 0.904 | 0.632 | **0.627** (macroF1 0.873) |

We chose `final_v3` on our own hand-labelled 2nd half, against the external and public scores:
the model's team calls drift later in the match even with the ball visible, and the rules, which
read the ball directly, do not.

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

With more time: a dedicated video-ball channel trained on simulated stoppage/tracklet noise, and
stoppage-ball gating when player pace is below 1.3 m/s and no player is within 5 m of the ball.

## 7. Reproduction

```bash
uv sync                                   # Python 3.11 via uv
# match data into data/ (DATA.md); external data:
bash scripts/download_external.sh && uv run python -m mtah.external
uv run python run.py --name final_v3      # -> evaluation/results/final_v3.csv
python evaluation/validate_submission.py evaluation/results/final_v3.csv
```

Deterministic: the output's sha256 starts `3b01921e`. First run ~30–60 min (caches), then ~11 min.
Every number above comes from a script in `analysis/` (INVESTIGATION.md maps each to its script).
