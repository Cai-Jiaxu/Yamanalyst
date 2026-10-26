# True physical possession latch v2 — result

Date 2026-10-26.

## Versions
- v1 (latch.py / true_latch.py): emission-bias by raw proximity (2m, 3 frames, +1.0 log bias).
  bench 0.6422. REJECTED.
- v1b (true_latch.py with state machine but bug): unconfirmed switches never blocked
  (out[i] stayed at baseline label). Zero delta — NOT a valid negative, implementation bug.
- v2 (true_latch_v2.py): fixed state override, vector rel velocity, direction cosine,
  same-carrier ID persistence. Measured strong negative.

## v2 formula
- Player positions normalized 0..1 -> meters via pitch (105.3 x 68.0).
- Player velocity: 3-frame central difference at 25fps, m/s.
- Per frame with ball seen: nearest home/away player.
- Control: distance <=1.5m, team margin >=0.5m, |v_ball - v_player| <=3 m/s (vector norm),
  direction cos >=0.5 (when both speeds >0.5 m/s).
- Persistence: trailing 10-frame window requires >=8 qualifying frames with same carrier ID
  (window allows up to 2 non-qualifying frames; this is a count-based filter, not a
  maximum-consecutive-gap mechanism).
- State machine: latch retains current team; H<->A switch only after opposite control
  confirmed. D/X reset latch. Ball-missing gaps >2 frames block switch acceptance.
- Outlier rejection: player >15 m/s, ball >50 m/s.

## Toy unit tests (8/8 pass)
- T1 enable=False returns labels exact
- T2 unconfirmed H->A blocked (output stays H)
- T3 confirmed H->A accepted
- T4 D resets latch
- T5 long ball gap (3+ frames) blocks switch
- T6 opposite-vector (no confirmation) blocked
- T7 single-frame confirmation accepted (edge case)
- T8 D/X intervals preserved

## Identity (enable=False, real data)
bench 0.6612 | ext 0.6163 | masked 0.7061 | public 0.6968/0.9098 | val 0.5792
Exact match to frozen final_v5.

## v2 result (enable=True)
| metric | baseline | v2 latch | delta |
|---|---|---|---|
| bench | 0.6612 | 0.3030 | -0.3582 |
| ext | 0.6163 | 0.1679 | -0.4483 |
| masked | 0.7061 | 0.4380 | -0.2680 |
| public spellF1 | 0.6968 | 0.5120 | -0.1848 |
| public macroF1 | 0.9098 | 0.8805 | -0.0293 |
| val spellF1 | 0.5792 | 0.4825 | -0.0967 |

Held-back frames: 432,308 (summed across all eval cases: public + 16 masked + 9 ext;
these are repeated time-frames where baseline proposed opposite team but latch held current,
not unique transition episodes).
Accepted latch switches: 1,386 (frames where opposite control confirmed and switch accepted).
v2 control signal on public: h_cand=15,105, a_cand=10,062, h_conf=11,839, a_conf=8,167
(total ~20k confirmed frames, ~10% of 199k). Relative speed median 4.45 m/s, p90 15.5.

## Interpretation
The physical control signal confirms on ~20k of 199k public frames (~10%). The held-back
count of 432,308 is repeated frame-level opposing-label proposals summed across all eval
cases (public + 16 masked + 9 ext) — not 434k unique H↔A transition episodes. Gating those
frame-level proposals on sparse physical evidence collapses predictions into long
monolithic stretches of one team.

This does NOT prove physics is unhelpful — it shows the strict variant (1.5m, 10-frame
same-carrier, 3 m/s vector norm) is too sparse to serve as a per-frame transition gate.
The model's emissions encode richer tactical context that raw ball-nearest-player physics
alone does not.

## REJECTED. Best remains frozen final_v5: bench 0.6612, public 0.7226/0.9098.

## Artifacts
- analysis/autonomous_research/experiments/physical_latch/true_latch_v2.py (current)
- analysis/autonomous_research/experiments/physical_latch/true_latch_v2_console.txt
- analysis/autonomous_research/experiments/physical_latch/latch.py (v1, archived)
- analysis/autonomous_research/experiments/physical_latch/true_latch.py (v1b, archived)
