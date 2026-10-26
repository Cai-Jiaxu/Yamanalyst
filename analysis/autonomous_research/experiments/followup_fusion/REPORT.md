# Follow-up fusion audit — projected out-ball / broadcast coverage

Date 2026-10-25. Decoder-level cycle on cached P_R1.npz, no refit, no manual labels.

## Corrected coverage

### Item 1 — broadcast tactical/broadcast fusion
My first early assessment ("blocked/closed from games.py constants") was WRONG.
`analysis/broadcast_align.py` already aligns broadcast ATD to the tactical timeline by
player-configuration chamfer (not a fixed offset). Measured now:

- 80% of broadcast samples match tactical at <1.5 m; median chamfer 0.56 m.
- Consecutive confident samples agree within 3 frames 83% of the time.
- Offset range -7850..31655 frames (wide because of replays/close-ups; no single offset works).
- Broadcast fills 6.5% of tactical missing-ball frames (40.7% -> 38.0%).

So broadcast is feasible, not blocked. The distinct remaining angle is confidence/replay/
time-warp filtering (the 20% non-matching samples are replays/close-ups), not a repeat of the
matched alignment. R2's perfect-ball upper bound (+0.005 bench) does not prove other fusion
signals cannot help. The "+4014 offset" has no source anywhere in the repo — unsupported, not
a verified physical offset; real per-half tracking offsets are 1702057/1702058.

### Item 3 — projected out-ball / kickoff priors
- Kickoff priors already implemented (mtah/kickoff.py evidence()), R1 -> final_v2.
- Tactical ball coords are normalized 0..1 but range x -0.065..1.114, y -0.107..1.489
  (ball can sit outside the pitch). hvalid=1.0, reproj median 0.24 m.
- No existing out-of-bounds feature in feats/features/rules.
- External matches have no real homography (hvalid=1 by construction), so a homogated prior
  only applies on our match (public + masked), not ext.

## Cycle: projected out-ball prior

When the ball is SEEN outside pitch bounds (normalized x<0/>1 or y<0/>1) AND reproj error
< thresh, add a D-emission boost w_out in the frozen v5 decoder.

| w_out | reproj thr | bench | ext | masked | public sp/mf | val sp/mf |
|---|---|---|---|---|---|---|
| 0.0 (v5 baseline) | 0.5 | 0.6612 | 0.6163 | 0.7061 | 0.6968 / 0.9098 | 0.5792 / 0.8411 |
| 2.0 | 0.5 | 0.6612 | 0.6163 | 0.7061 | 0.6968 / 0.9131 | 0.5692 / 0.8415 |
| 4.0 | 0.5 | 0.6612 | 0.6163 | 0.7061 | 0.6968 / 0.9129 | 0.5692 / 0.8417 |
| 8.0 | 0.5 | 0.6612 | 0.6163 | 0.7061 | 0.6968 / 0.9129 | 0.5561 / 0.8403 |
| 4.0 | 1.0 | 0.6612 | 0.6163 | 0.7061 | 0.6968 / 0.9129 | 0.5692 / 0.8417 |

3,927 confident out-of-pitch frames (out of 4,964 out-of-bounds).

## Result: REJECTED, no adoption
- bench/ext/masked unchanged (0.6612 / 0.6163 / 0.7061) across all weights.
- public macroF1 nudges +0.003 (0.9098 -> 0.9131) but val spellF1 drops -0.010
  (0.5792 -> 0.5692 at w=4). No gain on bench/ext/masked; the out-of-pitch frames are already
  decoded as D by existing rules, so the prior is redundant.
- The confident out-of-pitch frames are already decoded as D by the existing rules/dead logic;
  the prior is redundant. No gate clears the +0.005 adoption bar on the independent yardsticks.

## Limitations
- Prior only applies on our match (external has no real homography); ext cannot move.
- This is one geometric projection, not a confidence/replay time-warp fusion.
- Game 3 stays frozen; not used as selector.

## Artifacts
- analysis/autonomous_research/experiments/followup_fusion/outball_cycle.py
- analysis/autonomous_research/experiments/followup_fusion/outball_console.txt

## Correction: f["out"] already exists
mtah/features.py:110 already builds `out` in METRES (OUT_MARGIN=0.3 m):
`out = (bx < -M) | (bx > L+M) | (by < -M) | (by > W+M)`, feeds both `dead_mask` (rules.py:48)
and the 226-dim model (feats.py:125). My earlier "no out-of-bounds feature" note was wrong; the
projected-out-ball prior was doubly redundant (existing rule + existing model feature).

## Cycle 2: replay-filtered broadcast fill as decoder prior

Distinct from broadcast_align.py static coverage: built a per-tactical-frame fusion ball-fill
array with temporal monotonic offset per half and replay-jump exclusion (drop samples whose
offset jumps >3 frames between consecutive confident matches). When tactical ball is missing but
the replay-filtered broadcast says the ball is live, suppress the D emission.

- Alignment: 647 confident matches -> 270 after replay-jump filter (jump>3).
- Replay-filtered fill covers 2.3% of graded missing frames (40.7% -> 39.7%).

| w_fill | bench | ext | masked | public sp/mf | val sp/mf |
|---|---|---|---|---|---|
| 0 (v5) | 0.6612 | 0.6163 | 0.7061 | 0.6968/0.9098 | 0.5792/0.8411 |
| 1.0 | 0.6612 | 0.6163 | 0.7061 | 0.6968/0.9098 | 0.5792/0.8411 |
| 2.0 | 0.6612 | 0.6163 | 0.7061 | 0.6968/0.9098 | 0.5792/0.8411 |
| 4.0 | 0.6638 | 0.6163 | 0.7113 | 0.6968/0.9098 | 0.5792/0.8411 |

### Result: BELOW GATE, no adoption
- ext unchanged (correct: external matches have no broadcast, cannot see this prior).
- At w=4.0: bench +0.0026 (0.6612 -> 0.6638), masked +0.0052 (0.7061 -> 0.7113).
- bench delta +0.0026 is BELOW the +0.005 adoption bar; masked +0.0052 alone does not clear it.
- public/val flat. The replay filter (jump>3) is conservative and keeps only 270/647 matches,
  so the fill is small (2.3% of missing). A looser filter would keep more replay frames and is
  not pursued because ext cannot validate it and public/val are flat.

### Physical-prior caveat (independent review correction)
"Broadcast ball seen" is NOT proof of live possession. The broadcast ATD can track a returned
or spare ball during dead-ball periods, so suppressing D on filled frames is a proxy, not a
control-recovery of true live ball positions. This cycle did NOT recover actual ball positions;
it only applied a false-evidence D-suppression. The +0.0026 is not a "perfectly fused ball"
bound. It is a simplistic replay-jump filter on top of cached probabilities, and no universal
upper bound should be claimed from it. R2's +0.005 oracle remains the only real upper bound.

### Replay-filter note
jump>3 frames between consecutive confident samples (samples spaced SAMPLE=125 frames = 5 s at
25 fps) is the monotonic-offset cut; 83% of consecutive samples already agreed within 3 frames
in broadcast_align.py, so this filter removes the ~17% offset jumps (replays/close-ups). It is
not a fixed global offset; it is a per-half temporal consistency cut.

### Fill array for downstream combine
- Saved: cache/ar_followup_fusion_fill.npy  (bool (199319,), fill frac 0.1384)
- Script: analysis/autonomous_research/experiments/followup_fusion/fusion_cycle.py
- Predefined w_fill=4.0 (NOT selected on public; frozen as a fixed margin for any combine).
- Sequence agent may combine this fixed fill prior with its own DoG decoder (sigma 12, w 0.5)
  as ONE fixed combo, no public tuning. DoG original config remains primary.

## Cycle 3: dense frame-correspondence actual ball recovery (corrected)

The broadcast ATD is per-frame tracking already available. Built a piecewise-constant time-warp
from the 647 chamfer anchors: between adjacent anchors whose offset agrees within 3 frames (live
segment, gap <=125), every tactical frame maps to a broadcast frame via the stable offset, then
we READ the actual broadcast ball tracking at that frame. No ball-position interpolation.

- 647 anchors -> 422 live segments, 52,612 tactical frames mapped (26.4%).
- After gating (broadcast hvalid, reproj<0.5 m, bounded pitch, speed<80 m/s): 1,941 recovered
  actual ball positions (3.0% of 64,643 missing graded).
- Re-predicted with the 3 saved final_v1 fold models on rebuilt 226-dim features, frozen v5
  decoder.

| case | baseline sp/mf | injected sp/mf |
|---|---|---|
| public | 0.6968 / 0.9098 | 0.6968 / 0.9098 |

**Result: flat score, but large probability reaction at injected frames.**

| case | injected frames | label diff | prob max-diff | base sp/mf | injected sp/mf |
|---|---|---|---|---|---|
| public | ~1941 | 533 | 0.9675 | 0.6968/0.9098 | 0.6968/0.9098 |

The recovered positions flip model probabilities by up to 0.97 at injected frames and change 533
decoder labels, but spellF1/macroF1 are unchanged. The 533 label flips do not cross spell
boundaries in a way that changes spellF1. This is a public-window diagnostic only; public has no
artificial hide, so the recovered positions land at frames the model already handles.

**Correct masked recovery (identity verified, 2 seeds):**

Identity check: re-predicted baseline matches cached P_R1 exactly (seed0 0.7310/0.8945, seed1
0.6897/0.9273).

| seed | recovered / hidden | label diff | prob max-diff | base sp/mf | injected sp/mf |
|---|---|---|---|---|---|
| 0 | 85 / 4523 | 0 | 0.0764 | 0.7310/0.8945 | 0.7310/0.8945 |
| 1 | 0 / 4366 | 0 | 0.0000 | 0.6897/0.9273 | 0.6897/0.9273 |

Only 85 of 4523 hidden frames (1.9%) on seed0 overlap with recovered positions; seed1 has zero
overlap. Even on the 85 frames, no label changes. Dense broadcast correspondence is real but the
overlap with artificial masked gaps is tiny. No score change.

**Honest bounds**: recovering 3% of missing frames is a coverage number, not a performance
threshold. No claim that 3% implies <+0.005 bench; that would be inferring a metric from a percent.
R2's +0.005 oracle is the only real upper bound, and it applies to perfect ball recovery at 30%
missing, not to this 3% subset.

## All artifacts
- analysis/autonomous_research/experiments/followup_fusion/outball_cycle.py (+console)
- analysis/autonomous_research/experiments/followup_fusion/fusion_cycle.py (+console)
- analysis/autonomous_research/experiments/followup_fusion/recov_dense_cycle.py (+console)
- analysis/autonomous_research/experiments/followup_fusion/recov_masked_eval.py (+console)
- cache/ar_followup_fusion_fill.npy (bool fill array, w_fill=4 frozen)
- cache/ar_followup_fusion_recpos.npy (1,941 recovered actual positions)
