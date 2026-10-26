# Phys228 typed transition head integration — result

Date 2026-10-26. Integration of 3 saved phys228 binary head models (228 features:
full_plus3 226 + 2 phys features). No new models trained.

## Setup
- Models: cache/ar_phys228_fold{0,1,2}.joblib, HistGBDT, 100 trees, classes [0,1], pos_rate ~1.2%.
- Extra features: closing_speed_margin = gradient(da - dh); ball_motion_aligned = vx * sign(da - dh).
- Inputs from masked f dict only; no raw player positions.
- Cost delta: cost_team_t = 12 - λ*(logodds(p_head) - logodds(prior=0.012)). λ=2, amp=±4.
- D/dead cost unchanged. min_dead=25, stop_gate=100, kickoff=half, w_shape=0.75.

## Identity check (λ=0)
bench 0.6612 | ext 0.6163 | masked 0.7061 | public 0.6968/0.9098 | val 0.5792
Exact match to frozen final_v5. Custom Viterbi λ=0 identity verified.

## λ=2 result
| metric | λ=0 | phys228 λ=2 | delta | 226 head λ=2 |
|---|---|---|---|---|
| bench | 0.6612 | 0.6577 | -0.0035 | 0.6573 |
| ext | 0.6163 | 0.6220 | +0.0057 | 0.6220 |
| masked | 0.7061 | 0.6934 | -0.0127 | 0.6926 |
| public spellF1 | 0.6968 | 0.6710 | -0.0258 | 0.6710 |
| public macroF1 | 0.9098 | 0.9103 | +0.0005 | 0.9102 |
| val spellF1 | 0.5792 | 0.5835 | +0.0043 | 0.5835 |

## Result: REJECTED
- phys228 head λ=2 is within +0.0004 bench of 226 head λ=2: ext +0.0057 same, bench/masked
  slightly less negative but still below gate. The +2 phys features (gradient of nearest-opp
  distance margin, ball-velocity-aligned direction proxy) do not add measurable signal in this
  100-tree head at fixed λ=2. This does NOT rule out a different physical model architecture
  with controlled persistence/latching; it only says these two proxy features under this head
  do not move the metric.
- bench -0.0035, masked -0.0127, public spellF1 -0.0258. Fails adoption rule.
- No promotion. Best remains frozen final_v5.

## Semantic corrections
- "No public overfit" → no public labels enter training; masked/public are development yardsticks
  with repeated selection across this session.
- "Features don't add signal" → under-measured: +0.0004 bench vs 226 head at fixed λ=2,
  negligible; cannot rule out a different physical model architecture.
- Extra features are gradient(da - dh) and vx * sign(da - dh): a distance-margin rate and a
  ball-velocity direction proxy, not true player-relative velocity or persistence control.

## Hashes
- cache/ar_phys228_fold0.joblib edef66e5…44c
- cache/ar_phys228_fold1.joblib 1678e008…5e9
- cache/ar_phys228_fold2.joblib 6f8ff6d8…71d
- cache/ar_phys228_head_probs.npz acb262bc…078
- cache/P_R1.npz 5c742d75…9d7f

## Artifacts
- analysis/autonomous_research/experiments/followup_phys228_integration/integrate_phys.py
- analysis/autonomous_research/experiments/followup_phys228_integration/integrate_console.txt
- cache/ar_phys228_head_probs.npz (saved head predictions, ignored)
