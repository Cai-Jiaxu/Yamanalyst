# Typed transition head integration — result

Date 2026-10-26. Integration of the 3 saved binary turnover head models into the frozen v5
decoder. No new models trained.

## Setup
- Head models: cache/ar_followup_baseline226_fold{0,1,2}.joblib, HistGBDT, 100 trees,
  226 features, classes [0,1], pos_rate ~1.2%.
- Target: H↔A turnover within 6 frames, excludes D.
- Cost delta: cost_team_t = 12 - λ*(logodds(p_head) - logodds(prior=0.012)).
  High head prob → lower H↔A cost. D/dead cost unchanged.
- λ=2, amp=±4. min_dead=25, stop_gate=100, kickoff=half, w_shape=0.75.

## Identity check (λ=0)
bench 0.6612 | ext 0.6163 | masked 0.7061 | public 0.6968/0.9098 | val 0.5792
Exact match to frozen final_v5. Custom Viterbi λ=0 identity verified.

## λ=2 result
| metric | λ=0 | λ=2 | delta |
|---|---|---|---|
| bench | 0.6612 | 0.6573 | -0.0039 |
| ext | 0.6163 | 0.6220 | +0.0057 |
| masked | 0.7061 | 0.6926 | -0.0135 |
| public spellF1 | 0.6968 | 0.6710 | -0.0258 |
| public macroF1 | 0.9098 | 0.9102 | +0.0004 |
| val spellF1 | 0.5792 | 0.5835 | +0.0043 |

Head probabilities: mean ~0.008 public, ~0.012 ext; max ~0.68 public, ~0.84 ext.
The head is conservative (rare positive), so the cost delta is small and only activates on
high-confidence turnover frames.

## Result: REJECTED
- bench drops -0.0039, masked drops -0.0135, public spellF1 drops -0.0258.
- ext +0.0057 is the only positive, but bench and masked fail the adoption rule
  (bench delta ≥ +0.005 AND ext/masked not down >0.005).
- No promotion. Best remains frozen final_v5.

## Artifacts
- analysis/autonomous_research/experiments/followup_typed_integration/integrate_head.py
- analysis/autonomous_research/experiments/followup_typed_integration/integrate_console.txt
- cache/ar_followup_typed_head_probs.npz (saved head predictions, ignored)

## Diagnostic: head probability stratification (external matches)

Head prob on true H↔A turnover frames is 3-5x higher than non-turnover HA frames
(mean 0.06-0.11 vs 0.017-0.026). The head has real discrimination signal.

Cost delta (extra_team, λ=2) percentiles:
- Turnover frames: median -2.5 to -3.1 (lower cost = more willing to switch), 90th +0.2 to +0.9.
- Non-turnover HA: median +0.4 to +1.7 (higher cost = less willing to switch).

Ball availability correlation:
- Head prob on ball-seen HA: 0.020-0.027.
- Head prob on ball-missing HA: 0.013-0.021.
- Small availability correlation; head slightly more confident when ball is seen.

Interpretation: the head discriminates turnover vs non-turnover, but the cost delta is modest.
At λ=2 the adjustment hurts bench (-0.0039) and masked (-0.0135). The head's signal is real but
weak enough that the cost adjustment overshoots on the benchmark. No causal claim; this is
correlational diagnostic on external eval ground truth.
