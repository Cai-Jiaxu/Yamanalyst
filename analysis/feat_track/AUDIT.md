# Candidate feature audit: strictly causal, half-reset (compliance version)

Per auditor: the NEW centered windows are NEW future dependence and must NOT inherit the
full_plus3 assumption. All new blocks are now PAST-ONLY and reset at half boundaries. The existing
full_plus3 code is unchanged and keeps its disclosed assumption.

## Time dependence (final)

| block | lookahead | lookback | half-break reset | notes |
|---|---|---|---|---|
| c_carrier_tightness | 0 | trailing w=12 tightness mean + backward closing | yes | carrier sign/tightness instantaneous |
| c_centroid_compression | 0 | backward centroid velocity w=12 | yes | no future frame |
| c_margin_dwell | 0 | causal run-length; trailing flip rate w=50; backward slope w=12 | yes | flips across a half break are not counted |
| c_melee_uncertainty | 0 | causal swap flag; trailing distinct near-ball slots w=50 | yes | |
| c_ballfree_cluster | 0 | trailing smooth w=25 | yes | ball-free |

No block reads the label, undegraded external observations, or any future frame. The only
non-past quantity was removed; every rolling/dwell/derivative is computed within a half segment
using `_half_id(m)` (m.halves for both our Match and ExtMatch).

## Half-break / boundary audit

- Our match halves: (1, 11185, 90651), (2, 110808, 190197). Ungraded frames between and outside
  are segment 0; windows never blend across the gap.
- External matches: halves concatenated with a 1-frame gap; `m.halves` is the (K,3) array, used
  the same way.
- External train folds never see our-match labels; our-match probs come from the mean of fold
  models. No label leakage.

## Source eligibility

- Ball-gated blocks: NaN when ball unseen (external degraded and our-match masked stretches).
- c_ballfree_cluster: valid on external degraded and our match, including long ball gaps.
- Audio: RESEARCHED/IMPLEMENTED but NOT EVALUATED (no run performed). External matches have no
  audio, so those columns would be all-null at training and unlearnable; logged as degenerate
  control, not a candidate gain.

## Status

- Code: analysis/feat_track/feat_candidates.py (smoke verified: composed 245 cols, halves parsed).
- Screen runner: analysis/feat_track/feat_screen.py (fresh full_plus3 screen control + paired deltas, JSONL).
- Registry: analysis/feat_track/REGISTRY.json (isolated).
