# Feature track, 2026-10-26: causal tracking features (not adopted)

Five new label-free feature blocks added on top of `feats.full_plus3`, screened, and the two best
given the full training recipe. **None beat `final_v5`.** Evidence entry:
[docs/INVESTIGATION.md §26.2](../../docs/INVESTIGATION.md). Moved here from `analysis/` root on
2026-10-26 (the earlier hand-off note, whose open questions were answered by these runs, is in the
local `_archive/feat_track/`).

## Files

| file | what |
|---|---|
| `feat_candidates.py` | the 5 extractors and `compose(base, names)` → a featfn for `Bench` |
| `AUDIT.md` | compliance: every new block is past-only and resets at half boundaries |
| `test_feat_candidates.py` | unit tests of the causal primitive `_trail` (no-future, half reset) |
| `feat_screen.py` | fast screen (9 base matches, no augmentation, 300 trees) with a fresh `full_plus3` control |
| `feat_full.py` | full recipe (600 trees, mirrors, extra-noise copies), cached with `key=full_cmp__…` |
| `feat_full_diverse.py` | full recipe for `c_ballfree_cluster`, `c_melee_uncertainty` — **started, never finished** (see below) |
| `run_screen_when_free.sh` | waits for a PID, then runs the screen under `caffeinate` |
| `feat_screen_results.jsonl`, `feat_full_results.jsonl` | one JSON line per run (aggregates only) |
| `feat_screen_run.log`, `feat_full_run.log` | console logs |
| `REGISTRY.json` | locked baseline, gates, per-candidate status |

Run from the repo root: `.venv/bin/python analysis/feat_track/feat_screen.py` (~10 min per block),
`caffeinate -i .venv/bin/python analysis/feat_track/feat_full.py` (~45–50 min per block; one refit
at a time).

## The blocks

| block | cols | idea | needs ball? |
|---|---|---|---|
| `c_carrier_tightness` | 5 | carrier-to-nearest-opponent distance and closing speed | yes |
| `c_centroid_compression` | 3 | team centroid velocity toward the ball | yes |
| `c_margin_dwell` | 4 | stability / flip rate of which team is nearest (a duel should not flip the label) | yes |
| `c_melee_uncertainty` | 3 | nearest-slot identity swaps, distinct bodies near the ball | yes |
| `c_ballfree_cluster` | 4 | ball-free inter-team spacing and longitudinal overlap | no |

## Results

Screen (control `full_plus3`: bench 0.573, ext 0.590, masked 0.557, public 0.625):

| block | bench | Δbench | Δext | Δmasked | Δpublic | next step taken |
|---|---|---|---|---|---|---|
| `c_margin_dwell` | 0.604 | +0.031 | +0.004 | +0.057 | +0.025 | full fit |
| `c_carrier_tightness` | 0.591 | +0.018 | +0.004 | +0.033 | +0.007 | full fit |
| `c_ballfree_cluster` | 0.591 | +0.018 | +0.006 | +0.029 | −0.013 | full fit started, not finished |
| `c_melee_uncertainty` | 0.590 | +0.017 | +0.006 | +0.027 | −0.013 | full fit queued, not run |
| `c_centroid_compression` | 0.589 | +0.016 | +0.002 | +0.030 | −0.013 | dropped |

Full recipe (baseline `final_v5` fold models: bench 0.6612, ext 0.6163, masked 0.7061, public 0.6968):

| block | bench | ext | masked | public | val_h2 | <2 s recall | spurious/100 | verdict |
|---|---|---|---|---|---|---|---|---|
| `c_margin_dwell` | 0.6584 (−0.003) | 0.6149 | 0.7018 | 0.693 | 0.585 | 0.178 | 3.78 | not adopted |
| `c_carrier_tightness` | 0.6586 (−0.003) | 0.6162 | 0.7009 | 0.710 | 0.576 | 0.186 | 3.48 | not adopted |
| both | 0.6495 (−0.012) | 0.6175 | 0.6815 | 0.684 | 0.578 | 0.197 | 3.48 | not adopted |

**Lesson.** This is the fourth time a screen gain of +0.014 to +0.040 vanished under the full
recipe (R8, R17, and these two). The mirrored and extra-noise copies already supply what these
blocks add. The fast screen is useful only to drop clear losers; it does not predict a full-fit
gain. Any new feature should be judged on the full recipe, and its cost (~50 min) budgeted.

**Unfinished.** `feat_full_diverse.py` was started at 15:43 on 2026-10-26: it built the mirrored and
augmented feature caches for `c_ballfree_cluster` (`cache/{mir,aug}_*_cmp__full_plus3__c_ballfree_cluster_v4.npz`,
~4.5 GB) and stopped before fitting; no result line was written. Rerunning it reuses those caches.
Given the lesson above, expect ≤ +0.005.
