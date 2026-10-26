# Baseline snapshot: `final_v5`, locked 2026-10-26

The reference every experiment is compared against. Read-only evidence: nothing here changes
`mtah/`, `run.py` or a prediction file.

| file | job |
|---|---|
| [OPERATING_INTERFACE.md](OPERATING_INTERFACE.md) | **developer handbook**: task, evaluator, `Bench` yardsticks, shipped config, how to write a featfn, the adoption gate, leak rules |
| [ELIGIBILITY_AND_REBUILD.md](ELIGIBILITY_AND_REBUILD.md) | cached-P reconstruction vs fresh rebuild, GT provenance per yardstick, future-context and audio eligibility, version and cache hashes |
| `reconstruct_baseline.py` | rescores cached fold probabilities `cache/P_R1.npz` with the shipped decoder (~40 s warm) |
| `baseline_metrics.json` | its output: bench 0.6612, ext 0.6163, masked 0.7061, public (fold) 0.6968 / 0.9098, val_h2 0.5627 |
| `fresh_rebuild.log` | `run.py` from scratch on 2026-10-26: sha256 `4089da23…`, public 0.7226 / 0.9098 |
| `report.html` | rendered summary of the above (moved from `artifacts/baseline_report/`) |

Two different numbers, never to be mixed: **fold-ensemble dev numbers** (bench, ext, masked,
public 0.697) come from `reconstruct_baseline.py`; the **full-fit delivered file** scores public
0.7226 / 0.9098 with the official evaluator.

```bash
.venv/bin/python baseline_snapshot/reconstruct_baseline.py     # needs cache/P_R1.npz
```
