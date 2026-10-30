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

## Rebuild the missing probability cache

From the repository root, after `uv sync`, installing the tactical assets (or setting
`MTAH_DATA_DIR`), and parsing all nine external matches with `uv run python -m mtah.external`:

```bash
uv run python baseline_snapshot/rebuild_probabilities.py --check
uv run python baseline_snapshot/rebuild_probabilities.py
uv run python baseline_snapshot/reconstruct_baseline.py
```

The rebuild fits three match-held-out models using `full_plus3`, `final.MODEL`, mirrors,
and all three extra noise draws. External predictions are out of fold; public and masked
predictions average the three models. It never reuses a fold-model joblib cache. Allow roughly
an hour for the refit, longer with cold feature caches; run one refit at a time per machine.
The submission CSV and locked `baseline_metrics.json` are unchanged by the rebuild command.
The reconstruction command does rewrite `baseline_metrics.json`, so review its diff afterwards.

Existing output is refused by default. Use `--output cache/P_R1_rebuilt.npz` to compare with an
original cache, or explicitly `--overwrite` to replace it. A validated NPZ is published atomically;
an aggregate-only `.rebuild.json` report sits alongside it in the gitignored cache directory.
The NPZ contains only the case arrays expected by existing consumers (columns H, A, D).

The script compares bench, ext, masked, and public spellF1 plus public macroF1 against the
locked snapshot, with absolute tolerance 0.0001. A mismatch returns exit code 1 and retains
the rebuilt cache and differences for investigation; it does not establish baseline equivalence.
Human validation labels are optional and do not enter training. Only aggregate results are printed.

Synthetic contract checks (no match data needed):

```bash
uv run python -m unittest baseline_snapshot.test_rebuild_probabilities
```
