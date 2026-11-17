# Baseline snapshot: `final_v5`, locked 2026-10-26

The reference every experiment is compared against. Read-only evidence: nothing here changes
`mtah/`, `run.py` or a prediction file.

> **2026-11-17:** the submission is now `final_v6` (`final_v5` + one more pass of the stoppage gate,
> docs/INVESTIGATION.md §27–§28). This folder still describes `final_v5`, the development baseline;
> its scripts use `mtah.final.DECODER_V5`, so every number below is unchanged.

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

### Verified on Windows, 2026-11-02

The full rebuild passed using Python 3.11.9 and the locked dependencies. Tactical assets were
extracted locally and all nine public external matches were downloaded and parsed. The rebuilt
cache contains fresh fold predictions; no original-machine probability or model cache was used.

| checked metric | rebuilt value | difference from locked reference |
|---|---|---|
| bench spellF1 | 0.6611595515410369 | 0 |
| ext spellF1 | 0.6162642272941614 | 0 |
| masked spellF1 | 0.7060548757879124 | 0 |
| public spellF1 (fold ensemble) | 0.696774193548387 | 0 |
| public macroF1 (fold ensemble) | 0.9098282105426794 | 0 |

The existing `reconstruct_baseline.py` consumer also passed with the same metrics. Its output
was saved locally as `cache/P_R1.reconstructed_metrics.json`, and the committed
`baseline_metrics.json` was restored unchanged. The rebuild report is
`cache/P_R1.rebuild.json`; these local cache artifacts remain gitignored.

All three synthetic rebuild tests and the handover's feature, semi-Markov, and class-order
test scripts passed in the project environment. `val_h2` was not rechecked because the private
human-label files were not present. No submission CSV or production model settings changed.

### Full submission verification on Windows, 2026-11-02

Two independent `run.py` full-model refits in the existing checkout produced the exact locked
submission SHA-256:

```text
4089da237ca8780a6041bad8cc78261f29e9c5f5ef9fe8d3bdd9d9a8e2951c8a
```

| output under `evaluation/results/` | validator | official public spellF1 | official public macroF1 |
|---|---|---|---|
| `windows_repro_20261102_b.csv` | pass, 199319 rows | 0.722581 | 0.909848 |
| `windows_repro_20261102_c.csv` | pass, 199319 rows | 0.722581 | 0.909848 |

The original Windows writer emitted CRLF. A diagnostic run reproduced the expected hash after
normalizing CRLF to LF, confirming the predictions matched. `run.py` now explicitly writes
UTF-8 with LF line endings; both subsequent fresh runs match the expected bytes directly.
No model or decoder parameters changed. The new byte-level serialization test failed before
the fix and passes afterwards:

```bash
uv run python -m unittest discover -s tests -p test_submission_io.py
```

The rebuild contract tests and all three handover regression scripts also passed. Aggregate
results and official evaluation outputs are kept locally under `cache/submission_repro_20261102/`;
the CRLF diagnostic CSV is archived there. Generated submissions and reports remain gitignored.
These runs reused prepared data and feature caches in the existing checkout. The final
clean-clone replay at the freeze remains required.
