# Audio and auxiliary sources (`analysis/audio_sources/`)

Source-discovery workspace for MTAH 2027: can the tactical video's **audio track** (referee
whistle, crowd) or other generic pretrained sources help H/A/D/X? Moved here from the repo root
(`research_sources_hack2027/`) on 2026-10-26; nothing in it touches `mtah/`, `run.py` or
`evaluation/`.

**Status: nothing adopted.** Read [docs/FINAL_SOURCE_SYNTHESIS.md](docs/FINAL_SOURCE_SYNTHESIS.md)
first, then [docs/ADDENDUM_2026-10-26.md](docs/ADDENDUM_2026-10-26.md) (it supersedes earlier
claims where they conflict). Summary of measured results is in
[docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md#audio-and-auxiliary-sources).

## Layout

| path | what | committed? |
|---|---|---|
| `bin/` | label-free extraction and experiment scripts | yes |
| `docs/` | inventory, log, proposals, synthesis, addendum | yes |
| `logs/` | console logs of the runs (aggregates only) | yes |
| `cache/` | derived per-frame arrays from the match audio | **no** (match-derived; gitignored) |
| `pylibs/` | isolated pip packages (imageio-ffmpeg, soundfile, onnxruntime, panns_inference, librosa), ~400 MB | **no** (gitignored; rebuild below) |

## Docs

| file | job |
|---|---|
| [docs/SOURCE_INVENTORY.md](docs/SOURCE_INVENTORY.md) | every source considered: available, permitted, feasible |
| [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) | chronological log |
| [docs/PRIORITIZED_PROPOSALS.md](docs/PRIORITIZED_PROPOSALS.md) | P0/P1/P2 ideas; only items marked [MEASURED] were run |
| [docs/AUDIO_D_FUSION_EXPERIMENT.md](docs/AUDIO_D_FUSION_EXPERIMENT.md) | v1 hard-D relabel experiment |
| [docs/FINAL_RESEARCH.md](docs/FINAL_RESEARCH.md) | first final report |
| [docs/FINAL_SOURCE_SYNTHESIS.md](docs/FINAL_SOURCE_SYNTHESIS.md) | measured vs hypothesis, verdicts |
| [docs/ADDENDUM_2026-10-26.md](docs/ADDENDUM_2026-10-26.md) | tonal whistle detector, AV container audit, PANNs pilot, corpora |

## Setup and run (from the repo root)

```bash
# isolated packages, so the project .venv stays untouched (unverified after the move; rebuilds pylibs/)
UV_NATIVE_TLS=1 uv pip install --target=analysis/audio_sources/pylibs imageio-ffmpeg soundfile panns_inference librosa torchlibrosa
cd analysis/audio_sources
PYTHONPATH=./pylibs ../../.venv/bin/python bin/extract_audio_features.py \
  --video ../../data/tactical/BAR-ATH_tactical.mp4 --out cache/audio_tactical_features.npz
PYTHONPATH=./pylibs ../../.venv/bin/python bin/extract_whistle_tonal.py   # ~30 s CPU
```

`bin/panns_pilot.py` expects the PANNs Cnn14 weights at `~/panns_data/Cnn14_mAP=0.431.pth`
(outside the repo, 327 MB, from Zenodo; see the addendum §7). The experiment scripts
(`hardD_fusion_v2.py`, `soft_nudge_experiment.py`, `audio_d_fusion_experiment.py`) now compute the
repo root from their own location; they write candidate CSVs to `evaluation/results/research_*.csv`
(the earlier ones are in the local `_archive/evaluation_results/rejected_candidates/`).

Data rule: the audio is match data. Never send audio, cached arrays or per-frame outputs to a
hosted LLM; print aggregates only.
