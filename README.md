# Yamanalyst: MTAH 2027 Phase 1, ball possession

Our pipeline for [More than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027):
for each of the 199,319 tactical-video frames of FC Barcelona vs Athletic Club, output `H` / `A` /
`D`. Submissions are ranked on **spellF1** (whole possessions matched by temporal IoU). Deadline
**17 Nov 2026**.

**New to the project? Start with [HANDOVER.md](HANDOVER.md)**: state, findings, the four
workstreams and the first-day checklist.

| | what | public spellF1 / macroF1 | where |
|---|---|---|---|
| official baseline | starter's nearest player + ±50-frame vote (`baseline_smooth50`) | 0.326 / 0.621 | `starter/baseline.py`, `evaluation/results/baseline_*.csv` |
| previous submission, still the development baseline | `final_v5`, sha256 `4089da237ca8780a…` | 0.723 / 0.910 (bench 0.661) | [docs/INVESTIGATION.md](docs/INVESTIGATION.md) §24; locked numbers in [baseline_snapshot/](baseline_snapshot/) |
| **current submission** | **`final_v6`** = `final_v5` + one-sided stoppage gate, sha256 `349a7df54c8e0364…` | **0.747 / 0.923** (bench 0.666) | `run.py`, `mtah/final.py`; [docs/INVESTIGATION.md](docs/INVESTIGATION.md) §27–§28 |

`final_v6` changes 5 spells (299 frames) of `final_v5`; nothing else tried since `final_v5` has
beaten it (§26, §27).

## Documents

| | where | job |
|---|---|---|
| organiser | [challenge/](challenge/) | task, data, metric, submission, rules, FAQ; unchanged from Metrica's starter |
| ours | [docs/ROADMAP.md](docs/ROADMAP.md) | **source of truth for open work**, decisions, dates |
| ours | [docs/INVESTIGATION.md](docs/INVESTIGATION.md) | **source of truth for evidence**: every measured number and its script, plus the literature review |
| ours | [docs/REVIEWS.md](docs/REVIEWS.md) | verdicts on outside advice |
| ours | [HANDOVER.md](HANDOVER.md) | onboarding, the four workstreams and their owners, interfaces, milestones |
| ours | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | one-line index of every experiment (adopted and not), pointing to its record |
| ours | [baseline_snapshot/](baseline_snapshot/) | locked `final_v5` numbers (the development baseline) and the developer interface for experiments |
| ours | [docs/CLEANUP_REPORT.md](docs/CLEANUP_REPORT.md) | what the 2026-10-26 cleanup moved or archived, and which checks were run |
| ours | [Yamanalyst.md](Yamanalyst.md) | the technical document we submit; `bash scripts/build_pdf.sh <team_name>` exports `<team_name>.pdf` |

## Data rules (read first)

- **Never give match data, the videos, or anything derived from them to a hosted LLM**:
  ChatGPT, Claude, Gemini (including Gemini in Colab, Drive or Antigravity), Copilot chat and so
  on. This is in the T&Cs. Code you write with an LLM is fine; data and outputs are not. Scripts
  in this repo print aggregates only.
- **Never commit** `data/`, `cache/`, `external/`, `models/`, `colab_upload/`, `work/`,
  `_archive/`, `analysis/audio_sources/{cache,pylibs}/`, any `*.npz`/`*.joblib`, or a label CSV
  (`evaluation/results/*.csv`, `analysis/**/*.csv`). They are all gitignored; keep it that way. This repo becomes
  public at submission.
- Labels must come from an algorithm. Hand labels are allowed only for training or validation.

## Setup

```bash
# Python 3.11 via uv (https://docs.astral.sh/uv/)
uv sync                      # behind a corporate TLS proxy: UV_SYSTEM_CERTS=1 uv sync
```

1. **Match data** (from the enrolment download link): unpack into `data/tactical/` and
   `data/broadcast/`, as described in [challenge/DATA.md](challenge/DATA.md). Or point `MTAH_DATA_DIR` at it.
2. **Public external data**, used for training. It is about 2.6 GB: IDSSE (CC BY 4.0) and the
   Metrica sample games.
   ```bash
   bash scripts/download_external.sh
   uv run python -m mtah.external
   ```

## Make the submission

```bash
uv run python run.py                      # -> evaluation/results/final.csv, validated
uv run python run.py --name final_v6      # the same, under the submission's name
uv run python evaluation/run_evaluation.py --no-detail     # public-window scoreboard
```

`run.py` is deterministic: it reproduces `final_v6` byte for byte (sha256
`349a7df54c8e0364e91e060b6279a2e7236bf5bfefc813e0bb63795b2e640ea3`). `final_v5` was `4089da23…`
(set `DECODER = DECODER_V5` in `mtah/final.py` to regenerate it), `final_v4` was `0bc6e8c0…`, `final_v3` `3b01921e…`, `final_v2` `59e7b27f…`.
`final_v2` differs from `final_v1` (`34f6c958…`) only on the 2nd half's 2,934 pre-kickoff frames
(docs/INVESTIGATION.md §10).
The first run builds caches (about 30–60 min); later runs take about 11 min. Older methods are
still available with `--method rules|fused_own|fused_ext`.

## The files for the form

```bash
uv run python run.py --name final_v6                      # sha256 349a7df54c8e0364…
mkdir -p submission && cp evaluation/results/final_v6.csv submission/<team_name>.csv
bash scripts/build_pdf.sh <team_name> && mv <team_name>.pdf submission/
uv run python evaluation/validate_submission.py submission/<team_name>.csv
```

`submission/` is gitignored: the CSV is a full-match label file and must not be committed.

## Checks that need no data

```bash
bash scripts/check.sh        # compiles every tracked .py, then all synthetic tests
```

`tests/test_decode.py` pins the shipped decoder settings and the stoppage gate, and
`tests/test_submission_io.py` the CSV bytes, so a change that would move the submission's sha
fails a test before it reaches `run.py`.

## How the pipeline works

```
raw ATD ──► mtah/data.py      cache: players, ball, homography, camera centre
        ──► mtah/features.py  ball cleaning, distances, team speeds, out-of-pitch
        ──► mtah/rules.py     touch episodes + dead-ball state machine (its label is a FEATURE)
        ──► mtah/feats.py     full_plus3: ball-free team shape at ±1/3/5/10 s, ball/touch block,
                              rule internals, k-nearest, pitch control, set-piece geometry
        ──► mtah/final.py     gradient-boosted trees trained on 9 EXTERNAL matches with true
                              labels (mtah/external.py), noised to our tracker (mtah/noise.py),
                              plus mirrored and extra-noise copies
        ──► mtah/decode.py    3-state Viterbi: switch costs and a 1-s minimum dead ball
                              (+ dead ball before the 2nd-half kickoff, mtah/kickoff.py,
                              + stoppage gate for short live blips next to a dead ball)
        ──► evaluation/results/final.csv
```

## How to try an idea

Everything is judged on the shared benchmark, `mtah/bench.py`:

```python
from mtah.bench import Bench
from mtah import feats
b = Bench()                                   # ~5 min the first time (cached after)
b.evaluate("my idea", feats.full_plus3, model_params={"max_iter": 600},
           dec_params={"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 12.0, "min_dead": 25},
           use_rules=False)
# -> bench | ext (idsse, metrica) | masked | public, plus an error breakdown
#    (split / merge / drift / missed / spurious per 100 true possessions)
```

A change goes into `mtah/final.py` only if it raises `bench`, which means external and masked
together. Decoder-only changes can reuse fitted probabilities (`b.probs(...)` then `b.score(...)`);
see `analysis/decoder_sweep.py` and `analysis/round2.py`. Differences under about 0.005 are noise.

## Video ball detection (GPU, Colab)

The tracker loses the ball on 41% of graded frames. The fine-tuned detector works (79% of top
picks within 2 m on held-out frames), but merging its track into the current model **lowers** the
bench (docs/INVESTIGATION.md §9), and so does a better DP linker (§9.1), so the submission does not use
it. A separate "video ball" channel (docs/ROADMAP.md R6) was closed: even a perfect ball restored
in the gaps adds only +0.005 to the bench (R2, §12).
`mtah/video_ft.py` fine-tunes a small YOLO on our own video, using the ATD ball projected through
the homography as automatic labels, then detects on the missing frames.

1. `bash scripts/build_colab_bundle.sh` → `colab_upload/MTAH/{mtah_code.zip, ball_detector.ipynb}`
2. Upload both to `MyDrive/MTAH/`, next to `MyDrive/MTAH/data/tactical/` (the 5 tactical files).
3. Open `ball_detector.ipynb` in Colab on a GPU runtime, **with Gemini off**, and run all cells.
   It writes `ball_det_ft.npz` to Drive.
4. Copy it to `cache/ball_det_ft.npz` here, then `uv run python -m mtah.video_ft score` compares
   the bench with and without the video ball track.

`mtah/video.py` also runs locally on Apple silicon (MPS); see its docstring. Model weights:
`bash scripts/download_models.sh`.

## Repository map

| path | what |
|---|---|
| `run.py` | the one command that writes the submission |
| `mtah/` | the pipeline (table above) plus `bench.py`, `masking.py`, `score.py`, `report.py`, `kickoff.py`, and measured-but-unused `video*.py`, `mbr.py`, `refiner.py` |
| `analysis/` | every experiment behind docs/INVESTIGATION.md (`yardsticks.py` scores any decoder on every yardstick). Re-runnable; prints aggregates only |
| `analysis/autonomous_research/` | post-`final_v5` integrity review and decoder/training experiments (registry + reports) |
| `analysis/feat_track/` | post-`final_v5` causal tracking-feature blocks, screen and full-fit runners |
| `analysis/audio_sources/` | audio track research: extraction, whistle detector, PANNs pilot (own isolated `pylibs/`) |
| `baseline_snapshot/` | locked baseline metrics, reconstruction script, developer handbook |
| `work/labels/` | person-made validation labels (`val_h2_*.csv`, `ball_probe_300.csv`); gitignored |
| `_archive/` | local, gitignored archive of duplicates and superseded material (docs/CLEANUP_REPORT.md) |
| `colab/` | GPU notebook |
| `scripts/` | downloads (external data, model weights), Colab bundle, `check.sh` (all data-free checks) |
| `tests/` | synthetic tests of the shipped path: CSV bytes, decoder post-processing, shipped config |
| `evaluation/` | the official scorer, validator and public ground truth (unchanged from the starter) |
| `challenge/` | the organiser's documents, unchanged |
| `docs/` | our roadmap, evidence and review verdicts |

## Working on it as a team

- `main` only: `git pull` before you start, and keep commits small, with the bench numbers in
  the message when a result changes.
- New experiments go in `analysis/`; promote to `mtah/` only what the submission uses.
- Before submitting: `run.py` twice on a clean clone (identical sha), the validator,
  [challenge/SUBMISSION.md](challenge/SUBMISSION.md), and making the repo public.
