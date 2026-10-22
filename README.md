# Yamanalyst: MTAH 2027 Phase 1, ball possession

Our pipeline for [More than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027):
for each of the 199,319 tactical-video frames of FC Barcelona vs Athletic Club, output `H` / `A` /
`D`. Submissions are ranked on **spellF1** (whole possessions matched by temporal IoU). Deadline
**17 Nov 2026**.

The official challenge docs from Metrica's starter repository are kept as they were:
[STARTER_README.md](STARTER_README.md), [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md),
[EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md), [RULES.md](RULES.md),
[FAQ.md](FAQ.md).

## Where we are

| submission | public spellF1 | public macroF1 | bench* |
|---|---|---|---|
| `baseline_smooth50` (official floor; 0.2433 on the full match) | 0.326 | 0.621 | — |
| `rules_v2` (hand-written rules) | 0.461 | 0.852 | — |
| **`final_v1` (current submission)** | **0.700** | **0.898** | **0.656** |

\* `bench` is the yardstick we choose on, not the public window: the mean spellF1 over 9 external
matches noised to look like ours, and over our public window with the ball hidden the way the
full match loses it. The public 5 minutes are the best-tracked part of the match, with only 17
possessions. **Never pick a version because it scores better there.**

How we got here, with every number reproducible: [INVESTIGATION.md](INVESTIGATION.md).
External reviews and our verdicts on each point: [RECOMMENDATIONS_REVIEW.md](RECOMMENDATIONS_REVIEW.md).

## Data rules (read first)

- **Never give match data, the videos, or anything derived from them to a hosted LLM**:
  ChatGPT, Claude, Gemini (including Gemini in Colab, Drive or Antigravity), Copilot chat and so
  on. This is in the T&Cs. Code you write with an LLM is fine; data and outputs are not. Scripts
  in this repo print aggregates only.
- **Never commit** `data/`, `cache/`, `external/`, `models/`, `colab_upload/`, `work/` or
  `evaluation/results/*.csv`. They are all gitignored; keep it that way. This repo becomes
  public at submission.
- Labels must come from an algorithm. Hand labels are allowed only for training or validation.

## Setup

```bash
# Python 3.11 via uv (https://docs.astral.sh/uv/)
uv sync                      # behind a corporate TLS proxy: UV_SYSTEM_CERTS=1 uv sync
```

1. **Match data** (from the enrolment download link): unpack into `data/tactical/` and
   `data/broadcast/`, as described in [DATA.md](DATA.md). Or point `MTAH_DATA_DIR` at it.
2. **Public external data**, used for training. It is about 2.6 GB: IDSSE (CC BY 4.0) and the
   Metrica sample games.
   ```bash
   bash scripts/download_external.sh
   uv run python -m mtah.external
   ```

## Make the submission

```bash
uv run python run.py                      # -> evaluation/results/final.csv, validated
uv run python run.py --name final_v1      # the same, under another name
uv run python evaluation/run_evaluation.py --no-detail     # public-window scoreboard
```

`run.py` is deterministic: it reproduces `final_v1` byte for byte (sha256 `34f6c958cdee1f05…`).
The first run builds caches (about 30–60 min); later runs take about 11 min. Older methods are
still available with `--method rules|fused_own|fused_ext`.

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
bench (INVESTIGATION.md §9), so `final_v1` does not use it. The next step, if anyone picks it up,
is a separate "video ball" feature channel trained with simulated video-like noise.
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
| `mtah/` | the pipeline (table above) plus `bench.py`, `masking.py`, `score.py`, `report.py`, `kickoff.py` (measured, not used), `video*.py` |
| `analysis/` | every experiment behind INVESTIGATION.md. Re-runnable; prints aggregates only |
| `colab/` | GPU notebook |
| `scripts/` | downloads (external data, model weights), Colab bundle |
| `evaluation/` | the official scorer, validator and public ground truth (unchanged from the starter) |
| `PLAN.md` | the first plan (historical) |

## Working on it as a team

- `main` only: `git pull` before you start, and keep commits small, with the bench numbers in
  the message when a result changes.
- New experiments go in `analysis/`; promote to `mtah/` only what the submission uses.
- Before submitting: `run.py` twice on a clean clone (identical sha), the validator,
  [SUBMISSION.md](SUBMISSION.md), and making the repo public.
