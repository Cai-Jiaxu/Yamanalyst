# Research log: audio / auxiliary source discovery

Chronological. Every command that produced an artifact is logged with its
absolute path. Aggregates only; no per-frame match data is printed here.

## Setup

- Project root: `/Users/bytedance/Desktop/BARCA/more-than-a-hack-2027`
- Research dir (new, isolated): `/Users/bytedance/Desktop/BARCA/more-than-a-hack-2027/analysis/audio_sources/`
- Layout: `bin/` scripts, `cache/` outputs, `logs/` command logs, `docs/` notes,
  `pylibs/` isolated pip-installed packages (does NOT touch the project `.venv`).
- Did not modify `mtah/`, `run.py`, `evaluation/`, or any existing prediction CSV.

## Step 1 — read prior evidence

Read: `README.md`, `Yamanalyst.md`, `challenge/DATA.md`, `challenge/CHALLENGE.md`,
`challenge/SUBMISSION.md`, `challenge/RULES.md`, `challenge/FAQ.md`,
`docs/ROADMAP.md`, `docs/INVESTIGATION.md` §0, §9, §9.1, §12,
`docs/REVIEWS.md` (grep for audio/whistle/crowd/broadcast/uncertainty),
`baseline_snapshot/OPERATING_INTERFACE.md`,
`analysis/autonomous_research/REGISTRY.json`.

Conclusion: audio/whistle has never been touched. REVIEWS #4 blocked it
only because media was missing.

## Step 2 — probe stream coverage

- No system ffmpeg on PATH. Installed `imageio-ffmpeg==0.6.0` (bundles a
  static ffmpeg binary for macOS aarch64) plus `soundfile`, `onnxruntime`,
  into `analysis/audio_sources/pylibs/` via:
  ```
  VIRTUAL_ENV=/Users/bytedance/Desktop/BARCA/more-than-a-hack-2027/.venv \
    uv pip install --target=./pylibs imageio-ffmpeg soundfile onnxruntime
  ```
- Bundled binary at:
  `analysis/audio_sources/pylibs/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1`
- Stream probe (ffmpeg -i, no output file):
  - tactical mp4: Audio aac LC, 48000 Hz stereo, 236 kb/s; Video h264 1920x1080
    25 fps; Duration 02:12:52.76.
  - broadcast mp4: Video h264 Main 1920x1080 25 fps; Audio aac LC 48000 Hz
    stereo 317 kb/s; Duration 01:49:39.48.

## Step 3 — decode tactical audio and extract per-40ms features

Script: `analysis/audio_sources/bin/extract_audio_features.py`
Run:
```
cd analysis/audio_sources
PYTHONPATH=./pylibs .venv/bin/python bin/extract_audio_features.py \
  --video ../data/tactical/BAR-ATH_tactical.mp4 \
  --out cache/audio_tactical_features.npz
```
Log: `analysis/audio_sources/logs/audio_tactical.log`
Cached:
- `analysis/audio_sources/cache/audio_tactical_features.npz` (2.9 MB)
  arrays: `frame` (int32, 199317), `rms_db`, `whistle_db`, `crowd_db`,
  `speech_db` (float32, length 199317), `sr`, `fps`, `hop`.
- `analysis/audio_sources/cache/audio_tactical_features_summary.json`

Key measured numbers (aggregates only):
- decoded 127,563,093 samples @ 16 kHz mono; duration 7972.69 s.
- per-hop frames: 199,317 (matches the 199,319-frame tactical timeline to
  within 2 frames -> no offset to discover).
- RMS dBFS mean -59.85, p50 -59.65, p95 -52.83, p99 -51.71.
- whistle band (2-5 kHz) p99 -38.59 dB; crowd band (0.15-2 kHz) p95 -32.26 dB.

## Step 4 — alignment sanity check vs known kickoff frames

Script: `analysis/audio_sources/bin/check_alignment.py`
Log: `analysis/audio_sources/logs/alignment_check.log`

Peak density per segment:
- pre-match 4.6 / min; 1st half 15.5 / min; halftime 0.37 / min;
  2nd half 10.5 / min; post 0.33 / min.

Nearest naive peak to known kickoffs:
- 1st half kickoff (f=11185): nearest peak at 10962, -8.9 s.
- 1st half end (f=90651): nearest peak at 90111, -21.6 s.
- 2nd half kickoff (f=113742): nearest peak at 102661, -443 s (i.e. no peak
  within seconds; the closest is in the 1st half).
- 2nd half end (f=190197): nearest peak at 190536, +13.6 s.

Honest reading: the crude band-energy peak is not a whistle detector;
it fires on crowd noise and ball impacts. Halftime near-silence confirms
the audio track is real pitch-side ambient and well-aligned in time.

## What was NOT tested (do not infer that it was)

- Broadcast audio decode.
- Pretrained classifier (YAMNet / PANNs) inference.
- Tonal / Hough whistle detector.
- Alignment of broadcast audio to tactical audio.
- Any attach of audio features to the Bench (no `b.evaluate` run).
- Any change to `mtah/feats.py` or `mtah/final.py`.
