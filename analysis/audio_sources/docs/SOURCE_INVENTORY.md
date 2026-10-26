# Source inventory: H / A / D / X auxiliary signals

Date: 2026-10-26. Author: independent source-discovery subagent.
Scope: new input streams not already consumed by `mtah/final.py`
(which uses tactical ATD, tactical homography, smart-tagging features,
9 external labelled matches, ball-free team shape).

Legend:
- **[MEASURED]** I ran code locally and observed the number.
- **[EXTERNAL]** Claim comes from a cited paper/repo; not re-tested here.
- **[HYPOTHESIS]** My proposal; untested.
- **[BLOCKED]** Ruled out by T&Cs or by prior measurement.

---

## 1. Local media inventory (this match, on disk)

| stream | path | codec / shape | duration | alignment to submission row |
|---|---|---|---|---|
| Tactical video | `data/tactical/BAR-ATH_tactical.mp4` | h264 1920x1080, 25 fps | 2 h 12 min 52.76 s | **row i == frame i** (199,319 frames) |
| Tactical audio | same mp4, stream #0:0 | AAC-LC, 48 kHz stereo, 236 kb/s | 7972.69 s | **[MEASURED]** decoded to 16 kHz mono, 199,317 hops at 25 fps. Off-by-2 vs frame count 199,319; no offset to discover. |
| Tactical ATD | `data/tactical/BAR-ATH_tactical_atd.{xml,txt}` | Metrica EPTS, 199,319 rows | 1:1 with video | used by baseline |
| Tactical homography | `data/tactical/BAR-ATH_tactical_homography.csv` | 100% valid | 1:1 | used by baseline |
| Tactical smart tagging | `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | 638 intervals | seconds-from-video-start | used as a feature family |
| Broadcast video | `data/broadcast/BAR-ATH_broadcast.mp4` | h264 1920x1080, 25 fps | 1 h 49 min 39.48 s | **own timeline, NOT aligned**; cuts/replays |
| Broadcast audio | same mp4, stream #0:1 | AAC-LC, 48 kHz stereo, 317 kb/s | same as video | same timeline |
| Broadcast ATD | `data/broadcast/BAR-ATH_broadcast_atd.{xml,txt}` | 164,488 rows | own timeline | prior alignment work fills only 6.5% of tactical ball gaps (REVIEWS #1) |

### Audio feasibility (what I actually measured)

Decoded tactical audio with a locally-bundled ffmpeg (imageio-ffmpeg 0.6.0,
macOS aarch64 static binary). No cloud service involved.

Aggregates only (`cache/audio_tactical_features_summary.json`):

| quantity | value |
|---|---|
| decoded samples | 127,563,093 @ 16 kHz mono |
| duration | 7972.69 s |
| per-hop frames (25 fps) | 199,317 |
| broadband RMS dBFS: mean / p50 / p95 / p99 | -59.85 / -59.65 / -52.83 / -51.71 |
| 2-5 kHz "whistle band" dBFS: mean / p95 / p99 | -52.93 / -41.99 / -38.59 |
| 0.15-2 kHz "crowd band" dBFS: mean / p95 | -39.54 / -32.26 |
| naive whistle peaks (whistle p99 AND broadband p90) | 1421 hops |

Segment density of those naive peaks:

| segment | frames | peaks | peaks/min |
|---|---:|---:|---:|
| pre-match | 11,185 | 34 | 4.6 |
| 1st half play | 79,467 | 823 | 15.5 |
| halftime | 20,156 | 5 | **0.37** |
| 2nd half play | 79,390 | 557 | 10.5 |
| post-match | 9,121 | 2 | 0.33 |

Interpretation [MEASURED + HYPOTHESIS]:
- Halftime near-silence (0.37 peaks/min) confirms the audio track is
  pitch-side ambient, not a silent track. It really does carry crowd sound.
- Naive loud-band peaks fire 10-15/min during play. A real match has on the
  order of 50-100 referee whistles (one every 1-2 min). So the simple band-
  energy peak has ~10x false positives; it is NOT a whistle detector.
- Nearest-peak offsets to known kickoff frames were -8.9 s, -21.6 s, +13.6 s
  (not within 1-2 s). Honest reading: this crude threshold does not yet
  localise whistle events; a tonal/Hough-based detector is needed.

### What I did NOT do (be honest)
- Did not decode broadcast audio.
- Did not run a pretrained audio classifier (YAMNet / PANNs).
- Did not attach any H/A/D label to the features; the cached npz is
  label-free by construction.
- Did not modify `mtah/`, `run.py`, or any existing prediction CSV.

---

## 2. External sources (literature, repos, datasets)

### 2a. Whistle / dead-ball boundary from audio

| source | what it gives | url | license | local-runnable? |
|---|---|---|---|---|
| IFAB Laws on whistle use | Ref whistle fires at: kickoff, half start/end, free kick, penalty, after goal, end of half. Direct prior on D<->H/A transitions. | https://www.theifab.com/laws/latest/body-language-communication-and-whistle/ | public | n/a (rulebook) |
| Classic whistle DSP: Hough transform on 3.5-4.5 kHz spectrogram lines > 0.1 s | 98% on RoboCup dataset (Backer 2014); adopted by later work. | https://staff.fnwi.uva.nl/a.visser/education/bachelorAI/daniel_vermaas_thesis_bsc.pdf (survey of Backer et al.) | cited | yes, pure numpy/scipy |
| UPC 2014: whistle detector on 5 full soccer matches, hand-annotated | audio descriptors + SVM; explicit false-positive handling for vuvuzelas / supporter whistles. | https://gtav.upc.edu/en/publications/papers/2014/the-importance-of-audio-descriptors-in-automatic-soccer-highlights-generation/@@download/file/Audio_Descriptors_Soccer_Highlights_SSD.pdf | paper only | method, no released weights |
| RoboCup 2017 whistle dataset | labelled whistle/audio clips | https://sibylle.informatik.uni-bremen.de/public/datasets/whistle-2017/ | research use | yes, but robot-soccer mic, not stadium |
| BHuman whistle detector (C++) | reference DSP | https://github.com/robocupmipt/BHumanCodeRelease | open source | port to python ~1 day |

### 2b. Crowd / event energy from audio

| source | what it gives | url |
|---|---|---|
| Biadgie & Join, CIVR 2003: HMM on MFCC for 4 crowd-response levels | baseline crowd-event classifier | https://www.dcs.gla.ac.uk/~jj/publications/conference/2003/biallie-civr03.pdf |
| PANNs-CNN14, Kong et al. 2019 (arXiv 1912.10211) | pretrained on AudioSet, 527 classes incl. Whistle, Cheering, Crowd, Shout, Speech. PyTorch weights downloadable. | https://github.com/qiuqiangkong/panns_inference ; paper https://arxiv.org/pdf/1912.10211.pdf |
| YAMNet (Google) | MobileNet-v1 on AudioSet, 521 classes, TFLite, ~3.6 MB, runs on CPU at 12.8 ms hop | https://www.tensorflow.org/hub/tutorials/yamnet ; TFLite https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite |
| Cross-feed audio sync via crowd chant correlation (ICASSP 2017) | could align broadcast audio to tactical audio if needed | https://zenodo.org/records/268407/files/icassp17rev.pdf |
| SoccerNet-Echoes | broadcast commentary ASR, Whisper-based; commentary only, not pitch-side | https://xplorestaging.ieee.org/document/10935951/keywords |

### 2c. External match event feeds (for training priors, NOT labels for this match)

| source | what | url | notes |
|---|---|---|---|
| IDSSE / Sportec Open DFL (Bassek et al., Sci Data 2025) | 7 Bundesliga matches, per-frame BallPossession + BallStatus, event feed. CC BY 4.0. | https://www.nature.com/articles/s41597-025-04505-y | **already the training set** of the shipped model. |
| Metrica sample games 1-2 | tracking + broadcast video, open. | https://github.com/metrica-sports/sample-data | already used. |
| StatsBomb open data | event-level (passes, shots, fouls, cards) for World Cups / Copa America / Womens Super League. | https://github.com/statsbomb/open-data | event timestamps, NOT per-frame possession; different competition and convention. Could give a prior on set-piece cadence only. NOT a label source. |
| PFF | per-event data | not reachable without personal sign-up | prior team already ruled out. |
| Opta event feed for THIS match | commercial, not available | n/a | forbidden / unavailable. |

### 2d. Uncertainty / X-class

- CHALLENGE.md is explicit: `X` is a real allowed label. On spellF1 (the
  ranking metric) an `X` inside a possession is filled by the previous label,
  so it costs nothing mid-possession; across a hand-over it bleeds the old
  team into the new possession. (challenge/SUBMISSION.md, CHALLENGE.md §"The
  four labels".)
- OPERATING_INTERFACE §8: future-context (bidirectional) features are
  allowed, because the model runs offline.
- Selective-prediction / abstention literature: standard temperature scaling
  + threshold on max softmax, or risk-coverage curves. The shipped HGB
  already emits 3-class probabilities; converting `max_p < tau` to X is a
  one-line decoder change. **Not yet measured** on our bench.

---

## 3. Eligibility excerpts (verbatim, for the record)

- RULES.md: "Any method at all, subject to two conditions: organisers can
  access and inspect it; technical document explains it. Pre-trained models,
  commercial or open. LLMs, hosted or local. External data."
- FAQ: "participants shall not upload, disclose or otherwise make available
  to any third-party hosted or cloud-based LLM any videos, datasets,
  confidential information ... or any extracts thereof." -> **all audio
  extraction must stay local**. The bundled ffmpeg + local PANNs/YAMNet
  weights satisfy this; mediakit-cli (cloud) does not.
- DATA.md: "Everything is fair game, in any combination. A system built only
  on the ball track is as legitimate as one that reads the video."
- OPERATING_INTERFACE §8: "Because the model is applied offline (not live),
  future-context features are acceptable as long as they are computed
  identically for training external matches and our match."
- RULES.md: "Improving that data, or replacing it with your own, is entirely
  legitimate."
- Forbidden: private full-match GT (any derived form), hand-labelling
  submitted frames, cloud-LLM ingestion of match media.

---

## 4. What is already closed (do not repeat)

From docs/INVESTIGATION.md and REVIEWS.md:
- Video ball detection (YOLO fine-tuned on tactical camera): built,
  measured, dropped. Even a perfect ball restored in gaps adds +0.005 bench.
- DP tracklet linker over video ball detections: measured, drops bench.
- Broadcast ATD fusion: only fills 6.5% of tactical ball gaps; dropped.
- Smart Tagging as a team prior: team tag agrees 77% on live frames; below
  model 90%; dropped as a vote, kept as a feature.
- PathCRF (ball-free possession path): reviewed; our ball-free shape model
  is at the same level (88-93% team accuracy).
- CatBoost averaging, learned GRU/TCN refiner, touch-sequence kick features,
  MBR decoding, in-Viterbi dwell: all tried, none adopted.
- **Audio / whistle / crowd: NOT previously tried.** REVIEWS #4 listed
  "whistle audio" as blocked only because the videos had not yet arrived.
  They are local now.
