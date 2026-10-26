# Consolidated Research Findings: Audio / Generic Pretrained Sources for H/A/D/X

Owner: independent source discovery sub-agent. Date: 2026-10-26.
Project: more-than-a-hack-2027.

This is the concise final synthesis. Every conclusion is tagged **[measured]**
or **[hypothesis]**. URLs are next to the source they support.

---

## 1. Local media stream coverage

**[measured]** Tactical mp4 has AAC-LC 48 kHz stereo audio track.
- Source: hand-rolled box parser + ffmpeg probe.
- File: `data/tactical/BAR-ATH_tactical.mp4`.
- Audio mdhd 7972.6933 s; video mdhd 7972.7600 s; video stsz 199,319 frames @ 25 fps.
- elst media_time=0 on both tracks. No edit-list offset.
- Container start_time=0.

**[measured]** Broadcast mp4 has its own independent timeline.
- File: `data/broadcast/BAR-ATH_broadcast.mp4`.
- No mapping to tactical provided.

**[measured]** Tactical audio decodes to 16 kHz mono via ffmpeg in ~13.5 s
for the full match. Real pitch-side ambient confirmed: halftime near-silence
(0.37 loud peaks/min) vs 10-15/min in play.

**[measured]** Semantic AV delay is unknown. Container PTS offset = 0, but
acoustic perception latency, model detection latency, and sub-frame drift
over 2.2 hours are unmeasured. No per-frame alignment bound is claimed.

---

## 2. Tonal whistle detector (hand-rolled)

**[measured]** Script `bin/extract_whistle_tonal.py`. Runtime 30.5 s CPU.
STFT 1024 @ 16 kHz, 10 ms hop. Spectral concentration in 2-5 kHz + frequency
stability. Cache `cache/whistle_tonal.npz` (199,315 frames x 5 float32 cols).

**[measured]** Event counts vs threshold:
| thr | events |
|---:|---:|
| 0.20 | 790 |
| 0.30 | 206 |
| 0.40 | 45 |
| 0.50 | 3 |

**[measured]** At thr=0.30, detector events land near known kickoff frames:
- 1H kickoff f=11185 -> event at f=11152 (-1.32 s).
- 2H kickoff f=113742 -> event at f=113708 (-1.36 s).
- 2H end f=190197 -> event at f=190216 (+0.76 s).

**[measured]** No precision/recall reported. Only 3 of 206 events are
accounted for by known kickoffs. The rest are unlabelled tonal events.

**[hypothesis]** A threshold around 0.3-0.4 balances false positives vs
sensitivity. Not validated against hand labels.

---

## 3. PANNs Cnn14 generic pretrained detector (pilot)

**[measured]** Weights load locally on CPU. No media upload.
- Paper: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks
  for Audio Pattern Recognition", IEEE/ACM TASLP 2020.
  https://arxiv.org/abs/1912.10211
- Code: https://github.com/qiuqiangkong/panns_inference (MIT license).
- Weights: https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
  (327 MB, AudioSet-derived, research use).
- AudioSet label list:
  https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

**[measured]** Runtime: load 20.9 s, inference 44.4 ms per 1 s chunk on
Apple Silicon CPU. Full-match extrapolation ~6 min. NOT measured end-to-end.

**[measured]** In a 60 s window around 1H kickoff, peak probabilities:
- Crowd (idx 69) p=0.187 at t=-29.5 s (pregame).
- Applause (idx 67) p=0.135 at t=+22.5 s.
- Cheering (idx 66) p=0.080 at t=-9.5 s.
- Whistle (idx 402) p=0.002 at t=-1.5 s.
- Buzzer (idx 398) p=0.003 at t=-1.5 s.

**[measured]** Whistle and Buzzer probabilities are near the model's output
floor. The tonal detector and PANNs both peak near t=-1.5 s in this one
hand-picked window, but this is exploratory peak co-occurrence, NOT event
accuracy.

**[hypothesis]** Lower Whistle p is due to 1 s chunk dilution. Not tested.

**[hypothesis]** Cnn14_DecisionLevelMax (frame-level, ~10 ms hop) would
sharper Whistle scores. Not run. Weights:
https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth

**[hypothesis]** Crowd/Cheering/Applause scores would be useful possession
features. Not evaluated against bench.

---

## 4. YAMNet (alternative, not run)

**[hypothesis]** YAMNet (Google, Apache 2.0, 521 AudioSet classes incl.
Whistle, Crowd, Cheering) would be a second generic detector.
- Tutorial: https://www.tensorflow.org/hub/tutorials/yamnet
- TFLite: https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite
- Not run because PANNs already worked in PyTorch (project venv has torch).
  YAMNet would require TF or TFLite runtime install.

---

## 5. External audiovisual corpora (training data, not direct features)

**[measured]** IDSSE Bundesliga (already in baseline training set) has no
audio in the released bundle.
- Paper: https://www.nature.com/articles/s41597-025-04505-y
- This means external training matches see all-null audio columns,
  which cannot learn splits. `audio_present=0` is not a solution.

**[measured]** SoccerNet v2 provides event-level timestamps (goal, corner,
card, sub) on 550 broadcast games, but NO per-frame possession labels.
Gated/NDA for some content.
- Data: https://www.soccer-net.org/data
- Echoes (commentary ASR, free): https://github.com/SoccerNet/sn-echoes

**[measured]** RoboCup whistle-2017 has labelled whistle clips from a
robot-soccer mic (different acoustic distribution).
- https://sibylle.informatik.uni-bremen.de/public/datasets/whistle-2017/

**[measured]** StatsBomb open data has event-level football data but no
audio and no per-frame possession.
- https://github.com/statsbomb/open-data

**[hypothesis]** SoccerNet broadcast audio could be used as auxiliary
pretraining for a whistle/crowd classifier. Not downloaded. Cost: hours
of download + inference. Eligibility: gated account required.

---

## 6. Future-context eligibility

**[measured]** Primary docs (challenge/*.md) never explicitly authorize
future-context features and never forbid them.
- The only "window" mentions are the starter baseline's causal smoothing
  window (EVALUATION.md, STARTER_README.md).
- OPERATING_INTERFACE section 8's "future-context acceptable" is baseline
  team design choice, not competition authority.

**[measured]** The shipped baseline `mtah/feats.full_plus3` uses symmetric
+/-1/3/5/10 s shape windows. This is inherited, not newly authorized by
this research. No new future-derived candidates added.

---

## 7. Feasible next steps (not yet done)

1. **[measured]** Full-match tonal detector already runs in 30.5 s.
   Output cache is ready for feature integration.
2. **[hypothesis]** Full-match PANNs Cnn14 extraction (~6 min CPU) would
   produce per-1 s chunk scores for 11 event classes. Feasible but not
   run. Does not require refitting the baseline; just adds columns.
3. **[hypothesis]** Cnn14_DecisionLevelMax frame-level model would give
   ~10 ms hop Whistle/Buzzer scores. Needs second 300 MB weight download.
4. **[not allowed as training substitute]** None of these are possession
   labels. They are raw model outputs. Downstream classification eval
   against the bench (0.6612 dev, +0.005 gate) is required before adoption.

---

## 8. Executable setup instructions

```bash
cd analysis/audio_sources

# 1. Install isolated deps (project venv untouched)
VIRTUAL_ENV=../.venv uv pip install --target=pylibs imageio-ffmpeg soundfile onnxruntime librosa torchlibrosa panns_inference

# 2. Download AudioSet labels (one-time)
mkdir -p ~/panns_data
curl -sSL -o ~/panns_data/class_labels_indices.csv \
  "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

# 3. Download PANNs Cnn14 weights (one-time, 327 MB)
curl -sSL -o ~/panns_data/Cnn14_mAP=0.431.pth \
  "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

# 4. Run tonal whistle extraction on tactical mp4
PYTHONPATH=./pylibs ../.venv/bin/python bin/extract_whistle_tonal.py

# 5. Run PANNs pilot around 1H kickoff
PYTHONPATH=./pylibs ../.venv/bin/python bin/panns_pilot.py
```

---

## 9. Artifact inventory (this workspace)

All under `/Users/bytedance/Desktop/BARCA/more-than-a-hack-2027/analysis/audio_sources/`:

Public-code-only bundle: `dist/research_sources_hack2027_public.zip` (now in the local, gitignored `_archive/audio_sources/dist/`)
(scripts + docs only; excludes caches, weights, audio, media, target outputs).

| Path | Type |
|---|---|
| `README.md` | workspace overview |
| `bin/extract_audio_features.py` | RMS band extractor |
| `bin/extract_whistle_tonal.py` | tonal whistle detector |
| `bin/check_alignment.py` | naive peak vs kickoff |
| `bin/whistle_diag.py` | threshold sweep |
| `bin/mp4_timeline_audit.py` | elst/stsz/mdhd audit |
| `bin/panns_pilot.py` | PANNs Cnn14 pilot |
| `docs/SOURCE_INVENTORY.md` | source inventory |
| `docs/RESEARCH_LOG.md` | command log |
| `docs/PRIORITIZED_PROPOSALS.md` | P0-P3 proposals |
| `docs/ADDENDUM_2026-10-26.md` | post-review corrections |
| `docs/FINAL_RESEARCH.md` | this file |
| `logs/*.log` | run logs |
| `cache/*.npz`, `*.json` | local feature caches (NOT in public bundle) |
| `pylibs/` | isolated pip deps (NOT in public bundle) |
