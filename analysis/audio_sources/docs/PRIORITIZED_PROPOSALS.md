# Prioritized experimental proposals (auxiliary sources)

Ordered by expected value / cost. Each proposal is label-free extraction
first; attaching to the Bench is a separate later step. Nothing here has
been run except where flagged [MEASURED].

Adoption gate (from OPERATING_INTERFACE §7): bench must rise >= 0.005 vs
0.6612, ext must not fall > 0.005, masked must not fall > 0.005, public
sanity drop < 0.02.

---

## P0 (cheap, high-information, do next)

### P0.1 Tonal whistle detector on tactical audio
- **Why.** IFAB says a whistle fires at every D<->H/A boundary (kickoff,
  half, free kick, penalty, after goal). Naive band-energy peaks fire
  10-15/min; a tonal detector should get to ~1/min (real whistle rate).
- **Method.** Spectrogram at 5 ms hop, look for a sustained 3-4 kHz line
  for >= 0.1 s (Backer 2014 / UPC 2014). Output: per-frame
  `whistle_score` and `whistle_event` boolean.
- **Cost.** ~1 day of local CPU. Already have decoded 16 kHz mono cached
  as a script; re-decode on demand.
- **Deliverable.** `cache/audio_tactical_whistle.npz` with per-frame
  whistle score. Compare event times to known kickoff frames (11185,
  113742, 190197) and to the 638 smart-tag set-piece starts.
- **Expected effect [HYPOTHESIS].** A clean whistle event prior gives the
  decoder a strong D-edge signal. The shipped model's biggest errors are
  at switches (drift 19/100, missed 22/100); a D-edge prior directly
  attacks drift.
- **Risk.** Tactical-camera mic may be far from the ref; whistles may be
  attenuated. Halftime near-silence suggests the mic works, but we have
  not yet heard a whistle.

### P0.2 Crowd-energy curve as a dead-ball / intensity prior
- **Why.** We already have `crowd_db` per frame [MEASURED]. A sustained
  drop in crowd energy is a dead-ball cue independent of the ball.
- **Method.** Smooth `crowd_db` with a 1 s median, compute its derivative,
  output `calm_1s`, `calm_3s`. No labels.
- **Cost.** Hours. Cache exists.
- **Expected [HYPOTHESIS].** The shipped model already has a
  `pace < 1.3 m/s` stoppage bias; crowd calm is an orthogonal channel.
  Most likely a small feature, not a winner.

### P0.3 Decode broadcast audio and compare crowd bands
- **Why.** Broadcast audio carries commentary. Silero VAD (ONNX, local)
  can separate speech from crowd. If commentary tracks the live play, it
  is a timing reference; if it goes quiet, it may indicate a replay or
  stoppage.
- **Cost.** Same ffmpeg pipe. ~1 h including install.
- **Risk.** Broadcast audio is on an unsynced timeline. Without sync it
  is not a per-frame feature. Cross-feed sync (ICASSP 2017 crowd-chant
  correlation) is possible but is a separate project.

---

## P1 (medium cost, plausible wins)

### P1.1 YAMNet / PANNs local inference for 527 audio classes
- **Why.** Pretrained on AudioSet, includes "Whistle", "Cheering",
  "Crowd", "Shout", "Speech", "Bicycle bell", "Applause". Run locally on
  CPU.
- **Models.**
  - YAMNet TFLite (~3.6 MB): https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite
  - PANNs-CNN14: https://github.com/qiuqiangkong/panns_inference
    (PyTorch weights, 32 kHz, fits our existing torch 2.5.1).
- **Cost.** 1-2 days to integrate; ~1 h CPU to run on the full match.
- **Expected [HYPOTHESIS].** Gives a calibrated per-class probability
  curve at ~100 ms hop. Whistle + Crowd + Speech classes directly map
  to D-edge and live-play priors.
- **Risk.** AudioSet training distribution is YouTube clips; stadium
  pitch-side mic is out-of-distribution. Expect calibration drift; may
  need a small label-free threshold sweep against the known kickoff
  frames.

### P1.2 Bidirectional smoothing of existing probabilities
- **Why.** OPERATING_INTERFACE §8 explicitly allows future-context
  features. The shipped Viterbi is already bidirectional in time. A
  small additional bidirectional smooth (e.g. forward-backward on
  probabilities with a 0.5 s window) is decoder-only, no refit.
- **Cost.** Hours. Reuses cached `P_R1` probabilities.
- **Expected [HYPOTHESIS].** Likely within noise floor (<0.005). The
  existing decoder already has switch costs.

---

## P2 (speculative, defer)

### P2.1 Broadcast-to-tactical alignment via audio
- Cross-correlate crowd-band energy between the two audio tracks
  (ICASSP 2017). Reuses P0.2 curves. If alignment is within a frame,
  broadcast cuts/replays become a visual cue. Prior ATD alignment work
  found median 0.56 m error but only 6.5% ball fill; audio alignment may
  be cleaner.
- **Cost.** 2-3 days.
- **Why defer.** The prior broadcast integration was dropped for low
  marginal value. Audio alone does not change that unless the visual
  content of broadcast gives something the tactical camera does not
  (jersey colour, close-up referee gestures).

### P2.2 Referee gesture / visual referee detection
- REVIEWS #4 flagged "jersey colour at the ball" and referee gestures
  as blocked. Now that video is local, a pretrained pose/gesture
  detector (e.g. MoveNet, HRNet) could detect raised-arm / card signals
  as D-edge events.
- **Risk.** Tactical camera is wide-angle; referee is small in frame.
  Prior video-ball work shows visual detection on this camera is hard.
- **Cost.** Days.

### P2.3 X / abstention policy sweep
- The shipped CSV emits 0 X on graded frames (X only on ungraded).
  Sweep: on frames where max-tree-probability < tau, emit X instead of
  the argmax. Measure spellF1/macroF1 on public + val_h2.
- **Why cheap.** Decoder-only, cached P. No refit.
- **Why defer.** CHALLENGE.md warns: X across a hand-over bleeds the
  old label forward, so naive abstention can hurt spellF1. Needs a
  careful transition-aware design.

---

## P3 (likely dead ends, recorded so they are not re-litigated)

- Cloud mediakit-cli audio/video tools: **[BLOCKED]** by T&Cs (cloud
  ingestion of match media). Do not use.
- StatsBomb open data as labels: different competition, event-level not
  frame-level, different possession convention. Auxiliary prior at best.
- Opta / commercial event feed for this match: unavailable.
- PFF sign-up: prior team already ruled out.
- Sending match audio to Whisper / hosted ASR: **[BLOCKED]** by T&Cs.
  Local Whisper.cpp is possible but commentary is not a possession
  signal.

---

## Feature-worker interface proposal

If a candidate passes screen, the feature function should match the
signature in OPERATING_INTERFACE §6:

```python
def audio_feat(match, f, shapeX):
    # returns (n, k) float32 ndarray aligned to match.n frames.
    # columns (proposed):
    #   0: rms_db_40ms           (from cache/audio_tactical_features.npz)
    #   1: whistle_db_40ms
    #   2: crowd_db_40ms
    #   3: speech_db_40ms
    #   4: whistle_event_score    (from P0.1, once built)
    #   5: crowd_calm_1s          (from P0.2)
    #   6: crowd_calm_3s
    #   7+: YAMNet/PANNs class probs (from P1.1, once built)
    #
    # IMPORTANT: when running on external IDSSE / Metrica matches for
    # training, these columns are NaN (we have no audio for those matches).
    # The bench must see NaN the same way it sees missing ball; HGB handles
    # NaN natively.
```

Critical caveat for the parent: **external training matches (IDSSE,
Metrica) have no audio.** Audio features will be NaN on every external
training frame. The HGB can consume NaN, but the model will learn to
"not trust audio when absent". On our match audio is always present.
This is a train/test mismatch that must be simulated: when adding audio
features, either (a) train on our own audio only and pseudo-label, or
(b) add a binary `audio_present` column and leave audio columns NaN on
external matches so the trees learn to ignore them. Option (b) is the
honest version and likely to under-deliver; expect the audio channel to
add little to the ext yardstick and possibly help only on masked/public.
