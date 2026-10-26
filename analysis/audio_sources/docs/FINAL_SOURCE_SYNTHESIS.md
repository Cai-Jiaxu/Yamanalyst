# Final Source Synthesis: Audio / Generic Pretrained Sources for H/A/D/X

Date: 2026-10-26. Owner: independent source discovery sub-agent.
Status: final, pending auditor verification.

## 1. What was measured vs hypothesis

### Measured
- Tactical mp4 has AAC-LC 48 kHz stereo audio, decodes to 16 kHz mono in 13.5 s.
- Container PTS offset = 0 (elst media_time=0 both tracks). Semantic AV delay unknown.
- Tonal whistle detector (STFT concentration, 10 ms hop): 30.5 s CPU runtime.
  Events at thr=0.30/160ms: 3 (all kickoffs). At thr=0.25/80ms: 98.
- PANNs Cnn14 (AudioSet, 327 MB weights): loads locally on CPU, 44 ms/chunk.
  Whistle p=0.002 near output floor in pilot window. Crowd p=0.19, Applause p=0.14.
- Hard D-relabel v2: B_v2 (thr=0.25/80ms) hurts public spellF1 by 0.023, val_h2 by 0.024.
  A_v2 (thr=0.30/160ms): null (3 kickoff events outside scored windows).
- Soft D-prob nudge v2 (W=0.15): zero effect on decoder output.

### Hypothesis (not proven)
- Cnn14_DecisionLevelMax frame-level model would sharpen Whistle scores.
- SoccerNet audio could pretrain a better whistle/crowd classifier.
- Self-supervised pretraining on our unlabelled audio could help.
- Crowd/Cheering scores would be useful possession features.

## 2. Evaluator results (v2, corrected timing)

### Hard branch (post-hoc relabel on production CSV final_v5.csv)
Baseline: public spellF1 0.722581, macroF1 0.909848. val_h2 spellF1 0.581513, macroF1 0.859334.

| config | public spellF1 | public macroF1 | val_h2 spellF1 | val_h2 macroF1 |
|---|---:|---:|---:|---:|
| baseline | 0.722581 | 0.909848 | 0.581513 | 0.859334 |
| A_v2 thr=0.30/160ms | 0.722581 | 0.909848 | 0.581513 | 0.859334 |
| B_v2 thr=0.25/80ms | 0.700000 | 0.903682 | 0.558065 | 0.850067 |

### Soft branch (probability nudge on fold-ensemble P_R1.npz, unchanged decoder)
Baseline: public spellF1 0.696774, macroF1 0.909828. val_h2 spellF1 0.562712, macroF1 0.863147.
This is fold-ensemble, NOT production fullfit. Gap from production is fullfit vs fold CV.

| config | public spellF1 | val_h2 spellF1 |
|---|---:|---:|
| control W=0 | 0.696774 | 0.562712 |
| W=0.15 | 0.696774 | 0.562712 |

Soft branch gated ONLY on existing D probability (P[:,2] > 0.3). No ball speed,
player speed, ball-missing, or stoppage tracking context was included. This was
a tonal D-prob nudge, not broad contextual acoustic evidence fusion.

## 3. Why null ≠ audio futility

- Strict threshold (0.30/160ms) finds only 3 events, all kickoffs.
- Kickoffs are outside public GT window and outside val_h2.
- The null result at strict settings says nothing about mid-game whistles.
- Permissive threshold (0.25/80ms) has too many false positives, but true
  whistle precision is unknown (no hand labels).
- Available sources are NOT exhausted: PANNs frame-level model, SoccerNet
  audio, self-supervised pretraining all remain untested.

## 4. Genuine blockers

1. **No audio in external training matches.** All-null audio columns cannot
   learn splits. No model refit possible with audio as a feature.
2. **No semantic AV sync.** Container PTS offset = 0, but acoustic perception
   latency and sub-frame drift are unmeasured.
3. **No sound event labels for our match.** Tonal detector precision unknown.
4. **Crowd/impacts false-positive types plausible but not proven.**

## 5. PANNs pilot status

- Downloaded: YES (327 MB Cnn14 weights from Zenodo).
- Loaded: YES (CPU, 20.9 s).
- Inference: YES on 60 s kickoff window (44 ms/chunk).
- Full-match extraction: NOT RUN (estimated ~6 min CPU, not measured).
- Cnn14_DecisionLevelMax: NOT RUN (second 300 MB weights, frame-level).
- YAMNet: NOT RUN (TF install heavier than PANNs).

## 6. Key URLs

- PANNs paper: https://arxiv.org/abs/1912.10211
- PANNs code (MIT): https://github.com/qiuqiangkong/panns_inference
- Cnn14 weights: https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
- YAMNet: https://www.tensorflow.org/hub/tutorials/yamnet
- SoccerNet: https://www.soccer-net.org/data
- IDSSE paper: https://www.nature.com/articles/s41597-025-04505-y
- RoboCup whistle: https://sibylle.informatik.uni-bremen.de/public/datasets/whistle-2017/

## 7. Verdict

- Hard D-relabel: NOT ADOPTED (hurts at permissive, null at strict).
- Soft D-nudge at conservative settings: NOT ADOPTED (zero decoder effect).
- Audio as a feature family remains blocked by no-training-audio + no-labels.
- No promotion. No repeated threshold tuning.
