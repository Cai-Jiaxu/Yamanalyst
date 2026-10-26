# Audio D-Relabel Fusion: Controlled Experiment Result

Date: 2026-10-26. Owner: independent source discovery sub-agent.
Branch: supporting-change (post-hoc relabel), NOT feature-only training gain.

## Predeclared parameters (locked before scoring)

| param | pair A (strict) | pair B (permissive) |
|---|---:|---:|
| whistle threshold | 0.30 | 0.25 |
| min event persistence | 160 ms (16 hops) | 80 ms (8 hops) |
| trailing D window | +2.0 s (50 frames) | +2.0 s (50 frames) |
| relabel rule | H->D, A->D in [f0, f0+50]; X,D unchanged | same |
| half reset | clamp at halftime 90652 | same |
| causality | trailing only | trailing only |

No public17 tuning. Thresholds chosen from prior spectral-concentration
percentiles (p95=0.27, p99=0.33) before any score was seen.

## Extraction inputs

- Baseline predictions: `evaluation/results/final_v5.csv` (sha256 verified unchanged).
- Whistle scores: `cache/whistle_tonal.npz` (199,315 frames, STFT concentration).
- Public GT: `evaluation/gt/BAR-ATH_possession_gt_public.csv` (frames 11185..18684).
- val_h2 GT: `work/labels/val_h2_*.csv` (4 windows, 57 spells).

## Combined evidence semantics

Acoustic event onset f0 from tonal detector. Existing decoder label at f0.
If decoder says H or A (live possession), flip to D for trailing 2 s.
If decoder already says D or X, no change. This is NOT whistle=D hard
equivalence; it only fires where the tracker says live play and acoustic
says whistle.

## Results

### Pair A (strict: thr=0.30, 160 ms)

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| public spellF1 | 0.722581 | 0.722581 | +0.000000 |
| public macroF1 | 0.909848 | 0.909848 | +0.000000 |
| val_h2 spellF1 | 0.581513 | 0.581513 | +0.000000 |
| val_h2 macroF1 | 0.859334 | 0.859334 | +0.000000 |

Events: 3 (all at kickoffs frames 11152, 113708, 190216).
Relabeled: 51 frames (all H->D).
Score change: zero. The 3 events are outside the public GT window
(11185..18684) and outside val_h2 windows.

### Pair B (permissive: thr=0.25, 80 ms)

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| public spellF1 | 0.722581 | 0.700000 | **-0.022581** |
| public macroF1 | 0.909848 | 0.898556 | **-0.011293** |
| val_h2 spellF1 | 0.581513 | 0.553600 | **-0.027913** |
| val_h2 macroF1 | 0.859334 | 0.849485 | **-0.009849** |

Events: 98. Relabeled: 1501 frames (530 H->D, 971 A->D).
Score change: NEGATIVE on both public and val_h2, both spellF1 and macroF1.

## Interpretation

**Pair A is a null result.** At strict threshold, only 3 events detected
(all kickoffs), all outside scored windows. No measurable effect.

**Pair B is a negative result.** At permissive threshold, the tonal
detector produces many false positives (crowd chants, ball impacts,
echoes). Flipping live H/A to D at these points destroys correct
possession labels. spellF1 drops by 0.023-0.028, macroF1 by 0.010-0.011.

**Hypothesis falsified.** The claim "whistle events mark dead-ball
instants, so trailing live possession should be flipped to D" does not
hold. The detector's events at thr=0.25/80ms are dominated by non-whistle
tonal sounds. At thr=0.30/160ms, too few events fire to matter.

## Limitations

1. External training matches have no audio. No model refit is possible.
   This is a post-hoc relabel experiment, not a learned feature.
2. The tonal detector is uncalibrated. True precision at thr=0.25/80ms
   is unknown (no hand labels). The negative result suggests precision
   is low.
3. Only one fusion rule tested (trailing 2 s D window). Other rules
   (leading window, bidirectional, probabilistic soft fusion) were not
   tested.
4. val_h2 has 57 spells; spellF1 noise band is ~0.04. The -0.028 move
   is within noise but consistent across both public and val_h2.
5. No Cnn14_DecisionLevelMax frame-level model tried. PANNs Cnn14 clipwise
   Whistle p=0.002 is near output floor; not usable as a hard gate.
6. No audio on external matches means `ext` yardstick is N/A for this
   branch.

## Verdict

NOT ADOPTED. Audio D-relabel fusion does not improve and hurts when the
detector is permissive. The baseline decoder already handles dead balls
adequately. Tonal whistle scores remain available as diagnostic signals
but not as a hard label gate.

## Artifacts

- Script: `analysis/audio_sources/bin/audio_d_fusion_experiment.py`
- Log A: `analysis/audio_sources/logs/audio_d_fusion_A.log`
- Log B: `analysis/audio_sources/logs/audio_d_fusion_B.log`
- Summary: `analysis/audio_sources/cache/audio_d_fusion_summary.json`
- Candidate CSV: `evaluation/results/research_audio_d_relabel.csv`
  (research artifact, NOT a baseline submission; baseline sha256 unchanged)
