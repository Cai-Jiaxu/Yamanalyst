# Experiment index

One line per experiment, so nobody repeats one. This file is an **index only**: the numbers and
scripts live in the record it points to, and a new result goes into that record first
([INVESTIGATION.md](INVESTIGATION.md) for anything judged on the bench, plus the track's own
registry if it has one). Status of open work: [ROADMAP.md](ROADMAP.md). Last updated 2026-11-17.

Yardsticks: **bench** = mean of ext and masked spellF1 (`mtah/bench.py`); adopt only if bench
rises ≥ 0.005 and neither ext nor masked falls > 0.005 (ROADMAP rule 1). Locked baseline:
`final_v5` fold models, bench 0.6612 / ext 0.6163 / masked 0.7061
([baseline_snapshot/](../baseline_snapshot/)). The submission is `final_v6` (§28).

## Where the records are

| record | covers | format |
|---|---|---|
| [INVESTIGATION.md](INVESTIGATION.md) §0–§25 | everything up to `final_v5` (2026-10-21 → 25) | prose + tables, script per section |
| [INVESTIGATION.md](INVESTIGATION.md) §26 | summary of the three post-`final_v5` tracks | tables |
| [INVESTIGATION.md](INVESTIGATION.md) §27 | R20 stoppage gate v2 and the 2026-11-17 checks | tables, `analysis/r20_gate.py` |
| [analysis/autonomous_research/](../analysis/autonomous_research/) | integrity review; decoder, sequence, reweighting, latch, holdout | `REGISTRY.json`, `RESEARCH_LOG.md`, per-experiment `REPORT.md` / `LOG.md` |
| [analysis/feat_track/](../analysis/feat_track/) | 5 causal tracking feature blocks | `README.md`, `REGISTRY.json`, `*_results.jsonl` |
| [analysis/audio_sources/](../analysis/audio_sources/) | audio track, whistle detector, PANNs, external corpora | `docs/*.md`, `logs/` |
| [REVIEWS.md](REVIEWS.md) | verdicts on outside advice | table |

## Adopted (in `final_v6`; all but the last row were already in `final_v5`)

| what | effect | record |
|---|---|---|
| Train on 9 external matches with true labels, noised to our tracker | masked 0.35 → 0.47 | §6 |
| Rule label as a **feature**, no rule term in the decoder | +0.047 bench | §8 |
| Longer context ±5/10 s, ball/touch block, rule internals, 600 trees | 0.469 → 0.590 | §8 |
| k-nearest, approach speed, pitch control | +0.005 | §8 |
| Left/right and home/away mirrors + extra-noise copies | +0.039, largest single gain | §8 |
| Minimum dead ball 1 s; set-piece geometry | +0.004; +0.003 | §8 |
| Dead ball before each half's first kickoff (R1) | bench unchanged; 2nd-half pre-kickoff spells 19 → 2 | §10 |
| Stoppage-blip gate (R13) | public +0.022, masked +0.008 on `final_v3` | §19 |
| `final_v2` decoder instead of the rule vote (R14) | +0.025 bench | §20 |
| One-sided pass of the stoppage gate (R20 → `final_v6`) | bench +0.005, ext −0.002; delivered file public 0.723 → 0.747; 5 spells change, the 3 on labelled frames all spurious | §27–§28 |

## Not adopted — do not repeat without a new reason

| area | experiment | best result | record |
|---|---|---|---|
| decoder | boundary head on switch costs; EM label-shift prior; lower switch cost; boundary snapping | ±0.001 to −0.017 | §8 |
| decoder | kickoff evidence after every kickoff (incl. after goals) | −0.008 | §8, §10 |
| decoder | minimum dead-ball dwell inside Viterbi (R7) | 0.000 | §11 |
| decoder | rule labels voting in the decoder (`final_v3`/`v4`) | −0.025 bench; only eye labels favour it | §17–§20 |
| decoder | per-frame stacker of model + rules (R11) | public −0.08, ext −0.12 | §19 |
| decoder | segmental MBR for expected spellF1 (R15) | +0.001 | §21 |
| decoder | learned refiner, BiGRU / dilated TCN (R16) | ext +0.011–0.017, masked −0.02–0.03 | §22–§23 |
| decoder | duel switch cost; BOCPD-style smoother; semi-Markov durations | 0.660 / 0.633 / 0.657 | §26.1 |
| decoder | DoG logit sharpening | dev 0.667, full-fit public 0.686 | §26.1 |
| decoder | typed / physical transition heads (226/228 feats) | 0.657–0.658 | §26.1 |
| decoder | hard physical control latch | 0.303 | §26.1 |
| decoder | out-of-pitch emission boost | 0.000 | §26.1 |
| decoder | duration priors | null | ROADMAP "Not on the list" |
| decoder | R20 stoppage gate v2 as specified (walk + no player within 5 m) | −0.0002; matches 0 remaining stoppage spells | §27 |
| decoder | 4 other stoppage-gate variants (longer, faster pace, slower pace, distance only) | −0.0001 to −0.0004 | §27 |
| decoder | `min_poss` 25 / 50 / 75 frames on top of the one-sided gate | −0.0006 / −0.015 / −0.079 | §27 |
| features | velocity-projected pitch control (R8) | screen +0.014, full 0.000 | §13, §15 |
| features | touch-sequence / kick block for short possessions (R17) | screen +0.040, full +0.001 | §25 |
| features | carrier tightness; margin dwell; both | full 0.659 / 0.658 / 0.650 | §26.2 |
| features | centroid compression; melee uncertainty; ball-free cluster | screen only (+0.016–0.018) | §26.2 |
| model | CatBoost alone or averaged (R5) | −0.015 to −0.024 at screen | §13 |
| model | capacity > 300 trees; ball columns in the shape model | flat | §5 (W6) |
| training | noisy student on our own match (R4) | −0.016 | §14 |
| training | IDSSE relabelled to our convention | −0.002 | §8 |
| training | contest-duel reweighting (AR-E1) | 0.658 | §26.1 |
| training | milder noise where the ball is seen (R12) | closed: premise was eye-label timing | ROADMAP |
| video | fine-tuned YOLO ball merged into the track (W7) | 0.656 → 0.642/0.646 | §9 |
| video | DP linker for video detections | 0.629–0.635 | §9.1 |
| video | value of a perfect ball in the gaps (R2) | +0.005 upper bound → **video closed** | §12 |
| sources | broadcast ATD ball | fills 6.5% of gaps | Yamanalyst §3, §26.1 |
| sources | Smart Tagging as labels / anchors | team tags 77% agree; set pieces 3–12 s early | Yamanalyst §3, §26.1 |
| sources | goal-conditioned kickoff prior | blocked: no goal events | §26.1 |
| audio | hard D at whistle events; soft D nudge | null or −0.023 public | §26.3 |

## Audio and auxiliary sources

Measured: audio present and container-aligned; tonal whistle detector fires within 1.4 s of known
kickoffs; PANNs runs locally. Not measured: whistle precision/recall, semantic audio–video delay,
full-match PANNs. Open: ROADMAP R21. See
[analysis/audio_sources/docs/FINAL_SOURCE_SYNTHESIS.md](../analysis/audio_sources/docs/FINAL_SOURCE_SYNTHESIS.md).

## Holdouts already spent

- **Metrica sample game 3** was used once as a frozen holdout (AR-G3, 2026-10-26). It is consumed:
  do not tune on it, and report any further use as non-independent.
- The **public 5-minute window** has been looked at hundreds of times; it is a sanity check only.

## Naming a new experiment

`<track>-<short-name>-v<k>`, e.g. `feat-press-intensity-v1`. Use the same string as the
`Bench.probs(..., key=...)` cache key (change the key whenever the featfn, model params,
augmentation or decoder change), the JSONL `block`/`name` field, and the INVESTIGATION subsection
title. Add one line to this index when the verdict is in.
