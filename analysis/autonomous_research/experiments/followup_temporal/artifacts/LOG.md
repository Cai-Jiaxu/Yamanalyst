# Follow-up temporal experiments

## F1: asymmetric Gaussian peak-preserving filter on cached P_R1

Distinct from cycle2 transition-cost levers and geometric-HMM/semi-Markov cycles.
Fixed priors: sigma=0.5s, wide sigma=1.5s, sharpen weight 0.5, merge 1s.
No sweep.

| arm | bench | ext | masked | public | val |
|---|---|---|---|---|---|
| baseline v2g | 0.661 | 0.616 | 0.706 | 0.697 n14 | 0.579 |
| asymmetric gaussian + v2g | 0.667 | 0.629 | 0.706 | 0.682 n17 | 0.572 |

Result: bench +0.006, ext +0.013, masked unchanged. Meets +0.005 gate.
Public sanity -0.015 (n14->n17). Yellow flag; not promoted to production without
independent validation. Game 3 seen, not fresh holdout.

Literature: Solovyev et al. arXiv:2205.10450 dense detection anchors is SoccerNet
action spotting SOTA; +10-15 event F1 is for spot events, not spellF1. Transfer
hypothesised, not verified.

## F1 robustness (predeclated perturbation arms, no broad sweep)

Formula: sharp = logP + w * (gauss(logP, sigma) - gauss(logP, sigma*3)); Q=softmax(sharp); v2g Viterbi.
Script: analysis/autonomous_research/experiments/followup_temporal/peak_filter.py
SHA-256: d76197bc4cab8cca31fa8ae368c1d298a5b51410d0bd80f644ca01fc5f58c106

| arm | bench | ext | masked | public | val |
|---|---|---|---|---|---|
| baseline | 0.661 | 0.616 | 0.706 | 0.697/0.910 n14 | 0.579 |
| sigma=10f (0.4s) w=0.5 | 0.667 | 0.628 | 0.706 | 0.682/0.907 n17 | 0.579 |
| sigma=12f (0.5s) w=0.5 | 0.667 | 0.629 | 0.706 | 0.682/0.907 n17 | 0.572 |
| sigma=15f (0.6s) w=0.5 | 0.666 | 0.628 | 0.704 | 0.694/0.909 n17 | 0.570 |
| sigma=12f w=0.3 | 0.667 | 0.623 | 0.711 | 0.700/0.910 n15 | 0.582 |
| sigma=12f w=0.7 | 0.668 | 0.629 | 0.707 | 0.686/0.907 n18 | 0.565 |
| no-filter control | 0.661 | 0.616 | 0.706 | 0.697/0.910 n14 | 0.579 |

Gain robust: bench +0.005 to +0.007, ext +0.007 to +0.013 across 5 arms.
Public unstable (0.682-0.700, n15-18). w=0.3 arm most balanced (public back to 0.700).
NOT promoted. H/A vs D separate asymmetry ablation and NMS/min-fragment ablation not yet run.

Solovyev attribution: arXiv:2205.10450 dense detection anchors, SoccerNet-v2 action spotting.
The "+10-15 event F1" numeric is UNVERIFIED from paper tables; treated as hypothesised only.
Competition win vs SOTA claim not independently confirmed.

## F1 ablation (H/A vs D, sigma=12 w=0.5 fixed)

| arm | bench | ext | masked | public macro | val |
|---|---|---|---|---|---|
| baseline | 0.661 | 0.616 | 0.706 | 0.697/0.910 n14 | 0.579 |
| DoG H,A only | 0.667 | 0.629 | 0.706 | 0.682/0.903 n17 | 0.584 |
| DoG D only | 0.660 | 0.616 | 0.704 | 0.697/0.916 n14 | 0.566 |
| DoG all | 0.667 | 0.629 | 0.706 | 0.682/0.907 n17 | 0.572 |
| DoG all + same-team bridge | 0.667 | 0.629 | 0.706 | 0.682/0.907 n17 | 0.572 |

Naming corrected: difference-of-Gaussian (DoG) highpass sharpening, same weights all
classes; NOT asymmetric Gaussian. Gain is entirely from H/A (possession) sharpening.
D-only neutral. Same-team bridge has no effect.

Caches: P_R1['public'] is 199319 frames fold ensemble; no separate full-fit production
cache exists. No retrain. Candidate config preserved, NOT promoted.

Item 6 (pseudo anchors): SmartTagging XML schema not located in repo; whistle/audio
not confirmed available. No audio, no manual inputs. Stress test blocked.

## Production full-fit DoG candidate output (199319 frames)

Production model (final.predict, ~11 min) on public match, DoG sigma=12/w=0.5 all classes:
- candidate_dog_public.csv: 199319 rows, 325 spells, H=37590 A=44147 D=77120 X=40462
- candidate_dog_fusion.csv: same + tactical-missing D-suppression fill (13.8% of frames), 388 spells
- Determinism second run not completed (time); single production inference used.
- Fusion fill is exploratory proxy (tactical missing -> suppress D), not live ball fusion.
- Team tags (kickoff/free kick/corner) noisy 3-12s, prior external after goal -0.021 NOT adopted.
- Both CSVs preserved; DoG-alone remains primary candidate until combo verified.

## PRODUCTION FULL-FIT DEFECT (broken, preserved)

ar_candidate_dog.csv (hash 829fbadf...): official public spellF1 0.0000, macro 0.0175.
ar_candidate_dog_fusion.csv (hash 16089e35...): spellF1 0.0235, macro 0.1293.
Both in evaluation/results/ (gitignored). Preserved, NOT promoted.

Diagnostic: cached fold P_R1['public'] col means [0.330 H, 0.198 A, 0.473 D],
argmax D=100399 (50%). Broken production DoG output D=77120 (39% of graded),
H=37590 A=44147. Full-fit P distribution differs from fold P. No-filter production
control (required to isolate bug vs method) could not complete within time budget.
Cached-P DoG (bench 0.667) remains the valid dev candidate; production wrapper
integration bug unresolved.

## Final stable evidence (2026-10-26)

Official baseline (final_v5, audit reproduced): macroF1 0.909848, spellF1 0.722581.

Cached-P DoG (sigma=12, w=0.5, all classes): bench spellF1 0.667 vs baseline 0.661 (+0.006),
ext 0.629 vs 0.616 (+0.013), masked 0.706 unchanged, public spellF1 0.682 macro 0.907 n17.
Meets +0.005 dev gate on cached bench. NOT promoted (no independent validation; public dips).

Production full-fit DoG CSVs: BROKEN (spellF1 0.0). Integration bug, not method negative.
No-filter production control not completed (inference killed before P cache saved).

Anchors: smart tagging CSV available (638 rows: Throw-in 26, Corner 12, Goal Kick 10,
Free Kick 5, Kick Off 6). Whistle/audio NOT in schema; ffprobe unavailable. No audio extraction.
Item 6 agreement stress not executed.

No external data, no new training, no media upload.

## Corrected production DoG (schema fixed, still bad)

Schema bug fixed: run.py writes "frame,label" CSV; earlier broken CSVs had bare labels.
Canonical baseline ar_baseline_check.csv sha 4089da237ca8780a, spellF1 0.7226, macro 0.9098.

Corrected DoG (production full-fit P, rule_labels passed, frame,label CSV):
ar_candidate_dog.csv sha 5f117d5719153163, spellF1 0.0667, macro 0.1609.
Label counts H=42631 A=49230 D=66996 X=40462 vs baseline H=52668 A=33158 D=73031.

DoG on production full-fit P over-sharpens and destroys output. Cached-P DoG gain
(0.667 bench) does NOT transfer to production full-fit P. Candidate NOT promoted.
Production P cached at cache/ar_followup_production_P.npz (199319x3).

## Integration bug FIXED; DoG production is genuine negative

No-filter re-decode from cached production P (R=all X, matching final.predict use_rules=False):
sha 4089da237ca8780a, spellF1 0.7226, macro 0.9098 — EXACT match to run.py baseline.

Bug was: final.predict replaces R with all-X when use_rules=False; wrapper passed rule_labels.

DoG (sigma=12, w=0.5) on production full-fit P: sha 0d875a0bcf59e10f, spellF1 0.0000,
macro 0.0175. Label shift: H 52668->37590, A 33158->44147, D 73031->77120.
DoG over-sharpens production P and destroys possession structure.
Cached-P DoG gain (0.667 bench) does NOT transfer to production full-fit P.
This is a genuine method negative on production, not integration bug.

## DoG production failure diagnostic (independent operator check)

argmax flips: 199224 of 199319 frames (99.9%). Mean max P 0.891 -> mean max Q 0.623.
Example frame 138769: P=[0.003, 0.005, 0.993] confident D -> Q=[0.526, 0.474, 0.001]
D destroyed, split H/A. DoG highpass over-sharpens confident production P.

No-filter re-decode from cached P (R=all X): sha 4089da237ca8780a EXACT match.
Wrapper integration fixed. DoG sigma=12 w=0.5 is a genuine negative on production full-fit P.

## Softmax bug FIXED; DoG production valid result

Bug: Q normalized logits by row sum (inverses classes). Fixed with stable softmax.
Zero-weight identity: sha 4089da237ca8780a EXACT match to baseline.

DoG sigma=12 w=0.5 (correct softmax): sha 18fb0944d90f1409.
argmax flips 3075/199319 (1.5%), mean max P 0.891 -> Q 0.894.
Labels H=52468 A=33043 D=73346 X=40462 (vs baseline 52668/33158/73031/40462).
Public spellF1 0.6857 vs baseline 0.7226 (-0.037), macro 0.9049 vs 0.9098 (-0.005).

DoG is slightly negative on production public. NOT promoted.
Cached-P 0.667 bench gain does not transfer to production public.

## Item 6 tag agreement stress (not GT validation)

Smart tagging CSV: 638 rows, columns code/start_frame/end_frame/team/half/type.
Kickoff tags (6) overlap with algorithm D or team labels; set-piece tags (632) overlap
mixed H/A/D. Agreement is stress diagnostic only, not F1 truth. Whistle/audio not in schema.

## Item 6 kickoff anchor stress (predeclared, 6 tags)

Predeclared: kickoff tags should overlap D (dead ball). Avg D fraction across 6 kickoff
intervals: 0.586. First kickoff (start) falls in X gap. This is agreement stress, not F1 truth.
Set-piece tags (632) no predeclared expectation; counts only.
