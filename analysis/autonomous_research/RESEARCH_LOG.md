# Research log (autonomous research)

Chronological human-readable log. Machine-readable state lives in `REGISTRY.json` and `STATE.json`.
Short English, no em dashes. No raw match data here.

## 2026-10-25

### Family 5 integrity review
- Verified the official scorer from code: spellF1 averages F1 over IoU 0.3 to 0.7, greedy one-to-one
  same-team matching, X forward-filled within each half, runs never cross halftime.
- Public ground truth confirmed: 17 possessions, 7500 graded frames, H 3454 A 2577 D 1469.
- Main integrity finding: the masked bench reuses the same 17 possessions across 16 ball-mask
  seeds, so its sd is ball-placement noise, not possession variance. ext (about 3000 possessions)
  is the steadiest gate.
- Reviewed the 0.005 adoption gate as a development heuristic, not significance.

### Source verification
- Renner arXiv:2610.11571 confirmed: 203 rule-labelled matches, GNN-LSTM beats strongest baseline
  by 4.6 macro-F1 points. v1 preprint.
- Socceraction player_possession_time corrected: the code is a same-player and previous-action
  was dribble counter with no geometry. It is not a geometric controlled-possession reference.

### Provenance correction
- Traced run.py to mtah/final.py. No hand label reaches the delivered file because the only reader
  of work/labels is mtah/bench.py load_val, called inside Bench, which run.py never builds.
  use_rules=False only stops rule voting in the decoder; algorithmic rule features still enter.

### External disagreement audit
- Ran OOF disagreement audit on the 9 external matches, no relabel, no target labels.
- IDSSE off-diagonal 8.66 percent, mostly H/A team calls. Metrica off-diagonal 14.75 percent,
  mostly true alive called dead. Pattern is consistent with convention differences, not proof.
- Acknowledged: 9 matches are two datasets, not 9 independent clusters. Descriptive only.

### Reporting spine
- Created REGISTRY.json, STATE.json, this log. Recorded code and data hashes.
- No FINAL_REPORT.html yet. Waiting on sibling experiment tracks.

## 2026-10-25 evening (family 5 integrity, chronological)

- Independently read official scorer evaluation/possession_metrics.py. Confirmed spellF1, X forward-fill, no cross-half leak. Public GT 17 possessions, window 11185-18684.
- Feature provenance trace. Corrected earlier overstatement: no hand label reaches final_v5 structurally. use_rules=False stops rule votes; algorithmic rule features still enter.
- Built label_audit.py on cached OOF probs. IDSSE off-diagonal 8.66 pct, Metrica 14.75 pct. Consistent with convention difference, not proof.
- Decoder track reproduced v5: bench 0.661, ext 0.616, masked 0.706.
- Duel E3 rerun independently: flat to -0.001 bench, rejected.
- Sequence "BOCPD" rerun: memoryless geometric smoother, negative (h=1/400 bench 0.633).
- Semi-Markov v1 defect found: terminal boundary unreachable (26/354 gaps over 750), one all-H run. Returned to producer. Tests added.
- Corrected semi-Markov SUM rerun: bench 0.657, ext 0.614, masked 0.699. -0.004, not promoted. MEAN arm still collapses, separate issue.
- final_v5 reproducibility: sha256 4089da23..c8a byte-exact twice. Counts H52668 A33158 D73031 X40462.
- E1 duel reweight screen: w=0.5 +0.020 on degraded fast screen only. Full recipe pending.
- Corrected tuning disclosure: masked reuses the 17 public possessions; weight selection is public-derived. Not "never tuned public".
- Built provisional FINAL_REPORT.html (shot self-check pass) and TECHNICAL_SUMMARY.pdf (2 pages, verifier pass).
- No new gain claimed. Awaiting E1 full recipe.

## 2026-10-26 follow-up cycle

- E1 full reweight: bench 0.6581, ext 0.613, masked 0.703. Negative. Fresh Game 3 holdout: baseline 0.6128/0.8764, E1 0.6128/0.8770, tied. Consumed.
- DoG logit sharpening: dev bench 0.667. Production wrapper had logit-normalize bug (0.000 spell), retracted. Repaired softmax gives public 0.6857/0.9049 vs baseline 0.7226/0.9098. Not promoted.
- Typed transition head 226 features, lambda=2: bench 0.6573, ext 0.6220, masked 0.6926. Rejected.
- Phys228 head: bench 0.6577, ext 0.6220, masked 0.6934. Rejected.
- Fusion broadcast ball: two seeds identity exact. Recovery diagnostic 0 label change.
- Physical latch: proximity-only 0.6422 negative. v1 state bug invalid. True v2 vector velocity, 8/10 carrier continuity: bench 0.3030, ext 0.1679, masked 0.4380, public fold 0.5120/0.8805. Strong negative.
- Goal-conditioned prior blocked: no explicit scored-goal event, team, or clock in tags.
- Final pause after tested physical variant. 0.90 aspiration unmet. No promotion.
