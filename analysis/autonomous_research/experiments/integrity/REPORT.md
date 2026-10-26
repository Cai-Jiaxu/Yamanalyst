# Independent label-disagreement audit (family 5, empirical integrity)

Date: 2026-10-25. Read-only. External training matches only. No target manual labels, no
relabeling. Uses cached out-of-fold probabilities (`cache/P_R1.npz`), so each match is scored by a
fold model that never trained on it. Script:
`analysis/autonomous_research/experiments/integrity/label_audit.py`.
Run: `.venv/bin/python analysis/autonomous_research/experiments/integrity/label_audit.py`
Output cache: `cache/ar_integrity_labelaudit.npz`.

## 1. Provenance correction (why no hand labels reach the delivered file)

My earlier review wrote that `use_rules=False` implies no manual labels in inference. That is
inadequate. Traced directly:

- `run.py` calls `final.predict(m, f, rule_labels)`.
- `final.py external_training()` fits the HGB on the 9 external matches' true labels only, then
  predicts our match.
- The only reader of `work/labels/*.csv` is `mtah/bench.py:load_val()`, called inside
  `Bench.__init__`. `run.py` never imports or instantiates `Bench`.
- Therefore no hand label reaches `run.py` output. The reason is structural (the delivered path
  never builds the bench), not the decoder flag.
- `use_rules=False` only means the rule string does not vote in the Viterbi decoder. Rule-derived
  features from `mtah/rules.py` (an algorithmic touch and dead-ball state machine on tracking) still
  enter the trees. Those features are algorithmic, not manual.

## 2. What the audit measured

For each graded external frame, compare the fold model's argmax to the external true label.
Stratify by source and by distance to the nearest H/A/D change. Restrict to graded frames.

| source | matches | graded frames | off-diagonal | mean conf of disagree | mean conf of agree |
|---|---|---|---|---|---|
| IDSSE | 7 | 1,002,644 | 8.66% | 0.72 | 0.95 |
| Metrica | 2 | 286,112 | 14.75% | 0.81 | 0.94 |

Possession-aware: nearly every possession contains at least one disagreeing frame
(IDSSE 694/710 = 97.7%; Metrica 186/193 = 96.4%). This is not isolated label typos.

Confusion split of the off-diagonal frames:

| source | H<->A (team) | alive<->dead |
|---|---|---|
| IDSSE | 58.6% | 41.4% |
| Metrica | 26.9% | 73.1% |

Metrica drives most of its error from true-alive frames the model calls dead (true H -> pred D
5.39%, true A -> pred D 3.64%). IDSSE disagreements are more often team calls.

## 3. Interpretation (descriptive, not inferential)

- IDSSE disagreement is mostly team-call noise around possessions. This matches the known
  touch-happy convention (4.3 touches/min vs our 3.4).
- Metrica disagreement is mostly alive-versus-dead. Metrica event labels mark a possession from
  pass to recovery; the degraded tracking model calls many of those frames dead. This is a
  systematic dead-ball convention gap, not random label error.
- Confident-learning pruning would delete boundary and convention-rich frames, which carry the
  signal. It would not find fixable typos. The high model confidence on Metrica dead calls (0.81)
  is the model betting on its own degraded-tracking prior, not evidence that Metrica labels are
  wrong.

## 4. Caveats on sample size

Nine matches are two datasets, not nine independent clusters. IDSSE shares one tracking provider
and one label pipeline. Metrica is two sample games from one event-based construction. No
possession-level significance is claimed. Rates are descriptive. A paired possession-level effect
cannot be estimated from this set alone.

## 5. Recommendation

No relabeling. The audit supports the team's existing decision: do not globally relabel IDSSE
(already measured -0.002). The Metrica alive/dead gap is the more interesting systematic difference,
but it reflects event-vs-tracking convention, and a learned stage would inherit it (INVESTIGATION
section 23). The smallest follow-up, if assigned, is a possession-level stratification of the
Metrica true-alive-pred-dead frames: are they short holds, aerial balls, or contested recovers?
That requires no relabeling and only cached probabilities.
