# Release note: integrity extension (MTAH 2027)

This cycle adds read-only analysis scripts and reports under
`analysis/autonomous_research/`. It does not change the shipped model or decoder.
final_v5 remains the submission.

## Added
- External label-disagreement audit on cached out-of-fold probabilities.
- Independent reruns of decoder, sequence, and reweighting experiments.
- Holdout parser for a third Metrica sample match (local only, not used in training).
- Regression tests for class ordering and semi-Markov reachability.
- Consolidated HTML report and technical PDF.

## Result
No tested idea beat final_v5. Full contest-duel reweighting scored bench 0.658 versus
baseline 0.661. The shipped public score stands at spellF1 0.7226, macroF1 0.9098.

## Disclosed failures
- Semi-Markov v1 collapsed to one segment due to a boundary reachability bug, later fixed.
- A first full reweighting run used string class labels and produced a near-zero bench;
  the artifacts were discarded and retrained with integer labels.
- The 0.005 adoption margin is a development heuristic, not significance.
