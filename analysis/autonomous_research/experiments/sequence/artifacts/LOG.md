# Sequence experiments log — source families 3/4

Date: 2026-10-25.
Protocol: cached `P_R1.npz` fold probabilities; official spellF1 via
`analysis.yardsticks.Yard`; no public-GT fitting; no external label learning; no frame
rows printed; no `mtah/` or `run.py` changes. Scripts live in
`analysis/autonomous_research/experiments/sequence/`. Cache: `cache/ar_sequence_*`.

## Baseline (reproduced)

`v2g` = final_v2 decoder + stop_gate, on cached P_R1:

| bench | ext | idsse | metrica | masked | public spellF1/macro | public spells | val | valLo |
|---|---|---|---|---|---|---|---|---|
| 0.661 | 0.616 | 0.607 | 0.648 | 0.706 | 0.697 / 0.910 | 14 | 0.579 | 0.742 |

Matches audit (`AUDIT.md` 4.2). Provenance preserved: exact cached fold probabilities.

## E-BOCPD-1: convention-neutral changepoint smoother

Method: forward recursion `q_t = normalize((1-h) q_{t-1} P_t + h * prior * P_t)` with
uniform prior over H/A/D, then q fed into the same v5 Viterbi decoder. Hazard h predeclared.

| hazard (expected run) | bench | ext | masked | public spells |
|---|---|---|---|---|
| baseline | 0.661 | 0.616 | 0.706 | 14 |
| h=1/25 (~1 s) | 0.628 | 0.602 | 0.654 | 17 |
| h=1/50 (~2 s) | 0.632 | 0.601 | 0.663 | 17 |
| h=1/100 (~4 s) | 0.631 | 0.599 | 0.663 | 17 |
| h=1/200 (~8 s) | 0.629 | 0.597 | 0.662 | 17 |
| h=1/400 (~16 s) | 0.633 | 0.595 | 0.671 | 17 |

Argmax-only at h=1/400: bench 0.610, masked 0.644 (worse).

## Verdict

Negative at every predeclared hazard. Bench drops 0.028-0.033; masked drops 0.04-0.05.
Public spell count moves 14 to 17 (closer to true 17) but that does not transfer to ext or
masked. Same signature as the rejected GRU/TCN refiner (`INVESTIGATION.md` section 23):
more spells helps public count, hurts the yardsticks that represent the private match.

Interpretation (hypothesis, not causal claim): the existing Viterbi already encodes separate
H<->A and alive<->dead persistence costs; a generic exponential smoother ignores that and
cannot beat it.

## Next

Label-free semi-Markov duration regularizer with a fixed weak prior is the remaining
convention-neutral lever. Geometry-conditioned duel Viterbi belongs to the audit agent and
is not duplicated here.

## E-SEMIMARKOV: pruned explicit-duration span decoder (fixed generic priors)

Two arms on cached P_R1, same v2g post-processing:

| arm | bench | ext | masked | public spells |
|---|---|---|---|---|
| baseline v2g | 0.661 | 0.616 | 0.706 | 14 |
| semi-Markov SUM | 0.000 | 0.000 | 0.000 | 1 |
| semi-Markov MEAN | 0.000 | 0.000 | 0.000 | 1 |

Collapses to one segment. Toy test confirms the recurrence is mathematically correct
(flat prior + zero transition = global optimum), but with -12 per switch against span sums
over hundreds of frames, the objective favours one long span. No transition-cost sweep was
run (disallowed broad sweep). Negative; not promoted.

Naming correction: the earlier "BOCPD" is a fixed-hazard geometric-duration 3-state HMM,
not the Adams-MacKay run-length posterior. Registry: analysis/autonomous_research/experiments/sequence/registry.json.

## E-SEMIMARKOV v2 (corrected after bug)

Bug: production L_MAX=750 pruned candidate pairs. Natural argmax runs up to 1282 frames
broke the Viterbi chain; end boundary unreachable; backtrack followed uninitialized bp=(0,0),
producing one all-H segment. Toy tests T1 (forced blocks recovered) and T2 (every-frame
boundaries + zero prior = framewise Viterbi, 1.0 agreement) confirmed the recurrence.
Fix: remove L_MAX; duration prior shapes span length.

| arm | bench | ext | masked | public | val |
|---|---|---|---|---|---|
| baseline v2g | 0.661 | 0.616 | 0.706 | 0.697 n14 | 0.579 |
| semi-Markov SUM | 0.656 | 0.614 | 0.699 | 0.697 n14 | 0.590 |
| semi-Markov MEAN | 0.656 | 0.614 | 0.699 | 0.697 n14 | 0.590 |

Result: bench -0.005, val +0.011. Does not meet +0.005 bench gate. Not promoted.
SUM and MEAN identical (spans are already argmax-pure).

## E-SEMIMARKOV v3 (final corrected)

Tests T1-T4 all PASS (blocks, Viterbi equivalence 1.0, >L_MAX block, multi-switch).
Regular endpoints every 750 frames union argmax changes; reachability and backpointer
progress asserted. Invalid 0.000 run preserved as failed implementation.

| arm | bench | ext | masked | public | val |
|---|---|---|---|---|---|
| baseline v2g | 0.661 | 0.616 | 0.706 | 0.697 n14 | 0.579 |
| semi-Markov SUM | 0.657 | 0.614 | 0.699 | 0.697 n14 | 0.590 |
| semi-Markov MEAN (true norm) | 0.000 | 0.000 | 0.000 | 0.000 n1 | 0.000 |

SUM: bench -0.004, val +0.011. Does not meet +0.005 bench gate. Not promoted.
MEAN collapses (length signal removed; -12 switch vs per-frame mean ~0.3). Discarded.
Smoother renamed: recursive categorical-prior / geometric evidence smoother (not BOCPD).
