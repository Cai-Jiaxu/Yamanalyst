# TECHNICAL_SUMMARY.pdf structure (frozen outline, content pending experiment results)

Target: `analysis/autonomous_research/reports/TECHNICAL_SUMMARY.pdf`.
Per `challenge/SUBMISSION.md` section 3. Short, a few pages. Does not overwrite `Yamanalyst.md`.
This file is the frozen section skeleton. Numbers and experiment outcomes fill in after sibling
tracks finish and the strongest candidate is independently reproduced. Tone rules: use "consistent
with", never "prove". Cite original source URLs for every soft reference and numerical claim.

## Sections (locked)

1. **Approach**
   - What we built: per-frame H/A/D over 199,319 tactical frames, ranked on spellF1.
   - Core: gradient-boosted trees on 9 noised external matches, fixed-cost Viterbi decoder,
     stoppage gate, kickoff half-start rule.
   - Why this rather than alternatives: trees do not learn external label sequences; sequence models
     were measured and the result was consistent with a label-convention mismatch, not a capacity gap.

2. **Assets used and dropped**
   - Used: IDSSE (CC BY 4.0), Metrica sample (open attribution), open-source rule references.
   - Dropped and why: video ball channel, DP ball linker, segmental MBR, GRU/TCN refiner,
     boundary head, in-Viterbi dwell, per-frame stacker, rule vote in decoder, noisy student,
     CatBoost, velocity pitch control, touch-sequence features, global IDSSE relabel.
   - Family 5 audit result: external disagreement pattern consistent with convention differences.

3. **Possession rules implemented**
   - Loose balls, challenges, aerial duels without subsequent control, restarts, dead ball.
   - Where the line is drawn, with references.

4. **Limitations**
   - Private label convention unknown. Public window is 17 possessions. Nine external matches are
     two datasets. No live leaderboard. What we would fix with more time.

5. **Reproduction**
   - Exact commands: download, parse, `python run.py`, validate.
   - Pinned dependencies, sha256 of the delivered file.
   - How to reproduce the bench and the audit.

## Consolidated HTML
- `analysis/autonomous_research/reports/FINAL_REPORT.html`.
- Cite `final_v5` as the preserved baseline owned by the existing project. Do not claim a new
  improvement unless an independently reproduced candidate clears ext on its own.
- Deterministic single file, ECharts, validation limits prominent.

## Notes
- The possession CSV is handled by the audit engineer. We do not hold or publish local row content.
- Build PDF after results, using the html/pdf skill and verifier gates.

## Results ledger (internal; pending sections marked [pending])

Preserved best = final_v5 (owned by the existing project). No new candidate has beaten it.

Reproduced baseline (decoder track, independent): bench 0.661, ext 0.616, masked 0.706.
Official scorer public: spellF1 0.7226, macro 0.9098.

final_v5 reproducibility (independent local hash/count check):
- sha256 4089da23...c8a identical across original and two regenerations (R-008).
- 199319 rows. Counts: X 40462, H 52668, A 33158, D 73031 (R-009, corrected).
- Deterministic; production frozen. No full-match verified score beyond the public scorer.

Sequence track (independent-verified):
- Memoryless geometric evidence smoother: bench 0.633 at h=1/400, negative at every hazard.
- Semi-Markov SUM with fixed generic duration prior: bench 0.657, ext 0.614, masked 0.699,
  public macro 0.910. 0.004 below baseline. Not promoted.
- Semi-Markov true-normalized MEAN arm: implementation collapse, discarded.
- Invalid v1 run (bench 0.000) was a connectivity bug, preserved as failed implementation.

Decoder track (independent-verified):
- Duel-conditioned H/A switch cost: flat to -0.001 bench. Rejected.
- Transition cost levers C1-C4: none cleared the gate.
- [pending] convention e1_screen result.

E1 duel reweighting screen (pending full recipe):
- Fast screen (300 trees, step=5, no mirrors/noise), IDSSE-only duel weight 0.5 on 3.5 percent of frames.
- w=0.5 vs matched w=1.0: bench 0.589 vs 0.569 (+0.020), ext +0.007, masked +0.033.
- Screen-only, not comparable to full 0.661. Full 600-tree mirrored+noise recipe pending.
- Tuning disclosure: no labels enter E1 training, but the masked yardstick reuses the same 17 public possessions under ball masks, so choosing w on bench+ext+masked is public-derived development selection. The raw public fold number is a sanity check, not a selector. OOF bench and ext are also reused across many attempts. No independent holdout exists. Do not state "never tuned the public window" unqualified.
- If the full ext gate alone is positive, that is cross-domain signal, but repeated ext use still counts as development, not confirmation.
- Public numbers: fold-model 0.697 vs full-fit official 0.7226 are different protocols, stated explicitly.

Family 5 integrity:
- External OOF disagreement audit: IDSSE 8.66 percent off-diagonal, mostly team calls;
  Metrica 14.75 percent, mostly alive called dead. Consistent with convention differences.
- [pending] sports screen result.

Validation limits to state prominently in the published doc:
- Public window is 17 possessions. Not a leaderboard.
- Nine external matches are two datasets. No possession-level significance.
- 0.005 adoption gate is a development heuristic, not significance.
- All fold-model public numbers (0.697) differ from the full-fit public 0.7226 protocol.

The published HTML/PDF must contain no [pending] placeholders; unresolved items are omitted or
stated as limitations. No new gain claim.
