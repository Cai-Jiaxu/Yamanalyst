# Roadmap: what we do next, and why

**This file is the source of truth for open work.** Update its status column in the same commit
as the result. Everything else has one job:

| doc | job |
|---|---|
| `ROADMAP.md` (this) | open items, their order, decision rules and status |
| [INVESTIGATION.md](INVESTIGATION.md) | the evidence: every measured number and the script behind it |
| [RECOMMENDATIONS_REVIEW.md](RECOMMENDATIONS_REVIEW.md) | verdicts on outside advice (Do / Test / Defer / No) |
| [README.md](README.md) | how to run it and the current submission |
| [PLAN.md](PLAN.md) | the day-1 plan, historical |

Last updated 2026-10-23.

## Where we stand

`final_v2`: bench **0.656** (ext 0.617, masked 0.695), public spellF1 0.675 on the bench decoder
(0.700 for the delivered file). It is `final_v1` plus dead ball before the 2nd-half kickoff. Remaining error on external matches, per 100 true possessions:
missed 22, boundary drift 19, merges 11, spurious 3.6, splits 2.1; predicted/true 0.71.

Closed since the last plan: video ball as a drop-in for the ATD ball (lowers the bench at every
coverage, INVESTIGATION.md §9–9.1); R1 (INVESTIGATION.md §10).

## Dates

| date | what |
|---|---|
| **10 Nov** | registration closes: all 3–5 members registered under one team name |
| **14 Nov** | code freeze: last model change merged; only docs after this |
| 15 Nov | technical document done; repo made public; clean-clone rerun gives the same sha |
| **17 Nov** | submit `<team_name>.csv` + `<team_name>.pdf` + public repo URL on the Google Form |

## Rules for every experiment

1. Judge on `mtah/bench.py`: adopt only if bench rises by **≥ 0.005** and neither ext nor masked
   falls by more than 0.005. Smaller differences are noise.
2. **Every try goes into INVESTIGATION.md, kept or not.** We have run dozens of variants on one
   bench; picking the maximum of many noisy scores overstates the winner. A borderline win that
   needed several tries is a "no".
3. The public window is a sanity check, never the reason to pick.
4. Data never goes to a hosted LLM; scripts print aggregates only.
5. Refits take ~1 h. Keep the Mac awake (`caffeinate -i -w <pid>`, lid open): an overnight run
   slept for 8 h and made 20 min of progress. **Never run two refits at once**: they share the
   CPU and both take far longer.

## Open items, in order

| # | item | why (evidence) | cost | decision rule | status |
|---|---|---|---|---|---|
| R1 | **Half-start kickoff fix** in the decoder: dead ball before a half's first kickoff when there is a waiting period; nothing at after-goal kickoffs | `final_v1` has **19 label spells** (32% H/A) in the 2nd-half pre-kickoff stretch, which FAQ.md says is all `D`: ~9 spurious possessions on the full match, the costliest error type. The full kickoff evidence measured −0.008, but its harm came from after-goal kickoffs on external data | hours, decoder only (cached `P_R1`) | bench not worse; on our match the pre-kickoff stretch becomes D and the detected kickoff is within 5 frames of 113,742. The bench cannot see the gain (no pre-kickoff frames in ext or the public window), so this is adopted on a "does no harm + fixes a known error" basis | **done → `final_v2`**: bench 0.656 unchanged, kick found 5 frames early, pre-kickoff spells 19 → 2, possessions 292 → 282 (§10) |
| R2 | **Value-of-ball oracle**: how much would the bench gain with less missing ball, in-distribution? Ext matches degraded at 41% (as now), 25%, 10% missing; masked at 45%, 25%, 0% hidden | Decides whether R6 (a proper video channel) is worth its cost. W6 found ball columns add little once the leak was fixed, and masked (45% hidden) already scores above public (10%) | one refit (~1 h) | gain at 25% missing (about what perfect video at the linker's coverage would give) < 0.01 → close video; ≥ 0.02 → do R6 | **running** |
| R3 | **Hand-labelled validation** (team task, by a person, never an LLM): 3–4 five-minute windows from the late 2nd half in GT convention, plus ball visibility on ~300 genuine missing-ball frames | Our only in-domain labels are 17 possessions (masked seeds ±0.04–0.07). Allowed by RULES.md for validation | ~1 person-day | adds a third bench column; re-check `final_v1` and every R-item on it | not started, **needs an owner** |
| R4 | Noisy student: pseudo-label our own match (public window ± 30 s excluded) and add it to training | Review 1 #11: the model is now strong enough for in-domain pseudo-labels to help; the domain gap (Bundesliga / Metrica → LaLiga, our tracker) is real | one refit | bench rule | not started |
| R5 | Second model family (CatBoost) averaged with the trees | Review 1 #14; LightGBM cannot load here, CatBoost may | install + one refit | bench rule | not started |
| R6 | Video ball as its own channel: own columns and `seen`, video-like tracks simulated in external training (clumps, ~85% within 2 m, ~15% false positives), smoothing; detector retrained without the public window | INVESTIGATION.md §9.1; the only route left for video | days | only if R2 says ≥ 0.02 | **blocked on R2** |
| R7 | Minimum dead-ball dwell inside Viterbi (instead of post-hoc `min_dead`) | Review 4 #4; post-hoc gave +0.004 | hours, decoder only | bench rule | not started |
| R8 | Velocity-projected pitch control | Review 4 #2 | one refit | bench rule; drop if slot-speed spikes hurt | not started |
| R9 | **Technical document** (`<team_name>.md` → PDF), repo public, clean-clone check | SUBMISSION.md: approach, assets used and dropped, possession rules, limitations, reproduction | 1 day | due 15 Nov | not started |

Not on the list, and why: Smart Tagging fusion, `is_metrica`, X on boundaries, EM prior,
IDSSE relabel, boundary snapping, re-ranking video candidates, TCN refiner (defer). See
RECOMMENDATIONS_REVIEW.md for each verdict.

## Known issues and risks

- **Bench overfitting.** Many variants on one bench (§ rules 1–2). R3 is the real fix.
- **The masked test depends on one 5-minute window** (17 possessions, the best-tracked part of the
  match). The external half of the bench is the steadier one.
- **The detector was trained on public-window frames**, so any masked score with video in it is
  optimistic until it is retrained (R6).
- **`run.py`'s sha changes with any model or decoder change.** Update README.md's sha and
  public scores whenever the submission changes.
- **People tasks are the critical path**: registration (10 Nov), R3, making the repo public,
  the Google account for the form.
