# Roadmap: what we do next, and why

**This file is the source of truth for open work and decisions.** Update its status column in the
same commit as the result. Each document has one job and no other document repeats it:

| doc | job |
|---|---|
| `docs/ROADMAP.md` (this) | open items, their order, decision rules, status, dates |
| [docs/INVESTIGATION.md](INVESTIGATION.md) | the evidence: every measured number, the script behind it, and the literature |
| [docs/REVIEWS.md](REVIEWS.md) | verdicts on outside advice (Do / Test / Defer / No) |
| [docs/EXPERIMENTS.md](EXPERIMENTS.md) | one-line index of every experiment, pointing into the records below |
| [HANDOVER.md](../HANDOVER.md) | who owns which open item (Persons 1–4), interfaces, milestones |
| [baseline_snapshot/](../baseline_snapshot/) | the locked baseline numbers and the developer interface for experiments |
| [README.md](../README.md) | how to run it; which file is the current submission |
| [Yamanalyst.md](../Yamanalyst.md) | the technical document we submit (its numbers are copied from INVESTIGATION.md at the freeze) |
| [challenge/](../challenge/) | the organiser's documents, unchanged |

Last updated 2026-10-26.

## Where we stand

Submission **`final_v5`**: `final_v2` decoder + stoppage gate, no rule vote (INVESTIGATION.md §20,
§24). It reverses `final_v3`/`v4`, which were chosen on eye labels that reward the rules' boundary
timing. Since then measured and not adopted: MBR decoding (§21), GRU/TCN refiners (§22–23).
The one large error left is short possessions; R17's touch-sequence features did not move it (§25).
After `final_v5`, three more tracks (integrity review and decoder/training ideas, causal tracking
features, audio) were measured and none adopted (§26). The team of four takes over from here:
owners are in [HANDOVER.md](../HANDOVER.md).

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
6. **Fit once, reuse everywhere.** `Bench.probs(..., key=...)` saves the fold models and
   probabilities to `cache/` (I1); decoder and evaluation experiments then reuse them in seconds.
7. **Screen before a full refit** (I2): a candidate that loses by more than 0.005 on the fast
   screen (9 base matches, no mirror or noise copies, 300 trees) is dropped without a full refit.
   The screen only removes clear losers; it never adopts anything, because it leaves out the
   mirrors, our biggest single gain (+0.039).

## Process improvements (accepted from the fifth review, REVIEWS.md)

| # | item | status |
|---|---|---|
| I1 | Cache fold models in `Bench.probs(key=...)` (`cache/models_<key>.joblib`); probabilities are recomputed from them, so rebuilt cases stay correct | **done** |
| I2 | Fast screening tier: `Bench.screen()` (rule 7) | **done**; first use `analysis/round3.py` |
| I3 | R3 tooling: a local, person-run labelling tool that shows the video only to the person and writes a GT-format CSV; it must **not** pre-mark boundaries from any model (PLAN.md: label independently of model output) | **done**: `mtah/label_tool.py` → `work/labels/*.csv` (gitignored); the bench prints a `val` column when files exist |
| I4 | Colab GPU for screening GPU tree models / sequence models; anything adopted must still run deterministically on CPU from `run.py` | when R5 needs it |

Rejected from the same review: an LLM viewing frames or printing frame-level data (T&Cs),
model-suggested boundaries for labellers, and its unmeasured timing figures.

## Open items, in order

| # | item | why (evidence) | cost | decision rule | status |
|---|---|---|---|---|---|
| R1 | **Half-start kickoff fix** in the decoder: dead ball before a half's first kickoff when there is a waiting period; nothing at after-goal kickoffs | `final_v1` has **19 label spells** (32% H/A) in the 2nd-half pre-kickoff stretch, which FAQ.md says is all `D`: ~9 spurious possessions on the full match, the costliest error type. The full kickoff evidence measured −0.008, but its harm came from after-goal kickoffs on external data | hours, decoder only (cached `P_R1`) | bench not worse; on our match the pre-kickoff stretch becomes D and the detected kickoff is within 5 frames of 113,742. The bench cannot see the gain (no pre-kickoff frames in ext or the public window), so this is adopted on a "does no harm + fixes a known error" basis | **done → `final_v2`**: bench 0.656 unchanged, kick found 5 frames early, pre-kickoff spells 19 → 2, possessions 292 → 282 (§10) |
| R2 | **Value-of-ball oracle**: how much would the bench gain with less missing ball, in-distribution? Ext matches degraded at 41% (as now), 25%, 10% missing; masked at 45%, 25%, 0% hidden | Decides whether R6 (a proper video channel) is worth its cost. W6 found ball columns add little once the leak was fixed, and masked (45% hidden) already scores above public (10%) | one refit (~1 h) | gain at 25% missing (about what perfect video at the linker's coverage would give) < 0.01 → close video; ≥ 0.02 → do R6 | **done**: perfect ball at 30% missing gives bench +0.005 (ext +0.024, masked −0.014) → **video closed** (INVESTIGATION.md §12) |
| R3 | **Hand-labelled validation** (team task, by a person, never an LLM): 3–4 five-minute windows from the late 2nd half in GT convention, plus ball visibility on ~300 genuine missing-ball frames | Our only in-domain labels are 17 possessions (masked seeds ±0.04–0.07). Allowed by RULES.md for validation | ~1 person-day (a few hours per window; eye labels will not match the tracker-generated GT to the frame, so score team and dead-ball calls, not exact timing) | adds a third bench column `val_h2`; re-check the submission and every R-item on it | **done** (`INVESTIGATION.md` §16, `analysis/r3_eval.py`): 4 windows (`30,000` frames, 57 live spells) + 300-frame missing-ball probe (`67.0% D`, `17.3% O`, `15.7% V`) labelled in `work/labels/`. On `val_h2`, `final_v2.csv` reaches `spellF1 0.577` (`lowIoU 0.722`, `teamAcc 0.891`, `deadF1 0.938`, `collarMF1 0.895`); `R8` and `R4` reach `0.571` vs `0.558` for fold `R1`, while `R6` video ball drops to `0.536` (`deadF1 0.925`). More visible ball strictly raises `val_h2` `spellF1` (`0.440 → 0.531 → 0.555 → 0.558`) |
| R4 | Noisy student: pseudo-label our own match (public window ± 30 s excluded) and add it to training | Review 1 #11: the model is now strong enough for in-domain pseudo-labels to help; the domain gap (Bundesliga / Metrica → LaLiga, our tracker) is real | one refit | bench rule | **done, not adopted**: −0.016 (masked −0.032, public −0.041; INVESTIGATION.md §14) |
| R5 | Second model family (CatBoost) averaged with the trees | Review 1 #14; LightGBM cannot load here; **CatBoost 1.2.10 installs and runs** (venv only; add to `pyproject.toml` only if adopted, and run with `uv run --no-sync` until then) | screen, then one refit | rule 7 screen, then bench rule | **dropped at the screen**: CatBoost −0.024, HGB+CatBoost −0.015 (INVESTIGATION.md §13) |
| R6 | Video ball as its own channel: own columns and `seen`, video-like tracks simulated in external training (clumps, ~85% within 2 m, ~15% false positives), smoothing; detector retrained without the public window | INVESTIGATION.md §9.1; the only route left for video | days | only if R2 says ≥ 0.02 | **closed**: R2 upper bound +0.005 bench |
| R7 | Minimum dead-ball dwell inside Viterbi (instead of post-hoc `min_dead`) | Review 4 #4; post-hoc gave +0.004 | hours, decoder only | bench rule | **done, not adopted**: 0.656 = no change (INVESTIGATION.md §11) |
| R8 | Velocity-projected pitch control | Review 4 #2 | screen, then one refit | rule 7 screen, then bench rule | **done, not adopted**: screen +0.014, full fit 0.656 = no change (INVESTIGATION.md §13, §15) |
| R10 | **Why the model drifts in the 2nd half** | val_h2 team accuracy 0.78–0.94 vs 0.958 public | hours | — | **done** (§18): errors happen with the ball visible and are confident; model and rules each win in different windows |
| R11 | **Per-frame stacker** of model probabilities and rule evidence, fitted in-domain with leave-one-window-out CV over public + the 4 val windows (5 folds); a handful of parameters only (e.g. logistic on log P, rule vote, ball-near, on-ball distance margin) | §18: neither source dominates, so a per-frame choice beats any fixed blend | hours, decoder-level, no refit | must beat `final_v3` on every held-out window on average and not lose public by > 0.02 | **done**: stacker not adopted (public −0.08, ext −0.12); its stoppage gate adopted → **`final_v4`** (val_h2 0.632, possessions 286 → 278; INVESTIGATION.md §19) |
| R13 | **Stoppage-blip gate**: live spell ≤ 4 s between dead balls while players walk → dead | §16: ATD tracks a returned ball in stoppages | minutes, post-decode | does no harm on any yardstick | **done → `final_v4`, kept in `final_v5`** (`stop_gate=100` in `mtah/final.py`; §19, §24): on `final_v3` public +0.022, masked +0.008, ext/val unchanged |
| R12 | Make the model trust a visible ball (milder-noise augmentation where the ball is seen) | §18: model errs with the ball visible | one refit | val_h2 + ext | **closed**: its premise was the eye labels' timing (§20); mid-possession the model already matches the rules |
| R14 | **Audit the yardsticks**: where do H↔A changes land on real GT vs eye labels? | §18–19 chose on val_h2 against every real-GT yardstick | hours, cached `P_R1` | — | **done → `final_v5`** (§20): rules 1.6 s early on real GT, 0–2 frames on eye labels; v2 decoder + gate +0.025 bench |
| R15 | Segmental MBR decoding for expected spellF1 | Viterbi optimises path probability, not the metric | hours, decoder only | bench rule | **done, not adopted**: +0.001 (§21) |
| R16 | Learned temporal refiner (GRU / TCN) over the trees' probabilities | MS-TCN / ASRF; Kaggle CMI sleep | ~1–2 h CPU per run | bench rule | **done, not adopted**: ext +0.011–0.017, masked −0.02–0.03, public −0.04–0.06; learns external label conventions (§22–23) |
| R17 | **Short-possession recall**: touch-sequence and kick features (`feats.flight_block`, `full_plus5`) | 20–30% of true spells are < 2 s, recall ~0.19 (§20) | screen, then one refit | rule 7 screen; then bench rule, judged also per duration bin and spurious/100 | **done, not adopted** (§25): screen +0.040, full fit +0.001 (ext +0.009, masked −0.007); < 2 s recall unchanged at 0.186 |
| R18 | **Calibrate the eye labels** (team task, by a person, never an LLM): label the public window by eye without looking at the GT, compare with the real GT | decides what `val_h2` can still judge (§20) | ~1 person-hour | — | open (Person 3) |
| R19 | **Finish the diverse feature full fits** (`c_ballfree_cluster`, `c_melee_uncertainty`, `analysis/feat_track/feat_full_diverse.py`) | screened +0.017/+0.018, full fit started and interrupted (§26.2) | ~50 min each, caches half built | bench rule | open (Person 1); expect ≤ +0.005 |
| R20 | **Stoppage-ball gate v2**: live spell inside a stoppage when players walk (< 1.3 m/s) and no player within 5 m of the ball → dead | Yamanalyst §6; §16: ATD tracks a returned ball during stoppages | hours, post-decode on cached P | does no harm on any yardstick, gains on public/masked/val stoppage spells | open (Person 1) |
| R21 | **Whistle labels for audio**: a person marks referee whistles on 2–3 val windows, to give the tonal detector a precision/recall | §26.3: precision unknown, the one blocker that labels can lift | ~2 person-hours | — | open (Person 2) |
| R22 | **Noise floor of the bench**: refit `final_v5` with 2–3 other `random_state`s and record bench/ext/masked spread | every "not adopted" since R8 lands within 0.003; the 0.005 margin is a heuristic (§26.1) | 2–3 refits | — | open (Person 3) |
| R23 | **Privacy review before going public**: read every new committed file under `analysis/` and `baseline_snapshot/` for frame-level outputs | repo goes public at submission; research logs were written by several agents | ~1 h | nothing frame-level, no match-derived arrays | open (Person 4) |
| R9 | **Technical document** (`<team_name>.md` → PDF), repo public, clean-clone check | SUBMISSION.md: approach, assets used and dropped, possession rules, limitations, reproduction | 1 day | due 15 Nov | **draft**: `Yamanalyst.md` (refresh numbers at the freeze; export to PDF, `*.pdf` is gitignored); owner Person 4 |

Not on the list, and why: Smart Tagging fusion, `is_metrica`, X on boundaries, EM prior,
IDSSE relabel, boundary snapping, re-ranking video candidates (REVIEWS.md); ball-free deep
possessor models such as PathCRF (reported 92.4% team accuracy without the ball, about where our
shape model is, INVESTIGATION.md §0); more labelled matches (no ready source in our convention:
PFF needs a personal sign-up, SkillCorner has no per-frame possession); duration priors (§11 and
the sixth review: null).

## Next steps (2026-10-26)

Ordered milestones and owners: [HANDOVER.md §H](../HANDOVER.md#h-shared-milestones-and-completion-criteria).

1. Everyone reproduces `final_v5` on their own machine (README sha) — the gate for any other work.
2. R18 and R21 (person-run labelling), R22 (noise floor), R19/R20 (last modelling candidates).
3. 10 Nov registration; 14 Nov freeze: refresh Yamanalyst.md's numbers from INVESTIGATION.md, export the PDF.
4. 15 Nov: R23 privacy review, repo public, clean-clone `run.py` gives the README sha.

## Known issues and risks

- **Bench overfitting.** Many variants on one bench (rules 1–2). R3 (`val_h2`, 57 spells across 20 min of the 2nd half) now provides an independent 2nd-half check.
- **The masked test depends on one 5-minute window** (17 possessions, the best-tracked part of the
  match). The external half of the bench is the steadier one. On that window, *less* hidden ball
  scores *lower* (`0.675` at 0% extra hidden vs `0.695` at 45% hidden, §12): §16 (`analysis/r3_eval.py`)
  shows that visible ball slightly *raises* true-possession recall (`10.80` vs `10.66` of 17) and leaves
  live team accuracy (`0.958`) and dead-ball F1 (`0.969` vs `0.971`) unchanged, but triggers 2 short
  spurious spells (`1.84 s` and `2.28 s`) inside ground-truth stoppages (`D`) where the ATD tracks a
  fast-moving returned ball (`bspeed 7.8–12.4 m/s`) while players walk (`1.1–1.3 m/s`) $>5\text{ m}$
  away. `pace_weights` (`3×` bias on `pace < 1.3 m/s`) selectively masks those stoppage frames. On
  `val_h2` (where natural ball visibility is already `62.8%`), more visible ball strictly raises
  `spellF1` (`0.440 → 0.531 → 0.555 → 0.558`) and `teamAcc` (`0.856 → 0.890`), while `25%` slow-frame
  masking still trims stoppage-time spurious spells (`13.00 → 9.12`) and lifts relaxed-IoU `spellF1`
  (`0.734 → 0.748`).
- **The detector was trained on public-window frames**, so any masked score with video in it is
  optimistic until it is retrained (R6).
- **`run.py`'s sha changes with any model or decoder change.** Update README.md's sha and
  public scores whenever the submission changes.
- **People tasks are the critical path**: registration (10 Nov), R3, making the repo public,
  the Google account for the form.
