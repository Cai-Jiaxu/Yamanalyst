# Handover: MTAH 2027 Phase 1 ball possession, four workstreams

Written 2026-10-26 for a team of four (Person 1–4) taking over from here. It says what exists, what
has been proven, what has been ruled out, and who owns what next. It does not repeat the evidence:
it links to it. If this file and a linked record disagree, the record wins
([docs/INVESTIGATION.md](docs/INVESTIGATION.md) for numbers, [docs/ROADMAP.md](docs/ROADMAP.md) for
open-item status) — then fix this file.

**Contents.** [A. Overview](#a-project-overview) · [B. Current state](#b-current-state-of-the-project) ·
[C. Findings](#c-completed-work-and-important-findings) · [D. Workstreams](#d-four-workstreams) ·
[E. Why this split](#e-how-the-split-was-refined) · [F. Tasks](#f-task-breakdown-and-priorities) ·
[G. Interfaces](#g-interfaces-and-coordination) · [H. Milestones](#h-shared-milestones-and-completion-criteria) ·
[I. First day](#i-first-day-checklist)

---

## A. Project overview

**Task.** One match, FC Barcelona (home, `H`) vs Athletic Club (away, `A`). For each of the
**199,319** frames of the tactical video (25 fps) output `H`, `A`, `D` (dead ball) or `X`
(abstain). 158,857 frames are graded: 11185–90651 (1st half) and 110808–190197 (2nd half). Inputs:
the organiser's tracking (ATD: players, ball, homography), Smart Tagging events, tactical and
broadcast videos. Full definitions: [challenge/CHALLENGE.md](challenge/CHALLENGE.md),
[challenge/DATA.md](challenge/DATA.md).

**Metric.** Ranked on **`spellF1`**: whole possessions matched one-to-one (same team) by temporal
IoU, F1 averaged over IoU 0.3–0.7. Tie-break **`macroF1`** over H/A/D frames. One spurious 0.4 s
flip per minute costs about as much as missing every possession shorter than 3 s, so precision on
possession changes matters most. Official scorer: [evaluation/possession_metrics.py](evaluation/possession_metrics.py);
explanation: [challenge/EVALUATION.md](challenge/EVALUATION.md).

**Ground truth we can see.** Only the first 5 minutes (frames 11185–18684, 17 possessions):
[evaluation/gt/BAR-ATH_possession_gt_public.csv](evaluation/gt/BAR-ATH_possession_gt_public.csv).
It is the best-tracked part of the match (ball seen 88–99% there vs ~59% overall), so it flatters
any method. The private full-match GT is never available.

**Constraints that can disqualify us** ([challenge/RULES.md](challenge/RULES.md),
[challenge/SUBMISSION.md](challenge/SUBMISSION.md), [README.md](README.md#data-rules-read-first)):

- Labels must come from an algorithm. Hand labels only for training or validation.
- **Never give match data, video, audio or anything derived from them (incl. per-frame outputs) to a
  hosted LLM** (ChatGPT, Claude, Gemini, Copilot …). Code written with an LLM is fine. Scripts here
  print aggregates only; keep it that way.
- Do not redistribute the data; the repo becomes public at submission, so nothing match-derived
  may be committed (see `.gitignore`).
- Deliverables by **17 Nov 2026**: `<team_name>.csv`, `<team_name>.pdf` (or `.md`) technical
  document, public repo URL that reproduces the CSV exactly. Registration (3–5 members, one team
  name) closes **10 Nov**; our own code freeze is **14 Nov** ([ROADMAP dates](docs/ROADMAP.md#dates)).

**Goal of the remaining work.** Keep a valid, byte-reproducible submission at all times, and
replace it only with a candidate that beats it under the agreed protocol (§G.5).

## B. Current state of the project

### B.1 Baseline and best verified implementation

| | what | public spellF1 / macroF1 | bench (ext / masked) | how to get it |
|---|---|---|---|---|
| official starter baseline | nearest player to the ball, ±50-frame majority vote | 0.326 / 0.621 | — | `evaluation/results/baseline_smooth50.csv` (committed); `starter/baseline.py` |
| **best verified, current submission** | **`final_v5`** | **0.7226 / 0.9098** (delivered full fit) | **0.6612** (0.6163 / 0.7061), fold models | `uv run python run.py --name final_v5`; sha256 `4089da237ca8780a6041bad8cc78261f29e9c5f5ef9fe8d3bdd9d9a8e2951c8a` |

`final_v5` is also the **internal baseline** every experiment is compared with. Its locked numbers,
config and the developer interface are in [baseline_snapshot/](baseline_snapshot/README.md). Do not
mix the two families of numbers: fold-ensemble dev numbers (bench 0.661, public 0.697) vs the
delivered full-fit file (public 0.723).

Other yardstick for `final_v5`: `val_h2` (4 person-labelled 5-min windows from the 2nd half, 57
spells) 0.593 on the delivered file. Caveat: those eye labels reward the hand rules' boundary timing
(INVESTIGATION §20), so `val_h2` is used for timing-free checks only.

### B.2 Pipeline, end to end

```
data/tactical/ (ATD)  ──► mtah/data.py       cached arrays: players, ball, homography, camera
                      ──► mtah/features.py   ball cleaning, distances, speeds, out-of-pitch
                      ──► mtah/rules.py      touch episodes + dead-ball state machine (output is a FEATURE)
                      ──► mtah/shape.py      ball-free team shape
                      ──► mtah/feats.py      full_plus3: shape ±1/3/5/10 s, ball/touch, rule internals,
                                             k-nearest, pitch control, set-piece geometry
external/ (IDSSE ×7, Metrica ×2, true labels)
                      ──► mtah/external.py + mtah/noise.py   degraded to our tracker, + mirrors, + noise copies
                      ──► mtah/final.py      HistGradientBoosting (600 trees, seed 0) trained on external only
                      ──► mtah/decode.py     3-state Viterbi (switch cost 12/12, w_shape 0.75, min dead 1 s)
                                             + mtah/kickoff.py (dead ball before each half's kickoff)
                                             + stoppage gate (stop_gate=100)
                      ──► evaluation/results/<name>.csv ──► evaluation/validate_submission.py
```

Entry point: [run.py](run.py) (deterministic). Config: [mtah/final.py](mtah/final.py) (`FEATFN`,
`MODEL`, `DECODER`). Development harness: [mtah/bench.py](mtah/bench.py) `Bench` with
`probs` / `score` / `screen` / `evaluate`; [analysis/yardsticks.py](analysis/yardsticks.py) scores a
decoder on cached probabilities on every yardstick. Masked validation: [mtah/masking.py](mtah/masking.py).
Person-run labelling tool: [mtah/label_tool.py](mtah/label_tool.py) → `work/labels/`.

### B.3 Yardsticks

| yardstick | what | size | trust |
|---|---|---|---|
| `ext` | 9 external matches, 3 folds by match, degraded to our tracker | ~3,000 possessions | steadiest, but 2 datasets with their own label conventions |
| `masked` | public window with the ball hidden like the full match, 16 seeds | the same 17 possessions | sd 0.024 is ball-placement noise, not independent |
| `bench` | (ext + masked) / 2 | | the adoption metric |
| `public` | public window as delivered | 17 possessions | sanity only, never select on it |
| `val_h2` | eye labels, late 2nd half | 57 spells | timing-free checks only |
| Metrica game 3 | one-time holdout | | **consumed** on 2026-10-26 |

### B.4 How to run

```bash
uv sync                                         # Python 3.11; pinned in pyproject.toml / uv.lock
# behind a TLS-intercepting proxy: UV_NATIVE_TLS=1 uv sync  (worked on the original Mac;
#   README also mentions UV_SYSTEM_CERTS=1 — unverified which your uv needs)
bash scripts/download_external.sh && uv run python -m mtah.external     # ~2.6 GB external data
uv run python run.py --name final_v5            # first run 30–60 min (caches), then ~11 min
uv run python evaluation/validate_submission.py evaluation/results/final_v5.csv
uv run python evaluation/run_evaluation.py --no-detail                  # public-window board
.venv/bin/python baseline_snapshot/reconstruct_baseline.py              # dev bench, ~40 s; needs cache/P_R1.npz
```

LightGBM does not load on the original Mac (no libomp); the pipeline uses sklearn
HistGradientBoosting. A full refit on the bench takes ~45–60 min on an M2 Pro; never run two at once
on one machine and keep it awake (`caffeinate -i`).

### B.5 What was verified on 2026-10-26 (this handover)

See [docs/CLEANUP_REPORT.md §3](docs/CLEANUP_REPORT.md#3-validation-report) for the exact commands
and outputs: compile of all 129 Python files, three unit-test files, imports of the moved modules,
`run.py` rebuild sha, the validator, the evaluator and the cached-P reconstruction.

## C. Completed work and important findings

Index of every experiment: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Evidence:
[docs/INVESTIGATION.md](docs/INVESTIGATION.md) (§ numbers below).

### C.1 Confirmed (measured, with a script)

1. **External labelled data is the key asset.** Training on 9 external matches with true labels,
   degraded to our tracker, beat our own pseudo-labels by a wide margin (§6, §8).
2. **Mirrors and extra-noise copies** of every external match: +0.039, the largest single gain (§8).
3. **Rules as features, not decoder votes**: +0.047 (§8); re-confirmed (§20: rules switch ~1.6 s
   early vs the real GT; a rule vote costs 0.025–0.03).
4. **The ball is missing on 41% of graded frames**, mostly in gaps > 10 s, so the model reads
   possession from player shape (§3, §12).
5. **Video ball detection works but does not help**: fine-tuned YOLO 79% within 2 m, yet merging it
   lowers the bench; even a perfect ball in the gaps is worth only +0.005 → video closed (§9, §12).
   67% of genuine missing-ball frames are stoppages anyway (§16).
6. **2nd-half pre-kickoff frames are graded and all `D`** (FAQ); handled by `mtah/kickoff.py` (§10).
7. **Stoppage blips**: the ATD tracks a returned ball during stoppages; the stoppage gate removes
   short live spells between dead balls (§19).
8. **Remaining errors** (ext, per 100 true possessions): missed 22, drift 19, merged 11, spurious
   3.6, split 2.1; the decoder predicts 0.71 possessions per true one. Possessions < 2 s are
   20–30% of spells and their recall is 0.19 (§20, §25).
9. **The fast screen does not predict full-fit gains**: R8 (+0.014), R17 (+0.040),
   `c_margin_dwell` (+0.031), `c_carrier_tightness` (+0.018) all fell to ≤ +0.001 on the full
   recipe (§15, §25, §26.2).
10. **`run.py` is byte-deterministic** (sha `4089da23…`, several runs, §24, §26.4).
11. **No hand label reaches the submission** (only `Bench` reads `work/labels/`; §26.1).

### C.2 Ruled out (do not repeat without a new reason)

Decoder variants (MBR, refiners, semi-Markov, smoothers, transition heads, latch, rule votes,
stacker, DoG), model swaps (CatBoost, capacity), training tricks (noisy student, relabel, duel
reweighting), feature blocks (velocity pitch control, touch-sequence, 3 causal blocks), video,
broadcast ball, Smart Tagging labels, audio D-relabel/nudge. Numbers: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#not-adopted--do-not-repeat-without-a-new-reason).

### C.3 Unfinished or partial

- `c_ballfree_cluster` full fit started, interrupted (caches half built); `c_melee_uncertainty`,
  `c_centroid_compression` screened only ([analysis/feat_track/README.md](analysis/feat_track/README.md)).
- Audio: whistle detector precision unknown (no labels); semantic AV delay unmeasured; PANNs
  full-match extraction not run ([analysis/audio_sources/](analysis/audio_sources/README.md)).
- R18 eye-label calibration not done. Technical document is a draft ([Yamanalyst.md](Yamanalyst.md)).
- **P_R1 rebuild verified 2026-11-02:** `baseline_snapshot/rebuild_probabilities.py`
  rebuilt `cache/P_R1.npz` on Windows with the pinned Python 3.11 environment, tactical assets,
  and all nine external matches. All five checked metrics match the locked snapshot exactly
  (bench 0.6611595515410369); `reconstruct_baseline.py` also passed. The three rebuild tests
  and all three handover test scripts passed. Local human validation labels were not available,
  so `val_h2` was not rechecked. The committed reference metrics were preserved.
  See [rebuild instructions](baseline_snapshot/README.md#rebuild-the-missing-probability-cache).
- The research folders `analysis/autonomous_research/`, `analysis/feat_track/`,
  `analysis/audio_sources/`, `baseline_snapshot/` and this documentation are **not committed yet**.

### C.4 Open questions (hypotheses, not findings)

- Does any feature help possessions < 2 s? (R17 and the feat track say not from tracking alone.)
- Is a 0.005 bench gain outside run-to-run noise? Never measured (R22).
- Would a stoppage gate keyed on player pace and ball distance remove more spurious spells? (R20)
- Can referee whistles mark dead-ball boundaries reliably? (R21)
- Do future-context features (the shipped ±10 s windows) raise any eligibility concern? The
  challenge docs neither grant nor forbid them; the offline task allows them in practice
  ([baseline_snapshot/ELIGIBILITY_AND_REBUILD.md §3](baseline_snapshot/ELIGIBILITY_AND_REBUILD.md)).

## D. Four workstreams

| | workstream | owns (primary) | outcome they are judged on |
|---|---|---|---|
| **Person 1** | **Features and decoding candidates** | new predictive information: feature blocks in `mtah/feats.py` and `analysis/feat_track/`; post-decode / decoder candidates (R19, R20) | a ranked list of candidates with full-recipe results; at most one or two promoted |
| **Person 2** | **Data, sources and labels** | data loading and caches (`mtah/data.py`, `features.py`, `external.py`, `noise.py`), source investigations (audio, broadcast, tagging), person-made labels (R21) and `mtah/label_tool.py` | documented, reproducible, aligned inputs; source verdicts; labels delivered |
| **Person 3** | **Evaluation and error analysis** | `mtah/bench.py`, `masking.py`, `score.py`, `analysis/yardsticks.py`, the protocol, R18, R22, the verified-results record (`docs/INVESTIGATION.md`, `docs/EXPERIMENTS.md`) | every adoption claim backed by evidence; noise floor known; error analyses that point Person 1 at targets |
| **Person 4** | **Integration, reproducibility and submission** | `run.py`, `mtah/final.py`, `mtah/decode.py` merges, `pyproject.toml`/`uv.lock`, README, HANDOVER, ROADMAP status, `.gitignore`, R9, R23, the form | a valid, byte-reproducible `<team_name>.csv`, PDF and public repo on time |

Boundaries: Person 1 proposes, Person 3 verifies, Person 4 merges into the shipped path. Person 2
supplies inputs to Person 1 and labels to Person 3, and never changes `mtah/final.py`.

## E. How the split was refined

The proposed split was kept, with three adjustments that follow from the project's state:

1. **Decoder/post-processing candidates moved to Person 1.** The model is on a plateau; the two
   remaining modelling levers (R19 features, R20 stoppage gate) are both "new information" work and
   belong to one owner, so no two people edit `mtah/decode.py` in parallel.
2. **Person 2 owns labelling.** The cheapest unexplored information is person-made labels (whistles
   R21, eye-label calibration material for R18) and the audio channel; both are data work.
   Video ball is closed, so Person 2 does not restart it.
3. **Person 4 owns the people-and-deadline tasks** (registration, team name, Google account, making
   the repo public, privacy review), since these are on the critical path and are not modelling.

Shared tasks with a named owner: ROADMAP status upkeep (Person 4; each owner updates their rows),
INVESTIGATION entries (author writes, Person 3 reviews), cache sharing (Person 2), the freeze
decision (Person 3 recommends, Person 4 executes).

## F. Task breakdown and priorities

Each task: objective · files · done so far · remaining · inputs · deliverable · definition of done
(DoD) · validation · risks.

### Person 1 — Features and decoding candidates

**Immediate**

- **P1-I1 Learn the harness and the dead ends.** Objective: avoid repeating 30+ failed ideas.
  Files: [baseline_snapshot/OPERATING_INTERFACE.md](baseline_snapshot/OPERATING_INTERFACE.md) §4–§7,
  [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), INVESTIGATION §8, §20, §25, §26.
  Done: everything listed there. Remaining: read; run `reconstruct_baseline.py` once P_R1 exists.
  DoD: you can explain why R17 failed and what `key=` does. Risk: none.
- **P1-I2 Error targets.** Get from Person 3 the latest breakdown (short possessions, stoppage
  spurious spells, 2nd-half drift) so candidates aim at a measured error. Inputs: P3-N2.

**Next**

- **P1-N1 R20 stoppage-ball gate v2.** Objective: remove spurious live spells when a ball moves in
  a stoppage (§16: ball 7.8–12.4 m/s while players walk 1.1–1.3 m/s, > 5 m away). Files:
  `mtah/decode.py` (`stop_gate`), `analysis/r11_stack.py` (where the first gate was measured),
  `analysis/yardsticks.py`. Done: v1 gate in `final_v5`. Remaining: a post-decode rule using pace
  < 1.3 m/s and no player within 5 m, swept on cached `P_R1`. Inputs: `cache/P_R1.npz`, P_R1
  public/val windows. Deliverable: `analysis/r20_gate.py` + INVESTIGATION §27. DoD: verdict
  against rule 1 with ext, masked, public, val_h2 and spurious/100. Validation: decoder-only on
  cached P, seconds per config; limit to ≤ 6 configs and report all (multiple-chances rule).
  Risk: over-tuning to the 2 public stoppage blips.
- **P1-N2 R19 finish the diverse full fits.** Objective: close the feat track honestly. Files:
  `analysis/feat_track/feat_full_diverse.py`. Done: screens; half the caches. Remaining: run both
  (≈50 min each, one at a time). DoD: two result lines in `feat_full_results.jsonl`, registry and
  §26.2 updated. Risk: low value (expect ≤ +0.005); skip if compute is needed elsewhere.

**Later**

- **P1-L1 Short possessions (< 2 s, recall 0.19).** Only after P3-N2 shows where they are lost
  (detected but merged by the decoder, vs never in the probabilities). If the decoder merges them,
  try a duration-aware switch cost on cached P; if the probabilities miss them, features are needed
  and R17/feat-track show tracking-only touch features are not enough.
- **P1-L2 Audio as a decoder channel**, only if R21 gives whistle precision ≥ ~0.7: whistle →
  lower alive↔dead switch cost near the event. Not a tree feature (external data has no audio).

**Optional**: `c_melee_uncertainty` full fit; a ranked "next features" list in
`analysis/feat_track/README.md` for any future phase.

### Person 2 — Data, sources and labels

**Immediate**

- **P2-I1 Data on every machine.** Objective: everyone can run. Files: [challenge/DATA.md](challenge/DATA.md),
  `scripts/download_external.sh`, `mtah/data.py`, `mtah/external.py`. Remaining: confirm each member
  has the enrolment download (`data/tactical/`, `data/broadcast/`) and external data; set up a
  **private team share** (not git, not an LLM) for `work/labels/*.csv` and `cache/P_R1.npz`.
  DoD: all four can run `run.py` (with Person 4's P4-I1). Risk: enrolment per member; 42 GB of
  cache per machine.
- **P2-I2 Data dictionary.** Objective: one page of input schemas. Remaining: document what
  `mtah/data.py` caches (arrays, shapes, units, fps 25, coordinate frame metres, half ranges
  11185–90651 / 110808–190197, kickoff 113742), what `external.py` produces, and the noise
  model parameters in `noise.py`. Deliverable: `docs/DATA_PIPELINE.md`. DoD: Person 1 can write a
  featfn from it without reading `data.py`.

**Next**

- **P2-N1 R21 whistle labels.** Objective: measure the tonal whistle detector. Files:
  `analysis/audio_sources/bin/extract_whistle_tonal.py`, `docs/ADDENDUM_2026-10-26.md`. Done:
  detector, 206 events at thr 0.30, kickoffs within 1.4 s. Remaining: a **person** listens to 2–3
  val windows and marks whistle times (CSV of times only, in `work/labels/`, gitignored); compute
  precision/recall and median offset per threshold. Deliverable: §26.3 update + verdict. DoD:
  precision/recall at 3 thresholds. Risks: crowd noise; labelling time ~2 h; never use an LLM to
  listen.
- **P2-N2 Reproducible audio setup.** Rebuild `analysis/audio_sources/pylibs` from its README on a
  second machine and record any fix. DoD: `extract_whistle_tonal.py` runs on two machines.

**Later**

- **P2-L1 Data-quality report**: ball-seen fraction by half and by 5-min block, player count,
  tracking gaps, the stoppage ball phenomenon — aggregates only. Feeds P3-N2 and the technical
  document's limitations.
- **P2-L2 Semantic audio–video delay** around the 3 known boundaries, if P1-L2 goes ahead.

**Optional**: PANNs full-match extraction (~6 min CPU) and crowd-energy curve as a dead-ball cue
(`docs/PRIORITIZED_PROPOSALS.md` P0.2) — only with R21 labels to judge them.

### Person 3 — Evaluation and error analysis

**Immediate**

- **P3-I1 Verify the evaluation chain.** Files: `evaluation/possession_metrics.py`,
  `mtah/score.py`, `mtah/bench.py`, `analysis/yardsticks.py`. Done: scorer read and confirmed
  (§26.1); reconstruction reproduced (`baseline_metrics.json`). Remaining: on your machine run the
  evaluator on `final_v5.csv` (0.722581 / 0.909848) and `reconstruct_baseline.py` (bench 0.6612).
  DoD: both match to 4 decimals. Risk: needs P_R1 (P4-I2).

**Next**

- **P3-N1 R22 noise floor.** Objective: know what a "gain" is. Remaining: refit the `final_v5`
  recipe with 2–3 other `random_state`s (new `key=` each), report bench/ext/masked spread. Inputs:
  compute (~1 h each). Deliverable: INVESTIGATION §27 or §28, and an updated gate in ROADMAP rule
  1 if the spread is > 0.005. DoD: spread table with ≥ 3 seeds. Risk: seed changes only the trees,
  not the noise draws — say so.
- **P3-N2 Error analysis for targets.** Objective: tell Person 1 where errors are. Files:
  `analysis/err_locus.py`, `analysis/miss_by_len.py`, `analysis/drift.py`, `analysis/diagnose.py`.
  Remaining: on ext and val_h2, split errors by possession length, by ball-seen vs missing, by
  stoppage context, and "missed in the probabilities" vs "merged by the decoder". DoD: one table
  per split in INVESTIGATION, shared with P1. Validation: aggregates only.
- **P3-N3 R18 eye-label calibration.** A person labels the public window by eye without looking at
  the GT (`mtah/label_tool.py`), then compares with the real GT: offsets of boundaries, team
  agreement, dead-ball F1. DoD: tells us which `val_h2` columns remain trustworthy.

**Later**

- **P3-L1 Screen tier redesign.** The 300-tree screen gave four false positives. Test whether a
  cheaper "mirrors only, 300 trees" tier tracks the full recipe on the existing R8/R17/feat-track
  cases (caches exist). DoD: correlation table; ROADMAP rule 7 updated.
- **P3-L2 Freeze recommendation** (by 13 Nov): list of candidates that passed, with evidence.

**Optional**: per-seed masked distributions; bootstrap over external matches (with the caveat
that 9 matches are 2 datasets).

### Person 4 — Integration, reproducibility and submission

**Immediate**

- **P4-I1 Windows submission reproduction verified 2026-11-02; clean-clone replay pending.**
  Two independent full-fit runs in the existing checkout (`windows_repro_20261102_b` and `_c`)
  reproduce sha `4089da237ca8780a6041bad8cc78261f29e9c5f5ef9fe8d3bdd9d9a8e2951c8a`.
  Both pass the validator and official public scoring (spellF1 0.722581, macroF1 0.909848).
  `run.py` now writes explicit UTF-8/LF: the original Windows writer used CRLF, and its diagnostic
  output matched the baseline exactly after LF normalization. The serialization regression test
  and required test scripts passed. Remaining: the final fresh-clone replay at the freeze.
  Details: `baseline_snapshot/README.md`. DoD: expected sha on ≥ 2 machines and clean-clone check.
- **P4-I2 Make `cache/P_R1.npz` reproducible — verified 2026-11-02.** Script committed as
  `ff8421b`: `baseline_snapshot/rebuild_probabilities.py` (full_plus3, final.MODEL, mirrors and
  noise copies, fresh fold models). The full rebuild and `reconstruct_baseline.py` both reproduce
  bench 0.6611595515410369 on Windows; all five checked metrics have zero difference from the
  locked snapshot. Aggregate verification details are in `baseline_snapshot/README.md`.
- **P4-I3 Commit the cleaned repo** (after R23-light: confirm `git status` shows no CSV, npz,
  joblib, audio or video). Message with the cleanup summary. Commit to `main` (team convention).

**Next**

- **P4-N1 People and form tasks.** Registration of all members under one team name by 10 Nov; a
  Google account for the form; decide `<team_name>`. DoD: confirmation emails.
- **P4-N2 Integration of an approved candidate** (only after Person 3 signs off): change
  `mtah/final.py`, run `run.py` twice, update README sha and scores, Yamanalyst table,
  INVESTIGATION, ROADMAP. DoD: new sha reproducible twice; validator OK.
- **P4-N3 R23 privacy review** — *done 2026-10-27 for everything committed so far; repeat for files added later* — before making the repo public: read every committed file under
  `analysis/` and `baseline_snapshot/` (logs and console dumps from several agents) and remove or
  aggregate anything frame-level. DoD: written sign-off in CLEANUP_REPORT.

**Later**

- **P4-L1 R9 technical document**: refresh `Yamanalyst.md` from INVESTIGATION at the freeze,
  rename to `<team_name>.md`, export `<team_name>.pdf`. Also rename the CSV to `<team_name>.csv`.
- **P4-L2 Submission day**: validate the exact file uploaded, repo public, URL in the form,
  final clean-clone rerun gives the same sha.

**Optional**: retire unused deps from `pyproject.toml` only if `run.py` still reproduces the sha
(torch/ultralytics serve the unused video path); a `Makefile`.

## G. Interfaces and coordination

**G.1 File ownership.** Edit outside your area only after a message to its owner.

| path | owner | others |
|---|---|---|
| `run.py`, `mtah/final.py`, `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md`, `HANDOVER.md`, `Yamanalyst.md` | Person 4 | propose via message/PR |
| `mtah/feats.py`, `mtah/decode.py`, `mtah/kickoff.py`, `analysis/feat_track/`, new `analysis/r2x_*.py` | Person 1 | Person 4 merges into the shipped path |
| `mtah/data.py`, `features.py`, `external.py`, `noise.py`, `shape.py`, `label_tool.py`, `analysis/audio_sources/`, `work/labels/` | Person 2 | |
| `mtah/bench.py`, `masking.py`, `score.py`, `report.py`, `analysis/yardsticks.py`, `baseline_snapshot/`, `docs/INVESTIGATION.md`, `docs/EXPERIMENTS.md` | Person 3 | authors append their own sections |
| `docs/ROADMAP.md` | Person 4 (structure) | each owner updates their rows' status |
| `evaluation/`, `challenge/` | nobody (organiser's) | never edit |

**G.2 Data interfaces.**
- Feature function: `featfn(match, f, shapeX) -> (n_frames, k) float array`; `f` is
  `mtah.features.compute(match)`; must work identically on `ExtMatch` (degraded external) and our
  `Match`, return NaN where an input is missing, and never read labels.
  Compose with `analysis.feat_track.feat_candidates.compose("full_plus3", [...])`.
- Probabilities: `Bench.probs(featfn, model_params, extra=..., key=...)` → dict of per-yardstick
  `(n, 3)` arrays; cached as `cache/models_<key>.joblib` and (by the runners) `cache/P_<key>.npz`.
- Decoder parameters: a dict like `mtah.final.DECODER`; post-decode steps take and return a label
  array.
- Submission: `frame,label`, 199,319 rows, labels H/A/D/X; checked by `validate_submission.py`.
- Labels from people: GT-format CSV `frame,label` in `work/labels/` (gitignored), shared privately.

**G.3 Naming and recording.** Experiment name = cache key = INVESTIGATION subsection title:
`<track>-<short>-v<k>` ([docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#naming-a-new-experiment)). New
config = new key. Every run, kept or not, gets a line in the track's JSONL/registry and an
INVESTIGATION entry; verdicts go into EXPERIMENTS.md and ROADMAP.

**G.4 Before merging.** `python -m py_compile` on changed files; the unit tests
(`analysis/feat_track/test_feat_candidates.py`,
`analysis/autonomous_research/experiments/sequence/test_semimarkov.py`,
`analysis/autonomous_research/experiments/holdout/test_class_order.py`); if the shipped path
changed, `run.py` twice (same sha) and the validator. `git status` must show no data-derived file.
Keep commits small; `git pull` first (main-only workflow; the original team chose it — revisit if
conflicts appear).

**G.5 Evidence for a claimed improvement.** Full recipe (not the screen), with: bench, ext (idsse,
metrica), masked (mean and sd), public spell/macro, val_h2, error taxonomy, recall by length,
spurious/100, the `key`, and how many variants were tried. Adopt only per ROADMAP rule 1 (bench
≥ +0.005, ext and masked not below −0.005), and beyond the noise floor once R22 measures it.
Person 3 signs off; Person 4 integrates.

**G.6 Protecting the best version.** `evaluation/results/final_v5.csv` is never overwritten; a
new submission gets a new name (`final_v6`). README's sha changes only with a merged, verified
candidate. The freeze is 14 Nov.

**G.7 Compute.** One refit at a time per machine; announce long runs in the team channel with
expected end time; run them with `nohup caffeinate -i … &` so they survive app crashes and sleep.

**Remaining onboarding dependencies:** data and private labels on every machine (P2-I1),
registration (P4-N1). P_R1 can now be rebuilt locally; P4-I2 was verified on 2026-11-02.

## H. Shared milestones and completion criteria

| # | milestone | deliverables | depends on | accepted when | who |
|---|---|---|---|---|---|
| M1 | Everyone onboarded | read §A–C, README, OPERATING_INTERFACE | — | each person can state the metric, the baseline numbers and 3 dead ends | all |
| M2 | Baseline verified everywhere | `run.py` sha on each machine; P_R1 available; evaluator and reconstruction numbers | M1, P2-I1, P4-I1/I2 | sha `4089da23…` and bench 0.6612 reproduced on ≥ 2 machines; repo committed (P4-I3) | P4 (lead), P2, P3 |
| M3 | Workstreams running | R20, R19 (P1); R21, data dictionary (P2); R22, error analysis, R18 (P3); registration (P4) | M2 | each task has a result entry in INVESTIGATION/EXPERIMENTS | P1–P4 |
| M4 | Candidates judged | freeze recommendation with evidence | M3 | every candidate has a verdict under G.5 | P3 (lead), P1 |
| M5 | Code freeze (14 Nov) | final `mtah/final.py`; README sha; Yamanalyst numbers refreshed | M4 | `run.py` twice gives the README sha; validator OK | P4 |
| M6 | Submission (by 17 Nov) | `<team_name>.csv`, `<team_name>.pdf`, public repo, form | M5, R23 | clean clone reproduces the uploaded file; privacy sign-off; form submitted | P4 (lead), all review |

## I. First-day checklist

Everyone:

- [ ] Read [README.md](README.md) (data rules first), §A–C here, and
      [baseline_snapshot/OPERATING_INTERFACE.md](baseline_snapshot/OPERATING_INTERFACE.md).
- [ ] Confirm you are enrolled and have the match data; unpack into `data/tactical/`,
      `data/broadcast/` ([challenge/DATA.md](challenge/DATA.md)).
- [ ] `uv sync` (TLS proxy: see B.4), then `bash scripts/download_external.sh && uv run python -m mtah.external`.
- [ ] Start `uv run python run.py --name final_v5` (30–60 min first time) and check the sha
      `4089da237ca8780a…` and `validate_submission.py`.
- [ ] `uv run python evaluation/run_evaluation.py --no-detail` → `final_v5` 0.7226 / 0.9098.
- [ ] Rebuild `cache/P_R1.npz` with `baseline_snapshot/rebuild_probabilities.py` (or get it
      from the private share); get `work/labels/` from the private share, then
      `.venv/bin/python baseline_snapshot/reconstruct_baseline.py` → bench 0.6612.
- [ ] Skim [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) so you don't repeat a dead end.

First task each:

- **Person 1**: P1-N1 (R20 stoppage gate v2) on cached P.
- **Person 2**: P2-I1 (data + private share for P_R1 and labels), then P2-N1 (whistle labels).
- **Person 3**: P3-I1 (verify evaluation), then start P3-N1 (seed refits) overnight.
- **Person 4**: P4-I1 (clean-clone submission reproduction), P4-N1 (registration).
  P4-I2 (P_R1 rebuild) is verified; see the result above.

Current onboarding dependencies: match data and private labels on teammates' machines;
registration closes 10 Nov. The missing P_R1 rebuild tooling and verification are resolved.
