# Cleanup and validation report, 2026-10-26

What the handover cleanup changed, why, and which checks were actually run. Nothing was
permanently deleted: everything removed from the working tree is in the local, gitignored
`_archive/` (35 files, 25 MB). Restore any item with `mv _archive/<path> <original path>`.

## 1. Moved or archived

| from | to | why | evidence it was safe |
|---|---|---|---|
| `evaluation/results/ar_baseline_check.csv`, `ar_verified_v5_a.csv`, `ar_verified_v5_b.csv`, `snapshot_fresh_rebuild.csv` | `_archive/evaluation_results/duplicates_of_final_v5/` | duplicate rows on the scoreboard | sha256 identical to `final_v5.csv` (`4089da23…`) |
| `evaluation/results/ar_candidate_dog.csv`, `ar_candidate_dog_fusion.csv`, `research_audio_d_relabel.csv`, `research_hardD_A_v2.csv`, `research_hardD_B_v2.csv` | `_archive/evaluation_results/rejected_candidates/` | rejected candidates; `_fusion` has a broken schema | verdicts recorded in INVESTIGATION §26.1, §26.3; no script reads them back except the audio scripts that write them |
| `analysis/autonomous_research/source_only_additions.tgz` | `_archive/autonomous_research/` | a tarball of the same sources | all 95 files byte-identical to the live tree |
| `analysis/autonomous_research/experiments/reproducibility/score_in/*.csv` (3) | `_archive/autonomous_research/reproducibility_score_in/` | full-match label CSVs in a folder that was going to be committed | sha256 identical to `final_v5.csv` |
| `analysis/autonomous_research/reports/_shots/`, `artifacts/baseline_report/_shots/` | `_archive/…/report_screenshots/` | rendering screenshots of reports | regenerable; the HTML reports stay |
| `artifacts/followup_temporal_LOG.md`, `artifacts/sequence_experiments_LOG.md` | `_archive/artifacts/` | copies of the logs inside `analysis/autonomous_research/experiments/{followup_temporal,sequence}/artifacts/LOG.md` | first identical; second differs only in an attribution line |
| `artifacts/baseline_report/baseline_snapshot_report.html` | `baseline_snapshot/report.html` | belongs with the snapshot | paths inside updated |
| `research_sources_hack2027/` | `analysis/audio_sources/` | research lives in `analysis/`; root was cluttered | hard-coded absolute paths in 4 scripts replaced by paths computed from the script location; compiled |
| `research_sources_hack2027/dist/research_sources_hack2027_public.zip` | `_archive/audio_sources/dist/` | export of the same scripts and docs | listing matched the live files |
| `analysis/feat_*.py`, `test_feat_candidates.py`, `feat_*_results.jsonl`, `feat_*_run.log`, `run_screen_when_free.sh` | `analysis/feat_track/` | one folder per track, next to its `REGISTRY.json` | imports changed to `analysis.feat_track.*`, output paths updated; imports and tests run |
| `analysis/FEAT_AUDIT.md` | `analysis/feat_track/AUDIT.md` | same | — |
| `analysis/FEAT_TRACK_HANDOFF.md` | `_archive/feat_track/` | its "blocking" questions were answered by the runs that followed | content folded into `analysis/feat_track/README.md` |
| `docs/AUTONOMOUS_RESEARCH.md` | `_archive/docs/` | 16-line stub index | replaced by `docs/EXPERIMENTS.md` and INVESTIGATION §26.1 |
| `__pycache__/` (root), `runs/` | `_archive/generated/` | generated; `runs/detect/predict*` were 14 empty folders | no files in `runs/` |
| `evaluation/results/handover_check.csv` (this check's rebuild) | `_archive/evaluation_results/duplicates_of_final_v5/` | duplicate | identical to `final_v5.csv` (`cmp`) |

Kept on purpose, unchanged: `mtah/`, `run.py`, `analysis/*.py`, `evaluation/`, `challenge/`,
`starter/`, `colab/`, `scripts/`, `work/labels/`, `cache/` (42 GB, including ~4.5 GB of half-built
`c_ballfree_cluster` caches that `feat_full_diverse.py` would reuse), `colab_upload/`, `models/`,
`external/`, `data/`, `analysis/audio_sources/pylibs/` (needed to rerun the audio scripts).

## 2. Documentation changes

- **New:** `HANDOVER.md`, `docs/EXPERIMENTS.md` (experiment index), `docs/CLEANUP_REPORT.md`,
  `baseline_snapshot/README.md`, `analysis/feat_track/README.md`; `analysis/audio_sources/README.md`
  rewritten.
- **README.md:** baseline vs best-verified table, links to the new docs, repo map for the research
  folders, never-commit list extended.
- **docs/ROADMAP.md:** doc table, "Where we stand" mentions §26, new open items R19–R23 with
  owners, owners on R9/R18, next steps → milestones; **corrected** R13 (it said "awaiting adoption";
  the gate ships in `final_v5`, `stop_gate=100`).
- **docs/INVESTIGATION.md:** new §26 (the three post-`final_v5` tracks, numbers copied from their
  registries/logs); **corrected** §7 (said `final_v1` is the submission) and W9 (said "not started").
- **baseline_snapshot/*.md, report.html, analysis/audio_sources/docs/*.md:** paths updated after the moves.
- **.gitignore:** `_archive/`, `analysis/**/*.csv`, audio `cache/`/`pylibs/`/`dist/`, `*.npz`,
  `*.joblib`, `*.tgz`, `*.zip`, `runs/`.

One duplication left on purpose: `baseline_snapshot/OPERATING_INTERFACE.md` repeats parts of the
README's "how to try an idea" in more depth; README stays the short version.

## 3. Validation report

All run on 2026-10-26 on the original Mac (M2 Pro, `.venv` Python 3.11).

| check | command | result |
|---|---|---|
| syntax | `python -m py_compile` on all 129 `.py` files in `run.py games.py mtah analysis baseline_snapshot evaluation starter scripts` (excl. `pylibs`) | all compiled |
| unit tests | `.venv/bin/python analysis/feat_track/test_feat_candidates.py` | PASS no-future invariance, PASS half reset |
| unit tests | `.venv/bin/python analysis/autonomous_research/experiments/sequence/test_semimarkov.py` | T1–T4 PASS |
| unit tests | `.venv/bin/python analysis/autonomous_research/experiments/holdout/test_class_order.py` | PASS |
| imports after the move | import `analysis.feat_track.{feat_candidates,feat_screen,feat_full}`, `mtah.final`, `mtah.bench` | OK; 5 candidates; output paths point into `analysis/feat_track/` |
| shell syntax | `bash -n analysis/feat_track/run_screen_when_free.sh` | OK |
| end-to-end rebuild | `.venv/bin/python run.py --name handover_check` | exit 0, 486 s (warm caches), sha256 `4089da237ca8780a6041bad8cc78261f29e9c5f5ef9fe8d3bdd9d9a8e2951c8a`, `cmp` identical to `final_v5.csv`; public 0.7226 / 0.9098 |
| validator | `.venv/bin/python evaluation/validate_submission.py evaluation/results/final_v5.csv` | OK, 199,319 rows |
| official evaluator | `.venv/bin/python evaluation/run_evaluation.py --no-detail` | `final_v5` #1: spellF1 0.722581, macroF1 0.909848; `baseline_smooth50` 0.325581 / 0.620733 |
| dev bench reconstruction | `.venv/bin/python baseline_snapshot/reconstruct_baseline.py` | 40 s; bench 0.6612, ext 0.6163, masked 0.7061, public 0.6968 / 0.9098, val_h2 0.5627 (identical to the locked numbers) |
| doc links | relative links in README, HANDOVER, ROADMAP, EXPERIMENTS, INVESTIGATION and the new READMEs | 85 checked, all resolve |

Not run (explicitly unverified):
- The audio scripts after the path change (only compiled). `extract_whistle_tonal.py` needs
  `pylibs/`; `panns_pilot.py` needs weights at `~/panns_data/` outside the repo.
- `feat_screen.py` / `feat_full.py` after the move (imports checked; a full run takes 10–50 min).
- A clean clone on another machine, and `uv sync` from scratch (HANDOVER P4-I1).
- Whether `UV_NATIVE_TLS=1` or `UV_SYSTEM_CERTS=1` is the right proxy variable for a given uv.

## 4. Unresolved items and uncertainties

1. **`cache/P_R1.npz` has no generating script in the repo**; ~10 analysis scripts and the
   baseline reconstruction need it (HANDOVER P4-I2).
2. **Nothing new is committed.** The four research folders and all doc changes are in the working
   tree only (HANDOVER P4-I3). `git add -n` shows 143 small text files (largest 64 KB), no CSV,
   npz, joblib, audio or video.
3. **Privacy review (R23): done 2026-10-27** — a team member read the committed research files and cleared them. Original note: Logs and console dumps under `analysis/autonomous_research/`
   and `analysis/audio_sources/logs/` were written by several agents. They were not read line by line
   here (to keep match-derived output out of an LLM); a person must check them before the repo goes
   public. `analysis/autonomous_research/audit/reproduced_public_scoreboard.csv` and
   `experiments/reproducibility/score_out/` are now gitignored by the CSV/report rules.
4. `analysis/autonomous_research/` keeps its own historical status files (`STATE.json`,
   `COVERAGE.md`, `CODE_CHANGES.md`); they are records of that cycle, not current status.
   Current status is ROADMAP.
5. `evaluation/scoreboard.csv` and `evaluation/reports/` were regenerated by the evaluator run
   (both gitignored).

## 5. Rules compliance check, 2026-11-17 (public repo at `final_v6`)

Checked against `challenge/RULES.md`, `SUBMISSION.md`, `FAQ.md`, `DATA.md` and `LICENSE`.

| rule | check | result |
|---|---|---|
| Labels come from an algorithm | `run.py` → `mtah/final.py` → `mtah/decode.py`; the only reader of the public GT on that path is `mtah/report.py`, after the labels are written; `work/labels/` is read only by `mtah/bench.py` and `analysis/` | OK |
| No private ground truth | none in the tree or in history; `.gitignore` blocks `*_possession_gt.csv` | OK |
| Repository regenerates the exact file | `run.py --name final_v6`, two full refits, sha256 `349a7df54c8e0364…` (INVESTIGATION §28) | OK on macOS; not re-run on a second machine |
| Pinned dependencies, install and run instructions | `pyproject.toml`, `uv.lock` (`uv lock --check` passes), README | OK |
| Everything used is accessible to the organisers | IDSSE (figshare, CC BY 4.0) and Metrica sample games (public GitHub) fetched by `scripts/download_external.sh`; no private model or dataset | OK |
| Technical document covers approach, assets, possession rules, limitations, reproduction | `Yamanalyst.md` §1–§7 | OK |
| Match data not redistributed | every path ever committed on every pushed branch (281 paths): no video, audio, tracking, npz, joblib; the only CSVs are the organiser's public GT and four baselines, byte-identical to the starter repository | OK |
| No derived data that would reconstruct the match | all tracked logs, console dumps, JSON, HTML and notebooks scanned for per-frame or per-event rows and embedded images | one finding, below |
| Organiser files unchanged | `evaluation/*.py`, public GT, baselines identical to upstream; `challenge/` is the organiser's documents moved, not edited | OK |
| Licence | MIT text and Metrica Sports copyright notice kept | OK |

Finding: `analysis/audio_sources/logs/whistle_diag.log` listed 206 detected whistle events with
frame ranges and scores. It cannot reconstruct tracking or video, but it is a frame-indexed extract
derived from the match audio, which this repository's own data rules exclude. The list was removed
(the threshold summary stays) and `whistle_diag.py` now prints aggregates only. One example frame
index with its model probabilities was also removed from
`analysis/autonomous_research/experiments/followup_temporal/artifacts/LOG.md`. Both remain in
earlier commits; purging them needs a history rewrite and a force-push, which was not done.

Not checkable from the repository: team size (3–5) and registration, and that no team member gave
match data to a hosted LLM (FAQ, Terms and Conditions).
