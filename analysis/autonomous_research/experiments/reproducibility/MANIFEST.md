# Reproduction manifest — final_v5 (ar_verified)

Date: 2026-10-25. Reproducibility track. Production untouched; original `final_v5.csv`
and `run.py` unchanged. Two independent regenerations with the venv interpreter.

## Result

| file | sha256 (full) | rows | ordered | labels | public spellF1 | public macroF1 |
|---|---|---|---|---|---|---|
| final_v5.csv (original) | `4089da237ca8780a6041bad8cc78261f29e9c5f5ef9fe8d3bdd9d9a8e2951c8a` | 199319 | 0..199318 | H/A/D/X | 0.7226 | 0.9098 |
| ar_verified_v5_a.csv | `4089da237ca8780a6041bad8cc78261f29e9c5f5ef9fe8d3bdd9d9a8e2951c8a` | 199319 | 0..199318 | H/A/D/X | 0.7226 | 0.9098 |
| ar_verified_v5_b.csv | `4089da237ca8780a6041bad8cc78261f29e9c5f5ef9fe8d3bdd9d9a8e2951c8a` | 199319 | 0..199318 | H/A/D/X | 0.7226 | 0.9098 |

Byte-for-byte identical to the committed original. Determinism confirmed across two runs.

## Label distribution (all three identical)
H 52668 (26.4%) · A 33158 (16.6%) · D 73031 (36.6%) · X 40462 (20.3%).
X = the 40,462 ungraded frames (pre-match 11185 + halftime 20156 + post-whistle 9121),
correctly abstained. Over the 158,857 graded frames: H 33.2% / A 20.9% / D 46.0%.
269 possessions, mean 12.8 s.

## Commands
```
.venv/bin/python run.py --name ar_verified_v5_a     # real 11m37s
.venv/bin/python run.py --name ar_verified_v5_b     # real  9m55s
.venv/bin/python evaluation/run_evaluation.py \
    --results <score_in> --out <score_out> --gt public --bootstrap 0 --no-detail
.venv/bin/python evaluation/validate_submission.py evaluation/results/<name>.csv
```
Interpreter: `.venv/bin/python` (CPython 3.11.15). No `uv` used, so `uv.lock` was not
mutated. Caches reused (24 GB in `cache/`); no newfit required.

## Timing
- Run A: 2026-10-25 19:26:56 -> 19:38:33 (11m37s wall; user 52m42s CPU).
- Run B: first attempt killed by a shell cleanup deadline (exit -1, empty log, no file) —
  diagnosed as a harness cleanup timeout, not a training/cache failure; re-ran with direct
  file redirect and it completed 19:48:26 -> 19:58:21 (9m55s wall; user 52m06s CPU).

## Config (frozen final_v5)
- featfn: `mtah.feats.full_plus3`; model: HistGBDT 600 trees on 9 external matches with
  true labels, degraded to our tracker, plus left/right + home/away mirrors and extra-noise
  copies (`mtah/final.py`).
- decoder: `w_shape 0.75, cost_dead 12, cost_team 12, min_dead 25, kickoff=half,
  use_rules=False, stop_gate=100, w_bound=0`.

## Checks performed
- sha256 of all three files equal the original (above).
- csv schema: header `frame,label`; exactly 199,319 rows; frame column 0..199318 in
  order; every label in {H,A,D,X}; no blank cells.
- official `validate_submission.py`: OK on all three.
- isolated official scorer against public GT: all three rank 1..3 with identical
  macroF1 0.9098 / spellF1 0.7226, 14 predicted possessions (true 17), mean 18.3 s.

## Artifacts (this dir)
- `run_a.log`, `run_b.log` — full run.py output.
- `score_console.txt`, `score_out/scoreboard.csv`, `score_out/reports/` — isolated scorer.
- `MANIFEST.md` — this file.

## Not done
- No new submission default created; no push/publish. Original `final_v5.csv`, `run.py`,
  `mtah/` untouched. Failed decoder variants were NOT promoted.
