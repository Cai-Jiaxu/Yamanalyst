#!/usr/bin/env bash
# Every check that needs no match data: compile all tracked Python, then the synthetic tests.
# Usage: bash scripts/check.sh            (uses `uv run python`; set PYTHON=... to override)
# With data and caches present, also run:  uv run python baseline_snapshot/reconstruct_baseline.py
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-"uv run python"}
export PYTHONDONTWRITEBYTECODE=1

$PY - <<'PYEOF'
import os, py_compile, subprocess, sys, tempfile
files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard",
                                 "*.py"], text=True).split()
tmp = os.path.join(tempfile.gettempdir(), "mtah_check.pyc")
bad = []
for f in files:
    try:
        py_compile.compile(f, cfile=tmp, doraise=True)
    except py_compile.PyCompileError as e:
        bad.append((f, e))
print("compiled {} Python files, {} failures".format(len(files), len(bad)))
for f, e in bad:
    print("  FAIL", f, e)
sys.exit(1 if bad else 0)
PYEOF

$PY -m unittest discover -s tests -p "test_*.py"
$PY -m unittest baseline_snapshot.test_rebuild_probabilities
$PY analysis/feat_track/test_feat_candidates.py
$PY analysis/autonomous_research/experiments/sequence/test_semimarkov.py
$PY analysis/autonomous_research/experiments/holdout/test_class_order.py
echo "ALL CHECKS PASSED"
