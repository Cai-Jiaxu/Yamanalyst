"""One-at-a-time parameter sweep of the rule system: is each default on a plateau or a peak?

    uv run python analysis/sensitivity.py

Public spellF1 moves ~0.06 per possession here, so only look for cliffs and broad trends.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah import rules  # noqa: E402
from mtah.report import spells  # noqa: E402
from mtah.score import load_gt, score  # noqa: E402

GRID = {
    "r_touch": [1.0, 1.25, 1.5, 2.0, 2.5],
    "excl_margin": [0.0, 0.25, 0.5, 1.0],
    "hold_frames": [12, 18, 25, 38, 50],
    "retouch_s": [2.0, 3.0, 4.0, 6.0],
    "walk_speed": [1.0, 1.15, 1.25, 1.4, 1.6],
    "walk_frames": [50, 75, 100, 150, 200],
    "restart_speed": [2.0, 3.0, 4.0, 5.0],
    "still_frames": [25, 38, 50, 75],
}

m = Match(); f = compute(m); gt = load_gt()
runs = [(a, b + 1) for _, a, b in m.halves]
minutes = m.graded_mask().sum() / 25 / 60
for k, vals in GRID.items():
    cells = []
    for v in vals:
        p = dict(rules.PARAMS); p[k] = v
        lab = rules.run(m, f, p)
        s, _ = score(lab, gt)
        cells.append("{}:{:.3f}/{:.2f}pm".format(v, s["spellF1"], len(spells(lab, runs)) / minutes))
    print("{:<14} {}".format(k, "  ".join(cells)))
