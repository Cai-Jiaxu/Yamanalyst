"""One-at-a-time sweep of the rule parameters, scored against external TRUE labels on degraded
matches (rules alone, no model), next to our public window.

    uv run python analysis/rules_external.py
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402
from mtah.external import ExtMatch, available  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.noise import degrade, seed_for  # noqa: E402
from mtah import rules  # noqa: E402
from mtah.score import pm, load_gt, score  # noqa: E402

GRID = {
    "r_touch": [1.0, 1.5, 2.0, 2.5],
    "excl_margin": [0.0, 0.5, 1.0],
    "hold_frames": [12, 25, 50],
    "retouch_s": [2.0, 4.0, 6.0],
    "walk_speed": [1.0, 1.25, 1.5],
    "walk_frames": [50, 100, 200],
    "still_frames": [25, 38, 75],
}
cases = []
for name in available():
    em = ExtMatch(name)
    noisy = degrade(em, seed=seed_for(name))
    cases.append((name.split("_")[0], noisy, compute(noisy), pm.label_codes(em.labels)))
m = Match(); f = compute(m); gt = load_gt()


def evaluate(p):
    out = {"idsse": [], "metrica": []}
    for src, noisy, fn, g in cases:
        r = pm.evaluate(g, pm.label_codes(rules.run(noisy, fn, p)))
        out[src].append((r["spell_f1_mean"], r["n_pred_spells"] / r["n_gt_spells"]))
    pub, _ = score(rules.run(m, f, p), gt)
    return {k: np.mean(v, axis=0) for k, v in out.items()}, pub["spellF1"]


base, pub = evaluate(rules.PARAMS)
print("defaults: idsse {:.3f} (count x{:.2f})  metrica {:.3f} (x{:.2f})  public {:.3f}".format(
    *base["idsse"], *base["metrica"], pub))
for k, vals in GRID.items():
    cells = []
    for v in vals:
        r, pub = evaluate(dict(rules.PARAMS, **{k: v}))
        cells.append("{}: i{:.3f} m{:.3f} p{:.3f}".format(v, r["idsse"][0], r["metrica"][0], pub))
    print("{:<13} {}".format(k, " | ".join(cells)), flush=True)
