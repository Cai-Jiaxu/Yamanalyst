"""Rules vs rules+shape fusion, on the public window with and without the ball hidden.

    uv run python analysis/fusion.py
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match, CACHE_DIR  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah import rules, decode  # noqa: E402
from mtah.masking import masked_scores  # noqa: E402
from mtah.report import report  # noqa: E402

m = Match()
P = np.load(os.path.join(CACHE_DIR, "shape_proba.npy"))


def fused(p):
    return lambda m, f: decode.decode(m, rules.run(m, f), P, f["seen"], p)


f = compute(m)
report("rules", rules.run(m, f), m)
mean, _ = masked_scores(lambda m, f: rules.run(m, f), m, compute)
print("   masked spellF1 {:.3f} macroF1 {:.3f}".format(*mean))
for ct in (4, 8, 16, 32):
    for cd in (3, 6, 12):
        p = dict(decode.PARAMS, cost_team=ct, cost_dead=cd)
        report("fused team{} dead{}".format(ct, cd), fused(p)(m, f), m)
        mean, _ = masked_scores(fused(p), m, compute)
        print("   masked spellF1 {:.3f} macroF1 {:.3f}".format(*mean))
