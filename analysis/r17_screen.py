"""ROADMAP R17: touch-sequence features for short possessions, fast screen (rule 7: 9 base
matches, no augmentation, 300 trees) against the same-tier full_plus3 baseline, v2g decoder.

    uv run python analysis/r17_screen.py

Prints every yardstick plus ext recall at IoU 0.5 by true duration (<2 | 2-5 | 5-10 | >10 s) and
spurious spells per 100 true. Aggregates only.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.miss_by_len import P2, rec  # noqa: E402
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, feats  # noqa: E402
from mtah.bench import SCREEN, taxonomy  # noqa: E402
from mtah.score import pm  # noqa: E402

BINS = [0, 50, 125, 250, 10 ** 9]


def v2g(c, P):
    return decode.decode(c.m, np.full(c.m.n, "X", dtype="<U1"), P, c.f["seen"], P2, c.f)


def by_len(y, P):
    L, H, sp, ngt = [], [], 0, 0
    for k, c in y.b.ext.items():
        codes = pm.label_codes(v2g(c, P[k]))
        l, h, _ = rec(c.gt, codes)
        L.append(l), H.append(h)
        t = taxonomy(c.gt, codes)
        sp += t["spurious"]
        ngt += t["n_gt"]
    L, H = np.concatenate(L), np.concatenate(H)
    r = [H[(L >= a) & (L < b)].mean() for a, b in zip(BINS[:-1], BINS[1:])]
    return "ext recall by length " + " ".join("{:.3f}".format(v) for v in r) + \
        " | spurious/100 {:.1f}".format(100 * sp / ngt)


if __name__ == "__main__":
    y = Yard()
    for name in ("full_plus3", "full_plus5"):
        P = y.b.probs(getattr(feats, name), dict(SCREEN), key="screen_" + name)
        y.report("screen " + name, v2g, P=P)
        print("  " + by_len(y, P), flush=True)
