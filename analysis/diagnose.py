"""Segment-by-segment comparison of a prediction against the PUBLIC ground truth.

    uv run python analysis/diagnose.py [rules|lgbm|...]

Prints the GT segments of the public window (public data) next to what the prediction says over
the same frames, so systematic failure modes are visible. Times are seconds from kickoff.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.score import load_gt, pm  # noqa: E402

W0, W1 = 11185, 18685


def segs(lab, a, b):
    out = []
    i = a
    while i < b:
        j = i
        while j < b and lab[j] == lab[i]:
            j += 1
        out.append((i, j, lab[i]))
        i = j
    return out


def show(pred):
    gt = np.array(pm.LABELS)[load_gt()]
    pred = np.asarray(pred)
    for a, b, c in segs(gt, W0, W1):
        inner = segs(pred, a, b)
        desc = " ".join("{}{:.1f}".format(k, (j - i) / 25) for i, j, k in inner)
        print("{:6.1f}-{:6.1f}s  GT {} {:5.1f}s | pred: {}".format(
            (a - W0) / 25, (b - W0) / 25, c, (b - a) / 25, desc))


if __name__ == "__main__":
    from mtah.data import Match
    from mtah.features import compute
    from mtah import rules
    m = Match()
    show(rules.run(m, compute(m)))
