"""Learned temporal refiner (mtah/refiner.py) on cached fold probabilities, all yardsticks.

    uv run python analysis/refine_eval.py

Ext: each fold's matches are refined by a GRU trained on the other two folds' out-of-fold
probabilities (never on its own labels). Our match (public, masked, val): a GRU trained on all 9.
Aggregates only.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final, refiner  # noqa: E402
from mtah.bench import FOLDS  # noqa: E402

NOX = None


def Zc(y, c, key):
    return refiner.inputs(y.P[key], y.P.get("B:" + key), c.f, c.R)


def refined_P(y, seeds=(0,), **kw):
    t = time.time()
    ext = y.b.ext
    seqs = {k: (Zc(y, c, k), c.gt.astype(np.int64)) for k, c in ext.items()}
    for k, (Z, g) in seqs.items():
        g = g.copy()
        g[g > 2] = -1          # X (and anything else) ungraded
        seqs[k] = (Z, g)
    Q = {}
    for fold in FOLDS:
        train = [seqs[k] for k in ext if k not in fold]
        nets = [refiner.fit(train, seed=s, **kw) for s in seeds]
        for k in fold:
            Q[k] = np.mean([refiner.predict(n, seqs[k][0]) for n in nets], 0)
    nets = [refiner.fit(list(seqs.values()), seed=s, **kw) for s in seeds]
    for c in [y.b.public] + y.b.masked:
        Z = Zc(y, c, c.name)
        Q[c.name] = np.mean([refiner.predict(n, Z) for n in nets], 0)
    print("refiner fitted in {:.0f} s".format(time.time() - t), flush=True)
    return Q


def dec(mix=1.0, **kw):
    """Decode log(mix * Q + ...) : w_shape applies to the refined probabilities."""
    p = dict(decode.PARAMS, **dict(final.DECODER, use_rules=False, **kw))

    def fn(c, P):
        return decode.decode(c.m, np.full(c.m.n, "X", dtype="<U1"), P, c.f["seen"], p, c.f)
    return fn


def geo(P, Q, a):
    """Geometric blend P^(1-a) Q^a, renormalised."""
    L = (1 - a) * np.log(np.clip(P, 1e-4, 1)) + a * np.log(np.clip(Q, 1e-4, 1))
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


if __name__ == "__main__":
    y = Yard()
    y.report("v2g baseline", dec())
    Q = refined_P(y, seeds=(0, 1, 2))
    for a in (0.5, 1.0):
        PQ = {k: geo(y.P[k], Q[k], a) for k in Q}
        for cs in (1.0, 0.5):
            y.report("refined a={} cost x{}".format(a, cs),
                     dec(cost_team=12.0 * cs, cost_dead=12.0 * cs), P=PQ)
