"""Refiner, round 2: dilated TCN, and the train/test input mismatch removed.

    uv run python analysis/refine_eval2.py

Round 1 (analysis/refine_eval.py) trained on single fold-model out-of-fold probabilities but
refined our match's 3-model AVERAGE, which is smoother. Here our match is refined once per fold
model (cache/P_R1_folds.npz, analysis/fold_probs.py) and the three refined outputs averaged.
Boundary-head input dropped (not available per fold). Aggregates only.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.refine_eval import dec, geo  # noqa: E402
from analysis.yardsticks import Yard  # noqa: E402
from mtah import refiner  # noqa: E402
from mtah.bench import FOLDS  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402


def run(y, PF, seeds=(0, 1, 2), **kw):
    t = time.time()
    seqs = {}
    for k, c in y.b.ext.items():
        g = c.gt.astype(np.int64).copy()
        g[g > 2] = -1
        seqs[k] = (refiner.inputs(y.P[k], None, c.f, c.R), g)
    Qavg, Qfold = {}, {}
    for fold in FOLDS:
        nets = [refiner.fit([seqs[k] for k in seqs if k not in fold], seed=s, **kw) for s in seeds]
        for k in fold:
            Qavg[k] = Qfold[k] = np.mean([refiner.predict(n, seqs[k][0]) for n in nets], 0)
    nets = [refiner.fit(list(seqs.values()), seed=s, **kw) for s in seeds]
    for c in [y.b.public] + y.b.masked:
        Qavg[c.name] = np.mean([refiner.predict(n, refiner.inputs(y.P[c.name], None, c.f, c.R))
                                for n in nets], 0)
        Qfold[c.name] = np.mean([refiner.predict(n, refiner.inputs(PF["{}#{}".format(c.name, i)],
                                                                    None, c.f, c.R))
                                 for n in nets for i in range(3)], 0)
    print("refiner {} fitted in {:.0f} s".format(kw, time.time() - t), flush=True)
    return Qavg, Qfold


if __name__ == "__main__":
    y = Yard()
    PF = dict(np.load(os.path.join(CACHE_DIR, "P_R1_folds.npz")))
    y.report("v2g baseline", dec())
    Qa, Qf = run(y, PF, arch="tcn", epochs=20)
    for tag, Q in (("avg-in", Qa), ("per-fold", Qf)):
        for a in (0.5, 1.0):
            PQ = {k: geo(y.P[k], Q[k], a) for k in Q}
            for cs in (1.0, 0.5):
                y.report("tcn {} a={} cost x{}".format(tag, a, cs),
                         dec(cost_team=12.0 * cs, cost_dead=12.0 * cs), P=PQ)
