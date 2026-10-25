"""Segmental MBR decoding (mtah/mbr.py) against Viterbi on every yardstick (aggregates only).

    uv run python analysis/mbr_eval.py

Candidates: the full final_v2 + stop-gate decode at switch-cost scales CANDS. Samples: K paths
from the posterior of the base decoder's chain (same emissions and costs), at temperature TEMP.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final, mbr  # noqa: E402

BASE = dict(decode.PARAMS, **dict(final.DECODER, use_rules=False))
IDX = {"H": 0, "A": 1, "D": 2, "X": 2}


def make(cands=(0.5, 0.7, 1.0, 1.4, 2.0), k=24, temp=1.0, f1_ref=0.65, seed=0):
    def fn(c, P):
        nox = np.full(c.m.n, "X", dtype="<U1")
        labs = [decode.decode(c.m, nox, P, c.f["seen"],
                              dict(BASE, cost_team=BASE["cost_team"] * s,
                                   cost_dead=BASE["cost_dead"] * s), c.f) for s in cands]
        C = np.stack([np.vectorize(IDX.get)(l) for l in labs]).astype(np.int8)
        logE = decode.emissions(P, nox, c.f["seen"], BASE)
        if BASE.get("kickoff"):
            from mtah import kickoff
            logE = logE + kickoff.evidence(c.m, c.f, mode=BASE["kickoff"])
        T = np.zeros((3, 3))
        T[0, 1] = T[1, 0] = -BASE["cost_team"]
        T[0, 2] = T[2, 0] = T[1, 2] = T[2, 1] = -BASE["cost_dead"]
        out = labs[list(cands).index(1.0)].copy()
        rng = np.random.default_rng(seed)
        for _, a, b in c.m.halves:
            S = mbr.ffbs(logE[a:b + 1], T, k, rng, temp)
            ch = mbr.choose(C[:, a:b + 1], S, f1_ref)
            out[a:b + 1] = decode.STATES[ch]
        return out
    return fn


if __name__ == "__main__":
    y = Yard()
    nox = lambda c: np.full(c.m.n, "X", dtype="<U1")  # noqa: E731
    y.report("v2g viterbi", lambda c, P: decode.decode(c.m, nox(c), P, c.f["seen"], BASE, c.f))
    for temp in (1.0, 2.0):
        y.report("MBR temp {}".format(temp), make(temp=temp))
