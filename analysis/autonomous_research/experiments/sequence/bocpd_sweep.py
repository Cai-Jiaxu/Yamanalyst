"""Hazard sweep for the convention-neutral geometric HMM smoother (cached P_R1).

Predeclared hazard grid (expected run = 1/h): 25, 50, 100, 200 frames.
Reports bocpd->viterbi (the less-bad variant) plus argmax at one point.
No public-GT fitting; this is a fixed-prior sensitivity scan.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

STATE_PRIOR = np.ones(3) / 3


def bocpd_smooth(P, hz, prior=STATE_PRIOR):
    n = P.shape[0]
    eps = 1e-9
    logP = np.log(np.clip(P, eps, 1.0))
    lprior = np.log(np.clip(prior, eps, 1.0))
    l1mh = np.log1p(-hz)
    lh = np.log(hz)
    q = np.empty((n, 3))
    q[0] = logP[0] - np.logaddexp.reduce(logP[0])
    for t in range(1, n):
        cont = l1mh + q[t - 1] + logP[t]
        new = lh + lprior + logP[t]
        s = np.logaddexp(cont, new)
        q[t] = s - np.logaddexp.reduce(s)
    return np.exp(q)


def main():
    from analysis.yardsticks import Yard
    from mtah import decode, final
    t0 = time.time()
    y = Yard()
    BASE = dict(decode.PARAMS, **dict(final.DECODER, use_rules=False))

    def viterbi_with_q(c, P, q):
        nox = np.full(c.m.n, "X", dtype="<U1")
        return decode.decode(c.m, nox, q, c.f["seen"], BASE, c.f)

    # baseline once
    def baseline(c, P):
        nox = np.full(c.m.n, "X", dtype="<U1")
        return decode.decode(c.m, nox, P, c.f["seen"], BASE, c.f)
    y.report("baseline v2g", baseline)

    for exp in (25, 50, 100, 200):
        hz = 1.0 / exp

        def mk(h):
            def fn(c, P):
                q = bocpd_smooth(P, h)
                return viterbi_with_q(c, P, q)
            return fn
        y.report("bocpd->vit h=1/{}".format(exp), mk(hz))

    print("elapsed {:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
