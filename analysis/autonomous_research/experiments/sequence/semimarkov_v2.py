"""Corrected semi-Markov span decoder (regular endpoints, true mean arm).

Fixed generic duration priors (not fit to public). Two arms:
  sum  = span log-likelihood + log duration prior
  mean = true length-normalized mean logP + log duration prior
Boundaries: regular endpoints every 750 frames union argmax changes (connectivity guaranteed).
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from analysis.autonomous_research.experiments.sequence.test_semimarkov import (
    semimarkov, regular_boundaries)

FPS = 25
L_MAX = 30 * FPS
ALIVE_MEAN = 20.0 * FPS
DEAD_MEAN = 5.0 * FPS
STATES = np.array(["H", "A", "D"])


def make_dur_log(max_len, mode):
    dur = np.zeros((max_len + 1, 3))
    means = np.array([ALIVE_MEAN, ALIVE_MEAN, DEAD_MEAN])
    for s in range(3):
        L = np.arange(max_len + 1)
        dur[:, s] = -L / means[s] - np.log(means[s])
    return dur


def run_half(logP_half, cost_team, cost_dead, mode):
    cb = regular_boundaries(logP_half.argmax(1), L_MAX)
    T = np.zeros((3, 3))
    T[0, 1] = T[1, 0] = -cost_team
    T[0, 2] = T[2, 0] = T[1, 2] = T[2, 1] = -cost_dead
    dur = make_dur_log(len(logP_half), mode)
    # true mean arm: pass mode via closure not needed; semimarkov uses 'sum'.
    # For mean arm, override by scaling emissions to per-frame mean.
    if mode == "mean":
        # true mean: divide span sum by length inside semimarkov; we re-implement here
        return _semimarkov_mean(logP_half, cb, dur, T)
    return semimarkov(logP_half, cb, dur, T, max_len=L_MAX)


def _semimarkov_mean(logP, cb, dur, T):
    n = len(logP)
    K = len(cb)
    NEG = -1e9
    dp = np.full((K, 3), NEG)
    bp = np.zeros((K, 3, 2), dtype=np.int32)
    csum = np.vstack([np.zeros((1, 3)), np.cumsum(logP, axis=0)])
    dp[0] = 0.0
    for i in range(1, K):
        ce = cb[i]
        for j in range(i - 1, -1, -1):
            cs = cb[j]
            length = ce - cs
            if length > L_MAX:
                break
            for s in range(3):
                sc = (csum[ce, s] - csum[cs, s]) / length + dur[length, s]
                prev = dp[j] + T[:, s]
                b = int(prev.argmax())
                cand = prev[b] + sc
                if cand > dp[i, s]:
                    dp[i, s] = cand
                    bp[i, s] = (j, b)
    if dp[K - 1].max() <= NEG:
        raise RuntimeError("terminal unreachable")
    out = np.zeros(n, dtype=np.int8)
    i = K - 1
    s = int(dp[i].argmax())
    while i > 0:
        j, ps = bp[i, s]
        out[cb[j]:cb[i]] = s
        i, s = j, ps
    return out


def main():
    from analysis.yardsticks import Yard
    from mtah import decode, final
    t0 = time.time()
    y = Yard()
    BASE = dict(decode.PARAMS, **dict(final.DECODER, use_rules=False))

    def baseline(c, P):
        nox = np.full(c.m.n, "X", dtype="<U1")
        return decode.decode(c.m, nox, P, c.f["seen"], BASE, c.f)

    y.report("baseline v2g", baseline)

    def make(mode):
        def fn(c, P):
            nox = np.full(c.m.n, "X", dtype="<U1")
            out = nox.copy()
            logP = np.log(np.clip(P, 1e-9, 1.0))
            for _, a, b in c.m.halves:
                idx = run_half(logP[a:b + 1], BASE["cost_team"], BASE["cost_dead"], mode)
                out[a:b + 1] = STATES[idx]
            from mtah.decode import min_duration, stop_gate
            out = min_duration(out, BASE["min_dead"], 0)
            if BASE["stop_gate"]:
                out = stop_gate(out, c.f, BASE["stop_gate"])
            return out
        return fn

    y.report("semimarkov SUM fixed dur", make("sum"))
    y.report("semimarkov MEAN true normalized", make("mean"))
    print("elapsed {:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
