"""Pruned semi-Markov (explicit-duration) span decoder on cached P_R1.

Convention: no public-GT-derived density. Duration priors are fixed generic soccer values
(alive mean 20 s, dead mean 5 s). No broad sweep. Two narrow arms:
  SUM  = proper generative span log-likelihood (favors longer spans by evidence count)
  MEAN = length-normalized span score (removes the length bias)
Candidates are argmax run boundaries within each half, pruned to spans <= L_MAX frames.

Run:
  .venv/bin/python analysis/autonomous_research/experiments/sequence/semimarkov_span.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

# --- fixed generic priors (NOT fit to any labels) ---------------------------
FPS = 25
L_MAX = 30 * FPS          # 30 s max span, frames
ALIVE_MEAN = 20.0 * FPS  # generic expected possession length
DEAD_MEAN = 5.0 * FPS    # generic expected stoppage length
MIN_RUN = 5              # merge argmax runs shorter than this (prune)
# -----------------------------------------------------------------------------
STATES = np.array(["H", "A", "D"])


def candidate_boundaries(arg, half):
    """Sorted candidate start indices inside a half [a,b): argmax change points."""
    a, b = half
    x = arg[a:b + 1]
    ch = np.flatnonzero(x[1:] != x[:-1]) + 1
    # merge tiny runs: keep boundaries at changes, plus half start
    cb = np.concatenate(([0], ch)) + a
    # prune runs shorter than MIN_RUN
    keep = [cb[0]]
    prev = cb[0]
    for t in cb[1:]:
        if t - prev >= MIN_RUN:
            keep.append(t)
            prev = t
    keep.append(b + 1)  # half end
    return np.array(keep, dtype=int)


def span_score(cs, ce, s, logP, dur_mean, mode):
    """cs..ce exclusive frames, label s. logP (n,3)."""
    seg = logP[cs:ce, s]
    length = ce - cs
    if mode == "sum":
        ll = seg.sum()
    elif mode == "mean":
        ll = seg.mean() * length  # length-normalized then back-scaled; same argmax basis
    else:
        raise ValueError(mode)
    dur = -length / dur_mean - np.log(dur_mean)
    return ll + dur


def semimarkov_decode(logP, half, cost_team=12.0, cost_dead=12.0, mode="sum"):
    a, b = half
    n = b - a + 1
    cb = candidate_boundaries(logP[a:b + 1].argmax(1), (a, b))
    # work in local frames 0..n
    cb_local = cb - a
    dur_means = np.array([ALIVE_MEAN, ALIVE_MEAN, DEAD_MEAN])  # H, A, D
    K = len(cb_local)
    NEG = -1e9
    dp = np.full((K, 3), NEG)
    bp = np.zeros((K, 3, 2), dtype=np.int32)  # backpointer: (prev boundary idx, prev state)
    # first segment starts at cb_local[0]
    L = logP[a:b + 1]
    for s in range(3):
        dp[1, s] = span_score(cb_local[0], cb_local[1], s, L, dur_means[s], mode)
    # transition cost matrix
    T = np.zeros((3, 3))
    T[0, 1] = T[1, 0] = -cost_team
    T[0, 2] = T[2, 0] = T[1, 2] = T[2, 1] = -cost_dead
    for i in range(2, K):
        ce = cb_local[i]
        for j in range(max(1, i - 1), -1, -1):
            cs = cb_local[j]
            if ce - cs > L_MAX:
                break
            for s in range(3):
                sc = span_score(cs, ce, s, L, dur_means[s], mode)
                # best prev state at boundary j
                prev = dp[j] + T[:, s]
                best_prev = int(prev.argmax())
                cand = prev[best_prev] + sc
                if cand > dp[i, s]:
                    dp[i, s] = cand
                    bp[i, s] = (j, best_prev)
    # backtrack from last boundary
    out_local = np.zeros(n, dtype=np.int8)
    i = K - 1
    s = int(dp[i].argmax())
    while i > 1:
        j, ps = bp[i, s]
        out_local[cb_local[j]:cb_local[i]] = s
        i, s = j, ps
    out_local[cb_local[0]:cb_local[1]] = s
    out = np.empty(logP.shape[0], dtype="<U1")
    out[a:b + 1] = STATES[out_local]
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
                seg = semimarkov_decode(logP, (a, b), BASE["cost_team"], BASE["cost_dead"], mode)
                out[a:b + 1] = seg[a:b + 1]
            from mtah.decode import min_duration, stop_gate
            out = min_duration(out, BASE["min_dead"], 0)
            if BASE["stop_gate"]:
                out = stop_gate(out, c.f, BASE["stop_gate"])
            return out
        return fn

    y.report("semimarkov SUM fixed dur", make("sum"))
    y.report("semimarkov MEAN fixed dur", make("mean"))

    # registry
    here = os.path.dirname(os.path.abspath(__file__))
    reg = {
        "date": "2026-10-25",
        "script": os.path.relpath(__file__, ROOT),
        "fixed_priors": {
            "L_MAX_frames": L_MAX, "alive_mean_frames": ALIVE_MEAN,
            "dead_mean_frames": DEAD_MEAN, "min_run_frames": MIN_RUN,
            "cost_team": BASE["cost_team"], "cost_dead": BASE["cost_dead"],
        },
        "geometric_hmm_tests": [
            {"hazard": 1 / 25, "bench": 0.628, "ext": 0.602, "masked": 0.654},
            {"hazard": 1 / 50, "bench": 0.632, "ext": 0.601, "masked": 0.663},
            {"hazard": 1 / 100, "bench": 0.631, "ext": 0.599, "masked": 0.663},
            {"hazard": 1 / 200, "bench": 0.629, "ext": 0.597, "masked": 0.662},
            {"hazard": 1 / 400, "bench": 0.633, "ext": 0.595, "masked": 0.671},
        ],
        "note": "geometric HMM (not true BOCPD); semi-Markov arms use fixed generic duration priors",
    }
    with open(os.path.join(here, "registry.json"), "w") as f:
        json.dump(reg, f, indent=2)
    print("registry written; elapsed {:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
