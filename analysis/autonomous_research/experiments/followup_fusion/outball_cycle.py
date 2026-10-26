"""Projected out-ball prior: when the ball is SEEN outside the pitch bounds (normalized
x<0/>1 or y<0/>1) AND the tactical homography reprojection error is small (<thresh m),
force a dead-ball (D) prior in the decoder emissions. This is a decoder-level prior on the
cached fold probabilities P_R1.npz; no refit, no manual labels.

Distinct from existing rules: rules already mark BALL OUT from events; this uses the geometric
projection of the seen ball itself, gated by homography confidence. External matches have no
real homography (hvalid=1 by construction), so this prior only applies on our match
(public + masked), not ext.

Adoption rule: bench delta >= +0.005 AND ext/masked not down >0.005.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final, kickoff  # noqa: E402
from mtah.data import Match  # noqa: E402

STATES = np.array(["H", "A", "D"])
_TEAM = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def viterbi_team(logE, cost_team, cost_dead):
    n = len(logE)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logE[0].copy()
    for t in range(1, n):
        T = -cost_dead * _DEAD
        T[0, 1] = T[1, 0] = -cost_team
        cand = score[:, None] + T
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def out_mask(thresh_reproj):
    m = Match(source="tactical")
    bx, by = m.ball[:, 0], m.ball[:, 1]
    out = np.isfinite(bx) & ((bx < 0.0) | (bx > 1.0) | (by < 0.0) | (by > 1.0))
    conf = out & (m.reproj < thresh_reproj)
    return m, conf


def decode_ar(c, Pd, w_out, thresh):
    p = dict(decode.PARAMS)
    p.update(final.DECODER)
    m, f = c.m, c.f
    P = Pd[c.name]
    R = np.full(m.n, "X", dtype="<U1")
    logE = decode.emissions(P, R, f["seen"], p)
    # projected out-ball prior: boost D on confident out-of-pitch frames
    if w_out and c.name.startswith(("public", "masked")):
        _, omask = out_mask(thresh)
        if len(omask) == m.n:
            logE[omask, 2] += w_out  # D index
    logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        out[a:b + 1] = STATES[viterbi_team(logE[a:b + 1], p["cost_team"], p["cost_dead"])]
    out = decode.min_duration(out, p["min_dead"], 0)
    if p["stop_gate"]:
        out = decode.stop_gate(out, f, int(p["stop_gate"]))
    return out


def main():
    y = Yard(pfile="P_R1.npz")
    Pd = y.P
    print("=== projected out-ball prior (our match only) ===")
    for w, thr in [(0.0, 0.5), (2.0, 0.5), (4.0, 0.5), (8.0, 0.5), (4.0, 1.0)]:
        out = y.report("out_w{}_t{}".format(w, thr), lambda c, _p, w=w, t=thr: decode_ar(c, Pd, w, t), quiet=True)
        print("  w={:<4} thr={}: bench {:.4f} ext {:.4f} masked {:.4f} | public sp {:.4f}/mf {:.4f} | val sp {:.4f}/mf {:.4f}".format(
            w, thr, out["bench"], out["ext"], out["masked"], out["public"], out["pubM"], out["val"], out["valM"]))


if __name__ == "__main__":
    main()
