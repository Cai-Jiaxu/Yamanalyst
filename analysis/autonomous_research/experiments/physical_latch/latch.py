"""Physical possession latch: emit H/A bias based on ball-nearest-player control.

For each frame with ball seen, compute whether a team has physical control:
  - distance to nearest home player dh < 2.0 m
  - distance to nearest away player da < 2.0 m
Require 3 consecutive frames (0.12s at 25fps) of control evidence before boosting
the team's emission. When ball is missing, no boost (latch via existing viterbi transition
cost). D emission unchanged.

No new models trained. No public-tuned thresholds. Thresholds from physics:
  2.0 m control radius, 3-frame persistence, bias +1.0 log-units.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final, kickoff  # noqa: E402

STATES = np.array(["H", "A", "D"])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

# Physical thresholds (frozen, from physics reasoning)
CONTROL_DIST_M = 2.0   # meters: ball within this of a nearest player = control
PERSIST_FRAMES = 3     # consecutive frames required for control evidence
EMIT_BIAS = 1.0        # log-emission boost for controlling team


def control_signal(f):
    """Return (h_ctrl, a_ctrl) boolean arrays after persistence filter."""
    seen = f["seen"].astype(bool)
    dh = f["dh"]
    da = f["da"]
    # raw control: ball seen AND close to a player
    h_raw = seen & (dh < CONTROL_DIST_M)
    a_raw = seen & (da < CONTROL_DIST_M)
    # persistence: require N consecutive frames
    def persist(x, n=PERSIST_FRAMES):
        out = np.zeros_like(x)
        run = 0
        for i in range(len(x)):
            if x[i]:
                run += 1
            else:
                run = 0
            out[i] = run >= n
        return out
    return persist(h_raw), persist(a_raw), h_raw, a_raw


def viterbi_phys(logE, cost_team, cost_dead, h_ctrl, a_ctrl, bias):
    n = len(logE)
    # apply emission bias
    le = logE.copy()
    le[h_ctrl, 0] += bias  # H
    le[a_ctrl, 1] += bias   # A
    back = np.zeros((n, 3), dtype=np.int8)
    score = le[0].copy()
    for t in range(1, n):
        T = -cost_dead * _DEAD
        T[0, 1] = T[1, 0] = -cost_team
        cand = score[:, None] + T
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + le[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_phys(c, P, bias):
    p = dict(decode.PARAMS)
    p.update(final.DECODER)
    m, f = c.m, c.f
    Pd = P[c.name]
    R = np.full(m.n, "X", dtype="<U1")
    logE = decode.emissions(Pd, R, f["seen"], p)
    logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
    h_ctrl, a_ctrl, h_raw, a_raw = control_signal(f)
    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        out[a:b + 1] = STATES[viterbi_phys(
            logE[a:b + 1], p["cost_team"], p["cost_dead"],
            h_ctrl[a:b + 1], a_ctrl[a:b + 1], bias)]
    out = decode.min_duration(out, p["min_dead"], p["min_poss"])
    if p["stop_gate"]:
        out = decode.stop_gate(out, f, int(p["stop_gate"]))
    return out, h_ctrl, a_ctrl, h_raw, a_raw


def main():
    y = Yard(pfile="P_R1.npz")
    P = y.P

    # identity / no-op test: bias=0 should reproduce frozen v3 exactly
    print("=== bias=0 identity test ===", flush=True)
    def fn0(c, _p):
        out, *_ = decode_phys(c, P, 0.0)
        return out
    base = y.report("phys_bias0", fn0)
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        base["bench"], base["ext"], base["masked"], base["public"], base["pubM"], base["val"]))

    # bias = EMIT_BIAS
    print("\n=== bias={} physical latch ===".format(EMIT_BIAS), flush=True)
    # gather diagnostics on public case
    cpub = y.b.public
    out_pub, hc, ac, hr, ar = decode_phys(cpub, P, EMIT_BIAS)
    print("  public: h_ctrl frames={} a_ctrl frames={}".format(hc.sum(), ac.sum()))
    print("  public: h_raw frames={} a_raw frames={}".format(hr.sum(), ar.sum()))

    def fn1(c, _p):
        out, *_ = decode_phys(c, P, EMIT_BIAS)
        return out
    res = y.report("phys_bias{}".format(EMIT_BIAS), fn1)
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        res["bench"], res["ext"], res["masked"], res["public"], res["pubM"], res["val"]))
    print("  delta: bench {:+.4f} ext {:+.4f} masked {:+.4f} public {:+.4f} val {:+.4f}".format(
        res["bench"] - base["bench"], res["ext"] - base["ext"],
        res["masked"] - base["masked"], res["public"] - base["public"],
        res["val"] - base["val"]))


if __name__ == "__main__":
    main()
