"""Geometry-conditioned decoder experiments (decoder track).

Reuses cache/P_R1.npz fold probabilities (no refit, no manual labels). Mirrors
mtah.decode.decode exactly except the H<->A transition cost may vary per frame.

E3 hypothesis (sports review): when the visible ball has at least one home AND one
away player within ~2 m (a contested duel), a change of possession is less likely to
be "controlled" -- per the challenge rules a loose/contested ball stays with the last
team. So raise the H<->A switching cost during duel frames. Geometry-conditioned control
confirmation, not a global cost sweep.

The frozen final_v5 gate (stop_gate=100) and min_dead=25 are applied identically in
every run, so all numbers are comparable to the reproduced baseline.

Run: .venv/bin/python analysis/autonomous_research/experiments/decoder/e3_sweep.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final, kickoff  # noqa: E402

STATES = np.array(["H", "A", "D"])
_TEAM = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def duel_mask(f):
    """Visible ball with >=1 home AND >=1 away player within 2 m of the ball."""
    return f["seen"] & (f["nh2"] >= 1) & (f["na2"] >= 1)


def viterbi_team(logE, cost_team, cost_dead, extra_team, bonus=None):
    """Viterbi where the H<->A cost at frame t is -(cost_team + extra_team[t])."""
    n = len(logE)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logE[0].copy()
    off = 1.0 - np.eye(3)
    for t in range(1, n):
        T = -cost_dead * _DEAD
        T[0, 1] = T[1, 0] = -(cost_team + extra_team[t])
        cand = score[:, None] + T
        if bonus is not None:
            cand = cand + bonus[t] * off
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_ar(c, Pd, extra_duel=0.0):
    """final_v5 decoder + extra H<->A switch cost on duel frames. Pd is the full
    probability dict; Pd[c.name] is the (n,3) emission array."""
    p = dict(decode.PARAMS)
    p.update(final.DECODER)
    m, f = c.m, c.f
    P = Pd[c.name]
    R = np.full(m.n, "X", dtype="<U1")  # use_rules=False (final_v5)
    B = Pd.get("B:" + c.name)
    bonus = decode.boundary_bonus(B, p["w_bound"]) if B is not None and p["w_bound"] else None
    logE = decode.emissions(P, R, f["seen"], p)
    logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
    extra_team = extra_duel * duel_mask(f).astype(float)
    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        bb = None if bonus is None else bonus[a:b + 1]
        out[a:b + 1] = STATES[viterbi_team(logE[a:b + 1], p["cost_team"], p["cost_dead"],
                                           extra_team[a:b + 1], bb)]
    out = decode.min_duration(out, p["min_dead"], p["min_poss"])
    if p["stop_gate"]:
        out = decode.stop_gate(out, f, int(p["stop_gate"]))
    return out


def main():
    y = Yard(pfile="P_R1.npz")
    Pd = y.P
    base = y.report("baseline final_v5", lambda c, _pk: decode_ar(c, Pd, extra_duel=0.0))
    print("\n=== E3: extra H<->A switch cost during contested duels (>=1H & >=1A within 2 m) ===")
    for extra in [1.0, 2.0, 4.0, 8.0]:
        out = y.report("E3 extra_team={:+.1f}".format(extra),
                       lambda c, _pk, e=extra: decode_ar(c, Pd, extra_duel=e))
        print("    -> bench {:+.4f} | ext {:+.4f} | masked {:+.4f} | public {:+.4f} | "
              "val {:+.4f}".format(out["bench"] - base["bench"], out["ext"] - base["ext"],
                                   out["masked"] - base["masked"], out["public"] - base["public"],
                                   out["val"] - base["val"]))


if __name__ == "__main__":
    main()
