"""Cycle 2: error-driven decoder levers on the frozen final_v5 recipe.

Reuses cache/P_R1.npz fold probabilities (no refit, no manual labels). Reports both
spellF1 and macroF1 on ext / masked / public / val, plus the per-100-true-possession
error taxonomy (split/merge/drift/missed/spurious).

Baseline = final_v5: w_shape 0.75, cost_team 12, cost_dead 12, min_dead 25,
kickoff=half, use_rules=False, stop_gate=100, w_bound=0.

Hypotheses (targeting the measured error mix: missed 22.3, drift 19.4, merge 10.6,
spurious 3.6, split 2.0 per 100 true possessions; pred/true 0.71):
  C1  cost_dead 12 -> 8 : cheaper alive<->dead switches should recover short live spells
                          (spurious live spells during stoppages are already killed by
                          stop_gate=100, so the spurious channel should stay closed)
  C2  cost_dead 12 -> 6
  C3  w_bound 0 -> 0.5  : turn on the cached boundary head to land H<->A switches on
                          predicted change frames (targets boundary drift)
  C4  w_bound 0 -> 1.0

Adoption rule (ROADMAP 1): bench up >=0.005 and neither ext nor masked down >0.005.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final, kickoff  # noqa: E402
from mtah.bench import taxonomy  # noqa: E402
from mtah.score import pm  # noqa: E402

STATES = np.array(["H", "A", "D"])
_TEAM = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def viterbi_team(logE, cost_team, cost_dead, extra_team, bonus=None):
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


def decode_ar(c, Pd, **over):
    p = dict(decode.PARAMS)
    p.update(final.DECODER)
    p.update(over)
    m, f = c.m, c.f
    P = Pd[c.name]
    R = np.full(m.n, "X", dtype="<U1")
    B = Pd.get("B:" + c.name)
    bonus = decode.boundary_bonus(B, p["w_bound"]) if B is not None and p["w_bound"] else None
    logE = decode.emissions(P, R, f["seen"], p)
    logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        bb = None if bonus is None else bonus[a:b + 1]
        out[a:b + 1] = STATES[viterbi_team(logE[a:b + 1], p["cost_team"], p["cost_dead"],
                                           np.zeros(b - a + 1), bb)]
    out = decode.min_duration(out, p["min_dead"], p["min_poss"])
    if p["stop_gate"]:
        out = decode.stop_gate(out, f, int(p["stop_gate"]))
    return out


def ext_taxonomy(y, Pd, over):
    agg = dict(split=0, merge=0, drift=0, missed=0, spurious=0, n_gt=0, n_pred=0)
    for k, c in y.b.ext.items():
        codes = pm.label_codes(decode_ar(c, Pd, **over))
        t = taxonomy(c.gt, codes)
        for kk in agg:
            agg[kk] += t[kk]
    return agg


def show(name, out, agg):
    per = 100.0 / agg["n_gt"]
    print("  {:<16} bench {:.4f} | ext {:.4f} (idsse {:.3f} metrica {:.3f}) | masked {:.4f} "
          "| public sp {:.4f}/mf {:.4f} | val sp {:.4f}/mf {:.4f}".format(
              name, out["bench"], out["ext"], out["idsse"], out["metrica"], out["masked"],
              out["public"], out["pubM"], out["val"], out["valM"]))
    print("                errors/100: missed {:.1f} drift {:.1f} merge {:.1f} spurious {:.1f} "
          "split {:.1f} | pred/true {:.2f}".format(
              agg["missed"] * per, agg["drift"] * per, agg["merge"] * per,
              agg["spurious"] * per, agg["split"] * per, agg["n_pred"] / agg["n_gt"]))


def main():
    y = Yard(pfile="P_R1.npz")
    Pd = y.P
    variants = [
        ("v5 baseline", dict()),
        ("C1 cost_dead=8", dict(cost_dead=8.0)),
        ("C2 cost_dead=6", dict(cost_dead=6.0)),
        ("C3 w_bound=0.5", dict(w_bound=0.5)),
        ("C4 w_bound=1.0", dict(w_bound=1.0)),
    ]
    base = y.report("v5 baseline", lambda c, _p: decode_ar(c, Pd))
    print("\n=== cycle 2 levers (spellF1 / macroF1) ===")
    for name, over in variants:
        out = y.report(name, lambda c, _p, o=over: decode_ar(c, Pd, **o), quiet=True)
        agg = ext_taxonomy(y, Pd, over)
        show(name, out, agg)
    print("\n(base row echoed)")
    agg = ext_taxonomy(y, Pd, dict())
    show("v5 baseline", base, agg)


if __name__ == "__main__":
    main()
