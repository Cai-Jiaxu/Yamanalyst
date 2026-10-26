"""Reproduce the internal final_v5 fold benchmark from cached fold probabilities (aggregates only).

Does NOT refit any model. Reuses cache/P_R1.npz (the full-recipe fold probabilities:
9 external matches with true labels, degraded to our tracker, plus mirrors and extra-noise
copies) and applies the final_v5 decoder from mtah/final.py.

Run with the venv interpreter directly (avoids uv.lock mutation):
    .venv/bin/python analysis/autonomous_research/experiments/decoder/reproduce_v5.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final  # noqa: E402
from mtah.bench import taxonomy  # noqa: E402
from mtah.score import pm  # noqa: E402


def make_dec(**kw):
    """Return a fn(case, P) -> per-frame labels using the final_v5 decoder, overridden by kw."""
    p = dict(decode.PARAMS)
    p.update(final.DECODER)
    p.update(kw)

    def fn(c, P):
        R = np.full(c.m.n, "X", dtype="<U1") if p.get("use_rules") is False else c.R
        B = P.get("B:" + c.name) if p.get("w_bound", 0) else None
        return decode.decode(c.m, R, P, c.f["seen"], p, c.f, B)
    return fn, p


def ext_taxonomy(y, fn):
    """Per-100-true-possession error type counts across the external fold matches."""
    agg = dict(split=0, merge=0, drift=0, missed=0, spurious=0, n_gt=0, n_pred=0)
    for k, c in y.b.ext.items():
        codes = pm.label_codes(fn(c, y.P[k]))
        t = taxonomy(c.gt, codes)
        for kk in agg:
            agg[kk] += t[kk]
    return agg


def main():
    y = Yard(pfile="P_R1.npz")
    fn, p = make_dec()
    out = y.report("final_v5 (repro)", fn)
    agg = ext_taxonomy(y, fn)

    print("\n=== final_v5 fold-bench reproduction ===")
    print("decoder config:", {k: p[k] for k in
          ("w_shape", "cost_dead", "cost_team", "min_dead", "kickoff", "use_rules",
           "stop_gate", "w_bound")})
    print("bench   {:.4f}  (ext {:.4f} + masked {:.4f})".format(
        out["bench"], out["ext"], out["masked"]))
    print("ext     {:.4f}  idsse {:.4f}  metrica {:.4f}".format(
        out["ext"], out["idsse"], out["metrica"]))
    print("masked  {:.4f}  (16-seed mean)".format(out["masked"]))
    print("public  spellF1 {:.4f}  macroF1 {:.4f}  (n_pred_spells {})".format(
        out["public"], out["pubM"], out["npub"]))
    print("val_h2  spellF1 {:.4f}  macroF1 {:.4f}  valLo {:.4f}  valMid(team err) {:.4f}".format(
        out["val"], out["valM"], out["valLo"], out["valMid"]))
    per = 100.0 / agg["n_gt"]
    print("ext errors /100 true poss: missed {:.1f} drift {:.1f} merge {:.1f} spurious {:.1f} "
          "split {:.1f} | pred/true {:.2f}".format(
              agg["missed"] * per, agg["drift"] * per, agg["merge"] * per,
              agg["spurious"] * per, agg["split"] * per, agg["n_pred"] / agg["n_gt"]))


if __name__ == "__main__":
    main()
