"""Why do the model's team calls drift later in the match? (ROADMAP R10)

    uv run python analysis/drift.py

Per window (public + the four R3 2nd-half windows): live-frame confusion of final_v2 and of the
rules against the labels, and tracking-quality signals (players tracked per team, ball seen,
team pace, where play is). Aggregates only.
"""
import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import decode, final, rules, shape  # noqa: E402
from mtah.data import CACHE_DIR, Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.masking import W0, W1  # noqa: E402
from mtah.score import load_gt, pm  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    m = Match()
    f = compute(m)
    P = dict(np.load(os.path.join(CACHE_DIR, "P_R1.npz")))["public"]
    p = dict(decode.PARAMS, **dict(final.DECODER, w_rule=1.0))
    model = decode.decode(m, np.full(m.n, "X", dtype="<U1"), P, f["seen"],
                          dict(p, use_rules=False), f)
    R = rules.run(m, f)
    L, W = m.pitch
    sgn = shape.attack_sign(m)
    home = m.team == 0
    nh = np.isfinite(m.players[:, home, 0]).sum(1)
    na = np.isfinite(m.players[:, ~home, 0]).sum(1)
    cx = (np.nanmean(m.players[..., 0], axis=1) - 0.5) * L * sgn   # + = play in A's half
    wins = [("public", W0, W1, np.array(pm.LABELS)[load_gt()])]
    for path in sorted(glob.glob(os.path.join(ROOT, "work", "labels", "val_*.csv"))):
        lab = pm.read_label_csv(path)["label"].astype(str).str.strip().str.upper().values
        idx = np.flatnonzero(lab != "X")
        wins.append((os.path.basename(path)[7:-4], idx.min(), idx.max() + 1, lab))
    print("{:<8} {:>5} {:>5} | model: acc  H->A  A->H | rules: acc  H->A  A->H | P(H) true/model "
          "| tracked H/A | ball seen | centroid x | max P on errors".format("window", "H%", "A%"))
    for name, a, b, lab in wins:
        g, mo, ru = lab[a:b], model[a:b], R[a:b]
        live = np.isin(g, ["H", "A"]) & np.isin(mo, ["H", "A"])
        lr = np.isin(g, ["H", "A"]) & np.isin(ru, ["H", "A"])
        def conf(pred, mask):
            gg, pp = g[mask], pred[mask]
            return (np.mean(gg == pp), np.mean(pp[gg == "H"] == "A"), np.mean(pp[gg == "A"] == "H"))
        cm, cr = conf(mo, live), conf(ru, lr)
        err = live & (g != mo)
        pmax = P[a:b].max(1)[err].mean() if err.any() else np.nan
        print("{:<8} {:5.0%} {:5.0%} |       {:.3f} {:5.1%} {:5.1%} |        {:.3f} {:5.1%} {:5.1%} | "
              "{:.2f}/{:.2f} | {:4.1f}/{:4.1f} | {:8.0%} | {:+9.1f} | {:.2f}".format(
                  name, np.mean(g == "H"), np.mean(g == "A"), *cm, *cr,
                  np.mean(g[live] == "H"), np.mean(mo[live] == "H"),
                  nh[a:b].mean(), na[a:b].mean(), np.isfinite(m.ball[a:b, 0]).mean(),
                  np.nanmean(cx[a:b]), pmax), flush=True)
        sn = f["seen"][a:b]
        rows = []
        for tag, sel in (("ball seen", sn), ("ball missing", ~sn)):
            mm, rr = live & sel, lr & sel
            rows.append("{} ({:.0%} of live): model err {:.1%}, rules err {:.1%}".format(
                tag, mm.sum() / max(live.sum(), 1), np.mean(g[mm] != mo[mm]) if mm.any() else np.nan,
                np.mean(g[rr] != ru[rr]) if rr.any() else np.nan))
        gd = g == "D"
        rows.append("ball seen on D frames {:.0%}".format(sn[gd].mean() if gd.any() else np.nan))
        print("         " + " | ".join(rows), flush=True)


if __name__ == "__main__":
    main()
