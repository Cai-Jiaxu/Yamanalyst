"""Signed offset of predicted changes against true ones, per labelled window (aggregates only).

    uv run python analysis/boundary_offsets.py

For every true change of a kind (H<->A, dead->live, live->dead) the nearest predicted change of
the same kind within +-5 s; offset = predicted - true (positive = prediction late). Compares the
real GT (public window) with the eye labels (R3 windows): a convention gap shows as a shift of
the rules' and model's offsets between the two.
"""
import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import decode, final, rules  # noqa: E402
from mtah.data import CACHE_DIR, Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.masking import W0, W1  # noqa: E402
from mtah.score import load_gt, pm  # noqa: E402

warnings.filterwarnings("ignore")
R5 = 125


def changes(lab, kind):
    a, b = lab[:-1], lab[1:]
    if kind == "team":
        s = np.isin(a, ["H", "A"]) & np.isin(b, ["H", "A"]) & (a != b)
    elif kind == "restart":
        s = (a == "D") & np.isin(b, ["H", "A"])
    else:
        s = np.isin(a, ["H", "A"]) & (b == "D")
    return np.flatnonzero(s) + 1


def offsets(g, q, kind):
    tg, tq = changes(g, kind), changes(q, kind)
    out = []
    for t in tg:
        if len(tq):
            j = tq[np.argmin(np.abs(tq - t))]
            if abs(j - t) <= R5:
                out.append(j - t)
                continue
        out.append(np.nan)
    return np.array(out, dtype=float)


def main():
    m = Match()
    f = compute(m)
    P = dict(np.load(os.path.join(CACHE_DIR, "P_R1.npz")))["public"]
    R = rules.run(m, f)
    nox = np.full(m.n, "X", dtype="<U1")
    preds = {"model": decode.decode(m, nox, P, f["seen"],
                                    dict(decode.PARAMS, **dict(final.DECODER, use_rules=False,
                                                               stop_gate=0)), f),
             "v4": decode.decode(m, R, P, f["seen"], dict(decode.PARAMS, **final.DECODER), f),
             "rules": R}
    gt = np.array(pm.LABELS)[load_gt()]
    groups = {"public(GT)": [(W0, W1, gt)], "val(eye)": []}
    for path in sorted(glob.glob(os.path.join(ROOT, "work", "labels", "val_*.csv"))):
        lab = pm.read_label_csv(path)["label"].astype(str).str.strip().str.upper().values
        idx = np.flatnonzero(lab != "X")
        groups["val(eye)"].append((idx.min(), idx.max() + 1, lab))
    print("offset = predicted - true change, frames (25 = 1 s); hit = a same-kind change within 5 s")
    for kind in ("team", "restart", "stop"):
        for gname, wins in groups.items():
            line = "{:<8} {:<11} n={:3d} |".format(kind, gname, sum(len(changes(l[a:b], kind))
                                                                    for a, b, l in wins))
            for k, pr in preds.items():
                o = np.concatenate([offsets(l[a:b], pr[a:b], kind) for a, b, l in wins])
                h = o[np.isfinite(o)]
                line += " {} hit {:.2f} med {:+5.0f} iqr [{:+4.0f},{:+4.0f}] |{}".format(
                    k, np.isfinite(o).mean(), *(np.percentile(h, [50, 25, 75]) if len(h)
                                                 else [np.nan] * 3), "")
            print(line, flush=True)


if __name__ == "__main__":
    main()
