"""Recall of true possessions by duration, and spell-length mix, for the v2g decoder
(final_v2 + stop gate) on cached fold probabilities. Aggregates only.

    uv run python analysis/miss_by_len.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final  # noqa: E402
from mtah.score import pm  # noqa: E402

BINS = [0, 50, 125, 250, 500, 10 ** 9]   # frames: <2 s, 2-5, 5-10, 10-20, >20 s
P2 = dict(decode.PARAMS, **dict(final.DECODER, use_rules=False))


def rec(gt, pred):
    runs = pm.valid_runs(gt)
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    ov = pm.spell_overlaps(gs, ps)
    hit = pm.matched_gt_flags(ov, len(gs), 0.5)
    L = np.array([b - a for a, b, _ in gs])
    PL = np.array([b - a for a, b, _ in ps])
    return L, hit, PL


if __name__ == "__main__":
    y = Yard()
    groups = {"idsse": [], "metrica": [], "public": []}
    for k, c in y.b.ext.items():
        lab = decode.decode(c.m, np.full(c.m.n, "X", dtype="<U1"), y.P[k], c.f["seen"], P2, c.f)
        groups[k.split("_")[0]].append(rec(c.gt, pm.label_codes(lab)))
    c = y.b.public
    lab = decode.decode(c.m, np.full(c.m.n, "X", dtype="<U1"), y.P["public"], c.f["seen"], P2, c.f)
    groups["public"].append(rec(c.gt, pm.label_codes(lab)))
    print("duration bins: <2s | 2-5s | 5-10s | 10-20s | >20s ; share of true spells, recall@IoU0.5, "
          "share of predicted spells")
    for g, rows in groups.items():
        L = np.concatenate([r[0] for r in rows])
        H = np.concatenate([r[1] for r in rows])
        PL = np.concatenate([r[2] for r in rows])
        sh, rc, ps = [], [], []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            s = (L >= lo) & (L < hi)
            sh.append(s.mean())
            rc.append(H[s].mean() if s.any() else np.nan)
            ps.append(((PL >= lo) & (PL < hi)).mean())
        print("{:<8} n_true {:5d} n_pred {:5d} | share {} | recall {} | pred share {}".format(
            g, len(L), len(PL), " ".join("{:.2f}".format(v) for v in sh),
            " ".join("{:.2f}".format(v) for v in rc), " ".join("{:.2f}".format(v) for v in ps)))
