"""E1: contested micro-duel reweighting of the IDSSE training labels.

Hypothesis (predefined geometry, no relabel of our match, no training on the public window):
IDSSE labels count brief contested touches as possessions (4.3/min vs our GT ~3.4/min). A frame is a
'contested duel' when the ball is seen and BOTH a home and an away player are within 2 m of it
(`seen & nh2>=1 & na2>=1`). At such frames our GT attributes the ball to the last team (an aerial duel
without control does not switch), while IDSSE labels a team. Down-weighting these IDSSE frames teaches
the model NOT to flip possession at a contested duel, aligning training with our convention.

We fit the same 3-fold HistGradientBoosting models as mtah.bench, but pass sample_weight that is
`w_duel` at IDSSE duel frames and 1.0 elsewhere. Metrica frames are never reweighted (its convention
already matches ours). No mtah file is modified; this script reuses Bench's cached cases.

Fast screen protocol (300 trees, no mirrors/noise copies): compare w_duel=1.0 (baseline) against
w_duel in {0.5, 0.3, 0.1} on the SAME protocol. Adoption gate (heuristic, not significance):
reweighted screen bench must beat baseline by >=0.005 AND not regress ext or masked by >0.005.
"""
import json
import os
import time

import numpy as np

from mtah import feats
from mtah.bench import Bench, FOLDS
from mtah.score import pm
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = "cache/ar_convention_results.json"
LAB = {"H": 0, "A": 1, "D": 2}
DECODER = {"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 12.0, "min_dead": 25}


def duel_mask(case):
    seen = case.f["seen"].astype(bool)
    both = (case.f["nh2"] >= 1) & (case.f["na2"] >= 1)
    return seen & both


def fit_weighted(X, y_char, mask, w, max_iter=300, step=5):
    idx = np.flatnonzero(mask)[::step]
    y = np.array([LAB[c] for c in y_char[idx]])
    clf = HistGradientBoostingClassifier(
        early_stopping=False, random_state=0, max_iter=max_iter,
        learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0)
    clf.fit(X[idx], y, sample_weight=w[idx])
    return clf


def screen(b, w_duel, max_iter=300):
    featfn = feats.full_plus3
    X = {k: featfn(c.m, c.f, c.Xs) for k, c in b.ext.items()}
    lab = {k: np.array(pm.LABELS)[c.gt] for k, c in b.ext.items()}
    msk = {k: c.gt != pm.LABELS.index("X") for k, c in b.ext.items()}
    wt = {}
    duel_frac = {}
    for k, c in b.ext.items():
        w = np.ones(c.m.n, dtype=float)
        if k.startswith("idsse"):
            dm = duel_mask(c)
            duel_frac[k] = float(np.mean(dm[msk[k]]))
            w[dm] = w_duel
        wt[k] = w
    P, models = {}, []
    t0 = time.time()
    for i, fold in enumerate(FOLDS):
        train = [k for k in b.ext if k not in fold]
        clf = fit_weighted(np.vstack([X[k] for k in train]),
                           np.concatenate([lab[k] for k in train]),
                           np.concatenate([msk[k] for k in train]),
                           np.concatenate([wt[k] for k in train]),
                           max_iter=max_iter)
        models.append(clf)
        for k in fold:
            P[k] = clf.predict_proba(X[k])
    for c in [b.public] + b.masked:
        Xo = featfn(c.m, c.f, c.Xs)
        P[c.name] = np.mean([clf.predict_proba(Xo) for clf in models], axis=0)
    res = b.score("w_duel={}".format(w_duel), P, dec_params=DECODER,
                 use_rules=False, t0=t0)
    return {"w_duel": w_duel, "bench": float(res["bench"]),
            "ext": float(res["ext"][0]), "idsse": float(res["idsse"][0]),
            "metrica": float(res["metrica"][0]),
            "masked": float(res["masked"][0]), "public": float(res["public"][0]),
            "duel_frac_idsse_mean": float(np.mean([
                np.mean(duel_mask(c)[msk[k]]) for k, c in b.ext.items()
                if k.startswith("idsse")]))}


def main():
    b = Bench(verbose=True)
    rows = []
    if os.path.exists(RESULT_PATH):
        try:
            rows = json.load(open(RESULT_PATH)).get("rows", [])
        except Exception:
            rows = []
    done = {r["w_duel"] for r in rows}
    for w in (1.0, 0.5, 0.2, 0.0):
        if w in done:
            continue
        rows.append(screen(b, w))
        rows.sort(key=lambda x: x["w_duel"])
        with open(RESULT_PATH, "w") as f:
            json.dump({"date": time.strftime("%Y-%m-%d %H:%M"),
                       "protocol": "fast_screen_300_noaug",
                       "gate": "bench +>=0.005 vs w=1.0 and no ext/masked regression >0.005",
                       "rows": rows}, f, indent=2)
    base = rows[0]
    print("\n=== SUMMARY (fast screen) ===")
    for r in rows:
        print("w={:.2f} bench {:.3f} ({:+.3f}) | ext {:.3f} ({:+.3f}) | masked {:.3f} "
              "({:+.3f}) | pub {:.3f} | duelFrac {:.3f}".format(
                  r["w_duel"], r["bench"], r["bench"] - base["bench"],
                  r["ext"], r["ext"] - base["ext"], r["masked"],
                  r["masked"] - base["masked"], r["public"], r["duel_frac_idsse_mean"]))
    print("wrote", RESULT_PATH)


if __name__ == "__main__":
    main()
