"""Followup control screen: baseline226 matched recipe + typed transition head.
Fast 300-tree screen on 9 external matches, 3 folds. No mirrors/noise.
"""
import numpy as np, time, json, os
from sklearn.ensemble import HistGradientBoostingClassifier
from mtah import feats
from mtah.bench import Bench, FOLDS

MODEL_SCREEN = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                    min_samples_leaf=100, l2_regularization=1.0)
LABEL_MAP = {"H": 0, "A": 1, "D": 2}
OUT = "cache/ar_followup_control.json"


def build_fold(b, fold):
    Xs, ys, ms = [], [], []
    for name in b.ext:
        if name in fold:
            continue
        c = b.ext[name]
        Xb = feats.full_plus3(c.m, c.f, c.Xs)
        lab = np.array(["H", "A", "D", "X"])[c.gt]
        msk = c.gt != 3
        Xs.append(Xb)
        ys.append(lab)
        ms.append(msk)
    Xtr = np.vstack(Xs)
    ytr = np.concatenate(ys)
    mtr = np.concatenate(ms)
    ysub = np.array([LABEL_MAP[c] for c in ytr[mtr]])
    return Xtr[mtr], ysub


def main():
    t0 = time.time()
    b = Bench(verbose=True)
    results = {}

    # Variant 1: baseline226 (matched recipe)
    P = {}
    models = []
    for i, fold in enumerate(FOLDS):
        Xtr, ytr = build_fold(b, fold)
        print("fold", i, "rows", Xtr.shape, flush=True)
        clf = HistGradientBoostingClassifier(early_stopping=False,
                                            random_state=0, **MODEL_SCREEN)
        clf.fit(Xtr, ytr)
        models.append(clf)
        for k in fold:
            c = b.ext[k]
            Xb = feats.full_plus3(c.m, c.f, c.Xs)
            P[k] = clf.predict_proba(Xb)
    # public + masked (average over folds)
    for c in [b.public] + b.masked:
        Xo = feats.full_plus3(c.m, c.f, c.Xs)
        P[c.name] = np.mean([m.predict_proba(Xo) for m in models], axis=0)
    res = b.score("baseline226", P, dec_params=None, use_rules=False, t0=t0)
    results["baseline226"] = {"bench": float(res["bench"]),
                               "ext": float(res["ext"][0]),
                               "masked": float(res["masked"][0])}
    print("RESULT baseline226", json.dumps(results["baseline226"]), flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
