"""Phys228: +2 relative velocity features for transition head.
Features: opponent-vs-own closing speed margin, ball motion alignment.
100 trees, 3 folds, binary classes [0,1].
"""
import numpy as np, joblib, time, json
from sklearn.ensemble import HistGradientBoostingClassifier
from mtah import feats
from mtah.bench import Bench, FOLDS

MODEL = dict(max_iter=100, learning_rate=0.05, max_leaf_nodes=31,
             min_samples_leaf=100, l2_regularization=1.0)
LABEL_MAP = {"H": 0, "A": 1, "D": 2}
OUT = "cache/ar_phys228_results.json"


def phys_features(c):
    """+2 features from existing f dict:
    1. closing_speed_margin: (da - dh) / dt  (rate of change of nearest-opp margin)
    2. ball_motion_aligned: sign of vx * (bx - nearest_opp_x)
    """
    f = c.f
    seen = f["seen"]
    n = c.m.n
    d = np.zeros((n, 2))
    # Feature 1: rate of change of nearest-opp margin (closing speed)
    margin = f["da"] - f["dh"]
    d[:, 0] = np.gradient(margin)  # frames/sqrt(25fps)
    # Feature 2: ball velocity aligned with nearest opponent direction
    # vx is ball velocity; bx is ball x position; da is distance to nearest opp
    d[:, 1] = np.where(seen, f["vx"] * np.sign(f["da"] - f["dh"]), 0.0)
    return d


def transition_target(gt, window=6):
    n = len(gt)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        if gt[i] == 2:
            continue
        for j in range(i+1, min(i+window+1, n)):
            if gt[j] == 2:
                continue
            if gt[i] != gt[j]:
                y[i] = 1
                break
    return y


def main():
    t0 = time.time()
    b = Bench(verbose=True)
    P = {}
    models = []
    for i, fold in enumerate(FOLDS):
        Xs, ys = [], []
        for name in b.ext:
            if name in fold:
                continue
            c = b.ext[name]
            Xb = feats.full_plus3(c.m, c.f, c.Xs)
            Xb = np.hstack([Xb, phys_features(c)])
            yt = transition_target(c.gt)
            msk = c.gt != 3
            Xs.append(Xb[msk])
            ys.append(yt[msk])
        Xtr = np.vstack(Xs)
        ytr = np.concatenate(ys)
        print("phys228 fold", i, "rows", Xtr.shape, "pos_rate",
              round(float(ytr.mean()), 4), flush=True)
        clf = HistGradientBoostingClassifier(early_stopping=False,
                                            random_state=0, **MODEL)
        clf.fit(Xtr, ytr)
        models.append(clf)
        joblib.dump(clf, f"cache/ar_phys228_fold{i}.joblib")
        for k in fold:
            c = b.ext[k]
            Xb = feats.full_plus3(c.m, c.f, c.Xs)
            Xb = np.hstack([Xb, phys_features(c)])
            P[k] = clf.predict_proba(Xb)
    for c in [b.public] + b.masked:
        Xo = feats.full_plus3(c.m, c.f, c.Xs)
        Xo = np.hstack([Xo, phys_features(c)])
        P[c.name] = np.mean([m.predict_proba(Xo) for m in models], axis=0)
    res = b.score("phys228", P, dec_params=None, use_rules=False, t0=t0)
    rw = {"bench": float(res["bench"]), "ext": float(res["ext"][0]),
          "masked": float(res["masked"][0])}
    print("RESULT phys228", json.dumps(rw), flush=True)
    with open(OUT, "w") as f:
        json.dump(rw, f, indent=2)


if __name__ == "__main__":
    main()
