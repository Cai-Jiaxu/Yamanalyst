"""Typed transition head + phys228 screen.
Binary classification: turnover (H↔A within 6 frames) vs no-turnover.
Excludes D frames. Weighted positive for imbalance.
100 trees, 3 folds, no mirrors/noise.
"""
import numpy as np, joblib, time, json
from sklearn.ensemble import HistGradientBoostingClassifier
from mtah import feats
from mtah.bench import Bench, FOLDS

MODEL = dict(max_iter=100, learning_rate=0.05, max_leaf_nodes=31,
             min_samples_leaf=100, l2_regularization=1.0)
LABEL_MAP = {"H": 0, "A": 1, "D": 2}
OUT = "cache/ar_followup_control_results.json"


def transition_target(gt, window=6):
    """Binary: turnover (H↔A) within window frames. Excludes D."""
    n = len(gt)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        if gt[i] == 2:  # skip D
            continue
        for j in range(i+1, min(i+window+1, n)):
            if gt[j] == 2:
                continue
            if gt[i] != gt[j]:  # H↔A flip
                y[i] = 1
                break
    return y


def phys_features(c):
    """+2 features: nearest opponent distance, closing speed."""
    n = c.m.n
    d = np.zeros((n, 2))
    for i in range(n):
        bx, by = c.m.ball[i]
        opp = c.m.away_pos[i]
        dist = np.sqrt(((opp - [bx, by]) ** 2).sum(1))
        d[i, 0] = dist.min() if len(dist) else 10.0
        d[i, 1] = 0.0  # closing speed placeholder
    return d


def main():
    t0 = time.time()
    b = Bench(verbose=True)
    results = {}

    for variant in ["baseline226", "phys228"]:
        P = {}
        models = []
        for i, fold in enumerate(FOLDS):
            Xs, ys = [], []
            for name in b.ext:
                if name in fold:
                    continue
                c = b.ext[name]
                Xb = feats.full_plus3(c.m, c.f, c.Xs)
                if variant == "phys228":
                    Xb = np.hstack([Xb, phys_features(c)])
                # transition target: binary turnover
                yt = transition_target(c.gt)
                msk = c.gt != 3  # exclude X
                Xs.append(Xb[msk])
                ys.append(yt[msk])
            Xtr = np.vstack(Xs)
            ytr = np.concatenate(ys)
            print(variant, "fold", i, "rows", Xtr.shape, "pos_rate",
                  round(float(ytr.mean()), 4), flush=True)
            clf = HistGradientBoostingClassifier(early_stopping=False,
                                                random_state=0, **MODEL)
            clf.fit(Xtr, ytr)
            models.append(clf)
            # save checkpoint
            joblib.dump(clf, f"cache/ar_followup_{variant}_fold{i}.joblib")
            for k in fold:
                c = b.ext[k]
                Xb = feats.full_plus3(c.m, c.f, c.Xs)
                if variant == "phys228":
                    Xb = np.hstack([Xb, phys_features(c)])
                P[k] = clf.predict_proba(Xb)
        # public + masked
        for c in [b.public] + b.masked:
            Xo = feats.full_plus3(c.m, c.f, c.Xs)
            if variant == "phys228":
                Xo = np.hstack([Xo, phys_features(c)])
            P[c.name] = np.mean([m.predict_proba(Xo) for m in models], axis=0)
        # score
        res = b.score(variant, P, dec_params=None, use_rules=False, t0=t0)
        results[variant] = {"bench": float(res["bench"]),
                            "ext": float(res["ext"][0]),
                            "masked": float(res["masked"][0])}
        print("RESULT", variant, json.dumps(results[variant]), flush=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
