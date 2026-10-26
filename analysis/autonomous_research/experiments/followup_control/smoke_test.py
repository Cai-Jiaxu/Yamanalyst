"""Followup control: smoke test + checkpoint support.
3-iter fit on subsample to verify P keys (public/masked/ext), classes [0,1,2], 226 features.
"""
import numpy as np, joblib, time, json, os
from sklearn.ensemble import HistGradientBoostingClassifier
from mtah import feats
from mtah.bench import Bench, FOLDS

LABEL_MAP = {"H": 0, "A": 1, "D": 2}
OUT = "cache/ar_followup_control_smoke.json"


def smoke_test():
    b = Bench(verbose=True)
    # 3-iter fit on fold 0 training data
    fold = FOLDS[0]
    Xs, ys, ms = [], [], []
    for name in b.ext:
        if name in fold:
            continue
        c = b.ext[name]
        Xb = feats.full_plus3(c.m, c.f, c.Xs)
        lab = np.array(["H", "A", "D", "X"])[c.gt]
        msk = c.gt != 3
        Xs.append(Xb); ys.append(lab); ms.append(msk)
    Xtr = np.vstack(Xs)
    ytr = np.concatenate(ys)
    mtr = np.concatenate(ms)
    ysub = np.array([LABEL_MAP[c] for c in ytr[mtr]])
    # subsample 10k rows for speed
    idx = np.random.RandomState(0).choice(len(ysub), 10000, replace=False)
    clf = HistGradientBoostingClassifier(max_iter=3, early_stopping=False, random_state=0)
    clf.fit(Xtr[mtr][idx], ysub[idx])
    assert list(clf.classes_) == [0, 1, 2], f"classes wrong: {clf.classes_}"
    assert Xtr.shape[1] == 226, f"n_features wrong: {Xtr.shape[1]}"
    print("PASS: classes [0,1,2], 226 features", flush=True)

    # Build P dict and verify keys
    P = {}
    for i, f in enumerate(FOLDS):
        for k in f:
            c = b.ext[k]
            Xb = feats.full_plus3(c.m, c.f, c.Xs)
            P[k] = clf.predict_proba(Xb)
    for c in [b.public] + b.masked:
        Xo = feats.full_plus3(c.m, c.f, c.Xs)
        P[c.name] = clf.predict_proba(Xo)
    expected = ["public"] + [c.name for c in b.masked] + list(b.ext.keys())
    missing = [k for k in expected if k not in P]
    assert not missing, f"missing P keys: {missing}"
    print("PASS: P has all public/masked/ext keys", flush=True)

    # Save checkpoint
    joblib.dump(clf, "cache/ar_followup_smoke_model.joblib")
    print("PASS: checkpoint saved", flush=True)

    with open(OUT, "w") as f:
        json.dump({"status": "pass", "classes": [0, 1, 2], "n_features": 226,
                    "expected_keys": len(expected)}, f, indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    smoke_test()
