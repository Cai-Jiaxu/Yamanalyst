"""Ball-free possession model trained on the rule system's confident labels (noisy student).

    uv run python -m mtah.shape_model     # trains, evaluates on the public window, caches probs

Training frames: every 5th graded frame OUTSIDE the public window (plus a 1-minute buffer) where
the rule system is confident -- the ball was seen within +-0.5 s and the rule label does not
change within +-1 s. The model never sees the ball, so it learns a different view of the same
labels and can stand in for the rules where the ball is missing.

Evaluated on the public window against the real ground truth, with no ball input at all.
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import CACHE_DIR, Match  # noqa: E402
from mtah.masking import W0, W1  # noqa: E402

CLASSES = np.array(["H", "A", "D"])
STEP = 5
BUFFER = 1500
SEED = 0


def features(m):
    path = os.path.join(CACHE_DIR, "shape_X.npy")
    if os.path.isfile(path):
        return np.load(path)
    from mtah import shape
    X, names = shape.compute(m)
    np.save(path, X)
    with open(os.path.join(CACHE_DIR, "shape_names.json"), "w") as handle:
        json.dump(names, handle)
    return X


def confident(labels, seen, fps=25):
    n = len(labels)
    seen_near = np.convolve(seen.astype(float), np.ones(fps + 1), "same") > 0
    stable = np.ones(n, dtype=bool)
    for d in range(1, fps + 1):
        stable[d:] &= labels[d:] == labels[:-d]
        stable[:-d] &= labels[:-d] == labels[d:]
    return seen_near & stable & np.isin(labels, CLASSES)


BALL_FEATURES = ("bx", "by", "vx", "vy", "bspeed", "dv", "dh", "da", "nh2", "na2")


def full_features(m, f):
    """Shape features plus ball features; ball columns are NaN where the ball is missing, which
    the gradient-boosted trees route natively -- one model for both regimes."""
    ball = np.stack([f[k] for k in BALL_FEATURES], axis=1).astype(np.float32)
    return np.hstack([features(m), ball])


def train(X, labels, train_mask, offset=0):
    idx = np.flatnonzero(train_mask)[offset::STEP]
    y = np.searchsorted(CLASSES, labels[idx], sorter=np.argsort(CLASSES))
    y = np.argsort(CLASSES)[y]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=100, l2_regularization=1.0,
                                         early_stopping=False, random_state=SEED)
    clf.fit(X[idx], y)
    return clf


def predict_proba(clf, X):
    """(n, 3) probabilities in CLASSES order (H, A, D)."""
    return clf.predict_proba(X)


def fit_full(m, labels, f):
    """Bag of STEP models, one per frame offset of the subsample, probabilities averaged."""
    # Ball-free on purpose: the ball already reaches the decoder through the rule labels, and
    # adding ball columns here double-counts it (analysis/pipeline_eval.py: no masked gain,
    # more public blips).
    X = features(m)
    train_mask = m.graded_mask() & confident(labels, f["seen"])
    train_mask[W0 - BUFFER:W1 + BUFFER] = False
    P = np.zeros((m.n, 3))
    for offset in range(STEP):
        P += predict_proba(train(X, labels, train_mask, offset), X) / STEP
    return P


if __name__ == "__main__":
    from mtah.features import compute
    from mtah import rules
    from mtah.score import load_gt, pm
    m = Match()
    f = compute(m)
    labels = rules.run(m, f)
    P = fit_full(m, labels, f)
    np.save(os.path.join(CACHE_DIR, "shape_proba.npy"), P.astype(np.float32))

    gt = np.array(pm.LABELS)[load_gt()]
    w = slice(W0, W1)
    pred = CLASSES[P[w].argmax(1)]
    g = gt[w]
    alive = np.isin(g, ("H", "A"))
    team_pred = np.where(P[w, 0] > P[w, 1], "H", "A")
    print("public window, NO ball input:")
    print("  3-class frame accuracy  {:.3f}".format(np.mean(pred == g)))
    print("  team accuracy on alive  {:.3f}".format(np.mean(team_pred[alive] == g[alive])))
    print("  dead recall / precision {:.3f} / {:.3f}".format(
        np.mean(pred[g == "D"] == "D"), np.mean(g[pred == "D"] == "D") if (pred == "D").any() else 0))
    print("rules on the same frames, WITH ball:")
    print("  3-class frame accuracy  {:.3f}".format(np.mean(labels[w] == g)))


# --- training on public external tracking with TRUE labels (W8) ------------------------------
# analysis/external_eval.py: a model trained this way reads our public window far better than one
# trained on our own pseudo-labels (team accuracy 0.93 vs 0.83 with no ball input). External
# sources have no tactical camera, so these models use cam=False features everywhere.

EXT_NOISE_VERSION = "v4"


def external_set(names=None):
    """[(name, X, labels, graded_mask)] for each parsed external match, degraded to our noise.

    Degradation is seeded by the match name (noise.seed_for), so the set is reproducible; features are cached under cache/ because the degrade step takes ~20 s each.
    """
    from mtah import external, noise, shape
    out = []
    for name in external.available():
        if names is not None and name not in names:
            continue
        path = os.path.join(CACHE_DIR, "extX_{}_{}.npz".format(name, EXT_NOISE_VERSION))
        em = external.ExtMatch(name)
        if os.path.isfile(path):
            X = np.load(path)["X"]
        else:
            X, _ = shape.compute(noise.degrade(em, seed=noise.seed_for(name)), cam=False)
            np.savez_compressed(path, X=X)
        out.append((name, X, em.labels, em.graded_mask()))
    return out


def fit_external(names=None, offsets=range(STEP)):
    """Bagged ball-free model on external true labels. Returns predict(X) -> (n, 3) H, A, D."""
    data = external_set(names)
    if not data:
        raise IOError("no external data parsed -- run scripts/download_external.sh and "
                      "python -m mtah.external")
    X = np.vstack([d[1] for d in data])
    y = np.concatenate([d[2] for d in data])
    msk = np.concatenate([d[3] for d in data])
    clfs = [train(X, y, msk, o) for o in offsets]
    return lambda Xq: np.mean([predict_proba(c, Xq) for c in clfs], axis=0)
