"""Per-fold probabilities on our match (public + masked cases) from the cached final_v1 fold
models, so a refiner trained on single-model out-of-fold probabilities sees the same kind of
input on our match. Writes cache/P_R1_folds.npz. Aggregates only.

    uv run python analysis/fold_probs.py
"""
import os
import sys
import time

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import feats  # noqa: E402
from mtah.bench import Bench  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402

if __name__ == "__main__":
    b = Bench(verbose=False)
    models = joblib.load(os.path.join(CACHE_DIR, "models_final_v1.joblib"))
    models = models["models"] if isinstance(models, dict) else models
    P1 = dict(np.load(os.path.join(CACHE_DIR, "P_R1.npz")))
    out = {}
    for c in [b.public] + b.masked:
        t = time.time()
        X = feats.full_plus3(c.m, c.f, c.Xs)
        for i, clf in enumerate(models):
            out["{}#{}".format(c.name, i)] = clf.predict_proba(X).astype(np.float32)
        avg = np.mean([out["{}#{}".format(c.name, i)] for i in range(3)], 0)
        print("{} max |mean - P_R1| {:.2e}  {:.0f}s".format(
            c.name, np.abs(avg - P1[c.name]).max(), time.time() - t), flush=True)
    np.savez_compressed(os.path.join(CACHE_DIR, "P_R1_folds.npz"), **out)
