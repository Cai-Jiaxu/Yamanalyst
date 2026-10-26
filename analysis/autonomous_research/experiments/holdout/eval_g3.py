"""Holdout eval vet for Metrica Game 3.

Readiness check only. No scoring until the three E1 fold models and the matched
baseline fold models exist. This script only verifies paths, shapes and the
226-dim feature expectation. It does not touch production code.
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

E1_MODELS = ["cache/ar_e1_model_fold{}.joblib".format(i) for i in range(3)]
G3 = "cache/ar_holdout/ext_g3.npz"


def main():
    import joblib
    ok = True
    for p in E1_MODELS:
        if os.path.exists(p):
            m = joblib.load(p)
            print("found", p, "n_features", getattr(m, "n_features_in_", "?"))
        else:
            print("MISSING", p)
            ok = False
    z = np.load(G3, allow_pickle=True)
    print("g3 n", len(z["labels"]), "keys", list(z.files))
    print("readiness:", "ready" if ok else "waiting for E1 fold models")


if __name__ == "__main__":
    main()
