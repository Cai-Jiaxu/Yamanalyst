"""Paired holdout evaluation: baseline fold ensemble vs E1 fold ensemble on Metrica Game 3.

FROZEN inputs:
  - holdout:  cache/ar_holdout/ext_g3.npz   sha aa1589d7...a23d7
  - baseline: cache/models_final_v1.joblib    sha cd1bf353...39d4c  (3 folds, 226-dim)
  - E1:       cache/ar_e1_model_fold{0,1,2}.joblib  (producer; NOT yet saved)

Protocol:
  - Compute mtah.features.compute + feats.full_plus3 on the holdout (226-dim).
  - Average predict_proba over the 3 folds for each side (matched fold ensemble).
  - Apply the frozen v5 decoder (w_shape 0.75, cost_dead 12, cost_team 12, min_dead 25,
    kickoff=half, stop_gate=100, w_bound=0).
  - Score against event-derived holdout labels via mtah.score.pm spell F1.
  - Clean = primary. Stress = Bernoulli p=0.41 ball-missing, seed 20261025 (stress, not
    realistic replication).
  - No retune, no model selection on this holdout.

This script refuses to run until all three E1 fold joblibs exist. It does not fabricate
an E1 result.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from mtah.data import CACHE_DIR  # noqa: E402

HOLDOUT = os.path.join(ROOT, "cache", "ar_holdout", "ext_g3.npz")
BASELINE = os.path.join(CACHE_DIR, "models_final_v1.joblib")
E1_FOLDS = [os.path.join(CACHE_DIR, "ar_e1_model_fold{}.joblib".format(i)) for i in range(3)]
E1_SEED = 20261025
STRESS_P = 0.41


def e1_ready():
    return all(os.path.isfile(p) and os.path.getsize(p) > 1000 for p in E1_FOLDS)


def structural_check():
    """De-risk feature/label wiring without running either model. No fake eval."""
    z = np.load(HOLDOUT)
    print("holdout keys:", list(z.keys()))
    print("players", z["players"].shape, "labels", z["labels"].shape)
    print("halves", z["halves"].tolist(), "pitch", z["pitch"].tolist())
    if not e1_ready():
        print("E1 folds not yet saved; structural check only, no evaluation run.")
        return
    print("All 3 E1 folds present. Run main eval.")


if __name__ == "__main__":
    structural_check()
