"""In-process scoring against a frame-label ground truth, using the official metric library.

    from mtah.score import load_gt, score
    gt = load_gt()                 # public 5 minutes, codes array, X elsewhere
    score(labels, gt)              # {'spellF1', 'macroF1', 'n_pred', 'n_gt', ...}
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def load_gt(path=PUBLIC_GT):
    """Code array (pm.H/A/D/X) for a frame,label CSV."""
    return pm.label_codes(pm.read_label_csv(path))


def to_codes(labels):
    """Letters ('H','A','D','X') or codes -> code array."""
    labels = np.asarray(labels)
    if labels.dtype.kind in "US":
        return pm.label_codes(labels)
    return labels


def score(labels, gt_codes, bootstrap=0):
    r = pm.evaluate(gt_codes, to_codes(labels), bootstrap_n=bootstrap)
    out = {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
           "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
    return out, r
