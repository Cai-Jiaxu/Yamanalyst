"""Noisy-student round 2 and a ball+shape model, scored public and masked.

    uv run python analysis/student.py
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah import rules, decode, shape_model as sm  # noqa: E402
from mtah.masking import masked_scores, W0, W1  # noqa: E402
from mtah.report import report  # noqa: E402

m = Match(); f = compute(m)
R = rules.run(m, f)
X = sm.features(m)
graded = m.graded_mask()


def fit(labels, Xmat, conf):
    tm = graded & conf
    tm[W0 - sm.BUFFER:W1 + sm.BUFFER] = False
    clf = sm.train(Xmat, labels, tm)
    return clf.predict_proba(Xmat)


def evaluate(name, P):
    meth = lambda m, f: decode.decode(m, rules.run(m, f), P, f["seen"])
    report(name, meth(m, f), m)
    mean, per = masked_scores(meth, m, compute)
    print("   masked spellF1 {:.3f} macroF1 {:.3f}".format(*mean))
    return meth(m, f)


P1 = fit(R, X, sm.confident(R, f["seen"]))
F1 = evaluate("round1 (rules labels)", P1)

# round 2: labels from the fused output; confidence no longer needs the ball
stable = sm.confident(F1, np.ones(m.n, bool))
P2 = fit(F1, X, stable)
evaluate("round2 (fused labels)", P2)

# ball + shape: ball features join the shape features; NaN where the ball is missing
ball_cols = np.stack([f[k] for k in ("bx", "by", "vx", "vy", "bspeed", "dv", "dh", "da",
                                    "nh2", "na2")], 1).astype(np.float32)
sgn = None
Xb = np.hstack([X, ball_cols])
P3 = fit(R, Xb, sm.confident(R, f["seen"]))
evaluate("round1 ball+shape", P3)
