"""Ball+shape model capacity sweep, 16 masking seeds. uv run python analysis/capacity.py"""
import os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah import rules, decode, shape_model as sm  # noqa: E402
from mtah.masking import masked_scores, W0, W1  # noqa: E402
from mtah.score import load_gt, score  # noqa: E402

m = Match(); f = compute(m); gt = load_gt()
R = rules.run(m, f)
X = sm.features(m)
ball = np.stack([f[k] for k in ("bx", "by", "vx", "vy", "bspeed", "dv", "dh", "da", "nh2", "na2")],
                1).astype(np.float32)
Xb = np.hstack([X, ball])
tm = m.graded_mask() & sm.confident(R, f["seen"])
tm[W0 - sm.BUFFER:W1 + sm.BUFFER] = False
idx = np.flatnonzero(tm)[::sm.STEP]
y = np.array([{"H": 0, "A": 1, "D": 2}[c] for c in R[idx]])
for it, leaf, lr in ((600, 100, 0.05), (1000, 100, 0.05), (600, 30, 0.05),
                     (300, 30, 0.1)):
    clf = HistGradientBoostingClassifier(max_iter=it, learning_rate=lr, max_leaf_nodes=31,
                                         min_samples_leaf=leaf, l2_regularization=1.0,
                                         early_stopping=False, random_state=0).fit(Xb[idx], y)
    P = clf.predict_proba(Xb)
    meth = lambda m, f: decode.decode(m, rules.run(m, f), P, f["seen"])
    pub, _ = score(meth(m, f), gt)
    mean, per = masked_scores(meth, m, compute, seeds=range(16))
    print("iter {:>3} leaf {:>4}: public {:.3f}/{:.3f}  masked {:.3f}/{:.3f} (sd {:.3f})".format(
        it, leaf, pub["spellF1"], pub["macroF1"], mean[0], mean[1], per[:, 0].std()))
