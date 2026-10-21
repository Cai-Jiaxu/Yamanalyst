"""Score the exact run.py pipeline: public window, and masked (16 seeds).

The shape model is ball-free and the public window is excluded from its training, so its
probabilities do not change when the ball is hidden; only the rule labels are recomputed.

    uv run python analysis/pipeline_eval.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah import rules, decode, shape_model as sm  # noqa: E402
from mtah.masking import masked_scores  # noqa: E402
from mtah.report import report  # noqa: E402

m = Match(); f = compute(m)
P = sm.fit_full(m, rules.run(m, f), f)
meth = lambda m, f2: decode.decode(m, rules.run(m, f2), P, f2["seen"])  # noqa: E731
report("pipeline", meth(m, f), m)
mean, per = masked_scores(meth, m, compute, seeds=range(16))
print("   masked spellF1 {:.3f} macroF1 {:.3f} (sd {:.3f})".format(mean[0], mean[1], per[:, 0].std()))
