"""Which shape model should the pipeline use? Scored on OUR match: public, and masked (16 seeds).

    uv run python analysis/ext_pipeline.py

  own        bagged model on our own rule pseudo-labels, with camera features (fused_v2)
  ext        bagged model on external TRUE labels (all parsed matches), no camera
  avg        mean of own and ext probabilities
  union      one model on external true labels + our pseudo-labels, no camera
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah import rules, decode, shape, shape_model as sm  # noqa: E402
from mtah.masking import masked_scores, W0, W1  # noqa: E402
from mtah.report import report  # noqa: E402

m = Match(); f = compute(m)
R = rules.run(m, f)
Xnc, _ = shape.compute(m, cam=False)

P = {}
P["own"] = sm.fit_full(m, R, f)
P["ext"] = sm.fit_external()(Xnc)
P["avg"] = (P["own"] + P["ext"]) / 2

data = sm.external_set()
tm = m.graded_mask() & sm.confident(R, f["seen"])
tm[W0 - sm.BUFFER:W1 + sm.BUFFER] = False
X = np.vstack([d[1] for d in data] + [Xnc])
y = np.concatenate([d[2] for d in data] + [R])
msk = np.concatenate([d[3] for d in data] + [tm])
clfs = [sm.train(X, y, msk, o) for o in range(sm.STEP)]
P["union"] = np.mean([sm.predict_proba(c, Xnc) for c in clfs], axis=0)

for name, Pv in P.items():
    meth = lambda m, f2, Pv=Pv: decode.decode(m, rules.run(m, f2), Pv, f2["seen"])  # noqa: E731
    report(name, meth(m, f), m)
    mean, per = masked_scores(meth, m, compute, seeds=range(16))
    print("   masked spellF1 {:.3f} macroF1 {:.3f} (sd {:.3f})".format(mean[0], mean[1],
                                                                    per[:, 0].std()))
