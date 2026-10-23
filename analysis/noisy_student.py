"""ROADMAP R4: noisy student -- add our own match, pseudo-labelled, to the training data.

    uv run python analysis/noisy_student.py         # after analysis/ball_value.py (needs its models)

Each fold's teacher is that fold's final_v1 model (cache/models_final_v1.joblib), so a fold never
trains on labels from a model that saw its own test matches. The teacher's decoded labels are
kept where it is confident (max probability >= CONF and agreeing with the decoded label), on
graded frames outside the public window +- 30 s and the masked look-ahead tail, so the masked
and public scores stay honest. One student refit (~1 h). Aggregates only.
"""
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import decode, feats, final  # noqa: E402
from mtah.bench import Bench  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402
from mtah.masking import TAIL, W0, W1  # noqa: E402

warnings.filterwarnings("ignore")
CONF = 0.8
MARGIN = 750   # frames (30 s) either side of the public window


def main():
    import joblib
    featfn = getattr(feats, final.FEATFN)
    b = Bench()
    teachers = joblib.load(os.path.join(CACHE_DIR, "models_final_v1.joblib"))
    if isinstance(teachers, dict):
        teachers = teachers["models"]
    c = b.public
    Xo = featfn(c.m, c.f, c.Xs)
    keep = c.m.graded_mask()
    keep[max(0, W0 - MARGIN):W1 + TAIL + MARGIN] = False
    no_rules = np.full(c.m.n, "X", dtype="<U1")
    fold_extra = []
    for i, clf in enumerate(teachers):
        Pi = clf.predict_proba(Xo)
        lab = decode.decode(c.m, no_rules, Pi, c.f["seen"], dict(decode.PARAMS, **final.DECODER),
                            c.f)
        agree = np.array(["H", "A", "D"])[Pi.argmax(1)] == lab
        mask = keep & agree & (Pi.max(1) >= CONF) & (lab != "X")
        fold_extra.append([(Xo, lab, mask)])
        print("teacher {}: {:.0%} of graded frames kept (H {:.0%} A {:.0%} D {:.0%})".format(
            i, mask.sum() / c.m.graded_mask().sum(),
            *[np.mean(lab[mask] == k) for k in "HAD"]), flush=True)
    extra = {k: feats.mirrored(featfn).get(k, []) + feats.augmented(featfn).get(k, [])
             for k in b.ext}
    t = time.time()
    P = b.probs(featfn, final.MODEL, extra=extra, key="final_v2_student", fold_extra=fold_extra)
    print("fitted in {:.0f} s".format(time.time() - t), flush=True)
    b.score("R4 noisy student", P, final.DECODER, use_rules=False, t0=time.time())


if __name__ == "__main__":
    main()
