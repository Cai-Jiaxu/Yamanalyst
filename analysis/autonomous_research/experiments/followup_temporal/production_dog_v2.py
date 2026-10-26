"""Corrected DoG wrapper: softmax on logits, R=all X, cached P.

Bug fixed: Q was normalizing logits by row sum (inverses classes); now softmax.
Uses cached production P; no new fit. Writes frame,label CSV.
"""
import os, sys, hashlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from mtah.data import Match
from mtah.features import compute
from mtah import final, decode

m = Match()
f = compute(m)
P = np.load(os.path.join(ROOT, "cache/ar_followup_production_P.npz"))["P"]
p = dict(decode.PARAMS, **dict(final.DECODER, use_rules=False))
R_dec = np.full(m.n, "X", dtype="<U1")

def dog(P, sigma=12, w=0.5):
    sharp = np.log(np.clip(P, 1e-9, 1.0))
    logits = sharp.copy()
    for s in range(3):
        logits[:, s] = sharp[:, s] + w * (
            gaussian_filter1d(sharp[:, s], sigma) - gaussian_filter1d(sharp[:, s], sigma * 3))
    logits -= logits.max(1, keepdims=True)
    Q = np.exp(logits)
    Q /= Q.sum(1, keepdims=True)
    return Q

# zero-weight identity test
Q0 = dog(P, w=0.0)
assert np.allclose(Q0, P, atol=1e-6), "zero-weight identity failed"
assert (Q0 >= 0).all() and np.isfinite(Q0).all()
out0 = decode.decode(m, R_dec, P, f["seen"], p, f)
with open("/tmp/z.csv", "w") as h:
    h.write("frame,label\n")
    h.writelines("{},{}\n".format(i, c) for i, c in enumerate(out0))
print("zero-weight sha", hashlib.sha256(open("/tmp/z.csv", "rb").read()).hexdigest()[:16], flush=True)

# primary DoG
Q = dog(P, sigma=12, w=0.5)
amP = P.argmax(1); amQ = Q.argmax(1)
print("argmax flips", int((amP != amQ).sum()), "of", len(amP), flush=True)
print("mean max P", round(float(P.max(1).mean()), 4), "Q", round(float(Q.max(1).mean()), 4), flush=True)

out = decode.decode(m, R_dec, Q, f["seen"], p, f)
path = os.path.join(ROOT, "evaluation/results/ar_candidate_dog.csv")
with open(path, "w") as h:
    h.write("frame,label\n")
    h.writelines("{},{}\n".format(i, c) for i, c in enumerate(out))
print("DoG sha", hashlib.sha256(open(path, "rb").read()).hexdigest()[:16], flush=True)
print("labels", dict(zip(*np.unique(out, return_counts=True))), flush=True)

from analysis.yardsticks import Yard
from mtah.score import pm
y = Yard()
res = pm.evaluate(y.b.gt, pm.label_codes(out), bootstrap_n=0)
print("DoG spellF1", round(res["spell_f1_mean"], 4), "macro", round(res["macro_f1"], 4), flush=True)
