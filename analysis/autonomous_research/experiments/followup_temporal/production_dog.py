"""Production full-fit model + DoG sharpening before decode.

Wraps mtah.final.predict (full-fit model, cached features), applies DoG highpass
(sigma=12, w=0.5, all classes) to P, then re-decodes with frozen v5 DECODER.
Outputs 199319-row candidate CSV. No public selector. Run twice for determinism.
"""
import os
import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from mtah import final, decode, feats, shape, external
from mtah.data import Match

SIGMA = 12
W = 0.5


def dog(P):
    sharp = np.log(np.clip(P, 1e-9, 1.0))
    out = np.empty_like(sharp)
    for s in range(3):
        narrow = gaussian_filter1d(sharp[:, s], SIGMA)
        wide = gaussian_filter1d(sharp[:, s], SIGMA * 3)
        out[:, s] = sharp[:, s] + W * (narrow - wide)
    Q = np.exp(out)
    Q /= Q.sum(1, keepdims=True)
    return Q


def main():
    t0 = time.time()
    from analysis.yardsticks import Yard
    y = Yard()
    c = y.b.public
    m, f = c.m, c.f
    labels, P = final.predict(m, f, None)
    print("production P shape", P.shape, "elapsed {:.0f}s".format(time.time() - t0), flush=True)

    Q = dog(P)
    p = dict(decode.PARAMS, **dict(final.DECODER, use_rules=False))
    R_dec = np.full(m.n, "X", dtype="<U1")
    out1 = decode.decode(m, R_dec, Q, f["seen"], p, f)
    from mtah.decode import min_duration, stop_gate
    out1 = min_duration(out1, p["min_dead"], 0)
    if p["stop_gate"]:
        out1 = stop_gate(out1, f, p["stop_gate"])

    labels2, P2 = final.predict(m, f, None)
    Q2 = dog(P2)
    out2 = decode.decode(m, R_dec, Q2, f["seen"], p, f)
    out2 = min_duration(out2, p["min_dead"], 0)
    if p["stop_gate"]:
        out2 = stop_gate(out2, f, p["stop_gate"])

    agree = (out1 == out2).mean()
    print("determinism agree {:.4f}".format(agree), flush=True)
    print("out1 unique:", dict(zip(*np.unique(out1, return_counts=True))), flush=True)
    print("out1 spells:", int((out1[1:] != out1[:-1]).sum()) + 1, flush=True)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    np.savetxt(os.path.join(out_dir, "candidate_dog_public.csv"), out1, fmt="%s")
    print("saved candidate_dog_public.csv, elapsed {:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
