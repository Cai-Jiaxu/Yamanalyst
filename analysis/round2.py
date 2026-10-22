"""Round 2 on the rebuilt data (half gaps, edge-clamped shifts, leak-free pace-weighted masking).

    uv run python analysis/round2.py

R0  full_plus2 + mirrors + noise copies, 600 trees, with the boundary head; decoder tuned
    greedily one dimension at a time (each dimension keeps the best value found so far)
R1  R0 + set-piece geometry features, at R0's best decoder
R2  R0 with IDSSE training labels relabelled to the controlled-possession convention
Fitted probabilities are cached in cache/P_<key>.npz.
"""
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import feats  # noqa: E402
from mtah.bench import Bench  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402

warnings.filterwarnings("ignore")
MODEL = {"max_iter": 600}
BASE = {"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 12.0}
DIMENSIONS = [
    ("cost_team", [6.0, 9.0, 12.0, 18.0]),
    ("kickoff", [False, True]),
    ("min_dead", [0, 25, 50]),
    ("w_bound", [0.0, 0.5, 1.0, 2.0]),
    ("em_prior", [False, True]),
    ("w_shape", [0.6, 0.75, 1.0]),
    ("cost_dead", [9.0, 12.0, 16.0]),
]


def extras(featfn):
    a, b = feats.mirrored(featfn), feats.augmented(featfn)
    return {k: a.get(k, []) + b.get(k, []) for k in set(a) | set(b)}


def fit(b, key, featfn, relabel=None):
    path = os.path.join(CACHE_DIR, "P_{}.npz".format(key))
    if os.path.isfile(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    t = time.time()
    P = b.probs(featfn, MODEL, extras(featfn), boundary=True, relabel=relabel)
    np.savez_compressed(path, **P)
    print("fitted {} in {:.0f}s".format(key, time.time() - t), flush=True)
    return P


def main():
    b = Bench()
    P0 = fit(b, "R0", feats.full_plus2)
    best = dict(BASE)
    best_score = b.score("R0 base", P0, best, use_rules=False, t0=time.time())["bench"]
    for name, values in DIMENSIONS:
        for v in values:
            if best.get(name) == v:
                continue
            g = dict(best, **{name: v})
            r = b.score("R0 {}={}".format(name, v), P0, g, use_rules=False, t0=time.time())
            if r["bench"] > best_score + 0.002:
                best, best_score = g, r["bench"]
        print("  -> best so far {} bench {:.3f}".format(best, best_score), flush=True)

    P1 = fit(b, "R1", feats.full_plus3)
    b.score("R1 set-piece @best", P1, best, use_rules=False, t0=time.time())
    P2 = fit(b, "R2", feats.full_plus2, relabel=feats.controlled_labels)
    b.score("R2 IDSSE relabel @best", P2, best, use_rules=False, t0=time.time())
    print("FINAL best decoder:", best, flush=True)


if __name__ == "__main__":
    main()
