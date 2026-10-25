"""ROADMAP R17 full fit: touch-sequence / kick features (feats.full_plus5) with the final_v1
recipe (600 trees, mirrored + extra-noise copies). It passed the fast screen (r17_screen.py:
bench 0.573 -> 0.613). Scored with the final_v5 decoder on every yardstick, against P_R1.

    uv run python analysis/r17_full.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.r17_screen import by_len, v2g  # noqa: E402
from analysis.yardsticks import Yard  # noqa: E402
from mtah import feats, final  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402

if __name__ == "__main__":
    y = Yard()
    y.report("final_v5 (P_R1, full_plus3)", v2g)
    print("  " + by_len(y, y.P), flush=True)
    featfn = feats.full_plus5
    t = time.time()
    extra = {k: feats.mirrored(featfn).get(k, []) + feats.augmented(featfn).get(k, [])
             for k in y.b.ext}
    print("augmented features in {:.0f} s".format(time.time() - t), flush=True)
    P = y.b.probs(featfn, final.MODEL, extra=extra, key="full_plus5")
    np.savez_compressed(os.path.join(CACHE_DIR, "P_R17.npz"), **P)
    print("fitted in {:.0f} s".format(time.time() - t), flush=True)
    y.report("R17 full_plus5 (full fit)", v2g, P=P)
    print("  " + by_len(y, P), flush=True)
