"""ROADMAP R8 full fit: velocity-projected pitch control (feats.full_plus4) with the final_v1
recipe (600 trees, mirrored + extra-noise copies). It passed the fast screen (round3.py).

    uv run python analysis/r8_full.py
"""
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import feats, final  # noqa: E402
from mtah.bench import Bench  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    b = Bench()
    featfn = feats.full_plus4
    extra = {k: feats.mirrored(featfn).get(k, []) + feats.augmented(featfn).get(k, [])
             for k in b.ext}
    t = time.time()
    P = b.probs(featfn, final.MODEL, extra=extra, key="full_plus4")
    print("fitted in {:.0f} s".format(time.time() - t), flush=True)
    b.score("R8 full_plus4 (full fit)", P, final.DECODER, use_rules=False, t0=time.time())


if __name__ == "__main__":
    main()
