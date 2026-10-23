"""Round 3 fast screen (ROADMAP rule 7): R8 velocity pitch control, R5 CatBoost, against the
same-tier baseline. 9 base matches, no augmentation, 300 trees; models cached per key.

    uv run --no-sync python analysis/round3.py      # --no-sync keeps catboost (not in uv.lock)

A candidate more than 0.005 below the screen baseline is dropped; the rest get a full fit.
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
    for name, featfn, mp, key in (
            ("screen: full_plus3 (baseline)", feats.full_plus3, {}, None),
            ("screen: R8 full_plus4", feats.full_plus4, {}, None),
            ("screen: R5 catboost", feats.full_plus3, {"family": "catboost"}, "full_plus3_cb"),
            ("screen: R5 hgb + catboost", feats.full_plus3, {"family": "both"},
             "full_plus3_both")):
        t = time.time()
        b.screen(name, featfn, final.DECODER, model_params=mp, key=key)
        print("  ({:.0f} s)".format(time.time() - t), flush=True)


if __name__ == "__main__":
    main()
