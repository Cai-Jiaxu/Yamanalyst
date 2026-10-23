"""ROADMAP R7: minimum dead-ball length inside the Viterbi search (decode.viterbi_dwell) against
post-hoc absorption (min_dead), on final_v1's cached fold probabilities (cache/P_R1.npz).

    uv run python analysis/dwell.py

Decoder-only. A small, fixed grid (no search), so the comparison is not tuned to the bench.
"""
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import final  # noqa: E402
from mtah.bench import Bench  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    b = Bench()
    z = np.load(os.path.join(CACHE_DIR, "P_R1.npz"))
    P = {k: z[k] for k in z.files}
    for name, change in (("final_v2 (post-hoc min_dead 25)", {}),
                         ("dwell 25 in Viterbi", {"dwell": 25, "min_dead": 0}),
                         ("dwell 50 in Viterbi", {"dwell": 50, "min_dead": 0}),
                         ("dwell 25, cost_dead 9", {"dwell": 25, "min_dead": 0, "cost_dead": 9.0}),
                         ("dwell 25 + post-hoc min_dead 25", {"dwell": 25})):
        b.score(name, P, dict(final.DECODER, **change), use_rules=False, t0=time.time())


if __name__ == "__main__":
    main()
