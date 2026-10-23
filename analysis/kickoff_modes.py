"""ROADMAP R1: kickoff evidence modes on final_v1's cached fold probabilities (cache/P_R1.npz).

    uv run python analysis/kickoff_modes.py

Decoder-only, so it reuses the fit. The bench cannot see the main target (the 2nd half's 2,934
pre-kickoff frames: no external match and no public frame is before a half's kickoff), so it
also prints, for our full match, the label spells and H/A share before the 2nd-half kickoff.
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
KICK = 113742   # FAQ.md: the 2nd half is graded from 110,808 but kicks off here


def main():
    b = Bench()
    z = np.load(os.path.join(CACHE_DIR, "P_R1.npz"))
    P = {k: z[k] for k in z.files}
    for mode in (False, "half", "all"):
        b.score("kickoff={}".format(mode), P, dict(final.DECODER, kickoff=mode),
                use_rules=False, t0=time.time())
    # our full match, 2nd half before the kickoff: full-match model and decoder as in run.py
    import pandas as pd
    from mtah import decode
    from mtah.data import Match
    from mtah.features import compute
    m = Match()
    f = compute(m)
    lab = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "final_v1.csv"))["label"].values
    s = next(a for h, a, _ in m.halves if h == 2)
    seg = lab[s:KICK]
    print("final_v1 before the 2nd-half kickoff: {} spells, H/A share {:.0%}".format(
        1 + int((seg[1:] != seg[:-1]).sum()), np.isin(seg, ["H", "A"]).mean()))
    from mtah import kickoff
    E = kickoff.evidence(m, f, mode="half")
    print("'half' evidence forces D on {} frames of that stretch ({} frames long)".format(
        int((E[s:KICK, 2] > 0).sum()), KICK - s))


if __name__ == "__main__":
    main()
