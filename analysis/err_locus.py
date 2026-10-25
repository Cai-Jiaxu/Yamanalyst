"""Where are the team errors: near a true change, or mid-possession? (aggregates only)

    uv run python analysis/err_locus.py

For each labelled window (public GT + the R3 2nd-half windows), live frames are binned by their
distance to the nearest true label change. Mid-possession errors (> 3 s from any change) do not
depend on where a labeller puts a boundary, so they separate genuine team confusion from
boundary convention/lag.
"""
import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import decode, final, rules  # noqa: E402
from mtah.data import CACHE_DIR, Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.masking import W0, W1  # noqa: E402
from mtah.score import load_gt, pm  # noqa: E402

warnings.filterwarnings("ignore")
BINS = [(0, 25), (25, 75), (75, 10 ** 9)]


def dist_to_change(lab):
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    idx = np.arange(len(lab))
    if not len(ch):
        return np.full(len(lab), 10 ** 9)
    j = np.clip(np.searchsorted(ch, idx), 1, len(ch) - 1)
    return np.minimum(np.abs(idx - ch[j - 1]), np.abs(idx - ch[np.minimum(j, len(ch) - 1)]))


def main():
    m = Match()
    f = compute(m)
    P = dict(np.load(os.path.join(CACHE_DIR, "P_R1.npz")))["public"]
    R = rules.run(m, f)
    p2 = dict(decode.PARAMS, **dict(final.DECODER, use_rules=False, stop_gate=0))
    p4 = dict(decode.PARAMS, **final.DECODER)
    nox = np.full(m.n, "X", dtype="<U1")
    preds = {"model(v2)": decode.decode(m, nox, P, f["seen"], p2, f),
             "v4": decode.decode(m, R, P, f["seen"], p4, f),
             "rules": R}
    wins = [("public", W0, W1, np.array(pm.LABELS)[load_gt()])]
    for path in sorted(glob.glob(os.path.join(ROOT, "work", "labels", "val_*.csv"))):
        lab = pm.read_label_csv(path)["label"].astype(str).str.strip().str.upper().values
        idx = np.flatnonzero(lab != "X")
        wins.append((os.path.basename(path)[7:-4], idx.min(), idx.max() + 1, lab))
    print("live-frame team error by distance to nearest true change (<1 s | 1-3 s | >3 s); "
          "share of live frames in each bin first")
    for name, a, b, lab in wins:
        g = lab[a:b]
        d = dist_to_change(g)
        live = np.isin(g, ["H", "A"])
        shares = [np.mean(live & (d >= lo) & (d < hi)) / live.mean() for lo, hi in BINS]
        line = "{:<8} share {:.2f}/{:.2f}/{:.2f} |".format(name, *shares)
        for k, pr in preds.items():
            q = pr[a:b]
            errs = []
            for lo, hi in BINS:
                s = live & (d >= lo) & (d < hi) & np.isin(q, ["H", "A"])
                errs.append(np.mean(q[s] != g[s]) if s.any() else np.nan)
            dead_as_live = np.mean(np.isin(q[g == "D"], ["H", "A"])) if (g == "D").any() else np.nan
            line += " {} {:.3f}/{:.3f}/{:.3f} D->live {:.2f} |".format(k, *errs, dead_as_live)
        print(line, flush=True)


if __name__ == "__main__":
    main()
