"""One-line report for a full-match label array: public-window scores plus full-match shape.

The full-match numbers have no ground truth behind them; they are sanity checks. The public
window shows about 3.4 possessions per minute of play with a mean of 14 s, and a full match should
land in that neighbourhood -- a count far above it means flicker.
"""

import numpy as np

from mtah.score import load_gt, score

_GT = None


def spells(labels, runs):
    out = []
    for a, b in runs:
        w = labels[a:b]
        # forward-fill X like the scorer does
        last = ""
        filled = []
        for c in w:
            if c != "X":
                last = c
            filled.append(last)
        w = np.array(filled)
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], br, [len(w)]))
        for i in range(len(bounds) - 1):
            if w[bounds[i]] in ("H", "A"):
                out.append(bounds[i + 1] - bounds[i])
    return np.array(out)


def report(name, labels, m):
    global _GT
    if _GT is None:
        _GT = load_gt()
    s, _ = score(labels, _GT)
    runs = [(a, b + 1) for _, a, b in m.halves]
    sp = spells(labels, runs)
    graded = m.graded_mask()
    minutes = graded.sum() / m.fps / 60
    lab = labels[graded]
    print("{:<28} public spellF1 {:.4f} macroF1 {:.4f} poss {:>2}/{:<2} | full: {:>4} poss "
          "({:.2f}/min, mean {:4.1f} s, <1s {:>3})  H {:.0%} A {:.0%} D {:.0%}".format(
              name, s["spellF1"], s["macroF1"], s["n_pred"], s["n_gt"], len(sp),
              len(sp) / minutes, sp.mean() / m.fps if len(sp) else 0, int((sp < m.fps).sum()),
              np.mean(lab == "H"), np.mean(lab == "A"), np.mean(lab == "D")))
    return s
