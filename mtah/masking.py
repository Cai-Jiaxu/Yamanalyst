"""Masked validation: hide the ball in long stretches of the public window, the way the full
match loses it, and see what a method still recovers.

analysis/conventions.py: 85% of missing-ball frames in graded play sit in gaps > 10 s, and later
5-minute blocks lose 40-80% of the ball. The public window loses ~10%, so it barely tests that
regime. Gap lengths here are drawn log-uniform between 4 and 40 s until ~45% of the window is
hidden.
"""
import numpy as np

W0, W1 = 11185, 18685


def ball_mask(n, seed, frac=0.45, lo_s=4, hi_s=40, fps=25):
    rng = np.random.default_rng(seed)
    hide = np.zeros(n, dtype=bool)
    span = W1 - W0
    while hide[W0:W1].mean() < frac:
        length = int(np.exp(rng.uniform(np.log(lo_s), np.log(hi_s))) * fps)
        a = W0 + rng.integers(0, span - length)
        hide[a:a + length] = True
    return hide


def masked_scores(method, m, compute, seeds=range(8)):
    """Mean and per-seed spellF1/macroF1 of method(m, f) with the ball hidden."""
    from mtah.score import load_gt, score
    gt = load_gt()
    out = []
    for s in seeds:
        f = compute(m, hide=ball_mask(m.n, s))
        r, _ = score(method(m, f), gt)
        out.append((r["spellF1"], r["macroF1"]))
    out = np.array(out)
    return out.mean(axis=0), out
