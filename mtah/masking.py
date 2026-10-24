"""Masked validation: hide the ball in long stretches of the public window, the way the full
match loses it, and see what a method still recovers.

analysis/conventions.py: 85% of missing-ball frames in graded play sit in gaps > 10 s, and later
5-minute blocks lose 40-80% of the ball. The public window loses ~10%, so it barely tests that
regime. Gap lengths here are drawn log-uniform between 4 and 40 s until ~45% of the window is
hidden.

Two guards (from the code audit): the ball is also hidden for TAIL frames after the window, so
look-ahead features ('frames until the ball is seen', the dead-ball restart scan) cannot peek at
the well-tracked frames just past it; and gap starts can be weighted (e.g. by our own players'
slow pace, no labels involved), since real ball loss clusters in stoppages.
"""
import numpy as np

W0, W1 = 11185, 18685
TAIL = 750


def ball_mask(n, seed, frac=0.45, lo_s=4, hi_s=40, fps=25, weights=None, tail=TAIL,
              window=(W0, W1)):
    """weights: optional (n,) non-negative preference for where gaps start (window part used)."""
    w0, w1 = window
    rng = np.random.default_rng(seed)
    hide = np.zeros(n, dtype=bool)
    span = w1 - w0
    w = None
    if weights is not None:
        w = np.asarray(weights[w0:w1], dtype=float) + 1e-9
        w = w / w.sum()
    while hide[w0:w1].mean() < frac:
        length = int(np.exp(rng.uniform(np.log(lo_s), np.log(hi_s))) * fps)
        if w is None:
            a = w0 + rng.integers(0, span - length)
        else:
            a = w0 + min(int(rng.choice(span, p=w)), span - length)
        hide[a:a + length] = True
    hide[w1:min(n, w1 + tail)] = True
    return hide


def pace_weights(f, slow=1.3, bias=3.0):
    """Gap-start weights from our own players' pace: slow (stoppage-like) frames x bias."""
    from scipy.ndimage import uniform_filter1d
    pace = uniform_filter1d(np.nan_to_num((f["sh"] + f["sa"]) / 2), 25)
    return np.where(pace < slow, bias, 1.0)


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
