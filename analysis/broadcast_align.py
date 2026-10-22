"""Feasibility: can the broadcast ATD be aligned to the tactical timeline, and how much of the
tactical ball gap would the broadcast ball fill?

    uv run python analysis/broadcast_align.py

Alignment by player configuration: for a sampled broadcast frame, the best tactical frame is the
one minimising the mean distance from each visible broadcast player to the nearest tactical
player of the same team (a one-sided chamfer distance, in metres). Live broadcast footage should
match to well under 2 m; replays and close-ups will not match anything.

Prints aggregates only.
"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402

warnings.filterwarnings("ignore")
SAMPLE = 125        # broadcast frames between samples (5 s)
COARSE = 5          # tactical candidate stride for the global search
GOOD = 1.5          # m, a chamfer this low is a confident match


def team_points(m, i):
    P = m.players[i]
    ok = np.isfinite(P[:, 0])
    L, W = m.pitch
    return [np.stack([P[ok & (m.team == t), 0] * L, P[ok & (m.team == t), 1] * W], 1)
            for t in (0, 1)]


def chamfer_many(bpts, tac, cand):
    """Mean nearest-same-team distance from broadcast points to each candidate tactical frame."""
    L, W = tac.pitch
    total, count = np.zeros(len(cand)), 0
    for t, pts in enumerate(bpts):
        if not len(pts):
            continue
        tx = tac.players[cand][:, tac.team == t, 0] * L
        ty = tac.players[cand][:, tac.team == t, 1] * W
        d = np.hypot(tx[:, None, :] - pts[None, :, 0, None], ty[:, None, :] - pts[None, :, 1, None])
        total += np.nansum(np.nanmin(d, axis=2), axis=1)
        count += len(pts)
    return total / max(count, 1)


def main():
    tac = Match(source="tactical")
    bro = Match(source="broadcast")
    graded = tac.graded_mask()
    lo, hi = np.flatnonzero(graded)[[0, -1]]
    cand_all = np.arange(lo - 3000, hi + 3000, COARSE)

    rows = []
    for b in range(0, bro.n, SAMPLE):
        if not bro.hvalid[b]:
            continue
        pts = team_points(bro, b)
        if sum(len(p) for p in pts) < 8:
            continue
        c = chamfer_many(pts, tac, cand_all)
        j = np.nanargmin(c)
        fine = np.arange(cand_all[j] - COARSE, cand_all[j] + COARSE + 1)
        cf = chamfer_many(pts, tac, fine)
        k = np.nanargmin(cf)
        rows.append((b, fine[k], cf[k]))
    rows = np.array(rows)
    good = rows[rows[:, 2] < GOOD]
    print("broadcast samples tried {} | confident matches (<{} m) {} ({:.0%}) | median chamfer "
          "of confident {:.2f} m".format(len(rows), GOOD, len(good), len(good) / len(rows),
                                         np.median(good[:, 2])))
    off = good[:, 1] - good[:, 0]
    # piecewise-constant offsets: how many distinct offset regimes (cuts/replays) are there?
    d = np.abs(np.diff(off))
    print("offset (tactical - broadcast) range {:.0f}..{:.0f} frames; consecutive confident "
          "samples agree within 3 frames {:.0%} of the time".format(off.min(), off.max(),
                                                                     np.mean(d <= 3)))

    # coverage: for each confident sample, propagate its offset +-SAMPLE/2 frames and ask whether
    # the broadcast ball fills tactical frames where the tactical ball is missing
    t_seen = np.isfinite(tac.ball[:, 0])
    b_seen = np.isfinite(bro.ball[:, 0])
    covered = np.zeros(tac.n, dtype=bool)
    filled = np.zeros(tac.n, dtype=bool)
    for b, t, _ in good.astype(int):
        for k in range(-SAMPLE // 2, SAMPLE // 2):
            bb, tt = b + k, t + k
            if 0 <= bb < bro.n and 0 <= tt < tac.n:
                covered[tt] = True
                if b_seen[bb] and not t_seen[tt]:
                    filled[tt] = True
    miss = graded & ~t_seen
    print("graded frames covered by a confident alignment: {:.0%}".format(covered[graded].mean()))
    print("tactical ball missing on {:.1%} of graded frames; broadcast fills {:.1%} of those "
          "-> missing would fall to {:.1%}".format(miss.sum() / graded.sum(),
                                                  filled[miss].mean(),
                                                  (miss & ~filled).sum() / graded.sum()))


if __name__ == "__main__":
    main()
