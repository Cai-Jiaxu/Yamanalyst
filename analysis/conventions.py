"""What does the ground truth mean, in terms of the raw tracking? Public 5 minutes only.

    uv run python analysis/conventions.py

Prints aggregates only (rates, medians, frame offsets) -- never tracking rows.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402
from mtah.score import load_gt, pm  # noqa: E402

H, A, D, X = (pm.LABELS.index(c) for c in "HADX")
W0, W1 = 11185, 18685


def nearest(m):
    """Per frame: distance (m) to nearest home player, nearest away player."""
    L, Wd = m.pitch
    px, py = m.players[..., 0] * L, m.players[..., 1] * Wd
    bx, by = m.ball[:, 0:1] * L, m.ball[:, 1:2] * Wd
    d = np.hypot(px - bx, py - by)
    dh = np.nanmin(np.where(m.team[None] == 0, d, np.nan), axis=1)
    da = np.nanmin(np.where(m.team[None] == 1, d, np.nan), axis=1)
    return dh, da


def pct(x):
    return "{:5.1f}%".format(100 * np.mean(x)) if len(x) else "  n/a "


def main():
    import warnings
    warnings.filterwarnings("ignore")
    m = Match()
    gt = load_gt()
    L, Wd = m.pitch
    seen = np.isfinite(m.ball[:, 0])
    dh, da = nearest(m)
    bx, by = m.ball[:, 0] * L, m.ball[:, 1] * Wd
    out = (m.ball[:, 0] < 0) | (m.ball[:, 0] > 1) | (m.ball[:, 1] < 0) | (m.ball[:, 1] > 1)
    pspeed = np.nanmean(m.players[..., 2], axis=1)
    # the ATD ball speed column is empty; derive it from positions (m/s, centred difference)
    bspeed = np.full(m.n, np.nan)
    bspeed[1:-1] = np.hypot(bx[2:] - bx[:-2], by[2:] - by[:-2]) * m.fps / 2
    print("ATD ball speed column finite on {} of frames".format(pct(np.isfinite(m.ball[:, 2]))))
    ntracked = np.isfinite(m.players[..., 0]).sum(axis=1)
    win = np.zeros(m.n, bool)
    win[W0:W1] = True

    print("== per GT class, public window")
    print("class  frames  ball_seen  ball_out_of_pitch  ball_speed_med  player_speed_med  n_tracked")
    for c, name in ((H, "H"), (A, "A"), (D, "D")):
        s = win & (gt == c)
        print("  {}    {:5d}   {}      {}            {:5.2f}           {:5.2f}          {:4.1f}".format(
            name, s.sum(), pct(seen[s]), pct(out[s & seen]), np.nanmedian(bspeed[s]),
            np.nanmedian(pspeed[s]), np.mean(ntracked[s])))

    print("\n== nearest-player team vs GT team (alive, ball seen)")
    alive = win & np.isin(gt, (H, A)) & seen
    near_team = np.where(dh < da, H, A)
    dmin = np.minimum(dh, da)
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 99)):
        s = alive & (dmin >= lo) & (dmin < hi)
        print("  nearest {:>2}-{:<2} m: {:5d} frames, agrees {}".format(
            lo, hi, s.sum(), pct(near_team[s] == gt[s])))
    print("  overall: agrees {}".format(pct(near_team[alive] == gt[alive])))

    print("\n== transitions in the public window (offsets in frames, + = data event after GT)")
    trans = [t for t in range(W0 + 1, W1) if gt[t] != gt[t - 1]]
    for t in trans:
        a, b = pm.LABELS[gt[t - 1]], pm.LABELS[gt[t]]
        seg = slice(t - 75, t + 76)
        note = ""
        if b in "HA" and a in "HA":
            # first frame near the transition where the new team is within 1.5 m of the ball
            new_d = (dh if b == "H" else da)[seg]
            hits = np.flatnonzero(new_d < 1.5)
            first = hits[np.argmin(np.abs(hits - 75))] - 75 if len(hits) else None
            note = "new team <1.5m nearest at {!s:>4}  ball seen at t: {}".format(
                str(first), bool(seen[t]))
        elif b == "D":
            o = np.flatnonzero(out[seg] & seen[seg])
            note = "ball out-of-pitch nearest at {!s:>4}  ball seen at t: {}  vis +-1s {}".format(
                str(o[np.argmin(np.abs(o - 75))] - 75) if len(o) else None, bool(seen[t]),
                pct(seen[t - 25:t + 25]))
        elif a == "D":
            mv = np.flatnonzero((bspeed[seg] > 3) & seen[seg])
            near = np.minimum(dh, da)[seg]
            nd = np.flatnonzero(near < 1.5)
            note = "ball speed>3 nearest at {!s:>4}  player<1.5m nearest at {!s:>4}  ball seen {}".format(
                str(mv[np.argmin(np.abs(mv - 75))] - 75) if len(mv) else None,
                str(nd[np.argmin(np.abs(nd - 75))] - 75) if len(nd) else None, bool(seen[t]))
        print("  {}->{}  {}".format(a, b, note))

    print("\n== tactical camera centre as ball proxy (whole match, ball seen, graded)")
    g = m.graded_mask() & seen & np.isfinite(m.cam[:, 0])
    ex = (m.cam[g, 0] - m.ball[g, 0]) * L
    ey = (m.cam[g, 1] - m.ball[g, 1]) * Wd
    print("  |dx| median {:.1f} m, p90 {:.1f} m;  |dy| median {:.1f} m, p90 {:.1f} m;  corr x {:.3f}"
          .format(np.median(np.abs(ex)), np.percentile(np.abs(ex), 90), np.median(np.abs(ey)),
                  np.percentile(np.abs(ey), 90), np.corrcoef(m.cam[g, 0], m.ball[g, 0])[0, 1]))

    print("\n== ball gaps (graded frames)")
    gr = m.graded_mask()
    miss = gr & ~seen
    edges = np.diff(np.concatenate(([0], miss.astype(int), [0])))
    lens = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    for lo, hi in ((1, 5), (5, 25), (25, 75), (75, 250), (250, 10 ** 7)):
        s = (lens >= lo) & (lens < hi)
        print("  gap {:>4}-{:<8} frames: {:5d} gaps, {:5.1f}% of missing frames".format(
            lo, hi, s.sum(), 100 * lens[s].sum() / lens.sum()))


if __name__ == "__main__":
    main()
