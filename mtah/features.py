"""Per-frame physical features from the cached tracking. Metres and m/s throughout.

    f = compute(Match())      # dict of (n,) arrays

Ball cleaning: points implying > MAX_BALL_SPEED are dropped as detector glitches, then gaps of at
most MAX_INTERP frames are filled linearly. Long gaps stay NaN -- the analysis in
analysis/conventions.py shows 85% of missing ball frames sit in gaps over 10 s, which no
interpolation can bridge honestly.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

# Params
MAX_BALL_SPEED = 45.0     # m/s; a hard shot is ~35
MAX_INTERP = 12           # frames (~0.5 s) of ball gap filled by straight line
SPEED_SMOOTH = 5          # frames, centred window for ball speed
OUT_MARGIN = 0.3          # m past a line before the ball counts as out


def _interp_gaps(x, seen, max_gap):
    """Linear fill of NaN runs no longer than max_gap that have data on both sides."""
    x = x.copy()
    idx = np.flatnonzero(seen)
    if len(idx) < 2:
        return x, seen
    gaps = np.diff(idx)
    filled = seen.copy()
    for k in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
        a, b = idx[k], idx[k + 1]
        t = np.arange(1, b - a) / (b - a)
        x[a + 1:b] = x[a][None] + t[:, None] * (x[b] - x[a])[None]
        filled[a + 1:b] = True
    return x, filled


def clean_ball(m, hide=None):
    L, W = m.pitch
    xy = np.stack([m.ball[:, 0] * L, m.ball[:, 1] * W], axis=1).astype(np.float64)
    if hide is not None:
        xy[hide] = np.nan
    vball = getattr(m, "video_ball", None)
    if vball is not None:
        miss = ~np.isfinite(xy[:, 0]) & np.isfinite(vball[:, 0])
        xy[miss] = vball[miss]
    seen = np.isfinite(xy[:, 0])
    # drop isolated teleports: a point far from both neighbours is a false detection
    for _ in range(2):
        prev = np.vstack([xy[:1], xy[:-1]])
        nxt = np.vstack([xy[1:], xy[-1:]])
        jump_prev = np.hypot(*(xy - prev).T) * m.fps
        jump_next = np.hypot(*(xy - nxt).T) * m.fps
        bad = seen & (np.nan_to_num(jump_prev) > MAX_BALL_SPEED) & \
            (np.nan_to_num(jump_next) > MAX_BALL_SPEED)
        xy[bad] = np.nan
        seen = np.isfinite(xy[:, 0])
    xy, filled = _interp_gaps(xy, seen, MAX_INTERP)
    return xy, seen, filled


def compute(m, hide=None):
    """hide: optional boolean (n,) of frames whose ball observation is deleted first -- used to
    simulate the full match's missing-ball regime on the well-tracked public window."""
    L, W = m.pitch
    fps = m.fps
    ball, seen_raw, seen = clean_ball(m, hide)
    bx, by = ball[:, 0], ball[:, 1]

    v = np.full((m.n, 2), np.nan)
    v[1:-1] = (ball[2:] - ball[:-2]) * fps / 2
    vs = np.where(np.isfinite(v), v, 0.0)
    ok = np.isfinite(v[:, 0]).astype(float)
    norm = uniform_filter1d(ok, SPEED_SMOOTH)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_s = uniform_filter1d(vs, SPEED_SMOOTH, axis=0) / norm[:, None]
    v_s[norm < 0.5] = np.nan
    bspeed = np.hypot(v_s[:, 0], v_s[:, 1])

    # direction change of the ball (deg/frame-ish proxy for touches): angle between v[t-3], v[t+3]
    a = np.full((m.n, 2), np.nan)
    b = np.full((m.n, 2), np.nan)
    a[3:] = v_s[:-3]
    b[:-3] = v_s[3:]
    dv = np.hypot(*(b - a).T)  # m/s change across ~0.25 s

    px, py = m.players[..., 0] * L, m.players[..., 1] * W
    d = np.hypot(px - bx[:, None], py - by[:, None])
    home = (m.team == 0)[None]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dh = np.nanmin(np.where(home, d, np.nan), axis=1)
            da = np.nanmin(np.where(~home, d, np.nan), axis=1)
            nh2 = np.nansum(np.where(home, d < 2, False), axis=1)
            na2 = np.nansum(np.where(~home, d < 2, False), axis=1)
            sp = m.players[..., 2]
            sh = np.nanmean(np.where(home, sp, np.nan), axis=1)
            sa = np.nanmean(np.where(~home, sp, np.nan), axis=1)
            cxh = np.nanmean(np.where(home, px, np.nan), axis=1)
            cxa = np.nanmean(np.where(~home, px, np.nan), axis=1)
            cyh = np.nanmean(np.where(home, py, np.nan), axis=1)
            cya = np.nanmean(np.where(~home, py, np.nan), axis=1)
            sdh = np.nanstd(np.where(home, px, np.nan), axis=1) + \
                np.nanstd(np.where(home, py, np.nan), axis=1)
            sda = np.nanstd(np.where(~home, px, np.nan), axis=1) + \
                np.nanstd(np.where(~home, py, np.nan), axis=1)
            nplayers = np.isfinite(px).sum(axis=1)

    out = (bx < -OUT_MARGIN) | (bx > L + OUT_MARGIN) | (by < -OUT_MARGIN) | (by > W + OUT_MARGIN)
    return {
        "bx": bx, "by": by, "seen": seen, "seen_raw": seen_raw,
        "vx": v_s[:, 0], "vy": v_s[:, 1], "bspeed": bspeed, "dv": dv,
        "dh": dh, "da": da, "nh2": nh2, "na2": na2,
        "sh": sh, "sa": sa, "cxh": cxh, "cxa": cxa, "cyh": cyh, "cya": cya,
        "sdh": sdh, "sda": sda, "nplayers": nplayers,
        "out": out & seen,
        "camx": m.cam[:, 0] * L, "camy": m.cam[:, 1] * W,
    }
