"""Ball-free features: who has the ball, read from the shape and motion of the 22 players alone.

When the tracker loses the ball (41% of graded frames, mostly in long gaps) the players still
tell the story: the team in possession stretches and pushes up, the defending team compacts and
drops, everyone slows down at a stoppage. Every feature here is computed without the ball, so the
model trained on them works identically whether or not the ball is visible.

Coordinates are flipped per half so that +x is always the direction the HOME team attacks;
"home attacks +x" is decided from the teams' mean positions over the half (each side spends
most of the match in its own half's shape).
"""

import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d

SMOOTH = (25, 75)     # frames: +-0.5 s and +-1.5 s context windows
TREND = 25            # frames: motion measured over +-1 s
MAX_PLAYER_SPEED = 11.5  # m/s; faster per-slot motion is a track identity switch, not running


def _nan_stat(fn, x, mask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(np.where(mask, x, np.nan), axis=1)


def attack_sign(m):
    """(n,) +1/-1: multiply x by this so home attacks +x. 0 outside the halves."""
    sign = np.zeros(m.n)
    home = (m.team == 0)[None]
    x = m.players[..., 0]
    for _, a, b in m.halves:
        mh = np.nanmean(np.where(home, x[a:b + 1], np.nan))
        ma = np.nanmean(np.where(~home, x[a:b + 1], np.nan))
        sign[a:b + 1] = 1.0 if mh < ma else -1.0
    return sign


def _smooth(x, w):
    ok = np.isfinite(x).astype(float)
    num = uniform_filter1d(np.nan_to_num(x), w)
    den = uniform_filter1d(ok, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.3] = np.nan
    return out


def compute(m, cam=True):
    """(n, k) float32 feature matrix and the list of names. cam=False drops the tactical-camera
    columns, for sources that have no camera (external datasets) or models that must transfer."""
    L, W = m.pitch
    sgn = attack_sign(m)[:, None]
    x = (m.players[..., 0] - 0.5) * L * sgn      # metres from halfway, home attacks +x
    y = (m.players[..., 1] - 0.5) * W
    sp = m.players[..., 2].astype(np.float64)
    home = (m.team == 0)[None]
    ok = np.isfinite(x)

    # per-slot motion over +-TREND frames, in the home-attack frame
    vx = np.full_like(x, np.nan)
    vx[TREND:-TREND] = (x[2 * TREND:] - x[:-2 * TREND]) / (2 * TREND / m.fps)
    # a slot that changes identity inside the window produces an impossible speed; drop it
    vx[np.abs(vx) > MAX_PLAYER_SPEED] = np.nan

    cols = {}
    for name, mask, s in (("h", home & ok, 1.0), ("a", ~home & ok, -1.0)):
        # s flips away-team quantities into ITS attacking direction, so 'adv' always means
        # 'towards the goal this team attacks'
        cols[name + "_cx"] = _nan_stat(np.nanmean, x * s, mask)
        cols[name + "_cy"] = _nan_stat(np.nanmean, y, mask)
        cols[name + "_sx"] = _nan_stat(np.nanstd, x, mask)
        cols[name + "_sy"] = _nan_stat(np.nanstd, y, mask)
        cols[name + "_front"] = _nan_stat(np.nanmax, x * s, mask)
        xs = np.sort(np.where(mask, x * s, np.inf), axis=1)
        cols[name + "_line"] = np.where(np.isfinite(xs[:, 1]), xs[:, 1], np.nan)  # 2nd deepest
        cols[name + "_speed"] = _nan_stat(np.nanmean, sp, mask)
        cols[name + "_adv"] = _nan_stat(np.nanmean, vx * s, mask)
        cols[name + "_n"] = mask.sum(axis=1).astype(float)
        cols[name + "_in_opp_half"] = (np.where(mask, x * s, -np.inf) > 0).sum(axis=1).astype(float)

    # pressing: mean distance from each away player to the nearest home player, and vice versa
    hx, hy = np.where(home & ok, x, np.nan), np.where(home & ok, y, np.nan)
    ax, ay = np.where(~home & ok, x, np.nan), np.where(~home & ok, y, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dd = np.hypot(hx[:, :, None] - ax[:, None, :], hy[:, :, None] - ay[:, None, :])
        cols["press_on_a"] = np.nanmean(np.nanmin(dd, axis=1), axis=1)
        cols["press_on_h"] = np.nanmean(np.nanmin(dd, axis=2), axis=1)

    cols["cam_x"] = (m.cam[:, 0] - 0.5) * L * sgn[:, 0]
    cols["cam_y"] = (m.cam[:, 1] - 0.5) * W
    cam_vx = np.full(m.n, np.nan)
    cam_vx[TREND:-TREND] = (cols["cam_x"][2 * TREND:] - cols["cam_x"][:-2 * TREND]) / 2.0
    cols["cam_vx"] = cam_vx

    # differences carry most of the signal; let the trees see them directly
    for k in ("cx", "sx", "sy", "front", "line", "speed", "adv", "in_opp_half"):
        cols["d_" + k] = cols["h_" + k] - cols["a_" + k]
    cols["d_press"] = cols["press_on_a"] - cols["press_on_h"]

    base = list(cols)
    for w in SMOOTH:
        for k in base:
            cols["{}_m{}".format(k, w)] = _smooth(cols[k], 2 * w + 1)
    names = [k for k in cols if cam or not k.startswith("cam")]
    return np.stack([cols[k] for k in names], axis=1).astype(np.float32), names
