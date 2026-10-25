"""Production feature sets for the possession model (built in analysis/experiments.py).

    X = full_plus2(match, f, Xs)   # f = mtah.features.compute(match), Xs = shape.compute(m, cam=False)

full_plus2 = long-context ball-free shape + ball/touch block + rule label one-hot + rule
internals + extra ball geometry (k-nearest, approach speed, pitch control). mirrored() and
augmented() build the training-only copies of external matches (left/right and home/away
mirrors; extra noise draws at three ball-loss levels).
"""
import os
import warnings

import numpy as np

from mtah import shape

warnings.filterwarnings("ignore")
_, NAMES = None, None


def base_columns():
    """Indices of the un-smoothed shape columns (cam=False layout)."""
    global NAMES
    if NAMES is None:
        from mtah.data import Match
        _, NAMES = shape.compute(Match(), cam=False)
    return [i for i, k in enumerate(NAMES) if "_m" not in k]


KEY = ("d_cx", "d_sx", "d_line", "d_adv", "d_front", "d_in_opp_half", "d_press", "h_speed", "a_speed")


def _shift(x, k):
    """x moved by k frames (positive = later), edge-clamped -- unlike np.roll, the match end
    never wraps onto its start."""
    idx = np.clip(np.arange(len(x)) - k, 0, len(x) - 1)
    return x[idx]


def long_context(m, f, Xs):
    """+-5 s and +-10 s means of every base column, and lead-minus-lag trends of key columns."""
    cols = [Xs]
    base = Xs[:, base_columns()]
    for w in (125, 250):
        cols.append(np.stack([shape._smooth(base[:, j], 2 * w + 1) for j in range(base.shape[1])], 1))
    keys = [NAMES.index(k) for k in KEY]
    for w in (50, 125):
        for j in keys:
            s = shape._smooth(Xs[:, j], w + 1)
            lead = _shift(s, -(w // 2))
            lag = _shift(s, w // 2)
            cols.append((lead - lag)[:, None])
    return np.hstack(cols).astype(np.float32)


def ball_block(m, f):
    """Ball and touch features in the home-attack frame; NaN where the ball is not seen."""
    sgn = shape.attack_sign(m)
    L, W = m.pitch
    seen = f["seen"]
    bx = np.where(seen, (f["bx"] - L / 2) * sgn, np.nan)
    by = np.where(seen, f["by"] - W / 2, np.nan)
    vx = np.where(seen, f["vx"] * sgn, np.nan)
    dh = np.where(seen, f["dh"], np.nan)
    da = np.where(seen, f["da"], np.nan)
    # frames since the ball was last seen, and the nearest-team margin when it was
    idx = np.where(seen, np.arange(m.n), -1)
    last = np.maximum.accumulate(idx)
    since = np.where(last >= 0, np.arange(m.n) - last, 10 ** 5).astype(float)
    margin = np.where(seen, da - dh, np.nan)
    last_margin = np.where(last >= 0, margin[np.maximum(last, 0)], np.nan)
    cols = [bx, by, vx, f["bspeed"], dh, da, margin, f["nh2"].astype(float),
            f["na2"].astype(float), since, last_margin, seen.astype(float)]
    for w in (12, 50):
        cols += [shape._smooth(margin, 2 * w + 1), shape._smooth(bx, 2 * w + 1)]
    return np.stack(cols, 1)


def rule_onehot(R):
    return np.stack([(R == c).astype(np.float32) for c in ("H", "A", "D")], 1)


def make_ball(with_rules_feature):
    def fn(m, f, Xs):
        from mtah import rules
        cols = [Xs, ball_block(m, f)]
        if with_rules_feature:
            cols.append(rule_onehot(rules.run(m, f)))
        return np.hstack(cols).astype(np.float32)
    return fn


def full(m, f, Xs):
    """E6: long-context shape + ball/touch block + rule label one-hot."""
    from mtah import rules
    return np.hstack([long_context(m, f, Xs), ball_block(m, f),
                      rule_onehot(rules.run(m, f))]).astype(np.float32)


def _since(event, cap=750):
    """Frames since the last True in event (capped), and until the next one."""
    n = len(event)
    idx = np.where(event, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    since = np.where(last >= 0, np.arange(n) - last, cap)
    idx2 = np.where(event, np.arange(n), n + cap)
    nxt = np.minimum.accumulate(idx2[::-1])[::-1]
    until = np.minimum(nxt - np.arange(n), cap)
    return np.minimum(since, cap).astype(np.float32), until.astype(np.float32)


def touch_block(m, f, R):
    """The rule system's internals as features, so the model can learn its thresholds."""
    from mtah import rules
    from scipy.ndimage import uniform_filter1d
    ob = rules.on_ball(f, rules.PARAMS)
    cols = [(ob == "H").astype(np.float32), (ob == "A").astype(np.float32)]
    sh, uh = _since(ob == "H")
    sa, ua = _since(ob == "A")
    cols += [sh, uh, sa, ua, sh - sa, uh - ua]
    change = np.concatenate(([False], R[1:] != R[:-1]))
    sc, uc = _since(change)
    cols += [sc, uc]
    pace = uniform_filter1d(np.nan_to_num((f["sh"] + f["sa"]) / 2), 25)
    cols += [pace, uniform_filter1d(pace, 125), f["out"].astype(np.float32),
             uniform_filter1d(f["out"].astype(np.float32), 125)]
    return np.stack(cols, 1)


def ball_extra(m, f):
    """k-nearest distances, approach speeds, ball-gap look-ahead, simplified pitch control."""
    L, W = m.pitch
    seen = f["seen"]
    px, py = m.players[..., 0] * L, m.players[..., 1] * W
    d = np.hypot(px - f["bx"][:, None], py - f["by"][:, None])
    home = (m.team == 0)[None]
    dh = np.sort(np.where(home & np.isfinite(d), d, np.inf), axis=1)[:, :3]
    da = np.sort(np.where(~home & np.isfinite(d), d, np.inf), axis=1)[:, :3]
    dh[~np.isfinite(dh)] = np.nan
    da[~np.isfinite(da)] = np.nan
    # approach speed of each team's nearest player: d(distance)/dt of the per-frame minimum,
    # over +-3 frames (negative = closing in)
    def rate(x):
        r = np.full_like(x, np.nan)
        r[3:-3] = (x[6:] - x[:-6]) / (6 / m.fps)
        return r
    cols = [dh[:, 1], dh[:, 2], da[:, 1], da[:, 2], rate(dh[:, 0]), rate(da[:, 0])]
    _, until = _since(seen)
    cols.append(until)
    # simplified pitch control (Spearman-style): time to reach a point = 0.7 s reaction +
    # distance / 5 m/s; control = logistic of the arrival-time gap between the teams
    def control(tx, ty):
        tt = 0.7 + np.hypot(px - tx[:, None], py - ty[:, None]) / 5.0
        th = np.nanmin(np.where(home, tt, np.nan), axis=1)
        ta = np.nanmin(np.where(~home, tt, np.nan), axis=1)
        return 1 / (1 + np.exp(-(ta - th) / 0.45))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cols.append(np.where(seen, control(f["bx"], f["by"]), np.nan))
        cols.append(np.where(seen, control(f["bx"] + np.nan_to_num(f["vx"]),
                                           f["by"] + np.nan_to_num(f["vy"])), np.nan))
    return np.stack(cols, 1).astype(np.float32)


def full_plus2(m, f, Xs):
    """E8: E7 + ball_extra."""
    return np.hstack([full_plus(m, f, Xs), ball_extra(m, f)]).astype(np.float32)


def full_plus(m, f, Xs):
    """E7: E6 + rule internals."""
    from mtah import rules
    R = rules.run(m, f)
    return np.hstack([long_context(m, f, Xs), ball_block(m, f), rule_onehot(R),
                      touch_block(m, f, R)]).astype(np.float32)


def augmented(featfn, variants=((1, None, 3.0), (2, 0.25, 1.0), (3, 0.6, 6.0))):
    """{match: [(X, labels, mask)]}: extra noise draws per external match -- a second draw at
    our calibrated level, a milder one (25% ball lost, no dead bias), a harsher one (60%, 6x)."""
    from mtah import external, noise
    from mtah.data import CACHE_DIR
    from mtah.features import compute
    out = {}
    for name in external.available():
        em = external.ExtMatch(name)
        out[name] = []
        for k, miss, bias in variants:
            from mtah.shape_model import EXT_NOISE_VERSION
            path = os.path.join(CACHE_DIR, "aug_{}_{}_{}_{}.npz".format(
                name, k, featfn.__name__, EXT_NOISE_VERSION))
            if os.path.isfile(path):
                X = np.load(path)["X"]
            else:
                noisy = noise.degrade(em, seed=noise.seed_for(name) + 1000 * k, miss_frac=miss,
                                      dead_bias=bias)
                Xs, _ = shape.compute(noisy, cam=False)
                X = featfn(noisy, compute(noisy), Xs)
                np.savez_compressed(path, X=X)
            out[name].append((X, em.labels, em.graded_mask()))
    return out


def mirrored(featfn, kinds=("lr", "swap")):
    """{match: [(X, labels, mask)]}: mirrored copies of each external match, each degraded with
    its own noise draw, for fold-safe training augmentation."""
    from mtah import external, noise
    from mtah.data import CACHE_DIR
    from mtah.shape_model import EXT_NOISE_VERSION
    from mtah.features import compute
    out = {}
    for name in external.available():
        em = external.ExtMatch(name)
        out[name] = []
        for j, kind in enumerate(kinds):
            mm = em.mirrored(kind)
            path = os.path.join(CACHE_DIR, "mir_{}_{}_{}_{}.npz".format(
                name, kind, featfn.__name__, EXT_NOISE_VERSION))
            if os.path.isfile(path):
                X = np.load(path)["X"]
            else:
                noisy = noise.degrade(mm, seed=noise.seed_for(name) + 7919 * (j + 1))
                Xs, _ = shape.compute(noisy, cam=False)
                X = featfn(noisy, compute(noisy), Xs)
                np.savez_compressed(path, X=X)
            out[name].append((X, mm.labels, mm.graded_mask()))
    return out


def setpiece_block(m, f):
    """Set-piece geometry: who is over halfway, who is in each box, where the ball is relative to
    the lines and corners. Kickoffs, corners, goal kicks and throw-ins each have a signature."""
    from scipy.ndimage import uniform_filter1d
    L, W = m.pitch
    sgn = shape.attack_sign(m)[:, None]
    x = (m.players[..., 0] - 0.5) * L * sgn          # metres from halfway, home attacks +x
    y = (m.players[..., 1] - 0.5) * W
    ok = np.isfinite(x)
    home = (m.team == 0)[None] & ok
    away = (m.team == 1)[None] & ok
    in_box_y = np.abs(y) < 20.16
    att_box = (x > L / 2 - 16.5) & in_box_y           # the box home attacks
    own_box = (x < -L / 2 + 16.5) & in_box_y          # the box home defends
    cols = [
        (home & (x > 1.0)).sum(1), (away & (x < -1.0)).sum(1),      # over halfway
        (home & att_box).sum(1), (away & att_box).sum(1),
        (home & own_box).sum(1), (away & own_box).sum(1),
    ]
    cols = [c.astype(np.float32) for c in cols]
    cols += [uniform_filter1d(c, 51) for c in cols[:2]]
    seen = f["seen"]
    bx, by = f["bx"], f["by"]
    touch = np.minimum(by, W - by)
    goal = np.minimum(bx, L - bx)
    corner = np.hypot(goal, touch)
    cols += [np.where(seen, v, np.nan).astype(np.float32) for v in (touch, goal, corner)]
    return np.stack(cols, 1)


def full_plus3(m, f, Xs):
    """E11: full_plus2 + set-piece geometry."""
    return np.hstack([full_plus2(m, f, Xs), setpiece_block(m, f)]).astype(np.float32)


MAX_SLOT_SPEED = 11.5   # m/s; faster is an identity switch, not a sprint


def velocity_control(m, f):
    """ROADMAP R8: pitch control with each player carried 0.7 s along his own velocity before
    running (Spearman 2018), at the ball and at the ball 1 s ahead. Velocities are +-3-frame
    differences; implausible ones (identity switches) count as standing."""
    L, W = m.pitch
    seen = f["seen"]
    px, py = m.players[..., 0] * L, m.players[..., 1] * W
    vx, vy = np.zeros_like(px), np.zeros_like(py)
    vx[3:-3] = (px[6:] - px[:-6]) / (6 / m.fps)
    vy[3:-3] = (py[6:] - py[:-6]) / (6 / m.fps)
    bad = ~np.isfinite(vx) | ~np.isfinite(vy) | (np.hypot(vx, vy) > MAX_SLOT_SPEED)
    vx[bad], vy[bad] = 0.0, 0.0
    qx, qy = px + 0.7 * vx, py + 0.7 * vy
    home = (m.team == 0)[None]

    def control(tx, ty):
        tt = 0.7 + np.hypot(qx - tx[:, None], qy - ty[:, None]) / 5.0
        th = np.nanmin(np.where(home, tt, np.nan), axis=1)
        ta = np.nanmin(np.where(~home, tt, np.nan), axis=1)
        return 1 / (1 + np.exp(-(ta - th) / 0.45))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c0 = np.where(seen, control(f["bx"], f["by"]), np.nan)
        c1 = np.where(seen, control(f["bx"] + np.nan_to_num(f["vx"]),
                                    f["by"] + np.nan_to_num(f["vy"])), np.nan)
    return np.stack([c0, c1], 1).astype(np.float32)


def full_plus4(m, f, Xs):
    """R8: full_plus3 + velocity-projected pitch control."""
    return np.hstack([full_plus3(m, f, Xs), velocity_control(m, f)]).astype(np.float32)


def controlled_labels(labels, min_frames=25):
    """IDSSE -> our convention: an opposite-team spell shorter than min_frames with the SAME team
    on both sides is a touch, not a change of possession. Training labels only."""
    out = labels.copy()
    b = np.concatenate(([0], np.flatnonzero(out[1:] != out[:-1]) + 1, [len(out)]))
    for i in range(1, len(b) - 2):
        a, z = b[i], b[i + 1]
        prev, cur, nxt = out[a - 1], out[a], out[z]
        if cur in "HA" and prev == nxt and prev in "HA" and prev != cur and z - a < min_frames:
            out[a:z] = prev
    return out


def _team_code(t):
    return 1.0 if t == "H" else -1.0


def flight_block(m, f, cap=125, win=50):
    """ROADMAP R17: the touch SEQUENCE around each frame, for short possessions. The other blocks
    see touches through +-1..10 s smoothing, which averages a 1-2 s possession away. Per frame:
    team (+1 H / -1 A) and length of the last and next on-ball episode, the episode before the
    last and after the next (a second touch confirms control, CHALLENGE.md), frames since the
    last / until the next episode, episodes per team within +-2 s, and kick events (ball speed
    jump with a player close) with the kicker's team."""
    from mtah import rules
    n = m.n
    p = rules.PARAMS
    ob = rules.on_ball(f, p)
    eps = rules.episodes(ob, p, 0, n)
    t = np.arange(n)
    cols = []
    if eps:
        team = np.array([_team_code(e[0]) for e in eps], dtype=np.float32)
        st = np.array([e[1] for e in eps])
        en = np.array([e[2] for e in eps])
        ln = (en - st).astype(np.float32)
        i = np.searchsorted(st, t, "right") - 1            # last episode started at or before t
        j = i + 1                                           # next episode

        def take(v, k, fill=0.0):
            ok = (k >= 0) & (k < len(v))
            return np.where(ok, v[np.clip(k, 0, len(v) - 1)], fill).astype(np.float32)
        since = np.where(i >= 0, np.maximum(t - take(en, i, -cap), 0), cap)
        until = np.where(j < len(st), take(st, j, n + cap) - t, cap)
        cols += [take(team, i), take(ln, i), take(team, i - 1), take(team, j), take(ln, j),
                 take(team, j + 1), np.minimum(since, cap), np.minimum(until, cap)]
        hs = np.zeros(n + 1, np.float32)
        as_ = np.zeros(n + 1, np.float32)
        np.add.at(hs, st[team > 0] + 1, 1)
        np.add.at(as_, st[team < 0] + 1, 1)
        hs, as_ = np.cumsum(hs), np.cumsum(as_)
        lo, hi = np.clip(t - win, 0, n), np.clip(t + win + 1, 0, n)
        cols += [hs[hi] - hs[lo], as_[hi] - as_[lo]]
    else:
        cols += [np.zeros(n, np.float32)] * 6 + [np.full(n, cap, np.float32)] * 2 + \
                [np.zeros(n, np.float32)] * 2
    # kicks: ball speed rises by > 4 m/s within 4 frames to > 6 m/s, a player within 2.5 m
    bs = np.nan_to_num(f["bspeed"])
    seen = f["seen"].astype(bool)
    jump = np.zeros(n, bool)
    jump[1:-3] = seen[1:-3] & seen[4:] & (bs[4:] - bs[:-4] > 4.0) & (bs[4:] > 6.0)
    near = np.fmin(np.nan_to_num(f["dh"], nan=99), np.nan_to_num(f["da"], nan=99)) < 2.5
    kick = jump & near
    kick[1:] &= ~kick[:-1]
    kt = np.where(np.nan_to_num(f["dh"], nan=99) <= np.nan_to_num(f["da"], nan=99), 1.0, -1.0)
    ks, ku = _since(kick, cap)
    idx = np.where(kick, t, -1)
    last = np.maximum.accumulate(idx)
    idx2 = np.where(kick, t, n)
    nxt = np.minimum.accumulate(idx2[::-1])[::-1]
    cols += [ks, ku, np.where(last >= 0, kt[np.maximum(last, 0)], 0.0),
             np.where(nxt < n, kt[np.minimum(nxt, n - 1)], 0.0)]
    return np.stack([np.asarray(c, np.float32) for c in cols], 1)


def full_plus5(m, f, Xs):
    """R17: full_plus3 + touch-sequence / kick block."""
    return np.hstack([full_plus3(m, f, Xs), flight_block(m, f)]).astype(np.float32)
