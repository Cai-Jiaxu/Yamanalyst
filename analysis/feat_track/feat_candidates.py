"""Modular, label-free CANDIDATE feature extractors (strictly causal, half-reset).

Scope (owned by this track): tracking / spatial / temporal / uncertainty features that are NOT
already in `mtah.feats.full_plus3`. Model architecture, training recipe, decoder and scorer are
held fixed. Analysis-only: `mtah/` is untouched; no ground truth is read.

Compliance (per auditor): every NEW block uses PAST-ONLY information (no lookahead beyond the
current frame). Rolling / dwell / derivative quantities reset at half boundaries so no window
crosses a half break. This is stricter than the inherited full_plus3 centered assumption; the
existing full_plus3 code is left unchanged.

House fps = 25. Windows are in frames, TRAILING (inclusive of current frame).

Compose against the shipped base with `compose("full_plus3", ["c_margin_dwell", ...])`.
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")
FPS = 25


def _f(x):
    return np.asarray(x, dtype=np.float32)


def _half_id(m):
    """(n,) int: half label per frame (0 = outside graded halves / ungraded)."""
    hid = np.zeros(m.n, dtype=int)
    halves = getattr(m, "halves", None)
    if halves is not None:
        for row in halves:
            h, a, b = int(row[0]), int(row[1]), int(row[2])
            if b > a:
                hid[a:b + 1] = h
    return hid


def _seg_starts(hid):
    breaks = np.flatnonzero(np.diff(hid) != 0) + 1
    return np.concatenate([[0], breaks]), np.concatenate([breaks, [len(hid)]])


def _trail(x, w, hid):
    """Causal trailing mean over the last w frames (inclusive), reset at half breaks. No future."""
    out = np.full(len(x), np.nan, dtype=float)
    starts, ends = _seg_starts(hid)
    for s, e in zip(starts, ends):
        xa = x[s:e]
        ok = np.isfinite(xa).astype(float)
        val = np.where(np.isfinite(xa), xa, 0.0)
        cs = np.concatenate([[0.0], np.cumsum(val)])
        co = np.concatenate([[0.0], np.cumsum(ok)])
        idx = np.arange(len(xa))
        lo = np.maximum(0, idx - w)
        num = cs[idx + 1] - cs[lo]
        den = co[idx + 1] - co[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:e] = num / np.where(den < 0.3, np.nan, den)
    return out


def _back_diff(x, w, hid):
    """Causal backward rate: (x[i]-x[i-w])/(w/fps). NaN if the two ends differ in half."""
    out = np.full(len(x), np.nan, dtype=float)
    a, b = x[w:], x[:-w]
    good = np.isfinite(a) & np.isfinite(b) & (hid[w:] == hid[:-w])
    out[w:] = np.where(good, (a - b) / (w / FPS), np.nan)
    return out


def _slot_xy(m):
    L, W = m.pitch
    return m.players[..., 0] * L, m.players[..., 1] * W


# ---------------------------------------------------------------------------
# C1. Carrier tightness + opponent closing (B2 turnovers). Causal.
# ---------------------------------------------------------------------------
def c_carrier_tightness(m, f):
    px, py = _slot_xy(m)
    seen = f["seen"].astype(bool)
    hid = _half_id(m)
    home = (m.team == 0)[None]
    bx, by = f["bx"], f["by"]
    d = np.hypot(px - bx[:, None], py - by[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hslot = np.nanargmin(np.where(np.isfinite(np.where(home, d, np.inf)),
                                      np.where(home, d, np.inf), np.inf), axis=1)
        aslot = np.nanargmin(np.where(np.isfinite(np.where(~home, d, np.inf)),
                                      np.where(~home, d, np.inf), np.inf), axis=1)
    t = np.arange(m.n)
    hx, hy = px[t, hslot], py[t, hslot]
    ax, ay = px[t, aslot], py[t, aslot]
    home_carrier = f["dh"] <= f["da"]
    cx = np.where(home_carrier, hx, ax)
    cy = np.where(home_carrier, hy, ay)
    cd = np.hypot(px - cx[:, None], py - cy[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = np.where(home_carrier,
                         np.nanmin(np.where(~home, cd, np.nan), axis=1),
                         np.nanmin(np.where(home, cd, np.nan), axis=1))
        n_opp2 = np.where(home_carrier,
                          np.nansum(np.where(~home, cd < 2.0, 0.0), axis=1),
                          np.nansum(np.where(home, cd < 2.0, 0.0), axis=1)).astype(float)
    carrier_sign = np.where(home_carrier, 1.0, -1.0)
    closing = _back_diff(tight, 12, hid)  # trailing, half-reset
    cols = [
        np.where(seen, carrier_sign, np.nan),
        np.where(seen, tight, np.nan),
        np.where(seen, _trail(tight, 12, hid), np.nan),
        np.where(seen, closing, np.nan),
        np.where(seen, n_opp2, np.nan),
    ]
    return np.stack([_f(c) for c in cols], 1)


# ---------------------------------------------------------------------------
# C2. Team-centroid approach-to-ball velocity (B2 pressing). Causal backward diff.
# ---------------------------------------------------------------------------
def c_centroid_compression(m, f):
    px, py = _slot_xy(m)
    seen = f["seen"].astype(bool)
    hid = _half_id(m)
    home = (m.team == 0)[None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cxh = np.nanmean(np.where(home, px, np.nan), axis=1)
        cxa = np.nanmean(np.where(~home, px, np.nan), axis=1)
        cyh = np.nanmean(np.where(home, py, np.nan), axis=1)
        cya = np.nanmean(np.where(~home, py, np.nan), axis=1)
    vxh, vyh = _back_diff(cxh, 12, hid), _back_diff(cyh, 12, hid)
    vxa, vya = _back_diff(cxa, 12, hid), _back_diff(cya, 12, hid)
    dxh = f["bx"] - cxh
    dyh = f["by"] - cyh
    dxa = f["bx"] - cxa
    dya = f["by"] - cya
    nh = np.hypot(dxh, dyh)
    na = np.hypot(dxa, dya)
    ah = (vxh * dxh + vyh * dyh) / np.where(nh < 1e-6, np.nan, nh)
    aa = (vxa * dxa + vya * dya) / np.where(na < 1e-6, np.nan, na)
    cols = [np.where(seen, ah, np.nan), np.where(seen, aa, np.nan),
            np.where(seen, ah - aa, np.nan)]
    return np.stack([_f(c) for c in cols], 1)


# ---------------------------------------------------------------------------
# C3. Nearest-team margin dwell / oscillation (B2/B3). Causal run-length; trailing flip rate.
# ---------------------------------------------------------------------------
def c_margin_dwell(m, f):
    seen = f["seen"].astype(bool)
    hid = _half_id(m)
    margin = f["da"] - f["dh"]
    sign = np.sign(margin)
    sign[~seen] = 0.0
    # run length: consecutive frames same sign, reset at half break
    idx = np.arange(m.n)
    flip = np.zeros(m.n, bool)
    flip[1:] = (sign[1:] != sign[:-1]) & (sign[1:] != 0)
    flip &= (hid == np.concatenate([[hid[0]], hid[:-1]]))  # no flip counted across half break
    last_flip = np.maximum.accumulate(np.where(flip, idx, 0))
    run = np.where(seen, idx - last_flip, 0.0)
    run = np.minimum(run, 250.0)
    # trailing flip rate: fraction of flips in last 50 frames (~2 s), reset at half break
    zc = _trail(flip.astype(float), 50, hid)
    # trailing slope of smoothed margin (past only)
    sm = _trail(margin, 12, hid)
    slope = _back_diff(sm, 12, hid)
    cols = [np.where(seen, np.abs(margin), np.nan),
            np.where(seen, run, np.nan),
            np.where(seen, zc, np.nan),
            np.where(seen, slope, np.nan)]
    return np.stack([_f(c) for c in cols], 1)


# ---------------------------------------------------------------------------
# C4. Track-identity swap + melee body count (B2/masked). Swap causal; melee trailing distinct.
# ---------------------------------------------------------------------------
def c_melee_uncertainty(m, f):
    px, py = _slot_xy(m)
    seen = f["seen"].astype(bool)
    hid = _half_id(m)
    bx, by = f["bx"], f["by"]
    d = np.hypot(px - bx[:, None], py - by[:, None])
    nearest = np.nanargmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    prev = np.concatenate([nearest[:1], nearest[:-1]])
    swap = (nearest != prev).astype(float)
    swap[~seen | (hid != np.concatenate([[hid[0]], hid[:-1]]))] = np.nan
    near = np.isfinite(d) & (d < 3.0)
    # trailing distinct slots near ball in last 50 frames, reset at half break
    n, P = near.shape
    distinct = np.zeros(n)
    for p in range(P):
        col = near[:, p]
        last = -10 ** 9
        prev_hid = -1
        for i in range(n):
            if hid[i] != prev_hid:
                last = -10 ** 9
                prev_hid = hid[i]
            if col[i]:
                last = i
            if 0 <= i - last <= 50:
                distinct[i] += 1.0
    n_near = np.nansum(np.where(np.isfinite(d), d < 3.0, 0.0), axis=1).astype(float)
    cols = [np.where(seen, swap, np.nan),
            np.where(seen, distinct, np.nan),
            np.where(seen, n_near, np.nan)]
    return np.stack([_f(c) for c in cols], 1)


# ---------------------------------------------------------------------------
# C5. Ball-free midfield clustering. Causal trailing smooth, half-reset.
# ---------------------------------------------------------------------------
def c_ballfree_cluster(m, f):
    px, py = _slot_xy(m)
    hid = _half_id(m)
    home = (m.team == 0)[None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hx = np.where(home, px, np.nan)
        ax = np.where(~home, px, np.nan)
        h_lo, h_hi = np.nanmin(hx, axis=1), np.nanmax(hx, axis=1)
        a_lo, a_hi = np.nanmin(ax, axis=1), np.nanmax(ax, axis=1)
        overlap = np.minimum(h_hi, a_hi) - np.maximum(h_lo, a_lo)
        dd = np.hypot(np.where(home, py, np.nan)[:, :, None] -
                      np.where(~home, py, np.nan)[:, None, :],
                      hx[:, :, None] - ax[:, None, :])
        med = np.nanmedian(np.nanmin(dd, axis=1), axis=1)
    cols = [overlap, _trail(overlap, 25, hid), med, _trail(med, 25, hid)]
    return np.stack([_f(c) for c in cols], 1)


CANDIDATES = {
    "c_carrier_tightness": {
        "fn": c_carrier_tightness, "title": "Carrier tightness + trailing closing speed",
        "hypothesis": "Ball carrier tightly marked and opponent closing = imminent turnover.",
        "literature": "Bauer & Anzer 2021 counterpressing.",
        "source": "Needs ball seen; external degraded and our match/masked.",
        "future": "NONE. Trailing closing speed + trailing tightness mean; reset at half breaks.",
        "temporal": "trailing +-0.5 s window (w=12).", "targets": "B2"},
    "c_centroid_compression": {
        "fn": c_centroid_compression, "title": "Team-centroid approach-to-ball velocity",
        "hypothesis": "A team centroid accelerating toward the ball is pressing.",
        "literature": "Bauer & Anzer 2021 compactness.",
        "source": "Needs ball seen.",
        "future": "NONE. Backward difference only; half-reset.",
        "temporal": "trailing w=12 (~0.5 s).", "targets": "B2"},
    "c_margin_dwell": {
        "fn": c_margin_dwell, "title": "Nearest-team margin stability / trailing flip rate",
        "hypothesis": "A flipping small margin = duel that should not flip the label.",
        "literature": "Link & Hoernig 2017 controlled possession.",
        "source": "Needs ball seen.",
        "future": "NONE. Causal run-length; trailing flip rate; backward slope; half-reset.",
        "temporal": "trailing w=12/50.", "targets": "B2, B3"},
    "c_melee_uncertainty": {
        "fn": c_melee_uncertainty, "title": "Nearest-slot swap + trailing melee body count",
        "hypothesis": "Identity swap or many bodies around the ball make dh/da unreliable.",
        "literature": "House identity-switch precedent.",
        "source": "Needs ball seen.",
        "future": "NONE. Swap flag causal; trailing distinct count; half-reset.",
        "temporal": "trailing w=50 (~2 s).", "targets": "B2, masked"},
    "c_ballfree_cluster": {
        "fn": c_ballfree_cluster, "title": "Ball-free midfield clustering (trailing)",
        "hypothesis": "Collapsed shared x-band + tight median spacing localises a contested ball.",
        "literature": "Ball-free shape possession.",
        "source": "Ball-free; valid everywhere.",
        "future": "NONE. Trailing smooth; half-reset.",
        "temporal": "trailing w=25 (~1 s).", "targets": "ball-missing"},
}


def compose(base_name, names):
    from mtah import feats
    base = getattr(feats, base_name)
    blocks = [CANDIDATES[n]["fn"] for n in names]

    def featfn(m, f, Xs):
        cols = [base(m, f, Xs)]
        for b in blocks:
            cols.append(b(m, f))
        return np.hstack(cols).astype(np.float32)

    featfn.__name__ = "cmp__" + base_name + "__" + "__".join(names)
    featfn.candidates = list(names)
    return featfn


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from mtah.data import Match
    from mtah.features import compute
    from mtah import shape
    m = Match()
    f = compute(m)
    Xs, _ = shape.compute(m, cam=False)
    featfn = compose("full_plus3", list(CANDIDATES))
    X = featfn(m, f, Xs)
    print("match frames:", m.n, "composed:", X.shape, "dtype", X.dtype)
    print("halves:", m.halves)
    for name, c in CANDIDATES.items():
        z = c["fn"](m, f)
        print("  {:<22} cols={:<3} finite={:.2f}".format(name, z.shape[1], np.nanmean(np.isfinite(z))))
