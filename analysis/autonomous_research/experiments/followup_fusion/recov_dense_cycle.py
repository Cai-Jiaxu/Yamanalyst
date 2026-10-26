"""Dense frame-correspondence fusion: build a piecewise-constant time-warp from the confident
chamfer anchors, then for every tactical frame map to a broadcast frame and READ the actual
per-frame broadcast ball tracking. No ball-position interpolation, no detector.

Between two adjacent confident anchors whose offset agrees within OFF_TOL frames, the offset is
stable (live segment). Within that live interval, every tactical frame maps to a broadcast frame
via the stable offset, and we read bro.ball[broadcast_frame]. Replay offset-jump windows are
excluded. Gating: broadcast hvalid, reproj <REPROJ_MAX, speed <SPEED_MAX, bounded pitch.

External has no broadcast; measured on public/masked.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, feats, final, kickoff  # noqa: E402
from mtah.data import Match  # noqa: E402
from mtah.features import compute  # noqa: E402

STATES = np.array(["H", "A", "D"])
_TEAM = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

SAMPLE = 125
COARSE = 5
GOOD = 1.5
REPROJ_MAX = 0.50
SPEED_MAX = 80.0
OFF_TOL = 3          # adjacent anchors must agree within this for a live segment
GAP_MAX = 125       # max anchor gap to bridge (one sample interval)


def team_points(m, i):
    P = m.players[i]
    ok = np.isfinite(P[:, 0])
    L, W = m.pitch
    return [np.stack([P[ok & (m.team == t), 0] * L, P[ok & (m.team == t), 1] * W], 1)
            for t in (0, 1)]


def chamfer_many(bpts, tac, cand):
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


def anchors(tac, bro):
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
    return np.array(rows)


def build_time_map(tac, bro):
    """Return (n_tactical,) broadcast frame index, or -1 where no live correspondence."""
    rows = anchors(tac, bro)
    good = rows[rows[:, 2] < GOOD]
    good = good[np.argsort(good[:, 0])]  # by broadcast frame
    off = good[:, 1] - good[:, 0]

    # piecewise-constant live segments between adjacent anchors with stable offset
    t2b = np.full(tac.n, -1, dtype=np.int64)
    n_seg = 0
    for i in range(len(good) - 1):
        b0, t0, _ = good[i]
        b1, t1, _ = good[i + 1]
        dt_b = b1 - b0
        dt_t = t1 - t0
        # live segment if offset stable and gap is one sample interval
        if abs(off[i] - off[i + 1]) > OFF_TOL:
            continue
        if dt_b > GAP_MAX:
            continue
        offset = int(round(off[i]))
        a, bnd = int(min(t0, t1)), int(max(t0, t1))
        for tt in range(a, bnd + 1):
            bb = tt - offset
            if 0 <= bb < bro.n:
                t2b[tt] = bb
        n_seg += 1
    print("  anchors {} live segments {} mapped tactical frames {} ({:.1%})".format(
        len(good), n_seg, (t2b >= 0).sum(), (t2b >= 0).mean()))
    return t2b


def recovered_positions(tac, bro):
    t2b = build_time_map(tac, bro)
    rec = np.full((tac.n, 2), np.nan, dtype=np.float32)
    t_seen = np.isfinite(tac.ball[:, 0])
    b_seen = np.isfinite(bro.ball[:, 0])
    n = 0
    for tt in range(tac.n):
        bb = t2b[tt]
        if bb < 0:
            continue
        if t_seen[tt]:
            continue
        if not b_seen[bb]:
            continue
        if not bro.hvalid[bb] or bro.reproj[bb] > REPROJ_MAX:
            continue
        bx, by = bro.ball[bb, 0], bro.ball[bb, 1]
        if not (0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0):
            continue
        rec[tt] = (bx, by)
        n += 1
    # speed plausibility on the recovered track
    finite = np.isfinite(rec[:, 0])
    idx = np.flatnonzero(finite)
    removed = 0
    for j in range(1, len(idx)):
        dt = (idx[j] - idx[j - 1]) / tac.fps
        d = np.hypot(rec[idx[j], 0] - rec[idx[j - 1], 0], rec[idx[j], 1] - rec[idx[j - 1], 1])
        if dt > 0 and d / dt * np.max(tac.pitch) > SPEED_MAX:
            rec[idx[j]] = np.nan
            removed += 1
    graded = tac.graded_mask()
    print("  recovered actual broadcast ball positions: {} frames (after speed gate {} removed) "
          "of {} missing graded".format(n - removed, removed, int((graded & ~t_seen).sum())))
    return rec


def viterbi_team(logE, cost_team, cost_dead):
    back = np.zeros((len(logE), 3), dtype=np.int8)
    score = logE[0].copy()
    for t in range(1, len(logE)):
        T = -cost_dead * _DEAD
        T[0, 1] = T[1, 0] = -cost_team
        cand = score[:, None] + T
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(len(logE), dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(len(logE) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def main():
    import joblib
    from mtah.score import pm
    from mtah.data import CACHE_DIR
    tac = Match(source="tactical")
    bro = Match(source="broadcast")
    rec = recovered_positions(tac, bro)
    np.save(os.path.join(ROOT, "cache", "ar_followup_fusion_recpos.npy"), rec)

    models = joblib.load(os.path.join(CACHE_DIR, "models_final_v1.joblib"))
    models = models["models"] if isinstance(models, dict) else models
    y = Yard(pfile="P_R1.npz")
    b = y.b

    p = dict(decode.PARAMS)
    p.update(final.DECODER)

    def predict(c, inject):
        m = c.m
        ball = m.ball.copy()
        if inject:
            missing = np.isnan(ball[:, 0])
            have = np.isfinite(rec[:, 0])
            ball[missing & have, :2] = rec[missing & have]
        orig = m.ball
        m.ball = ball
        f = compute(m)
        X = feats.full_plus3(m, f, c.Xs)
        m.ball = orig
        P = np.mean([clf.predict_proba(X) for clf in models], axis=0)
        return P, f

    def decode_case(c, P, f):
        m = c.m
        R = np.full(m.n, "X", dtype="<U1")
        logE = decode.emissions(P, R, f["seen"], p)
        logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
        out = np.full(m.n, "X", dtype="<U1")
        for _, a, bnd in m.halves:
            out[a:bnd + 1] = STATES[viterbi_team(logE[a:bnd + 1], p["cost_team"], p["cost_dead"])]
        out = decode.min_duration(out, p["min_dead"], 0)
        if p["stop_gate"]:
            out = decode.stop_gate(out, f, int(p["stop_gate"]))
        return out

    print("=== baseline (no injection) ===")
    for c in [b.public] + b.masked:
        P, f = predict(c, inject=False)
        lab = decode_case(c, P, f)
        ev = pm.evaluate(c.gt, pm.label_codes(lab), bootstrap_n=0)
        print("  {}: sp {:.4f} mf {:.4f}".format(c.name, ev["spell_f1_mean"], ev["macro_f1"]))

    print("=== dense-correspondence recovered ball injection ===")
    for c in [b.public] + b.masked:
        P, f = predict(c, inject=True)
        lab = decode_case(c, P, f)
        ev = pm.evaluate(c.gt, pm.label_codes(lab), bootstrap_n=0)
        print("  {}: sp {:.4f} mf {:.4f}".format(c.name, ev["spell_f1_mean"], ev["macro_f1"]))


if __name__ == "__main__":
    main()
