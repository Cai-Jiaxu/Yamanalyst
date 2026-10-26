"""Truly fundamental fusion: recover ACTUAL broadcast ball positions (not a seen>validity
proxy), gate by homography reprojection confidence and temporal velocity plausibility, inject
them into the tactical ball array where the tactical ball is missing, rebuild features, re-predict
with the 3 saved final_v1 fold models (no refit), and run the frozen v5 decoder.

Distinct from fusion_cycle.py (which only suppressed D on "broadcast ball seen" = false-evidence
proxy): here we inject the actual ball (bx, by) positions, so ball-derived features (bx, by,
bspeed, control, nearest-player distance) change and the model re-predicts.

Gating:
  - broadcast hvalid frame
  - broadcast reproj error < REPROJ_MAX m
  - speed between recovered positions < SPEED_MAX m/s (plausibility)
  - recovered ball within bounded pitch [0,1] x [0,1] (normalized)
  - replay-jump filtered alignment (jump > JUMP frames drops)

External matches have no broadcast, so ext cannot move. Measured on public/masked (our match).
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
JUMP = 3


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


def recovered_positions(tac, bro):
    """Return (n_tactical, 2) recovered ball positions (normalized 0..1), NaN where not recovered."""
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
    good = good[np.argsort(good[:, 0])]
    off = good[:, 1] - good[:, 0]
    keep = np.ones(len(good), dtype=bool)
    last = None
    for i in range(len(good)):
        if last is not None and abs(off[i] - last) > JUMP:
            keep[i] = False
        else:
            last = off[i]
    kept = good[keep]

    rec = np.full((tac.n, 2), np.nan, dtype=np.float32)
    b_seen = np.isfinite(bro.ball[:, 0])
    t_seen = np.isfinite(tac.ball[:, 0])
    n_rec = 0
    for b, t, _ in kept.astype(int):
        bb = b
        tt = t
        if not (0 <= bb < bro.n and 0 <= tt < tac.n):
            continue
        if not b_seen[bb]:
            continue
        if t_seen[tt]:
            continue  # tactical already has the ball
        bx, by = bro.ball[bb, 0], bro.ball[bb, 1]
        # bounded pitch check
        if not (0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0):
            continue
        # reproj confidence on broadcast frame
        if bro.reproj[bb] > REPROJ_MAX:
            continue
        rec[tt] = (bx, by)
        n_rec += 1
    # velocity plausibility between consecutive recovered positions
    finite = np.isfinite(rec[:, 0])
    idx = np.flatnonzero(finite)
    for j in range(1, len(idx)):
        dt = (idx[j] - idx[j - 1]) / tac.fps
        d = np.hypot(rec[idx[j], 0] - rec[idx[j - 1], 0], rec[idx[j], 1] - rec[idx[j - 1], 1])
        if dt > 0 and d / dt * np.max(tac.pitch) > SPEED_MAX:
            rec[idx[j]] = np.nan  # implausible jump
            n_rec -= 1
    print("  recovered actual ball positions: {} frames (of {} missing graded)".format(
        n_rec, int((graded & ~t_seen).sum())))
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
    from mtah.data import CACHE_DIR
    tac = Match(source="tactical")
    bro = Match(source="broadcast")
    rec = recovered_positions(tac, bro)
    np.save(os.path.join(ROOT, "cache", "ar_followup_fusion_recpos.npy"), rec)

    models = joblib.load(os.path.join(CACHE_DIR, "models_final_v1.joblib"))
    models = models["models"] if isinstance(models, dict) else models

    y = Yard(pfile="P_R1.npz")
    b = y.b

    def predict_with_recov(c, inject):
        m = c.m
        ball = m.ball.copy()
        if inject:
            missing = np.isnan(ball[:, 0])
            have = np.isfinite(rec[:, 0])
            ball[missing & have] = rec[missing & have]
        # rebuild a lightweight match view: temporarily swap ball
        orig_ball = m.ball
        m.ball = ball
        f = compute(m)
        X = feats.full_plus3(m, f, c.Xs)
        m.ball = orig_ball
        P = np.mean([clf.predict_proba(X) for clf in models], axis=0)
        return P, f

    p = dict(decode.PARAMS)
    p.update(final.DECODER)

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

    # baseline (no injection) on public + masked
    print("=== baseline (cached P, no injection) ===")
    for c in [b.public] + b.masked:
        P, f = predict_with_recov(c, inject=False)
        lab = decode_case(c, P, f)
        ev = pm_evaluate(c, lab)
        print("  {}: sp {:.4f} mf {:.4f}".format(c.name, ev["spell_f1_mean"], ev["macro_f1"]))

    print("=== recovered-ball injection ===")
    for c in [b.public] + b.masked:
        P, f = predict_with_recov(c, inject=True)
        lab = decode_case(c, P, f)
        ev = pm_evaluate(c, lab)
        print("  {}: sp {:.4f} mf {:.4f}".format(c.name, ev["spell_f1_mean"], ev["macro_f1"]))


def pm_evaluate(c, lab):
    import evaluation.metrics as pm
    return pm.evaluate(c.gt, pm.label_codes(lab), bootstrap_n=0)


if __name__ == "__main__":
    main()
