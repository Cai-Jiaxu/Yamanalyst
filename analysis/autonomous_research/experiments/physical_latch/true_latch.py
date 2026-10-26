"""True physical possession latch with per-player velocity (generic team slots).

m.players shape varies per match (public 46, IDSSE 101, Metrica 76).
m.team[slot] = 0 home, 1 away. Third column is z/dummy.
Positions normalized 0..1; convert to meters via pitch dims.
Player velocity: 3-frame central difference at 25fps, m/s.
Control: ball seen, nearest player distance <=1.5m, team margin >=0.5m,
  |ball_speed - nearest_player_speed| <= 3 m/s, 10 consecutive frames (2-frame gap).
State machine: latch current team through passes; switch only after opposite
  control confirmed. D frames reset latch.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final, kickoff  # noqa: E402

STATES = np.array(["H", "A", "D"])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

CTRL_DIST_M = 1.5
CTRL_MARGIN_M = 0.5
CTRL_REL_SPEED = 3.0
CTRL_MIN_FRAMES = 10
CTRL_MAX_GAP = 2
LATCH_MAX_MISSING = 2
MAX_PLAYER_SPEED = 15.0   # reject outliers
MAX_BALL_SPEED = 50.0


def control_signal(c):
    f = c.f
    m = c.m
    seen = f["seen"].astype(bool)
    bx, by = f["bx"], f["by"]
    bspeed = f["bspeed"]

    pl = m.players  # (n, S, 3)
    pitch = m.pitch
    fps = m.fps
    xm = pl[:, :, 0] * pitch[0]
    ym = pl[:, :, 1] * pitch[1]

    home_idx = np.where(m.team == 0)[0]
    away_idx = np.where(m.team == 1)[0]

    # player velocities (central difference)
    pvx = np.full_like(xm, np.nan)
    pvy = np.full_like(ym, np.nan)
    pvx[1:-1] = (xm[2:] - xm[:-2]) / (2.0 / fps)
    pvy[1:-1] = (ym[2:] - ym[:-2]) / (2.0 / fps)
    pvx[0] = (xm[1] - xm[0]) / (1.0 / fps)
    pvy[0] = (ym[1] - ym[0]) / (1.0 / fps)
    pvx[-1] = (xm[-1] - xm[-2]) / (1.0 / fps)
    pvy[-1] = (ym[-1] - ym[-2]) / (1.0 / fps)

    n = m.n

    def nearest_rel_speed(team_idx):
        xt = xm[:, team_idx]
        yt = ym[:, team_idx]
        vxt = pvx[:, team_idx]
        vyt = pvy[:, team_idx]
        dx = xt - bx[:, None]
        dy = yt - by[:, None]
        d = np.sqrt(dx**2 + dy**2)
        d[~seen] = np.nan
        d_filled = np.where(np.isnan(d), 1e9, d)
        best = np.argmin(d_filled, axis=1)
        row = np.arange(n)
        pspeed = np.sqrt(vxt[row, best]**2 + vyt[row, best]**2)
        pspeed = np.where(pspeed > MAX_PLAYER_SPEED, np.nan, pspeed)
        bs = np.where(bspeed > MAX_BALL_SPEED, np.nan, bspeed)
        rel = np.abs(bs - pspeed)
        rel[~seen] = np.nan
        return rel

    h_rel = nearest_rel_speed(home_idx)
    a_rel = nearest_rel_speed(away_idx)

    dh = f["dh"]
    da = f["da"]
    h_cand = (seen & (dh < CTRL_DIST_M) & ((dh - da) < -CTRL_MARGIN_M)
              & ~np.isnan(h_rel) & (h_rel < CTRL_REL_SPEED))
    a_cand = (seen & (da < CTRL_DIST_M) & ((da - dh) < -CTRL_MARGIN_M)
              & ~np.isnan(a_rel) & (a_rel < CTRL_REL_SPEED))

    def persist(x, nf=CTRL_MIN_FRAMES, gap=CTRL_MAX_GAP):
        out = np.zeros_like(x)
        for i in range(len(x)):
            if i < nf - 1:
                continue
            out[i] = x[i - nf + 1:i + 1].sum() >= nf - gap
        return out

    return persist(h_cand), persist(a_cand), h_cand, a_cand, h_rel, a_rel


def latch_labels(labels, f, h_conf, a_conf, enable=True):
    if not enable:
        return labels.copy()
    seen = f["seen"].astype(bool)
    n = len(labels)
    out = labels.copy()
    current = None
    for i in range(n):
        lab = labels[i]
        if lab == "D":
            current = None
            out[i] = "D"
            continue
        miss_run = 0
        for j in range(i, max(i - 10, -1), -1):
            if not seen[j]:
                miss_run += 1
            else:
                break
        long_gap = miss_run > LATCH_MAX_MISSING
        if current is None:
            current = lab
            out[i] = lab
            continue
        wants = (lab != current) and (lab in ("H", "A"))
        if not wants:
            out[i] = current
            continue
        opp = h_conf[i] if lab == "H" else a_conf[i]
        if opp and not long_gap:
            current = lab
            out[i] = lab
    return out


def viterbi(logE, cost_team, cost_dead):
    n = len(logE)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logE[0].copy()
    for t in range(1, n):
        T = -cost_dead * _DEAD
        T[0, 1] = T[1, 0] = -cost_team
        cand = score[:, None] + T
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_baseline(c, P):
    p = dict(decode.PARAMS); p.update(final.DECODER)
    m, f = c.m, c.f
    Pd = P[c.name]
    R = np.full(m.n, "X", dtype="<U1")
    logE = decode.emissions(Pd, R, f["seen"], p)
    logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        out[a:b+1] = STATES[viterbi(logE[a:b+1], p["cost_team"], p["cost_dead"])]
    out = decode.min_duration(out, p["min_dead"], p["min_poss"])
    if p["stop_gate"]:
        out = decode.stop_gate(out, f, int(p["stop_gate"]))
    return out


def main():
    import time
    t0 = time.time()
    y = Yard(pfile="P_R1.npz")
    P = y.P

    print("=== enable=False identity ===", flush=True)
    def fn0(c, _p):
        return decode_baseline(c, P)
    base = y.report("tl_off", fn0)
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        base["bench"], base["ext"], base["masked"], base["public"], base["pubM"], base["val"]))

    print("\n=== control signal diagnostics (public) ===", flush=True)
    cpub = y.b.public
    hc, ac, hr_cand, ar_cand, h_rel, a_rel = control_signal(cpub)
    print("  h_cand={} a_cand={}".format(hr_cand.sum(), ar_cand.sum()))
    print("  h_conf={} a_conf={}".format(hc.sum(), ac.sum()))
    vh = ~np.isnan(h_rel); va = ~np.isnan(a_rel)
    if vh.any():
        print("  h_rel_speed: mean={:.3f} median={:.3f} p90={:.3f}".format(
            np.nanmean(h_rel), np.nanmedian(h_rel), np.nanpercentile(h_rel[vh], 90)))
    if va.any():
        print("  a_rel_speed: mean={:.3f} median={:.3f} p90={:.3f}".format(
            np.nanmean(a_rel), np.nanmedian(a_rel), np.nanpercentile(a_rel[va], 90)))

    print("\n=== enable=True true latch ===", flush=True)
    def fn1(c, _p):
        out = decode_baseline(c, P)
        hc, ac, _, _, _, _ = control_signal(c)
        return latch_labels(out, c.f, hc, ac, enable=True)
    res = y.report("tl_on", fn1)
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        res["bench"], res["ext"], res["masked"], res["public"], res["pubM"], res["val"]))
    print("  delta: bench {:+.4f} ext {:+.4f} masked {:+.4f} public {:+.4f} val {:+.4f}".format(
        res["bench"]-base["bench"], res["ext"]-base["ext"], res["masked"]-base["masked"],
        res["public"]-base["public"], res["val"]-base["val"]))
    print("  elapsed {:.1f}s".format(time.time()-t0))


if __name__ == "__main__":
    main()
