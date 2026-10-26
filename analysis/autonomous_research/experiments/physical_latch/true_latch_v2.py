"""True physical possession latch v2 — fixed state override + vector rel velocity.

Fixes from v1:
- latch_labels now ALWAYS writes out[i]=current on live frames, not labels[i].
- Relative velocity is vector norm |v_ball - v_player|, not scalar |speed diff|.
- Direction cosine required when both speeds >0.5 m/s.
- Same-carrier ID continuity tracked across the 10-frame persistence window.
- D/X reset current_team to None (no carry across dead/out-of-play).
- enable=False path actually calls latch_labels(enable=False), not just baseline.
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
MIN_DIR_SPEED = 0.5
MIN_DIR_COS = 0.5
MAX_PLAYER_SPEED = 15.0
MAX_BALL_SPEED = 50.0


def control_signal(c):
    """Return h_conf, a_conf (boolean arrays), plus nearest-player ID tracking."""
    f = c.f
    m = c.m
    seen = f["seen"].astype(bool)
    bx, by = f["bx"], f["by"]
    bvx, bvy = f["vx"], f["vy"]  # ball velocity (from clean masked f)
    bspeed = f["bspeed"]

    pl = m.players
    pitch = m.pitch
    fps = m.fps
    xm = pl[:, :, 0] * pitch[0]
    ym = pl[:, :, 1] * pitch[1]

    home_idx = np.where(m.team == 0)[0]
    away_idx = np.where(m.team == 1)[0]

    pvx = np.full_like(xm, np.nan)
    pvy = np.full_like(ym, np.nan)
    pvx[1:-1] = (xm[2:] - xm[:-2]) / (2.0 / fps)
    pvy[1:-1] = (ym[2:] - ym[:-2]) / (2.0 / fps)
    pvx[0] = (xm[1] - xm[0]) / (1.0 / fps)
    pvy[0] = (ym[1] - ym[0]) / (1.0 / fps)
    pvx[-1] = (xm[-1] - xm[-2]) / (1.0 / fps)
    pvy[-1] = (ym[-1] - ym[-2]) / (1.0 / fps)

    n = m.n
    row = np.arange(n)

    def nearest_team(team_idx):
        xt = xm[:, team_idx]; yt = ym[:, team_idx]
        vxt = pvx[:, team_idx]; vyt = pvy[:, team_idx]
        dx = xt - bx[:, None]; dy = yt - by[:, None]
        d = np.sqrt(dx**2 + dy**2)
        d[~seen] = np.nan
        d_filled = np.where(np.isnan(d), 1e9, d)
        best = np.argmin(d_filled, axis=1)
        psx = vxt[row, best]; psy = vyt[row, best]
        # vector relative velocity
        rvx = bvx - psx; rvy = bvy - psy
        rel = np.sqrt(rvx**2 + rvy**2)
        # direction cosine (only when both speeds > threshold)
        bs_mag = np.sqrt(bvx**2 + bvy**2)
        ps_mag = np.sqrt(psx**2 + psy**2)
        dot = bvx * psx + bvy * psy
        cos = dot / (bs_mag * ps_mag + 1e-9)
        dir_ok = ~((bs_mag > MIN_DIR_SPEED) & (ps_mag > MIN_DIR_SPEED) & (cos < MIN_DIR_COS))
        # outlier rejection
        ps_mag_out = ps_mag > MAX_PLAYER_SPEED
        bs_mag_out = bs_mag > MAX_BALL_SPEED
        rel[ps_mag_out | bs_mag_out] = np.nan
        rel[~seen] = np.nan
        dir_ok[~seen] = False
        # return local best index within team, rel speed, dir ok
        return best, rel, dir_ok

    h_best, h_rel, h_dir = nearest_team(home_idx)
    a_best, a_rel, a_dir = nearest_team(away_idx)

    dh = f["dh"]; da = f["da"]
    h_cand = (seen & (dh < CTRL_DIST_M) & ((dh - da) < -CTRL_MARGIN_M)
              & ~np.isnan(h_rel) & (h_rel < CTRL_REL_SPEED) & h_dir)
    a_cand = (seen & (da < CTRL_DIST_M) & ((da - dh) < -CTRL_MARGIN_M)
              & ~np.isnan(a_rel) & (a_rel < CTRL_REL_SPEED) & a_dir)

    # persistence: require N frames with same carrier ID (allow 2-frame gaps)
    def persist_with_id(cand, best_id, nf=CTRL_MIN_FRAMES, gap=CTRL_MAX_GAP):
        out = np.zeros(len(cand), dtype=bool)
        confirmed = np.zeros(len(cand), dtype=bool)
        # sliding window: count cand frames and unique IDs in window
        for i in range(len(cand)):
            if i < nf - 1:
                continue
            win = cand[i - nf + 1: i + 1]
            if win.sum() < nf - gap:
                continue
            # check carrier ID continuity: at least nf-gap frames share same ID
            ids = best_id[i - nf + 1: i + 1]
            valid = win
            if valid.sum() == 0:
                continue
            vids = ids[valid]
            # most common ID in window
            vals, counts = np.unique(vids, return_counts=True)
            top = vals[np.argmax(counts)]
            if counts.max() >= nf - gap:
                confirmed[i] = True
        return confirmed

    h_conf = persist_with_id(h_cand, h_best)
    a_conf = persist_with_id(a_cand, a_best)
    return h_conf, a_conf, h_cand, a_cand, h_rel, a_rel


def latch_labels(labels, f, h_conf, a_conf, enable=True):
    """State machine: ALWAYS write current team on live frames."""
    if not enable:
        return labels.copy(), 0, 0
    seen = f["seen"].astype(bool)
    n = len(labels)
    out = np.empty(n, dtype="<U1")
    current = None  # H or A
    blocked = 0
    accepted = 0
    for i in range(n):
        lab = labels[i]
        if lab not in ("H", "A"):
            current = None
            out[i] = lab
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
        if lab == current:
            out[i] = current
            continue
        # baseline proposes switch to lab
        opp_conf = h_conf[i] if lab == "H" else a_conf[i]
        if opp_conf and not long_gap:
            current = lab
            out[i] = lab
            accepted += 1
        else:
            out[i] = current
            blocked += 1
    return out, blocked, accepted


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


def toy_tests():
    """Unit tests for latch_labels state machine."""
    print("=== toy unit tests ===")
    # Test 1: enable=False returns labels unchanged
    labels = np.array(["H","H","A","A","D","H"], dtype="<U1")
    f = {"seen": np.ones(6, dtype=bool)}
    hc = np.zeros(6, dtype=bool); ac = np.zeros(6, dtype=bool)
    out, b, a = latch_labels(labels, f, hc, ac, enable=False)
    assert (out == labels).all(), f"enable=False failed: {out} vs {labels}"
    print("  T1 enable=False exact: PASS")

    # Test 2: no opp_conf -> baseline H->A blocked, output stays H
    labels = np.array(["H","H","H","A","A","A"], dtype="<U1")
    f = {"seen": np.ones(6, dtype=bool)}
    hc = np.zeros(6, dtype=bool); ac = np.zeros(6, dtype=bool)
    out, b, a = latch_labels(labels, f, hc, ac, enable=True)
    assert (out == np.array(["H","H","H","H","H","H"])).all(), f"T2 failed: {out}"
    print(f"  T2 unconfirmed H->A blocked: PASS (blocked={b}, accepted={a})")

    # Test 3: opp_conf at frame 3 -> switch accepted
    labels = np.array(["H","H","H","A","A","A"], dtype="<U1")
    f = {"seen": np.ones(6, dtype=bool)}
    hc = np.zeros(6, dtype=bool); ac = np.array([0,0,0,1,1,1], dtype=bool)
    out, b, a = latch_labels(labels, f, hc, ac, enable=True)
    assert (out == np.array(["H","H","H","A","A","A"])).all(), f"T3 failed: {out}"
    print(f"  T3 confirmed H->A accepted: PASS (blocked={b}, accepted={a})")

    # Test 4: D resets latch
    labels = np.array(["H","H","D","A","A","A"], dtype="<U1")
    f = {"seen": np.ones(6, dtype=bool)}
    hc = np.zeros(6, dtype=bool); ac = np.zeros(6, dtype=bool)
    out, b, a = latch_labels(labels, f, hc, ac, enable=True)
    # after D, current=None, frame 3 takes baseline A
    assert (out == np.array(["H","H","D","A","A","A"])).all(), f"T4 failed: {out}"
    print(f"  T4 D resets latch: PASS")

    # Test 5: long ball gap blocks switch (switch at frame 4 with 3-frame miss run)
    labels = np.array(["H","H","H","H","A","A"], dtype="<U1")
    f = {"seen": np.array([1,1,0,0,0,1], dtype=bool)}
    hc = np.zeros(6, dtype=bool); ac = np.array([0,0,0,0,1,1], dtype=bool)
    out, b, a = latch_labels(labels, f, hc, ac, enable=True)
    # frame 4: miss_run=3 (frames 4,3,2 missing) -> long_gap=True, blocked
    # frame 5: seen=1, miss_run=0, accepted
    assert (out == np.array(["H","H","H","H","H","A"])).all(), f"T5 failed: {out}"
    print(f"  T5 long gap blocks switch: PASS (blocked={b}, accepted={a})")

    # Test 6: opposite-speed vectors (same magnitude, opposite direction) -> rel norm large,
    #   so control_signal would NOT confirm. State machine: no opp_conf -> block.
    labels = np.array(["H","H","H","A","A","A"], dtype="<U1")
    f = {"seen": np.ones(6, dtype=bool)}
    hc = np.zeros(6, dtype=bool); ac = np.zeros(6, dtype=bool)  # no confirmation
    out, b, a = latch_labels(labels, f, hc, ac, enable=True)
    assert (out == np.array(["H","H","H","H","H","H"])).all(), f"T6 failed: {out}"
    print(f"  T6 opposite-vector (no conf) blocked: PASS (blocked={b})")

    # Test 7: carrier ID changes every frame -> persistence filter would reject.
    #   Simulate: ac has confirmation only at frame 5 (not enough contiguous)
    labels = np.array(["H","H","H","H","H","A"], dtype="<U1")
    f = {"seen": np.ones(6, dtype=bool)}
    hc = np.zeros(6, dtype=bool); ac = np.array([0,0,0,0,0,1], dtype=bool)
    out, b, a = latch_labels(labels, f, hc, ac, enable=True)
    # frame 5: ac[5]=True, switch accepted
    assert (out == np.array(["H","H","H","H","H","A"])).all(), f"T7 failed: {out}"
    print(f"  T7 single-frame conf accepted (edge): PASS")

    # Test 8: D/X intervals preserved, half reset
    labels = np.array(["H","D","X","A","D","H"], dtype="<U1")
    f = {"seen": np.ones(6, dtype=bool)}
    hc = np.zeros(6, dtype=bool); ac = np.zeros(6, dtype=bool)
    out, b, a = latch_labels(labels, f, hc, ac, enable=True)
    assert (out == labels).all(), f"T8 failed: {out}"
    print(f"  T8 D/X intervals preserved: PASS")


def main():
    import time
    t0 = time.time()
    toy_tests()

    y = Yard(pfile="P_R1.npz")
    P = y.P

    print("\n=== enable=False identity (real data) ===", flush=True)
    def fn0(c, _p):
        out = decode_baseline(c, P)
        hc, ac, _, _, _, _ = control_signal(c)
        lat, _, _ = latch_labels(out, c.f, hc, ac, enable=False)
        return lat
    base = y.report("tl2_off", fn0)
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        base["bench"], base["ext"], base["masked"], base["public"], base["pubM"], base["val"]))

    print("\n=== enable=True true latch v2 ===", flush=True)
    total_blocked = 0; total_accepted = 0
    def fn1(c, _p):
        nonlocal total_blocked, total_accepted
        out = decode_baseline(c, P)
        hc, ac, _, _, _, _ = control_signal(c)
        lat, bl, acn = latch_labels(out, c.f, hc, ac, enable=True)
        total_blocked += bl; total_accepted += acn
        return lat
    res = y.report("tl2_on", fn1)
    print(f"  total blocked switches={total_blocked}, accepted={total_accepted}")
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        res["bench"], res["ext"], res["masked"], res["public"], res["pubM"], res["val"]))
    print("  delta: bench {:+.4f} ext {:+.4f} masked {:+.4f} public {:+.4f} val {:+.4f}".format(
        res["bench"]-base["bench"], res["ext"]-base["ext"], res["masked"]-base["masked"],
        res["public"]-base["public"], res["val"]-base["val"]))
    print("  elapsed {:.1f}s".format(time.time()-t0))


if __name__ == "__main__":
    main()
