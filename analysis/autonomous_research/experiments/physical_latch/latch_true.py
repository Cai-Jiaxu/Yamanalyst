"""True physical possession latch: stateful post-viterbi H/A retention.

State machine on top of frozen v3 decoded labels:
- D frames pass through unchanged (dead ball resets latch state).
- Between D frames, latch retains current team.
- A team switch (H->A or A->H) is only accepted if the NEW team has confirmed
  continuous control evidence for 10 consecutive frames (allowing max 2-frame gaps):
    * ball seen
    * distance to that team's nearest player <= 1.5 m
    * distance margin: that team's distance - other team's distance <= -0.5 m
      (i.e., this team is at least 0.5 m closer)
    * relative ball-player speed <= 3 m/s (ball not escaping the nearest player)
- When ball is missing for >2 frames inside a live stretch, no switch accepted
  (latch holds through short gaps, but does not blindly extend long gaps).
- No manual labels, no training. Thresholds from physics reasoning only.

Identity test: enable=False reproduces baseline labels exactly.
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

# Physics thresholds (frozen, no public tuning)
CTRL_DIST_M = 1.5       # ball within this of a nearest player
CTRL_MARGIN_M = 0.5     # this team must be at least 0.5m closer than other
CTRL_REL_SPEED = 3.0    # m/s, ball not escaping nearest player
CTRL_MIN_FRAMES = 10    # 10 consecutive frames = 0.4s at 25fps
CTRL_MAX_GAP = 2        # allow up to 2-frame gaps within confirmation window
LATCH_MAX_MISSING = 2   # don't accept switches during ball gaps > 2 frames


def control_evidence(f):
    """Return h_confirmed, a_confirmed boolean arrays after persistence filter."""
    seen = f["seen"].astype(bool)
    dh = f["dh"]
    da = f["da"]
    bspeed = f["bspeed"]
    sdh = f["sdh"]
    sda = f["sda"]
    # relative ball-player speed: |ball_speed - nearest_player_speed|
    rel_h = np.abs(bspeed - sdh)
    rel_a = np.abs(bspeed - sda)

    # h candidate: ball seen, close to home, home is closer than away by margin, rel speed low
    h_cand = (seen
              & (dh < CTRL_DIST_M)
              & ((dh - da) < -CTRL_MARGIN_M)
              & (rel_h < CTRL_REL_SPEED))
    a_cand = (seen
              & (da < CTRL_DIST_M)
              & ((da - dh) < -CTRL_MARGIN_M)
              & (rel_a < CTRL_REL_SPEED))

    def persist(x, n=CTRL_MIN_FRAMES, max_gap=CTRL_MAX_GAP):
        """Mark frames where the preceding n frames (allowing max_gap misses) all qualify."""
        out = np.zeros_like(x)
        for i in range(len(x)):
            if i < n - 1:
                continue
            window = x[i - n + 1: i + 1]
            if window.sum() >= n - max_gap:
                out[i] = True
        return out

    return persist(h_cand), persist(a_cand), h_cand, a_cand


def latch_labels(labels, f, enable=True):
    """Stateful latch: keep current team until opposite control is confirmed.

    labels: array of H/A/D from baseline viterbi.
    Returns new labels array.
    """
    if not enable:
        return labels.copy()
    h_conf, a_conf, h_cand, a_cand = control_evidence(f)
    seen = f["seen"].astype(bool)
    n = len(labels)
    out = labels.copy()
    current_team = None  # H or A
    for i in range(n):
        lab = labels[i]
        if lab == "D":
            current_team = None
            out[i] = "D"
            continue
        # live frame (H or A in baseline)
        # if ball missing for >2 consecutive, don't latch-switch (but keep D reset)
        # determine ball-missing run length
        miss_run = 0
        for j in range(i, max(i - 10, -1), -1):
            if not seen[j]:
                miss_run += 1
            else:
                break
        long_gap = miss_run > LATCH_MAX_MISSING

        if current_team is None:
            # start of live stretch: take baseline label
            current_team = lab
            out[i] = lab
            continue
        # we have a current_team; check if baseline wants to switch
        wants_switch = (lab != current_team) and (lab in ("H", "A"))
        if not wants_switch:
            out[i] = current_team
            continue
        # baseline wants to switch to lab; only accept if opposite control confirmed
        opp_conf = h_conf[i] if lab == "H" else a_conf[i]
        if opp_conf and not long_gap:
            current_team = lab
            out[i] = lab
        # else: latch holds current_team
    return out


def decode_latch(c, P, enable):
    p = dict(decode.PARAMS)
    p.update(final.DECODER)
    m, f = c.m, c.f
    Pd = P[c.name]
    R = np.full(m.n, "X", dtype="<U1")
    logE = decode.emissions(Pd, R, f["seen"], p)
    logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        out[a:b + 1] = STATES[viterbi(logE[a:b + 1], p["cost_team"], p["cost_dead"])]
    out = decode.min_duration(out, p["min_dead"], p["min_poss"])
    if p["stop_gate"]:
        out = decode.stop_gate(out, f, int(p["stop_gate"]))
    # apply latch
    out = latch_labels(out, f, enable=enable)
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


def main():
    y = Yard(pfile="P_R1.npz")
    P = y.P

    print("=== enable=False identity test ===", flush=True)
    def fn0(c, _p):
        return decode_latch(c, P, enable=False)
    base = y.report("latch_off", fn0)
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        base["bench"], base["ext"], base["masked"], base["public"], base["pubM"], base["val"]))

    # diagnostics on public
    cpub = y.b.public
    _, _, h_cand, a_cand = control_evidence(cpub.f)
    h_conf = np.zeros_like(h_cand); a_conf = np.zeros_like(a_cand)
    # recompute confirmed for diagnostic
    def persist(x, n=CTRL_MIN_FRAMES, max_gap=CTRL_MAX_GAP):
        out = np.zeros_like(x)
        for i in range(len(x)):
            if i < n - 1: continue
            out[i] = x[i - n + 1:i + 1].sum() >= n - max_gap
        return out
    h_conf = persist(h_cand); a_conf = persist(a_cand)
    print("\n=== diagnostics (public) ===", flush=True)
    print("  h_candidate frames={} a_candidate frames={}".format(h_cand.sum(), a_cand.sum()))
    print("  h_confirmed frames={} a_confirmed frames={}".format(h_conf.sum(), a_conf.sum()))

    print("\n=== enable=True physical latch ===", flush=True)
    def fn1(c, _p):
        return decode_latch(c, P, enable=True)
    res = y.report("latch_on", fn1)
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        res["bench"], res["ext"], res["masked"], res["public"], res["pubM"], res["val"]))
    print("  delta: bench {:+.4f} ext {:+.4f} masked {:+.4f} public {:+.4f} val {:+.4f}".format(
        res["bench"] - base["bench"], res["ext"] - base["ext"],
        res["masked"] - base["masked"], res["public"] - base["public"],
        res["val"] - base["val"]))


if __name__ == "__main__":
    main()
