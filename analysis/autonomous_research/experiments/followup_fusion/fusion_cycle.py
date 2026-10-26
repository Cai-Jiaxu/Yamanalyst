"""NEW fusion cycle: confidence/replay-filtered broadcast ball fill, applied as a decoder
prior on cached fold probabilities (P_R1.npz). No refit, no new video model, no manual labels.

What is distinct from broadcast_align.py:
  - broadcast_align.py already reports coverage/fill as a STATIC aggregate. This cycle builds a
    per-tactical-frame fusion ball-fill array with TEMPORAL MONOTONIC OFFSET per half and REPLAY
    EXCLUSION (drop offset jumps > jump_thresh frames between consecutive confident samples),
    then uses it as a decoder emission prior.
  - When the tactical ball is missing (f["seen"] False) but the replay-filtered broadcast fill
    says the broadcast ball is live at that tactical frame, suppress the D emission (the ball is
    in play, not dead). This is the controlled-recovery ablation: how much does a clean,
    replay-filtered fusion fill move the frozen v5 decoder on our match.

External matches have no broadcast, so ext cannot move. Measured on public/masked (our match).
Adoption rule: bench >= +0.005 AND ext/masked not down >0.005.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final, kickoff  # noqa: E402
from mtah.data import Match  # noqa: E402

STATES = np.array(["H", "A", "D"])
_TEAM = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

SAMPLE = 125
COARSE = 5
GOOD = 1.5


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


def align_matches(tac, bro):
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


def fusion_fill_array(tac, bro, jump_thresh):
    """Replay-filtered per-tactical-frame broadcast-ball-fill mask.

    Returns (n_tactical,) bool: True where a confident, temporally-consistent alignment says
    the broadcast ball is live (seen) at that tactical frame.
    """
    rows = align_matches(tac, bro)
    good = rows[rows[:, 2] < GOOD]
    good = good[np.argsort(good[:, 0])]  # by broadcast frame
    # monotonic offset per half: drop samples whose offset jumps > jump_thresh from previous kept
    off = good[:, 1] - good[:, 0]
    keep = np.ones(len(good), dtype=bool)
    last_off = None
    for i in range(len(good)):
        if last_off is not None and abs(off[i] - last_off) > jump_thresh:
            keep[i] = False
        else:
            last_off = off[i]
    kept = good[keep]
    print("  alignment: {} confident, {} after replay-jump filter (jump>{})".format(
        len(good), len(kept), jump_thresh))

    b_seen = np.isfinite(bro.ball[:, 0])
    fill = np.zeros(tac.n, dtype=bool)
    for b, t, _ in kept.astype(int):
        for k in range(-SAMPLE // 2, SAMPLE // 2):
            bb, tt = b + k, t + k
            if 0 <= bb < bro.n and 0 <= tt < tac.n and b_seen[bb]:
                fill[tt] = True
    t_seen = np.isfinite(tac.ball[:, 0])
    graded = tac.graded_mask()
    miss = graded & ~t_seen
    print("  replay-filtered fill: {:.1%} of graded missing frames would be filled -> "
          "missing {:.1%} -> {:.1%}".format(
              fill[miss].mean(), miss.sum() / graded.sum(),
              (miss & ~fill).sum() / graded.sum()))
    return fill


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


def decode_ar(c, Pd, fill, w_fill):
    p = dict(decode.PARAMS)
    p.update(final.DECODER)
    m, f = c.m, c.f
    P = Pd[c.name]
    R = np.full(m.n, "X", dtype="<U1")
    logE = decode.emissions(P, R, f["seen"], p)
    # fusion prior: tactical ball missing but broadcast says live -> suppress D
    if w_fill and c.name.startswith(("public", "masked")) and len(fill) == m.n:
        suppress = ~f["seen"] & fill
        logE[suppress, 2] -= w_fill  # D index
    logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        out[a:b + 1] = STATES[viterbi_team(logE[a:b + 1], p["cost_team"], p["cost_dead"])]
    out = decode.min_duration(out, p["min_dead"], 0)
    if p["stop_gate"]:
        out = decode.stop_gate(out, f, int(p["stop_gate"]))
    return out


def main():
    tac = Match(source="tactical")
    bro = Match(source="broadcast")
    print("=== fusion: replay-filtered broadcast fill as decoder prior ===")
    fill = fusion_fill_array(tac, bro, jump_thresh=3)
    np.save(os.path.join(ROOT, "cache", "ar_followup_fusion_fill.npy"), fill)
    y = Yard(pfile="P_R1.npz")
    Pd = y.P
    for w in [0.0, 1.0, 2.0, 4.0]:
        out = y.report("fill_w{}".format(w), lambda c, _p, w=w: decode_ar(c, Pd, fill, w), quiet=True)
        print("  w={}: bench {:.4f} ext {:.4f} masked {:.4f} | public sp {:.4f}/mf {:.4f} | val sp {:.4f}/mf {:.4f}".format(
            w, out["bench"], out["ext"], out["masked"], out["public"], out["pubM"], out["val"], out["valM"]))


if __name__ == "__main__":
    main()
