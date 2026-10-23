"""Link per-frame video ball candidates into tracklets by DP, vs the greedy merge_track.

    uv run python analysis/video_link.py

Scored on the public window with the ATD ball hidden (8 masking seeds; the ATD ball is the
answer there). Caveat: these frames were in the detector's training pool and the ball is
visible on them, so precision here is optimistic. Aggregates only."""
import os, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match
from mtah.masking import W0, W1, ball_mask
from mtah.video_ft import merge_track

m = Match()
L, W = m.pitch
FPS = m.fps
z = np.load(os.path.join(ROOT, "cache", "ball_det_ft.npz"))
DET = z["det"][np.argsort(z["det"][:, 0], kind="mergesort")]
bx, by = m.ball[:, 0] * L, m.ball[:, 1] * W
seen = np.isfinite(bx)


def dedup(det, r=1.0):
    out = []
    fr = det[:, 0].astype(int)
    fu, st = np.unique(fr, return_index=True)
    en = np.append(st[1:], len(det))
    for s, e in zip(st, en):
        d = det[s:e][np.argsort(-det[s:e, 3])]
        keep = []
        for k in range(len(d)):
            if all(np.hypot(d[k, 1] - d[j, 1], d[k, 2] - d[j, 2]) >= r for j in keep):
                keep.append(k)
        out.append(d[keep])
    return np.vstack(out) if out else det[:0]


def link(det, anchors, n, c0=0.08, max_dt=12, vmax=32.0, tol=1.5, skip_pen=0.15,
         start_pen=3.0, w_v=0.3, anchor_rew=4.0):
    """det: (k,4) frame,x,y,conf (deduped). anchors: (a,3) frame,x,y known positions.
    Returns (n,2) track filled only between linked nodes."""
    nodes = np.vstack([np.c_[det[:, :3], np.log(np.maximum(det[:, 3], 1e-3) / c0)],
                       np.c_[anchors, np.full(len(anchors), anchor_rew)]])
    nodes = nodes[np.argsort(nodes[:, 0], kind="mergesort")]
    f = nodes[:, 0].astype(int)
    K = len(nodes)
    best = np.full(K, -np.inf)
    back = np.full(K, -1)          # predecessor node in the same tracklet, -1 = tracklet start
    pre = np.full(K, -1)           # for starts: end node of the previous tracklet (or -1)
    run_best, run_arg = 0.0, -1    # best total over nodes with frame < current frame
    fu, st = np.unique(f, return_index=True)
    en = np.append(st[1:], K)
    lo = 0
    pending = []
    for fi, s, e in zip(fu, st, en):
        # nodes from earlier frames become eligible for the prefix max
        for j in pending:
            if best[j] > run_best:
                run_best, run_arg = best[j], j
        pending = list(range(s, e))
        while f[lo] < fi - max_dt:
            lo += 1
        for j in range(s, e):
            cand_best, cand_back = run_best - start_pen, -1
            if lo < s:
                i = np.arange(lo, s)
                dt = fi - f[i]
                dist = np.hypot(nodes[i, 1] - nodes[j, 1], nodes[i, 2] - nodes[j, 2])
                ok = dist <= vmax * dt / FPS + tol
                if ok.any():
                    sc = best[i] - skip_pen * (dt - 1) - w_v * dist / (dt / FPS) / vmax
                    sc[~ok] = -np.inf
                    k = int(np.argmax(sc))
                    if sc[k] > cand_best:
                        cand_best, cand_back = sc[k], i[k]
            best[j] = nodes[j, 3] + cand_best
            back[j] = cand_back
            pre[j] = run_arg if cand_back == -1 else -1
    # backtrack from the best overall node
    for j in pending:
        if best[j] > run_best:
            run_best, run_arg = best[j], j
    track = np.full((n, 2), np.nan)
    j = run_arg
    while j >= 0:
        i = back[j]
        if i >= 0:
            a, b = f[i], f[j]
            t = np.arange(0, b - a + 1) / (b - a)
            track[a:b + 1] = nodes[i, 1:3][None] + t[:, None] * (nodes[j, 1:3] - nodes[i, 1:3])[None]
            j = i
        else:
            track[f[j]] = nodes[j, 1:3]
            j = pre[j]
    return track


def anchors_for(hidden_frames_mask):
    """ATD positions on the visible frame just before / after each hidden stretch."""
    e = np.diff(np.concatenate(([0], hidden_frames_mask.astype(np.int8), [0])))
    a, b = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    out = []
    for s in a - 1:
        if s >= 0 and seen[s]:
            out.append((s, bx[s], by[s]))
    for s in b:
        if s < m.n and seen[s]:
            out.append((s, bx[s], by[s]))
    return np.array(out, dtype=float).reshape(-1, 3)


def evaluate(name, tracks_and_hidden):
    cov, acc, good = [], [], []
    for tr, hid in tracks_and_hidden:
        h = np.flatnonzero(hid)
        ok = np.isfinite(tr[h, 0])
        err = np.hypot(tr[h, 0] - bx[h], tr[h, 1] - by[h])
        cov.append(ok.mean())
        acc.append((err[ok] < 2.0).mean() if ok.any() else np.nan)
        good.append((ok & (err < 2.0)).mean())
    print("{:<44s} coverage {:5.1%} | within 2 m when filled {:5.1%} | correct coverage {:5.1%}"
          .format(name, np.mean(cov), np.nanmean(acc), np.mean(good)), flush=True)


if __name__ == "__main__":
    DD = dedup(DET)
    print("dedup: {} -> {} candidates".format(len(DET), len(DD)))
    seeds = range(8)
    cases = []
    for s in seeds:
        hid = ball_mask(m.n, s)
        hid[W1:] = False
        hid &= seen            # score only where the ATD answer exists
        cases.append(hid)

    def sub(det, hid):
        return det[hid[det[:, 0].astype(int)]]

    for conf_min in (0.25, 0.15, 0.05):
        res = []
        for hid in cases:
            p = os.path.join(tempfile.gettempdir(), "mtah_tmp_det.npz")
            np.savez(p, det=sub(DET, hid))
            res.append((merge_track(m, p, conf_min=conf_min), hid))
        evaluate("greedy merge_track conf>={}".format(conf_min), res)
    for kw in ({}, {"start_pen": 1.5}, {"start_pen": 6.0}, {"c0": 0.05}, {"c0": 0.12},
               {"max_dt": 25}, {"anchor_rew": 0.0}):
        res = [(link(sub(DD, hid), anchors_for(hid), m.n, **kw), hid) for hid in cases]
        evaluate("DP linker {}".format(kw or "default"), res)
    # raw ceiling: any candidate within 2 m on the hidden frames
    cov = []
    for hid in cases:
        h = np.flatnonzero(hid)
        d = sub(DET, hid)
        hit = np.zeros(m.n, bool)
        ok = np.hypot(d[:, 1] - bx[d[:, 0].astype(int)], d[:, 2] - by[d[:, 0].astype(int)]) < 2
        hit[d[ok, 0].astype(int)] = True
        cov.append(hit[h].mean())
    print("ceiling: some raw candidate within 2 m on {:.1%} of hidden frames".format(np.mean(cov)))

    # real missing frames: coverage only (no answer exists there)
    miss = m.graded_mask() & ~seen
    tr = link(sub(DD, miss), anchors_for(miss), m.n)
    g = merge_track(m, os.path.join(ROOT, "cache", "ball_det_ft.npz"))
    print("real missing graded frames: DP linker fills {:.1%} | greedy merge_track fills {:.1%}".format(
        np.isfinite(tr[miss, 0]).mean(), np.isfinite(g[miss, 0]).mean()))
