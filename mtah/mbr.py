"""Segmental minimum-Bayes-risk decoding for spellF1.

Viterbi returns the single most probable path; spellF1 rewards matched possessions, not frames.
Here a few candidate paths (Viterbi at several switch-cost scales) compete stretch by stretch,
each judged by its expected spellF1 against paths sampled from the decoder's own posterior
(forward filtering, backward sampling). spellF1 = 2 TP / (G + N) does not split over stretches,
but its first-order change does: a local choice that adds dTP matches and dN spells moves it by
(2 dTP - F1 dN) / (G + N), so each stretch takes the candidate with the largest
E[2 TP] - F1_ref * N. Stretches are cut where every candidate agrees on a dead ball, so no
possession crosses a cut. (Segmental MBR as in speech recognition, Goel & Byrne 2004.)
"""
import numpy as np

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def ffbs(logE, T, k, rng, temp=1.0):
    """k state paths sampled from the chain posterior p(s) ~ exp((sum logE + sum T) / temp)."""
    logE = logE / temp
    T = T / temp
    n = len(logE)
    alpha = np.zeros((n, 3))
    alpha[0] = logE[0]
    for t in range(1, n):
        a = alpha[t - 1][:, None] + T
        m = a.max(0)
        alpha[t] = m + np.log(np.exp(a - m).sum(0)) + logE[t]
        alpha[t] -= alpha[t].max()
    out = np.zeros((k, n), dtype=np.int8)
    p = np.exp(alpha[-1] - alpha[-1].max())
    out[:, -1] = rng.choice(3, size=k, p=p / p.sum())
    u = rng.random((n, k))
    for t in range(n - 1, 0, -1):
        w = alpha[t - 1][:, None] + T[:, out[:, t]]        # (3, k)
        w = np.exp(w - w.max(0))
        c = np.cumsum(w / w.sum(0), 0)
        out[:, t - 1] = (u[t] [None, :] > c).sum(0).clip(0, 2)
    return out


def spells(path):
    """[(start, stop, state)] runs of H (0) / A (1)."""
    b = np.concatenate(([0], np.flatnonzero(path[1:] != path[:-1]) + 1, [len(path)]))
    return [(b[i], b[i + 1], path[b[i]]) for i in range(len(b) - 1) if path[b[i]] < 2]


def tp(ref, cand):
    """Mean over IoU thresholds of greedy one-to-one matches (the official matching)."""
    ov = []
    for i, (a, b, k) in enumerate(ref):
        for j, (c, d, q) in enumerate(cand):
            if q != k or d <= a or b <= c:
                continue
            ov.append(((min(b, d) - max(a, c)) / (max(b, d) - min(a, c)), i, j))
    ov.sort(key=lambda x: (-x[0], x[1], x[2]))
    tot = 0
    for th in IOUS:
        gi, pj = set(), set()
        for iou, i, j in ov:
            if iou < th:
                break
            if i in gi or j in pj:
                continue
            gi.add(i), pj.add(j)
        tot += len(gi)
    return tot / len(IOUS)


def choose(cands, samples, f1_ref=0.65, min_cut=50):
    """cands (c, n) and samples (k, n) state paths over one half -> (n,) chosen path."""
    dead_all = (cands == 2).all(0)
    e = np.diff(np.concatenate(([0], dead_all.astype(np.int8), [0])))
    s, z = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    cuts = [(a + b) // 2 for a, b in zip(s, z) if b - a >= min_cut]
    edges = [0] + cuts + [cands.shape[1]]
    out = cands[0].copy()
    for a, b in zip(edges[:-1], edges[1:]):
        cs = [spells(c[a:b]) for c in cands]
        if all(len(x) == len(cs[0]) and x == cs[0] for x in cs[1:]):
            continue
        ss = [spells(x[a:b]) for x in samples]
        best, bu = 0, -np.inf
        for ci, c in enumerate(cs):
            u = 2 * np.mean([tp(r, c) for r in ss]) - f1_ref * len(c)
            if u > bu + 1e-9:
                best, bu = ci, u
        out[a:b] = cands[best][a:b]
    return out
