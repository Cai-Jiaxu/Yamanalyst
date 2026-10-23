"""Audit of cache/ball_det_ft.npz: coverage, confidence of the true candidate, rank, duplicates,
distance to players. Aggregates only.

    uv run python analysis/video_audit.py
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match
from mtah.masking import W0, W1

m = Match()
L, W = m.pitch
z = np.load(os.path.join(ROOT, "cache", "ball_det_ft.npz"))
det, wanted = z["det"], z["frames"]
bx, by = m.ball[:, 0] * L, m.ball[:, 1] * W
px, py = m.players[..., 0] * L, m.players[..., 1] * W
graded = m.graded_mask()
seen = np.isfinite(bx)
miss = graded & ~seen

det = det[np.argsort(det[:, 0], kind="mergesort")]
fr = det[:, 0].astype(int)
fu, st = np.unique(fr, return_index=True)
en = np.append(st[1:], len(det))
groups = {f: (s, e) for f, s, e in zip(fu, st, en)}

print("frames detected on: {}, candidates: {}, mean per frame-with-cand {:.2f}".format(
    len(wanted), len(det), len(det) / max(len(fu), 1)))

# ---- A. coverage on genuinely missing graded frames
mw = np.flatnonzero(miss)
has = np.array([f in groups for f in mw])
topc = np.array([det[groups[f][0]:groups[f][1], 3].max() if f in groups else 0 for f in mw])
print("\n[A] missing graded frames: {} | any candidate {:.1%} | top conf>=0.25 {:.1%} | >=0.45 {:.1%}"
      .format(len(mw), has.mean(), (topc >= .25).mean(), (topc >= .45).mean()))

# ---- B. public window frames where ATD sees the ball (NB: these frames were in the detector's
# training pool, so this is optimistic)
pw = np.arange(W0, W1 + 750)
pw = pw[seen[pw] & np.isin(pw, wanted)]
n = len(pw)
top_ok = any_ok = with_c = 0
rank_hist = np.zeros(6, int)
conf_true, conf_wrongtop = [], []
oob_wrongtop = far_wrongtop = far_true = 0
dup_frames = 0
nearp_true_all = []
for f in pw:
    if f not in groups:
        continue
    with_c += 1
    d = det[groups[f][0]:groups[f][1]]
    dist = np.hypot(d[:, 1] - bx[f], d[:, 2] - by[f])
    o = np.argsort(-d[:, 3])
    hit = dist[o] < 2.0
    if hit.any():
        any_ok += 1
        r = int(np.argmax(hit))
        rank_hist[r] += 1
        conf_true.append(d[o[r], 3])
        pd_ = np.nanmin(np.hypot(px[f] - d[o[r], 1], py[f] - d[o[r], 2]))
        far_true += pd_ >= 4.0
    if hit[0]:
        top_ok += 1
    else:
        t = d[o[0]]
        conf_wrongtop.append(t[3])
        oob_wrongtop += not (0 <= t[1] <= L and 0 <= t[2] <= W)
        far_wrongtop += np.nanmin(np.hypot(px[f] - t[1], py[f] - t[2])) >= 4.0
    if len(d) > 1:
        dd = np.hypot(d[:, None, 1] - d[None, :, 1], d[:, None, 2] - d[None, :, 2])
        dup_frames += (dd[np.triu_indices(len(d), 1)] < 0.8).any()
nw = n - with_c
print("\n[B] public-window ATD-seen frames: {} | with cand {:.1%} | any within 2 m {:.1%} | "
      "top-1 within 2 m {:.1%} (of all) / {:.1%} (given cand)".format(
          n, with_c / n, any_ok / n, top_ok / n, top_ok / max(with_c, 1)))
print("    rank of true candidate when present (1..5):", (rank_hist[:5] / max(any_ok, 1)).round(3))
ct = np.array(conf_true)
print("    conf of the true candidate: median {:.2f}, <0.25 on {:.1%}, <0.45 on {:.1%}".format(
    np.median(ct), (ct < .25).mean(), (ct < .45).mean()))
cw = np.array(conf_wrongtop)
print("    wrong top-1: n={}, median conf {:.2f}, out of pitch {:.1%}, >=4 m from every player {:.1%}"
      .format(len(cw), np.median(cw) if len(cw) else np.nan, oob_wrongtop / max(len(cw), 1),
              far_wrongtop / max(len(cw), 1)))
print("    true candidate >=4 m from every player: {:.1%}".format(far_true / max(any_ok, 1)))
print("    frames with two candidates < 0.8 m apart (tile duplicates): {:.1%}".format(
    dup_frames / max(with_c, 1)))

# ---- C. how often is the (ATD) ball >= 4 m from every player, over graded seen frames
gs = np.flatnonzero(graded & seen)
dmin = np.nanmin(np.hypot(px[gs] - bx[gs, None], py[gs] - by[gs, None]), axis=1)
print("\n[C] graded ATD-seen frames: ball >= 4 m from every player on {:.1%} (>=8 m {:.1%})".format(
    (dmin >= 4).mean(), (dmin >= 8).mean()))
# and right before the ATD loses the ball
e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
starts, ends = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
lens = ends - starts
pre = starts[(starts > 0) & seen[np.maximum(starts - 1, 0)]] - 1
if len(pre):
    dpre = np.nanmin(np.hypot(px[pre] - bx[pre, None], py[pre] - by[pre, None]), axis=1)
    out_pre = ~((bx[pre] >= 0) & (bx[pre] <= L) & (by[pre] >= 0) & (by[pre] <= W))
    print("    last ATD frame before a gap ({} gaps): ball >=4 m from players {:.1%}, out of pitch {:.1%}"
          .format(len(pre), (dpre >= 4).mean(), out_pre.mean()))
print("    gaps: {} | >10 s hold {:.1%} of missing frames".format(
    len(lens), lens[lens > 250].sum() / lens.sum()))
