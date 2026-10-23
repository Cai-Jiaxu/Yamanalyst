"""Does video-ball jitter fake ball speed and change dead-ball detection (rules.dead_mask)?
Aggregates only.

    uv run python analysis/video_jitter.py
"""
import os, sys, warnings
import numpy as np
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "analysis"))
from mtah import rules
from mtah.features import compute
from mtah.masking import W0, W1, ball_mask, pace_weights
from mtah.video_ft import merge_track
from mtah.score import load_gt, pm
import video_link as linker

m = linker.m
L, W = m.pitch
bx, by = linker.bx, linker.by
DETP = os.path.join(ROOT, "cache", "ball_det_ft.npz")


def smooth(track, win=15):
    out = track.copy()
    ok = np.isfinite(track[:, 0])
    e = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if b - a >= win:
            out[a:b] = savgol_filter(track[a:b], win, 2, axis=0)
    return out


DD = linker.dedup(linker.DET)
tracks = {"greedy@0.25": merge_track(m, DETP, conf_min=0.25),
          "greedy@0.05": merge_track(m, DETP, conf_min=0.05),
          "DP": linker.link(DD, np.zeros((0, 3)), m.n)}
tracks["DP+savgol15"] = smooth(tracks["DP"])

# (1) jitter on frames where the ATD ball is ~stationary (public window, ATD seen)
w = np.arange(W0 + 3, W1 - 3)
v_atd = np.hypot(bx[w + 1] - bx[w - 1], by[w + 1] - by[w - 1]) * m.fps / 2
stat = w[np.isfinite(v_atd) & (v_atd < 0.5)]
print("stationary ATD frames in window:", len(stat))
for k, tr in tracks.items():
    v = np.full(m.n, np.nan)
    v[1:-1] = np.hypot(tr[2:, 0] - tr[:-2, 0], tr[2:, 1] - tr[:-2, 1]) * m.fps / 2
    vs = v[stat]
    ok = np.isfinite(vs)
    print("  {:<12s} speed on stationary-ball frames: median {:.1f} m/s, >3 m/s on {:.0%} (n={})".format(
        k, np.nanmedian(vs), (vs[ok] > 3).mean() if ok.any() else np.nan, ok.sum()))

# (2) dead-ball share and agreement with GT D on masked cases (rules dead_mask)
gt = load_gt()
GD = (gt == pm.D)[W0:W1]
weights = pace_weights(compute(m))
win = slice(W0, W1)


def dead_stats(track):
    m.video_ball = track
    share, rec, prec = [], [], []
    for s in range(8):
        f = compute(m, hide=ball_mask(m.n, s, weights=weights))
        d = rules.dead_mask(f, rules.PARAMS, m.fps)[win]
        share.append((d.mean(), (d & GD).sum() / GD.sum(), (d & GD).sum() / max(d.sum(), 1)))
    m.video_ball = None
    return "share {:.1%} | recall of GT D {:.1%} | precision {:.1%}".format(*np.mean(share, axis=0))

print("rules dead share in public window, masked (8 seeds):")
print("  GT D share {:.1%}".format(GD.mean()))
print("  no video      ", dead_stats(None))
for k, tr in tracks.items():
    print("  {:<12s} ".format(k), dead_stats(tr))
