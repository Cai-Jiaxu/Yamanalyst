"""Bench (masked + public) for: no video, greedy merge_track at conf >= 0.25 / 0.05, and the DP
linker of analysis/video_link.py (mask-agnostic: no ATD anchors). Same structure as
analysis/video_score.py. ~1 h on a laptop that stays awake. Aggregates only.

    uv run python analysis/video_link_bench.py
"""
import os, sys, time, warnings
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from mtah import feats, final, rules  # noqa: E402
from mtah.bench import Bench, Case  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.masking import ball_mask, pace_weights  # noqa: E402
from mtah.video_ft import merge_track  # noqa: E402
import video_link as linker  # noqa: E402

warnings.filterwarnings("ignore")
featfn = getattr(feats, final.FEATFN)
b = Bench()
extra = {k: feats.mirrored(featfn).get(k, []) + feats.augmented(featfn).get(k, []) for k in b.ext}
t = time.time()
P, models = b.probs(featfn, final.MODEL, extra=extra, return_models=True)
print("fitted in {:.0f} s".format(time.time() - t), flush=True)
b.score("final_v1 (no video)", P, final.DECODER, use_rules=False, t0=time.time())
DETP = os.path.join(ROOT, "cache", "ball_det_ft.npz")


def with_track(label, track):
    m = b.m
    m.video_ball = track
    miss = m.graded_mask() & ~np.isfinite(m.ball[:, 0])
    f = compute(m)
    cases = [Case("public", m, f, rules.run(m, f), b.public.Xs, b.gt)]
    m.video_ball = None
    weights = pace_weights(compute(m))
    m.video_ball = track
    for s in range(len(b.masked)):
        fk = compute(m, hide=ball_mask(m.n, s, weights=weights))
        cases.append(Case("masked{}".format(s), m, fk, rules.run(m, fk), b.public.Xs, b.gt))
    P2 = dict(P)
    for c in cases:
        Xo = featfn(c.m, c.f, c.Xs)
        P2[c.name] = np.mean([clf.predict_proba(Xo) for clf in models], axis=0)
    saved = (b.public, b.masked)
    b.public, b.masked = cases[0], cases[1:]
    b.score("{} ({:.1%} of missing filled)".format(label, np.isfinite(track[miss, 0]).mean()),
            P2, final.DECODER, use_rules=False, t0=time.time())
    b.public, b.masked = saved
    m.video_ball = None


with_track("greedy conf>=0.25 (current)", merge_track(b.m, DETP, conf_min=0.25))
with_track("greedy conf>=0.05", merge_track(b.m, DETP, conf_min=0.05))
DD = linker.dedup(linker.DET)
with_track("DP linker c0=0.08", linker.link(DD, np.zeros((0, 3)), b.m.n))
with_track("DP linker c0=0.12", linker.link(DD, np.zeros((0, 3)), b.m.n, c0=0.12))
