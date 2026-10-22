"""Does the fine-tuned video ball track (cache/ball_det_ft.npz) improve the submission model?

    uv run python analysis/video_score.py

Fold models are fitted once on the external data (which the video does not touch); our match's
public and masked cases are then rebuilt with Match.video_ball attached and re-predicted.
Caveat: in the masked cases the hidden frames are ones the ATD did see, which are easier for the
video detector too, so the masked gain is an upper bound; the public number is unaffected by that.
"""
import os, sys, time, warnings
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import feats, final, rules  # noqa: E402
from mtah.bench import Bench, Case  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.masking import ball_mask, pace_weights  # noqa: E402
from mtah.video_ft import merge_track  # noqa: E402

warnings.filterwarnings("ignore")
featfn = getattr(feats, final.FEATFN)
b = Bench()
extra = {k: feats.mirrored(featfn).get(k, []) + feats.augmented(featfn).get(k, []) for k in b.ext}
t = time.time()
P, models = b.probs(featfn, final.MODEL, extra=extra, return_models=True)
print("fitted in {:.0f} s".format(time.time() - t), flush=True)
b.score("final_v1 (no video)", P, final.DECODER, use_rules=False, t0=time.time())


def with_video(conf_min):
    m = b.m
    m.video_ball = merge_track(m, os.path.join(ROOT, "cache", "ball_det_ft.npz"), conf_min=conf_min)
    miss = m.graded_mask() & ~np.isfinite(m.ball[:, 0])
    f = compute(m)
    cases = [Case("public", m, f, rules.run(m, f), b.public.Xs, b.gt)]
    weights = pace_weights(compute(m.__class__()))  # same gap placement as the bench
    for s in range(len(b.masked)):
        fk = compute(m, hide=ball_mask(m.n, s, weights=weights))
        cases.append(Case("masked{}".format(s), m, fk, rules.run(m, fk), b.public.Xs, b.gt))
    P2 = dict(P)
    for c in cases:
        Xo = featfn(c.m, c.f, c.Xs)
        P2[c.name] = np.mean([clf.predict_proba(Xo) for clf in models], axis=0)
    saved = (b.public, b.masked)
    b.public, b.masked = cases[0], cases[1:]
    b.score("video conf>={} ({:.1%} of missing filled)".format(
        conf_min, np.isfinite(m.video_ball[miss, 0]).mean()), P2, final.DECODER,
        use_rules=False, t0=time.time())
    b.public, b.masked = saved
    m.video_ball = None


for c in (0.25, 0.15):
    with_video(c)
