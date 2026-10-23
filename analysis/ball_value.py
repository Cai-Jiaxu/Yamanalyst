"""ROADMAP R2: how much is less missing ball worth to final_v1, in-distribution?

    uv run python analysis/ball_value.py            # all three levels
    uv run python analysis/ball_value.py 0.10       # one level (models are cached after the fit)

The fold models are fitted once, as for final_v1, and saved to cache/models_final_v1.joblib
for later decoder or evaluation work. Then:
  ext     each external match keeps its bench degradation (same players, same noise), and whole
          ball gaps are restored (true ball + the usual 0.25 m noise) until the missing share
          drops to 25% and 10%. Restoring whole gaps keeps the "exact or missing" pattern the
          model was trained on, so this is an upper bound on what a perfect video channel adds.
  masked  our public window with 25% and 10% hidden instead of 45%.
Each match is scored by the fold model that did not train on it. Aggregates only.
"""
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import feats, final, noise  # noqa: E402
from mtah.bench import FOLDS, Bench, Case  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.masking import ball_mask, pace_weights  # noqa: E402

warnings.filterwarnings("ignore")
MODELS = os.path.join(CACHE_DIR, "models_final_v1.joblib")


def restore(c, em, target, seed):
    """c.m with whole ball gaps restored until <= target of graded frames miss the ball."""
    rng = np.random.default_rng(seed)
    ball = c.m.ball.copy()
    play = em.graded_mask()
    miss = play & ~np.isfinite(ball[:, 0])
    e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    runs = list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))
    L, W = em.pitch
    for k in rng.permutation(len(runs)):
        if miss.sum() <= target * play.sum():
            break
        a, b = runs[k]
        ball[a:b, 0] = em.ball[a:b, 0] + rng.normal(0, noise.BALL_NOISE / L, b - a)
        ball[a:b, 1] = em.ball[a:b, 1] + rng.normal(0, noise.BALL_NOISE / W, b - a)
        miss[a:b] = False
    return c.m.copy_with(ball=ball)


def main():
    from mtah import external
    import joblib
    featfn = getattr(feats, final.FEATFN)
    b = Bench()
    if os.path.isfile(MODELS):
        models = joblib.load(MODELS)
        P = None
    else:
        extra = {k: feats.mirrored(featfn).get(k, []) + feats.augmented(featfn).get(k, [])
                 for k in b.ext}
        t = time.time()
        P, models = b.probs(featfn, final.MODEL, extra=extra, return_models=True)
        joblib.dump(models, MODELS)
        print("fitted in {:.0f} s -> {}".format(time.time() - t, MODELS), flush=True)
    fold_of = {k: i for i, fold in enumerate(FOLDS) for k in fold}

    def predict(cases):
        out = {}
        for c in cases:
            X = featfn(c.m, c.f, c.Xs)
            if c.name in fold_of:
                out[c.name] = models[fold_of[c.name]].predict_proba(X)
            else:
                out[c.name] = np.mean([clf.predict_proba(X) for clf in models], axis=0)
        return out

    base_ext, base_masked = b.ext, b.masked
    weights = pace_weights(b.public.f)
    levels = ((None, 0.45), (0.25, 0.25), (0.10, 0.10))
    if len(sys.argv) > 1:
        levels = [(float(sys.argv[1]), float(sys.argv[1]))]
    for ext_target, mask_frac in levels:
        ext = {}
        for k, c in base_ext.items():
            if ext_target is None:
                ext[k] = c
                continue
            m2 = restore(c, external.ExtMatch(k), ext_target, noise.seed_for(k))
            f2 = compute(m2)
            ext[k] = Case(k, m2, f2, c.R, c.Xs, c.gt)
        masked = [Case("masked{}".format(s), b.m, fk, None, b.public.Xs, b.gt) for s, fk in
                  ((s, compute(b.m, hide=ball_mask(b.m.n, s, frac=mask_frac, weights=weights)))
                   for s in range(len(base_masked)))]
        b.ext, b.masked = ext, masked
        P2 = predict(list(ext.values()) + masked + [b.public])
        miss = np.mean([np.mean(~np.isfinite(c.m.ball[c.m.graded_mask(), 0]))
                        for c in ext.values()])
        b.score("ext miss {:.0%} / masked {:.0%}".format(miss, mask_frac), P2, final.DECODER,
                use_rules=False, t0=time.time())
    b.ext, b.masked = base_ext, base_masked


if __name__ == "__main__":
    main()
