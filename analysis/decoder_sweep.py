"""Decoder-only sweep on the shared benchmark: fit the models once, then vary the decoder.

    uv run python analysis/decoder_sweep.py [E-key of the feature set, default E0]
"""
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from mtah.bench import Bench, shape_only  # noqa: E402
import experiments as ex  # noqa: E402

warnings.filterwarnings("ignore")
FEATS = {"E0": shape_only, "E1": ex.long_context, "E2a": ex.make_ball(False),
         "E2c": ex.make_ball(True), "E6": ex.full, "E7": ex.full_plus, "E8": ex.full_plus2}

GRID = [
    {},
    {"q_rule": 0.9}, {"q_rule": 0.995},
    {"w_rule": 0.5}, {"w_rule": 2.0},
    {"seen_r": 0}, {"seen_r": 50},
    {"w_shape": 0.5}, {"w_shape": 2.0},
    {"snap_r": 12}, {"snap_r": 25}, {"snap_r": 50},
    {"cost_team": 8.0}, {"cost_team": 16.0}, {"cost_dead": 4.0}, {"cost_dead": 9.0},
]

GRID_NORULES = [
    {},
    {"w_shape": 0.5}, {"w_shape": 0.75}, {"w_shape": 1.5}, {"w_shape": 2.0},
    {"cost_dead": 3.0}, {"cost_dead": 9.0}, {"cost_dead": 12.0},
    {"snap_r": 12}, {"snap_r": 25}, {"snap_r": 50},
]

GRID_MIN = [{"w_shape": 0.75, "cost_dead": 12.0}] + [
    {"w_shape": 0.75, "cost_dead": 12.0, "min_dead": d, "min_poss": q}
    for d in (0, 25, 50) for q in (0, 12, 25) if d or q]

def _both_extras(featfn):
    a, b = ex.mirrored(featfn), ex.augmented(featfn)
    return {k: a.get(k, []) + b.get(k, []) for k in set(a) | set(b)}


# fit configurations beyond a bare feature set: (featfn, extra builder, model params)
CONFIGS = {
    "E10": (ex.full_plus2, _both_extras, {"max_iter": 600}),
}

GRID_FINAL = [{"w_shape": 0.75, "cost_dead": cd, "cost_team": ct, "w_bound": wb}
              for ct in (6.0, 9.0, 12.0) for cd in (9.0, 12.0) for wb in (0.0, 1.0)] + [
    {"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 9.0, "w_bound": 0.0, "min_dead": 25},
    {"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 9.0, "w_bound": 0.0, "em_prior": True},
    {"w_shape": 1.0, "cost_dead": 12.0, "cost_team": 9.0, "w_bound": 0.0},
    {"w_shape": 0.5, "cost_dead": 12.0, "cost_team": 9.0, "w_bound": 0.0},
]

GRID_BOUND = [{"w_shape": 0.75, "cost_dead": 12.0, "cost_team": ct, "w_bound": wb}
              for wb in (0.0, 0.5, 1.0, 2.0) for ct in (12.0, 18.0, 24.0)] + [
    {"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 12.0, "w_bound": wb, "min_dead": md,
     "min_poss": mp} for wb in (0.0, 1.0) for md, mp in ((25, 0), (50, 0), (25, 12), (0, 12))]

GRID_2D = [{"w_shape": w, "cost_dead": cd, "cost_team": ct}
           for w in (0.6, 0.75, 1.0) for cd in (9.0, 12.0, 16.0) for ct in (12.0, 18.0)]

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "E0"
    use_rules = (sys.argv[2] != "norules") if len(sys.argv) > 2 else True
    grid = sys.argv[3] if len(sys.argv) > 3 else ""
    b = Bench()
    t = time.time()
    import numpy as np
    from mtah.data import CACHE_DIR
    path = os.path.join(CACHE_DIR, "P_{}_{}.npz".format(key, "bound"))
    if os.path.isfile(path):
        z = np.load(path)
        P = {k: z[k] for k in z.files}
    else:
        if key in CONFIGS:
            featfn, extra_fn, params = CONFIGS[key]
            P = b.probs(featfn, params, extra_fn(featfn), boundary=True)
        else:
            P = b.probs(FEATS[key], boundary=True)
        np.savez_compressed(path, **P)
    print("fitted {} in {:.0f}s".format(key, time.time() - t), flush=True)
    GRID_EM = [{"w_shape": 0.75, "cost_dead": 12.0, "em_prior": e} for e in (False, True)]
    chosen = {"2d": GRID_2D, "min": GRID_MIN, "bound": GRID_BOUND, "em": GRID_EM,
              "final": GRID_FINAL}.get(grid, GRID if use_rules else GRID_NORULES)
    for g in chosen:
        b.score("{} {}".format(key, g or "defaults"), P, g, use_rules=use_rules, t0=time.time())
