"""Feature-set experiments on the shared benchmark (mtah/bench.py).

    uv run python analysis/experiments.py E1 E2 ...

Each experiment prints one bench line; 'bench' = mean of external spellF1 and our masked spellF1.
"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import shape  # noqa: E402
from mtah.bench import Bench, shape_only  # noqa: E402

warnings.filterwarnings("ignore")
from mtah.feats import *  # noqa: E402,F401,F403
from mtah.feats import _since  # noqa: E402,F401


EXPERIMENTS = {
    "E0": lambda b: b.evaluate("E0 shape-only + rules (current)", shape_only),
    "E1": lambda b: b.evaluate("E1 long context + rules", long_context),
    "E2a": lambda b: b.evaluate("E2a shape+ball + rules", make_ball(False)),
    "E2b": lambda b: b.evaluate("E2b shape+ball, no rules in decoder", make_ball(False),
                                use_rules=False),
    "E2c": lambda b: b.evaluate("E2c shape+ball+rulefeat, no rules", make_ball(True),
                                use_rules=False),
    "E2d": lambda b: b.evaluate("E2d shape+ball+rulefeat + rules", make_ball(True)),
    "E7": lambda b: b.evaluate("E7 full + rule internals, no rules", full_plus,
                               use_rules=False),
    "E7big": lambda b: b.evaluate("E7 + 600 iters", full_plus, use_rules=False,
                                  model_params={"max_iter": 600}),
    "E8": lambda b: b.evaluate("E8 E7 + ball extra, no rules", full_plus2, use_rules=False,
                               dec_params={"w_shape": 0.75, "cost_dead": 12.0}),
    "E9": lambda b: b.evaluate("E9 E8 + lr/swap mirrors", full_plus2, use_rules=False,
                               dec_params={"w_shape": 0.75, "cost_dead": 12.0},
                               extra=mirrored(full_plus2)),
    "E3b": lambda b: b.evaluate("E3b E8 + noise-level copies", full_plus2, use_rules=False,
                                dec_params={"w_shape": 0.75, "cost_dead": 12.0},
                                extra=augmented(full_plus2)),
    "E3": lambda b: b.evaluate("E3 shape-only + noise augmentation", shape_only,
                               extra=augmented(shape_only)),
    "E5a": lambda b: b.evaluate("E5a current + snap 25", shape_only, dec_params={"snap_r": 25}),
    "E5b": lambda b: b.evaluate("E5b current + snap 50", shape_only, dec_params={"snap_r": 50}),
}

if __name__ == "__main__":
    b = Bench()
    for key in sys.argv[1:]:
        EXPERIMENTS[key](b)
