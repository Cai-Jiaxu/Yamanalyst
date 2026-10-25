"""Re-check final_v3/v4 against final_v2 + stop gate on every yardstick (aggregates only).

    uv run python analysis/rev_v4.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, final  # noqa: E402


def dec(**kw):
    p = dict(decode.PARAMS, **dict(final.DECODER, **kw))

    def fn(c, P):
        R = c.R if p.get("use_rules") else np.full(c.m.n, "X", dtype="<U1")
        return decode.decode(c.m, R, P, c.f["seen"], p, c.f)
    return fn


if __name__ == "__main__":
    y = Yard()
    y.report("rules only", lambda c, P: c.R)
    y.report("final_v2 (no rule vote)", dec(use_rules=False, stop_gate=0))
    y.report("final_v2 + stop gate", dec(use_rules=False))
    y.report("final_v3 (rule vote 0.25)", dec(stop_gate=0))
    y.report("final_v4 (v3 + gate)", dec())
