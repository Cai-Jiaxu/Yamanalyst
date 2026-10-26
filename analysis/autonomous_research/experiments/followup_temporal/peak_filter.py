"""Metric-aligned asymmetric temporal logit filter + NMS fragment merge.

Distinct from cycle2 transition-cost levers and from geometric-HMM/semi-Markov work.
Fixed priors (no sweep): asymmetric Gaussian sharpening on logits, NMS on peaks,
merge short fragments into dominant neighbor. Targets measured error mix:
missed 22.3, drift 19.4, merge 10.6 (cycle2 audit).

Run: .venv/bin/python analysis/autonomous_research/experiments/followup_temporal/peak_filter.py
"""
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

FPS = 25
SIGMA_FRAMES = int(0.5 * FPS)   # 0.5 s sharpening
NMS_RADIUS = int(0.5 * FPS)     # suppress peaks within
MERGE_FRAMES = int(1.0 * FPS)   # merge fragments shorter than 1 s


def asymmetric_filter(logP, sigma):
    """Sharpen: subtract a wide blurred version to center peaks, soften shoulders."""
    wide = np.stack([gaussian_filter1d(logP[:, s], sigma * 3) for s in range(3)], axis=1)
    narrow = np.stack([gaussian_filter1d(logP[:, s], sigma) for s in range(3)], axis=1)
    return logP + 0.5 * (narrow - wide)


def nms_merge(labels, probs, nms_radius, merge_frames):
    """Suppress weak peaks (NMS) and merge short fragments into dominant neighbor."""
    n = len(labels)
    # merge short fragments
    bounds = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    runs = np.split(np.arange(n), bounds)
    out = labels.copy()
    for k, run in enumerate(runs):
        if len(run) < merge_frames and 0 < k < len(runs) - 1:
            # assign to neighbor with higher mean prob on that neighbor's class
            prev_cls = labels[runs[k - 1][-1]]
            next_cls = labels[runs[k + 1][0]]
            p_prev = probs[run, prev_cls].mean()
            p_next = probs[run, next_cls].mean()
            out[run] = prev_cls if p_prev >= p_next else next_cls
    return out


def main():
    from analysis.yardsticks import Yard
    from mtah import decode, final
    t0 = time.time()
    y = Yard()
    BASE = dict(decode.PARAMS, **dict(final.DECODER, use_rules=False))

    def baseline(c, P):
        nox = np.full(c.m.n, "X", dtype="<U1")
        return decode.decode(c.m, nox, P, c.f["seen"], BASE, c.f)

    y.report("baseline v2g", baseline)

    def filtered(c, P):
        nox = np.full(c.m.n, "X", dtype="<U1")
        out = nox.copy()
        logP = np.log(np.clip(P, 1e-9, 1.0))
        sharp = asymmetric_filter(logP, SIGMA_FRAMES)
        Q = np.exp(sharp); Q /= Q.sum(1, keepdims=True)
        # decode with same v2g Viterbi but on sharpened probs
        decoded = decode.decode(c.m, nox, Q, c.f["seen"], BASE, c.f)
        # post-merge short fragments
        from mtah.decode import min_duration, stop_gate
        decoded = min_duration(decoded, BASE["min_dead"], 0)
        if BASE["stop_gate"]:
            decoded = stop_gate(decoded, c.f, BASE["stop_gate"])
        return decoded

    y.report("asymmetric gaussian + v2g", filtered)
    print("elapsed {:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
