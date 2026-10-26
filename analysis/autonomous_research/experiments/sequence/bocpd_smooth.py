"""Convention-neutral fixed-prior geometric HMM forward smoother on cached P_R1.

Scope (see analysis/autonomous_research/literature/sequence/REVIEW ...):
- No public-GT fitting. All parameters are generic weak priors predeclared below.
- No external label learning. The only input is cached per-frame 3-class probabilities.
- Output labels are scored with the existing analysis.yardsticks.Yard (official spellF1).

Method: a run-length forward filter over (run-length, state). At each frame, either the
run continues (same state, probability 1-h) or a new run starts at a fresh uniformly-chosen
state (probability h). The observation evidence is P[t, state]. This is the Adams-MacKay
run-length recursion with a non-informative uniform changepoint-state prior, deliberately
NOT modeling H<->A vs alive<->dead transition costs (those stay in the baseline Viterbi).

We do not touch mtah/ or run.py. We only read cache/P_R1.npz and call existing scorers.

Run:
    .venv/bin/python analysis/autonomous_research/experiments/sequence/bocpd_smooth.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

# --- predeclared generic weak priors (NOT fit to any labels) -----------------
HZ = 1.0 / 400.0          # hazard: expected run ~400 frames ~16 s, generic soccer
K_CAP = 800               # run-length cap, frames (~32 s)
N_STATES = 3              # H, A, D
STATE_PRIOR = np.ones(N_STATES) / N_STATES   # uniform; convention-neutral
# -----------------------------------------------------------------------------


def bocpd_smooth(P, hz=HZ, prior=STATE_PRIOR):
    """Forward changepoint smoother with geometric (memoryless) run length.

    q_t[s] = normalize( (1-h) * q_{t-1}[s] * P[t,s]  +  h * prior[s] * P[t,s] )

    - continue term: the run keeps its current state belief and multiplies the new
      observation likelihood.
    - changepoint term: a fresh state drawn from the uniform prior, then multiplied by
      the new observation likelihood.
    No public or external labels are used; hz is a fixed weak prior.
    """
    n = P.shape[0]
    eps = 1e-9
    logP = np.log(np.clip(P, eps, 1.0))
    lprior = np.log(np.clip(prior, eps, 1.0))
    l1mh = np.log1p(-hz)
    lh = np.log(hz)
    q = np.empty((n, N_STATES))
    # first frame: just the observation
    q[0] = logP[0] - np.logaddexp.reduce(logP[0])
    for t in range(1, n):
        cont = l1mh + q[t - 1] + logP[t]
        new = lh + lprior + logP[t]
        s = np.logaddexp(cont, new)
        q[t] = s - np.logaddexp.reduce(s)
    return np.exp(q)


def labels_from_q(q, min_dead=25, sg=100, f=None):
    """Argmax of smoothed posterior, then the same post-processing as v5 decoder
    (min_dead absorb, stop_gate) so the comparison isolates the temporal smoothing."""
    from mtah.decode import STATES, min_duration, stop_gate
    idx = q.argmax(axis=1)
    out = STATES[idx]
    out = min_duration(out, min_dead, 0)
    if sg and f is not None:
        out = stop_gate(out, f, sg)
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

    # baseline v2g (same as mbr_eval)
    y.report("baseline v2g", baseline)

    # geometric HMM-smoothed posteriors -> argmax + same post-processing
    def bocpd_argmax(c, P):
        q = bocpd_smooth(P)
        return labels_from_q(q, BASE["min_dead"], BASE["stop_gate"], c.f)

    y.report("bocpd argmax h=1/400", bocpd_argmax)

    # geometric HMM-smoothed posteriors fed back into the same Viterbi decoder
    def bocpd_viterbi(c, P):
        q = bocpd_smooth(P)
        nox = np.full(c.m.n, "X", dtype="<U1")
        # decode expects probabilities; pass smoothed q as P_shape
        return decode.decode(c.m, nox, q, c.f["seen"], BASE, c.f)

    y.report("bocpd -> viterbi h=1/400", bocpd_viterbi)

    # cache the run for reproducibility
    out_path = os.path.join(ROOT, "cache", "ar_sequence_bocpd.npz")
    np.savez(out_path, hz=HZ, kcap=K_CAP)
    print("cached", out_path, "in {:.0f}s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
