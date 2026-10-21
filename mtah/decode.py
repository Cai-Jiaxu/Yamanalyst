"""Fuse the rule system and the ball-free shape model with a 3-state Viterbi decoder.

States H, A, D. Per-frame log-emission is

    w_shape * log P_shape(state)                       always (the model never sees the ball)
  + w_rule  * log(q if state == rule_label else ...)   only where the ball was seen nearby

so the rules decide wherever the ball is visible -- and keep their touch-accurate boundaries --
while the shape model carries the long gaps. Switching costs are paid per transition, separately
for H<->A and alive<->dead: analysis/ceiling.py shows spurious switches are the most expensive
error on spellF1, so these are the knobs that trade recall of short possessions for precision.
"""

import numpy as np

STATES = np.array(["H", "A", "D"])

PARAMS = {
    "w_shape": 1.0,
    "w_rule": 1.0,
    "q_rule": 0.97,       # how much to trust the rule label where the ball is seen
    "seen_r": 12,         # frames: ball seen within this counts as 'seen nearby'
    "cost_team": 12.0,    # nats per H<->A switch (mid-plateau, analysis/decoder_tune.py)
    "cost_dead": 6.0,     # nats per alive<->dead switch
}


def viterbi(logE, T):
    """logE (n, 3), T (3, 3) log transition. Returns state index path."""
    n = len(logE)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logE[0].copy()
    for t in range(1, n):
        cand = score[:, None] + T          # from i to j
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def emissions(P_shape, rule_labels, seen, p):
    eps = 1e-4
    logE = p["w_shape"] * np.log(np.clip(P_shape, eps, 1.0))
    near = np.convolve(seen.astype(float), np.ones(2 * p["seen_r"] + 1), "same") > 0
    q = p["q_rule"]
    rl = np.full((len(rule_labels), 3), np.log((1 - q) / 2))
    for k, s in enumerate(STATES):
        rl[rule_labels == s, k] = np.log(q)
    use = near & np.isin(rule_labels, STATES)
    logE[use] += p["w_rule"] * rl[use]
    return logE


def decode(m, rule_labels, P_shape, seen, p=PARAMS):
    out = np.full(m.n, "X", dtype="<U1")
    T = np.zeros((3, 3))
    T[0, 1] = T[1, 0] = -p["cost_team"]
    T[0, 2] = T[2, 0] = T[1, 2] = T[2, 1] = -p["cost_dead"]
    logE = emissions(P_shape, rule_labels, seen, p)
    for _, a, b in m.halves:
        out[a:b + 1] = STATES[viterbi(logE[a:b + 1], T)]
    return out
