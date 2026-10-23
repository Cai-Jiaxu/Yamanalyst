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
    "min_dead": 0,        # frames; dead spells shorter than this are absorbed (min_duration())
    "min_poss": 0,        # frames; possessions shorter than this, flanked by one team, absorbed
    "em_prior": False,    # adapt class priors to this match by EM (label_shift())
    "kickoff": False,     # add kickoff evidence (mtah/kickoff.py): True / 'all' or 'half'
                          # (see kickoff.evidence)
    "w_bound": 0.0,       # weight of the boundary head's log-odds on the switch cost
    "snap_r": 0,          # frames; >0 moves each boundary to the nearest physical event (snap())
}


def viterbi(logE, T, bonus=None):
    """logE (n, 3), T (3, 3) log transition. Returns state index path.

    bonus: optional (n,) added to every off-diagonal transition INTO frame t -- the boundary
    head's evidence that a change happens here (positive = cheaper to switch)."""
    n = len(logE)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logE[0].copy()
    off = 1.0 - np.eye(3)
    for t in range(1, n):
        cand = score[:, None] + T          # from i to j
        if bonus is not None:
            cand = cand + bonus[t] * off
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def label_shift(P, train_prior, iters=50):
    """Saerens-Latinne-Decaestecker EM: re-weight posteriors to this match's own class mix.

    train_prior is the (H, A, D) frequency of the model's training labels. The match prior is
    re-estimated from the adapted posteriors until it settles. Unsupervised -- no labels used.
    """
    P = np.clip(P, 1e-6, 1.0)
    train_prior = np.asarray(train_prior, dtype=float)
    prior = train_prior.copy()
    for _ in range(iters):
        Q = P * (prior / train_prior)
        Q /= Q.sum(axis=1, keepdims=True)
        prior = Q.mean(axis=0)
    return Q


def emissions(P_shape, rule_labels, seen, p):
    eps = 1e-4
    if p.get("em_prior"):
        P_shape = label_shift(P_shape, p["train_prior"])
    logE = p["w_shape"] * np.log(np.clip(P_shape, eps, 1.0))
    near = np.convolve(seen.astype(float), np.ones(2 * p["seen_r"] + 1), "same") > 0
    q = p["q_rule"]
    rl = np.full((len(rule_labels), 3), np.log((1 - q) / 2))
    for k, s in enumerate(STATES):
        rl[rule_labels == s, k] = np.log(q)
    use = near & np.isin(rule_labels, STATES)
    logE[use] += p["w_rule"] * rl[use]
    return logE


def snap(labels, f, r, fps=25):
    """Move every boundary to the nearest physical event within +-r frames, where one exists.

    The ground truth places H<->A at the new team's first touch and D->alive at the frame the
    ball starts moving (analysis/conventions.py); the Viterbi path, smoothed by the shape model,
    can land a second or two off. Only the boundary moves -- no segment is created or removed --
    and a boundary never moves past its neighbours.
    """
    from mtah import rules
    ob = rules.on_ball(f, rules.PARAMS)
    moving = f["seen"] & (f["bspeed"] > rules.PARAMS["restart_speed"])
    out = labels.copy()
    n = len(labels)
    b = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for i, t in enumerate(b):
        a, c = labels[t - 1], labels[t]
        if "X" in (a, c):
            continue
        lo = max(b[i - 1] + 1 if i else 0, t - r)
        hi = min(b[i + 1] - 1 if i + 1 < len(b) else n - 1, t + r)
        if c in "HA" and a in "HA":
            # first touch of a new-team episode: on-ball for c here, not on the frame before
            cand = np.flatnonzero((ob[lo:hi + 1] == c) & (np.roll(ob, 1)[lo:hi + 1] != c)) + lo
        elif a == "D" and c in "HA":
            cand = np.flatnonzero(moving[lo:hi + 1] & ~np.roll(moving, 1)[lo:hi + 1]) + lo
        else:
            continue
        if not len(cand):
            continue
        new = cand[np.argmin(np.abs(cand - t))]
        if new < t:
            out[new:t] = c
        elif new > t:
            out[t:new] = a
    return out


def min_duration(labels, min_dead, min_poss):
    """Absorb implausibly short spells into their surroundings.

    A dead spell shorter than min_dead becomes the label before it (restarts take seconds; a
    sub-second D is a decoder wobble). A possession shorter than min_poss with the SAME team on
    both sides is a blip inside one possession and becomes that team. A short possession between
    two different labels is a genuine candidate and is left alone.
    """
    out = labels.copy()
    for _ in range(2):
        b = np.concatenate(([0], np.flatnonzero(out[1:] != out[:-1]) + 1, [len(out)]))
        for i in range(1, len(b) - 2):
            a, z = b[i], b[i + 1]
            prev, cur, nxt = out[a - 1], out[a], out[z]
            if "X" in (prev, cur, nxt):
                continue
            if cur == "D" and z - a < min_dead:
                out[a:z] = prev
            elif cur in "HA" and z - a < min_poss and prev == nxt and prev in "HA":
                out[a:z] = prev
    return out


def boundary_bonus(B, w):
    """w * (log b_t - log mean b): zero on an average frame, positive near likely changes,
    negative deep inside spells."""
    eps = 1e-4
    lb = np.log(np.clip(B, eps, 1.0))
    return w * (lb - np.log(np.clip(B.mean(), eps, 1.0)))


def decode(m, rule_labels, P_shape, seen, p=PARAMS, f=None, B=None):
    out = np.full(m.n, "X", dtype="<U1")
    T = np.zeros((3, 3))
    T[0, 1] = T[1, 0] = -p["cost_team"]
    T[0, 2] = T[2, 0] = T[1, 2] = T[2, 1] = -p["cost_dead"]
    logE = emissions(P_shape, rule_labels, seen, p)
    if p.get("kickoff") and f is not None:
        from mtah import kickoff
        mode = p["kickoff"] if isinstance(p["kickoff"], str) else "all"
        logE = logE + kickoff.evidence(m, f, mode=mode)
    bonus = boundary_bonus(B, p["w_bound"]) if B is not None and p.get("w_bound", 0) else None
    for _, a, b in m.halves:
        out[a:b + 1] = STATES[viterbi(logE[a:b + 1], T,
                                      None if bonus is None else bonus[a:b + 1])]
    if p.get("min_dead", 0) or p.get("min_poss", 0):
        out = min_duration(out, p.get("min_dead", 0), p.get("min_poss", 0))
    if p.get("snap_r", 0) > 0 and f is not None:
        out = snap(out, f, p["snap_r"], m.fps)
    return out
