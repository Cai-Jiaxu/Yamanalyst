"""Rule system v2: touch episodes -> controlled possession, plus a dead-ball state machine.

Conventions, from analysis/conventions.py and analysis/diagnose.py on the public window:
  * D -> alive happens on the frame the ball starts moving (ball speed crosses ~3 m/s).
  * H <-> A is placed at the new team's first touch, but only counts if that team then CONTROLS
    the ball: it keeps it for a while, or touches it again before the opponent does (a completed
    pass or a dribble). A touch-and-clear does not change possession.
  * A contested or loose ball stays with the last team (CHALLENGE.md).
  * Stoppages the tracker misses (ball still 'seen', never out of bounds) show up as both teams
    walking for several seconds.
And from analysis/ceiling.py: a spurious blip inside a possession costs far more than a missed
short possession, so every rule errs on the side of not switching.

    labels = run(Match(), compute(Match()), PARAMS)   # (n,) array of 'H','A','D','X'
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

PARAMS = {
    "r_touch": 1.5,        # m, a player this close to the ball can be touching it
    "excl_margin": 0.5,    # m, the other team must be at least this much farther
    "merge_gap": 4,        # frames, on-ball runs of one team closer than this are one episode
    "hold_frames": 25,     # an episode this long is control on its own (~1 s)
    "retouch_s": 4.0,      # s, a second same-team episode within this, before any opponent
                           # episode, confirms control (a pass or a dribble)
    "still_speed": 1.0,    # m/s, ball below this is stationary
    "still_frames": 38,    # frames (~1.5 s) stationary with no one on it -> dead
    "still_r": 1.0,        # m, 'no one on it' = nearest player farther than this
    "restart_speed": 3.0,  # m/s, a dead ball moving faster than this has been restarted
    "missing_dead": 125,   # frames of missing ball with slow players -> dead
    "slow_players": 1.3,   # m/s, mean player speed below this counts as slow
    "walk_speed": 1.25,    # m/s, both teams' mean speed below this ...
    "walk_frames": 100,    # ... for this long (4 s) is a stoppage even with the ball 'in play'
}

H, A, D, X = "H", "A", "D", "X"


def _runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def dead_mask(f, p, fps=25):
    """Boolean (n,) where the ball is judged dead, restarts resolved."""
    n = len(f["bx"])
    dead = f["out"].copy()

    # stationary ball nobody is standing on: a set piece being prepared
    near = np.fmin(f["dh"], f["da"])
    still = f["seen"] & (f["bspeed"] < p["still_speed"]) & ~(near < p["still_r"])
    for a, b in _runs(still):
        if b - a >= p["still_frames"]:
            dead[a:b] = True

    # everyone walking: a stoppage whatever the ball track says
    pace = uniform_filter1d(np.nan_to_num((f["sh"] + f["sa"]) / 2, nan=0.0), 25)
    for a, b in _runs(pace < p["walk_speed"]):
        if b - a >= p["walk_frames"]:
            dead[a:b] = True

    # long missing ball while players are slow: a stoppage the tracker could not see
    slow = pace < p["slow_players"]
    for a, b in _runs(~f["seen"]):
        if b - a >= p["missing_dead"] and np.mean(slow[a:b]) > 0.5:
            dead[a:b] = True

    # a dead stretch lasts until the ball is seen moving again
    moving = f["seen"] & (f["bspeed"] > p["restart_speed"])
    i = 0
    while i < n:
        if dead[i]:
            j = i
            while j < n and (dead[j] or not moving[j]):
                j += 1
            dead[i:j] = True
            i = j
        i += 1
    return dead


def on_ball(f, p):
    """(n,) team code on the ball this frame: 'H', 'A' or '' (nobody, contested, unseen)."""
    dh, da = f["dh"], f["da"]
    out = np.full(len(dh), "", dtype="<U1")
    h = f["seen"] & (dh < p["r_touch"]) & ~(da < dh + p["excl_margin"])
    a = f["seen"] & (da < p["r_touch"]) & ~(dh < da + p["excl_margin"])
    out[h] = H
    out[a] = A
    return out


def episodes(ob, p, start, stop):
    """[(team, first, last_exclusive)] maximal on-ball runs per team, small gaps merged."""
    eps = []
    for team in (H, A):
        for a, b in _runs(ob[start:stop] == team):
            a, b = a + start, b + start
            if eps and eps[-1][0] == team and a - eps[-1][2] < p["merge_gap"]:
                eps[-1] = (team, eps[-1][1], b)
            else:
                eps.append((team, a, b))
    eps.sort(key=lambda e: e[1])
    # re-merge after interleaving the two teams' runs
    merged = []
    for e in eps:
        if merged and merged[-1][0] == e[0] and e[1] - merged[-1][2] < p["merge_gap"]:
            merged[-1] = (e[0], merged[-1][1], e[2])
        else:
            merged.append(e)
    return merged


def controlled(eps, k, p, fps):
    """Does episode k amount to its team taking control?"""
    team, a, b = eps[k]
    if b - a >= p["hold_frames"]:
        return True
    horizon = b + p["retouch_s"] * fps
    for team2, a2, _ in eps[k + 1:]:
        if a2 > horizon:
            return False
        return team2 == team  # the very next episode decides
    return False


def possession(f, p, dead, graded_runs, fps=25):
    """Labels H/A/D for graded frames, X elsewhere."""
    n = len(dead)
    ob = on_ball(f, p)
    ob[dead] = ""
    labels = np.full(n, X, dtype="<U1")
    for start, stop in graded_runs:
        eps = episodes(ob, p, start, stop)
        # change points: (frame, team) where a controlled episode of a new team starts
        switch = np.full(stop - start, "", dtype="<U1")
        for k, (team, a, _) in enumerate(eps):
            if controlled(eps, k, p, fps):
                switch[a - start] = team
        first_touch = np.full(stop - start, "", dtype="<U1")
        for team, a, _ in eps:
            first_touch[a - start] = team

        owner = ""
        for t in range(start, stop):
            if dead[t]:
                labels[t] = D
                owner = ""  # a restart re-decides possession from scratch
                continue
            if owner == "":
                # just restarted (or first frame): first team on the ball owns it
                if first_touch[t - start]:
                    owner = first_touch[t - start]
                elif ob[t]:
                    owner = ob[t]
            elif switch[t - start] and switch[t - start] != owner:
                owner = switch[t - start]
            labels[t] = owner if owner else X
        # a restart nobody was seen touching yet: back-fill from the first label after it
        _backfill(labels, start, stop)
    return labels


def _backfill(labels, start, stop):
    """X inside alive stretches right after D: take the next known label (the restarter)."""
    t = stop - 1
    nxt = X
    while t >= start:
        c = labels[t]
        if c in (H, A):
            nxt = c
        elif c == D:
            nxt = X
        elif c == X and nxt != X:
            labels[t] = nxt
        t -= 1


def fill_x(labels, graded_runs):
    """Never leave X on a graded frame: the scorer would forward-fill it for spellF1 anyway, and
    macroF1 charges it as a miss. Forward-fill, then back-fill the head of each half."""
    for start, stop in graded_runs:
        last = X
        for t in range(start, stop):
            if labels[t] == X:
                labels[t] = last
            else:
                last = labels[t]
        first = next((c for c in labels[start:stop] if c != X), D)
        for t in range(start, stop):
            if labels[t] != X:
                break
            labels[t] = first
    return labels


def run(m, f, p=PARAMS):
    dead = dead_mask(f, p, m.fps)
    graded = [(a, b + 1) for _, a, b in m.halves]
    return fill_x(possession(f, p, dead, graded, m.fps), graded)
