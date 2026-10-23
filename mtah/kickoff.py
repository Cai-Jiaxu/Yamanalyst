"""Algorithmic kickoff detection: half starts and restarts after goals.

A kickoff is the one moment every player stands in his own half: the formation holds for at
least FORM_FRAMES, then the ball leaves the centre spot (or, with the ball unseen, the formation
breaks as players cross halfway). The kicking team is the side at the ball.

Used by the decoder (mtah/decode.py, 'kickoff'): every graded frame before a half's first
kickoff is dead ball -- the 2nd half here is graded 2,934 frames before its kickoff -- and the
kicking team owns the first moments after each kickoff.

    kicks = detect(m, f)      # [(frame, 'H' or 'A', half)]
"""

import warnings

import numpy as np

from mtah import shape

OWN_TOL = 1.0        # m past halfway still counts as own half
MAX_OUT = 1          # players per team allowed over the line (a kicker steps across)
MIN_PLAYERS = 7      # per team, so a sparse frame cannot fake the formation
FORM_FRAMES = 75     # the formation must hold this long (3 s: teams line up and wait)
STILL_PACE = 1.5     # m/s, median mean-player speed during the formation (standing, not
                     # open play that happens to sit in own halves for a moment)
SPOT_R = 4.0         # m, ball this close to the centre spot is 'on the spot'
MOVE_SPEED = 3.0     # m/s, the ball has been kicked
SEARCH = 75          # frames after the formation ends in which to look for the kick
MERGE = 1500         # frames: formations closer than this are one stoppage; keep the last kick
BRIDGE = 50          # frames: a formation broken for less than this (a player drifting over the
                     # line, a slot dropping out) is still one formation. Unbridged, our 2nd-half
                     # kickoff was found 29 s early, at the first break


PRE = 3000           # frames before a half's graded start in which its formation can form
HALF_WAIT = (250, 4500)  # frames: 'half' mode imposes pre-kickoff dead ball only for a waiting
                     # period of 10 s to 3 min (ours: 2,934). Later 'first' kickoffs on external
                     # data are restarts after goals, up to 35 min in (bench: IDSSE -0.021)
FIRST_WITHIN = 15000  # a half's first kickoff is looked for in its first 10 minutes


def _sign(m):
    """Attack sign per frame, extended PRE frames back from each half (the 1st half here is
    graded from the kickoff frame itself, so the formation forms before grading starts)."""
    sgn = shape.attack_sign(m)
    out = sgn.copy()
    for _, s, e in m.halves:
        out[max(0, s - PRE):s] = sgn[s]
    return out


def formation(m):
    """(n,) bool: both teams entirely in their own halves."""
    L, W = m.pitch
    sgn = _sign(m)[:, None]
    x = (m.players[..., 0] - 0.5) * L * sgn      # home attacks +x
    ok = np.isfinite(x)
    home = (m.team == 0)[None] & ok
    away = (m.team == 1)[None] & ok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nh, na = home.sum(1), away.sum(1)
        oh = (home & (x >= OWN_TOL)).sum(1)
        oa = (away & (x <= -OWN_TOL)).sum(1)
    return (nh >= MIN_PLAYERS) & (na >= MIN_PLAYERS) & (oh <= MAX_OUT) & (oa <= MAX_OUT) & \
        (sgn[:, 0] != 0)


def _runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def _bridge(mask, gap):
    out = mask.copy()
    runs = _runs(mask)
    for (_, b), (a, _) in zip(runs[:-1], runs[1:]):
        if a - b < gap:
            out[b:a] = True
    return out


def detect(m, f):
    L, W = m.pitch
    form = _bridge(formation(m), BRIDGE)
    on_spot = f["seen"] & (np.hypot(f["bx"] - L / 2, f["by"] - W / 2) < SPOT_R)
    kicked = on_spot & (f["bspeed"] > MOVE_SPEED)
    px, py = m.players[..., 0] * L, m.players[..., 1] * W
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pace = np.nanmean(m.players[..., 2], axis=1)
    out = []
    for a, b in _runs(form):
        # standing test excludes the last 2 s, when players start running at the kick
        if b - a < FORM_FRAMES or not np.nanmedian(pace[a:max(a + 25, b - 50)]) < STILL_PACE:
            continue
        hits = np.flatnonzero(kicked[a:b + SEARCH]) + a
        t = int(hits[0]) if len(hits) else int(b)
        half = next((h for h, s, e in m.halves if s - PRE <= t <= e), 0)
        if not half:
            continue
        start = next(s for h, s, e in m.halves if h == half)
        first = t - start < FIRST_WITHIN and not any(k[2] == half for k in out)
        if not len(hits) and not first:
            continue  # mid-half: only a formation with the ball seen leaving the spot counts
        t = max(t, start)
        # kicking team: nearest player to the ball if seen, else to the centre spot
        tx, ty = (f["bx"][t], f["by"][t]) if f["seen"][t] else (L / 2, W / 2)
        d = np.hypot(px[t] - tx, py[t] - ty)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dh = np.nanmin(np.where(m.team == 0, d, np.nan))
            da = np.nanmin(np.where(m.team == 1, d, np.nan))
        team = "H" if dh < da else "A"
        out.append((t, team, half, bool(len(hits))))
    merged = []
    for k in out:
        if merged and k[0] - merged[-1][0] < MERGE and k[2] == merged[-1][2]:
            merged[-1] = k
        else:
            merged.append(k)
    return merged


def evidence(m, f, n_after=25, strength=8.0, mode="all"):
    """(n, 3) log-evidence (H, A, D) to add to the decoder's emissions.

    mode 'all': dead ball before each half's first kickoff, and the kicking team after every
    kickoff. 'half': only the dead ball before a half's first kickoff, and only for a waiting
    period within HALF_WAIT (a half graded from the kick itself gets nothing).

    Pre-kickoff dead ball is imposed only when the kick itself was SEEN (ball leaving the centre
    spot): a formation-only detection can be late, and forcing D on everything before a late
    detection is catastrophic (bench: IDSSE spellF1 0.612 -> 0.540 before this guard)."""
    E = np.zeros((m.n, 3))
    kicks = detect(m, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pace = np.nanmean(m.players[..., 2], axis=1)
    for h, s, e in m.halves:
        first = [k for k in kicks if k[2] == h]
        if not (first and first[0][3]):
            continue
        t0 = first[0][0]
        # only a stretch that looks like waiting for the kickoff (players standing or walking)
        # becomes dead ball; a 'first' detection after minutes of open play is a restart after
        # a goal whose half-opening kickoff we did not see -- impose nothing before it
        wait = HALF_WAIT[0] < t0 - s <= HALF_WAIT[1]
        if (wait or mode == "all") and (t0 - s <= 25 or np.nanmedian(pace[s:t0]) < STILL_PACE):
            E[s:t0, 2] += strength * 4   # before a half's first kickoff the ball is dead
            E[s:t0, :2] -= strength * 4
    for t, team, _, _ in (kicks if mode == "all" else []):
        k = 0 if team == "H" else 1
        E[t:t + n_after, k] += strength
    return E
