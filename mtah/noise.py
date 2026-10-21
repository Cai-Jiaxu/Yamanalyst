"""Degrade clean, corrected external tracking until it looks like our automatic, uncorrected ATD.

    noisy = degrade(ext_match, seed=0)

Calibrated at runtime against our own tactical ATD (aggregates only):
  * ball: our graded missing fraction (~41%), with gap lengths resampled from our actual gap
    lengths -- mostly > 10 s. Gaps start more often during dead ball (DEAD_BIAS), because in our
    data players move at 1.0 m/s on average when the ball is missing vs 2.0 m/s when seen.
  * players: random dropout spans until the mean tracked count matches ours (~20.5 of 22).
  * team mislabels: short spans where a player's track is reassigned to the other team (seen in
    the public window: GT says H while an 'A' slot sits on the ball).
  * small Gaussian position noise on players and ball.
"""

import numpy as np

# Params
DEAD_BIAS = 3.0            # gap-start weight on dead-ball frames relative to live ones
POS_NOISE = 0.25           # m, players
BALL_NOISE = 0.25          # m
MISLABEL_PER_MIN = 0.5     # team-mislabel spans per minute of play
MISLABEL_S = (2.0, 10.0)   # span length range, seconds
DROP_SPAN_S = (2.0, 30.0)  # player dropout span length range, seconds

_REF = None


def seed_for(name):
    """Stable per-match seed, independent of which other matches happen to be present."""
    import zlib
    return zlib.crc32(name.encode())


def reference_stats():
    """Aggregates from our match: graded ball gap lengths, missing fraction, mean tracked."""
    global _REF
    if _REF is None:
        from mtah.data import Match
        m = Match()
        g = m.graded_mask()
        miss = g & ~np.isfinite(m.ball[:, 0])
        e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
        lens = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)
        tracked = np.isfinite(m.players[g, :, 0]).sum(axis=1).mean()
        _REF = {"gap_lengths": lens, "miss_frac": miss.sum() / g.sum(), "tracked": tracked}
    return _REF


def _ball_mask(labels, play, rng, ref):
    n = len(labels)
    w = np.where(labels == "D", DEAD_BIAS, 1.0) * play
    w = w / w.sum()
    hide = np.zeros(n, dtype=bool)
    target = ref["miss_frac"] * play.sum()
    while (hide & play).sum() < target:
        start = rng.choice(n, p=w)
        length = int(rng.choice(ref["gap_lengths"]))
        hide[start:start + length] = True
    return hide


def degrade(m, seed=0, ref=None):
    ref = ref or reference_stats()
    rng = np.random.default_rng(seed)
    L, W = m.pitch
    play = m.graded_mask()

    ball = m.ball.copy()
    ball[:, 0] += rng.normal(0, BALL_NOISE / L, m.n)
    ball[:, 1] += rng.normal(0, BALL_NOISE / W, m.n)
    ball[_ball_mask(m.labels, play, rng, ref)] = np.nan
    ball[:, 2] = np.nan  # our ATD ball speed column is ~90% empty; never rely on it

    players = m.players.copy()
    P = players.shape[1]
    players[..., 0] += rng.normal(0, POS_NOISE / L, players.shape[:2])
    players[..., 1] += rng.normal(0, POS_NOISE / W, players.shape[:2])

    # dropout spans until the mean tracked count during play matches ours
    def tracked():
        return np.isfinite(players[play, :, 0]).sum(axis=1).mean()
    play_idx = np.flatnonzero(play)
    while tracked() > ref["tracked"]:
        for _ in range(50):
            t = rng.choice(play_idx)
            k = rng.integers(P)
            length = int(rng.uniform(*DROP_SPAN_S) * m.fps)
            players[t:t + length, k] = np.nan

    # team mislabels: move a span of one player's track into a new slot of the other team
    team = list(m.team)
    extra = []
    n_spans = int(MISLABEL_PER_MIN * play.sum() / m.fps / 60)
    for _ in range(n_spans):
        t = rng.choice(play_idx)
        k = rng.integers(P)
        length = int(rng.uniform(*MISLABEL_S) * m.fps)
        slot = np.full((m.n, 3), np.nan, dtype=np.float32)
        slot[t:t + length] = players[t:t + length, k]
        players[t:t + length, k] = np.nan
        extra.append(slot)
        team.append(1 - m.team[k])
    if extra:
        players = np.concatenate([players, np.stack(extra, axis=1)], axis=1)
    return m.copy_with(players=players, ball=ball, team=np.array(team, dtype=np.int8))
