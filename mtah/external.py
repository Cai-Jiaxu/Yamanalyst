"""Public tracking datasets in the same in-memory shape as mtah.data.Match, with true labels.

    uv run python -m mtah.external          # parse every downloaded match into cache/ext_*.npz

IDSSE (Bassek et al., Sci Data 2025, CC BY 4.0): seven Bundesliga matches, DFL positional XML at
25 fps. The ball's FrameSet carries BallPossession (1 home, 2 away) and BallStatus (1 alive, 0
dead) on every frame -- the same "corrected tracking says who owns it, and whether it is dead"
construction our ground truth uses.

An ExtMatch exposes the attributes the feature code reads (players, ball, team, cam, halves,
pitch, fps, n) plus `labels` ('H','A','D', 'X' outside play). Coordinates are converted to the
ATD convention: normalised 0-1, origin top-left.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import CACHE_DIR  # noqa: E402

EXT_DIR = os.path.join(ROOT, "external")
IDSSE_MATCHES = ("J03WPY", "J03WN1", "J03WMX", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
SECTIONS = ("firstHalf", "secondHalf")


def _idsse_meta(path):
    root = ET.parse(path).getroot()
    teams = {}
    for team in root.iter("Team"):
        teams[team.get("TeamId")] = team.get("Role")
    env = next((el for el in root.iter() if el.get("PitchX")), None)
    L = float(env.get("PitchX")) if env is not None else 105.0
    W = float(env.get("PitchY")) if env is not None else 68.0
    return teams, (L, W)


def parse_idsse(match):
    """Stream one positions XML (~400 MB) into arrays. Returns a dict for np.savez."""
    teams, (L, W) = _idsse_meta(os.path.join(EXT_DIR, "idsse", match + "_meta.xml"))
    path = os.path.join(EXT_DIR, "idsse", match + "_tracking.xml")

    # pass 1 collects everything per FrameSet; frames are then laid out half by half
    sets = []  # (section, team_role or 'ball', N array, X, Y, S, poss, status)
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "FrameSet":
            continue
        section = el.get("GameSection")
        team_id = el.get("TeamId")
        frames = el.findall("Frame")
        N = np.array([int(f.get("N")) for f in frames])
        X = np.array([float(f.get("X")) for f in frames])
        Y = np.array([float(f.get("Y")) for f in frames])
        S = np.array([float(f.get("S") or "nan") for f in frames]) / 3.6  # DFL km/h -> m/s
        if team_id == "BALL":
            poss = np.array([int(f.get("BallPossession") or 0) for f in frames])
            status = np.array([int(f.get("BallStatus") or 0) for f in frames])
            sets.append((section, "ball", N, X, Y, S, poss, status))
        elif teams.get(team_id) in ("home", "guest", "away"):
            role = "home" if teams[team_id] == "home" else "away"
            sets.append((section, role, N, X, Y, S, None, None))
        el.clear()

    # frame index: each half's N range, halves concatenated with a 1-frame gap of X
    offsets, halves, pos = {}, [], 0
    for h, sec in enumerate(SECTIONS, start=1):
        balls = [s for s in sets if s[0] == sec and s[1] == "ball"]
        if not balls:
            continue
        n0, n1 = balls[0][2].min(), balls[0][2].max()
        offsets[sec] = pos - n0
        halves.append((h, pos, pos + (n1 - n0)))
        pos += (n1 - n0) + 2
    n = pos

    persons = [s for s in sets if s[1] != "ball"]
    # one slot per (person FrameSet); the same person in both halves gets two slots, which is
    # harmless: features are team-level sets and a slot is NaN outside its own half
    P = len(persons)
    players = np.full((n, P, 3), np.nan, dtype=np.float32)
    team = np.zeros(P, dtype=np.int8)
    for k, (sec, role, N, X, Y, S, _, _) in enumerate(persons):
        idx = N + offsets[sec]
        players[idx, k, 0] = (X + L / 2) / L
        players[idx, k, 1] = (W / 2 - Y) / W
        players[idx, k, 2] = S
        team[k] = 0 if role == "home" else 1
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    labels = np.full(n, "X", dtype="<U1")
    for sec, _, N, X, Y, S, poss, status in [s for s in sets if s[1] == "ball"]:
        idx = N + offsets[sec]
        ball[idx, 0] = (X + L / 2) / L
        ball[idx, 1] = (W / 2 - Y) / W
        ball[idx, 2] = S
        lab = np.where(status == 0, "D", np.where(poss == 1, "H", np.where(poss == 2, "A", "D")))
        labels[idx] = lab
    return {"players": players, "ball": ball, "team": team, "labels": labels,
            "halves": np.array(halves), "pitch": np.array([L, W])}


def _metrica_tracking(path):
    """(period (n,), frame (n,), players (n, P, 2), ball (n, 2)) from a Metrica sample CSV."""
    import pandas as pd
    t = pd.read_csv(path, skiprows=2)
    cols = list(t.columns)
    xy = []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            xy.append(t.iloc[:, [i, i + 1]].values)
    bi = cols.index("Ball")
    return (t["Period"].values, t["Frame"].values, np.stack(xy, axis=1),
            t.iloc[:, [bi, bi + 1]].values)


def metrica_labels(events, n, fps=25):
    """Per-frame H/A/D from Metrica sample events (frames are 1-based in the files).

    Possession goes to the team of each PASS, RECOVERY, SET PIECE or SHOT at its start frame.
    CHALLENGE and BALL LOST alone change nothing -- the opponent's next RECOVERY or PASS does --
    which is the 'controlled possession' rule of CHALLENGE.md. Dead ball runs from a BALL OUT
    (its end frame), a shot that goes out or in, or a foul, until the next restart event.
    """
    lab = np.full(n, "X", dtype="<U1")
    ev = events.sort_values(["Start Frame", "End Frame"]).reset_index(drop=True)
    marks = []  # (frame, state)
    for _, e in ev.iterrows():
        team = "H" if e["Team"] == "Home" else "A"
        typ, sub = e["Type"], str(e["Subtype"]) if e["Subtype"] == e["Subtype"] else ""
        s0, s1 = int(e["Start Frame"]) - 1, int(e["End Frame"]) - 1
        if typ in ("PASS", "RECOVERY", "SET PIECE"):
            marks.append((s0, team))
        elif typ == "SHOT":
            marks.append((s0, team))
            if "OUT" in sub or "GOAL" in sub:
                marks.append((max(s1, s0 + 1), "D"))
        elif typ == "BALL OUT":
            marks.append((max(s1, s0), "D"))
        elif typ == "FAULT RECEIVED" or (typ == "CHALLENGE" and "FAULT" in sub):
            marks.append((s0, "D"))
    marks.sort(key=lambda x: x[0])
    state = "X"
    k = 0
    for t in range(n):
        while k < len(marks) and marks[k][0] <= t:
            state = marks[k][1]
            k += 1
        lab[t] = state
    return lab


def parse_metrica(game):
    import pandas as pd
    base = os.path.join(EXT_DIR, "metrica", "Sample_Game_{}_".format(game))
    per, frame, hxy, ball = _metrica_tracking(base + "RawTrackingData_Home_Team.csv")
    _, _, axy, _ = _metrica_tracking(base + "RawTrackingData_Away_Team.csv")
    n = int(frame.max())
    idx = frame - 1
    P = hxy.shape[1] + axy.shape[1]
    players = np.full((n, P, 3), np.nan, dtype=np.float32)
    players[idx, :, :2] = np.concatenate([hxy, axy], axis=1)
    # the sample files carry no speed; derive it (m/s) from positions over +-2 frames
    L, W = 105.0, 68.0
    d = np.full((n, P), np.nan)
    d[2:-2] = np.hypot((players[4:, :, 0] - players[:-4, :, 0]) * L,
                       (players[4:, :, 1] - players[:-4, :, 1]) * W) * 25 / 4
    players[..., 2] = d
    team = np.array([0] * hxy.shape[1] + [1] * axy.shape[1], dtype=np.int8)
    b = np.full((n, 3), np.nan, dtype=np.float32)
    b[idx, :2] = ball
    halves = []
    for p in (1, 2):
        f = idx[per == p]
        halves.append((p, int(f.min()), int(f.max())))
    events = pd.read_csv(base + "RawEventsData.csv")
    labels = metrica_labels(events, n)
    outside = np.ones(n, dtype=bool)
    for _, a, z in halves:
        outside[a:z + 1] = False
    labels[outside] = "X"
    return {"players": players, "ball": b, "team": team, "labels": labels,
            "halves": np.array(halves), "pitch": np.array([L, W])}


def cache_path(name):
    return os.path.join(CACHE_DIR, "ext_{}.npz".format(name))


class ExtMatch:
    """Match-like view of a parsed external match. cam is NaN: no tactical camera here."""

    def __init__(self, name):
        z = np.load(cache_path(name))
        self.game, self.source = name, "external"
        self.players, self.ball, self.team = z["players"], z["ball"], z["team"]
        self.labels = z["labels"]
        self.n = len(self.ball)
        self.halves = [tuple(int(v) for v in h) for h in z["halves"]]
        self.pitch = tuple(float(v) for v in z["pitch"])
        self.fps = 25
        self.cam = np.full((self.n, 2), np.nan, dtype=np.float32)
        self.H = self.reproj = self.hvalid = self.tags = None

    def graded_mask(self):
        m = np.zeros(self.n, dtype=bool)
        for _, a, b in self.halves:
            m[a:b + 1] = True
        return m & (self.labels != "X")

    def copy_with(self, players=None, ball=None, team=None):
        """Shallow copy with replaced arrays -- used by the noise simulator."""
        other = object.__new__(ExtMatch)
        other.__dict__.update(self.__dict__)
        if players is not None:
            other.players = players
        if ball is not None:
            other.ball = ball
        if team is not None:
            other.team = team
        return other


def available():
    """Cache names of every parsed external match."""
    names = ["idsse_" + m for m in IDSSE_MATCHES] + ["metrica_{}".format(g) for g in (1, 2)]
    return [k for k in names if os.path.isfile(cache_path(k))]


def _summary(name, d):
    lab = d["labels"]
    play = lab != "X"
    print("{}: {} frames, {} slots, ball seen {:.1%}, H {:.0%} A {:.0%} D {:.0%}".format(
        name, len(lab), d["players"].shape[1], np.isfinite(d["ball"][play, 0]).mean(),
        np.mean(lab[play] == "H"), np.mean(lab[play] == "A"), np.mean(lab[play] == "D")))


def build_all():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for g in (1, 2):
        out = cache_path("metrica_{}".format(g))
        src = os.path.join(EXT_DIR, "metrica", "Sample_Game_{}_RawEventsData.csv".format(g))
        if os.path.isfile(out) or not os.path.isfile(src):
            continue
        d = parse_metrica(g)
        np.savez_compressed(out, **d)
        _summary("metrica_{}".format(g), d)
    for match in IDSSE_MATCHES:
        out = cache_path("idsse_" + match)
        src = os.path.join(EXT_DIR, "idsse", match + "_tracking.xml")
        if os.path.isfile(out) or not os.path.isfile(src):
            continue
        d = parse_idsse(match)
        np.savez_compressed(out, **d)
        _summary("idsse_" + match, d)


if __name__ == "__main__":
    build_all()
