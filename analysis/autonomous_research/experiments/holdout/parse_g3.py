"""Parse Metrica Sample Game 3 (JSON tracking + events) into the project's external-match npz.

Source (recorded, not redistributed):
  https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_3
  events.json  (1.7 MB)
  metadata.xml (80 KB)
  tracking.txt (52 MB, 143761 frames @25fps)

License: repo ships no LICENSE file; usage is permissive-by-custom, attribution requested,
no redistribution. This script only parses locally and writes cache/ar_holdout/. It does
NOT touch mtah/external.py, run.py, or any production file.

Label rule is identical to mtah.external.metrica_labels (controlled possession):
  H/A at a frame = team of the most recent PASS / RECOVERY / SET PIECE / SHOT.
  CHALLENGE / BALL LOST alone do not switch. Dead runs from BALL OUT / shot out / foul.
Team A (FIFATMA, local) -> H ; Team B (FIFATMB, visiting) -> A.

Output: cache/ar_holdout/ext_g3.npz with the same keys as ext_metrica_{1,2}:
  players (n,P,3) normalised pitch coords + speed, ball (n,3), team (P,) int,
  labels (n,) H/A/D/X, halves, pitch.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "..", "cache", "ar_holdout", "external", "metrica_g3")
OUT = os.path.join(ROOT, "cache", "ar_holdout", "ext_g3.npz")
FPS = 25
L, W = 105.0, 68.0


def parse_tracking(path):
    """Game3 tracking.txt: '<frame>:<px,py>;<px,py>;...;<bx,by>'  (22 players + ball).

    Returns (frame_idx, players(n,P,2), ball(n,2)). Coordinates already normalised 0..1
    with origin top-left, matching the project convention.
    """
    frames = []
    players = []
    ball = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            head, rest = line.split(":", 1)
            # rest = p1x,p1y;p2x,p2y;...;p22x,p22y:bx,by
            player_part, _, ball_part = rest.rpartition(":")
            pts = [p.split(",") for p in player_part.split(";")]
            xy = []
            for p in pts:
                if p[0] == "NaN" or p[0] == "":
                    xy.append((np.nan, np.nan))
                else:
                    xy.append((float(p[0]), float(p[1])))
            bx, by = ball_part.split(",")
            bx = np.nan if bx in ("NaN", "") else float(bx)
            by = np.nan if by in ("NaN", "") else float(by)
            frames.append(int(head))
            players.append(xy)
            ball.append((bx, by))
    frames = np.array(frames)
    players = np.array(players, dtype=np.float32)  # (F, P, 2)
    ball = np.array(ball, dtype=np.float32)        # (F, 2)
    return frames, players, ball


def _sub_names(e):
    s = e.get("subtypes")
    if s is None:
        return []
    if isinstance(s, dict):
        return [s.get("name", "")]
    return [x.get("name", "") for x in s if isinstance(x, dict)]


def derive_labels(events, n):
    """Mirror mtah.external.metrica_labels but driven by the JSON event schema."""
    lab = np.full(n, "X", dtype="<U1")
    marks = []  # (frame, state)
    for e in events:
        team = "H" if e["team"]["name"] == "Team A" else "A"
        typ = e["type"]["name"]
        subs = _sub_names(e)
        sub = " ".join(subs)
        s0 = int(e["start"]["frame"]) - 1
        s1 = int(e["end"]["frame"]) - 1
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


def main():
    frames, players, ball = parse_tracking(os.path.join(SRC, "tracking.txt"))
    # frames are 1-based contiguous (metadata: second_half_end=143761); array length = max frame
    n = int(frames.max())
    P = players.shape[1]
    assert int(frames.min()) == 1, frames.min()
    assert len(frames) == n, (len(frames), n)
    print("tracking: n_frames={} P_players={} frame_range={}..{}".format(
        n, P, frames.min(), frames.max()))

    # place into dense array (frames are 1-based contiguous)
    dens = np.full((n, P, 2), np.nan, dtype=np.float32)
    dens[frames - 1] = players
    bdens = np.full((n, 2), np.nan, dtype=np.float32)
    bdens[frames - 1] = ball

    # speed (m/s) over +-2 frames, same as parse_metrica
    spd = np.full((n, P), np.nan, dtype=np.float32)
    spd[2:-2] = np.hypot((dens[4:, :, 0] - dens[:-4, :, 0]) * L,
                         (dens[4:, :, 1] - dens[:-4, :, 1]) * W) * FPS / 4
    out_players = np.concatenate([dens, spd[..., None]], axis=2).astype(np.float32)
    out_ball = np.concatenate([bdens, np.full((n, 1), np.nan, dtype=np.float32)], axis=1).astype(np.float32)

    # team slots: Game3 tracking order is home(Team A) then away(Team B). Confirm P/2 each.
    half = P // 2
    team = np.array([0] * half + [1] * (P - half), dtype=np.int8)

    events = json.load(open(os.path.join(SRC, "events.json")))["data"]
    labels = derive_labels(events, n)

    # halves from metadata: first 1..69661, second 69662..143761 (1-based) -> 0-based inclusive
    halves = [(1, 0, 69660), (2, 69661, 143760)]
    outside = np.ones(n, dtype=bool)
    for _, a, b in halves:
        outside[a:b + 1] = False
    labels[outside] = "X"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, players=out_players, ball=out_ball, team=team,
                        labels=labels, halves=np.array(halves), pitch=np.array([L, W]))

    # aggregate diagnostics only
    play = labels != "X"
    print("wrote", OUT)
    print("labels: H {:.1%} A {:.1%} D {:.1%} X {:.1%} (of {})".format(
        np.mean(labels == "H"), np.mean(labels == "A"), np.mean(labels == "D"),
        np.mean(labels == "X"), n))
    print("graded play frames: {} | ball seen on graded: {:.1%}".format(
        int(play.sum()), np.isfinite(out_ball[play, 0]).mean()))
    print("players/team: home {} away {}".format(half, P - half))


if __name__ == "__main__":
    main()
