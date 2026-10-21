"""Parse the provided assets once into numpy arrays under cache/, then load them in milliseconds.

    uv run python -m mtah.data            # builds cache/BAR-ATH_tactical.npz

Everything is indexed by VIDEO frame (submission row). The ATD counts from 1, so record frame_id
lands in row frame_id - 1. Coordinates stay normalised 0-1 here; metres are a feature-time choice.

Nothing in this module prints data values -- only shapes and coverage -- because the match data
must not leave this machine (see OVERVIEW.md on hosted LLMs).
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """(n_slots,) int8, 0 home / 1 away. First team in the EPTS metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def parse_atd(raw_path, n_slots, n_frames):
    """(players (n_frames, n_slots, 3), ball (n_frames, 3)) float32, NaN where unobserved.

    The raw line is "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>". Rewriting both separators to commas
    turns every well-formed line into 1 + 3*n_slots + 3 numbers, which pandas' C parser reads far
    faster than a Python loop.
    """
    width = 1 + 3 * n_slots + 3
    with open(raw_path) as handle:
        text = handle.read()
    lines = [line for line in text.replace(";", ",").replace(":", ",").split("\n")
             if line.count(",") == width - 1]
    table = pd.read_csv(io.StringIO("\n".join(lines)), header=None, dtype=np.float64,
                        na_values=["NaN"]).values

    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    rows = table[:, 0].astype(np.int64) - 1
    keep = (rows >= 0) & (rows < n_frames)
    players[rows[keep]] = table[keep, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball[rows[keep]] = table[keep, 1 + 3 * n_slots:]
    return players, ball, int(keep.sum())


def parse_homography(path, n_frames):
    """(H (n,3,3), reprojection_error (n,), valid (n,), camera_centre (n,2) normalised pitch).

    camera_centre is the pitch point under the image centre -- the tactical camera follows play,
    so it is a ball-location proxy for the 41% of frames where the tracker loses the ball.
    """
    table = pd.read_csv(path)
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    H = np.full((n_frames, 3, 3), np.nan)
    H[table["frame"].values] = table[cols].values.reshape(-1, 3, 3)
    err = np.full(n_frames, np.nan)
    err[table["frame"].values] = table["reprojection_error"].values
    valid = np.zeros(n_frames, dtype=bool)
    valid[table["frame"].values] = table["valid"].values.astype(bool)

    centre = np.full((n_frames, 2), np.nan)
    ok = valid & np.isfinite(H).all(axis=(1, 2))
    inv = np.linalg.inv(H[ok])
    p = inv @ np.array([960.0, 540.0, 1.0])
    centre[ok] = p[:, :2] / p[:, 2:3]
    return H.astype(np.float32), err.astype(np.float32), valid, centre.astype(np.float32)


def parse_smart_tagging(path, fps):
    """DataFrame [code, start_frame, end_frame, team, half, type, ...] -- one row per instance."""
    rows = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        rows.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start_frame": int(round(float(inst.findtext("start")) * fps)),
            "end_frame": int(round(float(inst.findtext("end")) * fps)),
            "team": (tags.get("Team") or "").strip(),
            "half": (tags.get("Half") or "").strip(),
            "type": (tags.get("Type") or "").strip(),
        })
    return pd.DataFrame(rows)


def cache_path(game=GAME, source="tactical"):
    return os.path.join(CACHE_DIR, "{}_{}.npz".format(game, source))


def build(game=GAME, source="tactical"):
    names = ["atd_meta", "atd_raw", "homography"]
    meta, raw, homo = require(names, game, source=source)
    n = source_frames(source, game)
    team = team_of_each_track(meta)
    players, ball, n_records = parse_atd(raw, len(team), n)
    H, err, valid, centre = parse_homography(homo, n)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path(game, source), players=players, ball=ball, team=team,
                        H=H, reproj=err, hvalid=valid, cam=centre)
    print("{} {}: {} ATD records -> {} frames, {} slots, ball seen {:.1%}, homography valid {:.1%}"
          .format(game, source, n_records, n, len(team), np.isfinite(ball[:, 0]).mean(),
                  valid.mean()))

    if source == "tactical":
        tags = parse_smart_tagging(require(["smart_tagging"], game)[0], GAMES[game]["fps"])
        tags.to_csv(os.path.join(CACHE_DIR, "{}_smart_tagging.csv".format(game)), index=False)
        print("smart tagging: {} instances".format(len(tags)))


class Match:
    """Everything for one source of one game, loaded from the cache."""

    def __init__(self, game=GAME, source="tactical"):
        path = cache_path(game, source)
        if not os.path.isfile(path):
            build(game, source)
        z = np.load(path)
        self.game, self.source = game, source
        self.players, self.ball, self.team = z["players"], z["ball"], z["team"]
        self.H, self.reproj, self.hvalid, self.cam = z["H"], z["reproj"], z["hvalid"], z["cam"]
        self.n = len(self.ball)
        self.pitch = GAMES[game]["pitch"]
        self.fps = GAMES[game]["fps"]
        self.halves = half_video_blocks(game)  # [(half, first, last)] inclusive
        tag_path = os.path.join(CACHE_DIR, "{}_smart_tagging.csv".format(game))
        self.tags = pd.read_csv(tag_path, keep_default_na=False) if os.path.isfile(tag_path) \
            else None

    def graded_mask(self):
        m = np.zeros(self.n, dtype=bool)
        for _, a, b in self.halves:
            m[a:b + 1] = True
        return m

    def half_of(self):
        """(n,) int8: 1, 2, or 0 for ungraded frames."""
        h = np.zeros(self.n, dtype=np.int8)
        for half, a, b in self.halves:
            h[a:b + 1] = half
        return h


if __name__ == "__main__":
    build()
