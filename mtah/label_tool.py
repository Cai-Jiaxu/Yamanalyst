"""Hand-label a validation window from the tactical video (ROADMAP R3). Run by a PERSON.

    uv run python -m mtah.label_tool --start 150000 --out work/labels/val_h2_150000.csv

Opens the video in a local window. You mark where each spell starts; every frame from a mark
to the next one takes that mark's label. Nothing from any model or from the tracking is shown,
so the labels are an independent check (PLAN.md). The CSV goes under work/ (gitignored): hand
labels are derived from the match video and must never be committed or given to an LLM.
mtah/bench.py scores every work/labels/*.csv as its 'val' column.

Convention (INVESTIGATION.md section 2), H = Barcelona (home), A = Athletic:
  H <-> A   on the new team's FIRST TOUCH of a controlled possession. A touch without control
            (a block, a deflection, a clearance that the other side recovers) does not switch.
  -> D      the frame the ball goes dead (out of play, whistle for a foul / offside / goal).
  D -> H/A  the frame the restart kick moves the ball, to the team taking it.

Keys
  k        play / pause                      h  a  d   spell starts HERE (H / A / D)
  j  l     back / forward 1 frame            x         delete the mark at or before here
  J  L     back / forward 1 s (25 frames)    s         save
  i  o     back / forward 10 s               q         save and quit
  1  2     play at 1x / 4x
Re-running with the same --out resumes from the saved marks.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import source_frames, video_path  # noqa: E402

FPS = 25
LABEL_DIR = os.path.join(ROOT, "work", "labels")


def labels_from_marks(marks, start, end, n):
    """(n,) array: 'X' outside [start, end), else the label of the latest mark at or before
    each frame ('X' before the first mark)."""
    out = np.full(n, "X", dtype="<U1")
    keys = sorted(f for f in marks if start <= f < end)
    for i, f in enumerate(keys):
        stop = keys[i + 1] if i + 1 < len(keys) else end
        out[f:stop] = marks[f]
    return out


def marks_from_labels(labels):
    """Inverse of labels_from_marks for a saved file: {frame: label} at every change."""
    marks = {}
    prev = "X"
    for f in np.flatnonzero(labels != "X"):
        if labels[f] != prev or labels[f - 1] == "X":
            marks[int(f)] = str(labels[f])
        prev = labels[f]
    return marks


def save(path, marks, start, end, n):
    import pandas as pd
    lab = labels_from_marks(marks, start, end, n)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"frame": np.arange(n), "label": lab}).to_csv(path, index=False)
    return int((lab != "X").sum())


def main():
    import cv2
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True, help="first video frame of the window")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.6, help="display size")
    a = ap.parse_args()
    n = source_frames("tactical")
    start, end = a.start, min(n, a.start + int(a.minutes * 60 * FPS))
    marks = {}
    if os.path.isfile(a.out):
        marks = marks_from_labels(pd.read_csv(a.out, dtype={"label": "object"})["label"].values)
        print("resumed {} marks from {}".format(len(marks), a.out))
    cap = cv2.VideoCapture(video_path())
    pos, playing, speed, img, at = start, False, 1, None, None

    def read(f):
        nonlocal at, img
        if at is None or f != at + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if ok:
            img, at = frame, f
        return ok

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    while True:
        pos = int(np.clip(pos, start, end - 1))
        if at != pos:
            read(pos)
        view = cv2.resize(img, None, fx=a.scale, fy=a.scale)
        cur = labels_from_marks(marks, start, end, end)[pos]
        text = "frame {}  ({:+.1f} s)  label {}  marks {}  {}".format(
            pos, (pos - start) / FPS, cur, len(marks),
            "MARK HERE: " + marks[pos] if pos in marks else "")
        cv2.putText(view, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(view, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("label", view)
        key = cv2.waitKey(max(1, int(1000 / FPS / speed)) if playing else 0) & 0xFF
        if key == 255:
            if playing:
                pos += 1 if speed == 1 else 4
            continue
        c = chr(key)
        step = {"j": -1, "l": 1, "J": -FPS, "L": FPS, "i": -10 * FPS, "o": 10 * FPS}
        if c in step:
            pos += step[c]
            playing = False
        elif c == "k":
            playing = not playing
        elif c in "12":
            speed = 1 if c == "1" else 4
        elif c in "had":
            marks[pos] = c.upper()
        elif c == "x":
            before = [f for f in marks if f <= pos]
            if before:
                del marks[max(before)]
        elif c in "sq":
            k = save(a.out, marks, start, end, n)
            print("saved {} labelled frames, {} marks -> {}".format(k, len(marks), a.out))
            if c == "q":
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
