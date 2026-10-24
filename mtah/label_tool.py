"""Hand-label a validation window from the tactical video (ROADMAP R3). Run by a PERSON.

    uv run python -m mtah.label_tool --window 1
    uv run python -m mtah.label_tool --start 150000 --out work/labels/val_h2_150000.csv
    uv run python -m mtah.label_tool --probe

Opens the video in a local window. You mark where each spell starts; every frame from a mark
to the next one takes that mark's label. Nothing from any model or from the tracking is shown,
so the labels are an independent check (PLAN.md). The CSV goes under work/ (gitignored): hand
labels are derived from the match video and must never be committed or given to an LLM.
mtah/bench.py and analysis/r3_eval.py score every work/labels/val_*.csv.

Stratified 2nd-half 5-minute windows (--window 1..4):
  1  122500..130000  (72.5% ball seen, ~18 spells) -- high-visibility window for masking check
  2  137500..145000  (60.9% ball seen, ~14 spells) -- match-average ball visibility
  3  167500..175000  (49.3% ball seen, ~11 spells) -- heavy missing-ball stretch
  4  182500..190000  (68.7% ball seen, ~19 spells) -- late stoppage-time stretch

Convention (INVESTIGATION.md section 2), H = Barcelona (home), A = Athletic:
  H <-> A   on the new team's FIRST TOUCH of a controlled possession. A touch without control
            (a block, a deflection, a clearance that the other side recovers) does not switch.
  -> D      the frame the ball goes dead (out of play, whistle for a foul / offside / goal).
  D -> H/A  the frame the restart kick moves the ball, to the team taking it.

Keys (window mode)
  k        play / pause                      h  a  d   spell starts HERE (H / A / D)
  j  l     back / forward 1 frame            x         delete the mark at or before here
  J  L     back / forward 1 s (25 frames)    s         save
  i  o     back / forward 10 s               q         save and quit
  1  2     play at 1x / 4x
Re-running with the same --out / --window resumes from the saved marks.

Keys (--probe mode: 300 genuine missing-ball frames, ROADMAP R3)
  v  d  o  mark frame as Visible-live (V), Dead/stoppage (D), or Occluded/off-screen (O) & advance
  j  l     previous / next probe frame       J  L      step -1 / +1 video frame for context
  s  q     save / save and quit (to work/labels/ball_probe_300.csv)
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
WINDOWS = {
    1: (122500, "val_h2_122500.csv"),
    2: (137500, "val_h2_137500.csv"),
    3: (167500, "val_h2_167500.csv"),
    4: (182500, "val_h2_182500.csv"),
}
PROBE_PATH = os.path.join(LABEL_DIR, "ball_probe_300.csv")


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


def probe_frames(n_samples=300, seed=0):
    """Deterministic stratified sample of genuine missing-ball frames in graded play."""
    from mtah.data import Match
    m = Match()
    miss = np.flatnonzero(m.graded_mask() & ~np.isfinite(m.ball[:, 0]))
    rng = np.random.default_rng(seed)
    idx = np.linspace(0, len(miss) - 1, n_samples, dtype=int)
    jitter = rng.integers(-25, 26, size=n_samples)
    idx = np.clip(idx + jitter, 0, len(miss) - 1)
    return np.unique(miss[idx])[:n_samples]


def run_probe(out_path, scale):
    import cv2
    import pandas as pd
    frames = probe_frames()
    marks = {}
    if os.path.isfile(out_path):
        df = pd.read_csv(out_path, dtype={"state": "object"})
        marks = {int(f): str(s) for f, s in zip(df["frame"], df["state"]) if str(s) in ("V", "D", "O")}
        print("resumed {}/{} probe frames from {}".format(len(marks), len(frames), out_path))
    cap = cv2.VideoCapture(video_path())
    idx = next((i for i, f in enumerate(frames) if int(f) not in marks), 0)
    offset = 0
    cv2.namedWindow("probe", cv2.WINDOW_NORMAL)
    while True:
        idx = int(np.clip(idx, 0, len(frames) - 1))
        f = int(frames[idx]) + offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        view = cv2.resize(img, None, fx=scale, fy=scale)
        cur = marks.get(int(frames[idx]), "-")
        text = "probe {}/{}  frame {} ({:+d})  state [{}]  (v=live-vis, d=dead, o=occl, j/l=prev/next, q=save)".format(
            idx + 1, len(frames), int(frames[idx]), offset, cur)
        cv2.putText(view, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(view, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("probe", view)
        key = cv2.waitKey(0) & 0xFF
        if key == 255:
            continue
        c = chr(key)
        if c in "vdo":
            marks[int(frames[idx])] = c.upper()
            offset = 0
            if idx + 1 < len(frames):
                idx += 1
        elif c == "j":
            idx -= 1
            offset = 0
        elif c == "l":
            idx += 1
            offset = 0
        elif c == "J":
            offset -= 1
        elif c == "L":
            offset += 1
        elif c in "sq":
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pd.DataFrame({
                "frame": frames,
                "state": [marks.get(int(x), "X") for x in frames],
            }).to_csv(out_path, index=False)
            print("saved {}/{} probe marks -> {}".format(len(marks), len(frames), out_path))
            if c == "q":
                break
    cap.release()
    cv2.destroyAllWindows()


def main():
    import cv2
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, choices=sorted(WINDOWS),
                    help="preset 2nd-half 5-minute window (1..4)")
    ap.add_argument("--start", type=int, help="first video frame of the window")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--out", help="output CSV path under work/labels/")
    ap.add_argument("--probe", action="store_true",
                    help="label ball visibility on 300 genuine missing-ball frames")
    ap.add_argument("--scale", type=float, default=0.6, help="display size")
    a = ap.parse_args()
    if a.probe:
        run_probe(a.out or PROBE_PATH, a.scale)
        return
    if a.window is not None:
        w_start, w_name = WINDOWS[a.window]
        start = a.start if a.start is not None else w_start
        out_path = a.out or os.path.join(LABEL_DIR, w_name)
    elif a.start is not None and a.out:
        start, out_path = a.start, a.out
    else:
        ap.error("specify --window {1..4}, or --start <frame> --out <path>, or --probe")
    n = source_frames("tactical")
    end = min(n, start + int(a.minutes * 60 * FPS))
    marks = {}
    if os.path.isfile(out_path):
        marks = marks_from_labels(pd.read_csv(out_path, dtype={"label": "object"})["label"].values)
        print("resumed {} marks from {}".format(len(marks), out_path))
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
            k = save(out_path, marks, start, end, n)
            print("saved {} labelled frames, {} marks -> {}".format(k, len(marks), out_path))
            if c == "q":
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
