"""Local ball detection on the tactical video (W7). Everything runs on this machine.

    uv run python -m mtah.video bench            # speed + accuracy vs the ATD ball (aggregates)
    uv run python -m mtah.video run              # detect on every graded missing-ball frame

A COCO-pretrained YOLO ('sports ball', class 32) runs on the frames we ask for; every candidate
is mapped to the pitch through that frame's homography (pitch -> pixel, inverted), candidates
off the pitch are dropped, and the TOP_K per frame are cached in cache/ball_det_<model>.npz as
(frame, x_m, y_m, conf). No frame, crop or detection ever leaves the machine; ultralytics'
analytics ('sync') is switched off before the model loads.
"""

import os
import sys
import time

import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import video_path, source_frames  # noqa: E402
from mtah.data import CACHE_DIR, Match  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")
BALL_CLASS = 32
CONF_MIN = 0.05
TOP_K = 5
PITCH_MARGIN = 3.0     # m outside the lines still kept (balls go out)


def load_model(name):
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    return YOLO(os.path.join(MODEL_DIR, name + ".pt"))


def frames(path, wanted):
    """Yield (frame_index, image) for the sorted frame indices in `wanted`, decoding sequentially
    and only retrieving the frames asked for."""
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n != source_frames("tactical"):
        raise ValueError("video has {} frames, the submission contract says {}".format(
            n, source_frames("tactical")))
    wanted = np.asarray(sorted(wanted))
    k, i = 0, 0
    while k < len(wanted):
        if i == wanted[k]:
            ok, img = cap.read()
            if not ok:
                break
            yield i, img
            k += 1
        else:
            if wanted[k] - i > 250:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(wanted[k]))
                i = int(wanted[k])
                continue
            cap.grab()
        i += 1
    cap.release()


def to_pitch(H, px, py):
    """Pixel -> metres on the pitch through the inverse of the pitch->pixel homography."""
    inv = np.linalg.inv(H)
    v = inv @ np.stack([px, py, np.ones_like(px)])
    return v[0] / v[2], v[1] / v[2]


TILES_2x2 = [(0, 0, 1024, 576), (896, 0, 1920, 576), (0, 504, 1024, 1080),
             (896, 504, 1920, 1080)]   # 128 px / 72 px overlap so no ball is cut in half


def detect(model, m, wanted, imgsz=1920, device="mps", batch=8, log_every=2000, tiles=None):
    """Array (k, 4): frame, x_m, y_m, conf -- up to TOP_K on-pitch candidates per frame.

    tiles: optional [(x0, y0, x1, y1)] crops run separately (each enlarged to imgsz), boxes mapped
    back to full-frame pixels -- the standard fix for objects a few pixels across."""
    L, W = m.pitch
    rows, buf, idx = [], [], []
    t0 = time.time()

    def flush():
        crops, owner, offs = [], [], []
        for fi, img in zip(idx, buf):
            for (x0, y0, x1, y1) in (tiles or [(0, 0, img.shape[1], img.shape[0])]):
                crops.append(img[y0:y1, x0:x1])
                owner.append(fi)
                offs.append((x0, y0))
        per = {}
        names = getattr(model, "names", {}) or {}
        cls_ids = [0] if len(names) == 1 else [BALL_CLASS]
        for k in range(0, len(crops), batch):
            res = model.predict(crops[k:k + batch], imgsz=imgsz, classes=cls_ids,
                                conf=CONF_MIN, device=device, verbose=False)
            for j, r in enumerate(res):
                if r.boxes is None or not len(r.boxes):
                    continue
                b = r.boxes.xyxy.cpu().numpy().copy()
                ox, oy = offs[k + j]
                b[:, [0, 2]] += ox
                b[:, [1, 3]] += oy
                per.setdefault(owner[k + j], []).append((b, r.boxes.conf.cpu().numpy()))
        for fi, parts in per.items():
            xyxy = np.vstack([p[0] for p in parts])
            conf = np.concatenate([p[1] for p in parts])
            cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]   # ground contact: box bottom
            x, y = to_pitch(m.H[fi].astype(np.float64), cx, cy)
            x, y = x * L, y * W
            keep = (x > -PITCH_MARGIN) & (x < L + PITCH_MARGIN) & (y > -PITCH_MARGIN) & \
                (y < W + PITCH_MARGIN)
            order = np.argsort(-conf[keep])[:TOP_K]
            for j in order:
                rows.append((fi, x[keep][j], y[keep][j], conf[keep][j]))
        buf.clear()
        idx.clear()

    done = 0
    for fi, img in frames(video_path(), wanted):
        buf.append(img)
        idx.append(fi)
        if len(buf) == batch:
            flush()
        done += 1
        if log_every and done % log_every == 0:
            el = time.time() - t0
            print("  {} / {} frames, {:.1f} fps, eta {:.0f} min".format(
                done, len(wanted), done / el, (len(wanted) - done) / (done / el) / 60),
                flush=True)
    if buf:
        flush()
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def accuracy(det, m, frames_used, radius=2.0):
    """Against the ATD ball on frames where the tracker sees it: share of frames with a candidate
    within `radius` m (recall), and how often the top candidate is that close (precision)."""
    L, W = m.pitch
    bx, by = m.ball[:, 0] * L, m.ball[:, 1] * W
    hit_any, hit_top, has = 0, 0, 0
    for fi in frames_used:
        d = det[det[:, 0] == fi]
        if not len(d):
            continue
        has += 1
        dist = np.hypot(d[:, 1] - bx[fi], d[:, 2] - by[fi])
        hit_any += dist.min() < radius
        hit_top += dist[np.argmax(d[:, 3])] < radius
    n = len(frames_used)
    return {"frames": n, "with_candidate": has / n, "recall_any": hit_any / n,
            "top1_within_r_given_candidate": hit_top / max(has, 1)}


def cache_path(model_name):
    return os.path.join(CACHE_DIR, "ball_det_{}.npz".format(model_name))


def bench(n=300, seed=0):
    m = Match()
    seen = np.flatnonzero(m.graded_mask() & np.isfinite(m.ball[:, 0]))
    sample = np.sort(np.random.default_rng(seed).choice(seen, n, replace=False))
    configs = [("yolo11m", 1024, TILES_2x2, 4),
               ("yolo11x", 1024, TILES_2x2, 4)]
    for name, imgsz, tiles, batch in configs:
        model = load_model(name)
        t = time.time()
        det = detect(model, m, sample, imgsz=imgsz, log_every=0, tiles=tiles, batch=batch)
        fps = n / (time.time() - t)
        acc = accuracy(det, m, sample)
        print("{} @{} tiles={}: {:.1f} fps | candidate on {:.0%} of frames | some candidate within 2 m "
              "on {:.0%} | top-1 within 2 m when present {:.0%}".format(
                  name, imgsz, len(tiles or [1]), fps, acc["with_candidate"], acc["recall_any"],
                  acc["top1_within_r_given_candidate"]), flush=True)


def run(model_name="yolo11m", imgsz=1920):
    """Detect on every graded frame where the ATD ball is missing, plus the public window (for
    masked validation) -- cached."""
    m = Match()
    graded = m.graded_mask()
    wanted = graded & ~np.isfinite(m.ball[:, 0])
    wanted[11185:18685 + 750] = True
    wanted = np.flatnonzero(wanted)
    print("detecting on {} frames with {} @{}".format(len(wanted), model_name, imgsz), flush=True)
    det = detect(load_model(model_name), m, wanted, imgsz=imgsz)
    np.savez_compressed(cache_path(model_name), det=det, frames=wanted)
    print("saved {} candidates".format(len(det)), flush=True)



def probe(n_gaps=12, step=5, seed=0, model=None, imgsz=1024, tiles=TILES_2x2, batch=4,
          device="mps"):
    """Yield on GENUINE missing-ball frames (no ground truth there, so measure plausibility):
    a sampled frame counts if its top candidate is near a player (< 4 m) and within a physically
    possible distance of the previous sample's top candidate (< 30 m/s * dt)."""
    m = Match()
    L, W = m.pitch
    miss = m.graded_mask() & ~np.isfinite(m.ball[:, 0])
    e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    gaps = [(a, b) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)) if b - a > 250]
    rng = np.random.default_rng(seed)
    pick = [gaps[i] for i in sorted(rng.choice(len(gaps), n_gaps, replace=False))]
    wanted = np.concatenate([np.arange(a, b, step) for a, b in pick])
    model = model or load_model("yolo11m")
    det = detect(model, m, wanted, imgsz=imgsz, tiles=tiles, batch=batch, device=device,
                 log_every=0)
    px, py = m.players[..., 0] * L, m.players[..., 1] * W
    ok, prev, conf_ok = 0, None, 0
    for fi in wanted:
        d = det[det[:, 0] == fi]
        if not len(d):
            prev = None
            continue
        top = d[np.argmax(d[:, 3])]
        conf_ok += top[3] >= 0.25
        near = np.nanmin(np.hypot(px[int(fi)] - top[1], py[int(fi)] - top[2])) < 4.0
        cons = prev is not None and np.hypot(top[1] - prev[1], top[2] - prev[2]) < \
            30.0 * (fi - prev[0]) / 25
        ok += bool(near and cons)
        prev = top
    print("missing-ball probe: {} frames from {} gaps | top candidate conf>=0.25 on {:.0%} | "
          "near a player AND consistent with previous sample on {:.0%}".format(
              len(wanted), n_gaps, conf_ok / len(wanted), ok / len(wanted)), flush=True)
    return {"frames": len(wanted), "conf_ok": conf_ok / len(wanted), "plausible": ok / len(wanted)}


if __name__ == "__main__":
    {"bench": bench, "run": run, "probe": probe}[sys.argv[1]](*sys.argv[2:])
