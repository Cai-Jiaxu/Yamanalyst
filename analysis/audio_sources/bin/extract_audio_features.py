#!/usr/bin/env python3
"""
Label-free audio feature extraction prototype (research only).

Goal: from the tactical mp4 audio track, derive per-40 ms-frame (25 fps)
scalar features that a downstream feature worker could attach as columns.
NO labels are attached. NO match audio is sent anywhere. Everything runs
locally.

Outputs (cached under ./cache/):
  audio_tactical_features.npz
      frame: int32 array, length N, tactical video frame index 0..N-1
      rms_db: float32, broadband RMS in dBFS per 40 ms hop
      whistle_db: float32, RMS in 2.0-5.0 kHz band per 40 ms hop
      crowd_db: float32, RMS in 0.15-2.0 kHz band per 40 ms hop
      speech_db: float32, RMS in 0.3-3.4 kHz band per 40 ms hop (proxy)
  audio_tactical_summary.txt: aggregate stats only

Usage:
  PYTHONPATH=./pylibs ../../.venv/bin/python bin/extract_audio_features.py \
      --video ../../../data/tactical/BAR-ATH_tactical.mp4 \
      --out cache/audio_tactical_features.npz
"""
import argparse, os, sys, subprocess, struct, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pylibs"))
import imageio_ffmpeg


def decode_audio_mono(video_path, sr=16000):
    """Pipe 16 kHz mono s16 PCM out of ffmpeg. Returns int16 numpy array."""
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ff, "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-",
    ]
    p = subprocess.run(cmd, capture_output=True, check=True)
    a = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return a, sr


def band_rms(x, sr, lo, hi, hop=640):
    """Per-hop RMS of band-passed x, zero-phase FIR via FFT.
    x: 1D float32 mono @ sr. hop: samples per 40 ms @ 16k = 640.
    Returns float32 array of length n_hops, in dB.
    """
    n = (len(x) // hop) * hop
    x = x[:n].reshape(-1, hop)
    # FFT across hops for band energy (cheap, hop is 640 samples = 20 Hz bin).
    X = np.fft.rfft(x * np.hanning(hop)[None, :], axis=1)
    freqs = np.fft.rfftfreq(hop, d=1.0 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    e = (np.abs(X[:, mask]) ** 2).sum(axis=1) / hop
    db = 10.0 * np.log10(e + 1e-12)
    return db.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    print(f"[audio] decoding {args.video} @ {args.sr} Hz mono ...", flush=True)
    x, sr = decode_audio_mono(args.video, sr=args.sr)
    dur_s = len(x) / sr
    print(f"[audio] decoded {len(x)} samples, {dur_s:.2f} s", flush=True)

    hop = int(round(sr / args.fps))  # 640 @ 16k/25
    # broadband RMS per hop
    n = (len(x) // hop) * hop
    xh = x[:n].reshape(-1, hop)
    rms = np.sqrt((xh ** 2).mean(axis=1))
    rms_db = (20.0 * np.log10(rms + 1e-9)).astype(np.float32)

    whistle_db = band_rms(x, sr, 2000.0, 5000.0, hop=hop)
    crowd_db = band_rms(x, sr, 150.0, 2000.0, hop=hop)
    speech_db = band_rms(x, sr, 300.0, 3400.0, hop=hop)

    N = len(rms_db)
    frames = np.arange(N, dtype=np.int32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        frame=frames,
        rms_db=rms_db,
        whistle_db=whistle_db[:N],
        crowd_db=crowd_db[:N],
        speech_db=speech_db[:N],
        sr=np.int32(sr),
        fps=np.float32(args.fps),
        hop=np.int32(hop),
    )

    # ---- aggregates only (no per-frame printing) ----
    def pct(a, q):
        return float(np.percentile(a, q))

    summary = {
        "video": os.path.basename(args.video),
        "audio_duration_s": round(dur_s, 2),
        "n_hops": int(N),
        "fps": args.fps,
        "sr": sr,
        "rms_db": {
            "mean": round(float(rms_db.mean()), 2),
            "p05": round(pct(rms_db, 5), 2),
            "p50": round(pct(rms_db, 50), 2),
            "p95": round(pct(rms_db, 95), 2),
            "p99": round(pct(rms_db, 99), 2),
        },
        "whistle_db": {
            "mean": round(float(whistle_db.mean()), 2),
            "p50": round(pct(whistle_db, 50), 2),
            "p95": round(pct(whistle_db, 95), 2),
            "p99": round(pct(whistle_db, 99), 2),
        },
        "crowd_db": {
            "mean": round(float(crowd_db.mean()), 2),
            "p50": round(pct(crowd_db, 50), 2),
            "p95": round(pct(crowd_db, 95), 2),
        },
    }
    # count loud whistle peaks: above p99 of whistle_db AND above broadband
    thr_w = pct(whistle_db, 99)
    thr_r = pct(rms_db, 90)
    peaks = int(((whistle_db > thr_w) & (rms_db > thr_r)).sum())
    summary["whistle_peak_hops_est"] = peaks
    summary["whistle_peak_thr_db"] = round(thr_w, 2)

    s = json.dumps(summary, indent=2)
    print("[audio] summary:\n" + s, flush=True)
    with open(args.out.replace(".npz", "_summary.json"), "w") as f:
        f.write(s + "\n")
    print(f"[audio] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
