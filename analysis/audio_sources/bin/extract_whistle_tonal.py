#!/usr/bin/env python3
"""
Tonal whistle detector (label-free) on tactical audio.

Hypothesis under test: a referee whistle is a SUSTAINED NARROW-BAND tone
in the 2-5 kHz range, lasting >= ~0.15 s. Naive band-energy (the earlier
prototype) fires on any loud broadband event (crowd, ball impact). This
detector adds two constraints:
  (a) spectral concentration: the peak FFT bin carries a large fraction of
      the band energy (whistles are ~pure sine; crowd noise is wideband).
  (b) frequency stability: the peak bin persists within +/- 1 bin across
      consecutive hops (a sine stays at one frequency; a shout wanders).

Input: 16 kHz mono float32 (re-decoded from the mp4 via ffmpeg pipe).
Output cache: per-10ms-hop arrays. We then aggregate to 25 fps (40 ms)
by taking the max concentration in each 40 ms window.

NO labels are attached. NO audio is played back. Aggregates only.
"""
import argparse, os, sys, subprocess, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pylibs"))
import imageio_ffmpeg


def decode(video, sr=16000):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", video,
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, check=True)
    a = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return a, time.time() - t0


def tonal_features(x, sr=16000, win=1024, hop=160):
    """Per-hop STFT features at 10 ms hop (160 samples @ 16k).
    Returns:
      peak_freq_hz: (T,) dominant peak frequency in 2-5 kHz
      concentration: (T,) peak_energy / band_energy (0..1)
      peak_db: (T,) peak bin energy in dB
    """
    n = (len(x) - win) // hop
    # frame into (n, win)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(win)[None, :]
    X = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    band = (freqs >= 2000.0) & (freqs <= 5000.0)
    Xb = X[:, band]
    fb = freqs[band]
    mag = np.abs(Xb)
    peak = mag.argmax(axis=1)
    peak_freq = fb[peak]
    peak_energy = mag[np.arange(n), peak] ** 2
    band_energy = (mag ** 2).sum(axis=1) + 1e-12
    concentration = peak_energy / band_energy
    peak_db = 10.0 * np.log10(peak_energy / win + 1e-12)
    return peak_freq.astype(np.float32), concentration.astype(np.float32), peak_db.astype(np.float32), fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    t_start = time.time()
    x, dt_decode = decode(args.video, sr=args.sr)
    print(f"[tonal] decoded {len(x)} samples in {dt_decode:.1f}s", flush=True)

    t0 = time.time()
    peak_freq, conc, peak_db, freqs = tonal_features(x, sr=args.sr)
    dt_feat = time.time() - t0
    print(f"[tonal] STFT features: {len(peak_freq)} hops in {dt_feat:.1f}s", flush=True)

    # Aggregate 10 ms hops -> 25 fps (40 ms) frames. 4 hops per frame.
    hop_s = 160 / args.sr  # 0.01 s
    hops_per_frame = int(round((1.0 / args.fps) / hop_s))  # 4
    n_frames = len(conc) // hops_per_frame
    conc_f = conc[:n_frames * hops_per_frame].reshape(n_frames, hops_per_frame).max(axis=1)
    peak_db_f = peak_db[:n_frames * hops_per_frame].reshape(n_frames, hops_per_frame).max(axis=1)
    # mode peak frequency (argmax concentration per 4-hop block)
    pf = peak_freq[:n_frames * hops_per_frame].reshape(n_frames, hops_per_frame)
    # frequency stability: std of peak freq across the 4 hops (whistle -> low std)
    pf_std = pf.std(axis=1)

    # Whistle event score: concentration * (1 - norm(pf_std)) * loudness
    # Normalize pf_std to [0,1] via a soft cap at 2 bins (~156 Hz).
    stab = np.exp(-pf_std / freqs[1] / 2.0).astype(np.float32)  # freqs[1] ~= bin width
    loud = (peak_db_f - peak_db_f.mean())
    loud = (loud - loud.min()) / (loud.max() - loud.min() + 1e-9)
    score = (conc_f * stab * (0.5 + 0.5 * loud)).astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        frame=np.arange(n_frames, dtype=np.int32),
        conc_db=conc_f,            # spectral concentration 0..1
        peak_db=peak_db_f,         # peak bin dB
        freq_std_hz=pf_std.astype(np.float32),  # frequency stability (lower = more whistle-like)
        stab=stab,                 # stability score 0..1
        whistle_score=score,       # combined event score
        sr=np.int32(args.sr),
        fps=np.float32(args.fps),
        win=np.int32(1024),
        hop=np.int32(160),
    )

    # aggregates only
    def pct(a, q): return float(np.percentile(a, q))
    summary = {
        "n_frames": int(n_frames),
        "decode_s": round(dt_decode, 1),
        "feat_s": round(dt_feat, 1),
        "total_s": round(time.time() - t_start, 1),
        "conc_p50": round(pct(conc_f, 50), 3),
        "conc_p95": round(pct(conc_f, 95), 3),
        "conc_p99": round(pct(conc_f, 99), 3),
        "freq_std_p50_hz": round(pct(pf_std, 50), 1),
        "freq_std_p95_hz": round(pct(pf_std, 95), 1),
        "whistle_score_p95": round(pct(score, 95), 3),
        "whistle_score_p99": round(pct(score, 99), 3),
    }
    # candidate event count at two thresholds
    for thr in (0.5, 0.7, 0.8):
        mask = score > thr
        # group consecutive frames into events, require >=4 frames (160 ms)
        n_events = 0
        run = 0
        for v in mask:
            if v:
                run += 1
            else:
                if run >= 4: n_events += 1
                run = 0
        if run >= 4: n_events += 1
        summary[f"events_thr{thr}_min160ms"] = n_events
    print("[tonal] summary:\n" + json.dumps(summary, indent=2), flush=True)
    with open(args.out.replace(".npz", "_summary.json"), "w") as f:
        f.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
