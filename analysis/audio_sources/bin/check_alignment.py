#!/usr/bin/env python3
"""
Label-free alignment sanity check: does the whistle-band energy spike near
known kickoff / half-end frames? Uses ONLY aggregate frame offsets, no labels.

Known frame boundaries (from challenge/DATA.md):
  - pre-match:       0 .. 11184
  - 1st half:    11185 .. 90651
  - halftime:    90652 .. 110807
  - 2nd half:   110808 .. 190197 (kickoff actually at 113742)
  - after:      190198 .. 199318

Prints ONLY: for each known boundary, the offset (in frames) of the nearest
whistle-band peak within +-6 s. No per-frame audio, no labels.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
d = np.load(os.path.join(ROOT, "cache/audio_tactical_features.npz"))
frame = d["frame"]
wh = d["whistle_db"]
rms = d["rms_db"]
N = len(frame)

# thresholds from summary
thr_w = float(np.percentile(wh, 99))
thr_r = float(np.percentile(rms, 90))
peak_mask = (wh > thr_w) & (rms > thr_r)
peak_frames = frame[peak_mask]

# Known events (frame index, name)
known = [
    (11185,  "1st half kickoff"),
    (90651,  "1st half end"),
    (110808, "2nd half start-of-grading (pre-kickoff)"),
    (113742, "2nd half kickoff"),
    (190197, "2nd half end"),
]

print(f"N frames={N}, whistle p99={thr_w:.1f} dB, rms p90={thr_r:.1f} dB, peaks={len(peak_frames)}")
print(f"{'event':45s} {'frame':>7s}  nearest_peak  offset_f  offset_s")
for f0, name in known:
    if len(peak_frames) == 0:
        print(f"{name:45s} {f0:>7d}  (no peaks)")
        continue
    idx = np.searchsorted(peak_frames, f0)
    cands = []
    for j in (idx-1, idx, idx+1):
        if 0 <= j < len(peak_frames):
            cands.append(peak_frames[j])
    nearest = min(cands, key=lambda p: abs(p - f0))
    off = nearest - f0
    print(f"{name:45s} {f0:>7d}  {nearest:>11d}  {off:>+7d}  {off/25.0:>+6.2f}s")

# Also: distribution of peak density per known segment
segments = [
    (0,     11184, "pre-match"),
    (11185, 90651, "1st half play"),
    (90652, 110807,"halftime"),
    (110808,190197,"2nd half play"),
    (190198,199318,"after"),
]
print()
print(f"{'segment':20s} {'frames':>8s} {'peaks':>6s} {'per_min':>8s}")
for a, b, name in segments:
    m = (peak_frames >= a) & (peak_frames <= b)
    n_peaks = int(m.sum())
    dur_min = (b - a + 1) / 25.0 / 60.0
    per_min = n_peaks / max(dur_min, 1e-6)
    print(f"{name:20s} {b-a+1:>8d} {n_peaks:>6d} {per_min:>8.2f}")
