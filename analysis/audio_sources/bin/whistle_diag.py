#!/usr/bin/env python3
"""Label-free diagnostic: where do tonal whistle events land, and how does
the event count scale with threshold? No labels attached."""
import os, json
import numpy as np

d = np.load(os.path.join(os.path.dirname(__file__), '..', 'cache', 'whistle_tonal.npz'))
score = d['whistle_score']
conc = d['conc_db']
freq_std = d['freq_std_hz']
frame = d['frame']

# event sweep
print("threshold  events>=160ms  first_event_frame  last_event_frame")
for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
    mask = score > thr
    events = []
    run_start = None
    for i, v in enumerate(mask):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            if i - run_start >= 4:
                events.append((run_start, i-1))
            run_start = None
    if run_start is not None and len(mask) - run_start >= 4:
        events.append((run_start, len(mask)-1))
    if events:
        print(f"{thr:6.2f}     {len(events):4d}         {events[0][0]:>8d}         {events[-1][1]:>8d}")
    else:
        print(f"{thr:6.2f}     {0:4d}")

# Known kickoff frames
known = [(11185, "1H kickoff"), (90651, "1H end"), (110808, "2H grade start"),
         (113742, "2H kickoff"), (190197, "2H end")]

# At thr=0.3, list events and nearest known kickoff
print()
print("Events at thr=0.3:")
thr = 0.3
mask = score > thr
events = []
run_start = None
for i, v in enumerate(mask):
    if v and run_start is None:
        run_start = i
    elif not v and run_start is not None:
        if i - run_start >= 4:
            events.append((run_start, i-1))
        run_start = None
if run_start is not None and len(mask) - run_start >= 4:
    events.append((run_start, len(mask)-1))

for s, e in events:
    mid = (s + e) // 2
    dur = (e - s + 1) / 25.0
    # nearest known
    best = min(known, key=lambda kv: abs(kv[0] - mid))
    off = mid - best[0]
    print(f"  frames {s:>7d}..{e:>7d}  mid={mid:>7d}  dur={dur:5.2f}s  score_max={score[s:e+1].max():.3f}  nearest={best[1]} f={best[0]} off={off:+d} ({off/25:+.2f}s)")
