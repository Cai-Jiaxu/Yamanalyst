#!/usr/bin/env python3
"""Label-free diagnostic: where do tonal whistle events land, and how does
the event count scale with threshold? No labels attached. Prints aggregates only."""
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
print("Events at thr=0.3 (summary):")
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

# Aggregates only: a per-event list is frame-level data derived from the match audio and must not
# reach a committed log (README "Data rules"). For each published half boundary, report the offset
# of the nearest event; for the rest, counts and durations.
mids = np.array([(s + e) // 2 for s, e in events])
durs = np.array([(e - s + 1) / 25.0 for s, e in events])
print(f"  {len(events)} events, duration median {np.median(durs):.2f}s max {durs.max():.2f}s, "
      f"score_max median {np.median([score[s:e+1].max() for s, e in events]):.3f}")
for f0, name in known:
    off = mids[np.argmin(np.abs(mids - f0))] - f0
    print(f"  nearest event to {name} (f={f0}): {off:+d} frames ({off/25:+.2f}s)")
