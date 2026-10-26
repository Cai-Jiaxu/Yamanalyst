#!/usr/bin/env python
"""Soft audio-evidence probability nudge through UNCHANGED decoder.

Temporal correctness: event nudge starts at event CONFIRMATION END + causal
lag (not onset). The 80/160ms detection window is not available until the
event ends, so no future info is used during confirmation.

Predeclared parameters (physics/scaling prior, NOT eval-tuned):
- Whistle threshold: 0.30
- Min event persistence: 160 ms (16 hops)
- Nudge start: event_end + 5 frames (+0.2 s causal lag)
- Nudge window: +50 frames (+2.0 s) after lag
- Weight: w=0.15 max D-prob increase (before renorm)
- Context gate: nudge only where model already suspects D (P[:,2] > 0.3)
- Within-half clamp: derived from m.halves
- No hard live->D label. Decoder unchanged.
- Ext/masked probabilities untouched (no audio available).

Control: W=0 reproduces fold-ensemble baseline (P_R1.npz decoder output).
This is NOT the production fullfit CSV baseline (final_v5.csv).
"""
import sys, os, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from mtah.bench import Bench
from mtah.final import DECODER

# --- Predeclared (locked before scoring) ---
WHISTLE_THR = 0.30
MIN_EVENT_HOPS = 16       # 160 ms at 10 ms hop
STFT_LATENCY_FRAMES = 3   # 94ms (64ms window + 30ms to last hop) rounded up
CAUSAL_LAG_FRAMES = 5     # +0.2s after event confirmation + STFT latency
PERSIST_FRAMES = 50       # +2.0 s nudge window
D_CONTEXT_THR = 0.30      # nudge only where P[:,2] > this

def main():
    t0 = time.time()
    W = float(os.environ.get('W_NUDGE', '0.15'))
    print(f'[config] W_NUDGE={W}, thr={WHISTLE_THR}, min_hops={MIN_EVENT_HOPS}, '
          f'stft_latency={STFT_LATENCY_FRAMES}f, causal_lag={CAUSAL_LAG_FRAMES}f, '
          f'persist={PERSIST_FRAMES}f, D_ctx_thr={D_CONTEXT_THR}')

    # 1. Load Bench (for match halves and decoder)
    b = Bench(verbose=False)
    m = b.public.m
    halves = m.halves  # [(1, f_start, f_end), (2, f_start, f_end)]
    print(f'[match] halves: {halves}')

    # 2. Load cached probabilities
    z = np.load(os.path.join(ROOT, 'cache/P_R1.npz'), allow_pickle=True)
    P = {k: z[k].copy() for k in z.keys()}
    P_pub = P['public'].copy()  # (199319, 3)
    print(f'[load] public P shape: {P_pub.shape}')

    # 3. Load tonal whistle scores
    w = np.load(os.path.join(ROOT, 'analysis/audio_sources/cache/whistle_tonal.npz'))
    ws = w['whistle_score']
    wf = w['frame']

    # 4. Detect events, record END frame (confirmation point)
    above = ws >= WHISTLE_THR
    events_end = []  # (end_frame, duration_hops)
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= MIN_EVENT_HOPS:
                # event ends at frame wf[j-1]; nudge starts there + lag
                events_end.append((int(wf[j-1]), j - i))
            i = j
        else:
            i += 1
    print(f'[events] {len(events_end)} confirmed events at thr={WHISTLE_THR}')

    # 5. Soft nudge: start at event_end + lag, within same half
    n_nudged_frames = 0
    n_nudged_events = 0
    for f_end_ev, dur_hops in events_end:
        f_start = f_end_ev + STFT_LATENCY_FRAMES + CAUSAL_LAG_FRAMES
        f_end = f_end_ev + STFT_LATENCY_FRAMES + CAUSAL_LAG_FRAMES + PERSIST_FRAMES
        # Within-half clamp: find which half f_end_ev is in, clamp to that half's end
        half_end = None
        half_start = None
        for (h, hs, he) in halves:
            if hs <= f_end_ev <= he:
                half_start = hs
                half_end = he
                break
        if half_end is None:
            continue  # outside graded halves
        f_end = min(f_end, half_end)
        # Also don't start before half start
        f_start = max(f_start, half_start)
        if f_start > f_end:
            continue
        event_nudged = False
        for f in range(f_start, f_end + 1):
            # Context gate: only nudge where model already suspects D
            if P_pub[f, 2] < D_CONTEXT_THR:
                continue
            decay = 1.0 - (f - f_start) / max(1, PERSIST_FRAMES)
            d_inc = W * decay
            P_pub[f, 2] += d_inc
            s = P_pub[f].sum()
            P_pub[f] /= s
            n_nudged_frames += 1
            event_nudged = True
        if event_nudged:
            n_nudged_events += 1

    print(f'[nudge] events with context trigger: {n_nudged_events}/{len(events_end)}')
    print(f'[nudge] frames modified: {n_nudged_frames}')

    P['public'] = P_pub

    # 6. Run unchanged decoder
    res = b.score('soft_audio_nudge', P, dec_params=DECODER, use_rules=False,
                  verbose=True, t0=t0)

    print()
    print(f'=== W={W} results (fold-ensemble baseline) ===')
    print(f'public spellF1={res["public"][0]:.6f}  macroF1={res["public"][1]:.6f}')
    if res['val'] is not None:
        print(f'val_h2 spellF1={res["val"][0]:.6f}  macroF1={res["val"][1]:.6f}')
    print(f'bench={res["bench"]:.6f} (ext {res["ext"][0]:.4f} masked {res["masked"][0]:.4f})')

    summary = {
        'W_NUDGE': W,
        'whistle_thr': WHISTLE_THR,
        'min_event_hops': MIN_EVENT_HOPS,
        'lag_frames': STFT_LATENCY_FRAMES + CAUSAL_LAG_FRAMES,
        'persist_frames': PERSIST_FRAMES,
        'd_context_thr': D_CONTEXT_THR,
        'n_events': len(events_end),
        'n_nudged_events': n_nudged_events,
        'n_nudged_frames': n_nudged_frames,
        'baseline_type': 'fold-ensemble (P_R1.npz decoder output)',
        'note': 'NOT production fullfit CSV (final_v5.csv)',
        'public_spellF1': float(res['public'][0]),
        'public_macroF1': float(res['public'][1]),
        'val_h2_spellF1': None if res['val'] is None else float(res['val'][0]),
        'val_h2_macroF1': None if res['val'] is None else float(res['val'][1]),
        'bench': float(res['bench']),
        'ext_spellF1': float(res['ext'][0]),
        'masked_spellF1': float(res['masked'][0]),
    }
    tag = 'control' if W == 0 else f'w{W}'
    out_path = os.path.join(ROOT, f'analysis/audio_sources/cache/soft_nudge_{tag}_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[save] {out_path}')

if __name__ == '__main__':
    main()
