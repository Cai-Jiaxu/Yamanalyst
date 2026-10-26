#!/usr/bin/env python
"""Hard D-relabel fusion, corrected causal timing.

Event nudge starts at event CONFIRMATION END + STFT latency + causal lag.
STFT window 64ms (1024 @ 16k), hop 10ms, 4 hops per 25fps frame.
Frame f score available at f + 94ms (2.35 frames). Round up to 3 frames.

Within-half clamp derived from m.halves.
Unique snapshot names per config.
"""
import sys, os, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
import possession_metrics as pm

# --- Predeclared ---
WHISTLE_THR = float(os.environ.get('WHISTLE_THR', '0.30'))
MIN_EVENT_MS = float(os.environ.get('MIN_EVENT_MS', '160'))
TRAIL_S = 2.0
FPS = 25.0
STFT_LATENCY_FRAMES = 3   # 94ms rounded up
CAUSAL_LAG_FRAMES = 5     # +0.2s after confirmation
TRAIL_FRAMES = int(round(TRAIL_S * FPS))
MIN_HOPS = int(round(MIN_EVENT_MS / 10.0))
TAG = os.environ.get('TAG', 'run')

def main():
    t0 = time.time()
    out_dir = os.path.join(ROOT, 'analysis', 'audio_sources')

    # Load match halves
    from mtah.bench import Bench
    b = Bench(verbose=False)
    halves = b.public.m.halves
    print(f'[match] halves: {halves}')

    # Load baseline predictions
    base = pd.read_csv(os.path.join(ROOT, 'evaluation/results/final_v5.csv'))
    base_labels = base['label'].astype(str).str.strip().str.upper().values
    print(f'[load] baseline: {len(base_labels)} frames')

    # Load whistle scores
    w = np.load(os.path.join(out_dir, 'cache/whistle_tonal.npz'))
    ws = w['whistle_score']
    wf = w['frame']

    # Detect events, record END frame
    above = ws >= WHISTLE_THR
    events_end = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= MIN_HOPS:
                events_end.append(int(wf[j-1]))
            i = j
        else:
            i += 1
    print(f'[events] thr={WHISTLE_THR} min={MIN_EVENT_MS}ms -> {len(events_end)} events (end-timestamped)')

    # Relabel: start at event_end + STFT_LATENCY + CAUSAL_LAG
    cand = base_labels.copy()
    n_relabel = 0
    for f_end_ev in events_end:
        f0 = f_end_ev + STFT_LATENCY_FRAMES + CAUSAL_LAG_FRAMES
        f1 = f0 + TRAIL_FRAMES
        # Within-half clamp
        for (h, hs, he) in halves:
            if hs <= f_end_ev <= he:
                f1 = min(f1, he)
                break
        else:
            continue
        f0 = max(f0, hs)
        if f0 > f1:
            continue
        for f in range(f0, f1 + 1):
            if cand[f] == 'H':
                cand[f] = 'D'; n_relabel += 1
            elif cand[f] == 'A':
                cand[f] = 'D'; n_relabel += 1
    print(f'[fusion] relabeled {n_relabel} frames')

    # Write unique snapshot
    cand_path = os.path.join(ROOT, f'evaluation/results/research_hardD_{TAG}.csv')
    pd.DataFrame({'frame': np.arange(199319), 'label': cand}).to_csv(cand_path, index=False)
    print(f'[write] {cand_path}')

    # Score public
    pub_gt = pm.read_label_csv(os.path.join(ROOT, 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    pub_gt_codes = pm.label_codes(pub_gt['label'].astype(str).str.strip().str.upper().values)
    base_codes = pm.label_codes(base_labels)
    cand_codes = pm.label_codes(cand)
    pub_base = pm.evaluate(pub_gt_codes, base_codes)
    pub_cand = pm.evaluate(pub_gt_codes, cand_codes)

    # Score val_h2
    import glob
    lab_paths = sorted(p for p in glob.glob(os.path.join(ROOT, 'work/labels/*.csv'))
                       if not os.path.basename(p).startswith('ball_probe'))
    val_labels = None
    for path in lab_paths:
        x = pd.read_csv(path)['label'].astype(str).str.strip().str.upper().values
        val_labels = x.copy() if val_labels is None else np.where(x != 'X', x, val_labels)
    val_codes = pm.label_codes(val_labels)
    val_base = pm.evaluate(val_codes, base_codes)
    val_cand = pm.evaluate(val_codes, cand_codes)

    print(f'[public] baseline spellF1={pub_base["spell_f1_mean"]:.6f} macroF1={pub_base["macro_f1"]:.6f}')
    print(f'[public] candidate spellF1={pub_cand["spell_f1_mean"]:.6f} macroF1={pub_cand["macro_f1"]:.6f}')
    print(f'[val_h2] baseline spellF1={val_base["spell_f1_mean"]:.6f} macroF1={val_base["macro_f1"]:.6f}')
    print(f'[val_h2] candidate spellF1={val_cand["spell_f1_mean"]:.6f} macroF1={val_cand["macro_f1"]:.6f}')

    summary = {
        'tag': TAG, 'thr': WHISTLE_THR, 'min_ms': MIN_EVENT_MS,
        'stft_latency_f': STFT_LATENCY_FRAMES, 'causal_lag_f': CAUSAL_LAG_FRAMES,
        'trail_f': TRAIL_FRAMES, 'n_events': len(events_end), 'n_relabel': n_relabel,
        'baseline_public_spellF1': pub_base['spell_f1_mean'],
        'baseline_public_macroF1': pub_base['macro_f1'],
        'candidate_public_spellF1': pub_cand['spell_f1_mean'],
        'candidate_public_macroF1': pub_cand['macro_f1'],
        'baseline_val_spellF1': val_base['spell_f1_mean'],
        'baseline_val_macroF1': val_base['macro_f1'],
        'candidate_val_spellF1': val_cand['spell_f1_mean'],
        'candidate_val_macroF1': val_cand['macro_f1'],
    }
    with open(os.path.join(out_dir, f'cache/hardD_{TAG}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[done] {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
