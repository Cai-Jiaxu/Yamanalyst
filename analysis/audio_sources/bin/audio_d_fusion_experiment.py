#!/usr/bin/env python
"""Controlled audio-evidence fusion experiment (post-hoc, fixed decoder).

Predeclared parameters (NOT tuned on public17):
- Tonal whistle threshold: 0.30 (from cache/whistle_tonal.npz)
- Min event persistence: 160 ms (4 hops at 10 ms hop)
- Trailing D window: +2.0 s = 50 frames at 25 fps after whistle onset
- Relabel: within [f0, f0+50], H->D and A->D only. Leave X, D unchanged.
- Half reset: do not cross halftime (frames 90652..110807).
- Causal trailing only. No future context.

Hypothesis under test (falsifiable):
  Whistle events mark dead-ball instants. Flipping trailing 2 s of live
  possession to D should improve spellF1. If whistles mark restarts, this
  will HURT spellF1.

This is a supporting-change branch, NOT a feature-only training gain.
External training matches have no audio, so no model refit is possible.
"""
import sys, os, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
import possession_metrics as pm

# --- Predeclared parameters (locked before scoring) ---
# Falsification pair A (strict): thr=0.30, min persist 160 ms
# Falsification pair B (permissive): thr=0.25, min persist 80 ms
# Both predeclared before seeing any score.
WHISTLE_THR = float(os.environ.get('WHISTLE_THR', '0.30'))
MIN_EVENT_MS = float(os.environ.get('MIN_EVENT_MS', '160'))
TRAIL_S = 2.0           # trailing D window after whistle onset
FPS = 25.0
HALF2_START = 110808    # 2H graded start
HALF1_END = 90651       # 1H graded end (halftime 90652..110807)
TRAIL_FRAMES = int(round(TRAIL_S * FPS))
MIN_HOPS = int(round(MIN_EVENT_MS / 10.0))

def main():
    t0 = time.time()
    out_dir = os.path.join(ROOT, 'analysis', 'audio_sources')
    log = []

    # 1. Load baseline predictions
    base = pd.read_csv(os.path.join(ROOT, 'evaluation/results/final_v5.csv'))
    assert len(base) == 199319, len(base)
    base_labels = base['label'].astype(str).str.strip().str.upper().values
    log.append(f'[load] baseline labels: {len(base_labels)}')
    counts_base = pd.Series(base_labels).value_counts().to_dict()
    log.append(f'[load] baseline label counts: {counts_base}')

    # 2. Load cached tonal whistle scores
    w = np.load(os.path.join(out_dir, 'cache/whistle_tonal.npz'))
    whistle_score = w['whistle_score']  # (199315,)
    frame_idx = w['frame']               # (199315,) int32
    log.append(f'[load] whistle_score shape: {whistle_score.shape}')

    # 3. Detect events at threshold (same logic as whistle_diag.py)
    above = whistle_score >= WHISTLE_THR
    # Group contiguous above-threshold frames into events
    events = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            event_frames = frame_idx[i:j]
            event_dur_hops = j - i
            if event_dur_hops >= MIN_HOPS:
                events.append(int(event_frames[0]))
            i = j
        else:
            i += 1
    log.append(f'[events] thr={WHISTLE_THR} min_persist={MIN_EVENT_MS}ms -> {len(events)} events')

    # 4. Apply trailing D relabel
    cand_labels = base_labels.copy()
    relabeled_count = 0
    relabeled_h = 0
    relabeled_a = 0
    for f0 in events:
        f_lo = f0
        f_hi = min(f0 + TRAIL_FRAMES, 199318)
        # Half reset: do not cross halftime
        if f_lo <= HALF1_END and f_hi > HALF1_END:
            f_hi = HALF1_END
        # Relabel H->D, A->D. Leave X, D.
        for f in range(f_lo, f_hi + 1):
            lab = cand_labels[f]
            if lab == 'H':
                cand_labels[f] = 'D'
                relabeled_h += 1
                relabeled_count += 1
            elif lab == 'A':
                cand_labels[f] = 'D'
                relabeled_a += 1
                relabeled_count += 1
    log.append(f'[fusion] relabeled {relabeled_count} frames '
                f'(H->D: {relabeled_h}, A->D: {relabeled_a})')
    counts_cand = pd.Series(cand_labels).value_counts().to_dict()
    log.append(f'[fusion] candidate label counts: {counts_cand}')

    # 5. Write candidate CSV
    cand_path = os.path.join(ROOT, 'evaluation/results/research_audio_d_relabel.csv')
    pd.DataFrame({'frame': np.arange(199319), 'label': cand_labels}).to_csv(cand_path, index=False)
    log.append(f'[write] {cand_path}')

    # 6. Score on public GT
    pub_gt = pm.read_label_csv(os.path.join(ROOT, 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    pub_gt_codes = pm.label_codes(pub_gt['label'].astype(str).str.strip().str.upper().values)
    base_codes = pm.label_codes(base_labels)
    cand_codes = pm.label_codes(cand_labels)
    pub_res_base = pm.evaluate(pub_gt_codes, base_codes)
    pub_res_cand = pm.evaluate(pub_gt_codes, cand_codes)
    log.append(f'[public] baseline spellF1={pub_res_base["spell_f1_mean"]:.6f} '
                f'macroF1={pub_res_base["macro_f1"]:.6f}')
    log.append(f'[public] candidate spellF1={pub_res_cand["spell_f1_mean"]:.6f} '
                f'macroF1={pub_res_cand["macro_f1"]:.6f}')

    # 7. Score on val_h2 (work/labels/*.csv)
    import glob
    lab_paths = sorted(p for p in glob.glob(os.path.join(ROOT, 'work/labels/*.csv'))
                       if not os.path.basename(p).startswith('ball_probe'))
    val_labels = None
    for path in lab_paths:
        x = pd.read_csv(path)['label'].astype(str).str.strip().str.upper().values
        val_labels = x.copy() if val_labels is None else np.where(x != 'X', x, val_labels)
    if val_labels is not None:
        val_codes = pm.label_codes(val_labels)
        val_res_base = pm.evaluate(val_codes, base_codes)
        val_res_cand = pm.evaluate(val_codes, cand_codes)
        log.append(f'[val_h2] baseline spellF1={val_res_base["spell_f1_mean"]:.6f} '
                    f'macroF1={val_res_base["macro_f1"]:.6f}')
        log.append(f'[val_h2] candidate spellF1={val_res_cand["spell_f1_mean"]:.6f} '
                    f'macroF1={val_res_cand["macro_f1"]:.6f}')
    else:
        log.append('[val_h2] no labels found')

    # 8. Print deltas
    d_public_spell = pub_res_cand['spell_f1_mean'] - pub_res_base['spell_f1_mean']
    d_public_macro = pub_res_cand['macro_f1'] - pub_res_base['macro_f1']
    log.append(f'[delta] public spellF1 {d_public_spell:+.6f}  macroF1 {d_public_macro:+.6f}')
    if val_codes is not None:
        d_val_spell = val_res_cand['spell_f1_mean'] - val_res_base['spell_f1_mean']
        d_val_macro = val_res_cand['macro_f1'] - val_res_base['macro_f1']
        log.append(f'[delta] val_h2 spellF1 {d_val_spell:+.6f}  macroF1 {d_val_macro:+.6f}')

    log.append(f'[done] {time.time()-t0:.1f} s')
    for line in log:
        print(line)

    # Save summary
    summary = {
        'params': {
            'whistle_thr': WHISTLE_THR,
            'min_event_ms': MIN_EVENT_MS,
            'trail_s': TRAIL_S,
            'trail_frames': TRAIL_FRAMES,
            'relabel_rule': 'H->D, A->D in [f0, f0+trail]; X,D unchanged',
            'half_reset': 'do not cross halftime 90652..110807',
        },
        'n_events': len(events),
        'relabeled_frames': relabeled_count,
        'relabeled_h_to_d': relabeled_h,
        'relabeled_a_to_d': relabeled_a,
        'baseline_public_spellF1': pub_res_base['spell_f1_mean'],
        'baseline_public_macroF1': pub_res_base['macro_f1'],
        'candidate_public_spellF1': pub_res_cand['spell_f1_mean'],
        'candidate_public_macroF1': pub_res_cand['macro_f1'],
        'delta_public_spellF1': d_public_spell,
        'delta_public_macroF1': d_public_macro,
    }
    if val_codes is not None:
        summary['baseline_val_h2_spellF1'] = val_res_base['spell_f1_mean']
        summary['baseline_val_h2_macroF1'] = val_res_base['macro_f1']
        summary['candidate_val_h2_spellF1'] = val_res_cand['spell_f1_mean']
        summary['candidate_val_h2_macroF1'] = val_res_cand['macro_f1']
        summary['delta_val_h2_spellF1'] = d_val_spell
        summary['delta_val_h2_macroF1'] = d_val_macro
    with open(os.path.join(out_dir, 'cache/audio_d_fusion_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
