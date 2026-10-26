#!/usr/bin/env python
"""PANNs Cnn14 pilot on tactical audio around 1H kickoff.

Label-free, local-only. Generic AudioSet weights. No media upload.
CPU inference. Compares per-chunk event scores vs tonal detector events.
"""
import sys, os, time, subprocess, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pylibs'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(ROOT)), '.venv/lib/python3.11/site-packages'))

FFMPEG = os.path.join(ROOT, 'pylibs/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1')
SRC = os.path.join(os.path.dirname(os.path.dirname(ROOT)), 'data/tactical/BAR-ATH_tactical.mp4')

# AudioSet class indices of interest
CLASSES = {
    8: 'Shout',
    10: 'Whoop',
    11: 'Yell',
    66: 'Cheering',
    67: 'Applause',
    69: 'Crowd',
    70: 'Hubbub',
    318: 'Air_horn',
    398: 'Buzzer',
    402: 'Whistle',
    403: 'Steam_whistle',
}

def decode_segment(t_start, t_end, sr=32000):
    cmd = [FFMPEG, '-v', 'error', '-ss', str(t_start), '-to', str(t_end),
           '-i', SRC, '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)

def main():
    t0 = time.time()
    # Window around 1H kickoff: frame 11185 / 25 = 447.4 s.
    # Take 30 s before to 30 s after = 60 s total.
    t_kick = 11185 / 25.0
    t_lo, t_hi = t_kick - 30.0, t_kick + 30.0
    print(f'[decode] {t_lo:.2f}..{t_hi:.2f} s around 1H kickoff at {t_kick:.2f} s')

    t_decode0 = time.time()
    audio = decode_segment(t_lo, t_hi, sr=32000)
    t_decode = time.time() - t_decode0
    print(f'[decode] {len(audio)/32000:.2f} s audio, {t_decode:.1f} s ffmpeg')

    # Import torch + panns
    import torch
    from panns_inference import AudioTagging, labels
    device = 'cpu'
    print(f'[model] loading AudioTagging Cnn14 on {device}')
    t_load0 = time.time()
    at = AudioTagging(checkpoint_path='/Users/bytedance/panns_data/Cnn14_mAP=0.431.pth',
                      device=device)
    t_load = time.time() - t_load0
    print(f'[model] loaded in {t_load:.1f} s')

    # Chunk into 1.0 s windows
    chunk_s = 1.0
    n_chunk = int(chunk_s * 32000)
    n_chunks = len(audio) // n_chunk
    scores = np.zeros((n_chunks, len(CLASSES)), dtype=np.float32)
    t_inf0 = time.time()
    for i in range(n_chunks):
        seg = audio[i*n_chunk:(i+1)*n_chunk][None, :]
        (clipwise, _) = at.inference(seg)
        for j, idx in enumerate(CLASSES):
            scores[i, j] = clipwise[0, idx]
    t_inf = time.time() - t_inf0
    print(f'[infer] {n_chunks} chunks x 1.0 s in {t_inf:.1f} s '
          f'({t_inf/n_chunks*1000:.0f} ms/chunk CPU)')

    # Print top-scoring chunks per class
    print()
    print('=== Per-class peak chunks (relative to kickoff) ===')
    for j, idx in enumerate(CLASSES):
        col = scores[:, j]
        top = np.argsort(col)[-3:][::-1]
        for k in top:
            t_rel = (k + 0.5) - 30.0  # relative to kickoff
            print(f'  {CLASSES[idx]:14s} idx={idx:4d}  chunk={k:3d} t_rel={t_rel:+6.2f} s  p={col[k]:.3f}')
    print()

    # Whistle peak specifically
    whistle_col = scores[:, list(CLASSES).index(402)]
    print('=== Whistle (idx 402) per-chunk scores (relative to kickoff) ===')
    for i in range(n_chunks):
        t_rel = (i + 0.5) - 30.0
        bar = '#' * int(whistle_col[i] * 40)
        print(f'  chunk={i:3d} t_rel={t_rel:+6.2f} s  p={whistle_col[i]:.3f} {bar}')

    # Save cache
    out = os.path.join(ROOT, 'cache/panns_kickoff_window.npz')
    np.savez(out,
             t_lo=t_lo, t_kick=t_kick, t_hi=t_hi,
             chunk_s=chunk_s,
             class_indices=np.array(list(CLASSES.keys())),
             class_names=np.array(list(CLASSES.values())),
             scores=scores,
             whistle_score=whistle_col)
    print(f'\n[cache] {out}')
    print(f'[total] {time.time()-t0:.1f} s')

    summary = {
        'model': 'Cnn14 (AudioSet, mAP=0.431)',
        'weights_path': '/Users/bytedance/panns_data/Cnn14_mAP=0.431.pth',
        'weights_bytes': os.path.getsize('/Users/bytedance/panns_data/Cnn14_mAP=0.431.pth'),
        'sample_rate': 32000,
        'device': 'cpu',
        'decode_s': round(t_decode, 1),
        'load_s': round(t_load, 1),
        'infer_s': round(t_inf, 1),
        'ms_per_chunk': round(t_inf / n_chunks * 1000, 1),
        'n_chunks': n_chunks,
        'window': [t_lo, t_hi],
        'whistle_peak_chunk': int(np.argmax(whistle_col)),
        'whistle_peak_prob': float(whistle_col.max()),
        'whistle_peak_t_rel': float(np.argmax(whistle_col) + 0.5 - 30.0),
    }
    with open(os.path.join(ROOT, 'cache/panns_kickoff_window_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
