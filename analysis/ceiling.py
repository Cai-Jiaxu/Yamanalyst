"""How much does each kind of error cost on spellF1? Uses only the PUBLIC ground truth.

    uv run python analysis/ceiling.py

The public window has 17 possessions -- too few to read costs off reliably -- so each experiment
also runs on a synthetic full match: the public segments resampled (with replacement) into a
158k-frame, two-half label track with the same segment-length and transition statistics. The
synthetic track is only a ruler for the metric; it says nothing about the real match.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.score import load_gt, pm  # noqa: E402

FPS = 25
H, A, D, X = (pm.LABELS.index(c) for c in "HADX")


def rle(codes, start, stop):
    out = []
    w = codes[start:stop]
    b = np.flatnonzero(w[1:] != w[:-1]) + 1
    bounds = np.concatenate(([0], b, [len(w)]))
    for i in range(len(bounds) - 1):
        out.append((int(w[bounds[i]]), int(bounds[i + 1] - bounds[i])))
    return out


def synthetic_gt(public, rng, n_half=79400):
    """Two halves of resampled public segments. Team labels re-drawn so H/A alternate sensibly."""
    segs = rle(public, 11185, 18685)
    gt = np.full(n_half * 2 + 20000, X, dtype=np.int8)
    for h in range(2):
        start = h * (n_half + 20000)
        pos, prev_team = start, H
        while pos < start + n_half:
            code, length = segs[rng.integers(len(segs))]
            if code != D:
                # a possession follows a different team unless a stoppage sits between them
                code = A if prev_team == H else H
                if gt[pos - 1] == D and rng.random() < 0.5:
                    code = prev_team
                prev_team = code
            end = min(pos + length, start + n_half)
            gt[pos:end] = code
            pos = end
    return gt


def spell(gt, pred):
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]


def transitions_of(gt):
    return np.flatnonzero((gt[1:] != gt[:-1]) & (gt[1:] != X) & (gt[:-1] != X)) + 1


def drop_short(gt, seconds):
    """Spells shorter than `seconds` taken over by the label before them: undetectable spells."""
    out = gt.copy()
    for start, stop in pm.valid_runs(gt):
        pos = start
        for code, n in rle(gt, start, stop):
            if code in (H, A) and n < seconds * FPS and pos > start:
                out[pos:pos + n] = out[pos - 1]
            pos += n
    return out


def jitter(gt, sigma_s, rng, only=None):
    """Move every transition by N(0, sigma) frames, keeping order. only: 'team' or 'dead'."""
    out = gt.copy()
    for t in transitions_of(gt):
        a, b = gt[t - 1], gt[t]
        is_dead = (a == D) or (b == D)
        if only == "team" and is_dead or only == "dead" and not is_dead:
            continue
        d = int(round(rng.normal(0, sigma_s * FPS)))
        if d > 0:
            out[t:t + d] = np.where(gt[t:t + d] == b, a, out[t:t + d])
        elif d < 0:
            out[t + d:t] = np.where(gt[t + d:t] == a, b, out[t + d:t])
    return out


def blips(gt, n_per_min, length_s, rng):
    """Spurious opposite-team blips dropped uniformly inside possessions."""
    out = gt.copy()
    alive = np.flatnonzero(np.isin(gt, (H, A)))
    n = int(n_per_min * len(alive) / FPS / 60)
    for c in rng.choice(alive, n, replace=False):
        L = int(length_s * FPS)
        if gt[c] in (H, A):
            out[c:c + L] = np.where(gt[c:c + L] == H, A, np.where(gt[c:c + L] == A, H, gt[c:c + L]))
    return out


def no_dead(gt):
    """Never predict D: stoppages vanish and the possessions either side merge if same team."""
    out = gt.copy()
    out[out == D] = X
    return out


def main():
    public = load_gt()
    rng = np.random.default_rng(0)
    rulers = [("public5min", public)] + [("synth{}".format(i), synthetic_gt(public, rng))
                                         for i in range(3)]
    for name, g in rulers:
        print("{:<11} {} possessions".format(name, len(pm.possession_spells(g, pm.valid_runs(g)))))

    def row(label, fn, seeds=3):
        vals = []
        for _, g in rulers:
            vals.append(np.mean([spell(g, fn(g, np.random.default_rng(s))) for s in range(seeds)]))
        print("{:<46} public {:.3f}   synth {:.3f}".format(label, vals[0], np.mean(vals[1:])))

    print("\n-- undetectable short possessions (absorbed by the previous label)")
    for s in (0.5, 1, 2, 3, 5):
        row("miss every possession < {} s".format(s), lambda g, r, s=s: drop_short(g, s), 1)

    print("\n-- boundary timing noise, every transition ~ N(0, sigma)")
    for s in (0.12, 0.25, 0.5, 1.0, 2.0):
        row("sigma {} s, all transitions".format(s), lambda g, r, s=s: jitter(g, s, r))
    for s in (0.5, 1.0, 2.0):
        row("sigma {} s, only alive<->dead".format(s), lambda g, r, s=s: jitter(g, s, r, "dead"))
        row("sigma {} s, only H<->A".format(s), lambda g, r, s=s: jitter(g, s, r, "team"))

    print("\n-- spurious opposite-team blips inside possessions")
    for n, L in ((0.5, 0.4), (1, 0.4), (2, 0.4), (1, 1.0), (4, 0.4)):
        row("{} blips/min of {} s".format(n, L), lambda g, r, n=n, L=L: blips(g, n, L, r))

    print("\n-- dead ball")
    row("never predict D (team labels perfect)", lambda g, r: no_dead(g), 1)


if __name__ == "__main__":
    main()
