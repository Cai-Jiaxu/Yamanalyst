"""Independent label-disagreement audit on external training matches only.

Read-only. Uses cached out-of-fold probabilities (cache/P_R1.npz): each external match was
scored by a fold model NOT trained on it, so disagreements are honest out-of-sample. No target
manual labels, no relabeling. Prints aggregates only. Writes cache/ar_integrity_labelaudit.npz.

Usage: .venv/bin/python analysis/autonomous_research/experiments/integrity/label_audit.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from mtah import external  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CACHE = os.path.join(ROOT, "cache")
CODE = {"H": 0, "A": 1, "D": 2, "X": -1}


def boundary_distance(lab_codes):
    """For each graded frame, distance (frames) to the nearest H/A/D change. X ignored."""
    n = len(lab_codes)
    dist = np.full(n, 10 ** 9, dtype=int)
    changes = np.zeros(n, dtype=bool)
    valid = lab_codes >= 0  # not X
    for i in range(1, n):
        if valid[i] and valid[i - 1] and lab_codes[i] != lab_codes[i - 1]:
            changes[i] = True
    last = -10 ** 9
    for i in range(n):
        if changes[i]:
            last = i
        if valid[i]:
            dist[i] = min(dist[i], i - last)
    nxt = 10 ** 9
    for i in range(n - 1, -1, -1):
        if changes[i]:
            nxt = i
        if valid[i]:
            dist[i] = min(dist[i], nxt - i)
    return dist


def main():
    z = np.load(os.path.join(CACHE, "P_R1.npz"), allow_pickle=True)
    rows = []
    for name in external.available():
        em = external.ExtMatch(name)
        lab = np.array(em.labels)
        graded = em.graded_mask()
        P = z[name]
        y = np.array([CODE[c] for c in lab])
        pred = P.argmax(1)
        conf = P.max(1)
        disagree = (pred != y) & graded
        dist = boundary_distance(y)
        src = "idsse" if name.startswith("idsse") else "metrica"
        # possession-aware: runs of H/A; count possessions and how many contain a disagreement
        pos_runs = []
        in_run = False
        start = 0
        for i in range(len(y)):
            alive = graded[i] and y[i] in (0, 1)
            if alive and not in_run:
                start, in_run = i, True
            elif not alive and in_run:
                pos_runs.append((start, i)); in_run = False
        if in_run:
            pos_runs.append((start, len(y)))
        n_pos = len(pos_runs)
        pos_with_dis = sum(1 for a, b in pos_runs if disagree[a:b].any())
        rows.append(dict(
            name=name, src=src, n=int(graded.sum()),
            dis=float(disagree.sum()), dis_rate=float(disagree.mean(where=graded)),
            conf_dis=float(conf[disagree].mean()) if disagree.any() else float("nan"),
            conf_ok=float(conf[(~disagree) & graded].mean()),
            n_pos=n_pos, pos_with_dis=pos_with_dis,
            near2=float(disagree[(dist <= 2) & graded].sum()),
            near5=float(disagree[(dist <= 5) & graded].sum()),
            near15=float(disagree[(dist <= 15) & graded].sum()),
        ))

    print(f"{'match':<16} src     n_graded  dis%   conf_dis conf_ok  n_pos pos_w_dis  <=2f <=5f <=15f")
    for r in rows:
        print(f"{r['name']:<16} {r['src']:<7} {r['n']:>8} {100*r['dis_rate']:5.2f} "
              f"{r['conf_dis']:6.3f}  {r['conf_ok']:6.3f}  {r['n_pos']:>5} {r['pos_with_dis']:>6}   "
              f"{r['near2']:5.0f} {r['near5']:5.0f} {r['near15']:5.0f}")

    # aggregate by source
    for src in ("idsse", "metrica"):
        rs = [r for r in rows if r["src"] == src]
        n = sum(r["n"] for r in rs); d = sum(r["dis"] for r in rs)
        p = sum(r["n_pos"] for r in rs); pd_ = sum(r["pos_with_dis"] for r in rs)
        near2 = sum(r["near2"] for r in rs); near5 = sum(r["near5"] for r in rs)
        near15 = sum(r["near15"] for r in rs)
        print(f"\n{src}: {len(rs)} matches, {n} graded frames, disagreement {100*d/n:.2f}%, "
              f"possessions with any disagreement {pd_}/{p} ({100*pd_/p:.1f}%), "
              f"disagreements within <=2f: {near2:.0f}, <=5f: {near5:.0f}, <=15f: {near15:.0f}")

    np.savez_compressed(os.path.join(CACHE, "ar_integrity_labelaudit.npz"),
                        rows=np.array([{k: r[k] for k in r} for r in rows], dtype=object))
    print("\nwrote cache/ar_integrity_labelaudit.npz")
    print("NOTE: 9 matches = 7 idsse + 2 metrica. These are NOT 9 independent clusters for "
          "significance; they are two datasets. Treat per-source rates as descriptive, not "
          "inferential.")


if __name__ == "__main__":
    main()
