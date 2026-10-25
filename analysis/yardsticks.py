"""All yardsticks for a decoder on cached fold probabilities (cache/P_R1.npz), aggregates only.

    from analysis.yardsticks import Yard
    y = Yard()                       # Bench cases + val windows + cached P
    y.report("name", lambda case, P: labels)

Columns: bench = (ext + masked) / 2 on the official spellF1; public; val = official spellF1 on
the eye-labelled 2nd-half windows; valLo = spellF1 at IoU 0.1-0.3 and valMid = team error more
than 3 s from any labelled change. analysis/boundary_offsets.py shows the eye labels put H<->A
changes where the rules do (IQR 0..+2 frames) while the real GT puts them a median 1.6 s later,
so the official val spellF1 rewards rule timing; valLo and valMid do not depend on exact timing.
"""
import glob
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.bench import Bench  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402
from mtah.score import pm  # noqa: E402

warnings.filterwarnings("ignore")


def val_windows(n):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "work", "labels", "val_*.csv"))):
        lab = pm.read_label_csv(p)["label"].astype(str).str.strip().str.upper().values
        assert len(lab) == n
        out.append(pm.label_codes(lab))
    return out


def mid_team_err(gt, pred, far=75):
    g = np.array(pm.LABELS)[gt]
    q = np.array(pm.LABELS)[pred]
    ch = np.flatnonzero((g[1:] != g[:-1])) + 1
    idx = np.arange(len(g))
    j = np.clip(np.searchsorted(ch, idx), 1, len(ch) - 1)
    d = np.minimum(np.abs(idx - ch[j - 1]), np.abs(idx - ch[j]))
    s = np.isin(g, ["H", "A"]) & (d > far) & np.isin(q, ["H", "A"])
    return np.mean(q[s] != g[s])


class Yard:
    def __init__(self, pfile="P_R1.npz"):
        t = time.time()
        self.b = Bench(verbose=False)
        self.P = dict(np.load(os.path.join(CACHE_DIR, pfile)))
        self.vals = val_windows(self.b.m.n)
        print("yardsticks ready in {:.0f} s".format(time.time() - t), flush=True)

    def cases(self):
        return list(self.b.ext.values()) + [self.b.public] + self.b.masked

    def report(self, name, fn, P=None, quiet=False):
        """fn(case, P) -> label array for that case."""
        P = P or self.P
        ext = {"idsse": [], "metrica": []}
        for k, c in self.b.ext.items():
            r = pm.evaluate(c.gt, pm.label_codes(fn(c, P[k])), bootstrap_n=0)
            ext[k.split("_")[0]].append(r["spell_f1_mean"])
        lab_pub = pm.label_codes(fn(self.b.public, P["public"]))
        pub = pm.evaluate(self.b.gt, lab_pub, bootstrap_n=0)
        masked = [pm.evaluate(c.gt, pm.label_codes(fn(c, P[c.name])), bootstrap_n=0)["spell_f1_mean"]
                  for c in self.b.masked]
        v_off, v_lo, v_mid, v_mf = [], [], [], []
        for g in self.vals:
            runs = pm.valid_runs(g)
            v_off.append(pm.evaluate(g, lab_pub, bootstrap_n=0)["spell_f1_mean"])
            v_lo.append(pm.spell_scores(g, lab_pub, runs, ious=(0.1, 0.2, 0.3))["mean_f1"])
            v_mid.append(mid_team_err(g, lab_pub))
            v_mf.append(pm.evaluate(g, lab_pub, bootstrap_n=0)["macro_f1"])
        e = np.mean(ext["idsse"] + ext["metrica"])
        out = dict(bench=(e + np.mean(masked)) / 2, ext=e, idsse=np.mean(ext["idsse"]),
                   metrica=np.mean(ext["metrica"]), masked=np.mean(masked),
                   public=pub["spell_f1_mean"], pubM=pub["macro_f1"], val=np.mean(v_off),
                   valLo=np.mean(v_lo), valMid=np.mean(v_mid), valM=np.mean(v_mf),
                   npub=pub["n_pred_spells"])
        if not quiet:
            print("{:<30} bench {bench:.3f} | ext {ext:.3f} (idsse {idsse:.3f} metrica {metrica:.3f})"
                  " | masked {masked:.3f} | public {public:.3f}/{pubM:.3f} n{npub} | val {val:.3f}"
                  " valLo {valLo:.3f} valMid {valMid:.3f} valM {valM:.3f}".format(name, **out),
                  flush=True)
        return out
