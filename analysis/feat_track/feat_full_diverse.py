"""Full 600-tree recipe for the information-diverse screened-only blocks.

ballfree_cluster and melee_uncertainty target the missingness/ball-missing regime
(ext screen +0.0059 / +0.0062), distinct from dwell/carrier. Same locked recipe.
Writes to analysis/feat_track/feat_full_results.jsonl.
"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
from analysis.feat_track.feat_candidates import compose
from analysis.feat_track.feat_screen import by_len, v2g
from analysis.yardsticks import Yard
from mtah import feats, final
from mtah.data import CACHE_DIR
OUT = os.path.join(ROOT, "analysis", "feat_track", "feat_full_results.jsonl")
CONFIGS = [
    ("c_ballfree_cluster", ["c_ballfree_cluster"]),
    ("c_melee_uncertainty", ["c_melee_uncertainty"]),
]

def log(rec):
    rec["t"] = time.strftime("%H:%M:%S")
    with open(OUT, "a") as fh: fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)

if __name__ == "__main__":
    y = Yard()
    for label, names in CONFIGS:
        featfn = compose("full_plus3", names)
        key = "full_" + featfn.__name__
        t = time.time()
        try:
            extra = {k: feats.mirrored(featfn).get(k, []) +
                        feats.augmented(featfn).get(k, []) for k in y.b.ext}
            P = y.b.probs(featfn, final.MODEL, extra=extra, key=key)
            np.savez_compressed(os.path.join(CACHE_DIR, "P_" + key + ".npz"), **P)
            r = y.report(label + " (full fit)", v2g, P=P)
            bl = by_len(y, P)
            rec = dict(what="full_fit", block=label, members=names, key=key,
                       secs=round(time.time() - t, 1), bench=float(r["bench"]),
                       ext=float(r["ext"]), masked=float(r["masked"]),
                       public=float(r["public"]), public_macro=float(r["pubM"]),
                       rec_lt2s=float(bl["rec_lt2s"]), rec_gt10s=float(bl["rec_gt10s"]),
                       spur_per_100=float(bl["spur_per_100"]))
            base = dict(bench=0.6611595515, ext=0.6162642273, masked=0.7060548758,
                        public=0.6967741935)
            for k in base: rec["d_" + k] = round(rec[k] - base[k], 5)
            log(rec)
        except Exception as e:  # noqa: BLE001
            log(dict(what="error", block=label, members=names, key=key,
                     secs=round(time.time() - t, 1), err=repr(e)))
