"""Full-recipe refit for screen-selected label-free candidate blocks.

Mirrors the shipped final_v5 recipe exactly:
  - features: full_plus3 + candidate block(s) (analysis.feat_track.feat_candidates.compose)
  - model:    final.MODEL (HistGradientBoosting max_iter=600, seed 0)
  - train:    9 external matches + left/right mirrors + home/away swap + 3 noise draws
              (feats.mirrored / feats.augmented)
  - decoder:  P2 = final.DECODER with use_rules=False (locked v5 decoder)

Compared against the locked unrounded baseline (baseline_metrics.json):
  bench 0.66116 / ext 0.61626 / masked 0.70605 / public 0.69677.
Promote gates: bench +>=0.005, ext/masked drop <=0.005, public drop <=0.02.

Writes one JSONL line per completed full fit to analysis/feat_track/feat_full_results.jsonl.
Run detached: caffeinate -i .venv/bin/python analysis/feat_track/feat_full.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from analysis.feat_track.feat_candidates import compose  # noqa: E402
from analysis.feat_track.feat_screen import by_len, v2g  # noqa: E402
from analysis.yardsticks import Yard  # noqa: E402
from mtah import feats, final  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "feat_track", "feat_full_results.jsonl")

# Evidence-selected after the 300-tree screen (deltas vs fresh full_plus3 control):
#   margin_dwell      bench+0.031 ext+0.004 masked+0.057 public+0.025  (best; only + public)
#   carrier_tightness bench+0.018 ext+0.004 masked+0.033 public+0.007  (2nd; + public)
#   combo             test complementarity of the two public-positive blocks
# Dropped by evidence (public -0.0125, no unique lift): centroid_compression,
# melee_uncertainty, ballfree_cluster.
CONFIGS = [
    ("c_margin_dwell", ["c_margin_dwell"]),
    ("c_carrier_tightness", ["c_carrier_tightness"]),
    ("combo_margin_carrier", ["c_margin_dwell", "c_carrier_tightness"]),
]


def log(rec):
    rec["t"] = time.strftime("%H:%M:%S")
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    y = Yard()
    for label, names in CONFIGS:
        featfn = compose("full_plus3", names)
        key = "full_" + featfn.__name__
        t = time.time()
        try:
            extra = {k: feats.mirrored(featfn).get(k, []) +
                        feats.augmented(featfn).get(k, [])
                     for k in y.b.ext}
            P = y.b.probs(featfn, final.MODEL, extra=extra, key=key)
            np.savez_compressed(os.path.join(CACHE_DIR, "P_" + key + ".npz"), **P)
            r = y.report(label + " (full fit)", v2g, P=P)
            bl = by_len(y, P)
            rec = dict(what="full_fit", block=label, members=names, key=key,
                       secs=round(time.time() - t, 1),
                       bench=float(r["bench"]), ext=float(r["ext"]),
                       idsse=float(r["idsse"]), metrica=float(r["metrica"]),
                       masked=float(r["masked"]), public=float(r["public"]),
                       public_macro=float(r["pubM"]),
                       val=float(r["val"]) if r["val"] is not None else None,
                       val_macro=float(r["valM"]) if r.get("valM") is not None else None,
                       rec_lt2s=float(bl["rec_lt2s"]), rec_2_5s=float(bl["rec_2_5s"]),
                       rec_5_10s=float(bl["rec_5_10s"]), rec_gt10s=float(bl["rec_gt10s"]),
                       spur_per_100=float(bl["spur_per_100"]))
            # paired deltas vs locked unrounded baseline
            base = dict(bench=0.6611595515, ext=0.6162642273, masked=0.7060548758,
                        public=0.6967741935)
            for k in base:
                rec["d_" + k] = round(rec[k] - base[k], 5)
            log(rec)
        except Exception as e:  # noqa: BLE001
            log(dict(what="error", block=label, members=names, key=key,
                     secs=round(time.time() - t, 1), err=repr(e)))
