"""Read-only reconstruction of the shipped final_v5 dev bench.

Loads the cached fold probabilities P_R1.npz (produced by the shipped pipeline,
no refit) and scores them with the EXACT final_v5 decoder params, mtah/final.py DECODER_V5.
It changes nothing: no new features, no retraining, no decoder tuning.

Output: baseline_snapshot/baseline_metrics.json (aggregate numbers only).

Run:
    .venv/bin/python baseline_snapshot/reconstruct_baseline.py
"""

import json
import os
import time

import numpy as np

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mtah.bench import Bench  # noqa: E402
from mtah.final import DECODER_V5 as DECODER, FEATFN, MODEL, SEED  # noqa: E402  (locked final_v5)

CACHE_P = os.path.join(ROOT, "cache", "P_R1.npz")
OUT = os.path.join(ROOT, "baseline_snapshot", "baseline_metrics.json")


def main():
    t0 = time.time()
    assert os.path.isfile(CACHE_P), "missing cached probabilities %s" % CACHE_P
    z = np.load(CACHE_P, allow_pickle=True)
    P = {k: z[k] for k in z.keys()}

    b = Bench(verbose=True)
    # use_rules=False matches shipped final_v5: algorithmic rule FEATURES are
    # already baked into P; the rule LABEL does not vote in the decoder.
    res = b.score("final_v5_snapshot", P, dec_params=DECODER, use_rules=False,
                  verbose=True, t0=t0)

    out = {
        "elapsed_s": round(time.time() - t0, 1),
        "featfn": FEATFN,
        "model": MODEL,
        "seed": SEED,
        "decoder": DECODER,
        "cache_P": os.path.relpath(CACHE_P, ROOT),
        "bench_spellF1": float(res["bench"]),
        "ext_spellF1": float(res["ext"][0]),
        "ext_macroF1": float(res["ext"][1]),
        "idsse_spellF1": float(res["idsse"][0]),
        "metrica_spellF1": float(res["metrica"][0]),
        "masked_spellF1": float(res["masked"][0]),
        "masked_macroF1": float(res["masked"][1]),
        "masked_sd_spellF1": float(res["masked_sd"]),
        "public_spellF1": float(res["public"][0]),
        "public_macroF1": float(res["public"][1]),
        "val_spellF1": None if res["val"] is None else float(res["val"][0]),
        "val_macroF1": None if res["val"] is None else float(res["val"][1]),
        "val_possessions": getattr(b, "val_poss", None),
        "taxonomy_ext_per100": {k: round(100 * v / res["taxonomy"]["n_gt"], 2)
                                for k, v in res["taxonomy"].items()
                                if k in ("split", "merge", "drift", "missed", "spurious")},
        "pred_to_true_ratio": round(res["taxonomy"]["n_pred"] / res["taxonomy"]["n_gt"], 3),
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("\n=== baseline_metrics.json ===")
    print(json.dumps(out, indent=2, sort_keys=True))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
