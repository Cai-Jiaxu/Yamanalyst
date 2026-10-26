"""Fast screen (ROADMAP rule 7) for modular label-free candidate feature blocks.

Tier: 9 base external matches, 300 trees, no mirror/noise copies. v2g decoder
(final_v2 + stop_gate, use_rules=False). Aggregates only. Writes one JSONL line per
completed result so partial progress survives interruption.

    .venv/bin/python analysis/feat_track/feat_screen.py

Cache keys: screen_cmp__full_plus3__<block>. The fresh full_plus3 control screen is reused from
the existing cache when present; otherwise it is refit (one fold chain) and logged as the control.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from analysis.feat_track.feat_candidates import CANDIDATES, compose  # noqa: E402
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, feats  # noqa: E402
from mtah.bench import SCREEN  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "feat_track", "feat_screen_results.jsonl")


def v2g(c, P):
    from analysis.miss_by_len import P2
    return decode.decode(c.m, np.full(c.m.n, "X", dtype="<U1"), P, c.f["seen"], P2, c.f)


def log(rec):
    rec["t"] = time.strftime("%H:%M:%S")
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def by_len(y, P):
    from analysis.miss_by_len import rec
    from mtah.bench import taxonomy
    from mtah.score import pm
    L, H, sp, ngt = [], [], 0, 0
    for k, c in y.b.ext.items():
        codes = pm.label_codes(v2g(c, P[k]))
        l, h, _ = rec(c.gt, codes)
        L.append(l), H.append(h)
        t = taxonomy(c.gt, codes)
        sp += t["spurious"]
        ngt += t["n_gt"]
    L, H = np.concatenate(L), np.concatenate(H)
    bins = [0, 50, 125, 250, 10 ** 9]
    r = [H[(L >= a) & (L < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
    return dict(rec_lt2s=float(r[0]), rec_2_5s=float(r[1]), rec_5_10s=float(r[2]),
                rec_gt10s=float(r[3]), spur_per_100=float(100 * sp / ngt))


if __name__ == "__main__":
    y = Yard()
    # fresh screen control = full_plus3 at screen tier
    base = feats.full_plus3
    Pb = y.b.probs(base, dict(SCREEN), key="screen_full_plus3")
    rb = y.report("screen full_plus3 (control)", v2g, P=Pb)
    rec0 = dict(what="control", block="full_plus3", **{k: rb[k] for k in
                ("bench", "ext", "masked", "public", "val", "valM")})
    rec0.update(by_len(y, Pb))
    log(rec0)

    for name in CANDIDATES:
        try:
            featfn = compose("full_plus3", [name])
            t = time.time()
            P = y.b.probs(featfn, dict(SCREEN), key="screen_" + featfn.__name__)
            r = y.report("screen +" + name, v2g, P=P)
            rec = dict(what="candidate", block=name, secs=round(time.time() - t, 1),
                       **{k: r[k] for k in ("bench", "ext", "masked", "public", "val", "valM")})
            for k in ("bench", "ext", "masked", "public"):
                rec["d_" + k] = round(r[k] - rb[k], 4)
            rec.update(by_len(y, P))
            log(rec)
        except Exception as e:  # noqa: BLE001
            log(dict(what="error", block=name, err=repr(e)))
