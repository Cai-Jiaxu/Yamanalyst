"""Rules back in the decoder? Scored on ext, masked, public AND the R3 2nd-half windows.

    uv run python analysis/fusion_val.py          # rules in the decoder, all labels
    uv run python analysis/fusion_val.py --ha     # team-only rule evidence, lower weights

final_v2 uses the rules only as model features (use_rules=False): that won on the bench, but on
the hand-labelled 2nd half rules_v2 alone scores 0.655 against 0.558. Decoder-only on cached
P_R1, small fixed grid. Prints per-window spellF1 so a gain must hold in every window.
"""
import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import decode, final  # noqa: E402
from mtah.bench import Bench  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402
from mtah.score import pm  # noqa: E402

warnings.filterwarnings("ignore")
LABELS = os.path.join(ROOT, "work", "labels")
GRID3 = [
    ("final_v2", False, {}, False),
    ("rules w 0.5, gate 0.6", True, {"w_rule": 0.5, "rule_gate": 0.6}, False),
    ("rules w 1, gate 0.6", True, {"w_rule": 1.0, "rule_gate": 0.6}, False),
    ("rules w 0.5, gate 0.8", True, {"w_rule": 0.5, "rule_gate": 0.8}, False),
    ("rules w 1, gate 0.8", True, {"w_rule": 1.0, "rule_gate": 0.8}, False),
]
GRID2 = [
    ("final_v2", False, {}, False),
    ("rules all, w 0.25", True, {"w_rule": 0.25}, False),
    ("rules H/A only, w 0.25", True, {"w_rule": 0.25}, True),
    ("rules H/A only, w 0.5", True, {"w_rule": 0.5}, True),
    ("rules H/A only, w 1", True, {"w_rule": 1.0}, True),
]
GRID = [
    ("final_v2 (rules as features only)", False, {}),
    ("+ rules in decoder, w_rule 0.5", True, {"w_rule": 0.5}),
    ("+ rules in decoder, w_rule 1", True, {"w_rule": 1.0}),
    ("+ rules in decoder, w_rule 2", True, {"w_rule": 2.0}),
    ("+ rules w1, w_shape 0.5", True, {"w_rule": 1.0, "w_shape": 0.5}),
    ("+ rules w1, costs 8/8", True, {"w_rule": 1.0, "cost_team": 8.0, "cost_dead": 8.0}),
]


def windows():
    out = []
    for p in sorted(glob.glob(os.path.join(LABELS, "val_*.csv"))):
        lab = pm.read_label_csv(p)["label"].astype(str).str.strip().str.upper().values
        out.append((os.path.basename(p)[7:-4], pm.label_codes(lab)))
    return out


def main():
    b = Bench(verbose=False)
    z = np.load(os.path.join(CACHE_DIR, "P_R1.npz"))
    P = {k: z[k] for k in z.files}
    W = windows()
    c = b.public
    print("{:<36} {:>6} {:>6} {:>6} {:>6} | {:>6} | {}".format(
        "decoder", "bench", "ext", "masked", "public", "val", " ".join(w for w, _ in W)))
    grid = GRID3 if "--gate" in sys.argv else GRID2 if "--ha" in sys.argv else \
        [g + (False,) for g in GRID]
    for name, use_rules, change, ha in grid:
        p = dict(final.DECODER, **change)
        saved = {}
        if ha:   # rule evidence for the team only: dead-ball votes removed
            for case in [b.public] + b.masked + list(b.ext.values()):
                saved[id(case)] = case.R
                case.R = np.where(case.R == "D", "X", case.R)
        r = b.score(name, P, p, use_rules=use_rules, verbose=False)
        R = c.R if use_rules else np.full(c.m.n, "X", dtype="<U1")
        codes = pm.label_codes(decode.decode(c.m, R, P["public"], c.f["seen"],
                                             dict(decode.PARAMS, **p), c.f))
        per = [pm.evaluate(g, codes)["spell_f1_mean"] for _, g in W]
        print("{:<36} {:6.3f} {:6.3f} {:6.3f} {:6.3f} | {:6.3f} | {}".format(
            name, r["bench"], r["ext"][0], r["masked"][0], r["public"][0],
            r["val"][0] if r["val"] is not None else np.nan,
            " ".join("{:6.3f}".format(x) for x in per)), flush=True)
        for case in [b.public] + b.masked + list(b.ext.values()):
            if id(case) in saved:
                case.R = saved[id(case)]


if __name__ == "__main__":
    main()
