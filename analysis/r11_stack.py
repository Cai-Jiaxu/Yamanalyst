"""ROADMAP R11: a per-frame stacker of model probabilities and rule evidence, plus the
stoppage-blip gate, scored against final_v3's decoder.

    uv run python analysis/r11_stack.py

Stacker: multinomial logistic regression (a handful of inputs, L2) fitted in-domain on the
labelled frames of our match -- the public window and the four R3 2nd-half windows -- with
leave-one-window-out CV: each window is scored by a stacker that never saw it. Its output
replaces P in the final_v2 decoder (no rule votes: the stacker already weighs them per frame).
Ext uses a stacker fitted on all 5 windows; masked (public-window GT) one fitted on the 4 val
windows only, so neither is scored by a stacker that saw its labels.

Stoppage gate: a live spell <= 100 frames with dead ball on both sides while the players walk
(mean team pace < 1.3 m/s) becomes dead ball -- the ATD tracking a returned ball in a stoppage.

Decision rule (ROADMAP R11): beat final_v3 on the held-out windows on average, public not more
than 0.02 lower. Aggregates only.
"""
import glob
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import decode, final  # noqa: E402
from mtah.bench import Bench  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402
from mtah.score import pm  # noqa: E402

warnings.filterwarnings("ignore")
LABELS = os.path.join(ROOT, "work", "labels")
NOR = None   # filled with the "no rules" label array per match length


def windows(n):
    """[(name, code array over the match with X elsewhere)] for the 4 val windows."""
    out = []
    for p in sorted(glob.glob(os.path.join(LABELS, "val_*.csv"))):
        lab = pm.read_label_csv(p)["label"].astype(str).str.strip().str.upper().values
        assert len(lab) == n
        out.append((os.path.basename(p)[7:-4], pm.label_codes(lab)))
    return out


def stack_X(P, f, R, seen_r=12):
    """Per-frame stacker inputs: log P (3), rule vote x ball-near (3), near, on-ball margin,
    ball seen, ball speed (clipped)."""
    seen = f["seen"].astype(bool)
    near = np.convolve(seen.astype(float), np.ones(2 * seen_r + 1), "same") > 0
    lp = np.log(np.clip(P, 1e-4, 1.0))
    rv = np.stack([(R == s) & near for s in "HAD"], 1).astype(float)
    margin = np.where(seen, np.clip(np.nan_to_num(f["da"] - f["dh"]), -10, 10) / 10, 0.0)
    bs = np.where(seen, np.clip(np.nan_to_num(f["bspeed"]), 0, 20) / 20, 0.0)
    return np.column_stack([lp, rv, near, margin, seen, bs])


def fit(Xs, ys):
    X, y = np.vstack(Xs), np.concatenate(ys)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X, y)
    return clf


def proba(clf, X):
    """(n, 3) in H, A, D order."""
    Q = clf.predict_proba(X)
    order = [list(clf.classes_).index(pm.LABELS.index(s)) for s in "HAD"]
    return Q[:, order]


def stop_gate(labels, f, max_len=100, slow=1.3):
    pace = uniform_filter1d(np.nan_to_num((f["sh"] + f["sa"]) / 2), 25)
    out = labels.copy()
    b = np.concatenate(([0], np.flatnonzero(out[1:] != out[:-1]) + 1, [len(out)]))
    for i in range(1, len(b) - 2):
        a, z = b[i], b[i + 1]
        if out[a] in "HA" and out[a - 1] == "D" and out[z] == "D" and z - a <= max_len \
                and pace[a:z].mean() < slow:
            out[a:z] = "D"
    return out


def run(c, P, p, rules_in_decoder, gate):
    R = c.R if rules_in_decoder else np.full(c.m.n, "X", dtype="<U1")
    lab = decode.decode(c.m, R, P, c.f["seen"], dict(decode.PARAMS, **p), c.f)
    return stop_gate(lab, c.f) if gate else lab


def f1(gt, lab):
    return pm.evaluate(gt, pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"]


def main():
    b = Bench(verbose=False)
    z = np.load(os.path.join(CACHE_DIR, "P_R1.npz"))
    P = {k: z[k] for k in z.files}
    c = b.public
    W = [("public", b.gt)] + windows(c.m.n)
    Xpub = stack_X(P["public"], c.f, c.R)
    lab_idx = {w: np.flatnonzero(g != pm.LABELS.index("X")) for w, g in W}

    def fit_on(names):
        return fit([Xpub[lab_idx[w]] for w in names], [g[lab_idx[w]] for w, g in W if w in names])

    v2 = dict(final.DECODER, use_rules=False)
    v3 = dict(final.DECODER)
    names = [w for w, _ in W]
    loo = {w: proba(fit_on([x for x in names if x != w]), Xpub) for w in names}
    clf_all = fit_on(names)
    clf_val = fit_on(names[1:])
    print("stacker coefficients (all 5 windows), rows H/A/D; cols logPH logPA logPD rH rA rD near "
          "margin seen bspeed")
    for row in proba_coef(clf_all):
        print("  " + " ".join("{:+6.2f}".format(x) for x in row))

    configs = [
        ("final_v2 decoder", "v2", False),
        ("final_v3 (submitted)", "v3", False),
        ("final_v3 + stop gate", "v3", True),
        ("final_v2 + stop gate", "v2", True),
        ("R11 stacker, w_shape 0.75", "st0.75", False),
        ("R11 stacker, w_shape 1.0", "st1.0", False),
        ("R11 stacker w 0.75 + gate", "st0.75", True),
    ]
    print("{:<28} {:>6} {:>6} {:>6} | {:>6} {} | {:>6}".format(
        "config", "bench", "ext", "masked", "public", " ".join("{:>6}".format(w[:6]) for w in
                                                              names[1:]), "valavg"))
    for name, kind, gate in configs:
        if kind.startswith("st"):
            ws = float(kind[2:])
            p = dict(v2, w_shape=ws)
            per = [f1(g, run(c, loo[w], p, False, gate)) for w, g in W]
            ext = np.mean([f1(e.gt, run(e, proba(clf_all, stack_X(P[k], e.f, e.R)), p, False,
                                        gate)) for k, e in b.ext.items()])
            msk = np.mean([f1(mc.gt, run(mc, proba(clf_val, stack_X(P[mc.name], mc.f, mc.R)), p,
                                         False, gate)) for mc in b.masked])
        else:
            p = v2 if kind == "v2" else v3
            ru = kind == "v3"
            lab = run(c, P["public"], p, ru, gate)
            per = [f1(g, lab) for _, g in W]
            ext = np.mean([f1(e.gt, run(e, P[k], p, ru, gate)) for k, e in b.ext.items()])
            msk = np.mean([f1(mc.gt, run(mc, P[mc.name], p, ru, gate)) for mc in b.masked])
        print("{:<28} {:6.3f} {:6.3f} {:6.3f} | {:6.3f} {} | {:6.3f}".format(
            name, (ext + msk) / 2, ext, msk, per[0], " ".join("{:6.3f}".format(x) for x in per[1:]),
            np.mean(per[1:])), flush=True)


def proba_coef(clf):
    order = [list(clf.classes_).index(pm.LABELS.index(s)) for s in "HAD"]
    return clf.coef_[order]


if __name__ == "__main__":
    main()
