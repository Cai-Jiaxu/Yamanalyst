"""Phys228 typed transition head integration.

Loads cache/ar_phys228_fold{0,1,2}.joblib (228 features: full_plus3 226 + 2 phys features),
predicts on public/16-masked/ext cases, applies the same fixed λ=2 cost delta as the 226 head.

Compare: original v5 (P_R1) → 226 head λ=2 → phys228 head λ=2, to isolate feature effect.
No new models trained. No producer artifact edits.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, feats, final, kickoff  # noqa: E402
from mtah.data import CACHE_DIR  # noqa: E402

STATES = np.array(["H", "A", "D"])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

FOLD_TRAINS = {
    0: {"idsse_J03WOH", "idsse_J03WQQ", "idsse_J03WOY", "idsse_J03WR9", "metrica_1", "metrica_2"},
    1: {"idsse_J03WPY", "idsse_J03WN1", "idsse_J03WMX", "idsse_J03WR9", "metrica_1", "metrica_2"},
    2: {"idsse_J03WPY", "idsse_J03WN1", "idsse_J03WMX", "idsse_J03WOH", "idsse_J03WQQ", "idsse_J03WOY"},
}
POS_RATE = 0.012


def phys_features(c):
    """+2 features from f dict (from producer phys228_screen.py)."""
    f = c.f
    seen = f["seen"]
    n = c.m.n
    d = np.zeros((n, 2))
    margin = f["da"] - f["dh"]
    d[:, 0] = np.gradient(margin)
    d[:, 1] = np.where(seen, f["vx"] * np.sign(f["da"] - f["dh"]), 0.0)
    return d


def viterbi_team(logE, cost_team, cost_dead, extra_team):
    n = len(logE)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logE[0].copy()
    for t in range(1, n):
        T = -cost_dead * _DEAD
        T[0, 1] = T[1, 0] = -(cost_team + extra_team[t])
        cand = score[:, None] + T
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def head_prob_for_case(case, models):
    """228-dim: full_plus3 + phys_features. Public/masked avg all 3; ext uses held-out fold."""
    X = np.hstack([feats.full_plus3(case.m, case.f, case.Xs), phys_features(case)])
    name = case.name
    if name.startswith(("public", "masked")):
        return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    for fold, trains in FOLD_TRAINS.items():
        if name not in trains:
            return models[fold].predict_proba(X)[:, 1]
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def decode_with_head(c, Pd, head_probs, lam):
    p = dict(decode.PARAMS)
    p.update(final.DECODER)
    m, f = c.m, c.f
    P = Pd[c.name]
    R = np.full(m.n, "X", dtype="<U1")
    logE = decode.emissions(P, R, f["seen"], p)
    logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])

    eps = 1e-6
    hp = np.clip(head_probs, eps, 1 - eps)
    logodds = np.log(hp / (1 - hp))
    logodds_prior = np.log(POS_RATE / (1 - POS_RATE))
    corrected = logodds - logodds_prior
    extra_team = -lam * corrected

    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        out[a:b + 1] = STATES[viterbi_team(logE[a:b + 1], p["cost_team"], p["cost_dead"],
                                           extra_team[a:b + 1])]
    out = decode.min_duration(out, p["min_dead"], p["min_poss"])
    if p["stop_gate"]:
        out = decode.stop_gate(out, f, int(p["stop_gate"]))
    return out


def main():
    import joblib
    models = [joblib.load(os.path.join(CACHE_DIR, "ar_phys228_fold{}.joblib".format(i)))
              for i in range(3)]
    y = Yard(pfile="P_R1.npz")
    Pd = y.P

    print("=== computing phys228 head probabilities ===", flush=True)
    head_P = {}
    for c in [y.b.public] + y.b.masked + list(y.b.ext.values()):
        head_P[c.name] = head_prob_for_case(c, models)
        print("  {}: mean p={:.4f} max p={:.4f}".format(c.name, head_P[c.name].mean(),
                                                        head_P[c.name].max()), flush=True)
    np.savez_compressed(os.path.join(CACHE_DIR, "ar_phys228_head_probs.npz"), **head_P)

    def mk(lam):
        def fn(c, _p):
            return decode_with_head(c, Pd, head_P[c.name], lam)
        return fn

    print("\n=== λ=0 identity (phys228 head) ===", flush=True)
    base = y.report("phys_lambda0", mk(0.0))
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        base["bench"], base["ext"], base["masked"], base["public"], base["pubM"], base["val"]))

    print("\n=== λ=2 (phys228 head) ===", flush=True)
    out = y.report("phys_lambda2", mk(2.0))
    print("  bench {:.4f} ext {:.4f} masked {:.4f} public {:.4f}/{:.4f} val {:.4f}".format(
        out["bench"], out["ext"], out["masked"], out["public"], out["pubM"], out["val"]))
    print("  delta: bench {:+.4f} ext {:+.4f} masked {:+.4f} public {:+.4f} val {:+.4f}".format(
        out["bench"] - base["bench"], out["ext"] - base["ext"], out["masked"] - base["masked"],
        out["public"] - base["public"], out["val"] - base["val"]))


if __name__ == "__main__":
    main()
