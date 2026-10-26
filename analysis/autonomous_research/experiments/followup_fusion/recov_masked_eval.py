"""Full masked-variant evaluation with dense recovered broadcast ball positions.
Public may have few/no injected frames (it's the original graded window), so public flat is
expected. The masked variants have artificial ball missing, where recovered positions can land.

Reports: label-diff counts, prob max-diff, and official spellF1/macroF1 per masked variant.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, feats, final  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.score import pm  # noqa: E402

STATES = np.array(["H", "A", "D"])
_TEAM = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def viterbi_team(logE, cost_team, cost_dead):
    back = np.zeros((len(logE), 3), dtype=np.int8)
    score = logE[0].copy()
    for t in range(1, len(logE)):
        T = -cost_dead * _DEAD
        T[0, 1] = T[1, 0] = -cost_team
        cand = score[:, None] + T
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(len(logE), dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(len(logE) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def main():
    import joblib
    from mtah.data import CACHE_DIR
    rec = np.load(os.path.join(ROOT, "cache", "ar_followup_fusion_recpos.npy"))
    models = joblib.load(os.path.join(CACHE_DIR, "models_final_v1.joblib"))
    models = models["models"] if isinstance(models, dict) else models
    y = Yard(pfile="P_R1.npz")
    b = y.b
    p = dict(decode.PARAMS)
    p.update(final.DECODER)

    def predict(c, inject):
        m = c.m
        ball = m.ball.copy()
        if inject:
            missing = np.isnan(ball[:, 0])
            have = np.isfinite(rec[:, 0])
            ball[missing & have, :2] = rec[missing & have]
        orig = m.ball
        m.ball = ball
        f = compute(m)
        X = feats.full_plus3(m, f, c.Xs)
        m.ball = orig
        P = np.mean([clf.predict_proba(X) for clf in models], axis=0)
        return P, f

    def decode_case(c, P, f):
        m = c.m
        R = np.full(m.n, "X", dtype="<U1")
        logE = decode.emissions(P, R, f["seen"], p)
        from mtah import kickoff
        logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
        out = np.full(m.n, "X", dtype="<U1")
        for _, a, bnd in m.halves:
            out[a:bnd + 1] = STATES[viterbi_team(logE[a:bnd + 1], p["cost_team"], p["cost_dead"])]
        out = decode.min_duration(out, p["min_dead"], 0)
        if p["stop_gate"]:
            out = decode.stop_gate(out, f, int(p["stop_gate"]))
        return out

    print("=== per-case: baseline vs injected (label diffs, prob max-diff, score) ===")
    sp_base, sp_inj = [], []
    for c in [b.public] + b.masked:
        P0, f0 = predict(c, inject=False)
        P1, f1 = predict(c, inject=True)
        lab0 = decode_case(c, P0, f0)
        lab1 = decode_case(c, P1, f1)
        diff = (lab0 != lab1)
        pdiff = np.abs(P1 - P0).max()
        ev0 = pm.evaluate(c.gt, pm.label_codes(lab0), bootstrap_n=0)
        ev1 = pm.evaluate(c.gt, pm.label_codes(lab1), bootstrap_n=0)
        n_inj = int(np.isfinite(rec[:, 0]).sum() & np.isnan(c.m.ball[:, 0]).sum()) if c.name.startswith("masked") else int(np.isfinite(rec[:, 0]).sum())
        print("  {:<10}: injected_frames ~{:<6} label_diff {:<5} prob_maxdiff {:.4f} | base sp {:.4f}/mf {:.4f} -> inj sp {:.4f}/mf {:.4f}".format(
            c.name, n_inj, int(diff.sum()), pdiff, ev0["spell_f1_mean"], ev0["macro_f1"],
            ev1["spell_f1_mean"], ev1["macro_f1"]))
        if c.name.startswith("masked"):
            sp_base.append(ev0["spell_f1_mean"])
            sp_inj.append(ev1["spell_f1_mean"])
    print("masked mean: base {:.4f} -> inj {:.4f} (delta {:+.4f})".format(
        np.mean(sp_base), np.mean(sp_inj), np.mean(sp_inj) - np.mean(sp_base)))


if __name__ == "__main__":
    main()
