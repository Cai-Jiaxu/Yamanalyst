"""Correct masked recovery: reconstruct the exact per-seed hide mask, then inject recovered
broadcast positions ONLY where the hide would otherwise delete the ball. Baseline (same hide,
no injection) must reproduce cached P_R1 masked scores before trusting the injected run.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from analysis.yardsticks import Yard  # noqa: E402
from mtah import decode, feats, final, kickoff  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.masking import ball_mask, pace_weights  # noqa: E402
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

    # reconstruct weights as Bench does
    f_full = compute(b.m)
    weights = pace_weights(f_full)

    def predict_with_seed(seed, inject):
        hide = ball_mask(b.m.n, seed, weights=weights)
        m = b.m
        ball = m.ball.copy()
        if inject:
            # where hide is True but we have a recovered position, keep recovered; else NaN
            have = np.isfinite(rec[:, 0])
            fill = hide & have
            ball[fill, :2] = rec[fill]
            # remaining hide frames stay NaN
            still_hide = hide & ~have
            ball[still_hide] = np.nan
        else:
            ball[hide] = np.nan
        orig = m.ball
        m.ball = ball
        f = compute(m, hide=None)
        X = feats.full_plus3(m, f, b.public.Xs)
        m.ball = orig
        P = np.mean([clf.predict_proba(X) for clf in models], axis=0)
        return P, f, hide

    def decode_case(m, P, f):
        R = np.full(m.n, "X", dtype="<U1")
        logE = decode.emissions(P, R, f["seen"], p)
        logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
        out = np.full(m.n, "X", dtype="<U1")
        for _, a, bnd in m.halves:
            out[a:bnd + 1] = STATES[viterbi_team(logE[a:bnd + 1], p["cost_team"], p["cost_dead"])]
        out = decode.min_duration(out, p["min_dead"], 0)
        if p["stop_gate"]:
            out = decode.stop_gate(out, f, int(p["stop_gate"]))
        return out

    print("=== identity check: re-predicted baseline vs cached P_R1 ===")
    for seed in [0, 1]:
        P0, f0, hide = predict_with_seed(seed, inject=False)
        lab0 = decode_case(b.m, P0, f0)
        ev0 = pm.evaluate(b.masked[seed].gt, pm.label_codes(lab0), bootstrap_n=0)
        cached_P = y.P["masked{}".format(seed)]
        cached_lab = decode_case(b.m, cached_P, b.masked[seed].f)
        cached_ev = pm.evaluate(b.masked[seed].gt, pm.label_codes(cached_lab), bootstrap_n=0)
        print("  seed{}: hide {} frames | re-pred sp {:.4f}/mf {:.4f} | cached P_R1 sp {:.4f}/mf {:.4f}".format(
            seed, int(hide.sum()), ev0["spell_f1_mean"], ev0["macro_f1"],
            cached_ev["spell_f1_mean"], cached_ev["macro_f1"]))

    print("=== injected recovery on seeds 0,1 ===")
    for seed in [0, 1]:
        P0, f0, hide = predict_with_seed(seed, inject=False)
        P1, f1, _ = predict_with_seed(seed, inject=True)
        lab0 = decode_case(b.m, P0, f0)
        lab1 = decode_case(b.m, P1, f1)
        ev0 = pm.evaluate(b.masked[seed].gt, pm.label_codes(lab0), bootstrap_n=0)
        ev1 = pm.evaluate(b.masked[seed].gt, pm.label_codes(lab1), bootstrap_n=0)
        diff = (lab0 != lab1).sum()
        pdiff = np.abs(P1 - P0).max()
        n_fill = int((hide & np.isfinite(rec[:, 0])).sum())
        print("  seed{}: recovered {} of {} hidden frames | label_diff {} prob_maxdiff {:.4f} | base {:.4f}/{:.4f} -> inj {:.4f}/{:.4f}".format(
            seed, n_fill, int(hide.sum()), int(diff), pdiff,
            ev0["spell_f1_mean"], ev0["macro_f1"], ev1["spell_f1_mean"], ev1["macro_f1"]))


if __name__ == "__main__":
    main()
