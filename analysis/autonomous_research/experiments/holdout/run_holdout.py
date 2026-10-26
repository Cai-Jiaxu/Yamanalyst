"""Paired holdout eval: baseline fold ensemble vs E1 fold ensemble on Metrica Game 3.

Frozen inputs:
  holdout  cache/ar_holdout/ext_g3.npz   sha aa1589d7...a23d7
  baseline cache/models_final_v1.joblib    sha cd1bf353...39d4c  (3 folds, 226-dim, classes [0,1,2])
  E1       cache/ar_e1_model_fold{0,1,2}.joblib  (3 folds, 226-dim, classes [0,1,2])

Protocol: matched fold ensembles averaged, frozen v5 decoder (w_shape 0.75, cost_dead 12,
cost_team 12, min_dead 25, kickoff=half, stop_gate=100, w_bound=0). Clean = primary.
Stress = Bernoulli p=0.41 ball-missing, seed 20261025 (stress, not realistic).
No retune, no model selection.
"""
import os
import sys

import joblib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from mtah import feats, rules, decode, kickoff  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah import shape as shape_mod  # noqa: E402
from mtah.score import pm  # noqa: E402

HOLDOUT = os.path.join(ROOT, "cache", "ar_holdout", "ext_g3.npz")
BASELINE = os.path.join(ROOT, "cache", "models_final_v1.joblib")
E1_FOLDS = [os.path.join(ROOT, "cache", "ar_e1_model_fold{}.joblib".format(i)) for i in range(3)]
STRESS_P = 0.41
STRESS_SEED = 20261025

STATES = np.array(["H", "A", "D"])
_TEAM = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
DEC = dict(w_shape=0.75, cost_dead=12.0, cost_team=12.0, min_dead=25,
           kickoff="half", use_rules=False, stop_gate=100, w_bound=0.0)


class HoldoutMatch:
    def __init__(self, z):
        self.players, self.ball, self.team = z["players"], z["ball"], z["team"]
        self.labels = z["labels"]
        self.n = len(self.ball)
        self.halves = [tuple(int(v) for v in h) for h in z["halves"]]
        self.pitch = tuple(float(v) for v in z["pitch"])
        self.fps = 25
        self.source = "holdout_g3"
        self.cam = np.full((self.n, 2), np.nan, dtype=np.float32)


def viterbi_team(logE, cost_team, cost_dead):
    n = len(logE)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logE[0].copy()
    for t in range(1, n):
        T = -cost_dead * _DEAD
        T[0, 1] = T[1, 0] = -cost_team
        cand = score[:, None] + T
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_labels(m, f, P):
    p = dict(decode.PARAMS)
    p.update(DEC)
    logE = decode.emissions(P, f["rule_labels"], f["seen"], p)
    logE = logE + kickoff.evidence(m, f, mode=DEC["kickoff"])
    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        out[a:b + 1] = STATES[viterbi_team(logE[a:b + 1], DEC["cost_team"], DEC["cost_dead"])]
    out = decode.min_duration(out, DEC["min_dead"], 0)
    out = decode.stop_gate(out, f, int(DEC["stop_gate"]))
    return out


def fold_avg(models, X):
    return np.mean([clf.predict_proba(X).astype(np.float32) for clf in models], axis=0)


def score_it(m, gt_codes, pred_labels):
    pred_codes = pm.label_codes(pred_labels)
    r = pm.evaluate(gt_codes, pred_codes, bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"]


def main():
    z = np.load(HOLDOUT, allow_pickle=True)
    m = HoldoutMatch(z)
    gt = pm.label_codes(m.labels)

    # clean features
    f_clean = compute(m)
    R = rules.run(m, f_clean)
    f_clean["rule_labels"] = R
    Xs_clean, _ = shape_mod.compute(m, cam=False)
    X_clean = feats.full_plus3(m, f_clean, Xs_clean)

    # stress: deterministic ball-missing mask
    rng = np.random.default_rng(STRESS_SEED)
    graded = m.labels != "X"
    miss = rng.random(m.n) < STRESS_P
    hide = graded & miss
    f_stress = compute(m, hide=hide)
    R_s = rules.run(m, f_stress)
    f_stress["rule_labels"] = R_s
    Xs_stress, _ = shape_mod.compute(m, cam=False)
    X_stress = feats.full_plus3(m, f_stress, Xs_stress)

    base = joblib.load(BASELINE)
    base = base["models"] if isinstance(base, dict) else base
    e1 = [joblib.load(p) for p in E1_FOLDS]

    print("=== baseline fold ensemble ===")
    P_base_clean = fold_avg(base, X_clean)
    P_base_stress = fold_avg(base, X_stress)
    print("  clean P shape", P_base_clean.shape, "classes_ order [0,1,2]=H,A,D")

    print("=== E1 fold ensemble ===")
    P_e1_clean = np.mean([clf.predict_proba(X_clean).astype(np.float32) for clf in e1], axis=0)
    P_e1_stress = np.mean([clf.predict_proba(X_stress).astype(np.float32) for clf in e1], axis=0)
    print("  clean P shape", P_e1_clean.shape)

    for tag, f, P in [
        ("clean", f_clean, P_base_clean),
        ("clean", f_clean, P_e1_clean),
    ]:
        pred = decode_labels(m, f, P)
        sp, mf = score_it(m, gt, pred)
        print("{:10s} spellF1 {:.4f} macroF1 {:.4f}".format(tag, sp, mf))

    for tag, f, P in [
        ("stress", f_stress, P_base_stress),
        ("stress", f_stress, P_e1_stress),
    ]:
        pred = decode_labels(m, f, P)
        sp, mf = score_it(m, gt, pred)
        print("{:10s} spellF1 {:.4f} macroF1 {:.4f}".format(tag, sp, mf))


if __name__ == "__main__":
    main()
