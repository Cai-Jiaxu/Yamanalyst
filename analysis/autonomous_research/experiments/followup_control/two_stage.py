"""Two-stage: cached P_R1 primary + typed H/A transition head cost delta.
Reuses viterbi_team from e3_sweep.py.
cost_team_t = 12 - λ*(logodds_head - logodds_prior), clipped ±4.
High turnover posterior -> LOWER switch cost.
"""
import numpy as np, joblib, time, json
from mtah import feats, decode, final, kickoff
from mtah.bench import Bench, FOLDS
from mtah.score import pm

STATES = np.array(["H", "A", "D"])
_TEAM = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
_DEAD = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
OUT = "cache/ar_followup_two_stage.json"


def viterbi_team(logE, cost_team, cost_dead, extra_team, bonus=None):
    n = len(logE)
    back = np.zeros((n, 3), dtype=np.int8)
    score = logE[0].copy()
    off = 1.0 - np.eye(3)
    for t in range(1, n):
        T = -cost_dead * _DEAD
        T[0, 1] = T[1, 0] = -(cost_team + extra_team[t])
        cand = score[:, None] + T
        if bonus is not None:
            cand = cand + bonus[t] * off
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logE[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_with_delta(c, P, trans_post, lam=2.0, prior=0.012):
    p = dict(decode.PARAMS)
    p.update(final.DECODER)
    m, f = c.m, c.f
    R = np.full(m.n, "X", dtype="<U1")
    B = P.get("B:" + c.name)
    bonus = decode.boundary_bonus(B, p["w_bound"]) if B is not None and p["w_bound"] else None
    logE = decode.emissions(P[c.name], R, f["seen"], p)
    logE = logE + kickoff.evidence(m, f, mode=p["kickoff"])
    # Cost delta: high turnover -> LOWER switch cost
    logodds_head = np.log(trans_post / (1 - trans_post + 1e-10) + 1e-10)
    logodds_prior = np.log(prior / (1 - prior))
    extra_team = -lam * (logodds_head - logodds_prior)
    extra_team = np.clip(extra_team, -4, 4)
    out = np.full(m.n, "X", dtype="<U1")
    for _, a, b in m.halves:
        bb = None if bonus is None else bonus[a:b + 1]
        out[a:b + 1] = STATES[viterbi_team(logE[a:b + 1], p["cost_team"], p["cost_dead"],
                                           extra_team[a:b + 1], bb)]
    out = decode.min_duration(out, p["min_dead"], p["min_poss"])
    if p["stop_gate"]:
        out = decode.stop_gate(out, f, int(p["stop_gate"]))
    return out


def main():
    t0 = time.time()
    b = Bench(verbose=True)

    # Load baseline models for external matches
    base_models = joblib.load("cache/models_final_v1.joblib")
    # Load transition head models
    trans_models = [joblib.load(f"cache/ar_followup_baseline226_fold{i}.joblib")
                    for i in range(3)]

    # Build P dict
    P = {}
    cached = np.load("cache/P_R1_folds.npz", allow_pickle=True)
    P["public"] = np.mean([cached["public#{}".format(i)] for i in range(3)], axis=0)
    for s in range(16):
        key = "masked{}".format(s)
        keys = ["{}#{}".format(key, i) for i in range(3)]
        if all(k in cached.files for k in keys):
            P[key] = np.mean([cached[k] for k in keys], axis=0)
    # External
    for i, fold in enumerate(FOLDS):
        for k in fold:
            c = b.ext[k]
            Xb = feats.full_plus3(c.m, c.f, c.Xs)
            P[k] = base_models[i].predict_proba(Xb)

    # λ=0 baseline
    DECODER = {"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 12.0,
               "min_dead": 25, "w_bound": 0.0, "kickoff": "half",
               "use_rules": False, "stop_gate": 100}
    res0 = b.score("lambda0", P, dec_params=DECODER, use_rules=False, t0=t0)
    print("λ=0 bench:", float(res0["bench"]), flush=True)

    with open(OUT, "w") as f:
        json.dump({"lambda0_bench": float(res0["bench"]),
                    "lambda0_ext": float(res0["ext"][0]),
                    "lambda0_masked": float(res0["masked"][0])}, f, indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
