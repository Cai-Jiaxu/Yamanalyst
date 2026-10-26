"""E1 full recipe: contest-duel reweighting under the frozen v5 training recipe.

Trains the reweighted arm only (w_duel=0.5 on IDSSE frames where ball seen AND
>=1 home AND >=1 away player within 2 m). Baseline arm = cached P_R1_folds.npz
(public/masked) decoded with the same frozen decoder; baseline ext/bench anchor
is the audited v5 (ext 0.616, bench 0.6612).

Frozen flags match mtah/final.py: 600-tree HistGradientBoosting, STEP=5, lr/swap
mirrors, three extra-noise draws per external match, early_stopping=False,
seed 0, decoder stop_gate=100 / kickoff='half'. No production code mutated.
"""
import os
import time
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

from mtah import feats
from mtah.bench import Bench, MODEL_DEFAULTS, FOLDS
from mtah.shape_model import STEP

W_DUEL = 0.5
SEEN, NH2, NA2 = 184, 180, 181  # ball_block columns (verified corr == 1.0)
# Match production _fit: H=0, A=1, D=2 (NOT sklearn alphabetical A,D,H)
LABEL_MAP = {"H": 0, "A": 1, "D": 2}
FULL_DECODER = {"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 12.0,
                "min_dead": 25, "w_bound": 0.0, "kickoff": "half",
                "use_rules": False, "stop_gate": 100}
MODEL_FULL = dict(MODEL_DEFAULTS, max_iter=600)
OUT = "cache/ar_e1_full_folds.npz"
RESULT_PATH = "cache/ar_e1_full_results.json"
# audited v5 anchors (analysis/autonomous_research/audit/AUDIT.md)
AUDIT_EXT = 0.616
AUDIT_BENCH = 0.6612


def duel_weight(X, is_idsse):
    if not is_idsse:
        return np.ones(X.shape[0], dtype=np.float32)
    seen = X[:, SEEN] > 0.5
    duel = seen & (X[:, NH2] >= 1) & (X[:, NA2] >= 1)
    w = np.ones(X.shape[0], dtype=np.float32)
    w[duel] = W_DUEL
    return w


def build_fold(b, fold, Xb, labb, mskb):
    mir = feats.mirrored(feats.full_plus3)
    aug = feats.augmented(feats.full_plus3)
    Xs, ys, ms, ws = [], [], [], []
    for name in b.ext:
        if name in fold:
            continue
        is_idsse = name.startswith("idsse")
        Xs.append(Xb[name]); ys.append(labb[name]); ms.append(mskb[name])
        ws.append(duel_weight(Xb[name], is_idsse))
        for (xc, yc, mc) in mir.get(name, []):
            Xs.append(xc); ys.append(yc); ms.append(mc)
            ws.append(duel_weight(xc, is_idsse))
        for (xc, yc, mc) in aug.get(name, []):
            Xs.append(xc); ys.append(yc); ms.append(mc)
            ws.append(duel_weight(xc, is_idsse))
    Xtr = np.vstack(Xs)
    ytr = np.concatenate(ys)
    mtr = np.concatenate(ms).astype(bool)
    wtr = np.concatenate(ws)
    idx = np.flatnonzero(mtr)[::STEP]
    ysub = np.array([LABEL_MAP[c] for c in ytr[idx]])
    return Xtr[idx], ysub, wtr[idx]


def main():
    import sys
    score_only = "--score-only" in sys.argv
    t0 = time.time()
    b = Bench(verbose=True)
    from mtah.score import pm
    Xb = {k: feats.full_plus3(c.m, c.f, c.Xs) for k, c in b.ext.items()}
    labb = {k: np.array(pm.LABELS)[c.gt] for k, c in b.ext.items()}
    mskb = {k: c.gt != pm.LABELS.index("X") for k, c in b.ext.items()}

    if score_only:
        # Load saved fold models, predict, score. No retraining.
        models = [joblib.load("cache/ar_e1_model_fold{}.joblib".format(i))
                  for i in range(len(FOLDS))]
        P = {}
        for c in [b.public] + b.masked:
            Xo = feats.full_plus3(c.m, c.f, c.Xs)
            P[c.name] = np.mean([m.predict_proba(Xo) for m in models], axis=0)
        for i, fold in enumerate(FOLDS):
            for k in fold:
                P[k] = models[i].predict_proba(Xb[k])
    else:
        cached = {}
        if os.path.exists(OUT):
            z = np.load(OUT, allow_pickle=True)
            for k in z.files:
                cached[k] = z[k]
            print("resumed with", len(cached), "cached keys", flush=True)
        P = {}
        for c in [b.public] + b.masked:
            if c.name in cached:
                P[c.name] = cached[c.name]
        extP = {k: cached[k] for k in cached if not k.startswith("model#")
                and k != c.name}
        # reload ext predictions from cache (bare keys)
        for k in b.ext:
            if k in cached:
                extP[k] = cached[k]

        for i, fold in enumerate(FOLDS):
            mp = "cache/ar_e1_model_fold{}.joblib".format(i)
            if os.path.exists(mp):
                print("fold", i, "model exists, loading", flush=True)
                clf = joblib.load(mp)
            else:
                print("fold", i, "training 600 trees mirrors+noise w_duel=0.5 ...",
                      flush=True)
                Xtr, ytr, wtr = build_fold(b, fold, Xb, labb, mskb)
                print("  rows", Xtr.shape, "down-weighted frac",
                      round(float((wtr == W_DUEL).mean()), 4), flush=True)
                clf = HistGradientBoostingClassifier(early_stopping=False,
                                                     random_state=0, **MODEL_FULL)
                clf.fit(Xtr, ytr, sample_weight=wtr)
                joblib.dump(clf, mp)
            for k in fold:
                extP[k] = clf.predict_proba(Xb[k])
            for c in [b.public] + b.masked:
                Xo = feats.full_plus3(c.m, c.f, c.Xs)
                P[c.name] = clf.predict_proba(Xo) if c.name not in P else \
                    P[c.name] + clf.predict_proba(Xo)
            print("  fold", i, "done, elapsed", int(time.time() - t0), "s", flush=True)

        nf = len(FOLDS)
        P = {k: v / nf for k, v in P.items()}
        P.update(extP)

    # assert keys before scoring
    expected = [b.public.name] + [c.name for c in b.masked] + list(b.ext.keys())
    missing = [k for k in expected if k not in P]
    assert not missing, "missing P keys: {}".format(missing)

    res = b.score("e1_full_w0.5", P, dec_params=FULL_DECODER, use_rules=False, t0=t0)
    # res[k] is [spellF1, macroF1]; official bench uses [0] = spellF1
    rw = {
        "bench": float(res["bench"]),
        "ext": float(res["ext"][0]),
        "idsse": float(res["idsse"][0]),
        "metrica": float(res["metrica"][0]),
        "masked": float(res["masked"][0]),
        "public": float(res["public"][0]),
    }

    out = {"date": time.strftime("%Y-%m-%d %H:%M"),
           "protocol": "full_600_mirrors_noise_step5_gate100",
           "w_duel": W_DUEL,
           "reweighted": rw,
           "baseline_audit": {"ext": AUDIT_EXT, "bench": AUDIT_BENCH}}
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== E1 FULL RESULT ===")
    print("reweighted:", rw)
    print("baseline audit: ext", AUDIT_EXT, "bench", AUDIT_BENCH)
    print("dbench = {:+.4f}  dext = {:+.4f}  dmasked = {:+.4f}".format(
        rw["bench"] - AUDIT_BENCH, rw["ext"] - AUDIT_EXT, rw["masked"]))
    print("wrote", RESULT_PATH)


if __name__ == "__main__":
    main()
