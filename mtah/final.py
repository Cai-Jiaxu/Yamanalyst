"""The submission model: trained on every external match, applied to our match.

Chosen on the shared benchmark (mtah/bench.py, analysis/experiments.py,
analysis/decoder_sweep.py):
  features   mtah.feats.full_plus3 (full_plus2 + set-piece geometry)
  training   9 public external matches with TRUE labels, each degraded to our ATD's noise, plus
             left/right and home/away mirrors and three extra noise draws per match
  model      HistGradientBoosting, 600 trees (frame classifier H/A/D) and, when the decoder uses
             it, a second one for 'boundary within +-6 frames'
  decoder    3-state Viterbi on the model's probabilities alone (no hand-weighted rule term:
             the rule labels enter as features instead), with DECODER settings below
"""

import os

import numpy as np

from mtah import decode, external, feats, noise, shape
from mtah.bench import BOUNDARY_R, MODEL_DEFAULTS, boundary_target  # noqa: F401
from mtah.data import CACHE_DIR
from mtah.features import compute
from mtah.shape_model import EXT_NOISE_VERSION, STEP

MODEL = {"max_iter": 600}
# analysis/round2.py on the leak-free bench: base 0.649 -> min_dead 0.653 -> set-piece 0.656.
# Not adopted (measured): boundary head (+-0.001), EM prior (-0.017), IDSSE relabel (-0.002),
# full kickoff evidence (-0.008). Adopted in final_v2: kickoff='half', dead ball before the 2nd
# half's kickoff only (bench unchanged, 0.656; fixes 19 spurious spells in our 2,934 pre-kickoff
# frames, which the bench cannot see -- analysis/kickoff_modes.py).
# final_v3/v4 let the rule labels vote in the decoder (w_rule 0.25), chosen on the eye-labelled
# 2nd half. Reverted in final_v5: those labels put H<->A changes where the rules do, while the
# real GT puts them a median 1.6 s later, so they reward rule timing (the rules alone score best
# there and worst on every real-GT yardstick; INVESTIGATION.md section 20).
# final_v4 added stop_gate (short live blips inside stoppages -> dead ball; section 19), kept.
# final_v5 = final_v2 decoder + stop_gate: bench 0.636 -> 0.661, ext 0.589 -> 0.616,
# public 0.667 -> 0.697 on the fold models (analysis/rev_v4.py).
# final_v6 = final_v5 + stop_gate_sides=1: after the shipped gate, a second pass also turns a short
# slow live spell with dead ball on ONE side into dead ball (analysis/r20_gate.py, INVESTIGATION.md
# section 27): bench 0.6612 -> 0.6663, ext 0.6163 -> 0.6146, public 0.697 -> 0.720 on the fold
# models; 5 spells change on the full match, the 3 that fall on labelled frames are all spurious.
DECODER_V5 = {"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 12.0, "min_dead": 25,
              "w_bound": 0.0, "kickoff": "half", "use_rules": False, "stop_gate": 100}
DECODER = dict(DECODER_V5, stop_gate_sides=1)
# DECODER_V5 is the locked development baseline of baseline_snapshot/ (bench 0.6612).
FEATFN = "full_plus3"
SEED = 0


def external_training():
    """[(X, labels, mask)] for every external match and its augmented copies."""
    featfn = getattr(feats, FEATFN)
    rows = []
    mir = feats.mirrored(featfn)
    aug = feats.augmented(featfn)
    for name in external.available():
        em = external.ExtMatch(name)
        path = os.path.join(CACHE_DIR, "final_{}_{}_{}.npz".format(
            name, featfn.__name__, EXT_NOISE_VERSION))
        if os.path.isfile(path):
            X = np.load(path)["X"]
        else:
            noisy = noise.degrade(em, seed=noise.seed_for(name))
            Xs, _ = shape.compute(noisy, cam=False)
            X = featfn(noisy, compute(noisy), Xs)
            np.savez_compressed(path, X=X)
        rows.append((X, em.labels, em.graded_mask()))
        rows += mir.get(name, []) + aug.get(name, [])
    return rows


def _fit(X, y, mask, boundary=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    idx = np.flatnonzero(mask)[::STEP]
    if boundary:
        target = y.astype(int)
    else:
        target = np.array([{"H": 0, "A": 1, "D": 2}[c] for c in y[idx]])
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                         **dict(MODEL_DEFAULTS, **MODEL))
    clf.fit(X[idx], target[idx] if boundary else target)
    return clf


def predict(m, f, R):
    """Labels for our match, plus the probabilities (for reporting)."""
    rows = external_training()
    X = np.vstack([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    mask = np.concatenate([r[2] for r in rows])
    Xs, _ = shape.compute(m, cam=False)
    Xo = getattr(feats, FEATFN)(m, f, Xs)
    P = _fit(X, y, mask).predict_proba(Xo)
    B = None
    if DECODER.get("w_bound", 0):
        bt = np.concatenate([boundary_target(r[1]) for r in rows])
        B = _fit(X, bt, mask, boundary=True).predict_proba(Xo)[:, 1]
    p = dict(decode.PARAMS, **DECODER)
    if p.get("em_prior"):
        p["train_prior"] = np.array([np.mean(y[mask] == k) for k in "HAD"])
    R_dec = R if p.get("use_rules") else np.full(m.n, "X", dtype="<U1")
    return decode.decode(m, R_dec, P, f["seen"], p, f, B), P
