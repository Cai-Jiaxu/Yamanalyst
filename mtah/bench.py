"""One benchmark for every modelling decision: three yardsticks, computed the same way each time.

    from mtah.bench import Bench
    b = Bench()
    b.evaluate(name, featfn, model_params={}, dec_params={}, use_rules=True)

  ext      9 public external matches degraded to our noise, 3 folds by match: each match is scored
           against its TRUE labels by a model trained on the other two folds (~3,000 possessions)
  masked   our match, public window, ball hidden in 4-40 s stretches (16 seeds): the full-match proxy
  public   our match, public window as delivered (17 possessions; a sanity check, not a ranking)

The model used on our match is the mean of the three fold models, so no extra fit is needed.
featfn(match, f, shapeX) -> (n, k) features; shapeX is the cached ball-free shape matrix
(cam=False) of that match and f the mtah.features dict -- for the masked variants of our match f
is recomputed with the ball hidden, so ball-aware features are scored honestly.
"""

import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah import decode, external, noise, rules, shape, shape_model as sm
from mtah.data import CACHE_DIR, Match
from mtah.features import compute
from mtah.masking import W0, W1, ball_mask, pace_weights
from mtah.score import load_gt, pm

FOLDS = [["idsse_J03WPY", "idsse_J03WN1", "idsse_J03WMX"],
         ["idsse_J03WOH", "idsse_J03WQQ", "idsse_J03WOY"],
         ["idsse_J03WR9", "metrica_1", "metrica_2"]]
MASK_SEEDS = range(16)
BENCH_VERSION = "v3"  # rules/features cache; shape matrices follow EXT_NOISE_VERSION
MODEL_DEFAULTS = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
                  "min_samples_leaf": 100, "l2_regularization": 1.0}


SCREEN = {"max_iter": 300}   # ROADMAP rule 7: fast screen, with no mirror or noise copies


class Averaged:
    """Mean of several classifiers' probabilities (ROADMAP R5: a second model family)."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([c.predict_proba(X) for c in self.models], axis=0)


def _catboost(X, y, params):
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(iterations=int(params.get("max_iter", 600)), depth=6,
                             learning_rate=params.get("cb_learning_rate", 0.1), l2_leaf_reg=3.0,
                             loss_function="MultiClass", random_seed=0, thread_count=8,
                             verbose=False, allow_writing_files=False)
    return clf.fit(X, y)


def _fit(X, labels, mask, params, step=sm.STEP):
    """params['family']: 'hgb' (default, sklearn trees), 'catboost', or 'both' (their mean)."""
    idx = np.flatnonzero(mask)[::step]
    y = np.array([{"H": 0, "A": 1, "D": 2}[c] for c in labels[idx]])
    params = dict(params)
    family = params.pop("family", "hgb")
    out = []
    if family in ("hgb", "both"):
        hp = {k: v for k, v in params.items() if k != "cb_learning_rate"}
        clf = HistGradientBoostingClassifier(early_stopping=False, random_state=0,
                                             **dict(MODEL_DEFAULTS, **hp))
        out.append(clf.fit(X[idx], y))
    if family in ("catboost", "both"):
        out.append(_catboost(X[idx], y, params))
    return out[0] if len(out) == 1 else Averaged(out)


def taxonomy(gt, pred):
    """Counts of possession-level error types (on the abstention-filled prediction).

    split    true possessions covered by >= 2 predicted spells of the same team
    merge    predicted spells covering >= 2 true possessions of the same team
    drift    true possessions whose best same-team overlap has IoU in [0.1, 0.5)
    missed   true possessions with no same-team overlap at all
    spurious predicted spells with no same-team overlap at all
    """
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(filled, runs)
    out = dict(split=0, merge=0, drift=0, missed=0, spurious=0, n_gt=len(gs), n_pred=len(ps))
    starts = np.array([p[0] for p in ps])
    p_hits = np.zeros(len(ps), dtype=int)
    for a, b, k in gs:
        lo = max(np.searchsorted(starts, a, "right") - 1, 0)
        best, hits = 0.0, 0
        for j in range(lo, len(ps)):
            pa, pb, pk = ps[j]
            if pa >= b:
                break
            if pb <= a or pk != k:
                continue
            hits += 1
            p_hits[j] += 1
            best = max(best, (min(b, pb) - max(a, pa)) / (max(b, pb) - min(a, pa)))
        out["split"] += hits >= 2
        out["missed"] += hits == 0
        out["drift"] += 0.1 <= best < 0.5
    out["merge"] = int((p_hits >= 2).sum())
    out["spurious"] = int((p_hits == 0).sum())
    return out


BOUNDARY_R = 6  # frames either side of a true change that count as 'boundary here'


def boundary_target(labels):
    """1 within +-BOUNDARY_R frames of any change between H, A and D (X excluded)."""
    ch = np.zeros(len(labels), dtype=bool)
    ch[1:] = (labels[1:] != labels[:-1]) & (labels[1:] != "X") & (labels[:-1] != "X")
    k = np.ones(2 * BOUNDARY_R + 1)
    return np.convolve(ch.astype(float), k, "same") > 0


def _fit_boundary(Xt, yt, mt, params, step=sm.STEP):
    X = np.vstack(Xt)
    y = np.concatenate([boundary_target(l) for l in yt]).astype(int)
    idx = np.flatnonzero(np.concatenate(mt))[::step]
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=0,
                                         **dict(MODEL_DEFAULTS, **params))
    return clf.fit(X[idx], y[idx])


class Case:
    """Everything needed to score one match: a (possibly degraded) match, its features, rule
    labels, cached shape matrix and ground-truth codes."""

    def __init__(self, name, match, f, R, Xs, gt):
        self.name, self.m, self.f, self.R, self.Xs, self.gt = name, match, f, R, Xs, gt


class Bench:
    def __init__(self, verbose=True):
        t = time.time()
        self.ext = {}
        data = {d[0]: d for d in sm.external_set()}
        for name in external.available():
            em = external.ExtMatch(name)
            path = os.path.join(CACHE_DIR, "bench_{}_{}.npz".format(name, BENCH_VERSION))
            noisy = noise.degrade(em, seed=noise.seed_for(name))
            if os.path.isfile(path):
                z = np.load(path, allow_pickle=True)
                f = z["f"].item()
                R = z["R"]
            else:
                f = compute(noisy)
                R = rules.run(noisy, f)
                np.savez_compressed(path, f=np.array(f, dtype=object), R=R)
            self.ext[name] = Case(name, noisy, f, R, data[name][1], pm.label_codes(em.labels))

        self.m = Match()
        self.gt = load_gt()
        Xs, _ = shape.compute(self.m, cam=False)
        f = compute(self.m)
        self.public = Case("public", self.m, f, rules.run(self.m, f), Xs, self.gt)
        # hand-labelled validation windows (ROADMAP R3, mtah/label_tool.py), scored on the same
        # probabilities as the public window
        self.val = None
        vgt = load_val()
        if vgt is not None:
            self.val = Case("val", self.m, f, self.public.R, Xs, vgt)
            lab = np.array(pm.LABELS)[vgt]
            self.val_poss = int(np.sum(np.isin(lab[1:], ["H", "A"]) & (lab[1:] != lab[:-1])))
        self.masked = []
        weights = pace_weights(f)
        for s in MASK_SEEDS:
            fk = compute(self.m, hide=ball_mask(self.m.n, s, weights=weights))
            self.masked.append(Case("masked{}".format(s), self.m, fk, rules.run(self.m, fk), Xs,
                                    self.gt))
        if verbose:
            print("bench ready in {:.0f} s: {} external matches, {} mask seeds".format(
                time.time() - t, len(self.ext), len(self.masked)), flush=True)

    def probs(self, featfn, model_params=None, extra=None, boundary=False, relabel=None,
              return_models=False, key=None, fold_extra=None):
        """Fit the three fold models once; return {case name: (n, 3) probabilities}.

        extra: optional {match name: [(X, labels, mask), ...]} -- augmented copies of a match,
        added only to folds that train on that match (never to the fold that scores it).
        key: optional cache name. The fold models are saved to cache/models_<key>.joblib and
        reloaded instead of refitted next time; the key must change whenever featfn, params,
        extra or relabel do. Probabilities are always recomputed, so cases may be rebuilt.
        fold_extra: optional [[(X, labels, mask), ...] per fold] added to that fold's training
        only -- e.g. our own match pseudo-labelled by that fold's model (ROADMAP R4)."""
        import joblib
        model_params = model_params or {}
        path = os.path.join(CACHE_DIR, "models_{}.joblib".format(key)) if key else None
        cached = joblib.load(path) if path and os.path.isfile(path) else None
        if isinstance(cached, list):    # analysis/ball_value.py's format: models only
            cached = {"models": cached, "bmodels": []}
        if cached is not None and boundary and not cached["bmodels"]:
            cached = None
        X = {k: featfn(c.m, c.f, c.Xs) for k, c in self.ext.items()}
        lab = {k: np.array(pm.LABELS)[c.gt] for k, c in self.ext.items()}
        # training labels may follow a different convention than the scoring labels
        tlab = {k: (relabel(v) if relabel and k.startswith("idsse") else v)
                for k, v in lab.items()}
        msk = {k: c.gt != pm.LABELS.index("X") for k, c in self.ext.items()}
        P, models, bmodels = {}, [], []
        for i, fold in enumerate(FOLDS):
            if cached is not None:
                clf = cached["models"][i]
                models.append(clf)
                if boundary:
                    bclf = cached["bmodels"][i]
                    bmodels.append(bclf)
            else:
                train = [k for k in self.ext if k not in fold]
                Xt = [X[k] for k in train]
                yt, mt = [tlab[k] for k in train], [msk[k] for k in train]
                for k in train:
                    for xe, ye, me in (extra or {}).get(k, []):
                        if relabel and k.startswith("idsse"):
                            ye = relabel(ye)
                        Xt.append(xe), yt.append(ye), mt.append(me)
                for xe, ye, me in (fold_extra or [[]] * len(FOLDS))[i]:
                    Xt.append(xe), yt.append(ye), mt.append(me)
                clf = _fit(np.vstack(Xt), np.concatenate(yt), np.concatenate(mt), model_params)
                models.append(clf)
                if boundary:
                    bclf = _fit_boundary(Xt, yt, mt, model_params)
                    bmodels.append(bclf)
            for k in fold:
                P[k] = clf.predict_proba(X[k])
                if boundary:
                    P["B:" + k] = bclf.predict_proba(X[k])[:, 1]
        for c in [self.public] + self.masked:
            Xo = featfn(c.m, c.f, c.Xs)
            P[c.name] = np.mean([clf.predict_proba(Xo) for clf in models], axis=0)
            if boundary:
                P["B:" + c.name] = np.mean([b.predict_proba(Xo)[:, 1] for b in bmodels], axis=0)
        if path and cached is None:
            joblib.dump({"models": models, "bmodels": bmodels}, path)
        return (P, models) if return_models else P

    def train_prior(self):
        """(H, A, D) frequency over all external labelled frames -- the models' training mix."""
        lab = np.concatenate([c.gt[c.gt != pm.LABELS.index("X")] for c in self.ext.values()])
        return np.array([np.mean(lab == pm.LABELS.index(k)) for k in "HAD"])

    def score(self, name, P, dec_params=None, use_rules=True, verbose=True, t0=None):
        p = dict(decode.PARAMS, **(dec_params or {}))
        if p.get("em_prior") and "train_prior" not in p:
            p["train_prior"] = self.train_prior()
        res = {"idsse": [], "metrica": []}
        self._tax = {}
        for k, c in self.ext.items():
            res[k.split("_")[0]].append(self._score(c, P[k], p, use_rules, self._tax,
                                                    P.get("B:" + k)))
        pub = self._score(self.public, P["public"], p, use_rules, B=P.get("B:public"))
        val = None if self.val is None else \
            self._score(self.val, P["public"], p, use_rules, B=P.get("B:public"))
        masked = np.array([self._score(c, P[c.name], p, use_rules, B=P.get("B:" + c.name))
                           for c in self.masked])
        out = {
            "idsse": np.mean(res["idsse"], axis=0), "metrica": np.mean(res["metrica"], axis=0),
            "ext": np.mean(res["idsse"] + res["metrica"], axis=0),
            "public": np.array(pub), "masked": masked.mean(axis=0),
            "masked_sd": masked[:, 0].std(),
            "val": None if val is None else np.array(val),
        }
        out["bench"] = (out["ext"][0] + out["masked"][0]) / 2
        out["taxonomy"] = dict(self._tax)
        if verbose:
            print("{:<34} bench {:.3f} | ext {:.3f}/{:.3f} (idsse {:.3f} metrica {:.3f}) | "
                  "masked {:.3f}/{:.3f} | public {:.3f}/{:.3f} | {:.0f}s".format(
                      name, out["bench"], *out["ext"], out["idsse"][0], out["metrica"][0],
                      *out["masked"], *out["public"], time.time() - (t0 or time.time())),
                  flush=True)
            if val is not None:
                R = self.val.R if use_rules else np.full(self.val.m.n, "X", dtype="<U1")
                vc = pm.label_codes(decode.decode(self.val.m, R, P["public"],
                                                  self.val.f["seen"], p, self.val.f,
                                                  P.get("B:public")))
                vr = pm.evaluate(self.val.gt, vc, bootstrap_n=0)
                print("{:<34} val_h2 ({} possessions) spellF1/macroF1 {:.3f}/{:.3f} | "
                      "teamAcc {:.3f} | deadF1 {:.3f}".format(
                          "", self.val_poss, *val, vr["team_acc_both_alive"],
                          vr["alive_dead_f1"]), flush=True)
            t = out["taxonomy"]
            print("{:<34} ext errors per 100 true possessions: split {:.1f} merge {:.1f} drift "
                  "{:.1f} missed {:.1f} spurious {:.1f} | pred/true {:.2f}".format(
                      "", *(100 * t[k] / t["n_gt"] for k in ("split", "merge", "drift", "missed",
                                                             "spurious")),
                      t["n_pred"] / t["n_gt"]), flush=True)
        return out

    def screen(self, name, featfn, dec_params=None, model_params=None, key=None):
        """Fast screen (ROADMAP rule 7): 9 base matches, no augmentation, 300 trees. It only
        drops clear losers; adoption needs the full fit."""
        key = "screen_" + (key or featfn.__name__)
        P = self.probs(featfn, dict(SCREEN, **(model_params or {})), key=key)
        return self.score(name, P, dec_params, use_rules=False, t0=time.time())

    def evaluate(self, name, featfn, model_params=None, dec_params=None, use_rules=True,
                 extra=None, verbose=True, boundary=False):
        t0 = time.time()
        P = self.probs(featfn, model_params, extra, boundary)
        return self.score(name, P, dec_params, use_rules, verbose, t0)

    @staticmethod
    def _score(c, P, p, use_rules, tax=None, B=None):
        R = c.R if use_rules else np.full(c.m.n, "X", dtype="<U1")
        labels = decode.decode(c.m, R, P, c.f["seen"], p, c.f, B)
        codes = pm.label_codes(labels)
        r = pm.evaluate(c.gt, codes)
        if tax is not None:
            for k, v in taxonomy(c.gt, codes).items():
                tax[k] = tax.get(k, 0) + v
        return r["spell_f1_mean"], r["macro_f1"]


def load_val(folder=None):
    """Code array over the whole match from every work/labels/*.csv (X elsewhere), or None."""
    import glob
    folder = folder or os.path.join(os.path.dirname(CACHE_DIR), "work", "labels")
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*.csv"))
                   if not os.path.basename(p).startswith("ball_probe"))
    if not paths:
        return None
    lab = None
    for path in paths:
        x = pm.read_label_csv(path)["label"].astype(str).str.strip().str.upper().values
        lab = x.copy() if lab is None else np.where(x != "X", x, lab)
    return pm.label_codes(lab)


def shape_only(m, f, Xs):
    return Xs
