"""W8: validate and train on public external tracking (IDSSE) degraded to look like our ATD.

    uv run python analysis/external_eval.py

Sections
  1. label conventions: IDSSE possessions/min and durations vs our public window (3.4/min, 14 s)
  2. current pipeline on every degraded IDSSE match, scored against its TRUE labels
     (shape model trained on our own match's pseudo-labels, camera columns dropped)
  3. shape model trained on degraded IDSSE with TRUE labels, leave-one-match-out
  4. that IDSSE-trained model applied to OUR match: public team accuracy, public + masked spellF1
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match, CACHE_DIR  # noqa: E402
from mtah.external import ExtMatch, available  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.noise import degrade, seed_for  # noqa: E402
from mtah import rules, decode, shape, shape_model as sm  # noqa: E402
from mtah.score import pm, load_gt, score  # noqa: E402
from mtah.masking import masked_scores, W0, W1  # noqa: E402
from mtah.report import spells  # noqa: E402

CLASSES = sm.CLASSES


def spellf1(labels, truth):
    gt = pm.label_codes(truth)
    r = pm.evaluate(gt, pm.label_codes(labels))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def fit(X, y_lab, mask, it=300):
    from sklearn.ensemble import HistGradientBoostingClassifier
    idx = np.flatnonzero(mask)[::sm.STEP]
    y = np.searchsorted(np.array(["A", "D", "H"]), y_lab[idx])  # A,D,H order
    clf = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=100, l2_regularization=1.0,
                                         early_stopping=False, random_state=0).fit(X[idx], y)
    return lambda Xq: clf.predict_proba(Xq)[:, [2, 0, 1]]  # -> H, A, D


ids = available()
print("external matches parsed:", ids)
ext = {}
for i, mid in enumerate(ids):
    em = ExtMatch(mid)
    noisy = degrade(em, seed=seed_for(mid))
    f = compute(noisy)
    Xs, _ = shape.compute(noisy, cam=False)
    ext[mid] = (em, noisy, f, Xs)

print("\n== 1. label conventions (true labels)")
gtpub = np.array(pm.LABELS)[load_gt()]
pub_sp = spells(gtpub, [(W0, W1)])
print("  ours public: {:.2f} poss/min, mean {:.1f} s, D {:.0%}".format(
    len(pub_sp) / 5, pub_sp.mean() / 25, np.mean(gtpub[W0:W1] == "D")))
for mid, (em, noisy, f, _) in ext.items():
    runs = [(a, b + 1) for _, a, b in em.halves]
    sp = spells(em.labels, runs)
    g = em.graded_mask()
    print("  {}: {:.2f} poss/min, mean {:.1f} s, <1s {:.0%}, D {:.0%}, ball missing after "
          "noise {:.0%}".format(mid, len(sp) / (g.sum() / 1500), sp.mean() / 25,
                                np.mean(sp < 25), np.mean(em.labels[g] == "D"),
                                1 - np.mean(f["seen_raw"][g])))

print("\n== 2. current pipeline on degraded IDSSE vs true labels")
m = Match(); fm = compute(m)
Rm = rules.run(m, fm)
Xm_nocam, _ = shape.compute(m, cam=False)
tm = m.graded_mask() & sm.confident(Rm, fm["seen"])
tm[W0 - sm.BUFFER:W1 + sm.BUFFER] = False
own_model = fit(Xm_nocam, Rm, tm)
rows = []
for mid, (em, noisy, f, Xs) in ext.items():
    R = rules.run(noisy, f)
    Rclean = rules.run(em, compute(em))
    F = decode.decode(noisy, R, own_model(Xs), f["seen"])
    rows.append((spellf1(Rclean, em.labels), spellf1(R, em.labels), spellf1(F, em.labels)))
    print("  {}  rules(clean) {:.3f}/{:.3f}  rules(noisy) {:.3f}/{:.3f} poss {}/{}  fused(noisy) "
          "{:.3f}/{:.3f} poss {}/{}".format(mid, *rows[-1][0][:2], *rows[-1][1], *rows[-1][2]))
r = np.array([[x[0][0], x[1][0], x[2][0]] for x in rows])
print("  MEAN spellF1  rules(clean) {:.3f}  rules(noisy) {:.3f}  fused(noisy) {:.3f}".format(*r.mean(0)))

print("\n== 3. shape model trained on IDSSE TRUE labels, leave-one-match-out")
for mid, (em, noisy, f, Xs) in ext.items():
    others = [k for k in ext if k != mid]
    X = np.vstack([ext[k][3] for k in others])
    y = np.concatenate([ext[k][0].labels for k in others])
    msk = np.concatenate([ext[k][0].graded_mask() for k in others])
    model = fit(X, y, msk)
    P = model(Xs)
    g = em.graded_mask()
    acc = np.mean(CLASSES[P[g].argmax(1)] == em.labels[g])
    alive = g & np.isin(em.labels, ("H", "A"))
    tacc = np.mean(np.where(P[alive, 0] > P[alive, 1], "H", "A") == em.labels[alive])
    own = own_model(Xs)
    acc_own = np.mean(CLASSES[own[g].argmax(1)] == em.labels[g])
    F = decode.decode(noisy, rules.run(noisy, f), P, f["seen"])
    print("  {}  3-class acc {:.3f} (own-pseudo model {:.3f})  team acc alive {:.3f}  fused "
          "spellF1 {:.3f}/{:.3f}".format(mid, acc, acc_own, tacc, *spellf1(F, em.labels)[:2]))

print("\n== 4. IDSSE-trained shape model applied to OUR match")
X = np.vstack([v[3] for v in ext.values()])
y = np.concatenate([v[0].labels for v in ext.values()])
msk = np.concatenate([v[0].graded_mask() for v in ext.values()])
ext_model = fit(X, y, msk)
for name, model in (("own pseudo-labels", own_model), ("IDSSE true labels", ext_model)):
    P = model(Xm_nocam)
    w = slice(W0, W1)
    g = gtpub[w]
    alive = np.isin(g, ("H", "A"))
    tacc = np.mean(np.where(P[w][alive, 0] > P[w][alive, 1], "H", "A") == g[alive])
    meth = lambda m, f2, P=P: decode.decode(m, rules.run(m, f2), P, f2["seen"])  # noqa: E731
    pub, _ = score(meth(m, fm), load_gt())
    mean, per = masked_scores(meth, m, compute, seeds=range(16))
    print("  {:<18} public team acc {:.3f}  3-class {:.3f}  public spellF1 {:.3f}  masked "
          "{:.3f}/{:.3f}".format(name, tacc, np.mean(CLASSES[P[w].argmax(1)] == g),
                                 pub["spellF1"], mean[0], mean[1]))
