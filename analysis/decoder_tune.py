"""Tune the decoder's switching costs on external TRUE labels (leave-one-match-out models),
reported per source next to our own public and masked scores.

    uv run python analysis/decoder_tune.py

Metrica's event-derived labels are closest to our convention (3.4-3.7 possessions/min vs our
3.4); IDSSE's frame flags switch more often (4.3/min). Pick a cell that is good on all three
yardsticks, from the middle of a plateau.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402
from mtah.external import ExtMatch  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.noise import degrade, seed_for  # noqa: E402
from mtah import external, rules, decode, shape, shape_model as sm  # noqa: E402
from mtah.masking import masked_scores  # noqa: E402
from mtah.score import pm, load_gt, score  # noqa: E402

GRID = [(4, 3), (8, 3), (8, 6), (12, 6), (16, 6), (16, 12), (24, 12)]

data = sm.external_set()
names = [d[0] for d in data]
cases = []
for i, name in enumerate(external.available()):
    em = ExtMatch(name)
    noisy = degrade(em, seed=seed_for(name))
    fn = compute(noisy)
    others = [k for k in names if k != name]
    P = sm.fit_external(others, offsets=[0])(dict((d[0], d[1]) for d in data)[name])
    cases.append((name, noisy, rules.run(noisy, fn), P, fn["seen"], pm.label_codes(em.labels)))
    print("prepared", name, flush=True)

m = Match(); f = compute(m)
Xnc, _ = shape.compute(m, cam=False)
Pm = sm.fit_external(offsets=[0])(Xnc)
gt = load_gt()

print("\ncost_team cost_dead | idsse spellF1 | metrica spellF1 | ours public | ours masked")
for ct, cd in GRID:
    p = dict(decode.PARAMS, cost_team=ct, cost_dead=cd)
    res = {"idsse": [], "metrica": []}
    for name, noisy, R, P, seen, g in cases:
        lab = decode.decode(noisy, R, P, seen, p)
        res[name.split("_")[0]].append(pm.evaluate(g, pm.label_codes(lab))["spell_f1_mean"])
    meth = lambda m, f2, p=p: decode.decode(m, rules.run(m, f2), Pm, f2["seen"], p)  # noqa: E731
    pub, _ = score(meth(m, f), gt)
    mean, _ = masked_scores(meth, m, compute, seeds=range(16))
    print("{:>9} {:>9} | {:.3f} (n={}) | {:.3f} (n={}) | {:.3f} | {:.3f}".format(
        ct, cd, np.mean(res["idsse"]), len(res["idsse"]), np.mean(res["metrica"]),
        len(res["metrica"]), pub["spellF1"], mean[0]), flush=True)
