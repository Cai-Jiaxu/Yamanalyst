"""Clean no-filter production control: capture P from final.predict, verify baseline.

Monkeypatches decode.decode at runtime to capture P; no module mutation.
Saves cached P for deterministic re-decode. Scores no-filter output vs v5 baseline.
"""
import os, sys, time, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, ROOT)

from analysis.yardsticks import Yard
from mtah.score import pm
from mtah import final, decode

t0 = time.time()
y = Yard()
c = y.b.public
m, f = c.m, c.f

labels, P = final.predict(m, f, None)
print("captured P shape", P.shape, "col means", P.mean(0), flush=True)
print("decoded labels unique", dict(zip(*np.unique(labels, return_counts=True))), flush=True)

# save P IMMEDIATELY after fit (most expensive step); score after
out = os.path.join(ROOT, "cache/ar_followup_production_P.npz")
np.savez_compressed(out, P=P, labels=labels)
print("saved P to", out, flush=True)

# score no-filter
codes = pm.label_codes(labels)
res = pm.evaluate(y.b.gt, codes, bootstrap_n=0)
print("NO-FILTER production spellF1", round(res['spell_f1_mean'], 4),
      "macro", round(res['macro_f1'], 4), flush=True)
print("elapsed", time.time() - t0, flush=True)
