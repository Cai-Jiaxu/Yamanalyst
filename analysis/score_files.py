"""Score delivered submission files on the public GT and the eye-labelled 2nd-half windows.

    uv run python analysis/score_files.py evaluation/results/final_v4.csv evaluation/results/final_v5.csv

val = official spellF1 on val_h2 (rewards rule timing, INVESTIGATION.md section 20); valLo =
spellF1 at IoU 0.1-0.3; valMid = team error > 3 s from a labelled change. Aggregates only.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.yardsticks import mid_team_err, val_windows  # noqa: E402
from mtah.score import load_gt, pm  # noqa: E402
from games import half_video_blocks  # noqa: E402

if __name__ == "__main__":
    gt = load_gt()
    vals = val_windows(len(gt))
    for path in sys.argv[1:]:
        lab = pm.label_codes(pm.read_label_csv(path))
        r = pm.evaluate(gt, lab, bootstrap_n=0)
        v, lo, mid = [], [], []
        for g in vals:
            v.append(pm.evaluate(g, lab, bootstrap_n=0)["spell_f1_mean"])
            lo.append(pm.spell_scores(g, lab, pm.valid_runs(g), ious=(0.1, 0.2, 0.3))["mean_f1"])
            mid.append(mid_team_err(g, lab))
        runs = [(a, b + 1) for _, a, b in half_video_blocks("BAR-ATH")]
        poss = pm.possession_spells(pm.fill_abstentions(lab, runs), runs)
        print("{:<28} public {:.3f}/{:.3f} n{} | val {:.3f} valLo {:.3f} valMid {:.3f} | "
              "full-match spells {}".format(os.path.basename(path), r["spell_f1_mean"],
                                            r["macro_f1"], r["n_pred_spells"], np.mean(v),
                                            np.mean(lo), np.mean(mid), len(poss)))
