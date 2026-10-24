"""ROADMAP R3: hand-labelled 2nd-half validation & the 'less hidden ball scores lower' analysis.

    uv run python analysis/r3_eval.py            # fast public diagnosis + score work/labels/*.csv
    uv run python analysis/r3_eval.py --sweep    # full 0%/10%/25%/45%/100% ball-hiding sweep

1. Settles the 'less hidden ball scores lower' question (ROADMAP R2/R3, INVESTIGATION.md §12, §16)
   on the public window by decomposing spellF1 into recall (TP / N_gt) and precision (TP / N_pred),
   and diagnosing the spurious spells triggered when the ATD ball is tracked moving inside
   ground-truth stoppages (D).
2. Prints the 4 stratified 2nd-half 5-minute validation windows (--window 1..4 in mtah/label_tool.py).
3. When work/labels/*.csv files exist (created by a person via mtah/label_tool.py), scores every
   window and their union (val_h2) across rules_v2, final_v1 (R0), final_v2 (R1), R7 (dwell),
   R8 (velocity pitch control) and the delivered submission CSV, reporting both exact metrics and
   eye-label timing-tolerant metrics (teamAcc on agreed live frames, deadF1, +-1 s collar macroF1,
   and relaxed-IoU spellF1), plus the ball-hiding test (0% vs 45% vs 100% extra hiding) on val_h2.
4. Summarises work/labels/ball_probe_300.csv when present.

Aggregates only: no frame-level labels or coordinates are printed.
"""
import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluation import possession_metrics as pm  # noqa: E402
from mtah import decode, feats, final, rules, shape  # noqa: E402
from mtah.bench import MASK_SEEDS, load_gt, load_val, taxonomy  # noqa: E402
from mtah.data import CACHE_DIR, Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.label_tool import LABEL_DIR, PROBE_PATH, WINDOWS  # noqa: E402
from mtah.masking import W0, W1, ball_mask, pace_weights  # noqa: E402

warnings.filterwarnings("ignore")
MODELS_V1 = os.path.join(CACHE_DIR, "models_final_v1.joblib")
MODELS_R4 = os.path.join(CACHE_DIR, "models_final_v2_student.joblib")
MODELS_R8 = os.path.join(CACHE_DIR, "models_full_plus4.joblib")
P_R1_PATH = os.path.join(CACHE_DIR, "P_R1.npz")
P_R2_PATH = os.path.join(CACHE_DIR, "P_R2.npz")


def collar_mask(gt_codes, radius=25):
    """True on graded frames at least `radius` frames (1.0 s at 25 fps) away from any GT transition."""
    graded = gt_codes != pm.X
    ch = np.zeros(len(gt_codes), dtype=bool)
    ch[1:] = graded[1:] & graded[:-1] & (gt_codes[1:] != gt_codes[:-1])
    near = np.convolve(ch.astype(float), np.ones(2 * radius + 1), mode="same") > 0
    return graded & ~near


def eye_metrics(gt_codes, pred_codes, collar_frames=25):
    """Exact and timing-tolerant metrics for eye-labelled windows (ROADMAP R3)."""
    ev = pm.evaluate(gt_codes, pred_codes, bootstrap_n=0)
    tx = taxonomy(gt_codes, pred_codes)
    tp = float(np.mean([v["n_matched"] for v in ev["spell_f1"].values()]))
    runs = pm.valid_runs(gt_codes)
    low_iou = pm.spell_scores(gt_codes, pred_codes, runs, ious=(0.1, 0.2, 0.3))["mean_f1"]

    cmask = collar_mask(gt_codes, radius=collar_frames)
    if cmask.any():
        c_gt = np.where(cmask, gt_codes, pm.X)
        c_ev = pm.evaluate(c_gt, pred_codes, bootstrap_n=0)
        collar_mf1 = c_ev["macro_f1"]
        collar_tacc = c_ev["team_acc_both_alive"]
    else:
        collar_mf1, collar_tacc = ev["macro_f1"], ev["team_acc_both_alive"]

    return {
        "spellF1": ev["spell_f1_mean"],
        "lowIoU_F1": low_iou,
        "macroF1": ev["macro_f1"],
        "collar_macroF1": collar_mf1,
        "teamAcc": ev["team_acc_both_alive"],
        "collar_teamAcc": collar_tacc,
        "deadF1": ev["alive_dead_f1"],
        "tp": tp,
        "n_gt": ev["n_gt_spells"],
        "n_pred": ev["n_pred_spells"],
        "spurious": tx["spurious"],
        "missed": tx["missed"],
        "drift": tx["drift"],
    }


def diagnose_public_paradox(m, f, gt, full_sweep=False):
    """Aggregate diagnosis of why public (0% hidden, 0.675) scores below masked (45% hidden, 0.695)."""
    print("=== 1. Public Window 'Less Hidden Ball Scores Lower' Diagnosis (INVESTIGATION.md §16) ===")
    no_rules = np.full(m.n, "X", dtype="<U1")
    p = dict(decode.PARAMS, **final.DECODER)
    P_cache = dict(np.load(P_R1_PATH))
    weights = pace_weights(f)

    c0 = pm.label_codes(decode.decode(m, no_rules, P_cache["public"], f["seen"], p, f))
    r0 = eye_metrics(gt, c0)

    m_metrics = []
    d_pub = gt == pm.D
    l_pub = (gt == pm.H) | (gt == pm.A)
    d_seen_masked, l_seen_masked = [], []
    for s in MASK_SEEDS:
        fk = compute(m, hide=ball_mask(m.n, s, frac=0.45, weights=weights))
        ck = pm.label_codes(decode.decode(m, no_rules, P_cache["masked{}".format(s)],
                                          fk["seen"], p, fk))
        m_metrics.append(eye_metrics(gt, ck))
        d_seen_masked.append(float(np.mean(fk["seen"][d_pub])))
        l_seen_masked.append(float(np.mean(fk["seen"][l_pub])))

    def avg(key):
        return float(np.mean([x[key] for x in m_metrics]))

    print("  {:<22} | spellF1 | TP /17 (rec)  | N_pred (prec) | spur | teamAcc | deadF1 | macroF1".format(
        "condition"))
    print("  " + "-" * 92)
    print("  {:<22} |  {:.3f}  | {:5.2f} ({:.3f}) | {:5.2f} ({:.3f}) | {:4.2f} |  {:.3f}  | {:.3f}  |  {:.3f}".format(
        "public (0% extra hide)", r0["spellF1"], r0["tp"], r0["tp"] / r0["n_gt"],
        r0["n_pred"], r0["tp"] / r0["n_pred"], r0["spurious"], r0["teamAcc"], r0["deadF1"], r0["macroF1"]))
    print("  {:<22} |  {:.3f}  | {:5.2f} ({:.3f}) | {:5.2f} ({:.3f}) | {:4.2f} |  {:.3f}  | {:.3f}  |  {:.3f}".format(
        "masked (45% pace-wt)", avg("spellF1"), avg("tp"), avg("tp") / r0["n_gt"],
        avg("n_pred"), avg("tp") / avg("n_pred"), avg("spurious"), avg("teamAcc"), avg("deadF1"), avg("macroF1")))

    if full_sweep and os.path.isfile(MODELS_V1):
        import joblib
        models = joblib.load(MODELS_V1)
        if isinstance(models, dict):
            models = models["models"]
        Xs, _ = shape.compute(m, cam=False)
        for frac in (0.10, 0.25, 1.00):
            seeds = [0] if frac == 1.00 else MASK_SEEDS
            rows = []
            for s in seeds:
                hide = np.ones(m.n, dtype=bool) if frac == 1.00 else ball_mask(
                    m.n, s, frac=frac, weights=weights)
                fk = compute(m, hide=hide)
                Xo = feats.full_plus3(m, fk, Xs)
                Pk = np.mean([clf.predict_proba(Xo) for clf in models], axis=0)
                ck = pm.label_codes(decode.decode(m, no_rules, Pk, fk["seen"], p, fk))
                rows.append(eye_metrics(gt, ck))
            tp_m = float(np.mean([x["tp"] for x in rows]))
            np_m = float(np.mean([x["n_pred"] for x in rows]))
            print("  {:<22} |  {:.3f}  | {:5.2f} ({:.3f}) | {:5.2f} ({:.3f}) | {:4.2f} |  {:.3f}  | {:.3f}  |  {:.3f}".format(
                "masked ({:.0%} hide)".format(frac),
                float(np.mean([x["spellF1"] for x in rows])), tp_m, tp_m / r0["n_gt"],
                np_m, tp_m / np_m, float(np.mean([x["spurious"] for x in rows])),
                float(np.mean([x["teamAcc"] for x in rows])),
                float(np.mean([x["deadF1"] for x in rows])),
                float(np.mean([x["macroF1"] for x in rows]))))

    gt_spells = pm.possession_spells(gt, [(W0, W1)])
    p0_spells = pm.possession_spells(c0, [(W0, W1)])
    spur_mask = np.zeros(m.n, dtype=bool)
    spur_durs = []
    for pa, pb, pk in p0_spells:
        ov = [min(b, pb) - max(a, pa) for a, b, k in gt_spells if k == pk and min(b, pb) > max(a, pa)]
        if not ov:
            spur_mask[pa:pb] = True
            spur_durs.append((pb - pa) / 25.0)
    pace = (f["sh"] + f["sa"]) / 2
    print("\n  Spurious-spell mechanism on public (0% extra hide):")
    print("    * Count: {} spells (durations: {} s, {} frames total)".format(
        len(spur_durs), ", ".join("{:.2f}".format(d) for d in spur_durs), int(spur_mask.sum())))
    print("    * True label during spurious frames: D = {:.1%}, H = {:.1%}, A = {:.1%}".format(
        float(np.mean(gt[spur_mask] == pm.D)),
        float(np.mean(gt[spur_mask] == pm.H)),
        float(np.mean(gt[spur_mask] == pm.A))))
    print("    * Tracking state during spurious frames: ball_seen = {:.1%}, bspeed = {:.2f} m/s, "
          "min(dh,da) = {:.2f} m, player pace = {:.2f} m/s".format(
              float(np.mean(f["seen"][spur_mask])),
              float(np.nanmean(f["bspeed"][spur_mask])),
              float(np.nanmean(np.minimum(f["dh"][spur_mask], f["da"][spur_mask]))),
              float(np.nanmean(pace[spur_mask]))))
    print("    * Dead-ball ball visibility: public D_seen = {:.1%} (Live {:.1%}) -> "
          "masked D_seen = {:.1%} (Live {:.1%})".format(
              float(np.mean(f["seen"][d_pub])), float(np.mean(f["seen"][l_pub])),
              float(np.mean(d_seen_masked)), float(np.mean(l_seen_masked))))


def print_r3_windows(m, f):
    """Print the 4 stratified 2nd-half 5-minute windows and commands for human labelling."""
    print("\n=== 2. R3 Stratified 2nd-Half Validation Windows (5 min = 7,500 frames each) ===")
    no_rules = np.full(m.n, "X", dtype="<U1")
    p = dict(decode.PARAMS, **final.DECODER)
    P_pub = dict(np.load(P_R1_PATH))["public"]
    codes = pm.label_codes(decode.decode(m, no_rules, P_pub, f["seen"], p, f))
    pace = (f["sh"] + f["sa"]) / 2
    for idx, (start, fname) in sorted(WINDOWS.items()):
        end = start + 7500
        path = os.path.join(LABEL_DIR, fname)
        status = "DONE" if os.path.isfile(path) else "TODO"
        spells = pm.possession_spells(codes[start:end], [(0, 7500)])
        print("  window {}: {:6d}..{:6d} ({:5.1f}..{:5.1f} min) | ball_seen {:5.1%} | "
              "pace {:.2f} m/s | ~{:2d} spells | [{}]  uv run python -m mtah.label_tool --window {}".format(
                  idx, start, end, start / 1500.0, end / 1500.0,
                  float(np.mean(f["seen"][start:end])),
                  float(np.nanmean(pace[start:end])),
                  len(spells), status, idx))
    p_status = "DONE" if os.path.isfile(PROBE_PATH) else "TODO"
    print("  probe   : 300 genuine missing-ball frames (stratified)              | [{}]  "
          "uv run python -m mtah.label_tool --probe".format(p_status))


def score_r3_labels(m, f):
    """Score all saved work/labels/*.csv across R-items and run the ball-hiding test on val_h2."""
    paths = sorted(p for p in glob.glob(os.path.join(LABEL_DIR, "*.csv"))
                   if not os.path.basename(p).startswith("ball_probe"))
    if not paths:
        print("\n=== 3. R3 Hand-Labelled Evaluation ===")
        print("  No hand-labelled windows found in work/labels/ yet.")
        print("  Run one of the `uv run python -m mtah.label_tool --window <1..4>` commands above,")
        print("  then re-run `uv run python analysis/r3_eval.py`.")
        return

    print("\n=== 3. R3 Hand-Labelled Evaluation ({} file(s) in work/labels/) ===".format(len(paths)))
    vgt = load_val(LABEL_DIR)
    no_rules = np.full(m.n, "X", dtype="<U1")
    R = rules.run(m, f)
    P_r1 = dict(np.load(P_R1_PATH))["public"]
    p_v1 = {**decode.PARAMS, **final.DECODER, "kickoff": None}
    p_v2 = {**decode.PARAMS, **final.DECODER}
    p_r7 = {**decode.PARAMS, **final.DECODER, "min_dead": 0, "dwell": 25}

    candidates = [
        ("rules_v2", pm.label_codes(R)),
        ("R0 final_v1 (kickoff=None)", pm.label_codes(decode.decode(m, no_rules, P_r1, f["seen"], p_v1, f))),
        ("R1 final_v2 (kickoff='half')", pm.label_codes(decode.decode(m, no_rules, P_r1, f["seen"], p_v2, f))),
    ]
    if os.path.isfile(P_R2_PATH):
        P_r2 = dict(np.load(P_R2_PATH))["public"]
        candidates.append(("R2 50% unmasked copies",
                           pm.label_codes(decode.decode(m, no_rules, P_r2, f["seen"], p_v2, f))))
    Xs = None
    if os.path.isfile(MODELS_R4):
        import joblib
        m_r4 = joblib.load(MODELS_R4)
        if isinstance(m_r4, dict):
            m_r4 = m_r4["models"]
        Xs, _ = shape.compute(m, cam=False)
        Xo = feats.full_plus3(m, f, Xs)
        P_r4 = np.mean([clf.predict_proba(Xo) for clf in m_r4], axis=0)
        candidates.append(("R4 pseudo-label student",
                           pm.label_codes(decode.decode(m, no_rules, P_r4, f["seen"], p_v2, f))))
    det_path = os.path.join(CACHE_DIR, "ball_det_ft.npz")
    if os.path.isfile(det_path) and os.path.isfile(MODELS_V1):
        import joblib
        from analysis import video_link as vlink
        m_v1 = joblib.load(MODELS_V1)
        if isinstance(m_v1, dict):
            m_v1 = m_v1["models"]
        if Xs is None:
            Xs, _ = shape.compute(m, cam=False)
        dd = vlink.dedup(vlink.DET)
        tr_raw = vlink.link(dd, np.zeros((0, 3)), m.n, c0=0.08)
        tr_nospot = tr_raw.copy()
        for sx, sy in ((11.0, 34.0), (94.3, 34.0), (52.65, 34.0)):
            bad = np.hypot(tr_nospot[:, 0] - sx, tr_nospot[:, 1] - sy) < 1.3
            tr_nospot[bad] = np.nan
        for r6_name, tr in (("R6 DP video ball (c0=0.08)", tr_raw),
                            ("R6 DP video ball (no spots)", tr_nospot)):
            m.video_ball = tr
            fv = compute(m)
            Xo_v = feats.full_plus3(m, fv, Xs)
            P_r6 = np.mean([clf.predict_proba(Xo_v) for clf in m_v1], axis=0)
            candidates.append((r6_name,
                               pm.label_codes(decode.decode(m, no_rules, P_r6, fv["seen"], p_v2, fv))))
        m.video_ball = None
    candidates.append(("R7 in-Viterbi dwell=25",
                       pm.label_codes(decode.decode(m, no_rules, P_r1, f["seen"], p_r7, f))))
    if os.path.isfile(MODELS_R8):
        import joblib
        m_r8 = joblib.load(MODELS_R8)
        if isinstance(m_r8, dict):
            m_r8 = m_r8["models"]
        if Xs is None:
            Xs, _ = shape.compute(m, cam=False)
        Xo = feats.full_plus4(m, f, Xs)
        P_r8 = np.mean([clf.predict_proba(Xo) for clf in m_r8], axis=0)
        candidates.append(("R8 velocity pitch control",
                           pm.label_codes(decode.decode(m, no_rules, P_r8, f["seen"], p_v2, f))))
    for sub_name in ("Yamanalyst.csv", os.path.join("work", "submission.csv"),
                     os.path.join("evaluation", "results", "final_v2.csv")):
        sub_path = os.path.join(ROOT, sub_name)
        if os.path.isfile(sub_path):
            sub_lab = pm.read_label_csv(sub_path)["label"].astype(str).str.strip().str.upper().values
            candidates.append(("submission ({})".format(os.path.basename(sub_name)),
                               pm.label_codes(sub_lab)))
            break

    print("  {:<28} | spellF1 | lowIoU | macroF1 | collarMF1 | teamAcc | deadF1 | TP/N_gt | N_pred | spur".format(
        "method (on combined val_h2)"))
    print("  " + "-" * 106)
    for name, codes in candidates:
        r = eye_metrics(vgt, codes)
        print("  {:<28} |  {:.3f}  | {:.3f}  |  {:.3f}  |   {:.3f}   |  {:.3f}  | {:.3f}  | {:4.1f}/{:<2d} |  {:4d}  | {:3d}".format(
            name, r["spellF1"], r["lowIoU_F1"], r["macroF1"], r["collar_macroF1"],
            r["teamAcc"], r["deadF1"], r["tp"], r["n_gt"], r["n_pred"], r["spurious"]))

    # Per-window breakdown for final_v2
    if len(paths) > 1:
        print("\n  Per-window breakdown (R1 final_v2):")
        c_v2 = candidates[2][1]
        for path in paths:
            w_lab = pm.read_label_csv(path)["label"].astype(str).str.strip().str.upper().values
            w_gt = pm.label_codes(w_lab)
            r = eye_metrics(w_gt, c_v2)
            idx_graded = np.flatnonzero(w_gt != pm.X)
            seen_w = float(np.mean(f["seen"][idx_graded])) if len(idx_graded) else 0.0
            print("    {:<24} | ball_seen {:5.1%} | spellF1 {:.3f} | lowIoU {:.3f} | "
                  "teamAcc {:.3f} | deadF1 {:.3f} | collarMF1 {:.3f} | spells {}/{}".format(
                      os.path.basename(path), seen_w, r["spellF1"], r["lowIoU_F1"],
                      r["teamAcc"], r["deadF1"], r["collar_macroF1"], r["n_pred"], r["n_gt"]))

    # Settle the 'less hidden ball scores lower' question on val_h2
    if os.path.isfile(MODELS_V1):
        import joblib
        models = joblib.load(MODELS_V1)
        if isinstance(models, dict):
            models = models["models"]
        Xs, _ = shape.compute(m, cam=False)
        weights = pace_weights(f)
        runs = pm.valid_runs(vgt)
        print("\n  === 4. 'Less Hidden Ball Scores Lower' Test on val_h2 ===")
        print("  {:<24} | ball_seen | spellF1 | lowIoU | TP/{:<2d} | N_pred | spur | teamAcc | deadF1 | collarMF1".format(
            "val_h2 masking level", int(eye_metrics(vgt, candidates[2][1])["n_gt"])))
        print("  " + "-" * 100)
        idx_graded = np.flatnonzero(vgt != pm.X)
        for frac in (0.0, 0.25, 0.45, 1.0):
            seeds = [0] if frac in (0.0, 1.0) else range(8)
            rows, seen_rates = [], []
            for s in seeds:
                if frac == 0.0:
                    fk = f
                elif frac == 1.0:
                    fk = compute(m, hide=np.ones(m.n, dtype=bool))
                else:
                    hide = np.zeros(m.n, dtype=bool)
                    for w0, w1 in runs:
                        hide |= ball_mask(m.n, s + 100 * w0, frac=frac, weights=weights,
                                          window=(w0, w1))
                    fk = compute(m, hide=hide)
                Xo = feats.full_plus3(m, fk, Xs)
                Pk = np.mean([clf.predict_proba(Xo) for clf in models], axis=0)
                ck = pm.label_codes(decode.decode(m, no_rules, Pk, fk["seen"], p_v2, fk))
                rows.append(eye_metrics(vgt, ck))
                seen_rates.append(float(np.mean(fk["seen"][idx_graded])))
            print("  {:<24} |   {:5.1%}   |  {:.3f}  | {:.3f}  | {:5.2f} | {:5.2f}  | {:4.2f} |  {:.3f}  | {:.3f}  |   {:.3f}".format(
                "extra hide {:4.0%}".format(frac),
                float(np.mean(seen_rates)),
                float(np.mean([x["spellF1"] for x in rows])),
                float(np.mean([x["lowIoU_F1"] for x in rows])),
                float(np.mean([x["tp"] for x in rows])),
                float(np.mean([x["n_pred"] for x in rows])),
                float(np.mean([x["spurious"] for x in rows])),
                float(np.mean([x["teamAcc"] for x in rows])),
                float(np.mean([x["deadF1"] for x in rows])),
                float(np.mean([x["collar_macroF1"] for x in rows]))))


def score_probe():
    """Summarise work/labels/ball_probe_300.csv if present."""
    if not os.path.isfile(PROBE_PATH):
        return
    import pandas as pd
    df = pd.read_csv(PROBE_PATH)
    valid = df[df["state"].isin(["V", "D", "O"])]
    if valid.empty:
        return
    print("\n=== 5. Genuine Missing-Ball Frame Probe ({}/{} labelled) ===".format(
        len(valid), len(df)))
    for code, desc in (("V", "Visible live play"), ("D", "Dead ball / stoppage"),
                       ("O", "Occluded / off-camera")):
        print("  {} ({:<22}): {:5.1%} ({})".format(
            code, desc, float(np.mean(valid["state"] == code)), int(np.sum(valid["state"] == code))))


def main():
    full_sweep = "--sweep" in sys.argv
    m = Match()
    f = compute(m)
    gt = load_gt()
    diagnose_public_paradox(m, f, gt, full_sweep=full_sweep)
    print_r3_windows(m, f)
    score_r3_labels(m, f)
    score_probe()


if __name__ == "__main__":
    main()
