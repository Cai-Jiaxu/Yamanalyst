"""ROADMAP R20: stoppage-ball gate v2, post-decode on cached fold probabilities (cache/P_R1.npz).

    uv run python analysis/r20_gate.py                       # the six configs, every yardstick
    uv run python analysis/r20_gate.py --diagnose            # what the remaining spurious spells look like
    uv run python analysis/r20_gate.py --sensitivity         # length x pace grid for the one-sided gate
    uv run python analysis/r20_gate.py --min-poss            # short-possession absorption on top of G2
    uv run python analysis/r20_gate.py --write g2_check      # final_v5.csv + G2; same bytes as final_v6

The shipped gate (decode.stop_gate, final_v5) turns a live spell of <= 100 frames with dead ball on
BOTH sides into dead ball when the players walk (< 1.3 m/s). R20 asked whether a gate that also
uses "no player within 5 m of the ball" removes more spurious stoppage spells. Six configs were
fixed before scoring and all are reported (ROADMAP rule 2). Verdict in INVESTIGATION.md section 27:
G1 (R20 as specified) is null; G2 (one side) became final_v6 (stop_gate_sides=1 in mtah/final.py).

Low memory on purpose (< 1 GB): unlike mtah.bench.Bench it keeps no feature matrices, only each
case's pre-gate labels and the few physical arrays a gate needs, so it also runs on a small VM.
It first checks that it reproduces baseline_snapshot/baseline_metrics.json exactly.
Aggregates only: nothing frame-level is printed.
"""
import argparse
import gc
import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import decode, external, final, noise  # noqa: E402
from mtah.bench import BENCH_VERSION, MASK_SEEDS, load_val, taxonomy  # noqa: E402
from mtah.data import CACHE_DIR, Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.masking import ball_mask, pace_weights  # noqa: E402
from mtah.score import load_gt, pm  # noqa: E402

warnings.filterwarnings("ignore")
KEEP = ("sh", "sa", "dh", "da", "seen")
MAX_LEN, SLOW = final.DECODER["stop_gate"], 1.3

# name -> kwargs of gate(); fixed before any of them was scored
CONFIGS = [
    ("G1 R20 as specified: 1 side, <=250, far 5 m", dict(max_len=250, sides=1, far=5.0)),
    ("G2 1 side, <=100, pace < 1.3", dict(sides=1)),
    ("G3 1 side, <=100, pace < 1.0", dict(sides=1, slow=1.0)),
    ("G4 2 sides, <=250, pace < 1.3", dict(max_len=250)),
    ("G5 2 sides, <=100, pace < 1.6", dict(slow=1.6)),
    ("G6 2 sides, <=100, far 5 m, any pace", dict(slow=99.0, far=5.0)),
]


def build_cases():
    """{name: dict(pre, f, gt)}: labels of the final_v5 decoder with the gate switched off."""
    p0 = dict(decode.PARAMS, **final.DECODER)
    p0["stop_gate"] = 0
    z = np.load(os.path.join(CACHE_DIR, "P_R1.npz"))
    cases = {}

    def add(name, m, f, gt):
        pre = decode.decode(m, np.full(m.n, "X", dtype="<U1"), z[name], f["seen"], p0, f)
        cases[name] = dict(pre=pre, f={k: np.asarray(f[k]) for k in KEEP}, gt=gt)

    for name in external.available():
        em = external.ExtMatch(name)
        noisy = noise.degrade(em, seed=noise.seed_for(name))
        f = np.load(os.path.join(CACHE_DIR, "bench_{}_{}.npz".format(name, BENCH_VERSION)),
                    allow_pickle=True)["f"].item()
        add(name, noisy, f, pm.label_codes(em.labels))
        del em, noisy, f
        gc.collect()
    m = Match()
    gt = load_gt()
    f = compute(m)
    add("public", m, f, gt)
    weights = pace_weights(f)
    for s in MASK_SEEDS:
        add("masked{}".format(s), m, compute(m, hide=ball_mask(m.n, s, weights=weights)), gt)
    return cases, load_val()


def v1(c):
    return decode.stop_gate(c["pre"], c["f"], MAX_LEN)


def gate(c, max_len=MAX_LEN, slow=SLOW, sides=2, far=None, labels=None):
    """The shipped gate, then one more pass with the given settings. far: also require that on
    more than half of the spell's ball-seen frames no player is within `far` metres of the ball."""
    out = v1(c) if labels is None else labels
    if far is None:
        return decode.stop_gate(out, c["f"], max_len, slow, sides)
    f = c["f"]
    md = np.fmin(f["dh"], f["da"])
    cand = decode.stop_gate(out, f, max_len, slow, sides)
    res = out.copy()
    ch = np.diff(np.concatenate(([0], (cand != out).astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(ch == 1), np.flatnonzero(ch == -1)):
        s = f["seen"][a:b]
        if s.any() and np.mean(md[a:b][s] > far) > 0.5:
            res[a:b] = "D"
    return res


def score(cases, val, fn):
    ext = {"idsse": [], "metrica": []}
    tax, mtax = {}, {}
    for k, c in cases.items():
        if k == "public" or k.startswith("masked"):
            continue
        codes = pm.label_codes(fn(c))
        ext[k.split("_")[0]].append(pm.evaluate(c["gt"], codes, bootstrap_n=0)["spell_f1_mean"])
        for a, b in taxonomy(c["gt"], codes).items():
            tax[a] = tax.get(a, 0) + b
    masked = []
    for k, c in cases.items():
        if k.startswith("masked"):
            codes = pm.label_codes(fn(c))
            masked.append(pm.evaluate(c["gt"], codes, bootstrap_n=0)["spell_f1_mean"])
            for a, b in taxonomy(c["gt"], codes).items():
                mtax[a] = mtax.get(a, 0) + b
    pub_codes = pm.label_codes(fn(cases["public"]))
    pub = pm.evaluate(cases["public"]["gt"], pub_codes, bootstrap_n=0)
    e = ext["idsse"] + ext["metrica"]
    r = dict(bench=(np.mean(e) + np.mean(masked)) / 2, ext=np.mean(e), idsse=np.mean(ext["idsse"]),
             metrica=np.mean(ext["metrica"]), masked=np.mean(masked), masked_sd=np.std(masked),
             public=pub["spell_f1_mean"], public_macro=pub["macro_f1"],
             ext_spur=100 * tax["spurious"] / tax["n_gt"], ext_missed=100 * tax["missed"] / tax["n_gt"],
             masked_spur=mtax["spurious"] / len(masked),
             public_spur=taxonomy(cases["public"]["gt"], pub_codes)["spurious"],
             val=np.nan, val_spur=-1, val_pred=-1)
    if val is not None:
        r["val"] = pm.evaluate(val, pub_codes, bootstrap_n=0)["spell_f1_mean"]
        t = taxonomy(val, pub_codes)
        r["val_spur"], r["val_pred"] = t["spurious"], t["n_pred"]
    return r


def show(name, r, base=None):
    print("{:<40} bench {bench:.4f} | ext {ext:.4f} (idsse {idsse:.4f} metrica {metrica:.4f}) | "
          "masked {masked:.4f} | public {public:.4f}/{public_macro:.4f} | val {val:.4f}".format(name, **r))
    print("{:<40} spurious: ext {ext_spur:.2f}/100 masked {masked_spur:.2f}/seed public {public_spur}"
          " val {val_spur} of {val_pred} | ext missed {ext_missed:.2f}/100".format("", **r))
    if base is not None:
        print("{:<40} vs final_v5: bench {:+.4f} ext {:+.4f} masked {:+.4f} public {:+.4f} val {:+.4f}"
              .format("", *(r[k] - base[k] for k in ("bench", "ext", "masked", "public", "val"))),
              flush=True)


def delivered(cases, val, write=None):
    """The one-sided gate (G2) on the delivered full-fit file: scores and how much changes."""
    import pandas as pd
    path = os.path.join(ROOT, "evaluation", "results", "final_v5.csv")
    if not os.path.isfile(path):
        print("no evaluation/results/final_v5.csv here: delivered-file check skipped")
        return
    lab = pd.read_csv(path)["label"].values.astype("<U1")
    c = cases["public"]
    new = gate(c, sides=1, labels=lab)
    for name, x in (("final_v5 delivered", lab), ("final_v5 + G2", new)):
        codes = pm.label_codes(x)
        p = pm.evaluate(c["gt"], codes, bootstrap_n=0)
        s = "{:<22} public {:.6f}/{:.6f} spurious {}".format(
            name, p["spell_f1_mean"], p["macro_f1"], taxonomy(c["gt"], codes)["spurious"])
        if val is not None:
            v = pm.evaluate(val, codes, bootstrap_n=0)
            t = taxonomy(val, codes)
            s += " | val {:.4f}/{:.4f} deadF1 {:.3f} teamAcc {:.3f} spurious {} of {}".format(
                v["spell_f1_mean"], v["macro_f1"], v["alive_dead_f1"], v["team_acc_both_alive"],
                t["spurious"], t["n_pred"])
        print(s + " | live spells, full match {}".format(
            int(np.sum(np.isin(x[1:], ["H", "A"]) & (x[1:] != x[:-1])))))
    ch = new != lab
    starts = np.flatnonzero(np.diff(np.concatenate(([0], ch.astype(np.int8)))) == 1)
    public = (starts >= 11185) & (starts < 18685)
    in_val = np.zeros(len(starts), dtype=bool) if val is None else val[starts] != pm.LABELS.index("X")
    print("spells relabelled on the full match: {} ({} frames); {} in the public window, {} in the "
          "val windows, {} unlabelled".format(len(starts), int(ch.sum()), int(public.sum()),
                                              int(in_val.sum()), int((~public & ~in_val).sum())))
    if write:
        from run import write as write_csv
        import hashlib
        out = os.path.join(ROOT, "evaluation", "results", write + ".csv")
        assert "final_v5" not in write, "never overwrite the submission"
        write_csv(out, new)
        with open(out, "rb") as fh:
            print("wrote {}  sha256 {}".format(os.path.relpath(out, ROOT),
                                               hashlib.sha256(fh.read()).hexdigest()))


def diagnose(cases, val):
    """Counts of predicted live spells after the shipped gate, spurious (no same-team overlap with
    a true possession) vs the rest, and how many each candidate condition would relabel."""
    from scipy.ndimage import uniform_filter1d
    names = np.array(pm.LABELS)

    def rows(c, gt):
        f, lab = c["f"], v1(c)
        pace = uniform_filter1d(np.nan_to_num((f["sh"] + f["sa"]) / 2), 25)
        md = np.fmin(f["dh"], f["da"])
        runs = pm.valid_runs(gt)
        gs = pm.possession_spells(gt, runs)
        ps = pm.possession_spells(pm.fill_abstentions(pm.label_codes(lab), runs), runs)
        out = []
        for a, b, k in ps:
            seen = f["seen"][a:b]
            out.append(dict(
                spur=not any(ga < b and gb > a and gk == k for ga, gb, gk in gs), n=b - a,
                pace=pace[a:b].mean(), dead=np.mean(names[gt[a:b]] == "D") >= 0.5,
                far=bool(seen.any()) and np.mean(md[a:b][seen] > 5.0) > 0.5,
                nd=int(a > 0 and lab[a - 1] == "D") + int(b < len(lab) and lab[b] == "D")))
        return out

    groups = {"ext": sum((rows(c, c["gt"]) for k, c in cases.items()
                          if k != "public" and not k.startswith("masked")), []),
              "masked (16 seeds)": sum((rows(c, c["gt"]) for k, c in cases.items()
                                        if k.startswith("masked")), []),
              "public": rows(cases["public"], cases["public"]["gt"])}
    if val is not None:
        groups["val_h2"] = rows(cases["public"], val)
    conds = [("2 sides, <=100, pace<1.3 (shipped, left over)", lambda r: r["nd"] == 2 and r["n"] <= 100 and r["pace"] < 1.3),
             ("1+ side, <=250, pace<1.3, far 5 m (R20)", lambda r: r["nd"] >= 1 and r["n"] <= 250 and r["pace"] < 1.3 and r["far"]),
             ("1+ side, <=100, pace<1.3 (G2)", lambda r: r["nd"] >= 1 and r["n"] <= 100 and r["pace"] < 1.3)]
    print("\npredicted live spells after the shipped gate; each cell = spurious inside true dead "
          "ball / other spurious / not spurious")
    for g, R in groups.items():
        def cell(fn):
            return "{}/{}/{}".format(sum(fn(r) for r in R if r["spur"] and r["dead"]),
                                     sum(fn(r) for r in R if r["spur"] and not r["dead"]),
                                     sum(fn(r) for r in R if not r["spur"]))
        print("{:<18} all {:<12} <=100 frames {:<12}".format(g, cell(lambda r: True), cell(lambda r: r["n"] <= 100))
              + " | ".join("{}: {}".format(n, cell(fn)) for n, fn in conds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--min-poss", action="store_true", help="min_poss 25/50/75 on top of G2")
    ap.add_argument("--write", default=None, help="write final_v5 + G2 under this name")
    args = ap.parse_args()
    cases, val = build_cases()
    base = score(cases, val, v1)
    with open(os.path.join(ROOT, "baseline_snapshot", "baseline_metrics.json")) as fh:
        ref = json.load(fh)
    for k, rk in (("bench", "bench_spellF1"), ("ext", "ext_spellF1"), ("masked", "masked_spellF1"),
                  ("public", "public_spellF1"), ("public_macro", "public_macroF1")):
        assert abs(base[k] - ref[rk]) < 1e-9, (k, base[k], ref[rk])
    print("baseline reproduced: matches baseline_snapshot/baseline_metrics.json to 1e-9")
    show("final_v5 (shipped gate)", base)
    show("no gate", score(cases, val, lambda c: c["pre"]), base)
    for name, kw in CONFIGS:
        show(name, score(cases, val, lambda c: gate(c, **kw)), base)
    # the decoder switch gives the same labels as G2
    p = dict(decode.PARAMS, **final.DECODER)
    for c in cases.values():
        both = decode.stop_gate(decode.stop_gate(c["pre"], c["f"], p["stop_gate"]), c["f"],
                                p["stop_gate"], sides=1)
        assert np.array_equal(both, gate(c, sides=1))
    if args.min_poss:
        g2 = score(cases, val, lambda c: gate(c, sides=1))
        for n in (25, 50, 75):
            show("G2 + min_poss {} (delta vs G2)".format(n), score(
                cases, val, lambda c: decode.min_duration(gate(c, sides=1), 0, n)), g2)
    if args.diagnose:
        diagnose(cases, val)
    if args.sensitivity:
        print("\none-sided gate, length x pace (sensitivity only, not a selection):")
        for ml in (50, 75, 100, 125, 150):
            for slow in (1.1, 1.2, 1.3, 1.4, 1.5):
                r = score(cases, val, lambda c: gate(c, max_len=ml, slow=slow, sides=1))
                print("<= {:>3} frames, pace < {:.1f}: bench {:+.4f} ext {:+.4f} masked {:+.4f} "
                      "public {:+.4f} val {:+.4f}".format(ml, slow, *(r[k] - base[k] for k in (
                          "bench", "ext", "masked", "public", "val"))), flush=True)
    print()
    delivered(cases, val, args.write)


if __name__ == "__main__":
    main()
