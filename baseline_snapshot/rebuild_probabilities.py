"""Rebuild cache/P_R1.npz from the shipped recipe, without an original-machine cache.

    uv run python baseline_snapshot/rebuild_probabilities.py --check
    uv run python baseline_snapshot/rebuild_probabilities.py

Fits three match-held-out models with mirrors and extra noise draws. This is NOT
the full-fit submission model. Only aggregate metrics are printed. Existing output
is protected unless --overwrite is supplied. Run only one refit per machine.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "cache" / "P_R1.npz"
REFERENCE = ROOT / "baseline_snapshot" / "baseline_metrics.json"


def prerequisites():
    """Check paths without loading match data or importing ML dependencies."""
    from games import GAME, asset_paths

    missing = []
    if not (ROOT / "cache" / (GAME + "_tactical.npz")).is_file():
        paths = asset_paths()
        missing.extend(paths[k] for k in
                       ("atd_meta", "atd_raw", "homography", "smart_tagging")
                       if not Path(paths[k]).is_file())
    # Read the fold names without importing Bench (which imports sklearn).
    import ast
    tree = ast.parse((ROOT / "mtah" / "bench.py").read_text(encoding="utf-8"))
    folds = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "FOLDS" for t in node.targets))
    missing.extend(str(ROOT / "cache" / ("ext_" + name + ".npz"))
                   for fold in folds for name in fold
                   if not (ROOT / "cache" / ("ext_" + name + ".npz")).is_file())
    return missing


def compare_metrics(result, reference, tolerance=0.0001):
    actual = {"bench_spellF1": float(result["bench"]),
              "ext_spellF1": float(result["ext"][0]),
              "masked_spellF1": float(result["masked"][0]),
              "public_spellF1": float(result["public"][0]),
              "public_macroF1": float(result["public"][1])}
    comparison = {k: {"actual": v, "reference": reference[k],
                      "delta": v - reference[k]} for k, v in actual.items()}
    return comparison, all(abs(v["delta"]) <= tolerance for v in comparison.values())


def validate_probabilities(probabilities, bench):
    import numpy as np

    cases = list(bench.ext.values()) + [bench.public] + bench.masked
    if set(probabilities) != {c.name for c in cases}:
        raise ValueError("Probability cache does not contain exactly the benchmark cases")
    for case in cases:
        p = probabilities[case.name]
        if (p.shape != (case.m.n, 3) or not np.isfinite(p).all()
                or (p < 0).any() or (p > 1).any()
                or not np.allclose(p.sum(axis=1), 1.0)):
            raise ValueError("Invalid probability array for " + case.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check input paths without training")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output cache")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if output.exists() and not args.overwrite and not args.check:
        parser.error("output already exists; use a different --output or explicitly --overwrite")
    missing = prerequisites()
    if missing:
        print("Missing prerequisites:\n  " + "\n  ".join(missing))
        print("Install with uv sync; unpack tactical assets (or set MTAH_DATA_DIR); then run\n"
              "  bash scripts/download_external.sh\n  uv run python -m mtah.external")
        return 1
    if args.check:
        print("Input paths ready. Run with uv run and no --check to fit the pinned recipe.")
        return 0

    import numpy as np
    from mtah import feats, final
    from mtah.bench import Bench

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if (final.FEATFN != reference["featfn"] or final.MODEL != reference["model"]
            or final.SEED != reference["seed"] or final.DECODER_V5 != reference["decoder"]):
        parser.error("shipped configuration differs from the locked baseline; review before rebuilding")
    started = time.time()
    print("Rebuilding baseline fold probabilities; no existing fold-model cache will be reused.",
          flush=True)
    bench = Bench()
    featfn = getattr(feats, final.FEATFN)
    mirrors = feats.mirrored(featfn)
    augmented = feats.augmented(featfn)
    extra = {name: mirrors.get(name, []) + augmented.get(name, []) for name in bench.ext}
    # key=None deliberately forces fresh folds, avoiding unrelated/stale joblib models.
    probabilities = bench.probs(featfn, final.MODEL, extra=extra, key=None)
    validate_probabilities(probabilities, bench)
    result = bench.score("P_R1 rebuild", probabilities, dec_params=final.DECODER_V5,
                         use_rules=False, t0=started)
    comparison, matches = compare_metrics(result, reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete, validated archive; interruption leaves the original intact.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as fh:
            temporary = Path(fh.name)
            np.savez_compressed(fh, **probabilities)
        if args.overwrite:
            os.replace(temporary, output)
        else:
            # Exclusive publication also protects against another writer starting meanwhile.
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report = {"matches_reference": matches, "tolerance": 0.0001,
              "metrics": comparison, "elapsed_s": round(time.time() - started, 1),
              "featfn": final.FEATFN, "model": final.MODEL, "seed": final.SEED}
    output.with_suffix(".rebuild.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Wrote", output)
    if not matches:
        print("Scores differ from the locked baseline. Cache retained for investigation; "
              "do not claim baseline equivalence.")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
