"""Regenerate the submission from the provided data. The single command the repo promises.

    bash scripts/download_external.sh         # once: public IDSSE + Metrica sample data
    uv run python -m mtah.external            # once: parse them into cache/
    uv run python run.py                      # -> evaluation/results/<name>.csv
    uv run python run.py --method rules --name rules_v2

Deterministic: no randomness anywhere in the default pipeline, so the same data gives the same
file byte for byte.
"""

import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from mtah.data import Match  # noqa: E402
from mtah.features import compute  # noqa: E402
from mtah.report import report  # noqa: E402

METHODS = ("fused_ext", "fused_own", "rules")


def predict(method, m, f):
    from mtah import rules
    rule_labels = rules.run(m, f)
    if method == "rules":
        return rule_labels
    from mtah import decode, shape, shape_model
    if method == "fused_ext":
        # ball-free model trained on public external tracking with true labels, degraded to
        # our noise (W8, analysis/ext_pipeline.py)
        X, _ = shape.compute(m, cam=False)
        P = shape_model.fit_external()(X)
        return decode.decode(m, rule_labels, P, f["seen"])
    if method == "fused_own":
        # ball-free model trained on our own rule pseudo-labels (fused_v2)
        P = shape_model.fit_full(m, rule_labels, f)
        return decode.decode(m, rule_labels, P, f["seen"])
    raise ValueError(method)


def write(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, c) for i, c in enumerate(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="fused_ext", choices=METHODS)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or args.method

    m = Match()
    f = compute(m)
    labels = predict(args.method, m, f)
    path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
    write(path, labels)
    report(name, labels, m)
    with open(path, "rb") as handle:
        print("wrote {}  sha256 {}".format(os.path.relpath(path, ROOT),
                                           hashlib.sha256(handle.read()).hexdigest()[:16]))
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                    path], check=True)


if __name__ == "__main__":
    main()
