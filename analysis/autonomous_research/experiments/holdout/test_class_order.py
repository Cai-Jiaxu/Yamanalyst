"""Regression: string vs integer class ordering must map to pm.LABELS (H,A,D).

sklearn HistGBDT sorts string classes alphabetically: [A, D, H]. Any code that
passes string labels and then assumes proba columns are [H,A,D] is wrong.
The production baseline uses integer labels (H=0,A=1,D=2), which is safe.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import numpy as np
from mtah.score import pm
from sklearn.ensemble import HistGradientBoostingClassifier


def test_string_classes_not_had():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.array(["H", "A", "D"] * 20)
    c = HistGradientBoostingClassifier(max_iter=5).fit(X, y)
    order = list(c.classes_)
    assert order == ["A", "D", "H"], order
    # adapter: reorder proba columns into pm.LABELS order (H,A,D)
    idx = [order.index(s) for s in pm.LABELS[:3]]
    P = c.predict_proba(X)[:, idx]
    assert list(pm.LABELS[:3]) == ["H", "A", "D"]


def test_integer_path_is_identity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = np.array([0, 1, 2] * 20)  # H,A,D
    c = HistGradientBoostingClassifier(max_iter=5).fit(X, y)
    assert list(c.classes_) == [0, 1, 2]


if __name__ == "__main__":
    test_string_classes_not_had()
    test_integer_path_is_identity()
    print("class-order tests PASS")
