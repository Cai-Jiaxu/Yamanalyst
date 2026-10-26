"""Unit tests for the candidate feature extractor compliance.

Checks, on the causal primitive `_trail` (the only temporal operator all five
blocks share):
  1. No-future invariance: output[t] depends only on x[..=t].
  2. Half reset: a trailing window never crosses a half boundary.
Run: .venv/bin/python analysis/feat_track/test_feat_candidates.py
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from analysis.feat_track.feat_candidates import _trail  # noqa: E402


def test_no_future_invariance():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    hid = np.zeros(n, dtype=int)  # single half
    w = 25
    out = _trail(x, w, hid)
    # Corrupt everything after frame t and re-run; prefix must be identical.
    for t in (10, 100, 250, 399):
        x2 = x.copy()
        x2[t + 1:] = 1e6  # huge future mutation
        out2 = _trail(x2, w, hid)
        assert np.allclose(out[:t + 1], out2[:t + 1], equal_nan=True), \
            f"no-future violated at t={t}"
    print("PASS no-future invariance")


def test_half_reset():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    # half boundary at frame 100; fill second half with large offset.
    hid = np.zeros(n, dtype=int)
    hid[100:] = 1
    x[100:] += 50.0
    w = 50
    out = _trail(x, w, hid)
    # At frame 110 (10 into second half), window may only use x[100..110].
    # So the trailing mean must be ~mean(x[100:111]) = 50 + small, never
    # contaminated by first-half ~0 values.
    t = 110
    expected = np.nanmean(x[100:t + 1])
    got = out[t]
    assert abs(got - expected) < 1e-9, f"half-reset leak: got {got} exp {expected}"
    # And it must NOT equal a full 50-frame window that would mix halves.
    leaky = np.nanmean(x[t - w + 1:t + 1])
    assert abs(got - leaky) > 1.0, "window appears to cross half boundary"
    print("PASS half reset (no boundary crossing)")


if __name__ == "__main__":
    test_no_future_invariance()
    test_half_reset()
    print("ALL TESTS PASSED")
