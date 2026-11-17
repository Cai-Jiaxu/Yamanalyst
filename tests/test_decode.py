"""Decoder post-processing on synthetic labels: no match data needed.

Pins the shipped behaviour (final_v6: the two-sided stoppage gate of final_v5, then the one-sided
pass of R20, and min_dead) so that a later change cannot move the submission by accident.
"""

import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import decode  # noqa: E402


def labels(*runs):
    """labels(("D", 50), ("H", 40)) -> array of 90 one-letter labels."""
    return np.array([c for c, n in runs for _ in range(n)], dtype="<U1")


def pace(n, value):
    """f dict with both teams at `value` m/s on every frame."""
    return {"sh": np.full(n, value), "sa": np.full(n, value)}


class StopGateTests(unittest.TestCase):
    def test_short_slow_spell_between_dead_balls_becomes_dead(self):
        x = labels(("H", 200), ("D", 60), ("A", 50), ("D", 60), ("H", 200))
        out = decode.stop_gate(x, pace(len(x), 1.0), 100)
        self.assertTrue((out[200:370] == "D").all())
        self.assertTrue((out[:200] == "H").all() and (out[370:] == "H").all())

    def test_kept_when_players_run_or_spell_is_long(self):
        x = labels(("H", 200), ("D", 60), ("A", 50), ("D", 60), ("H", 200))
        np.testing.assert_array_equal(decode.stop_gate(x, pace(len(x), 2.5), 100), x)
        y = labels(("H", 200), ("D", 60), ("A", 150), ("D", 60), ("H", 200))
        np.testing.assert_array_equal(decode.stop_gate(y, pace(len(y), 1.0), 100), y)

    def test_default_needs_dead_ball_on_both_sides(self):
        x = labels(("H", 200), ("D", 60), ("A", 50), ("H", 200))
        np.testing.assert_array_equal(decode.stop_gate(x, pace(len(x), 1.0), 100), x)

    def test_one_sided_gate_accepts_one_dead_neighbour(self):
        x = labels(("H", 200), ("D", 60), ("A", 50), ("H", 200))
        out = decode.stop_gate(x, pace(len(x), 1.0), 100, sides=1)
        self.assertTrue((out[260:310] == "D").all())
        self.assertEqual(int((out != x).sum()), 50)

    def test_one_sided_gate_does_not_chain(self):
        # A then H, both short and slow, after a dead ball: only A touches a dead ball as passed in
        x = labels(("H", 300), ("D", 60), ("A", 50), ("H", 50), ("A", 300))
        out = decode.stop_gate(x, pace(len(x), 1.0), 100, sides=1)
        self.assertTrue((out[360:410] == "D").all())
        self.assertTrue((out[410:460] == "H").all())

    def test_first_and_last_spell_are_never_touched(self):
        x = labels(("H", 50), ("D", 60), ("A", 50))
        for sides in (1, 2):
            np.testing.assert_array_equal(decode.stop_gate(x, pace(len(x), 1.0), 100, sides=sides), x)


class MinDurationTests(unittest.TestCase):
    def test_short_dead_spell_takes_the_previous_label(self):
        x = labels(("H", 100), ("D", 10), ("A", 100))
        out = decode.min_duration(x, 25, 0)
        self.assertTrue((out[:110] == "H").all() and (out[110:] == "A").all())

    def test_long_dead_spell_is_kept(self):
        x = labels(("H", 100), ("D", 30), ("A", 100))
        np.testing.assert_array_equal(decode.min_duration(x, 25, 0), x)


class ViterbiTests(unittest.TestCase):
    def test_switch_cost_removes_a_weak_blip(self):
        logE = np.log(np.full((300, 3), 0.05))
        logE[:, 0] = np.log(0.9)
        logE[140:150, 0], logE[140:150, 1] = np.log(0.3), np.log(0.65)   # 10 weak frames for A
        T = np.zeros((3, 3))
        T[0, 1] = T[1, 0] = T[0, 2] = T[2, 0] = T[1, 2] = T[2, 1] = -12.0
        self.assertTrue((decode.viterbi(logE, T) == 0).all())
        self.assertEqual(int((decode.viterbi(logE, np.zeros((3, 3))) == 1).sum()), 10)


class ShippedConfigTests(unittest.TestCase):
    def test_submission_decoder_is_final_v6(self):
        from mtah import final
        v5 = {"w_shape": 0.75, "cost_dead": 12.0, "cost_team": 12.0, "min_dead": 25,
              "w_bound": 0.0, "kickoff": "half", "use_rules": False, "stop_gate": 100}
        self.assertEqual(final.DECODER_V5, v5)          # the locked baseline_snapshot config
        self.assertEqual(final.DECODER, dict(v5, stop_gate_sides=1))
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "baseline_snapshot", "baseline_metrics.json")) as fh:
            self.assertEqual(json.load(fh)["decoder"], v5)
        self.assertEqual(decode.PARAMS["stop_gate_sides"], 2)   # off unless a config asks for it
        self.assertEqual((final.FEATFN, final.MODEL, final.SEED), ("full_plus3", {"max_iter": 600}, 0))


if __name__ == "__main__":
    unittest.main()
