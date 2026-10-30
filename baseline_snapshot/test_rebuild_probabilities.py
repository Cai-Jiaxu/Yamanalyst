"""Synthetic checks: no match data, training, or cache writes required."""

from contextlib import redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from baseline_snapshot.rebuild_probabilities import (
    compare_metrics, main, validate_probabilities,
)


class RebuildTests(unittest.TestCase):
    def test_metric_drift_is_detected(self):
        result = {"bench": 0.66, "ext": [0.62], "masked": [0.70],
                  "public": [0.69, 0.91]}
        reference = {"bench_spellF1": 0.66, "ext_spellF1": 0.62,
                     "masked_spellF1": 0.70, "public_spellF1": 0.69,
                     "public_macroF1": 0.91}
        self.assertTrue(compare_metrics(result, reference)[1])
        result["ext"][0] += 0.001
        self.assertFalse(compare_metrics(result, reference)[1])

    def test_probability_contract(self):
        case = SimpleNamespace(name="public", m=SimpleNamespace(n=2))
        bench = SimpleNamespace(ext={}, public=case, masked=[])
        good = np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]])
        validate_probabilities({"public": good}, bench)
        for invalid in ({}, {"public": good[:, :2]},
                        {"public": np.full((2, 3), np.nan)},
                        {"public": good * 0.5},
                        {"public": np.array([[-0.1, 0.6, 0.5]] * 2)}):
            with self.subTest(keys=list(invalid)):
                with self.assertRaises(ValueError):
                    validate_probabilities(invalid, bench)

    def test_existing_output_is_preserved_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "P_R1.npz"
            path.write_bytes(b"existing baseline")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--output", str(path)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_bytes(), b"existing baseline")


if __name__ == "__main__":
    unittest.main()
