"""Submission serialization must produce identical bytes on every platform."""

from pathlib import Path
import tempfile
import unittest

from run import write


class SubmissionIOTests(unittest.TestCase):
    def test_csv_uses_canonical_lf_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "submission.csv"
            write(path, ["H", "A", "D", "X"])
            self.assertEqual(path.read_bytes(),
                             b"frame,label\n0,H\n1,A\n2,D\n3,X\n")


if __name__ == "__main__":
    unittest.main()
