"""Independent checks of published record integrity and aggregate outcomes."""

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import csv
from tudtf.verify import verify_hashes, verify_decisions

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/reference"


class ArchiveTests(unittest.TestCase):
    def test_all_reference_hashes(self):
        self.assertEqual(verify_hashes(REFERENCE, REFERENCE / "SHA256.json"), 13)

    def test_every_decision_and_block_summary(self):
        audit = verify_decisions(REFERENCE)
        self.assertEqual(audit["decision_records"], 62208)
        self.assertEqual(audit["blocks_checked"], 864)

    def test_modified_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "data.csv").write_bytes(b"value\n1\n")
            expected = hashlib.sha256((p / "data.csv").read_bytes()).hexdigest()
            (p / "SHA256.json").write_text(json.dumps({"data.csv": expected}))
            (p / "data.csv").write_bytes(b"value\n2\n")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_hashes(p, p / "SHA256.json")

    def test_manifest_cannot_escape_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "SHA256.json").write_text(json.dumps({"../outside": "x"}))
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_hashes(p, p / "SHA256.json")

    def test_published_overall_values_match_block_records(self):
        totals = {}
        with (REFERENCE / "block_results.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                key = (row["network"], row["controller"])
                counts, violations, cost = totals.get(key, (0, 0.0, 0.0))
                totals[key] = (
                    counts + 1,
                    violations + float(row["violation_rate"]),
                    cost + float(row["cost_eur"]),
                )
        with (REFERENCE / "overall_results.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                count, violations, cost = totals[(row["network"], row["controller"])]
                self.assertAlmostEqual(
                    violations / count, float(row["violation_rate"]), places=12
                )
                self.assertAlmostEqual(cost / count, float(row["cost_eur"]), places=7)


if __name__ == "__main__":
    unittest.main()
