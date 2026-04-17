"""Regression test for the deploy.sh status dashboard TSV parser.

The parser must read the *current* CVReport schema, not the legacy
`experiment_id` / `score` columns. A parser that reads the wrong columns
produces blank or misleading status output, which is exactly the failure
mode the review flagged.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v2.core.cv_report import RESULTS_TSV_HEADER, header_line
from v2.ops.status_tsv import render


class TestStatusTSVParser(unittest.TestCase):

    def test_parses_new_schema_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            body = [
                header_line(),
                "\t".join(["exp_200", "full", "keep", "0.625000", "1.2345",
                           "0.1200", "500", "240", "false",
                           "[0.60,0.63,0.65,0.62,0.61]",
                           "full mode=full folds=[0.60,0.63,0.65,0.62,0.61]"]),
                "\t".join(["exp_201", "full", "revert", "-0.200000", "0.9000",
                           "0.3000", "0", "0", "true",
                           "[-0.2,-0.2,-0.2,-0.2,-0.2]",
                           "GATE_FAILURE mode=full folds=[-0.2,-0.2,-0.2,-0.2,-0.2]"]),
                "",
            ]
            path.write_text("\n".join(body))
            out = render(path)
            self.assertIn("exp_200", out)
            self.assertIn("0.625", out)      # stability score rendered
            self.assertIn("1.23", out)       # pooled PF rendered
            self.assertIn("keep", out)
            self.assertIn("exp_201", out)
            self.assertIn("-0.200", out)
            self.assertIn("revert", out)

    def test_rejects_legacy_schema(self):
        """A legacy header must be called out loudly, not silently parsed as blanks."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            path.write_text(
                "experiment_id\tscore\tstatus\tdescription\n"
                "exp_old\t0.5\tkeep\tlegacy row\n"
            )
            out = render(path)
            self.assertIn("ERROR", out)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            path.write_text(header_line() + "\n")
            out = render(path)
            self.assertIn("(no experiments yet)", out)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = render(Path(tmp) / "nonexistent.tsv")
            self.assertIn("(no results.tsv yet)", out)

    def test_header_columns_match_canonical(self):
        """If someone changes RESULTS_TSV_HEADER, the parser stays in sync."""
        # Hard-coded column list here on purpose — it fails loudly if the
        # canonical header shifts without this test being updated.
        expected = [
            "experiment", "screening_mode", "status", "stability_score",
            "pooled_pf", "pooled_dd", "pooled_trades", "pooled_traded_days",
            "any_gate_failure", "per_fold_scores", "description",
        ]
        self.assertEqual(RESULTS_TSV_HEADER, expected)


if __name__ == "__main__":
    unittest.main()
