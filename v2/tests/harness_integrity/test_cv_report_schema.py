"""CVReport round-trip via JSON and TSV serialization.

Invariant: every field in the schema survives a JSON round-trip with the same
value; the TSV row has the right number of tab-separated columns; scope-mixed
fields do not exist at the top level.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.core.cv_report import (
    CVReport,
    FoldSlice,
    PooledSlice,
    RESULTS_TSV_HEADER,
    SCHEMA_VERSION,
    StabilitySlice,
    header_line,
)


def _minimal_report() -> CVReport:
    fold = FoldSlice(
        fold_idx=4,
        window_id="abcd1234abcd1234",
        test_window_start="2025-01-01",
        test_window_end="2025-03-01",
        train_window_start="2024-01-01",
        train_window_end="2024-12-31",
        seed=2026,
        score=0.42,
        gate_failure=None,
        metrics={"total_trades": 100, "traded_days": 45, "profit_factor": 1.2},
        baseline_scores={"random": 0.1, "atm": 0.2, "rules": 0.15, "trailing": 0.18},
        n_trades=100,
        n_test_days=60,
        n_traded_days=45,
        train_seconds=123.4,
    )
    return CVReport(
        experiment_id="exp_test",
        screening_mode="full",
        folds=[fold],
        pooled=PooledSlice(
            profit_factor=1.2, max_account_drawdown=0.1, win_rate=0.55,
            call_pct=0.52, put_pct=0.48, net_pnl_dollars=123.0,
            total_trades=100, total_eval_days=60, traded_days=45,
            positive_day_rate=0.6, daily_sortino=0.9,
        ),
        stability=StabilitySlice(
            mean_fold_score=0.42, min_fold_score=0.42, max_fold_score=0.42,
            std_fold_score=0.0, per_fold_scores=[0.42],
            per_fold_gate_failures=[False], any_fold_gate_failure=False,
        ),
        aggregate_baselines={"random": 0.1, "atm": 0.2, "rules": 0.15, "trailing": 0.18},
        beats_all_baselines=True,
        training_config_fingerprint="cfgfp",
        training_env_overrides={"SOFT_TEMP": "0.08", "SIDE_MODE": "skew"},
        policy_fingerprint="polfp",
        dataset_fingerprint="dsetfp",
        evaluator_fingerprint="evalfp",
        training_seconds=456.0,
    )


class TestCVReportSchema(unittest.TestCase):

    def test_json_round_trip(self):
        r = _minimal_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cv_report.json")
            r.to_json(path)
            loaded = CVReport.from_json(path)
        self.assertEqual(loaded.experiment_id, r.experiment_id)
        self.assertEqual(loaded.schema_version, SCHEMA_VERSION)
        self.assertEqual(loaded.folds[0].window_id, r.folds[0].window_id)
        self.assertEqual(loaded.pooled.profit_factor, r.pooled.profit_factor)
        self.assertEqual(loaded.stability.per_fold_scores, r.stability.per_fold_scores)

    def test_schema_mismatch_rejected(self):
        r = _minimal_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv_report.json"
            d = r.to_dict()
            d["schema_version"] = "bogus_v99"
            path.write_text(json.dumps(d, default=str))
            with self.assertRaises(ValueError):
                CVReport.from_json(str(path))

    def test_tsv_header_matches_columns(self):
        self.assertEqual(header_line().split("\t"), RESULTS_TSV_HEADER)

    def test_to_tsv_row_has_right_number_of_columns(self):
        row = _minimal_report().to_tsv_row()
        self.assertEqual(len(row.split("\t")), len(RESULTS_TSV_HEADER))

    def test_top_level_has_no_blended_fields(self):
        """No scope-mixing: top-level fields must be in {pooled, stability, folds}."""
        r = _minimal_report()
        forbidden = {"profit_factor", "daily_sortino", "max_account_drawdown",
                     "win_rate", "net_pnl_dollars", "positive_day_rate"}
        fields = {f for f in r.to_dict().keys()}
        self.assertFalse(
            forbidden & fields,
            f"CVReport top-level must not carry scope-mixed fields: {forbidden & fields}",
        )

    def test_fingerprints_are_separate(self):
        """training_config_fingerprint and evaluator_fingerprint must be distinct fields."""
        r = _minimal_report()
        d = r.to_dict()
        self.assertIn("training_config_fingerprint", d)
        self.assertIn("evaluator_fingerprint", d)
        self.assertNotEqual(
            r.training_config_fingerprint, r.evaluator_fingerprint,
            "Distinct semantics require distinct values in a realistic test.",
        )
        self.assertNotIn(
            "config_fingerprint", d,
            "Legacy ambiguous `config_fingerprint` must be gone — use training_config_fingerprint.",
        )

    def test_training_env_overrides_round_trip(self):
        r = _minimal_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cv_report.json")
            r.to_json(path)
            loaded = CVReport.from_json(path)
        self.assertEqual(loaded.training_env_overrides, r.training_env_overrides)
        self.assertEqual(loaded.training_env_overrides.get("SOFT_TEMP"), "0.08")


if __name__ == "__main__":
    unittest.main()
