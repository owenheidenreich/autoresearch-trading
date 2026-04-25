"""Lock the calibration audit findings.

Asserts that the cal_pf > 4 → OOS collapse signal is present in the
audit output, and that W12 is the dominant offender. If a future
retrain shifts the calibration distribution materially, this test
breaks and we revisit the deployment guard.
"""
from __future__ import annotations

import json
import os
import unittest


class CalibrationAuditTests(unittest.TestCase):

    REPORT = "v3/artifacts/calibration_audit/spx_combined_3seed_001.json"

    def setUp(self) -> None:
        if not os.path.exists(self.REPORT):
            self.skipTest(
                f"{self.REPORT} not produced — run "
                "`.venv/bin/python -m v3.analysis.per_seed_calibration_audit` first"
            )
        with open(self.REPORT) as f:
            self.audit = json.load(f)

    def test_w12_is_in_top_degradation_list(self) -> None:
        top = self.audit["top10_val_to_oos_degradation"]
        self.assertGreater(len(top), 0)
        top_windows = {r["window"] for r in top[:5]}
        self.assertIn(12, top_windows, "W12 should be in the top-5 worst val→OOS degradations")

    def test_high_cal_pf_records_correlate_with_oos_collapse(self) -> None:
        recs = [
            r
            for r in self.audit["records"]
            if r.get("cal_pf") is not None
            and r.get("oos_pf") is not None
            and (r.get("oos_trades") or 0) >= 5
        ]
        high = [r for r in recs if r["cal_pf"] > 4.0]
        if not high:
            self.skipTest("no cal_pf > 4 records (champion may have shifted)")
        # Median val→OOS pct should be < -50% on high cal_pf records
        pcts = [r["val_to_oos_pf_pct"] for r in high]
        median = sorted(pcts)[len(pcts) // 2]
        self.assertLess(
            median,
            -0.5,
            f"high cal_pf records should show severe OOS degradation, got median {median:.2%}",
        )

    def test_w12_is_top_high_cal_pf_window(self) -> None:
        recs = [
            r for r in self.audit["records"]
            if r.get("cal_pf") is not None and r["cal_pf"] > 4.0
        ]
        if not recs:
            self.skipTest("no cal_pf > 4 records")
        from collections import Counter
        per_window = Counter(r["window"] for r in recs)
        # W12 should be the modal high-cal_pf window, with at least 3 hits
        w12_count = per_window.get(12, 0)
        self.assertGreaterEqual(w12_count, 3, f"W12 should have >=3 high cal_pf records, got {w12_count}")
        # Some other window can match but not exceed W12
        max_other = max((c for w, c in per_window.items() if w != 12), default=0)
        self.assertGreaterEqual(
            w12_count, max_other,
            f"W12 ({w12_count}) should be >= every other window's count (max other {max_other})"
        )


if __name__ == "__main__":
    unittest.main()
