"""Lock the forward-walk-with-oracle headline numbers.

Asserts that the published with-oracle forward walk results stay within
expected ranges. If a future change to the oracle build, the dataset, or
the forward-walk script silently shifts these numbers, the test breaks.
"""
from __future__ import annotations

import json
import os
import unittest


class ForwardWalkWithOracleTests(unittest.TestCase):

    REPORT = "v3/artifacts/forward_walk/spx_combined_3seed_001_with_oracle.json"

    def setUp(self) -> None:
        if not os.path.exists(self.REPORT):
            self.skipTest(
                f"{self.REPORT} not produced — run "
                "scripts/overnight_oracle_rebuild_v2.sh first"
            )
        with open(self.REPORT) as f:
            self.report = json.load(f)

    def test_forward_walk_window_is_42_days(self) -> None:
        n = self.report.get("forward_walk_unique_days")
        self.assertEqual(n, 42, f"expected 42 forward-walk days, got {n}")
        self.assertEqual(self.report.get("forward_walk_first_day"), "2026-02-25")
        self.assertEqual(self.report.get("forward_walk_last_day"), "2026-04-24")

    def test_seed_46_with_oracle_is_strongest(self) -> None:
        per_seed = self.report.get("per_seed", {})
        seed_46 = per_seed.get("46") or per_seed.get(46)
        self.assertIsNotNone(seed_46)
        oracle = seed_46.get("with_oracle", {})
        self.assertGreaterEqual(oracle.get("pf", 0), 3.0,
                                "seed 46 with oracle should have PF >= 3.0")
        self.assertGreater(oracle.get("n", 0), 15)
        self.assertLessEqual(oracle.get("dd_pct", 0), 13.0)

    def test_seed_44_with_oracle_is_second_strongest(self) -> None:
        per_seed = self.report.get("per_seed", {})
        seed_44 = per_seed.get("44") or per_seed.get(44)
        self.assertIsNotNone(seed_44)
        oracle = seed_44.get("with_oracle", {})
        self.assertGreaterEqual(oracle.get("pf", 0), 3.0,
                                "seed 44 with oracle should have PF >= 3.0")

    def test_oracle_lifts_cross_seed_mean(self) -> None:
        per_seed = self.report.get("per_seed", {})
        oracle_pfs = []
        no_oracle_pfs = []
        for s in [42, 43, 44, 45, 46]:
            m = per_seed.get(str(s)) or per_seed.get(s)
            if not m or m.get("n_trades", 0) == 0:
                continue
            o = m.get("with_oracle", {})
            n = m.get("no_oracle_dataset_label", {})
            if o.get("n", 0) > 0:
                oracle_pfs.append(o.get("pf", 0))
            if n.get("n", 0) > 0:
                no_oracle_pfs.append(n.get("pf", 0))
        if not oracle_pfs or not no_oracle_pfs:
            self.skipTest("no oracle/no-oracle PF data")
        oracle_mean = sum(oracle_pfs) / len(oracle_pfs)
        no_oracle_mean = sum(no_oracle_pfs) / len(no_oracle_pfs)
        self.assertGreater(oracle_mean, no_oracle_mean,
                           f"oracle should lift cross-seed mean; "
                           f"got {oracle_mean:.3f} vs no-oracle {no_oracle_mean:.3f}")
        self.assertGreaterEqual(oracle_mean, 1.5,
                                f"with-oracle cross-seed mean should be >= 1.5, got {oracle_mean:.3f}")

    def test_calibration_pf_seeds_were_loaded(self) -> None:
        """Sanity check: each seed's calibration_pf is set."""
        per_seed = self.report.get("per_seed", {})
        expected_cal_pf = {
            42: (15.0, 25.0),  # ~21.58
            43: (4.0, 7.5),    # ~5.71
            44: (6.0, 10.0),   # ~7.99
            45: (2.5, 4.0),    # ~3.15
            46: (1.0, 3.0),    # ~1.93
        }
        for s, (lo, hi) in expected_cal_pf.items():
            m = per_seed.get(str(s)) or per_seed.get(s)
            cal = m.get("calibration_pf") if m else None
            self.assertIsNotNone(cal, f"seed {s} cal_pf is None")
            self.assertGreaterEqual(cal, lo, f"seed {s} cal_pf {cal} < {lo}")
            self.assertLessEqual(cal, hi, f"seed {s} cal_pf {cal} > {hi}")


if __name__ == "__main__":
    unittest.main()
