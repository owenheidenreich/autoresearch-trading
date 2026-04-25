"""Lock the spx_combined_3seed_001 K-consensus floor.

The 5-seed champion (`spx_combined_3seed_001`, seeds 42-46) has agg PF
1.881 unfiltered. K-of-5 side-consensus filtering (per-seed mode) lifts
PF and tightens DD without retraining. This test asserts the K=2
recipe — the recommended deployment filter — keeps clearing PF >= 2.0
on the 5-seed mean and trade count >= 150 per seed.

If a future change to chosen_objective_pnl labeling or seed artifacts
breaks this, we want to catch it before it breaks the deployment.
"""
from __future__ import annotations

import json
import os
import unittest


class EnsembleConsensusFloorTests(unittest.TestCase):

    REPORT = "v3/artifacts/ensemble_consensus/spx_combined_3seed_001.json"

    def setUp(self) -> None:
        if not os.path.exists(self.REPORT):
            self.skipTest(
                f"{self.REPORT} not produced — run "
                "`.venv/bin/python -m v3.analysis.ensemble_consensus` first"
            )
        with open(self.REPORT) as f:
            self.report = json.load(f)

    def test_5_seed_baseline_matches_champion(self) -> None:
        baseline = self.report["five_seed_mean_baseline"]
        # Champion adoption locked these values (champion_adoption_2026_04_25.md)
        self.assertGreaterEqual(baseline["mean_pf"], 1.85)
        self.assertLessEqual(baseline["mean_pf"], 1.92)
        self.assertGreaterEqual(baseline["min_pf"], 1.60)
        self.assertLessEqual(baseline["mean_dd_pct"], 13.5)

    def test_K2_per_seed_consensus_lifts_pf(self) -> None:
        agg = self.report["per_seed_K_aggregated"]["2"]
        self.assertGreaterEqual(
            agg["mean_pf"], 2.0, "K=2 should lift mean PF to >= 2.0"
        )
        self.assertGreaterEqual(
            agg["min_pf"], 1.75, "K=2 min seed PF should clear retired floor"
        )
        self.assertGreaterEqual(
            agg["min_n_trades"], 150, "K=2 should keep >= 150 trades on weakest seed"
        )
        self.assertLessEqual(agg["max_dd_pct"], 13.0)

    def test_K3_consensus_further_lifts_pf(self) -> None:
        agg = self.report["per_seed_K_aggregated"]["3"]
        self.assertGreaterEqual(agg["mean_pf"], 2.5)
        self.assertGreaterEqual(agg["min_pf"], 2.0)
        self.assertLessEqual(agg["max_dd_pct"], 13.0)

    def test_qualifying_bar_counts_monotone(self) -> None:
        bars = self.report["qualifying_bars_by_K"]
        prev = float("inf")
        for K in ["1", "2", "3", "4", "5"]:
            self.assertLessEqual(bars[K], prev)
            prev = bars[K]


if __name__ == "__main__":
    unittest.main()
