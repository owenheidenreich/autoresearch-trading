"""Tests for the oracle gate code surface.

⚠️ The original gate's PF claims were retracted (see
oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md): two features in the
original feature set were oracle labels. The pickle was deleted; these
tests are skipped unless the gate is re-trained with the now-clean
FEATURES list. They no longer assert the +24% lift number.
"""
from __future__ import annotations

import os
import unittest

import pandas as pd

from v3.live_shadow.oracle_gate import (
    DEFAULT_MODEL_PATH,
    OracleGate,
    evaluate_gate_on_forward_walk,
)


class OracleGateTests(unittest.TestCase):

    def setUp(self) -> None:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            self.skipTest(
                f"{DEFAULT_MODEL_PATH} not present — run "
                "`v3.live_shadow.oracle_gate train` first"
            )
        self.gate = OracleGate.load()

    def test_gate_loads(self) -> None:
        self.assertIsNotNone(self.gate.classifier)
        self.assertEqual(len(self.gate.feature_names), 16)
        self.assertGreater(self.gate.n_train, 1500)

    def test_no_label_leak_features(self) -> None:
        """Ensure gate FEATURES list does not contain oracle labels."""
        from v3.live_shadow.oracle_gate import FEATURES
        leaky = {"time_stop_margin_raw", "side_margin_raw",
                 "best_forward_pnl_call", "best_forward_pnl_put",
                 "worst_forward_pnl_call", "worst_forward_pnl_put",
                 "entry_value_raw", "time_stop_pnl_call", "time_stop_pnl_put",
                 "time_stop_value_raw", "opportunity_oracle_entry"}
        for f in FEATURES:
            self.assertNotIn(f, leaky, f"Feature {f} is an oracle label — would leak future PnL")

    def test_predict_prob_returns_probability(self) -> None:
        # construct a feature dict
        features = {f: 0.0 for f in self.gate.feature_names}
        p = self.gate.predict_prob_oracle_better(features)
        self.assertIsInstance(p, float)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_use_oracle_returns_bool(self) -> None:
        features = {f: 0.0 for f in self.gate.feature_names}
        b = self.gate.use_oracle_exit(features)
        self.assertIsInstance(b, bool)

    def test_forward_walk_at_least_baseline(self) -> None:
        """With clean features, gate should at least not catastrophically
        underperform always-oracle baseline."""
        result = evaluate_gate_on_forward_walk(self.gate)
        # n trades
        self.assertGreaterEqual(result["n"], 50)
        # Gate should be within reasonable range of baseline (no PF lift
        # claim — that was retracted as label leakage).
        self.assertGreaterEqual(result["gate_pf"], 1.5,
                                f"gate PF too low: {result['gate_pf']:.3f}")


if __name__ == "__main__":
    unittest.main()
