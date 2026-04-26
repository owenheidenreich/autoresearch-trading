"""Lock the trained oracle gate's headline numbers."""
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

    def test_top_feature_is_time_stop_margin_raw(self) -> None:
        importance = self.gate.train_metadata.get("feature_importance", {})
        ranked = sorted(importance.items(), key=lambda r: -r[1])
        self.assertEqual(ranked[0][0], "time_stop_margin_raw",
                         f"top feature should be time_stop_margin_raw, got {ranked[0]}")

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

    def test_forward_walk_lift(self) -> None:
        result = evaluate_gate_on_forward_walk(self.gate)
        # Gate should match or beat baseline always-oracle on FW
        self.assertGreaterEqual(result["gate_pf"], result["baseline_pf_always_oracle"],
                                "gate should >= baseline on forward-walk")
        # PF should be at least 2.0 (we found 2.27 at threshold 0.45)
        self.assertGreaterEqual(result["gate_pf"], 2.0,
                                f"gate forward-walk PF should be >= 2.0, got {result['gate_pf']:.3f}")
        # n trades
        self.assertGreaterEqual(result["n"], 50)


if __name__ == "__main__":
    unittest.main()
