from __future__ import annotations

import unittest

from v2.core.metrics import ReplayMetrics
from v2 import train


class TestCheckpointSelection(unittest.TestCase):

    def test_training_env_vars_capture_checkpoint_mode(self):
        self.assertIn("CKPT_SELECTION_MODE", train._TRAINING_ENV_VARS)

    def test_replay_selection_prefers_gate_pass(self):
        gate_fail = ReplayMetrics(
            profit_factor=2.0,
            max_account_drawdown=0.40,
            win_rate=0.60,
            positive_day_rate=0.55,
            trades_per_day=2.0,
            score=-0.2,
            gate_failure="excessive_drawdown (40.0% > 25%)",
        )
        gate_pass = ReplayMetrics(
            profit_factor=1.1,
            max_account_drawdown=0.12,
            win_rate=0.52,
            positive_day_rate=0.50,
            trades_per_day=2.5,
            score=0.15,
            gate_failure=None,
        )
        self.assertGreater(
            train._replay_selection_key(gate_pass),
            train._replay_selection_key(gate_fail),
        )

    def test_replay_selection_breaks_gate_fail_ties_by_pf_then_dd(self):
        weaker = ReplayMetrics(
            profit_factor=0.72,
            max_account_drawdown=0.90,
            win_rate=0.45,
            positive_day_rate=0.30,
            trades_per_day=8.0,
            score=-0.2,
            gate_failure="excessive_drawdown",
        )
        stronger = ReplayMetrics(
            profit_factor=0.81,
            max_account_drawdown=0.68,
            win_rate=0.47,
            positive_day_rate=0.41,
            trades_per_day=7.0,
            score=-0.2,
            gate_failure="excessive_drawdown",
        )
        self.assertGreater(
            train._replay_selection_key(stronger),
            train._replay_selection_key(weaker),
        )


if __name__ == "__main__":
    unittest.main()
