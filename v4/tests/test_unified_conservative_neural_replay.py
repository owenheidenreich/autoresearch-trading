from __future__ import annotations

import pandas as pd

from v4.model.unified_conservative_neural_policy import ConservativeNeuralPolicyConfig
from v4.model.unified_conservative_neural_replay import (
    choose_conservative_holding_exit,
    select_conservative_entry_candidate,
    summarize_replay_trades,
)


def test_select_conservative_entry_candidate_requires_gate_and_affordability() -> None:
    frame = pd.DataFrame(
        [
            {"predicted_advantage": 400.0, "positive_probability": 0.70, "tail_probability": 0.20, "entry_premium": 12_000.0},
            {"predicted_advantage": 300.0, "positive_probability": 0.60, "tail_probability": 0.20, "entry_premium": 500.0},
            {"predicted_advantage": 200.0, "positive_probability": 0.90, "tail_probability": 0.10, "entry_premium": 300.0},
        ]
    )

    idx, reason = select_conservative_entry_candidate(
        frame,
        equity=10_000.0,
        policy_config=ConservativeNeuralPolicyConfig(min_advantage_margin=250.0),
    )

    assert idx == 1
    assert reason == "challenger_entry_selected"


def test_choose_conservative_holding_exit_uses_first_failed_hold_gate() -> None:
    path = pd.DataFrame([{"state": 0}, {"state": 1}, {"state": 2}])
    predictions = pd.DataFrame(
        [
            {"predicted_advantage": 300.0, "positive_probability": 0.70, "tail_probability": 0.20},
            {"predicted_advantage": 50.0, "positive_probability": 0.70, "tail_probability": 0.20},
            {"predicted_advantage": 400.0, "positive_probability": 0.70, "tail_probability": 0.20},
        ]
    )

    idx, reason = choose_conservative_holding_exit(
        path,
        predictions,
        policy_config=ConservativeNeuralPolicyConfig(min_advantage_margin=250.0),
    )

    assert idx == 1
    assert reason == "lifecycle_conservative_exit"


def test_summarize_replay_trades_preserves_serial_account_metrics() -> None:
    summary = summarize_replay_trades(
        [
            {
                "session": "2026-01-02",
                "decision_time": "2026-01-02T15:00:00+00:00",
                "duration_minutes": 5.0,
                "pnl": 100.0,
                "account_equity_after": 10_100.0,
                "right": "C",
                "offset": 0.0,
                "source": "protocol101_defer",
                "exit_reason": "protocol101_baseline_exit",
            }
        ],
        event_count=10,
        starting_cash=10_000.0,
        skipped={"overlap": 2},
    )

    assert summary["total_pnl"] == 100.0
    assert summary["ending_equity"] == 10_100.0
    assert summary["skipped_overlap"] == 2
    assert summary["source_counts"] == {"protocol101_defer": 1}
