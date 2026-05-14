from __future__ import annotations

from v4.scripts.run_protocol136_account_scaled_daily_stop import (
    decide,
    evaluate_experiment,
    scaled_daily_stop_policy,
)
from v4.sim.protocol101_position_sizing import effective_daily_stop


def test_scaled_daily_stop_uses_more_negative_equity_scaled_threshold() -> None:
    policy = scaled_daily_stop_policy(10_000.0, 0.005, "test")

    assert effective_daily_stop(policy, 10_000.0) == -750.0
    assert effective_daily_stop(policy, 300_000.0) == -1500.0


def test_evaluate_experiment_requires_positive_stress_incremental() -> None:
    current = [
        {
            "starting_cash": 10_000.0,
            "stress_per_side": 0.25,
            "policy_kind": "candidate",
            "total_pnl": 100.0,
            "max_drawdown": -100.0,
            "worst_day_pnl": -100.0,
        }
    ]
    rows = [
        {
            "starting_cash": 10_000.0,
            "stress_per_side": 0.25,
            "policy_kind": "candidate",
            "total_pnl": 90.0,
            "incremental_pnl": -10.0,
            "max_drawdown": -100.0,
            "worst_day_pnl": -100.0,
            "passes_cash_level_gate": True,
        }
    ]

    result = evaluate_experiment("test", rows, current)

    assert result["passes"] is False


def test_decide_pauses_after_three_failed_hypotheses() -> None:
    assert decide([], 3) == "pause_after_three_failed_daily_stop_hypotheses"

