from __future__ import annotations

from v4.scripts.run_protocol135_starting_cash_sensitivity import (
    compare_to_baseline,
    decide,
    stress_trades,
)


def test_stress_trades_applies_round_trip_per_side_cost() -> None:
    rows = [{"pnl": 100.0}]

    assert stress_trades(rows, 0.25)[0]["pnl"] == 50.0


def test_compare_to_baseline_requires_incremental_return_and_risk_control() -> None:
    baseline = {
        "starting_cash": 10_000.0,
        "total_pnl": 100_000.0,
        "max_drawdown_pct": -0.06,
        "worst_day_pnl": -2_000.0,
    }
    candidate = {
        "starting_cash": 10_000.0,
        "total_pnl": 120_000.0,
        "max_drawdown_pct": -0.07,
        "worst_day_pnl": -2_100.0,
        "risk_of_ruin": False,
    }

    result = compare_to_baseline(candidate, baseline)

    assert result["incremental_pnl"] == 20_000.0
    assert result["passes_cash_level_gate"] is True


def test_decide_blocks_025_stress_failure() -> None:
    rows = [
        {
            "policy_kind": "candidate",
            "stress_per_side": 0.25,
            "passes_cash_level_gate": False,
            "risk_of_ruin": False,
        }
    ]

    assert decide(rows) == "fragile_sizer_fails_025_starting_cash_sensitivity"

