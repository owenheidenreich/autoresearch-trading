from __future__ import annotations

from v4.scripts.run_protocol133_sizing_split_stress import (
    decide,
    group_rows,
    stress_trades,
)


def test_stress_trades_subtracts_round_trip_cost_per_contract_basis() -> None:
    rows = [{"pnl": 100.0}, {"pnl": -50.0}]

    stressed = stress_trades(rows, 0.25)

    assert stressed[0]["pnl"] == 50.0
    assert stressed[1]["pnl"] == -100.0


def test_group_rows_reports_split_pnl_and_contracts() -> None:
    rows = [
        {"policy": "p", "segment": "q1", "realized_pnl": 100.0, "quantity": 1},
        {"policy": "p", "segment": "q1", "realized_pnl": 200.0, "quantity": 2},
        {"policy": "p", "segment": "q2", "realized_pnl": 0.0, "quantity": 0},
    ]

    grouped = group_rows(rows, stress=0.0, group_col="segment")

    q1 = next(row for row in grouped if row["group"] == "q1")
    assert q1["total_pnl"] == 300.0
    assert q1["taken_trades"] == 2
    assert q1["total_contracts"] == 3


def test_decide_requires_025_stress_acceptance_for_best_candidate() -> None:
    rows = [
        {
            "stress_per_side": 0.0,
            "policy": "candidate",
            "passes_acceptance": True,
            "total_pnl": 100.0,
            "max_drawdown_pct": -0.01,
        },
        {
            "stress_per_side": 0.25,
            "policy": "candidate",
            "passes_acceptance": False,
            "total_pnl": 50.0,
            "max_drawdown_pct": -0.01,
        },
    ]

    assert decide(rows) == "fragile_sizing_candidate_fails_025_stress"

