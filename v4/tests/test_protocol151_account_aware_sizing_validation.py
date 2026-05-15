from __future__ import annotations

from v4.scripts.run_protocol151_account_aware_sizing_validation import (
    compare_summaries,
    evaluate_gate,
    stress_trades,
)


def test_stress_trades_applies_round_trip_cost() -> None:
    rows = [{"pnl": 100.0}, {"pnl": -25.0}]

    stressed = stress_trades(rows, 0.25)

    assert stressed[0]["pnl"] == 50.0
    assert stressed[1]["pnl"] == -75.0


def test_compare_summaries_requires_incremental_and_risk_quality() -> None:
    baseline = {
        "starting_cash": 10_000.0,
        "total_pnl": 100_000.0,
        "max_drawdown": -5_000.0,
        "max_drawdown_pct": -0.10,
        "worst_day_pnl": -2_000.0,
        "risk_of_ruin": False,
        "max_quantity": 1,
        "taken_trades": 10,
        "skipped_trades": 0,
        "total_contracts": 10,
    }
    candidate = {
        **baseline,
        "total_pnl": 130_000.0,
        "max_drawdown": -5_500.0,
        "max_drawdown_pct": -0.09,
        "worst_day_pnl": -2_200.0,
        "max_quantity": 3,
        "total_contracts": 14,
    }

    row = compare_summaries(baseline, candidate, 0.25)

    assert row["incremental_positive"] is True
    assert row["return_over_drawdown_improved"] is True
    assert row["worst_day_not_materially_worse"] is True


def test_evaluate_gate_rejects_negative_segment_even_if_aggregate_is_positive() -> None:
    validation = [
        {
            "incremental_positive": True,
            "candidate_risk_of_ruin": False,
            "return_over_drawdown_improved": True,
            "worst_day_not_materially_worse": True,
        }
    ]
    concentration = {
        "top_day_share_of_positive": 0.10,
        "top_month_share_of_positive": 0.20,
        "scaled_positive_fraction": 0.80,
    }

    gate = evaluate_gate(
        validation,
        [{"segment": "q3_2025", "incremental_pnl": -1.0}],
        concentration,
        {"total_pnl": 100.0},
        {"total_pnl": 200.0, "max_quantity": 2},
    )

    assert gate["passed"] is False
    assert "negative_incremental_segment" in gate["reasons"]


def test_evaluate_gate_passes_clean_multi_contract_candidate() -> None:
    validation = [
        {
            "incremental_positive": True,
            "candidate_risk_of_ruin": False,
            "return_over_drawdown_improved": True,
            "worst_day_not_materially_worse": True,
        }
    ]
    concentration = {
        "top_day_share_of_positive": 0.10,
        "top_month_share_of_positive": 0.20,
        "scaled_positive_fraction": 0.80,
    }

    gate = evaluate_gate(
        validation,
        [{"segment": "q3_2025", "incremental_pnl": 10.0}],
        concentration,
        {"total_pnl": 100.0},
        {"total_pnl": 200.0, "max_quantity": 3},
    )

    assert gate["passed"] is True
