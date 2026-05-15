from __future__ import annotations

from pathlib import Path

import pandas as pd

from v4.scripts.run_protocol152_multi_contract_promotion_gauntlet import (
    build_timing_rows,
    decide,
    evaluate_timing_gate,
    summarize_timing,
)


def test_build_timing_rows_applies_challenger_quantity(tmp_path: Path) -> None:
    timing_path = tmp_path / "timing.csv"
    pd.DataFrame(
        [
            {
                "split": "q3_2025",
                "seed": 1,
                "session": "2025-07-01",
                "candidate_uid": "u1",
                "contract_id": "SPXW-20250701-06000.000-C",
                "delay_seconds": 1,
                "status": "ok",
                "original_pnl": 100.0,
                "delayed_pnl": 80.0,
            }
        ]
    ).to_csv(timing_path, index=False)
    trades = [{"trade_number": 1, "candidate_uid": "u1"}]
    baseline = [{"trade_number": 1, "quantity": 1}]
    challenger = [{"trade_number": 1, "quantity": 3}]

    rows = build_timing_rows(
        trades=trades,
        baseline_rows=baseline,
        challenger_rows=challenger,
        timing_path=timing_path,
    )

    assert rows[0]["baseline_delayed_pnl"] == 80.0
    assert rows[0]["challenger_delayed_pnl"] == 240.0
    assert rows[0]["incremental_delayed_pnl"] == 160.0


def test_timing_gate_blocks_incomplete_coverage_without_policy_failure() -> None:
    summary = [
        {
            "split": "q3_2025",
            "delay_seconds": 1,
            "coverage": 0.90,
            "challenger_delayed_pnl": 100.0,
            "incremental_delayed_pnl": 10.0,
        },
        {
            "split": "q3_2025",
            "delay_seconds": 5,
            "coverage": 0.90,
            "challenger_delayed_pnl": 90.0,
            "incremental_delayed_pnl": 8.0,
        },
    ]

    gate = evaluate_timing_gate(summary)

    assert gate["passed_for_available_coverage"] is True
    assert gate["promotion_passed"] is False
    assert "incomplete_highres_timing_coverage" in gate["reasons"]


def test_decide_rejects_policy_timing_failure() -> None:
    base_gate = {"passed": True}
    timing_gate = {
        "passed_for_available_coverage": False,
        "promotion_coverage_passed": True,
    }

    assert decide(base_gate, timing_gate) == "reject_multi_contract_challenger_timing_fragile"


def test_summarize_timing_reports_split_incremental() -> None:
    rows = [
        {
            "split": "q3_2025",
            "delay_seconds": 1,
            "status": "ok",
            "baseline_delayed_pnl": 80.0,
            "challenger_delayed_pnl": 160.0,
            "incremental_delayed_pnl": 80.0,
            "baseline_original_pnl": 100.0,
            "challenger_original_pnl": 200.0,
            "incremental_original_pnl": 100.0,
            "is_scaled": True,
            "is_skipped": False,
        }
    ]

    summary = summarize_timing(rows)

    assert summary[0]["incremental_delayed_pnl"] == 80.0
    assert summary[0]["scaled_trades"] == 1
