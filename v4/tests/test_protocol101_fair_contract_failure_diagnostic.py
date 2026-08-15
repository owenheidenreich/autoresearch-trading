"""Tests for fair-contract failure diagnostic helpers."""
from __future__ import annotations

from datetime import datetime, timezone

from v4.scripts.run_protocol101_fair_contract_failure_diagnostic import (
    offset_bucket,
    pnl_metrics,
    premium_bucket,
    summarize_split,
    time_bucket,
)


def test_bucket_helpers_are_stable() -> None:
    assert time_bucket(datetime(2026, 3, 3, 14, 31, tzinfo=timezone.utc)) == "open_0931_0959"
    assert offset_bucket(0) == "atm_0_5"
    assert offset_bucket(25) == "mid_25_35"
    assert premium_bucket(0.75) == "lt_1"
    assert premium_bucket(4.5) == "3_to_7_5"


def test_pnl_metrics_and_summary_count_missed_profitable_decisions() -> None:
    selected = [
        {"label_net_pnl": 100.0, "time_bucket": "open_0931_0959", "right": "C"},
        {"label_net_pnl": -25.0, "time_bucket": "open_0931_0959", "right": "P"},
    ]
    decisions = [
        {"top_label_net_pnl": 150.0, "selected_count": 1},
        {"top_label_net_pnl": 80.0, "selected_count": 0},
        {"top_label_net_pnl": -10.0, "selected_count": 0},
    ]

    metrics = pnl_metrics(selected)
    summary = summarize_split(
        candidates=selected,
        decisions=decisions,
        selected=selected,
        profitable_label_threshold=20.0,
    )

    assert metrics["total_pnl"] == 75.0
    assert metrics["win_rate"] == 0.5
    assert summary["missed_profitable_decisions"] == 1
