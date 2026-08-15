"""Tests for fair-contract cross-attempt stability diagnostics."""
from __future__ import annotations

from v4.scripts.run_protocol101_fair_contract_stability_diagnostic import (
    build_bucket_metrics,
    build_stability_pairs,
    enrich_trade,
    pnl_metrics,
)


def test_pnl_metrics_uses_stressed_pnl_and_drawdown() -> None:
    rows = [
        {"stressed_pnl": "100"},
        {"stressed_pnl": "-40"},
        {"stressed_pnl": "60"},
    ]

    metrics = pnl_metrics(rows)

    assert metrics["trades"] == 3
    assert metrics["total_pnl"] == 120.0
    assert metrics["profit_factor"] == 4.0
    assert metrics["max_drawdown"] == -40.0


def test_enrich_trade_adds_causal_buckets() -> None:
    row = enrich_trade(
        {
            "split": "validation",
            "session": "2026-03-03",
            "decision_time": "2026-03-03T14:32:00+00:00",
            "right": "P",
            "offset": "15",
            "entry_ask": "3.25",
            "stressed_pnl": "50",
        },
        search_dir="memory://search",
        attempt_id="attempt_x",
        registry={"config": {"target_mode": "profit_classifier"}},
    )

    assert row["time_bucket"] == "open_0931_0959"
    assert row["offset_bucket"] == "near_10_20"
    assert row["premium_bucket"] == "3_to_7_5"
    assert row["right_time_bucket"] == "P_open_0931_0959"
    assert row["target_mode"] == "profit_classifier"


def test_stability_pairs_identify_stable_positive_and_split_flip() -> None:
    trades = [
        {
            "attempt_id": "a",
            "split": "validation",
            "time_bucket": "open",
            "right": "C",
            "offset_bucket": "near",
            "premium_bucket": "hi",
            "right_time_bucket": "C_open",
            "right_offset_bucket": "C_near",
            "stressed_pnl": "100",
        },
        {
            "attempt_id": "a",
            "split": "diagnostic_test",
            "time_bucket": "open",
            "right": "C",
            "offset_bucket": "near",
            "premium_bucket": "hi",
            "right_time_bucket": "C_open",
            "right_offset_bucket": "C_near",
            "stressed_pnl": "80",
        },
        {
            "attempt_id": "b",
            "split": "validation",
            "time_bucket": "open",
            "right": "P",
            "offset_bucket": "near",
            "premium_bucket": "hi",
            "right_time_bucket": "P_open",
            "right_offset_bucket": "P_near",
            "stressed_pnl": "80",
        },
        {
            "attempt_id": "b",
            "split": "diagnostic_test",
            "time_bucket": "open",
            "right": "P",
            "offset_bucket": "near",
            "premium_bucket": "hi",
            "right_time_bucket": "P_open",
            "right_offset_bucket": "P_near",
            "stressed_pnl": "-40",
        },
    ]

    buckets = build_bucket_metrics(trades)
    pairs = build_stability_pairs(buckets, min_bucket_trades=1, min_profit_factor=1.0)
    by_attempt = {
        row["attempt_id"]: row
        for row in pairs
        if row["dimension"] == "overall" and row["bucket"] == "all"
    }

    assert by_attempt["a"]["status"] == "stable_positive_candidate"
    assert by_attempt["b"]["status"] == "split_sign_flip"
