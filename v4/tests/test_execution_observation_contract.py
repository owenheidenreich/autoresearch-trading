from __future__ import annotations

import pandas as pd

from v4.foundation.execution_observation_contract import BLOCKED, PASS, coverage_for_frame, validate_observation


def _row() -> dict:
    return {
        "decision_timestamp": "2026-05-26T14:30:00+00:00",
        "raw_quote_timestamp": "2026-05-26T14:29:59.800000+00:00",
        "received_timestamp": "2026-05-26T14:30:00+00:00",
        "quote_age_ms": 200,
        "bid": 10.0,
        "ask": 10.2,
        "bid_size": 5,
        "ask_size": 4,
        "spread": 0.2,
        "premium": 1020.0,
        "side": "C",
        "moneyness": 0.0,
        "time_bucket": "post_open_morning",
        "intended_ask_entry": 10.2,
        "submitted_limit": 10.2,
        "fill_status": "filled",
        "cancel_status": "not_cancelled",
        "timeout_status": "not_timeout",
        "latency_ms": 150,
        "exit_bid": 11.0,
        "post_fill_pnl": 80.0,
    }


def test_execution_observation_contract_accepts_complete_packet() -> None:
    assert validate_observation(_row()) == {"status": PASS, "errors": []}


def test_execution_observation_contract_rejects_missing_quote_time() -> None:
    row = _row()
    row.pop("raw_quote_timestamp")

    result = validate_observation(row)

    assert result["status"] == BLOCKED
    assert "missing_field:raw_quote_timestamp" in result["errors"]


def test_execution_observation_coverage_reports_missing_fields() -> None:
    frame = pd.DataFrame([{"timestamp": "2026-05-26T14:30:00+00:00", "bid": 10.0, "ask": 10.2}])

    result = coverage_for_frame(frame)

    assert result["status"] == BLOCKED
    assert "raw_quote_timestamp" in result["missing_fields"]
    assert result["field_coverage"]["decision_timestamp"] == 1.0
