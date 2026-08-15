from __future__ import annotations

from pathlib import Path

import pandas as pd

from v4.live.paper_trade_log import load_trade_log, make_trade_log_event, validate_trade_event, validate_trade_log
from v4.scripts.run_protocol244_premium_blend_paper_runtime_shell import (
    PROTOCOL_ID,
    append_runtime_event,
    decide,
    validate_shell_order_intent,
)


def test_paper_log_accepts_challenger_protocol_id() -> None:
    row = make_trade_log_event(
        event_type="model_decision",
        protocol_id=PROTOCOL_ID,
        session="2026-05-20",
        run_id="test_challenger",
        account={"account_id_redacted": None, "cash": 10_000.0, "equity": 10_000.0},
        model_decision={"action": "wait"},
        risk_gate={"passed": True, "reason": "wait"},
    )

    assert validate_trade_event(row).status == "pass"


def test_paper_log_rejects_unknown_protocol_id() -> None:
    row = make_trade_log_event(
        event_type="model_decision",
        protocol_id="unknown_protocol",
        session="2026-05-20",
        run_id="test_unknown",
        account={"account_id_redacted": None, "cash": 10_000.0, "equity": 10_000.0},
    )

    result = validate_trade_event(row)
    assert result.status == "fail"
    assert any("protocol_id must be one of" in error for error in result.errors)


def test_protocol244_append_runtime_event_never_calls_broker(tmp_path: Path) -> None:
    log = tmp_path / "shell.jsonl"
    append_runtime_event(
        log,
        event_type="model_decision",
        session="2026-05-20",
        run_id="test_shell",
        paper_cash=10_000.0,
        reason="wait",
        timestamp="2026-05-20T14:00:00+00:00",
        model_decision={"action": "wait", "score": 0.0, "threshold": 1.0},
    )

    rows = load_trade_log(log)
    summary = validate_trade_log(rows)
    assert summary["status"] == "pass"
    assert summary["broker_order_endpoint_called_rows"] == 0
    assert rows[0]["protocol_id"] == PROTOCOL_ID
    assert rows[0]["timestamp"] == "2026-05-20T14:00:00+00:00"


def test_protocol244_validate_shell_order_intent_enforces_affordability() -> None:
    row = pd.Series(
        {
            "decision_dt": pd.Timestamp("2026-05-20T14:00:00Z"),
            "contract_id": "SPXW-20260520-07400.000-C",
            "entry_underlying_price": 7400.0,
            "offset": 0.0,
            "right": "C",
            "entry_ask": 125.0,
            "entry_bid": 124.5,
        }
    )

    result = validate_shell_order_intent(row, paper_cash=10_000.0, quantity=1)
    assert result["passed"] is False
    assert "insufficient_paper_cash" in result["reason"]


def test_protocol244_decide_blocks_broker_rows() -> None:
    result = {"decisions": 1}
    validation = {"status": "pass", "broker_order_endpoint_called_rows": 1}

    assert decide(result, validation) == "blocked_broker_endpoint_was_called"
