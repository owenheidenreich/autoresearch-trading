from __future__ import annotations

from pathlib import Path

from v4.live.paper_trade_log import (
    append_trade_event,
    export_trade_log_csv,
    executor_result_event,
    load_trade_log,
    make_trade_log_event,
    trade_log_path,
    validate_trade_event,
    validate_trade_log,
)


def _event(event_type: str = "paper_order_dry_run") -> dict:
    return make_trade_log_event(
        event_type=event_type,
        timestamp="2026-03-20T14:35:00+00:00",
        session="2026-03-20",
        run_id="test_run",
        trade_uid="t1",
        selected_contract={"symbol": "SPX", "root": "SPXW", "trading_class": "SPXW", "expiry": "20260320", "strike": 6700.0, "right": "C"},
        order={"action": "BUY", "quantity": 1, "limit_price": 10.0},
        account={"account_id_redacted": "DU***40", "cash": 10_000.0, "equity": 10_000.0, "open_positions": 0},
        market_snapshot={"option_nbbo": {"bid": 9.9, "ask": 10.0, "quote_age_ms": 100}, "underlying": {"spx": 6700.0, "vix": 16.0}},
        model_decision={"action": "enter", "score": 2.0, "threshold": 1.0},
        risk_gate={"passed": True, "reason": "pass"},
    )


def test_trade_log_path_partitions_by_session_and_run_id(tmp_path: Path) -> None:
    path = trade_log_path(root=tmp_path, session="2026-03-20", run_id="paper/run")

    assert path == tmp_path / "2026-03-20" / "paper_run.jsonl"


def test_append_load_validate_and_export_trade_log(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    append_trade_event(path, _event("model_decision"))
    append_trade_event(path, _event("paper_order_dry_run"))

    rows = load_trade_log(path)
    summary = validate_trade_log(rows)
    csv_summary = export_trade_log_csv(path, tmp_path / "log.csv")

    assert summary["status"] == "pass"
    assert summary["event_counts"]["paper_order_dry_run"] == 1
    assert csv_summary["rows"] == 2
    assert (tmp_path / "log.csv").exists()


def test_trade_log_validation_rejects_real_money_and_raw_account() -> None:
    event = _event()
    event["real_money_trading"] = True
    event["account_id"] = "DU12345"

    result = validate_trade_event(event)

    assert result.status == "fail"
    assert "real_money_trading must be false" in result.errors
    assert "raw account_id must not be logged" in result.errors


def test_executor_result_event_builds_analyzable_order_row() -> None:
    result = {
        "status": "dry_run_pass",
        "reason": "paper_order_validated_not_submitted",
        "broker_order_endpoint_called": False,
        "intent": {"action": "BUY", "symbol": "SPX", "expiry": "20260320", "strike": 6700.0, "right": "P", "quantity": 1, "limit_price": 8.5},
        "quote": {"bid": 8.4, "ask": 8.5, "quote_age_ms": 100},
        "context": {"context_age_ms": 100},
        "permission": {"passed": True, "reasons": []},
        "validation": {"passed": True, "reasons": [], "account_cash": 10_000.0, "open_positions": 0},
    }

    event = executor_result_event(result=result, run_id="run", mode="paper_executor")

    assert event["event_type"] == "paper_order_dry_run"
    assert event["selected_contract"]["right"] == "P"
    assert event["risk_gate"]["passed"] is True
    assert validate_trade_event(event).status == "pass"
