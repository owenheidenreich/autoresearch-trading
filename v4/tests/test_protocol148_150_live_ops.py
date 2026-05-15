from __future__ import annotations

from pathlib import Path

from v4.live.paper_trade_log import append_trade_event, flatten_trade_event, load_trade_log, make_trade_log_event
from v4.scripts.run_protocol148_protocol101_post_session_analyzer import analyze_session_rows
from v4.scripts.run_protocol149_protocol101_live_log_visual import write_html
from v4.scripts.run_protocol150_protocol101_paper_order_enablement_gate import evaluate_gate


def _base_event(event_type: str, *, tmp_path: Path, reason: str = "pass", extra: dict | None = None) -> dict:
    return make_trade_log_event(
        event_type=event_type,
        timestamp="2026-05-15T14:30:00+00:00",
        session="2026-05-15",
        run_id="test_session",
        trade_uid="t1",
        selected_contract={
            "symbol": "SPX",
            "root": "SPXW",
            "trading_class": "SPXW",
            "expiry": "20260515",
            "strike": 6700.0,
            "right": "C",
        },
        order={"action": "BUY", "quantity": 1, "limit_price": 10.0},
        account={"account_id_redacted": "DU***40", "cash": 10_000.0, "equity": 10_000.0, "open_positions": 0},
        market_snapshot={
            "option_nbbo": {"bid": 9.9, "ask": 10.0, "quote_age_ms": 100},
            "underlying": {"spx": 6700.0, "vix": 16.0},
        },
        model_decision={"action": "wait", "score": 2.0, "threshold": 1.0},
        risk_gate={"passed": reason == "pass", "reason": reason},
        extra=extra,
    )


def test_protocol148_detects_live_parity_and_fill_pnl(tmp_path: Path) -> None:
    parity_log = tmp_path / "parity.out.log"
    capture_log = tmp_path / "capture.out.log"
    parity_log.write_text('{"decision":"ready_for_protocol101_no_order_live_capture","status":"port_open","ibkr_connected":true}\n')
    capture_log.write_text('{"decision":"pass","captured_rows":3}\n')
    trade_log = tmp_path / "session.jsonl"
    append_trade_event(
        trade_log,
        _base_event(
            "risk_gate",
            tmp_path=tmp_path,
            extra={
                "protocol124": {"name": "run_protocol124_protocol101_live_data_parity_checkpoint", "returncode": 0, "stdout_log": str(parity_log)},
                "live_capture": {"name": "run_protocol081_live_shadow_router", "returncode": 0, "stdout_log": str(capture_log)},
            },
        ),
    )
    entry = _base_event("paper_entry_fill", tmp_path=tmp_path)
    entry["order"]["avg_fill_price"] = 10.0
    entry["order"]["filled"] = 1
    exit_ = _base_event("paper_exit_fill", tmp_path=tmp_path)
    exit_["order"] = {"action": "SELL", "quantity": 1, "limit_price": 12.0, "avg_fill_price": 12.0, "filled": 1}
    append_trade_event(trade_log, entry)
    append_trade_event(trade_log, exit_)

    analysis = analyze_session_rows(load_trade_log(trade_log), trade_log=trade_log)

    assert analysis["startup_and_data"]["live_parity_ready"] is True
    assert analysis["startup_and_data"]["live_capture_pass"] is True
    assert analysis["paper_fill_pnl"]["total_pnl"] == 200.0


def test_protocol149_writes_separate_live_dashboard(tmp_path: Path) -> None:
    trade_log = tmp_path / "session.jsonl"
    row = _base_event("heartbeat", tmp_path=tmp_path)
    append_trade_event(trade_log, row)
    analysis = analyze_session_rows([row], trade_log=trade_log)
    html_path = tmp_path / "live_session.html"

    write_html(html_path, {"analysis": analysis, "launchd": {}}, [flatten_trade_event(row)])

    text = html_path.read_text()
    assert "Protocol101 Live Session" in text
    assert "Event Timeline" in text
    assert "heartbeat" in text


def test_protocol150_blocks_until_permission_and_live_capture_pass() -> None:
    analysis = {
        "validation": {"status": "pass"},
        "broker_order_endpoint_called_rows": 0,
        "startup_and_data": {"live_capture_pass": True, "live_parity_ready": True},
    }
    blocked = evaluate_gate(analysis, {"passed": False, "reasons": ["paper_order_env_not_set"]})
    allowed = evaluate_gate(analysis, {"passed": True, "reasons": []})

    assert blocked["passed"] is False
    assert "paper_order_env_not_set" in blocked["reasons"]
    assert allowed["passed"] is True
