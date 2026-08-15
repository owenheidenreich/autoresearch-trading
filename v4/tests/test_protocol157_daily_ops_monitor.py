from __future__ import annotations

import json
from pathlib import Path

from v4.live.paper_trade_log import append_trade_event, load_trade_log, make_trade_log_event
from v4.scripts.run_protocol148_protocol101_post_session_analyzer import analyze_session_rows
from v4.scripts.run_protocol157_protocol101_daily_ops_monitor import (
    FIVEDAY_LAUNCHD_LABELS,
    build_monitor_payload,
    event_table,
    important_timeline_rows,
    latest_market_snapshot_summary,
    write_html,
)


def _event(event_type: str, *, extra: dict | None = None, order: dict | None = None) -> dict:
    return make_trade_log_event(
        event_type=event_type,
        timestamp="2026-05-19T14:31:00+00:00",
        session="2026-05-19",
        run_id="monitor_test",
        trade_uid="t1",
        selected_contract={
            "root": "SPXW",
            "trading_class": "SPXW",
            "expiry": "20260519",
            "strike": 7350.0,
            "right": "C",
        },
        order=order or {"action": "BUY", "quantity": 1, "limit_price": 10.0},
        account={
            "account_id_redacted": "DU***45",
            "starting_cash": 10_000.0,
            "cash": 10_000.0,
            "equity": 10_000.0,
            "realized_daily_pnl": 0.0,
            "open_positions": 0,
        },
        market_snapshot={
            "underlying": {"spx": 7350.0, "vix": 18.0},
            "option_nbbo": {"bid": 9.9, "ask": 10.0, "quote_age_ms": 100},
        },
        model_decision={"action": "enter", "score": 2.0, "threshold": 1.0},
        risk_gate={"passed": True, "reason": "pass"},
        broker_order_endpoint_called=event_type in {"paper_order_submitted", "paper_entry_fill", "paper_exit_fill"},
        extra=extra,
    )


def test_protocol157_monitor_summarizes_live_shadow_and_paper_pnl(tmp_path: Path) -> None:
    shadow = tmp_path / "live_router_shadow_observations.jsonl"
    shadow.write_text(
        json.dumps(
            {
                "decision_time": "2026-05-19T14:31:00+00:00",
                "contract_id": "SPXW-20260519-07350.000-C",
                "decision": {"action": "hold"},
                "position_state": "holding",
                "router": {"market_data_type": "live"},
                "context": {"spx": 7350.0},
                "order_intent": None,
            }
        )
        + "\n"
    )
    protocol147_dir = tmp_path / "protocol147" / "2026-05-19" / "monitor_test"
    capture_dir = protocol147_dir / "cycle_0000" / "live_capture"
    capture_dir.mkdir(parents=True)
    (capture_dir / "ibkr-live-capture_summary.json").write_text(
        json.dumps({"decision": "pass", "captured_rows": 1, "shadow_log": str(shadow)})
    )
    (protocol147_dir / "summary.json").write_text(
        json.dumps({"decision": "completed_no_order_analysis", "cycles_run": 1, "live_capture_passes": 1, "live_capture_blocks": 0})
    )
    log = tmp_path / "monitor_test.jsonl"
    append_trade_event(log, _event("heartbeat", extra={"shadow_log": str(shadow)}))
    append_trade_event(log, _event("paper_order_submitted"))
    entry = _event("paper_entry_fill")
    entry["order"]["avg_fill_price"] = 10.0
    entry["order"]["filled"] = 1
    exit_ = _event("paper_exit_fill", order={"action": "SELL", "quantity": 1, "limit_price": 12.0, "avg_fill_price": 12.0, "filled": 1})
    append_trade_event(log, entry)
    append_trade_event(log, exit_)
    rows = load_trade_log(log)
    analysis = analyze_session_rows(rows, trade_log=log)

    payload = build_monitor_payload(
        trade_log=log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=protocol147_dir,
        launchd={},
        entitlement={"decision": "pass", "ibkr_connected": True},
        runtime_flag={},
        ibkr_account_snapshot={
            "status": "pass",
            "source": "ibkr_account_summary",
            "checked_at_utc": "2026-05-19T20:05:00+00:00",
            "account_id_redacted": "DU***45",
            "paper_account_confirmed": True,
            "real_money_trading": False,
            "broker_order_endpoint_called": False,
            "values": {
                "net_liquidation": 26745.12,
                "cash": 25100.25,
                "realized_pnl": -30.0,
                "unrealized_pnl": 12.5,
                "available_funds": 24500.0,
                "buying_power": 98000.0,
                "currency": "USD",
            },
        },
    )

    assert payload["decision"] == "observe_paper_broker_activity_logged"
    assert payload["paper"]["paper_orders_submitted"] == 1
    assert payload["paper"]["reconstructed_closed_pnl"] == 200.0
    assert payload["shadow"]["live_rows"] == 1
    assert payload["shadow"]["latest_spx"] == 7350.0
    assert payload["shadow"]["latest_contracts_observed"][0]["contract_id"] == "SPXW-20260519-07350.000-C"
    assert payload["paper"]["contracts_traded"][0]["contract"] == "SPXW-20260519-7350.0-C"
    assert payload["paper"]["account_source"] == "ibkr_account_summary"
    assert payload["trader_status"]["account"]["source"] == "ibkr_account_summary"
    assert payload["trader_status"]["account"]["equity"] == 26745.12
    assert payload["trader_status"]["account"]["realized_daily_pnl"] == -30.0


def test_protocol157_resolves_stale_subscription_block_when_live_data_is_current(tmp_path: Path) -> None:
    shadow = tmp_path / "live_router_shadow_observations.jsonl"
    shadow.write_text(
        json.dumps(
            {
                "decision_time": "2026-05-19T14:31:00+00:00",
                "contract_id": "SPXW-20260519-07350.000-C",
                "decision": {"action": "hold"},
                "position_state": "holding",
                "router": {"market_data_type": "live"},
                "order_intent": None,
            }
        )
        + "\n"
    )
    protocol147_dir = tmp_path / "protocol147" / "2026-05-19" / "monitor_test"
    capture_dir = protocol147_dir / "cycle_0000" / "live_capture"
    capture_dir.mkdir(parents=True)
    (capture_dir / "ibkr-live-capture_summary.json").write_text(
        json.dumps({"decision": "pass", "captured_rows": 1, "shadow_log": str(shadow)})
    )
    (protocol147_dir / "summary.json").write_text(
        json.dumps({"decision": "completed_no_order_analysis", "cycles_run": 1, "live_capture_passes": 0, "live_capture_blocks": 1})
    )
    log = tmp_path / "monitor_test.jsonl"
    blocked = _event("risk_gate")
    blocked["risk_gate"]["reason"] = "blocked_live_subscriptions_delayed_plumbing_passed"
    append_trade_event(log, blocked)
    append_trade_event(log, _event("model_decision", extra={"shadow_log": str(shadow)}))
    rows = load_trade_log(log)
    analysis = analyze_session_rows(rows, trade_log=log)

    payload = build_monitor_payload(
        trade_log=log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=protocol147_dir,
        launchd={},
        entitlement={"decision": "pass", "ibkr_connected": True},
        runtime_flag={},
    )

    assert payload["failures"]["has_failures"] is False
    assert payload["failures"]["resolved_reason_counts"] == {"trade_log:blocked_live_subscriptions_delayed_plumbing_passed": 1}


def test_protocol157_resolves_initial_no_valid_nbbo_after_later_quotes(tmp_path: Path) -> None:
    log = tmp_path / "monitor_test.jsonl"
    blocked = _event("paper_order_blocked")
    blocked["risk_gate"] = {"passed": False, "reason": "no_valid_spxw_nbbo_quotes"}
    blocked["model_decision"] = {"action": "blocked", "reason": "no_valid_spxw_nbbo_quotes"}
    append_trade_event(log, blocked)
    snapshot = _event("market_snapshot")
    snapshot["market_snapshot"]["option_nbbo"] = {"quote_count": 12}
    append_trade_event(log, snapshot)
    candidate = _event("candidate_set", extra={"candidate_count": 0, "candidate_gate_diagnostics": {"filter_reason": "below_min_edge"}})
    candidate["risk_gate"] = {"passed": True, "reason": "candidate_set_built"}
    append_trade_event(log, candidate)
    rows = load_trade_log(log)
    analysis = analyze_session_rows(rows, trade_log=log)

    payload = build_monitor_payload(
        trade_log=log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=None,
        launchd={},
        entitlement={"decision": "pass", "ibkr_connected": True},
        runtime_flag={},
    )

    assert payload["failures"]["has_failures"] is False
    assert payload["failures"]["resolved_reason_counts"] == {"trade_log:no_valid_spxw_nbbo_quotes": 1}


def test_protocol157_uses_fiveday_launchd_profile_when_packet_is_active(tmp_path: Path) -> None:
    log = tmp_path / "monitor_test.jsonl"
    append_trade_event(log, _event("market_snapshot"))
    append_trade_event(log, _event("model_decision"))
    rows = load_trade_log(log)
    analysis = analyze_session_rows(rows, trade_log=log)
    launchd = {
        label: {"loaded": True, "state": "running" if "daily-monitor" in label else "not running", "last_exit_code": 0}
        for label in FIVEDAY_LAUNCHD_LABELS
    }
    launchd.update(
        {
            "com.autoresearch.ibgateway.paper": {"loaded": False, "state": None, "last_exit_code": None},
            "com.autoresearch.protocol101.paper-preflight": {"loaded": False, "state": None, "last_exit_code": None},
            "com.autoresearch.protocol101.paper-session": {"loaded": False, "state": None, "last_exit_code": None},
            "com.autoresearch.protocol101.daily-monitor": {"loaded": False, "state": None, "last_exit_code": None},
            "com.autoresearch.premiumblend.no-order-surface-check": {"loaded": False, "state": None, "last_exit_code": None},
        }
    )

    payload = build_monitor_payload(
        trade_log=log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=None,
        launchd=launchd,
        entitlement={"decision": "pass", "ibkr_connected": True},
        runtime_flag={},
    )

    assert payload["startup"]["launchd_profile"] == "fiveday"
    assert payload["startup"]["launchd_all_loaded"] is True
    assert payload["startup"]["launchd_loaded_count"] == len(FIVEDAY_LAUNCHD_LABELS)
    assert payload["failures"]["has_failures"] is False


def test_protocol157_live_feed_accepts_market_data_type_on_underlying_payload() -> None:
    row = _event("market_snapshot")
    row["market_snapshot"] = {
        "context": {"source": "ibkr_live", "context_age_ms": 0},
        "option_nbbo": {"quote_count": 42},
        "underlying": {
            "spx": 7500.0,
            "vix": 17.0,
            "spx_market_data_type": "live",
            "vix_market_data_type": "live",
        },
    }

    summary = latest_market_snapshot_summary([row])

    assert summary["is_live"] is True
    assert summary["spx_market_data_type"] == "live"
    assert summary["vix_market_data_type"] == "live"


def test_protocol157_resolves_stale_entitlement_block_from_current_live_paper_log(tmp_path: Path) -> None:
    log = tmp_path / "monitor_test.jsonl"
    market = _event("market_snapshot")
    market["mode"] = "paper-submit"
    market["market_snapshot"] = {
        "context": {"source": "ibkr_live", "context_age_ms": 0},
        "option_nbbo": {"quote_count": 42},
        "underlying": {
            "spx": 7560.44,
            "vix": 15.7,
            "spx_market_data_type": "live",
            "vix_market_data_type": "live",
        },
    }
    candidate = _event(
        "candidate_set",
        extra={
            "candidate_count": 0,
            "candidate_gate_diagnostics": {
                "filter_reason": "below_min_edge",
                "max_edge": -10.9,
                "min_edge": 25.0,
                "top_rejected_contracts": [
                    {"contract_id": "SPXW-20260604-07520.000-C"},
                    {"contract_id": "SPXW-20260604-07560.000-P"},
                ],
            },
            "live_index_context": {
                "row_count": 5318,
                "span_minutes": 91.8,
                "first_timestamp": "2026-06-04T13:30:12+00:00",
                "last_timestamp": "2026-06-04T15:02:00+00:00",
            },
        },
    )
    candidate["mode"] = "paper-submit"
    candidate["risk_gate"] = {"passed": True, "reason": "candidate_set_built"}
    decision = _event("model_decision")
    decision["mode"] = "paper-submit"
    decision["model_decision"] = {
        "action": "wait",
        "selected_action": "wait",
        "reason": "no_candidates",
        "threshold": -1.365,
        "no_entry_reason": "no_candidates",
        "action_mask": {"wait": True, "enter": False, "candidate_count": 0},
        "raw_logits": [],
        "candidate_logits": [],
        "wait_logit": None,
    }
    account = _event("paper_account_state")
    account["account"] = {
        "account_id_redacted": "DU***40",
        "starting_cash": 10000.0,
        "cash": 10035.51,
        "equity": 10035.51,
        "realized_daily_pnl": 0.0,
        "open_positions": 0,
    }
    for row in [market, candidate, decision, account]:
        append_trade_event(log, row)
    rows = load_trade_log(log)
    analysis = analyze_session_rows(rows, trade_log=log)

    payload = build_monitor_payload(
        trade_log=log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=None,
        launchd={
            "com.autoresearch.protocol101.fiveday.paper-preflight": {
                "loaded": True,
                "state": "not running",
                "last_exit_code": 1,
            }
        },
        entitlement={
            "decision": "blocked",
            "blocked_reason": "missing_live_market_data_entitlements",
            "ibkr_connected": True,
            "ibkr_port": 4002,
        },
        runtime_flag={},
    )

    assert payload["startup"]["effective_entitlement_decision"] == "pass_current_live_paper_log"
    assert payload["startup"]["entitlement_ready"] is True
    assert payload["startup"]["live_capture_pass"] is True
    assert payload["current_live_trade_log_evidence"]["live_context_ready"] is True
    assert payload["failures"]["has_failures"] is False
    assert payload["failures"]["resolved_reason_counts"] == {
        "entitlement:blocked": 1,
        "entitlement_blocked:missing_live_market_data_entitlements": 1,
        "launchd_last_exit_nonzero:com.autoresearch.protocol101.fiveday.paper-preflight": 1,
    }
    assert payload["trader_status"]["feed"] == "Live feed"
    assert payload["trader_status"]["status"] == "Flat and watching"
    assert payload["trader_status"]["latest_ladder"]["summary"] == "7520-7560 C/P around SPX unknown"


def test_protocol157_html_flags_broker_equity_delta_without_logged_trades(tmp_path: Path) -> None:
    log = tmp_path / "monitor_test.jsonl"
    append_trade_event(log, _event("market_snapshot"))
    append_trade_event(log, _event("model_decision"))
    rows = load_trade_log(log)
    analysis = analyze_session_rows(rows, trade_log=log)
    payload = build_monitor_payload(
        trade_log=log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=None,
        launchd={},
        entitlement={"decision": "pass", "ibkr_connected": True},
        runtime_flag={},
        ibkr_account_snapshot={
            "status": "pass",
            "source": "ibkr_account_summary",
            "checked_at_utc": "2026-05-27T16:37:18+00:00",
            "account_id_redacted": "DU***40",
            "paper_account_confirmed": True,
            "real_money_trading": False,
            "broker_order_endpoint_called": False,
            "values": {
                "net_liquidation": 10035.50,
                "cash": 10000.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "available_funds": 10000.0,
                "buying_power": 40000.0,
                "currency": "USD",
            },
        },
    )
    out = tmp_path / "daily_monitor.html"

    write_html(out, payload, rows)
    html = out.read_text()

    assert "Broker Equity Delta" in html
    assert "$35.50" in html
    assert "Broker equity moved without logged Protocol101 trades" in html
    assert "monitor_refresh=" in html
    assert "Cache-Control" in html


def test_protocol157_resolves_stale_entry_bridge_exception_after_later_clean_cycle(tmp_path: Path) -> None:
    protocol147_dir = tmp_path / "protocol147" / "2026-05-19" / "monitor_test"
    protocol147_dir.mkdir(parents=True)
    (protocol147_dir / "summary.json").write_text(
        json.dumps({"decision": "completed_no_order_analysis", "cycles_run": 2, "live_capture_passes": 1, "live_capture_blocks": 0})
    )
    log = tmp_path / "monitor_test.jsonl"
    blocked = _event("paper_order_blocked")
    blocked["mode"] = "paper-submit"
    blocked["model_decision"] = {"action": "wait", "reason": "protocol158_exception"}
    blocked["risk_gate"] = {"passed": False, "reason": "protocol158_exception"}
    append_trade_event(log, blocked)
    bridge_gate = _event("risk_gate")
    bridge_gate["risk_gate"] = {"passed": False, "reason": "entry_bridge_blocked_protocol158_exception"}
    append_trade_event(log, bridge_gate)
    clean = _event("candidate_set", extra={"candidate_gate_diagnostics": {"filter_reason": "below_min_edge"}})
    clean["mode"] = "paper-submit"
    append_trade_event(log, clean)
    rows = load_trade_log(log)
    analysis = analyze_session_rows(rows, trade_log=log)

    payload = build_monitor_payload(
        trade_log=log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=protocol147_dir,
        launchd={},
        entitlement={"decision": "pass", "ibkr_connected": True},
        runtime_flag={},
    )

    assert payload["failures"]["has_failures"] is False
    assert payload["trader_status"]["feed"] != "Blocked"
    assert payload["failures"]["resolved_reason_counts"] == {
        "trade_log:entry_bridge_blocked_protocol158_exception": 1,
        "trade_log:protocol158_exception": 1,
    }


def test_protocol157_resolves_stale_ibkr_connection_block_after_live_market_snapshot(tmp_path: Path) -> None:
    log = tmp_path / "monitor_test.jsonl"
    blocked = _event("paper_order_blocked")
    blocked["risk_gate"] = {"passed": False, "reason": "blocked_ibkr_connection"}
    blocked["model_decision"] = {"action": "blocked", "reason": "blocked_ibkr_connection"}
    append_trade_event(log, blocked)
    failed = _event("paper_error")
    failed["risk_gate"] = {"passed": False, "reason": "preflight_failed"}
    failed["model_decision"] = {"action": "blocked", "reason": "preflight_failed"}
    append_trade_event(log, failed)
    live = _event("market_snapshot")
    live["mode"] = "paper-submit"
    live["market_snapshot"]["context"] = {
        "source": "ibkr_live",
        "spx_market_data_type": "live",
        "vix_market_data_type": "live",
    }
    append_trade_event(log, live)
    rows = load_trade_log(log)
    analysis = analyze_session_rows(rows, trade_log=log)

    payload = build_monitor_payload(
        trade_log=log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=None,
        launchd={
            "com.autoresearch.ibgateway.paper": {"loaded": True, "state": "running", "last_exit_code": 1},
        },
        entitlement={"decision": "pass", "ibkr_connected": True},
        runtime_flag={},
    )

    assert payload["failures"]["has_failures"] is False
    assert payload["trader_status"]["feed"] != "Blocked"
    assert payload["failures"]["resolved_reason_counts"] == {
        "launchd_last_exit_nonzero:com.autoresearch.ibgateway.paper": 1,
        "trade_log:blocked_ibkr_connection": 1,
        "trade_log:preflight_failed": 1,
    }


def test_protocol157_action_timeline_filters_repetitive_wait_rows() -> None:
    wait_row = {"event_type": "risk_gate", "model_action": "wait", "risk_reason": "no_entry_intent"}
    trade_row = {"event_type": "paper_order_submitted", "model_action": "enter", "risk_reason": "pass"}

    assert important_timeline_rows([wait_row, trade_row], limit=10) == [trade_row]


def test_protocol157_action_timeline_filters_resolved_blockers() -> None:
    resolved_block = {"event_type": "paper_order_blocked", "model_action": "blocked", "risk_reason": "blocked_ibkr_connection"}
    active_block = {"event_type": "paper_order_blocked", "model_action": "blocked", "risk_reason": "runtime_flag_not_enabled"}

    rows = important_timeline_rows(
        [resolved_block, active_block],
        limit=10,
        resolved_reasons={"trade_log:blocked_ibkr_connection": 3},
    )

    assert rows == [active_block]


def test_protocol157_event_table_displays_pacific_time_not_raw_utc() -> None:
    html = event_table(
        [
            {
                "timestamp": "2026-05-20T13:30:19.419664+00:00",
                "event_type": "paper_order_submitted",
                "model_action": "enter",
                "risk_reason": "pass",
            }
        ],
        limit=10,
    )

    assert "06:30:19 PT" in html
    assert "2026-05-20T13:30" not in html


def test_protocol157_last_model_read_ignores_shadow_capture_plumbing(tmp_path: Path) -> None:
    log = tmp_path / "monitor_test.jsonl"
    append_trade_event(log, _event("market_snapshot"))
    paper_decision = _event("model_decision")
    paper_decision["mode"] = "paper-submit"
    append_trade_event(log, paper_decision)
    blocked = _event("model_decision")
    blocked["mode"] = "no-order-shadow"
    blocked["model_decision"] = {"action": "blocked", "reason": "live_capture_blocked"}
    blocked["risk_gate"] = {"passed": False, "reason": "live_capture_blocked"}
    append_trade_event(log, blocked)
    rows = load_trade_log(log)
    analysis = analyze_session_rows(rows, trade_log=log)

    payload = build_monitor_payload(
        trade_log=log,
        rows=rows,
        analysis=analysis,
        protocol147_dir=None,
        launchd={},
        entitlement={"decision": "pass", "ibkr_connected": True},
        runtime_flag={},
    )

    assert payload["trader_status"]["latest_model_decision"]["action"] == "enter"
    assert payload["trader_status"]["latest_model_decision"]["reason"] is None


def test_protocol157_html_includes_contracts_and_failures(tmp_path: Path) -> None:
    payload = {
        "session": "2026-05-19",
        "run_id": "monitor_test",
        "decision": "pass_live_shadow_monitor_ready",
        "next_action": "review",
        "startup": {
            "launchd_loaded_count": 3,
            "launchd_expected_count": 3,
            "entitlement_decision": "pass",
        },
        "paper": {
            "paper_orders_submitted": 0,
            "model_decision_rows": 0,
            "reconstructed_closed_pnl": 0.0,
            "broker_order_endpoint_called_rows": 0,
            "latest_account": {"open_positions": 0},
            "contracts_traded": [],
        },
        "shadow": {
            "live_rows": 10,
            "total_rows": 10,
            "delayed_rows": 0,
            "action_counts": {"hold": 10},
            "position_state_counts": {"holding": 10},
            "top_contracts_observed": [],
        },
        "failures": {"reason_counts": {}},
        "launchd": {},
    }
    path = tmp_path / "daily_monitor.html"

    write_html(path, payload, [])

    text = path.read_text()
    assert "Protocol101 Daily Monitor" in text
    assert "Contracts Bought/Sold" in text
    assert "Failures And Blockers" in text
