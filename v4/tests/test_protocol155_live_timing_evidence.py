from __future__ import annotations

from pathlib import Path

from v4.live.paper_trade_log import append_trade_event, make_trade_log_event
from v4.scripts.run_protocol155_one_contract_live_timing_evidence import analyze_timing_evidence, decide


def _event(
    event_type: str,
    *,
    timestamp: str,
    trade_uid: str = "t1",
    quantity: int = 1,
    fill_price: float | None = None,
    action: str = "BUY",
    delay_quote_scenarios: dict | None = None,
) -> dict:
    order = {"action": action, "quantity": quantity, "limit_price": fill_price or 10.0}
    if fill_price is not None:
        order["avg_fill_price"] = fill_price
        order["filled"] = quantity
    return make_trade_log_event(
        event_type=event_type,
        timestamp=timestamp,
        session="2026-05-19",
        run_id="protocol155_test",
        trade_uid=trade_uid,
        mode="paper",
        selected_contract={
            "contract_id": "SPXW-20260519-06700.000-C",
            "symbol": "SPX",
            "root": "SPXW",
            "trading_class": "SPXW",
            "expiry": "20260519",
            "strike": 6700.0,
            "right": "C",
        },
        order=order,
        account={"account_id_redacted": "DU***45", "cash": 10_000.0, "equity": 10_000.0, "open_positions": 0},
        market_snapshot={
            "option_nbbo": {"bid": 9.9, "ask": 10.0, "bid_size": 20, "ask_size": 20, "quote_age_ms": 100},
            "context": {"context_age_ms": 100},
            "underlying": {
                "spx": 6700.0,
                "vix": 16.0,
                "spx_timestamp": timestamp,
                "vix_timestamp": timestamp,
            },
        },
        model_decision={"action": "enter", "score": 2.0, "threshold": 1.0},
        risk_gate={"passed": True, "reason": "pass"},
        broker_order_endpoint_called=event_type in {"paper_order_submitted", "paper_entry_fill", "paper_exit_fill"},
        extra={
            "timing": {
                "intended_entry_time": "2026-05-19T14:30:00+00:00",
                "entry_fill_at": "2026-05-19T14:30:01+00:00",
                "exit_decision_at": "2026-05-19T14:35:00+00:00",
                "exit_fill_at": "2026-05-19T14:35:01+00:00",
                "delay_quote_scenarios": delay_quote_scenarios or {},
            },
            "delay_quote_scenarios": delay_quote_scenarios or {},
        },
    )


def _closed_log(tmp_path: Path, *, quantity: int = 1, delay_scenarios: bool = True) -> Path:
    scenarios = (
        {
            "1": {"entry_ask": 10.01, "exit_bid": 12.0},
            "5": {"entry_ask": 10.05, "exit_bid": 11.95},
            "15": {"entry_ask": 10.10, "exit_bid": 11.90},
            "30": {"entry_ask": 10.20, "exit_bid": 11.80},
        }
        if delay_scenarios
        else {}
    )
    path = tmp_path / "session.jsonl"
    append_trade_event(
        path,
        _event(
            "model_decision",
            timestamp="2026-05-19T14:30:00+00:00",
            quantity=quantity,
            delay_quote_scenarios=scenarios,
        ),
    )
    append_trade_event(path, _event("paper_order_submitted", timestamp="2026-05-19T14:30:00.500000+00:00", quantity=quantity))
    append_trade_event(path, _event("paper_entry_fill", timestamp="2026-05-19T14:30:01+00:00", quantity=quantity, fill_price=10.0))
    append_trade_event(
        path,
        _event("paper_exit_fill", timestamp="2026-05-19T14:35:01+00:00", quantity=quantity, fill_price=12.0, action="SELL"),
    )
    return path


def test_protocol155_passes_complete_one_contract_timing_log(tmp_path: Path) -> None:
    path = _closed_log(tmp_path)
    from v4.live.paper_trade_log import load_trade_log

    analysis = analyze_timing_evidence(load_trade_log(path), trade_log=path)

    assert analysis["operational_checks"]["one_contract_operational_only"] is True
    assert analysis["timing_checks"]["entry_latency_inside_budget"] is True
    assert analysis["timing_checks"]["delay_stress_available"] is True
    assert analysis["replay_summary_rows"][0]["total_pnl"] == 200.0
    assert decide(analysis) == "pass_protocol155_live_timing_evidence_ready_for_multi_contract_review"


def test_protocol155_rejects_multi_contract_paper_rows(tmp_path: Path) -> None:
    path = _closed_log(tmp_path, quantity=2)
    from v4.live.paper_trade_log import load_trade_log

    analysis = analyze_timing_evidence(load_trade_log(path), trade_log=path)

    assert analysis["operational_checks"]["one_contract_operational_only"] is False
    assert decide(analysis) == "reject_protocol155_multi_contract_operational_violation"


def test_protocol155_blocks_when_delay_stress_missing(tmp_path: Path) -> None:
    path = _closed_log(tmp_path, delay_scenarios=False)
    from v4.live.paper_trade_log import load_trade_log

    analysis = analyze_timing_evidence(load_trade_log(path), trade_log=path)

    assert analysis["timing_checks"]["delay_stress_available"] is False
    assert decide(analysis) == "blocked_protocol155_timing_evidence_incomplete"
