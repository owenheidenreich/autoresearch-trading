from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from v4.live.ibkr_paper_guard import PaperOrderIntent
from v4.live.paper_trade_log import load_trade_log
from v4.live.protocol101_live_entry import LiveIndexState, selected_contract_payload
from v4.scripts.run_protocol158_protocol101_live_entry_paper_bridge import (
    append_live_index_context,
    append_time_bucket_block,
    blocked_payload,
    decide,
    is_blocking_reason,
    live_context_ready,
    load_live_index_context,
    quote_freshness_from_ticker,
    runtime_flag_summary,
    validate_intent,
    variant_for,
)


def _args(tmp_path: Path, *, mode: str = "intent-shadow") -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        paper_cash=10_000.0,
        open_positions=0,
        trade_log_root=tmp_path,
        min_edge=25.0,
        min_live_context_minutes=30.0,
    )


def test_protocol158_variant_for_finds_frozen_surface_variant() -> None:
    variant = variant_for("surface_structure_aplus_side_value_rank")

    assert variant.action_space == "surface"
    assert variant.market_mode == "structure"
    assert variant.token_mode == "aplus"


def test_protocol158_decision_modes_are_fail_closed() -> None:
    assert decide(args=_args(Path(".")), decisions=[], executor_results=[], paper_submit_blocked=0) == "blocked_no_live_entry_decisions_emitted"
    assert (
        decide(
            args=_args(Path("."), mode="intent-shadow"),
            decisions=[{"action": "wait"}],
            executor_results=[],
            paper_submit_blocked=0,
        )
        == "pass_live_entry_intent_shadow_logged"
    )
    assert (
        decide(
            args=_args(Path("."), mode="paper-submit"),
            decisions=[{"action": "enter"}],
            executor_results=[],
            paper_submit_blocked=1,
        )
        == "blocked_paper_submit_runtime_flag_missing"
    )
    assert (
        decide(
            args=_args(Path("."), mode="paper-submit"),
            decisions=[{"action": "wait"}],
            executor_results=[],
            paper_submit_blocked=0,
        )
        == "pass_no_entry_intents_to_paper_submit"
    )


def test_protocol158_blocked_payload_writes_valid_trade_log(tmp_path: Path) -> None:
    log = tmp_path / "session.jsonl"

    payload = blocked_payload(
        _args(tmp_path),
        session="2026-05-19",
        run_id="test",
        trade_log=log,
        reason="outside_regular_market_hours",
    )
    rows = load_trade_log(log)

    assert payload["decision"] == "blocked_outside_regular_market_hours"
    assert rows[0]["event_type"] == "paper_order_blocked"
    assert rows[0]["broker_order_endpoint_called"] is False


def test_protocol158_runtime_flag_summary_redacts_shape() -> None:
    summary = runtime_flag_summary({"paper_orders_enabled": True, "account_id_redacted": "DU***45"})

    assert summary["paper_orders_enabled"] is True
    assert summary["account_id_redacted"] == "DU***45"


def test_protocol158_validate_intent_passes_when_no_entry() -> None:
    result = validate_intent(intent=None, selected_contract={}, args=_args(Path(".")))

    assert result["passed"] is True
    assert result["reason"] == "no_entry_intent"


def test_protocol158_quote_freshness_is_derived_from_ticker_time() -> None:
    observed = datetime(2026, 5, 24, 14, 30, 0, 500000, tzinfo=timezone.utc)
    ticker = SimpleNamespace(time=datetime(2026, 5, 24, 14, 30, 0, tzinfo=timezone.utc))

    freshness = quote_freshness_from_ticker(ticker, observed_at=observed)

    assert freshness["quote_age_ms"] == 500
    assert freshness["quote_age_source"] == "time"
    assert freshness["quote_timestamp"] == "2026-05-24T14:30:00+00:00"
    assert freshness["received_timestamp"] == "2026-05-24T14:30:00.500000+00:00"


def test_protocol158_unknown_quote_time_fails_intent_validation_closed() -> None:
    intent = PaperOrderIntent(action="BUY", symbol="SPX", expiry="20260524", strike=6000.0, right="C", quantity=1, limit_price=10.0)
    selected_contract = {
        "symbol": "SPX",
        "trading_class": "SPXW",
        "expiry": "20260524",
        "strike": 6000.0,
        "right": "C",
        "exchange": "SMART",
        "currency": "USD",
        "bid": 9.9,
        "ask": 10.0,
        "quote_age_ms": None,
    }

    result = validate_intent(intent=intent, selected_contract=selected_contract, args=_args(Path(".")))

    assert result["passed"] is False
    assert "missing_quote_age" in result["reasons"]


def test_protocol101_selected_contract_payload_preserves_quote_freshness() -> None:
    intent = PaperOrderIntent(action="BUY", symbol="SPX", expiry="20260524", strike=6000.0, right="P", quantity=1, limit_price=12.5)
    lookup = {
        "SPXW_20260524_6000_P": {
            "expiry": "20260524",
            "strike": 6000.0,
            "right": "P",
            "bid": 12.4,
            "ask": 12.5,
            "quote_age_ms": 375,
            "quote_timestamp": "2026-05-24T14:30:00+00:00",
            "received_timestamp": "2026-05-24T14:30:00.375000+00:00",
        }
    }

    payload = selected_contract_payload(intent, lookup)

    assert payload["contract_id"] == "SPXW_20260524_6000_P"
    assert payload["bid"] == 12.4
    assert payload["ask"] == 12.5
    assert payload["quote_age_ms"] == 375
    assert payload["quote_timestamp"] == "2026-05-24T14:30:00+00:00"


def test_protocol158_default_blocking_reasons_fail_risk_gate() -> None:
    assert is_blocking_reason("protocol158_exception") is True
    assert is_blocking_reason("missing_live_spx_or_vix") is True
    assert is_blocking_reason("outside_regular_market_hours") is True
    assert is_blocking_reason("no_entry_intent") is False


def test_protocol158_live_context_ready_requires_minute_span() -> None:
    assert live_context_ready({"span_minutes": 30.0, "minute_row_count": 30}, min_context_minutes=30.0) is True
    assert live_context_ready({"span_minutes": 29.9, "minute_row_count": 30}, min_context_minutes=30.0) is False
    assert live_context_ready({"span_minutes": 30.0, "minute_row_count": 29}, min_context_minutes=30.0) is False


def test_protocol158_live_context_ready_requires_opening_context_when_reported() -> None:
    assert (
        live_context_ready(
            {
                "span_minutes": 60.0,
                "minute_row_count": 60,
                "opening_context_ready": False,
                "missing_opening_minutes": 9,
            },
            min_context_minutes=30.0,
        )
        is False
    )
    assert (
        live_context_ready(
            {
                "span_minutes": 60.0,
                "minute_row_count": 60,
                "opening_context_ready": True,
            },
            min_context_minutes=30.0,
        )
        is True
    )


def test_protocol158_outside_time_bucket_block_takes_precedence(tmp_path: Path) -> None:
    log = tmp_path / "session.jsonl"

    append_time_bucket_block(
        _args(tmp_path),
        log,
        "2026-01-02",
        "test_run",
        0.42,
        datetime(2026, 1, 2, 14, 45, tzinfo=timezone.utc),
        {"span_minutes": 10.0, "minute_row_count": 10},
        option_quotes=[{"contract_id": "C1"}],
    )
    rows = load_trade_log(log)
    candidate = next(row for row in rows if row["event_type"] == "candidate_set")
    decision = next(row for row in rows if row["event_type"] == "model_decision")

    assert candidate["model_decision"]["reason"] == "candidate_set_blocked_outside_time_bucket"
    assert candidate["candidate_gate_diagnostics"]["filter_reason"] == "outside_time_bucket"
    assert candidate["candidate_gate_diagnostics"]["context_ready"] is False
    assert decision["model_decision"]["no_entry_reason"] == "outside_time_bucket"


def test_protocol158_live_index_context_persists_same_session(tmp_path: Path) -> None:
    path = tmp_path / "context.jsonl"
    append_live_index_context(path, session="2026-05-19", timestamp=datetime(2026, 5, 19, 15, 0, tzinfo=timezone.utc), spx=6000.0, vix=18.0)
    append_live_index_context(path, session="2026-05-20", timestamp=datetime(2026, 5, 20, 15, 0, tzinfo=timezone.utc), spx=6100.0, vix=19.0)
    state = LiveIndexState()

    loaded = load_live_index_context(path, state, session="2026-05-19")

    assert loaded == 1
    assert len(state.rows) == 1
    assert state.rows[0]["spx"] == 6000.0


def test_protocol158_live_index_context_can_seed_prior_session_close(tmp_path: Path) -> None:
    path = tmp_path / "context.jsonl"
    append_live_index_context(path, session="2026-05-19", timestamp=datetime(2026, 5, 19, 20, 0, tzinfo=timezone.utc), spx=6000.0, vix=18.0)
    append_live_index_context(path, session="2026-05-20", timestamp=datetime(2026, 5, 20, 13, 30, tzinfo=timezone.utc), spx=6100.0, vix=19.0)
    state = LiveIndexState()

    loaded = load_live_index_context(path, state, session="2026-05-20", include_prior_session_close=True)

    assert loaded == 2
    assert state.previous_session_close(datetime(2026, 5, 20, 14, 0, tzinfo=timezone.utc)) == 6000.0
