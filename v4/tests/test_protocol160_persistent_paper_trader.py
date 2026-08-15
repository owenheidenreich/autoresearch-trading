from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from v4.live.paper_trade_log import load_trade_log
from v4.scripts.run_protocol160_protocol101_persistent_paper_trader import (
    append_decision_shadow,
    blocked_payload,
    clean_window_health_reasons,
    contract_refresh_reason,
    decide,
    entry_decision_interval_seconds,
    keep_running,
    paper_account_payload,
    session_deadline,
    should_evaluate_entry,
)


NY = ZoneInfo("America/New_York")


def _args(tmp_path: Path, *, mode: str = "paper-submit", skip_market_clock: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        paper_cash=10_000.0,
        skip_market_clock=skip_market_clock,
        entry_decision_mode="minute",
        entry_decision_interval_seconds=60.0,
        decision_interval_seconds=None,
    )


def test_protocol160_contract_ladder_refreshes_initially() -> None:
    reason = contract_refresh_reason(
        spx=6000.0,
        ladder_atm=None,
        last_refresh_monotonic=0.0,
        refresh_seconds=60.0,
        drift_points=15.0,
    )

    assert reason == "initial_ladder"


def test_protocol160_contract_ladder_refreshes_on_atm_drift() -> None:
    reason = contract_refresh_reason(
        spx=6020.0,
        ladder_atm=6000,
        last_refresh_monotonic=999999999.0,
        refresh_seconds=3600.0,
        drift_points=15.0,
    )

    assert reason == "atm_drift"


def test_protocol160_decision_modes_are_fail_closed() -> None:
    assert (
        decide(
            mode="paper-submit",
            decisions=[],
            executor_results=[],
            paper_submit_blocked=0,
            broker_order_endpoint_called=False,
        )
        == "blocked_no_persistent_live_decisions_emitted"
    )
    assert (
        decide(
            mode="intent-shadow",
            decisions=[{"action": "wait"}],
            executor_results=[],
            paper_submit_blocked=0,
            broker_order_endpoint_called=False,
        )
        == "pass_persistent_live_intent_shadow_logged"
    )
    assert (
        decide(
            mode="paper-submit",
            decisions=[{"action": "enter"}],
            executor_results=[],
            paper_submit_blocked=1,
            broker_order_endpoint_called=False,
        )
        == "blocked_persistent_paper_submit_runtime_flag_missing"
    )
    assert (
        decide(
            mode="paper-submit",
            decisions=[{"action": "wait"}],
            executor_results=[],
            paper_submit_blocked=0,
            broker_order_endpoint_called=False,
        )
        == "pass_persistent_no_entry_intents_to_paper_submit"
    )


def test_protocol160_blocked_payload_writes_valid_trade_log(tmp_path: Path) -> None:
    log = tmp_path / "session.jsonl"

    payload = blocked_payload(
        _args(tmp_path),
        session="2026-05-20",
        run_id="test_persistent",
        trade_log=log,
        reason="outside_regular_market_hours",
    )
    rows = load_trade_log(log)

    assert payload["decision"] == "blocked_outside_regular_market_hours"
    assert rows[0]["event_type"] == "paper_order_blocked"
    assert rows[0]["broker_order_endpoint_called"] is False


def test_protocol160_keep_running_respects_market_close() -> None:
    args = _args(Path("."), skip_market_clock=False)
    past_close = datetime(2000, 1, 1, 16, 0, tzinfo=NY)

    assert keep_running(args, close_deadline=past_close) is False


def test_protocol160_session_deadline_uses_new_york_wall_time() -> None:
    now = datetime(2026, 5, 20, 12, 0, tzinfo=NY)

    assert session_deadline(now, "15:55").isoformat() == "2026-05-20T15:55:00-04:00"


def test_protocol160_minute_mode_emits_only_once_per_minute(tmp_path: Path) -> None:
    args = _args(tmp_path)
    first = datetime(2026, 5, 20, 14, 31, 2, tzinfo=ZoneInfo("UTC"))

    ok, next_mono, minute_key = should_evaluate_entry(
        args=args,
        now_utc=first,
        next_decision_monotonic=0.0,
        last_entry_decision_minute=None,
    )
    again, _, same_minute = should_evaluate_entry(
        args=args,
        now_utc=datetime(2026, 5, 20, 14, 31, 45, tzinfo=ZoneInfo("UTC")),
        next_decision_monotonic=next_mono,
        last_entry_decision_minute=minute_key,
    )
    next_minute, _, _ = should_evaluate_entry(
        args=args,
        now_utc=datetime(2026, 5, 20, 14, 32, 0, tzinfo=ZoneInfo("UTC")),
        next_decision_monotonic=next_mono,
        last_entry_decision_minute=same_minute,
    )

    assert ok is True
    assert again is False
    assert next_minute is True


def test_protocol160_interval_seconds_prefers_explicit_entry_interval(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.entry_decision_interval_seconds = 60.0
    args.decision_interval_seconds = 5.0

    assert entry_decision_interval_seconds(args) == 60.0


def test_protocol160_account_payload_records_redacted_equity() -> None:
    class Row:
        def __init__(self, tag: str, value: str, account: str = "DU12340", currency: str = "USD") -> None:
            self.tag = tag
            self.value = value
            self.account = account
            self.currency = currency

    class FakeIb:
        def accountSummary(self, account: str = "") -> list[Row]:
            return [
                Row("NetLiquidation", "10035.50"),
                Row("TotalCashValue", "10000.00"),
                Row("AvailableFunds", "10000.00"),
                Row("BuyingPower", "40000.00"),
                Row("RealizedPnL", "0.00"),
                Row("UnrealizedPnL", "0.00"),
                Row("GrossPositionValue", "0.00"),
            ]

        def accountValues(self, account: str = "") -> list[Row]:
            return []

    payload = paper_account_payload(FakeIb(), account_id="DU12340", open_positions=0, fallback_cash=10_000.0)

    assert payload["account_id_redacted"] == "DU***40"
    assert payload["cash"] == 10_000.0
    assert payload["equity"] == 10035.5
    assert payload["paper_account_confirmed"] is True
    assert "DU12340" not in str(payload)


def test_protocol160_clean_window_health_requires_fresh_complete_live_ladder() -> None:
    class Ticker:
        marketDataType = 1

    quotes = [
        {
            "strike": float(6000 + offset),
            "right": right,
            "quote_age_ms": 1_000.0,
        }
        for offset in range(-50, 51, 5)
        for right in ("C", "P")
    ]
    context = {
        "opening_context_ready": True,
        "span_minutes": 30.0,
        "minute_row_count": 31,
    }

    assert (
        clean_window_health_reasons(
            option_quotes=quotes,
            spx=6000.0,
            spx_ticker=Ticker(),
            vix_ticker=Ticker(),
            context_summary=context,
            min_context_minutes=30.0,
        )
        == ()
    )

    quotes[0]["quote_age_ms"] = 90_001.0
    assert "stale_quote" in clean_window_health_reasons(
        option_quotes=quotes,
        spx=6000.0,
        spx_ticker=Ticker(),
        vix_ticker=Ticker(),
        context_summary=context,
        min_context_minutes=30.0,
    )


def test_protocol160_decision_shadow_serializes_canonical_arrays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decision_shadow.jsonl"
    append_decision_shadow(
        path,
        session="2026-07-25",
        run_id="shadow-test",
        mode="intent-shadow",
        timestamp=datetime(2026, 7, 25, 14, 32, tzinfo=ZoneInfo("UTC")),
        action="wait",
        reason="outside_time_bucket",
        canonical_input={
            "option_ladder": np.asarray([[1.0, np.nan]]),
            "candidate_mask": np.asarray([[True, False]]),
        },
    )

    row = json.loads(path.read_text().strip())
    assert row["schema_version"] == "Protocol101DecisionShadowV1"
    assert row["canonical_input"]["option_ladder"] == [[1.0, None]]
    assert row["broker_order_endpoint_called"] is False
