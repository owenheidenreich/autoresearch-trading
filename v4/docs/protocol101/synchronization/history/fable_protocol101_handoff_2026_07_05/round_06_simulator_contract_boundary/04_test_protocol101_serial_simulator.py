"""Tests for the canonical Protocol101 serial simulator contract."""
from __future__ import annotations

from datetime import datetime, timezone

from v4.model.protocol101_serial_simulator import (
    CASH_BASIS,
    DAILY_LOSS_BASIS,
    PROTOCOL101_SERIAL_SIMULATOR_VERSION,
    STRESS_APPLICATION,
    SerialCandidate,
    SerialSimulatorConfig,
    simulate_serial_candidates,
)


def _candidate(
    *,
    minute: int,
    raw_pnl: float,
    ask: float = 3.0,
    session: str = "2026-01-02",
) -> SerialCandidate:
    return SerialCandidate(
        split="validation",
        session=session,
        decision_time=datetime(2026, 1, 2, 14, minute, tzinfo=timezone.utc),
        contract_id=f"SPXW-{session}-06500.000-P",
        right="P",
        offset=-20.0,
        entry_ask=ask,
        score=1.0,
        raw_label_pnl=raw_pnl,
        cooldown_minutes=1.0,
        feature_hash="a" * 64,
        source_quote_time=f"2026-01-02T14:{minute:02d}:00+00:00",
        source_context_time=f"2026-01-02T14:{minute - 1:02d}:00+00:00",
    )


def test_simulator_contract_metadata_is_explicit() -> None:
    config = SerialSimulatorConfig()

    assert config.simulator_version == PROTOCOL101_SERIAL_SIMULATOR_VERSION
    assert config.daily_loss_basis == DAILY_LOSS_BASIS == "raw_realized_net_pnl"
    assert config.cash_basis == CASH_BASIS == "raw_realized_net_pnl"
    assert config.stress_application == STRESS_APPLICATION == "metrics_only"


def test_daily_loss_stop_uses_raw_realized_pnl_not_stressed_metrics() -> None:
    trades, state = simulate_serial_candidates(
        [
            _candidate(minute=32, raw_pnl=-480.0),
            _candidate(minute=34, raw_pnl=-10.0),
            _candidate(minute=36, raw_pnl=100.0),
        ],
        config=SerialSimulatorConfig(
            max_daily_loss=500.0,
            stress_per_trade=20.0,
        ),
    )

    assert [trade.raw_label_pnl for trade in trades] == [-480.0, -10.0, 100.0]
    assert [trade.stressed_pnl for trade in trades] == [-500.0, -30.0, 80.0]
    assert state.skipped["daily_loss_stop"] == 0
    assert state.realized_raw_pnl_by_split_session["validation:2026-01-02"] == -390.0


def test_daily_loss_stop_blocks_after_raw_loss_threshold_is_crossed() -> None:
    trades, state = simulate_serial_candidates(
        [
            _candidate(minute=32, raw_pnl=-600.0),
            _candidate(minute=34, raw_pnl=900.0),
            _candidate(minute=36, raw_pnl=900.0),
        ],
        config=SerialSimulatorConfig(max_daily_loss=500.0),
    )

    assert [trade.raw_label_pnl for trade in trades] == [-600.0]
    assert state.skipped["daily_loss_stop"] == 2


def test_cash_affordability_uses_raw_realized_pnl_not_stressed_metrics() -> None:
    trades, state = simulate_serial_candidates(
        [
            _candidate(minute=32, raw_pnl=-480.0, ask=100.0),
            _candidate(minute=34, raw_pnl=50.0, ask=95.1),
        ],
        config=SerialSimulatorConfig(
            starting_cash=10_000.0,
            contract_multiplier=100.0,
            stress_per_trade=500.0,
        ),
    )

    assert len(trades) == 2
    assert trades[0].cash_after == 9_520.0
    assert trades[1].cash_before == 9_520.0
    assert state.skipped["unaffordable"] == 0

