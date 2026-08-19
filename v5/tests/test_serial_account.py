"""Regression tests for the serial, compounding account.

Written 2026-08-16 after finding that `simulate_session` opened every session at
a constant $10,000 and measured the daily breaker against that constant. The
charter declares a serial account that carries the prior session's ending
balance, so re-seeding silently understated compounding and mis-sized the
breaker as equity moved.
"""
from __future__ import annotations

import inspect

import pytest

from v5.ops.causal_day_simulator import (
    STARTING_EQUITY_USD,
    SimulationResult,
    SimulatorError,
    simulate_serial_account,
    simulate_session,
)


def _result(session: str, *, equity: float, pnl: float) -> SimulationResult:
    import pandas as pd

    return SimulationResult(
        session=session,
        risk_mode="ticket_only",
        trade_cap=1,
        starting_equity_usd=equity,
        ending_cash_usd=equity + pnl,
        realised_pnl_usd=pnl,
        trades=pd.DataFrame(),
        events=pd.DataFrame(),
        considered_ladder=pd.DataFrame(),
        blocked_terminal_position=False,
        unresolved_position=None,
    )


def test_simulate_session_accepts_a_session_starting_equity() -> None:
    """The account must be an input, not a constant baked into the replay."""

    parameter = inspect.signature(simulate_session).parameters["starting_equity_usd"]
    assert parameter.default == STARTING_EQUITY_USD


def test_equity_compounds_across_sessions() -> None:
    seen: list[float] = []

    def replay(session: str, equity: float) -> SimulationResult:
        seen.append(equity)
        return _result(session, equity=equity, pnl=100.0)

    walk = simulate_serial_account(
        ["2024-01-02", "2024-01-03", "2024-01-04"], replay, starting_equity_usd=10_000.0
    )
    assert seen == [10_000.0, 10_100.0, 10_200.0]
    assert walk.ending_equity_usd == pytest.approx(10_300.0)
    assert not walk.ruined


def test_losses_carry_forward_too() -> None:
    def replay(session: str, equity: float) -> SimulationResult:
        return _result(session, equity=equity, pnl=-500.0)

    walk = simulate_serial_account(["2024-01-02", "2024-01-03"], replay)
    assert walk.ending_equity_usd == pytest.approx(STARTING_EQUITY_USD - 1000.0)


def test_the_account_must_never_re_seed() -> None:
    """The exact defect this file exists to prevent."""

    def re_seeding_replay(session: str, equity: float) -> SimulationResult:
        # Ignores the carried equity and restarts at the constant, which is what
        # simulate_session did unconditionally before 2026-08-16.
        return _result(session, equity=STARTING_EQUITY_USD, pnl=-900.0)

    with pytest.raises(SimulatorError, match="must not re-seed"):
        simulate_serial_account(["2024-01-02", "2024-01-03"], re_seeding_replay)


def test_walk_stops_at_the_survival_floor() -> None:
    def replay(session: str, equity: float) -> SimulationResult:
        return _result(session, equity=equity, pnl=-3000.0)

    walk = simulate_serial_account(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], replay
    )
    assert walk.ruined
    assert len(walk.sessions) == 2  # 10,000 -> 7,000 -> 4,000 breaches 50%


def test_sessions_must_be_chronological() -> None:
    def replay(session: str, equity: float) -> SimulationResult:
        return _result(session, equity=equity, pnl=0.0)

    with pytest.raises(SimulatorError, match="chronological"):
        simulate_serial_account(["2024-01-05", "2024-01-02"], replay)


def test_unresolved_terminal_still_carries_realised_pnl() -> None:
    """A blocked terminal must not silently restore the account."""

    import pandas as pd

    def replay(session: str, equity: float) -> SimulationResult:
        return SimulationResult(
            session=session,
            risk_mode="ticket_only",
            trade_cap=1,
            starting_equity_usd=equity,
            ending_cash_usd=None,
            realised_pnl_usd=-250.0,
            trades=pd.DataFrame(),
            events=pd.DataFrame(),
            considered_ladder=pd.DataFrame(),
            blocked_terminal_position=True,
            unresolved_position=None,
        )

    walk = simulate_serial_account(["2024-01-02", "2024-01-03"], replay)
    assert walk.ending_equity_usd == pytest.approx(STARTING_EQUITY_USD - 500.0)


def test_zero_or_negative_starting_equity_is_refused() -> None:
    with pytest.raises(SimulatorError, match="starting equity must be positive"):
        simulate_session(
            raw_quotes=None,  # never reached; the guard runs first
            session="2024-01-02",
            policy=None,
            trade_cap=1,
            starting_equity_usd=0.0,
        )
