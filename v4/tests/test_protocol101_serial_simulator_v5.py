from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from v4.model.protocol101_regimen_repair import ExitReason
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION,
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    Protocol101LegacySyntheticExitArtifactError,
    SerialCandidateV5,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)


def _ns(session: str, hhmm: str) -> int:
    clock = hhmm if hhmm.count(":") == 2 else f"{hhmm}:00"
    return int(
        pd.Timestamp(
            f"{session} {clock}", tz="America/New_York"
        ).tz_convert("UTC").value
    )


def _candidate(
    *,
    session: str = "2025-01-02",
    decision: str = "10:00",
    source: str = "10:01",
    realized: str = "10:01",
    deadline: str = "10:10",
    ask: float = 2.0,
    bid: float = 2.1,
    reason: ExitReason = ExitReason.TAKE_PROFIT,
    contract: str = "c1",
    split: str = "validation",
) -> SerialCandidateV5:
    source_ns = _ns(session, source)
    realized_ns = _ns(session, realized)
    return SerialCandidateV5(
        split=split,
        session=session,
        decision_time_ns=_ns(session, decision),
        contract_id=contract,
        right="C",
        canonical_strike_slot=0,
        policy_index=0,
        entry_ask=ask,
        score=1.0,
        raw_label_pnl_after_campaign_fee=(bid - ask) * 100.0 - 3.0,
        label_mid_pnl_before_campaign_fee=(bid - ask) * 100.0,
        label_realized_exit_time_ns=realized_ns,
        label_source_exit_quote_time_ns=source_ns,
        label_exit_quote_age_ms=(realized_ns - source_ns) / 1_000_000.0,
        label_exit_reason_code=int(reason),
        label_executable_exit_bid=bid,
        label_policy_deadline_ns=_ns(session, deadline),
        feature_hash="feature",
        strategy="test",
        source_simulator_version=PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    )


def test_actual_stop_exit_releases_early_SIM_ACTUAL_EXIT_001() -> None:
    first = _candidate(
        source="10:02",
        realized="10:02",
        reason=ExitReason.STOP_LOSS,
        bid=1.5,
    )
    second = _candidate(
        decision="10:03",
        source="10:04",
        realized="10:04",
        deadline="10:13",
        contract="c2",
    )
    trades, state = simulate_serial_candidates_v5([first, second])
    assert [trade.contract_id for trade in trades] == ["c1", "c2"]
    assert state.skipped["overlap"] == 0


def test_deadline_source_does_not_release_occupancy_SIM_DEADLINE_OCCUPANCY_001() -> None:
    first = _candidate(
        source="10:05",
        realized="10:10",
        deadline="10:10",
        reason=ExitReason.MAX_HOLD,
    )
    second = _candidate(
        decision="10:06",
        source="10:07",
        realized="10:07",
        deadline="10:16",
        contract="c2",
    )
    trades, state = simulate_serial_candidates_v5([first, second])
    assert [trade.contract_id for trade in trades] == ["c1"]
    assert state.skipped["overlap"] == 1
    assert state.skipped_events[0].pending_source_quote_time_ns_or_null == _ns(
        "2025-01-02", "10:05"
    )
    assert state.skipped_events[0].pending_realized_exit_time_ns_or_null == _ns(
        "2025-01-02", "10:10"
    )


def test_exit_realizes_before_same_time_decision_SIM_SAME_TIME_001() -> None:
    first = _candidate(
        source="10:05",
        realized="10:10",
        deadline="10:10",
        reason=ExitReason.MAX_HOLD,
    )
    second = _candidate(
        decision="10:10",
        source="10:11",
        realized="10:11",
        deadline="10:20",
        contract="c2",
    )
    trades, state = simulate_serial_candidates_v5([first, second])
    assert len(trades) == 2
    assert state.skipped["overlap"] == 0


def test_decision_before_realized_exit_is_overlap_SIM_OVERLAP_001() -> None:
    first = _candidate(
        source="10:01",
        realized="10:10",
        deadline="10:10",
        reason=ExitReason.MAX_HOLD,
    )
    second = _candidate(
        decision="10:09",
        source="10:09:30",
        realized="10:09:30",
        deadline="10:19",
        contract="c2",
    )
    _, state = simulate_serial_candidates_v5([first, second])
    assert state.skipped["overlap"] == 1


def test_cash_event_uses_realized_clock_and_source_pnl_SIM_CASH_001() -> None:
    item = _candidate(
        source="10:05",
        realized="10:10",
        deadline="10:10",
        bid=1.5,
        reason=ExitReason.MAX_HOLD,
    )
    trades, state = simulate_serial_candidates_v5([item])
    assert trades[0].raw_label_pnl_after_campaign_fee == -53.0
    events = state.equity_events_by_account["validation"]
    assert events[-1]["event_time_ns"] == _ns("2025-01-02", "10:10")
    assert events[-1]["equity"] == 9_947.0


def test_daily_stop_waits_for_occupancy_exit_SIM_DAILYSTOP_001() -> None:
    loss = _candidate(
        source="10:01",
        realized="10:10",
        deadline="10:10",
        ask=6.0,
        bid=0.97,
        reason=ExitReason.MAX_HOLD,
    )
    overlap = _candidate(
        decision="10:05",
        source="10:06",
        realized="10:06",
        deadline="10:15",
        contract="c2",
    )
    stopped = _candidate(
        decision="10:10",
        source="10:11",
        realized="10:11",
        deadline="10:20",
        contract="c3",
    )
    trades, state = simulate_serial_candidates_v5([loss, overlap, stopped])
    assert len(trades) == 1
    assert state.skipped["overlap"] == 1
    assert state.skipped["daily_loss_stop"] == 1


def test_cash_continuity_across_sessions_SIM_CONTINUITY_001() -> None:
    first = _candidate(bid=2.1)
    second = _candidate(
        session="2025-01-03",
        contract="c2",
        bid=2.2,
    )
    _, state = simulate_serial_candidates_v5([first, second])
    assert state.session_start_equity["validation:2025-01-02"] == 10_000.0
    assert state.session_start_equity["validation:2025-01-03"] == pytest.approx(
        10_007.0
    )
    assert state.cash_by_account["validation"] == pytest.approx(10_024.0)


def test_equity_events_and_frequency_use_realized_clock_SIM_DRAWDOWN_001() -> None:
    first = _candidate(bid=1.0, reason=ExitReason.STOP_LOSS)
    second = _candidate(
        decision="10:02",
        source="10:03",
        realized="10:03",
        deadline="10:12",
        contract="c2",
        bid=3.0,
    )
    trades, state = simulate_serial_candidates_v5([first, second])
    events = state.equity_events_by_account["validation"]
    assert len(trades) == 2
    assert [event["event_time_ns"] for event in events[1:]] == [
        _ns("2025-01-02", "10:01"),
        _ns("2025-01-02", "10:03"),
    ]
    equities = [event["equity"] for event in events]
    assert max(equities) - min(equities) == pytest.approx(103.0)


def test_end_of_stream_uses_stored_exit_SIM_END_001() -> None:
    item = _candidate(
        source="10:05",
        realized="10:10",
        deadline="10:10",
        reason=ExitReason.MAX_HOLD,
    )
    _, state = simulate_serial_candidates_v5([item])
    assert state.equity_events_by_account["validation"][-1][
        "event_time_ns"
    ] == _ns("2025-01-02", "10:10")


def test_legacy_artifact_is_rejected_SIM_LEGACY_001() -> None:
    item = replace(
        _candidate(),
        source_simulator_version=PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION,
    )
    with pytest.raises(Protocol101LegacySyntheticExitArtifactError):
        simulate_serial_candidates_v5([item])


def test_affordability_reserve_and_fee_once_SIM_FEE_001() -> None:
    item = _candidate(bid=2.1)
    accepted, accepted_state = simulate_serial_candidates_v5(
        [item], config=SerialSimulatorV5Config(starting_cash=203.0)
    )
    rejected, rejected_state = simulate_serial_candidates_v5(
        [item], config=SerialSimulatorV5Config(starting_cash=202.99)
    )
    assert len(accepted) == 1
    assert accepted_state.cash_by_account["validation"] == pytest.approx(210.0)
    assert rejected == []
    assert rejected_state.skipped["unaffordable"] == 1


def test_stress_does_not_change_cash_or_occupancy_SIM_STRESS_001() -> None:
    item = _candidate(bid=2.1)
    baseline_trades, baseline = simulate_serial_candidates_v5([item])
    stress_trades, stressed = simulate_serial_candidates_v5(
        [item],
        config=SerialSimulatorV5Config(stress_per_trade_dollars=50.0),
    )
    assert baseline.cash_by_account == stressed.cash_by_account
    assert (
        baseline_trades[0].label_realized_exit_time_ns
        == stress_trades[0].label_realized_exit_time_ns
    )
    assert stress_trades[0].stressed_pnl == pytest.approx(
        baseline_trades[0].raw_label_pnl_after_campaign_fee - 50.0
    )
