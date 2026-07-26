"""Tests for the canonical Protocol101 serial simulator contract."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from v4.model.protocol101_serial_simulator import (
    CASH_BASIS,
    DAILY_LOSS_BASIS,
    PROTOCOL101_SERIAL_SIMULATOR_VERSION,
    STRESS_APPLICATION,
    SerialCandidate,
    SerialSimulatorConfig,
    simulate_serial_candidates,
)
from v4.model.supervised_pilot import DecisionCandidates, simulate_model_policy


def _candidate(
    *,
    minute: int,
    raw_pnl: float,
    ask: float = 3.0,
    session: str = "2026-01-02",
    max_hold: float | None = None,
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
        max_hold_minutes=max_hold,
        feature_hash="a" * 64,
        source_quote_time=f"2026-01-02T14:{minute:02d}:00+00:00",
        source_context_time=f"2026-01-02T14:{minute - 1:02d}:00+00:00",
    )


def test_simulator_contract_metadata_is_explicit() -> None:
    config = SerialSimulatorConfig()

    assert config.simulator_version == PROTOCOL101_SERIAL_SIMULATOR_VERSION
    assert config.effective_simulator_version() == PROTOCOL101_SERIAL_SIMULATOR_VERSION
    assert config.daily_loss_basis == DAILY_LOSS_BASIS == "raw_realized_net_pnl"
    assert config.cash_basis == CASH_BASIS == "raw_realized_net_pnl"
    assert config.stress_application == STRESS_APPLICATION == "metrics_only"


def test_disabled_cooldown_hold_guard_changes_effective_version() -> None:
    config = SerialSimulatorConfig(require_cooldown_equals_max_hold_when_present=False)

    assert config.effective_simulator_version().endswith("_cooldown_hold_guard_disabled")
    assert config.semantics()["simulator_version"].endswith("_cooldown_hold_guard_disabled")


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


def test_entry_cutoff_blocks_after_1530_et() -> None:
    trades, state = simulate_serial_candidates(
        [
            SerialCandidate(
                split="validation",
                session="2026-01-02",
                decision_time=datetime(2026, 1, 2, 20, 31, tzinfo=timezone.utc),
                contract_id="SPXW-20260102-06500.000-P",
                right="P",
                offset=-20.0,
                entry_ask=3.0,
                score=1.0,
                raw_label_pnl=100.0,
                cooldown_minutes=1.0,
            )
        ],
        config=SerialSimulatorConfig(no_new_entries_after_et="15:30"),
    )

    assert trades == []
    assert state.skipped["after_entry_cutoff"] == 1


def test_entry_cutoff_allows_exactly_1530_et() -> None:
    trades, state = simulate_serial_candidates(
        [
            SerialCandidate(
                split="validation",
                session="2026-01-02",
                decision_time=datetime(2026, 1, 2, 20, 30, tzinfo=timezone.utc),
                contract_id="SPXW-20260102-06500.000-P",
                right="P",
                offset=-20.0,
                entry_ask=3.0,
                score=1.0,
                raw_label_pnl=100.0,
                cooldown_minutes=1.0,
            )
        ],
        config=SerialSimulatorConfig(no_new_entries_after_et="15:30"),
    )

    assert len(trades) == 1
    assert state.skipped["after_entry_cutoff"] == 0


def test_naive_decision_time_fails_closed() -> None:
    trades, state = simulate_serial_candidates(
        [
            SerialCandidate(
                split="validation",
                session="2026-01-02",
                decision_time=datetime(2026, 1, 2, 14, 32),
                contract_id="SPXW-20260102-06500.000-P",
                right="P",
                offset=-20.0,
                entry_ask=3.0,
                score=1.0,
                raw_label_pnl=100.0,
                cooldown_minutes=1.0,
            )
        ],
        config=SerialSimulatorConfig(),
    )

    assert trades == []
    assert state.skipped["tz_naive_decision_time"] == 1


def test_forced_flat_caps_synthetic_exit_time() -> None:
    trades, state = simulate_serial_candidates(
        [
            SerialCandidate(
                split="validation",
                session="2026-01-02",
                decision_time=datetime(2026, 1, 2, 20, 29, tzinfo=timezone.utc),
                contract_id="SPXW-20260102-06500.000-P",
                right="P",
                offset=-20.0,
                entry_ask=3.0,
                score=1.0,
                raw_label_pnl=100.0,
                cooldown_minutes=45.0,
            )
        ],
        config=SerialSimulatorConfig(forced_flat_before_et="15:55"),
    )

    assert len(trades) == 1
    assert datetime.fromisoformat(trades[0].synthetic_exit_time).astimezone().isoformat()
    exit_et = datetime.fromisoformat(trades[0].synthetic_exit_time).astimezone(
        timezone(timedelta(hours=-5))
    )
    assert exit_et.hour == 15
    assert exit_et.minute == 55
    assert state.semantics["forced_flat_before"] == "15:55"


def test_cooldown_hold_mismatch_fails_when_policy_hold_is_available() -> None:
    trades, state = simulate_serial_candidates(
        [_candidate(minute=32, raw_pnl=100.0, max_hold=2.0)],
        config=SerialSimulatorConfig(),
    )

    assert trades == []
    assert state.skipped["cooldown_hold_mismatch"] == 1


def test_hashes_separate_config_stream_and_payload_purposes() -> None:
    first = [_candidate(minute=32, raw_pnl=100.0)]
    same_key_different_payload = [_candidate(minute=32, raw_pnl=125.0)]
    different_key_same_config = [_candidate(minute=34, raw_pnl=100.0)]

    _trades_a, state_a = simulate_serial_candidates(first, config=SerialSimulatorConfig())
    _trades_b, state_b = simulate_serial_candidates(
        same_key_different_payload,
        config=SerialSimulatorConfig(),
    )
    _trades_c, state_c = simulate_serial_candidates(
        different_key_same_config,
        config=SerialSimulatorConfig(),
    )

    assert state_a.simulator_config_hash == state_b.simulator_config_hash == state_c.simulator_config_hash
    assert state_a.simulator_semantics_hash == state_a.simulator_config_hash
    assert state_a.candidate_stream_hash == state_b.candidate_stream_hash
    assert state_a.candidate_payload_hash != state_b.candidate_payload_hash
    assert state_a.candidate_stream_hash != state_c.candidate_stream_hash


def test_no_trade_enters_inside_same_session_pending_window() -> None:
    trades, state = simulate_serial_candidates(
        [
            _candidate(minute=32, raw_pnl=100.0),
            _candidate(minute=32, raw_pnl=200.0),
            _candidate(minute=33, raw_pnl=300.0),
        ],
        config=SerialSimulatorConfig(),
    )

    assert len(trades) == 2
    assert state.skipped["overlap"] == 1
    previous_exit_by_session = {}
    for trade in trades:
        key = (trade.split, trade.session)
        decision_time = datetime.fromisoformat(trade.decision_time)
        previous_exit = previous_exit_by_session.get(key)
        assert previous_exit is None or decision_time >= previous_exit
        previous_exit_by_session[key] = datetime.fromisoformat(trade.synthetic_exit_time)


def test_v2_acceptance_matches_simulate_model_policy_when_cooldown_equals_hold() -> None:
    base_time = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    labels = [-120.0, 180.0, -600.0, 100.0, 90.0, 80.0]
    asks = [5.0, 6.0, 7.0, 4.0, 4.0, 4.0]
    decisions = []
    candidates = []
    for idx, (label, ask) in enumerate(zip(labels, asks)):
        decision_time = base_time + timedelta(minutes=idx * 2)
        decisions.append(
            DecisionCandidates(
                session="2026-01-02",
                decision_time=decision_time,
                features=np.zeros((1, 71), dtype=np.float32),
                labels=np.array([label], dtype=np.float32),
                offsets=np.array([-20.0], dtype=np.float32),
                rights=np.array(["P"], dtype=object),
                market_last=np.zeros(7, dtype=np.float32),
                contract_ids=np.array([f"contract-{idx}"], dtype=object),
                entry_asks=np.array([ask], dtype=np.float32),
            )
        )
        candidates.append(
            SerialCandidate(
                split="validation",
                session="2026-01-02",
                decision_time=decision_time,
                contract_id=f"contract-{idx}",
                right="P",
                offset=-20.0,
                entry_ask=ask,
                score=1.0,
                raw_label_pnl=label,
                cooldown_minutes=2.0,
                max_hold_minutes=2.0,
            )
        )

    legacy_trades = simulate_model_policy(
        decisions,
        [np.array([1.0], dtype=np.float32) for _ in decisions],
        threshold=0.0,
        cooldown_minutes=2,
        strategy="legacy_equivalence",
        max_daily_loss=500.0,
    )
    v2_trades, state = simulate_serial_candidates(
        candidates,
        config=SerialSimulatorConfig(max_daily_loss=500.0),
    )

    assert [trade.decision_time for trade in v2_trades] == [
        trade.decision_time for trade in legacy_trades
    ]
    assert [trade.raw_label_pnl for trade in v2_trades] == [
        trade.pnl for trade in legacy_trades
    ]
    assert state.semantics["candidate_stream_hash"]
    assert state.semantics["simulator_semantics_hash"]


def test_fuzzed_v2_acceptance_matches_simulate_model_policy_for_equal_cooldown_hold() -> None:
    rng = np.random.default_rng(17)
    base_time = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    for trial in range(25):
        decisions = []
        candidates = []
        minute = 0
        for idx in range(12):
            minute += int(rng.integers(1, 4))
            decision_time = base_time + timedelta(minutes=minute)
            label = float(rng.choice([-700.0, -200.0, -50.0, 30.0, 120.0, 450.0]))
            ask = float(rng.choice([2.0, 4.0, 8.0, 25.0, 120.0]))
            contract_id = f"trial-{trial}-contract-{idx}"
            decisions.append(
                DecisionCandidates(
                    session="2026-01-02",
                    decision_time=decision_time,
                    features=np.zeros((1, 71), dtype=np.float32),
                    labels=np.array([label], dtype=np.float32),
                    offsets=np.array([-20.0], dtype=np.float32),
                    rights=np.array(["P"], dtype=object),
                    market_last=np.zeros(7, dtype=np.float32),
                    contract_ids=np.array([contract_id], dtype=object),
                    entry_asks=np.array([ask], dtype=np.float32),
                )
            )
            candidates.append(
                SerialCandidate(
                    split="validation",
                    session="2026-01-02",
                    decision_time=decision_time,
                    contract_id=contract_id,
                    right="P",
                    offset=-20.0,
                    entry_ask=ask,
                    score=1.0,
                    raw_label_pnl=label,
                    cooldown_minutes=2.0,
                    max_hold_minutes=2.0,
                )
            )
        legacy_trades = simulate_model_policy(
            decisions,
            [np.array([1.0], dtype=np.float32) for _ in decisions],
            threshold=0.0,
            cooldown_minutes=2,
            strategy="legacy_equivalence",
            max_daily_loss=500.0,
            starting_cash=10_000.0,
        )
        v2_trades, _state = simulate_serial_candidates(
            candidates,
            config=SerialSimulatorConfig(
                max_daily_loss=500.0,
                starting_cash=10_000.0,
            ),
        )

        assert [trade.decision_time for trade in v2_trades] == [
            trade.decision_time for trade in legacy_trades
        ]
        assert [trade.raw_label_pnl for trade in v2_trades] == [
            trade.pnl for trade in legacy_trades
        ]
