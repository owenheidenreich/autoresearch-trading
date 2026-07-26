"""Tests for strict replay of fair-contract selected candidates."""
from __future__ import annotations

from v4.scripts.run_protocol101_fair_contract_selected_candidate_replay_gate import (
    STRICT_REPLAY_IMPLEMENTATION_VERSION,
    build_checks,
    metrics_for_trades,
    replay_selected_candidates,
    waiting_payload,
)


def _row(
    *,
    split: str = "validation",
    session: str = "2026-01-02",
    decision_time: str = "2026-01-02T14:32:00+00:00",
    ask: float = 3.0,
    pnl: float = 80.0,
    cooldown: int = 25,
) -> dict:
    return {
        "split": split,
        "session": session,
        "decision_time": decision_time,
        "contract_id": f"SPXW-{session}-06500.000-C",
        "right": "C",
        "offset": "0",
        "entry_ask": str(ask),
        "score": "50",
        "label_net_pnl": str(pnl),
        "feature_hash": "a" * 64,
        "source_quote_time": decision_time,
        "source_context_time": decision_time,
        "cooldown_minutes": str(cooldown),
    }


def test_waiting_payload_is_safe() -> None:
    payload = waiting_payload("selected_export_not_pass")

    assert payload["status"] == "waiting_for_owner_approved_training_result"
    assert payload["implementation_version"] == STRICT_REPLAY_IMPLEMENTATION_VERSION
    assert payload["simulator_version"] == "protocol101_serial_simulator_v2"
    assert payload["daily_loss_basis"] == "raw_realized_net_pnl"
    assert payload["cash_basis"] == "raw_realized_net_pnl"
    assert payload["stress_application"] == "metrics_only"
    assert payload["exit_time_semantics"] == "synthetic_exit_at_entry_plus_cooldown"
    assert payload["cooldown_anchor"] == "entry"
    assert payload["no_new_entries_after"] == "15:30"
    assert payload["forced_flat_before"] == "15:55"
    assert payload["fee_model"] == "none_in_state"
    assert payload["account_continuity"] == "cash_compounds_across_sessions_within_split"
    assert payload["strict_replay_executed"] is False
    assert payload["broker_endpoint_called"] is False
    assert payload["paper_submit_allowed"] is False


def test_replay_enforces_affordability() -> None:
    trades, state = replay_selected_candidates(
        [_row(ask=200.0)],
        starting_cash=10_000.0,
        contract_multiplier=100.0,
        stress_per_side=0.10,
    )

    assert trades == []
    assert state["skipped"]["unaffordable"] == 1


def test_replay_enforces_serial_overlap_by_session() -> None:
    trades, state = replay_selected_candidates(
        [
            _row(decision_time="2026-01-02T14:32:00+00:00"),
            _row(decision_time="2026-01-02T14:33:00+00:00"),
        ],
        starting_cash=10_000.0,
        contract_multiplier=100.0,
        stress_per_side=0.10,
    )

    assert len(trades) == 1
    assert state["skipped"]["overlap"] == 1


def test_replay_enforces_session_trade_cap_and_daily_loss_stop() -> None:
    rows = [
        _row(decision_time="2026-01-02T14:32:00+00:00", pnl=-600.0, cooldown=1),
        _row(decision_time="2026-01-02T14:34:00+00:00", pnl=900.0, cooldown=1),
        _row(decision_time="2026-01-02T14:36:00+00:00", pnl=900.0, cooldown=1),
    ]

    capped, capped_state = replay_selected_candidates(
        rows,
        starting_cash=10_000.0,
        contract_multiplier=100.0,
        stress_per_side=0.0,
        max_trades_per_session=2,
    )
    stopped, stopped_state = replay_selected_candidates(
        rows,
        starting_cash=10_000.0,
        contract_multiplier=100.0,
        stress_per_side=0.0,
        max_daily_loss=500.0,
    )

    assert [trade.stressed_pnl for trade in capped] == [-600.0, 900.0]
    assert capped_state["skipped"]["session_trade_cap"] == 1
    assert [trade.stressed_pnl for trade in stopped] == [-600.0]
    assert stopped_state["skipped"]["daily_loss_stop"] == 2
    assert stopped_state["daily_loss_basis"] == "raw_realized_net_pnl"


def test_replay_daily_loss_ignores_metrics_stress_haircut() -> None:
    rows = [
        _row(decision_time="2026-01-02T14:32:00+00:00", pnl=-480.0, cooldown=1),
        _row(decision_time="2026-01-02T14:34:00+00:00", pnl=-10.0, cooldown=1),
        _row(decision_time="2026-01-02T14:36:00+00:00", pnl=100.0, cooldown=1),
    ]

    trades, state = replay_selected_candidates(
        rows,
        starting_cash=10_000.0,
        contract_multiplier=100.0,
        stress_per_side=0.10,
        max_daily_loss=500.0,
    )

    assert [trade.raw_label_pnl for trade in trades] == [-480.0, -10.0, 100.0]
    assert [trade.stressed_pnl for trade in trades] == [-500.0, -30.0, 80.0]
    assert state["skipped"]["daily_loss_stop"] == 0
    assert state["realized_raw_pnl_by_split_session"]["validation:2026-01-02"] == -390.0
    assert state["stress_per_trade_dollars"] == 20.0
    assert state["simulator_config_hash"]
    assert state["candidate_stream_hash"]
    assert state["candidate_payload_hash"]
    assert state["simulator_semantics_hash"]
    assert state["simulator_semantics_hash"] == state["simulator_config_hash"]


def test_replay_blocks_entries_after_cutoff() -> None:
    trades, state = replay_selected_candidates(
        [_row(decision_time="2026-01-02T20:31:00+00:00")],
        starting_cash=10_000.0,
        contract_multiplier=100.0,
        stress_per_side=0.10,
    )

    assert trades == []
    assert state["skipped"]["after_entry_cutoff"] == 1


def test_replay_forced_flat_caps_synthetic_exit_time() -> None:
    trades, state = replay_selected_candidates(
        [_row(decision_time="2026-01-02T20:29:00+00:00", cooldown=45)],
        starting_cash=10_000.0,
        contract_multiplier=100.0,
        stress_per_side=0.10,
    )

    assert len(trades) == 1
    assert trades[0].synthetic_exit_time == "2026-01-02T15:55:00-05:00"
    assert state["forced_flat_before"] == "15:55"


def test_replay_metrics_and_checks_pass_for_clean_candidate_set() -> None:
    rows = [
        _row(split="validation", session="2026-01-02", pnl=200.0),
        _row(split="diagnostic_test", session="2026-03-03", pnl=250.0),
    ]
    trades, state = replay_selected_candidates(
        rows,
        starting_cash=10_000.0,
        contract_multiplier=100.0,
        stress_per_side=0.10,
    )
    by_split = {}
    for trade in trades:
        by_split.setdefault(trade.split, []).append(trade)
    metrics = {
        split: metrics_for_trades(split_trades, starting_cash=10_000.0)
        for split, split_trades in by_split.items()
    }
    checks = build_checks(
        metrics_by_split=metrics,
        skipped=state["skipped"],
        selected_export={
            "status": "pass",
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
        },
        min_validation_trades=1,
        min_diagnostic_trades=1,
        min_profit_factor=1.0,
        max_drawdown_pct_of_start=0.35,
    )

    assert len(trades) == 2
    assert metrics["validation"]["total_pnl"] == 180.0
    assert all(check["pass"] for check in checks.values())
