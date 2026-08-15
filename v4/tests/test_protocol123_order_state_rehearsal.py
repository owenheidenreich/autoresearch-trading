from __future__ import annotations

from v4.scripts.run_protocol123_protocol101_order_state_rehearsal import (
    build_order_record,
    decide,
    max_concurrent_positions,
    stress_replay,
)
from v4.sim.order_state import OrderState


def _trade(uid: str = "t1", *, decision_ms: int = 0, exit_ms: int = 60_000, pnl: float = 100.0) -> dict:
    return {
        "candidate_uid": uid,
        "contract_id": "SPXW-20260306-06700.000-C",
        "session": "2026-03-06",
        "decision_time": "2026-03-06T15:00:00+00:00",
        "exit_time": "2026-03-06T15:01:00+00:00",
        "decision_ms": decision_ms,
        "exit_ms": exit_ms,
        "entry_bid": 1.90,
        "entry_ask": 2.00,
        "exit_bid": 3.00,
        "exit_ask": 3.10,
        "premium_paid": 200.0,
        "quote_gap_seconds": 0.0,
        "pnl": pnl,
    }


def test_protocol123_build_order_record_reaches_exit_filled() -> None:
    record = build_order_record(_trade(), order_id="o1")

    assert record.final_state == OrderState.EXIT_FILLED
    assert record.intended_size == 1
    assert [event.to_state for event in record.history][-2:] == [
        OrderState.EXIT_SUBMITTED,
        OrderState.EXIT_FILLED,
    ]


def test_protocol123_stress_replay_is_adverse() -> None:
    base = stress_replay([_trade(pnl=100.0)], starting_cash=10_000.0, stress_per_side=0.0)
    stressed = stress_replay([_trade(pnl=100.0)], starting_cash=10_000.0, stress_per_side=0.25)

    assert base["total_pnl"] == 100.0
    assert stressed["total_pnl"] == 50.0
    assert stressed["unaffordable_trades"] == 0


def test_protocol123_detects_overlap() -> None:
    assert max_concurrent_positions(
        [
            _trade("a", decision_ms=0, exit_ms=120_000),
            _trade("b", decision_ms=60_000, exit_ms=180_000),
        ]
    ) == 2


def test_protocol123_decision_requires_10000_baseline() -> None:
    decision = decide(
        {
            "errors": [],
            "all_exit_filled": True,
            "all_one_contract": True,
            "all_spxw": True,
            "max_concurrent_positions": 1,
        },
        [{"stress_per_side": 0.25, "total_pnl": 10.0}],
        skipped=[],
        starting_cash=10_000.0,
    )

    assert decision == "pass_10000_order_state_rehearsal_live_data_pending"
