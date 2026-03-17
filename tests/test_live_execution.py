from __future__ import annotations

from training.prepare import ACTION_BUY_CALL_ATM
from training.live.contracts import DecisionIntent, RiskUpdateIntent
from training.live.execution import OCOExecutionEngine


def _intent() -> DecisionIntent:
    return DecisionIntent(
        action=ACTION_BUY_CALL_ATM,
        contract={"symbol": "SPXW"},
        qty=1,
        entry_order="MKT",
        stop_price=1.00,
        take_profit_price=2.00,
        confidence=0.8,
        reason_codes=["unit_test"],
        reference_price=1.25,
    )


def test_monotonic_risk_updates() -> None:
    engine = OCOExecutionEngine(dry_run=True, max_position_size=2)
    state = engine.place_entry(_intent())

    ok_up = engine.apply_risk_update(
        RiskUpdateIntent(position_id=state.position_id, new_stop_price=1.10, new_take_profit_price=2.30)
    )
    assert ok_up is True
    assert engine.positions[state.position_id].current_stop == 1.10
    assert engine.positions[state.position_id].current_take_profit == 2.30

    bad_stop = engine.apply_risk_update(
        RiskUpdateIntent(position_id=state.position_id, new_stop_price=1.05)
    )
    assert bad_stop is False
    assert engine.positions[state.position_id].current_stop == 1.10

    bad_tp = engine.apply_risk_update(
        RiskUpdateIntent(position_id=state.position_id, new_take_profit_price=2.20)
    )
    assert bad_tp is False
    assert engine.positions[state.position_id].current_take_profit == 2.30

