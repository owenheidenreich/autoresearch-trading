from __future__ import annotations

from v4.scripts.run_protocol138_base_contract_protection import decide
from v4.sim.protocol101_position_sizing import AccountState, PositionSizingPolicy, choose_quantity


def test_exposure_cap_can_preserve_base_contract() -> None:
    policy = PositionSizingPolicy(
        name="test",
        max_contracts=3,
        premium_exposure_fraction=0.20,
        exposure_cap_applies_to_initial_contract=False,
    )
    state = AccountState(cash=10_000.0, peak_equity=10_000.0, daily_realized_pnl={}, recent_trade_pnls=[])
    trade = {"session": "2026-03-06", "premium_paid": 3_000.0, "pnl": 100.0}

    assert choose_quantity(trade, state, policy) == (1, "")


def test_decide_rejects_negative_external_block() -> None:
    assert (
        decide(
            [{"segment": "q4_2024_external", "incremental_pnl": -1.0}],
            [{"incremental_pnl": 1.0}],
            {"max_drawdown": -100.0},
            {"max_drawdown": -100.0},
        )
        == "reject_base_contract_protection_still_external_negative"
    )

