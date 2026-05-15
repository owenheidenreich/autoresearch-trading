from __future__ import annotations

from v4.scripts.run_protocol131_multi_contract_pnl_tracking import decide, with_acceptance
from v4.sim.protocol101_position_sizing import (
    AccountState,
    PositionSizingPolicy,
    choose_quantity,
    conservative_profit_ladder_policy,
    large_account_research_sizer_policy,
    max_contracts_by_equity,
    simulate_position_sizing,
    strict_exposure_ladder_policy,
)


def _trade(
    *,
    trade_number: int = 1,
    decision_time: str = "2026-03-06T15:00:00+00:00",
    exit_time: str = "2026-03-06T15:05:00+00:00",
    premium: float = 1_000.0,
    pnl: float = 100.0,
) -> dict:
    return {
        "trade_number": trade_number,
        "candidate_uid": f"t{trade_number}",
        "session": decision_time[:10],
        "decision_time": decision_time,
        "exit_time": exit_time,
        "contract_id": "SPXW-20260306-06700.000-C",
        "side": "CALL",
        "premium_paid": premium,
        "entry_ask": premium / 100.0,
        "pnl": pnl,
    }


def test_choose_quantity_scales_only_after_equity_and_recent_profit() -> None:
    policy = conservative_profit_ladder_policy()
    trade = _trade(premium=1_000.0)

    no_recent = AccountState(cash=25_000.0, peak_equity=25_000.0, daily_realized_pnl={}, recent_trade_pnls=[])
    positive_recent = AccountState(cash=25_000.0, peak_equity=25_000.0, daily_realized_pnl={}, recent_trade_pnls=[500.0])

    assert choose_quantity(trade, no_recent, policy) == (1, "")
    assert choose_quantity(trade, positive_recent, policy) == (2, "")


def test_choose_quantity_respects_drawdown_and_daily_stop() -> None:
    policy = conservative_profit_ladder_policy()
    trade = _trade(premium=1_000.0)
    drawdown = AccountState(cash=25_000.0, peak_equity=30_000.0, daily_realized_pnl={}, recent_trade_pnls=[500.0])
    daily_stopped = AccountState(
        cash=25_000.0,
        peak_equity=25_000.0,
        daily_realized_pnl={"2026-03-06": -750.0},
        recent_trade_pnls=[500.0],
    )

    assert choose_quantity(trade, drawdown, policy) == (1, "")
    assert choose_quantity(trade, daily_stopped, policy) == (0, "daily_loss_stop")


def test_strict_exposure_ladder_skips_high_premium_early_trade() -> None:
    policy = strict_exposure_ladder_policy()
    state = AccountState(cash=10_000.0, peak_equity=10_000.0, daily_realized_pnl={}, recent_trade_pnls=[])

    assert choose_quantity(_trade(premium=3_000.0), state, policy) == (0, "premium_exposure_cap")


def test_simulator_tracks_cash_quantity_and_daily_stop() -> None:
    policy = PositionSizingPolicy(
        name="test",
        max_contracts=2,
        equity_for_two_contracts=0.0,
        premium_exposure_fraction=1.0,
        daily_new_entry_stop_loss=-750.0,
        require_recent_positive_for_scaling=False,
    )
    trades = [
        _trade(trade_number=1, pnl=-800.0),
        _trade(
            trade_number=2,
            decision_time="2026-03-06T15:10:00+00:00",
            exit_time="2026-03-06T15:15:00+00:00",
            pnl=1_000.0,
        ),
    ]

    result = simulate_position_sizing(trades, policy)

    assert result["rows"][0]["quantity"] == 2
    assert result["rows"][0]["realized_pnl"] == -1600.0
    assert result["rows"][1]["quantity"] == 0
    assert result["rows"][1]["skip_reason"] == "daily_loss_stop"


def test_protocol131_acceptance_rejects_worse_loss_clustering() -> None:
    baseline = {
        "policy": "one_contract_baseline",
        "total_pnl": 100_000.0,
        "max_drawdown_pct": -0.05,
        "worst_day_pnl": -2_000.0,
        "candidate_trades": 100,
        "skipped_trades": 0,
        "risk_of_ruin": False,
    }
    scaled = {
        "policy": "scaled",
        "total_pnl": 200_000.0,
        "max_drawdown_pct": -0.06,
        "worst_day_pnl": -5_000.0,
        "candidate_trades": 100,
        "skipped_trades": 0,
        "risk_of_ruin": False,
    }

    enriched = with_acceptance(scaled, baseline)

    assert enriched["acceptance_checks"]["worst_day_ok"] is False
    assert decide([baseline, scaled]) == "reject_multi_contract_risk_not_improved_enough"


def test_large_account_policy_is_not_architecturally_capped_at_three() -> None:
    policy = large_account_research_sizer_policy(starting_cash=1_000_000.0)

    assert max_contracts_by_equity(1_100_000.0, policy) == 20


def test_large_account_policy_can_choose_double_digit_contracts_only_with_confidence_and_profit() -> None:
    policy = large_account_research_sizer_policy(starting_cash=1_000_000.0)
    high_confidence = {
        **_trade(premium=2_000.0),
        "score": 5.0,
        "threshold": 0.0,
    }
    low_confidence = {
        **_trade(premium=2_000.0),
        "score": 2.0,
        "threshold": 0.0,
    }
    state = AccountState(
        cash=1_100_000.0,
        peak_equity=1_100_000.0,
        daily_realized_pnl={},
        recent_trade_pnls=[10_000.0],
    )

    assert choose_quantity(high_confidence, state, policy) == (11, "")
    assert choose_quantity(low_confidence, state, policy) == (2, "")
