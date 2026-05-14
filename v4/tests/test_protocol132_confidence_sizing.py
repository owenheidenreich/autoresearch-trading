from __future__ import annotations

from dataclasses import replace

from v4.scripts.run_protocol132_confidence_sizing_autoresearch import (
    MAX_FAILED_HYPOTHESES,
    decide,
    evaluate_candidate,
)
from v4.sim.protocol101_position_sizing import (
    AccountState,
    confidence_ladder_policy,
    choose_quantity,
    score_margin,
)


def _trade(*, score: float = 0.5, threshold: float = -2.0, premium: float = 1_000.0) -> dict:
    return {
        "session": "2026-03-06",
        "decision_time": "2026-03-06T15:00:00+00:00",
        "exit_time": "2026-03-06T15:05:00+00:00",
        "contract_id": "SPXW-20260306-06700.000-C",
        "premium_paid": premium,
        "entry_ask": premium / 100.0,
        "score": score,
        "threshold": threshold,
        "pnl": 100.0,
    }


def test_score_margin_is_entry_time_confidence_signal() -> None:
    assert score_margin(_trade(score=0.5, threshold=-2.0)) == 2.5


def test_confidence_policy_reduces_quantity_when_margin_is_too_low() -> None:
    policy = confidence_ladder_policy()
    state = AccountState(
        cash=80_000.0,
        peak_equity=80_000.0,
        daily_realized_pnl={},
        recent_trade_pnls=[2_000.0],
    )

    low = choose_quantity(_trade(score=-0.2, threshold=-1.0), state, policy)
    high = choose_quantity(_trade(score=1.5, threshold=-2.0), state, policy)

    assert low == (1, "")
    assert high == (3, "")


def test_profit_cushion_blocks_scaling_before_real_account_growth() -> None:
    policy = replace(confidence_ladder_policy(), min_profit_for_scaling=20_000.0)
    state = AccountState(
        cash=25_000.0,
        peak_equity=25_000.0,
        daily_realized_pnl={},
        recent_trade_pnls=[2_000.0],
    )

    assert choose_quantity(_trade(score=1.5, threshold=-2.0), state, policy) == (1, "")


def test_evaluate_candidate_requires_worst_day_to_stay_controlled() -> None:
    baseline = {
        "total_pnl": 300_000.0,
        "max_drawdown_pct": -0.06,
        "worst_day_pnl": -2_000.0,
        "candidate_trades": 100,
    }
    candidate = {
        "total_pnl": 500_000.0,
        "max_drawdown_pct": -0.07,
        "worst_day_pnl": -4_000.0,
        "candidate_trades": 100,
        "skipped_trades": 0,
        "risk_of_ruin": False,
    }

    result = evaluate_candidate(candidate, baseline)

    assert result["total_improved"] is True
    assert result["drawdown_ok"] is True
    assert result["worst_day_ok"] is False
    assert result["passes_acceptance"] is False


def test_decide_pauses_after_three_failed_hypotheses() -> None:
    assert decide([], MAX_FAILED_HYPOTHESES) == "pause_after_three_failed_sizing_hypotheses"

