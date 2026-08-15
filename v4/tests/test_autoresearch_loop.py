"""Tests for the bounded SPXW 0DTE autoresearch loop."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from v4.model.action_pilot import ActionDecision
from v4.scripts.run_autoresearch_loop import (
    AutoresearchTrial,
    aggregate_results,
    simulate_true_no_trade_policy,
)


def _decision(*, minute_utc: int, call_pnl: float = 50.0, put_pnl: float = -20.0) -> ActionDecision:
    return ActionDecision(
        session="2026-03-02",
        decision_time=datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        + timedelta(minutes=minute_utc),
        features=np.zeros(284, dtype=np.float32),
        labels=np.array([0.0, call_pnl, put_pnl], dtype=np.float32),
        offsets=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        market_last=np.array([1, 0, 1, -1, 0, 0, -1], dtype=np.float32),
    )


def _trial() -> AutoresearchTrial:
    return AutoresearchTrial(
        name="post_open_test",
        time_filter="post_open_only",
        allowed_buckets=("post_open_morning",),
        min_edge_vs_no_trade=0.0,
        max_trades_per_day=99,
        daily_loss_stop=None,
    )


def _metrics(
    total_pnl: float,
    *,
    positive_day_fraction: float = 0.6,
    top_day_profit_share: float = 0.3,
) -> dict:
    return {
        "trades": 20,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / 20.0,
        "median_pnl": 1.0,
        "win_rate": 0.55,
        "profit_factor": 1.2,
        "max_drawdown": -100.0,
        "sessions_traded": 10,
        "mean_daily_pnl": total_pnl / 10.0,
        "median_daily_pnl": 1.0,
        "positive_day_fraction": positive_day_fraction,
        "top_day_profit_share": top_day_profit_share,
        "selection_reward": total_pnl,
    }


def test_true_no_trade_output_can_abstain() -> None:
    trades = simulate_true_no_trade_policy(
        [_decision(minute_utc=65), _decision(minute_utc=80)],
        np.array(
            [
                [25.0, 10.0, 5.0],  # no-trade wins
                [0.0, 20.0, 5.0],  # call wins
            ],
            dtype=np.float32,
        ),
        trial=_trial(),
        cooldown_minutes=10,
        strategy="test",
    )

    assert len(trades) == 1
    assert trades[0].right == "C"
    assert trades[0].pnl == 50.0


def test_time_filter_does_not_consume_cooldown() -> None:
    trades = simulate_true_no_trade_policy(
        [
            _decision(minute_utc=45, call_pnl=-100.0),  # 09:45 ET, disallowed
            _decision(minute_utc=65, call_pnl=50.0),  # 10:05 ET, allowed
        ],
        np.array([[0.0, 20.0, 5.0], [0.0, 20.0, 5.0]], dtype=np.float32),
        trial=_trial(),
        cooldown_minutes=45,
        strategy="test",
    )

    assert len(trades) == 1
    assert trades[0].decision_time.endswith("15:05:00+00:00")


def test_champion_selection_ignores_audit_test_metrics() -> None:
    results = [
        {
            "policy_index": 1,
            "policy_name": "policy",
            "seed": 11,
            "trial": {"name": "good_validation"},
            "selection_metrics": _metrics(500.0),
            "audit_test_metrics": _metrics(-1000.0),
            "selection_reward": 500.0,
        },
        {
            "policy_index": 1,
            "policy_name": "policy",
            "seed": 11,
            "trial": {"name": "good_audit_only"},
            "selection_metrics": _metrics(100.0),
            "audit_test_metrics": _metrics(5000.0),
            "selection_reward": 100.0,
        },
    ]

    aggregate = aggregate_results(results)

    assert aggregate["champion"]["trial_name"] == "good_validation"


def test_champion_prefers_validation_floor_eligible_trial() -> None:
    results = [
        {
            "policy_index": 1,
            "policy_name": "policy",
            "seed": 11,
            "trial": {"name": "high_reward_bad_breadth"},
            "selection_metrics": _metrics(1000.0, positive_day_fraction=0.2),
            "audit_test_metrics": _metrics(1000.0),
            "selection_reward": 1000.0,
        },
        {
            "policy_index": 1,
            "policy_name": "policy",
            "seed": 11,
            "trial": {"name": "lower_reward_floor_cleared"},
            "selection_metrics": _metrics(500.0, positive_day_fraction=0.6),
            "audit_test_metrics": _metrics(-1000.0),
            "selection_reward": 500.0,
        },
    ]

    aggregate = aggregate_results(results)

    assert aggregate["champion"]["trial_name"] == "lower_reward_floor_cleared"
    assert aggregate["eligible_trial_count"] == 1
