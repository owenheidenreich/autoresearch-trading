"""Tests for risk-controlled broad purchase helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from v4.model.action_pilot import ActionDecision
from v4.scripts.evaluate_risk_controlled_purchase_signal import (
    RiskConfig,
    simulate_risk_controlled_policy,
)


def _decision(*, minute_utc: int, call_pnl: float, put_pnl: float = -1.0) -> ActionDecision:
    return ActionDecision(
        session="2026-03-02",
        decision_time=datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        + timedelta(minutes=minute_utc),
        features=np.zeros(220, dtype=np.float32),
        labels=np.array([0.0, call_pnl, put_pnl], dtype=np.float32),
        offsets=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        market_last=np.array([1, 0, 1, -1, 0, 0, -1], dtype=np.float32),
    )


def test_time_filter_is_applied_before_cooldown() -> None:
    config = RiskConfig(
        threshold=0.0,
        time_filter="post_open_only",
        allowed_buckets=("post_open_morning",),
        max_trades_per_day=99,
        daily_loss_stop=None,
    )
    decisions = [
        _decision(minute_utc=45, call_pnl=-100.0),  # 09:45 ET, disallowed.
        _decision(minute_utc=65, call_pnl=50.0),  # 10:05 ET, allowed.
    ]
    predictions = np.array([[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32)

    trades = simulate_risk_controlled_policy(
        decisions,
        predictions,
        config=config,
        cooldown_minutes=45,
        strategy="test",
    )

    assert len(trades) == 1
    assert trades[0].pnl == 50.0
    assert trades[0].decision_time.endswith("15:05:00+00:00")


def test_daily_loss_stop_halts_after_realized_loss() -> None:
    config = RiskConfig(
        threshold=0.0,
        time_filter="post_open_only",
        allowed_buckets=("post_open_morning",),
        max_trades_per_day=99,
        daily_loss_stop=-100.0,
    )
    decisions = [
        _decision(minute_utc=65, call_pnl=-150.0),
        _decision(minute_utc=80, call_pnl=500.0),
    ]
    predictions = np.array([[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32)

    trades = simulate_risk_controlled_policy(
        decisions,
        predictions,
        config=config,
        cooldown_minutes=10,
        strategy="test",
    )

    assert len(trades) == 1
    assert trades[0].pnl == -150.0
