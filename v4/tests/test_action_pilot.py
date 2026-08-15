"""Tests for the decision-level action pilot."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import torch

from v4.model.action_pilot import (
    ActionDecision,
    decision_aware_action_loss,
    metrics_for_trades,
    simulate_action_baseline,
)


def test_action_baseline_enforces_cooldown() -> None:
    decisions = [
        ActionDecision(
            session="2026-03-02",
            decision_time=datetime(2026, 3, 2, 14, 31 + i, tzinfo=timezone.utc),
            features=np.zeros(220, dtype=np.float32),
            labels=np.array([0.0, -10.0, 20.0], dtype=np.float32),
            offsets=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            market_last=np.array([1, 0, 1, -1, 0, 0, -1], dtype=np.float32),
        )
        for i in range(3)
    ]

    trades = simulate_action_baseline(decisions, kind="atm_put", cooldown_minutes=45)
    metrics = metrics_for_trades(trades)

    assert metrics["trades"] == 1
    assert metrics["total_pnl"] == 20.0


def test_decision_aware_loss_penalizes_false_positive_trade() -> None:
    target = torch.tensor([[0.0, -1.0, -2.0]], dtype=torch.float32)
    bad_false_trade = torch.tensor([[0.0, 0.4, -0.2]], dtype=torch.float32)
    good_no_trade = torch.tensor([[0.4, -0.2, -0.3]], dtype=torch.float32)

    _, bad_parts = decision_aware_action_loss(
        bad_false_trade,
        target,
        target_scale=100.0,
    )
    _, good_parts = decision_aware_action_loss(
        good_no_trade,
        target,
        target_scale=100.0,
    )

    assert bad_parts["false_trade_margin"] > good_parts["false_trade_margin"]


def test_decision_aware_loss_penalizes_missed_profitable_trade() -> None:
    target = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float32)
    bad_no_trade = torch.tensor([[0.5, 0.2, -0.2]], dtype=torch.float32)
    good_trade = torch.tensor([[0.1, 0.6, -0.2]], dtype=torch.float32)

    _, bad_parts = decision_aware_action_loss(
        bad_no_trade,
        target,
        target_scale=100.0,
    )
    _, good_parts = decision_aware_action_loss(
        good_trade,
        target,
        target_scale=100.0,
    )

    assert bad_parts["missed_trade_margin"] > good_parts["missed_trade_margin"]
