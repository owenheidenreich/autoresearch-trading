from __future__ import annotations

import numpy as np
import pytest
import torch

from v5.research.causal_day_action_value_targets import (
    ActionValueTargetError,
    ActionValueTargetSession,
    action_value_loss,
    collate_action_values,
    shuffled_session_surface,
    validate_alignment,
)
from v5.research.causal_day_architectures import ArchitectureScores
from v5.research.causal_day_fit_cache import CachedSession


def _cached() -> CachedSession:
    return CachedSession(
        session="2025-08-01",
        minutes=np.asarray(["09:35", "09:36", "09:37"]),
        candle_lengths=np.asarray([1, 2, 3]),
        candles=np.zeros((3, 18), np.float32),
        ladder_offsets=np.asarray([0, 2, 4, 5]),
        ladder=np.zeros((5, 23), np.float32),
        action_mask=np.asarray([True, False, True, True, False]),
        targets=np.zeros((5, 3), np.float32),
        contract_ids=np.asarray(["a", "x", "b", "c", "y"]),
    )


def _targets() -> ActionValueTargetSession:
    return ActionValueTargetSession(
        "2025-08-01",
        np.asarray([10.0, np.nan, 30.0, -5.0, np.nan], np.float32),
        np.asarray([30.0, 0.0, 0.0], np.float32),
    )


def test_target_alignment_and_collation_preserve_whole_ladder_mask() -> None:
    cached = _cached()
    targets = _targets()
    validate_alignment(cached, targets)
    q_enter, q_wait = collate_action_values(cached, targets, np.asarray([0, 1]))
    assert q_enter.shape == (2, 2)
    assert torch.equal(q_wait, torch.tensor([30.0, 0.0]))
    assert torch.isnan(q_enter[0, 1])


def test_joint_loss_trains_wait_and_enter_on_one_common_scale() -> None:
    contract = torch.tensor([[0.0, -torch.inf], [0.0, 0.0]], requires_grad=True)
    wait = torch.tensor([0.0, 0.0], requires_grad=True)
    scores = ArchitectureScores(
        contract_logits=contract,
        abstain_logits=wait,
        exit_logits=torch.full((2, 2), -torch.inf),
        roles=("morning_entry", "morning_entry"),
        architecture="compact_interaction_entry",
    )
    q_enter = torch.tensor([[1000.0, float("nan")], [2000.0, -1000.0]])
    q_wait = torch.tensor([3000.0, 0.0])
    mask = torch.tensor([[True, False], [True, True]])
    loss = action_value_loss(scores, q_enter, q_wait, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert wait.grad is not None and wait.grad[0] < 0
    assert contract.grad is not None and contract.grad[0, 0] < 0


def test_action_value_target_transform_is_not_clipped() -> None:
    """Values beyond the old magnitude ceiling remain exactly attainable."""

    scores = ArchitectureScores(
        contract_logits=torch.tensor([[2.5]]),
        abstain_logits=torch.tensor([-1.5]),
        exit_logits=torch.full((1, 2), -torch.inf),
        roles=("morning_entry",),
        architecture="compact_interaction_entry",
    )
    loss = action_value_loss(
        scores,
        torch.tensor([[2_500.0]]),
        torch.tensor([-1_500.0]),
        torch.tensor([[True]]),
    )
    assert loss.item() == 0.0


def test_shuffled_null_is_deterministic_preserves_values_and_recomputes_wait() -> None:
    cached = _cached()
    targets = _targets()
    first = shuffled_session_surface(cached, targets, seed=7)
    second = shuffled_session_surface(cached, targets, seed=7)
    assert np.array_equal(first.q_enter_bid_usd, second.q_enter_bid_usd, equal_nan=True)
    original = np.sort(targets.q_enter_bid_usd[cached.action_mask])
    shuffled = np.sort(first.q_enter_bid_usd[cached.action_mask])
    assert np.array_equal(original, shuffled)
    minute_best = [first.q_enter_bid_usd[0], np.nanmax(first.q_enter_bid_usd[2:4]), np.nan]
    expected_wait = [max(0.0, minute_best[1]), 0.0, 0.0]
    assert np.allclose(first.q_wait_bid_usd, expected_wait)


def test_target_on_ineligible_node_is_refused() -> None:
    targets = _targets()
    targets.q_enter_bid_usd[1] = 5.0
    with pytest.raises(ActionValueTargetError, match="ineligible"):
        validate_alignment(_cached(), targets)


def test_wait_only_training_row_still_trains_wait() -> None:
    wait = torch.zeros(1, requires_grad=True)
    scores = ArchitectureScores(
        contract_logits=torch.zeros(1, 1),
        abstain_logits=wait,
        exit_logits=torch.full((1, 2), -torch.inf),
        roles=("morning_entry",),
        architecture="compact_interaction_entry",
    )
    loss = action_value_loss(
        scores,
        torch.tensor([[float("nan")]]),
        torch.tensor([1000.0]),
        torch.tensor([[False]]),
    )
    loss.backward()
    assert wait.grad is not None and wait.grad.item() < 0.0
