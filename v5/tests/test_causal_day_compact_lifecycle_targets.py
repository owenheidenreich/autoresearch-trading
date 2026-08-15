from __future__ import annotations

import numpy as np
import pytest
import torch

from v5.research.causal_day_architectures import ArchitectureScores
from v5.research.causal_day_compact_lifecycle_targets import (
    LifecycleTargetError,
    build_exit_action_targets,
    exit_action_value_loss,
)


def _scores(values: torch.Tensor) -> ArchitectureScores:
    rows = len(values)
    return ArchitectureScores(
        contract_logits=torch.empty((rows, 0)),
        abstain_logits=torch.zeros(rows),
        exit_logits=values,
        roles=tuple("morning_exit" for _ in range(rows)),
        architecture="compact_shared_lifecycle",
    )


def test_hold_target_uses_best_strictly_later_executable_sale() -> None:
    got = build_exit_action_targets(np.asarray([-100.0, 0.0, 50.0, 20.0]))

    assert np.array_equal(got.q_sell_usd, [-100.0, 0.0, 50.0, 20.0])
    assert np.allclose(got.q_hold_usd[:-1], [50.0, 50.0, 20.0])
    assert np.isnan(got.q_hold_usd[-1])
    assert np.array_equal(got.hold_available, [True, True, True, False])


def test_exit_loss_is_zero_for_exact_scaled_action_values() -> None:
    targets = build_exit_action_targets(np.asarray([-100.0, 0.0, 50.0, 20.0]))
    predicted = torch.tensor(
        [
            [0.05, -0.10],
            [0.05, 0.00],
            [0.02, 0.05],
            [0.00, 0.02],
        ],
        dtype=torch.float32,
    )
    loss = exit_action_value_loss(
        _scores(predicted),
        torch.tensor(targets.q_sell_usd, dtype=torch.float32),
        torch.tensor(targets.q_hold_usd, dtype=torch.float32),
        torch.tensor(targets.hold_available),
    )
    assert loss.item() == pytest.approx(0.0)


def test_exit_targets_refuse_missing_sale_values() -> None:
    with pytest.raises(LifecycleTargetError, match="finite sale values"):
        build_exit_action_targets(np.asarray([0.0, np.nan]))


def test_forced_liquidation_may_not_receive_a_hold_target() -> None:
    with pytest.raises(LifecycleTargetError, match="must remain unavailable"):
        exit_action_value_loss(
            _scores(torch.zeros((2, 2))),
            torch.zeros(2),
            torch.zeros(2),
            torch.tensor([True, False]),
        )
