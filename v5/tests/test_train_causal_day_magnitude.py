from __future__ import annotations

import numpy as np
import torch

from v5.ops.train_causal_day_magnitude import shuffled_target_assignments, train_stage
from v5.research.causal_day_architectures import declared_dimensions
from v5.research.causal_day_fit_cache import FeatureScaler, cache_path
from v5.research.causal_day_magnitude import build_magnitude_policy


def test_null_shuffles_outcomes_but_preserves_horizon_vectors(tmp_path) -> None:
    root = tmp_path / "cache"
    path = cache_path(root, "2025-01-02")
    path.parent.mkdir(parents=True)
    ladder = np.zeros((6, 23), np.float32)
    ladder[:, 20] = [-5, -5, -5, -10, -10, -10]
    ladder[:, 21] = [1, 1, 1, 0, 0, 0]
    targets = np.asarray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]],
        np.float32,
    )
    np.savez_compressed(
        path,
        session=np.asarray("2025-01-02"),
        minutes=np.asarray(["09:35"]),
        candle_lengths=np.asarray([1]),
        candles=np.zeros((1, 18), np.float32),
        ladder_offsets=np.asarray([0, 6]),
        ladder=ladder,
        action_mask=np.ones(6, bool),
        targets=targets,
        contract_ids=np.asarray(list("abcdef")),
    )
    first = shuffled_target_assignments([path], 7)["2025-01-02"]
    second = shuffled_target_assignments([path], 7)["2025-01-02"]
    assert np.array_equal(first, second)
    assert {tuple(row) for row in first} == {tuple(row) for row in targets}
    assert all(tuple(row) in {tuple(value) for value in targets} for row in first)


def test_one_canonical_scalar_horizon_training_step_is_executable(tmp_path) -> None:
    root = tmp_path / "cache"
    path = cache_path(root, "2025-01-02")
    path.parent.mkdir(parents=True)
    ladder = np.zeros((4, 23), np.float32)
    ladder[:, 20] = [-5, -10, -5, -10]
    ladder[:, 21] = [1, 1, 0, 0]
    np.savez_compressed(
        path,
        session=np.asarray("2025-01-02"),
        minutes=np.asarray(["09:35"]),
        candle_lengths=np.asarray([1]),
        candles=np.zeros((1, 18), np.float32),
        ladder_offsets=np.asarray([0, 4]),
        ladder=ladder,
        action_mask=np.ones(4, bool),
        targets=np.asarray(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], np.float32
        ),
        contract_ids=np.asarray(list("abcd")),
    )
    model = build_magnitude_policy("shallow_joint", declared_dimensions())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = train_stage(
        model,
        optimizer,
        paths=[path],
        scaler=FeatureScaler(np.zeros(18), np.ones(18), np.zeros(23), np.ones(23)),
        weights=torch.ones(5),
        device=torch.device("cpu"),
        architecture="shallow_joint",
        horizon=60,
        null=False,
        stage=1,
        epochs=1,
    )
    assert len(losses) == 1
    assert np.isfinite(losses[0])
