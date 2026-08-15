from __future__ import annotations

import numpy as np
import torch

from v5.ops.train_causal_day_action_value import score_sessions, train_stage
from v5.research.causal_day_action_value_targets import (
    ActionValueTargetSession,
    target_path,
)
from v5.research.causal_day_compact_interaction import CompactInteractionEntryPolicy
from v5.research.causal_day_fit_cache import FeatureScaler, cache_path


def _write_fixture(feature_root, target_root) -> None:
    path = cache_path(feature_root, "2025-08-01")
    path.parent.mkdir(parents=True)
    ladder = np.zeros((4, 23), np.float32)
    ladder[:, 20] = [-5.0, -10.0, -5.0, -10.0]
    ladder[:, 21] = [1.0, 1.0, 0.0, 0.0]
    np.savez_compressed(
        path,
        session=np.asarray("2025-08-01"),
        minutes=np.asarray(["09:35", "09:36"]),
        candle_lengths=np.asarray([1, 2]),
        candles=np.zeros((2, 18), np.float32),
        ladder_offsets=np.asarray([0, 2, 4]),
        ladder=ladder,
        action_mask=np.asarray([True, True, False, False]),
        targets=np.zeros((4, 3), np.float32),
        contract_ids=np.asarray(["a", "b", "c", "d"]),
    )
    target = target_path(target_root, "2025-08-01")
    target.parent.mkdir(parents=True)
    values = ActionValueTargetSession(
        "2025-08-01",
        np.asarray([1000.0, -500.0, np.nan, np.nan], np.float32),
        np.asarray([0.0, 0.0], np.float32),
    )
    np.savez_compressed(
        target,
        session=np.asarray(values.session),
        q_enter_bid_usd=values.q_enter_bid_usd,
        q_wait_bid_usd=values.q_wait_bid_usd,
    )


def test_one_joint_training_stage_and_scoring_path_is_executable(tmp_path) -> None:
    feature_root = tmp_path / "features"
    target_root = tmp_path / "targets"
    _write_fixture(feature_root, target_root)
    model = CompactInteractionEntryPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.wait.weight.detach().clone()
    losses = train_stage(
        model,
        optimizer,
        feature_paths=[cache_path(feature_root, "2025-08-01")],
        target_root=target_root,
        scaler=FeatureScaler(np.zeros(18), np.ones(18), np.zeros(23), np.ones(23)),
        device=torch.device("cpu"),
        null=False,
        stage=1,
        epochs=1,
    )
    assert len(losses) == 1 and np.isfinite(losses[0])
    assert not torch.equal(before, model.wait.weight.detach())
    candidate, minute = score_sessions(
        model,
        feature_paths=[cache_path(feature_root, "2025-08-01")],
        scaler=FeatureScaler(np.zeros(18), np.ones(18), np.zeros(23), np.ones(23)),
        device=torch.device("cpu"),
        null=False,
        fold=1,
    )
    assert len(candidate) == 2
    assert len(minute) == 2
    assert set(candidate["contract_id"]) == {"a", "b"}
    assert np.isfinite(candidate["predicted_q_enter_bid_120m_usd"]).all()
    assert np.isfinite(minute["predicted_q_wait_bid_120m_usd"]).all()
