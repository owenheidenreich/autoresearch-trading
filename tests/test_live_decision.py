from __future__ import annotations

import numpy as np
import torch

from training.prepare import ACTION_DO_NOTHING
from training.live.decision import ModelDecisionEngine


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_input_shape = tuple(int(v) for v in x.shape)
        batch = x.shape[0]
        gate_logits = torch.tensor([[0.0, 4.0]], dtype=torch.float32, device=x.device).repeat(batch, 1)
        dir_logits = torch.tensor([[0.1, 0.2, 3.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=x.device).repeat(batch, 1)
        return gate_logits, dir_logits


def test_infer_truncates_extra_features_to_model_width() -> None:
    model = _DummyModel()
    engine = ModelDecisionEngine(
        model=model,
        lookback=3,
        num_features=60,
        min_trade_prob=0.2,
    )

    feature_window = np.zeros((5, 64), dtype=np.float32)
    result = engine.infer(feature_window)

    assert model.last_input_shape == (1, 3, 60)
    assert result.action != ACTION_DO_NOTHING
    assert "feature_dim_truncated" in result.reason_codes


def test_infer_rejects_too_narrow_feature_window() -> None:
    model = _DummyModel()
    engine = ModelDecisionEngine(
        model=model,
        lookback=3,
        num_features=60,
    )

    feature_window = np.zeros((5, 59), dtype=np.float32)
    result = engine.infer(feature_window)

    assert result.action == ACTION_DO_NOTHING
    assert "feature_dim_too_small" in result.reason_codes
    assert model.last_input_shape is None
