from __future__ import annotations

import numpy as np
import torch

from training.prepare import ACTION_DO_NOTHING, ACTION_BUY_CALL_OTM10
from training.live.decision import ModelDecisionEngine


class _DummyModelV17(torch.nn.Module):
    """Mimics v17 PredictionModel: returns (pred_returns, pred_conf, action_out)."""

    ACCOUNT_STATE_DIM = 5

    def __init__(self) -> None:
        super().__init__()
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor, account_state=None):
        self.last_input_shape = tuple(int(v) for v in x.shape)
        batch = x.shape[0]
        pred_returns = torch.tensor([[0.001, 0.002, 0.003]], dtype=torch.float32).repeat(batch, 1)
        pred_conf = torch.tensor([0.5], dtype=torch.float32).repeat(batch)
        action_out = torch.tensor([[0.8, 0.5, 0.1]], dtype=torch.float32).repeat(batch, 1)
        return pred_returns, pred_conf, action_out


class _DummyModelV18(torch.nn.Module):
    """Mimics v18 TradingModel: returns 5-tuple."""

    def __init__(self) -> None:
        super().__init__()
        # gate_head attribute signals v18
        self.gate_head = torch.nn.Identity()
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor):
        self.last_input_shape = tuple(int(v) for v in x.shape)
        batch = x.shape[0]
        market_pred = torch.tensor([[0.001, 0.002, 0.003]], dtype=torch.float32).repeat(batch, 1)
        entry_gate = torch.tensor([0.8], dtype=torch.float32).repeat(batch)
        risk_params = torch.tensor([[1.5, 2.0, 0.7]], dtype=torch.float32).repeat(batch, 1)
        exit_signal = torch.tensor([0.1], dtype=torch.float32).repeat(batch)
        # Direction class 2 = call_otm10
        dir_logits = torch.tensor([[0.1, 0.2, 3.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(batch, 1)
        return market_pred, entry_gate, risk_params, exit_signal, dir_logits


def test_infer_truncates_extra_features_to_model_width() -> None:
    model = _DummyModelV17()
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
    model = _DummyModelV17()
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


def test_v18_infer_uses_direction_head() -> None:
    model = _DummyModelV18()
    engine = ModelDecisionEngine(
        model=model,
        lookback=3,
        num_features=60,
        min_trade_prob=0.2,
        model_version="v18",
    )

    feature_window = np.zeros((5, 60), dtype=np.float32)
    result = engine.infer(feature_window)

    # Direction class 2 = call_otm10
    assert result.action == ACTION_BUY_CALL_OTM10
    assert abs(result.trade_prob - 0.8) < 1e-5
    assert abs(result.risk_conviction - 0.7) < 1e-5
    assert "trade_signal" in result.reason_codes


def test_v18_infer_no_trade_when_gate_low() -> None:
    model = _DummyModelV18()
    # Override entry_gate to return low value
    original_forward = model.forward
    def low_gate_forward(x):
        out = original_forward(x)
        return out[0], torch.tensor([0.1]).repeat(x.shape[0]), out[2], out[3], out[4]
    model.forward = low_gate_forward

    engine = ModelDecisionEngine(
        model=model,
        lookback=3,
        num_features=60,
        min_trade_prob=0.5,
        model_version="v18",
    )

    feature_window = np.zeros((5, 60), dtype=np.float32)
    result = engine.infer(feature_window)

    assert result.action == ACTION_DO_NOTHING
    assert "no_trade" in result.reason_codes
