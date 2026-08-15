"""Conservative neural policy primitives for the unified serial DP scope."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


ROLE_LABEL = "CHALLENGER_UNIFIED_CONSERVATIVE_NEURAL_POLICY_V1"
TRAINING_SPEC_LABEL = "PREREGISTERED_UNIFIED_CONSERVATIVE_NEURAL_TRAINING_V1"


@dataclass(frozen=True)
class ConservativeNeuralPolicyConfig:
    hidden_dim: int = 128
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    epochs: int = 3
    batch_size: int = 4096
    target_scale: float = 500.0
    target_clip: float = 5000.0
    min_advantage_margin: float = 250.0
    positive_probability_min: float = 0.55
    tail_probability_max: float = 0.35
    regression_weight: float = 0.35
    tail_weight: float = 0.35
    positive_class_weight: float = 1.0
    tail_class_weight: float = 1.0
    positive_regression_weight: float = 1.0
    negative_regression_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConservativeAdvantageMLP(nn.Module):
    """Small multi-head MLP for advantage, positive edge, and tail risk."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.advantage_head = nn.Linear(hidden_dim, 1)
        self.positive_logit_head = nn.Linear(hidden_dim, 1)
        self.tail_logit_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.body(features)
        return {
            "advantage": self.advantage_head(encoded).squeeze(-1),
            "positive_logit": self.positive_logit_head(encoded).squeeze(-1),
            "tail_logit": self.tail_logit_head(encoded).squeeze(-1),
        }


def conservative_action_allowed(
    *,
    predicted_advantage: float,
    positive_probability: float,
    tail_probability: float,
    config: ConservativeNeuralPolicyConfig = ConservativeNeuralPolicyConfig(),
) -> dict[str, Any]:
    adjusted = float(predicted_advantage)
    allowed = (
        adjusted >= float(config.min_advantage_margin)
        and float(positive_probability) >= float(config.positive_probability_min)
        and float(tail_probability) <= float(config.tail_probability_max)
    )
    return {
        "allowed": bool(allowed),
        "decision": "allow_challenger_action" if allowed else "defer_to_protocol101",
        "predicted_advantage": adjusted,
        "positive_probability": float(positive_probability),
        "tail_probability": float(tail_probability),
        "required_margin": float(config.min_advantage_margin),
        "required_positive_probability": float(config.positive_probability_min),
        "max_tail_probability": float(config.tail_probability_max),
    }


def loss_for_batch(
    outputs: dict[str, torch.Tensor],
    *,
    advantage_target: torch.Tensor,
    positive_target: torch.Tensor,
    tail_target: torch.Tensor,
    config: ConservativeNeuralPolicyConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    advantage_loss_unreduced = nn.functional.huber_loss(outputs["advantage"], advantage_target, delta=1.0, reduction="none")
    regression_weights = torch.where(
        positive_target >= 0.5,
        torch.full_like(positive_target, float(config.positive_regression_weight)),
        torch.full_like(positive_target, float(config.negative_regression_weight)),
    )
    advantage_loss = (advantage_loss_unreduced * regression_weights).mean()
    positive_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["positive_logit"],
        positive_target,
        pos_weight=torch.as_tensor(float(config.positive_class_weight), dtype=positive_target.dtype, device=positive_target.device),
    )
    tail_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["tail_logit"],
        tail_target,
        pos_weight=torch.as_tensor(float(config.tail_class_weight), dtype=tail_target.dtype, device=tail_target.device),
    )
    total = positive_loss + config.regression_weight * advantage_loss + config.tail_weight * tail_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "advantage_loss": float(advantage_loss.detach().cpu()),
        "positive_loss": float(positive_loss.detach().cpu()),
        "tail_loss": float(tail_loss.detach().cpu()),
    }
