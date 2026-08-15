from __future__ import annotations

import torch

from v4.model.unified_conservative_neural_policy import (
    ConservativeAdvantageMLP,
    ConservativeNeuralPolicyConfig,
    conservative_action_allowed,
    loss_for_batch,
)


def test_conservative_advantage_mlp_outputs_three_heads() -> None:
    model = ConservativeAdvantageMLP(input_dim=4, hidden_dim=8)

    out = model(torch.zeros((3, 4), dtype=torch.float32))

    assert set(out) == {"advantage", "positive_logit", "tail_logit"}
    assert out["advantage"].shape == (3,)
    assert out["positive_logit"].shape == (3,)
    assert out["tail_logit"].shape == (3,)


def test_conservative_action_gate_requires_margin_positive_prob_and_tail_prob() -> None:
    config = ConservativeNeuralPolicyConfig(min_advantage_margin=250.0, positive_probability_min=0.55, tail_probability_max=0.35)

    assert conservative_action_allowed(predicted_advantage=300.0, positive_probability=0.7, tail_probability=0.2, config=config)["allowed"]
    assert not conservative_action_allowed(predicted_advantage=200.0, positive_probability=0.7, tail_probability=0.2, config=config)["allowed"]
    assert not conservative_action_allowed(predicted_advantage=300.0, positive_probability=0.4, tail_probability=0.2, config=config)["allowed"]
    assert not conservative_action_allowed(predicted_advantage=300.0, positive_probability=0.7, tail_probability=0.5, config=config)["allowed"]


def test_loss_for_batch_is_finite() -> None:
    outputs = {
        "advantage": torch.zeros(2),
        "positive_logit": torch.zeros(2),
        "tail_logit": torch.zeros(2),
    }

    loss, parts = loss_for_batch(
        outputs,
        advantage_target=torch.tensor([0.5, -0.5]),
        positive_target=torch.tensor([1.0, 0.0]),
        tail_target=torch.tensor([0.0, 1.0]),
        config=ConservativeNeuralPolicyConfig(),
    )

    assert torch.isfinite(loss)
    assert parts["loss"] > 0.0
