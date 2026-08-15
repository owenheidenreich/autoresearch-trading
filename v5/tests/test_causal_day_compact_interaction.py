from __future__ import annotations

import torch

from v5.research.causal_day_action_value_attainability import (
    action_value_selector_attainability,
)
from v5.research.causal_day_architectures import (
    CausalPolicyBatch,
    computed_parameter_counts,
)
from v5.research.causal_day_compact_interaction import (
    CONTRACT_BASE_INDICES,
    MONEYNESS_INDEX,
    SIDE_INDEX,
    CompactInteractionEntryPolicy,
    computed_parameter_count,
)
from v5.research.causal_day_tensorizer import CANDLE_FEATURES, LADDER_FEATURES


def _batch(*, body: float) -> CausalPolicyBatch:
    candles = torch.zeros(1, 2, len(CANDLE_FEATURES))
    candles[0, 1, CANDLE_FEATURES.index("body_points")] = body
    ladder = torch.zeros(1, 2, len(LADDER_FEATURES))
    # The fold-standardized is_call channel is positive for calls and negative
    # for puts.  Equal remaining fields isolate the declared interaction.
    ladder[0, 0, SIDE_INDEX] = 1.0
    ladder[0, 1, SIDE_INDEX] = -1.0
    ladder[:, :, MONEYNESS_INDEX] = -1.0
    return CausalPolicyBatch(
        candles=candles,
        candle_mask=torch.tensor([[True, True]]),
        ladder=ladder,
        ladder_mask=torch.ones(1, 2, dtype=torch.bool),
        entry_action_mask=torch.ones(1, 2, dtype=torch.bool),
        account=torch.zeros(1, 5),
        position=torch.zeros(1, 10),
        clock=torch.zeros(1, 5),
        roles=("morning_entry",),
    )


def test_computed_count_is_inside_the_conservative_budget() -> None:
    # Three 13->1 state maps (42) plus one 5->1 contract base (6).
    assert computed_parameter_count() == 48
    assert 29 <= computed_parameter_count() <= 50


def test_canonical_architecture_registry_builds_and_counts_the_compact_model() -> None:
    counts = computed_parameter_counts()
    assert counts["compact_interaction_entry"] == computed_parameter_count() == 48


def test_relative_action_selector_has_an_unclipped_built_model_witness() -> None:
    proof = action_value_selector_attainability()
    assert proof.rule_kind == "relative_action_value"
    assert proof.target_clip_bounds is None
    assert proof.prediction_clip_bounds is None
    assert proof.absolute_threshold is None
    assert proof.witness_enter == 1_000.0
    assert proof.witness_wait == 0.0


def test_chart_state_can_reverse_call_put_ordering() -> None:
    model = CompactInteractionEntryPolicy().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        body_index_in_state = 0
        model.direction_state.weight[0, body_index_in_state] = 1.0
    up = model(_batch(body=2.0)).contract_logits
    down = model(_batch(body=-2.0)).contract_logits
    assert up.argmax(dim=1).item() == 0
    assert down.argmax(dim=1).item() == 1


def test_wait_is_a_trainable_action_in_the_same_action_vector() -> None:
    model = CompactInteractionEntryPolicy()
    scores = model(_batch(body=0.0))
    actions = scores.active_logits(0)
    assert actions.shape == (3,)
    loss = torch.nn.functional.cross_entropy(actions.unsqueeze(0), torch.tensor([0]))
    loss.backward()
    assert model.wait.weight.grad is not None
    assert model.wait.bias.grad is not None


def test_masked_contracts_are_never_actions() -> None:
    model = CompactInteractionEntryPolicy().eval()
    batch = _batch(body=1.0)
    batch = CausalPolicyBatch(
        candles=batch.candles,
        candle_mask=batch.candle_mask,
        ladder=batch.ladder,
        ladder_mask=batch.ladder_mask,
        entry_action_mask=torch.tensor([[True, False]]),
        account=batch.account,
        position=batch.position,
        clock=batch.clock,
        roles=batch.roles,
    )
    scores = model(batch)
    assert torch.isfinite(scores.contract_logits[0, 0])
    assert torch.isneginf(scores.contract_logits[0, 1])
    assert len(CONTRACT_BASE_INDICES) == 5
