from __future__ import annotations

import torch

from v5.research.causal_day_architectures import CausalPolicyBatch
from v5.research.causal_day_compact_shared_lifecycle import (
    LADDER_CONTEXT_INDICES,
    MAX_PROJECTED_CONSERVATIVE_PARAMETERS,
    CompactSharedLifecyclePolicy,
    computed_parameter_count,
)
from v5.research.causal_day_compact_interaction import SIDE_INDEX
from v5.research.causal_day_tensorizer import (
    ACCOUNT_FEATURES,
    CANDLE_FEATURES,
    CLOCK_FEATURES,
    LADDER_FEATURES,
    POSITION_FEATURES,
)


def _batch(*, body: float = 0.0, origin: float = 1.0) -> CausalPolicyBatch:
    candles = torch.zeros(4, 3, len(CANDLE_FEATURES))
    candles[:, -1, CANDLE_FEATURES.index("body_points")] = body
    ladder = torch.zeros(4, 3, len(LADDER_FEATURES))
    ladder[:, 0, SIDE_INDEX] = 1.0
    ladder[:, 1, SIDE_INDEX] = -1.0
    position = torch.zeros(4, POSITION_FEATURES)
    position[:, 9] = origin
    return CausalPolicyBatch(
        candles=candles,
        candle_mask=torch.ones(4, 3, dtype=torch.bool),
        ladder=ladder,
        ladder_mask=torch.ones(4, 3, dtype=torch.bool),
        entry_action_mask=torch.tensor(
            [[1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.bool
        ),
        account=torch.zeros(4, ACCOUNT_FEATURES),
        position=position,
        clock=torch.zeros(4, CLOCK_FEATURES),
        roles=("morning_entry", "morning_exit", "afternoon_entry", "afternoon_exit"),
    )


def test_built_count_fits_worst_projected_conservative_budget() -> None:
    assert computed_parameter_count() <= MAX_PROJECTED_CONSERVATIVE_PARAMETERS
    assert computed_parameter_count() == sum(
        value.numel() for value in CompactSharedLifecyclePolicy().parameters()
    )


def test_one_shared_model_routes_entry_wait_hold_and_sell() -> None:
    scores = CompactSharedLifecyclePolicy()(_batch())
    assert scores.contract_logits.shape == (4, 3)
    assert scores.abstain_logits.shape == (4,)
    assert scores.exit_logits.shape == (4, 2)
    assert scores.active_logits(0).shape == (4,)
    assert scores.active_logits(1).shape == (2,)
    assert torch.isneginf(scores.contract_logits[0, 2])
    assert torch.isneginf(scores.contract_logits[1]).all()


def test_chart_state_can_reverse_call_put_ordering() -> None:
    model = CompactSharedLifecyclePolicy().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.shared_state.weight[0, 0] = 1.0
        model.direction.weight[0, 0] = 1.0
    up = model(_batch(body=2.0)).contract_logits[0]
    down = model(_batch(body=-2.0)).contract_logits[0]
    assert up[:2].argmax().item() == 0
    assert down[:2].argmax().item() == 1


def test_context_only_ladder_node_remains_visible_to_shared_state() -> None:
    model = CompactSharedLifecyclePolicy().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # First ladder-summary input follows candle+clock and is the mean of
        # the first declared context field. WAIT reads that latent value.
        summary_offset = 8 + CLOCK_FEATURES
        model.shared_state.weight[0, summary_offset] = 1.0
        model.wait.weight[0, 0] = 1.0
    base = _batch()
    changed_ladder = base.ladder.clone()
    changed_ladder[:, 2, LADDER_CONTEXT_INDICES[0]] = 9.0
    changed = CausalPolicyBatch(
        candles=base.candles,
        candle_mask=base.candle_mask,
        ladder=changed_ladder,
        ladder_mask=base.ladder_mask,
        entry_action_mask=base.entry_action_mask,
        account=base.account,
        position=base.position,
        clock=base.clock,
        roles=base.roles,
    )
    assert not torch.allclose(model(base).abstain_logits, model(changed).abstain_logits)


def test_opening_regime_is_position_state_not_current_wall_clock() -> None:
    model = CompactSharedLifecyclePolicy().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # The last selected position field is origin regime.
        model.exit.weight[1, -1] = 1.0
    morning_origin = model(_batch(origin=1.0)).exit_logits
    afternoon_origin = model(_batch(origin=-1.0)).exit_logits
    assert torch.all(morning_origin[:, 1] > afternoon_origin[:, 1])


def test_masked_future_values_cannot_change_scores() -> None:
    torch.manual_seed(4)
    model = CompactSharedLifecyclePolicy().eval()
    clean = _batch()
    candles = torch.cat((clean.candles, torch.full((4, 1, len(CANDLE_FEATURES)), 1e6)), dim=1)
    mask = torch.cat((clean.candle_mask, torch.zeros((4, 1), dtype=torch.bool)), dim=1)
    dirty = CausalPolicyBatch(
        candles=candles,
        candle_mask=mask,
        ladder=clean.ladder,
        ladder_mask=clean.ladder_mask,
        entry_action_mask=clean.entry_action_mask,
        account=clean.account,
        position=clean.position,
        clock=clean.clock,
        roles=clean.roles,
    )
    a = model(clean)
    b = model(dirty)
    assert torch.allclose(a.abstain_logits, b.abstain_logits)
    assert torch.allclose(a.exit_logits, b.exit_logits)
    finite = torch.isfinite(a.contract_logits)
    assert torch.allclose(a.contract_logits[finite], b.contract_logits[finite])
