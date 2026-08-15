"""Built-model attainability proof for the compact action-value selector."""
from __future__ import annotations

import torch

from v5.research.causal_day_action_value_targets import TARGET_SCALE_USD
from v5.research.causal_day_architectures import (
    CausalPolicyBatch,
    build_architecture,
    declared_dimensions,
)
from v5.research.causal_day_compact_interaction import ARCHITECTURE_NAME
from v5.research.causal_day_policy_gate import (
    SelectorAttainability,
    assert_selector_attainable,
)


def action_value_selector_attainability() -> SelectorAttainability:
    """Construct a firing witness from the canonical built architecture.

    Action values are divided by one common scale in the loss and are never
    clipped. The model's final contract and WAIT maps are linear and likewise
    unbounded. Zeroing every parameter, then setting only the canonical
    contract bias to +1 scaled unit, creates an exact current ENTER=$1,000,
    WAIT=$0 witness under the production comparison.
    """

    dimensions = declared_dimensions()
    model = build_architecture(ARCHITECTURE_NAME, dimensions).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.contract_base.bias.fill_(1.0)
    batch = CausalPolicyBatch(
        candles=torch.zeros(1, 1, dimensions.candle_features),
        candle_mask=torch.ones(1, 1, dtype=torch.bool),
        ladder=torch.zeros(1, 1, dimensions.ladder_features),
        ladder_mask=torch.ones(1, 1, dtype=torch.bool),
        entry_action_mask=torch.ones(1, 1, dtype=torch.bool),
        account=torch.zeros(1, dimensions.account_features),
        position=torch.zeros(1, dimensions.position_features),
        clock=torch.zeros(1, dimensions.clock_features),
        roles=("morning_entry",),
    )
    scores = model(batch)
    proof = SelectorAttainability(
        selector_name="first_enter_above_structural_wait_floor",
        rule_kind="relative_action_value",
        target_clip_bounds=None,
        prediction_clip_bounds=None,
        absolute_threshold=None,
        no_trade_floor=0.0,
        witness_enter=float(scores.contract_logits[0, 0]) * TARGET_SCALE_USD,
        witness_wait=float(scores.abstain_logits[0]) * TARGET_SCALE_USD,
        proof_source=(
            "computed canonical compact model with all weights zero and only "
            "contract_base.bias=1; action_value_loss uses q/TARGET_SCALE_USD without clamp"
        ),
    )
    assert_selector_attainable(proof)
    return proof
