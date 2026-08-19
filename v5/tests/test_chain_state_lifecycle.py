"""The declared member must be capacity-neutral, and must refuse absent state."""
from __future__ import annotations

import pytest
import torch

from v5.research.causal_day_architectures import ArchitectureInputError
from v5.research.causal_day_chain_state_lifecycle import (
    ChainPolicyBatch,
    CONTRACT_WIDTH,
    STATE_WIDTH,
    ChainStateLifecyclePolicy,
    computed_parameter_count,
)
from v5.research.causal_day_compact_shared_lifecycle import (
    computed_parameter_count as baseline_parameter_count,
)
from v5.research.causal_day_tensorizer import (
    ACCOUNT_FEATURES,
    CANDLE_FEATURES,
    CLOCK_FEATURES,
    LADDER_FEATURES,
    POSITION_FEATURES,
)
from v5.research.chain_internal_features import (
    CHAIN_STATE_FEATURES,
    CONTRACT_CHAIN_FEATURES,
)

LADDER_WIDTH = len(LADDER_FEATURES) + len(CONTRACT_CHAIN_FEATURES)


def _batch(rows: int = 2, contracts: int = 4, *, chain: bool = True) -> CausalPolicyBatch:
    return ChainPolicyBatch(
        candles=torch.zeros((rows, 5, len(CANDLE_FEATURES))),
        candle_mask=torch.ones((rows, 5), dtype=torch.bool),
        ladder=torch.zeros((rows, contracts, LADDER_WIDTH)),
        ladder_mask=torch.ones((rows, contracts), dtype=torch.bool),
        entry_action_mask=torch.ones((rows, contracts), dtype=torch.bool),
        account=torch.zeros((rows, ACCOUNT_FEATURES)),
        position=torch.zeros((rows, POSITION_FEATURES)),
        clock=torch.zeros((rows, CLOCK_FEATURES)),
        roles=("morning_entry",) * rows,
        chain=torch.zeros((rows, len(CHAIN_STATE_FEATURES))) if chain else None,
    )


def test_the_member_is_capacity_neutral_against_the_frozen_baseline() -> None:
    """118 against 120, counted from both built modules rather than asserted."""

    assert computed_parameter_count() == 118
    assert baseline_parameter_count() == 120
    assert computed_parameter_count() <= baseline_parameter_count()


def test_the_entry_and_exit_split_is_what_the_budget_is_charged_against() -> None:
    model = ChainStateLifecyclePolicy()
    entry_modules = ("shared_state", "contract_base", "direction", "depth", "wait")
    entry = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.split(".")[0] in entry_modules
    )
    assert entry == 94
    assert computed_parameter_count() - entry == 24


def test_the_frozen_baseline_module_is_not_imported_for_mutation() -> None:
    """The member sits beside the pinned baseline; it must not subclass it."""

    from v5.research import causal_day_compact_shared_lifecycle as frozen

    assert not issubclass(ChainStateLifecyclePolicy, frozen.CompactSharedLifecyclePolicy)


def test_a_batch_without_chain_state_is_refused_rather_than_zero_filled() -> None:
    """Training this member on silently absent state is the failure it avoids."""

    model = ChainStateLifecyclePolicy()
    with pytest.raises(ArchitectureInputError, match="requires chain-internal state"):
        model(_batch(chain=False))


def test_a_chain_tensor_of_the_wrong_width_is_refused() -> None:
    model = ChainStateLifecyclePolicy()
    batch = _batch()
    wrong = ChainPolicyBatch(**{**batch.__dict__, "chain": torch.zeros((2, 3))})
    with pytest.raises(ArchitectureInputError, match="chain state must be"):
        model(wrong)


def test_non_finite_chain_state_is_refused(  ) -> None:
    model = ChainStateLifecyclePolicy()
    batch = _batch()
    poisoned = batch.chain.clone()
    poisoned[0, 0] = float("nan")
    with pytest.raises(ArchitectureInputError, match="non-finite"):
        model(ChainPolicyBatch(**{**batch.__dict__, "chain": poisoned}))


def test_state_width_accounts_for_every_declared_group() -> None:
    assert STATE_WIDTH == 4 + CLOCK_FEATURES + len(CHAIN_STATE_FEATURES) + 6 + 2
    assert CONTRACT_WIDTH == 4 + len(CONTRACT_CHAIN_FEATURES)


def test_chain_state_can_reorder_the_ladder() -> None:
    """The whole point: state must change which contract is preferred.

    V5's failure was an additive offset that left the ranking untouched. Here the
    same ladder under two different chain states must be able to rank differently,
    because state reaches contracts through side and moneyness interactions.
    """

    torch.manual_seed(0)
    model = ChainStateLifecyclePolicy()
    batch = _batch(rows=1, contracts=6)
    ladder = batch.ladder.clone()
    ladder[0, :, LADDER_FEATURES.index("is_call")] = torch.tensor(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    )
    ladder[0, :, LADDER_FEATURES.index("moneyness_itm_points")] = torch.tensor(
        [-5.0, -10.0, -15.0, -5.0, -10.0, -15.0]
    )

    orders = set()
    for scale in (-4.0, 4.0):
        chain = torch.full((1, len(CHAIN_STATE_FEATURES)), scale)
        scores = model(ChainPolicyBatch(**{**batch.__dict__, "ladder": ladder, "chain": chain}))
        orders.add(tuple(torch.argsort(scores.contract_logits[0], descending=True).tolist()))

    assert len(orders) > 1, "chain state cannot reorder the ladder; this is V5 again"
