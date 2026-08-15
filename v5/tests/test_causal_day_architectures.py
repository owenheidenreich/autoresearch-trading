from __future__ import annotations

import torch

from v5.research import causal_day_architectures as arch


DIMS = arch.ArchitectureDimensions(
    candle_features=5,
    ladder_features=7,
    account_features=3,
    position_features=4,
    clock_features=3,
    hidden_size=8,
)


def _batch() -> arch.CausalPolicyBatch:
    torch.manual_seed(7)
    candles = torch.randn(4, 6, 5)
    candle_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    ladder = torch.randn(4, 5, 7)
    ladder_mask = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    entry_action_mask = torch.tensor(
        [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    return arch.CausalPolicyBatch(
        candles=candles,
        candle_mask=candle_mask,
        ladder=ladder,
        ladder_mask=ladder_mask,
        entry_action_mask=entry_action_mask,
        account=torch.randn(4, 3),
        position=torch.randn(4, 4),
        clock=torch.randn(4, 3),
        roles=(
            "morning_entry",
            "morning_exit",
            "afternoon_entry",
            "afternoon_exit",
        ),
    )


def test_every_declared_architecture_has_the_common_forward_contract() -> None:
    for name in (
        "shallow_joint",
        "shallow_four_head",
        "neural_joint",
        "neural_four_head",
        "four_independent",
    ):
        torch.manual_seed(11)
        model = arch.build_architecture(name, DIMS)
        scores = model(_batch())
        assert scores.contract_logits.shape == (4, 5)
        assert scores.abstain_logits.shape == (4,)
        assert scores.exit_logits.shape == (4, 2)
        assert arch.trainable_parameter_count(model) > 0
        assert torch.isneginf(scores.contract_logits[0, 2:]).all()
        assert scores.active_logits(0).shape == (6,)
        assert scores.active_logits(1).shape == (2,)


def test_masked_future_candles_and_ladder_nodes_cannot_change_scores() -> None:
    torch.manual_seed(17)
    model = arch.build_architecture("neural_four_head", DIMS).eval()
    clean = _batch()
    dirty = arch.CausalPolicyBatch(
        candles=torch.where(
            clean.candle_mask.unsqueeze(-1), clean.candles, clean.candles + 1_000_000.0
        ),
        candle_mask=clean.candle_mask,
        ladder=torch.where(
            clean.ladder_mask.unsqueeze(-1), clean.ladder, clean.ladder - 1_000_000.0
        ),
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
    assert torch.equal(torch.isneginf(a.contract_logits), torch.isneginf(b.contract_logits))
    finite = torch.isfinite(a.contract_logits)
    assert torch.allclose(a.contract_logits[finite], b.contract_logits[finite])


def test_shared_and_independent_forms_encode_the_intended_ownership() -> None:
    shared = arch.build_architecture("neural_four_head", DIMS)
    independent = arch.build_architecture("four_independent", DIMS)
    assert hasattr(shared, "encoder")
    assert set(independent.encoders) == set(arch.ROLES)
    assert len({id(value) for value in independent.encoders.values()}) == 4


def test_four_heads_route_by_origin_role_not_wall_clock_proxy() -> None:
    torch.manual_seed(23)
    model = arch.build_architecture("neural_four_head", DIMS).eval()
    with torch.no_grad():
        model.exit_heads["morning_exit"].weight.zero_()
        model.exit_heads["morning_exit"].bias.copy_(torch.tensor([1.0, 2.0]))
        model.exit_heads["afternoon_exit"].weight.zero_()
        model.exit_heads["afternoon_exit"].bias.copy_(torch.tensor([3.0, 4.0]))
    scores = model(_batch())
    assert torch.equal(scores.active_logits(1), torch.tensor([1.0, 2.0]))
    assert torch.equal(scores.active_logits(3), torch.tensor([3.0, 4.0]))


def test_entry_logits_are_masked_to_affordable_actions_not_whole_context_chain() -> None:
    torch.manual_seed(29)
    model = arch.build_architecture("neural_four_head", DIMS).eval()
    batch = _batch()
    scores = model(batch)
    assert torch.isfinite(scores.contract_logits[0, 0])
    assert torch.isneginf(scores.contract_logits[0, 1:]).all()
    assert torch.isfinite(scores.contract_logits[2, 0])
    assert torch.isneginf(scores.contract_logits[2, 1:]).all()
