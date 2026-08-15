from __future__ import annotations

import pandas as pd
import pytest
import torch

from v5.research.causal_day_architectures import (
    ArchitectureDimensions,
    CausalPolicyBatch,
    build_architecture,
)
from v5.research.causal_day_attribution import (
    activation_drift,
    decompose_contract_scores,
    score_decomposition_metrics,
    time_band,
)


DIMS = ArchitectureDimensions(
    candle_features=4,
    ladder_features=6,
    account_features=2,
    position_features=3,
    clock_features=5,
    hidden_size=3,
)


def _batch() -> CausalPolicyBatch:
    torch.manual_seed(101)
    return CausalPolicyBatch(
        candles=torch.randn(2, 4, 4),
        candle_mask=torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool
        ),
        ladder=torch.randn(2, 3, 6),
        ladder_mask=torch.ones(2, 3, dtype=torch.bool),
        entry_action_mask=torch.ones(2, 3, dtype=torch.bool),
        account=torch.randn(2, 2),
        position=torch.randn(2, 3),
        clock=torch.randn(2, 5),
        roles=("morning_entry", "afternoon_entry"),
    )


def test_additive_decomposition_reproduces_every_contract_score() -> None:
    torch.manual_seed(103)
    model = build_architecture("neural_four_head", DIMS).eval()
    batch = _batch()
    value = decompose_contract_scores(model, batch)
    expected = model(batch).contract_logits
    assert value.reconstruction_max_abs_error <= 1e-6
    assert torch.allclose(value.full, expected, atol=1e-6, rtol=0.0)
    assert torch.allclose(
        value.full,
        value.state_offset.unsqueeze(1) + value.contract_node,
        atol=1e-6,
        rtol=0.0,
    )


def test_candles_and_clock_cannot_change_contract_ranking_within_a_role() -> None:
    torch.manual_seed(107)
    model = build_architecture("neural_four_head", DIMS).eval()
    clean = _batch()
    changed = CausalPolicyBatch(
        candles=clean.candles * -9.0 + 40.0,
        candle_mask=clean.candle_mask,
        ladder=clean.ladder,
        ladder_mask=clean.ladder_mask,
        entry_action_mask=clean.entry_action_mask,
        account=clean.account,
        position=clean.position,
        clock=clean.clock * 13.0 - 7.0,
        roles=clean.roles,
    )
    a = model(clean).contract_logits
    b = model(changed).contract_logits
    assert torch.equal(a.argmax(dim=1), b.argmax(dim=1))
    assert torch.allclose(a[:, 1:] - a[:, :1], b[:, 1:] - b[:, :1], atol=1e-6)


def test_magnitude_only_loss_cannot_train_wait_or_exit_heads() -> None:
    """Job 39's training path consumed only ``contract_logits``."""

    torch.manual_seed(109)
    model = build_architecture("neural_four_head", DIMS)
    scores = model(_batch())
    loss = scores.contract_logits[torch.isfinite(scores.contract_logits)].sum()
    loss.backward()
    for head in model.entry_heads.values():
        assert head.abstain.weight.grad is None
        assert head.abstain.bias.grad is None
    for head in model.exit_heads.values():
        assert head.weight.grad is None
        assert head.bias.grad is None


@pytest.mark.parametrize(
    ("minute", "expected"),
    [
        ("09:35", "opening_0935_0949"),
        ("09:50", "magic_0950_1010"),
        ("10:11", "morning_rest_1011_1245"),
        ("12:46", "afternoon_pre_algo_1246_1319"),
        ("13:20", "algo_1320_1340"),
        ("15:00", "afternoon_rest_1341_1500"),
    ],
)
def test_time_bands_are_total_and_predeclared(minute: str, expected: str) -> None:
    assert time_band(minute) == expected


def test_activation_drift_uses_each_folds_frozen_cutoff() -> None:
    rows = []
    for fold in range(1, 6):
        for minute, score in (("09:35", float(fold)), ("10:00", float(fold + 1))):
            rows.append(
                {
                    "session": f"2025-01-{fold:02d}",
                    "entry_minute": minute,
                    "contract_id": f"{fold}-{minute}",
                    "fold": fold,
                    "full_score": score,
                    "state_offset": score / 2,
                }
            )
    result = activation_drift(
        pd.DataFrame(rows),
        fold_cutoffs={1: 10.0, 2: 10.0, 3: 3.5, 4: 4.5, 5: 5.5},
    )
    assert result["severe_calibration_drift"] is True
    assert [row["cutoff_above_every_scored_minute"] for row in result["folds"]] == [
        True,
        True,
        False,
        False,
        False,
    ]


def test_decomposition_metrics_prove_node_only_argmax() -> None:
    frame = pd.DataFrame(
        [
            {
                "session": "2025-01-02",
                "entry_minute": minute,
                "contract_id": contract,
                "state_offset": offset,
                "contract_node_contribution": node,
                "full_score": offset + node,
            }
            for minute, offset in (("09:35", 2.0), ("09:36", -1.0))
            for contract, node in (("C", 0.5), ("P", -0.5))
        ]
    )
    result = score_decomposition_metrics(frame)
    assert result["maximum_abs_full_minus_state_minus_node"] == 0.0
    assert result["state_offset_constant_within_every_minute"] is True
    assert result["full_argmax_equals_contract_node_argmax_rate"] == 1.0
