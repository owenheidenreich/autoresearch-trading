from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from v5.research import causal_day_magnitude as magnitude
from v5.research.causal_day_architectures import (
    CausalPolicyBatch,
    computed_parameter_counts,
    declared_dimensions,
)


DIMENSIONS = declared_dimensions()


def _batch(roles=("morning_entry", "afternoon_entry")) -> CausalPolicyBatch:
    batch = len(roles)
    return CausalPolicyBatch(
        candles=torch.zeros((batch, 5, 18)),
        candle_mask=torch.ones((batch, 5), dtype=torch.bool),
        ladder=torch.zeros((batch, 4, 23)),
        ladder_mask=torch.ones((batch, 4), dtype=torch.bool),
        entry_action_mask=torch.tensor([[True, False, True, False]] * batch),
        account=torch.zeros((batch, 5)),
        position=torch.zeros((batch, 10)),
        clock=torch.zeros((batch, 5)),
        roles=roles,
    )


def test_the_reopened_models_are_the_exact_width_three_models_the_gate_counts() -> None:
    counts = computed_parameter_counts()
    for name in magnitude.PERMITTED_ARCHITECTURES:
        model = magnitude.build_magnitude_policy(name, DIMENSIONS)
        got = model(_batch())
        assert got.contract_logits.shape == (2, 4)
        assert torch.isfinite(got.contract_logits[:, (0, 2)]).all()
        assert torch.isneginf(got.contract_logits[:, (1, 3)]).all()
        assert magnitude.parameter_count(model) == counts[name]
        assert magnitude.parameter_count(model) <= 377


def test_folds_are_strictly_chronological_and_cover_150_score_sessions() -> None:
    sessions = [f"2025-01-{value:03d}" for value in range(243)]
    folds = magnitude.chronological_folds(sessions)
    assert len(folds) == 5
    assert sum(len(value.score_sessions) for value in folds) == 150
    assert all(max(value.train_sessions) < min(value.score_sessions) for value in folds)
    assert set().union(*(set(value.score_sessions) for value in folds)) == set(sessions[93:])


def test_magnitude_loss_uses_only_action_nodes() -> None:
    prediction = torch.zeros((1, 2), requires_grad=True)
    targets = torch.tensor([[-5.0, 999.0]])
    mask = torch.tensor([[True, False]])
    weights = torch.ones(5)
    loss = magnitude.magnitude_loss(prediction, targets, mask, weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad[0, 0].abs() > 0
    assert prediction.grad[0, 1].abs() == 0


def test_stratum_weights_are_fold_derived_and_capped() -> None:
    targets = np.asarray([[-5, 1, 5], [-4, 11, 20], [-3, 21, 30], [30, 30, 30]], float)
    got = magnitude.stratum_weights(targets)
    assert got.shape == (3, 5)
    assert np.isfinite(got).all()
    assert (got >= 0.0).all()
    assert (got > 0.0).any(axis=1).all()


def test_clock_walk_obeys_threshold_tie_break_and_occupancy() -> None:
    rows = []
    for minute, contract, score, spread in (
        ("10:00", "b", 31.0, 2.0),
        ("10:00", "a", 31.0, 1.0),
        ("10:30", "c", 40.0, 1.0),
        ("11:01", "d", 35.0, 1.0),
    ):
        rows.append(
            {
                "session": "2025-01-02",
                "entry_minute": minute,
                "contract_id": contract,
                "spread_usd": spread,
                "moneyness_itm_points": -5.0,
                "predicted_depth_60m": score,
                "clock_exit_minute_60m": "11:00" if minute == "10:00" else "12:00",
                "net_bid_60m_usd": 1.0,
                "net_mid_60m_usd": 4.0,
            }
        )
    got = magnitude.select_clock_trades(
        pd.DataFrame(rows), horizon=60, depth_threshold=30, trade_cap=2
    )
    assert got["contract_id"].tolist() == ["a", "d"]
    assert got["gross_mid_usd"].tolist() == pytest.approx([7.08, 7.08])
