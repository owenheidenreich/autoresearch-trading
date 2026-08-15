from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from v4.model.hypothesis_protocol import (
    GeneralizationWindow,
    MarketStructureCache,
    SurfaceDecision,
    SurfaceVariant,
    load_surface_decisions,
    registered_surface_variants,
    surface_loss,
    window_seed,
)


def test_window_seed_is_stable_for_same_dates() -> None:
    window = GeneralizationWindow(
        train_start="2026-01-02",
        train_end="2026-01-30",
        calibration_start="2026-02-02",
        calibration_end="2026-02-13",
        selection_start="2026-02-17",
        selection_end="2026-02-27",
        audit_start="2026-03-02+2025-10-01",
        audit_end="2026-03-31+2025-12-31",
    )

    assert window.window_id == GeneralizationWindow(**window.__dict__).window_id
    assert window_seed(11, window.window_id) == window_seed(11, window.window_id)
    assert window_seed(22, window.window_id) != window_seed(11, window.window_id)


def test_market_structure_features_are_fixed_width() -> None:
    cache = MarketStructureCache()
    features = cache.features_for(datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc))

    assert features.shape == (28,)
    assert np.isfinite(features).all()
    assert features[0] > 0


def test_market_structure_lag_preserves_decision_time_bucket_semantics() -> None:
    cache = object.__new__(MarketStructureCache)
    cache.decision_context_lag_minutes = 1
    close = np.linspace(7400.0, 7500.0, 391, dtype=float)
    high = close + 1.0
    low = close - 1.0
    cache.spx_by_day = {"2026-06-30": {"close": close, "high": high, "low": low}}
    cache.spy_vwap_by_day = {
        "2026-06-30": {
            "close": close.copy(),
            "vwap": np.cumsum(close) / np.arange(1, len(close) + 1),
            "std": np.maximum.accumulate(np.linspace(1.0, 2.0, len(close))),
        }
    }
    cache.vix_by_day = {"2026-06-30": np.full(len(close), 17.0)}
    cache.omar_by_day = {
        "2026-06-30": {"high": high[0], "low": low[0], "mid": close[0], "range": 2.0}
    }
    cache.prev_close_by_day = {"2026-06-30": 7390.0}

    at_ten = cache.features_for(datetime(2026, 6, 30, 14, 0, tzinfo=timezone.utc))
    at_eleven_thirty = cache.features_for(datetime(2026, 6, 30, 15, 30, tzinfo=timezone.utc))
    at_one_thirty = cache.features_for(datetime(2026, 6, 30, 17, 30, tzinfo=timezone.utc))

    assert np.isclose(at_ten[23], 29.0 / 390.0)
    assert at_ten[24:28].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert at_eleven_thirty[24:28].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert at_one_thirty[24:28].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_surface_decisions_keep_atm_action_space_narrow() -> None:
    cache = MarketStructureCache()
    variant = SurfaceVariant(
        name="atm_structure_pressure_huber",
        action_space="atm",
        market_mode="structure",
        token_mode="pressure",
        loss_mode="huber",
    )
    decisions = load_surface_decisions(
        [Path("data/processed/spxw_0dte_neural_derived/2026-01-02.pkl")],
        policy_index=1,
        variant=variant,
        market_cache=cache,
    )

    tradable_counts = [int(d.token_mask.sum()) for d in decisions if d.token_mask.any()]
    assert tradable_counts
    assert max(tradable_counts) <= 2


def test_surface_decision_side_loss_is_finite() -> None:
    _ = SurfaceDecision(
        session="2026-01-02",
        decision_time=datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
        scalar_features=np.zeros(4, dtype=np.float32),
        token_features=np.zeros((4, 3), dtype=np.float32),
        token_mask=np.array([True, True, True, True]),
        labels=np.array([50.0, -10.0, -20.0, 70.0], dtype=np.float32),
        offsets=np.array([-5.0, -5.0, 0.0, 0.0], dtype=np.float32),
        rights=np.array(["C", "P", "C", "P"], dtype=object),
        contract_ids=np.array(["c1", "p1", "c2", "p2"], dtype=object),
        market_last=np.zeros(7, dtype=np.float32),
    )
    pred = torch.tensor([[0.0, 0.5, -0.1, -0.2, 0.7]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.5, -0.1, -0.2, 0.7]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True, True, True]])
    is_call = torch.tensor([[True, False, True, False]])
    is_put = torch.tensor([[False, True, False, True]])

    loss, parts = surface_loss(
        pred,
        target,
        mask,
        is_call,
        is_put,
        target_scale=100.0,
        loss_mode="decision_side",
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(parts["side_rank"])


def test_surface_contract_rank_loss_is_finite() -> None:
    pred = torch.tensor([[0.0, 0.1, 0.2, 0.9, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 1.5, -0.2, 0.0, 0.8]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True, True, True]])
    is_call = torch.tensor([[True, False, True, False]])
    is_put = torch.tensor([[False, True, False, True]])
    pattern_target = torch.ones((1, 4), dtype=torch.float32)
    value_target = torch.ones((1, 4), dtype=torch.float32)
    pattern_logit = torch.zeros((1, 4), dtype=torch.float32)
    value_logit = torch.zeros((1, 4), dtype=torch.float32)

    loss, parts = surface_loss(
        pred,
        target,
        mask,
        is_call,
        is_put,
        target_scale=100.0,
        loss_mode="aplus_side_value_rank",
        pattern_logit=pattern_logit,
        value_logit=value_logit,
        pattern_target=pattern_target,
        value_target=value_target,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(parts["contract_rank_margin"])
    assert parts["contract_rank_margin"] > 0


def test_registered_surface_variants_are_unique() -> None:
    names = [variant.name for variant in registered_surface_variants()]

    assert len(names) == len(set(names))
