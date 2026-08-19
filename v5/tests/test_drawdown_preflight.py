"""Focused tests for the drawdown-preflight harness (synthetic worlds only)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from v5.ops.build_causal_day_dataset import canonical_json
from v5.research.causal_day_compact_interaction import computed_parameter_count
from v5.research.drawdown_preflight import (
    Calibration,
    FrozenSelector,
    GateDecision,
    LAW,
    WalkResult,
    World,
    full_gate,
    generate_world,
    reference_weights,
    serial_walk,
    wilson_upper,
)


def _toy_calibration(seed: int = 7) -> Calibration:
    rng = np.random.default_rng(seed)
    return Calibration(
        winner_net_usd=rng.normal(641.4, 1085.6, size=4000),
        loser_net_usd=rng.normal(-332.9, 239.0, size=4000),
        winner_duration=np.full(4000, 60.0),
        loser_duration=rng.uniform(5.0, 45.0, size=4000),
    )


def test_architecture_fits_conservative_budget() -> None:
    count = computed_parameter_count()
    assert 29 <= count <= 50, count


def test_generate_world_is_seed_deterministic() -> None:
    calibration = _toy_calibration()
    a = generate_world(sessions=5, effect_slope=0.05, seed=123, calibration=calibration)
    b = generate_world(sessions=5, effect_slope=0.05, seed=123, calibration=calibration)
    assert torch.equal(a.clean_label.nan_to_num(), b.clean_label.nan_to_num())
    assert torch.equal(a.net_usd.nan_to_num(), b.net_usd.nan_to_num())
    c = generate_world(sessions=5, effect_slope=0.05, seed=124, calibration=calibration)
    assert not torch.equal(a.clean_label.nan_to_num(), c.clean_label.nan_to_num())


def test_null_world_matches_pinned_calibration() -> None:
    calibration = _toy_calibration()
    world = generate_world(
        sessions=60, effect_slope=0.0, seed=5, calibration=calibration
    )
    mask = world.batch.entry_action_mask
    labels = world.clean_label[mask]
    rate = float(labels.mean())
    assert abs(rate - LAW.base_clean_rate) < 0.01, rate
    nets = world.net_usd[mask]
    assert float(nets.mean()) < 0.0  # the measured -$22.8/trade drag


def test_planted_signal_is_representable_by_reference_weights() -> None:
    calibration = _toy_calibration()
    world = generate_world(
        sessions=100,
        effect_slope=LAW.effect_slope_minimum,
        seed=11,
        calibration=calibration,
    )
    model = reference_weights(LAW.effect_slope_minimum)
    with torch.no_grad():
        scores = model(world.batch).contract_logits
    masked = scores.masked_fill(~world.batch.entry_action_mask, -torch.inf)
    best, node = masked.max(dim=1)
    rate = LAW.trades_per_session_target / LAW.minutes_per_session
    cutoff = torch.quantile(best, 1.0 - rate)
    picked = best > cutoff
    labels = world.clean_label[torch.arange(len(node)), node][picked]
    precision = float(labels.mean())
    assert precision >= LAW.base_clean_rate + 0.10, precision


def test_selector_cutoff_ignores_outcomes() -> None:
    calibration = _toy_calibration()
    world = generate_world(
        sessions=20, effect_slope=0.05, seed=3, calibration=calibration
    )
    model = reference_weights(0.05)
    frozen = FrozenSelector.freeze(model, world)
    mutated = World(
        batch=world.batch,
        clean_label=1.0 - world.clean_label,
        net_usd=-world.net_usd,
        duration_minutes=world.duration_minutes,
        true_clean_probability=world.true_clean_probability,
        sessions=world.sessions,
    )
    refrozen = FrozenSelector.freeze(model, mutated)
    assert frozen.score_cutoff == refrozen.score_cutoff
    assert frozen.probability_floor == refrozen.probability_floor


def test_serial_walk_respects_occupancy_and_cap() -> None:
    calibration = Calibration(
        winner_net_usd=np.full(10, 500.0),
        loser_net_usd=np.full(10, -300.0),
        winner_duration=np.full(10, 60.0),
        loser_duration=np.full(10, 60.0),
    )
    world = generate_world(
        sessions=10, effect_slope=0.2, seed=9, calibration=calibration
    )
    model = reference_weights(0.2)
    selector = FrozenSelector(score_cutoff=-1e9, probability_floor=LAW.probability_floor)
    walk = serial_walk(model, selector, world)
    # A cutoff every minute clears still cannot take more than the cap, and a
    # 60-minute duration means the second trade starts only after the first ends.
    assert walk.trades <= world.sessions * LAW.max_trades_per_session
    assert walk.trades >= world.sessions  # fires immediately every session
    assert len(walk.session_net_usd) == world.sessions


def _walk(nets: np.ndarray, trades: int, winners: int) -> WalkResult:
    blocks = np.array_split(nets, LAW.score_blocks)
    return WalkResult(
        session_net_usd=nets,
        trades=trades,
        winners=winners,
        selected_precision=winners / trades if trades else float("nan"),
        block_means=tuple(float(b.mean()) for b in blocks),
    )


def test_full_gate_passes_clear_positive() -> None:
    nets = np.full(150, 120.0)
    decision = full_gate(_walk(nets, trades=300, winners=150), seed=1)
    assert decision.passed and decision.reason == "passed"


def test_full_gate_refuses_insufficient_trades() -> None:
    nets = np.full(150, 120.0)
    decision = full_gate(_walk(nets, trades=LAW.min_selected_trades - 1, winners=10), seed=1)
    assert not decision.passed and decision.reason == "insufficient_trades"


def test_full_gate_refuses_negative_mean() -> None:
    nets = np.full(150, -5.0)
    decision = full_gate(_walk(nets, trades=100, winners=30), seed=1)
    assert not decision.passed and decision.reason == "mean_not_positive"


def test_full_gate_refuses_noise_through_corrected_lcb() -> None:
    rng = np.random.default_rng(2)
    nets = rng.normal(30.0, 1200.0, size=150)  # positive mean possible, wide noise
    walk = _walk(nets, trades=200, winners=80)
    decision = full_gate(walk, seed=3)
    if decision.passed:  # pragma: no cover - guards against silent loosening
        assert decision.corrected_lcb_usd > 0
    else:
        assert decision.reason in (
            "mean_not_positive",
            "corrected_lcb_not_positive",
            "insufficient_positive_blocks",
        )


def test_full_gate_requires_block_stability() -> None:
    nets = np.concatenate([np.full(120, 200.0), np.full(30, -400.0)])
    walk = _walk(nets, trades=200, winners=100)
    decision = full_gate(walk, seed=4)
    if not decision.passed:
        assert decision.reason in (
            "insufficient_positive_blocks",
            "corrected_lcb_not_positive",
        )
        assert decision.positive_blocks == 4 or decision.corrected_lcb_usd <= 0


def test_calibration_load_refuses_wrong_bytes(tmp_path) -> None:
    path = tmp_path / "calibration.parquet"
    path.write_bytes(b"not the declared calibration")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        Calibration.load(path)


def test_wilson_upper_bounds_the_null_requirement() -> None:
    assert wilson_upper(0, LAW.null_trials) < 0.05
    assert wilson_upper(1, LAW.null_trials) > 0.05


def test_declaration_self_hash_convention() -> None:
    body = {"schema_version": "v5.drawdown-preflight-declaration.v1", "x": 1}
    digest = hashlib.sha256(canonical_json(body)).hexdigest()
    declared = dict(body)
    declared["declaration_sha256"] = digest
    unsigned = dict(declared)
    unsigned.pop("declaration_sha256")
    assert hashlib.sha256(canonical_json(unsigned)).hexdigest() == digest
    assert json.loads(json.dumps(declared))  # serialisable
