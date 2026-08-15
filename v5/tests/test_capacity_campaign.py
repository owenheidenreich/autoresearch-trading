from __future__ import annotations

import dataclasses

import torch

from v5.research.capacity_campaign import (
    LAW,
    evaluate,
    generate_world,
    reference_weights,
    run_trial,
    train_entry_phase,
    wilson_lower,
)

FAST = dataclasses.replace(
    LAW,
    training_sessions=(24,),
    trials_per_cell=1,
    max_epochs=4,
    plateau_patience=2,
    score_sessions=16,
)


def test_world_generation_is_deterministic_and_masked() -> None:
    first = generate_world(sessions=8, effect_usd_per_sd=40.0, seed=7, law=FAST)
    second = generate_world(sessions=8, effect_usd_per_sd=40.0, seed=7, law=FAST)
    assert torch.equal(first.batch.ladder, second.batch.ladder)
    assert torch.allclose(
        first.true_entry_ev_usd, second.true_entry_ev_usd, equal_nan=True
    )
    off_mask = ~first.batch.entry_action_mask
    assert torch.isnan(first.true_entry_ev_usd[off_mask]).all()
    assert first.batch.batch_size == 8 * FAST.minutes_per_session


def test_null_world_has_no_positive_entry_anywhere() -> None:
    world = generate_world(sessions=8, effect_usd_per_sd=0.0, seed=3, law=FAST)
    eligible = world.true_entry_ev_usd[world.batch.entry_action_mask]
    assert float(eligible.max()) < 0.0


def test_planted_edge_is_representable_by_the_real_architecture() -> None:
    model = reference_weights(LAW.effect_medium_usd_per_sd)
    train = generate_world(
        sessions=30, effect_usd_per_sd=LAW.effect_medium_usd_per_sd, seed=11, law=FAST
    )
    score = generate_world(
        sessions=30, effect_usd_per_sd=LAW.effect_medium_usd_per_sd, seed=12, law=FAST
    )
    result = evaluate(model, train, score, law=FAST)
    assert result.oracle_ev_per_minute > 0.0
    assert (
        result.oof_policy_ev_per_minute
        >= LAW.recovery_fraction_of_oracle * result.oracle_ev_per_minute
    )


def test_reference_weights_stay_flat_in_the_null_world() -> None:
    model = reference_weights(0.0)
    train = generate_world(sessions=12, effect_usd_per_sd=0.0, seed=21, law=FAST)
    score = generate_world(sessions=12, effect_usd_per_sd=0.0, seed=22, law=FAST)
    result = evaluate(model, train, score, law=FAST)
    assert result.null_clean
    assert result.oof_policy_ev_per_minute == 0.0


def test_training_freezes_the_exit_head() -> None:
    world = generate_world(sessions=6, effect_usd_per_sd=40.0, seed=5, law=FAST)
    torch.manual_seed(0)
    model = train_entry_phase(world, seed=5, law=FAST)
    assert not any(p.requires_grad for p in model.exit.parameters())
    trained = [p for n, p in model.named_parameters() if not n.startswith("exit")]
    assert all(p.requires_grad for p in trained)


def test_run_trial_is_deterministic() -> None:
    a = run_trial(
        training_sessions=10, effect_usd_per_sd=40.0, trial_seed=99, law=FAST
    )
    b = run_trial(
        training_sessions=10, effect_usd_per_sd=40.0, trial_seed=99, law=FAST
    )
    assert a == b


def test_convergence_training_keeps_improving_past_a_tiny_budget() -> None:
    world = generate_world(sessions=12, effect_usd_per_sd=120.0, seed=41, law=FAST)

    def training_loss(model) -> float:
        with torch.no_grad():
            scores = model(world.batch)
            mask = world.batch.entry_action_mask
            target = world.realized_entry_usd
            return float(
                torch.mean((scores.contract_logits[mask] - target[mask]) ** 2)
            )

    short = train_entry_phase(world, seed=41, law=FAST)
    longer_law = dataclasses.replace(FAST, max_epochs=40, plateau_patience=10)
    longer = train_entry_phase(world, seed=41, law=longer_law)
    assert training_loss(longer) < training_loss(short)


def test_wilson_lower_is_conservative() -> None:
    assert wilson_lower(0, 0) == 0.0
    assert 0.0 < wilson_lower(40, 40) < 1.0
    assert wilson_lower(20, 40) < 0.5
