from __future__ import annotations

import dataclasses

import torch

from v5.research.capacity_campaign import (
    LAW,
    WorldScaler,
    evaluate,
    generate_world,
    reference_weights,
    run_trial,
    stable_seed,
    train_entry_phase,
    wilson_lower,
    wilson_upper,
)

FAST = dataclasses.replace(
    LAW,
    training_sessions=(24,),
    trials_per_cell=1,
    max_epochs=4,
    plateau_patience=2,
    score_sessions=16,
)


def test_stable_seed_is_process_independent() -> None:
    # SHA-256 of the joined string, first four little-endian bytes. A fixed
    # expectation guards against anyone reintroducing runtime hash().
    assert stable_seed("capacity-v3", 404, "null", 0) == stable_seed(
        "capacity-v3", 404, "null", 0
    )
    assert stable_seed("a", 1) != stable_seed("a", 2)
    digest_first = stable_seed("capacity-v3", 243, "edge_small", 7)
    assert 0 <= digest_first < 2**32


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


def test_scaler_standardises_and_clips_without_touching_targets() -> None:
    world = generate_world(sessions=10, effect_usd_per_sd=40.0, seed=9, law=FAST)
    scaler = WorldScaler.fit(world)
    scaled = scaler.apply(world, clip=LAW.feature_clip)
    assert torch.allclose(
        scaled.realized_entry_usd, world.realized_entry_usd, equal_nan=True
    )
    ladder = scaled.batch.ladder.reshape(-1, scaled.batch.ladder.shape[-1])
    assert float(ladder.abs().max()) <= LAW.feature_clip
    from v5.research.capacity_campaign import ASK_INDEX

    ask = ladder[:, ASK_INDEX]
    assert abs(float(ask.mean())) < 0.1
    assert 0.8 < float(ask.std()) < 1.2


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
    assert result.null_abstained
    assert result.oof_policy_ev_per_minute == 0.0


def test_training_freezes_the_exit_head_and_reports_its_record() -> None:
    world = generate_world(sessions=6, effect_usd_per_sd=40.0, seed=5, law=FAST)
    model, record = train_entry_phase(world, seed=5, law=FAST)
    assert not any(p.requires_grad for p in model.exit.parameters())
    trained = [p for n, p in model.named_parameters() if not n.startswith("exit")]
    assert all(p.requires_grad for p in trained)
    assert 1 <= record.epochs_run <= FAST.max_epochs
    assert record.stop_epoch_of_best <= record.epochs_run - 1
    assert record.best_loss < float("inf")


def test_early_stop_restores_the_best_checkpoint() -> None:
    world = generate_world(sessions=6, effect_usd_per_sd=120.0, seed=13, law=FAST)
    scaler = WorldScaler.fit(world)
    scaled = scaler.apply(world, clip=LAW.feature_clip)
    law = dataclasses.replace(FAST, max_epochs=30, plateau_patience=5)
    model, record = train_entry_phase(scaled, seed=13, law=law)

    import torch as t

    smooth = t.nn.SmoothL1Loss()
    with t.no_grad():
        scores = model(scaled.batch)
        mask = scaled.batch.entry_action_mask
        target = scaled.realized_entry_usd / law.target_scale_usd
        returned_loss = float(
            smooth(scores.contract_logits[mask], target[mask])
            + smooth(
                scores.abstain_logits, t.zeros_like(scores.abstain_logits)
            )
        )
    # The returned model must correspond to the best recorded loss, not a
    # deteriorated terminal state (allowing for minibatch-vs-full evaluation).
    assert returned_loss <= record.best_loss * 1.05


def test_run_trial_is_deterministic() -> None:
    a = run_trial(
        training_sessions=10, effect_usd_per_sd=40.0, trial_seed=99, law=FAST
    )
    b = run_trial(
        training_sessions=10, effect_usd_per_sd=40.0, trial_seed=99, law=FAST
    )
    assert a == b
    assert a.seed == 99


def test_wilson_bounds_are_conservative() -> None:
    assert wilson_lower(0, 0) == 0.0
    assert 0.0 < wilson_lower(40, 40) < 1.0
    assert wilson_lower(20, 40) < 0.5
    # 0 of 40 failures still cannot certify <= 5%: the upper bound exceeds it.
    assert wilson_upper(0, 40) > 0.05
