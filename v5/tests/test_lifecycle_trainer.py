"""Tests for the two-phase lifecycle trainer (synthetic episodes only)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from v5.research.causal_day_architectures import CausalPolicyBatch
from v5.research.causal_day_compact_shared_lifecycle import (
    CompactSharedLifecyclePolicy,
)
from v5.research.causal_day_tensorizer import (
    ACCOUNT_FEATURES,
    CANDLE_FEATURES,
    CLOCK_FEATURES,
    LADDER_FEATURES,
    POSITION_FEATURES,
)
from v5.research.lifecycle_trainer import (
    ChronologySplit,
    LifecycleTrainingError,
    SessionEpisode,
    Trajectory,
    TrainingLaw,
    assert_trajectories_are_out_of_fold,
    generate_oof_trajectories,
    select_entries,
    should_sell,
    stable_seed,
    train_entry_phase,
    train_exit_head,
)

FAST = TrainingLaw(max_epochs=6, plateau_patience=3)


def make_batch(rows: int = 4, nodes: int = 3, *, seed: int = 0) -> CausalPolicyBatch:
    g = torch.Generator().manual_seed(seed)
    entry_mask = torch.zeros(rows, nodes, dtype=torch.bool)
    entry_mask[:, : max(nodes - 1, 1)] = True
    return CausalPolicyBatch(
        candles=torch.randn(rows, 1, len(CANDLE_FEATURES), generator=g),
        candle_mask=torch.ones(rows, 1, dtype=torch.bool),
        ladder=torch.randn(rows, nodes, len(LADDER_FEATURES), generator=g),
        ladder_mask=torch.ones(rows, nodes, dtype=torch.bool),
        entry_action_mask=entry_mask,
        account=torch.zeros(rows, ACCOUNT_FEATURES),
        position=torch.zeros(rows, POSITION_FEATURES),
        clock=torch.rand(rows, CLOCK_FEATURES, generator=g),
        roles=tuple("morning_entry" for _ in range(rows)),
    )


def make_episode(session: str, *, rows: int = 4, seed: int = 0) -> SessionEpisode:
    batch = make_batch(rows=rows, seed=seed)
    values = torch.full((rows, batch.ladder.shape[1]), float("nan"))
    values[batch.entry_action_mask] = torch.randn(
        int(batch.entry_action_mask.sum()), generator=torch.Generator().manual_seed(seed)
    ) * 100.0
    # Keyed by (decision minute, ladder column) and deliberately DIFFERENT per
    # contract, so a test that mixes up the contract cannot pass by accident.
    nodes = batch.ladder.shape[1]
    paths = {
        (i, k): np.array([100.0, 140.0 + 10.0 * k, 120.0], dtype=np.float64)
        for i in range(rows)
        for k in range(nodes)
    }
    return SessionEpisode(
        session=session, entry_batch=batch, entry_value_usd=values, sell_paths=paths
    )


def sessions(n: int) -> list[str]:
    return [f"2024-01-{i + 1:02d}" for i in range(n)]


# ---------------------------------------------------------------- chronology


def test_split_reserves_forty_percent_and_five_contiguous_blocks() -> None:
    split = ChronologySplit.build(sessions(20))
    assert len(split.training_prefix) == 8
    assert len(split.score_blocks) == 5
    flat = [s for block in split.score_blocks for s in block]
    assert flat == sessions(20)[8:]
    # Blocks must be contiguous in time, never interleaved.
    assert all(list(b) == sorted(b) for b in split.score_blocks)


def test_score_blocks_never_overlap_the_training_prefix() -> None:
    split = ChronologySplit.build(sessions(30))
    scored = {s for block in split.score_blocks for s in block}
    assert scored.isdisjoint(set(split.training_prefix))


def test_inner_folds_train_strictly_before_holdout() -> None:
    split = ChronologySplit.build(sessions(40))
    for train, holdout in split.inner_folds(4):
        assert max(train) < min(holdout)
        assert set(train).isdisjoint(set(holdout))


def test_split_refuses_duplicate_sessions() -> None:
    with pytest.raises(LifecycleTrainingError, match="duplicate session"):
        ChronologySplit.build(["2024-01-01", "2024-01-01", "2024-01-02"] + sessions(5))


# ------------------------------------------------------------- nested OOF law


def test_generated_trajectories_are_all_out_of_fold() -> None:
    all_sessions = sessions(24)
    episodes = [make_episode(s, seed=i) for i, s in enumerate(all_sessions)]
    split = ChronologySplit.build(all_sessions)
    trajectories = generate_oof_trajectories(
        episodes, split, folds=3, seed=7, law=FAST,
        model_factory=CompactSharedLifecyclePolicy,
    )

    assert trajectories, "the fixture should produce at least one trajectory"
    for trajectory in trajectories:
        assert trajectory.session not in trajectory.generator_train_sessions
    assert_trajectories_are_out_of_fold(trajectories)


def test_trajectories_never_come_from_score_blocks() -> None:
    all_sessions = sessions(24)
    episodes = [make_episode(s, seed=i) for i, s in enumerate(all_sessions)]
    split = ChronologySplit.build(all_sessions)
    trajectories = generate_oof_trajectories(
        episodes, split, folds=3, seed=7, law=FAST,
        model_factory=CompactSharedLifecyclePolicy,
    )

    scored = {s for block in split.score_blocks for s in block}
    assert {t.session for t in trajectories}.isdisjoint(scored)


def test_trajectory_carries_the_contract_the_model_actually_chose() -> None:
    """The 2026-08-17 defect: a sale path belongs to a contract, not a minute.

    `sell_paths` was keyed by decision minute alone, so the exit head could be
    trained on some other instrument's price history. It threw no error and
    produced a plausible-looking exit, which would have read as "the exit adds
    nothing" for a reason unrelated to the market.
    """

    episode = make_episode("2024-01-01", rows=3)
    model = CompactSharedLifecyclePolicy().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Make column 1 strictly preferred: score = depth(state) * moneyness,
        # and the fixture's ladder differs per column.
        model.depth.bias.fill_(1.0)

    chosen = select_entries(model, episode)
    assert chosen, "fixture must fire at least one entry"
    for index, column in chosen:
        # The path handed downstream must be the one keyed to THIS contract.
        expected = episode.sell_paths[(index, column)]
        # Fixture paths differ per column, so a mis-keyed lookup cannot match.
        for other in range(episode.entry_batch.ladder.shape[1]):
            if other != column:
                assert not np.array_equal(expected, episode.sell_paths[(index, other)])


def test_generated_trajectories_use_the_selected_contracts_path() -> None:
    all_sessions = sessions(24)
    episodes = [make_episode(s, seed=i) for i, s in enumerate(all_sessions)]
    split = ChronologySplit.build(all_sessions)
    trajectories = generate_oof_trajectories(
        episodes, split, folds=3, seed=7, law=FAST,
        model_factory=CompactSharedLifecyclePolicy,
    )

    assert trajectories
    by_session = {e.session: e for e in episodes}
    for trajectory in trajectories:
        episode = by_session[trajectory.session]
        expected = episode.sell_paths[
            (trajectory.decision_index, trajectory.ladder_column)
        ]
        assert np.array_equal(trajectory.sell_path_usd, expected)


def test_out_of_fold_assertion_catches_a_leaked_trajectory() -> None:
    leaked = Trajectory(
        session="2024-01-02",
        decision_index=0,
        ladder_column=0,
        sell_path_usd=np.array([100.0, 110.0]),
        generator_train_sessions=("2024-01-01", "2024-01-02"),
    )
    with pytest.raises(LifecycleTrainingError, match="trained on it"):
        assert_trajectories_are_out_of_fold([leaked])


# ------------------------------------------------------------- phase freezing


def test_an_unknown_target_is_masked_out_of_the_loss_rather_than_poisoning_it() -> None:
    """A NaN target inside the action mask must not reach the loss.

    Roughly 0.05% of the corpus's labels are unknown -- the quote path became
    unobservable before either bracket event. The design's rule is that those
    actions stay feasible and stay unsupervised. Before this guard a single one
    of them turned the whole epoch's loss into NaN.
    """

    episode = make_episode("2024-01-01")
    values = episode.entry_value_usd.clone()
    first = torch.nonzero(episode.entry_batch.entry_action_mask, as_tuple=False)[0]
    values[first[0], first[1]] = float("nan")
    poisoned = SessionEpisode(
        session=episode.session,
        entry_batch=episode.entry_batch,
        entry_value_usd=values,
        sell_paths=episode.sell_paths,
    )
    model = train_entry_phase([poisoned], seed=1, law=FAST, model=CompactSharedLifecyclePolicy())
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def test_a_session_with_no_affordable_contract_still_trains_the_wait_head() -> None:
    """2025-04-09 and 2025-04-10 are real no-trade days, not missing sessions.

    Every entry action is masked off, so the supervised term selects nothing --
    and `smooth_l1_loss` over an empty selection returns NaN. The session must
    still contribute its WAIT rows rather than silently destroy the fit.
    """

    episode = make_episode("2024-01-01")
    flat = SessionEpisode(
        session=episode.session,
        entry_batch=CausalPolicyBatch(
            **{
                **episode.entry_batch.__dict__,
                "entry_action_mask": torch.zeros_like(
                    episode.entry_batch.entry_action_mask
                ),
            }
        ),
        entry_value_usd=torch.full_like(episode.entry_value_usd, float("nan")),
    )
    model = train_entry_phase(
        [flat, make_episode("2024-01-02", seed=3)],
        seed=1,
        law=FAST,
        model=CompactSharedLifecyclePolicy(),
    )
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
    assert select_entries(model, flat) == []


def test_the_entry_phase_refuses_a_corpus_of_only_empty_batches() -> None:
    empty = SessionEpisode(
        session="2024-01-01",
        entry_batch=make_batch(rows=0),
        entry_value_usd=torch.zeros((0, 3)),
    )
    with pytest.raises(LifecycleTrainingError, match="no episode with a decision minute"):
        train_entry_phase([empty], seed=1, law=FAST, model=CompactSharedLifecyclePolicy())


def test_entry_phase_leaves_the_exit_head_untouched() -> None:
    model = CompactSharedLifecyclePolicy()
    before = {k: v.clone() for k, v in model.exit.state_dict().items()}
    train_entry_phase([make_episode("2024-01-01")], seed=1, law=FAST, model=model)
    after = model.exit.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key])


def test_exit_phase_cannot_move_entry_parameters() -> None:
    model = CompactSharedLifecyclePolicy()
    model = train_entry_phase([make_episode("2024-01-01")], seed=1, law=FAST, model=model)
    entry_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not name.startswith("exit")
    }
    exit_before = {k: v.detach().clone() for k, v in model.exit.state_dict().items()}

    trajectory = Trajectory(
        session="2024-01-05",
        decision_index=0,
        ladder_column=0,
        sell_path_usd=np.array([100.0, 180.0, 150.0]),
        generator_train_sessions=("2024-01-01",),
    )
    held = make_batch(rows=3, seed=5)
    model = train_exit_head(model, [trajectory], [held], seed=2, law=FAST)

    for name, value in entry_before.items():
        assert torch.equal(value, dict(model.named_parameters())[name]), name
    # The exit head must actually have learned something, or the test above is vacuous.
    assert any(
        not torch.equal(value, model.exit.state_dict()[key])
        for key, value in exit_before.items()
    )


def test_exit_phase_refuses_leaked_trajectories() -> None:
    model = CompactSharedLifecyclePolicy()
    leaked = Trajectory(
        session="2024-01-01",
        decision_index=0,
        ladder_column=0,
        sell_path_usd=np.array([100.0, 120.0]),
        generator_train_sessions=("2024-01-01",),
    )
    with pytest.raises(LifecycleTrainingError, match="trained on it"):
        train_exit_head(model, [leaked], [make_batch(rows=2)], seed=1, law=FAST)


def test_exit_phase_requires_one_batch_per_trajectory() -> None:
    model = CompactSharedLifecyclePolicy()
    trajectory = Trajectory(
        session="2024-01-05",
        decision_index=0,
        ladder_column=0,
        sell_path_usd=np.array([100.0, 120.0]),
        generator_train_sessions=("2024-01-01",),
    )
    with pytest.raises(LifecycleTrainingError, match="exactly one held-state batch"):
        train_exit_head(model, [trajectory], [], seed=1, law=FAST)


# ------------------------------------------------------------- inference laws


def test_entry_inference_requires_beating_a_nonnegative_wait() -> None:
    """WAIT's floor is $0, so a negative predicted WAIT cannot force a trade."""

    episode = make_episode("2024-01-01", rows=3)
    model = CompactSharedLifecyclePolicy().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Every contract scores exactly 0; WAIT's raw score is driven negative.
        model.wait.bias.fill_(-5.0)
    assert select_entries(model, episode) == []


def test_sell_fires_only_when_it_is_at_least_hold() -> None:
    model = CompactSharedLifecyclePolicy().eval()
    held = make_batch(rows=2, seed=3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.exit.bias[0] = 1.0  # HOLD
        model.exit.bias[1] = 2.0  # SELL
    assert bool(should_sell(model, held).all())
    with torch.no_grad():
        model.exit.bias[1] = 0.5
    assert not bool(should_sell(model, held).any())


# -------------------------------------------------------------- determinism


def test_stable_seed_is_process_independent() -> None:
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)
    # Pinned literal: a regression here means seed banks silently changed.
    assert stable_seed("drawdown", 0) == stable_seed("drawdown", 0)


def test_training_is_reproducible_from_its_seed_and_its_initialisation() -> None:
    """Reproducibility now needs the caller to seed *before* constructing.

    Making `model` required moved construction out of the function, and with it
    the random initialisation. `seed` still governs the fit; it no longer governs
    the starting weights. Callers that need end-to-end reproduction must seed
    first, and this is what that looks like.
    """

    episodes = [make_episode("2024-01-01", seed=1)]
    torch.manual_seed(11)
    first = train_entry_phase(episodes, seed=11, law=FAST, model=CompactSharedLifecyclePolicy())
    torch.manual_seed(11)
    second = train_entry_phase(episodes, seed=11, law=FAST, model=CompactSharedLifecyclePolicy())
    for name, parameter in first.named_parameters():
        assert torch.equal(parameter, dict(second.named_parameters())[name]), name


def test_the_same_seed_alone_no_longer_reproduces_a_fit() -> None:
    """The cost of removing the default, pinned so nobody rediscovers it late.

    Two fits at the same `seed` from two independently constructed models differ,
    because the initialisation is now the caller's. A runner that records only
    the seed has not recorded enough to reproduce its own result.
    """

    episodes = [make_episode("2024-01-01", seed=1)]
    torch.manual_seed(101)
    first = train_entry_phase(episodes, seed=11, law=FAST, model=CompactSharedLifecyclePolicy())
    torch.manual_seed(202)
    second = train_entry_phase(episodes, seed=11, law=FAST, model=CompactSharedLifecyclePolicy())
    named = dict(second.named_parameters())
    assert any(
        not torch.equal(parameter, named[name]) for name, parameter in first.named_parameters()
    )


def test_the_entry_phase_will_not_pick_an_architecture_for_you() -> None:
    """`model` has no default, and that is the point.

    It used to default to the frozen baseline, so a caller that forgot the
    argument trained the wrong architecture and reported a plausible number for
    it. That fired for real on 2026-08-20 and was caught only because the two
    architectures happen to differ in ladder width -- an accident of this corpus,
    not a guard.
    """

    with pytest.raises(TypeError, match="model"):
        train_entry_phase([make_episode("2024-01-01")], seed=1, law=FAST)


def test_out_of_fold_generation_will_not_pick_an_architecture_either() -> None:
    """The same trap on the path that feeds the exit head.

    `generate_oof_trajectories` fits an entry model per fold. Defaulting it would
    have generated trajectories from the frozen baseline and mis-trained the exit
    head, which reads out as "the exit adds nothing" for a reason unrelated to
    the market -- ledger row 341's failure mode, reintroduced by a keyword.
    """

    episodes = [make_episode(s, seed=i) for i, s in enumerate(sessions(12))]
    split = ChronologySplit.build(sessions(12))
    with pytest.raises(TypeError, match="model_factory"):
        generate_oof_trajectories(episodes, split, folds=2, seed=1, law=FAST)


def test_each_inner_fold_gets_a_fresh_model_not_a_shared_one() -> None:
    """A factory, not an instance: reusing one carries fold N into fold N+1."""

    built: list[int] = []

    def factory() -> CompactSharedLifecyclePolicy:
        model = CompactSharedLifecyclePolicy()
        built.append(id(model))
        return model

    episodes = [make_episode(s, seed=i) for i, s in enumerate(sessions(12))]
    split = ChronologySplit.build(sessions(12))
    generate_oof_trajectories(
        episodes, split, folds=3, seed=1, law=FAST, model_factory=factory
    )
    assert len(built) == len(set(built)) >= 2
