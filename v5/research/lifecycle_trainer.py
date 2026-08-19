"""Two-phase trainer for the compact shared SPXW 0DTE lifecycle policy.

This implements the frozen post-acquisition law in
`v5/work/entry-exit-attribution/COMPACT_SHARED_LIFECYCLE_PROTOCOL_V1.md` under
the amendments adopted by `v5/governance/DEVELOPMENT_CHARTER_2026_08.md`.

The protocol's two hard requirements are both structural, so both are enforced
here rather than left to a runner's discipline:

1. **Nested out-of-fold trajectories.** The exit head learns to manage entries,
   so it must be trained on entries the entry model did not memorise. An entry
   policy may never generate a trajectory for a session it trained on. Every
   prior exit result in this project was measured on *random* entries, which is
   the one regime where a conditional exit's value cancels (ledger row 341), so
   this ordering is the finding made mechanical.
2. **Frozen entry before exit.** The shared representation and all entry
   parameters are frozen before the exit head trains, so exit fitting cannot
   move the entries it is supposed to manage. `train_exit_head` asserts this
   bitwise after training rather than trusting `requires_grad`.

Chronology: the earliest 40% of sessions form the training prefix; the rest
split into five contiguous score blocks. Scaling, targets, fitting, and any
calibration read strictly earlier sessions only. Score blocks are never touched
by fitting — `ChronologySplit` exposes them but no training function accepts
them.

This module is data-source agnostic: it consumes `SessionEpisode` records, so it
is exercised on known-answer synthetic episodes today and binds to the real
dataset builder when the corpus exists. Nothing here contacts a vendor, spends
money, promotes a model, or authorizes an order.
"""
from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from v5.research.causal_day_architectures import CausalPolicyBatch
from v5.research.causal_day_compact_lifecycle_targets import (
    build_exit_action_targets,
    exit_action_value_loss,
)
from v5.research.causal_day_compact_shared_lifecycle import (
    CompactSharedLifecyclePolicy,
)

TRAINING_PREFIX_FRACTION = 0.40
SCORE_BLOCKS = 5
TARGET_SCALE_USD = 1_000.0


class LifecycleTrainingError(RuntimeError):
    """A protocol requirement was violated before or during a fit."""


def stable_seed(*values: object) -> int:
    """SHA-256 seed derivation; never the process-salted builtin `hash()`."""

    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class SessionEpisode:
    """One complete causal session: entry decisions and held-position paths.

    `entry_batch` holds every causal decision minute of the session.
    `entry_value_usd` is the executable ENTER-versus-WAIT dollar value per
    contract action; WAIT's structural floor is $0. It is NaN off the entry
    mask, and also NaN *on* it wherever the label is unknown -- the path became
    unobservable before either bracket event. Those actions stay feasible and
    stay unsupervised: `train_entry_phase` masks them out of the loss rather
    than removing them from the action set.
    `sell_paths` maps **(decision-minute index, ladder column)** to the
    executable sale value at each minute the position would be held, under the
    simulator's first-later-bid / validated-settlement law.

    The key carries the contract, not only the minute. A sale path belongs to
    the exact contract that was bought — a 4500 call and a 4600 put entered in
    the same minute have entirely different paths — so keying by minute alone
    would train the exit head on some other instrument's price history. That
    defect throws no error and produces a plausible-looking exit, which would
    then read as "the exit adds nothing" for a reason unrelated to the market.
    """

    session: str
    entry_batch: CausalPolicyBatch
    entry_value_usd: Tensor
    sell_paths: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    held_batch_builder: Callable[[int, int], CausalPolicyBatch] | None = None


@dataclass(frozen=True)
class ChronologySplit:
    """Deterministic chronological split; score blocks are fit-forbidden."""

    training_prefix: tuple[str, ...]
    score_blocks: tuple[tuple[str, ...], ...]

    @staticmethod
    def build(sessions: Sequence[str]) -> "ChronologySplit":
        ordered = sorted(sessions)
        if len(ordered) != len(set(ordered)):
            raise LifecycleTrainingError("duplicate session in the chronology")
        if len(ordered) < SCORE_BLOCKS + 1:
            raise LifecycleTrainingError("too few sessions to form the declared split")
        cut = math.floor(TRAINING_PREFIX_FRACTION * len(ordered))
        if cut < 1:
            raise LifecycleTrainingError("training prefix is empty")
        blocks = np.array_split(np.array(ordered[cut:], dtype=object), SCORE_BLOCKS)
        return ChronologySplit(
            training_prefix=tuple(ordered[:cut]),
            score_blocks=tuple(tuple(b.tolist()) for b in blocks),
        )

    def inner_folds(self, folds: int) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
        """Nested chronological folds inside the training prefix.

        Each fold returns `(train_sessions, holdout_sessions)` where every
        training session is strictly earlier than every holdout session, so an
        entry model never sees the future of the trajectories it generates.
        """

        if folds < 2:
            raise LifecycleTrainingError("nested trajectory generation needs at least two folds")
        prefix = list(self.training_prefix)
        if len(prefix) < folds + 1:
            raise LifecycleTrainingError("training prefix too small for the declared inner folds")
        chunks = [list(c) for c in np.array_split(np.array(prefix, dtype=object), folds)]
        out: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for index in range(1, folds):
            train = [s for chunk in chunks[:index] for s in chunk]
            holdout = chunks[index]
            if train and holdout:
                out.append((tuple(train), tuple(holdout)))
        if not out:
            raise LifecycleTrainingError("no usable nested fold was produced")
        return tuple(out)


@dataclass(frozen=True)
class TrainingLaw:
    """Production-mirrored optimisation constants (amendment A10)."""

    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    max_epochs: int = 200
    plateau_tolerance: float = 1e-4
    plateau_patience: int = 20
    smooth_l1_beta: float = 0.25


LAW = TrainingLaw()

ENTRY_MODULES = ("shared_state", "contract_base", "direction", "depth", "wait")


def _entry_state(model: CompactSharedLifecyclePolicy) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.split(".")[0] in ENTRY_MODULES
    }


def _plateau_fit(
    model: CompactSharedLifecyclePolicy,
    parameters: list[Tensor],
    step: Callable[[], Tensor],
    law: TrainingLaw,
) -> tuple[int, float]:
    """Shared optimisation loop with best-checkpoint restore.

    The V2 capacity harness returned the deteriorated terminal model and biased
    its own recovery down; restoring the best checkpoint is now law.
    """

    optimizer = torch.optim.AdamW(
        parameters, lr=law.learning_rate, weight_decay=law.weight_decay
    )
    best_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    epochs = 0
    for epoch in range(law.max_epochs):
        epochs = epoch + 1
        optimizer.zero_grad(set_to_none=True)
        loss = step()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, law.gradient_clip)
        optimizer.step()
        value = float(loss)
        if value < best_loss * (1.0 - law.plateau_tolerance):
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= law.plateau_patience:
                break
    model.load_state_dict(best_state)
    return epochs, best_loss


def train_entry_phase(
    episodes: Sequence[SessionEpisode],
    *,
    seed: int,
    law: TrainingLaw = LAW,
    model: CompactSharedLifecyclePolicy | None = None,
) -> CompactSharedLifecyclePolicy:
    """Fit the shared representation and entry heads. Exit head frozen."""

    if not episodes:
        raise LifecycleTrainingError("entry phase requires at least one episode")
    torch.manual_seed(seed)
    model = model or CompactSharedLifecyclePolicy()
    for parameter in model.exit.parameters():
        parameter.requires_grad_(False)
    trainable = [
        p
        for name, p in model.named_parameters()
        if name.split(".")[0] in ENTRY_MODULES and p.requires_grad
    ]
    before_exit = {k: v.detach().clone() for k, v in model.exit.state_dict().items()}

    def step() -> Tensor:
        total = torch.zeros((), dtype=torch.float32)
        count = 0
        for episode in episodes:
            if episode.entry_batch.batch_size == 0:
                continue
            scores = model(episode.entry_batch)
            target = episode.entry_value_usd / TARGET_SCALE_USD
            # Supervised only where a target exists. An action whose label is
            # unknown -- the path became unobservable before either bracket
            # event -- is masked out of the loss and never dropped from the
            # action set, so the policy may still choose it at inference while
            # nothing pretends to know what it was worth. Two real sessions
            # (2025-04-09/10) have no affordable contract at all: on those the
            # supervised term is empty, and `smooth_l1_loss` over an empty
            # selection returns NaN, which would silently poison every epoch.
            mask = episode.entry_batch.entry_action_mask & torch.isfinite(target)
            predicted = scores.contract_logits
            if mask.any():
                total = total + torch.nn.functional.smooth_l1_loss(
                    predicted[mask], target[mask], beta=law.smooth_l1_beta
                )
            # WAIT is trained toward its structural floor of $0, not left
            # gradient-free as the first capacity harness left it. It is trained
            # on the no-trade sessions too: standing down all day is a decision
            # the policy has to make, not a session to be skipped.
            total = total + torch.nn.functional.smooth_l1_loss(
                scores.abstain_logits,
                torch.zeros_like(scores.abstain_logits),
                beta=law.smooth_l1_beta,
            )
            count += 1
        if count == 0:
            raise LifecycleTrainingError("entry phase received no episode with a decision minute")
        return total / count

    _plateau_fit(model, trainable, step, law)
    after_exit = model.exit.state_dict()
    for key, value in before_exit.items():
        if not torch.equal(value, after_exit[key]):
            raise LifecycleTrainingError("entry phase modified the frozen exit head")
    return model.eval()


@dataclass(frozen=True)
class Trajectory:
    """One held position produced by an entry model that never saw its session."""

    session: str
    decision_index: int
    ladder_column: int
    sell_path_usd: np.ndarray
    generator_train_sessions: tuple[str, ...]


@torch.no_grad()
def select_entries(
    model: CompactSharedLifecyclePolicy, episode: SessionEpisode
) -> list[tuple[int, int]]:
    """`(decision minute, ladder column)` pairs the model prefers to waiting.

    Mirrors the protocol's inference law: there is no absolute threshold; enter
    only when the best contract's predicted value strictly exceeds WAIT, whose
    feasible floor is $0.

    Returns the chosen **contract** alongside the minute. An entry decision is
    not "buy at 10:14" but "buy this exact contract at 10:14", and everything
    downstream — the sale path, the held-state batch, the realised P&L — is a
    property of that contract.
    """

    scores = model(episode.entry_batch)
    best, node = scores.contract_logits.masked_fill(
        ~episode.entry_batch.entry_action_mask, -torch.inf
    ).max(dim=1)
    wait = torch.clamp(scores.abstain_logits, min=0.0)
    fire = torch.isfinite(best) & (best > wait)
    return [
        (int(index), int(node[index]))
        for index in torch.nonzero(fire, as_tuple=False).flatten().tolist()
    ]


def generate_oof_trajectories(
    episodes: Sequence[SessionEpisode],
    split: ChronologySplit,
    *,
    folds: int,
    seed: int,
    law: TrainingLaw = LAW,
) -> list[Trajectory]:
    """Nested out-of-fold entry trajectories for exit training.

    For each nested fold, an entry model trains on strictly earlier sessions and
    selects entries on the held-out sessions only. The resulting trajectories
    carry their generator's training sessions so the disjointness is auditable
    after the fact, not merely asserted here.
    """

    by_session = {e.session: e for e in episodes}
    trajectories: list[Trajectory] = []
    for fold_index, (train_sessions, holdout_sessions) in enumerate(
        split.inner_folds(folds)
    ):
        train_episodes = [by_session[s] for s in train_sessions if s in by_session]
        if not train_episodes:
            continue
        model = train_entry_phase(
            train_episodes, seed=stable_seed(seed, "inner-entry", fold_index), law=law
        )
        for session in holdout_sessions:
            episode = by_session.get(session)
            if episode is None:
                continue
            if session in train_sessions:  # pragma: no cover - guarded by construction
                raise LifecycleTrainingError(
                    f"nested fold leaked: {session} is both trained and generated"
                )
            for index, column in select_entries(model, episode):
                path = episode.sell_paths.get((index, column))
                if path is None or len(path) < 2:
                    continue
                trajectories.append(
                    Trajectory(
                        session=session,
                        decision_index=index,
                        ladder_column=column,
                        sell_path_usd=np.asarray(path, dtype=np.float64),
                        generator_train_sessions=tuple(train_sessions),
                    )
                )
    return trajectories


def assert_trajectories_are_out_of_fold(trajectories: Iterable[Trajectory]) -> None:
    """Refuse any trajectory whose generator trained on its own session."""

    for trajectory in trajectories:
        if trajectory.session in trajectory.generator_train_sessions:
            raise LifecycleTrainingError(
                f"trajectory for {trajectory.session} came from a model trained on it"
            )


def train_exit_head(
    model: CompactSharedLifecyclePolicy,
    trajectories: Sequence[Trajectory],
    held_batches: Sequence[CausalPolicyBatch],
    *,
    seed: int,
    law: TrainingLaw = LAW,
) -> CompactSharedLifecyclePolicy:
    """Fit only the exit head on out-of-fold trajectories.

    The shared representation and every entry parameter are frozen first and
    verified bitwise afterwards, so exit fitting provably cannot move the
    entries it manages.
    """

    if not trajectories:
        raise LifecycleTrainingError("exit phase requires at least one trajectory")
    if len(trajectories) != len(held_batches):
        raise LifecycleTrainingError("each trajectory needs exactly one held-state batch")
    assert_trajectories_are_out_of_fold(trajectories)

    torch.manual_seed(seed)
    frozen_before = _entry_state(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.split(".")[0] == "exit")
    trainable = [p for p in model.exit.parameters()]

    targets = [build_exit_action_targets(t.sell_path_usd) for t in trajectories]

    def step() -> Tensor:
        total = torch.zeros((), dtype=torch.float32)
        for batch, target in zip(held_batches, targets):
            scores = model(batch)
            total = total + exit_action_value_loss(
                scores,
                torch.as_tensor(target.q_sell_usd, dtype=torch.float32),
                torch.as_tensor(target.q_hold_usd, dtype=torch.float32),
                torch.as_tensor(target.hold_available, dtype=torch.bool),
            )
        return total / len(held_batches)

    _plateau_fit(model, trainable, step, law)

    frozen_after = _entry_state(model)
    for name, value in frozen_before.items():
        if not torch.equal(value, frozen_after[name]):
            raise LifecycleTrainingError(f"exit phase modified frozen entry parameter {name}")
    return model.eval()


@torch.no_grad()
def should_sell(model: CompactSharedLifecyclePolicy, held_batch: CausalPolicyBatch) -> Tensor:
    """Inference law: SELL fires when its value is at least HOLD's.

    Forced liquidation stays simulator-owned; this only answers the question the
    policy is allowed to answer.
    """

    exits = model(held_batch).exit_logits
    return exits[:, 1] >= exits[:, 0]
