"""Known-answer development harness for the compact shared lifecycle entry phase.

Classification: DEVELOPMENT DIAGNOSTIC, never a binding power measurement.
The 2026-08-15 second-round external review established that campaigns V1 and
V2 measured a simplified, badly-conditioned training implementation rather
than data capacity: raw-dollar MSE through unscaled features, an early stop
that returned the deteriorated last model instead of the best checkpoint, and
process-salted `hash()` seeds that made the declared seed bank irreproducible.

This version mirrors the production training law of
`v5/ops/train_causal_day_action_value.py`: SHA-256 stable seeds, training-world
feature standardisation with clipping, dollar targets scaled by 1000,
Smooth-L1 loss, AdamW with weight decay, gradient clipping, and best-checkpoint
restoration. Its verdicts still bind only the exact synthetic task, effect
family, noise law and training law tested; they can never close the lifecycle
member, certify full-pipeline power, or authorize a fit or purchase.

Nothing here touches real targets, real economics, a vendor, or reserved data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor

from v5.research.causal_day_architectures import CausalPolicyBatch
from v5.research.causal_day_compact_interaction import (
    CONTRACT_BASE_FEATURES,
    SIDE_INDEX,
    STATE_CANDLE_INDICES,
)
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

BODY_INDEX = CANDLE_FEATURES.index("body_points")
SPREAD_INDEX = LADDER_FEATURES.index("spread")
ASK_INDEX = LADDER_FEATURES.index("ask")


def stable_seed(*values: object) -> int:
    """SHA-256 seed derivation, mirroring the production trainer.

    Never use language-runtime `hash()`: it is salted per process, which made
    the V1/V2 seed banks irreproducible and falsified V2's identical-seeds
    declaration claim.
    """

    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class CampaignLaw:
    """Every constant a trial depends on; hashed into the declaration."""

    minutes_per_session: int = 150
    ladder_nodes: int = 16
    eligible_nodes: int = 6
    autocorrelation_minutes: float = 20.0
    signal_feature: str = "body_points"
    observation_noise_sd: float = 0.5
    distractor_sd: float = 1.0
    value_noise_usd: float = 100.0
    base_drag_usd: float = 15.0
    spread_low_usd: float = 2.0
    spread_high_usd: float = 10.0
    effect_small_usd_per_sd: float = 40.0
    effect_medium_usd_per_sd: float = 120.0
    score_sessions: int = 120
    # The actual entry-training prefix sizes of the frozen five-fold split at
    # a 1,011-session corpus, plus the owned-corpus anchor 243.
    training_sessions: tuple[int, ...] = (243, 404, 526, 648, 769, 890)
    trials_per_cell: int = 40
    max_epochs: int = 400
    plateau_tolerance: float = 1e-4
    plateau_patience: int = 20
    minibatch_rows: int = 8192
    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    target_scale_usd: float = 1000.0
    feature_clip: float = 10.0
    recovery_fraction_of_oracle: float = 0.5
    recovery_rate_required: float = 0.8
    null_entry_rate_tolerance: float = 0.05

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


LAW = CampaignLaw()


@dataclass(frozen=True)
class World:
    """One generated block of sessions plus the truth that scored it."""

    batch: CausalPolicyBatch
    realized_entry_usd: Tensor  # (rows, ladder_nodes); NaN off the entry mask
    true_entry_ev_usd: Tensor  # (rows, ladder_nodes); NaN off the entry mask
    sessions: int


@dataclass(frozen=True)
class WorldScaler:
    """Training-world standardisation for candle and ladder features."""

    candle_mean: Tensor
    candle_sd: Tensor
    ladder_mean: Tensor
    ladder_sd: Tensor

    @staticmethod
    def fit(world: World) -> "WorldScaler":
        candles = world.batch.candles.reshape(-1, world.batch.candles.shape[-1])
        ladder = world.batch.ladder.reshape(-1, world.batch.ladder.shape[-1])
        return WorldScaler(
            candle_mean=candles.mean(dim=0),
            candle_sd=candles.std(dim=0).clamp(min=1e-6),
            ladder_mean=ladder.mean(dim=0),
            ladder_sd=ladder.std(dim=0).clamp(min=1e-6),
        )

    def apply(self, world: World, *, clip: float) -> World:
        batch = world.batch
        candles = ((batch.candles - self.candle_mean) / self.candle_sd).clamp(
            -clip, clip
        )
        ladder = ((batch.ladder - self.ladder_mean) / self.ladder_sd).clamp(
            -clip, clip
        )
        return World(
            batch=CausalPolicyBatch(
                candles=candles,
                candle_mask=batch.candle_mask,
                ladder=ladder,
                ladder_mask=batch.ladder_mask,
                entry_action_mask=batch.entry_action_mask,
                account=batch.account,
                position=batch.position,
                clock=batch.clock,
                roles=batch.roles,
            ),
            realized_entry_usd=world.realized_entry_usd,
            true_entry_ev_usd=world.true_entry_ev_usd,
            sessions=world.sessions,
        )


def _clock_matrix(minutes: int) -> np.ndarray:
    """Five deterministic causal clock fields for one session."""

    t = np.arange(minutes, dtype=np.float64)
    fraction = t / max(minutes - 1, 1)
    morning = (fraction < 0.5).astype(np.float64)
    return np.stack(
        (
            fraction,
            morning,
            1.0 - morning,
            np.minimum(fraction * 2.0, 1.0),
            1.0 - fraction,
        ),
        axis=1,
    )


def generate_world(
    *, sessions: int, effect_usd_per_sd: float, seed: int, law: CampaignLaw = LAW
) -> World:
    """Generate sessions whose true entry values are known exactly.

    The latent signal is an AR(1) per session with the measured 20-minute
    autocorrelation time, observed through `body_points` with noise. The true
    expected value of entering node k at minute t is
    `effect * x_t * side_k - (base_drag + spread_k)`; WAIT is exactly zero.
    With effect 0 every possible entry is strictly negative, so a disciplined
    policy's only correct action in the null world is WAIT.
    """

    rng = np.random.default_rng(seed)
    minutes, nodes = law.minutes_per_session, law.ladder_nodes
    rows = sessions * minutes
    phi = math.exp(-1.0 / law.autocorrelation_minutes)
    innovation_sd = math.sqrt(1.0 - phi * phi)

    latent = np.empty((sessions, minutes))
    latent[:, 0] = rng.normal(0.0, 1.0, size=sessions)
    for minute in range(1, minutes):
        latent[:, minute] = phi * latent[:, minute - 1] + rng.normal(
            0.0, innovation_sd, size=sessions
        )

    candles = np.zeros((rows, 1, len(CANDLE_FEATURES)))
    flat_latent = latent.reshape(rows)
    for index in STATE_CANDLE_INDICES:
        candles[:, 0, index] = rng.normal(0.0, law.distractor_sd, size=rows)
    candles[:, 0, BODY_INDEX] = flat_latent + rng.normal(
        0.0, law.observation_noise_sd, size=rows
    )

    clock = np.tile(_clock_matrix(minutes), (sessions, 1))

    side = np.where(np.arange(nodes) % 2 == 0, 1.0, -1.0)
    moneyness = -np.linspace(2.0, 32.0, nodes)
    spread = rng.uniform(law.spread_low_usd, law.spread_high_usd, size=(rows, nodes))
    ladder = np.zeros((rows, nodes, len(LADDER_FEATURES)))
    ladder[:, :, SIDE_INDEX] = side
    ladder[:, :, LADDER_FEATURES.index("moneyness_itm_points")] = moneyness
    ladder[:, :, SPREAD_INDEX] = spread
    ladder[:, :, ASK_INDEX] = rng.uniform(100.0, 900.0, size=(rows, nodes))
    for name in CONTRACT_BASE_FEATURES:
        if name in ("ask", "spread"):
            continue
        ladder[:, :, LADDER_FEATURES.index(name)] = rng.normal(
            0.0, 1.0, size=(rows, nodes)
        )

    entry_mask = np.zeros((rows, nodes), dtype=bool)
    entry_mask[:, : law.eligible_nodes] = True

    drag = law.base_drag_usd + spread
    true_ev = effect_usd_per_sd * flat_latent[:, None] * side[None, :] - drag
    realized = true_ev + rng.normal(0.0, law.value_noise_usd, size=(rows, nodes))
    off_mask = ~entry_mask
    true_ev = np.where(off_mask, np.nan, true_ev)
    realized = np.where(off_mask, np.nan, realized)

    batch = CausalPolicyBatch(
        candles=torch.as_tensor(candles, dtype=torch.float32),
        candle_mask=torch.ones(rows, 1, dtype=torch.bool),
        ladder=torch.as_tensor(ladder, dtype=torch.float32),
        ladder_mask=torch.ones(rows, nodes, dtype=torch.bool),
        entry_action_mask=torch.as_tensor(entry_mask),
        account=torch.zeros(rows, ACCOUNT_FEATURES),
        position=torch.zeros(rows, POSITION_FEATURES),
        clock=torch.as_tensor(clock, dtype=torch.float32),
        roles=tuple("morning_entry" for _ in range(rows)),
    )
    return World(
        batch=batch,
        realized_entry_usd=torch.as_tensor(realized, dtype=torch.float32),
        true_entry_ev_usd=torch.as_tensor(true_ev, dtype=torch.float32),
        sessions=sessions,
    )


def _row_batch(world: World, rows: Tensor) -> CausalPolicyBatch:
    b = world.batch
    return CausalPolicyBatch(
        candles=b.candles[rows],
        candle_mask=b.candle_mask[rows],
        ladder=b.ladder[rows],
        ladder_mask=b.ladder_mask[rows],
        entry_action_mask=b.entry_action_mask[rows],
        account=b.account[rows],
        position=b.position[rows],
        clock=b.clock[rows],
        roles=tuple("morning_entry" for _ in range(len(rows))),
    )


@dataclass(frozen=True)
class TrainingRecord:
    epochs_run: int
    stop_epoch_of_best: int
    best_loss: float


def train_entry_phase(
    world: World, *, seed: int, law: CampaignLaw = LAW
) -> tuple[CompactSharedLifecyclePolicy, TrainingRecord]:
    """Fit the real entry phase under the production-mirrored training law.

    Exit head frozen, as the protocol orders. Smooth-L1 on dollar targets
    scaled by `target_scale_usd`, AdamW, gradient clipping, and an
    outcome-blind plateau stop that RESTORES the best-loss checkpoint — the V2
    trainer returned the deteriorated terminal state, which biased recovery
    down. The stop reads training loss only and never touches the score world.
    """

    torch.manual_seed(seed)
    model = CompactSharedLifecyclePolicy()
    for parameter in model.exit.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=law.learning_rate, weight_decay=law.weight_decay
    )
    rows = world.batch.batch_size
    generator = torch.Generator().manual_seed(seed)
    scale = law.target_scale_usd
    smooth_l1 = torch.nn.SmoothL1Loss()
    best_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(law.max_epochs):
        epochs_run = epoch + 1
        order = torch.randperm(rows, generator=generator)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, rows, law.minibatch_rows):
            index = order[start : start + law.minibatch_rows]
            scores = model(_row_batch(world, index))
            target = world.realized_entry_usd[index] / scale
            mask = world.batch.entry_action_mask[index]
            predicted = scores.contract_logits
            entry_loss = smooth_l1(predicted[mask], target[mask])
            wait_loss = smooth_l1(
                scores.abstain_logits, torch.zeros_like(scores.abstain_logits)
            )
            loss = entry_loss + wait_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, law.gradient_clip)
            optimizer.step()
            epoch_loss += float(loss)
            batches += 1
        epoch_loss /= max(batches, 1)
        if epoch_loss < best_loss * (1.0 - law.plateau_tolerance):
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= law.plateau_patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), TrainingRecord(
        epochs_run=epochs_run,
        stop_epoch_of_best=best_epoch,
        best_loss=best_loss,
    )


@dataclass(frozen=True)
class TrialResult:
    seed: int
    oof_policy_ev_per_minute: float
    oracle_ev_per_minute: float
    entry_rate: float
    train_apparent_edge_usd: float
    recovered: bool
    null_abstained: bool
    epochs_run: int
    stop_epoch_of_best: int


@torch.no_grad()
def evaluate(
    model: CompactSharedLifecyclePolicy,
    train_world: World,
    score_world: World,
    *,
    seed: int = 0,
    training: TrainingRecord | None = None,
    law: CampaignLaw = LAW,
) -> TrialResult:
    """Score the policy against the generator's truth.

    This is a stateless per-minute diagnostic: no serial account, occupancy,
    controls, or session-level inference. `recovered` and `null_abstained`
    are development statistics for the entry training law only — abstention in
    a world where every entry is negative by construction is NOT a
    full-pipeline false-pass rate.
    """

    def policy_metrics(world: World) -> tuple[Tensor, Tensor]:
        scores = model(world.batch)
        predicted = scores.contract_logits.masked_fill(
            ~world.batch.entry_action_mask, -torch.inf
        )
        best, node = predicted.max(dim=1)
        enter = best > scores.abstain_logits
        return enter, node

    enter, node = policy_metrics(score_world)
    picked_ev = score_world.true_entry_ev_usd[
        torch.arange(len(node)), node
    ].nan_to_num()
    policy_ev = torch.where(enter, picked_ev, torch.zeros_like(picked_ev))
    oracle = score_world.true_entry_ev_usd.nan_to_num(nan=-torch.inf).max(dim=1).values
    oracle_ev = torch.clamp(oracle, min=0.0)

    train_enter, train_node = policy_metrics(train_world)
    train_realized = train_world.realized_entry_usd[
        torch.arange(len(train_node)), train_node
    ].nan_to_num()
    taken = train_enter.sum().clamp(min=1).to(torch.float32)
    apparent = (
        torch.where(train_enter, train_realized, torch.zeros_like(train_realized)).sum()
        / taken
    )

    result_policy = float(policy_ev.mean())
    result_oracle = float(oracle_ev.mean())
    entry_rate = float(enter.to(torch.float32).mean())
    return TrialResult(
        seed=seed,
        oof_policy_ev_per_minute=result_policy,
        oracle_ev_per_minute=result_oracle,
        entry_rate=entry_rate,
        train_apparent_edge_usd=float(apparent),
        recovered=result_policy >= law.recovery_fraction_of_oracle * result_oracle
        and result_oracle > 0.0,
        null_abstained=entry_rate <= law.null_entry_rate_tolerance,
        epochs_run=training.epochs_run if training else 0,
        stop_epoch_of_best=training.stop_epoch_of_best if training else 0,
    )


def run_trial(
    *,
    training_sessions: int,
    effect_usd_per_sd: float,
    trial_seed: int,
    law: CampaignLaw = LAW,
) -> TrialResult:
    """One seeded world, one entry fit, one out-of-fold read of the truth."""

    train_world = generate_world(
        sessions=training_sessions,
        effect_usd_per_sd=effect_usd_per_sd,
        seed=trial_seed,
        law=law,
    )
    score_world = generate_world(
        sessions=law.score_sessions,
        effect_usd_per_sd=effect_usd_per_sd,
        seed=stable_seed("score-world", trial_seed),
        law=law,
    )
    scaler = WorldScaler.fit(train_world)
    scaled_train = scaler.apply(train_world, clip=law.feature_clip)
    scaled_score = scaler.apply(score_world, clip=law.feature_clip)
    model, record = train_entry_phase(scaled_train, seed=trial_seed, law=law)
    return evaluate(
        model,
        scaled_train,
        scaled_score,
        seed=trial_seed,
        training=record,
        law=law,
    )


def wilson_lower(successes: int, trials: int, z: float = 1.6449) -> float:
    if trials == 0:
        return 0.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4 * trials * trials))
    return (centre - margin) / denominator


def wilson_upper(successes: int, trials: int, z: float = 1.6449) -> float:
    if trials == 0:
        return 1.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4 * trials * trials))
    return (centre + margin) / denominator


def reference_weights(
    effect_usd_per_sd: float, law: CampaignLaw = LAW
) -> CompactSharedLifecyclePolicy:
    """Hand-built weights proving the planted edge is representable.

    Constructed for UNSCALED worlds: `direction(state) * side` carries
    `effect * x * side` through a small linearised tanh; `contract_base`
    carries the per-contract drag. If these weights recover most of the
    oracle, a failed trained fit is a training-law result rather than an
    architecture excuse.
    """

    epsilon = 0.05
    model = CompactSharedLifecyclePolicy().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        body_position = STATE_CANDLE_INDICES.index(
            CANDLE_FEATURES.index("body_points")
        )
        model.shared_state.weight[0, body_position] = epsilon
        model.direction.weight[0, 0] = effect_usd_per_sd / epsilon
        model.contract_base.weight[0, CONTRACT_BASE_FEATURES.index("spread")] = -1.0
        model.contract_base.bias[0] = -law.base_drag_usd
    return model
