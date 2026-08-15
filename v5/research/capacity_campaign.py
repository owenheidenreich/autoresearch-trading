"""Known-answer capacity campaign for the compact shared lifecycle entry phase.

The 2026-08-15 external adversarial review found that the 120-parameter
lifecycle design is compared against a full-corpus evidence budget although its
own chronology trains the first outer fit on a 404-session prefix, and that the
20-observations-per-parameter rule this project inherited was never measured.
This module measures the capacity law instead of arguing about it: it plants a
synthetic entry edge the architecture can provably represent, trains the real
entry phase on n sessions, and reads recovery and null discipline against the
generator's true expected values, which are known exactly because we own the
world.

Nothing here touches real targets, real economics, a vendor, or reserved data.
Worlds are synthetic; their dependence structure is calibrated to the measured
autocorrelation and noise scale recorded in
`research/findings/EFFECTIVE_SAMPLE_SIZE_2026_08_14.md` (10-27 minute
autocorrelation times) and the print-artifact finding (parity-residual SD
$102.60 against a ~$20-26 round trip).
"""
from __future__ import annotations

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
    training_sessions: tuple[int, ...] = (243, 404, 650, 890)
    trials_per_cell: int = 40
    max_epochs: int = 400
    plateau_tolerance: float = 1e-4
    plateau_patience: int = 20
    minibatch_rows: int = 8192
    learning_rate: float = 0.01
    recovery_fraction_of_oracle: float = 0.5
    recovery_rate_required: float = 0.8
    null_entry_rate_tolerance: float = 0.05
    null_clean_rate_required: float = 0.95

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


def train_entry_phase(
    world: World, *, seed: int, law: CampaignLaw = LAW
) -> CompactSharedLifecyclePolicy:
    """Fit the real entry phase: exit head stays frozen, as the protocol orders.

    The loss is the protocol's Phase-A value law: masked entry logits regress
    on realized entry dollars and WAIT regresses on its structural floor $0.

    Training runs to convergence rather than a fixed epoch count — the V1
    campaign proved a frozen 60-epoch budget binds before sample size does
    (receipt `capacity_known_answer_2026_08_15/receipt.json`). The stop reads
    only the epoch-mean training loss, so it is outcome-blind: it stops after
    `plateau_patience` consecutive epochs whose relative improvement is below
    `plateau_tolerance`, or at `max_epochs`.
    """

    torch.manual_seed(seed)
    model = CompactSharedLifecyclePolicy()
    for parameter in model.exit.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=law.learning_rate)
    rows = world.batch.batch_size
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    stale_epochs = 0
    for _ in range(law.max_epochs):
        order = torch.randperm(rows, generator=generator)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, rows, law.minibatch_rows):
            index = order[start : start + law.minibatch_rows]
            scores = model(_row_batch(world, index))
            target = world.realized_entry_usd[index]
            mask = world.batch.entry_action_mask[index]
            predicted = scores.contract_logits
            entry_loss = torch.mean((predicted[mask] - target[mask]) ** 2)
            wait_loss = torch.mean(scores.abstain_logits**2)
            loss = entry_loss + wait_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            batches += 1
        epoch_loss /= max(batches, 1)
        if epoch_loss < best_loss * (1.0 - law.plateau_tolerance):
            best_loss = epoch_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= law.plateau_patience:
                break
    return model.eval()


@dataclass(frozen=True)
class TrialResult:
    oof_policy_ev_per_minute: float
    oracle_ev_per_minute: float
    entry_rate: float
    train_apparent_edge_usd: float
    recovered: bool
    null_clean: bool


@torch.no_grad()
def evaluate(
    model: CompactSharedLifecyclePolicy,
    train_world: World,
    score_world: World,
    *,
    law: CampaignLaw = LAW,
) -> TrialResult:
    def policy_metrics(world: World) -> tuple[Tensor, Tensor, Tensor]:
        scores = model(world.batch)
        predicted = scores.contract_logits.masked_fill(
            ~world.batch.entry_action_mask, -torch.inf
        )
        best, node = predicted.max(dim=1)
        enter = best > scores.abstain_logits
        return enter, node, best

    enter, node, _ = policy_metrics(score_world)
    picked_ev = score_world.true_entry_ev_usd[
        torch.arange(len(node)), node
    ].nan_to_num()
    policy_ev = torch.where(enter, picked_ev, torch.zeros_like(picked_ev))
    oracle = score_world.true_entry_ev_usd.nan_to_num(nan=-torch.inf).max(dim=1).values
    oracle_ev = torch.clamp(oracle, min=0.0)

    train_enter, train_node, _ = policy_metrics(train_world)
    train_realized = train_world.realized_entry_usd[
        torch.arange(len(train_node)), train_node
    ].nan_to_num()
    taken = train_enter.sum().clamp(min=1).to(torch.float32)
    apparent = (torch.where(train_enter, train_realized, torch.zeros_like(train_realized)).sum() / taken)

    result_policy = float(policy_ev.mean())
    result_oracle = float(oracle_ev.mean())
    entry_rate = float(enter.to(torch.float32).mean())
    return TrialResult(
        oof_policy_ev_per_minute=result_policy,
        oracle_ev_per_minute=result_oracle,
        entry_rate=entry_rate,
        train_apparent_edge_usd=float(apparent),
        recovered=result_policy >= law.recovery_fraction_of_oracle * result_oracle
        and result_oracle > 0.0,
        null_clean=entry_rate <= law.null_entry_rate_tolerance,
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
        seed=trial_seed + 1_000_000,
        law=law,
    )
    model = train_entry_phase(train_world, seed=trial_seed, law=law)
    return evaluate(model, train_world, score_world, law=law)


def wilson_lower(successes: int, trials: int, z: float = 1.6449) -> float:
    if trials == 0:
        return 0.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4 * trials * trials))
    return (centre - margin) / denominator


def reference_weights(
    effect_usd_per_sd: float, law: CampaignLaw = LAW
) -> CompactSharedLifecyclePolicy:
    """Hand-built weights proving the planted edge is representable.

    `direction(state) * side` carries `effect * x * side` through a small
    linearised tanh; `contract_base` carries the per-contract drag. If these
    weights recover most of the oracle, a failed trained fit is a sample-size
    result rather than an architecture excuse.
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
