"""Known-answer preflight for the drawdown-ordered lifecycle experiment.

Authorized by the signed `STOP_OVERRIDE_DRAWDOWN_LIFECYCLE_2026_08_15.md` and
`LEDGER_REOPENING_DRAWDOWN_LIFECYCLE_2026_08_15.md`. The reopening's kill
condition 1 makes this preflight binding: the real fit may not run until this
harness recovers a planted minimum-size edge through the FULL gate — serial
account, occupancy, session bootstrap, family correction and chronological
blocks — in at least 80% of trials, with zero full-gate false passes in the
null worlds.

Fidelity notes, encoding the 2026-08-15 capacity-campaign lessons:
- production-mirrored training law (standardised clipped features, AdamW,
  gradient clipping, plateau stop with BEST-checkpoint restore);
- SHA-256 stable seeds, never process-salted `hash()`;
- dollar outcomes resampled from the MEASURED calibration draws
  (`drawdown_calibration_v1.parquet`: 32,976 owned quote paths under the
  declared stop-or-hold law), not from an assumed distribution;
- within-session clustering planted at the measured label ICC;
- "recovery" and "false pass" are decisions of the complete gate, never an
  entry-rate or per-minute score diagnostic.

Nothing here touches real targets, real economics, a vendor, or reserved data.
Synthetic-world iterations of this harness are development; the binding
requirement is that the final declared version passes before any real fit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from v5.research.causal_day_architectures import CausalPolicyBatch
from v5.research.causal_day_compact_interaction import (
    CompactInteractionEntryPolicy,
    CONTRACT_BASE_FEATURES,
    MONEYNESS_INDEX,
    SIDE_INDEX,
    STATE_CANDLE_INDICES,
)
from v5.research.causal_day_tensorizer import (
    ACCOUNT_FEATURES,
    CANDLE_FEATURES,
    LADDER_FEATURES,
    POSITION_FEATURES,
)
from v5.research.capacity_campaign import stable_seed

BODY_INDEX = CANDLE_FEATURES.index("body_points")
SPREAD_INDEX = LADDER_FEATURES.index("spread")
ASK_INDEX = LADDER_FEATURES.index("ask")

CALIBRATION_PATH = Path(
    "/Volumes/AR_TRADING_DATA/derived/drawdown_calibration_v1.parquet"
)
CALIBRATION_SHA256 = (
    "83384d6f842791b84e9e8d85e2d6a95df2ab2e1d0355b3490eaf6746c28ee61d"
)


@dataclass(frozen=True)
class PreflightLaw:
    """Every constant a trial depends on; hashed into the declaration."""

    # World geometry, matching the capacity-harness conventions.
    sessions_total: int = 250
    train_sessions: int = 100
    score_sessions: int = 150
    score_blocks: int = 5
    minutes_per_session: int = 150
    ladder_nodes: int = 16
    eligible_nodes: int = 6
    autocorrelation_minutes: float = 20.0
    observation_noise_sd: float = 0.5
    distractor_sd: float = 1.0
    # Label law, pinned from the 2026-08-15 measurement of 32,976 owned paths.
    base_clean_rate: float = 0.3183
    label_session_shock_sd: float = 0.066  # reproduces the measured ICC ~0.02
    probability_clip: tuple[float, float] = (0.02, 0.98)
    # Planted effects: probability-slope per (latent SD x side). The minimum
    # slope and the selector target were calibrated by an outcome-blind
    # synthetic probe grid on 2026-08-15 (slopes 0.082/0.10/0.12 x targets
    # 2/4): 0.10 at target 4 was the smallest configuration reaching the
    # required 80% full-gate recovery. Its realized selected precision ~52%
    # against the 31.8% base is therefore the experiment's sensitivity
    # statement: a real negative rules out edges of about that size, and
    # nothing smaller.
    effect_slope_null: float = 0.0
    effect_slope_minimum: float = 0.10
    # Selector law, mirroring the corrected V5 causal selector.
    trades_per_session_target: float = 4.0
    max_trades_per_session: int = 2
    probability_floor: float = 0.342  # measured break-even precision L/(W+L)
    # Production-mirrored training law.
    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    minibatch_rows: int = 8192
    max_epochs: int = 400
    plateau_tolerance: float = 1e-4
    plateau_patience: int = 20
    feature_clip: float = 10.0
    # Gate law, mirroring the signed reopening.
    bootstrap_draws: int = 20000
    alpha: float = 0.05
    family_size: int = 3
    min_selected_trades: int = 20
    required_positive_blocks: int = 4
    # Campaign law.
    planted_trials: int = 40
    null_trials: int = 60
    recovery_rate_required: float = 0.8
    null_false_pass_wilson_upper_limit: float = 0.05

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


LAW = PreflightLaw()


@dataclass(frozen=True)
class Calibration:
    """Empirical draws the generator resamples; never an assumed distribution."""

    winner_net_usd: np.ndarray
    loser_net_usd: np.ndarray
    winner_duration: np.ndarray
    loser_duration: np.ndarray

    @staticmethod
    def load(path: Path = CALIBRATION_PATH) -> "Calibration":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != CALIBRATION_SHA256:
            raise RuntimeError(
                f"calibration draws hash mismatch: {digest} != {CALIBRATION_SHA256}"
            )
        table = pd.read_parquet(path)
        clean = table["clean"].to_numpy(dtype=bool)
        net = table["net_usd"].to_numpy(dtype=np.float64)
        duration = table["duration_minutes"].to_numpy(dtype=np.float64)
        return Calibration(
            winner_net_usd=net[clean],
            loser_net_usd=net[~clean],
            winner_duration=duration[clean],
            loser_duration=duration[~clean],
        )


@dataclass(frozen=True)
class World:
    """One generated block of sessions plus the truth that scored it."""

    batch: CausalPolicyBatch
    clean_label: Tensor  # (rows, nodes) float 0/1; NaN off the entry mask
    net_usd: Tensor  # (rows, nodes); realized net under the declared law
    duration_minutes: Tensor  # (rows, nodes)
    true_clean_probability: Tensor  # (rows, nodes)
    sessions: int


def _clock_matrix(minutes: int) -> np.ndarray:
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
    *,
    sessions: int,
    effect_slope: float,
    seed: int,
    calibration: Calibration,
    law: PreflightLaw = LAW,
) -> World:
    """Sessions whose clean-run truth is known exactly.

    A latent AR(1) per session (measured 20-minute autocorrelation time) is
    observed through `body_points` with noise. The true clean-run probability
    of node k at minute t is
    `clip(base + session_shock + effect_slope * x_t * side_k)`; dollar
    outcomes are resampled from the measured winner/loser draws conditional on
    the realized label. With effect 0 the unconditional expectancy is the
    measured -$22.8/trade, so the null world's only gate-passing route is
    luck the bootstrap must refuse.
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
    flat_latent = latent.reshape(rows)

    candles = np.zeros((rows, 1, len(CANDLE_FEATURES)))
    for index in STATE_CANDLE_INDICES:
        candles[:, 0, index] = rng.normal(0.0, law.distractor_sd, size=rows)
    candles[:, 0, BODY_INDEX] = flat_latent + rng.normal(
        0.0, law.observation_noise_sd, size=rows
    )

    clock = np.tile(_clock_matrix(minutes), (sessions, 1))

    side = np.where(np.arange(nodes) % 2 == 0, 1.0, -1.0)
    moneyness = -np.linspace(2.0, 32.0, nodes)
    ladder = np.zeros((rows, nodes, len(LADDER_FEATURES)))
    ladder[:, :, SIDE_INDEX] = side
    ladder[:, :, MONEYNESS_INDEX] = moneyness
    ladder[:, :, SPREAD_INDEX] = rng.uniform(2.0, 10.0, size=(rows, nodes))
    ladder[:, :, ASK_INDEX] = rng.uniform(100.0, 900.0, size=(rows, nodes))
    for name in CONTRACT_BASE_FEATURES:
        if name in ("ask", "spread"):
            continue
        ladder[:, :, LADDER_FEATURES.index(name)] = rng.normal(
            0.0, 1.0, size=(rows, nodes)
        )

    entry_mask = np.zeros((rows, nodes), dtype=bool)
    entry_mask[:, : law.eligible_nodes] = True

    shock = np.repeat(
        rng.normal(0.0, law.label_session_shock_sd, size=sessions), minutes
    )
    low, high = law.probability_clip
    probability = np.clip(
        law.base_clean_rate
        + shock[:, None]
        + effect_slope * flat_latent[:, None] * side[None, :],
        low,
        high,
    )
    label = (rng.uniform(size=(rows, nodes)) < probability).astype(np.float64)

    winner_index = rng.integers(0, len(calibration.winner_net_usd), size=(rows, nodes))
    loser_index = rng.integers(0, len(calibration.loser_net_usd), size=(rows, nodes))
    net = np.where(
        label > 0.5,
        calibration.winner_net_usd[winner_index],
        calibration.loser_net_usd[loser_index],
    )
    duration = np.where(
        label > 0.5,
        calibration.winner_duration[winner_index],
        calibration.loser_duration[loser_index],
    )

    off = ~entry_mask
    probability = np.where(off, np.nan, probability)
    label = np.where(off, np.nan, label)
    net = np.where(off, np.nan, net)
    duration = np.where(off, np.nan, duration)

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
        clean_label=torch.as_tensor(label, dtype=torch.float32),
        net_usd=torch.as_tensor(net, dtype=torch.float32),
        duration_minutes=torch.as_tensor(duration, dtype=torch.float32),
        true_clean_probability=torch.as_tensor(probability, dtype=torch.float32),
        sessions=sessions,
    )


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
        return World(
            batch=CausalPolicyBatch(
                candles=((batch.candles - self.candle_mean) / self.candle_sd).clamp(
                    -clip, clip
                ),
                candle_mask=batch.candle_mask,
                ladder=((batch.ladder - self.ladder_mean) / self.ladder_sd).clamp(
                    -clip, clip
                ),
                ladder_mask=batch.ladder_mask,
                entry_action_mask=batch.entry_action_mask,
                account=batch.account,
                position=batch.position,
                clock=batch.clock,
                roles=batch.roles,
            ),
            clean_label=world.clean_label,
            net_usd=world.net_usd,
            duration_minutes=world.duration_minutes,
            true_clean_probability=world.true_clean_probability,
            sessions=world.sessions,
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


def train_entry_model(
    world: World, *, seed: int, law: PreflightLaw = LAW
) -> tuple[CompactInteractionEntryPolicy, TrainingRecord]:
    """Fit the real 48-parameter policy on the clean-run label.

    Binary cross-entropy over eligible nodes only; AdamW; gradient clipping;
    outcome-blind plateau stop that RESTORES the best-loss checkpoint. The
    abstain head receives no gradient because the declared selector is a
    frozen rank cutoff plus probability floor, not abstain competition.
    """

    torch.manual_seed(seed)
    model = CompactInteractionEntryPolicy()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=law.learning_rate, weight_decay=law.weight_decay
    )
    bce = torch.nn.BCEWithLogitsLoss()
    rows = world.batch.batch_size
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    epochs_run = 0
    for epoch in range(law.max_epochs):
        epochs_run = epoch + 1
        order = torch.randperm(rows, generator=generator)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, rows, law.minibatch_rows):
            index = order[start : start + law.minibatch_rows]
            scores = model(_row_batch(world, index))
            mask = world.batch.entry_action_mask[index]
            logits = scores.contract_logits[mask]
            target = world.clean_label[index][mask]
            loss = bce(logits, target)
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
            stale = 0
        else:
            stale += 1
            if stale >= law.plateau_patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), TrainingRecord(
        epochs_run=epochs_run, stop_epoch_of_best=best_epoch, best_loss=best_loss
    )


@dataclass(frozen=True)
class FrozenSelector:
    """Rank cutoff and probability floor, frozen from training sessions only."""

    score_cutoff: float
    probability_floor: float

    @staticmethod
    def freeze(
        model: CompactInteractionEntryPolicy,
        train_world: World,
        *,
        law: PreflightLaw = LAW,
    ) -> "FrozenSelector":
        with torch.no_grad():
            scores = model(train_world.batch).contract_logits
        best = scores.masked_fill(
            ~train_world.batch.entry_action_mask, -torch.inf
        ).max(dim=1).values
        rate = law.trades_per_session_target / law.minutes_per_session
        cutoff = float(torch.quantile(best, 1.0 - rate))
        if not math.isfinite(cutoff):
            raise RuntimeError("selector cutoff is not finite; refuse to freeze")
        # Attainability guard (the V4 lesson): at least one training minute
        # must clear both the cutoff and the probability floor.
        floor_logit = math.log(law.probability_floor / (1.0 - law.probability_floor))
        attainable = bool(((best > cutoff) & (best >= floor_logit)).any())
        if not attainable:
            raise RuntimeError(
                "no training minute clears cutoff and floor; selector unattainable"
            )
        return FrozenSelector(score_cutoff=cutoff, probability_floor=law.probability_floor)


@dataclass(frozen=True)
class WalkResult:
    session_net_usd: np.ndarray  # per scored session, zeros for no-trade
    trades: int
    winners: int
    selected_precision: float
    block_means: tuple[float, ...]


@torch.no_grad()
def serial_walk(
    model: CompactInteractionEntryPolicy,
    selector: FrozenSelector,
    score_world: World,
    *,
    law: PreflightLaw = LAW,
) -> WalkResult:
    """One-position serial account over the scored sessions.

    Occupancy consumes the trade's measured duration; at most
    `max_trades_per_session`; a session with no trade scores exactly $0, and
    those zeros stay in the primary estimand.
    """

    scores = model(score_world.batch).contract_logits
    masked = scores.masked_fill(~score_world.batch.entry_action_mask, -torch.inf)
    best, node = masked.max(dim=1)
    floor_logit = math.log(
        selector.probability_floor / (1.0 - selector.probability_floor)
    )
    fire = (best > selector.score_cutoff) & (best >= floor_logit)

    minutes = law.minutes_per_session
    sessions = score_world.sessions
    session_net = np.zeros(sessions)
    trades = 0
    winners = 0
    for s in range(sessions):
        occupied_until = -1.0
        taken = 0
        base = s * minutes
        for t in range(minutes):
            if taken >= law.max_trades_per_session:
                break
            if t <= occupied_until:
                continue
            row = base + t
            if not bool(fire[row]):
                continue
            k = int(node[row])
            net = float(score_world.net_usd[row, k])
            duration = float(score_world.duration_minutes[row, k])
            label = float(score_world.clean_label[row, k])
            session_net[s] += net
            occupied_until = t + duration
            taken += 1
            trades += 1
            winners += int(label > 0.5)

    per_block = sessions // law.score_blocks
    blocks = tuple(
        float(session_net[i * per_block : (i + 1) * per_block].mean())
        for i in range(law.score_blocks)
    )
    precision = winners / trades if trades else float("nan")
    return WalkResult(
        session_net_usd=session_net,
        trades=trades,
        winners=winners,
        selected_precision=precision,
        block_means=blocks,
    )


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    trades: int
    mean_session_net_usd: float
    corrected_lcb_usd: float
    positive_blocks: int
    selected_precision: float


def full_gate(
    walk: WalkResult, *, seed: int, law: PreflightLaw = LAW
) -> GateDecision:
    """The complete pass/fail decision the real experiment will apply."""

    mean_net = float(walk.session_net_usd.mean())
    rng = np.random.default_rng(seed)
    n = len(walk.session_net_usd)
    draws = rng.integers(0, n, size=(law.bootstrap_draws, n))
    means = walk.session_net_usd[draws].mean(axis=1)
    lcb = float(np.quantile(means, law.alpha / law.family_size))
    positive_blocks = sum(1 for b in walk.block_means if b > 0)

    if walk.trades < law.min_selected_trades:
        decision, reason = False, "insufficient_trades"
    elif mean_net <= 0:
        decision, reason = False, "mean_not_positive"
    elif lcb <= 0:
        decision, reason = False, "corrected_lcb_not_positive"
    elif positive_blocks < law.required_positive_blocks:
        decision, reason = False, "insufficient_positive_blocks"
    else:
        decision, reason = True, "passed"
    return GateDecision(
        passed=decision,
        reason=reason,
        trades=walk.trades,
        mean_session_net_usd=mean_net,
        corrected_lcb_usd=lcb,
        positive_blocks=positive_blocks,
        selected_precision=walk.selected_precision,
    )


@dataclass(frozen=True)
class TrialResult:
    seed: int
    effect_slope: float
    gate: GateDecision
    epochs_run: int
    stop_epoch_of_best: int
    score_cutoff: float


def run_trial(
    *,
    effect_slope: float,
    trial_seed: int,
    calibration: Calibration,
    law: PreflightLaw = LAW,
) -> TrialResult:
    """One seeded world pair, one production-law fit, one full-gate decision."""

    train_world = generate_world(
        sessions=law.train_sessions,
        effect_slope=effect_slope,
        seed=trial_seed,
        calibration=calibration,
        law=law,
    )
    score_world = generate_world(
        sessions=law.score_sessions,
        effect_slope=effect_slope,
        seed=stable_seed("score-world", trial_seed),
        calibration=calibration,
        law=law,
    )
    scaler = WorldScaler.fit(train_world)
    scaled_train = scaler.apply(train_world, clip=law.feature_clip)
    scaled_score = scaler.apply(score_world, clip=law.feature_clip)
    model, record = train_entry_model(scaled_train, seed=trial_seed, law=law)
    selector = FrozenSelector.freeze(model, scaled_train, law=law)
    walk = serial_walk(model, selector, scaled_score, law=law)
    gate = full_gate(walk, seed=stable_seed("gate-bootstrap", trial_seed), law=law)
    return TrialResult(
        seed=trial_seed,
        effect_slope=effect_slope,
        gate=gate,
        epochs_run=record.epochs_run,
        stop_epoch_of_best=record.stop_epoch_of_best,
        score_cutoff=selector.score_cutoff,
    )


def wilson_upper(successes: int, trials: int, z: float = 1.6449) -> float:
    if trials == 0:
        return 1.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4 * trials * trials))
    return (centre + margin) / denominator


def reference_weights(
    effect_slope: float, law: PreflightLaw = LAW
) -> CompactInteractionEntryPolicy:
    """Hand-built weights proving the planted signal is representable.

    For UNSCALED worlds: `direction(state) * side` carries `x * side`, which is
    exactly the planted probability ordering. If these weights achieve the
    operating-point precision, a failed trained fit indicts the training law,
    not the architecture.
    """

    model = CompactInteractionEntryPolicy().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        body_position = STATE_CANDLE_INDICES.index(
            CANDLE_FEATURES.index("body_points")
        )
        model.direction_state.weight[0, body_position] = 1.0
    return model
