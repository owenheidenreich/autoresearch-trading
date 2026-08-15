"""Aligned targets and loss for the compact WAIT-versus-ENTER policy.

All action values share one fixed dollar scale so argmax remains meaningful.
The shuffled null permutes the executable Q(enter) surface within a session and
then recomputes Q(wait), preserving the label law while removing association
with the observed market state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from v5.research.causal_day_action_advantage import _future_wait
from v5.research.causal_day_architectures import ArchitectureScores
from v5.research.causal_day_fit_cache import CachedSession


TARGET_SCALE_USD = 1_000.0
SMOOTH_L1_BETA = 0.25
WAIT_LOSS_WEIGHT = 0.5
ENTER_LOSS_WEIGHT = 0.5


class ActionValueTargetError(RuntimeError):
    """Target alignment or the frozen joint action loss is invalid."""


@dataclass(frozen=True)
class ActionValueTargetSession:
    session: str
    q_enter_bid_usd: np.ndarray
    q_wait_bid_usd: np.ndarray


def target_path(root: Path, session: str) -> Path:
    return root / "sessions" / f"{session}.npz"


def load_target_session(path: Path) -> ActionValueTargetSession:
    with np.load(path, allow_pickle=False) as value:
        return ActionValueTargetSession(
            session=str(value["session"].item()),
            q_enter_bid_usd=value["q_enter_bid_usd"].astype(np.float32),
            q_wait_bid_usd=value["q_wait_bid_usd"].astype(np.float32),
        )


def validate_alignment(cached: CachedSession, targets: ActionValueTargetSession) -> None:
    if cached.session != targets.session:
        raise ActionValueTargetError("feature and target sessions differ")
    if targets.q_enter_bid_usd.shape != (len(cached.ladder),):
        raise ActionValueTargetError("Q(enter) does not align to the whole ladder")
    if targets.q_wait_bid_usd.shape != (cached.minute_count,):
        raise ActionValueTargetError("Q(wait) does not align to every decision minute")
    if not np.isfinite(targets.q_enter_bid_usd[cached.action_mask]).all():
        raise ActionValueTargetError("eligible action lacks a finite Q(enter)")
    if not np.isnan(targets.q_enter_bid_usd[~cached.action_mask]).all():
        raise ActionValueTargetError("ineligible ladder node carries a Q(enter)")
    if not np.isfinite(targets.q_wait_bid_usd).all():
        raise ActionValueTargetError("decision minute lacks a finite Q(wait)")


def collate_action_values(
    cached: CachedSession,
    targets: ActionValueTargetSession,
    indices: np.ndarray,
) -> tuple[Tensor, Tensor]:
    validate_alignment(cached, targets)
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        raise ActionValueTargetError("cannot collate an empty minute batch")
    lengths = np.diff(cached.ladder_offsets)[indices]
    max_ladder = int(lengths.max())
    q_enter = np.full((len(indices), max_ladder), np.nan, dtype=np.float32)
    for row, minute_index in enumerate(indices):
        start = int(cached.ladder_offsets[minute_index])
        stop = int(cached.ladder_offsets[minute_index + 1])
        q_enter[row, : stop - start] = targets.q_enter_bid_usd[start:stop]
    q_wait = targets.q_wait_bid_usd[indices]
    return torch.from_numpy(q_enter), torch.from_numpy(q_wait.copy())


def action_value_loss(
    scores: ArchitectureScores,
    q_enter_bid_usd: Tensor,
    q_wait_bid_usd: Tensor,
    action_mask: Tensor,
) -> Tensor:
    """Equal-weight WAIT and per-minute ENTER Smooth-L1 losses."""

    if scores.contract_logits.shape != q_enter_bid_usd.shape:
        raise ActionValueTargetError("contract prediction and Q(enter) shapes differ")
    if scores.abstain_logits.shape != q_wait_bid_usd.shape:
        raise ActionValueTargetError("WAIT prediction and Q(wait) shapes differ")
    if action_mask.shape != q_enter_bid_usd.shape:
        raise ActionValueTargetError("action mask and Q(enter) shapes differ")
    if not torch.isfinite(q_wait_bid_usd).all():
        raise ActionValueTargetError("Q(wait) contains a non-finite value")
    usable = action_mask & torch.isfinite(q_enter_bid_usd)
    wait_raw = F.smooth_l1_loss(
        scores.abstain_logits,
        q_wait_bid_usd / TARGET_SCALE_USD,
        beta=SMOOTH_L1_BETA,
        reduction="none",
    )
    counts = usable.sum(dim=1)
    if not usable.any():
        return WAIT_LOSS_WEIGHT * wait_raw.mean()
    enter_raw = F.smooth_l1_loss(
        scores.contract_logits[usable],
        q_enter_bid_usd[usable] / TARGET_SCALE_USD,
        beta=SMOOTH_L1_BETA,
        reduction="none",
    )
    row_ids = torch.arange(len(usable), device=usable.device).unsqueeze(1).expand_as(usable)[usable]
    enter_sum = torch.zeros(len(usable), dtype=enter_raw.dtype, device=enter_raw.device)
    enter_sum.scatter_add_(0, row_ids, enter_raw)
    has_enter = counts > 0
    enter_mean = enter_sum[has_enter] / counts[has_enter].to(enter_raw.dtype)
    return ENTER_LOSS_WEIGHT * enter_mean.mean() + WAIT_LOSS_WEIGHT * wait_raw.mean()


def shuffled_session_surface(
    cached: CachedSession,
    targets: ActionValueTargetSession,
    *,
    seed: int,
) -> ActionValueTargetSession:
    """Permutation null preserving the session payoff distribution and action counts."""

    validate_alignment(cached, targets)
    rng = np.random.default_rng(seed)
    locations = np.flatnonzero(cached.action_mask)
    shuffled_enter = targets.q_enter_bid_usd.copy()
    shuffled_enter[locations] = targets.q_enter_bid_usd[rng.permutation(locations)]
    minute_best = np.full(cached.minute_count, np.nan, dtype=float)
    for minute_index in range(cached.minute_count):
        start = int(cached.ladder_offsets[minute_index])
        stop = int(cached.ladder_offsets[minute_index + 1])
        values = shuffled_enter[start:stop]
        finite = values[np.isfinite(values)]
        if len(finite):
            minute_best[minute_index] = float(np.max(finite))
    shuffled_wait = _future_wait(minute_best).astype(np.float32)
    return ActionValueTargetSession(cached.session, shuffled_enter, shuffled_wait)
