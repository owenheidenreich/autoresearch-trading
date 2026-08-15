"""Target semantics for the unfitted compact shared lifecycle design.

These helpers are exercised only on synthetic known-answer paths in v5.  Real
held-position targets may be built only after the same-day CBBO backfill and a
fresh permitted fit declaration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from v5.research.causal_day_architectures import ArchitectureScores


TARGET_SCALE_USD = 1_000.0
SMOOTH_L1_BETA = 0.25


class LifecycleTargetError(RuntimeError):
    """A held-position path cannot support the frozen action-value law."""


@dataclass(frozen=True)
class ExitActionTargets:
    q_sell_usd: np.ndarray
    q_hold_usd: np.ndarray
    hold_available: np.ndarray


def build_exit_action_targets(sell_if_requested_usd: np.ndarray) -> ExitActionTargets:
    """Return SELL-now and HOLD-for-a-strictly-later-sale oracle values.

    ``sell_if_requested_usd[t]`` must already implement the simulator's
    first-later-bid or validated-settlement rule.  The last row is forced
    liquidation, so HOLD is unavailable there rather than assigned a fake
    finite value.
    """

    sell = np.asarray(sell_if_requested_usd, dtype=np.float64)
    if sell.ndim != 1 or len(sell) < 2 or not np.isfinite(sell).all():
        raise LifecycleTargetError("exit path must contain at least two finite sale values")
    hold = np.full(len(sell), np.nan, dtype=np.float64)
    suffix_best = float(sell[-1])
    for index in range(len(sell) - 2, -1, -1):
        hold[index] = suffix_best
        suffix_best = max(suffix_best, float(sell[index]))
    available = np.isfinite(hold)
    return ExitActionTargets(sell.copy(), hold, available)


def exit_action_value_loss(
    scores: ArchitectureScores,
    q_sell_usd: Tensor,
    q_hold_usd: Tensor,
    hold_available: Tensor,
) -> Tensor:
    """Equal-weight Smooth-L1 over valid HOLD and SELL values."""

    if scores.exit_logits.shape != (len(q_sell_usd), 2):
        raise LifecycleTargetError("exit prediction shape differs from target rows")
    if q_sell_usd.shape != q_hold_usd.shape or q_sell_usd.shape != hold_available.shape:
        raise LifecycleTargetError("SELL, HOLD and availability shapes differ")
    if hold_available.dtype is not torch.bool:
        raise LifecycleTargetError("HOLD availability must be boolean")
    if not torch.isfinite(q_sell_usd).all():
        raise LifecycleTargetError("SELL targets must be finite")
    if not torch.isfinite(q_hold_usd[hold_available]).all():
        raise LifecycleTargetError("available HOLD target is non-finite")
    if torch.isfinite(q_hold_usd[~hold_available]).any():
        raise LifecycleTargetError("forced-liquidation HOLD target must remain unavailable")

    sell_loss = F.smooth_l1_loss(
        scores.exit_logits[:, 1],
        q_sell_usd / TARGET_SCALE_USD,
        beta=SMOOTH_L1_BETA,
    )
    if not hold_available.any():
        return sell_loss
    hold_loss = F.smooth_l1_loss(
        scores.exit_logits[hold_available, 0],
        q_hold_usd[hold_available] / TARGET_SCALE_USD,
        beta=SMOOTH_L1_BETA,
    )
    return 0.5 * sell_loss + 0.5 * hold_loss
