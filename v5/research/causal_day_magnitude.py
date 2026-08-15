"""Frozen magnitude objective and action law for the reopened causal-day fit.

The signed gate verifies parameter counts against the canonical architecture
builders.  Each declared horizon therefore receives its own canonical scalar
contract head; adding a three-output head would create parameters outside the
count the gate verifies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from v5.ops.build_causal_day_dataset import minute_number
from v5.research.causal_day_architectures import (
    ArchitectureDimensions,
    build_architecture,
    trainable_parameter_count,
)


HORIZONS = (60, 90, 120)
DEPTH_THRESHOLDS = (10, 20, 30)
PERMITTED_ARCHITECTURES = (
    "shallow_joint",
    "shallow_four_head",
    "neural_joint",
    "neural_four_head",
)
TARGET_CLIP_POINTS = (-25.0, 30.0)
TARGET_SCALE_POINTS = 30.0
TARGET_STRATA_EDGES = (0.0, 10.0, 20.0, 30.0)
MAX_STRATUM_WEIGHT = 20.0


class MagnitudeFitError(RuntimeError):
    """The frozen magnitude experiment contract was violated."""


@dataclass(frozen=True)
class ChronologicalFold:
    number: int
    train_sessions: tuple[str, ...]
    score_sessions: tuple[str, ...]


def chronological_folds(
    sessions: list[str] | tuple[str, ...],
    *,
    initial_train_sessions: int = 93,
    score_block_sessions: int = 30,
) -> tuple[ChronologicalFold, ...]:
    ordered = tuple(sorted(str(value) for value in sessions))
    if len(ordered) != initial_train_sessions + 5 * score_block_sessions:
        raise MagnitudeFitError(
            "the frozen five-fold design requires exactly "
            f"{initial_train_sessions + 5 * score_block_sessions} sessions"
        )
    folds = []
    for index in range(5):
        start = initial_train_sessions + index * score_block_sessions
        stop = start + score_block_sessions
        train = ordered[:start]
        score = ordered[start:stop]
        if not train or not score or max(train) >= min(score):
            raise MagnitudeFitError("chronological fold firewall failed")
        folds.append(ChronologicalFold(index + 1, train, score))
    return tuple(folds)


def build_magnitude_policy(
    name: str, dimensions: ArchitectureDimensions
) -> nn.Module:
    """Build the exact canonical model whose count the fit gate verifies."""

    if name not in PERMITTED_ARCHITECTURES:
        raise MagnitudeFitError(f"architecture {name!r} is outside the reopened family")
    return build_architecture(name, dimensions)


def parameter_count(module: nn.Module) -> int:
    return trainable_parameter_count(module)


def scale_targets(values: Tensor) -> Tensor:
    return values.clamp(*TARGET_CLIP_POINTS) / TARGET_SCALE_POINTS


def target_strata(values: Tensor) -> Tensor:
    edges = values.new_tensor(TARGET_STRATA_EDGES)
    return torch.bucketize(values.contiguous(), edges, right=False)


def stratum_weights(targets: np.ndarray) -> np.ndarray:
    """Fold-only inverse-frequency weights, capped before any fit."""

    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(HORIZONS):
        raise MagnitudeFitError("targets must have one column per declared horizon")
    weights = np.ones((len(HORIZONS), len(TARGET_STRATA_EDGES) + 1), dtype=np.float32)
    for column in range(len(HORIZONS)):
        strata = np.digitize(values[:, column], TARGET_STRATA_EDGES, right=False)
        counts = np.bincount(strata, minlength=weights.shape[1]).astype(float)
        present = counts > 0
        inverse = np.zeros_like(counts)
        inverse[present] = counts.sum() / (present.sum() * counts[present])
        inverse[present] = np.minimum(inverse[present], MAX_STRATUM_WEIGHT)
        normalizer = np.average(inverse[strata])
        weights[column] = inverse / normalizer
    return weights


def magnitude_loss(
    predictions: Tensor,
    targets_points: Tensor,
    action_mask: Tensor,
    weights: Tensor,
) -> Tensor:
    if predictions.shape != targets_points.shape:
        raise MagnitudeFitError("prediction and target shapes differ")
    if action_mask.shape != predictions.shape:
        raise MagnitudeFitError("action mask shape differs from contract surface")
    if weights.ndim != 1 or len(weights) != len(TARGET_STRATA_EDGES) + 1:
        raise MagnitudeFitError("one horizon requires one complete stratum-weight vector")
    usable = action_mask & torch.isfinite(targets_points)
    if not usable.any():
        raise MagnitudeFitError("magnitude batch contains no eligible actions")
    target = scale_targets(targets_points)
    raw = F.smooth_l1_loss(predictions[usable], target[usable], beta=0.25, reduction="none")
    strata = target_strata(targets_points)
    selected_weight = weights[strata[usable]]
    return (raw * selected_weight).sum() / selected_weight.sum()


def select_clock_trades(
    scored_candidates: pd.DataFrame,
    *,
    horizon: int,
    depth_threshold: int,
    trade_cap: int,
) -> pd.DataFrame:
    """Apply the frozen score/depth rule with clock occupancy.

    This table-mode walk is the broad-family evaluator.  It uses the same
    first-later-bid/settlement clock outcomes already produced by the causal
    simulator's dataset builder.  Final replay examples are still run through
    the event-driven simulator.
    """

    if horizon not in HORIZONS or depth_threshold not in DEPTH_THRESHOLDS:
        raise MagnitudeFitError("undeclared horizon or depth threshold")
    if trade_cap not in (1, 2, 3):
        raise MagnitudeFitError("undeclared trade cap")
    score_column = f"predicted_depth_{horizon}m"
    required = {
        "session",
        "entry_minute",
        "contract_id",
        "spread_usd",
        "moneyness_itm_points",
        score_column,
        f"clock_exit_minute_{horizon}m",
        f"net_bid_{horizon}m_usd",
        f"net_mid_{horizon}m_usd",
    }
    missing = sorted(required - set(scored_candidates.columns))
    if missing:
        raise MagnitudeFitError(f"scored candidates missing columns: {missing}")
    rows = []
    for _, session in scored_candidates.groupby("session", sort=True):
        trades = 0
        free_after = -1
        session = session.copy()
        session["_minute_number"] = session["entry_minute"].map(minute_number)
        for minute_number_value, block in session.groupby("_minute_number", sort=True):
            if trades >= trade_cap or minute_number_value <= free_after:
                continue
            eligible = block[pd.to_numeric(block[score_column], errors="coerce").ge(depth_threshold)]
            if eligible.empty:
                continue
            chosen = eligible.sort_values(
                [score_column, "spread_usd", "moneyness_itm_points", "contract_id"],
                ascending=[False, True, False, True],
                kind="mergesort",
            ).iloc[0].copy()
            exit_minute = str(chosen[f"clock_exit_minute_{horizon}m"])
            if not exit_minute or exit_minute == "nan":
                raise MagnitudeFitError("selected clock trade has no terminal accounting")
            free_after = minute_number(exit_minute)
            chosen["depth_threshold_points"] = depth_threshold
            chosen["horizon_minutes"] = horizon
            chosen["gross_mid_usd"] = float(chosen[f"net_mid_{horizon}m_usd"]) + 3.08
            chosen["net_mid_usd"] = float(chosen[f"net_mid_{horizon}m_usd"])
            chosen["net_bid_usd"] = float(chosen[f"net_bid_{horizon}m_usd"])
            rows.append(chosen)
            trades += 1
    return pd.DataFrame(rows)
