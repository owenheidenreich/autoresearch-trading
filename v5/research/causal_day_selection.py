"""Causal selector law for the corrected Job-39 magnitude policy.

The V4 defect was not a failed economic policy: an absolute activation
threshold was placed at the upper support boundary of a clipped regression
target.  This module is the selector-specific pre-run gate.  It deliberately
lives beside, rather than inside, the owner-controlled policy-fit gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import minute_number
from v5.research.causal_day_magnitude import HORIZONS, TARGET_CLIP_POINTS


class SelectorSpecificationError(RuntimeError):
    """A selector cannot be evaluated without violating its frozen law."""


@dataclass(frozen=True)
class RankCutoff:
    fold: int
    training_first_session: str
    training_last_session: str
    training_sessions: int
    training_minutes: int
    target_signal_minutes_per_session: int
    target_rank: int
    cutoff_points: float
    training_max_points: float
    scores_at_or_above_cutoff: int


def assert_absolute_threshold_attainable(
    threshold_points: float,
    *,
    label_clip_ceiling_points: float = TARGET_CLIP_POINTS[1],
) -> None:
    """Refuse the V4 defect class before predictions or economics are read."""

    from v5.research.causal_day_policy_gate import (
        SelectorAttainability,
        SelectorUnattainable,
        assert_selector_attainable,
    )

    try:
        assert_selector_attainable(
            SelectorAttainability(
                selector_name="absolute_magnitude_threshold",
                rule_kind="absolute_threshold",
                target_clip_bounds=(TARGET_CLIP_POINTS[0], float(label_clip_ceiling_points)),
                absolute_threshold=float(threshold_points),
                proof_source="causal_day_magnitude.scale_targets clip contract",
            )
        )
    except SelectorUnattainable as error:
        raise SelectorSpecificationError(str(error)) from error


def minute_max_scores(
    predictions: pd.DataFrame,
    *,
    score_column: str,
) -> pd.DataFrame:
    required = {"session", "entry_minute", score_column}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise SelectorSpecificationError(f"rank calibration is missing columns: {missing}")
    values = predictions[["session", "entry_minute", score_column]].copy()
    values[score_column] = pd.to_numeric(values[score_column], errors="raise")
    if not np.isfinite(values[score_column].to_numpy(dtype=float)).all():
        raise SelectorSpecificationError("rank calibration scores must be finite")
    return (
        values.groupby(["session", "entry_minute"], sort=True, as_index=False)[score_column]
        .max()
        .sort_values(["session", "entry_minute"], kind="mergesort")
        .reset_index(drop=True)
    )


def calibrate_rank_cutoff(
    training_predictions: pd.DataFrame,
    *,
    fold: int,
    score_column: str,
    target_signal_minutes_per_session: int,
) -> RankCutoff:
    """Freeze an average top-N operating point from strictly prior sessions.

    Contract scores are first collapsed to the best eligible contract in each
    minute.  The cutoff is the k-th largest training-minute maximum where
    k = N times the number of training sessions.  No label, trade outcome or
    scored-session statistic is an input.
    """

    if fold not in range(1, 6):
        raise SelectorSpecificationError("fold must be one of the five frozen folds")
    if target_signal_minutes_per_session <= 0:
        raise SelectorSpecificationError("rank target must be positive")
    minutes = minute_max_scores(training_predictions, score_column=score_column)
    sessions = tuple(sorted(minutes["session"].astype(str).unique()))
    if not sessions:
        raise SelectorSpecificationError("rank calibration has no training sessions")
    target_rank = target_signal_minutes_per_session * len(sessions)
    if target_rank <= 1 or target_rank >= len(minutes):
        raise SelectorSpecificationError("rank target is not interior to the training population")
    ordered = np.sort(minutes[score_column].to_numpy(dtype=float))[::-1]
    cutoff = float(ordered[target_rank - 1])
    maximum = float(ordered[0])
    if not np.isfinite(cutoff) or cutoff >= maximum:
        raise SelectorSpecificationError(
            "rank cutoff must be finite and strictly below the observed training maximum"
        )
    return RankCutoff(
        fold=fold,
        training_first_session=sessions[0],
        training_last_session=sessions[-1],
        training_sessions=len(sessions),
        training_minutes=len(minutes),
        target_signal_minutes_per_session=target_signal_minutes_per_session,
        target_rank=target_rank,
        cutoff_points=cutoff,
        training_max_points=maximum,
        scores_at_or_above_cutoff=int(np.count_nonzero(ordered >= cutoff)),
    )


def assert_cutoff_precedes_score_sessions(
    cutoff: RankCutoff,
    score_sessions: list[str] | tuple[str, ...],
) -> None:
    scored = tuple(sorted(str(value) for value in score_sessions))
    if not scored or cutoff.training_last_session >= scored[0]:
        raise SelectorSpecificationError(
            "rank cutoff is not isolated to sessions strictly before its score block"
        )


def select_rank_clock_trades(
    scored_candidates: pd.DataFrame,
    *,
    horizon: int,
    fold_cutoffs: Mapping[int, float],
    trade_cap: int,
) -> pd.DataFrame:
    """Walk scored sessions causally with fold-frozen rank cutoffs.

    Although the table contains a complete score block, each decision uses
    only the current minute's contract surface, the prior occupancy state and
    a cutoff frozen from earlier sessions.  Later scores from the same day do
    not enter an earlier action.
    """

    if horizon not in HORIZONS:
        raise SelectorSpecificationError("undeclared horizon")
    if trade_cap not in (1, 2, 3):
        raise SelectorSpecificationError("undeclared trade cap")
    if set(fold_cutoffs) != set(range(1, 6)):
        raise SelectorSpecificationError("one frozen cutoff is required for each fold")
    if not all(np.isfinite(float(value)) for value in fold_cutoffs.values()):
        raise SelectorSpecificationError("fold cutoffs must be finite")
    score_column = f"predicted_depth_{horizon}m"
    required = {
        "session",
        "entry_minute",
        "contract_id",
        "fold",
        "spread_usd",
        "moneyness_itm_points",
        score_column,
        f"clock_exit_minute_{horizon}m",
        f"net_mid_{horizon}m_usd",
    }
    missing = sorted(required - set(scored_candidates.columns))
    if missing:
        raise SelectorSpecificationError(f"scored candidates missing columns: {missing}")

    rows: list[pd.Series] = []
    for _, session in scored_candidates.groupby("session", sort=True):
        folds = session["fold"].astype(int).unique()
        if len(folds) != 1:
            raise SelectorSpecificationError("one session cannot span score folds")
        fold = int(folds[0])
        if fold not in fold_cutoffs:
            raise SelectorSpecificationError(f"no cutoff for fold {fold}")
        cutoff = float(fold_cutoffs[fold])
        trades = 0
        free_after = -1
        walk = session.copy()
        walk["_minute_number"] = walk["entry_minute"].map(minute_number)
        for minute_value, block in walk.groupby("_minute_number", sort=True):
            if trades >= trade_cap or minute_value <= free_after:
                continue
            eligible = block[
                pd.to_numeric(block[score_column], errors="coerce").ge(cutoff)
            ]
            if eligible.empty:
                continue
            chosen = eligible.sort_values(
                [score_column, "spread_usd", "moneyness_itm_points", "contract_id"],
                ascending=[False, True, False, True],
                kind="mergesort",
            ).iloc[0].copy()
            exit_minute = str(chosen[f"clock_exit_minute_{horizon}m"])
            if not exit_minute or exit_minute == "nan":
                raise SelectorSpecificationError("selected clock trade has no terminal accounting")
            free_after = minute_number(exit_minute)
            chosen["selector"] = "causal_training_prefix_rank"
            chosen["rank_cutoff_points"] = cutoff
            chosen["horizon_minutes"] = horizon
            chosen["gross_mid_usd"] = float(chosen[f"net_mid_{horizon}m_usd"]) + 3.08
            chosen["net_mid_usd"] = float(chosen[f"net_mid_{horizon}m_usd"])
            if f"net_bid_{horizon}m_usd" in chosen.index:
                chosen["net_bid_usd"] = float(chosen[f"net_bid_{horizon}m_usd"])
            rows.append(chosen)
            trades += 1
    return pd.DataFrame(rows)
