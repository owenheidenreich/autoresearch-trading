"""Causal estimator primitives for Protocol101 slot opportunity cost."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from v4.model.unified_conservative_policy import validate_no_future_feature_columns


ROLE_LABEL = "FOUNDATION_UNIFIED_SLOT_OPPORTUNITY_COST_ESTIMATOR_V1"
TRAINING_SPEC_LABEL = "CAUSAL_SLOT_OPPORTUNITY_COST_ESTIMATOR_TRAINING_V1"

SLOT_COST_LABEL_COLUMNS = (
    "blocked_protocol101_entries",
    "blocked_protocol101_pnl_0_00",
    "blocked_protocol101_pnl_0_10",
    "blocked_protocol101_pnl_0_25",
    "has_blocked_protocol101_entry",
)
SLOT_COST_FORBIDDEN_FEATURE_COLUMNS = (
    *SLOT_COST_LABEL_COLUMNS,
    "candidate_exit_time",
    "candidate_exit_dt",
    "candidate_pnl",
    "candidate_exit_reason",
    "q_wait",
    "session_oracle_value",
    "best_enter_value_at_decision",
    "oracle_action_uid",
    "q_enter",
    "a_enter",
    "next_flat_decision_index",
    "oracle_action",
    "q_exit_now_entry",
    "q_hold_entry",
    "a_hold_entry",
    "a_switch_entry",
)


@dataclass(frozen=True)
class SlotOpportunityCostEstimatorConfig:
    target_column: str = "blocked_protocol101_pnl_0_00"
    count_column: str = "blocked_protocol101_entries"
    positive_cost_threshold: float = 0.0
    target_clip: float = 10_000.0
    max_train_rows: int = 350_000
    max_eval_rows_per_split: int = 120_000
    train_positive_fraction: float = 0.40
    positive_sample_weight: float = 3.0
    random_seed: int = 1
    train_splits: tuple[str, ...] = ("q3_2025", "q4_2025")
    validation_split: str = "q1_2026"
    diagnostic_split: str = "recent_2026"
    uncertainty_quantile: float = 0.90

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_slot_cost_feature_columns(feature_columns: list[str] | tuple[str, ...]) -> None:
    """Reject realized labels, future exits, and oracle fields as estimator inputs."""

    validate_no_future_feature_columns(feature_columns)
    forbidden = sorted(set(str(column) for column in feature_columns) & set(SLOT_COST_FORBIDDEN_FEATURE_COLUMNS))
    if forbidden:
        raise ValueError(f"slot opportunity estimator feature leak columns: {forbidden}")


def build_slot_cost_targets(
    frame: pd.DataFrame,
    config: SlotOpportunityCostEstimatorConfig = SlotOpportunityCostEstimatorConfig(),
) -> pd.DataFrame:
    """Create nonnegative cost, blocked-entry count, and positive-cost targets."""

    target = pd.to_numeric(frame[config.target_column], errors="coerce").fillna(0.0)
    count = pd.to_numeric(frame[config.count_column], errors="coerce").fillna(0.0)
    out = pd.DataFrame(index=frame.index)
    out["target_cost"] = np.maximum(0.0, target.to_numpy(dtype=float))
    out["target_cost"] = np.minimum(float(config.target_clip), out["target_cost"])
    out["target_log_cost"] = np.log1p(out["target_cost"])
    out["target_blocked_entries"] = np.maximum(0.0, count.to_numpy(dtype=float))
    out["target_positive_cost"] = out["target_cost"].gt(float(config.positive_cost_threshold))
    return out


def sample_training_rows(
    frame: pd.DataFrame,
    *,
    positive_column: str,
    limit: int,
    positive_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Bound training size while preserving rare positive slot-cost rows."""

    if limit <= 0 or len(frame) <= limit:
        return frame.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    positive = frame[frame[positive_column].astype(bool)]
    negative = frame[~frame[positive_column].astype(bool)]
    positive_n = min(len(positive), int(round(limit * max(0.0, min(1.0, positive_fraction)))))
    negative_n = min(len(negative), int(limit) - positive_n)
    if positive_n + negative_n < int(limit):
        extra_positive = min(len(positive) - positive_n, int(limit) - positive_n - negative_n)
        positive_n += max(0, extra_positive)
    parts = []
    if positive_n > 0:
        parts.append(positive.sample(n=positive_n, random_state=int(seed), replace=False))
    if negative_n > 0:
        parts.append(negative.sample(n=negative_n, random_state=int(seed) + 1, replace=False))
    if not parts:
        return frame.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=int(seed) + 2).reset_index(drop=True)


def estimator_predictions(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Predict nonnegative slot cost, blocked-entry risk, and uncertainty."""

    feature_columns = list(bundle["feature_columns"])
    validate_slot_cost_feature_columns(feature_columns)
    features = coerce_feature_matrix(frame, feature_columns)
    cost_log = np.asarray(bundle["cost_model"].predict(features), dtype=float)
    predicted_cost = np.maximum(0.0, np.expm1(cost_log))
    predicted_cost = np.minimum(float(bundle["config"].get("target_clip", 10_000.0)), predicted_cost)
    entry_count = np.maximum(0.0, np.asarray(bundle["count_model"].predict(features), dtype=float))
    if hasattr(bundle["positive_model"], "predict_proba"):
        positive_probability = np.asarray(bundle["positive_model"].predict_proba(features)[:, 1], dtype=float)
    else:
        positive_probability = predicted_cost > 0.0
    uncertainty = np.full(len(frame), float(bundle.get("global_p90_abs_error", 0.0)), dtype=float)
    return pd.DataFrame(
        {
            "estimated_blocked_protocol101_cost": predicted_cost,
            "estimated_blocked_entries": entry_count,
            "blocked_cost_positive_probability": np.clip(positive_probability, 0.0, 1.0),
            "blocked_cost_uncertainty": uncertainty,
        },
        index=frame.index,
    )


def coerce_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    working = frame.reindex(columns=feature_columns).copy()
    for column in feature_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    return working.to_numpy(dtype=np.float32)
