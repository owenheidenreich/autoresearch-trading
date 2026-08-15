"""Bounded confidence calibration for the Protocol101 H2 policy-5 candidate."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from v4.model.protocol101_scoped_stage1_hgb import (
    CanonicalDecision,
    HGBUnitConfig,
    expected_calibration_error,
    selected_index_for_scores,
)


METHODS = ("training_tail_isotonic_v1", "training_tail_platt_v1")
INNER_SPLIT_COUNT = 3


@dataclass(frozen=True)
class CalibrationRow:
    session: str
    decision_time: str
    score: float
    outcome: float
    selected_contract_id: str


def calibration_rows(
    decisions: list[CanonicalDecision],
    scores_by_decision: list[np.ndarray],
    *,
    epsilon: float,
    config: HGBUnitConfig,
) -> list[CalibrationRow]:
    if len(decisions) != len(scores_by_decision):
        raise ValueError("decision/score length mismatch")
    rows: list[CalibrationRow] = []
    for decision, scores in zip(decisions, scores_by_decision):
        selected, top_score, _second, _margin, _confident = (
            selected_index_for_scores(
                decision,
                scores,
                epsilon=epsilon,
                k_slot=config.k_slot,
            )
        )
        rows.append(
            CalibrationRow(
                session=str(decision.session),
                decision_time=decision.decision_time.isoformat(),
                score=float(top_score),
                outcome=float(
                    float(decision.labels[selected]) - float(config.fee) > 0.0
                ),
                selected_contract_id=str(decision.contract_ids[selected]),
            )
        )
    return rows


def chronological_inner_splits(rows: list[CalibrationRow]) -> list[dict[str, Any]]:
    sessions = sorted({row.session for row in rows})
    if len(sessions) < 6:
        raise ValueError("fewer than six calibration sessions")
    block = max(1, len(sessions) // (INNER_SPLIT_COUNT + 1))
    first_validation = len(sessions) - INNER_SPLIT_COUNT * block
    splits: list[dict[str, Any]] = []
    for index in range(INNER_SPLIT_COUNT):
        start = first_validation + index * block
        end = len(sessions) if index == INNER_SPLIT_COUNT - 1 else start + block
        train_sessions = sessions[:start]
        validation_sessions = sessions[start:end]
        if not train_sessions or not validation_sessions:
            raise ValueError("invalid chronological calibration split")
        splits.append(
            {
                "inner_fold": index,
                "train_sessions": train_sessions,
                "validation_sessions": validation_sessions,
            }
        )
    return splits


def _arrays(rows: Iterable[CalibrationRow]) -> tuple[np.ndarray, np.ndarray]:
    materialized = list(rows)
    return (
        np.asarray([row.score for row in materialized], dtype=float),
        np.asarray([row.outcome for row in materialized], dtype=float),
    )


def fit_map(rows: list[CalibrationRow], *, method: str) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unsupported calibration method: {method}")
    x, y = _arrays(rows)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if not len(x):
        return {
            "method": "constant_no_calibration_rows",
            "requested_method": method,
            "sample_count": 0,
            "constant_probability": 0.0,
        }
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return {
            "method": "constant_training_tail_base_rate",
            "requested_method": method,
            "sample_count": int(len(x)),
            "constant_probability": float(np.mean(y)),
        }
    if method == "training_tail_isotonic_v1":
        model = IsotonicRegression(
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
        )
        model.fit(x, y)
        return {
            "method": method,
            "sample_count": int(len(x)),
            "x_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
        }
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=0,
    )
    model.fit(x.reshape(-1, 1), y.astype(int))
    return {
        "method": method,
        "sample_count": int(len(x)),
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
        "random_state": 0,
    }


def confidence_from_state(scores: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if "constant_probability" in state:
        return np.full(
            len(values),
            float(state["constant_probability"]),
            dtype=float,
        )
    method = state.get("method")
    if method == "training_tail_isotonic_v1":
        x = np.asarray(state.get("x_thresholds") or (), dtype=float)
        y = np.asarray(state.get("y_thresholds") or (), dtype=float)
        if not len(x) or len(x) != len(y):
            raise ValueError("invalid isotonic calibration state")
        return np.interp(values, x, y, left=y[0], right=y[-1])
    if method == "training_tail_platt_v1":
        logits = (
            float(state["coefficient"]) * values + float(state["intercept"])
        )
        logits = np.clip(logits, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-logits))
    raise ValueError(f"invalid calibration state: {method}")


def select_and_fit_map(rows: list[CalibrationRow]) -> dict[str, Any]:
    splits = chronological_inner_splits(rows)
    method_results: dict[str, Any] = {}
    for method in METHODS:
        fold_results: list[dict[str, Any]] = []
        weighted_numerator = 0.0
        observations = 0
        for split in splits:
            train_set = set(split["train_sessions"])
            validation_set = set(split["validation_sessions"])
            train_rows = [row for row in rows if row.session in train_set]
            validation_rows = [
                row for row in rows if row.session in validation_set
            ]
            state = fit_map(train_rows, method=method)
            scores, outcomes = _arrays(validation_rows)
            confidence = confidence_from_state(scores, state)
            ece = expected_calibration_error(confidence, outcomes, bins=10)
            weighted_numerator += ece * len(validation_rows)
            observations += len(validation_rows)
            fold_results.append(
                {
                    **split,
                    "train_observations": len(train_rows),
                    "validation_observations": len(validation_rows),
                    "fitted_state": state,
                    "ece": float(ece),
                }
            )
        method_results[method] = {
            "inner_folds": fold_results,
            "weighted_inner_ece": (
                weighted_numerator / observations if observations else 1.0
            ),
            "observations": observations,
        }
    selected = min(
        METHODS,
        key=lambda method: (
            float(method_results[method]["weighted_inner_ece"]),
            METHODS.index(method),
        ),
    )
    return {
        "selection_rule": (
            "minimum_observation_weighted_chronological_inner_ece_"
            "tie_to_isotonic"
        ),
        "method_results": method_results,
        "selected_method": selected,
        "final_state": fit_map(rows, method=selected),
    }


def apply_repair(
    *,
    calibration_decisions: list[CanonicalDecision],
    calibration_scores: list[np.ndarray],
    validation_decisions: list[CanonicalDecision],
    validation_scores: list[np.ndarray],
    epsilon: float,
    config: HGBUnitConfig,
) -> dict[str, Any]:
    calibration = calibration_rows(
        calibration_decisions,
        calibration_scores,
        epsilon=epsilon,
        config=config,
    )
    validation = calibration_rows(
        validation_decisions,
        validation_scores,
        epsilon=epsilon,
        config=config,
    )
    selection = select_and_fit_map(calibration)
    scores, outcomes = _arrays(validation)
    confidence = confidence_from_state(scores, selection["final_state"])
    ece = expected_calibration_error(confidence, outcomes, bins=10)
    validation_rows = [
        {
            "session": row.session,
            "decision_time": row.decision_time,
            "score": row.score,
            "outcome": row.outcome,
            "selected_contract_id": row.selected_contract_id,
            "calibrated_confidence": float(probability),
        }
        for row, probability in zip(validation, confidence)
    ]
    return {
        "method_selection": selection,
        "validation_ece": float(ece),
        "validation_observations": len(validation_rows),
        "validation_rows": validation_rows,
    }


def finite_probability(value: float) -> bool:
    return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
