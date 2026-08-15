"""Canonical bounded-HGB unit for scoped Protocol101 Stage-1 research.

The unit is deliberately model- and fold-local. It consumes only features
returned by ``protocol101_canonical_stage1_contract``, selects its action
threshold and score-noise epsilon on training-tail calibration sessions, and
evaluates later sessions through serial simulator v4.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from v4.model.protocol101_canonical_stage1_contract import (
    CONTRACT_ID,
    FEATURE_NAMES,
    HYPOTHESES,
    boundary_stable_mask,
    hypothesis_matrix,
)
from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_regimen_repair import (
    TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    assert_alpha_feature_names,
    assert_decision_row_identities,
    assert_processed_row_identities,
)
from v4.model.protocol101_serial_simulator import (
    PROTOCOL101_SERIAL_SIMULATOR_VERSION,
    SerialCandidate,
    SerialSimulatorConfig,
    simulate_serial_candidates,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)


ROUND_TRIP_FEE = 3.0
DAILY_LOSS_FRACTION = 0.05
K_ACTION = 2.0
K_SLOT = 2.0
TARGET_CLIP_LOW = -2.0
TARGET_CLIP_HIGH = 5.0
THRESHOLD_QUANTILES = (
    0.00,
    0.25,
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
)
POLICY_HOLD_MINUTES = {
    0: 10.0,
    1: 25.0,
    2: 45.0,
    3: 90.0,
    4: 120.0,
    5: 384.0,
    6: 384.0,
}


@dataclass(frozen=True)
class CanonicalDecision:
    """Candidate rows and labels for one causal decision minute."""

    session: str
    decision_time: pd.Timestamp
    features: np.ndarray
    labels: np.ndarray
    mid_labels: np.ndarray
    entry_asks: np.ndarray
    offsets: np.ndarray
    rights: np.ndarray
    contract_ids: np.ndarray
    strike_indices: np.ndarray
    right_indices: np.ndarray


@dataclass(frozen=True)
class RepairedCanonicalDecision:
    """Canonical decision with additive two-clock label metadata."""

    base: CanonicalDecision
    realized_exit_time_ns: np.ndarray
    source_exit_quote_time_ns: np.ndarray
    exit_quote_age_ms: np.ndarray
    exit_reason_codes: np.ndarray
    executable_exit_bids: np.ndarray
    policy_deadline_ns: np.ndarray
    invalid_reason_codes: np.ndarray
    canonical_strike_slots: np.ndarray
    source_quote_time_ns: np.ndarray
    source_context_time_ns: np.ndarray


@dataclass(frozen=True)
class HGBUnitConfig:
    """Frozen bounded model and selection settings for one unit."""

    hypothesis: str
    policy_index: int
    seed: int
    max_iter: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    max_train_examples: int = 350_000
    training_noise_scale: float = 1.0
    fee: float = ROUND_TRIP_FEE
    daily_loss_fraction: float = DAILY_LOSS_FRACTION
    k_action: float = K_ACTION
    k_slot: float = K_SLOT
    target_clip_low: float = TARGET_CLIP_LOW
    target_clip_high: float = TARGET_CLIP_HIGH

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _entry_ask(
    row: dict[str, Any],
    *,
    strike_idx: int,
    right_idx: int,
    contract_id: str,
) -> float | None:
    item = (row.get("contract_quote_metadata") or {}).get(contract_id) or {}
    ask = _finite_float(item.get("ask"))
    if ask is not None and ask > 0.0:
        return ask
    option_names = tuple(row.get("feature_names") or ())
    if "ask" not in option_names:
        return None
    ladder = np.asarray(row.get("option_ladder"), dtype=float)
    ask = _finite_float(ladder[strike_idx, right_idx, option_names.index("ask")])
    return ask if ask is not None and ask > 0.0 else None


def decision_from_row(
    row: dict[str, Any],
    *,
    session: str,
    hypothesis: str,
    policy_index: int,
    guard_margins: dict[str, float],
) -> CanonicalDecision | None:
    """Build one exact-contract decision without exposing quarantined alpha."""
    assert_decision_row_identities(
        row,
        split="canonical",
        session=session,
        boundary="canonical decision construction before target construction",
    )
    if hypothesis not in HYPOTHESES:
        raise ValueError(f"unknown hypothesis: {hypothesis}")
    labels_all = np.asarray(row.get("labels_net_pnl"), dtype=float)
    mid_labels_all = np.asarray(row.get("labels_mid_pnl"), dtype=float)
    if labels_all.ndim != 3 or not (0 <= int(policy_index) < labels_all.shape[2]):
        raise ValueError(f"policy {policy_index} absent from labels {labels_all.shape}")
    if mid_labels_all.shape != labels_all.shape:
        raise ValueError(
            f"mid-label shape {mid_labels_all.shape} != net-label shape {labels_all.shape}"
        )
    matrix = hypothesis_matrix(row, hypothesis)
    guard = boundary_stable_mask(row, guard_margins)
    ids = np.asarray(row.get("contract_ids"), dtype=object)
    offsets = np.asarray(row.get("strike_offsets"), dtype=float)
    rights = np.asarray(row.get("rights") or ("C", "P"), dtype=object)
    labels = labels_all[:, :, int(policy_index)]
    mid_labels = mid_labels_all[:, :, int(policy_index)]

    features: list[np.ndarray] = []
    selected_labels: list[float] = []
    selected_mid_labels: list[float] = []
    asks: list[float] = []
    selected_offsets: list[float] = []
    selected_rights: list[str] = []
    selected_ids: list[str] = []
    strike_indices: list[int] = []
    right_indices: list[int] = []
    for strike_idx, right_idx in np.argwhere(guard):
        label = _finite_float(labels[strike_idx, right_idx])
        mid_label = _finite_float(mid_labels[strike_idx, right_idx])
        contract_id = str(ids[strike_idx, right_idx])
        ask = _entry_ask(
            row,
            strike_idx=int(strike_idx),
            right_idx=int(right_idx),
            contract_id=contract_id,
        )
        if label is None or mid_label is None or ask is None:
            continue
        candidate_features = np.asarray(matrix[strike_idx, right_idx], dtype=float)
        if not np.isfinite(candidate_features[: len(HYPOTHESES["H0"])]).all():
            continue
        features.append(candidate_features)
        selected_labels.append(label)
        selected_mid_labels.append(mid_label)
        asks.append(ask)
        selected_offsets.append(float(offsets[strike_idx]))
        selected_rights.append(str(rights[right_idx]))
        selected_ids.append(contract_id)
        strike_indices.append(int(strike_idx))
        right_indices.append(int(right_idx))
    if not features:
        return None
    decision_time = pd.Timestamp(row.get("decision_time"))
    if decision_time.tzinfo is None:
        decision_time = decision_time.tz_localize("UTC")
    return CanonicalDecision(
        session=str(session),
        decision_time=decision_time.tz_convert("UTC"),
        features=np.vstack(features).astype(np.float64),
        labels=np.asarray(selected_labels, dtype=np.float64),
        mid_labels=np.asarray(selected_mid_labels, dtype=np.float64),
        entry_asks=np.asarray(asks, dtype=np.float64),
        offsets=np.asarray(selected_offsets, dtype=np.float64),
        rights=np.asarray(selected_rights, dtype=object),
        contract_ids=np.asarray(selected_ids, dtype=object),
        strike_indices=np.asarray(strike_indices, dtype=np.int64),
        right_indices=np.asarray(right_indices, dtype=np.int64),
    )


def load_decisions(
    session_paths: Iterable[tuple[str, Path]],
    *,
    hypothesis: str,
    policy_index: int,
    guard_margins: dict[str, float],
    max_rows_per_session: int | None = None,
) -> list[CanonicalDecision]:
    """Load governed processed rows into exact-contract candidate decisions."""
    decisions: list[CanonicalDecision] = []
    for session, path in session_paths:
        with path.open("rb") as handle:
            rows = pickle.load(handle)
        if not isinstance(rows, list):
            raise TypeError(f"{path} expected list rows")
        assert_processed_row_identities(
            rows,
            split="governed",
            session=session,
            boundary="processed-row load before target construction",
        )
        selected_rows = rows
        if max_rows_per_session is not None:
            selected_rows = rows[: int(max_rows_per_session)]
        for row in selected_rows:
            if not isinstance(row, dict):
                continue
            decision = decision_from_row(
                row,
                session=session,
                hypothesis=hypothesis,
                policy_index=policy_index,
                guard_margins=guard_margins,
            )
            if decision is not None:
                decisions.append(decision)
    return sorted(decisions, key=lambda item: (item.session, item.decision_time))


def _metadata_ns(value: Any) -> int:
    if value in (None, ""):
        return 0
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("repaired source timestamp must be timezone-aware")
    return int(timestamp.tz_convert("UTC").value)


def repaired_decision_from_row(
    row: dict[str, Any],
    *,
    session: str,
    hypothesis: str,
    policy_index: int,
    guard_margins: dict[str, float],
) -> RepairedCanonicalDecision | None:
    """Load one v2 row while keeping alpha and exit metadata separated."""

    if row.get("processed_row_schema_version") != TWO_CLOCK_PROCESSED_ROW_SCHEMA:
        raise ValueError("repaired decision requires the signed two-clock row schema")
    base = decision_from_row(
        row,
        session=session,
        hypothesis=hypothesis,
        policy_index=policy_index,
        guard_margins=guard_margins,
    )
    if base is None:
        return None
    indexes = (base.strike_indices, base.right_indices, int(policy_index))

    def selected(name: str, dtype: Any) -> np.ndarray:
        return np.asarray(row[name])[indexes].astype(dtype, copy=False)

    metadata = row.get("contract_quote_metadata") or {}
    source_quote_times: list[int] = []
    source_context_times: list[int] = []
    for contract_id in base.contract_ids:
        item = metadata.get(str(contract_id)) or {}
        source_quote_times.append(
            _metadata_ns(item.get("source_quote_time") or row.get("source_quote_time"))
        )
        source_context_times.append(
            _metadata_ns(
                item.get("source_context_time") or row.get("source_context_time")
            )
        )
    return RepairedCanonicalDecision(
        base=base,
        realized_exit_time_ns=selected(
            "label_realized_exit_time_ns", np.int64
        ),
        source_exit_quote_time_ns=selected(
            "label_source_exit_quote_time_ns", np.int64
        ),
        exit_quote_age_ms=selected("label_exit_quote_age_ms", np.float64),
        exit_reason_codes=selected("label_exit_reason_code", np.uint8),
        executable_exit_bids=selected(
            "label_executable_exit_bid", np.float64
        ),
        policy_deadline_ns=selected("label_policy_deadline_ns", np.int64),
        invalid_reason_codes=selected(
            "label_invalid_reason_code", np.uint8
        ),
        canonical_strike_slots=base.strike_indices.astype(np.int64),
        source_quote_time_ns=np.asarray(source_quote_times, dtype=np.int64),
        source_context_time_ns=np.asarray(
            source_context_times, dtype=np.int64
        ),
    )


def load_repaired_decisions(
    session_paths: Iterable[tuple[str, Path]],
    *,
    hypothesis: str,
    policy_index: int,
    guard_margins: dict[str, float],
    split: str,
    max_rows_per_session: int | None = None,
) -> list[RepairedCanonicalDecision]:
    """Load v2 decisions after a full-session identity preflight."""

    decisions: list[RepairedCanonicalDecision] = []
    for session, path in session_paths:
        with path.open("rb") as handle:
            rows = pickle.load(handle)
        if not isinstance(rows, list):
            raise TypeError(f"{path} expected list rows")
        assert_processed_row_identities(
            rows,
            split=split,
            session=session,
            boundary="repaired processed-row load before target construction",
        )
        selected_rows = (
            rows
            if max_rows_per_session is None
            else rows[: int(max_rows_per_session)]
        )
        for row in selected_rows:
            if not isinstance(row, dict):
                raise TypeError(f"{path} contains a non-mapping processed row")
            decision = repaired_decision_from_row(
                row,
                session=session,
                hypothesis=hypothesis,
                policy_index=policy_index,
                guard_margins=guard_margins,
            )
            if decision is not None:
                decisions.append(decision)
    return sorted(
        decisions,
        key=lambda item: (item.base.session, item.base.decision_time),
    )


def split_fit_calibration_sessions(
    sessions: list[str],
    *,
    calibration_fraction: float = 0.20,
) -> tuple[list[str], list[str]]:
    """Chronologically reserve the training tail for threshold calibration."""
    ordered = sorted(dict.fromkeys(str(item) for item in sessions))
    if len(ordered) < 2:
        raise ValueError("at least two training sessions are required")
    count = max(1, int(math.ceil(len(ordered) * float(calibration_fraction))))
    count = min(count, len(ordered) - 1)
    return ordered[:-count], ordered[-count:]


def _stack_training_rows(
    decisions: list[CanonicalDecision],
    *,
    feature_names: tuple[str, ...],
    noise_model: DivergenceNoiseModel,
    config: HGBUnitConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not decisions:
        raise ValueError("no fit decisions")
    features = np.vstack([item.features for item in decisions]).astype(np.float64)
    labels = np.concatenate([item.labels for item in decisions]).astype(np.float64)
    asks = np.concatenate([item.entry_asks for item in decisions]).astype(np.float64)
    offsets = np.concatenate([item.offsets for item in decisions]).astype(np.float64)
    frame = pd.DataFrame(features, columns=feature_names)
    frame["abs_offset"] = np.abs(offsets)
    noisy = noise_model.inject_dataframe(
        frame,
        feature_columns=feature_names,
        seed=int(config.seed),
        scale=float(config.training_noise_scale),
    )
    x = noisy.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    premium = asks * 100.0
    target = np.divide(
        labels - float(config.fee),
        premium,
        out=np.full_like(labels, np.nan),
        where=premium > 0.0,
    )
    target = np.clip(
        target,
        float(config.target_clip_low),
        float(config.target_clip_high),
    )
    finite_target = np.isfinite(target)
    x = x[finite_target]
    target = target[finite_target]
    if len(target) > int(config.max_train_examples):
        rng = np.random.default_rng(int(config.seed))
        indexes = np.sort(
            rng.choice(
                len(target),
                size=int(config.max_train_examples),
                replace=False,
            )
        )
        x = x[indexes]
        target = target[indexes]
    return x, target, {
        "fit_candidates_before_cap": int(finite_target.sum()),
        "fit_candidates_after_cap": int(len(target)),
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "target_mean": float(np.mean(target)),
    }


def fit_model(
    decisions: list[CanonicalDecision],
    *,
    noise_model: DivergenceNoiseModel,
    config: HGBUnitConfig,
) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    """Fit one bounded payoff regressor using training sessions only."""
    feature_names = tuple(HYPOTHESES[config.hypothesis])
    x, target, summary = _stack_training_rows(
        decisions,
        feature_names=feature_names,
        noise_model=noise_model,
        config=config,
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(config.learning_rate),
        max_iter=int(config.max_iter),
        max_depth=int(config.max_depth),
        min_samples_leaf=int(config.min_samples_leaf),
        l2_regularization=float(config.l2_regularization),
        early_stopping=False,
        random_state=int(config.seed),
    )
    model.fit(x, target)
    return model, {
        **summary,
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "contract_id": CONTRACT_ID,
        "model_family": "sklearn_hist_gradient_boosting_regressor",
        "model_iterations": int(getattr(model, "n_iter_", config.max_iter)),
    }


def score_decisions(
    model: HistGradientBoostingRegressor,
    decisions: list[CanonicalDecision],
    *,
    feature_names: tuple[str, ...],
    noise_model: DivergenceNoiseModel | None = None,
    noise_scale: float = 0.0,
    noise_seed: int = 0,
) -> list[np.ndarray]:
    """Score decisions, optionally under measured feature-noise replicas."""
    if not decisions:
        return []
    counts = [len(item.labels) for item in decisions]
    features = np.vstack([item.features for item in decisions]).astype(np.float64)
    if noise_model is not None and float(noise_scale) != 0.0:
        offsets = np.concatenate([item.offsets for item in decisions]).astype(float)
        frame = pd.DataFrame(features, columns=feature_names)
        frame["abs_offset"] = np.abs(offsets)
        frame = noise_model.inject_dataframe(
            frame,
            feature_columns=feature_names,
            seed=int(noise_seed),
            scale=float(noise_scale),
        )
        features = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    flat = np.asarray(model.predict(features), dtype=np.float64)
    out: list[np.ndarray] = []
    cursor = 0
    for count in counts:
        out.append(flat[cursor : cursor + count])
        cursor += count
    return out


def score_noise_epsilon(
    model: HistGradientBoostingRegressor,
    calibration: list[CanonicalDecision],
    *,
    feature_names: tuple[str, ...],
    noise_model: DivergenceNoiseModel,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    """Calibrate score epsilon from 0x/1x training-tail score replicas."""
    baseline = score_decisions(
        model,
        calibration,
        feature_names=feature_names,
    )
    noisy = score_decisions(
        model,
        calibration,
        feature_names=feature_names,
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=int(seed) + 100_000,
    )
    drift = np.concatenate(
        [np.abs(left - right) for left, right in zip(baseline, noisy)]
    )
    finite = drift[np.isfinite(drift)]
    epsilon = float(np.percentile(finite, 95)) if len(finite) else 0.0
    return epsilon, {
        "method": "training_tail_0x_vs_1x_measured_noise_p95",
        "sample_count": int(len(finite)),
        "p50_abs_score_drift": (
            float(np.percentile(finite, 50)) if len(finite) else 0.0
        ),
        "p95_abs_score_drift": epsilon,
        "p99_abs_score_drift": (
            float(np.percentile(finite, 99)) if len(finite) else 0.0
        ),
    }


def _ranked_indices(decision: CanonicalDecision, scores: np.ndarray) -> list[int]:
    return sorted(
        range(len(scores)),
        key=lambda index: (
            -float(scores[index]) if math.isfinite(float(scores[index])) else math.inf,
            int(decision.strike_indices[index]),
            int(decision.right_indices[index]),
        ),
    )


def _fallback_index(decision: CanonicalDecision) -> int:
    return min(
        range(len(decision.labels)),
        key=lambda index: (
            abs(float(decision.offsets[index])),
            int(decision.strike_indices[index]),
            int(decision.right_indices[index]),
            str(decision.contract_ids[index]),
        ),
    )


def selected_index_for_scores(
    decision: CanonicalDecision,
    scores: np.ndarray,
    *,
    epsilon: float,
    k_slot: float,
) -> tuple[int, float, float, float, bool]:
    ranked = _ranked_indices(decision, scores)
    if not ranked:
        raise ValueError("decision has no ranked candidates")
    top = ranked[0]
    top_score = float(scores[top])
    second_score = float(scores[ranked[1]]) if len(ranked) > 1 else -math.inf
    margin = top_score - second_score
    confident_slot = bool(margin > float(k_slot) * float(epsilon))
    selected = top if confident_slot else _fallback_index(decision)
    return selected, top_score, second_score, margin, confident_slot


def fit_confidence_map(
    decisions: list[CanonicalDecision],
    scores_by_decision: list[np.ndarray],
    *,
    epsilon: float,
    config: HGBUnitConfig,
) -> dict[str, Any]:
    """Fit a training-tail-only isotonic payoff-score confidence readout."""
    scores: list[float] = []
    wins: list[float] = []
    for decision, candidate_scores in zip(decisions, scores_by_decision):
        (
            selected,
            top_score,
            _second_score,
            _margin,
            _confident,
        ) = selected_index_for_scores(
            decision,
            candidate_scores,
            epsilon=epsilon,
            k_slot=config.k_slot,
        )
        scores.append(top_score)
        wins.append(
            float(
                float(decision.labels[selected]) - float(config.fee) > 0.0
            )
        )
    x = np.asarray(scores, dtype=float)
    y = np.asarray(wins, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if not len(x):
        return {
            "method": "constant_no_calibration_rows",
            "sample_count": 0,
            "constant_probability": 0.0,
        }
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return {
            "method": "constant_training_tail_base_rate",
            "sample_count": int(len(x)),
            "constant_probability": float(np.mean(y)),
        }
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(x, y)
    return {
        "method": "training_tail_isotonic_payoff_score_to_realized_win",
        "sample_count": int(len(x)),
        "x_thresholds": [float(value) for value in model.X_thresholds_],
        "y_thresholds": [float(value) for value in model.y_thresholds_],
    }


def confidence_from_state(scores: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    if "constant_probability" in state:
        return np.full(
            len(scores),
            float(state["constant_probability"]),
            dtype=float,
        )
    x = np.asarray(state.get("x_thresholds") or (), dtype=float)
    y = np.asarray(state.get("y_thresholds") or (), dtype=float)
    if not len(x) or len(x) != len(y):
        raise ValueError("invalid isotonic confidence state")
    return np.interp(np.asarray(scores, dtype=float), x, y, left=y[0], right=y[-1])


def expected_calibration_error(
    confidence: np.ndarray,
    outcomes: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    confidence = np.asarray(confidence, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    finite = np.isfinite(confidence) & np.isfinite(outcomes)
    confidence = np.clip(confidence[finite], 0.0, 1.0)
    outcomes = outcomes[finite]
    if not len(confidence):
        return 1.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = float(len(confidence))
    ece = 0.0
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        if index == int(bins) - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        ece += (
            float(mask.sum())
            / total
            * abs(float(np.mean(confidence[mask])) - float(np.mean(outcomes[mask])))
        )
    return float(ece)


def selection_rows(
    decisions: list[CanonicalDecision],
    scores_by_decision: list[np.ndarray],
    *,
    threshold: float,
    epsilon: float,
    config: HGBUnitConfig,
    split: str,
) -> tuple[list[SerialCandidate], list[dict[str, Any]]]:
    """Apply frozen action/slot semantics and emit serial entry intents."""
    candidates: list[SerialCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    for decision, scores in zip(decisions, scores_by_decision):
        (
            selected,
            top_score,
            second_score,
            margin,
            confident_slot,
        ) = selected_index_for_scores(
            decision,
            scores,
            epsilon=epsilon,
            k_slot=config.k_slot,
        )
        action_enter = bool(
            math.isfinite(top_score)
            and top_score > float(threshold) + float(config.k_action) * float(epsilon)
        )
        diagnostics.append(
            {
                "session": decision.session,
                "decision_time": decision.decision_time.isoformat(),
                "top_score": top_score,
                "second_score": second_score,
                "top2_margin": margin,
                "epsilon": float(epsilon),
                "threshold": float(threshold),
                "slot_confident": confident_slot,
                "fallback_used": not confident_slot,
                "action_enter": action_enter,
                "selected_contract_id": str(decision.contract_ids[selected]),
                "selected_offset": float(decision.offsets[selected]),
                "selected_right": str(decision.rights[selected]),
                "selected_label_before_fee": float(decision.labels[selected]),
                "selected_mid_label_before_fee": float(
                    decision.mid_labels[selected]
                ),
            }
        )
        if not action_enter:
            continue
        hold = POLICY_HOLD_MINUTES[int(config.policy_index)]
        candidates.append(
            SerialCandidate(
                split=str(split),
                session=decision.session,
                decision_time=decision.decision_time.to_pydatetime(),
                contract_id=str(decision.contract_ids[selected]),
                right=str(decision.rights[selected]),
                offset=float(decision.offsets[selected]),
                entry_ask=float(decision.entry_asks[selected]),
                score=top_score,
                raw_label_pnl=float(decision.labels[selected]) - float(config.fee),
                cooldown_minutes=hold,
                max_hold_minutes=hold,
                feature_hash=CONTRACT_ID,
                strategy=f"{config.hypothesis}_bounded_hgb",
                metadata={
                    "policy_index": int(config.policy_index),
                    "slot_confident": confident_slot,
                    "selected_candidate_score": float(scores[selected]),
                    "pessimistic_label_before_fee": float(
                        decision.labels[selected]
                    ),
                    "mid_label_before_fee": float(decision.mid_labels[selected]),
                },
            )
        )
    return candidates, diagnostics


def selection_rows_v5(
    decisions: list[RepairedCanonicalDecision],
    scores_by_decision: list[np.ndarray],
    *,
    threshold: float,
    epsilon: float,
    config: HGBUnitConfig,
    split: str,
    fold: str = "",
) -> tuple[list[SerialCandidateV5], list[dict[str, Any]]]:
    """Apply the frozen selector and carry both exit clocks into v5."""

    if len(decisions) != len(scores_by_decision):
        raise ValueError("repaired decision and score counts differ")
    candidates: list[SerialCandidateV5] = []
    diagnostics: list[dict[str, Any]] = []
    for repaired, scores in zip(decisions, scores_by_decision):
        decision = repaired.base
        (
            selected,
            top_score,
            second_score,
            margin,
            confident_slot,
        ) = selected_index_for_scores(
            decision,
            scores,
            epsilon=epsilon,
            k_slot=config.k_slot,
        )
        action_enter = bool(
            math.isfinite(top_score)
            and top_score
            > float(threshold) + float(config.k_action) * float(epsilon)
        )
        diagnostics.append(
            {
                "session": decision.session,
                "decision_time": decision.decision_time.isoformat(),
                "top_score": top_score,
                "second_score": second_score,
                "top2_margin": margin,
                "epsilon": float(epsilon),
                "threshold": float(threshold),
                "slot_confident": confident_slot,
                "fallback_used": not confident_slot,
                "action_enter": action_enter,
                "selected_contract_id": str(decision.contract_ids[selected]),
                "selected_offset": float(decision.offsets[selected]),
                "selected_right": str(decision.rights[selected]),
                "selected_label_before_fee": float(
                    decision.labels[selected]
                ),
                "selected_mid_label_before_fee": float(
                    decision.mid_labels[selected]
                ),
                "label_realized_exit_time_ns": int(
                    repaired.realized_exit_time_ns[selected]
                ),
                "label_source_exit_quote_time_ns": int(
                    repaired.source_exit_quote_time_ns[selected]
                ),
            }
        )
        if not action_enter:
            continue
        candidates.append(
            SerialCandidateV5(
                split=str(split),
                session=decision.session,
                decision_time_ns=int(decision.decision_time.value),
                contract_id=str(decision.contract_ids[selected]),
                right=str(decision.rights[selected]),
                canonical_strike_slot=int(
                    repaired.canonical_strike_slots[selected]
                ),
                policy_index=int(config.policy_index),
                entry_ask=float(decision.entry_asks[selected]),
                score=float(top_score),
                raw_label_pnl_after_campaign_fee=(
                    float(decision.labels[selected]) - float(config.fee)
                ),
                label_mid_pnl_before_campaign_fee=float(
                    decision.mid_labels[selected]
                ),
                label_realized_exit_time_ns=int(
                    repaired.realized_exit_time_ns[selected]
                ),
                label_source_exit_quote_time_ns=int(
                    repaired.source_exit_quote_time_ns[selected]
                ),
                label_exit_quote_age_ms=float(
                    repaired.exit_quote_age_ms[selected]
                ),
                label_exit_reason_code=int(
                    repaired.exit_reason_codes[selected]
                ),
                label_executable_exit_bid=float(
                    repaired.executable_exit_bids[selected]
                ),
                label_policy_deadline_ns=int(
                    repaired.policy_deadline_ns[selected]
                ),
                label_invalid_reason_code=int(
                    repaired.invalid_reason_codes[selected]
                ),
                feature_hash=CONTRACT_ID,
                source_quote_time_ns=int(
                    repaired.source_quote_time_ns[selected]
                ),
                source_context_time_ns=int(
                    repaired.source_context_time_ns[selected]
                ),
                strategy=f"{config.hypothesis}_bounded_hgb",
                source_simulator_version=(
                    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
                ),
                fold=str(fold),
                metadata={
                    "slot_confident": bool(confident_slot),
                    "selected_candidate_score": float(scores[selected]),
                    "pessimistic_label_before_fee": float(
                        decision.labels[selected]
                    ),
                    "mid_label_before_fee": float(
                        decision.mid_labels[selected]
                    ),
                },
            )
        )
    return candidates, diagnostics


def replay_candidates_v5(
    candidates: list[SerialCandidateV5],
    *,
    config: HGBUnitConfig,
) -> tuple[list[Any], Any, dict[str, Any]]:
    """Run repaired intents with no synthetic hold-time reconstruction."""

    trades, state = simulate_serial_candidates_v5(
        candidates,
        config=SerialSimulatorV5Config(
            starting_cash=10_000.0,
            max_daily_loss_fraction_of_session_start_equity=float(
                config.daily_loss_fraction
            ),
            affordability_reserve_per_trade=float(config.fee),
            campaign_round_trip_fee_dollars=float(config.fee),
        ),
    )
    pnl = np.asarray(
        [item.raw_label_pnl_after_campaign_fee for item in trades],
        dtype=float,
    )
    equity = np.asarray(
        [
            float(event["equity"])
            for events in state.equity_events_by_account.values()
            for event in events
        ]
        or [10_000.0],
        dtype=float,
    )
    peak = np.maximum.accumulate(equity)
    metrics = {
        "entry_intents": int(len(candidates)),
        "trades": int(len(trades)),
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "ending_equity": float(
            next(iter(state.cash_by_account.values()), 10_000.0)
        ),
        "minimum_equity": float(np.min(equity)),
        "max_drawdown": float(np.max(peak - equity)),
        "profitable_trades": int((pnl > 0.0).sum()),
        "skipped": dict(state.skipped),
        "simulator_version": state.semantics["simulator_version"],
        "affordability_reserve_per_trade": state.semantics[
            "affordability_reserve_per_trade"
        ],
        "campaign_round_trip_fee_dollars": state.semantics[
            "campaign_round_trip_fee_dollars"
        ],
        "candidate_stream_hash": state.candidate_stream_hash,
        "candidate_payload_hash": state.candidate_payload_hash,
        "trade_identity_hash": state.trade_identity_hash,
    }
    return trades, state, metrics


def replay_candidates_v5_at_fee(
    candidates: list[SerialCandidateV5],
    *,
    config: HGBUnitConfig,
    fee: float,
) -> tuple[list[Any], Any, dict[str, Any]]:
    """Replay identical v5 intents with one alternate campaign fee."""

    adjusted = [
        replace(
            item,
            raw_label_pnl_after_campaign_fee=(
                float(item.raw_label_pnl_after_campaign_fee)
                + float(config.fee)
                - float(fee)
            ),
        )
        for item in candidates
    ]
    return replay_candidates_v5(
        adjusted,
        config=replace(config, fee=float(fee)),
    )


def replay_candidates(
    candidates: list[SerialCandidate],
    *,
    config: HGBUnitConfig,
) -> tuple[list[Any], Any, dict[str, Any]]:
    trades, state = simulate_serial_candidates(
        candidates,
        config=SerialSimulatorConfig(
            starting_cash=10_000.0,
            max_daily_loss_fraction_of_session_start_equity=float(
                config.daily_loss_fraction
            ),
            affordability_reserve_per_trade=float(config.fee),
        ),
    )
    pnl = np.asarray([item.raw_label_pnl for item in trades], dtype=float)
    equity = np.asarray(
        state.equity_by_account.get("validation", [10_000.0]),
        dtype=float,
    )
    if "validation" not in state.equity_by_account and state.equity_by_account:
        equity = np.asarray(next(iter(state.equity_by_account.values())), dtype=float)
    peak = np.maximum.accumulate(equity)
    metrics = {
        "entry_intents": int(len(candidates)),
        "trades": int(len(trades)),
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "ending_equity": float(equity[-1]),
        "minimum_equity": float(np.min(equity)),
        "max_drawdown": float(np.max(peak - equity)),
        "profitable_trades": int((pnl > 0.0).sum()),
        "skipped": dict(state.skipped),
        "simulator_version": state.semantics["simulator_version"],
        "affordability_reserve_per_trade": state.semantics[
            "affordability_reserve_per_trade"
        ],
    }
    return trades, state, metrics


def replay_candidates_at_fee(
    candidates: list[SerialCandidate],
    *,
    config: HGBUnitConfig,
    fee: float,
) -> tuple[list[Any], Any, dict[str, Any]]:
    """Replay frozen entry intents under an alternate round-trip fee."""
    adjusted = [
        replace(
            item,
            raw_label_pnl=(
                float(item.raw_label_pnl) + float(config.fee) - float(fee)
            ),
        )
        for item in candidates
    ]
    return replay_candidates(adjusted, config=replace(config, fee=float(fee)))


def _selection_agreement(
    primary: list[dict[str, Any]],
    other: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(primary) != len(other):
        raise ValueError("selection diagnostic row count mismatch")
    actions = [
        bool(left["action_enter"]) == bool(right["action_enter"])
        for left, right in zip(primary, other)
    ]
    entered_pairs = [
        str(left["selected_contract_id"]) == str(right["selected_contract_id"])
        for left, right in zip(primary, other)
        if bool(left["action_enter"]) and bool(right["action_enter"])
    ]
    return {
        "decision_count": len(primary),
        "action_agreement": float(np.mean(actions)) if actions else 1.0,
        "mutual_enter_count": len(entered_pairs),
        "selected_contract_agreement_on_mutual_enters": (
            float(np.mean(entered_pairs)) if entered_pairs else 1.0
        ),
    }


def _fill_edge_band(
    trades: list[Any],
    intents: list[SerialCandidate],
    *,
    fee: float,
) -> dict[str, Any]:
    by_key = {
        (item.session, item.decision_time.isoformat(), item.contract_id): item
        for item in intents
    }
    mid_pnl = 0.0
    missing = 0
    for trade in trades:
        intent = by_key.get(
            (trade.session, trade.decision_time, trade.contract_id)
        )
        value = (
            None
            if intent is None
            else intent.metadata.get("mid_label_before_fee")
        )
        try:
            mid_pnl += float(value) - float(fee)
        except (TypeError, ValueError):
            missing += 1
    return {
        "pessimistic_ask_in_bid_out_net_pnl": float(
            sum(float(item.raw_label_pnl) for item in trades)
        ),
        "mid_path_net_pnl_same_executed_trades": float(mid_pnl),
        "executed_trades": len(trades),
        "missing_mid_path_labels": int(missing),
        "gating_rung": "pessimistic_ask_in_bid_out",
    }


def _fill_edge_band_v5(
    trades: list[Any],
    intents: list[SerialCandidateV5],
    *,
    fee: float,
) -> dict[str, Any]:
    by_key = {
        (item.session, int(item.decision_time_ns), item.contract_id): item
        for item in intents
    }
    mid_pnl = 0.0
    missing = 0
    for trade in trades:
        intent = by_key.get(
            (
                trade.session,
                int(trade.decision_time_ns),
                trade.contract_id,
            )
        )
        value = (
            None
            if intent is None
            else intent.label_mid_pnl_before_campaign_fee
        )
        try:
            mid_pnl += float(value) - float(fee)
        except (TypeError, ValueError):
            missing += 1
    return {
        "pessimistic_ask_in_bid_out_net_pnl": float(
            sum(
                float(item.raw_label_pnl_after_campaign_fee)
                for item in trades
            )
        ),
        "mid_path_net_pnl_same_executed_trades": float(mid_pnl),
        "executed_trades": len(trades),
        "missing_mid_path_labels": int(missing),
        "gating_rung": "pessimistic_ask_in_bid_out",
    }


def threshold_candidates(scores_by_decision: list[np.ndarray]) -> list[float]:
    top_scores = np.asarray(
        [
            float(np.nanmax(scores))
            for scores in scores_by_decision
            if len(scores) and np.isfinite(scores).any()
        ],
        dtype=float,
    )
    if not len(top_scores):
        return [math.inf]
    values = {
        float(np.quantile(top_scores, quantile))
        for quantile in THRESHOLD_QUANTILES
    }
    values.add(float(np.max(top_scores)) + 1e-9)
    return sorted(values)


def choose_threshold(
    decisions: list[CanonicalDecision],
    scores_by_decision: list[np.ndarray],
    *,
    epsilon: float,
    config: HGBUnitConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose abstention threshold on training-tail calibration only."""
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(scores_by_decision):
        candidates, _diagnostics = selection_rows(
            decisions,
            scores_by_decision,
            threshold=threshold,
            epsilon=epsilon,
            config=config,
            split="calibration",
        )
        trades, _state, metrics = replay_candidates(candidates, config=config)
        rows.append(
            {
                "threshold": float(threshold),
                "net_pnl": float(metrics["net_pnl"]),
                "trades": int(metrics["trades"]),
                "minimum_equity": float(metrics["minimum_equity"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "trade_pnls": [float(item.raw_label_pnl) for item in trades],
            }
        )
    best = max(
        rows,
        key=lambda row: (
            float(row["net_pnl"]),
            -int(row["trades"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"]), rows


def choose_threshold_v5(
    decisions: list[RepairedCanonicalDecision],
    scores_by_decision: list[np.ndarray],
    *,
    epsilon: float,
    config: HGBUnitConfig,
    fold: str = "",
) -> tuple[float, list[dict[str, Any]]]:
    """Choose the training-tail threshold exclusively through v5 replay."""

    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(scores_by_decision):
        candidates, _diagnostics = selection_rows_v5(
            decisions,
            scores_by_decision,
            threshold=threshold,
            epsilon=epsilon,
            config=config,
            split="calibration",
            fold=fold,
        )
        trades, _state, metrics = replay_candidates_v5(
            candidates,
            config=config,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "net_pnl": float(metrics["net_pnl"]),
                "trades": int(metrics["trades"]),
                "minimum_equity": float(metrics["minimum_equity"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "trade_pnls": [
                    float(item.raw_label_pnl_after_campaign_fee)
                    for item in trades
                ],
                "simulator_version": metrics["simulator_version"],
            }
        )
    best = max(
        rows,
        key=lambda row: (
            float(row["net_pnl"]),
            -int(row["trades"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"]), rows


def run_hgb_unit_v5(
    *,
    fit_decisions: list[RepairedCanonicalDecision],
    calibration_decisions: list[RepairedCanonicalDecision],
    validation_decisions: list[RepairedCanonicalDecision],
    noise_model: DivergenceNoiseModel,
    config: HGBUnitConfig,
    fold: str = "",
) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    """Fit and evaluate one fresh-campaign unit exclusively through v5."""

    feature_names = assert_alpha_feature_names(
        HYPOTHESES[config.hypothesis],
        allowed_feature_sets=HYPOTHESES.values(),
        boundary="fresh HGB unit before fit",
    )
    fit_base = [item.base for item in fit_decisions]
    calibration_base = [item.base for item in calibration_decisions]
    validation_base = [item.base for item in validation_decisions]
    model, fit_summary = fit_model(
        fit_base,
        noise_model=noise_model,
        config=config,
    )
    epsilon, epsilon_summary = score_noise_epsilon(
        model,
        calibration_base,
        feature_names=feature_names,
        noise_model=noise_model,
        seed=int(config.seed),
    )
    calibration_scores = score_decisions(
        model,
        calibration_base,
        feature_names=feature_names,
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=int(config.seed) + 100_000,
    )
    threshold, threshold_sweep = choose_threshold_v5(
        calibration_decisions,
        calibration_scores,
        epsilon=epsilon,
        config=config,
        fold=fold,
    )
    confidence_state = fit_confidence_map(
        calibration_base,
        calibration_scores,
        epsilon=epsilon,
        config=config,
    )
    validation_scores = score_decisions(
        model,
        validation_base,
        feature_names=feature_names,
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=int(config.seed) + 200_000,
    )
    intents, diagnostics = selection_rows_v5(
        validation_decisions,
        validation_scores,
        threshold=threshold,
        epsilon=epsilon,
        config=config,
        split="validation",
        fold=fold,
    )
    diagnostic_top_scores = np.asarray(
        [float(row["top_score"]) for row in diagnostics],
        dtype=float,
    )
    diagnostic_outcomes = np.asarray(
        [
            float(
                float(row["selected_label_before_fee"])
                - float(config.fee)
                > 0.0
            )
            for row in diagnostics
        ],
        dtype=float,
    )
    diagnostic_confidence = confidence_from_state(
        diagnostic_top_scores,
        confidence_state,
    )
    for row, confidence in zip(diagnostics, diagnostic_confidence):
        row["calibrated_confidence"] = float(confidence)
    validation_ece = expected_calibration_error(
        diagnostic_confidence,
        diagnostic_outcomes,
    )
    trades, state, metrics = replay_candidates_v5(
        intents,
        config=config,
    )
    fee_sensitivity: dict[str, Any] = {}
    for fee in (2.60, 3.00, 4.00):
        fee_trades, _fee_state, fee_metrics = replay_candidates_v5_at_fee(
            intents,
            config=config,
            fee=fee,
        )
        fee_sensitivity[f"{fee:.2f}"] = {
            "fee": fee,
            "trades": int(fee_metrics["trades"]),
            "net_pnl": float(fee_metrics["net_pnl"]),
            "minimum_equity": float(fee_metrics["minimum_equity"]),
            "max_drawdown": float(fee_metrics["max_drawdown"]),
            "realized_trade_count": len(fee_trades),
            "simulator_version": fee_metrics["simulator_version"],
        }
    noise_diagnostics: dict[str, Any] = {}
    for scale in (0.0, 0.5, 2.0):
        alternate_scores = score_decisions(
            model,
            validation_base,
            feature_names=feature_names,
            noise_model=noise_model,
            noise_scale=scale,
            noise_seed=int(config.seed) + 200_000,
        )
        alternate_intents, alternate_rows = selection_rows_v5(
            validation_decisions,
            alternate_scores,
            threshold=threshold,
            epsilon=epsilon,
            config=config,
            split="validation",
            fold=fold,
        )
        _alternate_trades, _alternate_state, alternate_metrics = (
            replay_candidates_v5(alternate_intents, config=config)
        )
        noise_diagnostics[f"{scale:.1f}x"] = {
            "noise_scale": scale,
            "metrics": alternate_metrics,
            "agreement_vs_primary_1x": _selection_agreement(
                diagnostics,
                alternate_rows,
            ),
        }
    return model, {
        "contract_id": CONTRACT_ID,
        "hypothesis": config.hypothesis,
        "feature_names": list(feature_names),
        "unexpected_model_features": sorted(
            set(feature_names) - set(FEATURE_NAMES)
        ),
        "authorized_features_not_used": sorted(
            set(FEATURE_NAMES) - set(feature_names)
        ),
        "policy_index": int(config.policy_index),
        "seed": int(config.seed),
        "fold": str(fold),
        "config": config.to_dict(),
        "fit": fit_summary,
        "calibration": {
            "decision_count": len(calibration_decisions),
            "epsilon": float(epsilon),
            "epsilon_summary": epsilon_summary,
            "threshold": float(threshold),
            "threshold_sweep": threshold_sweep,
            "confidence_map": confidence_state,
        },
        "validation": {
            "primary_noise_scale": 1.0,
            "primary_noise_seed": int(config.seed) + 200_000,
            "decision_count": len(validation_decisions),
            "candidate_count": int(
                sum(len(item.base.labels) for item in validation_decisions)
            ),
            "metrics": metrics,
            "expected_calibration_error": validation_ece,
            "calibration_observations": int(len(diagnostic_confidence)),
            "diagnostics": diagnostics,
            "entry_intents": [asdict(item) for item in intents],
            "trades": [asdict(item) for item in trades],
            "skipped_events": [
                asdict(item) for item in state.skipped_events
            ],
            "fee_sensitivity": fee_sensitivity,
            "fill_edge_band": _fill_edge_band_v5(
                trades,
                intents,
                fee=float(config.fee),
            ),
            "noise_diagnostics": noise_diagnostics,
            "simulator_semantics": state.semantics,
        },
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "two_clock_exit_semantics": state.semantics[
            "exit_time_semantics"
        ],
    }


def run_hgb_unit(
    *,
    fit_decisions: list[CanonicalDecision],
    calibration_decisions: list[CanonicalDecision],
    validation_decisions: list[CanonicalDecision],
    noise_model: DivergenceNoiseModel,
    config: HGBUnitConfig,
) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    """Fit, calibrate, score, select, and replay one governed model unit."""
    feature_names = tuple(HYPOTHESES[config.hypothesis])
    model, fit_summary = fit_model(
        fit_decisions,
        noise_model=noise_model,
        config=config,
    )
    epsilon, epsilon_summary = score_noise_epsilon(
        model,
        calibration_decisions,
        feature_names=feature_names,
        noise_model=noise_model,
        seed=int(config.seed),
    )
    calibration_scores = score_decisions(
        model,
        calibration_decisions,
        feature_names=feature_names,
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=int(config.seed) + 100_000,
    )
    threshold, threshold_sweep = choose_threshold(
        calibration_decisions,
        calibration_scores,
        epsilon=epsilon,
        config=config,
    )
    confidence_state = fit_confidence_map(
        calibration_decisions,
        calibration_scores,
        epsilon=epsilon,
        config=config,
    )
    validation_scores = score_decisions(
        model,
        validation_decisions,
        feature_names=feature_names,
        noise_model=noise_model,
        noise_scale=1.0,
        noise_seed=int(config.seed) + 200_000,
    )
    intents, diagnostics = selection_rows(
        validation_decisions,
        validation_scores,
        threshold=threshold,
        epsilon=epsilon,
        config=config,
        split="validation",
    )
    diagnostic_top_scores = np.asarray(
        [float(row["top_score"]) for row in diagnostics],
        dtype=float,
    )
    diagnostic_outcomes = np.asarray(
        [float(float(row["selected_label_before_fee"]) - config.fee > 0.0) for row in diagnostics],
        dtype=float,
    )
    diagnostic_confidence = confidence_from_state(
        diagnostic_top_scores,
        confidence_state,
    )
    for row, confidence in zip(diagnostics, diagnostic_confidence):
        row["calibrated_confidence"] = float(confidence)
    validation_ece = expected_calibration_error(
        diagnostic_confidence,
        diagnostic_outcomes,
    )
    trades, state, metrics = replay_candidates(intents, config=config)
    fee_sensitivity: dict[str, Any] = {}
    for fee in (2.60, 3.00, 4.00):
        fee_trades, _fee_state, fee_metrics = replay_candidates_at_fee(
            intents,
            config=config,
            fee=fee,
        )
        fee_sensitivity[f"{fee:.2f}"] = {
            "fee": fee,
            "trades": int(fee_metrics["trades"]),
            "net_pnl": float(fee_metrics["net_pnl"]),
            "minimum_equity": float(fee_metrics["minimum_equity"]),
            "max_drawdown": float(fee_metrics["max_drawdown"]),
            "realized_trade_count": len(fee_trades),
        }
    noise_diagnostics: dict[str, Any] = {}
    for scale in (0.0, 0.5, 2.0):
        alternate_scores = score_decisions(
            model,
            validation_decisions,
            feature_names=feature_names,
            noise_model=noise_model,
            noise_scale=scale,
            noise_seed=int(config.seed) + 200_000,
        )
        alternate_intents, alternate_rows = selection_rows(
            validation_decisions,
            alternate_scores,
            threshold=threshold,
            epsilon=epsilon,
            config=config,
            split="validation",
        )
        _alternate_trades, _alternate_state, alternate_metrics = (
            replay_candidates(alternate_intents, config=config)
        )
        noise_diagnostics[f"{scale:.1f}x"] = {
            "noise_scale": scale,
            "metrics": alternate_metrics,
            "agreement_vs_primary_1x": _selection_agreement(
                diagnostics,
                alternate_rows,
            ),
        }
    return model, {
        "contract_id": CONTRACT_ID,
        "hypothesis": config.hypothesis,
        "feature_names": list(feature_names),
        "unexpected_model_features": sorted(set(feature_names) - set(FEATURE_NAMES)),
        "authorized_features_not_used": sorted(
            set(FEATURE_NAMES) - set(feature_names)
        ),
        "policy_index": int(config.policy_index),
        "seed": int(config.seed),
        "config": config.to_dict(),
        "fit": fit_summary,
        "calibration": {
            "decision_count": len(calibration_decisions),
            "epsilon": float(epsilon),
            "epsilon_summary": epsilon_summary,
            "threshold": float(threshold),
            "threshold_sweep": threshold_sweep,
            "confidence_map": confidence_state,
        },
        "validation": {
            "primary_noise_scale": 1.0,
            "primary_noise_seed": int(config.seed) + 200_000,
            "decision_count": len(validation_decisions),
            "candidate_count": int(sum(len(item.labels) for item in validation_decisions)),
            "metrics": metrics,
            "expected_calibration_error": validation_ece,
            "calibration_observations": int(len(diagnostic_confidence)),
            "diagnostics": diagnostics,
            "entry_intents": [asdict(item) for item in intents],
            "trades": [asdict(item) for item in trades],
            "fee_sensitivity": fee_sensitivity,
            "fill_edge_band": _fill_edge_band(
                trades,
                intents,
                fee=float(config.fee),
            ),
            "noise_diagnostics": noise_diagnostics,
            "simulator_semantics": state.semantics,
        },
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_VERSION,
    }
