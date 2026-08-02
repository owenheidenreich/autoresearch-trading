"""Frozen offline entry-model machinery for the Path-D Tier-S study.

This module has no broker, network, registry, paper-trading, or protected-holdout
capability.  Public fit wrappers accept only a hash-bound ``FrozenFitAuthorization``
and load the corresponding verified dataset internally before invoking their private
implementation seam.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
import hashlib
import json
import math
import random
import re
from typing import Any, ClassVar, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
import torch
from torch import nn

from v4.research import pathd_entry_exit as prereg
from v4.research.pathd_entry_features import hgb_signed17_summaries


_HEX64 = re.compile(r"[0-9a-f]{64}")
_WEIGHT_ROLES = {"outer_weights", "nested_weights", "full_weights"}
_CALIBRATION_TO_WEIGHT_ROLE = {
    "outer_calibration": "outer_weights",
    "nested_calibration": "nested_weights",
    "full_calibration": "full_weights",
}


def _is_hex64(value: Any) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def _canonical(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": _canonical(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu().contiguous()
        return {
            "dtype": str(detached.dtype),
            "shape": list(detached.shape),
            "data": _canonical(detached.tolist()),
        }
    if type(value) is float:
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "+inf" if value > 0 else "-inf"}
        return value
    if type(value) is dict:
        return {str(key): _canonical(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_canonical(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _clone_jsonish(value: Any) -> Any:
    if type(value) is dict:
        return {key: _clone_jsonish(item) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(_clone_jsonish(item) for item in value)
    if type(value) is list:
        return [_clone_jsonish(item) for item in value]
    return value


def _readonly(value: Any, *, dtype: Any, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape drift: {array.shape} != {shape}")
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _field_values(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if type(value) is dict:
        return dict(value)
    raise TypeError("typed Path-D artifact required")


def _validate_artifact_hash(value: Any, *, hash_field: str = "artifact_sha256") -> dict[str, Any]:
    semantic = _field_values(value)
    digest = semantic.pop(hash_field, None)
    if not _is_hex64(digest) or digest != prereg.stable_hash(semantic):
        raise ValueError(f"{hash_field} drift")
    return _field_values(value)


def _seal_dataclass(cls: type[Any], values: Mapping[str, Any], *, hash_field: str = "artifact_sha256") -> Any:
    semantic = dict(values)
    semantic.pop(hash_field, None)
    return cls(**semantic, **{hash_field: prereg.stable_hash(semantic)})


def _coerce_exact_dataclass(value: Any, cls: type[Any]) -> Any:
    """Accept the in-memory type or its exact JSON object representation."""

    if type(value) is cls:
        return value
    expected = {field.name for field in fields(cls)}
    if type(value) is dict and set(value) == expected:
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{cls.__name__} JSON representation drift") from exc
    raise TypeError(f"exact {cls.__name__} or exact JSON representation required")


def _validate_nested_shape(
    value: Any, shape: tuple[int, ...], predicate: Any, *, name: str
) -> None:
    if not shape:
        if not predicate(value):
            raise ValueError(f"{name} leaf drift")
        return
    if type(value) not in (tuple, list) or len(value) != shape[0]:
        raise ValueError(f"{name} shape drift")
    for item in value:
        _validate_nested_shape(item, shape[1:], predicate, name=name)


def _authorization_sha256(authorization: prereg.FrozenFitAuthorization) -> str:
    if type(authorization) is not prereg.FrozenFitAuthorization:
        raise TypeError("exact FrozenFitAuthorization required")
    return prereg.stable_hash(authorization.to_dict())


@dataclass(frozen=True)
class EntryModelInputV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_model_input.v1"

    schema_version: str
    session: str
    decision_time_ns: int
    signed17_frame: Any
    signed17_history: Any
    hgb_summaries: np.ndarray
    current_offsets: np.ndarray
    current_rights: np.ndarray
    contract_ids: np.ndarray
    physical_action_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("entry model-input schema drift")
        if type(self.session) is not str or not self.session:
            raise TypeError("entry model-input session drift")
        if type(self.decision_time_ns) is not int:
            raise TypeError("entry model-input decision clock drift")
        summaries = _readonly(
            self.hgb_summaries, dtype=np.float64, shape=(42, 444), name="hgb_summaries"
        )
        offsets = _readonly(
            self.current_offsets, dtype=np.float64, shape=(42,), name="current_offsets"
        )
        rights = _readonly(
            self.current_rights, dtype=object, shape=(42,), name="current_rights"
        )
        identities = _readonly(
            self.contract_ids, dtype=object, shape=(42,), name="contract_ids"
        )
        mask = _readonly(
            self.physical_action_mask,
            dtype=np.bool_,
            shape=(42,),
            name="physical_action_mask",
        )
        if any(item not in {"C", "P"} for item in rights.tolist()):
            raise ValueError("entry model-input right drift")
        if len(set(identities.tolist())) != 42 or any(
            type(item) is not str or not item for item in identities.tolist()
        ):
            raise ValueError("entry model-input contract identity drift")
        object.__setattr__(self, "hgb_summaries", summaries)
        object.__setattr__(self, "current_offsets", offsets)
        object.__setattr__(self, "current_rights", rights)
        object.__setattr__(self, "contract_ids", identities)
        object.__setattr__(self, "physical_action_mask", mask)

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self)


@dataclass(frozen=True)
class EntryPredictionV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_prediction.v1"

    schema_version: str
    family: str
    model_artifact_sha256: str
    authorization_sha256: str
    dataset_sha256: str
    input_sha256: str
    prediction_values: Any
    prediction_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("entry prediction schema drift")
        for value in (
            self.model_artifact_sha256,
            self.authorization_sha256,
            self.dataset_sha256,
            self.input_sha256,
        ):
            if not _is_hex64(value):
                raise ValueError("entry prediction identity drift")
        semantic = {field.name: getattr(self, field.name) for field in fields(self)}
        digest = semantic.pop("prediction_sha256")
        if digest != _canonical_sha256(semantic):
            raise ValueError("entry prediction self-hash drift")

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self)


@dataclass(frozen=True)
class EntryModelBundleV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_model_bundle.v1"

    schema_version: str
    family: str
    fit_role: str
    outer_fold: int | None
    inner_fold: int | None
    authorization_sha256: str
    sessions_sha256_newline: str
    dataset_sha256: str
    artifact_sha256: str
    payload: Any


@dataclass(frozen=True)
class EntryCalibrationBundleV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_calibration_bundle.v1"

    schema_version: str
    model_artifact_sha256: str
    calibration_role: str
    outer_fold: int | None
    inner_fold: int | None
    authorization_sha256: str
    sessions_sha256_newline: str
    dataset_sha256: str
    artifact_sha256: str
    payload: Any


@dataclass(frozen=True)
class EntryComposerBundleV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_composer_bundle.v1"

    schema_version: str
    model_artifact_sha256: str
    calibration_artifact_sha256: str
    outer_fold: int | None
    inner_fold: int | None
    composer_spec_sha256: str
    artifact_sha256: str
    payload: Any


@dataclass(frozen=True)
class EntryNegativeControlBundleV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_negative_control_bundle.v1"

    schema_version: str
    control_id: str
    base_family: str
    fit_role: str
    outer_fold: int | None
    inner_fold: int | None
    authorization_sha256: str
    dataset_sha256: str
    seed_key_sha256: str
    artifact_sha256: str
    payload: Any


@dataclass(frozen=True)
class EntryNegativeControlCalibrationBundleV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_negative_control_calibration_bundle.v1"

    schema_version: str
    control_artifact_sha256: str
    control_id: str
    base_family: str
    calibration_role: str
    outer_fold: int | None
    inner_fold: int | None
    authorization_sha256: str
    dataset_sha256: str
    artifact_sha256: str
    payload: Any


@dataclass(frozen=True)
class EntryNegativeControlComposerBundleV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_negative_control_composer_bundle.v1"

    schema_version: str
    control_artifact_sha256: str
    calibration_artifact_sha256: str
    control_id: str
    base_family: str
    outer_fold: int | None
    inner_fold: int | None
    transform_stage: str
    composer_spec_sha256: str
    artifact_sha256: str
    payload: Any


@dataclass(frozen=True)
class EntryNegativeControlManifestV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_negative_control_manifest.v1"

    schema_version: str
    outer_fold: int
    required_control_ids: tuple[str, ...]
    required_base_families: tuple[str, ...]
    control_bundle_sha256s: tuple[str, ...]
    control_calibration_sha256s: tuple[str, ...]
    control_composer_sha256s: tuple[str, ...]
    sign_reversed_base_composer_sha256s: tuple[str, ...]
    seed_receipt_sha256s: tuple[str, ...]
    replay_config_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class EntryReplayEvaluationCellV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_replay_evaluation_cell.v1"

    schema_version: str
    outer_fold: int
    owner_policy_id: str
    replay_policy_id: str
    channel: str
    matched_random_seed: int | None
    authorization_sha256: str
    dataset_sha256: str
    evaluation: Any
    evaluation_sha256: str
    terminal_journal_sha256: str
    candidate_budget_sha256: str | None
    ordered_schedule_sha256: str | None
    structural_skip_trace_sha256: str | None
    realized_intents_and_fills_sha256: str | None
    cell_sha256: str


@dataclass(frozen=True)
class EntryActionCalibrationObservationV1:
    """Journal-derived action observation used by the fixed calibration audit.

    The canonical public owner of this schema is the research-replay module.  It is
    repeated here deliberately because the frozen fixed-vector test exercises the
    pure reconstruction statistic without importing execution machinery.
    """

    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_action_calibration_observation.v1"

    schema_version: str
    outer_fold: int
    action: str
    session: str
    trajectory_or_episode_id: str
    predicted_mean_micros: int
    predicted_lower_micros: int
    realized_micros: int
    source_policy_evaluation_sha256: str
    source_transition_sha256: str
    observation_sha256: str


@dataclass(frozen=True)
class EntryNegativeControlPanelV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_negative_control_panel.v1"

    schema_version: str
    holdout_caveat: str
    outer_fold: int
    manifest_sha256: str
    required_control_ids: tuple[str, ...]
    channels: tuple[str, ...]
    control_bundle_sha256s: tuple[str, ...]
    control_channel_evaluations: Any
    control_action_calibration_observations: Any
    control_time300_trajectory_evidence: Any
    artifact_sha256: str


@dataclass(frozen=True)
class EntryControlReplayConfigV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_control_replay_config.v1"

    schema_version: str
    outer_fold: int
    fill_law_hash: str
    control_exit_sha256: str
    required_policy_ids: tuple[str, ...]
    matched_random_seeds: tuple[int, ...]
    fee_paths: tuple[int, ...]
    sell_delay_rungs_ms: tuple[int, ...]
    headline_config_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class EntryControlReplayResultV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_control_replay_result.v1"

    schema_version: str
    holdout_caveat: str
    outer_fold: int
    replay_config_sha256: str
    candidate_outer_evaluation_sha256: str
    neural_outer_evaluation_sha256: str
    channels: tuple[str, ...]
    candidate_channel_evaluations: Any
    p5_channel_evaluations: Any
    matched_random_owner_policy_ids: tuple[str, ...]
    matched_random_seeds: tuple[int, ...]
    matched_random_channel_evaluations: Any
    candidate_action_calibration_observations: Any
    candidate_time300_trajectory_evidence: Any
    candidate_budget_sha256s: Any
    ordered_schedule_sha256s: Any
    structural_skip_trace_sha256s: Any
    realized_intents_and_fills_sha256s: Any
    artifact_sha256: str


@dataclass(frozen=True)
class EntryControlExitSelectionV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_control_exit_selection.v1"

    schema_version: str
    holdout_caveat: str
    outer_fold: int
    model_fit_sessions_sha256_newline: str
    candidate_policy_ids: tuple[str, ...]
    candidate_evaluation_sha256s: tuple[str, ...]
    valid_session_counts: tuple[int, ...]
    total_net_pnl_micros: tuple[int, ...]
    selected_policy_id: str
    selection_rule: str
    artifact_sha256: str


@dataclass(frozen=True)
class EntryControlExitCandidateEvaluationV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_control_exit_candidate_evaluation.v1"

    schema_version: str
    outer_fold: int
    policy_id: str
    model_fit_sessions_sha256_newline: str
    session_pnl_micros: tuple[int, ...]
    session_terminal_journal_sha256s: tuple[str, ...]
    valid_session_count: int
    total_net_pnl_micros: int
    evaluation_sha256: str


@dataclass(frozen=True)
class EntryMatchedRandomCandidateBudgetV1:
    """Outcome-blind exact candidate BUY-intent counts for one replay game."""

    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_matched_random_candidate_budget.v1"

    schema_version: str
    outer_fold: int
    owner_policy_id: str
    channel: str
    fee_path: int
    matching_fields: tuple[str, ...]
    ordered_cells_and_counts: tuple[tuple[tuple[Any, ...], int], ...]
    source_action_decision_sha256s: tuple[str, ...]
    candidate_budget_sha256: str


@dataclass(frozen=True)
class EntryNestedReplayConfigV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_nested_replay_config.v1"

    schema_version: str
    outer_fold: int
    inner_fold: int
    fill_law_hash: str
    policy_ids: tuple[str, ...]
    fee_path: int
    sell_delay_ms: int
    floor_on: bool
    exit_policy_id: str
    nested_account_scope_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class EntryActionDecisionV1:
    SCHEMA_VERSION: ClassVar[str] = "pathd.entry_action_decision.v1"

    schema_version: str
    action: str
    reason: str
    selected_action_index: int | None
    selected_source_neutral_contract_id: str | None
    selected_contract: Any
    reference_bid_micros: int | None
    reference_ask_micros: int | None
    buy_hard_limit_micros: int | None
    available_horizons: tuple[str, ...]
    mean_lcb_dollars: float | None
    mean_lcb_return: float | None
    q10_dollars: float | None
    q10_return: float | None
    physical_action_mask: tuple[bool, ...]
    dynamic_account_mask: tuple[bool, ...]
    combined_action_mask: tuple[bool, ...]
    authorization_sha256: str
    dataset_sha256: str
    example_sha256: str
    model_input_sha256: str
    prediction_sha256: str
    composer_sha256: str
    prior_journal_root_sha256: str
    fill_law_hash: str
    decision_sha256: str


def model_input_from_example(example: Any, /) -> EntryModelInputV1:
    source = getattr(example, "model_input", None)
    if source is None:
        raise TypeError("EntryExampleV1.model_input is required")

    def get(name: str) -> Any:
        if type(source) is dict:
            if name not in source:
                raise ValueError(f"entry example model input lacks {name}")
            return source[name]
        if not hasattr(source, name):
            raise ValueError(f"entry example model input lacks {name}")
        return getattr(source, name)

    frame = get("signed17_frame")
    history = get("signed17_history")
    offsets = np.asarray(get("current_offsets"), dtype=np.float64).reshape(-1)
    rights = np.asarray(get("current_rights"), dtype=object).reshape(-1)
    contracts = np.asarray(get("contract_ids"), dtype=object).reshape(-1)
    summaries = hgb_signed17_summaries(
        history, current_offsets=offsets, current_rights=rights
    )
    if (type(source) is dict and "hgb_summaries" in source) or (
        type(source) is not dict and hasattr(source, "hgb_summaries")
    ):
        supplied_summaries = np.asarray(get("hgb_summaries"), dtype=np.float64)
        if supplied_summaries.shape != summaries.shape or not np.array_equal(
            supplied_summaries, summaries, equal_nan=True
        ):
            raise ValueError("entry example HGB summaries are not causally reconstructible")
    facts = getattr(example, "action_execution_facts", None)
    if type(facts) not in (tuple, list) or len(facts) != 42:
        raise ValueError("entry example must carry all 42 execution facts")
    physical_values = [getattr(fact, "physical_eligible", None) for fact in facts]
    if any(type(value) is not bool for value in physical_values):
        raise TypeError("entry execution-fact physical mask drift")
    physical = np.asarray(physical_values, dtype=np.bool_)
    session = get("session")
    decision_time_ns = get("decision_time_ns")
    if any(
        getattr(fact, "session", None) != session
        or getattr(fact, "decision_time_ns", None) != decision_time_ns
        for fact in facts
    ):
        raise ValueError("entry execution fact/model-input identity drift")
    return EntryModelInputV1(
        schema_version=EntryModelInputV1.SCHEMA_VERSION,
        session=session,
        decision_time_ns=decision_time_ns,
        signed17_frame=frame,
        signed17_history=history,
        hgb_summaries=summaries,
        current_offsets=offsets,
        current_rights=rights,
        contract_ids=contracts,
        physical_action_mask=physical,
    )


def _seal_prediction(
    *, family: str, model_hash: str, authorization_hash: str,
    dataset_hash: str, inputs: EntryModelInputV1, values: Any,
) -> EntryPredictionV1:
    semantic = {
        "schema_version": EntryPredictionV1.SCHEMA_VERSION,
        "family": family,
        "model_artifact_sha256": model_hash,
        "authorization_sha256": authorization_hash,
        "dataset_sha256": dataset_hash,
        "input_sha256": inputs.canonical_sha256(),
        "prediction_values": values,
    }
    return EntryPredictionV1(
        **semantic, prediction_sha256=_canonical_sha256(semantic)
    )


def causal_probe_predict(inputs: EntryModelInputV1, /) -> EntryPredictionV1:
    if type(inputs) is not EntryModelInputV1:
        raise TypeError("causal probe accepts only EntryModelInputV1")
    summaries = np.asarray(inputs.hgb_summaries, dtype=np.float64)
    values = np.nan_to_num(summaries[:, :40], nan=0.0, posinf=0.0, neginf=0.0)
    return _seal_prediction(
        family="CAUSAL_PROBE",
        model_hash="0" * 64,
        authorization_hash="0" * 64,
        dataset_hash="0" * 64,
        inputs=inputs,
        values=values,
    )


def predict_entry(bundle: EntryModelBundleV1, inputs: EntryModelInputV1, /) -> EntryPredictionV1:
    validated = validate_entry_model_bundle(bundle)
    if type(inputs) is not EntryModelInputV1:
        raise TypeError("entry prediction accepts only EntryModelInputV1")
    payload = validated.payload
    if type(payload) is dict and "fixed_prediction_values" in payload:
        values = np.asarray(payload["fixed_prediction_values"], dtype=np.float64)
    else:
        values = _predict_fitted_payload(validated.family, payload, inputs)
    if values.shape != (42, 40) or not np.isfinite(values).all():
        raise ValueError("entry prediction tensor drift")
    return _seal_prediction(
        family=validated.family,
        model_hash=validated.artifact_sha256,
        authorization_hash=validated.authorization_sha256,
        dataset_hash=validated.dataset_sha256,
        inputs=inputs,
        values=values,
    )


def session_balanced_population_mean_std(
    values: Any, /, *, sessions: Sequence[str], valid: Any
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid)
    if array.ndim != 1 or mask.shape != array.shape or mask.dtype != np.bool_:
        raise ValueError("session-balanced vector shape/type drift")
    if len(sessions) != len(array) or any(type(item) is not str or not item for item in sessions):
        raise ValueError("session-balanced identity drift")
    retained = [index for index in range(len(array)) if bool(mask[index])]
    if not retained or any(not math.isfinite(float(array[index])) for index in retained):
        raise ValueError("session-balanced population is empty or nonfinite")
    ordered_sessions = tuple(dict.fromkeys(sessions[index] for index in retained))
    counts = {
        session: sum(sessions[index] == session for index in retained)
        for session in ordered_sessions
    }
    session_count = len(ordered_sessions)
    weighted = [
        (float(array[index]), 1.0 / (session_count * counts[sessions[index]]))
        for index in retained
    ]
    mean = math.fsum(value * weight for value, weight in weighted)
    variance = math.fsum(weight * (value - mean) ** 2 for value, weight in weighted)
    if not math.isfinite(mean) or not math.isfinite(variance) or variance < 0.0:
        raise ValueError("session-balanced statistic is nonfinite")
    return mean, math.sqrt(variance)


def weighted_lower_conformal_correction(
    raw_lower: Any, realized: Any, /, *, sessions: Sequence[str],
    identities: Sequence[str], alpha: float = 0.1,
) -> float:
    raw = np.asarray(raw_lower, dtype=np.float64)
    actual = np.asarray(realized, dtype=np.float64)
    if raw.ndim != 1 or actual.shape != raw.shape or len(sessions) != len(raw) or len(identities) != len(raw):
        raise ValueError("conformal vector shape drift")
    if type(alpha) is not float or not (0.0 < alpha < 1.0):
        raise ValueError("conformal alpha drift")
    if not np.isfinite(raw).all() or not np.isfinite(actual).all():
        raise ValueError("conformal value nonfinite")
    if len(set(identities)) != len(identities):
        raise ValueError("conformal identities are not unique")
    ordered_sessions = tuple(dict.fromkeys(sessions))
    if not ordered_sessions:
        raise ValueError("conformal population empty")
    counts = {session: sessions.count(session) for session in ordered_sessions}
    rows = sorted(
        (
            float(actual[index] - raw[index]),
            sessions[index],
            identities[index],
            1.0 / (len(ordered_sessions) * counts[sessions[index]]),
        )
        for index in range(len(raw))
    )
    grouped: list[tuple[float, float]] = []
    for residual, _session, _identity, weight in rows:
        if grouped and residual == grouped[-1][0]:
            grouped[-1] = (residual, grouped[-1][1] + weight)
        else:
            grouped.append((residual, weight))
    cumulative = 0.0
    for residual, weight in grouped:
        cumulative += weight
        if cumulative + 1e-15 >= alpha:
            return residual
    return grouped[-1][0]


def compose_enter_statistics(
    *, mean_dollars: Any, mean_returns: Any, q10_dollars: Any,
    q10_returns: Any, available: Any,
) -> dict[str, np.ndarray]:
    arrays = [np.asarray(value, dtype=np.float64) for value in (
        mean_dollars, mean_returns, q10_dollars, q10_returns
    )]
    if any(array.shape != (42, 10) for array in arrays):
        raise ValueError("entry composite component shape drift")
    horizon_mask = np.asarray(available)
    if horizon_mask.shape != (5,) or horizon_mask.dtype != np.bool_ or not horizon_mask.any():
        raise ValueError("entry available-horizon mask drift")
    component_mask = np.repeat(horizon_mask, 2)

    def decimal_ratio(value: float) -> tuple[int, int]:
        rendered = repr(float(value))
        sign = -1 if rendered.startswith("-") else 1
        rendered = rendered.lstrip("+-")
        if "e" in rendered.lower():
            mantissa, exponent_text = re.split("[eE]", rendered)
            exponent = int(exponent_text)
        else:
            mantissa, exponent = rendered, 0
        if "." in mantissa:
            whole, fraction = mantissa.split(".", 1)
            numerator = int((whole or "0") + fraction)
            exponent -= len(fraction)
        else:
            numerator = int(mantissa)
        return sign * numerator, exponent

    def stable_row_mean(array: np.ndarray) -> np.ndarray:
        selected = array[:, component_mask]
        result: list[float] = []
        for row in selected:
            if not np.isfinite(row).all():
                result.append(float("nan"))
                continue
            ratios = [decimal_ratio(float(value)) for value in row]
            minimum_exponent = min(exponent for _numerator, exponent in ratios)
            numerator = sum(
                value * 10 ** (exponent - minimum_exponent)
                for value, exponent in ratios
            )
            if minimum_exponent >= 0:
                numerator *= 10 ** minimum_exponent
                denominator = len(ratios)
            else:
                denominator = len(ratios) * 10 ** (-minimum_exponent)
            result.append(numerator / denominator)
        return np.asarray(result, dtype=np.float64)

    outputs = [stable_row_mean(array) for array in arrays]
    return {
        "mean_lcb_dollars": outputs[0],
        "mean_lcb_return": outputs[1],
        "q10_dollars": outputs[2],
        "q10_return": outputs[3],
    }


def wait_raw_lower_statistic(legal_action_q10_dollar_composites: Iterable[float], /) -> float:
    values = tuple(float(value) for value in legal_action_q10_dollar_composites)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("WAIT lower statistic nonfinite")
    return max(values, default=0.0)


def session_bootstrap_mean_lcb_correction(
    residuals: Any, /, *, sessions: Sequence[str], seed: int, resamples: int
) -> float:
    """Return the frozen nearest-rank 10% bootstrap correction.

    Rows are first reduced to one equally weighted mean per session.  The bootstrap
    then samples exactly ``S`` session means with replacement for each replicate,
    using NumPy PCG64 and the fixed stable nearest-rank convention.
    """

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(sessions) or len(values) == 0:
        raise ValueError("session-bootstrap population shape drift")
    if not np.isfinite(values).all():
        raise ValueError("session-bootstrap residual nonfinite")
    if (
        any(type(session) is not str or not session for session in sessions)
        or type(seed) is not int
        or seed < 0
        or type(resamples) is not int
        or resamples <= 0
    ):
        raise ValueError("session-bootstrap identity/configuration drift")
    ordered_sessions = tuple(dict.fromkeys(sessions))
    means = np.asarray(
        [
            math.fsum(
                float(values[index])
                for index, observed_session in enumerate(sessions)
                if observed_session == session
            )
            / sum(observed_session == session for observed_session in sessions)
            for session in ordered_sessions
        ],
        dtype=np.float64,
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    sampled = generator.integers(
        0, len(means), size=(resamples, len(means))
    )
    bootstrap_means = means[sampled].mean(axis=1)
    rank = min(resamples, max(1, math.ceil((resamples + 1) * 0.10)))
    return float(np.sort(bootstrap_means, kind="stable")[rank - 1])


def _validate_action_calibration_observation(
    value: Any, *, action: str, evidence_sessions: set[str]
) -> EntryActionCalibrationObservationV1:
    if type(value) is not EntryActionCalibrationObservationV1:
        raise TypeError("exact EntryActionCalibrationObservationV1 required")
    semantic = _field_values(value)
    digest = semantic.pop("observation_sha256")
    if (
        value.schema_version != value.SCHEMA_VERSION
        or type(value.outer_fold) is not int
        or not 1 <= value.outer_fold <= 5
        or value.action != action
        or value.session not in evidence_sessions
        or type(value.trajectory_or_episode_id) is not str
        or not value.trajectory_or_episode_id
        or any(
            type(getattr(value, name)) is not int
            for name in (
                "predicted_mean_micros",
                "predicted_lower_micros",
                "realized_micros",
            )
        )
        or not _is_hex64(value.source_policy_evaluation_sha256)
        or not _is_hex64(value.source_transition_sha256)
        or digest != prereg.stable_hash(semantic)
    ):
        raise ValueError("entry action calibration observation drift")
    return value


def reconstruct_entry_action_calibration_gate_inputs(
    observations: Sequence[EntryActionCalibrationObservationV1], /, *,
    action: str, evidence_sessions: Sequence[str], seed: int,
) -> dict[str, Any]:
    """Reconstruct the fixed action-coverage and adjacent-decile inputs.

    This is a pure reconstruction helper: it consumes already journal-derived,
    self-hashed observations and has no evidence loader or mutable state.
    """

    if action not in {"ENTER", "WAIT"}:
        raise ValueError("entry action calibration action drift")
    if (
        type(seed) is not int
        or seed < 0
        or type(evidence_sessions) not in (tuple, list)
        or not evidence_sessions
        or any(type(session) is not str or not session for session in evidence_sessions)
        or len(set(evidence_sessions)) != len(evidence_sessions)
    ):
        raise ValueError("entry action calibration evidence scope drift")
    session_set = set(evidence_sessions)
    rows = [
        _validate_action_calibration_observation(
            row, action=action, evidence_sessions=session_set
        )
        for row in observations
    ]
    identities = [
        (row.action, row.session, row.trajectory_or_episode_id) for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("entry action calibration trajectory reuse")
    rows.sort(
        key=lambda row: (
            row.predicted_mean_micros,
            row.session,
            row.trajectory_or_episode_id,
        )
    )
    count = len(rows)
    base_size, remainder = divmod(count, 10)
    bins: list[list[EntryActionCalibrationObservationV1]] = []
    offset = 0
    for index in range(10):
        size = base_size + int(index < remainder)
        bins.append(rows[offset : offset + size])
        offset += size
    session_means: list[dict[str, float]] = []
    for bucket in bins:
        grouped: dict[str, list[int]] = {}
        for row in bucket:
            grouped.setdefault(row.session, []).append(row.realized_micros)
        session_means.append(
            {
                session: float(math.fsum(values)) / len(values)
                for session, values in grouped.items()
            }
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    valid_counts: list[int] = []
    total_draws: list[int] = []
    upper_bounds: list[float] = []
    for lower_index in range(9):
        lower = session_means[lower_index]
        upper = session_means[lower_index + 1]
        differences: list[float] = []
        draws = 0
        while len(differences) < 5_000 and draws < 50_000:
            draws += 1
            sampled = generator.integers(
                0, len(evidence_sessions), size=len(evidence_sessions)
            )
            sampled_sessions = [evidence_sessions[int(index)] for index in sampled]
            lower_values = [lower[session] for session in sampled_sessions if session in lower]
            upper_values = [upper[session] for session in sampled_sessions if session in upper]
            if not lower_values or not upper_values:
                continue
            differences.append(
                float(math.fsum(lower_values)) / len(lower_values)
                - float(math.fsum(upper_values)) / len(upper_values)
            )
        valid_counts.append(len(differences))
        total_draws.append(draws)
        if differences:
            ordered = sorted(differences)
            rank = min(
                len(ordered),
                max(1, math.ceil((len(ordered) + 1) * 0.90)),
            )
            upper_bounds.append(float(ordered[rank - 1]))
        else:
            upper_bounds.append(0.0)
    return {
        "distinct_trajectory_count": count,
        "coverage_numerator": sum(
            row.realized_micros >= row.predicted_lower_micros for row in rows
        ),
        "coverage_denominator": count,
        "decile_trajectory_counts": [len(bucket) for bucket in bins],
        "decile_distinct_session_counts": [
            len({row.session for row in bucket}) for bucket in bins
        ],
        "adjacent_valid_bootstrap_replicates": valid_counts,
        "adjacent_total_draws": total_draws,
        "adjacent_upper_bounds_micros": upper_bounds,
    }


def select_entry_action_index(
    *, mean_lcb_dollars: Any, mean_lcb_returns: Any, q10_dollars: Any,
    q10_returns: Any, physical_action_mask: Any, dynamic_account_mask: Any,
    tie_break_keys: Sequence[Any],
) -> dict[str, Any]:
    numeric = [np.asarray(value, dtype=np.float64) for value in (
        mean_lcb_dollars, mean_lcb_returns, q10_dollars, q10_returns
    )]
    physical = np.asarray(physical_action_mask)
    dynamic = np.asarray(dynamic_account_mask)
    if any(value.shape != (42,) for value in numeric):
        raise ValueError("entry selection statistic shape drift")
    if physical.shape != (42,) or dynamic.shape != (42,) or physical.dtype != np.bool_ or dynamic.dtype != np.bool_:
        raise ValueError("entry selection mask drift")
    if len(tie_break_keys) != 42:
        raise ValueError("entry selection tie-break registry drift")
    combined = physical & dynamic
    legal = [
        index
        for index in range(42)
        if combined[index]
        and math.isfinite(float(numeric[0][index]))
        and math.isfinite(float(numeric[1][index]))
        and math.isfinite(float(numeric[2][index]))
        and math.isfinite(float(numeric[3][index]))
        and numeric[0][index] > 0.0
        and numeric[1][index] > 0.0
    ]
    if not legal:
        return {
            "action": "WAIT",
            "selected_action_index": None,
            "combined_action_mask": combined.tolist(),
        }
    selected = min(
        legal,
        key=lambda index: (
            -float(numeric[2][index]),
            -float(numeric[3][index]),
            tie_break_keys[index],
        ),
    )
    return {
        "action": "ENTER",
        "selected_action_index": int(selected),
        "combined_action_mask": combined.tolist(),
    }


def reverse_entry_calibrated_statistics(
    *, mean_lcb_dollars: Any, mean_lcb_returns: Any,
    q10_dollars: Any, q10_returns: Any,
) -> dict[str, np.ndarray]:
    arrays = [np.asarray(value, dtype=np.float64) for value in (
        mean_lcb_dollars, mean_lcb_returns, q10_dollars, q10_returns
    )]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("sign-reversed statistic shape drift")
    return {
        "mean_lcb_dollars": -arrays[0].copy(),
        "mean_lcb_return": -arrays[1].copy(),
        "q10_dollars": -arrays[2].copy(),
        "q10_return": -arrays[3].copy(),
    }


def build_hgb_regressor(
    *, seed: int, loss: str, quantile: float | None
) -> HistGradientBoostingRegressor:
    if type(seed) is not int or seed < 0:
        raise ValueError("HGB seed drift")
    if loss == "squared_error":
        if quantile is not None:
            raise ValueError("squared-error HGB cannot carry quantile")
    elif loss == "quantile":
        if type(quantile) is not float or quantile != 0.1:
            raise ValueError("entry q10 HGB requires quantile=0.1")
    else:
        raise ValueError("unsupported entry HGB loss")
    return HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=3,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=False,
        random_state=seed,
    )


class _EntryNeuralModule(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.token_projection = nn.Linear(34, 32, bias=True)
        self.token_activation = nn.GELU(approximate="none")
        self.temporal = nn.GRU(
            input_size=32, hidden_size=32, num_layers=1, bias=True,
            batch_first=True, dropout=0.0, bidirectional=False,
        )
        self.action_projection = nn.Linear(34, 64, bias=True)
        self.action_activation = nn.GELU(approximate="none")
        self.norm1 = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True, bias=True)
        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, dropout=0.10, bias=True,
            add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(0.10)
        self.norm2 = nn.LayerNorm(64, eps=1e-5, elementwise_affine=True, bias=True)
        self.ff1 = nn.Linear(64, 128, bias=True)
        self.ff_activation = nn.GELU(approximate="none")
        self.ff_inner_dropout = nn.Dropout(0.10)
        self.ff2 = nn.Linear(128, 64, bias=True)
        self.ff_output_dropout = nn.Dropout(0.10)
        self.output = nn.Linear(64, 40, bias=True)

    def forward(
        self, history_tokens: torch.Tensor, geometry: torch.Tensor,
        current_present: torch.Tensor,
    ) -> torch.Tensor:
        if history_tokens.ndim != 4 or history_tokens.shape[1:] != (90, 42, 34):
            raise ValueError("entry neural history shape drift")
        batch = history_tokens.shape[0]
        if geometry.shape != (batch, 42, 2):
            raise ValueError("entry neural geometry shape drift")
        if current_present.shape != (batch, 42) or current_present.dtype != torch.bool:
            raise ValueError("entry neural current-action mask drift")
        if history_tokens.dtype != torch.float32 or geometry.dtype != torch.float32:
            raise ValueError("entry neural inputs must be float32")
        if bool((~current_present.any(dim=1)).any()):
            raise ValueError("entry neural frame has no present action")
        present_history = current_present[:, None, :, None]
        clean_history = torch.where(
            present_history, history_tokens, torch.zeros_like(history_tokens)
        )
        clean_geometry = torch.where(
            current_present[:, :, None], geometry, torch.zeros_like(geometry)
        )
        token = self.token_activation(self.token_projection(clean_history))
        temporal_input = token.permute(0, 2, 1, 3).reshape(batch * 42, 90, 32)
        h0 = torch.zeros(
            (1, batch * 42, 32), dtype=history_tokens.dtype,
            device=history_tokens.device,
        )
        temporal_output, _hidden = self.temporal(temporal_input, h0)
        final = temporal_output[:, -1, :].reshape(batch, 42, 32)
        x = self.action_activation(
            self.action_projection(torch.cat((final, clean_geometry), dim=-1))
        )
        x = torch.where(current_present[:, :, None], x, torch.zeros_like(x))
        normalized = self.norm1(x)
        attention, _weights = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=~current_present, need_weights=False,
        )
        x = x + self.attention_dropout(attention)
        x = torch.where(current_present[:, :, None], x, torch.zeros_like(x))
        y = self.norm2(x)
        feedforward = self.ff2(
            self.ff_inner_dropout(self.ff_activation(self.ff1(y)))
        )
        x = x + self.ff_output_dropout(feedforward)
        x = torch.where(current_present[:, :, None], x, torch.zeros_like(x))
        output = self.output(x)
        return torch.where(
            current_present[:, :, None], output, torch.zeros_like(output)
        )


def build_entry_neural_module(*, seed: int) -> nn.Module:
    if type(seed) is not int or seed < 0:
        raise ValueError("entry neural seed drift")
    return _EntryNeuralModule(seed).cpu()


def _validate_control_id(control_id: Any) -> str:
    if control_id not in prereg.ENTRY_NEGATIVE_CONTROL_IDS:
        raise ValueError("unregistered entry negative control")
    return str(control_id)


def _validate_transform_population(
    *, target_bundles: Sequence[Any], target_validity: Sequence[Any],
    feature_histories: Sequence[Any], row_identities: Sequence[Any],
) -> None:
    count = len(row_identities)
    if count == 0 or any(
        len(values) != count
        for values in (target_bundles, target_validity, feature_histories)
    ):
        raise ValueError("negative-control population length drift")
    identity_fields = {
        "outer_fold", "session", "decision_time_ns",
        "source_neutral_contract_id", "expiry_yyyymmdd",
        "strike_milli_points", "right_code",
    }
    if any(
        type(identity) is not dict
        or set(identity) != identity_fields
        or type(identity["outer_fold"]) is not int
        or not 1 <= identity["outer_fold"] <= 5
        or type(identity["session"]) is not str
        or not identity["session"]
        or type(identity["decision_time_ns"]) is not int
        or type(identity["source_neutral_contract_id"]) is not str
        or not identity["source_neutral_contract_id"]
        or type(identity["expiry_yyyymmdd"]) is not int
        or type(identity["strike_milli_points"]) is not int
        or identity["right_code"] not in {"C", "P"}
        for identity in row_identities
    ):
        raise ValueError("negative-control row identity schema drift")
    canonical_keys = [
        (
            identity["outer_fold"], identity["session"],
            identity["decision_time_ns"],
            identity["source_neutral_contract_id"],
            identity["expiry_yyyymmdd"], identity["strike_milli_points"],
            identity["right_code"],
        )
        for identity in row_identities
    ]
    if canonical_keys != sorted(canonical_keys):
        raise ValueError("negative-control row population is not canonical")
    identity_hashes = [_canonical_sha256(identity) for identity in row_identities]
    if len(identity_hashes) != len(set(identity_hashes)):
        raise ValueError("negative-control row identities are not unique")
    first_target = target_bundles[0]
    first_validity = target_validity[0]
    if type(first_target) is not dict or type(first_validity) is not dict:
        raise TypeError("negative-control target bundle must be an object")
    if set(first_target) != set(first_validity):
        raise ValueError("negative-control target/validity keys drift")
    widths: dict[str, int] = {}
    for key in first_target:
        if type(first_target[key]) not in (tuple, list) or type(first_validity[key]) not in (tuple, list):
            raise TypeError("negative-control target components must be sequences")
        widths[key] = len(first_target[key])
        if widths[key] == 0 or len(first_validity[key]) != widths[key]:
            raise ValueError("negative-control target width drift")
    for target, validity in zip(target_bundles, target_validity, strict=True):
        if type(target) is not dict or type(validity) is not dict or set(target) != set(widths) or set(validity) != set(widths):
            raise ValueError("negative-control bundle key drift")
        for key, width in widths.items():
            if (
                type(target[key]) not in (tuple, list)
                or len(target[key]) != width
                or type(validity[key]) not in (tuple, list)
                or len(validity[key]) != width
                or any(type(flag) is not bool for flag in validity[key])
                or any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in target[key])
            ):
                raise ValueError("negative-control target component drift")
    for history in feature_histories:
        if type(history) is not dict:
            raise TypeError("negative-control feature history must be an object")


def _constant_target_bundle(
    target_bundles: Sequence[Mapping[str, Sequence[float]]],
    target_validity: Sequence[Mapping[str, Sequence[bool]]],
    row_identities: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key in target_bundles[0]:
        values: list[float] = []
        for component in range(len(target_bundles[0][key])):
            population = [
                (
                    float(target[key][component]),
                    str(identity["session"]),
                    prereg.stable_hash(identity),
                )
                for target, validity, identity in zip(
                    target_bundles, target_validity, row_identities, strict=True
                )
                if validity[key][component]
            ]
            if not population:
                raise ValueError("constant control has an empty target component")
            sessions = tuple(dict.fromkeys(session for _value, session, _identity in population))
            counts = {
                session: sum(row_session == session for _value, row_session, _identity in population)
                for session in sessions
            }
            weighted = sorted(
                (
                    value,
                    identity,
                    1.0 / (len(sessions) * counts[session]),
                )
                for value, session, identity in population
            )
            if "q10" in key.lower():
                cumulative = 0.0
                selected = weighted[-1][0]
                for value, _identity, weight in weighted:
                    cumulative += weight
                    if cumulative + 1e-15 >= 0.10:
                        selected = value
                        break
                values.append(selected)
            else:
                values.append(
                    math.fsum(value * weight for value, _identity, weight in weighted)
                )
        result[key] = values
    return result


def transform_entry_negative_control_bundle(
    *, control_id: str, target_bundles: Sequence[Any],
    target_validity: Sequence[Any], feature_histories: Sequence[Any],
    row_identities: Sequence[Any], seed: int,
) -> dict[str, Any]:
    control = _validate_control_id(control_id)
    if type(seed) is not int or seed not in prereg.MATCHED_RANDOM_SEEDS:
        raise ValueError("negative-control seed drift")
    _validate_transform_population(
        target_bundles=target_bundles,
        target_validity=target_validity,
        feature_histories=feature_histories,
        row_identities=row_identities,
    )
    identities = tuple(_clone_jsonish(item) for item in row_identities)
    source_identities = identities
    transformed_targets: tuple[Any, ...] = tuple(
        _clone_jsonish(item) for item in target_bundles
    )
    transformed_validity: tuple[Any, ...] = tuple(
        _clone_jsonish(item) for item in target_validity
    )
    transformed_histories: tuple[Any, ...] = tuple(
        _clone_jsonish(item) for item in feature_histories
    )
    if control.startswith("SHUFFLED_TARGET_"):
        keyed: list[tuple[str, int]] = []
        for index, identity in enumerate(identities):
            if type(identity) is not dict:
                raise TypeError("shuffled-target identity must be an object")
            digest = prereg.stable_hash({"attempt_seed_id": seed, **identity})
            keyed.append((digest, index))
        source_indexes = tuple(index for _digest, index in sorted(keyed))
        source_identities = tuple(identities[index] for index in source_indexes)
        transformed_targets = tuple(
            _clone_jsonish(target_bundles[index]) for index in source_indexes
        )
        transformed_validity = tuple(
            _clone_jsonish(target_validity[index]) for index in source_indexes
        )
        stage, refit = "TARGET_PRE_FIT", True
    elif control == "CONSTANT":
        constant = _constant_target_bundle(
            target_bundles, target_validity, row_identities
        )
        transformed_targets = tuple(_clone_jsonish(constant) for _ in identities)
        stage, refit = "PREDICTION_BASELINE", False
    elif control == "TIME_SHIFTED_FEATURES":
        shifted: list[dict[str, Any]] = []
        for history in feature_histories:
            if "current" not in history or "lag30" not in history:
                raise ValueError("time-shift control lacks current/lag30 history")
            row = _clone_jsonish(history)
            row["current"] = _clone_jsonish(history["lag30"])
            shifted.append(row)
        transformed_histories = tuple(shifted)
        stage, refit = "FEATURE_PRE_FIT", True
    else:
        stage, refit = "COMPOSER_POST_CALIBRATION", False
    semantic = {
        "control_id": control,
        "seed": seed,
        "row_identities": identities,
        "source_row_identities": source_identities,
        "target_bundles": transformed_targets,
        "target_validity": transformed_validity,
        "feature_histories": transformed_histories,
        "transform_stage": stage,
        "refit_required": refit,
    }
    return {
        **semantic,
        "transform_sha256": prereg.stable_hash(semantic),
    }


def _proposal_random_key(
    *, seed: int, outer_fold: int, proposal: Mapping[str, Any]
) -> tuple[int, str]:
    required = (
        "session", "decision_time_ns", "source_neutral_contract_id",
        "expiry_yyyymmdd", "strike_milli_points", "right_code",
    )
    if not all(name in proposal for name in required):
        raise ValueError("matched-random proposal lacks canonical key material")
    key = {
        "attempt_seed_id": seed,
        "outer_fold": outer_fold,
        **{name: proposal[name] for name in required},
    }
    digest = prereg.stable_hash(key)
    return int.from_bytes(bytes.fromhex(digest)[:8], "big"), digest


_MATCHED_RANDOM_FIELDS = (
    "outer_fold",
    "session",
    "call_or_put",
    "ATM_NEAR_WING",
    "decision_time_premium_band",
)
_MATCHED_RANDOM_MONEYNESS = ("ATM", "NEAR", "WING")
_MATCHED_RANDOM_PREMIUM_BANDS = (
    "cheap_le_1",
    "small_1_3",
    "medium_3_8",
    "large_8_20",
    "very_large_20p",
)


def _validated_matched_random_cell(row: Mapping[str, Any], /) -> tuple[Any, ...]:
    """Validate one cell after its bands were derived from a sealed decision."""

    if type(row) is not dict or any(name not in row for name in _MATCHED_RANDOM_FIELDS):
        raise ValueError("matched-random candidate row lacks a frozen matching field")
    cell = tuple(row[name] for name in _MATCHED_RANDOM_FIELDS)
    if (
        type(cell[0]) is not int
        or cell[0] not in range(1, 6)
        or type(cell[1]) is not str
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", cell[1]) is None
        or cell[2] not in {"C", "P"}
        or cell[3] not in _MATCHED_RANDOM_MONEYNESS
        or cell[4] not in _MATCHED_RANDOM_PREMIUM_BANDS
    ):
        raise ValueError("matched-random frozen matching cell drift")
    return cell


def _matched_random_moneyness_from_action_index(index: int, /) -> str:
    """Canonical ladder distance from ATM slot 10, independently per right."""

    if type(index) is not int or index not in range(42):
        raise ValueError("matched-random action index drift")
    distance = abs((index % 21) - 10)
    if distance <= 1:
        return "ATM"
    if distance <= 5:
        return "NEAR"
    return "WING"


def _matched_random_premium_band(ask_micros: int, /) -> str:
    if type(ask_micros) is not int or ask_micros <= 0:
        raise ValueError("matched-random decision ask drift")
    ask_cents = ask_micros / 10_000
    if ask_cents <= 100:
        return "cheap_le_1"
    if ask_cents <= 300:
        return "small_1_3"
    if ask_cents <= 800:
        return "medium_3_8"
    if ask_cents <= 2_000:
        return "large_8_20"
    return "very_large_20p"


def matched_random_candidate_row_from_decision(
    *, outer_fold: int, session: str, decision: EntryActionDecisionV1,
) -> dict[str, Any]:
    """Derive the complete outcome-blind match cell from a sealed ENTER decision."""

    if type(decision) is not EntryActionDecisionV1:
        raise TypeError("matched-random budget requires EntryActionDecisionV1")
    semantic = {
        field.name: _canonical(getattr(decision, field.name))
        for field in fields(decision)
        if field.name != "decision_sha256"
    }
    contract = decision.selected_contract
    index = decision.selected_action_index
    right = _member(contract, "right") if contract is not None else None
    if (
        decision.schema_version != decision.SCHEMA_VERSION
        or decision.decision_sha256 != prereg.stable_hash(semantic)
        or decision.action != "ENTER"
        or type(index) is not int
        or index not in range(42)
        or right not in {"C", "P"}
        or type(session) is not str
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", session) is None
        or outer_fold not in range(1, 6)
    ):
        raise ValueError("matched-random candidate decision identity drift")
    return {
        "outer_fold": outer_fold,
        "session": session,
        "call_or_put": right,
        "ATM_NEAR_WING": _matched_random_moneyness_from_action_index(index),
        "decision_time_premium_band": _matched_random_premium_band(
            decision.reference_ask_micros
        ),
        "action": "ENTER",
        "decision_sha256": decision.decision_sha256,
    }


def derive_matched_random_candidate_budget(
    *, outer_fold: int, owner_policy_id: str, channel: str, fee_path: int,
    candidate_buy_intents: Sequence[Mapping[str, Any]],
) -> EntryMatchedRandomCandidateBudgetV1:
    """Count decision-derived BUY intents before any comparator outcome is read.

    Public callers should construct rows only with
    :func:`matched_random_candidate_row_from_decision`; the outer producer does
    so directly from validated journal transitions.
    """

    if (
        type(outer_fold) is not int
        or outer_fold not in range(1, 6)
        or owner_policy_id not in prereg.entry_matched_random_owner_policy_ids()
        or channel not in prereg.ENTRY_REPLAY_CHANNELS
        or fee_path not in (3, 4)
    ):
        raise ValueError("matched-random candidate budget scope drift")
    rows = tuple(candidate_buy_intents)
    exact = {
        *_MATCHED_RANDOM_FIELDS,
        "action",
        "decision_sha256",
    }
    counts: dict[tuple[Any, ...], int] = {}
    decisions: list[str] = []
    for row in rows:
        if type(row) is not dict or set(row) != exact:
            raise ValueError("matched-random candidate budget row schema drift")
        if row["action"] != "ENTER" or not _is_hex64(row["decision_sha256"]):
            raise ValueError("matched-random budget accepts only sealed ENTER decisions")
        cell = _validated_matched_random_cell(row)
        if cell[0] != outer_fold:
            raise ValueError("matched-random budget cross-fold row")
        counts[cell] = counts.get(cell, 0) + 1
        decisions.append(row["decision_sha256"])
    if not rows or len(decisions) != len(set(decisions)):
        raise ValueError("matched-random budget is empty or reuses a decision")
    ordered = tuple(sorted(counts.items(), key=lambda item: _canonical(item[0])))
    semantic = {
        "schema_version": EntryMatchedRandomCandidateBudgetV1.SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "owner_policy_id": owner_policy_id,
        "channel": channel,
        "fee_path": fee_path,
        "matching_fields": _MATCHED_RANDOM_FIELDS,
        "ordered_cells_and_counts": ordered,
        "source_action_decision_sha256s": tuple(decisions),
    }
    return EntryMatchedRandomCandidateBudgetV1(
        **semantic,
        candidate_budget_sha256=prereg.stable_hash(_canonical(semantic)),
    )


def matched_random_budget_quotas(
    budget: EntryMatchedRandomCandidateBudgetV1, /
) -> dict[tuple[Any, ...], int]:
    row = _coerce_exact_dataclass(budget, EntryMatchedRandomCandidateBudgetV1)
    semantic = asdict(row)
    digest = semantic.pop("candidate_budget_sha256")
    if (
        row.schema_version != row.SCHEMA_VERSION
        or tuple(row.matching_fields) != _MATCHED_RANDOM_FIELDS
        or not _is_hex64(digest)
        or digest != prereg.stable_hash(_canonical(semantic))
    ):
        raise ValueError("matched-random candidate budget self-seal drift")
    quotas: dict[tuple[Any, ...], int] = {}
    for cell, count in row.ordered_cells_and_counts:
        canonical_cell = tuple(cell)
        if canonical_cell in quotas or canonical_cell[0] != row.outer_fold:
            raise ValueError("matched-random candidate budget cell drift")
        _validated_matched_random_cell(
            dict(zip(_MATCHED_RANDOM_FIELDS, canonical_cell, strict=True))
        )
        if type(count) is not int or count <= 0:
            raise ValueError("matched-random candidate budget count drift")
        quotas[canonical_cell] = count
    if not quotas or sum(quotas.values()) != len(row.source_action_decision_sha256s):
        raise ValueError("matched-random candidate budget arithmetic drift")
    return quotas


def build_matched_random_schedule(
    *, proposals: Sequence[Mapping[str, Any]], quotas: Mapping[Any, int],
    seed: int, policy_id: str, outer_fold: int,
) -> dict[str, Any]:
    if type(seed) is not int or seed not in prereg.MATCHED_RANDOM_SEEDS:
        raise ValueError("matched-random seed drift")
    expected_policy = f"MATCHED_RANDOM_{prereg.MATCHED_RANDOM_SEEDS.index(seed) + 1:02d}"
    if policy_id != expected_policy or type(outer_fold) is not int or not 1 <= outer_fold <= 5:
        raise ValueError("matched-random policy/fold drift")
    if type(quotas) is not dict or not quotas or any(
        type(value) is not int or value < 0 for value in quotas.values()
    ):
        raise ValueError("matched-random quota drift")
    rows = list(proposals)
    exact_fields = {
        "id", "cell", "session", "decision_time_ns",
        "source_neutral_contract_id", "expiry_yyyymmdd",
        "strike_milli_points", "right_code",
    }
    if not rows or any(type(row) is not dict or set(row) != exact_fields for row in rows):
        raise ValueError("matched-random proposal population drift")
    ids = [row.get("id") for row in rows]
    if any(type(value) is not str or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("matched-random proposal identity drift")
    ranked: list[tuple[Any, ...]] = []
    for row in rows:
        cell = row.get("cell")
        if (
            cell not in quotas
            or type(row.get("session")) is not str
            or not row["session"]
            or type(row.get("decision_time_ns")) is not int
            or type(row.get("source_neutral_contract_id")) is not str
            or not row["source_neutral_contract_id"]
            or type(row.get("expiry_yyyymmdd")) is not int
            or type(row.get("strike_milli_points")) is not int
            or row.get("right_code") not in {"C", "P"}
        ):
            raise ValueError("matched-random proposal cell/minute drift")
        rank, digest = _proposal_random_key(
            seed=seed, outer_fold=outer_fold, proposal=row
        )
        ranked.append(
            (
                rank,
                digest,
                json.dumps(_canonical(cell), sort_keys=True, separators=(",", ":")),
                row["decision_time_ns"],
                row["expiry_yyyymmdd"],
                row["strike_milli_points"],
                row["right_code"],
                row["source_neutral_contract_id"],
                row,
            )
        )
    remaining = dict(quotas)
    used_minutes: set[int] = set()
    selected: list[str] = []
    for *_sort_key, row in sorted(ranked):
        cell = row["cell"]
        minute = row["decision_time_ns"]
        if remaining[cell] <= 0 or minute in used_minutes:
            continue
        selected.append(row["id"])
        remaining[cell] -= 1
        used_minutes.add(minute)
        if all(value == 0 for value in remaining.values()):
            break
    if any(value != 0 for value in remaining.values()):
        raise ValueError("INSUFFICIENT_MATCHED_CONTROL: quota gap; no redraw")
    semantic = {
        "policy_id": policy_id,
        "outer_fold": outer_fold,
        "seed": seed,
        "ordered_ids": selected,
        "eligible_population_sha256": prereg.stable_hash(rows),
        "matching_budget_sha256": prereg.stable_hash(
            dict(quotas)
            if all(type(cell) is str for cell in quotas)
            else [
                {"cell": _canonical(cell), "count": quotas[cell]}
                for cell in sorted(
                    quotas,
                    key=lambda value: json.dumps(
                        _canonical(value), sort_keys=True, separators=(",", ":")
                    ),
                )
            ]
        ),
    }
    return {**semantic, "schedule_sha256": prereg.stable_hash(semantic)}


def realize_frozen_matched_random_schedule(
    schedule: Mapping[str, Any], /, *, proposal_population: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, str],
) -> dict[str, list[str]]:
    """Apply outcomes only to frozen attempts; never redraw or backfill."""

    if type(schedule) is not dict or set(schedule) != {
        "policy_id", "outer_fold", "seed", "ordered_ids",
        "eligible_population_sha256", "matching_budget_sha256", "schedule_sha256",
    }:
        raise ValueError("matched-random schedule schema drift")
    semantic = dict(schedule)
    digest = semantic.pop("schedule_sha256")
    if not _is_hex64(digest) or digest != prereg.stable_hash(semantic):
        raise ValueError("matched-random schedule self-hash drift")
    population = list(proposal_population)
    if schedule["eligible_population_sha256"] != prereg.stable_hash(population):
        raise ValueError("matched-random eligible population drift")
    ids = [row.get("id") if type(row) is dict else None for row in population]
    ordered = schedule["ordered_ids"]
    if (
        type(ordered) is not list
        or len(ordered) != len(set(ordered))
        or any(type(item) is not str or item not in ids for item in ordered)
        or type(outcomes) is not dict
        or set(outcomes) != set(ids)
        or any(value not in {"FILLED", "NO_FILL", "STRUCTURAL_SKIP"} for value in outcomes.values())
    ):
        raise ValueError("matched-random realization identity drift")
    return {
        "attempted_ids": list(ordered),
        "filled_ids": [item for item in ordered if outcomes[item] == "FILLED"],
        "no_fill_ids": [item for item in ordered if outcomes[item] == "NO_FILL"],
        "structural_skip_ids": [
            item for item in ordered if outcomes[item] == "STRUCTURAL_SKIP"
        ],
        "substituted_or_backfilled_ids": [],
    }


def seal_entry_model_bundle(
    *, family: str, authorization: prereg.FrozenFitAuthorization,
    dataset_sha256: str, payload: Any,
) -> EntryModelBundleV1:
    authorization_hash = _authorization_sha256(authorization)
    if family not in {"HGB", "NEURAL"} or authorization.role not in _WEIGHT_ROLES:
        raise ValueError("entry model family/fit-role drift")
    if not _is_hex64(dataset_sha256) or type(payload) is not dict:
        raise ValueError("entry model dataset/payload drift")
    values = {
        "schema_version": EntryModelBundleV1.SCHEMA_VERSION,
        "family": family,
        "fit_role": authorization.role,
        "outer_fold": authorization.outer_fold,
        "inner_fold": authorization.inner_fold,
        "authorization_sha256": authorization_hash,
        "sessions_sha256_newline": authorization.sessions_sha256_newline,
        "dataset_sha256": dataset_sha256,
        "payload": _clone_jsonish(payload),
    }
    result = _seal_dataclass(EntryModelBundleV1, values)
    validate_entry_model_bundle(result)
    if family == "HGB":
        key = id(authorization)
        prior = _HGB_BASELINE_BY_AUTHORIZATION_ID.get(key)
        if prior is None:
            _HGB_BASELINE_BY_AUTHORIZATION_ID[key] = (authorization, result.artifact_sha256)
    return result


def _valid_fit_scope(role: str, outer_fold: Any, inner_fold: Any) -> bool:
    if role.startswith("outer_"):
        return type(outer_fold) is int and 1 <= outer_fold <= 5 and inner_fold is None
    if role.startswith("nested_"):
        return (
            type(outer_fold) is int and 1 <= outer_fold <= 5
            and type(inner_fold) is int and 1 <= inner_fold <= 4
        )
    if role.startswith("full_"):
        return outer_fold is None and inner_fold is None
    return False


def validate_entry_model_bundle(bundle: Any, /) -> EntryModelBundleV1:
    bundle = _coerce_exact_dataclass(bundle, EntryModelBundleV1)
    if (
        bundle.schema_version != bundle.SCHEMA_VERSION
        or bundle.family not in {"HGB", "NEURAL"}
        or bundle.fit_role not in _WEIGHT_ROLES
        or not _valid_fit_scope(bundle.fit_role, bundle.outer_fold, bundle.inner_fold)
        or not _is_hex64(bundle.authorization_sha256)
        or not _is_hex64(bundle.sessions_sha256_newline)
        or not _is_hex64(bundle.dataset_sha256)
        or type(bundle.payload) is not dict
    ):
        raise ValueError("entry model bundle identity drift")
    _validate_artifact_hash(bundle)
    return bundle


def seal_entry_calibration_bundle(
    *, authorization: prereg.FrozenFitAuthorization,
    model_bundle: EntryModelBundleV1, dataset_sha256: str, payload: Any,
) -> EntryCalibrationBundleV1:
    model = validate_entry_model_bundle(model_bundle)
    authorization_hash = _authorization_sha256(authorization)
    if authorization.role not in _CALIBRATION_TO_WEIGHT_ROLE:
        raise ValueError("entry calibration role drift")
    if (
        model.fit_role != _CALIBRATION_TO_WEIGHT_ROLE[authorization.role]
        or model.outer_fold != authorization.outer_fold
        or model.inner_fold != authorization.inner_fold
        or not _is_hex64(dataset_sha256)
        or type(payload) is not dict
    ):
        raise ValueError("entry calibration/model scope drift")
    values = {
        "schema_version": EntryCalibrationBundleV1.SCHEMA_VERSION,
        "model_artifact_sha256": model.artifact_sha256,
        "calibration_role": authorization.role,
        "outer_fold": authorization.outer_fold,
        "inner_fold": authorization.inner_fold,
        "authorization_sha256": authorization_hash,
        "sessions_sha256_newline": authorization.sessions_sha256_newline,
        "dataset_sha256": dataset_sha256,
        "payload": _clone_jsonish(payload),
    }
    result = _seal_dataclass(EntryCalibrationBundleV1, values)
    validate_entry_calibration_bundle(result, model_bundle=model)
    return result


def validate_entry_calibration_bundle(
    bundle: Any, /, *, model_bundle: EntryModelBundleV1
) -> EntryCalibrationBundleV1:
    model = validate_entry_model_bundle(model_bundle)
    bundle = _coerce_exact_dataclass(bundle, EntryCalibrationBundleV1)
    expected_role = {
        "outer_weights": "outer_calibration",
        "nested_weights": "nested_calibration",
        "full_weights": "full_calibration",
    }[model.fit_role]
    if (
        bundle.schema_version != bundle.SCHEMA_VERSION
        or bundle.model_artifact_sha256 != model.artifact_sha256
        or bundle.calibration_role != expected_role
        or bundle.outer_fold != model.outer_fold
        or bundle.inner_fold != model.inner_fold
        or not _valid_fit_scope(bundle.calibration_role, bundle.outer_fold, bundle.inner_fold)
        or not _is_hex64(bundle.authorization_sha256)
        or not _is_hex64(bundle.sessions_sha256_newline)
        or not _is_hex64(bundle.dataset_sha256)
        or type(bundle.payload) is not dict
    ):
        raise ValueError("entry calibration bundle drift")
    _validate_artifact_hash(bundle)
    return bundle


def apply_entry_calibration(
    bundle: EntryCalibrationBundleV1, prediction: EntryPredictionV1, /
) -> EntryPredictionV1:
    if type(bundle) is not EntryCalibrationBundleV1 or type(prediction) is not EntryPredictionV1:
        raise TypeError("typed calibration and prediction required")
    if bundle.model_artifact_sha256 != prediction.model_artifact_sha256:
        raise ValueError("calibration/prediction model drift")
    values = np.asarray(prediction.prediction_values, dtype=np.float64)
    if (
        "mean_lcb_corrections" in bundle.payload
        or "q10_corrections" in bundle.payload
    ):
        mean = bundle.payload.get("mean_lcb_corrections")
        q10 = bundle.payload.get("q10_corrections")
        if (
            type(mean) not in (tuple, list)
            or type(q10) not in (tuple, list)
            or len(mean) != _ENTRY_HEAD_COUNT
            or len(q10) != _ENTRY_HEAD_COUNT
        ):
            raise ValueError("entry calibration correction-vector drift")
        offsets = np.asarray([*mean, *q10], dtype=np.float64)[None, :]
    else:
        # Retained only for the frozen synthetic bundle fixtures.
        offsets = bundle.payload.get("additive_offsets", 0.0)
    calibrated = values + np.asarray(offsets, dtype=np.float64)
    if calibrated.shape != values.shape or not np.isfinite(calibrated).all():
        raise ValueError("entry calibrated prediction drift")
    semantic = {
        "schema_version": EntryPredictionV1.SCHEMA_VERSION,
        "family": prediction.family,
        "model_artifact_sha256": prediction.model_artifact_sha256,
        "authorization_sha256": prediction.authorization_sha256,
        "dataset_sha256": prediction.dataset_sha256,
        "input_sha256": prediction.input_sha256,
        "prediction_values": calibrated,
    }
    return EntryPredictionV1(**semantic, prediction_sha256=_canonical_sha256(semantic))


def seal_entry_composer_bundle(
    *, model_bundle: EntryModelBundleV1,
    calibration_bundle: EntryCalibrationBundleV1, payload: Any,
) -> EntryComposerBundleV1:
    model = validate_entry_model_bundle(model_bundle)
    calibration = validate_entry_calibration_bundle(
        calibration_bundle, model_bundle=model
    )
    if type(payload) is not dict:
        raise TypeError("entry composer payload must be an object")
    values = {
        "schema_version": EntryComposerBundleV1.SCHEMA_VERSION,
        "model_artifact_sha256": model.artifact_sha256,
        "calibration_artifact_sha256": calibration.artifact_sha256,
        "outer_fold": model.outer_fold,
        "inner_fold": model.inner_fold,
        "composer_spec_sha256": prereg.stable_hash(prereg.entry_composer_spec()),
        "payload": _clone_jsonish(payload),
    }
    result = _seal_dataclass(EntryComposerBundleV1, values)
    validate_entry_composer_bundle(
        result, model_bundle=model, calibration_bundle=calibration
    )
    return result


def validate_entry_composer_bundle(
    bundle: Any, /, *, model_bundle: EntryModelBundleV1,
    calibration_bundle: EntryCalibrationBundleV1,
) -> EntryComposerBundleV1:
    model = validate_entry_model_bundle(model_bundle)
    calibration = validate_entry_calibration_bundle(
        calibration_bundle, model_bundle=model
    )
    bundle = _coerce_exact_dataclass(bundle, EntryComposerBundleV1)
    if (
        bundle.schema_version != bundle.SCHEMA_VERSION
        or bundle.model_artifact_sha256 != model.artifact_sha256
        or bundle.calibration_artifact_sha256 != calibration.artifact_sha256
        or bundle.outer_fold != model.outer_fold
        or bundle.inner_fold != model.inner_fold
        or bundle.composer_spec_sha256
        != prereg.stable_hash(prereg.entry_composer_spec())
        or type(bundle.payload) is not dict
    ):
        raise ValueError("entry composer bundle drift")
    _validate_artifact_hash(bundle)
    return bundle


_HGB_BASELINE_BY_AUTHORIZATION_ID: dict[int, tuple[prereg.FrozenFitAuthorization, str]] = {}


assert_fit_authorization_current = prereg.assert_fit_authorization_current


def load_authorized_entry_dataset(authorization: prereg.FrozenFitAuthorization) -> Any:
    from v4.research.pathd_entry_dataset import load_authorized_entry_dataset as loader

    return loader(authorization)


@dataclass(frozen=True)
class _EntryFitFrame:
    example: Any
    inputs: EntryModelInputV1
    targets: np.ndarray
    target_validity: np.ndarray


_HGB_SEEDS = (101, 102, 103)
_NEURAL_SEEDS = (211, 212, 213)
_ENTRY_HEAD_COUNT = len(prereg.ENTRY_HEAD_TARGETS)
_ENTRY_ACTION_COUNT = 42
_ENTRY_HISTORY_MINUTES = 90
_ENTRY_FEATURE_COUNT = 17
_ENTRY_HGB_MAX_EXAMPLES = 250_000
_HGB_PREDICTOR_CACHE: dict[int, tuple[Any, tuple[TreePredictor, ...]]] = {}
_NEURAL_MODULE_CACHE: dict[int, tuple[Any, nn.Module]] = {}
_ENTRY_DATASET_PREDICTION_CACHE: dict[
    tuple[str, str], dict[str, np.ndarray]
] = {}


def _member(value: Any, name: str, default: Any = None) -> Any:
    if type(value) is dict:
        return value.get(name, default)
    return getattr(value, name, default)


def _legacy_target_action_index(example: Any, inputs: EntryModelInputV1) -> int:
    audit = _member(example, "audit")
    identity = _member(audit, "arrival_contract_id")
    if identity is None:
        identity = _member(_member(audit, "future_audit", {}), "arrival_contract_id")
    matches = np.flatnonzero(inputs.contract_ids == identity).tolist()
    if len(matches) != 1:
        raise ValueError("legacy scalar entry target lacks one arrival action identity")
    return int(matches[0])


def _target_matrices(example: Any, inputs: EntryModelInputV1) -> tuple[np.ndarray, np.ndarray]:
    targets = _member(example, "targets")
    validity = _member(example, "target_validity")
    if type(targets) is not dict or type(validity) is not dict:
        raise TypeError("entry target bundle must be an exact object")
    values = np.zeros((_ENTRY_ACTION_COUNT, _ENTRY_HEAD_COUNT), dtype=np.float64)
    masks = np.zeros((_ENTRY_ACTION_COUNT, _ENTRY_HEAD_COUNT), dtype=np.bool_)
    present = [name for name in prereg.ENTRY_HEAD_TARGETS if name in targets]
    if not present or any(name not in validity for name in present):
        raise ValueError("entry target/validity head keys drift")
    sequence_mode = type(targets[present[0]]) in (tuple, list)
    if sequence_mode:
        for column, name in enumerate(prereg.ENTRY_HEAD_TARGETS):
            if name not in targets or name not in validity:
                continue
            target = targets[name]
            valid = validity[name]
            if (
                type(target) not in (tuple, list)
                or type(valid) not in (tuple, list)
                or len(target) != _ENTRY_ACTION_COUNT
                or len(valid) != _ENTRY_ACTION_COUNT
                or any(type(flag) is not bool for flag in valid)
            ):
                raise ValueError("all-action entry target vector drift")
            numeric = np.asarray(target, dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError("all-action entry target is nonfinite")
            values[:, column] = numeric
            masks[:, column] = np.asarray(valid, dtype=np.bool_)
    else:
        # Backward compatibility for the original synthetic one-action fixtures.
        action = _legacy_target_action_index(example, inputs)
        for column, name in enumerate(prereg.ENTRY_HEAD_TARGETS):
            if name not in targets or name not in validity:
                continue
            target = targets[name]
            valid = validity[name]
            if type(target) not in (int, float) or type(valid) is not bool:
                raise ValueError("mixed scalar/vector entry target representation")
            numeric = float(target)
            if not math.isfinite(numeric):
                raise ValueError("legacy entry target is nonfinite")
            values[action, column] = numeric
            masks[action, column] = valid
    masks &= inputs.physical_action_mask[:, None]
    return values, masks


def _shifted_history(history: Any) -> Any:
    values = np.asarray(_member(history, "values"), dtype=np.float64)
    present = np.asarray(_member(history, "contract_present"), dtype=np.bool_)
    available = np.asarray(_member(history, "minute_available"), dtype=np.bool_)
    if (
        values.shape != (_ENTRY_HISTORY_MINUTES, _ENTRY_ACTION_COUNT, _ENTRY_FEATURE_COUNT)
        or present.shape != (_ENTRY_HISTORY_MINUTES, _ENTRY_ACTION_COUNT)
        or available.shape != (_ENTRY_HISTORY_MINUTES,)
    ):
        raise ValueError("entry history shape drift")
    shifted_values = np.full_like(values, np.nan)
    shifted_present = np.zeros_like(present)
    shifted_available = np.zeros_like(available)
    shifted_values[30:] = values[:60]
    shifted_present[30:] = present[:60]
    shifted_available[30:] = available[:60]
    shifted_finite = (
        np.isfinite(shifted_values)
        & shifted_present[..., None]
        & shifted_available[:, None, None]
    )
    history_type = type(history)
    try:
        return history_type(
            schema_version=_member(history_type, "SCHEMA_VERSION", _member(history, "schema_version")),
            values=shifted_values,
            finite=shifted_finite,
            contract_present=shifted_present,
            minute_available=shifted_available,
            current_contract_ids=_member(history, "current_contract_ids"),
            decision_time_ns=_member(history, "decision_time_ns"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("cannot construct the registered 30-minute shifted history") from exc


def _shift_model_input(inputs: EntryModelInputV1) -> EntryModelInputV1:
    history = _shifted_history(inputs.signed17_history)
    return EntryModelInputV1(
        schema_version=EntryModelInputV1.SCHEMA_VERSION,
        session=inputs.session,
        decision_time_ns=inputs.decision_time_ns,
        signed17_frame=inputs.signed17_frame,
        signed17_history=history,
        hgb_summaries=hgb_signed17_summaries(
            history,
            current_offsets=inputs.current_offsets,
            current_rights=inputs.current_rights,
        ),
        current_offsets=inputs.current_offsets,
        current_rights=inputs.current_rights,
        contract_ids=inputs.contract_ids,
        physical_action_mask=inputs.physical_action_mask,
    )


def _prepare_fit_frames(
    dataset: Any, /, *, time_shifted: bool = False,
    target_override: Mapping[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[_EntryFitFrame, ...]:
    examples = _member(dataset, "examples")
    if type(examples) not in (tuple, list) or not examples:
        raise ValueError("entry fit dataset has no examples")
    prepared: list[_EntryFitFrame] = []
    frame_keys: list[tuple[str, int]] = []
    for example in examples:
        inputs = model_input_from_example(example)
        key = (inputs.session, inputs.decision_time_ns)
        frame_keys.append(key)
        targets, validity = _target_matrices(example, inputs)
        if target_override is not None:
            for action, contract_id in enumerate(inputs.contract_ids.tolist()):
                override = target_override.get((inputs.session, inputs.decision_time_ns, contract_id))
                if override is None:
                    raise ValueError("shuffled target population alignment drift")
                targets[action] = override[0]
                validity[action] = override[1]
            validity &= inputs.physical_action_mask[:, None]
        if time_shifted:
            inputs = _shift_model_input(inputs)
        prepared.append(
            _EntryFitFrame(
                example=example,
                inputs=inputs,
                targets=targets,
                target_validity=validity,
            )
        )
    if len(frame_keys) != len(set(frame_keys)):
        raise ValueError("duplicate entry decision frame")
    order = sorted(range(len(prepared)), key=lambda index: frame_keys[index])
    return tuple(prepared[index] for index in order)


def _target_scalers(frames: Sequence[_EntryFitFrame]) -> tuple[dict[str, float], ...]:
    scalers: list[dict[str, float]] = []
    for column, head in enumerate(prereg.ENTRY_HEAD_TARGETS):
        counts: dict[str, int] = {}
        for frame in frames:
            count = int(frame.target_validity[:, column].sum())
            counts[frame.inputs.session] = counts.get(frame.inputs.session, 0) + count
        sessions = tuple(session for session in sorted(counts) if counts[session] > 0)
        if not sessions:
            raise ValueError(f"entry target population invalid for {head}")
        weights = {
            session: 1.0 / (len(sessions) * counts[session]) for session in sessions
        }
        mean = math.fsum(
            float(frame.targets[action, column]) * weights[frame.inputs.session]
            for frame in frames
            if frame.inputs.session in weights
            for action in np.flatnonzero(frame.target_validity[:, column]).tolist()
        )
        variance = math.fsum(
            weights[frame.inputs.session]
            * (float(frame.targets[action, column]) - mean) ** 2
            for frame in frames
            if frame.inputs.session in weights
            for action in np.flatnonzero(frame.target_validity[:, column]).tolist()
        )
        standard_deviation = math.sqrt(variance) if variance >= 0.0 else float("nan")
        if (
            not math.isfinite(mean)
            or not math.isfinite(standard_deviation)
            or standard_deviation <= 0.0
        ):
            raise ValueError(f"entry target scaler invalid for {head}")
        scalers.append(
            {
                "head": head,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "valid_row_count": sum(counts[session] for session in sessions),
                "contributing_session_count": len(sessions),
            }
        )
    return tuple(scalers)


def _hgb_identity(
    frame: _EntryFitFrame, action: int, head: str
) -> dict[str, Any]:
    return {
        "session": frame.inputs.session,
        "decision_time_ns": frame.inputs.decision_time_ns,
        "source_neutral_contract_id": str(frame.inputs.contract_ids[action]),
        "target_axis": head,
    }


def _sample_hgb_rows(
    frames: Sequence[_EntryFitFrame], /, *, column: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    head = prereg.ENTRY_HEAD_TARGETS[column]
    population_counts: dict[str, int] = {}
    population_count = 0
    population_hasher = hashlib.sha256()
    population_hasher.update(b"[")
    first = True
    for frame in frames:
        actions = np.flatnonzero(frame.target_validity[:, column]).tolist()
        population_counts[frame.inputs.session] = (
            population_counts.get(frame.inputs.session, 0) + len(actions)
        )
        for action in actions:
            identity = _hgb_identity(frame, action, head)
            if not first:
                population_hasher.update(b",")
            population_hasher.update(
                json.dumps(
                    identity, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            )
            first = False
            population_count += 1
    population_hasher.update(b"]")
    sessions = tuple(
        session for session in sorted(population_counts) if population_counts[session] > 0
    )
    if population_count == 0 or not sessions:
        raise ValueError(f"HGB population empty for {head}")
    sample_count = min(_ENTRY_HGB_MAX_EXAMPLES, population_count)
    allocations = {session: 0 for session in sessions}
    allocated = 0
    while allocated < sample_count:
        progressed = False
        for session in sessions:
            if allocations[session] >= population_counts[session]:
                continue
            allocations[session] += 1
            allocated += 1
            progressed = True
            if allocated == sample_count:
                break
        if not progressed:
            raise AssertionError("HGB round-robin allocation stalled")
    rank_dtype = np.dtype(
        [("digest", "S32"), ("frame_index", "<i4"), ("action_index", "u1")]
    )
    ranked_arrays = {
        session: np.empty(population_counts[session], dtype=rank_dtype)
        for session in sessions
    }
    write_offsets = {session: 0 for session in sessions}
    for frame_index, frame in enumerate(frames):
        ranked_array = ranked_arrays[frame.inputs.session]
        for action in np.flatnonzero(frame.target_validity[:, column]).tolist():
            identity = _hgb_identity(frame, action, head)
            key_object = {
                "seed": seed,
                "session": identity["session"],
                "decision_time_ns": identity["decision_time_ns"],
                "source_neutral_contract_id": identity[
                    "source_neutral_contract_id"
                ],
                "target_axis": head,
            }
            digest = hashlib.sha256(
                json.dumps(
                    key_object, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).digest()
            destination = write_offsets[frame.inputs.session]
            ranked_array[destination] = (digest, frame_index, action)
            write_offsets[frame.inputs.session] = destination + 1
    ranked: dict[str, list[tuple[int, int]]] = {}
    for session in sessions:
        array = ranked_arrays[session]
        if write_offsets[session] != len(array):
            raise AssertionError("HGB compact rank population write drift")
        order = np.argsort(array["digest"], kind="stable")
        needed = allocations[session]
        chosen: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(order) and len(chosen) < needed:
            end = cursor + 1
            digest = array["digest"][order[cursor]]
            while end < len(order) and array["digest"][order[end]] == digest:
                end += 1
            tied = [
                (
                    json.dumps(
                        _hgb_identity(
                            frames[int(array["frame_index"][index])],
                            int(array["action_index"][index]),
                            head,
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    int(array["frame_index"][index]),
                    int(array["action_index"][index]),
                )
                for index in order[cursor:end]
            ]
            for _identity, selected_frame, selected_action in sorted(tied):
                if len(chosen) == needed:
                    break
                chosen.append((selected_frame, selected_action))
            cursor = end
        ranked[session] = chosen
    cursors = {session: 0 for session in sessions}
    selected_candidates: list[tuple[int, int]] = []
    while len(selected_candidates) < sample_count:
        for session in sessions:
            cursor = cursors[session]
            if cursor >= len(ranked[session]):
                continue
            selected_candidates.append(ranked[session][cursor])
            cursors[session] += 1
            if len(selected_candidates) == sample_count:
                break
    selected = [
        {
            "identity": _hgb_identity(
                frames[frame_index], action_index, head
            ),
            "frame_index": frame_index,
            "action_index": action_index,
            "target": float(frames[frame_index].targets[action_index, column]),
        }
        for frame_index, action_index in selected_candidates
    ]
    counts = {session: allocations[session] for session in sessions if allocations[session]}
    contributing = len(counts)
    weights = np.asarray(
        [
            sample_count
            * (1.0 / (contributing * counts[row["identity"]["session"]]))
            for row in selected
        ],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or not math.isclose(
        float(weights.mean()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("HGB sample-weight normalization drift")
    receipt_semantic = {
        "seed": seed,
        "head": head,
        "population_sha256": population_hasher.hexdigest(),
        "ordered_selected_identities_sha256": _canonical_sha256(
            [row["identity"] for row in selected]
        ),
        "per_session_counts": counts,
        "population_row_count": population_count,
        "selected_row_count": sample_count,
        "maximum_examples": _ENTRY_HGB_MAX_EXAMPLES,
    }
    return selected, {
        **receipt_semantic,
        "receipt_sha256": _canonical_sha256(receipt_semantic),
    }, weights


def _serialize_hgb_regressor(estimator: HistGradientBoostingRegressor) -> dict[str, Any]:
    baseline = np.asarray(estimator._baseline_prediction, dtype=np.float64).reshape(-1)
    stages: list[list[dict[str, Any]]] = []
    for stage in estimator._predictors:
        trees: list[dict[str, Any]] = []
        for predictor in stage:
            nodes = predictor.nodes
            names = tuple(nodes.dtype.names or ())
            required = {
                "value", "feature_idx", "num_threshold", "missing_go_to_left",
                "left", "right", "is_leaf", "is_categorical",
            }
            if not required.issubset(names) or bool(nodes["is_categorical"].any()):
                raise ValueError("unsupported HGB tree state")
            trees.append(
                {
                    "node_fields": list(names),
                    "node_dtype": [
                        [name, nodes.dtype.fields[name][0].str] for name in names
                    ],
                    "nodes": [
                        [
                            (
                                float(node[name])
                                if nodes.dtype.fields[name][0].kind == "f"
                                else int(node[name])
                            )
                            for name in names
                        ]
                        for node in nodes
                    ],
                }
            )
        stages.append(trees)
    return {
        "format": "pathd.sklearn_hgb_tree_state.v1",
        "baseline_prediction": baseline.tolist(),
        "n_features_in": int(estimator.n_features_in_),
        "stages": stages,
    }


def _predict_hgb_state(state: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    if type(state) is not dict or state.get("format") != "pathd.sklearn_hgb_tree_state.v1":
        raise ValueError("HGB serialized-state format drift")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != state.get("n_features_in"):
        raise ValueError("HGB serialized-state feature drift")
    baseline = state.get("baseline_prediction")
    stages = state.get("stages")
    if type(baseline) is not list or len(baseline) != 1 or type(stages) is not list:
        raise ValueError("HGB serialized-state structure drift")
    cache_key = id(state)
    cached = _HGB_PREDICTOR_CACHE.get(cache_key)
    if cached is not None and cached[0] is state:
        predictors = cached[1]
    else:
        rebuilt: list[TreePredictor] = []
        for stage in stages:
            if type(stage) is not list or len(stage) != 1:
                raise ValueError("HGB regression tree multiplicity drift")
            tree = stage[0]
            names = tree.get("node_fields") if type(tree) is dict else None
            dtype_rows = tree.get("node_dtype") if type(tree) is dict else None
            rows = tree.get("nodes") if type(tree) is dict else None
            if (
                type(names) is not list
                or type(dtype_rows) is not list
                or type(rows) is not list
                or not rows
                or len(dtype_rows) != len(names)
            ):
                raise ValueError("HGB serialized tree drift")
            dtype = np.dtype(
                [
                    (item[0], item[1])
                    for item in dtype_rows
                    if type(item) is list and len(item) == 2
                ]
            )
            if tuple(dtype.names or ()) != tuple(names):
                raise ValueError("HGB serialized node dtype drift")
            nodes = np.asarray([tuple(row) for row in rows], dtype=dtype)
            if bool(nodes["is_categorical"].any()):
                raise ValueError("categorical HGB node is forbidden")
            empty_bitsets = np.zeros((0, 8), dtype=np.uint32)
            rebuilt.append(TreePredictor(nodes, empty_bitsets, empty_bitsets))
        predictors = tuple(rebuilt)
        _HGB_PREDICTOR_CACHE[cache_key] = (state, predictors)
    output = np.full(x.shape[0], float(baseline[0]), dtype=np.float64)
    known_categories = np.zeros((0, 8), dtype=np.uint32)
    feature_map = np.zeros(x.shape[1], dtype=np.uint32)
    for predictor in predictors:
        output += predictor.predict(
            x,
            known_cat_bitsets=known_categories,
            f_idx_map=feature_map,
            n_threads=1,
        )
    if not np.isfinite(output).all():
        raise ValueError("HGB serialized prediction nonfinite")
    return output


def _fit_hgb_payload(
    frames: Sequence[_EntryFitFrame], /, *, input_transform: str = "NONE"
) -> dict[str, Any]:
    scalers = _target_scalers(frames)
    fits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    receipts: list[dict[str, Any]] = []
    fit_count = 0
    for seed in _HGB_SEEDS:
        seed_fits = {"mean": [], "q10": []}
        for column, head in enumerate(prereg.ENTRY_HEAD_TARGETS):
            selected, receipt, weights = _sample_hgb_rows(
                frames, column=column, seed=seed
            )
            scaler = scalers[column]
            x = np.asarray(
                [
                    frames[row["frame_index"]].inputs.hgb_summaries[
                        row["action_index"]
                    ]
                    for row in selected
                ],
                dtype=np.float64,
            )
            y = np.asarray(
                [
                    (row["target"] - scaler["mean"])
                    / scaler["standard_deviation"]
                    for row in selected
                ],
                dtype=np.float32,
            )
            if not np.isfinite(y).all():
                raise ValueError(f"HGB transformed target nonfinite for {head}")
            for branch, loss, quantile in (
                ("mean", "squared_error", None),
                ("q10", "quantile", 0.1),
            ):
                estimator = build_hgb_regressor(
                    seed=seed, loss=loss, quantile=quantile
                )
                estimator.fit(x, y, sample_weight=weights)
                seed_fits[branch].append(_serialize_hgb_regressor(estimator))
                fit_count += 1
            receipts.append(receipt)
        fits[str(seed)] = seed_fits
    if fit_count != 120:
        raise AssertionError(f"HGB exact fit budget drift: {fit_count} != 120")
    return {
        "format": "pathd.entry_hgb_fitted.v1",
        "input_transform": input_transform,
        "head_targets": list(prereg.ENTRY_HEAD_TARGETS),
        "target_scalers": list(scalers),
        "ensemble_seeds": list(_HGB_SEEDS),
        "ensemble_aggregation": "MEDIAN_PER_HEAD",
        "fits": fits,
        "fit_count": fit_count,
        "fits_per_weight_bundle": 120,
        "sampling_receipts": receipts,
    }


def _feature_scalers(frames: Sequence[_EntryFitFrame]) -> tuple[dict[str, float], ...]:
    session_counts: dict[str, np.ndarray] = {}
    session_sum_parts: dict[str, list[np.ndarray]] = {}
    for frame in frames:
        history = frame.inputs.signed17_history
        values = np.asarray(_member(history, "values"), dtype=np.float64)
        finite = np.asarray(_member(history, "finite"), dtype=np.bool_)
        if (
            values.shape
            != (_ENTRY_HISTORY_MINUTES, _ENTRY_ACTION_COUNT, _ENTRY_FEATURE_COUNT)
            or finite.shape != values.shape
        ):
            raise ValueError("neural history shape drift")
        counts = finite.sum(axis=(0, 1), dtype=np.int64)
        safe = np.where(finite, values, 0.0)
        sums = safe.sum(axis=(0, 1), dtype=np.float64)
        session_counts.setdefault(
            frame.inputs.session, np.zeros(_ENTRY_FEATURE_COUNT, dtype=np.int64)
        )[:] += counts
        session_sum_parts.setdefault(frame.inputs.session, []).append(sums)
    ordered_sessions = tuple(sorted(session_counts))
    contributing_counts = np.asarray(
        [
            sum(int(session_counts[session][feature]) > 0 for session in ordered_sessions)
            for feature in range(_ENTRY_FEATURE_COUNT)
        ],
        dtype=np.int64,
    )
    if bool((contributing_counts == 0).any()):
        raise ValueError("neural feature has no finite population")
    session_sums = {
        session: np.asarray(
            [
                math.fsum(float(part[feature]) for part in session_sum_parts[session])
                for feature in range(_ENTRY_FEATURE_COUNT)
            ],
            dtype=np.float64,
        )
        for session in ordered_sessions
    }
    means = np.asarray(
        [
            math.fsum(
                float(session_sums[session][feature])
                / (
                    int(contributing_counts[feature])
                    * int(session_counts[session][feature])
                )
                for session in ordered_sessions
                if session_counts[session][feature] > 0
            )
            for feature in range(_ENTRY_FEATURE_COUNT)
        ],
        dtype=np.float64,
    )
    variance_parts: list[np.ndarray] = []
    for frame in frames:
        history = frame.inputs.signed17_history
        values = np.asarray(_member(history, "values"), dtype=np.float64)
        finite = np.asarray(_member(history, "finite"), dtype=np.bool_)
        weights = np.asarray(
            [
                1.0
                / (
                    int(contributing_counts[feature])
                    * int(session_counts[frame.inputs.session][feature])
                )
                if session_counts[frame.inputs.session][feature] > 0
                else 0.0
                for feature in range(_ENTRY_FEATURE_COUNT)
            ],
            dtype=np.float64,
        )
        squared = np.where(finite, (values - means) ** 2, 0.0)
        variance_parts.append(
            squared.sum(axis=(0, 1), dtype=np.float64) * weights
        )
    variances = np.asarray(
        [
            math.fsum(float(part[feature]) for part in variance_parts)
            for feature in range(_ENTRY_FEATURE_COUNT)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(means).all() or not np.isfinite(variances).all() or bool(
        (variances < 0.0).any()
    ):
        raise ValueError("neural feature scaler nonfinite")
    results: list[dict[str, float]] = []
    for feature in range(_ENTRY_FEATURE_COUNT):
        standard_deviation = math.sqrt(float(variances[feature]))
        if standard_deviation == 0.0:
            standard_deviation = 1.0
        results.append(
            {
                "feature_index": feature,
                "mean": float(means[feature]),
                "standard_deviation": standard_deviation,
                "finite_token_count": sum(
                    int(session_counts[session][feature])
                    for session in ordered_sessions
                ),
                "contributing_session_count": int(contributing_counts[feature]),
            }
        )
    return tuple(results)


def _neural_inputs(
    frames: Sequence[_EntryFitFrame], feature_scalers: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_tokens: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
    current_masks: list[np.ndarray] = []
    means = np.asarray([row["mean"] for row in feature_scalers], dtype=np.float64)
    scales = np.asarray(
        [row["standard_deviation"] for row in feature_scalers], dtype=np.float64
    )
    for frame in frames:
        history = frame.inputs.signed17_history
        values = np.asarray(_member(history, "values"), dtype=np.float64)
        finite = np.asarray(_member(history, "finite"), dtype=np.bool_)
        present = np.asarray(_member(history, "contract_present"), dtype=np.bool_)
        available = np.asarray(_member(history, "minute_available"), dtype=np.bool_)
        if (
            values.shape != (_ENTRY_HISTORY_MINUTES, _ENTRY_ACTION_COUNT, _ENTRY_FEATURE_COUNT)
            or finite.shape != values.shape
            or present.shape != (_ENTRY_HISTORY_MINUTES, _ENTRY_ACTION_COUNT)
            or available.shape != (_ENTRY_HISTORY_MINUTES,)
        ):
            raise ValueError("neural input history drift")
        expected = np.isfinite(values) & present[..., None] & available[:, None, None]
        if not np.array_equal(finite, expected):
            raise ValueError("neural finite mask drift")
        standardized = np.zeros_like(values, dtype=np.float64)
        np.subtract(values, means, out=standardized, where=finite)
        np.divide(standardized, scales, out=standardized, where=finite)
        standardized[~finite] = 0.0
        tokens = np.concatenate(
            (standardized.astype(np.float32), finite.astype(np.float32)), axis=2
        )
        geometry = np.column_stack(
            (
                np.asarray(frame.inputs.current_offsets, dtype=np.float64) / 50.0,
                np.asarray(frame.inputs.current_rights, dtype=object) == "C",
            )
        ).astype(np.float32)
        current = present[-1].astype(np.bool_)
        if not current.any():
            raise ValueError("neural frame lacks a current exact action")
        history_tokens.append(tokens)
        geometries.append(geometry)
        current_masks.append(current)
    return (
        np.asarray(history_tokens, dtype=np.float32),
        np.asarray(geometries, dtype=np.float32),
        np.asarray(current_masks, dtype=np.bool_),
    )


def _neural_targets(
    frames: Sequence[_EntryFitFrame], target_scalers: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray([frame.targets for frame in frames], dtype=np.float64)
    valid = np.asarray([frame.target_validity for frame in frames], dtype=np.bool_)
    means = np.asarray([row["mean"] for row in target_scalers], dtype=np.float64)
    scales = np.asarray(
        [row["standard_deviation"] for row in target_scalers], dtype=np.float64
    )
    transformed = np.zeros_like(raw, dtype=np.float64)
    np.subtract(raw, means, out=transformed, where=valid)
    np.divide(transformed, scales, out=transformed, where=valid)
    transformed[~valid] = 0.0
    if not np.isfinite(transformed).all():
        raise ValueError("neural transformed target nonfinite")
    weights = np.zeros_like(raw, dtype=np.float64)
    denominators = np.zeros(_ENTRY_HEAD_COUNT, dtype=np.float64)
    for column in range(_ENTRY_HEAD_COUNT):
        sessions = tuple(
            dict.fromkeys(
                frame.inputs.session
                for frame_index, frame in enumerate(frames)
                if bool(valid[frame_index, :, column].any())
            )
        )
        if not sessions:
            raise ValueError("neural mandatory head has no valid rows")
        counts = {
            session: sum(
                int(valid[index, :, column].sum())
                for index, frame in enumerate(frames)
                if frame.inputs.session == session
            )
            for session in sessions
        }
        for index, frame in enumerate(frames):
            if frame.inputs.session not in counts:
                continue
            weights[index, valid[index, :, column], column] = 1.0 / (
                len(sessions) * counts[frame.inputs.session]
            )
        denominators[column] = math.fsum(
            float(weights[index, action, column])
            for index in range(len(frames))
            for action in range(_ENTRY_ACTION_COUNT)
            if valid[index, action, column]
        )
        if not math.isfinite(denominators[column]) or denominators[column] <= 0.0:
            raise ValueError("neural fixed head denominator drift")
    return (
        transformed.astype(np.float32),
        valid,
        weights.astype(np.float32),
        denominators.astype(np.float32),
    )


def _configure_neural_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError("neural interop-thread determinism cannot be established") from exc
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode(2)


def _serialize_neural_module(module: nn.Module) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        if value.dtype != torch.float32:
            raise ValueError("entry neural state dtype drift")
        state[name] = {
            "dtype": "float32",
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    return state


def _load_neural_module(seed: int, state: Mapping[str, Any]) -> nn.Module:
    if type(state) is not dict:
        raise ValueError("entry neural serialized state drift")
    cache_key = id(state)
    cached = _NEURAL_MODULE_CACHE.get(cache_key)
    if cached is not None and cached[0] is state:
        return cached[1]
    module = build_entry_neural_module(seed=seed)
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError("entry neural state key drift")
    loaded: dict[str, torch.Tensor] = {}
    for name, template in expected.items():
        item = state[name]
        if (
            type(item) is not dict
            or item.get("dtype") != "float32"
            or item.get("shape") != list(template.shape)
        ):
            raise ValueError("entry neural tensor metadata drift")
        tensor = torch.tensor(item.get("data"), dtype=torch.float32, device="cpu")
        if tuple(tensor.shape) != tuple(template.shape) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("entry neural tensor data drift")
        loaded[name] = tensor
    module.load_state_dict(loaded, strict=True)
    module.eval()
    _NEURAL_MODULE_CACHE[cache_key] = (state, module)
    return module


def _fit_neural_payload(
    frames: Sequence[_EntryFitFrame], /, *, input_transform: str = "NONE"
) -> dict[str, Any]:
    target_scalers = _target_scalers(frames)
    feature_scalers = _feature_scalers(frames)
    targets, valid, weights, denominators = _neural_targets(frames, target_scalers)
    modules: dict[str, Any] = {}
    training_receipts: list[dict[str, Any]] = []
    count = len(frames)
    for seed in _NEURAL_SEEDS:
        _configure_neural_determinism(seed)
        module = build_entry_neural_module(seed=seed).cpu()
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0001,
            amsgrad=False,
            maximize=False,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        module.train()
        epoch_losses: list[float] = []
        for epoch in range(20):
            module.train()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + epoch)
            permutation = torch.randperm(count, generator=generator, device="cpu")
            batch_losses: list[float] = []
            for offset in range(0, count, 64):
                indexes = permutation[offset : offset + 64].numpy()
                batch_frames = tuple(frames[int(index)] for index in indexes)
                batch_history, batch_geometry, batch_current = _neural_inputs(
                    batch_frames, feature_scalers
                )
                history_batch = torch.from_numpy(batch_history)
                geometry_batch = torch.from_numpy(batch_geometry)
                current_batch = torch.from_numpy(batch_current)
                target_batch = torch.from_numpy(targets[indexes])
                valid_batch = torch.from_numpy(valid[indexes])
                weight_batch = torch.from_numpy(weights[indexes])
                denominator_tensor = torch.from_numpy(denominators)
                optimizer.zero_grad(set_to_none=True)
                prediction = module(history_batch, geometry_batch, current_batch)
                mean_prediction = prediction[:, :, :_ENTRY_HEAD_COUNT]
                q10_prediction = prediction[:, :, _ENTRY_HEAD_COUNT:]
                mean_error = torch.where(
                    valid_batch,
                    (mean_prediction - target_batch) ** 2,
                    torch.zeros_like(mean_prediction),
                )
                difference = target_batch - q10_prediction
                q10_error = torch.where(
                    valid_batch,
                    torch.maximum(0.1 * difference, -0.9 * difference),
                    torch.zeros_like(q10_prediction),
                )
                mean_heads = (mean_error * weight_batch).sum(dim=(0, 1)) / denominator_tensor
                q10_heads = (q10_error * weight_batch).sum(dim=(0, 1)) / denominator_tensor
                loss = 0.5 * mean_heads.mean() + 0.5 * q10_heads.mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("entry neural training loss nonfinite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(), max_norm=1.0, norm_type=2.0,
                    error_if_nonfinite=True, foreach=False,
                )
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            epoch_losses.append(math.fsum(batch_losses))
        module.eval()
        modules[str(seed)] = _serialize_neural_module(module)
        training_receipts.append(
            {
                "seed": seed,
                "epoch_count": 20,
                "batch_size_decision_frames": 64,
                "frame_count": count,
                "epoch_loss_sums": epoch_losses,
                "state_sha256": _canonical_sha256(modules[str(seed)]),
            }
        )
    return {
        "format": "pathd.entry_neural_fitted.v1",
        "input_transform": input_transform,
        "head_targets": list(prereg.ENTRY_HEAD_TARGETS),
        "target_scalers": list(target_scalers),
        "feature_scalers": list(feature_scalers),
        "ensemble_seeds": list(_NEURAL_SEEDS),
        "ensemble_aggregation": "MEDIAN_PER_HEAD",
        "module_parameter_count": sum(
            parameter.numel() for parameter in build_entry_neural_module(seed=0).parameters()
        ),
        "epochs": 20,
        "batch_size_decision_frames": 64,
        "modules": modules,
        "training_receipts": training_receipts,
    }


def _predict_fitted_payload(
    family: str, payload: Mapping[str, Any], inputs: EntryModelInputV1
) -> np.ndarray:
    transform = payload.get("input_transform", "NONE")
    effective = _shift_model_input(inputs) if transform == "TIME_SHIFT_30" else inputs
    if transform not in {"NONE", "TIME_SHIFT_30"}:
        raise ValueError("entry fitted input transform drift")
    scalers = payload.get("target_scalers")
    if type(scalers) is not list or len(scalers) != _ENTRY_HEAD_COUNT:
        raise ValueError("entry fitted target-scaler state drift")
    means = np.asarray([row["mean"] for row in scalers], dtype=np.float64)
    scales = np.asarray(
        [row["standard_deviation"] for row in scalers], dtype=np.float64
    )
    if family == "HGB":
        if payload.get("format") != "pathd.entry_hgb_fitted.v1":
            raise ValueError("entry HGB fitted payload drift")
        fits = payload.get("fits")
        seed_predictions: list[np.ndarray] = []
        for seed in _HGB_SEEDS:
            state = fits.get(str(seed)) if type(fits) is dict else None
            if type(state) is not dict:
                raise ValueError("entry HGB ensemble state drift")
            columns: list[np.ndarray] = []
            for branch in ("mean", "q10"):
                branch_states = state.get(branch)
                if type(branch_states) is not list or len(branch_states) != _ENTRY_HEAD_COUNT:
                    raise ValueError("entry HGB head state drift")
                for column, estimator_state in enumerate(branch_states):
                    standardized = _predict_hgb_state(
                        estimator_state,
                        np.asarray(effective.hgb_summaries, dtype=np.float64),
                    )
                    columns.append(standardized * scales[column] + means[column])
            seed_predictions.append(np.column_stack(columns))
        result = np.median(np.stack(seed_predictions, axis=0), axis=0)
    elif family == "NEURAL":
        if payload.get("format") != "pathd.entry_neural_fitted.v1":
            raise ValueError("entry neural fitted payload drift")
        feature_scalers = payload.get("feature_scalers")
        if type(feature_scalers) is not list or len(feature_scalers) != _ENTRY_FEATURE_COUNT:
            raise ValueError("entry neural feature-scaler state drift")
        temporary = _EntryFitFrame(
            example=None,
            inputs=effective,
            targets=np.zeros((_ENTRY_ACTION_COUNT, _ENTRY_HEAD_COUNT)),
            target_validity=np.zeros((_ENTRY_ACTION_COUNT, _ENTRY_HEAD_COUNT), dtype=np.bool_),
        )
        history, geometry, current = _neural_inputs((temporary,), feature_scalers)
        modules = payload.get("modules")
        seed_predictions = []
        for seed in _NEURAL_SEEDS:
            state = modules.get(str(seed)) if type(modules) is dict else None
            module = _load_neural_module(seed, state)
            with torch.inference_mode():
                standardized = module(
                    torch.from_numpy(history),
                    torch.from_numpy(geometry),
                    torch.from_numpy(current),
                )[0].detach().cpu().numpy().astype(np.float64)
            original = standardized.copy()
            original[:, :_ENTRY_HEAD_COUNT] = (
                standardized[:, :_ENTRY_HEAD_COUNT] * scales + means
            )
            original[:, _ENTRY_HEAD_COUNT:] = (
                standardized[:, _ENTRY_HEAD_COUNT:] * scales + means
            )
            seed_predictions.append(original)
        result = np.median(np.stack(seed_predictions, axis=0), axis=0)
    else:
        raise ValueError("unsupported fitted entry family")
    if result.shape != (_ENTRY_ACTION_COUNT, 2 * _ENTRY_HEAD_COUNT) or not np.isfinite(result).all():
        raise ValueError("entry fitted prediction drift")
    return result


def _predict_fitted_inputs(
    family: str, payload: Mapping[str, Any], inputs: Sequence[EntryModelInputV1]
) -> tuple[np.ndarray, ...]:
    if not inputs:
        return ()
    transform = payload.get("input_transform", "NONE")
    if transform not in {"NONE", "TIME_SHIFT_30"}:
        raise ValueError("entry fitted batch transform drift")
    effective = tuple(
        _shift_model_input(item) if transform == "TIME_SHIFT_30" else item
        for item in inputs
    )
    scalers = payload.get("target_scalers")
    if type(scalers) is not list or len(scalers) != _ENTRY_HEAD_COUNT:
        raise ValueError("entry fitted batch scaler drift")
    target_means = np.asarray([row["mean"] for row in scalers], dtype=np.float64)
    target_scales = np.asarray(
        [row["standard_deviation"] for row in scalers], dtype=np.float64
    )
    chunks: list[np.ndarray] = []
    batch_size = 128 if family == "HGB" else 64
    for offset in range(0, len(effective), batch_size):
        batch = effective[offset : offset + batch_size]
        if family == "HGB":
            matrix = np.concatenate(
                [np.asarray(item.hgb_summaries, dtype=np.float64) for item in batch],
                axis=0,
            )
            fits = payload.get("fits")
            seed_predictions: list[np.ndarray] = []
            for seed in _HGB_SEEDS:
                state = fits.get(str(seed)) if type(fits) is dict else None
                if type(state) is not dict:
                    raise ValueError("entry HGB batch ensemble drift")
                columns: list[np.ndarray] = []
                for branch in ("mean", "q10"):
                    branch_states = state.get(branch)
                    if type(branch_states) is not list or len(branch_states) != _ENTRY_HEAD_COUNT:
                        raise ValueError("entry HGB batch head drift")
                    for column, estimator_state in enumerate(branch_states):
                        standardized = _predict_hgb_state(estimator_state, matrix)
                        columns.append(
                            standardized * target_scales[column] + target_means[column]
                        )
                seed_predictions.append(np.column_stack(columns))
            predicted = np.median(np.stack(seed_predictions, axis=0), axis=0)
            chunks.extend(
                predicted[index : index + _ENTRY_ACTION_COUNT]
                for index in range(0, len(predicted), _ENTRY_ACTION_COUNT)
            )
        elif family == "NEURAL":
            feature_scalers = payload.get("feature_scalers")
            if type(feature_scalers) is not list or len(feature_scalers) != _ENTRY_FEATURE_COUNT:
                raise ValueError("entry neural batch feature-scaler drift")
            temporary = tuple(
                _EntryFitFrame(
                    example=None,
                    inputs=item,
                    targets=np.zeros((_ENTRY_ACTION_COUNT, _ENTRY_HEAD_COUNT)),
                    target_validity=np.zeros(
                        (_ENTRY_ACTION_COUNT, _ENTRY_HEAD_COUNT), dtype=np.bool_
                    ),
                )
                for item in batch
            )
            history, geometry, current = _neural_inputs(temporary, feature_scalers)
            modules = payload.get("modules")
            seed_predictions = []
            for seed in _NEURAL_SEEDS:
                state = modules.get(str(seed)) if type(modules) is dict else None
                module = _load_neural_module(seed, state)
                with torch.inference_mode():
                    standardized = module(
                        torch.from_numpy(history),
                        torch.from_numpy(geometry),
                        torch.from_numpy(current),
                    ).detach().cpu().numpy().astype(np.float64)
                standardized[:, :, :_ENTRY_HEAD_COUNT] = (
                    standardized[:, :, :_ENTRY_HEAD_COUNT]
                    * target_scales[None, None, :]
                    + target_means[None, None, :]
                )
                standardized[:, :, _ENTRY_HEAD_COUNT:] = (
                    standardized[:, :, _ENTRY_HEAD_COUNT:]
                    * target_scales[None, None, :]
                    + target_means[None, None, :]
                )
                seed_predictions.append(standardized)
            chunks.extend(np.median(np.stack(seed_predictions, axis=0), axis=0))
        else:
            raise ValueError("unsupported fitted entry batch family")
    result = tuple(np.asarray(item, dtype=np.float64) for item in chunks)
    if len(result) != len(inputs) or any(
        item.shape != (_ENTRY_ACTION_COUNT, 2 * _ENTRY_HEAD_COUNT)
        or not np.isfinite(item).all()
        for item in result
    ):
        raise ValueError("entry fitted batch prediction drift")
    return result


def _predict_entry_from_dataset_cache(
    bundle: EntryModelBundleV1, inputs: EntryModelInputV1, dataset: Any
) -> EntryPredictionV1:
    dataset_hash = _member(dataset, "dataset_sha256")
    if dataset_hash != bundle.dataset_sha256 and _member(dataset, "role", "").endswith("weights"):
        raise ValueError("entry prediction cache weights-dataset drift")
    cache_key = (bundle.artifact_sha256, str(dataset_hash))
    cache = _ENTRY_DATASET_PREDICTION_CACHE.get(cache_key)
    if cache is None:
        examples = _member(dataset, "examples")
        if type(examples) not in (tuple, list) or not examples:
            raise ValueError("entry prediction cache dataset empty")
        model_inputs = tuple(model_input_from_example(example) for example in examples)
        keys = [item.canonical_sha256() for item in model_inputs]
        if len(keys) != len(set(keys)):
            raise ValueError("entry prediction cache input duplicate")
        predictions = _predict_fitted_inputs(bundle.family, bundle.payload, model_inputs)
        cache = {
            key: np.array(value, dtype=np.float64, copy=True, order="C")
            for key, value in zip(keys, predictions, strict=True)
        }
        for value in cache.values():
            value.setflags(write=False)
        _ENTRY_DATASET_PREDICTION_CACHE[cache_key] = cache
    input_hash = inputs.canonical_sha256()
    values = cache.get(input_hash)
    if values is None:
        raise ValueError("entry prediction input is absent from sealed dataset cache")
    return _seal_prediction(
        family=bundle.family,
        model_hash=bundle.artifact_sha256,
        authorization_hash=bundle.authorization_sha256,
        dataset_hash=bundle.dataset_sha256,
        inputs=inputs,
        values=values,
    )


def _fit_hgb_entry_bundle_impl(
    authorization: prereg.FrozenFitAuthorization, dataset: Any
) -> EntryModelBundleV1:
    frames = _prepare_fit_frames(dataset)
    payload = _fit_hgb_payload(frames)
    return seal_entry_model_bundle(
        family="HGB",
        authorization=authorization,
        dataset_sha256=_member(dataset, "dataset_sha256"),
        payload=payload,
    )


def _fit_neural_entry_bundle_impl(
    authorization: prereg.FrozenFitAuthorization, dataset: Any,
    hgb_bundle: EntryModelBundleV1,
) -> EntryModelBundleV1:
    baseline = validate_entry_model_bundle(hgb_bundle)
    if (
        baseline.family != "HGB"
        or baseline.authorization_sha256 != _authorization_sha256(authorization)
        or baseline.dataset_sha256 != _member(dataset, "dataset_sha256")
    ):
        raise ValueError("neural fit baseline scope drift")
    frames = _prepare_fit_frames(dataset)
    payload = _fit_neural_payload(frames)
    if payload["module_parameter_count"] != 45_768:
        raise AssertionError(
            f"entry neural parameter budget drift: {payload['module_parameter_count']} != 45768"
        )
    return seal_entry_model_bundle(
        family="NEURAL",
        authorization=authorization,
        dataset_sha256=_member(dataset, "dataset_sha256"),
        payload=payload,
    )


def _calibration_fold_scope(authorization: prereg.FrozenFitAuthorization) -> str:
    if authorization.role == "nested_calibration":
        return (
            f"NESTED_OUTER_{authorization.outer_fold}_INNER_"
            f"{authorization.inner_fold}"
        )
    if authorization.role == "outer_calibration":
        return f"OUTER_{authorization.outer_fold}"
    if authorization.role == "full_calibration":
        return "FULL_PRE_HOLDOUT"
    raise ValueError("entry calibration fold scope drift")


def _calibration_seed_receipt(
    authorization: prereg.FrozenFitAuthorization, /, *, purpose: str,
    model_family: str, statistic: str,
) -> dict[str, Any]:
    key = {
        "campaign": "pathd.tier_s.v1",
        "plan_sha256": prereg.sha256_path(prereg.PLAN_PATH),
        "purpose": purpose,
        "model_family": model_family,
        "fold_scope": _calibration_fold_scope(authorization),
        "statistic": statistic,
    }
    canonical_bytes = json.dumps(
        key, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    return {
        "key": key,
        "canonical_key_sha256": digest,
        "seed": int.from_bytes(bytes.fromhex(digest)[:4], "big"),
        "numpy_version": np.__version__,
        "replicate_count": 2_000,
    }


def _model_predictions_for_frames(
    family: str, payload: Mapping[str, Any], frames: Sequence[_EntryFitFrame]
) -> tuple[np.ndarray, ...]:
    fixed = payload.get("fixed_prediction_values")
    if fixed is not None:
        values = np.asarray(fixed, dtype=np.float64)
        if values.shape == (2 * _ENTRY_HEAD_COUNT,):
            values = np.broadcast_to(
                values[None, :], (_ENTRY_ACTION_COUNT, 2 * _ENTRY_HEAD_COUNT)
            )
        if values.shape != (_ENTRY_ACTION_COUNT, 2 * _ENTRY_HEAD_COUNT):
            raise ValueError("constant entry prediction shape drift")
        return tuple(np.array(values, copy=True) for _frame in frames)
    return _predict_fitted_inputs(
        family, payload, tuple(frame.inputs for frame in frames)
    )


def _fit_marginal_entry_corrections(
    authorization: prereg.FrozenFitAuthorization, /, *, model_family: str,
    frames: Sequence[_EntryFitFrame], predictions: Sequence[np.ndarray],
) -> dict[str, Any]:
    if len(frames) != len(predictions):
        raise ValueError("entry calibration prediction alignment drift")
    mean_corrections: list[float] = []
    q10_corrections: list[float] = []
    receipts: list[dict[str, Any]] = []
    for column, head in enumerate(prereg.ENTRY_HEAD_TARGETS):
        observed: list[float] = []
        raw_mean: list[float] = []
        raw_q10: list[float] = []
        sessions: list[str] = []
        identities: list[str] = []
        for frame, prediction in zip(frames, predictions, strict=True):
            for action in np.flatnonzero(frame.target_validity[:, column]).tolist():
                identity = {
                    "session": frame.inputs.session,
                    "decision_time_ns": frame.inputs.decision_time_ns,
                    "source_neutral_contract_id": str(
                        frame.inputs.contract_ids[action]
                    ),
                    "target_axis": head,
                }
                observed.append(float(frame.targets[action, column]))
                raw_mean.append(float(prediction[action, column]))
                raw_q10.append(
                    float(prediction[action, _ENTRY_HEAD_COUNT + column])
                )
                sessions.append(frame.inputs.session)
                identities.append(
                    json.dumps(
                        identity, sort_keys=True, separators=(",", ":"),
                        allow_nan=False,
                    )
                )
        distinct_sessions = tuple(dict.fromkeys(sessions))
        if len(distinct_sessions) < 10:
            raise ValueError(f"entry calibration has fewer than 10 sessions for {head}")
        if len(identities) != len(set(identities)):
            raise ValueError("entry calibration row identity duplicate")
        actual = np.asarray(observed, dtype=np.float64)
        mean_array = np.asarray(raw_mean, dtype=np.float64)
        q10_array = np.asarray(raw_q10, dtype=np.float64)
        if not (
            np.isfinite(actual).all()
            and np.isfinite(mean_array).all()
            and np.isfinite(q10_array).all()
        ):
            raise ValueError("entry calibration prediction/target nonfinite")
        seed_receipt = _calibration_seed_receipt(
            authorization,
            purpose="mean_lcb_bootstrap",
            model_family=model_family,
            statistic=f"HEAD::{head}",
        )
        mean_correction = session_bootstrap_mean_lcb_correction(
            actual - mean_array,
            sessions=sessions,
            seed=seed_receipt["seed"],
            resamples=2_000,
        )
        q10_correction = weighted_lower_conformal_correction(
            q10_array,
            actual,
            sessions=sessions,
            identities=identities,
            alpha=0.1,
        )
        mean_corrections.append(mean_correction)
        q10_corrections.append(q10_correction)
        receipt_semantic = {
            "head": head,
            "row_count": len(observed),
            "session_count": len(distinct_sessions),
            "population_sha256": _canonical_sha256(identities),
            "mean_lcb_seed_receipt": seed_receipt,
            "mean_lcb_correction": mean_correction,
            "q10_correction": q10_correction,
        }
        receipts.append(
            {
                **receipt_semantic,
                "receipt_sha256": _canonical_sha256(receipt_semantic),
            }
        )
    return {
        "mean_lcb_corrections": mean_corrections,
        "q10_corrections": q10_corrections,
        "head_calibration_receipts": receipts,
    }


def _future_audit_vectors(example: Any) -> dict[str, list[Any]]:
    audit = _member(example, "audit")
    future = _member(audit, "future_audit")
    if type(future) is not dict:
        raise ValueError("all-action calibration requires future_audit")
    names = (
        "entry_filled_by_action",
        "buy_hard_limit_micros_by_action",
        "entry_cash_debit_if_filled_micros_by_action",
        "arrival_bid_micros_by_action",
        "arrival_ask_micros_by_action",
        "arrival_available_at_ns_by_action",
        "terminal_executable_bid_micros_by_action",
        "hold_to_flat_net_pnl_micros_by_action",
    )
    result: dict[str, list[Any]] = {}
    for name in names:
        vector = future.get(name)
        if type(vector) not in (tuple, list) or len(vector) != _ENTRY_ACTION_COUNT:
            raise ValueError(f"all-action calibration audit vector missing: {name}")
        result[name] = list(vector)
    if any(type(value) is not bool for value in result["entry_filled_by_action"]):
        raise ValueError("entry-filled audit type drift")
    integer_vectors = names[1:]
    if any(
        value is not None and type(value) is not int
        for name in integer_vectors
        for value in result[name]
    ):
        raise ValueError("all-action calibration monetary/clock audit type drift")
    return result


def _calibration_dynamic_mask(
    frame: _EntryFitFrame, /, *, cash_micros: int,
    session_start_equity_micros: int, realized_session_pnl_micros: int,
    occupied: bool, pending: bool,
) -> tuple[np.ndarray, dict[str, list[Any]]]:
    vectors = _future_audit_vectors(frame.example)
    facts = _member(frame.example, "action_execution_facts")
    if type(facts) not in (tuple, list) or len(facts) != _ENTRY_ACTION_COUNT:
        raise ValueError("calibration execution-fact matrix drift")
    bids = [_member(fact, "bid_micros") for fact in facts]
    asks = [_member(fact, "ask_micros") for fact in facts]
    if any(type(value) is not int for value in (*bids, *asks)):
        raise ValueError("calibration current quote type drift")
    complete_ladder = all(
        bid >= 0 and ask > 0 and bid <= ask
        for bid, ask in zip(bids, asks, strict=True)
    )
    budget = math.floor(0.05 * session_start_equity_micros)
    stopped = realized_session_pnl_micros <= -budget
    mask = np.zeros(_ENTRY_ACTION_COUNT, dtype=np.bool_)
    hard_limits = vectors["buy_hard_limit_micros_by_action"]
    for action, (bid, ask, hard_limit) in enumerate(
        zip(bids, asks, hard_limits, strict=True)
    ):
        if hard_limit is None:
            continue
        total_cost = hard_limit * 100 + 3_000_000
        executable = bid > 0 and ask > bid and ask >= 1_000_000
        mask[action] = bool(
            complete_ladder
            and not occupied
            and not pending
            and not stopped
            and executable
            and total_cost <= cash_micros
            and total_cost <= budget
            and max(0, -realized_session_pnl_micros) + total_cost <= budget
        )
    return mask, vectors


def _action_tie_keys(frame: _EntryFitFrame) -> tuple[Any, ...]:
    facts = tuple(_member(frame.example, "action_execution_facts"))
    return tuple(
        (
            abs(float(frame.inputs.current_offsets[action])),
            float(frame.inputs.current_offsets[action]),
            str(frame.inputs.current_rights[action]),
            int(str(_member(_member(fact, "contract"), "expiry")).replace("-", "")),
            int(_member(_member(fact, "contract"), "strike_milli")),
            str(_member(fact, "source_neutral_contract_id")),
        )
        for action, fact in enumerate(facts)
    )


def _available_composite_columns(inputs: EntryModelInputV1) -> tuple[int, ...]:
    available = _entry_available_horizons(inputs)
    positions = {
        "h10": 0,
        "h20": 1,
        "h45": 2,
        "h90": 3,
        "remaining_session": 4,
    }
    return tuple(
        column
        for horizon in available
        for column in (4 * positions[horizon], 4 * positions[horizon] + 2)
    )


def _realized_dollar_composite(
    frame: _EntryFitFrame, action: int, columns: Sequence[int]
) -> float | None:
    if not columns or not bool(frame.target_validity[action, list(columns)].all()):
        return None
    return float(
        math.fsum(float(frame.targets[action, column]) for column in columns)
        / len(columns)
    )


def _fit_action_composite_corrections(
    frames: Sequence[_EntryFitFrame], predictions: Sequence[np.ndarray], /, *,
    mean_corrections: Sequence[float], q10_corrections: Sequence[float],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[_EntryFitFrame, np.ndarray]]] = {}
    for frame, prediction in zip(frames, predictions, strict=True):
        grouped.setdefault(frame.inputs.session, []).append((frame, prediction))
    cash = 10_000 * 1_000_000
    enter_rows: list[dict[str, Any]] = []
    wait_rows: list[dict[str, Any]] = []
    missing_enter_outcomes = 0
    missing_wait_outcomes = 0
    for session in sorted(grouped):
        session_rows = sorted(
            grouped[session], key=lambda item: item[0].inputs.decision_time_ns
        )
        session_start = cash
        realized = 0
        pending_intent: dict[str, Any] | None = None
        position: dict[str, Any] | None = None
        next_eligible_ns = -1
        last_wait_ns: int | None = None
        for frame, raw_prediction in session_rows:
            now = frame.inputs.decision_time_ns
            if pending_intent is not None and now >= pending_intent["arrival_ns"]:
                vectors = pending_intent["vectors"]
                action = pending_intent["action"]
                if vectors["entry_filled_by_action"][action]:
                    debit = vectors[
                        "entry_cash_debit_if_filled_micros_by_action"
                    ][action]
                    terminal_bid = vectors[
                        "terminal_executable_bid_micros_by_action"
                    ][action]
                    net_pnl = vectors[
                        "hold_to_flat_net_pnl_micros_by_action"
                    ][action]
                    if any(type(value) is not int for value in (debit, terminal_bid, net_pnl)):
                        raise ValueError("filled calibration intent lacks terminal audit")
                    if debit > cash:
                        raise ValueError("calibration arrival fill is unaffordable")
                    cash -= debit
                    position = {
                        "terminal_credit": terminal_bid * 100 - 1_500_000,
                        "net_pnl": net_pnl,
                    }
                else:
                    next_eligible_ns = pending_intent["decision_ns"] + 120_000_000_000
                pending_intent = None
                last_wait_ns = None
            if position is not None or pending_intent is not None or now < next_eligible_ns:
                last_wait_ns = None
                continue
            dynamic, vectors = _calibration_dynamic_mask(
                frame,
                cash_micros=cash,
                session_start_equity_micros=session_start,
                realized_session_pnl_micros=realized,
                occupied=position is not None,
                pending=pending_intent is not None,
            )
            columns = _available_composite_columns(frame.inputs)
            if not columns:
                last_wait_ns = None
                continue
            calibrated_mean = (
                raw_prediction[:, :_ENTRY_HEAD_COUNT]
                + np.asarray(mean_corrections, dtype=np.float64)[None, :]
            )
            calibrated_q10 = (
                raw_prediction[:, _ENTRY_HEAD_COUNT:]
                + np.asarray(q10_corrections, dtype=np.float64)[None, :]
            )
            dollar_columns = tuple(
                column for horizon in range(5) for column in (4 * horizon, 4 * horizon + 2)
            )
            mean_dollars = calibrated_mean[:, dollar_columns]
            mean_returns = calibrated_mean[:, tuple(column + 1 for column in dollar_columns)]
            q10_dollars = calibrated_q10[:, dollar_columns]
            q10_returns = calibrated_q10[:, tuple(column + 1 for column in dollar_columns)]
            available_horizons = _entry_available_horizons(frame.inputs)
            available_mask = np.asarray(
                [
                    horizon in available_horizons
                    for horizon in ("h10", "h20", "h45", "h90", "remaining_session")
                ],
                dtype=np.bool_,
            )
            statistics = compose_enter_statistics(
                mean_dollars=mean_dollars,
                mean_returns=mean_returns,
                q10_dollars=q10_dollars,
                q10_returns=q10_returns,
                available=available_mask,
            )
            selected = select_entry_action_index(
                mean_lcb_dollars=statistics["mean_lcb_dollars"],
                mean_lcb_returns=statistics["mean_lcb_return"],
                q10_dollars=statistics["q10_dollars"],
                q10_returns=statistics["q10_return"],
                physical_action_mask=frame.inputs.physical_action_mask,
                dynamic_account_mask=dynamic,
                tie_break_keys=_action_tie_keys(frame),
            )
            legal = np.flatnonzero(frame.inputs.physical_action_mask & dynamic).tolist()
            selected_action = selected["selected_action_index"]
            if selected_action is not None:
                realized_composite = _realized_dollar_composite(
                    frame, selected_action, columns
                )
                if realized_composite is not None:
                    identity = (
                        f"{session}:{now}:"
                        f"{frame.inputs.contract_ids[selected_action]}"
                    )
                    enter_rows.append(
                        {
                            "session": session,
                            "identity": identity,
                            "raw_lower": float(statistics["q10_dollars"][selected_action]),
                            "realized": realized_composite,
                        }
                    )
                else:
                    missing_enter_outcomes += 1
                pending_intent = {
                    "decision_ns": now,
                    "arrival_ns": now + 60_000_000_000,
                    "action": selected_action,
                    "vectors": vectors,
                }
                last_wait_ns = None
                continue
            is_new_episode = last_wait_ns is None or now != last_wait_ns + 60_000_000_000
            if is_new_episode:
                outcomes = [
                    _realized_dollar_composite(frame, action, columns)
                    for action in legal
                ]
                if any(value is None for value in outcomes):
                    missing_wait_outcomes += 1
                else:
                    predicted_mean = max(
                        0.0,
                        max(
                            (float(statistics["mean_lcb_dollars"][action]) for action in legal),
                            default=0.0,
                        ),
                    )
                    raw_lower = max(
                        0.0,
                        max(
                            (float(statistics["q10_dollars"][action]) for action in legal),
                            default=0.0,
                        ),
                    )
                    realized_wait = max(
                        0.0,
                        max((float(value) for value in outcomes), default=0.0),
                    )
                    wait_rows.append(
                        {
                            "session": session,
                            "identity": f"{session}:{now}:WAIT_EPISODE",
                            "predicted_mean": predicted_mean,
                            "raw_lower": raw_lower,
                            "realized": realized_wait,
                        }
                    )
            last_wait_ns = now
        if pending_intent is not None:
            vectors = pending_intent["vectors"]
            action = pending_intent["action"]
            if vectors["entry_filled_by_action"][action]:
                debit = vectors["entry_cash_debit_if_filled_micros_by_action"][action]
                terminal_bid = vectors["terminal_executable_bid_micros_by_action"][action]
                net_pnl = vectors["hold_to_flat_net_pnl_micros_by_action"][action]
                if any(type(value) is not int for value in (debit, terminal_bid, net_pnl)):
                    raise ValueError("terminal calibration intent lacks audit")
                cash -= debit
                position = {
                    "terminal_credit": terminal_bid * 100 - 1_500_000,
                    "net_pnl": net_pnl,
                }
        if position is not None:
            cash += position["terminal_credit"]
            realized += position["net_pnl"]
        if cash <= 0:
            raise ValueError("calibration-only causal account exhausted")

    def conformal(
        rows: Sequence[Mapping[str, Any]], *, minimum_rows: int,
        missing_outcome_count: int,
    ) -> dict[str, Any]:
        sessions = [str(row["session"]) for row in rows]
        distinct = tuple(dict.fromkeys(sessions))
        if missing_outcome_count:
            return {
                "status": "INVALID_TARGET_COVERAGE",
                "correction": None,
                "row_count": len(rows),
                "session_count": len(distinct),
                "missing_outcome_count": missing_outcome_count,
            }
        if len(distinct) < 10 or len(rows) < minimum_rows:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "correction": None,
                "row_count": len(rows),
                "session_count": len(distinct),
                "missing_outcome_count": 0,
            }
        correction = weighted_lower_conformal_correction(
            np.asarray([row["raw_lower"] for row in rows], dtype=np.float64),
            np.asarray([row["realized"] for row in rows], dtype=np.float64),
            sessions=sessions,
            identities=[str(row["identity"]) for row in rows],
            alpha=0.1,
        )
        semantic = {
            "status": "VALID",
            "correction": correction,
            "row_count": len(rows),
            "session_count": len(distinct),
            "missing_outcome_count": 0,
            "population_sha256": _canonical_sha256(list(rows)),
        }
        return {**semantic, "receipt_sha256": _canonical_sha256(semantic)}

    enter = conformal(
        enter_rows, minimum_rows=1,
        missing_outcome_count=missing_enter_outcomes,
    )
    wait = conformal(
        wait_rows, minimum_rows=30,
        missing_outcome_count=missing_wait_outcomes,
    )
    return {
        "enter_composite_q10_status": enter["status"],
        "enter_composite_q10_correction": enter["correction"],
        "enter_composite_calibration_receipt": enter,
        "wait_composite_q10_status": wait["status"],
        "wait_composite_q10_correction": wait["correction"],
        "wait_composite_calibration_receipt": wait,
        "enter_missing_outcome_intent_count": missing_enter_outcomes,
        "wait_missing_outcome_episode_count": missing_wait_outcomes,
        "calibration_ledger_rule": "START_10000_FEE3_ONE_SLOT_HOLD_TO_FLAT",
    }


def _fit_entry_calibration_payload(
    authorization: prereg.FrozenFitAuthorization, /, *, model_family: str,
    frames: Sequence[_EntryFitFrame], predictions: Sequence[np.ndarray],
) -> dict[str, Any]:
    marginal = _fit_marginal_entry_corrections(
        authorization,
        model_family=model_family,
        frames=frames,
        predictions=predictions,
    )
    actions = _fit_action_composite_corrections(
        frames,
        predictions,
        mean_corrections=marginal["mean_lcb_corrections"],
        q10_corrections=marginal["q10_corrections"],
    )
    return {
        "format": "pathd.entry_calibration_fitted.v1",
        "model_family_seed_token": model_family,
        **marginal,
        **actions,
    }


def _fit_entry_calibrators_impl(
    authorization: prereg.FrozenFitAuthorization,
    model_bundle: EntryModelBundleV1, dataset: Any,
) -> EntryCalibrationBundleV1:
    model = validate_entry_model_bundle(model_bundle)
    frames = _prepare_fit_frames(dataset)
    predictions = _model_predictions_for_frames(
        model.family, model.payload, frames
    )
    payload = _fit_entry_calibration_payload(
        authorization,
        model_family=model.family,
        frames=frames,
        predictions=predictions,
    )
    return seal_entry_calibration_bundle(
        authorization=authorization,
        model_bundle=model,
        dataset_sha256=_member(dataset, "dataset_sha256"),
        payload=payload,
    )


def fit_hgb_entry_bundle(
    authorization: prereg.FrozenFitAuthorization, /
) -> EntryModelBundleV1:
    if type(authorization) is not prereg.FrozenFitAuthorization or authorization.role not in _WEIGHT_ROLES:
        raise ValueError("HGB fit requires a weights authorization")
    dataset = load_authorized_entry_dataset(authorization)
    current = assert_fit_authorization_current(authorization)
    result = _fit_hgb_entry_bundle_impl(current, dataset)
    validated = validate_entry_model_bundle(result)
    if (
        validated.family != "HGB"
        or validated.authorization_sha256 != _authorization_sha256(current)
        or validated.dataset_sha256 != getattr(dataset, "dataset_sha256", None)
    ):
        raise ValueError("HGB implementation returned wrong authority/dataset")
    _HGB_BASELINE_BY_AUTHORIZATION_ID[id(authorization)] = (
        authorization, validated.artifact_sha256
    )
    return validated


def _baseline_matches_authorization(
    authorization: prereg.FrozenFitAuthorization,
    bundle: EntryModelBundleV1,
) -> bool:
    record = _HGB_BASELINE_BY_AUTHORIZATION_ID.get(id(authorization))
    return (
        record is not None
        and record[0] is authorization
        and record[1] == bundle.artifact_sha256
    )


def fit_neural_entry_bundle(
    authorization: prereg.FrozenFitAuthorization, /, *,
    hgb_bundle: EntryModelBundleV1,
) -> Any:
    if type(authorization) is not prereg.FrozenFitAuthorization or authorization.role not in _WEIGHT_ROLES:
        raise ValueError("neural fit requires a weights authorization")
    hgb = validate_entry_model_bundle(hgb_bundle)
    if (
        hgb.family != "HGB"
        or hgb.fit_role != authorization.role
        or hgb.outer_fold != authorization.outer_fold
        or hgb.inner_fold != authorization.inner_fold
        or hgb.authorization_sha256 != _authorization_sha256(authorization)
        or hgb.sessions_sha256_newline != authorization.sessions_sha256_newline
        or not _baseline_matches_authorization(authorization, hgb)
    ):
        raise ValueError("neural fit requires the exact frozen HGB baseline")
    dataset = load_authorized_entry_dataset(authorization)
    if getattr(dataset, "dataset_sha256", None) != hgb.dataset_sha256:
        raise ValueError("neural/HGB dataset drift")
    current = assert_fit_authorization_current(authorization)
    return _fit_neural_entry_bundle_impl(current, dataset, hgb)


def fit_entry_calibrators(
    authorization: prereg.FrozenFitAuthorization, /, *,
    model_bundle: EntryModelBundleV1,
) -> Any:
    if (
        type(authorization) is not prereg.FrozenFitAuthorization
        or authorization.role not in _CALIBRATION_TO_WEIGHT_ROLE
    ):
        raise ValueError("entry calibration requires a calibration authorization")
    model = validate_entry_model_bundle(model_bundle)
    if (
        model.fit_role != _CALIBRATION_TO_WEIGHT_ROLE[authorization.role]
        or model.outer_fold != authorization.outer_fold
        or model.inner_fold != authorization.inner_fold
    ):
        raise ValueError("entry calibration/model scope drift")
    dataset = load_authorized_entry_dataset(authorization)
    current = assert_fit_authorization_current(authorization)
    if (
        getattr(dataset, "authorization_sha256", _authorization_sha256(current))
        != _authorization_sha256(current)
        or getattr(dataset, "role", current.role) != current.role
    ):
        raise ValueError("entry calibration dataset authority drift")
    return _fit_entry_calibrators_impl(current, model, dataset)


def _negative_control_seed_key(
    authorization: prereg.FrozenFitAuthorization, *, base_family: str,
    control_id: str,
) -> str:
    return prereg.stable_hash(
        {
            "campaign": "pathd.tier_s.v1",
            "purpose": "entry_negative_control",
            "base_family": base_family,
            "control_id": control_id,
            "fit_role": authorization.role,
            "outer_fold": authorization.outer_fold,
            "inner_fold": authorization.inner_fold,
            "sessions_sha256_newline": authorization.sessions_sha256_newline,
        }
    )


def _constant_control_values(
    frames: Sequence[_EntryFitFrame], /
) -> tuple[list[float], dict[str, Any]]:
    means: list[float] = []
    q10s: list[float] = []
    receipts: list[dict[str, Any]] = []
    for column, head in enumerate(prereg.ENTRY_HEAD_TARGETS):
        observed: list[float] = []
        sessions: list[str] = []
        identities: list[str] = []
        for frame in frames:
            for action in np.flatnonzero(frame.target_validity[:, column]).tolist():
                observed.append(float(frame.targets[action, column]))
                sessions.append(frame.inputs.session)
                identities.append(
                    json.dumps(
                        _hgb_identity(frame, action, head),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
        if not observed or len(identities) != len(set(identities)):
            raise ValueError("constant control target population drift")
        distinct = tuple(dict.fromkeys(sessions))
        counts = {session: sessions.count(session) for session in distinct}
        mean = math.fsum(
            value / (len(distinct) * counts[session])
            for value, session in zip(observed, sessions, strict=True)
        )
        q10 = weighted_lower_conformal_correction(
            np.zeros(len(observed), dtype=np.float64),
            np.asarray(observed, dtype=np.float64),
            sessions=sessions,
            identities=identities,
            alpha=0.1,
        )
        means.append(mean)
        q10s.append(q10)
        semantic = {
            "head": head,
            "row_count": len(observed),
            "session_count": len(distinct),
            "population_sha256": _canonical_sha256(identities),
            "mean": mean,
            "q10": q10,
        }
        receipts.append({**semantic, "receipt_sha256": _canonical_sha256(semantic)})
    return [*means, *q10s], {
        "constant_head_receipts": receipts,
        "constant_seed_count": 0,
        "constant_estimator_fit_count": 0,
    }


def _shuffle_row_identity(
    authorization: prereg.FrozenFitAuthorization, frame: _EntryFitFrame,
    action: int,
) -> dict[str, Any]:
    fact = _member(frame.example, "action_execution_facts")[action]
    contract = _member(fact, "contract")
    expiry = str(_member(contract, "expiry"))
    return {
        "outer_fold": authorization.outer_fold if authorization.outer_fold is not None else 0,
        "session": frame.inputs.session,
        "decision_time_ns": frame.inputs.decision_time_ns,
        "source_neutral_contract_id": str(frame.inputs.contract_ids[action]),
        "expiry_yyyymmdd": int(expiry.replace("-", "")),
        "strike_milli_points": int(_member(contract, "strike_milli")),
        "right_code": str(_member(contract, "right")),
    }


def _shuffled_target_frames(
    authorization: prereg.FrozenFitAuthorization, frames: Sequence[_EntryFitFrame],
    *, seed: int,
) -> tuple[tuple[_EntryFitFrame, ...], dict[str, Any]]:
    row_count = len(frames) * _ENTRY_ACTION_COUNT
    ranks = np.empty(
        row_count, dtype=np.dtype([("digest", "S32"), ("flat_index", "<i8")])
    )
    destination_hasher = hashlib.sha256()
    destination_hasher.update(b"[")
    for flat_index in range(row_count):
        frame_index, action = divmod(flat_index, _ENTRY_ACTION_COUNT)
        identity = _shuffle_row_identity(
            authorization, frames[frame_index], action
        )
        if flat_index:
            destination_hasher.update(b",")
        encoded_identity = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        destination_hasher.update(encoded_identity)
        digest = hashlib.sha256(
            json.dumps(
                {"attempt_seed_id": seed, **identity},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).digest()
        ranks[flat_index] = (digest, flat_index)
    destination_hasher.update(b"]")
    digest_order = np.argsort(ranks["digest"], kind="stable")
    source_order = np.empty(row_count, dtype=np.int64)
    output_offset = 0
    cursor = 0
    while cursor < row_count:
        end = cursor + 1
        digest = ranks["digest"][digest_order[cursor]]
        while end < row_count and ranks["digest"][digest_order[end]] == digest:
            end += 1
        tied = [
            (
                json.dumps(
                    _shuffle_row_identity(
                        authorization,
                        frames[int(ranks["flat_index"][index]) // _ENTRY_ACTION_COUNT],
                        int(ranks["flat_index"][index]) % _ENTRY_ACTION_COUNT,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                int(ranks["flat_index"][index]),
            )
            for index in digest_order[cursor:end]
        ]
        for _identity, flat_index in sorted(tied):
            source_order[output_offset] = flat_index
            output_offset += 1
        cursor = end
    source_hasher = hashlib.sha256()
    source_hasher.update(b"[")
    transformed: list[_EntryFitFrame] = []
    for destination_frame, frame in enumerate(frames):
        targets = np.empty_like(frame.targets)
        validity = np.empty_like(frame.target_validity)
        for action in range(_ENTRY_ACTION_COUNT):
            destination_flat = destination_frame * _ENTRY_ACTION_COUNT + action
            source_flat = int(source_order[destination_flat])
            source_frame, source_action = divmod(source_flat, _ENTRY_ACTION_COUNT)
            targets[action] = frames[source_frame].targets[source_action]
            validity[action] = frames[source_frame].target_validity[source_action]
            identity = _shuffle_row_identity(
                authorization, frames[source_frame], source_action
            )
            if destination_flat:
                source_hasher.update(b",")
            source_hasher.update(
                json.dumps(
                    identity, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        transformed.append(
            _EntryFitFrame(
                example=frame.example,
                inputs=frame.inputs,
                targets=targets,
                target_validity=validity,
            )
        )
    source_hasher.update(b"]")
    semantic = {
        "attempt_seed_id": seed,
        "row_count": row_count,
        "destination_population_sha256": destination_hasher.hexdigest(),
        "ordered_source_population_sha256": source_hasher.hexdigest(),
    }
    return tuple(transformed), {
        **semantic, "receipt_sha256": _canonical_sha256(semantic)
    }


def _seal_negative_control_model(
    authorization: prereg.FrozenFitAuthorization, /, *, control_id: str,
    base_family: str, dataset_sha256: str, payload: Mapping[str, Any],
) -> EntryNegativeControlBundleV1:
    values = {
        "schema_version": EntryNegativeControlBundleV1.SCHEMA_VERSION,
        "control_id": control_id,
        "base_family": base_family,
        "fit_role": authorization.role,
        "outer_fold": authorization.outer_fold,
        "inner_fold": authorization.inner_fold,
        "authorization_sha256": _authorization_sha256(authorization),
        "dataset_sha256": dataset_sha256,
        "seed_key_sha256": _negative_control_seed_key(
            authorization, base_family=base_family, control_id=control_id
        ),
        "payload": _clone_jsonish(payload),
    }
    return validate_entry_negative_control_bundle(
        _seal_dataclass(EntryNegativeControlBundleV1, values)
    )


def _fit_entry_negative_control_bundle_impl(
    authorization: prereg.FrozenFitAuthorization, dataset: Any, *,
    base_family: str, control_id: str,
) -> EntryNegativeControlBundleV1:
    original = _prepare_fit_frames(dataset)
    transform_receipt: dict[str, Any]
    if control_id == "CONSTANT":
        fixed, receipt = _constant_control_values(original)
        model_payload = {
            "format": "pathd.entry_constant_control.v1",
            "input_transform": "NONE",
            "fixed_prediction_values": fixed,
        }
        transform_receipt = receipt
    elif control_id == "TIME_SHIFTED_FEATURES":
        shifted = _prepare_fit_frames(dataset, time_shifted=True)
        model_payload = (
            _fit_hgb_payload(shifted, input_transform="TIME_SHIFT_30")
            if base_family == "HGB"
            else _fit_neural_payload(shifted, input_transform="TIME_SHIFT_30")
        )
        transform_receipt = {
            "operation": "SIGNED17_HISTORY_ENDPOINT_MINUS_30_MINUTES",
            "frame_count": len(shifted),
            "transform_sha256": _canonical_sha256(
                [frame.inputs.canonical_sha256() for frame in shifted]
            ),
        }
    elif control_id.startswith("SHUFFLED_TARGET_"):
        attempt = int(control_id.rsplit("_", 1)[1])
        if not 1 <= attempt <= 8:
            raise ValueError("shuffled-target attempt drift")
        seed = prereg.SHUFFLED_TARGET_SEEDS[attempt - 1]
        shuffled, transform_receipt = _shuffled_target_frames(
            authorization, original, seed=seed
        )
        model_payload = (
            _fit_hgb_payload(shuffled)
            if base_family == "HGB"
            else _fit_neural_payload(shuffled)
        )
    else:
        raise ValueError("unsupported fitted entry negative control")
    payload = {
        "format": "pathd.entry_negative_control_fitted.v1",
        "control_id": control_id,
        "base_family": base_family,
        "model_payload": model_payload,
        "transform_receipt": transform_receipt,
    }
    return _seal_negative_control_model(
        authorization,
        control_id=control_id,
        base_family=base_family,
        dataset_sha256=_member(dataset, "dataset_sha256"),
        payload=payload,
    )


def _fit_entry_negative_control_calibrators_impl(
    authorization: prereg.FrozenFitAuthorization,
    control_bundle: EntryNegativeControlBundleV1, dataset: Any,
) -> EntryNegativeControlCalibrationBundleV1:
    control = validate_entry_negative_control_bundle(control_bundle)
    payload = control.payload
    if (
        payload.get("format") != "pathd.entry_negative_control_fitted.v1"
        or payload.get("control_id") != control.control_id
        or payload.get("base_family") != control.base_family
        or type(payload.get("model_payload")) is not dict
    ):
        raise ValueError("negative-control fitted payload drift")
    frames = _prepare_fit_frames(dataset)
    predictions = _model_predictions_for_frames(
        control.base_family, payload["model_payload"], frames
    )
    if control.control_id == "CONSTANT":
        family_token = "CONSTANT"
    elif control.control_id == "TIME_SHIFTED_FEATURES":
        family_token = f"TIME_SHIFTED_{control.base_family}"
    else:
        attempt = control.control_id.rsplit("_", 1)[1]
        family_token = f"SHUFFLED_TARGET_{control.base_family}_{attempt}"
    calibration_payload = _fit_entry_calibration_payload(
        authorization,
        model_family=family_token,
        frames=frames,
        predictions=predictions,
    )
    values = {
        "schema_version": EntryNegativeControlCalibrationBundleV1.SCHEMA_VERSION,
        "control_artifact_sha256": control.artifact_sha256,
        "control_id": control.control_id,
        "base_family": control.base_family,
        "calibration_role": authorization.role,
        "outer_fold": authorization.outer_fold,
        "inner_fold": authorization.inner_fold,
        "authorization_sha256": _authorization_sha256(authorization),
        "dataset_sha256": _member(dataset, "dataset_sha256"),
        "payload": calibration_payload,
    }
    return validate_entry_negative_control_calibration_bundle(
        _seal_dataclass(EntryNegativeControlCalibrationBundleV1, values),
        control_bundle=control,
    )


def fit_entry_negative_control_bundle(
    authorization: prereg.FrozenFitAuthorization, /, *, base_family: str,
    control_id: str,
) -> EntryNegativeControlBundleV1:
    if (
        type(authorization) is not prereg.FrozenFitAuthorization
        or authorization.role not in _WEIGHT_ROLES
        or base_family not in {"HGB", "NEURAL"}
    ):
        raise ValueError("negative-control fit authority/family drift")
    control = _validate_control_id(control_id)
    if control == "SIGN_REVERSED":
        raise ValueError("SIGN_REVERSED is composer-only and must not be refit")
    dataset = load_authorized_entry_dataset(authorization)
    current = assert_fit_authorization_current(authorization)
    result = _fit_entry_negative_control_bundle_impl(
        current, dataset, base_family=base_family, control_id=control
    )
    validated = validate_entry_negative_control_bundle(result)
    if (
        validated.control_id != control
        or validated.base_family != base_family
        or validated.fit_role != current.role
        or validated.outer_fold != current.outer_fold
        or validated.inner_fold != current.inner_fold
        or validated.authorization_sha256 != _authorization_sha256(current)
        or validated.dataset_sha256 != getattr(dataset, "dataset_sha256", None)
        or validated.seed_key_sha256
        != _negative_control_seed_key(
            current, base_family=base_family, control_id=control
        )
    ):
        raise ValueError("negative-control implementation identity drift")
    return validated


def validate_entry_negative_control_bundle(
    bundle: Any, /
) -> EntryNegativeControlBundleV1:
    bundle = _coerce_exact_dataclass(bundle, EntryNegativeControlBundleV1)
    if (
        bundle.schema_version != bundle.SCHEMA_VERSION
        or bundle.control_id not in prereg.ENTRY_NEGATIVE_CONTROL_IDS
        or bundle.control_id == "SIGN_REVERSED"
        or bundle.base_family not in {"HGB", "NEURAL"}
        or bundle.fit_role not in _WEIGHT_ROLES
        or not _valid_fit_scope(bundle.fit_role, bundle.outer_fold, bundle.inner_fold)
        or not _is_hex64(bundle.authorization_sha256)
        or not _is_hex64(bundle.dataset_sha256)
        or not _is_hex64(bundle.seed_key_sha256)
        or type(bundle.payload) is not dict
    ):
        raise ValueError("negative-control bundle drift")
    _validate_artifact_hash(bundle)
    return bundle


def fit_entry_negative_control_calibrators(
    authorization: prereg.FrozenFitAuthorization, /, *,
    control_bundle: EntryNegativeControlBundleV1,
) -> EntryNegativeControlCalibrationBundleV1:
    if (
        type(authorization) is not prereg.FrozenFitAuthorization
        or authorization.role not in _CALIBRATION_TO_WEIGHT_ROLE
    ):
        raise ValueError("negative-control calibration authority drift")
    control = validate_entry_negative_control_bundle(control_bundle)
    if (
        control.fit_role != _CALIBRATION_TO_WEIGHT_ROLE[authorization.role]
        or control.outer_fold != authorization.outer_fold
        or control.inner_fold != authorization.inner_fold
    ):
        raise ValueError("negative-control calibration scope drift")
    dataset = load_authorized_entry_dataset(authorization)
    current = assert_fit_authorization_current(authorization)
    result = _fit_entry_negative_control_calibrators_impl(
        current, control, dataset
    )
    validated = validate_entry_negative_control_calibration_bundle(
        result, control_bundle=control
    )
    if (
        validated.authorization_sha256 != _authorization_sha256(current)
        or validated.dataset_sha256 != getattr(dataset, "dataset_sha256", None)
    ):
        raise ValueError("negative-control calibrator implementation drift")
    return validated


def validate_entry_negative_control_calibration_bundle(
    bundle: Any, /, *, control_bundle: EntryNegativeControlBundleV1,
) -> EntryNegativeControlCalibrationBundleV1:
    control = validate_entry_negative_control_bundle(control_bundle)
    bundle = _coerce_exact_dataclass(
        bundle, EntryNegativeControlCalibrationBundleV1
    )
    expected_role = {
        "outer_weights": "outer_calibration",
        "nested_weights": "nested_calibration",
        "full_weights": "full_calibration",
    }[control.fit_role]
    if (
        bundle.schema_version != bundle.SCHEMA_VERSION
        or bundle.control_artifact_sha256 != control.artifact_sha256
        or bundle.control_id != control.control_id
        or bundle.base_family != control.base_family
        or bundle.calibration_role != expected_role
        or bundle.outer_fold != control.outer_fold
        or bundle.inner_fold != control.inner_fold
        or not _valid_fit_scope(
            bundle.calibration_role, bundle.outer_fold, bundle.inner_fold
        )
        or not _is_hex64(bundle.authorization_sha256)
        or not _is_hex64(bundle.dataset_sha256)
        or type(bundle.payload) is not dict
    ):
        raise ValueError("negative-control calibration bundle drift")
    _validate_artifact_hash(bundle)
    return bundle


def _negative_control_transform_stage(control_id: str) -> str:
    if control_id == "CONSTANT":
        return "PREDICTION_BASELINE"
    if control_id == "SIGN_REVERSED":
        return "COMPOSER_POST_CALIBRATION"
    if control_id == "TIME_SHIFTED_FEATURES":
        return "FEATURE_PRE_FIT"
    if control_id.startswith("SHUFFLED_TARGET_"):
        return "TARGET_PRE_FIT"
    raise ValueError("negative-control transform stage drift")


def build_entry_negative_control_composer(
    control_bundle: EntryNegativeControlBundleV1, /, *,
    calibration_bundle: EntryNegativeControlCalibrationBundleV1,
) -> EntryNegativeControlComposerBundleV1:
    control = validate_entry_negative_control_bundle(control_bundle)
    calibration = validate_entry_negative_control_calibration_bundle(
        calibration_bundle, control_bundle=control
    )
    values = {
        "schema_version": EntryNegativeControlComposerBundleV1.SCHEMA_VERSION,
        "control_artifact_sha256": control.artifact_sha256,
        "calibration_artifact_sha256": calibration.artifact_sha256,
        "control_id": control.control_id,
        "base_family": control.base_family,
        "outer_fold": control.outer_fold,
        "inner_fold": control.inner_fold,
        "transform_stage": _negative_control_transform_stage(control.control_id),
        "composer_spec_sha256": prereg.stable_hash(prereg.entry_composer_spec()),
        "payload": {
            "control_seed_key_sha256": control.seed_key_sha256,
            "calibration_dataset_sha256": calibration.dataset_sha256,
            "mean_lcb_additive_offsets": _clone_jsonish(
                calibration.payload.get("mean_lcb_corrections")
            ),
            "q10_additive_offsets": _clone_jsonish(
                calibration.payload.get("q10_corrections")
            ),
            "enter_composite_q10_status": calibration.payload.get(
                "enter_composite_q10_status"
            ),
            "enter_composite_q10_correction": calibration.payload.get(
                "enter_composite_q10_correction"
            ),
            "wait_composite_q10_status": calibration.payload.get(
                "wait_composite_q10_status"
            ),
            "wait_composite_q10_correction": calibration.payload.get(
                "wait_composite_q10_correction"
            ),
        },
    }
    result = _seal_dataclass(EntryNegativeControlComposerBundleV1, values)
    return validate_entry_negative_control_composer_bundle(result)


def build_entry_sign_reversed_composer(
    model_bundle: EntryModelBundleV1, /, *,
    calibration_bundle: EntryCalibrationBundleV1,
    composer_bundle: EntryComposerBundleV1,
) -> EntryNegativeControlComposerBundleV1:
    model = validate_entry_model_bundle(model_bundle)
    calibration = validate_entry_calibration_bundle(
        calibration_bundle, model_bundle=model
    )
    composer = validate_entry_composer_bundle(
        composer_bundle, model_bundle=model, calibration_bundle=calibration
    )
    values = {
        "schema_version": EntryNegativeControlComposerBundleV1.SCHEMA_VERSION,
        "control_artifact_sha256": model.artifact_sha256,
        "calibration_artifact_sha256": calibration.artifact_sha256,
        "control_id": "SIGN_REVERSED",
        "base_family": model.family,
        "outer_fold": model.outer_fold,
        "inner_fold": model.inner_fold,
        "transform_stage": "COMPOSER_POST_CALIBRATION",
        "composer_spec_sha256": prereg.stable_hash(prereg.entry_composer_spec()),
        "payload": {
            "base_composer_sha256": composer.artifact_sha256,
            "operation": "NEGATE_ALL_CALIBRATED_MEAN_LCB_AND_Q10_COMPONENTS",
            "recalibration_or_refit": "none",
        },
    }
    result = _seal_dataclass(EntryNegativeControlComposerBundleV1, values)
    return validate_entry_negative_control_composer_bundle(result)


def validate_entry_negative_control_composer_bundle(
    bundle: Any, /
) -> EntryNegativeControlComposerBundleV1:
    bundle = _coerce_exact_dataclass(
        bundle, EntryNegativeControlComposerBundleV1
    )
    if (
        bundle.schema_version != bundle.SCHEMA_VERSION
        or bundle.control_id not in prereg.ENTRY_NEGATIVE_CONTROL_IDS
        or bundle.base_family not in {"HGB", "NEURAL"}
        or not _is_hex64(bundle.control_artifact_sha256)
        or not _is_hex64(bundle.calibration_artifact_sha256)
        or bundle.transform_stage
        != _negative_control_transform_stage(bundle.control_id)
        or bundle.composer_spec_sha256
        != prereg.stable_hash(prereg.entry_composer_spec())
        or type(bundle.payload) is not dict
        or not (
            (bundle.outer_fold is None and bundle.inner_fold is None)
            or (
                type(bundle.outer_fold) is int
                and 1 <= bundle.outer_fold <= 5
                and (
                    bundle.inner_fold is None
                    or type(bundle.inner_fold) is int
                    and 1 <= bundle.inner_fold <= 4
                )
            )
        )
    ):
        raise ValueError("negative-control composer bundle drift")
    if bundle.control_id == "SIGN_REVERSED" and (
        bundle.payload.get("operation")
        != "NEGATE_ALL_CALIBRATED_MEAN_LCB_AND_Q10_COMPONENTS"
        or bundle.payload.get("recalibration_or_refit") != "none"
        or not _is_hex64(bundle.payload.get("base_composer_sha256"))
    ):
        raise ValueError("sign-reversed composer payload drift")
    _validate_artifact_hash(bundle)
    return bundle


def _negative_control_fit_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (family, control_id)
        for family in ("HGB", "NEURAL")
        for control_id in prereg.ENTRY_NEGATIVE_CONTROL_IDS
        if control_id != "SIGN_REVERSED"
    )


def _negative_control_all_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (family, control_id)
        for family in ("HGB", "NEURAL")
        for control_id in prereg.ENTRY_NEGATIVE_CONTROL_IDS
    )


def build_entry_negative_control_manifest(
    *, outer_fold: int, control_bundles: Sequence[Any],
    calibration_bundles: Sequence[Any], composer_bundles: Sequence[Any],
    replay_config_sha256: str,
) -> EntryNegativeControlManifestV1:
    if type(outer_fold) is not int or not 1 <= outer_fold <= 5:
        raise ValueError("negative-control manifest fold drift")
    if not _is_hex64(replay_config_sha256):
        raise ValueError("negative-control manifest replay-config drift")
    controls = tuple(validate_entry_negative_control_bundle(item) for item in control_bundles)
    expected_fit = _negative_control_fit_pairs()
    if (
        len(controls) != len(expected_fit)
        or tuple((item.base_family, item.control_id) for item in controls)
        != expected_fit
        or any(
            item.fit_role != "outer_weights"
            or item.outer_fold != outer_fold
            or item.inner_fold is not None
            for item in controls
        )
    ):
        raise ValueError("negative-control manifest fit-bundle matrix drift")
    if len(calibration_bundles) != len(controls):
        raise ValueError("negative-control manifest calibration count drift")
    calibrations = tuple(
        validate_entry_negative_control_calibration_bundle(
            calibration, control_bundle=control
        )
        for control, calibration in zip(controls, calibration_bundles, strict=True)
    )
    composers = tuple(
        validate_entry_negative_control_composer_bundle(item)
        for item in composer_bundles
    )
    expected_all = _negative_control_all_pairs()
    if (
        len(composers) != len(expected_all)
        or tuple((item.base_family, item.control_id) for item in composers)
        != expected_all
        or any(
            item.outer_fold != outer_fold or item.inner_fold is not None
            for item in composers
        )
    ):
        raise ValueError("negative-control manifest composer matrix drift")
    sign_reversed = tuple(
        item for item in composers if item.control_id == "SIGN_REVERSED"
    )
    values = {
        "schema_version": EntryNegativeControlManifestV1.SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "required_control_ids": tuple(prereg.ENTRY_NEGATIVE_CONTROL_IDS),
        "required_base_families": ("HGB", "NEURAL"),
        "control_bundle_sha256s": tuple(item.artifact_sha256 for item in controls),
        "control_calibration_sha256s": tuple(
            item.artifact_sha256 for item in calibrations
        ),
        "control_composer_sha256s": tuple(
            item.artifact_sha256 for item in composers
        ),
        "sign_reversed_base_composer_sha256s": tuple(
            item.payload["base_composer_sha256"] for item in sign_reversed
        ),
        "seed_receipt_sha256s": tuple(item.seed_key_sha256 for item in controls),
        "replay_config_sha256": replay_config_sha256,
    }
    result = _seal_dataclass(EntryNegativeControlManifestV1, values)
    return validate_entry_negative_control_manifest(result)


def validate_entry_negative_control_manifest(
    manifest: Any, /
) -> EntryNegativeControlManifestV1:
    manifest = _coerce_exact_dataclass(manifest, EntryNegativeControlManifestV1)
    fit_count = len(_negative_control_fit_pairs())
    all_count = len(_negative_control_all_pairs())
    hash_vectors = (
        (manifest.control_bundle_sha256s, fit_count),
        (manifest.control_calibration_sha256s, fit_count),
        (manifest.control_composer_sha256s, all_count),
        (manifest.sign_reversed_base_composer_sha256s, 2),
        (manifest.seed_receipt_sha256s, fit_count),
    )
    if (
        manifest.schema_version != manifest.SCHEMA_VERSION
        or type(manifest.outer_fold) is not int
        or not 1 <= manifest.outer_fold <= 5
        or tuple(manifest.required_control_ids)
        != tuple(prereg.ENTRY_NEGATIVE_CONTROL_IDS)
        or tuple(manifest.required_base_families) != ("HGB", "NEURAL")
        or not _is_hex64(manifest.replay_config_sha256)
        or any(
            type(vector) not in (tuple, list)
            or len(vector) != length
            or any(not _is_hex64(item) for item in vector)
            for vector, length in hash_vectors
        )
        or any(
            len(vector) != len(set(vector))
            for vector, _length in hash_vectors[:4]
        )
    ):
        raise ValueError("negative-control manifest drift")
    _validate_artifact_hash(manifest)
    return manifest


def validate_entry_control_replay_config(
    config: Any, /
) -> EntryControlReplayConfigV1:
    config = _coerce_exact_dataclass(config, EntryControlReplayConfigV1)
    if (
        config.schema_version != config.SCHEMA_VERSION
        or type(config.outer_fold) is not int
        or not 1 <= config.outer_fold <= 5
        or not _is_hex64(config.fill_law_hash)
        or not _is_hex64(config.control_exit_sha256)
        or tuple(config.required_policy_ids)
        != tuple(prereg.entry_matched_random_owner_policy_ids())
        or tuple(config.matched_random_seeds) != tuple(prereg.MATCHED_RANDOM_SEEDS)
        or tuple(config.fee_paths) != (3, 4)
        or tuple(config.sell_delay_rungs_ms) != (0, 1_000, 2_000, 5_000)
        or not _is_hex64(config.headline_config_sha256)
    ):
        raise ValueError("entry control replay config drift")
    _validate_artifact_hash(config)
    return config


ENTRY_CONTROL_EXIT_SELECTION_RULE = (
    "MAX_TOTAL_FEE3_NET_PNL_THEN_LEXICOGRAPHIC_CANONICAL_COMPARATOR_ID"
)


def entry_control_exit_policy_ids() -> tuple[str, ...]:
    fixed = tuple(
        f"FIXED_STOP_{stop}_TARGET_{target}_TIME_{seconds}"
        for stop in ("M25", "M50")
        for target in ("P25", "P50", "P100")
        for seconds in (60, 300, 900)
    )
    return (
        "EXIT_IMMEDIATE",
        "HOLD_TO_FLAT",
        "TIME_60",
        "TIME_300",
        "TIME_900",
        *fixed,
        "LEGACY_P5_LIFECYCLE",
    )


def validate_entry_control_exit_candidate_evaluation(
    value: Any, /
) -> EntryControlExitCandidateEvaluationV1:
    row = _coerce_exact_dataclass(value, EntryControlExitCandidateEvaluationV1)
    sessions = tuple(
        prereg.session_assignments()["folds"][row.outer_fold - 1]["model_fit"]
    ) if type(row.outer_fold) is int and row.outer_fold in range(1, 6) else ()
    if (
        row.schema_version != row.SCHEMA_VERSION
        or row.policy_id not in entry_control_exit_policy_ids()
        or row.model_fit_sessions_sha256_newline != prereg.canonical_session_hash(sessions)
        or len(row.session_pnl_micros) != len(sessions)
        or len(row.session_terminal_journal_sha256s) != len(sessions)
        or any(type(item) is not int for item in row.session_pnl_micros)
        or any(not _is_hex64(item) for item in row.session_terminal_journal_sha256s)
        or row.valid_session_count != len(sessions)
        or row.total_net_pnl_micros != sum(row.session_pnl_micros)
    ):
        raise ValueError("entry control-exit candidate evaluation drift")
    _validate_artifact_hash(row, hash_field="evaluation_sha256")
    return row


def select_entry_control_exit(
    *, outer_fold: int,
    evaluations: Sequence[EntryControlExitCandidateEvaluationV1],
) -> EntryControlExitSelectionV1:
    """Select the shared transparent exit from complete earlier-session replays."""

    rows = tuple(validate_entry_control_exit_candidate_evaluation(item) for item in evaluations)
    expected = entry_control_exit_policy_ids()
    if tuple(row.policy_id for row in rows) != expected:
        raise ValueError("entry control-exit panel is incomplete or out of order")
    if any(row.outer_fold != outer_fold for row in rows):
        raise ValueError("entry control-exit evaluation fold drift")
    best_pnl = max(row.total_net_pnl_micros for row in rows)
    selected = min(row.policy_id for row in rows if row.total_net_pnl_micros == best_pnl)
    session_hash = rows[0].model_fit_sessions_sha256_newline
    semantic = {
        "schema_version": EntryControlExitSelectionV1.SCHEMA_VERSION,
        "holdout_caveat": prereg.HOLDOUT_CAVEAT,
        "outer_fold": outer_fold,
        "model_fit_sessions_sha256_newline": session_hash,
        "candidate_policy_ids": tuple(row.policy_id for row in rows),
        "candidate_evaluation_sha256s": tuple(row.evaluation_sha256 for row in rows),
        "valid_session_counts": tuple(row.valid_session_count for row in rows),
        "total_net_pnl_micros": tuple(row.total_net_pnl_micros for row in rows),
        "selected_policy_id": selected,
        "selection_rule": ENTRY_CONTROL_EXIT_SELECTION_RULE,
    }
    result = EntryControlExitSelectionV1(
        **semantic, artifact_sha256=prereg.stable_hash(_canonical(semantic))
    )
    return validate_entry_control_exit_selection(result)


def validate_entry_control_exit_selection(
    selection: Any, /
) -> EntryControlExitSelectionV1:
    selection = _coerce_exact_dataclass(selection, EntryControlExitSelectionV1)
    count = len(selection.candidate_policy_ids)
    if (
        selection.schema_version != selection.SCHEMA_VERSION
        or selection.holdout_caveat != prereg.HOLDOUT_CAVEAT
        or type(selection.outer_fold) is not int
        or not 1 <= selection.outer_fold <= 5
        or not _is_hex64(selection.model_fit_sessions_sha256_newline)
        or type(selection.candidate_policy_ids) not in (tuple, list)
        or count == 0
        or any(type(item) is not str or not item for item in selection.candidate_policy_ids)
        or len(set(selection.candidate_policy_ids)) != count
        or type(selection.candidate_evaluation_sha256s) not in (tuple, list)
        or len(selection.candidate_evaluation_sha256s) != count
        or any(not _is_hex64(item) for item in selection.candidate_evaluation_sha256s)
        or type(selection.valid_session_counts) not in (tuple, list)
        or len(selection.valid_session_counts) != count
        or any(type(item) is not int or item < 25 for item in selection.valid_session_counts)
        or type(selection.total_net_pnl_micros) not in (tuple, list)
        or len(selection.total_net_pnl_micros) != count
        or any(type(item) is not int for item in selection.total_net_pnl_micros)
        or selection.selection_rule != ENTRY_CONTROL_EXIT_SELECTION_RULE
    ):
        raise ValueError("entry control-exit selection drift")
    best_pnl = max(selection.total_net_pnl_micros)
    expected = min(
        policy
        for policy, pnl in zip(
            selection.candidate_policy_ids,
            selection.total_net_pnl_micros,
            strict=True,
        )
        if pnl == best_pnl
    )
    if selection.selected_policy_id != expected:
        raise ValueError("entry control-exit selected policy drift")
    _validate_artifact_hash(selection)
    return selection


def validate_entry_nested_replay_config(
    config: Any, /
) -> EntryNestedReplayConfigV1:
    config = _coerce_exact_dataclass(config, EntryNestedReplayConfigV1)
    if (
        config.schema_version != config.SCHEMA_VERSION
        or type(config.outer_fold) is not int
        or not 1 <= config.outer_fold <= 5
        or type(config.inner_fold) is not int
        or not 1 <= config.inner_fold <= 4
        or not _is_hex64(config.fill_law_hash)
        or tuple(config.policy_ids) != ("HGB", "NEURAL")
        or config.fee_path != 3
        or config.sell_delay_ms != 1_000
        or type(config.floor_on) is not bool
        or not config.floor_on
        or type(config.exit_policy_id) is not str
        or not config.exit_policy_id
        or not _is_hex64(config.nested_account_scope_sha256)
    ):
        raise ValueError("entry nested replay config drift")
    _validate_artifact_hash(config)
    return config


def _expected_replay_policy_id(
    *, owner_policy_id: str, channel: str, matched_random_seed: int | None
) -> str:
    if matched_random_seed is None:
        return f"{owner_policy_id}::{channel}"
    return (
        f"MATCHED_RANDOM::{owner_policy_id}::SEED_{matched_random_seed}::{channel}"
    )


def _validate_entry_replay_cell(
    value: Any, *, outer_fold: int, owner_policy_id: str, channel: str,
    matched_random_seed: int | None,
) -> EntryReplayEvaluationCellV1:
    cell = _coerce_exact_dataclass(value, EntryReplayEvaluationCellV1)
    evaluation = cell.evaluation
    expected_policy = _expected_replay_policy_id(
        owner_policy_id=owner_policy_id,
        channel=channel,
        matched_random_seed=matched_random_seed,
    )
    if (
        cell.schema_version != cell.SCHEMA_VERSION
        or cell.outer_fold != outer_fold
        or cell.owner_policy_id != owner_policy_id
        or cell.replay_policy_id != expected_policy
        or cell.channel != channel
        or cell.matched_random_seed != matched_random_seed
        or not _is_hex64(cell.authorization_sha256)
        or not _is_hex64(cell.dataset_sha256)
        or type(evaluation) is not dict
        or evaluation.get("policy_id") != expected_policy
        or not _is_hex64(evaluation.get("result_sha256"))
        or cell.evaluation_sha256 != evaluation.get("result_sha256")
        or not _is_hex64(evaluation.get("terminal_journal_sha256"))
        or cell.terminal_journal_sha256
        != evaluation.get("terminal_journal_sha256")
    ):
        raise ValueError("entry replay evaluation cell identity drift")
    receipt_fields = (
        "candidate_budget_sha256",
        "ordered_schedule_sha256",
        "structural_skip_trace_sha256",
        "realized_intents_and_fills_sha256",
    )
    if matched_random_seed is None:
        if any(getattr(cell, name) is not None for name in receipt_fields):
            raise ValueError("nonrandom replay cell carried random receipts")
    elif (
        matched_random_seed not in prereg.MATCHED_RANDOM_SEEDS
        or any(not _is_hex64(getattr(cell, name)) for name in receipt_fields)
    ):
        raise ValueError("matched-random replay receipt drift")
    _validate_artifact_hash(cell, hash_field="cell_sha256")
    return cell


def validate_entry_negative_control_panel(
    panel: Any, /
) -> EntryNegativeControlPanelV1:
    panel = _coerce_exact_dataclass(panel, EntryNegativeControlPanelV1)
    controls = tuple(prereg.entry_negative_control_policy_ids())
    channels = tuple(prereg.ENTRY_REPLAY_CHANNELS)
    if (
        panel.schema_version != panel.SCHEMA_VERSION
        or panel.holdout_caveat != prereg.HOLDOUT_CAVEAT
        or type(panel.outer_fold) is not int
        or not 1 <= panel.outer_fold <= 5
        or not _is_hex64(panel.manifest_sha256)
        or tuple(panel.required_control_ids) != controls
        or tuple(panel.channels) != channels
        or type(panel.control_bundle_sha256s) not in (tuple, list)
        or len(panel.control_bundle_sha256s) != len(_negative_control_fit_pairs())
        or any(not _is_hex64(item) for item in panel.control_bundle_sha256s)
    ):
        raise ValueError("entry negative-control panel identity drift")
    _validate_nested_shape(
        panel.control_channel_evaluations,
        (len(controls), len(channels)),
        lambda item: item is None
        or type(item) in (dict, EntryReplayEvaluationCellV1),
        name="negative-control replay matrix",
    )
    if (
        type(panel.control_action_calibration_observations) not in (tuple, list)
        or len(panel.control_action_calibration_observations) != len(controls)
        or type(panel.control_time300_trajectory_evidence) not in (tuple, list)
        or len(panel.control_time300_trajectory_evidence) != len(controls)
    ):
        raise ValueError("entry negative-control evidence matrix drift")
    hashes: list[str] = []
    for control_index, control in enumerate(controls):
        cells = panel.control_channel_evaluations[control_index]
        for channel_index, channel in enumerate(channels):
            raw = cells[channel_index]
            if raw is None:
                continue
            cell = _validate_entry_replay_cell(
                raw,
                outer_fold=panel.outer_fold,
                owner_policy_id=control,
                channel=channel,
                matched_random_seed=None,
            )
            hashes.append(cell.cell_sha256)
        action_rows = panel.control_action_calibration_observations[control_index]
        time_rows = panel.control_time300_trajectory_evidence[control_index]
        if type(action_rows) not in (tuple, list) or type(time_rows) not in (tuple, list):
            raise ValueError("entry negative-control evidence row drift")
        if cells[2] is None and (len(action_rows) != 0 or len(time_rows) != 0):
            raise ValueError("orphan negative-control action/trajectory evidence")
    if len(hashes) != len(set(hashes)):
        raise ValueError("negative-control replay cell reuse")
    _validate_artifact_hash(panel)
    return panel


def validate_entry_control_replay_result(
    result: Any, /, *, config: EntryControlReplayConfigV1,
) -> EntryControlReplayResultV1:
    config = validate_entry_control_replay_config(config)
    result = _coerce_exact_dataclass(result, EntryControlReplayResultV1)
    channels = tuple(prereg.ENTRY_REPLAY_CHANNELS)
    owners = tuple(prereg.entry_matched_random_owner_policy_ids())
    seeds = tuple(prereg.MATCHED_RANDOM_SEEDS)
    if (
        result.schema_version != result.SCHEMA_VERSION
        or result.holdout_caveat != prereg.HOLDOUT_CAVEAT
        or result.outer_fold != config.outer_fold
        or result.replay_config_sha256 != config.artifact_sha256
        or not _is_hex64(result.candidate_outer_evaluation_sha256)
        or not _is_hex64(result.neural_outer_evaluation_sha256)
        or tuple(result.channels) != channels
        or tuple(result.matched_random_owner_policy_ids) != owners
        or tuple(result.matched_random_seeds) != seeds
    ):
        raise ValueError("entry control replay result identity drift")
    for rows, owner in (
        (result.candidate_channel_evaluations, "HGB"),
        (result.p5_channel_evaluations, "P5"),
    ):
        if type(rows) not in (tuple, list) or len(rows) != len(channels):
            raise ValueError("entry control candidate/P5 channel gap")
        for channel, raw in zip(channels, rows, strict=True):
            if raw is None:
                raise ValueError("mandatory candidate/P5 replay cell absent")
            _validate_entry_replay_cell(
                raw,
                outer_fold=result.outer_fold,
                owner_policy_id=owner,
                channel=channel,
                matched_random_seed=None,
            )
    dimensions = (len(owners), len(seeds), len(channels))
    _validate_nested_shape(
        result.matched_random_channel_evaluations,
        dimensions,
        lambda item: item is None
        or type(item) in (dict, EntryReplayEvaluationCellV1),
        name="matched-random replay matrix",
    )
    receipt_names = (
        "candidate_budget_sha256s",
        "ordered_schedule_sha256s",
        "structural_skip_trace_sha256s",
        "realized_intents_and_fills_sha256s",
    )
    for name in receipt_names:
        _validate_nested_shape(
            getattr(result, name),
            dimensions,
            lambda item: item is None or _is_hex64(item),
            name=name,
        )
    cell_hashes: list[str] = []
    for rows in (
        result.candidate_channel_evaluations,
        result.p5_channel_evaluations,
    ):
        for raw in rows:
            cell_hashes.append(
                _coerce_exact_dataclass(raw, EntryReplayEvaluationCellV1).cell_sha256
            )
    cell_fields = (
        "candidate_budget_sha256",
        "ordered_schedule_sha256",
        "structural_skip_trace_sha256",
        "realized_intents_and_fills_sha256",
    )
    for owner_index, owner in enumerate(owners):
        for seed_index, seed in enumerate(seeds):
            for channel_index, channel in enumerate(channels):
                raw = result.matched_random_channel_evaluations[owner_index][seed_index][channel_index]
                cell = None
                if raw is not None:
                    cell = _validate_entry_replay_cell(
                        raw,
                        outer_fold=result.outer_fold,
                        owner_policy_id=owner,
                        channel=channel,
                        matched_random_seed=seed,
                    )
                    cell_hashes.append(cell.cell_sha256)
                for matrix_name, cell_field in zip(
                    receipt_names, cell_fields, strict=True
                ):
                    observed = getattr(result, matrix_name)[owner_index][seed_index][channel_index]
                    expected = None if cell is None else getattr(cell, cell_field)
                    if observed != expected:
                        raise ValueError("matched-random receipt matrix drift")
    if len(cell_hashes) != len(set(cell_hashes)):
        raise ValueError("entry control replay cell reuse")
    if (
        type(result.candidate_action_calibration_observations) not in (tuple, list)
        or type(result.candidate_time300_trajectory_evidence) not in (tuple, list)
    ):
        raise ValueError("entry control replay evidence row drift")
    _validate_artifact_hash(result)
    return result


def _entry_available_horizons(inputs: EntryModelInputV1) -> tuple[str, ...]:
    terminal = datetime.fromisoformat(
        f"{inputs.session}T15:55:00"
    ).replace(tzinfo=ZoneInfo("America/New_York"))
    terminal_ns = int(terminal.timestamp() * 1_000_000_000)
    arrival_ns = inputs.decision_time_ns + 60_000_000_000
    if arrival_ns >= terminal_ns:
        return ()
    result = tuple(
        f"h{minutes}"
        for minutes in (10, 20, 45, 90)
        if arrival_ns + minutes * 60_000_000_000 <= terminal_ns
    )
    return (*result, "remaining_session")


def _journal_member(value: Any, name: str) -> Any:
    if type(value) is dict:
        return value.get(name)
    return getattr(value, name, None)


def _validate_composer_for_model(
    bundle: Any, model: EntryModelBundleV1
) -> EntryComposerBundleV1:
    bundle = _coerce_exact_dataclass(bundle, EntryComposerBundleV1)
    if (
        bundle.schema_version != bundle.SCHEMA_VERSION
        or bundle.model_artifact_sha256 != model.artifact_sha256
        or not _is_hex64(bundle.calibration_artifact_sha256)
        or bundle.outer_fold != model.outer_fold
        or bundle.inner_fold != model.inner_fold
        or bundle.composer_spec_sha256
        != prereg.stable_hash(prereg.entry_composer_spec())
        or type(bundle.payload) is not dict
    ):
        raise ValueError("entry composer/model binding drift")
    _validate_artifact_hash(bundle)
    return bundle


def _prediction_components(
    prediction: EntryPredictionV1, composer: EntryComposerBundleV1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(prediction.prediction_values, dtype=np.float64)
    if values.shape != (42, 40) or not np.isfinite(values).all():
        raise ValueError("entry composer prediction tensor drift")
    means = values[:, :20].copy()
    q10 = values[:, 20:].copy()
    mean_offset = np.asarray(
        composer.payload.get("mean_lcb_additive_offsets", 0.0), dtype=np.float64
    )
    q10_offset = np.asarray(
        composer.payload.get("q10_additive_offsets", 0.0), dtype=np.float64
    )
    try:
        means = means + mean_offset
        q10 = q10 + q10_offset
    except ValueError as exc:
        raise ValueError("entry composer correction shape drift") from exc
    if means.shape != (42, 20) or q10.shape != (42, 20):
        raise ValueError("entry composer corrected tensor drift")
    dollar_columns = tuple(
        column for horizon in range(5) for column in (4 * horizon, 4 * horizon + 2)
    )
    return (
        means[:, dollar_columns],
        means[:, tuple(column + 1 for column in dollar_columns)],
        q10[:, dollar_columns],
        q10[:, tuple(column + 1 for column in dollar_columns)],
    )


def _dynamic_entry_mask(
    *, example: Any, inputs: EntryModelInputV1, journal: Any,
) -> tuple[np.ndarray, tuple[int, ...]]:
    tip = _journal_member(journal, "tip_ledger")
    if tip is None:
        raise ValueError("entry composer journal has no causal ledger tip")
    session = _journal_member(tip, "session")
    cash = _journal_member(tip, "cash_micros")
    session_equity = _journal_member(tip, "session_start_equity_micros")
    realized = _journal_member(tip, "realized_session_pnl_micros")
    fee_path = _journal_member(tip, "fee_path")
    occupied = _journal_member(tip, "position") is not None
    pending = _journal_member(tip, "pending_intent_id") is not None
    if (
        session != inputs.session
        or any(type(value) is not int for value in (cash, session_equity, realized, fee_path))
        or fee_path not in {3, 4}
        or session_equity <= 0
    ):
        raise ValueError("entry composer causal account state drift")
    facts = getattr(example, "action_execution_facts", None)
    if type(facts) not in (tuple, list) or len(facts) != 42:
        raise ValueError("entry composer execution-fact matrix drift")
    bids = tuple(getattr(fact, "bid_micros", None) for fact in facts)
    asks = tuple(getattr(fact, "ask_micros", None) for fact in facts)
    if any(type(value) is not int for value in (*bids, *asks)):
        raise ValueError("entry composer quote type drift")
    complete_ladder = all(
        bid >= 0 and ask > 0 and bid <= ask for bid, ask in zip(bids, asks, strict=True)
    )
    local = datetime.fromtimestamp(
        inputs.decision_time_ns / 1_000_000_000,
        tz=ZoneInfo("UTC"),
    ).astimezone(ZoneInfo("America/New_York"))
    before_cutoff = (local.hour, local.minute, local.second) < (15, 30, 0)
    budget = math.floor(0.05 * session_equity)
    stopped = realized <= -budget
    round_trip_fee = fee_path * 1_000_000
    mask = np.zeros(42, dtype=np.bool_)
    hard_limits: list[int] = []
    for index, (bid, ask) in enumerate(zip(bids, asks, strict=True)):
        hard_limit = ask + (50_000 if ask < 3_000_000 else 100_000)
        hard_limits.append(hard_limit)
        total_cost = hard_limit * 100 + round_trip_fee
        d48 = total_cost <= budget
        d49 = max(0, -realized) + total_cost <= budget
        affordable = total_cost <= max(0, cash)
        executable = bid > 0 and ask > bid and ask >= 1_000_000
        mask[index] = bool(
            complete_ladder
            and before_cutoff
            and not occupied
            and not pending
            and not stopped
            and executable
            and affordable
            and d48
            and d49
        )
    return mask, tuple(hard_limits)


def compose_entry_action(
    bundle: EntryComposerBundleV1, model_bundle: EntryModelBundleV1,
    example: Any, /, *, journal: Any, dataset: Any, authorization: Any,
    fill_law: Any,
) -> EntryActionDecisionV1:
    """Compose one causal WAIT/ENTER decision from sealed model and ledger state."""

    model = validate_entry_model_bundle(model_bundle)
    composer = _validate_composer_for_model(bundle, model)
    current = prereg.assert_entry_evidence_authorization_current(authorization)
    from v4.path_d.execution import research_replay
    from v4.research.pathd_entry_dataset import validate_entry_evidence_dataset

    validated_dataset = validate_entry_evidence_dataset(
        dataset, authorization=current
    )
    active_journal = research_replay.validate_research_ledger_journal(
        journal, authorization=current
    )
    if (
        composer.outer_fold != current.outer_fold
        or getattr(validated_dataset, "authorization_sha256", None)
        != prereg.stable_hash(current.to_dict())
    ):
        raise ValueError("entry composer evidence authority drift")
    example_hash = getattr(example, "canonical_sha256", lambda: None)()
    matching = [
        item
        for item in validated_dataset.examples
        if item.canonical_sha256() == example_hash
    ]
    if len(matching) != 1 or matching[0] is not example:
        raise ValueError("entry composer example is not the exact sealed member")
    if (
        getattr(fill_law, "schema_version", None) != "pathd.research_fill_law.v1"
        or getattr(fill_law, "fill_law_hash", None)
        != prereg.preregistration_payload()[0]["fill_law"]["fill_law_hash"]
    ):
        raise ValueError("entry composer fill-law drift")
    inputs = model_input_from_example(example)
    prediction = (
        predict_entry(model, inputs)
        if "fixed_prediction_values" in model.payload
        else _predict_entry_from_dataset_cache(model, inputs, validated_dataset)
    )
    components = _prediction_components(prediction, composer)
    available_horizons = _entry_available_horizons(inputs)
    available_mask = np.asarray(
        [
            horizon in available_horizons
            for horizon in ("h10", "h20", "h45", "h90", "remaining_session")
        ],
        dtype=np.bool_,
    )
    dynamic_mask, hard_limits = _dynamic_entry_mask(
        example=example, inputs=inputs, journal=active_journal
    )
    if available_horizons:
        statistics = compose_enter_statistics(
            mean_dollars=components[0],
            mean_returns=components[1],
            q10_dollars=components[2],
            q10_returns=components[3],
            available=available_mask,
        )
    else:
        statistics = {
            name: np.full(42, np.nan, dtype=np.float64)
            for name in (
                "mean_lcb_dollars", "mean_lcb_return", "q10_dollars", "q10_return"
            )
        }
        dynamic_mask[:] = False
    facts = tuple(example.action_execution_facts)
    tie_keys = tuple(
        (
            abs(float(inputs.current_offsets[index])),
            float(inputs.current_offsets[index]),
            str(inputs.current_rights[index]),
            int(str(fact.contract.expiry).replace("-", "")),
            fact.contract.strike_milli,
            fact.source_neutral_contract_id,
        )
        for index, fact in enumerate(facts)
    )
    selected = select_entry_action_index(
        mean_lcb_dollars=statistics["mean_lcb_dollars"],
        mean_lcb_returns=statistics["mean_lcb_return"],
        q10_dollars=statistics["q10_dollars"],
        q10_returns=statistics["q10_return"],
        physical_action_mask=inputs.physical_action_mask,
        dynamic_account_mask=dynamic_mask,
        tie_break_keys=tie_keys,
    )
    index = selected["selected_action_index"]
    fact = None if index is None else facts[index]
    legal_indexes = np.flatnonzero(
        inputs.physical_action_mask & dynamic_mask
    ).tolist()
    if index is None:
        decision_mean_dollars = max(
            0.0,
            max(
                (float(statistics["mean_lcb_dollars"][item]) for item in legal_indexes),
                default=0.0,
            ),
        )
        decision_mean_return = max(
            0.0,
            max(
                (float(statistics["mean_lcb_return"][item]) for item in legal_indexes),
                default=0.0,
            ),
        )
        raw_wait_lower = max(
            0.0,
            max(
                (float(statistics["q10_dollars"][item]) for item in legal_indexes),
                default=0.0,
            ),
        )
        wait_correction = composer.payload.get(
            "wait_composite_q10_correction", 0.0
        )
        if type(wait_correction) not in (int, float) or not math.isfinite(
            float(wait_correction)
        ):
            raise ValueError("WAIT composite calibration correction drift")
        decision_q10_dollars = max(
            0.0, raw_wait_lower + float(wait_correction)
        )
        decision_q10_return = max(
            0.0,
            max(
                (float(statistics["q10_return"][item]) for item in legal_indexes),
                default=0.0,
            ),
        )
    else:
        enter_correction = composer.payload.get(
            "enter_composite_q10_correction", 0.0
        )
        if type(enter_correction) not in (int, float) or not math.isfinite(
            float(enter_correction)
        ):
            raise ValueError("ENTER composite calibration correction drift")
        decision_mean_dollars = float(statistics["mean_lcb_dollars"][index])
        decision_mean_return = float(statistics["mean_lcb_return"][index])
        decision_q10_dollars = (
            float(statistics["q10_dollars"][index]) + float(enter_correction)
        )
        decision_q10_return = float(statistics["q10_return"][index])
    semantic = {
        "schema_version": EntryActionDecisionV1.SCHEMA_VERSION,
        "action": selected["action"],
        "reason": "NO_LEGAL_POSITIVE_MEAN_LCB_ACTION" if index is None else "STRICT_MEAN_LCB_GATE_AND_Q10_RANK",
        "selected_action_index": index,
        "selected_source_neutral_contract_id": None if fact is None else fact.source_neutral_contract_id,
        "selected_contract": None if fact is None else fact.contract,
        "reference_bid_micros": None if fact is None else fact.bid_micros,
        "reference_ask_micros": None if fact is None else fact.ask_micros,
        "buy_hard_limit_micros": None if index is None else hard_limits[index],
        "available_horizons": tuple(available_horizons),
        "mean_lcb_dollars": decision_mean_dollars,
        "mean_lcb_return": decision_mean_return,
        "q10_dollars": decision_q10_dollars,
        "q10_return": decision_q10_return,
        "physical_action_mask": tuple(bool(value) for value in inputs.physical_action_mask),
        "dynamic_account_mask": tuple(bool(value) for value in dynamic_mask),
        "combined_action_mask": tuple(bool(value) for value in selected["combined_action_mask"]),
        "authorization_sha256": prereg.stable_hash(current.to_dict()),
        "dataset_sha256": validated_dataset.dataset_sha256,
        "example_sha256": example_hash,
        "model_input_sha256": inputs.canonical_sha256(),
        "prediction_sha256": prediction.prediction_sha256,
        "composer_sha256": composer.artifact_sha256,
        "prior_journal_root_sha256": active_journal.journal_root_sha256,
        "fill_law_hash": fill_law.fill_law_hash,
    }
    return EntryActionDecisionV1(
        **semantic,
        decision_sha256=prereg.stable_hash(_canonical(semantic)),
    )


_P5_OBJECTIVE_SPEC_PATH = (
    prereg.REPO_ROOT
    / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/objective_spec.json"
)
_P5_OBJECTIVE_SPEC_SHA256 = (
    "3b5148c79a52977b2d849f4185e5f5dd676604357b5e5d762bc4084ed77f41f0"
)


def compose_p5_under_cap_action(
    example: Any, /, *, journal: Any, dataset: Any, authorization: Any,
    fill_law: Any,
) -> EntryActionDecisionV1:
    """Compute the frozen P5 VWAP-side nearest-eligible comparator decision."""

    from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES
    from v4.path_d.execution import research_replay
    from v4.research.pathd_entry_dataset import validate_entry_evidence_dataset

    if prereg.sha256_path(_P5_OBJECTIVE_SPEC_PATH) != _P5_OBJECTIVE_SPEC_SHA256:
        raise RuntimeError("P5 objective authority byte drift")
    authority = json.loads(_P5_OBJECTIVE_SPEC_PATH.read_text(encoding="utf-8"))
    spec = authority.get("p5_under_cap_algorithm")
    if type(spec) is not dict or spec.get("name") != "P5_VWAP_SIDE_NEAREST_ELIGIBLE_UNDER_CAP_V2":
        raise RuntimeError("P5 objective authority semantic drift")
    current = prereg.assert_entry_evidence_authorization_current(authorization)
    validated_dataset = validate_entry_evidence_dataset(dataset, authorization=current)
    active_journal = research_replay.validate_research_ledger_journal(
        journal, authorization=current
    )
    example_hash = getattr(example, "canonical_sha256", lambda: None)()
    members = [
        row for row in validated_dataset.examples
        if row.canonical_sha256() == example_hash
    ]
    if len(members) != 1 or members[0] is not example:
        raise ValueError("P5 example is not the exact sealed dataset member")
    if (
        getattr(fill_law, "fill_law_hash", None)
        != prereg.preregistration_payload()[0]["fill_law"]["fill_law_hash"]
    ):
        raise ValueError("P5 fill-law drift")
    inputs = model_input_from_example(example)
    dynamic_mask, hard_limits = _dynamic_entry_mask(
        example=example, inputs=inputs, journal=active_journal
    )
    gap_index = tuple(FEATURE_NAMES).index("spx_vwap_gap_points")
    flattened = np.asarray(inputs.signed17_frame.values, dtype=np.float64).reshape(
        42, len(FEATURE_NAMES)
    )
    gaps = flattened[:, gap_index]
    finite_gaps = gaps[np.isfinite(gaps)]
    side = None if len(finite_gaps) == 0 else ("C" if float(finite_gaps[0]) >= 0.0 else "P")
    if len(finite_gaps) and not np.all(finite_gaps == finite_gaps[0]):
        raise ValueError("P5 SPX/VWAP state differs across the current ladder")
    facts = tuple(example.action_execution_facts)
    combined = inputs.physical_action_mask & dynamic_mask
    candidates = [] if side is None else [
        index for index in np.flatnonzero(combined).tolist()
        if str(inputs.current_rights[index]) == side
    ]
    selected_index = None
    if candidates:
        selected_index = min(
            candidates,
            key=lambda index: (
                abs(float(inputs.current_offsets[index])),
                float(inputs.current_offsets[index]),
                0 if str(inputs.current_rights[index]) == "C" else 1,
                int(str(facts[index].contract.expiry).replace("-", "")),
                int(facts[index].contract.strike_milli),
                str(facts[index].contract.right),
                str(facts[index].source_neutral_contract_id),
            ),
        )
    fact = None if selected_index is None else facts[selected_index]
    available_horizons = _entry_available_horizons(inputs)
    action = "ENTER" if selected_index is not None else "WAIT"
    reason = (
        "P5_VWAP_SIDE_NEAREST_ELIGIBLE_UNDER_CAP_V2"
        if selected_index is not None
        else (
            "P5_WAIT_MISSING_SPX_VWAP"
            if side is None
            else "P5_WAIT_NO_SELECTED_SIDE_ELIGIBLE"
        )
    )
    authority_hash = prereg.stable_hash(
        {
            "objective_spec_sha256": _P5_OBJECTIVE_SPEC_SHA256,
            "algorithm": spec,
            "side": side,
        }
    )
    semantic = {
        "schema_version": EntryActionDecisionV1.SCHEMA_VERSION,
        "action": action,
        "reason": reason,
        "selected_action_index": selected_index,
        "selected_source_neutral_contract_id": None if fact is None else fact.source_neutral_contract_id,
        "selected_contract": None if fact is None else fact.contract,
        "reference_bid_micros": None if fact is None else fact.bid_micros,
        "reference_ask_micros": None if fact is None else fact.ask_micros,
        "buy_hard_limit_micros": None if selected_index is None else hard_limits[selected_index],
        "available_horizons": available_horizons,
        "mean_lcb_dollars": 0.0,
        "mean_lcb_return": 0.0,
        "q10_dollars": 0.0,
        "q10_return": 0.0,
        "physical_action_mask": tuple(bool(value) for value in inputs.physical_action_mask),
        "dynamic_account_mask": tuple(bool(value) for value in dynamic_mask),
        "combined_action_mask": tuple(bool(value) for value in combined),
        "authorization_sha256": prereg.stable_hash(current.to_dict()),
        "dataset_sha256": validated_dataset.dataset_sha256,
        "example_sha256": example_hash,
        "model_input_sha256": inputs.canonical_sha256(),
        "prediction_sha256": authority_hash,
        "composer_sha256": _P5_OBJECTIVE_SPEC_SHA256,
        "prior_journal_root_sha256": active_journal.journal_root_sha256,
        "fill_law_hash": fill_law.fill_law_hash,
    }
    return EntryActionDecisionV1(
        **semantic, decision_sha256=prereg.stable_hash(_canonical(semantic))
    )


def compose_exit_action_from_calibrated_aref(
    *, mean_lcb_aref: float, q10_aref: float, q50_aref: float, q90_aref: float,
) -> dict[str, Any]:
    """Apply the frozen corrected-v3.1 HOLD/EXIT geometry or abstain invalid."""

    topology = prereg.preregistration_payload()[0]["calibration_and_statistics"][
        "aref_decision_critical_topology"
    ]
    composer = prereg.exit_action_composer_spec()
    values = (mean_lcb_aref, q10_aref, q50_aref, q90_aref)
    valid = all(type(value) in (int, float) and math.isfinite(float(value)) for value in values)
    valid = valid and float(q10_aref) <= float(q50_aref) <= float(q90_aref)
    if not valid:
        semantic = {
            "schema_version": "pathd.exit_action_decision.v1",
            "status": "invalid_result",
            "action": None,
            "utility_hold": None,
            "reason": "INVALID_OR_NONMONOTONE_COMPLETE_AREF_OUTPUT",
            "topology_sha256": composer["topology_sha256"],
        }
        return {**semantic, "decision_sha256": prereg.stable_hash(semantic)}
    if topology["direct_action_inputs"] != ["A_ref_mean", "A_ref_q10"]:
        raise RuntimeError("A_ref direct-action topology drift")
    utility = float(mean_lcb_aref) + 0.25 * min(float(q10_aref), 0.0)
    action = "HOLD" if utility > 0.0 else "EXIT"
    semantic = {
        "schema_version": "pathd.exit_action_decision.v1",
        "status": "VALID",
        "action": action,
        "utility_hold": utility,
        "reason": "FROZEN_AREF_MEAN_Q10_GEOMETRY",
        "topology_sha256": composer["topology_sha256"],
    }
    return {**semantic, "decision_sha256": prereg.stable_hash(semantic)}
