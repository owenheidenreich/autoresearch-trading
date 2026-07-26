"""Export reconstructable selected candidates for a fair-contract model.

The script is train-result driven. If no owner-approved training result exists,
it writes a waiting packet and exits cleanly. Once a model artifact exists, it
loads only the registered manifest/split files, scores candidates, applies the
chosen threshold and cooldown policy, and exports selected candidate rows with
the contract, quote, timestamp, score, label, and feature hash needed for the
next strict serial/lifecycle replay bridge.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch

from v4.model.supervised_pilot import (
    CLASSIFIER_TARGET_MODES,
    LISTWISE_TARGET_MODES,
    CandidateMLP,
    FEATURE_TRANSFORM_NONE,
    SELECTION_MODE_CHOICES,
    SELECTION_MODE_STABLE_ABS_OFFSET_10,
    SELECTION_MODE_STABLE_ABS_OFFSET_15,
    SELECTION_MODE_STABLE_ABS_OFFSET_20,
    SELECTION_MODE_TOP_SCORE,
    FeatureScaler,
    apply_candidate_feature_transform,
    candidate_feature_vector,
)


DEFAULT_RUNNER_PLAN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_runner/runner_plan.json"
)
DEFAULT_TRAINING_RESULT = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_runner/training_result.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_selected_candidate_export"
)
SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION = "selected_candidate_export_policy_index_v8_selection_mode"
STRICT_REPLAY_STARTING_CASH = 10_000.0
STRICT_REPLAY_CONTRACT_MULTIPLIER = 100.0
STRICT_REPLAY_CASH_STRESS_PER_TRADE = 20.0


@dataclass(frozen=True)
class LoadedFairModelMember:
    model: CandidateMLP
    scaler: FeatureScaler
    target_scale: float


@dataclass(frozen=True)
class LoadedFairModel:
    model: CandidateMLP | None
    scaler: FeatureScaler | None
    target_mode: str
    target_scale: float
    threshold: float
    policy_index: int
    policy_name: str
    cooldown_minutes: int
    model_path: Path
    entry_filter: str = "none"
    min_score_margin: float = 0.0
    max_score_ceiling: float = 0.0
    max_trades_per_session: int = 0
    max_daily_loss: float = 0.0
    feature_transform: str = FEATURE_TRANSFORM_NONE
    selection_mode: str = SELECTION_MODE_TOP_SCORE
    ensemble_members: tuple[LoadedFairModelMember, ...] = ()
    model_family: str = "mlp"
    sklearn_model: Any | None = None
    sklearn_models_by_right: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-plan", type=Path, default=DEFAULT_RUNNER_PLAN)
    parser.add_argument("--training-result", type=Path, default=DEFAULT_TRAINING_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "diagnostic_test"],
        choices=("train", "validation", "diagnostic_test"),
    )
    return parser.parse_args()


def load_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path} expected list, got {type(rows).__name__}")
    return rows


def feature_hash(vector: np.ndarray) -> str:
    clean = np.nan_to_num(np.asarray(vector, dtype=np.float32), nan=0.0, posinf=8.0, neginf=-8.0)
    return hashlib.sha256(clean.tobytes()).hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    if value is None:
        return int(default)
    return int(value)


def _config_str(config: dict[str, Any], key: str, default: str) -> str:
    value = config.get(key)
    if value is None or value == "":
        return default
    return str(value)


def _selection_mode_from_config(config: dict[str, Any]) -> str:
    mode = _config_str(config, "selection_mode", SELECTION_MODE_TOP_SCORE)
    if mode not in SELECTION_MODE_CHOICES:
        raise ValueError(f"unknown selection_mode in model config: {mode}")
    return mode


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        return datetime.fromisoformat(str(value)).isoformat()
    except ValueError:
        return str(value)


def load_model(training_result: dict[str, Any]) -> LoadedFairModel:
    model_path = Path(str(training_result.get("model_out") or ""))
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact missing: {model_path}")
    state = torch.load(model_path, map_location="cpu")
    config = state.get("config") or {}
    model_family = str(state.get("model_family") or training_result.get("model_family") or "mlp")
    if model_family == "sklearn_hist_gradient_boosting":
        estimator = state.get("sklearn_model")
        if estimator is None:
            raise ValueError("sklearn model artifact missing estimator")
        return LoadedFairModel(
            model=None,
            scaler=None,
            target_mode=str(config.get("target_mode") or "regression"),
            target_scale=float(config.get("target_scale") or 100.0),
            threshold=float(training_result.get("chosen_threshold")),
            policy_index=_config_int(config, "policy_index", 1),
            policy_name=_config_str(config, "policy_name", "ask_to_bid_stop50_target100_hold25m"),
            cooldown_minutes=int(config.get("cooldown_minutes") or 25),
            model_path=model_path,
            entry_filter=str(config.get("entry_filter") or "none"),
            min_score_margin=float(config.get("min_score_margin") or 0.0),
            max_score_ceiling=float(config.get("max_score_ceiling") or 0.0),
            max_trades_per_session=int(config.get("max_trades_per_session") or 0),
            max_daily_loss=float(config.get("max_daily_loss") or 0.0),
            feature_transform=str(config.get("feature_transform") or FEATURE_TRANSFORM_NONE),
            selection_mode=_selection_mode_from_config(config),
            model_family=model_family,
            sklearn_model=estimator,
        )
    if model_family == "sklearn_hist_gradient_boosting_by_right":
        estimators_by_right = state.get("sklearn_models_by_right") or {}
        if not estimators_by_right:
            raise ValueError("right-specialized sklearn model artifact missing estimators")
        return LoadedFairModel(
            model=None,
            scaler=None,
            target_mode=str(config.get("target_mode") or "regression"),
            target_scale=float(config.get("target_scale") or 100.0),
            threshold=float(training_result.get("chosen_threshold")),
            policy_index=_config_int(config, "policy_index", 1),
            policy_name=_config_str(config, "policy_name", "ask_to_bid_stop50_target100_hold25m"),
            cooldown_minutes=int(config.get("cooldown_minutes") or 25),
            model_path=model_path,
            entry_filter=str(config.get("entry_filter") or "none"),
            min_score_margin=float(config.get("min_score_margin") or 0.0),
            max_score_ceiling=float(config.get("max_score_ceiling") or 0.0),
            max_trades_per_session=int(config.get("max_trades_per_session") or 0),
            max_daily_loss=float(config.get("max_daily_loss") or 0.0),
            feature_transform=str(config.get("feature_transform") or FEATURE_TRANSFORM_NONE),
            selection_mode=_selection_mode_from_config(config),
            model_family=model_family,
            sklearn_models_by_right=dict(estimators_by_right),
        )
    ensemble_members: list[LoadedFairModelMember] = []
    model: CandidateMLP | None = None
    scaler: FeatureScaler | None = None
    for member_state in state.get("ensemble_members") or []:
        member_config = member_state.get("config") or config
        scaler_payload = member_state.get("scaler") or {}
        member_model = CandidateMLP(
            input_dim=int(member_state.get("input_dim") or state["input_dim"]),
            hidden_dim=int(member_config.get("hidden_dim") or config.get("hidden_dim") or 96),
        )
        member_model.load_state_dict(member_state["model_state"])
        member_model.eval()
        member_scaler = FeatureScaler(
            fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
            mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
            std=np.asarray(scaler_payload["std"], dtype=np.float32),
        )
        ensemble_members.append(
            LoadedFairModelMember(
                model=member_model,
                scaler=member_scaler,
                target_scale=float(member_config.get("target_scale") or config.get("target_scale") or 100.0),
            )
        )
    if not ensemble_members:
        scaler_payload = state.get("scaler") or {}
        model = CandidateMLP(
            input_dim=int(state["input_dim"]),
            hidden_dim=int(config.get("hidden_dim") or 96),
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        scaler = FeatureScaler(
            fill=np.asarray(scaler_payload["fill"], dtype=np.float32),
            mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
            std=np.asarray(scaler_payload["std"], dtype=np.float32),
        )
    return LoadedFairModel(
        model=model,
        scaler=scaler,
        target_mode=str(config.get("target_mode") or "regression"),
        target_scale=float(config.get("target_scale") or 100.0),
        threshold=float(training_result.get("chosen_threshold")),
        policy_index=_config_int(config, "policy_index", 1),
        policy_name=_config_str(config, "policy_name", "ask_to_bid_stop50_target100_hold25m"),
        cooldown_minutes=int(config.get("cooldown_minutes") or 25),
        model_path=model_path,
        entry_filter=str(config.get("entry_filter") or "none"),
        min_score_margin=float(config.get("min_score_margin") or 0.0),
        max_score_ceiling=float(config.get("max_score_ceiling") or 0.0),
        max_trades_per_session=int(config.get("max_trades_per_session") or 0),
        max_daily_loss=float(config.get("max_daily_loss") or 0.0),
        feature_transform=str(config.get("feature_transform") or FEATURE_TRANSFORM_NONE),
        selection_mode=_selection_mode_from_config(config),
        ensemble_members=tuple(ensemble_members),
        model_family=model_family,
    )


def candidate_records_from_row(
    row: dict[str, Any],
    *,
    session: str,
    split: str,
    policy_index: int,
) -> list[dict[str, Any]]:
    mask = np.asarray(row.get("candidate_mask"), dtype=bool)
    labels = np.asarray(row.get("labels_net_pnl"), dtype=np.float32)
    if labels.ndim != 3 or policy_index >= labels.shape[2]:
        return []
    contract_ids = np.asarray(row.get("contract_ids"), dtype=object)
    offsets = np.asarray(row.get("strike_offsets"))
    rights = list(row.get("rights") or [])
    metadata = row.get("contract_quote_metadata") or {}
    records: list[dict[str, Any]] = []
    for strike_idx, right_idx in np.argwhere(mask):
        label = _safe_float(labels[int(strike_idx), int(right_idx), policy_index])
        if label is None:
            continue
        contract_id = str(contract_ids[int(strike_idx), int(right_idx)])
        vector = candidate_feature_vector(row, int(strike_idx), int(right_idx))
        quote = metadata.get(contract_id) or {}
        records.append(
            {
                "split": split,
                "session": session,
                "decision_time": _iso(row.get("decision_time")),
                "source_quote_time": _iso(row.get("source_quote_time")),
                "source_context_time": _iso(row.get("source_context_time")),
                "feature_contract_version": row.get("feature_contract_version"),
                "contract_id": contract_id,
                "strike_idx": int(strike_idx),
                "right_idx": int(right_idx),
                "right": str(rights[int(right_idx)]),
                "offset": _safe_float(offsets[int(strike_idx)]),
                "feature_hash": feature_hash(vector),
                "label_net_pnl": label,
                "entry_bid": _safe_float(quote.get("bid")),
                "entry_ask": _safe_float(quote.get("ask")),
                "entry_mid": _safe_float(quote.get("mid")),
                "quote_age_ms": _safe_float(quote.get("quote_age_ms")),
                "tradable": True,
                "_features": vector,
            }
        )
    return records


def score_candidate_records(records: list[dict[str, Any]], loaded: LoadedFairModel) -> None:
    if not records:
        return
    raw_features = np.vstack([row["_features"] for row in records]).astype(np.float32)
    features = apply_candidate_feature_transform(raw_features, loaded.feature_transform)
    for row, vector in zip(records, features):
        row["_model_features"] = vector
        row["model_feature_hash"] = feature_hash(vector)
    if loaded.model_family == "sklearn_hist_gradient_boosting":
        if loaded.sklearn_model is None:
            raise ValueError("sklearn loaded model missing estimator")
        if loaded.target_mode in CLASSIFIER_TARGET_MODES:
            proba = loaded.sklearn_model.predict_proba(features)
            classes = list(getattr(loaded.sklearn_model, "classes_", []))
            if 1 in classes:
                positive_idx = classes.index(1)
            elif 1.0 in classes:
                positive_idx = classes.index(1.0)
            else:
                positive_idx = proba.shape[1] - 1
            final_scores = np.asarray(proba[:, positive_idx], dtype=np.float32)
        else:
            final_scores = (
                np.asarray(loaded.sklearn_model.predict(features), dtype=np.float32)
                * float(loaded.target_scale)
            )
        for row, score in zip(records, final_scores):
            row["score"] = float(score)
        return
    if loaded.model_family == "sklearn_hist_gradient_boosting_by_right":
        estimators_by_right = loaded.sklearn_models_by_right or {}
        final_scores = np.full(len(records), -1e9, dtype=np.float32)
        rights = np.asarray([str(row.get("right") or "") for row in records], dtype=object)
        for right, estimator in estimators_by_right.items():
            mask = np.asarray(rights == right, dtype=bool)
            if not mask.any():
                continue
            features_right = features[mask]
            if loaded.target_mode in CLASSIFIER_TARGET_MODES:
                proba = estimator.predict_proba(features_right)
                classes = list(getattr(estimator, "classes_", []))
                if 1 in classes:
                    positive_idx = classes.index(1)
                elif 1.0 in classes:
                    positive_idx = classes.index(1.0)
                else:
                    positive_idx = proba.shape[1] - 1
                final_scores[mask] = np.asarray(proba[:, positive_idx], dtype=np.float32)
            else:
                final_scores[mask] = (
                    np.asarray(estimator.predict(features_right), dtype=np.float32)
                    * float(loaded.target_scale)
                )
        for row, score in zip(records, final_scores):
            row["score"] = float(score)
        return
    score_arrays: list[np.ndarray] = []
    members = loaded.ensemble_members
    if members:
        for member in members:
            scaled = member.scaler.transform(features)
            with torch.no_grad():
                raw_scores = member.model(torch.from_numpy(scaled))
                if loaded.target_mode in CLASSIFIER_TARGET_MODES:
                    scores = torch.sigmoid(raw_scores).cpu().numpy().astype(np.float32)
                    scale = 1.0
                elif loaded.target_mode in LISTWISE_TARGET_MODES:
                    scores = raw_scores.cpu().numpy().astype(np.float32)
                    scale = 1.0
                else:
                    scores = raw_scores.cpu().numpy().astype(np.float32)
                    scale = member.target_scale
            score_arrays.append(scores * scale)
        final_scores = np.vstack(score_arrays).mean(axis=0).astype(np.float32)
    else:
        if loaded.model is None or loaded.scaler is None:
            raise ValueError("single-model artifact missing model or scaler")
        scaled = loaded.scaler.transform(features)
        with torch.no_grad():
            raw_scores = loaded.model(torch.from_numpy(scaled))
            if loaded.target_mode in CLASSIFIER_TARGET_MODES:
                final_scores = torch.sigmoid(raw_scores).cpu().numpy().astype(np.float32)
                scale = 1.0
            elif loaded.target_mode in LISTWISE_TARGET_MODES:
                final_scores = raw_scores.cpu().numpy().astype(np.float32)
                scale = 1.0
            else:
                final_scores = raw_scores.cpu().numpy().astype(np.float32)
                scale = loaded.target_scale
            final_scores = final_scores * scale
    for row, score in zip(records, final_scores):
        row["score"] = float(score)


def _vwap_aligned_right(row: dict[str, Any]) -> str | None:
    """Return the live-causal side allowed by the row's current SPX/VWAP state."""
    try:
        market_window = np.asarray(row.get("market_window"), dtype=np.float32)
        market_last = market_window[-1]
        spx_close = float(market_last[0])
        spx_vwap = float(market_last[2])
    except (IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(spx_close) or not math.isfinite(spx_vwap) or spx_close == spx_vwap:
        return None
    return "C" if spx_close > spx_vwap else "P"


def row_above_vwap_omar_pos_after_open(row: dict[str, Any]) -> bool:
    """Mirror the live-causal favorable-context gate used by the simulator."""
    decision_time = row.get("decision_time")
    if not isinstance(decision_time, datetime):
        try:
            decision_time = datetime.fromisoformat(str(decision_time))
        except ValueError:
            return False
    local = decision_time.astimezone(ZoneInfo("America/New_York"))
    if local.hour * 60 + local.minute < 10 * 60:
        return False
    try:
        market_window = np.asarray(row.get("market_window"), dtype=np.float32)
        market_last = market_window[-1]
        spx_close = float(market_last[0])
        spx_vwap = float(market_last[2])
        omar = float(market_last[3])
    except (IndexError, TypeError, ValueError):
        return False
    return (
        math.isfinite(spx_close)
        and math.isfinite(spx_vwap)
        and math.isfinite(omar)
        and spx_close > spx_vwap
        and omar > 0.0
    )


def candidate_allowed_by_entry_filter(
    candidate: dict[str, Any],
    row: dict[str, Any],
    entry_filter: str,
) -> bool:
    """Mirror the simulator's optional live-causal first-stage entry filter."""
    if entry_filter in {"", "none"}:
        return True
    if entry_filter == "vwap_aligned":
        allowed_right = _vwap_aligned_right(row)
        return allowed_right is not None and str(candidate.get("right")) == allowed_right
    if entry_filter == "premium_floor_3":
        ask = _safe_float(candidate.get("entry_ask"))
        return ask is not None and ask >= 3.0
    if entry_filter == "near_10_20_offset":
        offset = _safe_float(candidate.get("offset"))
        return offset is not None and 10.0 <= abs(offset) <= 20.0
    if entry_filter == "put_only":
        return str(candidate.get("right")) == "P"
    if entry_filter == "put_near_10_20_offset":
        offset = _safe_float(candidate.get("offset"))
        return (
            str(candidate.get("right")) == "P"
            and offset is not None
            and 10.0 <= abs(offset) <= 20.0
        )
    if entry_filter == "put_near_after_0940":
        decision_time = row.get("decision_time")
        if not isinstance(decision_time, datetime):
            try:
                decision_time = datetime.fromisoformat(str(decision_time))
            except ValueError:
                return False
        local = decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        offset = _safe_float(candidate.get("offset"))
        return (
            minutes >= 9 * 60 + 40
            and str(candidate.get("right")) == "P"
            and offset is not None
            and 10.0 <= abs(offset) <= 20.0
        )
    if entry_filter == "put_near_after_0940_vwap_m2_10":
        decision_time = row.get("decision_time")
        if not isinstance(decision_time, datetime):
            try:
                decision_time = datetime.fromisoformat(str(decision_time))
            except ValueError:
                return False
        local = decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        if minutes < 9 * 60 + 40:
            return False
        try:
            market_window = np.asarray(row.get("market_window"), dtype=np.float32)
            market_last = market_window[-1]
            spx_close = float(market_last[0])
            spx_vwap = float(market_last[2])
        except (IndexError, TypeError, ValueError):
            return False
        if not math.isfinite(spx_close) or not math.isfinite(spx_vwap):
            return False
        vwap_gap = spx_close - spx_vwap
        offset = _safe_float(candidate.get("offset"))
        return (
            -2.0 <= vwap_gap < 10.0
            and str(candidate.get("right")) == "P"
            and offset is not None
            and 10.0 <= abs(offset) <= 20.0
        )
    if entry_filter in {
        "put_near_after_0940_vwap_m2_10_omar_neg",
        "put_near_after_0940_vwap_m2_10_range_20_45",
        "put_near_after_0940_vwap_m2_10_near_vwap",
        "put_near_after_0940_vwap_m2_10_premium_gte_7_5",
        "put_near_after_0940_vwap_m2_10_mom15_nonpos",
        "put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos",
        "put_near_after_0940_vwap_m2_10_premium_gte_7_5_mom15_nonpos",
    }:
        decision_time = row.get("decision_time")
        if not isinstance(decision_time, datetime):
            try:
                decision_time = datetime.fromisoformat(str(decision_time))
            except ValueError:
                return False
        local = decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        if minutes < 9 * 60 + 40:
            return False
        try:
            market_window = np.asarray(row.get("market_window"), dtype=np.float32)
            market_last = market_window[-1]
            spx_close = float(market_last[0])
            spx_vwap = float(market_last[2])
            omar = float(market_last[3])
            session_range = float(market_last[4])
            momentum15 = float(market_last[6])
        except (IndexError, TypeError, ValueError):
            return False
        if not math.isfinite(spx_close) or not math.isfinite(spx_vwap):
            return False
        vwap_gap = spx_close - spx_vwap
        offset = _safe_float(candidate.get("offset"))
        base_allowed = (
            -2.0 <= vwap_gap < 10.0
            and str(candidate.get("right")) == "P"
            and offset is not None
            and 10.0 <= abs(offset) <= 20.0
        )
        if not base_allowed:
            return False
        if entry_filter == "put_near_after_0940_vwap_m2_10_omar_neg":
            return math.isfinite(omar) and omar < 0.0
        if entry_filter == "put_near_after_0940_vwap_m2_10_range_20_45":
            return math.isfinite(session_range) and 20.0 <= session_range < 45.0
        if entry_filter == "put_near_after_0940_vwap_m2_10_near_vwap":
            return -2.0 <= vwap_gap <= 2.0
        if entry_filter == "put_near_after_0940_vwap_m2_10_premium_gte_7_5":
            ask = _safe_float(candidate.get("entry_ask"))
            return ask is not None and ask >= 7.5
        if entry_filter == "put_near_after_0940_vwap_m2_10_mom15_nonpos":
            return math.isfinite(momentum15) and momentum15 <= 0.0
        if entry_filter == "put_near_after_0940_vwap_m2_10_omar_pos_mom15_nonpos":
            return math.isfinite(omar) and omar > 0.0 and math.isfinite(momentum15) and momentum15 <= 0.0
        if entry_filter == "put_near_after_0940_vwap_m2_10_premium_gte_7_5_mom15_nonpos":
            ask = _safe_float(candidate.get("entry_ask"))
            return ask is not None and ask >= 7.5 and math.isfinite(momentum15) and momentum15 <= 0.0
    if entry_filter in {
        "near_after_0940_vwap_m2_10_mom15_side",
        "near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5",
    }:
        decision_time = row.get("decision_time")
        if not isinstance(decision_time, datetime):
            try:
                decision_time = datetime.fromisoformat(str(decision_time))
            except ValueError:
                return False
        local = decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        if minutes < 9 * 60 + 40:
            return False
        try:
            market_window = np.asarray(row.get("market_window"), dtype=np.float32)
            market_last = market_window[-1]
            spx_close = float(market_last[0])
            spx_vwap = float(market_last[2])
            momentum15 = float(market_last[6])
        except (IndexError, TypeError, ValueError):
            return False
        if not math.isfinite(spx_close) or not math.isfinite(spx_vwap) or not math.isfinite(momentum15):
            return False
        vwap_gap = spx_close - spx_vwap
        offset = _safe_float(candidate.get("offset"))
        allowed_right = "C" if momentum15 > 0.0 else "P"
        base_allowed = (
            -2.0 <= vwap_gap < 10.0
            and str(candidate.get("right")) == allowed_right
            and offset is not None
            and 10.0 <= abs(offset) <= 20.0
        )
        if not base_allowed:
            return False
        if entry_filter == "near_after_0940_vwap_m2_10_mom15_side_premium_gte_7_5":
            ask = _safe_float(candidate.get("entry_ask"))
            return ask is not None and ask >= 7.5
        return True
    if entry_filter == "morning_1000_1129":
        decision_time = row.get("decision_time")
        if not isinstance(decision_time, datetime):
            try:
                decision_time = datetime.fromisoformat(str(decision_time))
            except ValueError:
                return False
        local = decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        return 10 * 60 <= minutes < 11 * 60 + 30
    if entry_filter == "morning_near_10_20_offset":
        decision_time = row.get("decision_time")
        if not isinstance(decision_time, datetime):
            try:
                decision_time = datetime.fromisoformat(str(decision_time))
            except ValueError:
                return False
        local = decision_time.astimezone(ZoneInfo("America/New_York"))
        minutes = local.hour * 60 + local.minute
        offset = _safe_float(candidate.get("offset"))
        return (
            10 * 60 <= minutes < 11 * 60 + 30
            and offset is not None
            and 10.0 <= abs(offset) <= 20.0
        )
    if entry_filter == "above_vwap_omar_pos_after_open":
        return row_above_vwap_omar_pos_after_open(row)
    raise ValueError(f"unknown entry_filter: {entry_filter}")


def top_candidate_with_margin(
    candidates: list[dict[str, Any]],
    *,
    selection_mode: str = SELECTION_MODE_TOP_SCORE,
) -> tuple[dict[str, Any], float] | None:
    """Return the simulator-equivalent top candidate and runner-up margin."""
    mode = str(selection_mode or SELECTION_MODE_TOP_SCORE)
    if mode not in SELECTION_MODE_CHOICES:
        raise ValueError(f"unknown selection_mode: {mode}")
    scored = [
        (idx, candidate, score)
        for idx, candidate in enumerate(candidates)
        if (score := _safe_float(candidate.get("score"))) is not None
    ]
    if not scored:
        return None
    scores = np.asarray([score for _idx, _candidate, score in scored], dtype=np.float32)
    if mode == SELECTION_MODE_TOP_SCORE:
        order = np.argsort(scores)
        top_pos = int(order[-1])
        _idx, best, best_score = scored[top_pos]
    else:
        target = {
            SELECTION_MODE_STABLE_ABS_OFFSET_10: 10.0,
            SELECTION_MODE_STABLE_ABS_OFFSET_15: 15.0,
            SELECTION_MODE_STABLE_ABS_OFFSET_20: 20.0,
        }[mode]
        _idx, best, best_score = min(
            scored,
            key=lambda item: (
                abs(abs(float(_safe_float(item[1].get("offset")) or 0.0)) - target),
                abs(float(_safe_float(item[1].get("offset")) or 0.0)),
                float(_safe_float(item[1].get("offset")) or 0.0),
                str(item[1].get("right") or ""),
                item[0],
            ),
        )
    if len(scored) == 1:
        margin = float("inf")
    else:
        if mode == SELECTION_MODE_TOP_SCORE:
            order = np.argsort(scores)
            second_score = float(scores[int(order[-2])])
        else:
            second_score = max(
                float(score)
                for idx, _candidate, score in scored
                if idx != _idx
            )
        margin = float(best_score) - second_score
    return best, margin


def select_records_for_split(
    *,
    split: str,
    paths: list[Path],
    loaded: LoadedFairModel,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    decision_groups: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    cash = float(STRICT_REPLAY_STARTING_CASH)
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    realized_pnl_by_session: dict[str, float] = {}
    decision_rows = 0
    candidate_rows = 0
    threshold_waits = 0
    cooldown_blocks = 0
    session_trade_cap_blocks = 0
    daily_loss_stop_blocks = 0
    entry_filter_blocks = 0
    affordability_blocks = 0
    entry_filter = str(getattr(loaded, "entry_filter", "none") or "none")
    selection_mode = str(getattr(loaded, "selection_mode", SELECTION_MODE_TOP_SCORE) or SELECTION_MODE_TOP_SCORE)
    min_score_margin = float(getattr(loaded, "min_score_margin", 0.0) or 0.0)
    max_score_ceiling = max(float(getattr(loaded, "max_score_ceiling", 0.0) or 0.0), 0.0)
    max_trades_per_session = max(int(getattr(loaded, "max_trades_per_session", 0) or 0), 0)
    max_daily_loss = max(float(getattr(loaded, "max_daily_loss", 0.0) or 0.0), 0.0)
    score_margin_waits = 0
    score_ceiling_waits = 0
    for path in paths:
        session = path.name.removesuffix(".pkl")
        for row in load_rows(path):
            decision_rows += 1
            candidates = candidate_records_from_row(
                row,
                session=session,
                split=split,
                policy_index=loaded.policy_index,
            )
            candidate_rows += len(candidates)
            if not candidates:
                continue
            decision_groups.append((session, row, candidates))
    all_candidates = [
        candidate
        for _session, _row, candidates in decision_groups
        for candidate in candidates
    ]
    score_candidate_records(all_candidates, loaded)
    for session, row, candidates in decision_groups:
        decision_time = row.get("decision_time")
        if not isinstance(decision_time, datetime):
            decision_time = datetime.fromisoformat(str(decision_time))
        next_time = next_time_by_session.get(session)
        if next_time is not None and decision_time < next_time:
            cooldown_blocks += 1
            continue
        if max_trades_per_session > 0 and trades_by_session.get(session, 0) >= max_trades_per_session:
            session_trade_cap_blocks += 1
            continue
        if max_daily_loss > 0.0 and realized_pnl_by_session.get(session, 0.0) <= -max_daily_loss:
            daily_loss_stop_blocks += 1
            continue
        eligible_candidates = [
            candidate
            for candidate in candidates
            if candidate_allowed_by_entry_filter(candidate, row, entry_filter)
        ]
        if not eligible_candidates:
            entry_filter_blocks += 1
            continue
        affordable_candidates = []
        for candidate in eligible_candidates:
            ask = _safe_float(candidate.get("entry_ask"))
            if ask is None or ask <= 0.0:
                continue
            if ask * STRICT_REPLAY_CONTRACT_MULTIPLIER <= cash + 1e-9:
                affordable_candidates.append(candidate)
        if not affordable_candidates:
            affordability_blocks += 1
            continue
        top = top_candidate_with_margin(
            affordable_candidates,
            selection_mode=selection_mode,
        )
        if top is None:
            threshold_waits += 1
            continue
        best, score_margin = top
        if min_score_margin > 0.0 and score_margin < min_score_margin:
            score_margin_waits += 1
            continue
        if max_score_ceiling > 0.0 and float(best["score"]) >= max_score_ceiling:
            score_ceiling_waits += 1
            continue
        if float(best["score"]) < loaded.threshold:
            threshold_waits += 1
            continue
        best = {key: value for key, value in best.items() if key != "_features"}
        best["selected_rank"] = 1
        best["threshold"] = loaded.threshold
        best["policy_index"] = loaded.policy_index
        best["policy_name"] = loaded.policy_name
        best["cooldown_minutes"] = loaded.cooldown_minutes
        best["target_mode"] = loaded.target_mode
        best["entry_filter"] = entry_filter
        best["selection_mode"] = selection_mode
        best["min_score_margin"] = min_score_margin
        best["max_score_ceiling"] = max_score_ceiling
        best["max_trades_per_session"] = max_trades_per_session
        best["max_daily_loss"] = max_daily_loss
        best["score_margin"] = score_margin
        best["model_path"] = str(loaded.model_path)
        best["model_family"] = str(getattr(loaded, "model_family", "mlp") or "mlp")
        selected.append(best)
        trades_by_session[session] = trades_by_session.get(session, 0) + 1
        realized_pnl_by_session[session] = realized_pnl_by_session.get(session, 0.0) + float(
            best.get("label_net_pnl") or 0.0
        )
        cash += (
            float(best.get("label_net_pnl") or 0.0)
            - float(STRICT_REPLAY_CASH_STRESS_PER_TRADE)
        )
        next_time_by_session[session] = decision_time + timedelta(minutes=loaded.cooldown_minutes)
    return selected, {
        "split": split,
        "sessions": len(paths),
        "decision_rows": decision_rows,
        "candidate_rows": candidate_rows,
        "score_batch_calls": 1 if all_candidates else 0,
        "selected_entries": len(selected),
        "threshold_waits": threshold_waits,
        "cooldown_blocks": cooldown_blocks,
        "session_trade_cap_blocks": session_trade_cap_blocks,
        "daily_loss_stop_blocks": daily_loss_stop_blocks,
        "entry_filter": entry_filter,
        "selection_mode": selection_mode,
        "entry_filter_blocks": entry_filter_blocks,
        "affordability_blocks": affordability_blocks,
        "min_score_margin": min_score_margin,
        "max_score_ceiling": max_score_ceiling,
        "max_trades_per_session": max_trades_per_session,
        "max_daily_loss": max_daily_loss,
        "score_margin_waits": score_margin_waits,
        "score_ceiling_waits": score_ceiling_waits,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fields})


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Selected Candidate Export",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Model loaded: `{str(payload.get('model_loaded')).lower()}`",
        f"- Model training executed here: `false`",
        f"- Threshold tuning executed here: `false`",
        f"- Broker endpoint called: `false`",
        f"- Paper-submit allowed: `false`",
        "",
        "## Split Summary",
        "",
    ]
    for row in payload.get("split_summary") or []:
        lines.append(
            f"- `{row['split']}`: selected=`{row['selected_entries']}`, "
            f"decisions=`{row['decision_rows']}`, candidates=`{row['candidate_rows']}`, "
            f"score_batch_calls=`{row.get('score_batch_calls', 0)}`, "
            f"entry_filter=`{row.get('entry_filter', 'none')}`, "
            f"selection_mode=`{row.get('selection_mode', 'top_score')}`, "
            f"entry_filter_blocks=`{row.get('entry_filter_blocks', 0)}`, "
            f"affordability_blocks=`{row.get('affordability_blocks', 0)}`, "
            f"min_score_margin=`{row.get('min_score_margin', 0.0)}`, "
            f"score_margin_waits=`{row.get('score_margin_waits', 0)}`, "
            f"max_score_ceiling=`{row.get('max_score_ceiling', 0.0)}`, "
            f"score_ceiling_waits=`{row.get('score_ceiling_waits', 0)}`."
        )
    if payload.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in payload["blockers"])
    lines.extend(["", "## Outputs", ""])
    for key, value in (payload.get("outputs") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def build_waiting_payload(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "Protocol101FairContractSelectedCandidateExportV1",
        "implementation_version": SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION,
        "status": "waiting_for_owner_approved_training_result",
        "decision": "no_candidate_model_available_for_selected_entry_export",
        "model_loaded": False,
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "split_summary": [],
        "blockers": [reason],
        "outputs": {},
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner_plan = load_json_optional(args.runner_plan)
    training_result = load_json_optional(args.training_result)
    if not runner_plan:
        payload = build_waiting_payload("runner_plan_missing")
    elif not training_result:
        payload = build_waiting_payload("training_result_missing")
    else:
        loaded = load_model(training_result)
        split_files = runner_plan.get("split_files") or {}
        all_selected: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for split in args.splits:
            paths = [Path(path) for path in split_files.get(split, [])]
            selected, summary = select_records_for_split(
                split=split,
                paths=paths,
                loaded=loaded,
            )
            all_selected.extend(selected)
            summaries.append(summary)
        selected_csv = args.out_dir / "selected_candidates.csv"
        selected_jsonl = args.out_dir / "selected_candidates.jsonl"
        write_csv(selected_csv, all_selected)
        with selected_jsonl.open("w") as handle:
            for row in all_selected:
                public = {key: value for key, value in row.items() if not key.startswith("_")}
                handle.write(json.dumps(public, sort_keys=True) + "\n")
        payload = {
            "schema_version": "Protocol101FairContractSelectedCandidateExportV1",
            "implementation_version": SELECTED_CANDIDATE_EXPORT_IMPLEMENTATION_VERSION,
            "status": "pass",
            "decision": "selected_candidates_exported_for_strict_replay_bridge",
            "model_loaded": True,
            "model_training_executed_here": False,
            "threshold_tuning_executed_here": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "policy_index": loaded.policy_index,
            "policy_name": loaded.policy_name,
            "threshold": loaded.threshold,
            "cooldown_minutes": loaded.cooldown_minutes,
            "model_family": loaded.model_family,
            "feature_transform": loaded.feature_transform,
            "entry_filter": loaded.entry_filter,
            "min_score_margin": loaded.min_score_margin,
            "max_score_ceiling": loaded.max_score_ceiling,
            "split_summary": summaries,
            "blockers": [],
            "outputs": {
                "selected_candidates_csv": str(selected_csv),
                "selected_candidates_jsonl": str(selected_jsonl),
            },
        }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "report.md").write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
