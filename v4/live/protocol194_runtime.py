"""No-order runtime parity and latency helpers for Protocol194.

Protocol194 is a research challenger, not the live-paper default. The helpers
in this module are deliberately broker-safe: they validate the live/training
candidate surface, run a live-safe inference mask, and emit JSONL-compatible
events that cannot carry order submission fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import time
from typing import Any

import numpy as np
import torch

from v4.live.protocol166_parity_contract import validate_candidate


SCHEMA_VERSION = "protocol194_runtime_v1"
PROTOCOL_ID = "protocol194_challenger"
EVENT_TYPES = ("candidate_set", "model_decision", "latency_probe", "risk_gate")
ACTIONS = ("enter", "wait", "blocked")
CONTRACT_MULTIPLIER = 100.0


@dataclass(frozen=True)
class Protocol194LatencyBudget:
    """Runtime budget required before Protocol194 can replace Protocol101."""

    max_total_decision_ms: float = 1_000.0
    max_candidate_validation_ms: float = 250.0
    max_model_inference_ms: float = 250.0
    max_option_quote_age_ms: float = 1_500.0
    max_context_age_ms: float = 5_000.0
    max_contracts: int = 1
    max_concurrent_positions: int = 1


@dataclass
class RuntimeValidationResult:
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_runtime_event(row: dict[str, Any], *, row_index: int = 0) -> RuntimeValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "schema_version",
        "protocol_id",
        "event_type",
        "timestamp",
        "session",
        "live_orders_enabled",
        "broker_endpoint_called",
        "selected_action",
        "latency",
        "candidate_set",
        "model_decision",
        "risk_gate",
        "paper_account_state",
    )
    for key in required:
        if key not in row:
            errors.append(f"row {row_index}: missing {key}")
    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"row {row_index}: schema_version must be {SCHEMA_VERSION}")
    if row.get("protocol_id") != PROTOCOL_ID:
        errors.append(f"row {row_index}: protocol_id must be {PROTOCOL_ID}")
    if row.get("event_type") not in EVENT_TYPES:
        errors.append(f"row {row_index}: invalid event_type {row.get('event_type')!r}")
    if row.get("selected_action") not in ACTIONS:
        errors.append(f"row {row_index}: invalid selected_action {row.get('selected_action')!r}")
    if row.get("live_orders_enabled") is not False:
        errors.append(f"row {row_index}: live_orders_enabled must be false")
    if row.get("broker_endpoint_called") is not False:
        errors.append(f"row {row_index}: broker_endpoint_called must be false")
    for key in ("order_intent", "submitted_order", "broker_order", "broker_order_id", "order_id"):
        if _truthy(row.get(key)):
            errors.append(f"row {row_index}: broker/order field {key!r} must be absent or null")

    candidate_set = _object(row.get("candidate_set"))
    candidate_count = _number(candidate_set.get("candidate_count"))
    valid_count = _number(candidate_set.get("valid_candidate_count"))
    if candidate_count is None or candidate_count <= 0:
        errors.append(f"row {row_index}: candidate_set.candidate_count must be positive")
    if valid_count is None or valid_count < 0:
        errors.append(f"row {row_index}: candidate_set.valid_candidate_count is required")
    if valid_count is not None and candidate_count is not None and valid_count > candidate_count:
        errors.append(f"row {row_index}: valid candidates cannot exceed total candidates")
    if candidate_set.get("root") != "SPXW":
        errors.append(f"row {row_index}: candidate_set.root must be SPXW")
    if candidate_set.get("settlement_style") != "PM":
        errors.append(f"row {row_index}: candidate_set.settlement_style must be PM")
    if _number(candidate_set.get("max_abs_offset")) is None:
        errors.append(f"row {row_index}: candidate_set.max_abs_offset is required")

    latency = _object(row.get("latency"))
    for key in ("candidate_validation_ms", "model_inference_ms", "total_decision_ms"):
        value = _number(latency.get(key))
        if value is None or value < 0:
            errors.append(f"row {row_index}: latency.{key} must be non-negative")
    if latency.get("budget_passed") is not True:
        warnings.append(f"row {row_index}: latency budget did not pass")

    account = _object(row.get("paper_account_state"))
    if _number(account.get("cash_available")) is None:
        errors.append(f"row {row_index}: paper_account_state.cash_available is required")
    if int(_number(account.get("max_contracts"), -1.0) or -1) != 1:
        errors.append(f"row {row_index}: max_contracts must remain 1 for this challenger gate")

    selected = row.get("selected_contract")
    if row.get("selected_action") == "enter":
        if not isinstance(selected, dict):
            errors.append(f"row {row_index}: enter requires selected_contract")
        else:
            selected_result = validate_candidate(selected, _account_for_protocol166(account))
            if selected_result["status"] != "pass":
                errors.extend(f"row {row_index}: selected_contract:{error}" for error in selected_result["errors"])
    elif selected:
        warnings.append(f"row {row_index}: selected_contract present for non-enter action")

    gate = _object(row.get("risk_gate"))
    if "passed" not in gate:
        errors.append(f"row {row_index}: risk_gate.passed is required")
    if row.get("selected_action") == "blocked" and not gate.get("reason"):
        errors.append(f"row {row_index}: blocked action requires risk_gate.reason")

    return RuntimeValidationResult("pass" if not errors else "fail", errors, warnings)


def validate_runtime_stream(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = [validate_runtime_event(row, row_index=index) for index, row in enumerate(rows)]
    errors = [error for result in results for error in result.errors]
    warnings = [warning for result in results for warning in result.warnings]
    event_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in rows:
        event_counts[str(row.get("event_type", "missing"))] = event_counts.get(str(row.get("event_type", "missing")), 0) + 1
        action_counts[str(row.get("selected_action", "missing"))] = action_counts.get(str(row.get("selected_action", "missing")), 0) + 1
    return {
        "status": "pass" if not errors else "fail",
        "rows": len(rows),
        "errors": errors,
        "warnings": warnings,
        "event_counts": event_counts,
        "action_counts": action_counts,
    }


def live_candidate_mask(candidates: Any, *, account_state: dict[str, Any]) -> np.ndarray:
    """Mask candidates using only fields that can exist live."""

    cash_value = _number(account_state.get("cash_available"), 0.0)
    open_position_value = _number(account_state.get("open_position_count"), 99.0)
    cash = 0.0 if cash_value is None else cash_value
    open_positions = 99 if open_position_value is None else int(open_position_value)
    mask = []
    for _, row in candidates.iterrows():
        candidate = candidate_from_row(row)
        result = validate_candidate(candidate, _account_for_protocol166(account_state))
        affordable = float(candidate.get("entry_premium", math.inf)) <= cash
        concurrency_ok = open_positions < int(account_state.get("max_concurrent_positions", 1))
        mask.append(result["status"] == "pass" and affordable and concurrency_ok)
    return np.asarray(mask, dtype=bool)


def candidate_from_row(row: Any) -> dict[str, Any]:
    bid = _finite(row.get("entry_bid"))
    ask = _finite(row.get("entry_ask"))
    mid = _finite(row.get("entry_mid"), (bid + ask) / 2.0 if math.isfinite(bid) and math.isfinite(ask) else math.nan)
    spread = _finite(row.get("entry_spread"), ask - bid if math.isfinite(bid) and math.isfinite(ask) else math.nan)
    return {
        "decision_time": str(row.get("decision_time", "")),
        "contract_id": str(row.get("contract_id", "")),
        "root": str(row.get("root", "")),
        "settlement_style": str(row.get("settlement_style", "")),
        "right": str(row.get("right", "")),
        "offset": _finite(row.get("offset")),
        "entry_bid": bid,
        "entry_ask": ask,
        "entry_mid": mid,
        "entry_spread": spread,
        "entry_bid_size": _finite(row.get("entry_bid_size")),
        "entry_ask_size": _finite(row.get("entry_ask_size")),
        "entry_premium": _finite(row.get("entry_premium"), ask * CONTRACT_MULTIPLIER if math.isfinite(ask) else math.nan),
        "entry_delta": _finite(row.get("entry_delta")),
        "entry_gamma": _finite(row.get("entry_gamma")),
        "entry_theta": _finite(row.get("entry_theta")),
        "entry_iv": _finite(row.get("entry_iv")),
    }


def latency_passed(latency: dict[str, Any], budget: Protocol194LatencyBudget = Protocol194LatencyBudget()) -> bool:
    return (
        (_number(latency.get("candidate_validation_ms"), math.inf) or math.inf) <= budget.max_candidate_validation_ms
        and (_number(latency.get("model_inference_ms"), math.inf) or math.inf) <= budget.max_model_inference_ms
        and (_number(latency.get("total_decision_ms"), math.inf) or math.inf) <= budget.max_total_decision_ms
    )


def build_runtime_event(
    *,
    session: str,
    timestamp: str,
    selected_action: str,
    candidate_set: dict[str, Any],
    model_decision: dict[str, Any],
    latency: dict[str, Any],
    risk_gate: dict[str, Any],
    paper_account_state: dict[str, Any],
    selected_contract: dict[str, Any] | None = None,
    event_type: str = "model_decision",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "event_type": event_type,
        "timestamp": timestamp,
        "session": session,
        "live_orders_enabled": False,
        "broker_endpoint_called": False,
        "selected_action": selected_action,
        "candidate_set": candidate_set,
        "model_decision": model_decision,
        "latency": latency,
        "risk_gate": risk_gate,
        "paper_account_state": paper_account_state,
        "selected_contract": selected_contract,
        "operational_default": "protocol101",
        "challenger_status": "no_order_runtime_parity_only",
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def _account_for_protocol166(account_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "account_equity": account_state.get("account_equity", account_state.get("cash_available", 0.0)),
        "cash_available": account_state.get("cash_available", 0.0),
        "open_position_count": account_state.get("open_position_count", account_state.get("open_positions", 0)),
        "max_concurrent_positions": account_state.get("max_concurrent_positions", 1),
        "max_contracts": account_state.get("max_contracts", 1),
    }


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _truthy(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return False
    return True


def _number(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _finite(value: Any, default: float = math.nan) -> float:
    number = _number(value, default)
    return number if number is not None else default
