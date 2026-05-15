"""Protocol 101 no-order live shadow event schema.

This module validates the expanded Tuesday shadow stream. It is intentionally
broker-safe: a valid row cannot carry live order submission intent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


SHADOW_SCHEMA_VERSION = "protocol101_shadow_v2"
EVENT_TYPES = (
    "market_snapshot",
    "candidate_set",
    "model_decision",
    "risk_gate",
    "paper_account_state",
    "exit_decision",
)
ACTIONS = ("enter", "wait", "hold", "exit", "blocked")
NO_ORDER_MODE = "no_order_shadow"
REQUIRED_TOP_LEVEL = (
    "schema_version",
    "protocol_id",
    "event_type",
    "timestamp",
    "session",
    "live_orders_enabled",
    "broker_endpoint_called",
    "market_snapshot",
    "model_decision",
    "selected_action",
    "risk_gate",
    "paper_account_state",
)


@dataclass
class SchemaValidationResult:
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_shadow_event(
    row: dict[str, Any],
    *,
    row_index: int = 0,
    max_contracts_per_position: int = 1,
) -> SchemaValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    for key in REQUIRED_TOP_LEVEL:
        if key not in row:
            errors.append(f"row {row_index}: missing {key}")

    if row.get("schema_version") != SHADOW_SCHEMA_VERSION:
        errors.append(f"row {row_index}: schema_version must be {SHADOW_SCHEMA_VERSION}")
    if row.get("protocol_id") != "protocol101":
        errors.append(f"row {row_index}: protocol_id must be protocol101")
    if row.get("event_type") not in EVENT_TYPES:
        errors.append(f"row {row_index}: event_type {row.get('event_type')!r} is invalid")
    if row.get("live_orders_enabled") is not False:
        errors.append(f"row {row_index}: live_orders_enabled must be false")
    if row.get("broker_endpoint_called") is not False:
        errors.append(f"row {row_index}: broker_endpoint_called must be false")

    for key in ("order_intent", "submitted_order", "broker_order", "broker_order_id", "order_id"):
        if _truthy(row.get(key)):
            errors.append(f"row {row_index}: broker/order field {key!r} must be absent or null")

    market = _object(row.get("market_snapshot"))
    underlying = _object(market.get("underlying"))
    if _positive(underlying.get("spx")) is None:
        errors.append(f"row {row_index}: market_snapshot.underlying.spx must be positive")
    if _positive(underlying.get("vix")) is None:
        errors.append(f"row {row_index}: market_snapshot.underlying.vix must be positive")
    if not isinstance(underlying.get("spx_timestamp"), str):
        errors.append(f"row {row_index}: SPX timestamp is required")
    if not isinstance(underlying.get("vix_timestamp"), str):
        errors.append(f"row {row_index}: VIX timestamp is required")

    option = _object(market.get("option_nbbo"))
    bid = _positive(option.get("bid"))
    ask = _positive(option.get("ask"))
    if bid is None:
        errors.append(f"row {row_index}: option bid must be positive")
    if ask is None:
        errors.append(f"row {row_index}: option ask must be positive")
    if bid is not None and ask is not None and ask < bid:
        errors.append(f"row {row_index}: option ask must be >= bid")
    if _number(option.get("quote_age_ms")) is None:
        errors.append(f"row {row_index}: option quote_age_ms is required")
    if not isinstance(option.get("timestamp"), str):
        errors.append(f"row {row_index}: option quote timestamp is required")

    contract = _object(row.get("selected_contract"))
    action = row.get("selected_action")
    max_contracts = max(1, int(max_contracts_per_position))
    if action not in ACTIONS:
        errors.append(f"row {row_index}: selected_action {action!r} is invalid")
    if action in {"enter", "hold", "exit"}:
        if contract.get("root") != "SPXW":
            errors.append(f"row {row_index}: selected contract root must be SPXW")
        if contract.get("settlement_style") != "PM":
            errors.append(f"row {row_index}: selected contract must be PM settled")
        quantity = _number(contract.get("quantity"))
        if quantity is not None and abs(quantity - int(quantity)) > 1e-9:
            errors.append(f"row {row_index}: selected quantity must be a whole contract count")
        if quantity is not None and (quantity <= 0 or quantity > max_contracts):
            errors.append(
                f"row {row_index}: selected quantity must be between 1 and {max_contracts} when present"
            )
    elif contract:
        warnings.append(f"row {row_index}: selected_contract present for non-position action")

    model = _object(row.get("model_decision"))
    if action == "enter":
        if _number(model.get("score")) is None:
            errors.append(f"row {row_index}: model score is required for enter")
        if _number(model.get("threshold")) is None:
            errors.append(f"row {row_index}: model threshold is required for enter")
    features = model.get("features")
    if features is not None and not isinstance(features, dict):
        errors.append(f"row {row_index}: model_decision.features must be an object when present")

    gate = _object(row.get("risk_gate"))
    if "passed" not in gate:
        errors.append(f"row {row_index}: risk_gate.passed is required")
    if action == "blocked" and not gate.get("reason"):
        errors.append(f"row {row_index}: blocked action requires risk_gate.reason")

    account = _object(row.get("paper_account_state"))
    if _positive(account.get("starting_cash")) is None:
        errors.append(f"row {row_index}: starting_cash must be positive")
    if _number(account.get("open_positions")) is None:
        errors.append(f"row {row_index}: open_positions is required")
    if _number(account.get("max_concurrent_positions")) is None:
        errors.append(f"row {row_index}: max_concurrent_positions is required")

    return SchemaValidationResult("pass" if not errors else "fail", errors, warnings)


def validate_shadow_stream(rows: list[dict[str, Any]], *, max_contracts_per_position: int = 1) -> dict[str, Any]:
    results = [
        validate_shadow_event(
            row,
            row_index=index,
            max_contracts_per_position=max_contracts_per_position,
        )
        for index, row in enumerate(rows)
    ]
    errors = [error for result in results for error in result.errors]
    warnings = [warning for result in results for warning in result.warnings]
    event_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in rows:
        event = str(row.get("event_type", "missing"))
        action = str(row.get("selected_action", "missing"))
        event_counts[event] = event_counts.get(event, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "status": "pass" if not errors else "fail",
        "rows": len(rows),
        "errors": errors,
        "warnings": warnings,
        "event_counts": event_counts,
        "action_counts": action_counts,
    }


def schema_contract(*, max_contracts_per_position: int = 1) -> dict[str, Any]:
    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "event_types": list(EVENT_TYPES),
        "actions": list(ACTIONS),
        "no_order_mode": NO_ORDER_MODE,
        "max_contracts_per_position": max(1, int(max_contracts_per_position)),
        "required_top_level_fields": list(REQUIRED_TOP_LEVEL),
        "hard_rules": [
            "live_orders_enabled must be false",
            "broker_endpoint_called must be false",
            "broker/order intent fields must be absent or null",
            "enter/hold/exit actions must reference PM-settled SPXW contracts within the configured quantity cap",
            "blocked actions must include a risk_gate.reason",
        ],
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


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None
