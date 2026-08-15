"""No-order shadow-feed parity checks for Protocol 066.

The shadow feed is a live-data rehearsal, not paper trading. Each observation
must contain the quotes, context, features, model output, and decision that the
frozen protocol would have seen, while carrying no broker order intent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import math

from v4.scripts.build_lifecycle_sequence_dataset import CAUSAL_STEP_FEATURE_COLUMNS


@dataclass(frozen=True)
class ShadowParityConfig:
    protocol_id: str = "protocol066"
    max_quote_age_ms: int = 1500
    max_context_age_ms: int = 5000
    intended_size: int = 1
    required_feature_columns: tuple[str, ...] = tuple(CAUSAL_STEP_FEATURE_COLUMNS)
    allowed_position_states: tuple[str, ...] = ("flat", "holding")
    allowed_actions: tuple[str, ...] = ("wait", "no_entry", "hold", "exit", "stop", "forced_flat")


@dataclass
class ShadowRowResult:
    row_index: int
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    action: str | None = None
    position_state: str | None = None
    contract_id: str | None = None


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _as_float(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    return float(value)


def _as_int_ms(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return int(numeric)


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _truthy_order_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return False
    return True


def load_shadow_observations(path: Path) -> list[dict[str, Any]]:
    """Load a JSON or JSONL shadow observation file."""

    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
        return rows

    value = json.loads(path.read_text())
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return value
    raise ValueError(f"{path} must contain a JSON object, JSON object list, or JSONL objects")


def validate_shadow_observation(
    row: dict[str, Any],
    *,
    row_index: int,
    config: ShadowParityConfig = ShadowParityConfig(),
) -> ShadowRowResult:
    errors: list[str] = []
    warnings: list[str] = []

    protocol_id = row.get("protocol_id") or row.get("protocol")
    if protocol_id is None:
        warnings.append("protocol_id is missing")
    elif str(protocol_id) != config.protocol_id:
        errors.append(f"protocol_id {protocol_id!r} != {config.protocol_id!r}")

    observation_ts = _as_int_ms(row.get("timestamp_ms"))
    if observation_ts is None:
        errors.append("timestamp_ms is missing or non-numeric")

    decision_time = row.get("decision_time")
    if not isinstance(decision_time, str) or not decision_time:
        errors.append("decision_time is missing")

    for key in ("order_intent", "order_id", "submitted_order", "broker_order", "broker_order_id"):
        if key in row and _truthy_order_value(row.get(key)):
            errors.append(f"broker/order field {key!r} must be absent or null in shadow mode")

    position_state = str(row.get("position_state", ""))
    if position_state not in config.allowed_position_states:
        errors.append(f"position_state {position_state!r} is invalid")

    intended_size = row.get("intended_size")
    if intended_size is None and position_state == "holding":
        errors.append("intended_size is required while holding")
    elif intended_size is not None:
        numeric_size = _as_float(intended_size)
        if numeric_size is None or abs(numeric_size - config.intended_size) > 1e-9:
            errors.append(f"intended_size must be {config.intended_size}")

    contract_id = row.get("contract_id") or row.get("raw_symbol")
    contract_id_text = str(contract_id) if contract_id is not None else None
    if position_state == "holding":
        if not contract_id_text:
            errors.append("contract_id/raw_symbol is required while holding")
        elif not contract_id_text.startswith("SPXW"):
            errors.append("holding contract must be PM-settled SPXW, not SPX/other roots")

    nbbo = _object(row.get("nbbo"))
    bid = _as_float(nbbo.get("bid", row.get("bid")))
    ask = _as_float(nbbo.get("ask", row.get("ask")))
    quote_ts = _as_int_ms(nbbo.get("timestamp_ms", nbbo.get("quote_timestamp_ms")))
    if bid is None or bid <= 0:
        errors.append("nbbo.bid must be positive")
    if ask is None or ask <= 0:
        errors.append("nbbo.ask must be positive")
    if bid is not None and ask is not None and ask < bid:
        errors.append("nbbo.ask must be >= nbbo.bid")
    if observation_ts is not None and quote_ts is not None:
        quote_age = observation_ts - quote_ts
        if quote_age < 0:
            errors.append("nbbo timestamp is after observation timestamp")
        elif quote_age > config.max_quote_age_ms:
            errors.append(f"nbbo quote age {quote_age}ms exceeds {config.max_quote_age_ms}ms")
    else:
        errors.append("nbbo.timestamp_ms is required for quote-age parity")

    context = _object(row.get("context"))
    spx = _as_float(context.get("spx", context.get("SPX")))
    vix = _as_float(context.get("vix", context.get("VIX")))
    context_ts = _as_int_ms(context.get("timestamp_ms"))
    if spx is None or spx <= 0:
        errors.append("context.spx must be positive")
    if vix is None or vix <= 0:
        errors.append("context.vix must be positive")
    if observation_ts is not None and context_ts is not None:
        context_age = observation_ts - context_ts
        if context_age < 0:
            errors.append("context timestamp is after observation timestamp")
        elif context_age > config.max_context_age_ms:
            errors.append(f"context age {context_age}ms exceeds {config.max_context_age_ms}ms")
    else:
        errors.append("context.timestamp_ms is required for context-age parity")
    source = context.get("source")
    if source is None:
        warnings.append("context.source is missing")

    features = _object(row.get("features"))
    if not features:
        errors.append("features object is missing")
    missing_features = [column for column in config.required_feature_columns if column not in features]
    if missing_features:
        errors.append(f"missing required features: {missing_features}")
    bad_features = [column for column in config.required_feature_columns if column in features and not _is_finite_number(features[column])]
    if bad_features:
        errors.append(f"non-finite required features: {bad_features}")
    extra_features = sorted(set(features) - set(config.required_feature_columns))
    if extra_features:
        warnings.append(f"extra features ignored: {extra_features[:10]}")

    model = _object(row.get("model"))
    if not model:
        errors.append("model object is missing")
    else:
        artifact = model.get("artifact") or model.get("artifact_path")
        if not isinstance(artifact, str) or not artifact:
            errors.append("model.artifact is required")
        for key in ("predicted_continuation_value", "override_threshold"):
            if key in model and not _is_finite_number(model[key]):
                errors.append(f"model.{key} must be finite when present")

    decision = _object(row.get("decision"))
    action = decision.get("action", row.get("action"))
    action_text = str(action) if action is not None else None
    if action_text not in config.allowed_actions:
        errors.append(f"decision.action {action_text!r} is invalid")
    elif position_state == "holding" and action_text in {"wait", "no_entry"}:
        errors.append("holding lifecycle observation cannot emit wait/no_entry")
    elif position_state == "flat" and action_text in {"exit", "stop", "forced_flat"}:
        errors.append("flat observation cannot emit exit/stop/forced_flat")

    status = "pass" if not errors else "fail"
    return ShadowRowResult(
        row_index=row_index,
        status=status,
        errors=errors,
        warnings=warnings,
        action=action_text,
        position_state=position_state or None,
        contract_id=contract_id_text,
    )


def summarize_shadow_parity(
    rows: list[dict[str, Any]],
    *,
    config: ShadowParityConfig = ShadowParityConfig(),
) -> dict[str, Any]:
    results = [
        validate_shadow_observation(row, row_index=index, config=config)
        for index, row in enumerate(rows)
    ]
    failed = [result for result in results if result.errors]
    warnings = [warning for result in results for warning in result.warnings]
    action_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    for result in results:
        if result.action:
            action_counts[result.action] = action_counts.get(result.action, 0) + 1
        if result.position_state:
            state_counts[result.position_state] = state_counts.get(result.position_state, 0) + 1

    checks = [
        {
            "name": "no_order_shadow_mode",
            "status": "pass" if not any("broker/order field" in error for result in results for error in result.errors) else "fail",
            "detail": "No broker order intent fields were present.",
        },
        {
            "name": "feature_schema",
            "status": "pass" if not any("required features" in error or "non-finite required features" in error for result in results for error in result.errors) else "fail",
            "detail": f"Required {config.protocol_id} feature columns: {len(config.required_feature_columns)}.",
        },
        {
            "name": "quote_context_freshness",
            "status": "pass"
            if not any(("quote age" in error or "context age" in error or "timestamp is after" in error) for result in results for error in result.errors)
            else "fail",
            "detail": f"Quote max age {config.max_quote_age_ms}ms; context max age {config.max_context_age_ms}ms.",
        },
        {
            "name": "spxw_one_contract_identity",
            "status": "pass"
            if not any(("SPXW" in error or "intended_size" in error or "contract_id" in error) for result in results for error in result.errors)
            else "fail",
            "detail": "Holding rows must be one SPXW contract.",
        },
    ]
    status = "pass" if rows and not failed else "blocked"
    if status == "pass" and warnings:
        status = "warn"
    return {
        "status": status,
        "protocol_id": config.protocol_id,
        "rows": len(rows),
        "passed_rows": len(rows) - len(failed),
        "failed_rows": len(failed),
        "warning_count": len(warnings),
        "action_counts": action_counts,
        "position_state_counts": state_counts,
        "checks": checks,
        "row_results": [
            {
                "row_index": result.row_index,
                "status": result.status,
                "errors": result.errors,
                "warnings": result.warnings,
                "action": result.action,
                "position_state": result.position_state,
                "contract_id": result.contract_id,
            }
            for result in results
        ],
    }


def shadow_observation_template(config: ShadowParityConfig = ShadowParityConfig()) -> dict[str, Any]:
    features = {column: 0.0 for column in config.required_feature_columns}
    return {
        "protocol_id": config.protocol_id,
        "timestamp_ms": 1770000000000,
        "decision_time": "2026-02-01T15:00:00+00:00",
        "position_state": "holding",
        "intended_size": 1,
        "contract_id": "SPXW-20260201-06700.000-C",
        "nbbo": {"bid": 1.0, "ask": 1.1, "timestamp_ms": 1769999999500},
        "context": {"spx": 6700.0, "vix": 20.0, "source": "official_live", "timestamp_ms": 1769999999000},
        "features": features,
        "model": {
            "artifact": f"persisted_artifact_for_{config.protocol_id}",
            "predicted_continuation_value": 25.0,
            "override_threshold": 0.0,
        },
        "decision": {"action": "hold", "reason": "shadow parity template"},
        "order_intent": None,
    }


def write_shadow_template(path: Path, config: ShadowParityConfig = ShadowParityConfig()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(shadow_observation_template(config), sort_keys=True) + "\n")
