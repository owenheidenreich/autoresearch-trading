"""Normalized Protocol101 decision traces for live/replay parity.

The trace contract is deliberately broker-safe. It records the decision state
needed to compare live IBKR observations with historical replay, but it does
not submit orders or call broker/data endpoints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any


SCHEMA_VERSION = "Protocol101DecisionTraceV1"
PROTOCOL_ID = "protocol101"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Protocol101DecisionTrace:
    schema_version: str = SCHEMA_VERSION
    protocol_id: str = PROTOCOL_ID
    source: str = UNKNOWN
    session: str = UNKNOWN
    run_id: str = UNKNOWN
    decision_ts: str = UNKNOWN
    source_quote_ts: str | None = None
    source_context_ts: str | None = None
    feature_contract_version: str | None = None
    decision_index: int | None = None
    mode: str = UNKNOWN
    candidate_count: int = 0
    candidate_ids: tuple[str, ...] = ()
    candidate_universe_hash: str | None = None
    feature_hash: str | None = None
    score_hash: str | None = None
    selected_action: str = UNKNOWN
    selected_contract_id: str | None = None
    selected_score: float | None = None
    decision_threshold: float | None = None
    threshold_distance: float | None = None
    lifecycle_action: str | None = None
    account_state_hash: str | None = None
    risk_gate_hash: str | None = None
    block_reasons: tuple[str, ...] = ()
    quote_freshness_ms: float | None = None
    context_freshness_ms: float | None = None
    execution_hash: str | None = None
    metadata_hash: str | None = None
    payload: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["candidate_ids"] = list(self.candidate_ids)
        out["block_reasons"] = list(self.block_reasons)
        return out


def normalize_decision_trace(row: dict[str, Any], *, source: str = UNKNOWN) -> Protocol101DecisionTrace:
    """Convert a live or historical row into the parity trace contract."""

    market = _object(row.get("market_snapshot"))
    model = _object(row.get("model_decision"))
    risk_gate = _object(row.get("risk_gate"))
    account_state = _object(row.get("paper_account_state") or row.get("account_state"))
    selected_contract = _object(row.get("selected_contract"))
    candidate_universe = row.get("candidate_universe", row.get("candidate_set"))
    features = row.get("features", model.get("features"))
    scores = row.get("model_scores", model.get("scores", model))
    execution = row.get("execution", row.get("order", row.get("fill")))
    metadata = row.get("metadata")

    selected_score = _finite(row.get("selected_score", model.get("score")))
    threshold = _finite(row.get("decision_threshold", model.get("threshold")))
    threshold_distance = None
    if selected_score is not None and threshold is not None:
        threshold_distance = round(selected_score - threshold, 12)

    candidate_ids = _candidate_ids(candidate_universe)
    if not candidate_ids:
        candidate_ids = _candidate_ids(row.get("candidate_ids"))
    block_reasons = _block_reasons(row, risk_gate)
    option_nbbo = _object(market.get("option_nbbo"))
    underlying = _object(market.get("underlying"))
    candidate_count_value = len(candidate_ids)
    if not candidate_ids:
        candidate_count_value = _optional_int(row.get("candidate_count")) or 0

    return Protocol101DecisionTrace(
        protocol_id=str(row.get("protocol_id", PROTOCOL_ID)),
        source=str(row.get("source", source)),
        session=str(row.get("session", UNKNOWN)),
        run_id=str(row.get("run_id", UNKNOWN)),
        decision_ts=str(row.get("decision_ts", row.get("decision_time", row.get("timestamp", UNKNOWN)))),
        source_quote_ts=_optional_str(row.get("source_quote_ts", row.get("source_quote_time"))),
        source_context_ts=_optional_str(row.get("source_context_ts", row.get("source_context_time"))),
        feature_contract_version=_optional_str(row.get("feature_contract_version")),
        decision_index=_optional_int(row.get("decision_index")),
        mode=str(row.get("mode", row.get("runtime_mode", UNKNOWN))),
        candidate_count=candidate_count_value,
        candidate_ids=tuple(candidate_ids),
        candidate_universe_hash=stable_hash(candidate_ids),
        feature_hash=str(row.get("feature_hash") or stable_hash(features)),
        score_hash=str(row.get("score_hash") or stable_hash(scores)),
        selected_action=str(row.get("selected_action", row.get("action", model.get("action", UNKNOWN)))),
        selected_contract_id=_selected_contract_id(row, selected_contract),
        selected_score=selected_score,
        decision_threshold=threshold,
        threshold_distance=threshold_distance,
        lifecycle_action=_optional_str(row.get("lifecycle_action", row.get("exit_action"))),
        account_state_hash=str(row.get("account_state_hash") or stable_hash(account_state)),
        risk_gate_hash=str(row.get("risk_gate_hash") or stable_hash(risk_gate)),
        block_reasons=tuple(block_reasons),
        quote_freshness_ms=_finite(row.get("quote_freshness_ms", option_nbbo.get("quote_age_ms"))),
        context_freshness_ms=_finite(row.get("context_freshness_ms", underlying.get("context_age_ms"))),
        execution_hash=None if execution is None else stable_hash(execution),
        metadata_hash=None if metadata is None else stable_hash(metadata),
        payload=row,
    )


def validate_decision_trace(trace: Protocol101DecisionTrace | dict[str, Any]) -> dict[str, Any]:
    row = trace.to_dict() if isinstance(trace, Protocol101DecisionTrace) else dict(trace)
    errors: list[str] = []
    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append("invalid_schema_version")
    if row.get("protocol_id") != PROTOCOL_ID:
        errors.append("invalid_protocol_id")
    for field_name in ("source", "session", "run_id", "decision_ts", "selected_action"):
        if not row.get(field_name) or row.get(field_name) == UNKNOWN:
            errors.append(f"missing_{field_name}")
    if int(row.get("candidate_count", 0)) < 0:
        errors.append("negative_candidate_count")
    if row.get("selected_score") is not None and not _is_finite_number(row.get("selected_score")):
        errors.append("invalid_selected_score")
    if row.get("decision_threshold") is not None and not _is_finite_number(row.get("decision_threshold")):
        errors.append("invalid_decision_threshold")
    return {"status": "pass" if not errors else "fail", "errors": errors}


def stable_hash(value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 12)
    return value


def _candidate_ids(candidate_universe: Any) -> list[str]:
    if isinstance(candidate_universe, dict):
        if isinstance(candidate_universe.get("candidates"), list):
            return _candidate_ids(candidate_universe["candidates"])
        return sorted(str(key) for key in candidate_universe)
    if not isinstance(candidate_universe, (list, tuple)):
        return []
    ids: list[str] = []
    for item in candidate_universe:
        if isinstance(item, dict):
            candidate_id = (
                item.get("candidate_id")
                or item.get("candidate_uid")
                or item.get("contract_id")
                or item.get("local_symbol")
                or item.get("conid")
            )
            if candidate_id is not None:
                ids.append(str(candidate_id))
        elif item is not None:
            ids.append(str(item))
    return sorted(ids)


def _block_reasons(row: dict[str, Any], risk_gate: dict[str, Any]) -> list[str]:
    reasons = row.get("block_reasons", row.get("blocked_reasons", risk_gate.get("reasons", risk_gate.get("reason"))))
    if reasons is None:
        return []
    if isinstance(reasons, str):
        return [reasons] if reasons else []
    if isinstance(reasons, (list, tuple, set)):
        return sorted(str(reason) for reason in reasons if str(reason))
    return [str(reasons)]


def _selected_contract_id(row: dict[str, Any], selected_contract: dict[str, Any]) -> str | None:
    value = row.get("selected_contract_id")
    if value is None:
        value = (
            selected_contract.get("contract_id")
            or selected_contract.get("candidate_id")
            or selected_contract.get("candidate_uid")
            or selected_contract.get("local_symbol")
            or selected_contract.get("conid")
        )
    return None if value is None else str(value)


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
