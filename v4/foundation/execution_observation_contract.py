"""Execution-observation packet contract for fill-realism work.

The contract is deliberately about observation quality, not model quality. It
defines the fields required before a future empirical fill/cancel/slippage
model can be calibrated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import pandas as pd


PASS = "pass"
BLOCKED = "blocked"


@dataclass(frozen=True)
class ExecutionObservationContract:
    schema_version: str = "execution_observation_packet_v1"
    required_fields: tuple[str, ...] = (
        "decision_timestamp",
        "raw_quote_timestamp",
        "received_timestamp",
        "quote_age_ms",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "spread",
        "premium",
        "side",
        "moneyness",
        "time_bucket",
        "intended_ask_entry",
        "submitted_limit",
        "fill_status",
        "cancel_status",
        "timeout_status",
        "latency_ms",
        "exit_bid",
        "post_fill_pnl",
    )
    required_outcomes: tuple[str, ...] = ("filled", "cancelled", "timeout", "missed")
    purpose: str = "execution_realism_packet_for_fill_cancel_slippage_calibration"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "decision_timestamp": ("decision_timestamp", "decision_time", "timestamp"),
    "raw_quote_timestamp": ("raw_quote_timestamp", "quote_timestamp", "quote_time"),
    "received_timestamp": ("received_timestamp", "receive_time", "local_received_timestamp"),
    "quote_age_ms": ("quote_age_ms", "quote_gap_ms", "quote_gap_seconds"),
    "bid": ("bid",),
    "ask": ("ask",),
    "bid_size": ("bid_size", "entry_bid_size"),
    "ask_size": ("ask_size", "entry_ask_size"),
    "spread": ("spread", "entry_spread"),
    "premium": ("premium", "entry_premium", "order_premium"),
    "side": ("side", "right", "order_side"),
    "moneyness": ("moneyness", "offset", "offset_points"),
    "time_bucket": ("time_bucket", "bucket"),
    "intended_ask_entry": ("intended_ask_entry", "reference_ask", "ask"),
    "submitted_limit": ("submitted_limit", "order_limit_price", "limit_price"),
    "fill_status": ("fill_status", "order_status"),
    "cancel_status": ("cancel_status", "order_cancel_status"),
    "timeout_status": ("timeout_status", "order_timeout_status"),
    "latency_ms": ("latency_ms", "latency_seconds"),
    "exit_bid": ("exit_bid", "bid"),
    "post_fill_pnl": ("post_fill_pnl", "pnl", "realized_pnl"),
}


def default_execution_observation_contract() -> ExecutionObservationContract:
    return ExecutionObservationContract()


def validate_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    contract = default_execution_observation_contract()
    errors: list[str] = []
    for field in contract.required_fields:
        key = resolve_field(row.keys(), field)
        if key is None:
            errors.append(f"missing_field:{field}")
            continue
        value = row.get(key)
        if value is None or value == "":
            errors.append(f"empty_field:{field}")
    bid = finite_value(value_for(row, "bid"))
    ask = finite_value(value_for(row, "ask"))
    if math.isfinite(bid) and math.isfinite(ask) and ask < bid:
        errors.append("crossed_quote")
    quote_age = finite_value(value_for(row, "quote_age_ms"))
    if math.isfinite(quote_age) and quote_age < 0:
        errors.append("negative_quote_age")
    return {"status": PASS if not errors else BLOCKED, "errors": errors}


def coverage_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    contract = default_execution_observation_contract()
    if frame.empty:
        return {
            "status": BLOCKED,
            "reason": "no_observation_rows",
            "row_count": 0,
            "required_fields": list(contract.required_fields),
            "missing_fields": list(contract.required_fields),
            "field_coverage": {},
        }
    coverage: dict[str, float] = {}
    missing: list[str] = []
    for field in contract.required_fields:
        key = resolve_field(frame.columns, field)
        if key is None:
            missing.append(field)
            coverage[field] = 0.0
            continue
        series = frame[key]
        coverage[field] = float(series.notna().mean())
    return {
        "status": PASS if not missing else BLOCKED,
        "reason": "all_required_fields_present" if not missing else "missing_required_fields",
        "row_count": int(len(frame)),
        "required_fields": list(contract.required_fields),
        "missing_fields": missing,
        "field_coverage": coverage,
    }


def resolve_field(columns: Any, canonical: str) -> str | None:
    column_set = {str(column) for column in columns}
    for alias in FIELD_ALIASES.get(canonical, (canonical,)):
        if alias in column_set:
            return alias
    return None


def value_for(row: Mapping[str, Any], canonical: str) -> Any:
    key = resolve_field(row.keys(), canonical)
    return None if key is None else row.get(key)


def finite_value(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default
