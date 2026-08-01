"""Append-only paper trade logging for Protocol 101 live/paper runs."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any


TRADE_LOG_SCHEMA_VERSION = "protocol101_paper_trade_log_v1"
DEFAULT_TRADE_LOG_ROOT = Path("v4/logs/paper_trading")
RETRYABLE_IO_ERRNOS = {errno.EAGAIN, errno.EDEADLK, errno.ETIMEDOUT}
ALLOWED_PROTOCOL_IDS = (
    "protocol101",
    "challenger_premium_leaning_blended_utility_v1",
)
EVENT_TYPES = (
    "market_snapshot",
    "candidate_set",
    "model_decision",
    "risk_gate",
    "paper_account_state",
    "paper_order_blocked",
    "paper_order_dry_run",
    "paper_order_submitted",
    "paper_order_status",
    "paper_entry_fill",
    "paper_exit_intent",
    "paper_exit_submitted",
    "paper_exit_fill",
    "paper_cancel_requested",
    "paper_cancel_confirmed",
    "paper_error",
    "heartbeat",
    "lifecycle_state",
)
REQUIRED_TOP_LEVEL = (
    "schema_version",
    "protocol_id",
    "event_type",
    "timestamp",
    "session",
    "run_id",
    "mode",
    "paper_trading",
    "real_money_trading",
    "broker_order_endpoint_called",
    "trade_uid",
    "selected_contract",
    "order",
    "account",
    "market_snapshot",
    "model_decision",
    "risk_gate",
)


@dataclass(frozen=True)
class TradeLogValidation:
    status: str
    errors: list[str]
    warnings: list[str]


OBSERVABILITY_SCHEMA_VERSION = "protocol101_paper_observability_v1"
RAW_ACCOUNT_ID_PATTERN = re.compile(r"\b(?:DU|U)\d{3,}\b")
PLACEHOLDER_STRINGS = {"", "unknown", "missing", "placeholder", "n/a", "na", "null", "none"}

OBSERVABILITY_REQUIRED_BY_EVENT_TYPE: dict[str, tuple[str, ...]] = {
    "*": (
        "run_id",
        "session_id",
        "event_id",
        "event_type",
        "mode",
        "paper_trading",
        "real_money",
        "real_money_trading",
        "live_orders_enabled",
        "broker_order_endpoint_called",
        "timestamp_utc",
        "received_timestamp_utc",
    ),
    "market_snapshot": (
        "market_snapshot.underlying.spx",
        "market_snapshot.underlying.vix",
        "market_snapshot.underlying.spx_raw_quote_timestamp_utc",
        "market_snapshot.underlying.vix_raw_quote_timestamp_utc",
        "market_snapshot.underlying.spx_quote_age_ms",
        "market_snapshot.underlying.vix_quote_age_ms",
        "market_snapshot.context.context_age_ms",
    ),
    "candidate_set": (
        "artifact_ids",
        "runtime_flag_digest",
        "candidate_set_hash",
        "feature_vector_hash",
        "candidate_count",
        "candidate_gate_diagnostics.filter_reason",
    ),
    "model_decision": (
        "artifact_ids",
        "runtime_flag_digest",
        "decision_timestamp_utc",
        "candidate_set_hash",
        "feature_vector_hash",
        "model_decision.action_mask",
        "model_decision.raw_logits",
        "model_decision.selected_action",
        "model_decision.threshold",
        "model_decision.no_entry_reason",
    ),
    "risk_gate": (
        "artifact_ids",
        "runtime_flag_digest",
        "risk_gate.passed",
        "risk_gate.reason",
        "risk_gate.guard_passed",
        "risk_gate.guard_block_reasons",
    ),
    "paper_order_blocked": (
        "runtime_flag_digest",
        "risk_gate.reason",
        "risk_gate.guard_block_reasons",
    ),
    "paper_order_dry_run": (
        "artifact_ids",
        "runtime_flag_digest",
        "intent_id",
        "order.action",
        "order.quantity",
        "order.limit_price",
        "order.contract_payload",
        "order.order_payload",
        "order.dry_run",
        "order.would_submit",
        "selected_contract.settlement",
        "selected_contract.raw_quote_timestamp_utc",
        "selected_contract.quote_age_ms",
    ),
    "paper_order_submitted": (
        "artifact_ids",
        "runtime_flag_digest",
        "intent_id",
        "order.action",
        "order.quantity",
        "order.limit_price",
        "order.contract_payload",
        "order.order_payload",
        "order.dry_run",
        "order.broker_order_id",
        "order.status",
        "selected_contract.settlement",
    ),
    "paper_order_status": (
        "artifact_ids",
        "runtime_flag_digest",
        "intent_id",
        "order.status",
        "order.final_status",
        "order.filled",
        "order.remaining",
        "selected_contract.settlement",
    ),
    "paper_entry_fill": (
        "artifact_ids",
        "runtime_flag_digest",
        "intent_id",
        "order.status",
        "order.avg_fill_price",
        "selected_contract.settlement",
        "lifecycle.position_detected",
    ),
    "paper_exit_intent": (
        "artifact_ids",
        "runtime_flag_digest",
        "intent_id",
        "lifecycle.position_detected",
        "selected_contract.settlement",
        "lifecycle.runtime_state_hash",
        "lifecycle.lifecycle_action",
    ),
    "paper_exit_submitted": (
        "artifact_ids",
        "runtime_flag_digest",
        "intent_id",
        "order.status",
        "lifecycle.exit_intent_id",
    ),
    "paper_exit_fill": (
        "artifact_ids",
        "runtime_flag_digest",
        "intent_id",
        "order.status",
        "order.avg_fill_price",
        "lifecycle.final_position_state",
    ),
    "lifecycle_state": (
        "lifecycle.position_detected",
        "lifecycle.runtime_state_hash",
        "lifecycle.lifecycle_action",
        "lifecycle.final_position_state",
    ),
}


def trade_log_path(
    *,
    root: Path = DEFAULT_TRADE_LOG_ROOT,
    session: str | None = None,
    run_id: str = "protocol101_paper",
) -> Path:
    clean_session = session or datetime.now(timezone.utc).date().isoformat()
    safe_run = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_id)
    return root / clean_session / f"{safe_run}.jsonl"


def make_trade_log_event(
    *,
    event_type: str,
    timestamp: str | datetime | None = None,
    session: str | None = None,
    run_id: str = "protocol101_paper",
    mode: str = "paper",
    trade_uid: str | None = None,
    selected_contract: dict[str, Any] | None = None,
    order: dict[str, Any] | None = None,
    account: dict[str, Any] | None = None,
    market_snapshot: dict[str, Any] | None = None,
    model_decision: dict[str, Any] | None = None,
    risk_gate: dict[str, Any] | None = None,
    broker_order_endpoint_called: bool = False,
    paper_trading: bool = True,
    real_money_trading: bool = False,
    protocol_id: str = "protocol101",
    session_id: str | None = None,
    event_id: str | None = None,
    intent_id: str | None = None,
    intent_chain_id: str | None = None,
    received_timestamp_utc: str | datetime | None = None,
    decision_timestamp_utc: str | datetime | None = None,
    source_script: str | None = None,
    artifact_ids: dict[str, Any] | None = None,
    runtime_flag_digest: str | None = None,
    live_orders_enabled: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = iso_timestamp(timestamp)
    received_ts = iso_timestamp(received_timestamp_utc or ts)
    decision_ts = iso_timestamp(decision_timestamp_utc or _decision_timestamp_from_payload(market_snapshot, model_decision) or ts)
    resolved_session = session or ts[:10]
    resolved_mode = str(mode)
    row = {
        "schema_version": TRADE_LOG_SCHEMA_VERSION,
        "observability_schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "protocol_id": str(protocol_id),
        "event_type": event_type,
        "timestamp": ts,
        "timestamp_utc": ts,
        "received_timestamp_utc": received_ts,
        "decision_timestamp_utc": decision_ts,
        "session": resolved_session,
        "session_id": session_id or resolved_session,
        "run_id": run_id,
        "mode": resolved_mode,
        "source_script": source_script or "protocol101_paper_runtime",
        "paper_trading": bool(paper_trading),
        "real_money": bool(real_money_trading),
        "real_money_trading": bool(real_money_trading),
        "live_orders_enabled": bool(resolved_mode == "paper-submit") if live_orders_enabled is None else bool(live_orders_enabled),
        "broker_order_endpoint_called": bool(broker_order_endpoint_called),
        "trade_uid": trade_uid or "",
        "intent_id": intent_id or "",
        "intent_chain_id": intent_chain_id or "",
        "artifact_ids": artifact_ids or {},
        "runtime_flag_digest": runtime_flag_digest or "",
        "selected_contract": selected_contract or {},
        "order": order or {},
        "account": account or {},
        "market_snapshot": market_snapshot or {},
        "model_decision": model_decision or {},
        "risk_gate": risk_gate or {},
    }
    if extra:
        row.update(extra)
    row["selected_contract"] = normalize_selected_contract(_object(row.get("selected_contract")))
    row["market_snapshot"] = normalize_market_snapshot(_object(row.get("market_snapshot")))
    if not row.get("intent_id"):
        row["intent_id"] = make_intent_id(row)
    if not row.get("intent_chain_id"):
        row["intent_chain_id"] = row.get("intent_id") or ""
    if not row.get("event_id"):
        row["event_id"] = event_id or make_event_id(row)
    return sanitize_json(row)


def append_trade_event(path: Path, row: dict[str, Any], *, validate: bool = True) -> None:
    if validate:
        result = validate_trade_event(row)
        if result.status != "pass":
            raise ValueError("; ".join(result.errors))
    line = json.dumps(sanitize_json(row), sort_keys=True, allow_nan=False) + "\n"

    def write_once() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)

    retry_io(write_once)


def load_trade_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    text = retry_io(lambda: path.read_text(encoding="utf-8"))
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def retry_io(operation, *, attempts: int = 6):
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as exc:
            if exc.errno not in RETRYABLE_IO_ERRNOS or attempt == attempts - 1:
                raise
            time.sleep(min(0.25 * (2**attempt), 2.0))
    raise RuntimeError("unreachable retry_io state")


def validate_trade_event(row: dict[str, Any]) -> TradeLogValidation:
    errors: list[str] = []
    warnings: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in row:
            errors.append(f"missing {key}")
    if row.get("schema_version") != TRADE_LOG_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TRADE_LOG_SCHEMA_VERSION}")
    if row.get("protocol_id") not in ALLOWED_PROTOCOL_IDS:
        errors.append(f"protocol_id must be one of {ALLOWED_PROTOCOL_IDS}")
    if row.get("event_type") not in EVENT_TYPES:
        errors.append(f"invalid event_type {row.get('event_type')!r}")
    if row.get("paper_trading") is not True:
        errors.append("paper_trading must be true")
    if row.get("real_money_trading") is not False:
        errors.append("real_money_trading must be false")
    if row.get("live_orders") is True:
        errors.append("live_orders must never be true in paper trade logs")
    if row.get("account_id"):
        errors.append("raw account_id must not be logged")

    contract = _object(row.get("selected_contract"))
    order = _object(row.get("order"))
    event_type = str(row.get("event_type") or "")
    if event_type.startswith("paper_order") or event_type.startswith("paper_entry") or event_type.startswith("paper_exit"):
        trading_class = str(contract.get("trading_class") or contract.get("tradingClass") or contract.get("root") or "")
        if trading_class and trading_class != "SPXW":
            errors.append("paper option order events must reference SPXW")
        quantity = _number(order.get("quantity") or order.get("totalQuantity"))
        if quantity is not None and quantity <= 0:
            errors.append("order quantity must be positive")
        limit_price = _number(order.get("limit_price") or order.get("lmtPrice"))
        if limit_price is not None and limit_price <= 0:
            errors.append("limit price must be positive")
    if bool(row.get("broker_order_endpoint_called")) and event_type not in {
        "paper_order_submitted",
        "paper_order_status",
        "paper_entry_fill",
        "paper_exit_submitted",
        "paper_exit_fill",
        "paper_cancel_requested",
        "paper_cancel_confirmed",
        "paper_error",
    }:
        errors.append("broker_order_endpoint_called is only valid for broker event rows")

    account = _object(row.get("account"))
    if account.get("account_id") and not account.get("account_id_redacted"):
        errors.append("account.account_id must not be logged; use account_id_redacted")
    if "account_id_redacted" not in account:
        warnings.append("account_id_redacted missing")

    return TradeLogValidation("pass" if not errors else "fail", errors, warnings)


def validate_trade_log(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = [validate_trade_event(row) for row in rows]
    errors = [error for result in results for error in result.errors]
    warnings = [warning for result in results for warning in result.warnings]
    event_counts: dict[str, int] = {}
    broker_rows = 0
    for row in rows:
        event = str(row.get("event_type", "missing"))
        event_counts[event] = event_counts.get(event, 0) + 1
        broker_rows += int(bool(row.get("broker_order_endpoint_called")))
    return {
        "status": "pass" if not errors else "fail",
        "rows": len(rows),
        "errors": errors,
        "warnings": warnings,
        "event_counts": event_counts,
        "broker_order_endpoint_called_rows": broker_rows,
    }


def validate_observability_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail-closed paper-readiness audit for Protocol101 reconstruction logs.

    This is deliberately stricter than `validate_trade_log`: it is used to mark
    readiness, not to preserve backward compatibility with older logs.
    """

    errors: list[str] = []
    warnings: list[str] = []
    event_counts: dict[str, int] = {}
    guard_pass_by_intent: set[str] = set()
    order_submit_by_intent: set[str] = set()

    for idx, row in enumerate(rows):
        event_type = str(row.get("event_type") or "missing")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        prefix = f"row {idx} {event_type}:"

        for path in required_observability_fields(event_type):
            if _missing_or_placeholder(_get_path(row, path)):
                errors.append(f"{prefix} missing required field {path}")

        if row.get("paper_trading") is not True:
            errors.append(f"{prefix} paper_trading must be true")
        if row.get("real_money") is not False or row.get("real_money_trading") is not False:
            errors.append(f"{prefix} real_money must be false")
        if contains_raw_account_id(row):
            errors.append(f"{prefix} raw account id appears in log row")

        mode = str(row.get("mode") or "")
        if mode in {"intent-shadow", "no-order-shadow", "paper-dry-run"} and row.get("broker_order_endpoint_called") is True:
            errors.append(f"{prefix} no-order/dry-run row called broker endpoint")

        order = _object(row.get("order"))
        quantity = _number(order.get("quantity") or order.get("totalQuantity"))
        if quantity is not None and quantity > 1:
            errors.append(f"{prefix} order quantity exceeds one contract")

        selected = _object(row.get("selected_contract"))
        selected_action = str(_object(row.get("model_decision")).get("selected_action") or _object(row.get("model_decision")).get("action") or "")
        has_selected_contract = bool(selected)
        if event_type in {"paper_order_dry_run", "paper_order_submitted", "paper_order_status", "paper_entry_fill", "paper_exit_intent", "paper_exit_submitted", "paper_exit_fill"} or selected_action == "enter":
            if has_selected_contract:
                if _missing_or_placeholder(selected.get("raw_quote_timestamp_utc") or selected.get("quote_timestamp")):
                    errors.append(f"{prefix} selected quote lacks raw timestamp")
                if _missing_or_placeholder(selected.get("quote_age_ms")):
                    errors.append(f"{prefix} selected quote lacks quote_age_ms")

        model = _object(row.get("model_decision"))
        action = str(model.get("selected_action") or model.get("action") or "")
        if event_type == "model_decision" and action in {"wait", "no_entry"}:
            if _missing_or_placeholder(model.get("no_entry_reason") or model.get("reason")):
                errors.append(f"{prefix} no-entry decision lacks structured no-entry reason")

        risk = _object(row.get("risk_gate"))
        intent_id = str(row.get("intent_id") or row.get("trade_uid") or "")
        if event_type == "risk_gate" and risk.get("passed") is True and intent_id:
            guard_pass_by_intent.add(intent_id)
        if risk.get("passed") is True and event_type == "paper_order_dry_run" and intent_id:
            guard_pass_by_intent.add(intent_id)

        if event_type == "paper_order_submitted":
            if intent_id not in guard_pass_by_intent:
                errors.append(f"{prefix} submitted order lacks prior guard pass")
            order_submit_by_intent.add(intent_id)
            if row.get("broker_order_endpoint_called") is not True:
                errors.append(f"{prefix} submitted order must show broker endpoint called")

        if event_type in {"paper_order_status", "paper_entry_fill", "paper_exit_fill"} and intent_id and intent_id not in order_submit_by_intent:
            warnings.append(f"{prefix} broker status/fill has no earlier submitted-order row in this fixture")

    return {
        "status": "pass" if not errors else "fail",
        "readiness_status": "ready" if not errors else "not_ready",
        "rows": len(rows),
        "errors": errors,
        "warnings": warnings,
        "event_counts": event_counts,
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
    }


def required_observability_fields(event_type: str) -> tuple[str, ...]:
    return OBSERVABILITY_REQUIRED_BY_EVENT_TYPE.get("*", ()) + OBSERVABILITY_REQUIRED_BY_EVENT_TYPE.get(str(event_type), ())


def flatten_trade_event(row: dict[str, Any]) -> dict[str, Any]:
    contract = _object(row.get("selected_contract"))
    order = _object(row.get("order"))
    account = _object(row.get("account"))
    market = _object(row.get("market_snapshot"))
    option = _object(market.get("option_nbbo"))
    underlying = _object(market.get("underlying"))
    model = _object(row.get("model_decision"))
    risk = _object(row.get("risk_gate"))
    timing = _object(row.get("timing"))
    return {
        "timestamp": row.get("timestamp"),
        "session": row.get("session"),
        "run_id": row.get("run_id"),
        "session_id": row.get("session_id"),
        "event_id": row.get("event_id"),
        "intent_id": row.get("intent_id"),
        "event_type": row.get("event_type"),
        "trade_uid": row.get("trade_uid"),
        "broker_order_endpoint_called": row.get("broker_order_endpoint_called"),
        "contract_id": contract.get("contract_id") or contract.get("local_symbol"),
        "trading_class": contract.get("trading_class") or contract.get("tradingClass") or contract.get("root"),
        "expiry": contract.get("expiry"),
        "strike": contract.get("strike"),
        "right": contract.get("right"),
        "action": order.get("action"),
        "quantity": order.get("quantity") or order.get("totalQuantity"),
        "limit_price": order.get("limit_price") or order.get("lmtPrice"),
        "order_status": order.get("status"),
        "filled": order.get("filled"),
        "remaining": order.get("remaining"),
        "avg_fill_price": order.get("avg_fill_price"),
        "cash": account.get("cash"),
        "equity": account.get("equity"),
        "realized_daily_pnl": account.get("realized_daily_pnl"),
        "open_positions": account.get("open_positions"),
        "spx": underlying.get("spx"),
        "vix": underlying.get("vix"),
        "bid": option.get("bid"),
        "ask": option.get("ask"),
        "bid_size": option.get("bid_size"),
        "ask_size": option.get("ask_size"),
        "quote_age_ms": option.get("quote_age_ms"),
        "raw_quote_timestamp_utc": option.get("raw_quote_timestamp_utc") or option.get("quote_timestamp"),
        "candidate_set_hash": row.get("candidate_set_hash"),
        "feature_vector_hash": row.get("feature_vector_hash"),
        "selected_action": model.get("selected_action") or model.get("action"),
        "raw_logits": json.dumps(model.get("raw_logits")) if model.get("raw_logits") is not None else None,
        "wait_logit": model.get("wait_logit"),
        "candidate_logits": json.dumps(model.get("candidate_logits")) if model.get("candidate_logits") is not None else None,
        "selected_margin": model.get("selected_margin") or model.get("margin"),
        "no_entry_reason": model.get("no_entry_reason"),
        "model_action": model.get("action"),
        "model_score": model.get("score"),
        "model_threshold": model.get("threshold"),
        "risk_passed": risk.get("passed"),
        "risk_reason": risk.get("reason"),
        "decision_emitted_at": timing.get("decision_emitted_at"),
        "intended_entry_time": timing.get("intended_entry_time"),
        "intended_exit_time": timing.get("intended_exit_time"),
        "broker_submit_at": timing.get("broker_submit_at"),
        "broker_ack_at": timing.get("broker_ack_at"),
        "entry_fill_at": timing.get("entry_fill_at"),
        "exit_decision_at": timing.get("exit_decision_at"),
        "exit_fill_at": timing.get("exit_fill_at"),
        "decision_to_submit_ms": timing.get("decision_to_submit_ms"),
        "decision_to_entry_fill_ms": timing.get("decision_to_entry_fill_ms"),
        "exit_decision_to_exit_fill_ms": timing.get("exit_decision_to_exit_fill_ms"),
        "blocked_reason": row.get("blocked_reason") or risk.get("reason"),
    }


def export_trade_log_csv(jsonl_path: Path, csv_path: Path) -> dict[str, Any]:
    rows = [flatten_trade_event(row) for row in load_trade_log(jsonl_path)]
    fieldnames = sorted({key for row in rows for key in row.keys()})

    def write_once() -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    retry_io(write_once)
    return {"rows": len(rows), "csv_path": str(csv_path), "jsonl_path": str(jsonl_path)}


def executor_result_event(
    *,
    result: dict[str, Any],
    run_id: str,
    mode: str,
    event_type: str | None = None,
    trade_uid: str | None = None,
    account: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent = _object(result.get("intent"))
    contract_preview = _object(result.get("contract_preview"))
    order_preview = _object(result.get("order_preview"))
    validation = _object(result.get("validation"))
    permission = _object(result.get("permission"))
    quote = _object(result.get("quote"))
    context = _object(result.get("context"))
    intent_id = str(result.get("intent_id") or make_intent_id({"run_id": run_id, "trade_uid": trade_uid or "", "selected_contract": intent, "order": intent}))
    resolved_event = event_type or event_type_from_executor_status(str(result.get("status") or ""))
    selected_contract = {
        "symbol": contract_preview.get("symbol") or intent.get("symbol"),
        "root": contract_preview.get("tradingClass") or intent.get("trading_class") or "SPXW",
        "trading_class": contract_preview.get("tradingClass") or intent.get("trading_class") or "SPXW",
        "settlement": intent.get("settlement") or "PM",
        "expiry": contract_preview.get("lastTradeDateOrContractMonth") or intent.get("expiry"),
        "strike": contract_preview.get("strike") or intent.get("strike"),
        "right": contract_preview.get("right") or intent.get("right"),
        "exchange": contract_preview.get("exchange") or intent.get("exchange"),
        "currency": contract_preview.get("currency") or intent.get("currency"),
    }
    selected_contract = normalize_selected_contract({**quote, **selected_contract})
    dry_run = bool(result.get("dry_run"))
    fill = _object(result.get("fill_summary"))
    order = {
        "action": order_preview.get("action") or intent.get("action"),
        "quantity": order_preview.get("totalQuantity") or intent.get("quantity"),
        "limit_price": order_preview.get("lmtPrice") or intent.get("limit_price"),
        "contract_payload": selected_contract,
        "order_payload": order_preview or intent,
        "status": result.get("status"),
        "final_status": fill.get("status") or result.get("status"),
        "broker_order_id": _object(result.get("trade_preview")).get("order_id") or result.get("broker_order_id"),
        "filled": fill.get("filled_quantity"),
        "remaining": fill.get("remaining_quantity"),
        "avg_fill_price": fill.get("avg_fill_price"),
        "dry_run": dry_run,
        "would_submit": bool(dry_run and result.get("status") == "dry_run_pass"),
        "submit_allowed": bool(result.get("status") in {"dry_run_pass", "submitted"}),
        "reason": result.get("reason"),
    }
    account_row = dict(_object(result.get("account")))
    account_row.update(account or {})
    account_row.setdefault("cash", validation.get("account_cash"))
    account_row.setdefault("open_positions", validation.get("open_positions"))
    return make_trade_log_event(
        event_type=resolved_event,
        run_id=run_id,
        mode=mode,
        trade_uid=trade_uid or executor_trade_uid(intent),
        intent_id=intent_id,
        intent_chain_id=intent_id,
        selected_contract=selected_contract,
        order=order,
        account=account_row,
        market_snapshot={"option_nbbo": quote, "context": context, "underlying": _object(context.get("underlying"))},
        model_decision={
            "source": "protocol142_executor_smoke",
            "action": intent.get("action"),
            "selected_action": intent.get("action"),
            "raw_logits": [],
            "candidate_logits": [],
            "wait_logit": None,
            "selected_margin": None,
            "threshold": None,
            "no_entry_reason": None,
            "action_mask": {"BUY": True, "SELL": True},
        },
        risk_gate={
            "passed": bool(permission.get("passed")) and bool(validation.get("passed")),
            "guard_passed": bool(permission.get("passed")) and bool(validation.get("passed")),
            "guard_block_reasons": [*permission.get("reasons", []), *validation.get("reasons", [])],
            "account_prefix_ok": permission.get("account_prefix_ok"),
            "paper_account_confirmed": permission.get("paper_account_confirmed"),
            "real_money_false_confirmed": bool(permission.get("real_money_false_confirmed")) and bool(validation.get("real_money_false_confirmed", True)),
            "quantity_ok": validation.get("quantity_ok"),
            "one_open_position_ok": validation.get("one_open_position_ok"),
            "quote_freshness_ok": validation.get("quote_freshness_ok"),
            "context_freshness_ok": validation.get("context_freshness_ok"),
            "affordability_ok": validation.get("affordability_ok"),
            "account_cash": validation.get("account_cash"),
            "open_positions": validation.get("open_positions"),
            "premium_required": validation.get("premium_required"),
            "reason": ",".join(
                str(reason)
                for reason in [*permission.get("reasons", []), *validation.get("reasons", [])]
            )
            or "pass",
            "permission": permission,
            "validation": validation,
        },
        broker_order_endpoint_called=bool(result.get("broker_order_endpoint_called")),
        artifact_ids=_object(result.get("artifact_ids")),
        runtime_flag_digest=str(result.get("runtime_flag_digest") or ""),
        extra={
            "order_intent": intent,
            "guard_config_digest": result.get("guard_config_digest"),
            "broker_status": fill,
        },
    )


def event_type_from_executor_status(status: str) -> str:
    if status == "dry_run_pass":
        return "paper_order_dry_run"
    if status == "submitted":
        return "paper_order_submitted"
    return "paper_order_blocked"


def executor_trade_uid(intent: dict[str, Any]) -> str:
    parts = [
        "protocol101",
        str(intent.get("expiry") or "unknown"),
        str(intent.get("strike") or "unknown"),
        str(intent.get("right") or "unknown"),
        str(intent.get("action") or "unknown"),
    ]
    return "-".join(part.replace(" ", "_") for part in parts)


def iso_timestamp(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return str(value)


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(sanitize_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_event_id(row: dict[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "event_id"}
    return stable_json_hash(payload)[:24]


def make_intent_id(row: dict[str, Any]) -> str:
    order = _object(row.get("order"))
    contract = _object(row.get("selected_contract"))
    if not order and not contract:
        return ""
    payload = {
        "run_id": row.get("run_id"),
        "trade_uid": row.get("trade_uid"),
        "action": order.get("action") or order.get("totalQuantity"),
        "quantity": order.get("quantity") or order.get("totalQuantity"),
        "limit_price": order.get("limit_price") or order.get("lmtPrice"),
        "symbol": contract.get("symbol"),
        "expiry": contract.get("expiry") or contract.get("lastTradeDateOrContractMonth"),
        "strike": contract.get("strike"),
        "right": contract.get("right"),
    }
    return f"intent_{stable_json_hash(payload)[:20]}"


def normalize_selected_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if not contract:
        return {}
    out = dict(contract)
    raw_ts = out.get("raw_quote_timestamp_utc") or out.get("quote_timestamp") or out.get("quote_time")
    if raw_ts is not None:
        out["raw_quote_timestamp_utc"] = iso_timestamp(raw_ts)
        out.setdefault("quote_timestamp", out["raw_quote_timestamp_utc"])
    received = out.get("received_timestamp_utc") or out.get("received_timestamp")
    if received is not None:
        out["received_timestamp_utc"] = iso_timestamp(received)
    decision = out.get("decision_timestamp_utc") or out.get("decision_timestamp")
    if decision is not None:
        out["decision_timestamp_utc"] = iso_timestamp(decision)
    return out


def normalize_market_snapshot(market: dict[str, Any]) -> dict[str, Any]:
    if not market:
        return {"underlying": {}, "option_nbbo": {}, "context": {}}
    out = dict(market)
    out["underlying"] = dict(_object(out.get("underlying")))
    out["option_nbbo"] = normalize_selected_contract(_object(out.get("option_nbbo")))
    out["context"] = dict(_object(out.get("context")))
    for prefix in ("spx", "vix"):
        raw = out["underlying"].get(f"{prefix}_raw_quote_timestamp_utc") or out["underlying"].get(f"{prefix}_quote_timestamp")
        if raw is not None:
            out["underlying"][f"{prefix}_raw_quote_timestamp_utc"] = iso_timestamp(raw)
    return out


def contains_raw_account_id(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) == "account_id_redacted":
                continue
            if contains_raw_account_id(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(contains_raw_account_id(item) for item in value)
    if isinstance(value, str):
        return bool(RAW_ACCOUNT_ID_PATTERN.search(value))
    return False


def _decision_timestamp_from_payload(market_snapshot: dict[str, Any] | None, model_decision: dict[str, Any] | None) -> str | None:
    model = _object(model_decision)
    market = _object(market_snapshot)
    option = _object(market.get("option_nbbo"))
    return (
        model.get("decision_timestamp_utc")
        or model.get("decision_timestamp")
        or option.get("decision_timestamp_utc")
        or option.get("decision_timestamp")
    )


def _get_path(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _missing_or_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in PLACEHOLDER_STRINGS
    if isinstance(value, dict):
        return not value
    return False


def sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return iso_timestamp(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (dict, list, tuple, str, bytes)):
        try:
            return sanitize_json(value.item())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    return value


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
