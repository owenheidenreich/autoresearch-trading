"""Append-only paper trade logging for Protocol 101 live/paper runs."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


TRADE_LOG_SCHEMA_VERSION = "protocol101_paper_trade_log_v1"
DEFAULT_TRADE_LOG_ROOT = Path("v4/logs/paper_trading")
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
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = iso_timestamp(timestamp)
    row = {
        "schema_version": TRADE_LOG_SCHEMA_VERSION,
        "protocol_id": "protocol101",
        "event_type": event_type,
        "timestamp": ts,
        "session": session or ts[:10],
        "run_id": run_id,
        "mode": mode,
        "paper_trading": bool(paper_trading),
        "real_money_trading": bool(real_money_trading),
        "broker_order_endpoint_called": bool(broker_order_endpoint_called),
        "trade_uid": trade_uid or "",
        "selected_contract": selected_contract or {},
        "order": order or {},
        "account": account or {},
        "market_snapshot": market_snapshot or {},
        "model_decision": model_decision or {},
        "risk_gate": risk_gate or {},
    }
    if extra:
        row.update(extra)
    return sanitize_json(row)


def append_trade_event(path: Path, row: dict[str, Any], *, validate: bool = True) -> None:
    if validate:
        result = validate_trade_event(row)
        if result.status != "pass":
            raise ValueError("; ".join(result.errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(sanitize_json(row), sort_keys=True, allow_nan=False) + "\n")


def load_trade_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_trade_event(row: dict[str, Any]) -> TradeLogValidation:
    errors: list[str] = []
    warnings: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in row:
            errors.append(f"missing {key}")
    if row.get("schema_version") != TRADE_LOG_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TRADE_LOG_SCHEMA_VERSION}")
    if row.get("protocol_id") != "protocol101":
        errors.append("protocol_id must be protocol101")
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


def flatten_trade_event(row: dict[str, Any]) -> dict[str, Any]:
    contract = _object(row.get("selected_contract"))
    order = _object(row.get("order"))
    account = _object(row.get("account"))
    market = _object(row.get("market_snapshot"))
    option = _object(market.get("option_nbbo"))
    underlying = _object(market.get("underlying"))
    model = _object(row.get("model_decision"))
    risk = _object(row.get("risk_gate"))
    return {
        "timestamp": row.get("timestamp"),
        "session": row.get("session"),
        "run_id": row.get("run_id"),
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
        "model_action": model.get("action"),
        "model_score": model.get("score"),
        "model_threshold": model.get("threshold"),
        "risk_passed": risk.get("passed"),
        "risk_reason": risk.get("reason"),
    }


def export_trade_log_csv(jsonl_path: Path, csv_path: Path) -> dict[str, Any]:
    rows = [flatten_trade_event(row) for row in load_trade_log(jsonl_path)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
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
    resolved_event = event_type or event_type_from_executor_status(str(result.get("status") or ""))
    selected_contract = {
        "symbol": contract_preview.get("symbol") or intent.get("symbol"),
        "root": contract_preview.get("tradingClass") or intent.get("trading_class") or "SPXW",
        "trading_class": contract_preview.get("tradingClass") or intent.get("trading_class") or "SPXW",
        "expiry": contract_preview.get("lastTradeDateOrContractMonth") or intent.get("expiry"),
        "strike": contract_preview.get("strike") or intent.get("strike"),
        "right": contract_preview.get("right") or intent.get("right"),
        "exchange": contract_preview.get("exchange") or intent.get("exchange"),
        "currency": contract_preview.get("currency") or intent.get("currency"),
    }
    order = {
        "action": order_preview.get("action") or intent.get("action"),
        "quantity": order_preview.get("totalQuantity") or intent.get("quantity"),
        "limit_price": order_preview.get("lmtPrice") or intent.get("limit_price"),
        "status": result.get("status"),
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
        selected_contract=selected_contract,
        order=order,
        account=account_row,
        market_snapshot={"option_nbbo": quote, "context": context, "underlying": _object(context.get("underlying"))},
        model_decision={"source": "protocol142_executor_smoke", "action": intent.get("action")},
        risk_gate={
            "passed": bool(permission.get("passed")) and bool(validation.get("passed")),
            "reason": ",".join(
                str(reason)
                for reason in [*permission.get("reasons", []), *validation.get("reasons", [])]
            )
            or "pass",
            "permission": permission,
            "validation": validation,
        },
        broker_order_endpoint_called=bool(result.get("broker_order_endpoint_called")),
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


def sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return iso_timestamp(value)
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
