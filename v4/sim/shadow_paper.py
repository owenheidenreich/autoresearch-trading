"""No-order shadow-paper accounting for router JSONL streams.

This module consumes the same shadow observations emitted by the live router.
It does not call broker endpoints and it does not decide trades. Its job is to
turn an already-frozen router stream into auditable one-contract lifecycle
accounting: observed entry, hold path, terminal exit/stop/forced-flat, and
mark-to-market state for streams that end while a trade is still open.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


_CONTRACT_MULTIPLIER = 100.0
_TERMINAL_ACTIONS = {"exit", "stop", "forced_flat"}
_ORDER_FIELDS = ("order_intent", "order_id", "submitted_order", "broker_order", "broker_order_id")


@dataclass(frozen=True)
class ShadowPaperConfig:
    """Configuration for no-order shadow-paper accounting."""

    protocol_id: str = "protocol081"
    contract_multiplier: float = _CONTRACT_MULTIPLIER
    intended_size: int = 1
    require_all_closed: bool = False
    require_terminal_final: bool = False
    enforce_global_one_position: bool = False


@dataclass
class ShadowTradeLedger:
    """One reconstructed trade lifecycle from a shadow observation stream."""

    trade_uid: str
    contract_id: str
    side: str
    entry_time: str
    first_observed_time: str
    exit_time: str | None
    last_observed_time: str
    entry_fill_price: float | None
    exit_fill_price: float | None
    mark_price: float | None
    status: str
    terminal_action: str | None
    row_count: int
    post_terminal_row_count: int
    realized_pnl: float | None
    mark_pnl: float | None
    min_mark_pnl: float | None
    max_mark_pnl: float | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def replay_shadow_paper(rows: list[dict[str, Any]], *, config: ShadowPaperConfig = ShadowPaperConfig()) -> dict[str, Any]:
    """Replay no-order shadow observations into trade-level paper accounting."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    global_errors: list[str] = []
    global_warnings: list[str] = []

    for index, row in enumerate(rows):
        protocol_id = row.get("protocol_id") or row.get("protocol")
        if protocol_id is not None and str(protocol_id) != config.protocol_id:
            global_errors.append(f"row {index}: protocol_id {protocol_id!r} != {config.protocol_id!r}")
        trade_uid = str(row.get("trade_uid") or "")
        if not trade_uid:
            trade_uid = f"missing-trade-uid-{index}"
            global_errors.append(f"row {index}: trade_uid is missing")
        grouped.setdefault(trade_uid, []).append(row)

    ledgers = [_replay_trade(trade_uid, trade_rows, config=config) for trade_uid, trade_rows in grouped.items()]
    checks = _checks(ledgers, global_errors, config=config)
    status = _status_from_checks(checks, ledgers, config=config)

    closed = [ledger for ledger in ledgers if ledger.status == "closed" and ledger.realized_pnl is not None]
    all_marks = [ledger for ledger in ledgers if ledger.mark_pnl is not None]
    max_concurrent = _max_concurrent_positions(ledgers)
    if max_concurrent > 1 and not config.enforce_global_one_position:
        global_warnings.append(
            "stream contains overlapping trade windows; this is allowed for offline selected-trade rehearsal, "
            "but live shadow paper should be run with enforce_global_one_position=true"
        )

    action_counts: dict[str, int] = {}
    side_counts: dict[str, int] = {}
    for ledger in ledgers:
        if ledger.terminal_action:
            action_counts[ledger.terminal_action] = action_counts.get(ledger.terminal_action, 0) + 1
        side_counts[ledger.side] = side_counts.get(ledger.side, 0) + 1

    summary = {
        "status": status,
        "protocol_id": config.protocol_id,
        "rows": len(rows),
        "trades": len(ledgers),
        "closed_trades": len(closed),
        "open_trades": len(ledgers) - len(closed),
        "terminal_action_counts": action_counts,
        "side_counts": side_counts,
        "max_concurrent_positions": max_concurrent,
        "realized_pnl_closed": _pnl_summary([ledger.realized_pnl for ledger in closed]),
        "mark_pnl_all_trades": _pnl_summary([ledger.mark_pnl for ledger in all_marks]),
        "checks": checks,
        "errors": global_errors,
        "warnings": global_warnings,
        "trade_ledgers": [_ledger_to_dict(ledger) for ledger in ledgers],
    }
    return summary


def _replay_trade(trade_uid: str, rows: list[dict[str, Any]], *, config: ShadowPaperConfig) -> ShadowTradeLedger:
    sorted_rows = sorted(rows, key=lambda row: (_as_int(row.get("timestamp_ms")) or 0, _as_int(row.get("sequence_step_index")) or 0))
    first = sorted_rows[0]
    last = sorted_rows[-1]
    errors: list[str] = []
    warnings: list[str] = []

    contract_ids = {str(row.get("contract_id") or row.get("raw_symbol") or "") for row in sorted_rows}
    contract_ids.discard("")
    contract_id = sorted(contract_ids)[0] if contract_ids else ""
    if not contract_id:
        errors.append("contract_id/raw_symbol is missing")
    elif not contract_id.startswith("SPXW"):
        errors.append("contract is not PM-settled SPXW")
    if len(contract_ids) > 1:
        errors.append(f"contract changed inside trade stream: {sorted(contract_ids)}")

    for row in sorted_rows:
        for key in _ORDER_FIELDS:
            if _truthy_order_value(row.get(key)):
                errors.append(f"broker/order field {key!r} must be absent or null")
        intended_size = _as_float(row.get("intended_size"))
        if intended_size is not None and abs(intended_size - config.intended_size) > 1e-9:
            errors.append(f"intended_size {intended_size} != {config.intended_size}")

    entry_fill = _infer_entry_fill_price(first)
    if entry_fill is None or entry_fill <= 0:
        errors.append("entry fill price could not be inferred from shadow row")

    terminal_index = next((index for index, row in enumerate(sorted_rows) if _action(row) in _TERMINAL_ACTIONS), None)
    terminal_row = sorted_rows[terminal_index] if terminal_index is not None else None
    terminal_action = _action(terminal_row) if terminal_row else None
    post_terminal_count = 0 if terminal_index is None else len(sorted_rows) - terminal_index - 1
    if post_terminal_count:
        message = f"{post_terminal_count} rows appear after terminal action {terminal_action!r}"
        if config.require_terminal_final:
            errors.append(message)
        else:
            warnings.append(message)
    exit_row = terminal_row or last
    exit_fill = _nbbo_bid(exit_row)
    mark_price = _nbbo_bid(last)
    if exit_fill is None or exit_fill <= 0:
        errors.append("exit/mark bid price is missing or non-positive")

    status = "closed" if terminal_row else "open_at_stream_end"
    if status != "closed":
        warnings.append("stream ended while trade was still open")

    marks = []
    if entry_fill is not None:
        for row in sorted_rows:
            bid = _nbbo_bid(row)
            if bid is not None:
                marks.append((bid - entry_fill) * config.contract_multiplier)

    realized_pnl = None
    if status == "closed" and entry_fill is not None and exit_fill is not None:
        realized_pnl = (exit_fill - entry_fill) * config.contract_multiplier
    mark_pnl = None
    if entry_fill is not None and mark_price is not None:
        mark_pnl = (mark_price - entry_fill) * config.contract_multiplier

    return ShadowTradeLedger(
        trade_uid=trade_uid,
        contract_id=contract_id,
        side=_side_from_contract(contract_id, first),
        entry_time=str(first.get("entry_decision_time") or first.get("decision_time") or ""),
        first_observed_time=str(first.get("decision_time") or ""),
        exit_time=str(exit_row.get("decision_time") or "") if status == "closed" else None,
        last_observed_time=str(last.get("decision_time") or ""),
        entry_fill_price=entry_fill,
        exit_fill_price=exit_fill if status == "closed" else None,
        mark_price=mark_price,
        status=status,
        terminal_action=terminal_action,
        row_count=len(sorted_rows),
        post_terminal_row_count=post_terminal_count,
        realized_pnl=realized_pnl,
        mark_pnl=mark_pnl,
        min_mark_pnl=min(marks) if marks else None,
        max_mark_pnl=max(marks) if marks else None,
        errors=sorted(set(errors)),
        warnings=warnings,
    )


def _infer_entry_fill_price(row: dict[str, Any]) -> float | None:
    features = _object(row.get("features"))
    for key in ("entry_ask", "entry_fill_price", "entry_fill_nbbo"):
        value = _as_float(features.get(key, row.get(key)))
        if value is not None and value > 0:
            return value

    bid = _nbbo_bid(row)
    bid_over_entry_ask = _as_float(features.get("bid_over_entry_ask"))
    if bid is not None and bid_over_entry_ask is not None and bid_over_entry_ask > 0:
        return bid / bid_over_entry_ask

    current_pnl = _as_float(features.get("current_pnl"))
    if bid is not None and current_pnl is not None:
        return bid - (current_pnl / _CONTRACT_MULTIPLIER)

    ask = _nbbo_ask(row)
    return ask if ask is not None and ask > 0 else None


def _checks(ledgers: list[ShadowTradeLedger], global_errors: list[str], *, config: ShadowPaperConfig) -> list[dict[str, Any]]:
    all_errors = [error for ledger in ledgers for error in ledger.errors] + global_errors
    checks = [
        {
            "name": "no_order_fields",
            "status": "pass" if not any("broker/order field" in error for error in all_errors) else "fail",
            "detail": "Shadow-paper replay found no broker order IDs, submitted orders, or order intents.",
        },
        {
            "name": "one_contract",
            "status": "pass" if not any("intended_size" in error for error in all_errors) else "fail",
            "detail": f"Every row must remain intended_size={config.intended_size}.",
        },
        {
            "name": "spxw_pm_contracts",
            "status": "pass" if not any("SPXW" in error or "contract" in error for error in all_errors) else "fail",
            "detail": "Only PM-settled SPXW contracts are allowed.",
        },
        {
            "name": "executable_bid_ask_prices",
            "status": "pass" if not any("price" in error or "bid" in error for error in all_errors) else "fail",
            "detail": "Entry is reconstructed from ask-equivalent fields; exit/mark uses bid.",
        },
        {
            "name": "all_trades_closed",
            "status": "pass"
            if all(ledger.status == "closed" for ledger in ledgers)
            else ("fail" if config.require_all_closed else "warn"),
            "detail": "A full-day paper/replay promotion stream should end flat.",
        },
        {
            "name": "terminal_action_final",
            "status": "pass"
            if all(ledger.post_terminal_row_count == 0 for ledger in ledgers)
            else ("fail" if config.require_terminal_final else "warn"),
            "detail": "After a live exit/stop/forced-flat action, later rows for the same trade should not remain holding.",
        },
        {
            "name": "global_one_position",
            "status": "pass"
            if _max_concurrent_positions(ledgers) <= 1
            else ("fail" if config.enforce_global_one_position else "warn"),
            "detail": "Live shadow paper should never hold more than one contract at a time.",
            "value": _max_concurrent_positions(ledgers),
        },
    ]
    return checks


def _status_from_checks(checks: list[dict[str, Any]], ledgers: list[ShadowTradeLedger], *, config: ShadowPaperConfig) -> str:
    if not ledgers:
        return "blocked"
    statuses = {check["status"] for check in checks}
    if "fail" in statuses or any(ledger.errors for ledger in ledgers):
        return "fail"
    if "warn" in statuses or any(ledger.warnings for ledger in ledgers):
        return "warn"
    return "pass"


def _max_concurrent_positions(ledgers: list[ShadowTradeLedger]) -> int:
    events: list[tuple[str, int]] = []
    for ledger in ledgers:
        start = ledger.entry_time or ledger.first_observed_time
        end = ledger.exit_time or ledger.last_observed_time
        if not start or not end:
            continue
        events.append((start, 1))
        events.append((end, -1))
    current = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def _pnl_summary(values: list[float | None]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return {"count": 0, "sum": 0.0, "min": None, "median": None, "max": None}
    return {
        "count": len(clean),
        "sum": float(sum(clean)),
        "min": clean[0],
        "median": clean[len(clean) // 2] if len(clean) % 2 else (clean[len(clean) // 2 - 1] + clean[len(clean) // 2]) / 2.0,
        "max": clean[-1],
    }


def _ledger_to_dict(ledger: ShadowTradeLedger) -> dict[str, Any]:
    return {
        "trade_uid": ledger.trade_uid,
        "contract_id": ledger.contract_id,
        "side": ledger.side,
        "entry_time": ledger.entry_time,
        "first_observed_time": ledger.first_observed_time,
        "exit_time": ledger.exit_time,
        "last_observed_time": ledger.last_observed_time,
        "entry_fill_price": ledger.entry_fill_price,
        "exit_fill_price": ledger.exit_fill_price,
        "mark_price": ledger.mark_price,
        "status": ledger.status,
        "terminal_action": ledger.terminal_action,
        "row_count": ledger.row_count,
        "post_terminal_row_count": ledger.post_terminal_row_count,
        "realized_pnl": ledger.realized_pnl,
        "mark_pnl": ledger.mark_pnl,
        "min_mark_pnl": ledger.min_mark_pnl,
        "max_mark_pnl": ledger.max_mark_pnl,
        "errors": ledger.errors,
        "warnings": ledger.warnings,
    }


def _action(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    decision = _object(row.get("decision"))
    action = decision.get("action", row.get("action"))
    return str(action) if action is not None else None


def _side_from_contract(contract_id: str, row: dict[str, Any]) -> str:
    if contract_id.endswith("-C"):
        return "call"
    if contract_id.endswith("-P"):
        return "put"
    features = _object(row.get("features"))
    if _as_float(features.get("entry_is_call")) == 1.0:
        return "call"
    if _as_float(features.get("entry_is_put")) == 1.0:
        return "put"
    return "unknown"


def _nbbo_bid(row: dict[str, Any]) -> float | None:
    nbbo = _object(row.get("nbbo"))
    return _as_float(nbbo.get("bid", row.get("bid")))


def _nbbo_ask(row: dict[str, Any]) -> float | None:
    nbbo = _object(row.get("nbbo"))
    return _as_float(nbbo.get("ask", row.get("ask")))


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _truthy_order_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return False
    return True
