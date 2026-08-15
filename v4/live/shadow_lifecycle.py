"""Utilities for making shadow JSONL streams obey live lifecycle constraints."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


TERMINAL_ACTIONS = {"exit", "stop", "forced_flat"}


@dataclass(frozen=True)
class StrictShadowLifecycleResult:
    """Strict serial replay result for a no-order shadow stream."""

    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def strict_serial_shadow_rows(rows: list[dict[str, Any]], *, require_terminal: bool = True) -> StrictShadowLifecycleResult:
    """Truncate terminal paths and keep only one open trade at a time.

    The input can be an offline selected-trade path rehearsal where many
    candidate trades overlap. The output is a live-like no-order stream:
    once a trade exits/stops/forced-flats, later rows for that trade are
    removed, and any candidate whose entry occurs while a prior selected trade
    is open is skipped.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    input_trade_order: dict[str, int] = {}
    for index, row in enumerate(rows):
        trade_uid = str(row.get("trade_uid") or f"missing-trade-uid-{index}")
        grouped.setdefault(trade_uid, []).append(row)
        input_trade_order.setdefault(trade_uid, index)

    candidates = []
    skipped_missing_terminal = 0
    post_terminal_rows_removed = 0
    for trade_uid, trade_rows in grouped.items():
        ordered = sorted(
            trade_rows,
            key=lambda row: (_as_int(row.get("timestamp_ms")) or 0, _as_int(row.get("sequence_step_index")) or 0),
        )
        terminal_index = next((index for index, row in enumerate(ordered) if _action(row) in TERMINAL_ACTIONS), None)
        if terminal_index is None:
            if require_terminal:
                skipped_missing_terminal += 1
                continue
            terminal_index = len(ordered) - 1
        terminal_row = ordered[terminal_index]
        truncated = ordered[: terminal_index + 1]
        post_terminal_rows_removed += len(ordered) - len(truncated)
        entry_time = _parse_time(truncated[0].get("entry_decision_time") or truncated[0].get("decision_time"))
        exit_time = _parse_time(terminal_row.get("decision_time"))
        if entry_time is None or exit_time is None:
            skipped_missing_terminal += 1
            continue
        candidates.append(
            {
                "trade_uid": trade_uid,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "terminal_action": _action(terminal_row),
                "rows": truncated,
                "input_order": input_trade_order[trade_uid],
            }
        )

    selected = []
    skipped_overlap = []
    open_until: datetime | None = None
    for candidate in sorted(candidates, key=lambda item: (item["entry_time"], item["input_order"], item["trade_uid"])):
        if open_until is not None and candidate["entry_time"] < open_until:
            skipped_overlap.append(candidate)
            continue
        selected.append(candidate)
        open_until = candidate["exit_time"]

    strict_rows = []
    for candidate in selected:
        strict_rows.extend(candidate["rows"])

    summary = {
        "input_rows": len(rows),
        "input_trades": len(grouped),
        "candidate_trades": len(candidates),
        "selected_trades": len(selected),
        "output_rows": len(strict_rows),
        "skipped_overlap_trades": len(skipped_overlap),
        "skipped_missing_terminal_trades": skipped_missing_terminal,
        "post_terminal_rows_removed": post_terminal_rows_removed,
        "terminal_action_counts": _count(candidate["terminal_action"] for candidate in selected),
        "selected_trade_uids": [candidate["trade_uid"] for candidate in selected],
        "skipped_overlap_trade_uids": [candidate["trade_uid"] for candidate in skipped_overlap],
    }
    return StrictShadowLifecycleResult(rows=strict_rows, summary=summary)


def _action(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    decision = row.get("decision")
    if isinstance(decision, dict) and decision.get("action") is not None:
        return str(decision["action"])
    action = row.get("action")
    return str(action) if action is not None else None


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
