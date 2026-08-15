"""Readiness packet for live no-order full-action parity.

This check is intentionally non-invasive: it does not connect to IBKR, request
market data, submit orders, or mutate runtime flags. On closed-market days it
can only certify that the next live no-order session is scaffolded.
"""
from __future__ import annotations

from datetime import datetime, time
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROLE_LABEL = "LIVE_NO_ORDER_FULL_ACTION_PARITY_READINESS_V1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/live_no_order_full_action_parity_readiness")
NY = ZoneInfo("America/New_York")

LIVE_PARITY_REQUIRED_FIELDS = (
    "decision_timestamp",
    "received_timestamp",
    "raw_quote_timestamp",
    "quote_age_ms",
    "candidate_count",
    "candidate_set_hash",
    "feature_vector_hash",
    "action_mask",
    "raw_logits",
    "selected_action",
    "selected_contract",
    "risk_gate",
    "latency_ms",
    "broker_endpoint_called",
    "live_orders_enabled",
)


def build_live_no_order_parity_readiness(repo_root: Path = Path("."), *, now: datetime | None = None) -> dict[str, Any]:
    root = repo_root.resolve()
    now_et = normalize_et(now or datetime.now(tz=NY))
    market = market_status(now_et)
    scaffolds = {
        "protocol166_contract": (root / "v4/live/protocol166_parity_contract.py").exists(),
        "protocol166_runner": (root / "v4/scripts/run_protocol166_live_training_parity_contract.py").exists(),
        "protocol158_quote_age_truth": protocol158_quote_age_truth(root),
        "formal_validation_governance": (root / "v4/audit/autoresearch/formal_validation_governance/summary.json").exists(),
    }
    missing = [name for name, present in scaffolds.items() if not present]
    if missing:
        decision = "live_no_order_full_action_parity_blocked_missing_scaffold"
    elif not market["is_regular_market_session"]:
        decision = "live_no_order_full_action_parity_waiting_for_open_market_session"
    else:
        decision = "live_no_order_full_action_parity_ready_to_run_no_order_session"
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "read-only readiness packet for next live no-order full-action parity session",
        "changes_paper_default": False,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "decision": decision,
        "market_status": market,
        "scaffolds": scaffolds,
        "missing_scaffolds": missing,
        "required_jsonl_fields": list(LIVE_PARITY_REQUIRED_FIELDS),
        "next_allowed_work": next_allowed_work(decision),
    }


def normalize_et(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=NY)
    return value.astimezone(NY)


def market_status(now_et: datetime) -> dict[str, Any]:
    weekday_open = now_et.weekday() < 5
    in_hours = time(9, 30) <= now_et.time() <= time(16, 0)
    is_open = bool(weekday_open and in_hours)
    return {
        "now_et": now_et.isoformat(),
        "weekday": now_et.strftime("%A"),
        "is_weekday": weekday_open,
        "is_regular_market_hours": in_hours,
        "is_regular_market_session": is_open,
        "holiday_calendar_checked": False,
        "note": "weekday/time check only; live no-order runner must still enforce exchange calendar and market data availability",
    }


def protocol158_quote_age_truth(root: Path) -> bool:
    bridge = root / "v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py"
    if not bridge.exists():
        return False
    text = bridge.read_text()
    return '"quote_age_ms": 0' not in text and "quote_freshness_from_ticker" in text


def next_allowed_work(decision: str) -> list[str]:
    if decision == "live_no_order_full_action_parity_ready_to_run_no_order_session":
        return [
            "Run only a no-order session with broker endpoint disabled and live_orders_enabled=false.",
            "Log the full required JSONL fields and reconstruct the decision offline before any paper-submit work.",
        ]
    if decision == "live_no_order_full_action_parity_waiting_for_open_market_session":
        return [
            "Market is closed; do not attempt broker/data endpoint collection from this packet.",
            "At the next regular market session, run no-order parity only with explicit human confirmation.",
        ]
    return [
        "Restore the missing scaffold before any live no-order parity attempt.",
        "Do not run paper-submit or live orders.",
    ]


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"Decision: `{payload['decision']}`",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Live orders: no",
        "Model training: no",
        "",
        "## Market Status",
        "",
        f"- Now ET: `{payload['market_status']['now_et']}`",
        f"- Weekday: `{payload['market_status']['weekday']}`",
        f"- Regular session now: `{payload['market_status']['is_regular_market_session']}`",
        "",
        "## Scaffold Checks",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in payload["scaffolds"].items())
    lines.extend(["", "## Required JSONL Fields", ""])
    lines.append(", ".join(f"`{field}`" for field in payload["required_jsonl_fields"]))
    lines.extend(["", "## Next Allowed Work", ""])
    lines.extend(f"- {item}" for item in payload["next_allowed_work"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.json"
    report = out_dir / "report.md"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report.write_text(render_report(payload))
    return summary, report
