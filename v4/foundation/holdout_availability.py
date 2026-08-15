"""Untouched holdout availability preflight.

This is a governance check only. It never collects data, purchases data, scores
models, or mutates the reservation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v4.model.neural_training_readiness import (
    BLOCKED,
    PASS,
    PAPER_DEFAULT_BASELINE,
    validate_untouched_holdout_reservation,
)


ROLE_LABEL = "UNTOUCHED_HOLDOUT_AVAILABILITY_PREFLIGHT_V1"
DEFAULT_RESERVATION_SUMMARY = Path("v4/audit/autoresearch/unified_untouched_holdout_reservation/summary.json")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/untouched_holdout_availability")


def build_holdout_availability(repo_root: Path = Path(".")) -> dict[str, Any]:
    root = repo_root.resolve()
    reservation_summary = read_json(root / DEFAULT_RESERVATION_SUMMARY)
    reservation = reservation_summary.get("reservation") if isinstance(reservation_summary.get("reservation"), dict) else {}
    validation = validate_untouched_holdout_reservation(reservation) if reservation else {
        "status": BLOCKED,
        "errors": ["missing_holdout_reservation"],
        "warnings": [],
        "data_available": False,
        "data_status": "missing",
        "split_label": "",
        "is_exposed_diagnostic_split": False,
    }
    protected_scored = bool(validation.get("data_status") == "scored_once_frozen")
    data_available = bool(validation.get("data_available"))
    if validation.get("status") != PASS:
        decision = "untouched_holdout_availability_blocked_invalid_reservation"
    elif data_available:
        decision = "untouched_holdout_data_available_frozen"
    else:
        decision = "untouched_holdout_data_pending_collection"
    return {
        "role_label": ROLE_LABEL,
        "what_is_this": "read-only untouched holdout data availability and no-score preflight",
        "changes_paper_default": False,
        "paper_default_baseline": PAPER_DEFAULT_BASELINE,
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "protected_holdout_scored": protected_scored,
        "data_available": data_available,
        "data_status": validation.get("data_status", "missing"),
        "split_label": validation.get("split_label", ""),
        "validation": validation,
        "decision": decision,
        "next_allowed_work": next_allowed_work(decision),
    }


def next_allowed_work(decision: str) -> list[str]:
    if decision == "untouched_holdout_data_available_frozen":
        return [
            "Do not score the frozen holdout until candidate, baseline, metrics, fill model, and parity packet are frozen.",
            "Use the holdout once for the final Protocol101 challenge packet only.",
        ]
    return [
        "Do not score any protected or future holdout data.",
        "Keep current exposed splits diagnostic only.",
        "Collect/freeze the reserved future block only after model and promotion packet rules are fixed.",
    ]


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"Decision: `{payload['decision']}`",
        "Does it change the paper-trading default: no",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training: no",
        "Protected holdout scored: no" if not payload["protected_holdout_scored"] else "Protected holdout scored: yes",
        "",
        "## Availability",
        "",
        f"- Data status: `{payload['data_status']}`",
        f"- Data available: `{payload['data_available']}`",
        f"- Split label: `{payload['split_label']}`",
        f"- Validation status: `{payload['validation']['status']}`",
        "",
        "## Next Allowed Work",
        "",
    ]
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
