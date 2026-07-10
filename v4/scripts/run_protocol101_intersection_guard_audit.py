"""Protocol101 intersection-guard audit for canonical Stage-1 preparation.

This offline audit measures how many burned-day static-ladder slots pass the
boundary-stable guard on both historical and IBKR planes. It records the
pessimistic guard policy that future vendor-only training should use, without
modifying runtime guard code or the frozen canonical v1.4 contract.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v4.scripts import run_protocol101_feature_recovery_group1_parity_audit as g1
from v4.scripts import run_protocol101_static_ladder_boundary_stable_policy_audit as static_audit


SCHEMA_VERSION = "Protocol101IntersectionGuardAuditV1"
BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_canonical_v1_intersection_guard_audit"
DEFAULT_STATIC_AUDIT_DIR = BASE_AUDIT / "protocol101_live_v2_static_ladder_boundary_stable_policy_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--static-audit-dir", type=Path, default=DEFAULT_STATIC_AUDIT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def slot_key(slot: dict[str, Any]) -> tuple[str, str]:
    return (f"{float(slot.get('strike')):.3f}", str(slot.get("right") or ""))


def audit(static_summary: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = static_summary["policy_replay_metrics"]
    margins = static_summary["frozen_margins"]
    first_et = metrics["required_first_decision_et"]
    last_et = metrics["required_last_decision_et"]
    trace_prefix = metrics["trace_prefix"]
    daily_rows: list[dict[str, Any]] = []
    aggregate = Counter()
    status_pairs = Counter()

    for session in metrics["required_sessions"]:
        _live_path, _historical_path, live_rows, historical_rows = static_audit.load_required_inputs(
            trace_prefix, session
        )
        window_keys = [
            ts for ts in sorted(set(live_rows) & set(historical_rows)) if g1.in_window(ts, first_et, last_et)
        ]
        daily = Counter()
        daily_status_pairs = Counter()
        for ts in window_keys:
            live_slots = {slot_key(item): item for item in static_audit.static_slots(live_rows[ts])}
            historical_slots = {
                slot_key(item): item for item in static_audit.static_slots(historical_rows[ts])
            }
            for key in sorted(set(live_slots) & set(historical_slots)):
                live_status = static_audit.guard_status(
                    live_slots[key],
                    margins,
                    cash=static_audit.cash_for_row(live_rows[ts]),
                )
                historical_status = static_audit.guard_status(
                    historical_slots[key],
                    margins,
                    cash=static_audit.cash_for_row(historical_rows[ts]),
                )
                historical_boundary = bool(historical_status["boundary_stable"])
                live_boundary = bool(live_status["boundary_stable"])
                daily["paired_static_slots"] += 1
                daily["historical_boundary_stable"] += int(historical_boundary)
                daily["ibkr_boundary_stable"] += int(live_boundary)
                daily["intersection_boundary_stable"] += int(historical_boundary and live_boundary)
                daily["historical_only_boundary_stable"] += int(historical_boundary and not live_boundary)
                daily["ibkr_only_boundary_stable"] += int(live_boundary and not historical_boundary)
                pair = f"{historical_status['status']}->{live_status['status']}"
                daily_status_pairs[pair] += 1
        aggregate.update(daily)
        status_pairs.update(daily_status_pairs)
        daily_rows.append(
            {
                "session": session,
                "decision_rows": len(window_keys),
                **{key: int(value) for key, value in daily.items()},
                "historical_excess_vs_intersection": int(
                    daily["historical_boundary_stable"] - daily["intersection_boundary_stable"]
                ),
                "ibkr_excess_vs_intersection": int(
                    daily["ibkr_boundary_stable"] - daily["intersection_boundary_stable"]
                ),
                "status_pair_counts": dict(sorted(daily_status_pairs.items())),
            }
        )

    historical_count = int(aggregate["historical_boundary_stable"])
    ibkr_count = int(aggregate["ibkr_boundary_stable"])
    intersection_count = int(aggregate["intersection_boundary_stable"])
    gap_closed_from_historical = (
        (historical_count - intersection_count) / historical_count if historical_count else 0.0
    )
    asymmetry_vs_ibkr = (
        (historical_count - ibkr_count) / ibkr_count if ibkr_count else 0.0
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "source_static_audit": str(DEFAULT_STATIC_AUDIT_DIR),
        "trace_prefix": trace_prefix,
        "sessions": metrics["required_sessions"],
        "decision_rows_checked": int(sum(row["decision_rows"] for row in daily_rows)),
        "paired_static_slots": int(aggregate["paired_static_slots"]),
        "historical_boundary_stable": historical_count,
        "ibkr_boundary_stable": ibkr_count,
        "intersection_boundary_stable": intersection_count,
        "historical_only_boundary_stable": int(aggregate["historical_only_boundary_stable"]),
        "ibkr_only_boundary_stable": int(aggregate["ibkr_only_boundary_stable"]),
        "historical_excess_vs_ibkr_fraction": asymmetry_vs_ibkr,
        "historical_excess_closed_by_intersection_fraction": gap_closed_from_historical,
        "vendor_only_training_guard_policy": {
            "policy_id": "protocol101_canonical_v1_intersection_guard_vendor_only_v1",
            "basis": "burned_day_exact_intersection_between_historical_and_ibkr_boundary_stable_guard_states",
            "use_boundary_stable_margins": margins,
            "extra_historical_acceptance_haircut_fraction": gap_closed_from_historical,
            "application": "training/evaluation pessimistic tradability only; not a live runtime guard change",
        },
        "status_pair_counts": dict(sorted(status_pairs.items())),
        "side_effects": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_downloaded": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
            "sealed_market_data_read": False,
        },
    }
    return summary, daily_rows


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and not args.force:
        raise SystemExit(f"{args.out_dir} exists; pass --force to overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    static_summary = load_json(args.static_audit_dir / "summary.json")
    summary, daily_rows = audit(static_summary)
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "daily_intersection_counts.json", {"daily": daily_rows})
    report = [
        "# Protocol101 Canonical Intersection Guard Audit",
        "",
        "Offline burned-day audit. No runtime guard, broker, paper, paid-data, promotion, launchd, or real-money state changed.",
        "",
        f"- Paired static slots: `{summary['paired_static_slots']}`",
        f"- Historical boundary-stable: `{summary['historical_boundary_stable']}`",
        f"- IBKR boundary-stable: `{summary['ibkr_boundary_stable']}`",
        f"- Exact intersection boundary-stable: `{summary['intersection_boundary_stable']}`",
        f"- Historical-only boundary-stable: `{summary['historical_only_boundary_stable']}`",
        f"- IBKR-only boundary-stable: `{summary['ibkr_only_boundary_stable']}`",
        f"- Historical excess vs IBKR: `{summary['historical_excess_vs_ibkr_fraction']:.4%}`",
        f"- Historical acceptance haircut to exact intersection: `{summary['historical_excess_closed_by_intersection_fraction']:.4%}`",
        "",
        "The `vendor_only_training_guard_policy` in `summary.json` is the pessimistic guard policy Stage-1 should reference.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
