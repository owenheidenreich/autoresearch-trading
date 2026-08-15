"""Build a post-session packet for the 2026-05-26 no-order evidence run."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any


DEFAULT_SESSION_DATE = "2026-05-26"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/tuesday_no_order_evidence_packet")
PROTOCOL245_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest")
PROTOCOL160_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_160_protocol101_persistent_paper_trader")
PAPER_FILL_ROOT = Path("v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation")
TRADE_LOG_ROOT = Path("v4/logs/paper_trading")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--session-date", default=DEFAULT_SESSION_DATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_packet(repo_root=args.repo_root, session_date=str(args.session_date))
    write_outputs(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def build_packet(*, repo_root: Path, session_date: str) -> dict[str, Any]:
    root = repo_root.resolve()
    protocol245 = collect_protocol_summaries(root / PROTOCOL245_ROOT / session_date)
    protocol160 = collect_protocol_summaries(root / PROTOCOL160_ROOT / session_date)
    paper_fill = collect_protocol_summaries(root / PAPER_FILL_ROOT / session_date)
    execution_observations = inspect_execution_observations(root / PAPER_FILL_ROOT / session_date)
    trade_log = inspect_trade_logs(root / TRADE_LOG_ROOT / session_date)
    fill = load_summary(root / "v4/audit/autoresearch/v4_aplus_hypothesis_272_fill_model_readiness/summary.json")
    holdout = load_summary(root / "v4/audit/autoresearch/untouched_holdout_availability/summary.json")
    no_order_readiness = load_summary(root / "v4/audit/autoresearch/live_no_order_full_action_parity_readiness/summary.json")
    neural = load_summary(root / "v4/audit/autoresearch/unified_neural_training_readiness/summary.json")
    section = load_summary(root / "v4/audit/autoresearch/project_section_readiness/summary.json")
    decision = decide(protocol245=protocol245, protocol160=protocol160, paper_fill=paper_fill, trade_log=trade_log)
    return {
        "role_label": "TUESDAY_NO_ORDER_EVIDENCE_PACKET_V1",
        "session_date": session_date,
        "generated_date": date.today().isoformat(),
        "decision": decision,
        "changes_paper_default": False,
        "model_training": False,
        "threshold_tuning": False,
        "protected_holdout_scored": False,
        "live_orders": False,
        "paper_order_permission": "approved_for_bounded_2026_05_26_observation_only",
        "protocol245_live_surface": protocol245,
        "protocol160_protocol101_intent_shadow": protocol160,
        "paper_fill_observations": paper_fill,
        "execution_observations": execution_observations,
        "trade_log": trade_log,
        "readiness_summaries": {
            "fill_model": compact_fill_summary(fill),
            "untouched_holdout": compact_keys(holdout, ["decision", "data_status", "data_available", "protected_holdout_scored"]),
            "live_no_order_parity": compact_keys(no_order_readiness, ["decision", "broker_endpoint_called", "live_orders"]),
            "unified_neural_training": compact_keys(neural, ["training_decision", "protocol101_challenge_decision", "challenge_blockers"]),
            "project_sections": compact_keys(section, ["section_1_2_decision", "model_hill_climb_decision", "model_hill_climb_blockers"]),
        },
        "remaining_human_confirmation_only": [
            "IBKR two-factor approval if Gateway asks for it.",
            "Any paper-submit observation run for real fill/cancel/timeout evidence.",
            "Any new model training, threshold tuning, protected holdout scoring, or challenger promotion.",
        ],
    }


def collect_protocol_summaries(path: Path) -> dict[str, Any]:
    summaries = sorted(path.glob("*/summary.json")) if path.exists() else []
    rows = [compact_protocol_summary(summary) for summary in summaries]
    return {
        "summary_root": str(path),
        "summary_count": len(rows),
        "decisions": [row["decision"] for row in rows],
        "runs": rows,
        "pass_count": sum(1 for row in rows if is_positive_decision(str(row.get("decision", "")))),
        "blocked_count": sum(1 for row in rows if str(row.get("decision", "")).startswith("blocked_")),
    }


def compact_protocol_summary(path: Path) -> dict[str, Any]:
    obj = load_summary(path)
    return {
        "path": str(path),
        "decision": obj.get("decision"),
        "run_id": obj.get("run_id"),
        "trade_log": obj.get("trade_log"),
        "decision_count": obj.get("decision_count"),
        "enter_intents": obj.get("enter_intents"),
        "paper_orders_submitted": obj.get("paper_orders_submitted"),
        "broker_order_endpoint_called": obj.get("broker_order_endpoint_called"),
        "ibkr_connected": obj.get("ibkr_connected"),
        "candidate_breadth": obj.get("candidate_breadth"),
        "latency_summary": obj.get("latency_summary"),
        "blocked_reason": obj.get("blocked_reason"),
    }


def is_positive_decision(decision: str) -> bool:
    return decision.startswith("pass_") or decision in {
        "paper_fill_observations_collected_for_execution_truth_review",
        "execution_truth_packet_ready_keep_stress_replay",
    }


def inspect_trade_logs(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.jsonl")) if path.exists() else []
    event_counts: dict[str, int] = {}
    broker_order_endpoint_called_rows = 0
    paper_order_submitted_rows = 0
    paper_fill_rows = 0
    parse_errors = 0
    rows = 0
    for file in files:
        with file.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                event_type = str(obj.get("event_type") or "")
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                if obj.get("broker_order_endpoint_called"):
                    broker_order_endpoint_called_rows += 1
                if event_type == "paper_order_submitted":
                    paper_order_submitted_rows += 1
                if event_type in {"paper_entry_fill", "paper_exit_fill"}:
                    paper_fill_rows += 1
    return {
        "root": str(path),
        "files": [str(file) for file in files],
        "file_count": len(files),
        "rows": rows,
        "event_counts": event_counts,
        "broker_order_endpoint_called_rows": broker_order_endpoint_called_rows,
        "paper_order_submitted_rows": paper_order_submitted_rows,
        "paper_fill_rows": paper_fill_rows,
        "parse_errors": parse_errors,
    }


def inspect_execution_observations(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*/execution_observations.jsonl")) if path.exists() else []
    rows = []
    parse_errors = 0
    for file in files:
        with file.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    parse_errors += 1
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("fill_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "files": [str(file) for file in files],
        "file_count": len(files),
        "rows": len(rows),
        "status_counts": status_counts,
        "filled_round_trips": sum(1 for row in rows if row.get("fill_status") == "filled" and row.get("exit_fill_status") == "filled"),
        "open_position_risk_rows": sum(1 for row in rows if row.get("open_position_risk")),
        "broker_order_endpoint_called_rows": sum(1 for row in rows if row.get("broker_order_endpoint_called")),
        "parse_errors": parse_errors,
    }


def decide(*, protocol245: dict[str, Any], protocol160: dict[str, Any], paper_fill: dict[str, Any], trade_log: dict[str, Any]) -> str:
    if trade_log["broker_order_endpoint_called_rows"] and not paper_fill["summary_count"]:
        return "fail_unattributed_broker_order_endpoint_rows"
    if paper_fill["pass_count"]:
        return "pass_tuesday_paper_fill_observations_ready_for_review"
    if paper_fill["summary_count"]:
        return "partial_tuesday_paper_fill_observations_review_required"
    if protocol245["pass_count"] and protocol160["pass_count"] and trade_log["rows"]:
        return "pass_tuesday_no_order_evidence_ready_for_review"
    if protocol245["summary_count"] or protocol160["summary_count"] or trade_log["rows"]:
        return "partial_tuesday_no_order_evidence_review_required"
    return "blocked_no_tuesday_evidence_found"


def compact_fill_summary(obj: dict[str, Any]) -> dict[str, Any]:
    readiness = obj.get("readiness") if isinstance(obj.get("readiness"), dict) else {}
    row_counts = obj.get("row_counts") if isinstance(obj.get("row_counts"), dict) else {}
    return {
        "decision": obj.get("decision"),
        "fill_observations": readiness.get("fill_observations", row_counts.get("fill_observations")),
        "required_fill_observations": readiness.get("required_fill_observations"),
        "status": readiness.get("status"),
        "reason": readiness.get("reason"),
    }


def compact_keys(obj: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: obj.get(key) for key in keys if key in obj}


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out_dir / "report.md").write_text(render_report(payload))


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Tuesday No-Order Evidence Packet",
        "",
        f"Session date: `{payload['session_date']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "No model training, threshold tuning, protected holdout scoring, or order submission is authorized by this packet.",
        "",
        "## Runtime Evidence",
        "",
        f"- Protocol245 summaries: `{payload['protocol245_live_surface']['summary_count']}`; passes: `{payload['protocol245_live_surface']['pass_count']}`",
        f"- Protocol160 summaries: `{payload['protocol160_protocol101_intent_shadow']['summary_count']}`; passes: `{payload['protocol160_protocol101_intent_shadow']['pass_count']}`",
        f"- Paper fill observation summaries: `{payload['paper_fill_observations']['summary_count']}`; passes: `{payload['paper_fill_observations']['pass_count']}`",
        f"- Execution observation rows: `{payload['execution_observations']['rows']}`; filled round trips: `{payload['execution_observations']['filled_round_trips']}`",
        f"- Trade log files: `{payload['trade_log']['file_count']}`; rows: `{payload['trade_log']['rows']}`",
        f"- Broker order endpoint rows: `{payload['trade_log']['broker_order_endpoint_called_rows']}`",
        f"- Paper order submitted rows: `{payload['trade_log']['paper_order_submitted_rows']}`",
        f"- Fill rows: `{payload['trade_log']['paper_fill_rows']}`",
        "",
        "## Readiness Snapshot",
        "",
    ]
    for name, summary in payload["readiness_summaries"].items():
        lines.append(f"- {name}: `{summary}`")
    lines.extend(["", "## Human Confirmation Only", ""])
    lines.extend(f"- {item}" for item in payload["remaining_human_confirmation_only"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
