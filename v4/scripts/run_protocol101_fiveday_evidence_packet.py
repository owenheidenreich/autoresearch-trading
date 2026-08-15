"""Build a post-session evidence packet for the Protocol101 five-day run.

This script reads local logs and summaries only. It does not call IBKR, submit
orders, download paid data, train models, tune thresholds, or mutate runtime
flags.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time
import json
from pathlib import Path
import subprocess
from typing import Any
from zoneinfo import ZoneInfo


TARGET_DATES = ("2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12")
FIVEDAY_LABELS = (
    "com.autoresearch.protocol101.fiveday.ibgateway.paper",
    "com.autoresearch.protocol101.fiveday.paper-preflight",
    "com.autoresearch.protocol101.fiveday.paper-session",
    "com.autoresearch.protocol101.fiveday.daily-monitor",
    "com.autoresearch.protocol101.fiveday.paper-shutdown",
    "com.autoresearch.protocol101.fiveday.evidence-postsession",
)
DAILY_LABELS = (
    "com.autoresearch.ibgateway.paper",
    "com.autoresearch.protocol101.paper-preflight",
    "com.autoresearch.protocol101.paper-session",
    "com.autoresearch.protocol101.daily-monitor",
    "com.autoresearch.ibgateway.paper-shutdown",
)
DEFAULT_TRADE_LOG_ROOT = Path("v4/logs/paper_trading")
DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/protocol101_fiveday_evidence_packet")
DEFAULT_RUNTIME_FLAG = Path("v4/runtime/protocol101_paper_order_enablement.json")
DEFAULT_ENTITLEMENT_SUMMARY = Path("v4/audit/ibkr_live_data_entitlements/summary.json")
LAUNCHD_LOG_DIR = Path.home() / "Library/Logs/autoresearch-trading"
PACIFIC = ZoneInfo("America/Los_Angeles")
MIN_FIRST_EVENT_PACIFIC = time(6, 35)
# The persistent trader polls on a minute loop, so a final regular-session
# decision just before the 13:00 PT close is the expected terminal sample.
MIN_LAST_EVENT_PACIFIC = time(12, 59)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--session-date", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--runtime-flag", type=Path, default=DEFAULT_RUNTIME_FLAG)
    parser.add_argument("--entitlement-summary", type=Path, default=DEFAULT_ENTITLEMENT_SUMMARY)
    parser.add_argument("--target-date", action="append", default=None)
    parser.add_argument("--label-prefix", default="com.autoresearch.protocol101.fiveday")
    parser.add_argument("--log-prefix", default="protocol101-fiveday")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = str(args.session_date or datetime.now(tz=PACIFIC).date().isoformat())
    run_id = str(args.run_id or f"daily_paper_autopilot_{session}")
    payload = build_packet(
        repo_root=args.repo_root,
        session=session,
        run_id=run_id,
        trade_log_root=args.trade_log_root,
        runtime_flag=args.runtime_flag,
        entitlement_summary=args.entitlement_summary,
        target_dates=tuple(args.target_date or TARGET_DATES),
        label_prefix=str(args.label_prefix),
        log_prefix=str(args.log_prefix),
    )
    out_dir = args.out_root / session / run_id
    write_outputs(out_dir, payload)
    print(json.dumps({"decision": payload["decision"], "report": str(out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def build_packet(
    *,
    repo_root: Path,
    session: str,
    run_id: str,
    trade_log_root: Path,
    runtime_flag: Path,
    entitlement_summary: Path,
    target_dates: tuple[str, ...] = TARGET_DATES,
    label_prefix: str = "com.autoresearch.protocol101.fiveday",
    log_prefix: str = "protocol101-fiveday",
) -> dict[str, Any]:
    root = repo_root.resolve()
    trade_log = inspect_trade_logs(root / trade_log_root / session, run_id=run_id)
    launchd = collect_launchd_status(label_prefix=label_prefix)
    launchd_logs = collect_launchd_logs(log_prefix=log_prefix)
    runtime = load_json(root / runtime_flag)
    entitlement = load_json(root / entitlement_summary)
    analyzer = load_json(root / "v4/audit/autoresearch/v4_aplus_hypothesis_148_protocol101_post_session_analyzer" / session / run_id / "summary.json")
    monitor = load_json(root / "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor" / session / run_id / "summary.json")
    decision = decide_packet(session=session, trade_log=trade_log, runtime=runtime, target_dates=target_dates)
    return {
        "role_label": "PROTOCOL101_FIVEDAY_PAPER_SUBMIT_EVIDENCE_PACKET_V1",
        "what_is_this": "post-session local evidence packet for the date-scoped Protocol101 paper-submit collection",
        "session": session,
        "run_id": run_id,
        "target_dates": list(target_dates),
        "generated_date": date.today().isoformat(),
        "generated_at_pacific": datetime.now(tz=PACIFIC).isoformat(),
        "decision": decision,
        "changes_paper_default": False,
        "model_training": False,
        "threshold_tuning": False,
        "paid_data_downloaded": False,
        "protected_holdout_scored": False,
        "paper_submit_mode_requested": True,
        "real_money_trading": False,
        "runtime_flag": summarize_runtime_flag(runtime),
        "trade_log": trade_log,
        "launchd": launchd,
        "launchd_logs": launchd_logs,
        "entitlement_summary": compact_keys(
            entitlement,
            ["decision", "ibkr_connected", "spx_live_price", "vix_live_price", "spxw_live_nbbo_rows"],
        ),
        "post_session_analyzer": compact_keys(analyzer, ["decision", "source_trade_log", "next_gate"]),
        "daily_monitor": compact_keys(monitor, ["decision", "source_trade_log", "paper_orders_submitted", "broker_order_endpoint_called"]),
        "completion_checks": completion_checks(session=session, trade_log=trade_log, runtime=runtime, target_dates=target_dates),
        "next_gate": next_gate(decision),
    }


def inspect_trade_logs(path: Path, *, run_id: str) -> dict[str, Any]:
    files = sorted(path.glob("*.jsonl")) if path.exists() else []
    preferred = [file for file in files if file.stem == run_id]
    active_files = preferred or files
    event_counts: Counter[str] = Counter()
    model_actions: Counter[str] = Counter()
    risk_reasons: Counter[str] = Counter()
    broker_order_endpoint_called_rows = 0
    paper_order_submitted_rows = 0
    paper_fill_rows = 0
    parse_errors = 0
    rows = 0
    first_timestamp = None
    last_timestamp = None
    first_timestamp_pacific = None
    last_timestamp_pacific = None
    for file in active_files:
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
                ts = obj.get("timestamp") or obj.get("timestamp_utc") or obj.get("decision_timestamp_utc")
                first_timestamp = first_timestamp or ts
                last_timestamp = ts or last_timestamp
                pacific_ts = parse_pacific_timestamp(ts)
                first_timestamp_pacific = first_timestamp_pacific or pacific_ts
                last_timestamp_pacific = pacific_ts or last_timestamp_pacific
                event_type = str(obj.get("event_type") or "missing")
                event_counts[event_type] += 1
                decision = obj.get("model_decision") if isinstance(obj.get("model_decision"), dict) else {}
                if decision.get("action"):
                    model_actions[str(decision.get("action"))] += 1
                risk = obj.get("risk_gate") if isinstance(obj.get("risk_gate"), dict) else {}
                if risk.get("reason"):
                    risk_reasons[str(risk.get("reason"))] += 1
                if obj.get("broker_order_endpoint_called"):
                    broker_order_endpoint_called_rows += 1
                if event_type == "paper_order_submitted":
                    paper_order_submitted_rows += 1
                if event_type in {"paper_entry_fill", "paper_exit_fill"}:
                    paper_fill_rows += 1
    return {
        "root": str(path),
        "files": [str(file) for file in active_files],
        "all_session_files": [str(file) for file in files],
        "file_count": len(active_files),
        "rows": rows,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "first_timestamp_pacific": first_timestamp_pacific.isoformat() if first_timestamp_pacific else None,
        "last_timestamp_pacific": last_timestamp_pacific.isoformat() if last_timestamp_pacific else None,
        "has_open_to_close_coverage": has_open_to_close_coverage(first_timestamp_pacific, last_timestamp_pacific),
        "event_counts": dict(sorted(event_counts.items())),
        "model_action_counts": dict(sorted(model_actions.items())),
        "risk_reason_counts": dict(risk_reasons.most_common()),
        "broker_order_endpoint_called_rows": broker_order_endpoint_called_rows,
        "paper_order_submitted_rows": paper_order_submitted_rows,
        "paper_fill_rows": paper_fill_rows,
        "parse_errors": parse_errors,
    }


def parse_pacific_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=PACIFIC)
    return parsed.astimezone(PACIFIC)


def has_open_to_close_coverage(first: datetime | None, last: datetime | None) -> bool:
    if first is None or last is None:
        return False
    return first.time() <= MIN_FIRST_EVENT_PACIFIC and last.time() >= MIN_LAST_EVENT_PACIFIC


def collect_launchd_status(*, label_prefix: str = "com.autoresearch.protocol101.fiveday") -> dict[str, Any]:
    suffixes = ("ibgateway.paper", "paper-preflight", "paper-session", "daily-monitor", "paper-shutdown", "evidence-postsession")
    labels = [*(f"{label_prefix}.{suffix}" for suffix in suffixes), *DAILY_LABELS]
    return {label: launchd_status(label) for label in labels}


def launchd_status(label: str) -> dict[str, Any]:
    target = f"gui/{_uid()}/{label}"
    result = subprocess.run(["launchctl", "print", target], text=True, capture_output=True, check=False)
    return {
        "label": label,
        "target": target,
        "loaded": result.returncode == 0,
        "state": first_launchctl_line(result.stdout, "state = "),
        "path": first_launchctl_line(result.stdout, "path = "),
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
    }


def _uid() -> int:
    import os

    return os.getuid()


def first_launchctl_line(stdout: str, needle: str) -> str | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(needle):
            return stripped.removeprefix(needle)
    return None


def collect_launchd_logs(*, log_prefix: str = "protocol101-fiveday") -> dict[str, Any]:
    names = {
        "gateway_stdout": f"{log_prefix}-ibgateway-paper.out.log",
        "gateway_stderr": f"{log_prefix}-ibgateway-paper.err.log",
        "preflight_stdout": f"{log_prefix}-paper-preflight.out.log",
        "preflight_stderr": f"{log_prefix}-paper-preflight.err.log",
        "session_stdout": f"{log_prefix}-paper-session.out.log",
        "session_stderr": f"{log_prefix}-paper-session.err.log",
        "monitor_stdout": f"{log_prefix}-daily-monitor.out.log",
        "monitor_stderr": f"{log_prefix}-daily-monitor.err.log",
        "shutdown_stdout": f"{log_prefix}-paper-shutdown.out.log",
        "shutdown_stderr": f"{log_prefix}-paper-shutdown.err.log",
        "postsession_stdout": f"{log_prefix}-evidence-postsession.out.log",
        "postsession_stderr": f"{log_prefix}-evidence-postsession.err.log",
    }
    return {key: file_status(LAUNCHD_LOG_DIR / filename) for key, filename in names.items()}


def file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=PACIFIC).isoformat(),
        "tail": tail_text(path),
    }


def tail_text(path: Path, *, lines: int = 20) -> list[str]:
    try:
        raw = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return raw[-lines:]


def summarize_runtime_flag(obj: dict[str, Any]) -> dict[str, Any]:
    return compact_keys(
        obj,
        [
            "paper_orders_enabled",
            "real_money_trading",
            "required_env",
            "scope",
            "source",
            "enabled_at",
            "run_id",
        ],
    )


def completion_checks(*, session: str, trade_log: dict[str, Any], runtime: dict[str, Any], target_dates: tuple[str, ...] = TARGET_DATES) -> dict[str, Any]:
    events = trade_log["event_counts"]
    return {
        "is_target_date": session in target_dates,
        "runtime_paper_orders_enabled": bool(runtime.get("paper_orders_enabled")),
        "runtime_real_money_false": runtime.get("real_money_trading") is False,
        "has_trade_log_rows": trade_log["rows"] > 0,
        "has_market_snapshots": int(events.get("market_snapshot", 0)) > 0,
        "has_candidate_sets": int(events.get("candidate_set", 0)) > 0,
        "has_model_decisions": int(events.get("model_decision", 0)) > 0,
        "has_risk_gates": int(events.get("risk_gate", 0)) > 0,
        "has_account_state": int(events.get("paper_account_state", 0)) > 0,
        "has_order_or_block_rows": any(int(events.get(name, 0)) > 0 for name in ("paper_order_blocked", "paper_order_submitted", "paper_order_status")),
        "has_open_to_close_coverage": bool(trade_log.get("has_open_to_close_coverage")),
        "parse_errors": int(trade_log["parse_errors"]),
    }


def decide_packet(*, session: str, trade_log: dict[str, Any], runtime: dict[str, Any], target_dates: tuple[str, ...] = TARGET_DATES) -> str:
    checks = completion_checks(session=session, trade_log=trade_log, runtime=runtime, target_dates=target_dates)
    if not checks["is_target_date"]:
        return "blocked_not_fiveday_target_date"
    if runtime.get("real_money_trading") is not False:
        return "fail_runtime_flag_real_money_not_false"
    if not bool(runtime.get("paper_orders_enabled")):
        return "partial_runtime_paper_orders_not_enabled"
    if trade_log["parse_errors"]:
        return "partial_trade_log_parse_errors"
    required = [
        "has_trade_log_rows",
        "has_market_snapshots",
        "has_candidate_sets",
        "has_model_decisions",
        "has_risk_gates",
        "has_account_state",
        "has_open_to_close_coverage",
    ]
    if all(checks[name] for name in required):
        return "complete_fiveday_paper_submit_session_ready_for_paired_replay_review"
    if checks["has_trade_log_rows"]:
        return "partial_fiveday_paper_submit_session_review_required"
    return "failed_no_fiveday_trade_log_rows"


def next_gate(decision: str) -> str:
    if decision.startswith("complete_"):
        return "Download the same session from Databento only after paid-data approval, then run paired replay diff."
    if decision.startswith("partial_"):
        return "Inspect launchd logs and paper JSONL to identify which required evidence fields were missing."
    return "Fix the launch/session blocker before counting this day toward the five-day evidence packet."


def compact_keys(obj: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: obj.get(key) for key in keys if key in obj}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"parse_error": str(path)}


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    (out_dir / "report.md").write_text(render_report(payload))


def render_report(payload: dict[str, Any]) -> str:
    trade_log = payload["trade_log"]
    checks = payload["completion_checks"]
    lines = [
        "# Protocol101 Five-Day Evidence Packet",
        "",
        f"Session: `{payload['session']}`",
        f"Run ID: `{payload['run_id']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "This packet reads local logs only. It does not train, tune, download paid data, score holdout data, or change the paper default.",
        "",
        "## Evidence Counts",
        "",
        f"- Trade log files: `{trade_log['file_count']}`",
        f"- Rows: `{trade_log['rows']}`",
        f"- First event PT: `{trade_log.get('first_timestamp_pacific')}`",
        f"- Last event PT: `{trade_log.get('last_timestamp_pacific')}`",
        f"- Open-to-close coverage: `{trade_log.get('has_open_to_close_coverage')}`",
        f"- Market snapshots: `{trade_log['event_counts'].get('market_snapshot', 0)}`",
        f"- Candidate sets: `{trade_log['event_counts'].get('candidate_set', 0)}`",
        f"- Model decisions: `{trade_log['event_counts'].get('model_decision', 0)}`",
        f"- Risk gates: `{trade_log['event_counts'].get('risk_gate', 0)}`",
        f"- Account-state rows: `{trade_log['event_counts'].get('paper_account_state', 0)}`",
        f"- Paper order submissions: `{trade_log['paper_order_submitted_rows']}`",
        f"- Fill rows: `{trade_log['paper_fill_rows']}`",
        f"- Broker endpoint rows: `{trade_log['broker_order_endpoint_called_rows']}`",
        "",
        "## Completion Checks",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in checks.items())
    lines.extend(
        [
            "",
            "## Runtime Flag",
            "",
            f"`{payload['runtime_flag']}`",
            "",
            "## Next Gate",
            "",
            payload["next_gate"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
