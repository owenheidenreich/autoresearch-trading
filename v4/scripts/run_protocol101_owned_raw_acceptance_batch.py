"""Run resumable Protocol101 owned-raw live-v1 build batches.

This orchestrator does not change the dataset builder or verifier semantics. It
wraps the existing builder one session at a time so a single stalled day cannot
silently block October 2024 through June 2025 acceptance work.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "Protocol101OwnedRawAcceptanceBatchV1"
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_SPX_DIR = Path("data/vendor/thetadata/index/spx_1m")
DEFAULT_VIX_DIR = Path("data/vendor/thetadata/index/vix_1m")
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_protocol101_owned_raw_acceptance_live_v1")
DEFAULT_PROCESSED_DIR = Path("data/processed/spxw_0dte_neural_protocol101_owned_raw_acceptance_live_v1")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_owned_raw_acceptance_batch")
MARKET_HOLIDAYS = {
    "2024-11-28",
    "2024-12-25",
    "2025-01-01",
    "2025-01-09",
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
}


@dataclass
class SessionBatchRecord:
    session: str
    status: str
    command: list[str]
    timeout_seconds: int
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    returncode: int | None
    stdout_log: str
    stderr_log: str
    build_summary_path: str
    sessions_built: int = 0
    sessions_skipped: list[str] | None = None
    sessions_existing_skipped: list[str] | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--official-spx-dir", type=Path, default=DEFAULT_SPX_DIR)
    parser.add_argument("--official-vix-dir", type=Path, default=DEFAULT_VIX_DIR)
    parser.add_argument("--context-mode", choices=("official", "auto", "derived"), default="official")
    parser.add_argument("--feature-contract", default="protocol101-live-v1")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--acceptance-timeout-seconds", type=int, default=1800)
    parser.add_argument("--role", default="diagnostics_only")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compute-live-policy-labels", action="store_true")
    parser.add_argument("--run-acceptance", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sessions_between(start: str, end: str) -> list[str]:
    cursor = pd.Timestamp(start).date()
    final = pd.Timestamp(end).date()
    sessions: list[str] = []
    while cursor <= final:
        session = cursor.isoformat()
        if cursor.weekday() < 5 and session not in MARKET_HOLIDAYS:
            sessions.append(session)
        cursor += timedelta(days=1)
    return sessions


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def status_from_build_summary(summary: dict[str, Any], session: str) -> str:
    if int(summary.get("sessions_built") or 0) > 0:
        return "built"
    if session in set(summary.get("sessions_existing_skipped") or []):
        return "skipped_existing"
    if session in set(summary.get("sessions_skipped") or []):
        return "skipped_missing_raw"
    return "no_rows_built"


def builder_command(args: argparse.Namespace, session: str, summary_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "v4.scripts.build_databento_neural_dataset",
        "--start-date",
        session,
        "--end-date",
        session,
        "--raw-root",
        str(args.raw_root),
        "--normalized-dir",
        str(args.normalized_dir),
        "--processed-dir",
        str(args.processed_dir),
        "--context-mode",
        str(args.context_mode),
        "--official-spx-dir",
        str(args.official_spx_dir),
        "--official-vix-dir",
        str(args.official_vix_dir),
        "--feature-contract",
        str(args.feature_contract),
        "--summary-out",
        str(summary_path),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.compute_live_policy_labels:
        cmd.append("--compute-live-policy-labels")
    return cmd


def verifier_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "v4.scripts.run_protocol101_owned_raw_acceptance_verifier",
        "--start-date",
        str(args.start_date),
        "--end-date",
        str(args.end_date),
        "--out-dir",
        str(out_dir),
        "--raw-root",
        str(args.raw_root),
        "--official-spx-dir",
        str(args.official_spx_dir),
        "--official-vix-dir",
        str(args.official_vix_dir),
        "--processed-dir",
        str(args.processed_dir),
        "--normalized-dir",
        str(args.normalized_dir),
        "--role",
        str(args.role),
    ]


def run_command(
    cmd: list[str],
    *,
    timeout_seconds: int,
    stdout_log: Path,
    stderr_log: Path,
) -> tuple[int | None, str]:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with stdout_log.open("w") as stdout, stderr_log.open("w") as stderr:
            completed = subprocess.run(
                cmd,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
                text=True,
            )
        return int(completed.returncode), ""
    except subprocess.TimeoutExpired as exc:
        with stderr_log.open("a") as stderr:
            stderr.write(f"\nTIMEOUT after {timeout_seconds}s: {exc}\n")
        return None, f"timeout_after_{timeout_seconds}s"


def run_session(args: argparse.Namespace, session: str) -> SessionBatchRecord:
    session_summary = args.out_dir / "build_summaries" / f"{session}.json"
    stdout_log = args.out_dir / "logs" / f"{session}.stdout.log"
    stderr_log = args.out_dir / "logs" / f"{session}.stderr.log"
    cmd = builder_command(args, session, session_summary)
    started = utc_now()
    start_time = time.monotonic()
    returncode, error = run_command(
        cmd,
        timeout_seconds=int(args.timeout_seconds),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )
    duration = time.monotonic() - start_time
    finished = utc_now()
    summary = load_json(session_summary)
    if error:
        status = "timeout"
    elif returncode != 0:
        status = "failed"
    else:
        status = status_from_build_summary(summary, session)
    return SessionBatchRecord(
        session=session,
        status=status,
        command=cmd,
        timeout_seconds=int(args.timeout_seconds),
        started_at_utc=started,
        finished_at_utc=finished,
        duration_seconds=round(duration, 3),
        returncode=returncode,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        build_summary_path=str(session_summary),
        sessions_built=int(summary.get("sessions_built") or 0),
        sessions_skipped=list(summary.get("sessions_skipped") or []),
        sessions_existing_skipped=list(summary.get("sessions_existing_skipped") or []),
        error=error,
    )


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_payload(args: argparse.Namespace, records: list[SessionBatchRecord], *, acceptance: dict[str, Any] | None = None) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    terminal_failures = [record.session for record in records if record.status in {"timeout", "failed"}]
    missing_raw = [record.session for record in records if record.status == "skipped_missing_raw"]
    acceptance_payload = acceptance or {"status": "not_run"}
    acceptance_failed = acceptance_payload.get("status") not in {"not_run", "not_run_due_to_build_failures", "pass"}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fail" if terminal_failures or acceptance_failed else "pass",
        "started_range": {"start_date": args.start_date, "end_date": args.end_date},
        "raw_root": str(args.raw_root),
        "normalized_dir": str(args.normalized_dir),
        "processed_dir": str(args.processed_dir),
        "context_mode": str(args.context_mode),
        "feature_contract": str(args.feature_contract),
        "compute_live_policy_labels": bool(args.compute_live_policy_labels),
        "skip_existing": bool(args.skip_existing),
        "timeout_seconds": int(args.timeout_seconds),
        "session_count": len(records),
        "counts_by_status": counts,
        "terminal_failures": terminal_failures,
        "missing_raw_sessions": missing_raw,
        "records": [asdict(record) for record in records],
        "acceptance": acceptance_payload,
    }


def terminal_build_failures(records: list[SessionBatchRecord]) -> list[str]:
    return [record.session for record in records if record.status in {"timeout", "failed"}]


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir / "acceptance"
    stdout_log = args.out_dir / "logs" / "acceptance.stdout.log"
    stderr_log = args.out_dir / "logs" / "acceptance.stderr.log"
    cmd = verifier_command(args, out_dir)
    started = utc_now()
    start_time = time.monotonic()
    returncode, error = run_command(
        cmd,
        timeout_seconds=int(args.acceptance_timeout_seconds),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )
    duration = time.monotonic() - start_time
    summary = load_json(out_dir / "summary.json")
    status = "timeout" if error else "failed" if returncode != 0 else str(summary.get("status") or "unknown")
    return {
        "status": status,
        "command": cmd,
        "returncode": returncode,
        "error": error,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "duration_seconds": round(duration, 3),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "summary_path": str(out_dir / "summary.json"),
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sessions_between(args.start_date, args.end_date)
    records: list[SessionBatchRecord] = []
    summary_path = args.out_dir / "summary.json"
    for session in sessions:
        record = run_session(args, session)
        records.append(record)
        write_payload(summary_path, build_payload(args, records))
        print(json.dumps({"session": session, "status": record.status}, sort_keys=True), flush=True)
        if record.status == "timeout":
            break
    build_failures = terminal_build_failures(records)
    if bool(args.run_acceptance) and build_failures:
        acceptance = {
            "status": "not_run_due_to_build_failures",
            "reason": "acceptance_requires_complete_successful_build_range",
            "terminal_failures": build_failures,
        }
    elif bool(args.run_acceptance):
        acceptance = run_acceptance(args)
    else:
        acceptance = {"status": "not_run"}
    payload = build_payload(args, records, acceptance=acceptance)
    write_payload(summary_path, payload)
    print(json.dumps({"status": payload["status"], "counts_by_status": payload["counts_by_status"]}, sort_keys=True), flush=True)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
