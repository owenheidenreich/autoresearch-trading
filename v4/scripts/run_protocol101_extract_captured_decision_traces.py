"""Extract replay-ready Protocol101 decision traces from captured paper logs.

This is the bridge between live paper JSONL logs and the paired replay diff.
It is intentionally offline-only: it reads captured logs, validates that they
contain full Protocol101DecisionTraceV1 fields, writes clean trace JSONL, and
optionally runs same-input or live-vs-historical diffs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from v4.live.protocol101_paired_replay_diff import (
    PairedReplayDiffConfig,
    build_paired_replay_diff,
)
from v4.scripts.run_protocol101_captured_trace_readiness import audit_candidate_set


DEFAULT_LIVE_ROOT = Path("v4/logs/paper_trading")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_extracted_captured_decision_traces")
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    parser.add_argument("--log-file", type=Path, action="append", default=[])
    parser.add_argument("--dates", nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--historical-traces", type=Path, default=None)
    parser.add_argument("--decision-ts-mode", choices=("floor-minute", "raw"), default="floor-minute")
    parser.add_argument("--threshold-adjacent-epsilon", type=float, default=0.02)
    parser.add_argument("--candidate-count-tolerance", type=int, default=0)
    return parser.parse_args()


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _iso_timestamp(value: Any, *, mode: str) -> str | None:
    dt = _parse_timestamp(value)
    if dt is None:
        return None
    if mode == "floor-minute":
        dt = dt.replace(second=0, microsecond=0)
    return dt.isoformat()


def _log_path(live_root: Path, session: str) -> Path:
    return live_root / session / f"daily_paper_autopilot_{session}.jsonl"


def discover_log_files(live_root: Path, dates: list[str], explicit: list[Path]) -> list[Path]:
    if explicit:
        return [path for path in explicit if path.exists()]
    if dates:
        return [path for session in dates if (path := _log_path(live_root, session)).exists()]
    if not live_root.exists():
        return []
    paths = sorted(live_root.glob("*/daily_paper_autopilot_*.jsonl"))
    if not paths:
        paths = sorted(live_root.glob("**/*.jsonl"))
    return paths


def _trace_key(row: dict[str, Any], *, mode: str) -> tuple[str, str]:
    session = str(row.get("session") or row.get("session_id") or "")
    decision_ts = _iso_timestamp(row.get("decision_timestamp_utc") or row.get("timestamp_utc") or row.get("timestamp"), mode=mode) or ""
    return session, decision_ts


def _hash_key(row: dict[str, Any], *, mode: str) -> tuple[str, str, str]:
    session, decision_ts = _trace_key(row, mode=mode)
    hash_value = str(row.get("candidate_universe_hash") or row.get("candidate_set_hash") or "")
    return session, decision_ts, hash_value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _index_related_rows(rows: list[dict[str, Any]], *, mode: str) -> dict[str, dict[tuple[str, str, str], dict[str, Any]]]:
    indexed: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {
        "model_decision": {},
        "risk_gate": {},
        "paper_account_state": {},
    }
    fallback: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        "model_decision": {},
        "risk_gate": {},
        "paper_account_state": {},
    }
    for row in rows:
        event_type = str(row.get("event_type") or "")
        if event_type not in indexed:
            continue
        key = _hash_key(row, mode=mode)
        minute_key = key[:2]
        indexed[event_type][key] = row
        fallback[event_type][minute_key] = row
    # Attach fallback under a reserved key for lookup helper.
    indexed["_fallback"] = fallback  # type: ignore[assignment]
    return indexed


def _find_related(indexed: dict[str, Any], event_type: str, candidate_row: dict[str, Any], *, mode: str) -> dict[str, Any]:
    key = _hash_key(candidate_row, mode=mode)
    found = indexed.get(event_type, {}).get(key)
    if isinstance(found, dict):
        return found
    minute_key = key[:2]
    fallback = indexed.get("_fallback", {}).get(event_type, {}).get(minute_key)
    return fallback if isinstance(fallback, dict) else {}


def _selected_contract_id(contract: dict[str, Any]) -> str:
    return str(
        contract.get("contract_id")
        or contract.get("candidate_id")
        or contract.get("candidate_uid")
        or contract.get("local_symbol")
        or contract.get("conid")
        or ""
    )


def enrich_candidate_trace(candidate_row: dict[str, Any], indexed: dict[str, Any], *, mode: str) -> dict[str, Any]:
    model_row = _find_related(indexed, "model_decision", candidate_row, mode=mode)
    risk_row = _find_related(indexed, "risk_gate", candidate_row, mode=mode)
    account_row = _find_related(indexed, "paper_account_state", candidate_row, mode=mode)
    model = model_row.get("model_decision") if isinstance(model_row.get("model_decision"), dict) else {}
    selected_contract = model.get("selected_contract") if isinstance(model.get("selected_contract"), dict) else {}
    if not selected_contract:
        selected_contract = model_row.get("selected_contract") if isinstance(model_row.get("selected_contract"), dict) else {}
    if not selected_contract:
        selected_contract = candidate_row.get("selected_contract") if isinstance(candidate_row.get("selected_contract"), dict) else {}
    risk_gate = risk_row.get("risk_gate") if isinstance(risk_row.get("risk_gate"), dict) else candidate_row.get("risk_gate")
    account_state = account_row.get("account") if isinstance(account_row.get("account"), dict) else account_row.get("paper_account_state")

    decision_ts = _iso_timestamp(
        candidate_row.get("decision_timestamp_utc") or candidate_row.get("timestamp_utc") or candidate_row.get("timestamp"),
        mode=mode,
    )
    source_quote_ts = candidate_row.get("source_quote_ts") or candidate_row.get("source_quote_time")
    source_context_ts = candidate_row.get("source_context_ts") or candidate_row.get("source_context_time")
    trace = dict(candidate_row)
    trace.update(
        {
            "source": "live_captured_ibkr",
            "decision_ts": decision_ts,
            "selected_action": model.get("selected_action") or model.get("action") or candidate_row.get("selected_action") or "wait",
            "selected_contract": selected_contract or {},
            "selected_contract_id": _selected_contract_id(selected_contract or {}),
            "selected_score": model.get("score") if model.get("score") is not None else model.get("selected_margin"),
            "decision_threshold": model.get("threshold") if model.get("threshold") is not None else candidate_row.get("model_threshold"),
            "source_quote_ts": source_quote_ts,
            "source_context_ts": source_context_ts,
            "risk_gate": risk_gate or {},
            "paper_account_state": account_state or {},
        }
    )
    if not trace.get("block_reasons"):
        gate = trace.get("candidate_gate_diagnostics") if isinstance(trace.get("candidate_gate_diagnostics"), dict) else {}
        reason = gate.get("filter_reason") or model.get("no_entry_reason") or model.get("reason")
        trace["block_reasons"] = [] if reason in {None, "", "candidates_available"} else [str(reason)]
    return trace


def extract_traces(paths: list[Path], *, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    by_session: dict[str, Counter[str]] = defaultdict(Counter)
    for path in paths:
        rows = _read_jsonl(path)
        indexed = _index_related_rows(rows, mode=mode)
        for line_number, row in enumerate(rows, start=1):
            if row.get("event_type") != "candidate_set":
                continue
            session = str(row.get("session") or row.get("session_id") or "")
            counters["candidate_set_rows"] += 1
            by_session[session]["candidate_set_rows"] += 1
            passed, errors = audit_candidate_set(row)
            if not passed:
                counters["trace_incomplete_candidate_set_rows"] += 1
                by_session[session]["trace_incomplete_candidate_set_rows"] += 1
                if len(failures) < 100:
                    failures.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "session": session,
                            "decision_ts": _iso_timestamp(row.get("decision_timestamp_utc") or row.get("timestamp_utc"), mode=mode) or "",
                            "errors": "|".join(errors),
                        }
                    )
                continue
            trace = enrich_candidate_trace(row, indexed, mode=mode)
            traces.append(trace)
            counters["full_trace_candidate_set_rows"] += 1
            by_session[session]["full_trace_candidate_set_rows"] += 1
    summary = {
        "candidate_set_rows": int(counters["candidate_set_rows"]),
        "full_trace_candidate_set_rows": int(counters["full_trace_candidate_set_rows"]),
        "trace_incomplete_candidate_set_rows": int(counters["trace_incomplete_candidate_set_rows"]),
        "by_session": {
            session: {key: int(value) for key, value in counter.items()}
            for session, counter in sorted(by_session.items())
        },
    }
    return traces, summary, failures


def load_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_diff_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol101 Extracted Captured Decision Traces",
        "",
        "This offline report extracts replay-ready live traces from captured paper JSONL logs.",
        "",
        f"- Status: `{payload['status']}`",
        f"- Extracted traces: `{payload['extracted_trace_rows']}`",
        f"- Candidate-set rows scanned: `{payload['candidate_set_rows']}`",
        f"- Incomplete candidate-set rows: `{payload['trace_incomplete_candidate_set_rows']}`",
        f"- Trace JSONL: `{payload.get('trace_jsonl') or ''}`",
        "",
        "## Same-Input Replay",
        "",
        f"- Status: `{payload.get('same_input_status') or 'not_run'}`",
        f"- Rows: `{payload.get('same_input_rows', 0)}`",
        "",
        "## Historical Paired Diff",
        "",
        f"- Status: `{payload.get('historical_diff_status') or 'not_run'}`",
        f"- Rows: `{payload.get('historical_diff_rows', 0)}`",
        "",
        "## By Session",
        "",
        "| session | candidate rows | full trace rows | incomplete rows |",
        "|---|---:|---:|---:|",
    ]
    for session, row in payload.get("by_session", {}).items():
        lines.append(
            f"| {session} | {row.get('candidate_set_rows', 0)} | {row.get('full_trace_candidate_set_rows', 0)} | {row.get('trace_incomplete_candidate_set_rows', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If status is `fail`, the captured logs are not sufficient for exact replay-level proof. Use the trace-readiness failure rows to determine whether the issue is old top-token logging or a current observability regression.",
            "",
            "## Artifacts",
            "",
            "- `summary.json`",
            "- `live_decision_traces.jsonl` when traces are available",
            "- `same_input_summary.json` when traces are available",
            "- `historical_paired_diff_summary.json` when `--historical-traces` is provided",
            "- `trace_extraction_failures.csv`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    paths = discover_log_files(args.live_root, list(args.dates or []), list(args.log_file or []))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces, extraction, failures = extract_traces(paths, mode=str(args.decision_ts_mode))
    trace_path = args.out_dir / "live_decision_traces.jsonl"
    if traces:
        write_jsonl(trace_path, traces)
    write_csv(args.out_dir / "trace_extraction_failures.csv", failures)

    same_input = None
    historical_diff = None
    if traces:
        config = PairedReplayDiffConfig(mode="same_input")
        same_input = build_paired_replay_diff(traces, traces, config=config)
        (args.out_dir / "same_input_summary.json").write_text(json.dumps(same_input, indent=2, sort_keys=True, default=str) + "\n")
        write_diff_csv(args.out_dir / "same_input_diff.csv", same_input.get("row_results", []))
    if traces and args.historical_traces and args.historical_traces.exists():
        historical_rows = load_trace_rows(args.historical_traces)
        config = PairedReplayDiffConfig(
            mode="cross_vendor",
            threshold_adjacent_epsilon=float(args.threshold_adjacent_epsilon),
            candidate_count_tolerance=int(args.candidate_count_tolerance),
        )
        historical_diff = build_paired_replay_diff(traces, historical_rows, config=config)
        (args.out_dir / "historical_paired_diff_summary.json").write_text(json.dumps(historical_diff, indent=2, sort_keys=True, default=str) + "\n")
        write_diff_csv(args.out_dir / "historical_paired_diff.csv", historical_diff.get("row_results", []))

    status = "pass" if traces and extraction["trace_incomplete_candidate_set_rows"] == 0 and (not same_input or same_input["status"] == "pass") else "fail"
    payload = {
        "protocol": "protocol101_extract_captured_decision_traces",
        "scope": "offline_existing_artifacts_only",
        "status": status,
        "inputs": {
            "live_root": str(args.live_root),
            "log_files": [str(path) for path in paths],
            "dates": list(args.dates or []),
            "historical_traces": None if args.historical_traces is None else str(args.historical_traces),
            "decision_ts_mode": str(args.decision_ts_mode),
        },
        "trace_jsonl": str(trace_path) if traces else None,
        "extracted_trace_rows": len(traces),
        **extraction,
        "same_input_status": None if same_input is None else same_input.get("status"),
        "same_input_rows": 0 if same_input is None else same_input.get("rows", 0),
        "historical_diff_status": None if historical_diff is None else historical_diff.get("status"),
        "historical_diff_rows": 0 if historical_diff is None else historical_diff.get("rows", 0),
        "next_step_if_fail": "collect_full_trace_logs_or_repair_current_live_trace_emission",
        "next_step_if_pass": "run_paired_live_historical_diff_on_matching_historical_day",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir, payload)
    print(json.dumps({"status": status, "out_dir": str(args.out_dir), "extracted_trace_rows": len(traces)}, indent=2, sort_keys=True))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
