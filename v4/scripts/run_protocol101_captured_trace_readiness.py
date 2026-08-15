"""Audit whether captured Protocol101 live logs support full parity replay.

Older paper logs may contain only top-token diagnostics. Newer logs should
include Protocol101DecisionTraceV1 fields with a full candidate universe,
token features, feature hashes, and model-score hashes. This script makes that
distinction explicit before we attempt same-input replay or paired diffs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATES = ("2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09")
DEFAULT_LIVE_ROOT = Path(
    "/Users/gduby/.autoresearch-trading/archive/"
    "ibkr_live_trading_sessions_2026-06-04_05_08_09_10_20260611T155134Z/"
    "live_runtime_paper_trading"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_captured_trace_readiness"
)
TRACE_SCHEMA = "Protocol101DecisionTraceV1"
LIVE_CONTRACT = "protocol101-live-v1"
UTC = timezone.utc


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


def _minute_key(value: Any) -> str | None:
    dt = _parse_timestamp(value)
    if dt is None:
        return None
    return dt.replace(second=0, microsecond=0).isoformat()


def _live_log_path(root: Path, session: str) -> Path:
    return root / session / f"daily_paper_autopilot_{session}.jsonl"


def _reason_rows(reasons: Counter[str], session: str) -> list[dict[str, Any]]:
    return [
        {"session": session, "reason": reason, "count": count}
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
    ]


def audit_candidate_set(row: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    diagnostics = row.get("candidate_gate_diagnostics")
    filter_reason = ""
    if isinstance(diagnostics, dict):
        filter_reason = str(diagnostics.get("filter_reason") or diagnostics.get("canonical_filter_reason") or "")
    block_reasons = row.get("block_reasons") or row.get("blocked_reasons") or []
    if isinstance(block_reasons, str):
        block_reason_set = {block_reasons}
    elif isinstance(block_reasons, (list, tuple, set)):
        block_reason_set = {str(reason) for reason in block_reasons}
    else:
        block_reason_set = set()
    pre_score_block_reasons = {
        "outside_time_bucket",
        "insufficient_live_index_context",
        "missing_opening_live_index_context",
        "insufficient_index_context",
    }
    pre_score_block = filter_reason in pre_score_block_reasons or bool(block_reason_set & pre_score_block_reasons)
    if row.get("decision_trace_schema_version") != TRACE_SCHEMA:
        errors.append("missing_protocol101_decision_trace_schema")
    if row.get("feature_contract_version") != LIVE_CONTRACT:
        errors.append("missing_live_feature_contract_version")
    candidate_universe = row.get("candidate_universe")
    if not isinstance(candidate_universe, list):
        errors.append("missing_candidate_universe")
        candidate_universe = []
    features = row.get("features")
    if not isinstance(features, dict):
        errors.append("missing_features_payload")
        token_features = []
    else:
        token_features = features.get("token_features")
        if not pre_score_block and not isinstance(token_features, list):
            errors.append("missing_features_token_features")
            token_features = []
        elif not isinstance(token_features, list):
            token_features = []
    model_scores = row.get("model_scores")
    if not isinstance(model_scores, dict):
        errors.append("missing_model_scores")
    elif not isinstance(model_scores.get("surface_scores"), list):
        errors.append("missing_model_scores_surface_scores")
    for field in ("candidate_universe_hash", "feature_hash", "score_hash"):
        if not row.get(field):
            errors.append(f"missing_{field}")

    # Candidate rows that represent a scored universe should carry token feature
    # hashes and full token features. Blocked pre-context rows may be empty, but
    # once a universe exists it must be reconstructable.
    if candidate_universe and not pre_score_block:
        missing_token_hash = 0
        missing_token_vector = 0
        token_by_contract = {}
        for token in token_features:
            if isinstance(token, dict):
                token_by_contract[str(token.get("contract_id") or "")] = token
        for candidate in candidate_universe:
            if not isinstance(candidate, dict):
                continue
            contract_id = str(candidate.get("contract_id") or "")
            if not candidate.get("token_feature_hash"):
                missing_token_hash += 1
            token_payload = token_by_contract.get(contract_id)
            if not token_payload or not isinstance(token_payload.get("token_features"), list):
                missing_token_vector += 1
        if missing_token_hash:
            errors.append("candidate_universe_missing_token_feature_hash")
        if missing_token_vector:
            errors.append("candidate_universe_missing_full_token_features")
    return not errors, errors


def audit_logs(root: Path, dates: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_session: dict[str, Any] = {}
    failure_rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []
    for session in sorted(dates):
        path = _live_log_path(root, session)
        counters: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        first_minute = None
        last_minute = None
        if not path.exists():
            by_session[session] = {
                "path": str(path),
                "status": "fail",
                "reason": "missing_live_log",
                "candidate_set_rows": 0,
            }
            failure_rows.append({"session": session, "path": str(path), "reason": "missing_live_log"})
            continue
        with path.open() as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counters["json_decode_errors"] += 1
                    continue
                if row.get("event_type") != "candidate_set":
                    continue
                counters["candidate_set_rows"] += 1
                minute = _minute_key(row.get("decision_timestamp_utc") or row.get("timestamp_utc"))
                first_minute = minute if first_minute is None else min(first_minute, minute or first_minute)
                last_minute = minute if last_minute is None else max(last_minute, minute or last_minute)
                passed, errors = audit_candidate_set(row)
                if passed:
                    counters["full_trace_candidate_set_rows"] += 1
                else:
                    counters["trace_incomplete_candidate_set_rows"] += 1
                    for error in errors:
                        reasons[error] += 1
                    if counters["trace_incomplete_candidate_set_rows"] <= 25:
                        failure_rows.append(
                            {
                                "session": session,
                                "line": line_number,
                                "decision_minute_utc": minute or "",
                                "errors": "|".join(errors),
                                "filter_reason": ((row.get("candidate_gate_diagnostics") or {}).get("filter_reason") if isinstance(row.get("candidate_gate_diagnostics"), dict) else ""),
                            }
                        )
        full = int(counters["full_trace_candidate_set_rows"])
        total = int(counters["candidate_set_rows"])
        status = "pass" if total > 0 and full == total else "fail"
        by_session[session] = {
            "path": str(path),
            "status": status,
            "candidate_set_rows": total,
            "full_trace_candidate_set_rows": full,
            "trace_incomplete_candidate_set_rows": int(counters["trace_incomplete_candidate_set_rows"]),
            "json_decode_errors": int(counters["json_decode_errors"]),
            "first_candidate_minute_utc": first_minute,
            "last_candidate_minute_utc": last_minute,
            "top_failure_reasons": dict(reasons.most_common(10)),
        }
        reason_rows.extend(_reason_rows(reasons, session))
    summary = {
        "protocol": "protocol101_captured_trace_readiness",
        "scope": "offline_existing_artifacts_only",
        "live_root": str(root),
        "dates": sorted(dates),
        "status": "pass" if by_session and all(row.get("status") == "pass" for row in by_session.values()) else "fail",
        "by_session": by_session,
        "next_step_if_fail": "collect_or_reconstruct_full_Protocol101DecisionTraceV1_rows_before_exact_same_input_replay",
        "next_step_if_pass": "run_protocol101_paired_live_historical_diff_or_same_input_replay",
    }
    return summary, failure_rows, reason_rows


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


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Protocol101 Captured Trace Readiness",
        "",
        "This report checks whether captured live logs contain enough `Protocol101DecisionTraceV1` detail for exact same-input replay and paired live/historical diffs.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Live root: `{summary['live_root']}`",
        "",
        "## By Session",
        "",
        "| session | status | candidate rows | full trace rows | incomplete rows | first minute UTC | last minute UTC | top failure reasons |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for session, row in summary["by_session"].items():
        lines.append(
            f"| {session} | {row.get('status')} | {row.get('candidate_set_rows', 0)} | "
            f"{row.get('full_trace_candidate_set_rows', 0)} | {row.get('trace_incomplete_candidate_set_rows', 0)} | "
            f"{row.get('first_candidate_minute_utc') or ''} | {row.get('last_candidate_minute_utc') or ''} | "
            f"`{row.get('top_failure_reasons') or {}}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A fail here does not mean the session was useless. It means the captured log cannot support exact replay-level proof. Minute-level and top-token forensic comparisons can still be useful, but full synchronization requires full candidate universe and feature vectors.",
            "",
            "## Artifacts",
            "",
            "- `summary.json`",
            "- `trace_failures.csv`",
            "- `failure_reason_counts.csv`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    args = parser.parse_args()
    summary, failures, reasons = audit_logs(args.live_root, set(args.dates))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(args.out_dir / "trace_failures.csv", failures)
    write_csv(args.out_dir / "failure_reason_counts.csv", reasons)
    write_report(args.out_dir, summary)
    print(json.dumps({"out_dir": str(args.out_dir), "status": summary["status"]}, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
