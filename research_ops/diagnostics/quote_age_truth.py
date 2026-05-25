#!/usr/bin/env python3
"""Audit whether logged Protocol101 quote ages are reconstructable.

This diagnostic is intentionally stdlib-only and read-only with respect to v4.
It reads JSONL logs supplied by path and writes artifacts to a caller-provided
iteration output directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUOTE_AGE_CANDIDATES = (
    ("market_snapshot", "option_nbbo", "quote_age_ms"),
    ("market_snapshot", "option", "quote_age_ms"),
    ("option_nbbo", "quote_age_ms"),
    ("quote", "quote_age_ms"),
    ("quote_age_ms",),
)

QUOTE_TIMESTAMP_CANDIDATES = (
    ("market_snapshot", "option_nbbo", "quote_timestamp"),
    ("market_snapshot", "option_nbbo", "quote_timestamp_ms"),
    ("market_snapshot", "option_nbbo", "timestamp"),
    ("market_snapshot", "option_nbbo", "time"),
    ("market_snapshot", "option_nbbo", "timeStamp"),
    ("market_snapshot", "option_nbbo", "rtTime"),
    ("market_snapshot", "option_nbbo", "lastTimestamp"),
    ("market_snapshot", "option_nbbo", "last_quote_timestamp"),
    ("market_snapshot", "option_nbbo", "last_quote_time"),
    ("market_snapshot", "option_nbbo", "bid_timestamp"),
    ("market_snapshot", "option_nbbo", "ask_timestamp"),
    ("market_snapshot", "option", "timestamp"),
    ("option_nbbo", "timestamp"),
    ("quote", "timestamp"),
)

DECISION_TIMESTAMP_CANDIDATES = (
    ("market_snapshot", "option_nbbo", "decision_timestamp"),
    ("market_snapshot", "option_nbbo", "decision_timestamp_ms"),
    ("timing", "decision_emitted_at"),
    ("timing", "decision_timestamp"),
    ("timing", "intended_entry_time"),
    ("decision_timestamp",),
)

RECEIVED_TIMESTAMP_CANDIDATES = (
    ("market_snapshot", "option_nbbo", "received_timestamp"),
    ("market_snapshot", "option_nbbo", "received_timestamp_ms"),
    ("market_snapshot", "option_nbbo", "observed_at"),
    ("market_snapshot", "option_nbbo", "observed_timestamp"),
    ("market_snapshot", "option_nbbo", "snapshot_timestamp"),
    ("market_snapshot", "option", "received_timestamp"),
    ("option_nbbo", "received_timestamp"),
    ("quote", "received_timestamp"),
)

EVENT_TIMESTAMP_CANDIDATES = (("timestamp",),)

CSV_COLUMNS = [
    "source_file",
    "line_number",
    "event_type",
    "timestamp",
    "run_id",
    "mode",
    "selected_action",
    "broker_order_endpoint_called",
    "quote_age_path",
    "quote_timestamp_path",
    "reference_timestamp_path",
    "received_timestamp_path",
    "persisted_quote_age_ms",
    "quote_timestamp",
    "reference_timestamp",
    "received_timestamp",
    "recomputed_quote_age_ms",
    "age_delta_ms",
    "has_quote_timestamp",
    "has_reference_timestamp",
    "has_received_timestamp",
    "classification",
    "trust_status",
    "reason",
]

NUMERIC_TEXT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
QUOTE_EVIDENCE_KEYS = {
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "quote_age_ms",
    "quote_timestamp",
    "quote_timestamp_ms",
    "timestamp",
    "time",
    "timeStamp",
    "rtTime",
    "lastTimestamp",
    "last_quote_timestamp",
    "last_quote_time",
    "bid_timestamp",
    "ask_timestamp",
    "received_timestamp",
    "received_timestamp_ms",
}


@dataclass(frozen=True)
class TimestampCandidate:
    path: str
    raw: Any
    parsed: dt.datetime | None
    malformed: bool = False


@dataclass(frozen=True)
class NumberCandidate:
    path: str
    raw: Any
    parsed: float | None
    malformed: bool = False


def path_label(path: tuple[str, ...]) -> str:
    return ".".join(path)


def get_path(row: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = row
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def parse_number(value: Any) -> float | None:
    if isinstance(value, bool) or is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_number(row: dict[str, Any], candidates: tuple[tuple[str, ...], ...]) -> NumberCandidate:
    for path in candidates:
        exists, raw = get_path(row, path)
        if not exists or is_missing(raw):
            continue
        parsed = parse_number(raw)
        return NumberCandidate(path_label(path), raw, parsed, parsed is None)
    return NumberCandidate("", None, None, False)


def parse_epoch(number: float) -> dt.datetime | None:
    magnitude = abs(number)
    if magnitude > 1e17:
        seconds = number / 1_000_000_000.0
    elif magnitude > 1e14:
        seconds = number / 1_000_000.0
    elif magnitude > 1e11:
        seconds = number / 1_000.0
    else:
        seconds = number
    try:
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def parse_timestamp(value: Any) -> dt.datetime | None:
    if isinstance(value, bool) or is_missing(value):
        return None
    if isinstance(value, (int, float)):
        return parse_epoch(float(value))
    text = str(value).strip()
    if NUMERIC_TEXT.fullmatch(text):
        return parse_epoch(float(text))
    clean = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = dt.datetime.fromisoformat(clean)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def first_timestamp(row: dict[str, Any], candidates: tuple[tuple[str, ...], ...]) -> TimestampCandidate:
    first_present: tuple[tuple[str, ...], Any] | None = None
    for path in candidates:
        exists, raw = get_path(row, path)
        if not exists or is_missing(raw):
            continue
        if first_present is None:
            first_present = (path, raw)
        parsed = parse_timestamp(raw)
        if parsed is not None:
            return TimestampCandidate(path_label(path), raw, parsed, False)
    if first_present is None:
        return TimestampCandidate("", None, None, False)
    path, raw = first_present
    return TimestampCandidate(path_label(path), raw, None, True)


def iso_or_empty(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat()


def rounded(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.3f}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def is_broker_row(row: dict[str, Any]) -> bool:
    return bool(row.get("broker_order_endpoint_called") or row.get("broker_endpoint_called"))


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def has_option_quote_evidence(row: dict[str, Any], age: NumberCandidate, quote_ts: TimestampCandidate) -> bool:
    market = as_dict(row.get("market_snapshot"))
    option = market.get("option_nbbo")
    has_quote_fields = isinstance(option, dict) and bool(QUOTE_EVIDENCE_KEYS.intersection(option.keys()))
    return has_quote_fields or bool(age.path) or bool(quote_ts.path)


def classify_event(
    row: dict[str, Any],
    *,
    source_file: str,
    line_number: int,
    max_age_ms: float,
    tolerance_ms: float,
) -> dict[str, str]:
    persisted = first_number(row, QUOTE_AGE_CANDIDATES)
    quote_ts = first_timestamp(row, QUOTE_TIMESTAMP_CANDIDATES)
    decision_ts = first_timestamp(row, DECISION_TIMESTAMP_CANDIDATES)
    received_ts = first_timestamp(row, RECEIVED_TIMESTAMP_CANDIDATES)
    event_ts = first_timestamp(row, EVENT_TIMESTAMP_CANDIDATES)

    reference = decision_ts
    if reference.parsed is None:
        reference = received_ts
    if reference.parsed is None:
        reference = event_ts

    recomputed: float | None = None
    age_delta: float | None = None
    classification = "unreconstructable"
    trust_status = "unknown"
    reason = "unclassified"

    if not persisted.path:
        classification = "missing"
        reason = "missing_persisted_quote_age"
    elif persisted.malformed or persisted.parsed is None:
        classification = "unreconstructable"
        reason = "malformed_persisted_quote_age"
    elif quote_ts.parsed is None:
        if quote_ts.malformed:
            classification = "unreconstructable"
            reason = "malformed_quote_timestamp"
        elif abs(persisted.parsed) <= 1e-9:
            classification = "placeholder"
            reason = "missing_raw_quote_timestamp_persisted_zero"
        else:
            classification = "unreconstructable"
            reason = "missing_raw_quote_timestamp"
    elif reference.parsed is None:
        classification = "unreconstructable"
        reason = "missing_or_malformed_reference_timestamp"
    else:
        raw_age = (reference.parsed - quote_ts.parsed).total_seconds() * 1000.0
        if raw_age < -1000.0:
            classification = "unreconstructable"
            reason = "quote_timestamp_after_reference_timestamp"
        else:
            recomputed = max(0.0, raw_age)
            age_delta = persisted.parsed - recomputed
            if persisted.parsed > max_age_ms or recomputed > max_age_ms:
                classification = "stale"
                trust_status = "fail"
                reason = "age_exceeds_max_age_ms"
            elif abs(age_delta) <= tolerance_ms:
                classification = "trustworthy"
                trust_status = "pass"
                reason = "recomputed_age_matches_persisted_age"
            elif abs(persisted.parsed) <= 1e-9 and recomputed > tolerance_ms:
                classification = "placeholder"
                reason = "persisted_zero_but_recomputed_nonzero"
            else:
                classification = "unreconstructable"
                reason = "persisted_age_mismatch"

    if classification in {"missing", "placeholder", "unreconstructable"}:
        trust_status = "unknown"

    return {
        "source_file": source_file,
        "line_number": str(line_number),
        "event_type": str(row.get("event_type") or ""),
        "timestamp": str(row.get("timestamp") or ""),
        "run_id": str(row.get("run_id") or ""),
        "mode": str(row.get("mode") or ""),
        "selected_action": str(row.get("selected_action") or as_dict(row.get("model_decision")).get("action") or ""),
        "broker_order_endpoint_called": bool_text(is_broker_row(row)),
        "quote_age_path": persisted.path,
        "quote_timestamp_path": quote_ts.path,
        "reference_timestamp_path": reference.path,
        "received_timestamp_path": received_ts.path,
        "persisted_quote_age_ms": rounded(persisted.parsed),
        "quote_timestamp": iso_or_empty(quote_ts.parsed),
        "reference_timestamp": iso_or_empty(reference.parsed),
        "received_timestamp": iso_or_empty(received_ts.parsed),
        "recomputed_quote_age_ms": rounded(recomputed),
        "age_delta_ms": rounded(age_delta),
        "has_quote_timestamp": bool_text(quote_ts.parsed is not None),
        "has_reference_timestamp": bool_text(reference.parsed is not None),
        "has_received_timestamp": bool_text(received_ts.parsed is not None),
        "classification": classification,
        "trust_status": trust_status,
        "reason": reason,
        "_has_option_quote_evidence": bool_text(has_option_quote_evidence(row, persisted, quote_ts)),
    }


def iter_log_files(log_roots: list[Path], log_files: list[Path]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    missing: list[str] = []
    seen: set[Path] = set()

    def add_file(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        files.append(resolved)

    for raw in log_files:
        path = raw.expanduser()
        if path.exists() and path.is_file():
            add_file(path)
        else:
            missing.append(str(path))

    for raw in log_roots:
        root = raw.expanduser()
        if not root.exists():
            missing.append(str(root))
            continue
        if root.is_file():
            add_file(root)
            continue
        for path in sorted(root.rglob("*.jsonl")):
            if path.is_file():
                add_file(path)

    return files, missing


def read_jsonl_rows(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    rows: list[tuple[int, dict[str, Any]]] = []
    json_errors = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue
            if isinstance(value, dict):
                rows.append((line_number, value))
            else:
                json_errors += 1
    return rows, json_errors


def analyze_logs(
    *,
    log_roots: list[Path],
    log_files: list[Path],
    out_dir: Path,
    max_age_ms: float,
    tolerance_ms: float,
) -> dict[str, Any]:
    files, missing_inputs = iter_log_files(log_roots, log_files)
    output_rows: list[dict[str, str]] = []
    json_error_count = 0

    for path in files:
        parsed_rows, file_errors = read_jsonl_rows(path)
        json_error_count += file_errors
        for line_number, row in parsed_rows:
            output_rows.append(
                classify_event(
                    row,
                    source_file=str(path),
                    line_number=line_number,
                    max_age_ms=max_age_ms,
                    tolerance_ms=tolerance_ms,
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "quote_age_rows.csv"
    json_path = out_dir / "quote_age_summary.json"
    md_path = out_dir / "quote_age_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in output_rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})

    summary = build_summary(
        rows=output_rows,
        files=files,
        missing_inputs=missing_inputs,
        json_error_count=json_error_count,
        max_age_ms=max_age_ms,
        tolerance_ms=tolerance_ms,
        csv_path=csv_path,
        json_path=json_path,
        md_path=md_path,
    )
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_summary(summary), encoding="utf-8")
    return summary


def build_summary(
    *,
    rows: list[dict[str, str]],
    files: list[Path],
    missing_inputs: list[str],
    json_error_count: int,
    max_age_ms: float,
    tolerance_ms: float,
    csv_path: Path,
    json_path: Path,
    md_path: Path,
) -> dict[str, Any]:
    classification_counts = Counter(row["classification"] for row in rows)
    trust_counts = Counter(row["trust_status"] for row in rows)
    quote_evidence_rows = [row for row in rows if row.get("_has_option_quote_evidence") == "true"]
    persisted_rows = [row for row in rows if row.get("persisted_quote_age_ms")]
    broker_rows = [row for row in rows if row.get("broker_order_endpoint_called") == "true"]
    broker_problem_rows = [row for row in broker_rows if row.get("classification") != "trustworthy"]
    paper_submit_rows = [
        row
        for row in quote_evidence_rows
        if str(row.get("mode") or "").lower() == "paper-submit"
        or "persistent-paper" in str(row.get("run_id") or "").lower()
    ]
    paper_submit_problem_rows = [row for row in paper_submit_rows if row.get("classification") != "trustworthy"]

    placeholder_count = classification_counts.get("placeholder", 0)
    placeholder_ratio = placeholder_count / len(persisted_rows) if persisted_rows else 0.0
    trustworthy_persisted = sum(1 for row in persisted_rows if row.get("classification") == "trustworthy")
    trustworthy_ratio = trustworthy_persisted / len(persisted_rows) if persisted_rows else 0.0

    verdict, verdict_reason = aggregate_verdict(
        rows=rows,
        persisted_count=len(persisted_rows),
        broker_problem_count=len(broker_problem_rows),
        broker_count=len(broker_rows),
        paper_submit_count=len(paper_submit_rows),
        paper_submit_problem_count=len(paper_submit_problem_rows),
        placeholder_ratio=placeholder_ratio,
        trustworthy_ratio=trustworthy_ratio,
    )

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "thresholds": {
            "max_age_ms": max_age_ms,
            "tolerance_ms": tolerance_ms,
        },
        "input": {
            "files_read": [str(path) for path in files],
            "missing_inputs": missing_inputs,
            "json_error_count": json_error_count,
        },
        "artifacts": {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "markdown_path": str(md_path),
        },
        "counts": {
            "total_rows": len(rows),
            "files_read_count": len(files),
            "quote_evidence_rows": len(quote_evidence_rows),
            "persisted_quote_age_rows": len(persisted_rows),
            "broker_endpoint_rows": len(broker_rows),
            "broker_problem_rows": len(broker_problem_rows),
            "paper_submit_quote_rows": len(paper_submit_rows),
            "paper_submit_problem_rows": len(paper_submit_problem_rows),
            "placeholder_ratio_among_persisted": round(placeholder_ratio, 6),
            "trustworthy_ratio_among_persisted": round(trustworthy_ratio, 6),
            "classification_counts": dict(sorted(classification_counts.items())),
            "trust_status_counts": dict(sorted(trust_counts.items())),
        },
    }


def aggregate_verdict(
    *,
    rows: list[dict[str, str]],
    persisted_count: int,
    broker_problem_count: int,
    broker_count: int,
    paper_submit_count: int,
    paper_submit_problem_count: int,
    placeholder_ratio: float,
    trustworthy_ratio: float,
) -> tuple[str, str]:
    if not rows:
        return "unknown", "no JSONL rows were available for inspection"
    if broker_problem_count:
        return "fail", "at least one broker endpoint row lacks trustworthy quote-age evidence"
    if paper_submit_problem_count:
        return "fail", "paper-submit quote rows include non-trustworthy quote-age evidence"
    if persisted_count and placeholder_ratio > 0.05:
        return "fail", "more than 5% of persisted quote-age rows are placeholders"
    if broker_count and trustworthy_ratio >= 0.95 and placeholder_ratio == 0.0:
        return "pass", "broker endpoint rows are trustworthy and persisted quote-age rows meet threshold"
    return "unknown", "available logs do not contain enough broker/paper-submit trustworthy evidence"


def render_markdown_summary(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    classification_counts = counts["classification_counts"]
    trust_counts = counts["trust_status_counts"]

    def bullet_counts(items: dict[str, int]) -> str:
        if not items:
            return "- None."
        return "\n".join(f"- `{key}`: {value}" for key, value in items.items())

    files = summary["input"]["files_read"]
    missing = summary["input"]["missing_inputs"]
    files_block = "\n".join(f"- `{path}`" for path in files) if files else "- None."
    missing_block = "\n".join(f"- `{path}`" for path in missing) if missing else "- None."

    return (
        "# Quote Age Truth Diagnostic Summary\n\n"
        f"Generated at: `{summary['generated_at']}`\n\n"
        "## Verdict\n\n"
        f"- Verdict: `{summary['verdict']}`\n"
        f"- Reason: {summary['verdict_reason']}\n"
        f"- Max age: `{summary['thresholds']['max_age_ms']}` ms\n"
        f"- Tolerance: `{summary['thresholds']['tolerance_ms']}` ms\n\n"
        "## Counts\n\n"
        f"- Total rows: `{counts['total_rows']}`\n"
        f"- Files read: `{counts['files_read_count']}`\n"
        f"- Quote evidence rows: `{counts['quote_evidence_rows']}`\n"
        f"- Persisted quote-age rows: `{counts['persisted_quote_age_rows']}`\n"
        f"- Broker endpoint rows: `{counts['broker_endpoint_rows']}`\n"
        f"- Broker problem rows: `{counts['broker_problem_rows']}`\n"
        f"- Paper-submit quote rows: `{counts['paper_submit_quote_rows']}`\n"
        f"- Paper-submit problem rows: `{counts['paper_submit_problem_rows']}`\n"
        f"- Placeholder ratio among persisted: `{counts['placeholder_ratio_among_persisted']}`\n"
        f"- Trustworthy ratio among persisted: `{counts['trustworthy_ratio_among_persisted']}`\n\n"
        "## Classification Counts\n\n"
        f"{bullet_counts(classification_counts)}\n\n"
        "## Trust Status Counts\n\n"
        f"{bullet_counts(trust_counts)}\n\n"
        "## Inputs Read\n\n"
        f"{files_block}\n\n"
        "## Missing Inputs\n\n"
        f"{missing_block}\n\n"
        "## Artifacts\n\n"
        f"- CSV: `{summary['artifacts']['csv_path']}`\n"
        f"- JSON: `{summary['artifacts']['json_path']}`\n"
        f"- Markdown: `{summary['artifacts']['markdown_path']}`\n\n"
        "## Interpretation\n\n"
        "- `trustworthy` means logged age matched recomputed age from raw timestamps. It does not prove fillability.\n"
        "- `placeholder` means a persisted age, especially zero, lacked raw timestamp support or contradicted recomputation.\n"
        "- Missing raw quote timestamp is never treated as pass.\n"
        "- `unknown` remains blocking evidence for paper-submit trust until live paper-submit rows carry reconstructable timestamps.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", action="append", default=[], help="directory or file containing JSONL logs")
    parser.add_argument("--log-file", action="append", default=[], help="specific JSONL log file")
    parser.add_argument("--out-dir", required=True, help="iteration artifact output directory")
    parser.add_argument("--max-age-ms", type=float, default=1500.0, help="stale quote threshold in milliseconds")
    parser.add_argument("--tolerance-ms", type=float, default=250.0, help="age agreement tolerance in milliseconds")
    args = parser.parse_args()

    summary = analyze_logs(
        log_roots=[Path(path) for path in args.log_root],
        log_files=[Path(path) for path in args.log_file],
        out_dir=Path(args.out_dir).expanduser().resolve(),
        max_age_ms=args.max_age_ms,
        tolerance_ms=args.tolerance_ms,
    )
    print(json.dumps({"verdict": summary["verdict"], "counts": summary["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
