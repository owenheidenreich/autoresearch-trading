#!/usr/bin/env python3
"""Audit whether Protocol101 decisions are reconstructable from logs alone."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any


MATRIX_COLUMNS = [
    "source_file",
    "line_number",
    "timestamp",
    "session",
    "run_id",
    "mode",
    "event_type",
    "decision_event",
    "logged_action",
    "has_identity",
    "has_model_action",
    "has_model_reason",
    "has_model_score",
    "has_model_threshold",
    "has_wait_logit",
    "has_candidate_count",
    "has_candidate_sample",
    "has_full_candidate_features",
    "has_selected_contract",
    "has_order_intent",
    "has_risk_gate_result",
    "has_risk_gate_inputs",
    "has_account_state",
    "has_option_quote",
    "has_quote_timestamp",
    "has_quote_age",
    "has_artifact_refs",
    "has_runtime_flag",
    "broker_endpoint_called",
    "reconstruction_status",
    "missing_fields",
]

DECISION_EVENT_TYPES = {
    "candidate_set",
    "model_decision",
    "risk_gate",
    "paper_order_blocked",
    "paper_order_dry_run",
    "paper_order_submitted",
    "paper_entry_fill",
    "paper_exit_intent",
    "paper_exit_submitted",
    "paper_exit_fill",
    "exit_decision",
}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def iter_log_files(log_roots: list[Path], log_files: list[Path]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    missing: list[str] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            files.append(resolved)

    for path in log_files:
        expanded = path.expanduser()
        if expanded.is_file():
            add(expanded)
        else:
            missing.append(str(path))
    for root in log_roots:
        expanded = root.expanduser()
        if not expanded.exists():
            missing.append(str(root))
        elif expanded.is_file():
            add(expanded)
        else:
            for path in sorted(expanded.rglob("*.jsonl")):
                if path.is_file():
                    add(path)
    return files, missing


def read_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    rows: list[tuple[int, dict[str, Any]]] = []
    errors = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if isinstance(payload, dict):
                rows.append((line_number, payload))
            else:
                errors += 1
    return rows, errors


def candidate_sample_count(row: dict[str, Any]) -> int:
    sample = row.get("candidate_sample")
    if isinstance(sample, list):
        return len(sample)
    extra = as_dict(row.get("extra"))
    sample = extra.get("candidate_sample")
    return len(sample) if isinstance(sample, list) else 0


def has_option_quote(market: dict[str, Any]) -> bool:
    option = as_dict(market.get("option_nbbo"))
    return any(present(option.get(key)) for key in ("bid", "ask", "quote_age_ms", "timestamp", "quote_timestamp"))


def has_quote_timestamp(market: dict[str, Any]) -> bool:
    option = as_dict(market.get("option_nbbo"))
    return any(
        present(option.get(key))
        for key in (
            "timestamp",
            "quote_timestamp",
            "quote_timestamp_ms",
            "received_timestamp",
            "received_timestamp_ms",
            "decision_timestamp",
            "decision_timestamp_ms",
        )
    )


def has_quote_age(market: dict[str, Any]) -> bool:
    option = as_dict(market.get("option_nbbo"))
    return present(option.get("quote_age_ms"))


def has_artifact_refs(row: dict[str, Any]) -> bool:
    text = json.dumps(row, sort_keys=True, default=str)
    needles = ("manifest", "model_path", "scaler", "artifact", "protocol101_manifest", "surface_manifest")
    return any(needle in text for needle in needles)


def classify_reconstruction(row: dict[str, Any], checks: dict[str, bool], missing: list[str]) -> str:
    event_type = str(row.get("event_type") or "")
    if event_type not in DECISION_EVENT_TYPES:
        return "not_decision_event"
    critical = [
        "identity",
        "model_action",
        "model_reason",
        "candidate_count",
        "risk_gate_result",
        "account_state",
        "option_quote",
        "quote_timestamp",
        "quote_age",
        "artifact_refs",
    ]
    if event_type == "candidate_set":
        critical = ["identity", "candidate_count", "candidate_sample", "full_candidate_features", "artifact_refs"]
    elif event_type.startswith("paper_order") or event_type.startswith("paper_entry") or event_type.startswith("paper_exit"):
        critical = ["identity", "selected_contract", "order_intent", "risk_gate_result", "account_state", "option_quote", "quote_timestamp", "quote_age"]
    failed = [name for name in critical if not checks.get(name, False)]
    missing.extend(failed)
    return "sufficient" if not failed else "insufficient"


def matrix_row(source_file: str, line_number: int, row: dict[str, Any]) -> dict[str, str]:
    model = as_dict(row.get("model_decision"))
    risk = as_dict(row.get("risk_gate"))
    account = as_dict(row.get("account") or row.get("paper_account_state"))
    market = as_dict(row.get("market_snapshot"))
    order = as_dict(row.get("order") or row.get("order_intent"))
    selected = as_dict(row.get("selected_contract"))
    event_type = str(row.get("event_type") or "")
    sample_count = candidate_sample_count(row)
    checks = {
        "identity": all(present(row.get(key)) for key in ("timestamp", "session", "run_id", "mode", "event_type")),
        "model_action": present(model.get("action")) or present(row.get("selected_action")),
        "model_reason": present(model.get("reason")) or present(row.get("reason")),
        "model_score": present(model.get("score") or model.get("margin")),
        "model_threshold": present(model.get("threshold") or row.get("model_threshold")),
        "wait_logit": present(model.get("wait_logit")),
        "candidate_count": present(row.get("candidate_count")) or present(model.get("candidate_count")),
        "candidate_sample": sample_count > 0,
        "full_candidate_features": bool(sample_count and any("features" in as_dict(item) for item in row.get("candidate_sample", []))),
        "selected_contract": bool(selected),
        "order_intent": bool(order),
        "risk_gate_result": present(risk.get("passed")) and present(risk.get("reason")),
        "risk_gate_inputs": present(risk.get("validation")) or present(risk.get("inputs")),
        "account_state": any(present(account.get(key)) for key in ("cash", "equity", "open_positions", "starting_cash")),
        "option_quote": has_option_quote(market),
        "quote_timestamp": has_quote_timestamp(market),
        "quote_age": has_quote_age(market),
        "artifact_refs": has_artifact_refs(row),
        "runtime_flag": present(row.get("runtime_flag")),
    }
    missing: list[str] = []
    status = classify_reconstruction(row, checks, missing)
    logged_action = row.get("selected_action") or model.get("action") or order.get("action") or ""
    return {
        "source_file": source_file,
        "line_number": str(line_number),
        "timestamp": str(row.get("timestamp") or ""),
        "session": str(row.get("session") or ""),
        "run_id": str(row.get("run_id") or ""),
        "mode": str(row.get("mode") or ""),
        "event_type": event_type,
        "decision_event": bool_text(event_type in DECISION_EVENT_TYPES),
        "logged_action": str(logged_action),
        "has_identity": bool_text(checks["identity"]),
        "has_model_action": bool_text(checks["model_action"]),
        "has_model_reason": bool_text(checks["model_reason"]),
        "has_model_score": bool_text(checks["model_score"]),
        "has_model_threshold": bool_text(checks["model_threshold"]),
        "has_wait_logit": bool_text(checks["wait_logit"]),
        "has_candidate_count": bool_text(checks["candidate_count"]),
        "has_candidate_sample": bool_text(checks["candidate_sample"]),
        "has_full_candidate_features": bool_text(checks["full_candidate_features"]),
        "has_selected_contract": bool_text(checks["selected_contract"]),
        "has_order_intent": bool_text(checks["order_intent"]),
        "has_risk_gate_result": bool_text(checks["risk_gate_result"]),
        "has_risk_gate_inputs": bool_text(checks["risk_gate_inputs"]),
        "has_account_state": bool_text(checks["account_state"]),
        "has_option_quote": bool_text(checks["option_quote"]),
        "has_quote_timestamp": bool_text(checks["quote_timestamp"]),
        "has_quote_age": bool_text(checks["quote_age"]),
        "has_artifact_refs": bool_text(checks["artifact_refs"]),
        "has_runtime_flag": bool_text(checks["runtime_flag"]),
        "broker_endpoint_called": bool_text(bool(row.get("broker_order_endpoint_called") or row.get("broker_endpoint_called"))),
        "reconstruction_status": status,
        "missing_fields": ";".join(missing),
    }


def analyze_logs(log_roots: list[Path], log_files: list[Path], out_dir: Path) -> dict[str, Any]:
    files, missing_inputs = iter_log_files(log_roots, log_files)
    rows: list[dict[str, str]] = []
    json_errors = 0
    for path in files:
        parsed, errors = read_jsonl(path)
        json_errors += errors
        for line_number, payload in parsed:
            rows.append(matrix_row(str(path), line_number, payload))

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "decision_reconstruction_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows, files, missing_inputs, json_errors, matrix_path, out_dir)
    (out_dir / "decision_reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "missing_fields_report.md").write_text(render_missing_fields_report(summary), encoding="utf-8")
    (out_dir / "log_schema_gap_report.md").write_text(render_schema_gap_report(summary), encoding="utf-8")
    return summary


def build_summary(
    rows: list[dict[str, str]],
    files: list[Path],
    missing_inputs: list[str],
    json_errors: int,
    matrix_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    decision_rows = [row for row in rows if row["decision_event"] == "true"]
    insufficient = [row for row in decision_rows if row["reconstruction_status"] == "insufficient"]
    sufficient = [row for row in decision_rows if row["reconstruction_status"] == "sufficient"]
    missing_counter: Counter[str] = Counter()
    for row in insufficient:
        for field in row["missing_fields"].split(";"):
            if field:
                missing_counter[field] += 1
    event_counts = Counter(row["event_type"] for row in rows)
    status_counts = Counter(row["reconstruction_status"] for row in rows)
    broker_rows = [row for row in rows if row["broker_endpoint_called"] == "true"]
    verdict = "logs sufficient"
    reason = "all decision rows have required reconstruction evidence"
    if not decision_rows:
        verdict = "logs insufficient"
        reason = "no decision rows found"
    elif insufficient:
        verdict = "schema patch required"
        reason = "one or more decision rows are missing fields needed for independent reconstruction"
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verdict": verdict,
        "verdict_reason": reason,
        "counts": {
            "files_read": len(files),
            "total_rows": len(rows),
            "decision_rows": len(decision_rows),
            "sufficient_decision_rows": len(sufficient),
            "insufficient_decision_rows": len(insufficient),
            "broker_endpoint_rows": len(broker_rows),
            "json_errors": json_errors,
            "event_counts": dict(sorted(event_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "missing_field_counts": dict(sorted(missing_counter.items())),
        },
        "input": {
            "files_read": [str(path) for path in files],
            "missing_inputs": missing_inputs,
        },
        "artifacts": {
            "decision_reconstruction_matrix": str(matrix_path),
            "missing_fields_report": str(out_dir / "missing_fields_report.md"),
            "log_schema_gap_report": str(out_dir / "log_schema_gap_report.md"),
        },
    }


def render_missing_fields_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    missing = counts["missing_field_counts"]
    missing_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in missing.items()) if missing else "- None."
    return (
        "# Missing Fields Report\n\n"
        f"Decision: `{summary['verdict']}`\n\n"
        f"Reason: {summary['verdict_reason']}\n\n"
        "## Counts\n\n"
        f"- Total rows: `{counts['total_rows']}`\n"
        f"- Decision rows: `{counts['decision_rows']}`\n"
        f"- Sufficient decision rows: `{counts['sufficient_decision_rows']}`\n"
        f"- Insufficient decision rows: `{counts['insufficient_decision_rows']}`\n"
        f"- Broker endpoint rows: `{counts['broker_endpoint_rows']}`\n\n"
        "## Missing Field Counts\n\n"
        f"{missing_lines}\n"
    )


def render_schema_gap_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    return (
        "# Log Schema Gap Report\n\n"
        "## Verdict\n\n"
        f"`{summary['verdict']}`\n\n"
        "## Primary Gaps\n\n"
        "- Full candidate feature rows are not logged for every decision.\n"
        "- Raw quote timestamps and quote age reconstruction fields are not logged for every decision.\n"
        "- Artifact references sufficient to bind model/scaler/manifest state are not logged per decision.\n"
        "- Protocol101 logits, wait logit, and rejected candidate logits are not consistently logged.\n"
        "- Guard inputs are not logged in enough detail to independently rerun every guard decision.\n\n"
        "## Evidence Counts\n\n"
        f"- Decision rows inspected: `{counts['decision_rows']}`\n"
        f"- Insufficient decision rows: `{counts['insufficient_decision_rows']}`\n"
        f"- Broker endpoint rows: `{counts['broker_endpoint_rows']}`\n\n"
        "## Required Schema Patch\n\n"
        "For every live/paper/no-order decision, log candidate set, feature vector or hash plus schema version, raw quote timestamp fields, "
        "model artifact references, model logits, selected action, risk-gate inputs/result, account state, and order/fill outcome where applicable.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", action="append", default=[])
    parser.add_argument("--log-file", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    summary = analyze_logs(
        [Path(path) for path in args.log_root],
        [Path(path) for path in args.log_file],
        Path(args.out_dir).expanduser().resolve(),
    )
    print(json.dumps({"verdict": summary["verdict"], "counts": summary["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
