#!/usr/bin/env python3
"""Audit whether logs contain enough evidence to compare replay/live features."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any


FEATURE_COLUMNS = [
    "source_file",
    "line_number",
    "timestamp",
    "run_id",
    "mode",
    "event_type",
    "comparison_scope",
    "live_value_present",
    "replay_value_present",
    "diff_present",
    "status",
    "missing_evidence",
]

LOGIT_COLUMNS = [
    "source_file",
    "line_number",
    "timestamp",
    "run_id",
    "mode",
    "model_action",
    "has_live_score",
    "has_live_threshold",
    "has_live_wait_logit",
    "has_live_candidate_logits",
    "has_replay_logits",
    "status",
    "missing_evidence",
]

CANDIDATE_COLUMNS = [
    "source_file",
    "line_number",
    "timestamp",
    "run_id",
    "mode",
    "candidate_count",
    "candidate_sample_count",
    "has_full_candidate_features",
    "has_surface_scores",
    "has_surface_gate_diagnostics",
    "has_replay_candidate_set",
    "status",
    "missing_evidence",
]


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def bool_text(value: bool) -> str:
    return "true" if value else "false"


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


def sample_items(row: dict[str, Any]) -> list[Any]:
    sample = row.get("candidate_sample")
    return sample if isinstance(sample, list) else []


def has_features(sample: list[Any]) -> bool:
    return any(present(as_dict(item).get("features")) or present(as_dict(item).get("feature_hash")) for item in sample)


def has_surface_scores(row: dict[str, Any], sample: list[Any]) -> bool:
    diagnostics = as_dict(row.get("candidate_gate_diagnostics"))
    if present(diagnostics.get("flat_score")) or present(diagnostics.get("top_surface_tokens")):
        return True
    for item in sample:
        payload = as_dict(item)
        if present(payload.get("edge")) or present(payload.get("surface_action_score")) or present(payload.get("surface_flat_score")):
            return True
    return False


def feature_rows_for(source_file: str, line_number: int, row: dict[str, Any]) -> list[dict[str, str]]:
    event_type = str(row.get("event_type") or "")
    if event_type not in {"candidate_set", "model_decision", "risk_gate"}:
        return []
    sample = sample_items(row)
    live_candidate_features = has_features(sample)
    live_quote = present(as_dict(as_dict(row.get("market_snapshot")).get("option_nbbo")).get("bid")) or present(
        as_dict(as_dict(row.get("market_snapshot")).get("option_nbbo")).get("ask")
    )
    live_surface = has_surface_scores(row, sample)
    live_artifacts = "manifest" in json.dumps(row, sort_keys=True, default=str)
    scopes = [
        ("Protocol051 surface scores", live_surface, False),
        ("Protocol101 candidate features", live_candidate_features, False),
        ("Quote/market state", live_quote, False),
        ("Model artifact references", live_artifacts, False),
    ]
    rows = []
    for scope, live_present, replay_present in scopes:
        missing = []
        if not live_present:
            missing.append("live_value")
        if not replay_present:
            missing.append("replay_reference")
        status = "comparable" if live_present and replay_present else "not_comparable"
        rows.append(
            {
                "source_file": source_file,
                "line_number": str(line_number),
                "timestamp": str(row.get("timestamp") or ""),
                "run_id": str(row.get("run_id") or ""),
                "mode": str(row.get("mode") or ""),
                "event_type": event_type,
                "comparison_scope": scope,
                "live_value_present": bool_text(live_present),
                "replay_value_present": bool_text(replay_present),
                "diff_present": bool_text(False),
                "status": status,
                "missing_evidence": ";".join(missing),
            }
        )
    return rows


def logit_row(source_file: str, line_number: int, row: dict[str, Any]) -> dict[str, str] | None:
    if str(row.get("event_type") or "") != "model_decision":
        return None
    model = as_dict(row.get("model_decision"))
    has_score = present(model.get("score") or model.get("margin"))
    has_threshold = present(model.get("threshold"))
    has_wait = present(model.get("wait_logit"))
    has_candidate_logits = present(model.get("candidate_logits")) or present(model.get("logits"))
    has_replay = present(model.get("replay_logits"))
    missing = []
    if not has_score:
        missing.append("live_score")
    if not has_threshold:
        missing.append("live_threshold")
    if not has_wait:
        missing.append("live_wait_logit")
    if not has_candidate_logits:
        missing.append("live_candidate_logits")
    if not has_replay:
        missing.append("replay_logits")
    return {
        "source_file": source_file,
        "line_number": str(line_number),
        "timestamp": str(row.get("timestamp") or ""),
        "run_id": str(row.get("run_id") or ""),
        "mode": str(row.get("mode") or ""),
        "model_action": str(model.get("action") or row.get("selected_action") or ""),
        "has_live_score": bool_text(has_score),
        "has_live_threshold": bool_text(has_threshold),
        "has_live_wait_logit": bool_text(has_wait),
        "has_live_candidate_logits": bool_text(has_candidate_logits),
        "has_replay_logits": bool_text(has_replay),
        "status": "comparable" if has_score and has_wait and has_candidate_logits and has_replay else "not_comparable",
        "missing_evidence": ";".join(missing),
    }


def candidate_row(source_file: str, line_number: int, row: dict[str, Any]) -> dict[str, str] | None:
    if str(row.get("event_type") or "") != "candidate_set":
        return None
    sample = sample_items(row)
    diagnostics = as_dict(row.get("candidate_gate_diagnostics"))
    has_full_features = has_features(sample)
    has_surface = has_surface_scores(row, sample)
    has_gate = bool(diagnostics)
    has_replay = present(row.get("replay_candidate_set"))
    missing = []
    if not has_full_features:
        missing.append("full_candidate_features")
    if not has_surface:
        missing.append("surface_scores")
    if not has_replay:
        missing.append("replay_candidate_set")
    return {
        "source_file": source_file,
        "line_number": str(line_number),
        "timestamp": str(row.get("timestamp") or ""),
        "run_id": str(row.get("run_id") or ""),
        "mode": str(row.get("mode") or ""),
        "candidate_count": str(row.get("candidate_count") or ""),
        "candidate_sample_count": str(len(sample)),
        "has_full_candidate_features": bool_text(has_full_features),
        "has_surface_scores": bool_text(has_surface),
        "has_surface_gate_diagnostics": bool_text(has_gate),
        "has_replay_candidate_set": bool_text(has_replay),
        "status": "comparable" if has_full_features and has_surface and has_replay else "not_comparable",
        "missing_evidence": ";".join(missing),
    }


def analyze_logs(log_roots: list[Path], log_files: list[Path], out_dir: Path) -> dict[str, Any]:
    files, missing_inputs = iter_log_files(log_roots, log_files)
    feature_rows: list[dict[str, str]] = []
    logit_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    json_errors = 0
    for path in files:
        parsed, errors = read_jsonl(path)
        json_errors += errors
        for line_number, payload in parsed:
            source = str(path)
            feature_rows.extend(feature_rows_for(source, line_number, payload))
            maybe_logit = logit_row(source, line_number, payload)
            if maybe_logit:
                logit_rows.append(maybe_logit)
            maybe_candidate = candidate_row(source, line_number, payload)
            if maybe_candidate:
                candidate_rows.append(maybe_candidate)

    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "feature_diff.csv"
    logit_path = out_dir / "logit_diff.csv"
    candidate_path = out_dir / "candidate_set_diff.csv"
    write_csv(feature_path, FEATURE_COLUMNS, feature_rows)
    write_csv(logit_path, LOGIT_COLUMNS, logit_rows)
    write_csv(candidate_path, CANDIDATE_COLUMNS, candidate_rows)
    summary = build_summary(feature_rows, logit_rows, candidate_rows, files, missing_inputs, json_errors, out_dir)
    (out_dir / "feature_parity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "feature_parity_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    feature_rows: list[dict[str, str]],
    logit_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    files: list[Path],
    missing_inputs: list[str],
    json_errors: int,
    out_dir: Path,
) -> dict[str, Any]:
    feature_status = Counter(row["status"] for row in feature_rows)
    logit_status = Counter(row["status"] for row in logit_rows)
    candidate_status = Counter(row["status"] for row in candidate_rows)
    comparable = (
        feature_status.get("comparable", 0)
        + logit_status.get("comparable", 0)
        + candidate_status.get("comparable", 0)
    )
    total = len(feature_rows) + len(logit_rows) + len(candidate_rows)
    verdict = "replay metrics usable" if total and comparable == total else "replay metrics not yet usable"
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verdict": verdict,
        "verdict_reason": "logs do not contain paired replay/live feature, candidate, and logit evidence"
        if verdict != "replay metrics usable"
        else "all inspected parity rows are comparable",
        "counts": {
            "files_read": len(files),
            "feature_rows": len(feature_rows),
            "logit_rows": len(logit_rows),
            "candidate_rows": len(candidate_rows),
            "json_errors": json_errors,
            "feature_status_counts": dict(sorted(feature_status.items())),
            "logit_status_counts": dict(sorted(logit_status.items())),
            "candidate_status_counts": dict(sorted(candidate_status.items())),
        },
        "input": {
            "files_read": [str(path) for path in files],
            "missing_inputs": missing_inputs,
        },
        "artifacts": {
            "feature_parity_report": str(out_dir / "feature_parity_report.md"),
            "feature_diff": str(out_dir / "feature_diff.csv"),
            "logit_diff": str(out_dir / "logit_diff.csv"),
            "candidate_set_diff": str(out_dir / "candidate_set_diff.csv"),
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    return (
        "# Feature Parity Report\n\n"
        f"Decision: `{summary['verdict']}`\n\n"
        f"Reason: {summary['verdict_reason']}\n\n"
        "## Counts\n\n"
        f"- Files read: `{counts['files_read']}`\n"
        f"- Feature comparison rows: `{counts['feature_rows']}`\n"
        f"- Logit comparison rows: `{counts['logit_rows']}`\n"
        f"- Candidate-set comparison rows: `{counts['candidate_rows']}`\n"
        f"- JSON errors: `{counts['json_errors']}`\n\n"
        "## Interpretation\n\n"
        "This diagnostic does not load models or rerun replay. It asks whether existing logs contain paired evidence "
        "needed to compare live Protocol051-to-Protocol101 features against replay for the same market state. "
        "When replay references, full live features, candidate tensors, or logits are absent, the result is not comparable.\n\n"
        "## Required Evidence For Pass\n\n"
        "- Live normalized market state and option ladder snapshot.\n"
        "- Replay-built market state for the same timestamp/contracts.\n"
        "- Candidate set identifiers before and after Protocol051 filtering.\n"
        "- Protocol051 surface scores and Protocol101 feature rows or hashes.\n"
        "- Protocol101 scaled tensor hash, wait logit, candidate logits, selected action, and threshold.\n"
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
