#!/usr/bin/env python3
"""Validate a local research_ops iteration packet with lightweight checks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "iteration_id",
    "title",
    "status",
    "created_date",
    "control_ref",
    "scope",
    "hard_constraints",
    "required_artifacts",
}

REQUIRED_ARTIFACTS = {
    "cartography_report": "cartography_report.md",
    "experiment_rfc": "experiment_rfc.md",
    "implementation_summary": "implementation_summary.md",
    "verifier_report": "verifier_report.md",
    "decision_memo": "decision_memo.md",
    "ceo_packet": "ceo_packet.md",
}

ASSUMPTION_HEADER = [
    "id",
    "priority",
    "layer",
    "assumption",
    "current_evidence",
    "risk_if_false",
    "falsification_test",
    "confidence_increases_if",
    "confidence_collapses_if",
    "required_artifacts",
    "blocked_actions",
    "status",
    "next_diagnostic",
]


def load_json(path: Path) -> tuple[dict[str, object] | None, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except Exception as exc:  # noqa: BLE001
        return None, [f"{path}: invalid JSON: {exc}"]


def validate_manifest(iteration_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = iteration_dir / "iteration_manifest.json"
    if not manifest_path.exists():
        return [f"{manifest_path}: missing manifest"]

    manifest, load_errors = load_json(manifest_path)
    errors.extend(load_errors)
    if manifest is None:
        return errors

    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    for field in missing:
        errors.append(f"{manifest_path}: missing required field {field}")

    if manifest.get("schema_version") != 1:
        errors.append(f"{manifest_path}: schema_version must be 1")

    required_artifacts = manifest.get("required_artifacts")
    if not isinstance(required_artifacts, dict):
        errors.append(f"{manifest_path}: required_artifacts must be an object")
        return errors

    missing_artifact_keys = sorted(set(REQUIRED_ARTIFACTS) - set(required_artifacts))
    for key in missing_artifact_keys:
        errors.append(f"{manifest_path}: missing required_artifacts.{key}")

    reports_dir = iteration_dir / "reports"
    for artifact_id, filename in REQUIRED_ARTIFACTS.items():
        artifact_path = reports_dir / filename
        if not artifact_path.exists():
            errors.append(f"{artifact_path}: missing {artifact_id} template copy")

    return errors


def validate_diagnostic_summaries(iteration_dir: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(iteration_dir.glob("**/diagnostic_summary*.json")):
        summary, load_errors = load_json(path)
        errors.extend(load_errors)
        if summary is None:
            continue
        required = {
            "schema_version",
            "iteration_id",
            "diagnostic_id",
            "status",
            "evidence_level",
            "summary",
            "artifacts",
            "assumptions_touched",
            "falsification_result",
        }
        for field in sorted(required - set(summary)):
            errors.append(f"{path}: missing required field {field}")
    return errors


def validate_assumption_registry(root: Path) -> list[str]:
    path = root / "research_ops" / "ASSUMPTION_REGISTRY.csv"
    if not path.exists():
        return [f"{path}: missing assumption registry"]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header != ASSUMPTION_HEADER:
        return [f"{path}: unexpected CSV header"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iteration", help="iteration directory or iteration id")
    parser.add_argument("--root", default=None, help="repository root; defaults to cwd")
    parser.add_argument("--registry", action="store_true", help="also validate ASSUMPTION_REGISTRY.csv header")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else Path.cwd().resolve()
    iteration = Path(args.iteration)
    if not iteration.is_absolute():
        direct = root / iteration
        by_id = root / "research_ops" / "iterations" / args.iteration
        iteration = direct if direct.exists() else by_id

    errors = []
    errors.extend(validate_manifest(iteration))
    errors.extend(validate_diagnostic_summaries(iteration))
    if args.registry:
        errors.extend(validate_assumption_registry(root))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {iteration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
