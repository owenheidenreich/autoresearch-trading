#!/usr/bin/env python3
"""Validate a local research_ops iteration folder.

This script validates only local files under research_ops. It must not inspect
broker APIs, import trading runtime code, run models, or mutate v4.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REQUIRED_FILES = [
    "manifest.yaml",
    "00_request.md",
    "01_cartography.md",
    "02_rfc.md",
    "03_implementation_summary.md",
    "04_verifier_report.md",
    "05_decision_memo.md",
]

REQUIRED_MANIFEST_FIELDS = [
    "iteration_id",
    "assumption_id",
    "title",
    "status",
    "created_at",
    "owner_role",
    "blocked_actions",
    "allowed_paths",
    "forbidden_paths",
    "expected_outputs",
]

REQUIRED_DECISION_SECTIONS = [
    "## Decision",
    "## Context",
    "## Evidence Reviewed",
    "## Decision Details",
    "## Assumptions Accepted Or Rejected",
    "## Follow-Up Actions",
]

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


def clean_yaml_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value


def read_yamlish(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    current_key = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if not current_key:
                continue
            data.setdefault(current_key, [])
            value = clean_yaml_value(stripped[2:])
            if isinstance(data[current_key], list):
                data[current_key].append(value)
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            data[key] = clean_yaml_value(value)
            current_key = ""
        else:
            data[key] = []
            current_key = key
    return data


def assumption_ids(root: Path) -> set[str]:
    path = root / "research_ops" / "ASSUMPTION_REGISTRY.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["id"] for row in reader}


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


def extract_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join(lines[start:end])


def forbidden_path_hits(manifest: dict[str, object], implementation_summary: str) -> list[str]:
    modified_section = extract_section(implementation_summary, "## Files Modified")
    hits: list[str] = []
    forbidden_paths = manifest.get("forbidden_paths", [])
    if not isinstance(forbidden_paths, list):
        return ["manifest.yaml: forbidden_paths must be a list"]
    for forbidden in forbidden_paths:
        base = str(forbidden).replace("**", "").replace("*", "").rstrip("/")
        if base and base in modified_section:
            hits.append(str(forbidden))
    return hits


def resolve_iteration(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = root / value
    by_id = root / "research_ops" / "iterations" / value
    return direct if direct.exists() else by_id


def validate_iteration(root: Path, iteration_dir: Path, check_registry: bool) -> list[str]:
    errors: list[str] = []
    if check_registry:
        errors.extend(validate_assumption_registry(root))

    for filename in REQUIRED_FILES:
        path = iteration_dir / filename
        if not path.exists():
            errors.append(f"{path}: missing required file")

    artifacts = iteration_dir / "artifacts"
    if not artifacts.exists() or not artifacts.is_dir():
        errors.append(f"{artifacts}: missing artifacts directory")

    manifest_path = iteration_dir / "manifest.yaml"
    if not manifest_path.exists():
        return errors

    manifest = read_yamlish(manifest_path)
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            errors.append(f"{manifest_path}: missing required field {field}")

    for field in ["blocked_actions", "allowed_paths", "forbidden_paths", "expected_outputs"]:
        if field in manifest and not isinstance(manifest[field], list):
            errors.append(f"{manifest_path}: {field} must be a list")

    assumption_id = str(manifest.get("assumption_id", ""))
    if assumption_id and assumption_id not in assumption_ids(root):
        errors.append(f"{manifest_path}: assumption_id not found in ASSUMPTION_REGISTRY.csv: {assumption_id}")

    implementation_path = iteration_dir / "03_implementation_summary.md"
    if implementation_path.exists():
        hits = forbidden_path_hits(manifest, implementation_path.read_text(encoding="utf-8"))
        for hit in hits:
            errors.append(f"{implementation_path}: forbidden path listed as modified: {hit}")

    decision_path = iteration_dir / "05_decision_memo.md"
    if decision_path.exists():
        decision_text = decision_path.read_text(encoding="utf-8")
        for heading in REQUIRED_DECISION_SECTIONS:
            if heading not in decision_text:
                errors.append(f"{decision_path}: missing required section {heading}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iteration", help="iteration directory or iteration id")
    parser.add_argument("--root", default=None, help="repository root; defaults to cwd")
    parser.add_argument("--registry", action="store_true", help="also validate ASSUMPTION_REGISTRY.csv header")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else Path.cwd().resolve()
    iteration_dir = resolve_iteration(root, args.iteration)
    errors = validate_iteration(root, iteration_dir, args.registry)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {iteration_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
