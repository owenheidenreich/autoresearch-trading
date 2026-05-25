#!/usr/bin/env python3
"""Write the research_ops CEO dashboard from machine-readable control state."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path


COMPLETED_STATUSES = {"completed", "complete", "closed", "accepted", "rejected", "falsified", "supported"}


def clean_yaml_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value


def read_yamlish(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    current_parent = ""
    current_key = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if stripped.startswith("- "):
            if current_key:
                values.setdefault(current_key, [])
                if isinstance(values[current_key], list):
                    values[current_key].append(clean_yaml_value(stripped[2:]))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0 and not value:
            current_parent = key
            current_key = key
            values.setdefault(key, [])
            continue
        full_key = f"{current_parent}.{key}" if indent > 0 and current_parent else key
        if value:
            values[full_key] = clean_yaml_value(value)
            current_key = ""
        else:
            values[full_key] = []
            current_key = full_key
    return values


def read_assumptions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def iteration_manifests(root: Path) -> list[tuple[Path, dict[str, object]]]:
    iterations = []
    for manifest_path in sorted((root / "research_ops" / "iterations").glob("*/manifest.yaml")):
        iterations.append((manifest_path.parent, read_yamlish(manifest_path)))
    return iterations


def latest_by_status(iterations: list[tuple[Path, dict[str, object]]], completed: bool) -> tuple[Path, dict[str, object]] | None:
    candidates = []
    for path, manifest in iterations:
        status = str(manifest.get("status", "")).lower()
        is_completed = status in COMPLETED_STATUSES
        if is_completed == completed:
            candidates.append((path, manifest))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: str(item[1].get("created_at", "")))[-1]


def extract_section(path: Path, heading: str) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        return []
    out = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        stripped = line.strip()
        if stripped:
            out.append(stripped)
    return [line for line in out if line not in {"- None yet.", "None yet.", "pending"}]


def normalize_section_items(items: list[str]) -> list[str]:
    normalized = []
    for item in items:
        clean = item.strip()
        while clean.startswith(("- ", "* ")):
            clean = clean[2:].strip()
        if clean:
            normalized.append(clean)
    return normalized


def list_or_none(items: list[str]) -> list[str]:
    return items if items else ["None recorded."]


def format_manifest(manifest_pair: tuple[Path, dict[str, object]] | None) -> list[str]:
    if manifest_pair is None:
        return ["None."]
    path, manifest = manifest_pair
    return [
        f"Iteration: `{manifest.get('iteration_id', path.name)}`",
        f"Assumption: `{manifest.get('assumption_id', 'unknown')}`",
        f"Title: {manifest.get('title', 'unknown')}",
        f"Status: `{manifest.get('status', 'unknown')}`",
    ]


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def code_or_text(value: object) -> str:
    text = str(value)
    return text if "`" in text else f"`{text}`"


def build_dashboard(root: Path) -> str:
    research_ops = root / "research_ops"
    state = read_yamlish(research_ops / "CURRENT_STATE.yaml")
    assumptions = read_assumptions(research_ops / "ASSUMPTION_REGISTRY.csv")
    iterations = iteration_manifests(root)
    active = latest_by_status(iterations, completed=False)
    completed = latest_by_status(iterations, completed=True)

    completed_path = completed[0] if completed else None
    latest_decision = (
        normalize_section_items(extract_section(completed_path / "05_decision_memo.md", "## Decision"))
        if completed_path
        else []
    )
    confirmed = (
        normalize_section_items(extract_section(completed_path / "04_verifier_report.md", "## Newly Confirmed Evidence"))
        if completed_path
        else []
    )
    falsified = (
        normalize_section_items(extract_section(completed_path / "04_verifier_report.md", "## Newly Falsified Assumptions"))
        if completed_path
        else []
    )
    next_iteration = (
        normalize_section_items(extract_section(completed_path / "05_decision_memo.md", "## Next Recommended Iteration"))
        if completed_path
        else []
    )

    p0_rows = [row for row in assumptions if row.get("priority") == "P0"]
    p0_lines = [
        f"`{row['id']}` {row['assumption']} - `{row['status']}` - next: `{row['next_diagnostic']}`"
        for row in p0_rows
    ]
    blocked_actions = state.get("blocked_actions", [])
    if not isinstance(blocked_actions, list):
        blocked_actions = []

    now = dt.date.today().isoformat()
    operational_default = state.get("current_default.name", state.get("control.operational_default", "unknown"))
    default_scope = state.get("current_default.scope", "unknown")
    real_money = state.get("current_default.real_money", "false")
    safety = state.get("audit_interpretation.next_phase", "execution-and-parity falsification")
    next_prompt = (
        next_iteration[0]
        if next_iteration
        else state.get("next_recommended_prompt", "Create the first verifier RFC for Protocol101 execution realism and replay/live fill parity.")
    )

    decisions_required = []
    if active:
        decisions_required.append(f"Complete or block active iteration `{active[1].get('iteration_id', active[0].name)}`.")
    if any(row.get("status") == "open" for row in p0_rows):
        decisions_required.append("Resolve open P0 assumptions before any promotion, threshold, runtime, or model-capacity work.")
    decisions_required.extend(latest_decision)

    return "\n\n".join(
        [
            "# CEO Dashboard",
            f"Last updated: {now}",
            "## 1. Current Operational Default\n\n"
            f"- Name: `{operational_default}`\n"
            f"- Scope: {default_scope}\n"
            f"- Real money authorized: `{real_money}`",
            "## 2. Current Safety Posture\n\n"
            f"- Posture: `{safety}`\n"
            "- Replay profitability remains a hypothesis until execution and parity assumptions are tested.",
            "## 3. Active Iteration\n\n" + bullet_list(format_manifest(active)),
            "## 4. Latest Completed Iteration\n\n" + bullet_list(format_manifest(completed)),
            "## 5. Decisions Required\n\n" + bullet_list(list_or_none(decisions_required)),
            "## 6. P0 Assumptions\n\n" + bullet_list(list_or_none(p0_lines)),
            "## 7. Blocked Actions\n\n" + bullet_list(list(blocked_actions)),
            "## 8. Newly Confirmed Evidence\n\n" + bullet_list(list_or_none(confirmed)),
            "## 9. Newly Falsified Assumptions\n\n" + bullet_list(list_or_none(falsified)),
            "## 10. Next Recommended Codex Prompt\n\n" + code_or_text(next_prompt),
            "## Dashboard Rule\n\n"
            "This dashboard is not primary evidence. Evidence lives in iteration artifacts, verifier reports, logs, manifests, and decision memos.",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="repository root; defaults to cwd")
    parser.add_argument("--dry-run", action="store_true", help="print dashboard instead of writing")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else Path.cwd().resolve()
    dashboard = build_dashboard(root)
    if args.dry_run:
        print(dashboard, end="")
    else:
        path = root / "research_ops" / "CEO_DASHBOARD.md"
        path.write_text(dashboard, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
