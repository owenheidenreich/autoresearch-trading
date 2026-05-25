#!/usr/bin/env python3
"""Create a local research_ops iteration folder.

This script is intentionally limited to file management under research_ops.
It must not import trading runtime, broker, model, or paid-data code.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from pathlib import Path


ITERATION_ID_RE = re.compile(r"^ITER-[0-9]{3}_[A-Za-z0-9][A-Za-z0-9_-]*$")

REQUIRED_FILES = [
    "00_request.md",
    "01_cartography.md",
    "02_rfc.md",
    "03_implementation_summary.md",
    "04_verifier_report.md",
    "05_decision_memo.md",
]

DEFAULT_BLOCKED_ACTIONS = [
    "no challenger promotion",
    "no threshold tuning",
    "no new model training",
    "no protected holdout scoring for exploration",
    "no assuming replay profitability proves live edge",
    "no assuming deterministic ask-entry/bid-exit proves fillability",
    "no runtime flag mutation as part of research",
    "no paid data downloads without approval",
]

DEFAULT_FORBIDDEN_PATHS = [
    "v4/runtime/**",
    "v4/ops/launchd/**",
    "v4/ops/ibkr/run_protocol101_paper_session.sh",
    "v4/live/ibkr_paper_executor.py",
    "v4/live/ibkr_paper_guard.py",
    "v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/**",
    "v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/**",
    "v4/raw/**",
    "v4/normalized/**",
    "v4/normalized_official_context/**",
    "data/processed/**",
]


def repo_root(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else Path.cwd().resolve()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def quote_yaml(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_yaml(path: Path, manifest: dict[str, object]) -> None:
    lines: list[str] = []
    for key, value in manifest.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {quote_yaml(str(item))}")
        else:
            lines.append(f"{key}: {quote_yaml(str(value))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assumption_exists(root: Path, assumption_id: str) -> bool:
    path = root / "research_ops" / "ASSUMPTION_REGISTRY.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return any(row.get("id") == assumption_id for row in csv.DictReader(handle))


def markdown_files(iteration_id: str, assumption_id: str, title: str) -> dict[str, str]:
    common = f"Iteration ID: `{iteration_id}`\nAssumption ID: `{assumption_id}`\nTitle: {title}\n"
    return {
        "00_request.md": (
            "# Iteration Request\n\n"
            f"{common}\n"
            "## Request\n\n"
            "State the user request or research control objective here.\n\n"
            "## Hard Constraints\n\n"
            "- Do not modify v4 trading logic without explicit approval.\n"
            "- Do not mutate runtime flags, launchd, broker behavior, paid data, training, thresholds, or model artifacts.\n"
        ),
        "01_cartography.md": (
            "# Cartography Report\n\n"
            f"{common}\n"
            "## Question\n\n"
            "What code, data, logs, docs, and artifacts define the current truth?\n\n"
            "## Sources Read\n\n"
            "| Source | Why read | Binding/stale/unknown |\n"
            "|---|---|---|\n"
            "|  |  |  |\n\n"
            "## Findings\n\n"
            "| Finding | Evidence | Risk |\n"
            "|---|---|---|\n"
            "|  |  |  |\n"
        ),
        "02_rfc.md": (
            "# Experiment RFC\n\n"
            f"{common}\n"
            "## Hypothesis\n\n"
            "## Method\n\n"
            "## Falsification Criteria\n\n"
            "## Protected Surfaces\n\n"
            "- Broker/API:\n"
            "- Paid data:\n"
            "- Runtime flags:\n"
            "- Models/scalers:\n"
            "- Launchd:\n"
            "- Holdouts:\n"
        ),
        "03_implementation_summary.md": (
            "# Implementation Summary\n\n"
            f"{common}\n"
            "## Change Summary\n\n"
            "No implementation yet.\n\n"
            "## Files Created\n\n"
            "- None yet.\n\n"
            "## Files Modified\n\n"
            "- None yet.\n\n"
            "## Commands Run\n\n"
            "```text\n\n"
            "```\n\n"
            "## Tests\n\n"
            "```text\n\n"
            "```\n"
        ),
        "04_verifier_report.md": (
            "# Verifier Report\n\n"
            f"{common}\n"
            "## Verification Question\n\n"
            "## Evidence Reviewed\n\n"
            "| Evidence | Location | Notes |\n"
            "|---|---|---|\n"
            "|  |  |  |\n\n"
            "## Results\n\n"
            "## Newly Confirmed Evidence\n\n"
            "- None yet.\n\n"
            "## Newly Falsified Assumptions\n\n"
            "- None yet.\n\n"
            "## Verdict\n\n"
            "blocked\n"
        ),
        "05_decision_memo.md": (
            "# Decision Memo\n\n"
            f"{common}\n"
            "## Decision\n\n"
            "pending\n\n"
            "## Context\n\n"
            "## Evidence Reviewed\n\n"
            "| Artifact | Location | Weight |\n"
            "|---|---|---|\n"
            "|  |  |  |\n\n"
            "## Decision Details\n\n"
            "```text\n"
            "Decision:\n"
            "Does this change PAPER_DEFAULT_PROTOCOL101:\n"
            "Does this authorize model training:\n"
            "Does this authorize broker/data/runtime action:\n"
            "Evidence reviewed:\n"
            "Risks accepted:\n"
            "Reversal condition:\n"
            "Owner:\n"
            "```\n\n"
            "## Assumptions Accepted Or Rejected\n\n"
            "| Assumption ID | Treatment | Rationale |\n"
            "|---|---|---|\n"
            "|  |  |  |\n\n"
            "## Follow-Up Actions\n\n"
            "- None yet.\n"
        ),
    }


def create_iteration(
    root: Path,
    iteration_id: str,
    assumption_id: str,
    title: str,
    owner_role: str,
    created_at: str,
    force: bool,
) -> Path:
    if not ITERATION_ID_RE.match(iteration_id):
        raise ValueError("iteration id must match ITER-001_short_slug")
    if not assumption_exists(root, assumption_id):
        raise ValueError(f"assumption_id not found in ASSUMPTION_REGISTRY.csv: {assumption_id}")

    iteration_dir = root / "research_ops" / "iterations" / iteration_id
    artifacts_dir = iteration_dir / "artifacts"
    if iteration_dir.exists() and not force:
        raise FileExistsError(f"iteration already exists: {iteration_dir}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / ".gitkeep").write_text("", encoding="utf-8")

    manifest = {
        "iteration_id": iteration_id,
        "assumption_id": assumption_id,
        "title": title,
        "status": "draft",
        "created_at": created_at,
        "owner_role": owner_role,
        "blocked_actions": DEFAULT_BLOCKED_ACTIONS,
        "allowed_paths": [
            f"research_ops/iterations/{iteration_id}/**",
            "research_ops/CEO_DASHBOARD.md",
            "research_ops/ASSUMPTION_REGISTRY.csv",
        ],
        "forbidden_paths": DEFAULT_FORBIDDEN_PATHS,
        "expected_outputs": REQUIRED_FILES,
    }
    write_yaml(iteration_dir / "manifest.yaml", manifest)

    for filename, content in markdown_files(iteration_id, assumption_id, title).items():
        path = iteration_dir / filename
        if path.exists() and not force:
            raise FileExistsError(f"file already exists: {path}")
        path.write_text(content, encoding="utf-8")

    return iteration_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, dest="iteration_id", help="iteration id, e.g. ITER-001_quote_age_truth")
    parser.add_argument("--assumption", required=True, dest="assumption_id", help="assumption id, e.g. A001")
    parser.add_argument("--title", required=True, help="human-readable title")
    parser.add_argument("--owner-role", default="research_ops_agent", help="role responsible for this iteration")
    parser.add_argument("--created-at", default=utc_now(), help="creation timestamp; defaults to current UTC time")
    parser.add_argument("--root", default=None, help="repository root; defaults to cwd")
    parser.add_argument("--force", action="store_true", help="overwrite an existing draft packet")
    args = parser.parse_args()

    iteration_dir = create_iteration(
        repo_root(args.root),
        args.iteration_id,
        args.assumption_id,
        args.title,
        args.owner_role,
        args.created_at,
        args.force,
    )
    print(iteration_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
