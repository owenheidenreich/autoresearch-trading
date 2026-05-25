#!/usr/bin/env python3
"""Create a local research_ops iteration packet.

This script is intentionally limited to file management under research_ops.
It must not import trading runtime, broker, model, or paid-data code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path


PACKET_TEMPLATES = {
    "cartography_report": "cartography_report.md",
    "experiment_rfc": "experiment_rfc.md",
    "implementation_summary": "implementation_summary.md",
    "verifier_report": "verifier_report.md",
    "decision_memo": "decision_memo.md",
    "ceo_packet": "ceo_packet.md",
}

HARD_CONSTRAINTS = [
    "Do not modify v4 trading logic.",
    "Do not modify Protocol101, Protocol051, Protocol066, Protocol081, model artifacts, or scalers.",
    "Do not modify runtime flags.",
    "Do not modify launchd.",
    "Do not modify IBKR paper execution behavior.",
    "Do not call broker APIs.",
    "Do not download paid data.",
    "Do not train models.",
    "Do not tune thresholds.",
]


def repo_root_from_args(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path.cwd().resolve()


def normalize_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-_")
    if not slug:
        raise ValueError("slug must contain at least one letter or digit")
    return slug


def build_manifest(iteration_id: str, title: str, date: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "iteration_id": iteration_id,
        "title": title,
        "status": "draft",
        "created_date": date,
        "control_ref": "v4-protocol101-control-2026-05-24",
        "scope": "research_ops_governance",
        "hard_constraints": HARD_CONSTRAINTS,
        "required_artifacts": {key: False for key in PACKET_TEMPLATES},
    }


def create_iteration(root: Path, slug: str, title: str, date: str, force: bool) -> Path:
    research_ops = root / "research_ops"
    templates_dir = research_ops / "templates"
    iterations_dir = research_ops / "iterations"
    iteration_id = f"{date.replace('-', '')}_{normalize_slug(slug)}"
    iteration_dir = iterations_dir / iteration_id
    reports_dir = iteration_dir / "reports"
    artifacts_dir = iteration_dir / "artifacts"

    if iteration_dir.exists() and not force:
        raise FileExistsError(f"iteration already exists: {iteration_dir}")

    reports_dir.mkdir(parents=True, exist_ok=force)
    artifacts_dir.mkdir(parents=True, exist_ok=force)

    manifest = build_manifest(iteration_id, title or slug, date)
    manifest_path = iteration_dir / "iteration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for artifact_id, filename in PACKET_TEMPLATES.items():
        src = templates_dir / filename
        dst = reports_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"missing template for {artifact_id}: {src}")
        if dst.exists() and not force:
            raise FileExistsError(f"artifact already exists: {dst}")
        shutil.copyfile(src, dst)

    return iteration_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="short iteration slug")
    parser.add_argument("--title", default="", help="human-readable title")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--root", default=None, help="repository root; defaults to cwd")
    parser.add_argument("--force", action="store_true", help="overwrite an existing draft packet")
    args = parser.parse_args()

    root = repo_root_from_args(args.root)
    iteration_dir = create_iteration(root, args.slug, args.title, args.date, args.force)
    print(iteration_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
