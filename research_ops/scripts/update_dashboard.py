#!/usr/bin/env python3
"""Update the generated current-state block in CEO_DASHBOARD.md."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


START = "<!-- research_ops:update_dashboard:start -->"
END = "<!-- research_ops:update_dashboard:end -->"


def read_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    current_parent = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" ") and line.endswith(":"):
            current_parent = line[:-1]
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"')
        full_key = f"{current_parent}.{key}" if raw_line.startswith(" ") and current_parent else key
        values[full_key] = value
    return values


def build_block(state: dict[str, str]) -> str:
    operational_default = state.get("current_default.name", state.get("control.operational_default", "unknown"))
    control_tag = state.get("control.tag", "unknown")
    next_phase = state.get("audit_interpretation.next_phase", state.get("stage", "unknown"))
    return "\n".join(
        [
            START,
            "## Generated Current State",
            "",
            f"- Stage: `{state.get('stage', 'unknown')}`",
            f"- Audit next phase: `{next_phase}`",
            f"- Control tag: `{control_tag}`",
            f"- Operational default: `{operational_default}`",
            f"- Next recommended prompt: `{state.get('next_recommended_prompt', 'unknown')}`",
            END,
        ]
    )


def replace_block(text: str, block: str) -> str:
    pattern = re.compile(rf"{re.escape(START)}.*?{re.escape(END)}", re.DOTALL)
    if pattern.search(text):
        return pattern.sub(block, text)
    marker = "\n## Dashboard Rule\n"
    if marker in text:
        return text.replace(marker, "\n" + block + "\n" + marker, 1)
    return text.rstrip() + "\n\n" + block + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="repository root; defaults to cwd")
    parser.add_argument("--dry-run", action="store_true", help="print updated dashboard instead of writing")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else Path.cwd().resolve()
    research_ops = root / "research_ops"
    dashboard_path = research_ops / "CEO_DASHBOARD.md"
    state_path = research_ops / "CURRENT_STATE.yaml"

    state = read_simple_yaml(state_path)
    block = build_block(state)
    updated = replace_block(dashboard_path.read_text(encoding="utf-8"), block)

    if args.dry_run:
        print(updated)
    else:
        dashboard_path.write_text(updated, encoding="utf-8")
        print(dashboard_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
