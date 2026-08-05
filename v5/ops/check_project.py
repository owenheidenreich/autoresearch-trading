#!/usr/bin/env python3
"""Read-only structural and navigation checks for the active v5 workspace."""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "v5"
STATUS = V5 / "STATUS.md"

REQUIRED = (
    V5 / "README.md",
    STATUS,
    V5 / "AGENTS.md",
    V5 / "CLAUDE.md",
    V5 / "TOOLBOX.md",
    V5 / "evidence/INDEX.md",
    V5 / "research/history/DO_NOT_RETEST.md",
    V5 / "work/measurement-review/BRIEF.md",
    V5 / "work/g1-direction/PLAN.md",
)
ROOT_POINTERS = {
    ROOT / "README.md": "v5/README.md",
    ROOT / "STATUS.md": "v5/STATUS.md",
    ROOT / "AGENTS.md": "v5/AGENTS.md",
    ROOT / "CLAUDE.md": "v5/AGENTS.md",
}
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
V4_IMPORT = re.compile(r"^\s*(?:from\s+v4(?:\.|\s)|import\s+v4(?:\.|\s|$))", re.MULTILINE)
NUMBERED_COPY = re.compile(r"^(.*) (\d+)(\.[^.]+)$")
BANNED_NAME = re.compile(r"SOURCE_OF_TRUTH|ROADMAP|HANDOFF|CURRENT_STATUS", re.IGNORECASE)


def _link_target(source: Path, raw: str) -> Path | None:
    target = raw.strip().strip("<>")
    if not target or target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = target.split("#", 1)[0]
    if not target:
        return None
    return Path(target) if Path(target).is_absolute() else source.parent / target


def scan() -> list[str]:
    problems: list[str] = []

    if (V5 / "docs").exists():
        problems.append("BANNED DIRECTORY  v5/docs exists; use a typed v5 directory")

    for path in REQUIRED:
        if not path.exists():
            problems.append(f"MISSING CORE      {path.relative_to(ROOT)}")

    if STATUS.exists():
        status_text = STATUS.read_text(errors="ignore")
    else:
        status_text = ""

    for path, marker in ROOT_POINTERS.items():
        if not path.is_file() or marker not in path.read_text(errors="ignore"):
            problems.append(f"BAD ROOT POINTER  {path.name} must point to {marker}")

    agents = V5 / "AGENTS.md"
    claude = V5 / "CLAUDE.md"
    if agents.exists() and claude.exists() and agents.read_bytes() != claude.read_bytes():
        problems.append("AGENT DRIFT       v5/AGENTS.md and v5/CLAUDE.md differ")

    work_root = V5 / "work"
    if work_root.exists():
        for directory in sorted(path for path in work_root.iterdir() if path.is_dir()):
            marker = f"work/{directory.name}/"
            if marker not in status_text:
                problems.append(f"UNREGISTERED WORK {directory.relative_to(ROOT)}")

    for path in sorted(V5.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if BANNED_NAME.search(path.name):
            problems.append(f"BANNED NAME       {rel}")
        if path.name == "PLAN.md" and not path.is_relative_to(work_root):
            problems.append(f"MISPLACED PLAN    {rel}")
        numbered = NUMBERED_COPY.match(path.name)
        if numbered:
            base = path.with_name(numbered.group(1) + numbered.group(3))
            if base.exists():
                problems.append(f"SYNC DUPLICATE    {rel} shadows {base.name}")
        if path.suffix == ".py" and V4_IMPORT.search(path.read_text(errors="ignore")):
            problems.append(f"V4 IMPORT         {rel} imports frozen v4 code")

    for source in sorted(V5.rglob("*.md")):
        text = source.read_text(errors="ignore")
        for raw in LINK.findall(text):
            target = _link_target(source, raw)
            if target is not None and not target.resolve(strict=False).exists():
                problems.append(
                    f"BROKEN LINK       {source.relative_to(ROOT)} -> {raw}"
                )

    return problems


def main() -> int:
    problems = scan()
    if problems:
        print(f"{len(problems)} v5 project problem(s):")
        for problem in problems:
            print("  " + problem)
        return 1
    print("v5 project OK - one front door, registered work, valid links, no v4 imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
