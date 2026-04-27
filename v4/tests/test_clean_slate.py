"""Verify v4 has no imports from v2 or v3.

Clean-slate is a hard rule from the research protocol. This test runs in CI to
guarantee no shared imports leak in over time.
"""
from __future__ import annotations

import re
from pathlib import Path

V4_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"^\s*(from|import)\s+(v2|v3)(\.|$|\s)", re.MULTILINE)


def test_no_v2_v3_imports() -> None:
    offenders: list[tuple[Path, int, str]] = []
    for path in V4_ROOT.rglob("*.py"):
        if "tests" in path.parts and path.name == "test_clean_slate.py":
            continue
        text = path.read_text()
        for match in FORBIDDEN.finditer(text):
            line_num = text[: match.start()].count("\n") + 1
            offenders.append((path.relative_to(V4_ROOT), line_num, match.group(0).strip()))
    assert not offenders, (
        "v4 must not import from v2 or v3. Found:\n"
        + "\n".join(f"  {p}:{ln}: {stmt}" for p, ln, stmt in offenders)
    )
