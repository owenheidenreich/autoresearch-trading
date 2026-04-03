"""Verify that v2 code does not import from v1 or legacy_v1.

v2 modules must be self-contained. They must not depend on:
- legacy_v1/ (frozen archive)
- training/ or training.live/ (v1 active code)
- tools/ or infra/ (v1 operational code)
- Bare module names that resolve to v1 (prepare, replay, train, run_loop, trading_rules)

This test scans all .py files under v2/ and fails if any forbidden import is found.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = REPO_ROOT / "v2"

# Patterns that indicate a v1 import
FORBIDDEN_PATTERNS = [
    # legacy_v1 archive
    r"\bfrom\s+legacy_v1\b",
    r"\bimport\s+legacy_v1\b",
    # v1 training package
    r"\bfrom\s+training\b",
    r"\bimport\s+training\b",
    # v1 tools/infra
    r"\bfrom\s+tools\b",
    r"\bimport\s+tools\b",
    r"\bfrom\s+infra\b",
    r"\bimport\s+infra\b",
    # Bare v1 module names (these resolve to training/ via sys.path)
    r"^\s*from\s+prepare\s+import\b",
    r"^\s*import\s+prepare\b",
    r"^\s*from\s+replay\s+import\b",
    r"^\s*import\s+replay\b",
    r"^\s*from\s+train\s+import\b",
    r"^\s*import\s+train\b",
    r"^\s*from\s+run_loop\s+import\b",
    r"^\s*import\s+run_loop\b",
    r"^\s*from\s+trading_rules\s+import\b",
    r"^\s*import\s+trading_rules\b",
    r"^\s*from\s+best_train\s+import\b",
    r"^\s*import\s+best_train\b",
]

COMPILED_PATTERNS = [re.compile(p) for p in FORBIDDEN_PATTERNS]


def _scan_file(filepath: Path) -> list[str]:
    """Return list of violation descriptions for a single file."""
    violations = []
    try:
        text = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return violations

    for line_num, line in enumerate(text.splitlines(), start=1):
        # Skip comments
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for pattern in COMPILED_PATTERNS:
            if pattern.search(line):
                rel = filepath.relative_to(REPO_ROOT)
                violations.append(f"{rel}:{line_num}: {line.strip()}")
                break  # one violation per line is enough
    return violations


def test_no_v1_imports_in_v2():
    """Scan all v2/ .py files for forbidden v1 imports."""
    if not V2_DIR.is_dir():
        return  # v2/ doesn't exist yet, nothing to check

    violations = []
    for root, _, files in os.walk(V2_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = Path(root) / fname
            violations.extend(_scan_file(fpath))

    assert not violations, (
        f"v2 code must not import from v1. Found {len(violations)} violation(s):\n"
        + "\n".join(f"  {v}" for v in violations)
    )
