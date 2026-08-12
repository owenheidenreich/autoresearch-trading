#!/usr/bin/env python3
"""Read-only structural and navigation checks for the repository.

Two scopes are enforced.  The v5 scope keeps the active workspace coherent.
The repository scope stops new planning files from accumulating in junk
drawers outside v5, which is how the previous sprawl happened.

Enumeration is deliberately filesystem-based rather than git-based: the
``docs/``, ``research_ops/`` and ``gpt-context-bundle/`` drawers survived every
earlier cleanup precisely because most of their files were never tracked.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import Sequence


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
    V5 / "research/findings/V5_WORKFLOW_CAPABILITY_AUDIT_2026_08_05.md",
    V5 / "research/training_twin.py",
    V5 / "research/feature_admission.py",
    V5 / "research/validation/candidate_packet.py",
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
NUMBERED_COPY = re.compile(r"^(.*) (\d+)(\.[^.]+)$")
BANNED_NAME = re.compile(r"SOURCE_OF_TRUTH|ROADMAP|HANDOFF|CURRENT_STATUS", re.IGNORECASE)

# --- repository scope -------------------------------------------------------

# Directories that are never walked: caches, virtualenvs, and version control.
PRUNED_DIRS = frozenset(
    {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}
)

# Protected evidence and already-reviewed quarantine batches.  Run receipts,
# audit reports and quarantined files are not documentation and must never be
# moved to satisfy a naming rule (v5/AGENTS.md 7).
EVIDENCE_TREES = (
    "_cleanup_quarantine",
    "archive",
    "archive_quarantine",
    "_history_backup",
    "v4/audit",
    "v4/artifacts",
    "v4/logs",
    "v4/runtime",
    "v4/promotion",
    "v4/ledger",
    "v2/artifacts",
    "v3/artifacts",
)

# Closed eras, frozen 2026-04-27 and earlier.  Their stray planning files are
# inert because nothing writes to these trees anymore.  They are reported as a
# deferred cleanup in STATUS rather than quarantined, so the exemption is named
# here explicitly instead of hidden inside a broad evidence exemption.
CLOSED_ERA_TREES = ("v2", "v3")

# A planning-shaped filename, matched on whole underscore-separated words so
# that "MODEL_PLANE_CONTRACT" does not read as a plan.
PLANNING_TOKENS = (
    "_PLAN_",
    "_PLANS_",
    "_HANDOFF_",
    "_ROADMAP_",
    "_SOURCE_OF_TRUTH_",
    "_CURRENT_STATUS_",
    "_NEXT_STEPS_",
    "_AUTHORITY_",
    "_STATUS_",
)

# The only two status filenames outside v5/work that are allowed to exist: the
# real status page and the root bootstrap pointer that redirects to it.
STATUS_EXEMPT = (ROOT / "STATUS.md", V5 / "STATUS.md")

# Frozen audit inputs.  This directory contains a captured copy of a v4-era
# AGENTS.md and of CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md because an
# audit was run against them.  They are the evidence of what that audit read
# and must not move, so the naming rules do not apply to them.
FROZEN_AUDIT_INPUTS = "v4/docs/protocol101/training/audits/frozen_inputs"

# The root documentation drawer is retired.  Only a pointer may live there.
ROOT_DOCS_ALLOWED = frozenset({"README.md"})

# Trees whose purpose is fully discharged.  They are quarantined, not deleted,
# so this rule is what stops them being recreated file by file.
RETIRED_TREES = ("research_ops", "gpt-context-bundle")
REQUIRED_TOOLBOX_MARKERS = (
    "research/training_twin.py",
    "research/feature_admission.py",
    "research/validation/candidate_packet.py",
    "v4/path_d/",
    "v4/research/autoresearch_v2/",
    "v4/scripts/export_protocol101_trade_charts.py",
)


def _link_target(source: Path, raw: str) -> Path | None:
    target = raw.strip().strip("<>")
    if not target or target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = target.split("#", 1)[0]
    if not target:
        return None
    return Path(target) if Path(target).is_absolute() else source.parent / target


def _v4_imports(source: str) -> list[str]:
    """Return v4 imports using the Python grammar rather than text matching."""

    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names if alias.name == "v4" or alias.name.startswith("v4."))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "v4" or module.startswith("v4."):
                found.append(module)
    return found


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

    toolbox = V5 / "TOOLBOX.md"
    toolbox_text = toolbox.read_text(errors="ignore") if toolbox.exists() else ""
    for marker in REQUIRED_TOOLBOX_MARKERS:
        if marker not in toolbox_text:
            problems.append(f"TOOLBOX GAP       missing classified path: {marker}")

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
        # `rglob` descends into caches that the repository scope already prunes.
        # Without this the check flagged iCloud-duplicated .pyc files as sync
        # duplicates, so `test_v5_project_structure_is_clean` passed or failed
        # on whether iCloud happened to have synced -- a guard that fails on
        # timing rather than on project health teaches its reader to ignore it.
        if PRUNED_DIRS.intersection(path.parts):
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
        if path.suffix == ".py":
            source = path.read_text(errors="ignore")
            try:
                imports = _v4_imports(source)
            except SyntaxError as exc:
                problems.append(f"PYTHON SYNTAX     {rel}:{exc.lineno}:{exc.msg}")
            else:
                if imports:
                    problems.append(
                        f"V4 IMPORT         {rel} imports frozen v4 code: {','.join(imports)}"
                    )

    for target, (source, raw) in sorted(
        v5_link_targets().items(), key=lambda item: (item[1][0], item[1][1])
    ):
        if not target.exists():
            problems.append(f"BROKEN LINK       {source.relative_to(ROOT)} -> {raw}")

    return problems


def v5_link_targets() -> dict[Path, tuple[Path, str]]:
    """Resolved targets of every markdown link inside v5.

    Serves two rules: a target that does not exist is a broken link, and a
    target that does exist is evidence v5 depends on, which exempts it from the
    repository naming rules.
    """

    targets: dict[Path, tuple[Path, str]] = {}
    for source in sorted(V5.rglob("*.md")):
        text = source.read_text(errors="ignore")
        for raw in LINK.findall(text):
            target = _link_target(source, raw)
            if target is None:
                continue
            targets.setdefault(target.resolve(strict=False), (source, raw))
    return targets


def _in_tree(rel: str, trees: Sequence[str]) -> bool:
    return any(rel == tree or rel.startswith(tree + "/") for tree in trees)


def _is_planning_name(stem: str) -> bool:
    padded = "_" + re.sub(r"[^A-Za-z0-9]+", "_", stem).upper().strip("_") + "_"
    return any(token in padded for token in PLANNING_TOKENS)


def active_code_text() -> str:
    """Python, JSON and shell under v4 and v5, excluding generated audit output.

    A planning-shaped document that active code names by filename is a live
    dependency: moving it would break the code, so it is exempt from the
    naming rule for as long as that reference exists.
    """

    chunks: list[str] = []
    for tree in ("v4", "v5"):
        for dirpath, dirnames, filenames in os.walk(ROOT / tree):
            dirnames[:] = [
                name for name in dirnames if name not in PRUNED_DIRS and name != "audit"
            ]
            for name in filenames:
                if name.endswith((".py", ".json", ".sh")):
                    chunks.append(Path(dirpath, name).read_text(errors="ignore"))
    return "\n".join(chunks)


def _markdown_paths() -> list[Path]:
    """Every markdown file in the repository, caches and virtualenvs pruned."""

    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(name for name in dirnames if name not in PRUNED_DIRS)
        for name in sorted(filenames):
            if name.endswith(".md"):
                found.append(Path(dirpath) / name)
    return found


def scan_repository() -> list[str]:
    """Rules that apply outside v5, so new sprawl cannot start somewhere else."""

    problems: list[str] = []
    evidence = set(v5_link_targets())
    exempt_status = {path.resolve(strict=False) for path in STATUS_EXEMPT}
    code_text = active_code_text()
    generic = {"README.md", "report.md", "summary.md", "index.md", "MANIFEST.md"}

    for path in _markdown_paths():
        rel = path.relative_to(ROOT).as_posix()
        resolved = path.resolve(strict=False)
        if path.is_symlink() and not resolved.exists():
            problems.append(f"BROKEN SYMLINK    {rel}")
            continue
        if _in_tree(rel, EVIDENCE_TREES) or rel.startswith(FROZEN_AUDIT_INPUTS + "/"):
            continue
        if _in_tree(rel, CLOSED_ERA_TREES):
            continue
        if path.is_relative_to(V5 / "work") or resolved in exempt_status:
            continue
        if resolved in evidence:
            continue
        if path.name not in generic and path.name in code_text:
            continue
        if _is_planning_name(path.stem):
            problems.append(
                f"STRAY PLANNING    {rel} belongs in v5/work/<registered-job>/ or quarantine"
            )

    docs = ROOT / "docs"
    if docs.is_dir():
        for path in sorted(docs.glob("*")):
            if path.name not in ROOT_DOCS_ALLOWED:
                problems.append(
                    f"RETIRED DRAWER    {path.relative_to(ROOT).as_posix()}; docs/ may hold only a pointer README"
                )

    for tree in RETIRED_TREES:
        if (ROOT / tree).exists():
            problems.append(f"RETIRED TREE      {tree}/ is quarantined and must not be recreated")

    # Numbered copies are macOS/iCloud sync artefacts.  Evidence and closed-era
    # trees are exempt for the same reason as the naming rules: a duplicate
    # inside protected evidence is reported to the owner, never silently moved.
    for path in _markdown_paths():
        rel = path.relative_to(ROOT).as_posix()
        if _in_tree(rel, EVIDENCE_TREES) or _in_tree(rel, CLOSED_ERA_TREES):
            continue
        numbered = NUMBERED_COPY.match(path.name)
        if not numbered:
            continue
        base = numbered.group(1) + numbered.group(3)
        if path.with_name(base).exists():
            problems.append(f"SYNC DUPLICATE    {rel} shadows {base}")
        else:
            problems.append(f"ORPHAN COPY       {rel} is a numbered copy with no base file")

    v4_readme = ROOT / "v4" / "README.md"
    if v4_readme.is_file() and "SINGLE_SOURCE_OF_TRUTH" in v4_readme.read_text(errors="ignore"):
        problems.append("BAD FRONT POINTER v4/README.md still names a non-v5 document as current truth")

    return problems


def main() -> int:
    failed = False
    for label, problems in (("v5 project", scan()), ("repository", scan_repository())):
        if problems:
            failed = True
            print(f"{len(problems)} {label} problem(s):")
            for problem in problems:
                print("  " + problem)
    if failed:
        return 1
    print("v5 project OK - one front door, registered work, valid links, no v4 imports")
    print("repository OK - no stray planning files, retired drawers, or duplicate copies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
