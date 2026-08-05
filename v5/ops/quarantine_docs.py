#!/usr/bin/env python3
"""Derive and apply a reviewed, reversible documentation quarantine batch.

Dry-run is the default.  ``--apply`` performs explicit, recoverable moves and
writes a manifest recording every original path, hash and reason.

Two selection modes exist, and both are deliberately narrow:

* **Retired tree.**  Every file under a tree whose purpose is fully discharged.
* **Stray planning file.**  A planning-shaped Markdown filename inside a live
  tree that v5 does not cite as evidence and that no active code references.

Anything v5 links to is kept wherever v5 can still reach it, and
``v5/ops/check_project.py``'s broken-link scan is the proof after the move.
This supersedes the batch-specific ``quarantine_v4_docs.py``, whose own record
is its manifest under ``_cleanup_quarantine/2026-08-05-docs/``.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from v5.ops.check_project import ROOT, _is_planning_name, v5_link_targets


BATCH = "2026-08-05b-docs"
QUARANTINE = ROOT / "_cleanup_quarantine" / BATCH
MANIFEST = QUARANTINE / "MANIFEST.md"

# Trees whose purpose is fully discharged.  `docs/` is replaced by a pointer
# README after the move; the other two go entirely.
RETIRED_TREES = ("docs", "research_ops", "gpt-context-bundle")

# Live trees swept for stray planning files.  v2/, v3/, archive*/ and the v4
# evidence trees are not swept: nothing writes to them, and their contents are
# protected history under v5/AGENTS.md 7.
SWEPT_TREES = ("v4/docs",)

# Frozen audit inputs are the evidence of what an audit read.  Never move them.
FROZEN_AUDIT_INPUTS = ROOT / "v4/docs/protocol101/training/audits/frozen_inputs"

GENERIC_NAMES = {"README.md", "report.md", "summary.md", "index.md", "MANIFEST.md"}
DOCS_POINTER = """# Documentation moved to v5

There is no repository documentation drawer. The active project is [`v5/`](../v5/README.md).

- Current state, jobs, and gates: [`v5/STATUS.md`](../v5/STATUS.md)
- Working rules: [`v5/AGENTS.md`](../v5/AGENTS.md)
- Plans and briefs: `v5/work/<registered-job>/`, registered in `v5/STATUS.md` first

Earlier contents of this directory were quarantined, not deleted. See
[`_cleanup_quarantine/2026-08-05b-docs/MANIFEST.md`](../_cleanup_quarantine/2026-08-05b-docs/MANIFEST.md).
"""


@dataclass(frozen=True)
class Selection:
    path: Path
    reason: str
    tracked: bool
    size: int
    sha256: str


def _run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True)


def _tracked_paths() -> set[str]:
    return set(_run("git", "ls-files").splitlines())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_code_text() -> str:
    """Python and non-audit JSON under v4 and v5, for reference checks."""

    chunks: list[str] = []
    for tree in ("v4", "v5"):
        for path in (ROOT / tree).rglob("*"):
            if not path.is_file() or "audit" in path.parts or "__pycache__" in path.parts:
                continue
            if path.suffix in {".py", ".json", ".sh"}:
                chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks)


def derive() -> tuple[list[Selection], list[Path]]:
    """Return the selected files and the broken symlinks to remove."""

    cited = {path for path in v5_link_targets() if path.exists()}
    cited_names = {path.name for path in cited} - GENERIC_NAMES
    code_text = _active_code_text()
    reasons: dict[Path, str] = {}
    broken_symlinks: list[Path] = []

    for tree in RETIRED_TREES:
        root = ROOT / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == ".DS_Store":
                continue
            # The pointer this tool writes into docs/ is the replacement, not a
            # candidate; without this the batch would re-sweep its own output.
            if path == ROOT / "docs" / "README.md":
                continue
            reasons[path] = f"retired tree {tree}/"

    for tree in SWEPT_TREES:
        for path in sorted((ROOT / tree).rglob("*.md")):
            if path.is_symlink() and not path.resolve(strict=False).exists():
                broken_symlinks.append(path)
                continue
            if not path.is_file() or path.is_relative_to(FROZEN_AUDIT_INPUTS):
                continue
            resolved = path.resolve(strict=False)
            if resolved in cited or path.name in cited_names:
                continue
            if path.name not in GENERIC_NAMES and path.name in code_text:
                continue
            if _is_planning_name(path.stem):
                reasons[path] = "stray planning file"
            elif _is_numbered_orphan(path):
                reasons[path] = "numbered copy with no base file"

    tracked = _tracked_paths()
    selections = [
        Selection(
            path=path,
            reason=reason,
            tracked=path.relative_to(ROOT).as_posix() in tracked,
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path, reason in sorted(reasons.items())
    ]
    return selections, broken_symlinks


def _is_numbered_orphan(path: Path) -> bool:
    from v5.ops.check_project import NUMBERED_COPY

    match = NUMBERED_COPY.match(path.name)
    if not match:
        return False
    return not path.with_name(match.group(1) + match.group(3)).exists()


def _assert_safe(selections: list[Selection]) -> None:
    """Refuse to move protected evidence, edited files, or live dependencies.

    The stray-planning rule already skips code-referenced files, but the
    retired-tree rule sweeps whole directories and originally bypassed that
    check. Retiring the root documentation drawer broke the section-readiness
    check in v4/foundation, which required two of those files by path. The
    guard below is repo-wide so no selection mode can repeat that.
    """

    for item in selections:
        rel = item.path.relative_to(ROOT).as_posix()
        for protected in ("v4/audit", "v4/artifacts", "v4/logs", "v4/runtime"):
            if rel.startswith(protected + "/"):
                raise RuntimeError(f"protected evidence candidate: {rel}")
        if item.path.is_relative_to(FROZEN_AUDIT_INPUTS):
            raise RuntimeError(f"frozen audit input candidate: {rel}")
        if not item.tracked:
            continue  # untracked files are expected to show as '??'
        status = _run("git", "status", "--porcelain=v1", "--", rel).strip()
        if status:
            raise RuntimeError(f"tracked candidate has local edits: {rel}: {status}")

    # Match the repo-relative path, not the bare filename: a real dependency is
    # a path constant such as "docs/NAME.md", whereas prose in a comment names
    # the file alone. Selection in derive() stays basename-based because keeping
    # an extra file is harmless and moving a needed one is not.
    code_text = _active_code_text()
    dangling = sorted(
        {
            item.path.relative_to(ROOT).as_posix()
            for item in selections
            if item.path.relative_to(ROOT).as_posix() in code_text
        }
    )
    if dangling:
        raise RuntimeError(
            "these candidates are still named by active code; repoint the reference first:\n  "
            + "\n  ".join(dangling)
        )


def _manifest(selections: list[Selection], broken_symlinks: list[Path]) -> str:
    review = date(2026, 8, 5) + timedelta(days=90)
    rows = [
        f"| `{item.path.relative_to(ROOT).as_posix()}` | "
        f"`{(QUARANTINE / item.path.relative_to(ROOT)).relative_to(ROOT).as_posix()}` | "
        f"{'tracked' if item.tracked else 'untracked'} | {item.reason} | {item.size} | `{item.sha256}` |"
        for item in selections
    ]
    removed = [
        f"- `{path.relative_to(ROOT).as_posix()}` -> `{path.readlink()}`"
        for path in broken_symlinks
    ]
    return "\n".join(
        [
            f"# Documentation Quarantine Manifest — {BATCH}",
            "",
            "Nothing in this batch was deleted. Original paths are preserved below the quarantine root.",
            f"The batch moved {len(selections)} files and removed {len(broken_symlinks)} broken symlink(s).",
            "",
            "## Selection rules",
            "",
            "A file was selected when it sat inside a retired tree (`docs/`, `research_ops/`,",
            "`gpt-context-bundle/`), or when it was a planning-shaped Markdown filename inside `v4/docs/`",
            "that v5 does not cite as evidence and that no active Python, JSON or shell file references.",
            "A numbered copy with no base file was selected as a sync artefact. Files cited by v5, files",
            "named by active code, `v4/audit`, `v4/artifacts`, `v4/logs`, `v4/runtime`, and the frozen audit",
            "inputs under `v4/docs/protocol101/training/audits/frozen_inputs/` were all excluded.",
            "",
            "The trees `v2/`, `v3/`, `archive/` and `archive_quarantine/` were **not** swept. They are closed",
            "eras and protected history; their stray planning files are recorded as deferred cleanup in",
            "`v5/STATUS.md` rather than moved here.",
            "",
            "## Rollback",
            "",
            "For the whole batch, revert the cleanup commit. For one file, move its quarantine path back to",
            "the original path with `git mv` (or plain `mv` for an untracked row), then rerun",
            "`v5/ops/check_project.py` and the v5 tests.",
            "",
            f"Deletion may be considered after **{review.isoformat()}**, but only with a new owner decision.",
            "",
            "## Removed broken symlinks",
            "",
            *(removed or ["None."]),
            "",
            "## Files",
            "",
            "| Original path | Quarantine path | Git state | Reason | Bytes | SHA-256 |",
            "|---|---|---|---|---:|---|",
            *rows,
            "",
        ]
    )


def apply_batch(selections: list[Selection], broken_symlinks: list[Path]) -> None:
    existing = [path for path in QUARANTINE.rglob("*") if path.is_file()]
    if existing:
        raise RuntimeError(f"quarantine batch already populated: {QUARANTINE}")
    _assert_safe(selections)
    moved: list[tuple[Path, Path, bool]] = []
    try:
        for item in selections:
            rel = item.path.relative_to(ROOT)
            target = QUARANTINE / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.tracked:
                subprocess.check_call(
                    ["git", "mv", str(rel), str(target.relative_to(ROOT))], cwd=ROOT
                )
            else:
                item.path.rename(target)
            moved.append((item.path, target, item.tracked))
        MANIFEST.write_text(_manifest(selections, broken_symlinks), encoding="utf-8")
        _write_docs_pointer()
        # Symlink removal is last and irreversible-by-rollback, so it runs only
        # once every move has succeeded and the manifest is on disk.
        for path in broken_symlinks:
            path.unlink()
        _prune_empty_dirs()
    except Exception:
        for original, target, tracked in reversed(moved):
            original.parent.mkdir(parents=True, exist_ok=True)
            if tracked:
                subprocess.call(
                    ["git", "mv", str(target.relative_to(ROOT)), str(original.relative_to(ROOT))],
                    cwd=ROOT,
                )
            elif target.exists():
                target.rename(original)
        _remove_empty_quarantine_dirs()
        raise


def _remove_empty_quarantine_dirs() -> None:
    """Leave no empty directory shell behind after a rolled-back attempt."""

    if not QUARANTINE.exists():
        return
    for path in sorted(QUARANTINE.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if not any(QUARANTINE.iterdir()):
        QUARANTINE.rmdir()


def _write_docs_pointer() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "README.md").write_text(DOCS_POINTER, encoding="utf-8")


def _prune_empty_dirs() -> None:
    """Remove directories emptied by the move, keeping `docs/` for its pointer."""

    for tree in RETIRED_TREES:
        root = ROOT / tree
        if not root.is_dir() or tree == "docs":
            continue
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file() and path.name == ".DS_Store":
                path.unlink()
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        if not any(root.iterdir()):
            root.rmdir()
    for path in sorted((ROOT / "docs").rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="perform the reviewed reversible moves")
    args = parser.parse_args()
    selections, broken_symlinks = derive()
    print(f"batch={BATCH} selected={len(selections)} broken_symlinks={len(broken_symlinks)}")
    for item in selections:
        print(f"{item.path.relative_to(ROOT)}\t{item.reason}\t{item.size}")
    for path in broken_symlinks:
        print(f"{path.relative_to(ROOT)}\tbroken symlink (remove)\t0")
    if args.apply:
        apply_batch(selections, broken_symlinks)
        print(f"manifest={MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
