#!/usr/bin/env python3
"""Derive and apply the reviewed 2026-08-05 v4 documentation quarantine.

Dry-run is the default. ``--apply`` performs only explicit, recoverable moves
after every candidate passes reference, scope, and Git-cleanliness checks.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = ROOT / "v4/docs"
QUARANTINE = ROOT / "_cleanup_quarantine/2026-08-05-docs"
MANIFEST = QUARANTINE / "MANIFEST.md"
GENERIC_NAMES = {"README.md", "report.md", "summary.md", "index.md"}
CLOSED_DIRS = (
    DOC_ROOT / "protocol101/training/history/closed_stage1_graph_and_goals_2026_07_28",
    DOC_ROOT / "protocol101/synchronization/history",
    DOC_ROOT / "protocol101/training/execution/CHATGPT_PRO_SCIENTIFIC_HANDOFF_2026_08_04",
)
CONTRACTS = DOC_ROOT / "protocol101/training/contracts"
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
NUMBERED = re.compile(r"^(.*) (\d+)(\.[^.]+)$")


@dataclass(frozen=True)
class Selection:
    path: Path
    reasons: tuple[str, ...]
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


def _code_and_config_text() -> str:
    chunks: list[str] = []
    for path in (ROOT / "v4").rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".py" or (path.suffix == ".json" and "audit" not in path.parts):
            chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks)


def _v5_text() -> str:
    return "\n".join(
        path.read_text(errors="ignore")
        for path in (ROOT / "v5").rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".json"}
    )


def _resolved_link(source: Path, raw: str) -> Path | None:
    target = raw.strip().strip("<>")
    if not target or target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = target.split("#", 1)[0]
    return (source.parent / target).resolve(strict=False) if target else None


def derive() -> tuple[list[Selection], int, int]:
    markdown = [path for path in DOC_ROOT.rglob("*.md") if path.is_file()]
    texts = {path: path.read_text(errors="ignore") for path in markdown}
    status_text = (ROOT / "v5/STATUS.md").read_text(errors="ignore")
    code_text = _code_and_config_text()
    v5_text = _v5_text()

    def keep(path: Path) -> bool:
        rel = path.relative_to(ROOT).as_posix()
        text = texts[path]
        numbered = NUMBERED.match(path.name)
        numbered_without_base = bool(
            numbered and not path.with_name(numbered.group(1) + numbered.group(3)).exists()
        )
        referenced_by_code = path.name not in GENERIC_NAMES and path.name in code_text
        referenced_by_v5 = rel in v5_text or (
            path.name not in GENERIC_NAMES and path.name in v5_text
        )
        return any(
            (
                referenced_by_code,
                referenced_by_v5,
                path.name in status_text,
                "STILL IN FORCE" in text,
                "Still authoritative" in text,
                path.is_relative_to(CONTRACTS),
                "2026_08" in path.name,
                numbered_without_base,
            )
        )

    reasons: dict[Path, set[str]] = {}
    for path in markdown:
        if keep(path):
            continue
        era = any(path.is_relative_to(directory) for directory in CLOSED_DIRS)
        orphan = not any(path.name in text for other, text in texts.items() if other != path)
        if era or orphan:
            reasons[path] = set()
            if era:
                reasons[path].add("closed era")
            if orphan:
                reasons[path].add("orphan")

    # A closed-era rule must never break a link from a surviving document.
    while True:
        protected: set[Path] = set()
        remaining = [path for path in markdown if path not in reasons]
        candidate_by_resolved = {path.resolve(): path for path in reasons}
        for source in remaining:
            for raw in LINK.findall(texts[source]):
                target = _resolved_link(source, raw)
                if target in candidate_by_resolved:
                    protected.add(candidate_by_resolved[target])
        if not protected:
            break
        for path in protected:
            reasons.pop(path, None)

    # Numbered copies are separate from the Markdown orphan rules. Generated
    # evidence under v4/audit is explicitly excluded.
    for path in (ROOT / "v4").rglob("*"):
        if not path.is_file() or "audit" in path.parts:
            continue
        match = NUMBERED.match(path.name)
        if not match:
            continue
        base = path.with_name(match.group(1) + match.group(3))
        if base.exists():
            reasons.setdefault(path, set()).add("numbered sync duplicate with base")

    tracked = _tracked_paths()
    selections = [
        Selection(
            path=path,
            reasons=tuple(sorted(selected_reasons)),
            tracked=path.relative_to(ROOT).as_posix() in tracked,
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path, selected_reasons in sorted(reasons.items())
    ]
    before = len(list(DOC_ROOT.rglob("*.md")))
    moved_markdown = sum(path.path.suffix == ".md" and path.path.is_relative_to(DOC_ROOT) for path in selections)
    return selections, before, before - moved_markdown


def _assert_clean(selections: list[Selection]) -> None:
    for item in selections:
        rel = item.path.relative_to(ROOT).as_posix()
        if item.path.is_relative_to(ROOT / "v4/audit"):
            raise RuntimeError(f"protected audit candidate: {rel}")
        status = _run("git", "status", "--porcelain=v1", "--", rel)
        if status.strip():
            raise RuntimeError(f"candidate is not clean: {rel}: {status.strip()}")


def _manifest(selections: list[Selection], before: int, after: int) -> str:
    review = date(2026, 8, 5) + timedelta(days=90)
    rows = []
    for item in selections:
        rel = item.path.relative_to(ROOT).as_posix()
        target = (QUARANTINE / rel).relative_to(ROOT).as_posix()
        rows.append(
            f"| `{rel}` | `{target}` | {'tracked' if item.tracked else 'untracked'} | "
            f"{', '.join(item.reasons)} | {item.size} | `{item.sha256}` |"
        )
    return "\n".join(
        [
            "# V4 Documentation Quarantine Manifest — 2026-08-05",
            "",
            "Nothing in this batch was deleted. Paths are preserved below the quarantine root.",
            f"The batch moved {len(selections)} files and reduced v4/docs Markdown paths from {before} to {after}.",
            "",
            "## Selection rules",
            "",
            "A document was kept when code/non-audit config or v5 referenced it, STATUS named it, its banner",
            "said STILL IN FORCE or Still authoritative, it was a signed training contract, or its name was",
            "from the 2026-08 current era. Otherwise a document was selected when it was in a declared closed",
            "era or no other v4 Markdown file referenced its filename. A resolved link from a surviving document",
            "overrode selection. Numbered copies were selected only when the real base existed; v4/audit and",
            "numbered files without a base were excluded.",
            "",
            "## Rollback",
            "",
            "For the whole batch, revert the cleanup commit. For one file, move its quarantine path back to the",
            "original path with `git mv` (or plain `mv` for an untracked row), then rerun the v5 checker and tests.",
            "",
            f"Deletion may be considered after **{review.isoformat()}**, but only with a new owner decision.",
            "",
            "## Files",
            "",
            "| Original path | Quarantine path | Git state | Reason | Bytes | SHA-256 |",
            "|---|---|---|---|---:|---|",
            *rows,
            "",
        ]
    )


def apply_batch(selections: list[Selection], before: int, after: int) -> None:
    if MANIFEST.exists() or QUARANTINE.exists():
        raise RuntimeError(f"quarantine target already exists: {QUARANTINE}")
    _assert_clean(selections)
    moved: list[tuple[Path, Path, bool]] = []
    try:
        for item in selections:
            rel = item.path.relative_to(ROOT)
            target = QUARANTINE / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.tracked:
                subprocess.check_call(["git", "mv", str(rel), str(target.relative_to(ROOT))], cwd=ROOT)
            else:
                item.path.rename(target)
            moved.append((item.path, target, item.tracked))
        MANIFEST.write_text(_manifest(selections, before, after), encoding="utf-8")
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
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="perform the reviewed reversible moves")
    args = parser.parse_args()
    selections, before, after = derive()
    print(f"selected={len(selections)} v4_docs_markdown_before={before} after={after}")
    for item in selections:
        print(f"{item.path.relative_to(ROOT)}\t{','.join(item.reasons)}\t{item.size}")
    if args.apply:
        apply_batch(selections, before, after)
        print(f"manifest={MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
