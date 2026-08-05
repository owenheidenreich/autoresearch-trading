#!/usr/bin/env python3
"""Documentation rot check.

Answers the question "how do we know the files STATUS.md points at are still legit?"
without anyone having to remember to ask it.

Run from the repo root:

    ./.venv/bin/python v4/ops/check_docs.py

Exit code 0 = clean, 1 = problems found. Reads only; changes nothing.

Four checks, each one a mistake that actually happened here:

1. BROKEN LINK      - STATUS.md points at a file that does not exist.
2. UNMARKED TRUTH   - a document claims to be status/truth/roadmap/gates but carries
                      no pointer to STATUS.md. Five of these were found on 2026-08-05,
                      including one literally named CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.
3. UNREGISTERED DOC - a handoff/review/brief exists that no register row mentions, so the
                      work it describes is invisible. Three of these existed on 2026-08-05.
4. STALE MEASUREMENT- STATUS.md still cites a diagnosis or number that a later document retracted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "STATUS.md"

# A doc that says one of these is claiming authority it may no longer have.
TRUTH_CLAIMS = re.compile(
    r"source of truth|current state|current status|road ?map|hill-climb gates|"
    r"single source|program status|current bootstrap",
    re.I,
)
# Documents that create work and therefore need a register row.
WORK_DOC = re.compile(r"HANDOFF|REVIEW|BRIEF|PREREGISTRATION|RESEARCH_|_PLAN", re.I)
# Retracted claims: (pattern that must NOT appear in STATUS.md, why).
RETRACTED = [
    (re.compile(r"file-sync fault|iCloud eviction|evicted to dataless", re.I),
     "the iCloud-eviction diagnosis was retracted 2026-08-05; the cause is macOS TCC permissions"),
    (re.compile(r"holdout is unspent|unspent holdout", re.I),
     "the protected holdout is SPENT (opened once, 2026-08-02)"),
]

# Generated evidence and frozen history are not authority claims. An audit report saying
# "current state" describes the run that produced it, which is exactly what it should do.
SKIP = ("/.git/", "/archive", "/_cleanup_quarantine/", "/v2/", "/v3/",
        "/node_modules/", "/.venv/",
        "/v4/audit/",        # generated run evidence
        "/history/",         # frozen, already-closed eras
        "/frozen_inputs/",   # deliberate point-in-time copies
        "/research_ops/iterations/",
        "_HANDOFF_2026_08_04/",  # bundled copies shipped to an external reviewer
        )


def scan() -> list[str]:
    problems: list[str] = []
    status_text = STATUS.read_text()

    # 1. broken links out of STATUS.md
    for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", status_text):
        if target.startswith("http") or target.startswith("#"):
            continue
        if not (ROOT / target).exists():
            problems.append(f"BROKEN LINK       STATUS.md -> {target}  ({label})")

    # 4. retracted claims still present in STATUS.md
    for pattern, why in RETRACTED:
        if pattern.search(status_text):
            problems.append(f"STALE CLAIM       STATUS.md contains '{pattern.pattern}' - {why}")

    register = status_text.split("## 1.")[0]  # section 0 only

    for md in sorted(ROOT.rglob("*.md")):
        rel = md.relative_to(ROOT).as_posix()
        if any(s.strip("/") in rel for s in (s.strip("/") for s in SKIP)):
            continue
        if rel in ("STATUS.md", "CLAUDE.md", "AGENTS.md"):
            continue
        try:
            head = md.read_text(errors="ignore")[:1500]
        except OSError:
            continue

        # 2. claims authority without pointing at STATUS.md
        if TRUTH_CLAIMS.search(head) and "STATUS.md" not in head:
            problems.append(f"UNMARKED TRUTH    {rel}  (claims authority, no STATUS.md pointer)")

        # 3. creates work but is in no register row
        if WORK_DOC.search(md.name) and md.name not in register:
            if "2026-08" in md.name:  # only current-era docs; history is not a live job
                problems.append(f"UNREGISTERED DOC  {rel}  (no row in STATUS.md section 0)")

        # 5. Finder/iCloud sync duplicate sitting next to the real file. One of these
        #    ("README 2.md", 2026-07-28) called itself the CANONICAL TRAINING FRONT PAGE
        #    while the real README.md beside it was three days newer.
        if md.stem.endswith(" 2"):
            real = md.with_name(md.stem[:-2] + md.suffix)
            if real.exists() and "DO NOT READ" not in head:
                problems.append(f"SYNC DUPLICATE    {rel}  (shadows {real.name}; banner it or remove it)")

    return problems


def main() -> int:
    problems = scan()
    if not problems:
        print("docs OK - links resolve, no unmarked truth files, register covers current work")
        return 0
    print(f"{len(problems)} documentation problem(s):\n")
    for p in problems:
        print("  " + p)
    print("\nFix by updating STATUS.md, or by adding a supersession banner to the offending file.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
