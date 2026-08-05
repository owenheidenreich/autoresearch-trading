"""Executable refusal check against the canonical v5 research history.

The check is intentionally read-only. A hit in the do-not-retest section or a
rejecting verdict in the protocol lineage blocks an unchanged experiment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DISTILLATION = REPO_ROOT / "v5/research/history/DO_NOT_RETEST.md"
FARM_LINEAGE = REPO_ROOT / "v5/research/history/PROTOCOL_FARM_LINEAGE_2026_07_19.md"


class LoopError(RuntimeError):
    """A research-loop precondition failed closed."""


_REJECTING_VERDICT = re.compile(
    r"\*\*(reject|falsified|null|abandoned|failed|no-op)"
    r"|exit-loop stop"
    r"|loop stop"
    r"|do not retest"
    r"|proof failure",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PriorArtHit:
    source: str
    line_number: int
    text: str
    in_do_not_retest: bool
    rejecting_verdict: bool = False

    @property
    def blocking(self) -> bool:
        return self.in_do_not_retest or self.rejecting_verdict


def _do_not_retest_span(text: str) -> tuple[int, int]:
    """Return the line span of the ledger's do-not-retest section."""

    lines = text.splitlines()
    start = end = None
    for index, line in enumerate(lines):
        if start is None and re.match(r"^##\s*4\.\s*Do-not-retest", line, re.IGNORECASE):
            start = index
        elif start is not None and line.startswith("## ") and index > start:
            end = index
            break
    if start is None:
        return (-1, -1)
    return (start, end if end is not None else len(lines))


def prior_art_check(mechanism: str, *, extra_terms: Sequence[str] = ()) -> list[PriorArtHit]:
    """Find matching prior work and mark every binding refusal.

    Proceeding after a blocking hit requires a materially different causal
    variable, target, data source, or game—not a renamed seed or threshold.
    """

    terms = [term.strip().lower() for term in [mechanism, *extra_terms] if term and term.strip()]
    if not terms:
        raise LoopError("prior_art_check requires a mechanism")

    hits: list[PriorArtHit] = []
    for path in (DISTILLATION, FARM_LINEAGE):
        if not path.exists():
            raise LoopError(
                f"canonical history document missing: {path}. The prior-art check cannot "
                "be skipped; restore the document or fix the canonical path."
            )
        text = path.read_text(encoding="utf-8")
        span = _do_not_retest_span(text) if path == DISTILLATION else (-1, -1)
        for index, line in enumerate(text.splitlines()):
            low = line.lower()
            if any(term in low for term in terms):
                hits.append(
                    PriorArtHit(
                        source=str(path.relative_to(REPO_ROOT)),
                        line_number=index + 1,
                        text=line.strip()[:400],
                        in_do_not_retest=bool(span[0] <= index < span[1]),
                        rejecting_verdict=bool(_REJECTING_VERDICT.search(line)),
                    )
                )
    return hits


def blocked_by_prior_art(hits: Sequence[PriorArtHit]) -> bool:
    """Return True when any canonical-history hit is binding."""

    return any(hit.blocking for hit in hits)


__all__ = [
    "LoopError",
    "PriorArtHit",
    "blocked_by_prior_art",
    "prior_art_check",
]
