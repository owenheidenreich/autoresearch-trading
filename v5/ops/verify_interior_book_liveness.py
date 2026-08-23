"""Fail-closed gate: a quoted book must never repeat itself minute over minute.

Why this exists, twice over
---------------------------
A vendor sometimes pads a session with a frozen book instead of truncating or
omitting it. The padded minutes carry a full, well-formed, entirely fictional
chain: every field present, every gate satisfied, and roughly three hours of
prices that never moved.

**This has now been found twice, both times by hand.** On 2026-08-18 a pre-fit
review caught `2022-11-25` -- the only early close in the 2022 portion -- whose
two-sided count was identical at 237 for every minute from 13:00 to 16:00 while
every healthy session drifts minute to minute. That session was excluded. The
scan was never turned into a gate.

On 2026-08-22 a corpus audit found **three more, all interior and all inside the
built corpus**: `2023-06-26` (10:29-10:30), `2023-10-19` (12:16-12:18), and
`2023-10-25`, which carries a **4-minute freeze at 10:17-10:20 and an 18-minute
freeze at 10:22-10:39**. None sits at a close, so no early-close calendar would
have caught them. Every existing gate passed them: they have all 390 minutes, a
terminal bar, live two-sided quotes, and plausible prices.

The separation from healthy data is clean rather than marginal. Measured over
seven control sessions spanning 2022 to 2026, including the narrow-ladder 2022
era: **zero repeated books, maximum identical run of 1 minute.** The three
defective sessions carry runs of 2, 3, 4 and 18. So the bar here is exact -- a
whole book that repeats even once is a pad, not a market.

Fail-closed
-----------
An unreadable file, a missing column, or an empty session is a **FAIL**, never a
skip. Every prior defect in this project's reporting layer shared one shape: the
watcher reported success while something was lost.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from v5.ops.build_causal_day_dataset import QUOTE_COLUMNS, prepare_quotes

SCHEMA_VERSION = "v5.interior-book-liveness.v1"

#: Fields whose joint repetition across consecutive minutes defines a frozen
#: book. Prices alone are not enough: a pad repeats the sizes too, and sizes are
#: what made `2022-11-25` legible in the first place.
BOOK_FIELDS = ("contract_id", "bid", "ask", "bid_size", "ask_size")

#: Longest run of identical consecutive books tolerated. One means "no minute may
#: repeat its predecessor". Justified by measurement, not preference: seven
#: control sessions across four years produced zero repeats.
MAX_IDENTICAL_RUN = 1


class BookLivenessError(RuntimeError):
    """The quoted book could not be verified, so it must not be trusted."""


@dataclass(frozen=True)
class FreezeRun:
    first_minute: str
    last_minute: str
    length: int

    def payload(self) -> dict[str, Any]:
        return {"first_minute": self.first_minute, "last_minute": self.last_minute,
                "length": self.length}


@dataclass(frozen=True)
class LivenessResult:
    session: str
    minutes: int
    longest_identical_run: int
    runs: tuple[FreezeRun, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return self.longest_identical_run <= MAX_IDENTICAL_RUN

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL_FROZEN_BOOK"

    def payload(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "verdict": self.verdict,
            "minutes": self.minutes,
            "longest_identical_run": self.longest_identical_run,
            "max_identical_run_allowed": MAX_IDENTICAL_RUN,
            "frozen_runs": [r.payload() for r in self.runs],
        }


def _minute_digests(quotes: pd.DataFrame) -> tuple[list[str], list[str]]:
    missing = [c for c in BOOK_FIELDS if c not in quotes.columns]
    if missing:
        raise BookLivenessError(f"quote frame is missing book fields: {missing}")
    if "minute" not in quotes.columns:
        raise BookLivenessError("quote frame carries no minute column")
    digests: list[str] = []
    minutes: list[str] = []
    for minute, group in quotes.groupby("minute", sort=True):
        ordered = group[list(BOOK_FIELDS)].sort_values("contract_id")
        digests.append(hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest())
        minutes.append(str(minute))
    return minutes, digests


def find_frozen_runs(quotes: pd.DataFrame) -> tuple[int, tuple[FreezeRun, ...]]:
    """Every maximal run of consecutive minutes sharing one book digest."""

    minutes, digests = _minute_digests(quotes)
    if not digests:
        raise BookLivenessError("session carries no quoted minute")
    runs: list[FreezeRun] = []
    start = 0
    for index in range(1, len(digests) + 1):
        if index == len(digests) or digests[index] != digests[start]:
            if index - start > 1:
                runs.append(FreezeRun(minutes[start], minutes[index - 1], index - start))
            start = index
    longest = max((r.length for r in runs), default=1)
    return longest, tuple(runs)


def verify_session(session: str, quote_path: Path) -> LivenessResult:
    """Fail-closed liveness verdict for one session's raw quote file."""

    path = Path(quote_path)
    if not path.is_file():
        raise BookLivenessError(f"{session}: no quote file at {path}")
    try:
        raw = pd.read_parquet(path, columns=list(QUOTE_COLUMNS))
    except Exception as exc:  # noqa: BLE001 - unreadable input is a FAIL, not a skip
        raise BookLivenessError(f"{session}: quote file unreadable: {exc}") from exc
    quotes = prepare_quotes(raw, session)
    if quotes.empty:
        raise BookLivenessError(f"{session}: quote file carries no usable row")
    longest, runs = find_frozen_runs(quotes)
    minutes = int(quotes["minute"].nunique())
    return LivenessResult(session=session, minutes=minutes,
                          longest_identical_run=longest, runs=runs)


def build_receipt(results: list[LivenessResult]) -> dict[str, Any]:
    failures = [r for r in results if not r.passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "book_fields": list(BOOK_FIELDS),
        "max_identical_run_allowed": MAX_IDENTICAL_RUN,
        "sessions_checked": len(results),
        "sessions_failed": len(failures),
        "verdict": "PASS" if not failures else "FAIL_FROZEN_BOOK",
        "failed_sessions": [r.payload() for r in failures],
        "results": [r.payload() for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--quote-file", type=Path, action="append", required=True,
                        metavar="SESSION=PATH")
    args = parser.parse_args(argv)

    results = []
    for item in args.quote_file:
        session, _, path = str(item).partition("=")
        results.append(verify_session(session, Path(path)))
    receipt = build_receipt(results)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "results"}, indent=2))
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
