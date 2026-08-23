"""The gate that would have caught four defective sessions instead of zero.

Structural tests run everywhere. The corpus tests are skipped when the SSD is not
mounted, but they are the ones that matter: they pin the three real sessions the
2026-08-22 audit found inside the built corpus, and the healthy neighbours that
must keep passing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from v5.ops.verify_interior_book_liveness import (
    MAX_IDENTICAL_RUN,
    BookLivenessError,
    build_receipt,
    find_frozen_runs,
    verify_session,
)

CORPUS_ROOT = Path("/Volumes/AR_TRADING_DATA/lifecycle_corpus_spx_tape_2022-06-01_2026-07-31")
BACKFILL_QUOTES = Path("/Volumes/AR_TRADING_DATA/lifecycle_repaired_2022-06-01_2025-07-31")
BUILD_RECEIPT = Path(
    "v4/audit/autoresearch/lifecycle_quote_backfill_2026_08_15/corpus_build_spx_tape_receipt.json"
)

# Found 2026-08-22, all interior, all inside the built corpus, all past every
# other gate. 2023-10-25 carries an 18-minute freeze at 10:22-10:39.
KNOWN_FROZEN = ("2023-06-26", "2023-10-19", "2023-10-25")
# Immediate neighbours: same era, same vendor, same ladder width.
KNOWN_HEALTHY = ("2023-06-27", "2023-10-18", "2023-10-24", "2023-10-26")

needs_corpus = pytest.mark.skipif(
    not (BACKFILL_QUOTES.is_dir() and BUILD_RECEIPT.is_file()),
    reason="built corpus not mounted",
)


def _book(rows: list[tuple[str, str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["minute", "contract_id", "bid", "ask", "bid_size", "ask_size"]
    )


def _drifting(minutes: int, *, start: float = 1.0) -> pd.DataFrame:
    rows = []
    for i in range(minutes):
        for c in ("A", "B"):
            rows.append((f"10:{i:02d}", c, start + i * 0.1, start + i * 0.1 + 0.2, 5.0, 5.0))
    return _book(rows)


# ------------------------------------------------------------------ structural


def test_a_drifting_book_passes() -> None:
    longest, runs = find_frozen_runs(_drifting(10))
    assert longest == 1 and runs == ()


def test_a_repeated_book_is_caught_even_for_one_minute() -> None:
    """The bar is exact: seven control sessions produced zero repeats."""

    frame = _drifting(10)
    # Make 10:05 an exact copy of 10:04.
    prior = frame[frame["minute"] == "10:04"].copy()
    prior["minute"] = "10:05"
    frame = pd.concat([frame[frame["minute"] != "10:05"], prior], ignore_index=True)
    longest, runs = find_frozen_runs(frame)
    assert longest == 2
    assert [(r.first_minute, r.last_minute, r.length) for r in runs] == [("10:04", "10:05", 2)]
    assert longest > MAX_IDENTICAL_RUN


def test_the_run_length_is_reported_not_just_the_fact() -> None:
    """2023-10-25's 18-minute freeze and a 2-minute one are not the same finding."""

    rows = []
    for i in range(12):
        minute = f"10:{i:02d}"
        # Minutes 3..8 all carry the identical book.
        value = 1.0 if 3 <= i <= 8 else 1.0 + i * 0.1
        for c in ("A", "B"):
            rows.append((minute, c, value, value + 0.2, 5.0, 5.0))
    longest, runs = find_frozen_runs(_book(rows))
    assert longest == 6
    assert runs[0].first_minute == "10:03" and runs[0].last_minute == "10:08"


def test_sizes_are_part_of_the_book_not_just_prices() -> None:
    """A pad repeats the sizes too; that is what made 2022-11-25 legible."""

    rows = []
    for i in range(6):
        for c in ("A", "B"):
            rows.append((f"10:{i:02d}", c, 1.0, 1.2, 5.0 if i != 3 else 9.0, 5.0))
    longest, _ = find_frozen_runs(_book(rows))
    # Minute 3 differs only in bid_size, so it breaks the run rather than joining it.
    assert longest == 3


def test_an_empty_session_fails_closed() -> None:
    with pytest.raises(BookLivenessError, match="no quoted minute"):
        find_frozen_runs(_book([]))


def test_a_missing_book_field_fails_closed() -> None:
    frame = _drifting(4).drop(columns=["bid_size"])
    with pytest.raises(BookLivenessError, match="missing book fields"):
        find_frozen_runs(frame)


def test_a_missing_quote_file_fails_closed_rather_than_skipping() -> None:
    with pytest.raises(BookLivenessError, match="no quote file"):
        verify_session("2023-10-25", Path("/nonexistent/nowhere.parquet"))


def test_the_receipt_names_the_failures() -> None:
    from v5.ops.verify_interior_book_liveness import FreezeRun, LivenessResult

    good = LivenessResult(session="2023-10-24", minutes=390, longest_identical_run=1)
    bad = LivenessResult(
        session="2023-10-25", minutes=390, longest_identical_run=18,
        runs=(FreezeRun("10:22", "10:39", 18),),
    )
    receipt = build_receipt([good, bad])
    assert receipt["verdict"] == "FAIL_FROZEN_BOOK"
    assert receipt["sessions_failed"] == 1
    assert receipt["failed_sessions"][0]["session"] == "2023-10-25"
    assert receipt["failed_sessions"][0]["frozen_runs"][0]["length"] == 18


# ---------------------------------------------------------------- real corpus


def _quote_path(session: str) -> Path:
    """The repaired-backfill naming, taken from the corpus index rather than guessed."""

    return BACKFILL_QUOTES / f"databento_spxw_0dte_{session}.parquet"


@needs_corpus
@pytest.mark.parametrize("session", KNOWN_FROZEN)
def test_the_three_audit_sessions_fail(session: str) -> None:
    path = _quote_path(session)
    if not path.is_file():
        pytest.skip(f"quote file for {session} not present at {path}")
    result = verify_session(session, path)
    assert not result.passed, f"{session} should fail the liveness gate"
    assert result.verdict == "FAIL_FROZEN_BOOK"
    assert result.runs, "a failing session must report where it froze"


@needs_corpus
@pytest.mark.parametrize("session", KNOWN_HEALTHY)
def test_healthy_neighbours_still_pass(session: str) -> None:
    """A gate that fails good data is worse than none."""

    path = _quote_path(session)
    if not path.is_file():
        pytest.skip(f"quote file for {session} not present at {path}")
    result = verify_session(session, path)
    assert result.passed, f"{session} is a healthy control and must pass"
    assert result.longest_identical_run == 1
