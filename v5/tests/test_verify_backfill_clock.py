"""The clock verifier must fail on exactly the defect that slipped through V4."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from v5.ops import verify_backfill_clock as verifier
from v5.ops.audit_causal_day_coverage import LAST_QUOTE_MINUTE, QUOTE_MINUTES

SESSION = "2024-03-15"


def _write(tmp_path: Path, minutes: list[str], name: str = SESSION) -> Path:
    stamps = pd.to_datetime([f"{SESSION} {m}" for m in minutes]).tz_localize(
        "America/New_York"
    ).tz_convert("UTC")
    path = tmp_path / f"{name}.cbbo-1m.parquet"
    pd.DataFrame({"ts_recv": stamps, "symbol": ["SPXW  240315C05000000"] * len(minutes)}).to_parquet(
        path, index=False
    )
    return path


def test_full_clock_passes(tmp_path: Path) -> None:
    _write(tmp_path, list(QUOTE_MINUTES))
    receipt = verifier.verify(tmp_path)
    assert receipt["gate"] == "PASS"
    assert receipt["sessions_failed"] == 0
    assert receipt["sessions_without_terminal_minute"] == 0


def test_the_v4_defect_is_caught(tmp_path: Path) -> None:
    """One missing terminal bar must fail the gate, not round away.

    This is the exact V4 shape: 389 of 390 minutes, ending 15:59.
    """

    _write(tmp_path, [m for m in QUOTE_MINUTES if m != LAST_QUOTE_MINUTE])
    receipt = verifier.verify(tmp_path)
    assert receipt["gate"] == "FAIL_CLOCK_SHORTFALL"
    assert receipt["sessions_without_terminal_minute"] == 1
    failure = receipt["failures"][0]
    assert failure["delivered_minutes"] == len(QUOTE_MINUTES) - 1
    assert failure["last_minute"] == "15:59"
    assert failure["missing_examples"] == [LAST_QUOTE_MINUTE]


def test_a_midday_gap_is_caught(tmp_path: Path) -> None:
    """The gate is about the whole clock, not only its final minute."""

    _write(tmp_path, [m for m in QUOTE_MINUTES if m != "12:00"])
    receipt = verifier.verify(tmp_path)
    assert receipt["gate"] == "FAIL_CLOCK_SHORTFALL"
    assert receipt["sessions_without_terminal_minute"] == 0
    assert receipt["failures"][0]["missing_examples"] == ["12:00"]


def test_one_bad_session_fails_the_whole_gate(tmp_path: Path) -> None:
    _write(tmp_path, list(QUOTE_MINUTES), name="2024-03-15")
    _write(tmp_path, list(QUOTE_MINUTES)[:-1], name="2024-03-18")
    receipt = verifier.verify(tmp_path)
    assert receipt["gate"] == "FAIL_CLOCK_SHORTFALL"
    assert receipt["sessions_ok"] == 1
    assert receipt["sessions_failed"] == 1


def test_empty_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(verifier.ClockVerificationError, match="no delivered quote files"):
        verifier.verify(tmp_path)


def _write_book(
    tmp_path: Path,
    minutes: list[str],
    *,
    freeze_from: str | None = None,
    contracts: int = 8,
    name: str = SESSION,
) -> Path:
    """A session with a top-of-book that drifts, optionally frozen from a minute.

    Mirrors the real signature: a live book changes every minute across every
    contract; a padded one repeats the last book exactly.
    """

    rows = []
    for index, minute in enumerate(minutes):
        step = index if freeze_from is None or minute < freeze_from else minutes.index(freeze_from)
        for contract in range(contracts):
            rows.append(
                {
                    "ts_recv": pd.Timestamp(f"{SESSION} {minute}", tz="America/New_York"),
                    "symbol": f"SPXW  240315C0500{contract}000",
                    "bid_px_00": 1.0 + step * 0.01 + contract,
                    "ask_px_00": 1.5 + step * 0.01 + contract,
                    "bid_sz_00": 10 + step,
                    "ask_sz_00": 12 + step,
                }
            )
    frame = pd.DataFrame(rows)
    frame["ts_recv"] = frame["ts_recv"].dt.tz_convert("UTC")
    path = tmp_path / f"{name}.cbbo-1m.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_a_live_full_session_is_build_eligible(tmp_path: Path) -> None:
    _write_book(tmp_path, list(QUOTE_MINUTES))
    receipt = verifier.verify(tmp_path)
    assert receipt["gate"] == "PASS"
    assert receipt["liveness_checked"] is True
    assert receipt["classification_counts"] == {verifier.CLASS_OK: 1}
    assert receipt["build_eligible_sessions"] == [SESSION]


def test_a_padded_early_close_is_refused_despite_a_full_clock(tmp_path: Path) -> None:
    """The 2022-11-25 defect: every minute present, book frozen from 13:00.

    A presence-only gate passes this session and hands the builder three hours
    of fabricated flat prices. This is the test that must fail if that check is
    ever weakened back to presence alone.
    """

    _write_book(tmp_path, list(QUOTE_MINUTES), freeze_from="13:00")
    receipt = verifier.verify(tmp_path)

    assert receipt["gate"] == "FAIL_CLOCK_SHORTFALL"
    failure = receipt["failures"][0]
    assert failure["has_terminal_minute"] is True  # presence alone would pass it
    assert failure["missing_required_minutes"] == 0
    assert failure["classification"] == verifier.CLASS_STALE_PADDED
    assert failure["last_live_minute"] == "13:00"
    assert failure["build_eligible"] is False
    assert receipt["build_eligible_sessions"] == []
    assert receipt["excluded_sessions"][verifier.CLASS_STALE_PADDED] == [SESSION]
    assert "stale_padded_warning" in receipt


def test_a_truncated_early_close_is_named_not_confused_with_padding(tmp_path: Path) -> None:
    """2023-11-24's shape: contiguous from the open, simply stops early."""

    early = [m for m in QUOTE_MINUTES if m <= "13:14"]
    _write_book(tmp_path, early)
    receipt = verifier.verify(tmp_path)

    failure = receipt["failures"][0]
    assert failure["classification"] == verifier.CLASS_EARLY_CLOSE
    assert failure["last_minute"] == "13:14"
    assert failure["stale_trailing_minutes"] == 0
    assert receipt["excluded_sessions"][verifier.CLASS_EARLY_CLOSE] == [SESSION]


def test_an_interior_gap_is_not_called_an_early_close(tmp_path: Path) -> None:
    """2025-07-30's shape: terminal bar present, minutes missing mid-session."""

    holed = [m for m in QUOTE_MINUTES if m not in {"11:20", "11:21", "11:22"}]
    _write_book(tmp_path, holed)
    receipt = verifier.verify(tmp_path)

    failure = receipt["failures"][0]
    assert failure["classification"] == verifier.CLASS_INTERIOR_GAP
    assert failure["has_terminal_minute"] is True
    assert failure["missing_required_minutes"] == 3


def test_liveness_is_unknown_rather_than_assumed_when_columns_are_absent(
    tmp_path: Path,
) -> None:
    """Missing evidence is never silently a pass."""

    _write(tmp_path, list(QUOTE_MINUTES))
    receipt = verifier.verify(tmp_path)
    assert receipt["liveness_checked"] is False
    assert receipt["failures"] == []


def test_a_post_close_repeat_outside_the_required_clock_is_not_padding(
    tmp_path: Path,
) -> None:
    """The owned era's wider 08:01-16:01 grid must not read as a frozen book.

    Liveness is judged only over the minutes the builder reads, so a repeated
    post-close bar cannot refuse an otherwise healthy session.
    """

    minutes = list(QUOTE_MINUTES) + ["16:01"]
    _write_book(tmp_path, minutes, freeze_from="16:01")
    receipt = verifier.verify(tmp_path)

    assert receipt["gate"] == "PASS"
    assert receipt["build_eligible_sessions"] == [SESSION]
