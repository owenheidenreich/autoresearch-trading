"""Tests for sanity and integrity checks."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest

from v4.checks import (
    check_ask_positive,
    check_bid_ask_ordering,
    check_bid_nonnegative,
    check_no_duplicate_keys,
    check_required_non_null,
    check_timestamp_monotonic_per_contract,
    run_all_integrity_checks,
    run_all_sanity_checks,
)
from v4.ingest.optionsdx import ingest_optionsdx_file

FIXTURE = Path(__file__).parent / "fixtures" / "optionsdx_spx_sample.csv"


# ---------- live data sanity (against the OptionsDX fixture) ----------

def test_optionsdx_fixture_passes_all_sanity_checks() -> None:
    result = ingest_optionsdx_file(FIXTURE)
    for r in run_all_sanity_checks(result.normalized):
        assert r.passed, f"{r.name} failed: {r.details}"


def test_optionsdx_fixture_passes_all_integrity_checks() -> None:
    result = ingest_optionsdx_file(FIXTURE)
    for r in run_all_integrity_checks(result.normalized):
        assert r.passed, f"{r.name} failed: {r.details}"


# ---------- bid/ask sanity ----------

def _two_row_table(bid_a: float | None, ask_a: float | None, bid_b: float | None = 1.0, ask_b: float | None = 2.0) -> pa.Table:
    """Minimal table for sanity checks — only the columns the checks use."""
    return pa.table(
        {
            "bid": pa.array([bid_a, bid_b], type=pa.float64()),
            "ask": pa.array([ask_a, ask_b], type=pa.float64()),
        }
    )


def test_bid_ask_ordering_catches_crossed_quote() -> None:
    bad = _two_row_table(bid_a=5.0, ask_a=4.0)
    res = check_bid_ask_ordering(bad)
    assert not res.passed
    assert res.bad_rows == 1


def test_bid_ask_ordering_passes_normal_quote() -> None:
    good = _two_row_table(bid_a=4.0, ask_a=5.0)
    res = check_bid_ask_ordering(good)
    assert res.passed


def test_bid_ask_ordering_ignores_null_pairs() -> None:
    """Rows where one side is missing are not 'crossed'."""
    partial = _two_row_table(bid_a=None, ask_a=5.0)
    res = check_bid_ask_ordering(partial)
    assert res.passed


def test_bid_nonnegative() -> None:
    bad = _two_row_table(bid_a=-1.0, ask_a=5.0)
    assert not check_bid_nonnegative(bad).passed


def test_ask_positive_rejects_zero() -> None:
    bad = _two_row_table(bid_a=0.0, ask_a=0.0)
    assert not check_ask_positive(bad).passed


# ---------- integrity ----------

def _normalized_minimal(rows: list[dict]) -> pa.Table:
    """Minimal subset of the normalized columns the integrity checks use."""
    return pa.table(
        {
            "event_time": pa.array(
                [r["event_time"] for r in rows], type=pa.timestamp("us", tz="UTC")
            ),
            "contract_id": pa.array([r["contract_id"] for r in rows], type=pa.string()),
        }
    )


def test_duplicate_keys_caught() -> None:
    t = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    rows = [
        {"event_time": t, "contract_id": "SPXW-20230102-04000.000-C"},
        {"event_time": t, "contract_id": "SPXW-20230102-04000.000-C"},  # dupe
        {"event_time": t, "contract_id": "SPXW-20230102-04050.000-C"},
    ]
    res = check_no_duplicate_keys(_normalized_minimal(rows))
    assert not res.passed
    assert res.bad_rows == 1


def test_no_duplicates_passes_clean() -> None:
    t = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    rows = [
        {"event_time": t, "contract_id": "SPXW-20230102-04000.000-C"},
        {"event_time": t, "contract_id": "SPXW-20230102-04050.000-C"},
    ]
    assert check_no_duplicate_keys(_normalized_minimal(rows)).passed


def test_timestamp_monotonic_catches_out_of_order_within_contract() -> None:
    t1 = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    t0 = datetime(2023, 1, 2, 14, 59, tzinfo=timezone.utc)
    rows = [
        {"event_time": t1, "contract_id": "SPXW-20230102-04000.000-C"},
        {"event_time": t0, "contract_id": "SPXW-20230102-04000.000-C"},  # backwards
    ]
    res = check_timestamp_monotonic_per_contract(_normalized_minimal(rows))
    assert not res.passed


def test_timestamp_monotonic_allows_different_contracts_at_same_time() -> None:
    t = datetime(2023, 1, 2, 15, 0, tzinfo=timezone.utc)
    rows = [
        {"event_time": t, "contract_id": "SPXW-20230102-04000.000-C"},
        {"event_time": t, "contract_id": "SPXW-20230102-04050.000-C"},
    ]
    assert check_timestamp_monotonic_per_contract(_normalized_minimal(rows)).passed


def test_required_non_null_detects_nulls() -> None:
    tbl = pa.table(
        {
            "a": pa.array([1, None, 3], type=pa.int64()),
            "b": pa.array(["x", "y", "z"], type=pa.string()),
        }
    )
    res = check_required_non_null(tbl, ["a", "b"])
    assert not res.passed
    assert "1" in res.details
