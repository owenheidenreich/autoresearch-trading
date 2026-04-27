"""Sanity checks on the Normalized layer.

Per the Data Contract Section 5 (leak-detection invariants), Phase 0 Gate 0
requires: bid <= ask, no zero-spread crosses without volume, monotonic
timestamps, no duplicate (event_time, contract_id) pairs.

Each check returns a CheckResult; a runner aggregates them into the
PIPELINE_INTEGRITY_REPORT.md (ticket 15).
"""
from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str = ""
    bad_rows: int = 0


def check_bid_ask_ordering(table: pa.Table) -> CheckResult:
    """Every row with both bid and ask must have bid <= ask."""
    bid = table["bid"]
    ask = table["ask"]
    both = pc.and_(pc.is_valid(bid), pc.is_valid(ask))
    crossed = pc.and_(both, pc.greater(bid, ask))
    bad = pc.sum(pc.cast(crossed, pa.int64())).as_py() or 0
    if bad == 0:
        return CheckResult(
            name="bid_ask_ordering",
            passed=True,
            details="all rows with both bid+ask have bid <= ask",
        )
    return CheckResult(
        name="bid_ask_ordering",
        passed=False,
        details=f"{bad} rows have bid > ask (crossed quote)",
        bad_rows=bad,
    )


def check_bid_nonnegative(table: pa.Table) -> CheckResult:
    """Bid must be >= 0 when present (negative bid is impossible)."""
    bid = table["bid"]
    valid_mask = pc.is_valid(bid)
    neg = pc.and_(valid_mask, pc.less(bid, 0.0))
    bad = pc.sum(pc.cast(neg, pa.int64())).as_py() or 0
    if bad == 0:
        return CheckResult(name="bid_nonnegative", passed=True)
    return CheckResult(
        name="bid_nonnegative",
        passed=False,
        details=f"{bad} rows have negative bid",
        bad_rows=bad,
    )


def check_ask_positive(table: pa.Table) -> CheckResult:
    """Ask must be > 0 when present.

    Some venues use ask=0 to indicate 'no offer' but vendors typically map
    that to NULL. We treat ask=0 as a data error.
    """
    ask = table["ask"]
    valid_mask = pc.is_valid(ask)
    nonpos = pc.and_(valid_mask, pc.less_equal(ask, 0.0))
    bad = pc.sum(pc.cast(nonpos, pa.int64())).as_py() or 0
    if bad == 0:
        return CheckResult(name="ask_positive", passed=True)
    return CheckResult(
        name="ask_positive",
        passed=False,
        details=f"{bad} rows have ask <= 0",
        bad_rows=bad,
    )


def run_all_sanity_checks(table: pa.Table) -> list[CheckResult]:
    """Run every sanity check; return list of results."""
    return [
        check_bid_ask_ordering(table),
        check_bid_nonnegative(table),
        check_ask_positive(table),
    ]
