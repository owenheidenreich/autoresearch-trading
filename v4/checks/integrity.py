"""Integrity checks on the Normalized layer.

Timestamp monotonicity (per contract), duplicate-key detection, no-missing-
trading-minutes, and deterministic-rebuild parity. These complement the
sanity checks in sanity.py.
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from .sanity import CheckResult


def check_no_duplicate_keys(
    table: pa.Table, keys: list[str] | None = None
) -> CheckResult:
    """Verify uniqueness of (event_time, contract_id) — the natural key
    in the Normalized layer.

    Duplicates indicate either a vendor data quality issue or a bug in
    ingest. Either way, halts CI.
    """
    keys = keys or ["event_time", "contract_id"]
    for k in keys:
        if k not in table.column_names:
            return CheckResult(
                name="no_duplicate_keys",
                passed=False,
                details=f"required key column missing: {k!r}",
            )

    sub = table.select(keys)
    grouped = sub.group_by(keys).aggregate([])
    n_rows = table.num_rows
    n_unique = grouped.num_rows
    if n_rows == n_unique:
        return CheckResult(
            name="no_duplicate_keys",
            passed=True,
            details=f"all {n_rows} rows are unique on {keys}",
        )
    return CheckResult(
        name="no_duplicate_keys",
        passed=False,
        details=f"{n_rows - n_unique} duplicate (event_time, contract_id) pair(s)",
        bad_rows=n_rows - n_unique,
    )


def check_timestamp_monotonic_per_contract(table: pa.Table) -> CheckResult:
    """Within each contract_id, event_time must be non-decreasing in storage order.

    Out-of-order timestamps within a single contract usually indicate a
    vendor sequencing issue (e.g., the SEC 0DTE limit-order paper's OPRA
    quote-after-trade case). The simulator must handle these explicitly;
    they should not silently exist in Normalized.

    Important: this check examines rows in their stored order. It does NOT
    sort first — a sort would mask exactly the disorder we're looking for.
    """
    if table.num_rows == 0:
        return CheckResult(
            name="timestamp_monotonic_per_contract",
            passed=True,
            details="empty table",
        )
    for col in ("event_time", "contract_id"):
        if col not in table.column_names:
            return CheckResult(
                name="timestamp_monotonic_per_contract",
                passed=False,
                details=f"required column missing: {col!r}",
            )

    cids = table["contract_id"].to_pylist()
    times = table["event_time"].to_pylist()
    last_seen: dict[str, object] = {}
    bad = 0
    for cid, ts in zip(cids, times, strict=True):
        prev = last_seen.get(cid)
        if prev is not None and ts < prev:
            bad += 1
        else:
            last_seen[cid] = ts
    if bad == 0:
        return CheckResult(
            name="timestamp_monotonic_per_contract",
            passed=True,
            details=f"event_time is non-decreasing within each contract ({len(last_seen)} contracts checked)",
        )
    return CheckResult(
        name="timestamp_monotonic_per_contract",
        passed=False,
        details=f"{bad} backwards-time event(s) within a contract",
        bad_rows=bad,
    )


def check_required_non_null(
    table: pa.Table, required_columns: list[str]
) -> CheckResult:
    """Verify each required column has no nulls."""
    bad_cols: list[tuple[str, int]] = []
    for col in required_columns:
        if col not in table.column_names:
            return CheckResult(
                name="required_non_null",
                passed=False,
                details=f"required column missing: {col!r}",
            )
        nulls = pc.sum(pc.cast(pc.is_null(table[col]), pa.int64())).as_py() or 0
        if nulls > 0:
            bad_cols.append((col, nulls))
    if not bad_cols:
        return CheckResult(
            name="required_non_null",
            passed=True,
            details=f"all {len(required_columns)} required columns are non-null",
        )
    return CheckResult(
        name="required_non_null",
        passed=False,
        details=f"null counts: {bad_cols}",
        bad_rows=sum(n for _, n in bad_cols),
    )


def run_all_integrity_checks(
    table: pa.Table, required_non_null_cols: list[str] | None = None
) -> list[CheckResult]:
    results = [
        check_no_duplicate_keys(table),
        check_timestamp_monotonic_per_contract(table),
    ]
    if required_non_null_cols:
        results.append(check_required_non_null(table, required_non_null_cols))
    return results
