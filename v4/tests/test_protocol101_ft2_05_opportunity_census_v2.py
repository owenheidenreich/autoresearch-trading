from __future__ import annotations

import math
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_V2_BUILDER = (
    Path(__file__).resolve().parents[1]
    / "audit/autoresearch/protocol101_ft2_05_opportunity_census/"
    "superseded/v2_pre_intent_recheck_20260729/"
    "run_protocol101_ft2_05_opportunity_census_v2.py"
)
_SPEC = importlib.util.spec_from_file_location("ft205_preserved_v2", _V2_BUILDER)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ONE_MINUTE_NS = _MODULE.ONE_MINUTE_NS
QuotePath = _MODULE.QuotePath
_d49_should_soft_close = _MODULE._d49_should_soft_close
_quote_at_minute = _MODULE._quote_at_minute
compute_horizon_labels = _MODULE.compute_horizon_labels


def _ns(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").value)


def _path(
    times: list[int],
    bids: list[float],
    asks: list[float] | None = None,
    ages: list[float] | None = None,
) -> QuotePath:
    return QuotePath(
        quote_ns=np.asarray(times, dtype=np.int64),
        bid=np.asarray(bids, dtype=float),
        ask=np.asarray(asks or [2.0] * len(times), dtype=float),
        quote_age_ms=np.asarray(ages or [0.0] * len(times), dtype=float),
    )


def test_tplus1_entry_fill_minute_is_not_a_descriptive_mark() -> None:
    entry_fill = _ns("2025-01-02 15:01:00")
    path = _path(
        [
            entry_fill,
            entry_fill + ONE_MINUTE_NS,
            entry_fill + 2 * ONE_MINUTE_NS,
            entry_fill + 3 * ONE_MINUTE_NS,
        ],
        [9.0, 1.80, 2.10, 2.40],
    )
    result = compute_horizon_labels(
        path=path,
        entry_fill_ns=entry_fill,
        session="2025-01-02",
        entry_ask=2.0,
        horizon=3,
    )
    assert result["nominal_minutes"] == 3
    assert result["available_minutes"] == 3
    assert result["ttfp_minutes"] == 2.0
    assert result["first_profit_time_ns"] == entry_fill + 2 * ONE_MINUTE_NS
    assert result["mfe_dollars"] == pytest.approx(37.0)


def test_no_bid_minute_is_full_loss_not_dropped() -> None:
    entry_fill = _ns("2025-01-02 15:01:00")
    path = _path(
        [
            entry_fill,
            entry_fill + ONE_MINUTE_NS,
            entry_fill + 2 * ONE_MINUTE_NS,
            entry_fill + 3 * ONE_MINUTE_NS,
        ],
        [1.95, 1.80, math.nan, 2.10],
    )
    result = compute_horizon_labels(
        path=path,
        entry_fill_ns=entry_fill,
        session="2025-01-02",
        entry_ask=2.0,
        horizon=3,
    )
    assert result["available_minutes"] == 3
    assert result["executable_bid_minutes"] == 2
    assert result["no_bid_minutes"] == 1
    assert result["early_dd_dollars"] == -203.0
    assert result["mnar_excluded_early_dd_dollars"] == pytest.approx(-23.0)
    assert result["underwater_minutes"] == 2


def test_quote_at_minute_requires_exact_fresh_executable_quote() -> None:
    timestamp = _ns("2025-01-02 15:01:00")
    path = _path(
        [timestamp],
        [1.90],
        asks=[2.00],
        ages=[90_001.0],
    )
    bid, ask, bid_valid, ask_valid = _quote_at_minute(path, timestamp)
    assert math.isnan(bid)
    assert math.isnan(ask)
    assert not bid_valid
    assert not ask_valid


def test_d49_soft_close_uses_one_dollar_premium_plus_fee_floor() -> None:
    assert not _d49_should_soft_close(
        session_start_equity=10_000.0,
        realized_session_pnl=-397.0,
    )
    assert _d49_should_soft_close(
        session_start_equity=10_000.0,
        realized_session_pnl=-397.01,
    )


def test_remaining_session_uses_exact_1555_boundary() -> None:
    entry_fill = _ns("2025-01-02 20:53:00")
    path = _path(
        [
            entry_fill,
            entry_fill + ONE_MINUTE_NS,
            entry_fill + 2 * ONE_MINUTE_NS,
        ],
        [2.0, 2.1, math.nan],
    )
    result = compute_horizon_labels(
        path=path,
        entry_fill_ns=entry_fill,
        session="2025-01-02",
        entry_ask=2.0,
        horizon="remaining_session",
    )
    assert result["deadline_ns"] == _ns("2025-01-02 20:55:00")
    assert result["nominal_minutes"] == 2
    assert result["no_bid_minutes"] == 1
    assert result["uwi_dollars"] == 203.0
