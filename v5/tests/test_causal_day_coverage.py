from __future__ import annotations

import pandas as pd

from v5.ops.audit_causal_day_coverage import (
    MAX_ENTRY_ASK_USD,
    causal_es_bar_minute,
    eligible_entry,
    minute_range,
    signed_moneyness,
)


def _row(right: str, strike: float, spot: float, ask: float) -> dict:
    return {
        "right": right,
        "strike": strike,
        "underlying_price": spot,
        "bid": max(0.05, ask - 0.10),
        "ask": ask,
        "bid_size": 1.0,
        "ask_size": 1.0,
        "quote_age_ms": 0.0,
    }


def test_signed_moneyness_is_positive_itm_for_both_sides() -> None:
    got = signed_moneyness(
        spot=pd.Series([101.0, 99.0, 99.0, 101.0]).to_numpy(),
        strike=pd.Series([100.0, 100.0, 100.0, 100.0]).to_numpy(),
        right=pd.Series(["C", "C", "P", "P"]).to_numpy(),
    )
    assert got.tolist() == [1.0, -1.0, 1.0, -1.0]


def test_entry_band_is_otm_near_atm_and_affordable() -> None:
    ceiling_points = MAX_ENTRY_ASK_USD / 100.0
    frame = pd.DataFrame(
        [
            _row("C", 125.0, 100.0, 1.0),  # -25 boundary, allowed
            _row("P", 75.0, 100.0, 1.0),   # -25 boundary, allowed
            _row("C", 125.01, 100.0, 1.0), # deep OTM, barred
            _row("P", 100.0, 100.0, 1.0),  # ATM, not OTM
            _row("C", 99.0, 100.0, 1.0),   # ITM, barred for entry
            _row("P", 99.0, 100.0, ceiling_points),
            _row("P", 99.0, 100.0, ceiling_points + 0.01),
        ]
    )
    assert eligible_entry(frame).tolist() == [True, True, False, False, False, True, False]


def test_stale_one_sided_and_crossed_quotes_are_unavailable() -> None:
    frame = pd.DataFrame(
        [
            _row("C", 105.0, 100.0, 1.0),
            {**_row("C", 105.0, 100.0, 1.0), "quote_age_ms": 90_001.0},
            {**_row("C", 105.0, 100.0, 1.0), "bid": 0.0},
            {**_row("C", 105.0, 100.0, 1.0), "bid": 1.0},
            {**_row("C", 105.0, 100.0, 1.0), "ask_size": 0.0},
        ]
    )
    assert eligible_entry(frame).tolist() == [True, False, False, False, False]


def test_completed_es_bar_ends_at_decision_boundary() -> None:
    assert causal_es_bar_minute("09:35") == "09:34"
    assert causal_es_bar_minute("13:30") == "13:29"


def test_minute_range_is_inclusive() -> None:
    assert minute_range("09:31", "09:33") == ("09:31", "09:32", "09:33")
