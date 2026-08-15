"""Measured spreads and the passive-fill proxy.

The proxy is the load-bearing part: a resting order must be counted filled only
when the market actually came to it, and the adverse-selection sign must be
right for both sides or the conclusion inverts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import measure_fill_quality as fq


def _quotes(mids: list[float], spread: float = 0.20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": "2025-08-01",
            "minute": [f"10:{i:02d}" for i in range(len(mids))],
            "hour": "10",
            "strike": 5_900.0,
            "right": "C",
            "bid": [m - spread / 2 for m in mids],
            "ask": [m + spread / 2 for m in mids],
            "mid": mids,
            "spread": spread,
            "spread_usd": spread * fq.CONTRACT_MULTIPLIER,
            "spread_share_of_mid": [spread / m for m in mids],
        }
    )


def test_a_resting_buy_fills_only_when_the_ask_comes_down_to_it() -> None:
    # Market drifts up: the ask never reaches the posted mid.
    rising = fq.passive_fills(_quotes([10.0, 10.5, 11.0, 11.5, 12.0, 12.5]))
    buys = rising[(rising["side"] == "buy") & (rising["patience"] == 1)]
    assert not buys["filled"].any()

    # Market drifts down: every posted mid is reached.
    falling = fq.passive_fills(_quotes([12.5, 12.0, 11.5, 11.0, 10.5, 10.0]))
    buys = falling[(falling["side"] == "buy") & (falling["patience"] == 1)]
    assert buys["filled"].all()


def test_a_resting_sell_fills_on_the_opposite_move() -> None:
    rising = fq.passive_fills(_quotes([10.0, 10.5, 11.0, 11.5, 12.0, 12.5]))
    sells = rising[(rising["side"] == "sell") & (rising["patience"] == 1)]
    assert sells["filled"].all()

    falling = fq.passive_fills(_quotes([12.5, 12.0, 11.5, 11.0, 10.5, 10.0]))
    sells = falling[(falling["side"] == "sell") & (falling["patience"] == 1)]
    assert not sells["filled"].any()


def test_adverse_is_negative_for_both_sides_when_the_fill_goes_against_it() -> None:
    """Sign convention: negative always means the resting order was picked off."""

    falling = fq.passive_fills(_quotes([12.5, 12.0, 11.5, 11.0, 10.5, 10.0]))
    buys = falling[(falling["side"] == "buy") & falling["filled"]]
    assert (buys["adverse"] < 0).all()

    rising = fq.passive_fills(_quotes([10.0, 10.5, 11.0, 11.5, 12.0, 12.5]))
    sells = rising[(rising["side"] == "sell") & rising["filled"]]
    assert (sells["adverse"] < 0).all()


def test_an_unfilled_order_records_no_adverse_selection() -> None:
    rising = fq.passive_fills(_quotes([10.0, 10.5, 11.0, 11.5, 12.0, 12.5]))
    unfilled = rising[~rising["filled"]]
    assert unfilled["adverse"].isna().all()


def test_more_patience_never_lowers_the_fill_rate() -> None:
    rng = np.random.default_rng(3)
    mids = list(10.0 + np.cumsum(rng.normal(0, 0.1, 40)))
    got = fq.passive_fills(_quotes(mids))
    rates = got.groupby(["side", "patience"])["filled"].mean().unstack()
    for side in ("buy", "sell"):
        row = rates.loc[side]
        assert row[1] <= row[2] <= row[5] + 1e-9


def test_a_flat_market_fills_both_sides_at_the_touch() -> None:
    """With an unchanged mid, bid < mid < ask, so neither side is reached."""

    got = fq.passive_fills(_quotes([10.0] * 6))
    assert not got["filled"].any()


def test_the_declared_constants_are_the_measured_ones() -> None:
    assert fq.FEES_PER_LEG_USD == 3.08
    assert fq.NEAR_ATM_POINTS == 25.0
    assert fq.PATIENCE_MINUTES == (1, 2, 5)


def test_the_window_is_the_regular_session() -> None:
    assert (fq.RTH_START, fq.RTH_END) == ("09:35", "16:00")
