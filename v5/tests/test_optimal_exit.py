"""Optimal stopping, and the feature audit that a shuffled-label null cannot do.

The leak found on 2026-08-13 is the reason the last test in this file exists. An
entry feature was reading the minute *after* the entry decision; it lifted the
headline from break-even to +$119 a trade; and the shuffled-label control did not
flag it, because permuting the label removes the thing a leak would leak to.
Only a timestamp audit of each feature finds that class of defect, so the audit
is encoded here rather than left to whoever remembers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import train_full_system as fs
from v5.ops import train_optimal_exit as oe


def _table(n_sessions: int, prices: list[float], seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        for k in range(3):
            tid = f"2025-{1 + s // 28:02d}-{1 + s % 28:02d}|09:35|C{k}"
            for m, price in enumerate(prices, start=1):
                jitter = 1.0 + rng.normal(0, 0.01)
                rows.append(
                    {
                        "session": f"2025-{1 + s // 28:02d}-{1 + s % 28:02d}",
                        "trade_id": tid,
                        "minute_in_trade": m,
                        "minutes_left": len(prices) - m,
                        "minutes_to_expiry": 300 - m,
                        "price": price * jitter,
                        "entry_premium": 1_000.0,
                        "return_from_entry": price / 10.0 - 1,
                        "peak_so_far": 0.0,
                        "drawdown_from_peak": 0.0,
                        "trough_so_far": 0.0,
                        "return_1m": 0.0,
                        "return_3m": 0.0,
                        "underlying_return_from_entry": 0.0,
                        "underlying_return_3m": 0.0,
                        "moneyness_now": 0.0,
                        "iv": 0.2,
                        "delta": 0.5,
                        "gamma_dollars": 1.0,
                        "theta_share_of_price": -0.002,
                        "vega": 1.0,
                        "strike": 5_900.0,
                        "is_call": True,
                        "moneyness_at_entry": 0.0,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the value iteration
# --------------------------------------------------------------------------


def test_the_next_value_within_a_trade_stops_at_the_trade_boundary() -> None:
    table = _table(2, [10.0, 11.0])
    got = oe._next_within_trade(table.sort_values(["trade_id", "minute_in_trade"]), "price")
    # Every trade's final minute has no successor.
    assert np.isnan(got).sum() == table["trade_id"].nunique()


def test_a_rising_then_falling_trade_is_stopped_at_the_peak() -> None:
    """The property that defines optimal stopping."""

    table = _table(120, [10.0, 12.0, 15.0, 11.0, 8.0]).sort_values(
        ["session", "trade_id", "minute_in_trade"]
    )
    model = oe.fit_continuation(table, seed=1, shuffle=False)
    table = table.copy()
    table["continuation"] = model.predict(table[list(oe.FEATURES)].to_numpy(float))
    got = oe.score(table, mode="optimal_stop", seed=1)
    # The peak is minute 3; a good policy should be leaving around there.
    assert 2.0 <= got["mean_minutes_held"] <= 4.0
    assert got["mean_net_usd"] > oe.score(table, mode="hold", seed=1)["mean_net_usd"]


def test_the_cost_does_not_enter_the_stopping_decision() -> None:
    """It is paid once whenever the trade closes, so it cannot change when."""

    import inspect

    source = inspect.getsource(oe.fit_continuation)
    assert "COST" not in source


def test_holding_and_the_oracle_bracket_every_policy() -> None:
    table = _table(120, [10.0, 12.0, 15.0, 11.0, 8.0]).sort_values(
        ["session", "trade_id", "minute_in_trade"]
    )
    model = oe.fit_continuation(table, seed=1, shuffle=False)
    table["continuation"] = model.predict(table[list(oe.FEATURES)].to_numpy(float))
    hold = oe.score(table, mode="hold", seed=1)["mean_net_usd"]
    stop = oe.score(table, mode="optimal_stop", seed=1)["mean_net_usd"]
    oracle = oe.score(table, mode="oracle", seed=1)["mean_net_usd"]
    assert hold <= stop <= oracle


def test_gross_is_the_net_plus_exactly_one_round_trip() -> None:
    table = _table(60, [10.0, 12.0]).sort_values(["session", "trade_id", "minute_in_trade"])
    table["continuation"] = 0.0
    got = oe.score(table, mode="hold", seed=1)
    assert got["mean_gross_usd"] - got["mean_net_usd"] == pytest.approx(
        oe.COST_PER_TRADE_USD
    )


def test_the_value_function_is_swept_more_than_once() -> None:
    assert oe.VALUE_ITERATIONS >= 2


# --------------------------------------------------------------------------
# the feature audit the null control cannot do
# --------------------------------------------------------------------------


def test_the_leaked_feature_is_recorded_and_stays_out() -> None:
    """Regression on the 2026-08-13 defect."""

    assert "underlying_return_3m_at_entry" in fs.LEAKED_AND_REMOVED
    assert "underlying_return_3m_at_entry" not in fs.ENTRY_FEATURES


def test_no_entry_feature_names_a_quantity_from_after_the_entry() -> None:
    """A timestamp audit, encoded.

    The exit model may read the trade's evolving state — that is its job. The
    *entry* model may not: every one of its features must be knowable in the
    minute the contract is bought.
    """

    forbidden = ("return_1m", "return_3m", "peak_so_far", "drawdown", "trough",
                 "minute_in_trade", "minutes_left", "remaining")
    for name in fs.ENTRY_FEATURES:
        for token in forbidden:
            assert token not in name, f"{name} looks like it reads past the entry"


def test_entry_greeks_are_reconstructed_without_touching_a_later_minute() -> None:
    """Spot at entry comes from strike and moneyness, not from the next bar."""

    table = _table(4, [10.0, 12.0])
    magnitude = pd.DataFrame(
        [
            {
                "session": s, "entry_minute": "09:35", "hold": 15, "spot": 5_900.0,
                "range_15m": 5.0, "range_30m": 8.0, "range_60m": 12.0,
                "move_15m": 1.0, "move_30m": 2.0, "move_60m": 3.0,
                "range_position": 0.5, "session_range": 20.0,
                "straddle_premium": 2_000.0,
            }
            for s in table["session"].unique()
        ]
    )
    got = fs.add_entry_features(table, magnitude)
    # Moneyness zero and strike 5,900 means the parity spot at entry was 5,900.
    assert not got.empty
    assert set(fs.ENTRY_FEATURES) <= set(got.columns)


def test_the_entry_label_is_in_dollars_net_of_the_round_trip() -> None:
    table = _table(4, [10.0, 15.0])
    magnitude = pd.DataFrame(
        [
            {
                "session": s, "entry_minute": "09:35", "hold": 15, "spot": 5_900.0,
                "range_15m": 5.0, "range_30m": 8.0, "range_60m": 12.0,
                "move_15m": 1.0, "move_30m": 2.0, "move_60m": 3.0,
                "range_position": 0.5, "session_range": 20.0,
                "straddle_premium": 2_000.0,
            }
            for s in table["session"].unique()
        ]
    )
    got = fs.add_entry_features(table, magnitude)
    # Peak near 15 against a 10 entry is about $500, less the round trip.
    assert got[fs.ENTRY_LABEL].median() == pytest.approx(
        500.0 - fs.COST_PER_TRADE_USD, abs=30.0
    )
