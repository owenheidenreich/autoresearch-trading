"""The exit policies must do what their names say."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import train_quoted_exit as tqe


def _trade(prices, continuation, trade_id="t1", entry=1.0):
    n = len(prices)
    return pd.DataFrame({
        "trade_id": trade_id, "session": "2025-09-03",
        "minute_in_trade": np.arange(n, dtype=float),
        "price": np.asarray(prices, float),
        "continuation": np.asarray(continuation, float),
        "entry_ask_usd": entry * 100.0, "entry_mid_usd": entry * 100.0,
    })


def test_oracle_takes_the_best_minute_on_the_path() -> None:
    got = tqe.score(_trade([1.0, 3.0, 2.0, 5.0, 1.0], [9] * 5),
                    mode="oracle", entry_column="entry_ask_usd", seed=1)
    assert got["mean_gross_usd"] == pytest.approx(400.0)  # (5.0 - 1.0) * 100


def test_hold_takes_the_declared_clock_minute() -> None:
    prices = [1.0] * 60
    prices[tqe.CLOCK_MINUTES] = 4.0
    got = tqe.score(_trade(prices, [99] * 60),
                    mode="hold", entry_column="entry_ask_usd", seed=1)
    assert got["mean_gross_usd"] == pytest.approx(300.0)
    assert got["mean_minutes_held"] == pytest.approx(tqe.CLOCK_MINUTES)


def test_optimal_stop_leaves_the_first_minute_price_beats_continuation() -> None:
    # Continuation is high everywhere except minute 2, where holding is worth less.
    got = tqe.score(_trade([1.0, 1.0, 3.0, 9.0], [99.0, 99.0, 2.0, 99.0]),
                    mode="optimal_stop", entry_column="entry_ask_usd", seed=1)
    assert got["mean_gross_usd"] == pytest.approx(200.0)  # sold at 3.0
    assert got["mean_minutes_held"] == pytest.approx(2.0)


def test_optimal_stop_rides_to_the_end_when_it_never_fires() -> None:
    got = tqe.score(_trade([1.0, 2.0, 3.0, 4.0], [99.0] * 4),
                    mode="optimal_stop", entry_column="entry_ask_usd", seed=1)
    assert got["mean_gross_usd"] == pytest.approx(300.0)
    assert got["mean_minutes_held"] == pytest.approx(3.0)


def test_a_policy_can_never_sell_at_the_entry_minute() -> None:
    """Minute zero is the fill; an exit there would be a costless round trip."""

    for mode in ("random", "optimal_stop", "hold"):
        got = tqe.score(_trade([5.0, 1.0, 1.0, 1.0], [0.0, 99.0, 99.0, 99.0]),
                        mode=mode, entry_column="entry_ask_usd", seed=2)
        assert got["mean_minutes_held"] >= 1.0, mode


def test_oracle_is_an_upper_bound_on_every_other_policy() -> None:
    rng = np.random.default_rng(0)
    prices = rng.lognormal(0.0, 0.3, 40)
    frame = _trade(prices, rng.random(40) * 2)
    oracle = tqe.score(frame, mode="oracle", entry_column="entry_ask_usd", seed=3)
    for mode in ("hold", "optimal_stop", "random"):
        other = tqe.score(frame, mode=mode, entry_column="entry_ask_usd", seed=3)
        assert other["mean_gross_usd"] <= oracle["mean_gross_usd"] + 1e-6


def test_entering_at_the_ask_never_looks_better_than_entering_at_the_mid() -> None:
    frame = _trade([2.0, 2.0], [99.0, 99.0])
    frame["entry_ask_usd"] = 150.0
    frame["entry_mid_usd"] = 100.0
    at_ask = tqe.score(frame, mode="hold", entry_column="entry_ask_usd", seed=1)
    at_mid = tqe.score(frame, mode="hold", entry_column="entry_mid_usd", seed=1)
    assert at_ask["mean_gross_usd"] < at_mid["mean_gross_usd"]


def test_fees_are_charged_once_and_only_once() -> None:
    got = tqe.score(_trade([1.0, 2.0], [99.0, 99.0]),
                    mode="hold", entry_column="entry_ask_usd", seed=1)
    assert got["mean_net_usd"] == pytest.approx(
        got["mean_gross_usd"] - tqe.FEES_PER_ROUND_TRIP_USD
    )


def test_trailing_stop_rides_a_sustained_move_instead_of_leaving_early() -> None:
    """The whole point: a clock bails at minute five, a trail stays with the trend."""

    rising = [1.0] + [1.0 + 0.05 * i for i in range(1, 40)]
    frame = _trade(rising, [0.0] * len(rising))
    trail = tqe.score(frame, mode="trail", entry_column="entry_ask_usd", seed=1, trail=0.25)
    clock = tqe.score(frame, mode="hold", entry_column="entry_ask_usd", seed=1, clock=5)
    assert trail["mean_gross_usd"] > clock["mean_gross_usd"]
    assert trail["mean_minutes_held"] > clock["mean_minutes_held"]


def test_trailing_stop_fires_after_giving_back_the_declared_fraction() -> None:
    prices = [1.0, 2.0, 4.0, 3.0, 9.0]      # peak 4.0, then 3.0 is a 25% giveback
    got = tqe.score(_trade(prices, [0.0] * 5), mode="trail",
                    entry_column="entry_ask_usd", seed=1, trail=0.25)
    assert got["mean_minutes_held"] == pytest.approx(3.0)
    assert got["mean_gross_usd"] == pytest.approx(200.0)   # sold 3.0, paid 1.0


def test_a_tighter_trail_never_holds_longer_than_a_looser_one() -> None:
    rng = np.random.default_rng(4)
    prices = np.abs(rng.lognormal(0.0, 0.25, 50)) + 0.5
    frame = _trade(prices, [0.0] * 50)
    held = [tqe.score(frame, mode="trail", entry_column="entry_ask_usd", seed=1,
                      trail=t)["mean_minutes_held"] for t in (0.15, 0.25, 0.35, 0.50)]
    assert held == sorted(held)


def test_trail_is_measured_from_the_peak_not_from_the_entry() -> None:
    """A -30% stop from entry and a 30% trail from the peak are different rules."""

    prices = [1.0, 1.5, 2.0, 1.5]   # never 30% below entry; is 25% below peak
    got = tqe.score(_trade(prices, [0.0] * 4), mode="trail",
                    entry_column="entry_ask_usd", seed=1, trail=0.25)
    assert got["mean_minutes_held"] == pytest.approx(3.0)
