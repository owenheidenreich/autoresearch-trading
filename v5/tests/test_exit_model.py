"""The trained exit and the entry that feeds it.

The tests that matter are the causal ones and the reference ones. A policy that
exits early looks skilful on a decaying asset, so the random-exit and
shuffled-label references have to behave correctly or the whole result is
unreadable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import train_entry_and_exit as ee
from v5.ops import train_exit_model as em


def _trade(trade_id: str, session: str, prices: list[float], entry: float = 10.0,
           predicted: list[float] | None = None) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame(
        {
            "session": session,
            "trade_id": trade_id,
            "minute_in_trade": range(1, n + 1),
            "minutes_left": [n - i for i in range(1, n + 1)],
            "minutes_to_close": [300 - i for i in range(1, n + 1)],
            "price": prices,
            "entry_premium": entry * em.CONTRACT_MULTIPLIER,
            "return_from_entry": [p / entry - 1 for p in prices],
            "peak_so_far": 0.0,
            "drawdown_from_peak": 0.0,
            "trough_so_far": 0.0,
            "return_1m": 0.0,
            "return_3m": 0.0,
            "moneyness_at_entry": 0.0,
            "underlying_return_from_entry": 0.0,
            "underlying_return_3m": 0.0,
            "remaining_max_gain": 0.0,
            "predicted_gain": predicted if predicted is not None else [1.0] * n,
            "cut_0.6": 0.5,
        }
    )


# --------------------------------------------------------------------------
# realising a trade
# --------------------------------------------------------------------------


def test_a_trade_realises_the_price_at_the_minute_it_exits() -> None:
    part = _trade("t", "2025-01-01", [11.0, 12.0, 9.0])
    assert em.realise(part, 1) == pytest.approx((12.0 - 10.0) * 100 - em.COST_PER_TRADE_USD)


def test_no_exit_means_the_last_price() -> None:
    part = _trade("t", "2025-01-01", [11.0, 12.0, 9.0])
    assert em.realise(part, None) == pytest.approx((9.0 - 10.0) * 100 - em.COST_PER_TRADE_USD)


def test_the_cost_is_charged_once_per_trade() -> None:
    part = _trade("t", "2025-01-01", [10.0])
    assert em.realise(part, 0) == pytest.approx(-em.COST_PER_TRADE_USD)


# --------------------------------------------------------------------------
# the policies and their references
# --------------------------------------------------------------------------


def _panel(n_sessions: int = 40) -> pd.DataFrame:
    """Trades that rise then fall, so exiting at the peak is clearly best."""

    frames = []
    for s in range(n_sessions):
        for k in range(4):
            frames.append(
                _trade(
                    f"t{s}-{k}", f"2025-01-{1 + s % 28:02d}",
                    [11.0, 13.0, 15.0, 12.0, 9.0],
                    predicted=[1.0, 1.0, 0.0, 0.0, 0.0],
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_the_model_exits_at_the_first_minute_below_the_threshold() -> None:
    got = em.score(_panel(), 0.5, mode="model", seed=1)
    # Prediction drops below 0.5 at minute 3, where the price is 15.
    assert got["mean_minutes_held"] == pytest.approx(3.0)
    assert got["mean_net_usd"] == pytest.approx((15.0 - 10.0) * 100 - em.COST_PER_TRADE_USD)


def test_holding_ignores_the_prediction_entirely() -> None:
    got = em.score(_panel(), 0.5, mode="hold", seed=1)
    assert got["mean_minutes_held"] == pytest.approx(5.0)
    assert got["mean_net_usd"] == pytest.approx((9.0 - 10.0) * 100 - em.COST_PER_TRADE_USD)


def test_the_oracle_takes_the_best_minute_and_beats_every_policy() -> None:
    oracle = em.score(_panel(), 0.5, mode="oracle", seed=1)
    model = em.score(_panel(), 0.5, mode="model", seed=1)
    hold = em.score(_panel(), 0.5, mode="hold", seed=1)
    assert oracle["mean_net_usd"] >= model["mean_net_usd"] > hold["mean_net_usd"]


def test_the_random_reference_carries_no_information() -> None:
    """It must land between the worst and best minute, not at either."""

    got = em.score(_panel(), 0.5, mode="random", seed=4)
    hold = em.score(_panel(), 0.5, mode="hold", seed=1)
    oracle = em.score(_panel(), 0.5, mode="oracle", seed=1)
    assert hold["mean_net_usd"] < got["mean_net_usd"] < oracle["mean_net_usd"]


def test_beats_hold_is_measured_against_the_same_trades() -> None:
    got = em.score(_panel(), 0.5, mode="model", seed=1)
    hold = em.score(_panel(), 0.5, mode="hold", seed=1)
    assert got["beats_hold_usd"] == pytest.approx(
        got["mean_net_usd"] - hold["mean_net_usd"]
    )


def test_a_quantile_policy_uses_the_cut_carried_on_the_row() -> None:
    got = em.score(_panel(), 0.6, mode="model_quantile", seed=1)
    assert got["mean_minutes_held"] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# the two entry labels, which is where the design was mis-specified
# --------------------------------------------------------------------------


def test_both_entry_labels_are_declared() -> None:
    assert ee.ENTRY_LABELS == ("trade_excursion", "trade_excursion_usd")


def test_the_dollar_label_is_the_percentage_label_times_the_premium() -> None:
    """A 50% excursion on a $200 ticket is worth a quarter of one on $800."""

    cheap = 100.0 * 0.50 * em.CONTRACT_MULTIPLIER / em.CONTRACT_MULTIPLIER
    rich = 400.0 * 0.50
    assert rich == 4 * cheap


def test_the_dollar_label_is_net_of_the_round_trip() -> None:
    exits = pd.concat(
        [_trade("2025-01-01|09:35|C", "2025-01-01", [12.0, 15.0, 11.0])],
        ignore_index=True,
    )
    magnitude = pd.DataFrame(
        [
            {
                "session": "2025-01-01", "entry_minute": "09:35", "hold": 15,
                "spot": 5_000.0, "range_15m": 5.0, "range_30m": 8.0, "range_60m": 12.0,
                "move_15m": 1.0, "move_30m": 2.0, "move_60m": 3.0,
                "range_position": 0.5, "session_range": 20.0,
                "straddle_premium": 2_000.0,
            }
        ]
    )
    got = ee.build(exits, magnitude)
    # Peak price 15 against a 10 entry: $500 gross, less the round trip.
    assert got["trade_excursion_usd"].iloc[0] == pytest.approx(
        500.0 - em.COST_PER_TRADE_USD
    )
    assert got["trade_excursion"].iloc[0] == pytest.approx(0.5)


def test_selection_rates_are_fractions_of_the_available_trades() -> None:
    assert all(0.0 < r <= 1.0 for r in ee.SELECTION_RATES)
    assert list(ee.SELECTION_RATES) == sorted(ee.SELECTION_RATES)


def test_every_entry_feature_is_known_at_the_entry_minute() -> None:
    """No feature may name a forward-looking quantity."""

    for name in ee.ENTRY_FEATURES:
        assert "remaining" not in name and "excursion" not in name, name
