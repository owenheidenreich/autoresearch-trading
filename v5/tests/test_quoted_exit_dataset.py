"""An exit must be able to sell only what a buyer was bidding."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import build_quoted_exit_dataset as bqe


def _session(path, *, seed: int, spread: float = 0.5):
    rng = np.random.default_rng(seed)
    minutes = [bqe._label(i) for i in range(bqe._index("09:30"), bqe._index("16:00") + 1)]
    strikes = np.arange(5000.0, 5105.0, 5.0)
    spot = 5050.0 + np.cumsum(rng.normal(0.0, 0.6, len(minutes)))
    rows = []
    for i, minute in enumerate(minutes):
        left = max(bqe._index("16:00") - bqe._index(minute), 1)
        for strike in strikes:
            for right in ("C", "P"):
                money = (spot[i] - strike) if right == "C" else (strike - spot[i])
                fair = max(money, 0.0) + 6.0 * np.sqrt(left / 390.0)
                rows.append({
                    "event_time": pd.Timestamp(f"2025-09-03 {minute}", tz="America/New_York"),
                    "strike": float(strike), "right": right,
                    "bid": float(fair - spread / 2), "ask": float(fair + spread / 2),
                    "mid": float(fair),
                })
    frame = pd.DataFrame(rows)
    frame[frame["bid"] > 0].to_parquet(path)
    return path


def test_bid_is_never_above_mid(tmp_path) -> None:
    got = bqe.session_paths(_session(tmp_path / "q.parquet", seed=1))
    assert (got["bid"] <= got["mid"] + 1e-9).all()


def test_entry_is_the_ask_and_costs_a_full_spread_over_the_bid(tmp_path) -> None:
    spread = 0.6
    got = bqe.session_paths(_session(tmp_path / "q.parquet", seed=2, spread=spread))
    first = got.groupby("trade_id").first()
    assert np.allclose(first["spread_usd"], spread * 100.0, atol=1e-6)
    assert np.allclose(first["entry_ask_usd"] - first["spread_usd"],
                       (first["entry_mid_usd"] - spread * 50.0), atol=1e-6)


def test_one_contract_per_side_per_entry_minute(tmp_path) -> None:
    got = bqe.session_paths(_session(tmp_path / "q.parquet", seed=3))
    per = got.groupby(["entry_minute", "is_call"])["trade_id"].nunique()
    assert (per == 1).all()


def test_the_chosen_contract_is_the_nearest_to_the_money(tmp_path) -> None:
    got = bqe.session_paths(_session(tmp_path / "q.parquet", seed=4))
    first = got.groupby("trade_id").first()
    assert (first["moneyness_now"].abs() <= bqe.NEAR_ATM_POINTS).all()


def test_path_state_never_reads_beyond_its_own_minute(tmp_path) -> None:
    """peak_so_far must be a running maximum, not the whole trade's maximum."""

    got = bqe.session_paths(_session(tmp_path / "q.parquet", seed=5))
    for _, trade in got.groupby("trade_id"):
        trade = trade.sort_values("minute_in_trade")
        peak = trade["peak_so_far"].to_numpy(float)
        assert np.all(np.diff(peak) >= -1e-9), "peak_so_far decreased"
        ret = trade["return_from_entry"].to_numpy(float)
        assert np.all(peak + 1e-9 >= ret)


def test_trough_is_a_running_minimum(tmp_path) -> None:
    got = bqe.session_paths(_session(tmp_path / "q.parquet", seed=6))
    for _, trade in got.groupby("trade_id"):
        trough = trade.sort_values("minute_in_trade")["trough_so_far"].to_numpy(float)
        assert np.all(np.diff(trough) <= 1e-9), "trough_so_far increased"


def test_a_trade_never_runs_past_the_declared_hold(tmp_path) -> None:
    got = bqe.session_paths(_session(tmp_path / "q.parquet", seed=7))
    assert got["minute_in_trade"].max() <= bqe.MAX_HOLD_MINUTES


def test_minute_in_trade_starts_at_zero_for_every_trade(tmp_path) -> None:
    got = bqe.session_paths(_session(tmp_path / "q.parquet", seed=8))
    assert (got.groupby("trade_id")["minute_in_trade"].min() == 0.0).all()
