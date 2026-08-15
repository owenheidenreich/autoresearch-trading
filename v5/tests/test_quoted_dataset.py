"""Priced at the touch: entry must cost the ask, exit must pay the bid.

The trade-corpus policy earned its whole apparent edge from print noise. The
point of this dataset is that the spread is charged rather than inferred, so
these tests check the charging, not the modelling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops import build_quoted_dataset as bqd
from v5.ops.build_decision_dataset import LABEL_HORIZON


def _session(path, *, seed: int, spread: float = 0.5,
             after_minute: str | None = None, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    minutes = [bqd._label(i) for i in range(bqd._index("09:30"), bqd._index("16:00") + 1)]
    strikes = np.arange(5000.0, 5105.0, 5.0)
    spot = 5050.0 + np.cumsum(rng.normal(0.0, 0.6, len(minutes)))
    rows = []
    for i, minute in enumerate(minutes):
        bump = scale if after_minute and minute > after_minute else 1.0
        left = max(bqd._index("16:00") - bqd._index(minute), 1)
        for strike in strikes:
            for right in ("C", "P"):
                money = (spot[i] - strike) if right == "C" else (strike - spot[i])
                fair = (max(money, 0.0) + 6.0 * np.sqrt(left / 390.0)) * bump
                rows.append({
                    "event_time": pd.Timestamp(f"2025-09-03 {minute}", tz="America/New_York"),
                    "strike": float(strike), "right": right,
                    "bid": float(fair - spread / 2), "ask": float(fair + spread / 2),
                    "mid": float(fair),
                })
    frame = pd.DataFrame(rows)
    frame = frame[frame["bid"] > 0]
    frame.to_parquet(path)
    return path


def test_entry_is_charged_the_ask_and_exit_paid_the_bid(tmp_path) -> None:
    spread = 0.5
    got = bqd.session_rows(_session(tmp_path / "q.parquet", seed=1, spread=spread),
                           tmp_path)
    assert got is not None
    # Mid-to-mid must beat touch-to-touch by exactly one full spread per contract.
    gap = got[f"gross_mid_{LABEL_HORIZON}m"] - got[f"gross_{LABEL_HORIZON}m"]
    inside = gap.dropna()
    assert inside.notna().any()
    assert np.allclose(inside, spread * 100.0, atol=1e-6)


def test_the_spread_is_never_charged_twice(tmp_path) -> None:
    """Crossing is paid in the prices, so only fees may be added on top."""

    got = bqd.session_rows(_session(tmp_path / "q.parquet", seed=2), tmp_path)
    assert (got["round_trip_usd"] == bqd.FEES_PER_ROUND_TRIP_USD).all()


def test_recorded_spread_matches_ask_minus_bid(tmp_path) -> None:
    got = bqd.session_rows(_session(tmp_path / "q.parquet", seed=3, spread=0.8), tmp_path)
    assert np.allclose(got["spread_usd"], got["entry_ask_usd"] - got["entry_bid_usd"])
    assert np.allclose(got["spread_usd"], 80.0, atol=1e-6)


def test_a_one_sided_or_crossed_quote_is_not_a_candidate(tmp_path) -> None:
    path = _session(tmp_path / "q.parquet", seed=4)
    frame = pd.read_parquet(path)
    bad = (frame["strike"] == 5050.0) & (frame["right"] == "C")
    frame.loc[bad, "bid"] = frame.loc[bad, "ask"] + 1.0  # crossed
    frame.to_parquet(path)

    got = bqd.session_rows(path, tmp_path)
    assert not ((got["strike"] == 5050.0) & got["is_call"]).any()


def test_no_feature_reads_a_minute_after_the_decision(tmp_path) -> None:
    clean = _session(tmp_path / "a.parquet", seed=5)
    dirty = _session(tmp_path / "b.parquet", seed=5, after_minute="12:00", scale=4.0)
    a, b = bqd.session_rows(clean, tmp_path), bqd.session_rows(dirty, tmp_path)
    early = a["entry_minute"] <= "12:00"
    for column in ("moneyness", "entry_premium", "entry_ask_usd", "spread_usd",
                   "iv", "delta", "range_position", "realised_vol_30m",
                   *[f"move_{b_}m_rel" for b_ in bqd.LOOKBACKS]):
        assert np.allclose(a.loc[early, column].to_numpy(float),
                           b.loc[early, column].to_numpy(float),
                           equal_nan=True), f"{column} reads the future"


def test_missing_volume_becomes_zero_not_a_gap(tmp_path) -> None:
    """No print in a minute means zero traded, which is a fact, not a hole."""

    got = bqd.session_rows(_session(tmp_path / "q.parquet", seed=6), tmp_path)
    assert got["contract_volume_15m"].notna().all()
    assert (got["contract_volume_15m"] == 0.0).all()


def test_features_come_from_the_mid_not_the_ask(tmp_path) -> None:
    wide = bqd.session_rows(_session(tmp_path / "w.parquet", seed=7, spread=2.0), tmp_path)
    tight = bqd.session_rows(_session(tmp_path / "t.parquet", seed=7, spread=0.1), tmp_path)
    # Same underlying fair value, very different spreads: the premium the model
    # sees must not move, or the spread leaks into every feature downstream.
    assert np.allclose(wide["entry_premium"], tight["entry_premium"], atol=1e-6)
    assert not np.allclose(wide["entry_ask_usd"], tight["entry_ask_usd"])
