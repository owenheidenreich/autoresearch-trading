from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from v4.research.pathd_opra_parity_features import bsm_price
from v4.research.pathd_options_chain_direction_screen import (
    HORIZON_NS,
    OptionsChainScreenError,
    _prior_art_receipt,
    _serial_replay,
    _valid_snapshot_rows,
    historical_risk_reversal_snapshot,
    live_risk_reversal_snapshot,
    risk_reversal_snapshot,
)


def _smiled_ladder(
    decision_time: datetime,
    *,
    spot: float = 6002.5,
    strikes: range = range(5940, 6066, 5),
) -> pd.DataFrame:
    expiry = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)
    years = (expiry - decision_time).total_seconds() / (365.0 * 24.0 * 60.0 * 60.0)
    rows = []
    for strike in strikes:
        volatility = 0.30 + 0.0005 * (6002.5 - float(strike))
        for right in ("C", "P"):
            mid = bsm_price(
                spot=spot,
                strike=float(strike),
                years=years,
                rate=0.04,
                dividend=0.0,
                volatility=volatility,
                right=right,
            )
            half_spread = min(0.001, mid / 4.0)
            rows.append(
                {
                    "raw_symbol": f"SYNTH-{strike}-{right}",
                    "strike": float(strike),
                    "right": right,
                    "bid": max(0.0, mid - half_spread),
                    "ask": mid + half_spread,
                }
            )
    return pd.DataFrame(rows)


def test_risk_reversal_uses_symmetric_interpolated_wings_and_shared_paths() -> None:
    now = datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc)
    rows = _smiled_ladder(now)
    historical = historical_risk_reversal_snapshot(
        rows, decision_time=pd.Timestamp(now)
    )
    live = live_risk_reversal_snapshot(rows, decision_time=pd.Timestamp(now))
    assert historical == live
    assert historical.implied_spot == pytest.approx(6002.5, abs=1e-10)
    assert historical.put_wing.target_strike == pytest.approx(5982.5, abs=1e-10)
    assert historical.call_wing.target_strike == pytest.approx(6022.5, abs=1e-10)
    assert historical.put_wing.lower_strike == 5980.0
    assert historical.put_wing.upper_strike == 5985.0
    assert historical.call_wing.lower_strike == 6020.0
    assert historical.call_wing.upper_strike == 6025.0
    assert historical.put_wing.upper_weight == pytest.approx(0.5, abs=1e-10)
    assert historical.call_wing.upper_weight == pytest.approx(0.5, abs=1e-10)
    assert historical.risk_reversal == pytest.approx(0.02, abs=1e-9)


def test_risk_reversal_fails_instead_of_extrapolating() -> None:
    now = datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc)
    rows = _smiled_ladder(now, spot=6002.5, strikes=range(5990, 6016, 5))
    with pytest.raises(ValueError, match="extrapolation"):
        risk_reversal_snapshot(rows, decision_time=pd.Timestamp(now))


def test_snapshot_filter_pins_event_age_and_rejects_bad_quotes() -> None:
    boundary = pd.Timestamp("2026-07-31T18:00:00Z")
    rows = pd.DataFrame(
        {
            "ts_event": [
                boundary - pd.Timedelta(seconds=10),
                boundary - pd.Timedelta(seconds=91),
                pd.NaT,
                boundary - pd.Timedelta(seconds=1),
            ],
            "ts_recv": [boundary] * 4,
            "strike": [100.0, 105.0, 110.0, 115.0],
            "right": ["C", "P", "C", "P"],
            "symbol": ["A", "B", "C", "D"],
            "bid_px_00": [1.0, 1.0, 1.0, 2.0],
            "ask_px_00": [1.1, 1.1, 1.1, 1.0],
        }
    )
    filtered, age_ns = _valid_snapshot_rows(rows, boundary_ns=int(boundary.value))
    assert filtered["raw_symbol"].tolist() == ["A"]
    assert age_ns.tolist() == [10_000_000_000]


def test_serial_replay_skips_overlap_and_reopens_at_exact_five_minutes() -> None:
    start = 1_000_000_000_000
    decisions = pd.DataFrame(
        {
            "session": ["2026-07-31"] * 3,
            "feature_boundary_ns": [start, start + 60_000_000_000, start + HORIZON_NS],
            "decision_time_ns": [start, start + 60_000_000_000, start + HORIZON_NS],
            "common_valid": [True, True, True],
            "h1_direction": [1, -1, 1],
            "future_spx_points": [2.0, 100.0, -1.0],
            "contemporaneous_spx_points": [0.5, 0.5, -0.5],
        }
    )
    trades = _serial_replay(decisions, action_column="h1_direction", strategy="H1")
    assert trades["decision_time_ns"].tolist() == [start, start + HORIZON_NS]
    assert trades["signed_spx_points"].tolist() == [2.0, -1.0]


def test_h1_prior_art_blocks_after_frozen_closeout() -> None:
    with pytest.raises(OptionsChainScreenError, match="H1 blocked by prior art"):
        _prior_art_receipt()
