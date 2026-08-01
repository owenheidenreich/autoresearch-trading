from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from v4.dataset.spxw_0dte_neural import NeuralDatasetConfig, _market_features, _market_window, _option_features
from v4.path_d.compat import LegacyHistoricalFeatureShim


def _bars(symbol: str, *, volume: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-02T14:30:00Z", periods=len(volume), freq="min"),
            "symbol": [symbol] * len(volume),
            "close": [6498.0, 6500.0, 6501.0, 6502.0, 6503.0, 6501.5, 6504.0, 6505.0, 6504.5, 6506.0, 6507.0, 6508.0, 6509.0, 6508.5, 6510.0, 6511.0][: len(volume)],
            "volume": volume,
        }
    )


def test_source_neutral_market_features_are_byte_and_hash_equivalent_to_legacy_fixture() -> None:
    spx = _bars("SPX", volume=[0.0] * 16)
    vix = pd.DataFrame(
        {"event_time": spx.event_time, "symbol": ["VIX"] * 16, "close": np.linspace(18.0, 19.5, 16), "volume": [0.0] * 16}
    )
    decision = pd.Timestamp("2026-01-02T14:45:00Z")
    legacy = _market_features(spx, vix, decision)
    shim = LegacyHistoricalFeatureShim().market_features(spx, vix, decision)
    assert legacy.tobytes(order="C") == shim.tobytes(order="C")
    assert hashlib.sha256(legacy.tobytes(order="C")).hexdigest() == hashlib.sha256(shim.tobytes(order="C")).hexdigest()


def test_source_neutral_weighted_vwap_and_window_are_byte_equivalent() -> None:
    spx = _bars("SPX", volume=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    vix = pd.DataFrame(
        {"event_time": spx.event_time, "symbol": ["VIX"] * 6, "close": [18.0] * 6, "volume": [0.0] * 6}
    )
    decision = pd.Timestamp("2026-01-02T14:35:00Z")
    config = NeuralDatasetConfig(market_window_minutes=4)
    legacy = _market_window(spx, vix, decision, config)
    shim = LegacyHistoricalFeatureShim().market_window(spx, vix, decision, 4)
    assert legacy.dtype == shim.dtype == np.dtype("float64")
    assert legacy.tobytes(order="C") == shim.tobytes(order="C")


def test_source_neutral_option_vector_is_byte_and_hash_equivalent_to_legacy_fixture() -> None:
    row = pd.Series(
        {
            "bid": 2.9, "ask": 3.1, "mid": 3.0, "bid_size": 10, "ask_size": 9,
            "option_ohlcv_volume": 20, "stat_open_interest": 100,
            "iv": 0.2, "delta": 0.5, "gamma": 0.04, "theta": -0.1,
            "strike": 6500.0, "right": "C", "underlying_price": 6498.0,
        }
    )
    decision = pd.Timestamp("2026-01-02T14:31:00Z")
    legacy = _option_features(
        row, decision_time=decision, atm_strike=6500, config=NeuralDatasetConfig()
    )
    shim = LegacyHistoricalFeatureShim().option_features(row, atm_strike=6500)
    assert legacy.tobytes(order="C") == shim.tobytes(order="C")
    assert hashlib.sha256(legacy.tobytes(order="C")).hexdigest() == hashlib.sha256(shim.tobytes(order="C")).hexdigest()
