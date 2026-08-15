from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from v4.dataset.spxw_0dte_neural import (
    OPTION_FEATURE_NAMES,
    NeuralDatasetConfig,
    _market_window,
)
from v4.live.protocol101_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
    DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT,
    MODEL_SCORING_FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE,
    MODEL_SCORING_VENDOR_SENSITIVE_OPTION_FEATURES,
    candidate_filter_diagnostics,
    candidate_is_tradable_values,
    feature_contract_metadata,
    feature_contract_model_transform,
    feature_contract_requires_model_scoring_greeks,
    feature_contract_version,
    is_live_feature_contract,
    option_feature_values,
    quote_source_metadata,
    strike_ladder_context,
)
from v4.live.protocol101_live_entry import LiveIndexState
from v4.model.hypothesis_protocol import MarketStructureCache


def test_live_feature_contract_zeroes_runtime_unavailable_fields() -> None:
    values = option_feature_values(
        {
            "bid": 9.8,
            "ask": 10.0,
            "mid": 9.9,
            "bid_size": 20,
            "ask_size": 25,
            "option_ohlcv_volume": 999,
            "stat_open_interest": 1234,
            "iv": 0.20,
            "delta": 0.50,
            "gamma": 0.01,
            "theta": -0.30,
        },
        decision_time=pd.Timestamp("2026-01-02T15:00:00Z"),
        spx=6000.0,
        strike=6000.0,
        right="C",
        atm_strike=6000,
        feature_names=OPTION_FEATURE_NAMES,
        live_contract=True,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )

    assert values["option_ohlcv_volume"] == 0.0
    assert values["stat_open_interest"] == 0.0
    assert np.isclose(values["spread"], 0.2)
    assert candidate_is_tradable_values(values)


def test_quote_source_metadata_records_contract_version_and_age() -> None:
    decision_time = datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc)

    metadata = quote_source_metadata(
        {"quote_time": "2026-01-02T14:59:59Z", "receive_time": "2026-01-02T15:00:00Z"},
        decision_time=pd.Timestamp(decision_time),
        feature_contract_name=FEATURE_CONTRACT_VERSION,
    )

    assert metadata["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert metadata["source_quote_time"] == "2026-01-02T14:59:59+00:00"
    assert np.isclose(metadata["quote_age_ms"], 1000.0)


def test_candidate_filter_diagnostics_explains_rejection_without_changing_helper_contract() -> None:
    values = {
        "bid": 1.0,
        "ask": 1.8,
        "mid": 1.4,
        "bid_size": 10,
        "ask_size": 10,
        "quote_age_ms": 91_000,
    }

    diagnostics = candidate_filter_diagnostics(
        values,
        DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT,
        require_greeks=False,
        enforce_freshness=True,
    )

    assert diagnostics["passed"] is False
    assert diagnostics["freshness_pass"] is False
    assert set(diagnostics["reasons"]) == {"spread_abs_too_wide", "spread_frac_too_wide", "stale_quote"}
    assert candidate_is_tradable_values(
        values,
        DEFAULT_PROTOCOL101_LIVE_FEATURE_CONTRACT,
        require_greeks=False,
    ) is False


def test_strike_ladder_context_records_shared_rounding_policy() -> None:
    context = strike_ladder_context(spx_for_ladder=6002.6, strike_step=5)

    assert context["atm_strike"] == 6005
    assert context["spx_for_ladder"] == 6002.6
    assert context["strike_step"] == 5
    assert context["rounding_tie_policy"] == "python_round_half_to_even"


def test_microstructure_masked_contract_is_live_and_declares_model_transform() -> None:
    assert is_live_feature_contract(FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED)
    assert feature_contract_version(FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED) == FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
    assert (
        feature_contract_model_transform(FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED)
        == MODEL_SCORING_FEATURE_TRANSFORM_MASK_VENDOR_SENSITIVE_OPTION_QUOTE_GREEK_MICROSTRUCTURE
    )

    metadata = feature_contract_metadata(FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED)

    assert metadata["version"] == FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
    assert metadata["raw_vendor_fields_preserved"] is True
    assert metadata["tradability_uses_raw_bid_ask_mid"] is True
    assert metadata["fills_and_pnl_use_raw_bid_ask"] is True
    assert tuple(metadata["model_scoring_masked_option_features"]) == MODEL_SCORING_VENDOR_SENSITIVE_OPTION_FEATURES
    assert {"bid", "ask", "mid", "delta", "gamma", "theta"}.issubset(
        metadata["model_scoring_masked_option_features"]
    )
    assert feature_contract_requires_model_scoring_greeks(FEATURE_CONTRACT_VERSION)
    assert not feature_contract_requires_model_scoring_greeks(
        FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED
    )


def test_quote_source_metadata_preserves_microstructure_masked_contract_version() -> None:
    metadata = quote_source_metadata(
        {"quote_time": "2026-01-02T14:59:59Z"},
        decision_time=pd.Timestamp("2026-01-02T15:00:00Z"),
        feature_contract_name=FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
    )

    assert metadata["feature_contract_version"] == FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED


def test_captured_vendor_greeks_can_be_recomputed_by_shared_contract() -> None:
    values = option_feature_values(
        {
            "bid": 9.8,
            "ask": 10.0,
            "mid": 9.9,
            "bid_size": 20,
            "ask_size": 25,
            "underlying_price": 6000.0,
            "settlement_time_utc": "2026-01-02T21:00:00Z",
            "iv": 0.90,
            "delta": 0.90,
            "gamma": 0.90,
            "theta": -0.90,
            "prefer_repaired_greeks": True,
        },
        decision_time=pd.Timestamp("2026-01-02T15:00:00Z"),
        spx=6000.0,
        strike=6000.0,
        right="C",
        atm_strike=6000,
        feature_names=OPTION_FEATURE_NAMES,
        live_contract=True,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )

    assert not np.isclose(values["iv"], 0.90)
    assert not np.isclose(values["theta"], -0.90)


def test_live_contract_does_not_fall_back_to_vendor_greeks_when_repair_fails() -> None:
    values = option_feature_values(
        {
            "bid": 0.9,
            "ask": 1.1,
            "mid": 1.0,
            "bid_size": 10,
            "ask_size": 10,
            "underlying_price": 6500.0,
            "settlement_time_utc": "2026-01-02T21:00:00Z",
            "iv": 0.20,
            "delta": 0.99,
            "gamma": 0.001,
            "theta": -0.10,
            "prefer_repaired_greeks": True,
        },
        decision_time=pd.Timestamp("2026-01-02T15:00:00Z"),
        spx=6500.0,
        strike=6400.0,
        right="C",
        atm_strike=6500,
        feature_names=OPTION_FEATURE_NAMES,
        live_contract=True,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )

    assert all(np.isnan(values[name]) for name in ("iv", "delta", "gamma", "theta"))
    assert not candidate_is_tradable_values(values)


def test_live_and_historical_structure_features_match_same_canonical_minutes(tmp_path) -> None:
    session = "2026-01-02"
    spx_dir = tmp_path / "spx"
    vix_dir = tmp_path / "vix"
    spx_dir.mkdir()
    vix_dir.mkdir()
    start = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
    rows = []
    vix_rows = []
    for idx in range(75):
        ts = start + pd.Timedelta(minutes=idx)
        close = 6800.0 + idx * 0.65 + np.sin(idx / 3.0) * 3.0
        high = close + 1.25 + (idx % 4) * 0.2
        low = close - 1.10 - (idx % 3) * 0.15
        rows.append(
            {
                "event_time": ts.isoformat(),
                "open": close - 0.25,
                "high": high,
                "low": low,
                "close": close,
                "volume": 0.0,
            }
        )
        vix_rows.append(
            {
                "event_time": ts.isoformat(),
                "open": 18.0 + idx * 0.01,
                "high": 18.0 + idx * 0.01,
                "low": 18.0 + idx * 0.01,
                "close": 18.0 + idx * 0.01,
                "volume": 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(spx_dir / "spx.csv", index=False)
    pd.DataFrame(vix_rows).to_csv(vix_dir / "vix.csv", index=False)

    cache = MarketStructureCache(
        source="index_bars",
        index_spx_dir=spx_dir,
        index_vix_dir=vix_dir,
        decision_context_lag_minutes=1,
    )
    live_state = LiveIndexState()
    for row, vix_row in zip(rows, vix_rows):
        minute = pd.Timestamp(row["event_time"])
        live_state.add(timestamp=minute, spx=row["open"], vix=vix_row["close"])
        live_state.add(timestamp=minute + pd.Timedelta(seconds=15), spx=row["high"], vix=vix_row["close"])
        live_state.add(timestamp=minute + pd.Timedelta(seconds=30), spx=row["low"], vix=vix_row["close"])
        live_state.add(timestamp=minute + pd.Timedelta(seconds=45), spx=row["close"], vix=vix_row["close"])

    for decision_time in (
        pd.Timestamp(f"{session} 09:45", tz="America/New_York").to_pydatetime(),
        pd.Timestamp(f"{session} 10:01", tz="America/New_York").to_pydatetime(),
        pd.Timestamp(f"{session} 10:44", tz="America/New_York").to_pydatetime(),
    ):
        historical = cache.features_for(decision_time)
        live = live_state.structure_features(decision_time)
        assert np.allclose(live, historical, atol=1e-5)


def test_live_and_historical_market_windows_match_same_canonical_minutes() -> None:
    session = "2026-01-02"
    start = pd.Timestamp(f"{session} 09:30", tz="America/New_York").tz_convert("UTC")
    rows = []
    vix_rows = []
    live_state = LiveIndexState()
    for idx in range(80):
        ts = start + pd.Timedelta(minutes=idx)
        close = 6800.0 + idx * 0.55 + np.cos(idx / 5.0) * 2.0
        vix_close = 18.5 + idx * 0.015
        rows.append(
            {
                "event_time": ts,
                "symbol": "SPX",
                "close": close,
                "volume": 0.0,
            }
        )
        vix_rows.append(
            {
                "event_time": ts,
                "symbol": "VIX",
                "close": vix_close,
                "volume": 0.0,
            }
        )
        live_state.add(timestamp=ts, spx=close, vix=vix_close)

    config = NeuralDatasetConfig(feature_contract="protocol101-live-v1")
    spx = pd.DataFrame(rows)
    vix = pd.DataFrame(vix_rows)
    decision_time = pd.Timestamp(f"{session} 10:45", tz="America/New_York")
    historical_context_time = decision_time.tz_convert("UTC") - pd.Timedelta(minutes=1)

    historical = _market_window(
        spx,
        vix,
        historical_context_time,
        config,
        session_only=True,
    )
    live = live_state.market_window(decision_time)

    assert np.allclose(live, historical, atol=1e-5)
