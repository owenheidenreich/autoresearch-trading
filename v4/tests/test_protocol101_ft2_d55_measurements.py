from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v4.scripts import run_protocol101_ft2_d55_measurements as d55


def test_census_v4_premium_band_boundaries() -> None:
    assert d55.premium_band(1.0) == "cheap_le_1"
    assert d55.premium_band(1.01) == "small_1_3"
    assert d55.premium_band(3.0) == "small_1_3"
    assert d55.premium_band(3.01) == "medium_3_8"
    assert d55.premium_band(8.0) == "medium_3_8"
    assert d55.premium_band(8.01) == "large_8_20"
    assert d55.premium_band(20.0) == "large_8_20"
    assert d55.premium_band(20.01) == "very_large_20p"


def test_join_contract_minute_detects_hidden_dip_and_fill_bias() -> None:
    timestamp = pd.Timestamp("2025-07-01T14:31:00Z")
    minute = pd.DataFrame(
        {
            "raw_symbol": ["SPXW  250701C06200000"],
            "bar_time": [timestamp],
            "bid": [1.00],
            "ask": [1.20],
            "minute_mid": [1.10],
            "minute_quote_age_is_zero": [True],
            "minute_quote_gap_is_null": [True],
            "premium_band": ["small_1_3"],
            "moneyness_band": ["atm"],
        }
    )
    one_second = pd.DataFrame(
        {
            "raw_symbol": ["SPXW  250701C06200000"],
            "bar_time": [timestamp],
            "stream_rows": [3],
            "bbo_change_count": [2],
            "one_second_ts_event_nonnull": [0],
            "executable_1s_rows": [3],
            "bid_first": [1.00],
            "bid_last": [1.00],
            "bid_min": [0.80],
            "bid_max": [1.00],
            "bid_mean": [0.90],
            "bid_median": [0.90],
            "bid_std": [0.10],
            "ask_first": [1.20],
            "ask_last": [1.30],
            "ask_min": [1.20],
            "ask_max": [1.40],
            "ask_mean": [1.30],
            "ask_median": [1.30],
            "ask_std": [0.10],
            "mid_first": [1.10],
            "mid_last": [1.15],
            "mid_min": [0.95],
            "mid_max": [1.20],
            "spread_first": [0.20],
            "spread_last": [0.30],
            "spread_mean": [0.30],
            "spread_std": [0.10],
            "realized_variance": [0.01],
            "one_second_realized_volatility": [0.10],
            "intraminute_mid_return": [1.15 / 1.10 - 1.0],
            "intraminute_mid_range_fraction": [(1.20 - 0.95) / 1.10],
            "spread_change": [0.10],
        }
    )
    joined = d55.join_contract_minutes(minute, one_second)
    row = joined.iloc[0]
    assert bool(row["hidden_adverse_dip"])
    assert bool(row["breach_and_recover"])
    assert np.isclose(row["adverse_gap_points"], 0.20)
    assert np.isclose(row["entry_error_mean"], -0.10)
    assert bool(row["entry_minute_fill_optimistic"])
    assert np.isclose(row["exit_error_mean"], 0.10)
    assert bool(row["exit_minute_fill_optimistic"])


def test_floor_response_detects_hidden_cross_and_first_gap() -> None:
    symbol = "SPXW  250701C06200000"
    first = pd.Timestamp("2025-07-01T14:31:00Z")
    second = pd.Timestamp("2025-07-01T14:32:00Z")
    joined = pd.DataFrame(
        {
            "raw_symbol": [symbol, symbol],
            "bar_time": [first, second],
            "premium_band": ["small_1_3", "small_1_3"],
            "moneyness_band": ["atm", "atm"],
            "bid": [1.00, 0.95],
            "bid_min": [0.90, 0.85],
            "prior_minute_bid": [np.nan, 1.00],
            "prior_is_consecutive": [False, True],
        }
    )
    one_second = pd.DataFrame(
        {
            "symbol": [symbol, symbol],
            "bar_time": [second, second],
            "ts_recv": [
                pd.Timestamp("2025-07-01T14:31:10Z"),
                pd.Timestamp("2025-07-01T14:31:20Z"),
            ],
            "bid_px_00": [0.89, 0.85],
            "executable_bbo": [True, True],
        }
    )
    floor = d55.floor_measurements(joined, one_second)
    row = floor[np.isclose(floor["floor_ratio"], 0.90)].iloc[0]
    assert bool(row["intraminute_cross"])
    assert not bool(row["completed_minute_cross"])
    assert bool(row["hidden_cross_recover"])
    assert np.isclose(row["first_cross_bid"], 0.89)
    assert np.isclose(row["floor_trigger_gap_points"], 0.01)


def test_floor_response_counts_exact_floor_touch_as_cross() -> None:
    symbol = "SPXW  250701C06200000"
    second = pd.Timestamp("2025-07-01T14:32:00Z")
    joined = pd.DataFrame(
        {
            "raw_symbol": [symbol],
            "bar_time": [second],
            "premium_band": ["small_1_3"],
            "moneyness_band": ["atm"],
            "bid": [0.95],
            "bid_min": [0.90],
            "prior_minute_bid": [1.00],
            "prior_is_consecutive": [True],
        }
    )
    one_second = pd.DataFrame(
        {
            "symbol": [symbol],
            "bar_time": [second],
            "ts_recv": [pd.Timestamp("2025-07-01T14:31:10Z")],
            "bid_px_00": [0.90],
            "executable_bbo": [True],
        }
    )

    row = d55.floor_measurements(joined, one_second).query(
        "floor_ratio == 0.90"
    ).iloc[0]

    assert bool(row["intraminute_cross"])
    assert not bool(row["completed_minute_cross"])
    assert bool(row["hidden_cross_recover"])
    assert np.isclose(row["floor_trigger_gap_points"], 0.0)


def test_trust_decision_accepts_zero_rates() -> None:
    overall = {
        "executable_1s_coverage_rate": 1.0,
        "forward_fill_rate": 0.0,
        "hidden_adverse_dip_rate": 0.0,
        "adverse_gap_p95_points": 0.0,
        "entry_fill_abs_error_p95_points": 0.0,
        "exit_fill_abs_error_p95_points": 0.0,
    }
    floor = pd.DataFrame(
        [
            {
                "floor_ratio": 0.90,
                "hidden_cross_recover_rate": 0.0,
                "floor_trigger_gap_p95_points": 0.0,
            }
        ]
    )
    result = d55.decision(overall, floor)
    assert result["all_checks_pass"]
    assert result["minute_resolution_exit_floor_validation"] == "TRUSTWORTHY"


def test_streamed_floor_pool_matches_direct_grouping(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "floor_ratio": [0.9, 0.9, 0.9, 0.9],
            "premium_band": ["small_1_3"] * 4,
            "moneyness_band": ["atm"] * 4,
            "intraminute_cross": [True, False, True, False],
            "completed_minute_cross": [False, False, True, False],
            "hidden_cross_recover": [True, False, False, False],
            "floor_trigger_gap_points": [0.1, np.nan, 0.3, np.nan],
            "floor_trigger_gap_dollars": [10.0, np.nan, 30.0, np.nan],
        }
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    frame.iloc[:2].to_parquet(first, index=False)
    frame.iloc[2:].to_parquet(second, index=False)

    overall, bands = d55.pooled_floor_summaries([first, second])
    direct_overall = d55.grouped_floor_summary(frame, ["floor_ratio"])
    direct_bands = d55.grouped_floor_summary(
        frame,
        ["floor_ratio", "premium_band", "moneyness_band"],
    )

    for streamed, direct in (
        (overall.iloc[0], direct_overall.iloc[0]),
        (bands.iloc[0], direct_bands.iloc[0]),
    ):
        assert streamed["eligible_contract_minutes"] == direct[
            "eligible_contract_minutes"
        ]
        assert np.isclose(
            streamed["hidden_cross_recover_rate"],
            direct["hidden_cross_recover_rate"],
        )
        assert np.isclose(
            streamed["floor_trigger_gap_p95_points"],
            direct["floor_trigger_gap_p95_points"],
        )
