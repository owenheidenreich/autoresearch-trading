"""Tests for SPXW 0DTE neural decision rows."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa

from v4.dataset.spxw_0dte_neural import (
    LabelPolicy,
    NeuralDatasetConfig,
    _policy_exit_deadline,
    _market_window,
    build_neural_dataset,
)
from v4.live.protocol101_feature_contract import FEATURE_CONTRACT_VERSION
from v4.schema.normalized import NORMALIZED_SCHEMA
from v4.schema.types import SCHEMA_VERSION, VendorSource


def _record(
    *,
    ts: datetime,
    right: str = "C",
    root: str = "SPXW",
    settlement_style: str = "PM",
    strike: Decimal = Decimal("6500.000000"),
    bid: float = 2.90,
    ask: float = 3.10,
    iv: float = 0.20,
    delta: float | None = None,
    gamma: float = 0.04,
    theta: float = -0.10,
) -> dict:
    mid = (bid + ask) / 2.0
    strike_int = int(strike)
    return {
        "event_time": ts,
        "receive_time": ts,
        "timestamp_source": "test",
        "contract_id": f"{root}-20260102-{strike_int:05d}.000-{right}",
        "raw_symbol": f"{root}  260102{right}{strike_int * 1000:08d}",
        "instrument_id": 1,
        "root": root,
        "expiry": date(2026, 1, 2),
        "strike": strike,
        "right": right,
        "settlement_style": settlement_style,
        "settlement_time_utc": datetime(2026, 1, 2, 21, 0, tzinfo=timezone.utc),
        "min_price_increment": 0.05,
        "contract_multiplier": 100,
        "bid": bid,
        "ask": ask,
        "bid_size": 10,
        "ask_size": 10,
        "mid": mid,
        "last_trade": mid,
        "last_trade_size": 1,
        "quote_time": ts,
        "last_trade_time": None,
        "quote_age_ms": 0,
        "quote_gap_seconds": None,
        "open_interest": 100,
        "open_interest_asof_date": date(2026, 1, 2),
        "stat_open_interest": 100,
        "volume": 20,
        "volume_asof_time": ts,
        "option_ohlcv_volume": 20,
        "underlying_price": 6500.0,
        "iv": iv,
        "iv_source": "black_scholes_v4",
        "delta": delta if delta is not None else (0.50 if right == "C" else -0.50),
        "gamma": gamma,
        "theta": theta,
        "vega": 0.0,
        "rho": 0.0,
        "charm": None,
        "vanna": None,
        "vomma": None,
        "greek_source": "black_scholes_v4",
        "greek_computation_ts": ts,
        "risk_free_rate_used": 0.05,
        "dividend_yield_used": 0.0,
        "vendor_source": VendorSource.DATABENTO_OPRA.value,
        "ingest_run_id": "test-run",
        "schema_version": SCHEMA_VERSION,
    }


def _table(records: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(records, schema=NORMALIZED_SCHEMA)


def _spx_bars() -> pd.DataFrame:
    times = pd.date_range("2026-01-02T14:30:00Z", periods=5, freq="min")
    return pd.DataFrame(
        {
            "event_time": times,
            "symbol": ["SPX"] * len(times),
            "close": [6498.0, 6500.0, 6501.0, 6502.0, 6503.0],
            "volume": [0] * len(times),
        }
    )


def _spx_bars_late() -> pd.DataFrame:
    times = pd.to_datetime(
        [
            "2026-01-02T20:27:00Z",
            "2026-01-02T20:28:00Z",
            "2026-01-02T20:29:00Z",
            "2026-01-02T20:30:00Z",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "event_time": times,
            "symbol": ["SPX"] * len(times),
            "close": [6500.0, 6500.0, 6500.0, 6500.0],
            "volume": [0] * len(times),
        }
    )


def _spx_bars_for(start: str, periods: int, close: float = 6500.0) -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "event_time": times,
            "symbol": ["SPX"] * len(times),
            "close": [close] * len(times),
            "volume": [0] * len(times),
        }
    )


def _row_at(rows: list[dict], timestamp: datetime) -> dict:
    wanted = pd.Timestamp(timestamp)
    for row in rows:
        if pd.Timestamp(row["decision_time"]) == wanted:
            return row
    raise AssertionError(f"missing decision row {wanted.isoformat()}")


def test_live_contract_market_window_does_not_backfill_from_prior_session() -> None:
    config = NeuralDatasetConfig(market_window_minutes=2)
    spx = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-01-01T20:59:00Z", "2026-01-02T14:30:00Z"]),
            "close": [6400.0, 6500.0],
            "volume": [0.0, 0.0],
        }
    )
    vix = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-01-01T20:59:00Z", "2026-01-02T14:30:00Z"]),
            "close": [18.0, 17.0],
            "volume": [0.0, 0.0],
        }
    )

    legacy = _market_window(spx, vix, pd.Timestamp("2026-01-02T14:30:00Z"), config)
    live_style = _market_window(
        spx,
        vix,
        pd.Timestamp("2026-01-02T14:30:00Z"),
        config,
        session_only=True,
    )

    assert legacy[0, 0] == 6400.0
    assert np.isnan(live_style[0, 0])
    assert live_style[-1, 0] == 6500.0


def test_build_neural_dataset_uses_spxw_five_point_ladder_and_executable_labels() -> None:
    t0 = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc)
    table = _table(
        [
            _record(ts=t0, right="C", bid=2.90, ask=3.10),
            _record(ts=t0, right="P", bid=2.80, ask=3.00),
            _record(ts=t1, right="C", bid=5.00, ask=5.20),
            _record(ts=t1, right="P", bid=1.70, ask=1.90),
        ]
    )
    config = NeuralDatasetConfig(
        market_window_minutes=2,
        label_policies=(LabelPolicy(0.50, 0.50, 5),),
    )

    rows = build_neural_dataset(table, _spx_bars(), config=config)

    assert rows
    row0 = rows[0]
    assert row0["atm_strike"] == 6500
    assert np.all(row0["strike_offsets"] % 5 == 0)
    assert row0["candidate_mask"].any()
    assert row0["contract_ids"][10, 0].startswith("SPXW-20260102")
    assert row0["labels_net_pnl"][10, 0, 0] == 190.0
    assert row0["labels_net_pnl"][10, 0, 0] != row0["labels_mid_pnl"][10, 0, 0]
    assert row0["decision_grid"] == "calendar_v2"
    assert row0["context_required_minutes"] == 2.0
    assert row0["context_minute_rows"] == 2
    assert row0["context_span_minutes"] == 1.0
    assert row0["context_ready"] is True
    assert rows[1]["context_ready"] is True


def test_calendar_grid_emits_empty_mask_row_for_filter_failing_quote_minute() -> None:
    t0 = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, 14, 33, tzinfo=timezone.utc)
    table = _table(
        [
            _record(ts=t0, right="C", bid=2.90, ask=3.10),
            _record(ts=t0, right="P", bid=2.80, ask=3.00),
            _record(ts=t1, right="C", bid=2.00, ask=3.50),
            _record(ts=t1, right="P", bid=2.00, ask=3.50),
            _record(ts=t2, right="C", bid=5.00, ask=5.20),
            _record(ts=t2, right="P", bid=1.70, ask=1.90),
        ]
    )
    config = NeuralDatasetConfig(market_window_minutes=2)

    rows = build_neural_dataset(table, _spx_bars_for("2026-01-02T14:30:00Z", 4), config=config)
    row0 = _row_at(rows, t0)
    row1 = _row_at(rows, t1)
    row2 = _row_at(rows, t2)

    assert len(rows) == 360
    assert row0["candidate_mask"].any()
    assert row1["decision_grid"] == "calendar_v2"
    assert row1["candidate_mask"].sum() == 0
    assert np.isnan(row1["option_ladder"]).all()
    assert row1["contract_quote_metadata"]
    assert np.isnan(row1["labels_net_pnl"]).all()
    assert row2["candidate_mask"].any()


def test_calendar_grid_uses_early_close_last_decision() -> None:
    t0 = datetime(2024, 11, 29, 14, 31, tzinfo=timezone.utc)
    table = _table([_record(ts=t0, right="C", bid=2.90, ask=3.10)])

    rows = build_neural_dataset(
        table,
        _spx_bars_for("2024-11-29T14:30:00Z", 210),
        config=NeuralDatasetConfig(market_window_minutes=2),
    )

    assert len(rows) == 209
    assert pd.Timestamp(rows[0]["decision_time"]).isoformat() == "2024-11-29T14:31:00+00:00"
    assert pd.Timestamp(rows[-1]["decision_time"]).isoformat() == "2024-11-29T17:59:00+00:00"


def test_label_deadline_caps_policy_holds_at_forced_flat() -> None:
    config = NeuralDatasetConfig()
    decision_time = pd.Timestamp("2026-01-02T20:29:00Z")

    deadlines = [
        _policy_exit_deadline(decision_time, policy, config).tz_convert("America/New_York").time()
        for policy in config.label_policies
    ]

    assert deadlines == [pd.Timestamp("2026-01-02T15:39:00", tz="America/New_York").time(), pd.Timestamp("2026-01-02T15:54:00", tz="America/New_York").time(), config.forced_flat_before]


def test_policy2_late_label_uses_forced_flat_not_post_forced_quote() -> None:
    t0 = datetime(2026, 1, 2, 20, 29, tzinfo=timezone.utc)
    forced = datetime(2026, 1, 2, 20, 55, tzinfo=timezone.utc)
    after_forced = datetime(2026, 1, 2, 20, 59, tzinfo=timezone.utc)
    table = _table(
        [
            _record(ts=t0, right="C", bid=2.90, ask=3.10),
            _record(ts=forced, right="C", bid=4.00, ask=4.20),
            _record(ts=after_forced, right="C", bid=100.00, ask=100.20),
        ]
    )
    config = NeuralDatasetConfig(
        market_window_minutes=2,
        max_spread_abs=1.0,
        label_policies=(LabelPolicy(0.65, 1.50, 45),),
    )

    rows = build_neural_dataset(table, _spx_bars_late(), config=config)

    assert rows
    row0 = _row_at(rows, t0)
    strike_idx = list(row0["strike_offsets"]).index(0)
    assert np.isclose(row0["labels_net_pnl"][strike_idx, 0, 0], 90.0)


def test_build_neural_dataset_rejects_am_or_non_spxw_rows() -> None:
    t0 = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    table = _table(
        [
            _record(ts=t0, root="SPX", settlement_style="AM"),
            _record(ts=t0, root="SPXW", settlement_style="AM"),
        ]
    )

    rows = build_neural_dataset(table, _spx_bars())

    assert rows == []


def test_build_neural_dataset_repairs_missing_greeks_from_executable_ask() -> None:
    t0 = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc)
    table = _table(
        [
            _record(
                ts=t0,
                right="C",
                strike=Decimal("6480.000000"),
                bid=19.80,
                ask=20.30,
                iv=np.nan,
                delta=np.nan,
                gamma=np.nan,
                theta=np.nan,
            ),
            _record(
                ts=t1,
                right="C",
                strike=Decimal("6480.000000"),
                bid=21.00,
                ask=21.50,
                iv=np.nan,
                delta=np.nan,
                gamma=np.nan,
                theta=np.nan,
            ),
        ]
    )
    config = NeuralDatasetConfig(
        market_window_minutes=2,
        max_spread_abs=1.0,
        label_policies=(LabelPolicy(0.50, 0.50, 5),),
    )

    rows = build_neural_dataset(table, _spx_bars(), config=config)

    assert rows
    row0 = rows[0]
    strike_idx = list(row0["strike_offsets"]).index(-20)
    call_idx = 0
    feature_idx = {name: idx for idx, name in enumerate(row0["feature_names"])}
    assert row0["candidate_mask"][strike_idx, call_idx]
    assert np.isfinite(row0["option_ladder"][strike_idx, call_idx, feature_idx["iv"]])
    assert np.isfinite(row0["option_ladder"][strike_idx, call_idx, feature_idx["gamma"]])


def test_build_neural_dataset_live_contract_emits_causal_metadata_and_zeroes_unavailable_fields() -> None:
    t0 = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc)
    table = _table(
        [
            _record(ts=t0, right="C", bid=2.90, ask=3.10),
            _record(ts=t1, right="C", bid=5.00, ask=5.20),
        ]
    )
    config = NeuralDatasetConfig(
        market_window_minutes=2,
        feature_contract=FEATURE_CONTRACT_VERSION,
        compute_policy_labels=False,
        label_policies=(LabelPolicy(0.50, 0.50, 5),),
    )

    rows = build_neural_dataset(table, _spx_bars(), config=config)

    assert rows
    row0 = rows[0]
    feature_idx = {name: idx for idx, name in enumerate(row0["feature_names"])}
    strike_idx = list(row0["strike_offsets"]).index(0)
    assert row0["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert row0["source_quote_time"].isoformat() == "2026-01-02T14:31:00+00:00"
    assert row0["source_context_time"].isoformat() == "2026-01-02T14:30:00+00:00"
    assert row0["market_window"][-1, 0] == 6498.0
    assert row0["option_ladder"][strike_idx, 0, feature_idx["option_ohlcv_volume"]] == 0.0
    assert row0["option_ladder"][strike_idx, 0, feature_idx["stat_open_interest"]] == 0.0
    assert row0["labels_net_pnl"].shape == (21, 2, 1)
    assert np.count_nonzero(row0["labels_net_pnl"]) == 0
    assert np.count_nonzero(row0["labels_mid_pnl"]) == 0
    metadata = row0["contract_quote_metadata"][row0["contract_ids"][strike_idx, 0]]
    assert metadata["feature_contract_version"] == FEATURE_CONTRACT_VERSION


def test_build_neural_dataset_live_contract_can_compute_offline_training_labels() -> None:
    t0 = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc)
    table = _table(
        [
            _record(ts=t0, right="C", bid=2.90, ask=3.10),
            _record(ts=t1, right="C", bid=5.00, ask=5.20),
        ]
    )
    config = NeuralDatasetConfig(
        market_window_minutes=2,
        feature_contract=FEATURE_CONTRACT_VERSION,
        compute_policy_labels=True,
        label_policies=(LabelPolicy(0.50, 0.50, 5),),
    )

    rows = build_neural_dataset(table, _spx_bars(), config=config)

    assert rows
    row0 = rows[0]
    feature_idx = {name: idx for idx, name in enumerate(row0["feature_names"])}
    strike_idx = list(row0["strike_offsets"]).index(0)
    assert row0["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert row0["option_ladder"][strike_idx, 0, feature_idx["option_ohlcv_volume"]] == 0.0
    assert row0["option_ladder"][strike_idx, 0, feature_idx["stat_open_interest"]] == 0.0
    assert row0["labels_net_pnl"].shape == (21, 2, 1)
    assert row0["labels_net_pnl"][strike_idx, 0, 0] > 0.0


def test_live_contract_diagnostic_context_lag_override_is_marked_and_isolated() -> None:
    t0 = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc)
    table = _table(
        [
            _record(ts=t0, right="C", bid=2.90, ask=3.10),
            _record(ts=t1, right="C", bid=5.00, ask=5.20),
        ]
    )
    config = NeuralDatasetConfig(
        market_window_minutes=2,
        feature_contract=FEATURE_CONTRACT_VERSION,
        compute_policy_labels=False,
        diagnostic_index_context_lag_minutes=0,
        diagnostic_source_policy="test_same_minute_context",
    )

    rows = build_neural_dataset(table, _spx_bars(), config=config)

    assert rows
    row0 = rows[0]
    assert row0["source_context_time"].isoformat() == "2026-01-02T14:31:00+00:00"
    assert row0["market_window"][-1, 0] == 6500.0
    assert row0["feature_contract"]["version"] == FEATURE_CONTRACT_VERSION
    assert row0["feature_contract"]["diagnostic_only"] is True
    assert row0["feature_contract"]["diagnostic_index_context_lag_minutes"] == 0
    assert row0["feature_contract"]["diagnostic_source_policy"] == "test_same_minute_context"
    assert row0["feature_contract"]["production_contract_mutation"] is False


def test_build_neural_dataset_emits_empty_row_when_greeks_cannot_be_repaired() -> None:
    t0 = datetime(2026, 1, 2, 14, 31, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 2, 14, 32, tzinfo=timezone.utc)
    table = _table(
        [
            _record(
                ts=t0,
                right="C",
                strike=Decimal("6480.000000"),
                bid=18.80,
                ask=19.20,
                iv=np.nan,
                delta=np.nan,
                gamma=np.nan,
                theta=np.nan,
            ),
            _record(
                ts=t1,
                right="C",
                strike=Decimal("6480.000000"),
                bid=18.90,
                ask=19.30,
                iv=np.nan,
                delta=np.nan,
                gamma=np.nan,
                theta=np.nan,
            ),
        ]
    )
    config = NeuralDatasetConfig(market_window_minutes=2, max_spread_abs=1.0)

    rows = build_neural_dataset(table, _spx_bars(), config=config)

    assert len(rows) == 360
    row0 = _row_at(rows, t0)
    assert row0["decision_grid"] == "calendar_v2"
    assert row0["candidate_mask"].sum() == 0
    assert np.isnan(row0["option_ladder"]).all()
    assert np.isnan(row0["labels_net_pnl"]).all()
