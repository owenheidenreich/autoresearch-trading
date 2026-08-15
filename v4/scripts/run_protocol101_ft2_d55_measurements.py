"""Measure minute-vs-one-second distortion for the FT2-D55 pilot.

The runner consumes only the owner-authorized D55 raw files and already-owned
normalized minute files.  It performs descriptive measurement; it does not
fit a model, tune a policy, score protected data, or modify a signed contract.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from v4.model.protocol101_divergence_noise import moneyness_band
from v4.scripts.run_protocol101_ft2_d55_acquisition import (
    AUTHORITY_PATH,
    AUTHORITY_SHA256,
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_RAW_ROOT,
    GOAL,
    SCHEMA,
    sha256_file,
    utc_now,
    write_json,
)


NY_TZ = "America/New_York"
FLOOR_RATIOS = (0.70, 0.80, 0.90, 0.95)
MARKET_FIRST_BAR_ET = 9 * 60 + 31
MARKET_LAST_BAR_ET = 15 * 60 + 55
CONTRACT_MULTIPLIER = 100.0
TRUST_THRESHOLDS = {
    "min_executable_1s_coverage_rate": 0.95,
    "max_forward_fill_rate": 0.05,
    "max_hidden_adverse_dip_rate": 0.05,
    "max_p95_adverse_gap_points": 0.25,
    "max_p95_abs_entry_fill_error_points": 0.25,
    "max_p95_abs_exit_fill_error_points": 0.25,
    "floor_ratio_for_gate": 0.90,
    "max_hidden_floor_cross_rate": 0.05,
    "max_p95_floor_trigger_gap_points": 0.25,
}
DISTORTION_ANALYSIS_COLUMNS = [
    "session",
    "premium_band",
    "moneyness_band",
    "stream_rows",
    "executable_1s_rows",
    "has_1s_stream",
    "has_executable_1s",
    "forward_fill",
    "hidden_adverse_dip",
    "breach_and_recover",
    "adverse_gap_points",
    "entry_error_mean",
    "entry_minute_fill_optimistic",
    "ask_std",
    "exit_error_mean",
    "exit_minute_fill_optimistic",
    "bid_std",
    "last_1s_bid_diff",
    "last_1s_ask_diff",
    "minute_quote_age_is_zero",
    "minute_quote_gap_is_null",
    "intraminute_mid_return",
    "one_second_realized_volatility",
    "intraminute_mid_range_fraction",
    "bbo_change_count",
    "spread_change",
    "next_minute_mid_return",
    "next_minute_bid_over_current_ask_return",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Reuse hash-verified per-session measurements without raw recompute.",
    )
    return parser.parse_args()


def premium_band(ask: float | int | None) -> str:
    try:
        value = float(ask)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value) or value <= 0:
        return "unknown"
    if value <= 1.0:
        return "cheap_le_1"
    if value <= 3.0:
        return "small_1_3"
    if value <= 8.0:
        return "medium_3_8"
    if value <= 20.0:
        return "large_8_20"
    return "very_large_20p"


def measurement_spec() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101FT2D55MeasurementSpecV1",
        "goal": GOAL,
        "authority": {
            "path": str(AUTHORITY_PATH),
            "sha256": AUTHORITY_SHA256,
        },
        "scope": {
            "descriptive_only": True,
            "model_training_or_fitting": False,
            "signed_contract_changes": False,
            "graph_changes": False,
            "census_changes": False,
        },
        "time_alignment": {
            "minute_bar_timestamp": "T",
            "one_second_interval": "(T-60 seconds, T]",
            "implementation": "bar_time=ts_recv.ceil('min')",
            "market_bar_end_window_et": "[09:31,15:55]",
            "reason": (
                "Databento cbbo-1m snapshots are interval-end stamped; this is "
                "the existing project CBBO-1m-vs-1s alignment convention."
            ),
        },
        "tradable_contract_minute": (
            "SPXW PM-settled 0DTE normalized row in the governed market window "
            "with finite bid>0, finite ask>0, and ask>=bid"
        ),
        "bands": {
            "premium": {
                "cheap_le_1": "0<ask<=1",
                "small_1_3": "1<ask<=3",
                "medium_3_8": "3<ask<=8",
                "large_8_20": "8<ask<=20",
                "very_large_20p": "ask>20",
            },
            "moneyness": {
                "atm": "abs(strike-canonical_ATM)<=10",
                "near": "10<abs(strike-canonical_ATM)<=25",
                "wing": "abs(strike-canonical_ATM)>25",
            },
            "canonical_ATM": (
                "np.rint(underlying_price/5)*5, matching the project "
                "round-half-to-even five-point rule"
            ),
        },
        "measurements": {
            "forward_fill": (
                "No price-or-size BBO state change in the one-second stream "
                "during the interval; a missing interval also counts stale."
            ),
            "hidden_adverse_excursion": (
                "minute_bid - minimum positive executable one-second bid; a "
                "breach-and-recover has min_1s_bid<minute_bid and "
                "last_1s_bid>=minute_bid"
            ),
            "fill_price_distortion": {
                "entry_error": (
                    "minute assumed ask minus one-second ask; negative is "
                    "optimistic for a buyer"
                ),
                "exit_error": (
                    "minute assumed bid minus one-second bid; positive is "
                    "optimistic for a seller"
                ),
                "population": (
                    "equal-weight contract-minutes, using first/mean/median/"
                    "worst executable observations within the 60-second interval"
                ),
            },
            "floor_slippage": {
                "limitation": (
                    "The signed design has no instantiated forecast-to-floor "
                    "equation yet. D55 therefore measures a preregistered "
                    "one-step committed-floor response surface, not a policy."
                ),
                "committed_floor": (
                    "prior completed-minute executable bid times floor_ratio"
                ),
                "floor_ratios": list(FLOOR_RATIOS),
                "hidden_cross": (
                    "one-second bid crosses the committed floor but the "
                    "completed-minute bid does not"
                ),
                "gap_loss": (
                    "committed floor minus first observed executable bid at or "
                    "below the floor"
                ),
            },
            "entry_signal_exploratory": {
                "features": [
                    "intraminute_mid_return",
                    "one_second_realized_volatility",
                    "intraminute_mid_range_fraction",
                    "bbo_change_count",
                    "spread_change",
                ],
                "targets": [
                    "next_minute_mid_return",
                    "next_minute_bid_over_current_ask_return",
                ],
                "statistics": [
                    "Spearman correlation",
                    "top-minus-bottom feature-quintile target mean",
                ],
                "claim": "descriptive only; not a fitted model or entry-edge claim",
            },
        },
        "trust_decision_thresholds": TRUST_THRESHOLDS,
        "cost_context_usd": {
            "pilot_reference": 24.12,
            "all_available_1s_history": 280,
            "minute_backfill_low": 41,
            "minute_backfill_high": 347,
        },
    }


def _market_mask(timestamp: pd.Series) -> pd.Series:
    local = timestamp.dt.tz_convert(NY_TZ)
    minute = local.dt.hour * 60 + local.dt.minute
    return minute.between(MARKET_FIRST_BAR_ET, MARKET_LAST_BAR_ET)


def load_minute_session(normalized_dir: Path, session: str) -> pd.DataFrame:
    path = normalized_dir / f"databento_spxw_0dte_{session}_derived_context.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(
        path,
        columns=[
            "quote_time",
            "raw_symbol",
            "root",
            "expiry",
            "settlement_style",
            "strike",
            "right",
            "bid",
            "ask",
            "underlying_price",
            "quote_age_ms",
            "quote_gap_seconds",
        ],
    )
    frame["bar_time"] = pd.to_datetime(
        frame["quote_time"], utc=True, errors="coerce"
    )
    frame["raw_symbol"] = frame["raw_symbol"].astype(str)
    frame["bid"] = pd.to_numeric(frame["bid"], errors="coerce")
    frame["ask"] = pd.to_numeric(frame["ask"], errors="coerce")
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    frame["underlying_price"] = pd.to_numeric(
        frame["underlying_price"], errors="coerce"
    )
    valid_identity = (
        frame["root"].astype(str).eq("SPXW")
        & frame["expiry"].astype(str).eq(session)
        & frame["settlement_style"].astype(str).eq("PM")
    )
    valid_quote = (
        frame["bid"].gt(0)
        & frame["ask"].gt(0)
        & frame["ask"].ge(frame["bid"])
    )
    frame = frame[
        valid_identity
        & valid_quote
        & frame["bar_time"].notna()
        & _market_mask(frame["bar_time"])
    ].copy()
    frame = (
        frame.sort_values(["raw_symbol", "bar_time"])
        .drop_duplicates(["raw_symbol", "bar_time"], keep="last")
        .reset_index(drop=True)
    )
    frame["minute_mid"] = (frame["bid"] + frame["ask"]) / 2.0
    frame["atm_strike"] = np.rint(frame["underlying_price"] / 5.0) * 5.0
    frame["abs_offset"] = (frame["strike"] - frame["atm_strike"]).abs()
    frame["moneyness_band"] = [
        moneyness_band(value) for value in frame["abs_offset"]
    ]
    frame["premium_band"] = [premium_band(value) for value in frame["ask"]]
    frame["quote_age_ms"] = pd.to_numeric(
        frame["quote_age_ms"], errors="coerce"
    )
    frame["quote_gap_seconds"] = pd.to_numeric(
        frame["quote_gap_seconds"], errors="coerce"
    )
    frame["minute_quote_age_is_zero"] = frame["quote_age_ms"].eq(0)
    frame["minute_quote_gap_is_null"] = frame["quote_gap_seconds"].isna()
    return frame


def load_one_second_session(raw_root: Path, session: str) -> pd.DataFrame:
    path = raw_root / session / f"{session}.{SCHEMA}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).reset_index()
    if "ts_recv" not in frame.columns and len(frame.columns):
        frame = frame.rename(columns={frame.columns[0]: "ts_recv"})
    required = {
        "ts_recv",
        "symbol",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    frame["ts_recv"] = pd.to_datetime(
        frame["ts_recv"], utc=True, errors="coerce"
    )
    frame["bar_time"] = frame["ts_recv"].dt.ceil("min")
    frame["symbol"] = frame["symbol"].astype(str)
    for column in ("bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["ts_recv"].notna()
        & _market_mask(frame["bar_time"])
    ].copy()
    if "ts_event" in frame.columns:
        frame["ts_event_nonnull"] = pd.to_datetime(
            frame["ts_event"], utc=True, errors="coerce"
        ).notna()
    else:
        frame["ts_event_nonnull"] = False
    frame = frame.sort_values(["symbol", "ts_recv"]).reset_index(drop=True)
    state_columns = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    state = frame[state_columns].fillna(-9.87654321e99)
    prior = state.groupby(frame["symbol"], sort=False).shift(1)
    frame["bbo_changed"] = state.ne(prior).any(axis=1)
    valid = (
        frame["bid_px_00"].gt(0)
        & frame["ask_px_00"].gt(0)
        & frame["ask_px_00"].ge(frame["bid_px_00"])
    )
    frame["executable_bbo"] = valid
    frame["mid"] = np.where(
        valid,
        (frame["bid_px_00"] + frame["ask_px_00"]) / 2.0,
        np.nan,
    )
    frame["spread"] = np.where(
        valid,
        frame["ask_px_00"] - frame["bid_px_00"],
        np.nan,
    )
    frame["log_mid"] = np.log(frame["mid"].where(frame["mid"] > 0))
    frame["log_mid_change"] = frame.groupby(
        ["symbol", "bar_time"], sort=False
    )["log_mid"].diff()
    frame["squared_log_mid_change"] = frame["log_mid_change"] ** 2
    return frame


def aggregate_one_second(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["symbol", "bar_time"]
    stream = (
        frame.groupby(keys, sort=False)
        .agg(
            stream_rows=("ts_recv", "size"),
            bbo_change_count=("bbo_changed", "sum"),
            one_second_ts_event_nonnull=("ts_event_nonnull", "sum"),
        )
        .reset_index()
    )
    valid = frame[frame["executable_bbo"]].copy()
    if valid.empty:
        return stream.rename(columns={"symbol": "raw_symbol"})
    prices = (
        valid.groupby(keys, sort=False)
        .agg(
            executable_1s_rows=("ts_recv", "size"),
            bid_first=("bid_px_00", "first"),
            bid_last=("bid_px_00", "last"),
            bid_min=("bid_px_00", "min"),
            bid_max=("bid_px_00", "max"),
            bid_mean=("bid_px_00", "mean"),
            bid_median=("bid_px_00", "median"),
            bid_std=("bid_px_00", "std"),
            ask_first=("ask_px_00", "first"),
            ask_last=("ask_px_00", "last"),
            ask_min=("ask_px_00", "min"),
            ask_max=("ask_px_00", "max"),
            ask_mean=("ask_px_00", "mean"),
            ask_median=("ask_px_00", "median"),
            ask_std=("ask_px_00", "std"),
            mid_first=("mid", "first"),
            mid_last=("mid", "last"),
            mid_min=("mid", "min"),
            mid_max=("mid", "max"),
            spread_first=("spread", "first"),
            spread_last=("spread", "last"),
            spread_mean=("spread", "mean"),
            spread_std=("spread", "std"),
            realized_variance=("squared_log_mid_change", "sum"),
        )
        .reset_index()
    )
    out = stream.merge(prices, on=keys, how="left")
    out["one_second_realized_volatility"] = np.sqrt(
        out["realized_variance"].clip(lower=0)
    )
    out["intraminute_mid_return"] = (
        out["mid_last"] / out["mid_first"] - 1.0
    )
    out["intraminute_mid_range_fraction"] = (
        (out["mid_max"] - out["mid_min"]) / out["mid_first"]
    )
    out["spread_change"] = out["spread_last"] - out["spread_first"]
    return out.rename(columns={"symbol": "raw_symbol"})


def join_contract_minutes(
    minute: pd.DataFrame,
    one_second_agg: pd.DataFrame,
) -> pd.DataFrame:
    out = minute.merge(
        one_second_agg,
        on=["raw_symbol", "bar_time"],
        how="left",
        validate="one_to_one",
    )
    out["stream_rows"] = out["stream_rows"].fillna(0).astype(np.int64)
    out["bbo_change_count"] = (
        out["bbo_change_count"].fillna(0).astype(np.int64)
    )
    out["executable_1s_rows"] = (
        out["executable_1s_rows"].fillna(0).astype(np.int64)
    )
    out["has_1s_stream"] = out["stream_rows"].gt(0)
    out["has_executable_1s"] = out["executable_1s_rows"].gt(0)
    out["forward_fill"] = out["bbo_change_count"].eq(0)
    out["adverse_gap_points"] = (
        out["bid"] - out["bid_min"]
    ).clip(lower=0)
    out["hidden_adverse_dip"] = (
        out["has_executable_1s"] & out["bid_min"].lt(out["bid"] - 1e-12)
    )
    out["breach_and_recover"] = (
        out["hidden_adverse_dip"] & out["bid_last"].ge(out["bid"] - 1e-12)
    )
    out["entry_error_first"] = out["ask"] - out["ask_first"]
    out["entry_error_mean"] = out["ask"] - out["ask_mean"]
    out["entry_error_median"] = out["ask"] - out["ask_median"]
    out["entry_error_worst"] = out["ask"] - out["ask_max"]
    out["exit_error_first"] = out["bid"] - out["bid_first"]
    out["exit_error_mean"] = out["bid"] - out["bid_mean"]
    out["exit_error_median"] = out["bid"] - out["bid_median"]
    out["exit_error_worst"] = out["bid"] - out["bid_min"]
    out["entry_minute_fill_optimistic"] = out["entry_error_mean"].lt(0)
    out["exit_minute_fill_optimistic"] = out["exit_error_mean"].gt(0)
    out["last_1s_bid_diff"] = out["bid"] - out["bid_last"]
    out["last_1s_ask_diff"] = out["ask"] - out["ask_last"]
    out = out.sort_values(["raw_symbol", "bar_time"]).reset_index(drop=True)
    groups = out.groupby("raw_symbol", sort=False)
    out["prior_bar_time"] = groups["bar_time"].shift(1)
    out["prior_minute_bid"] = groups["bid"].shift(1)
    out["next_bar_time"] = groups["bar_time"].shift(-1)
    out["next_minute_bid"] = groups["bid"].shift(-1)
    out["next_minute_mid"] = groups["minute_mid"].shift(-1)
    out["prior_is_consecutive"] = (
        out["bar_time"] - out["prior_bar_time"]
    ).eq(pd.Timedelta(minutes=1))
    out["next_is_consecutive"] = (
        out["next_bar_time"] - out["bar_time"]
    ).eq(pd.Timedelta(minutes=1))
    out["next_minute_mid_return"] = np.where(
        out["next_is_consecutive"],
        out["next_minute_mid"] / out["minute_mid"] - 1.0,
        np.nan,
    )
    out["next_minute_bid_over_current_ask_return"] = np.where(
        out["next_is_consecutive"],
        out["next_minute_bid"] / out["ask"] - 1.0,
        np.nan,
    )
    return out


def floor_measurements(
    joined: pd.DataFrame,
    one_second: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = [
        "raw_symbol",
        "bar_time",
        "premium_band",
        "moneyness_band",
        "bid",
        "bid_min",
        "prior_minute_bid",
        "prior_is_consecutive",
    ]
    base = joined[base_columns].copy()
    base = base[
        base["prior_is_consecutive"]
        & base["prior_minute_bid"].gt(0)
        & base["bid_min"].notna()
    ].copy()
    if base.empty:
        return pd.DataFrame()
    valid = one_second[one_second["executable_bbo"]][
        ["symbol", "bar_time", "ts_recv", "bid_px_00"]
    ].rename(
        columns={
            "symbol": "raw_symbol",
            "bid_px_00": "one_second_bid",
        }
    )
    path = valid.merge(
        base[
            [
                "raw_symbol",
                "bar_time",
                "prior_minute_bid",
            ]
        ],
        on=["raw_symbol", "bar_time"],
        how="inner",
    )
    results: list[pd.DataFrame] = []
    for ratio in FLOOR_RATIOS:
        rows = base.copy()
        rows["floor_ratio"] = ratio
        rows["committed_floor"] = rows["prior_minute_bid"] * ratio
        rows["intraminute_cross"] = rows["bid_min"].le(
            rows["committed_floor"]
        )
        rows["completed_minute_cross"] = rows["bid"].le(
            rows["committed_floor"]
        )
        rows["hidden_cross_recover"] = (
            rows["intraminute_cross"] & ~rows["completed_minute_cross"]
        )
        candidate = path[
            path["one_second_bid"].le(path["prior_minute_bid"] * ratio)
        ].copy()
        if not candidate.empty:
            candidate = (
                candidate.sort_values(["raw_symbol", "bar_time", "ts_recv"])
                .drop_duplicates(["raw_symbol", "bar_time"], keep="first")
                .rename(
                    columns={
                        "ts_recv": "first_cross_time",
                        "one_second_bid": "first_cross_bid",
                    }
                )
            )
            rows = rows.merge(
                candidate[
                    [
                        "raw_symbol",
                        "bar_time",
                        "first_cross_time",
                        "first_cross_bid",
                    ]
                ],
                on=["raw_symbol", "bar_time"],
                how="left",
            )
        else:
            rows["first_cross_time"] = pd.NaT
            rows["first_cross_bid"] = np.nan
        rows["floor_trigger_gap_points"] = (
            rows["committed_floor"] - rows["first_cross_bid"]
        ).clip(lower=0)
        rows["floor_trigger_gap_dollars"] = (
            rows["floor_trigger_gap_points"] * CONTRACT_MULTIPLIER
        )
        results.append(rows)
    return pd.concat(results, ignore_index=True)


def _finite(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric)]


def _quantile(series: pd.Series, q: float) -> float | None:
    values = _finite(series)
    return float(values.quantile(q)) if len(values) else None


def _mean(series: pd.Series) -> float | None:
    values = _finite(series)
    return float(values.mean()) if len(values) else None


def _rate(series: pd.Series) -> float | None:
    if len(series) == 0:
        return None
    return float(series.fillna(False).astype(bool).mean())


def distortion_summary_row(
    frame: pd.DataFrame,
    dimensions: dict[str, Any],
) -> dict[str, Any]:
    executable = frame[frame["has_executable_1s"]]
    entry = frame[frame["entry_error_mean"].notna()]
    exit_rows = frame[frame["exit_error_mean"].notna()]
    adverse = executable[executable["adverse_gap_points"].notna()]
    return {
        **dimensions,
        "contract_minutes": int(len(frame)),
        "one_second_stream_rows": int(frame["stream_rows"].sum()),
        "one_second_executable_rows": int(frame["executable_1s_rows"].sum()),
        "one_second_stream_coverage_rate": _rate(frame["has_1s_stream"]),
        "executable_1s_coverage_rate": _rate(frame["has_executable_1s"]),
        "forward_fill_rate": _rate(frame["forward_fill"]),
        "hidden_adverse_dip_rate": _rate(executable["hidden_adverse_dip"]),
        "breach_and_recover_rate": _rate(executable["breach_and_recover"]),
        "adverse_gap_mean_points": _mean(adverse["adverse_gap_points"]),
        "adverse_gap_p50_points": _quantile(
            adverse["adverse_gap_points"], 0.50
        ),
        "adverse_gap_p90_points": _quantile(
            adverse["adverse_gap_points"], 0.90
        ),
        "adverse_gap_p95_points": _quantile(
            adverse["adverse_gap_points"], 0.95
        ),
        "adverse_gap_p99_points": _quantile(
            adverse["adverse_gap_points"], 0.99
        ),
        "entry_fill_error_mean_points": _mean(entry["entry_error_mean"]),
        "entry_fill_error_p50_points": _quantile(
            entry["entry_error_mean"], 0.50
        ),
        "entry_fill_abs_error_p95_points": _quantile(
            entry["entry_error_mean"].abs(), 0.95
        ),
        "entry_fill_optimistic_rate": _rate(
            entry["entry_minute_fill_optimistic"]
        ),
        "entry_one_second_ask_std_mean": _mean(entry["ask_std"]),
        "exit_fill_error_mean_points": _mean(exit_rows["exit_error_mean"]),
        "exit_fill_error_p50_points": _quantile(
            exit_rows["exit_error_mean"], 0.50
        ),
        "exit_fill_abs_error_p95_points": _quantile(
            exit_rows["exit_error_mean"].abs(), 0.95
        ),
        "exit_fill_optimistic_rate": _rate(
            exit_rows["exit_minute_fill_optimistic"]
        ),
        "exit_one_second_bid_std_mean": _mean(exit_rows["bid_std"]),
        "minute_vs_last_1s_bid_abs_p95_points": _quantile(
            frame["last_1s_bid_diff"].abs(), 0.95
        ),
        "minute_vs_last_1s_ask_abs_p95_points": _quantile(
            frame["last_1s_ask_diff"].abs(), 0.95
        ),
        "minute_quote_age_zero_rate": _rate(
            frame["minute_quote_age_is_zero"]
        ),
        "minute_quote_gap_null_rate": _rate(
            frame["minute_quote_gap_is_null"]
        ),
    }


def grouped_distortion_summary(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    if not group_columns:
        return pd.DataFrame([distortion_summary_row(frame, {})])
    rows = []
    group_arg: str | list[str]
    group_arg = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in frame.groupby(group_arg, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        dimensions = {
            column: value for column, value in zip(group_columns, key)
        }
        rows.append(distortion_summary_row(group, dimensions))
    return pd.DataFrame(rows)


def floor_summary_row(
    frame: pd.DataFrame,
    dimensions: dict[str, Any],
) -> dict[str, Any]:
    crossed = frame[frame["intraminute_cross"]]
    return {
        **dimensions,
        "eligible_contract_minutes": int(len(frame)),
        "intraminute_cross_rate": _rate(frame["intraminute_cross"]),
        "completed_minute_cross_rate": _rate(
            frame["completed_minute_cross"]
        ),
        "hidden_cross_recover_rate": _rate(frame["hidden_cross_recover"]),
        "crossed_contract_minutes": int(len(crossed)),
        "floor_trigger_gap_mean_points": _mean(
            crossed["floor_trigger_gap_points"]
        ),
        "floor_trigger_gap_p50_points": _quantile(
            crossed["floor_trigger_gap_points"], 0.50
        ),
        "floor_trigger_gap_p90_points": _quantile(
            crossed["floor_trigger_gap_points"], 0.90
        ),
        "floor_trigger_gap_p95_points": _quantile(
            crossed["floor_trigger_gap_points"], 0.95
        ),
        "floor_trigger_gap_p99_points": _quantile(
            crossed["floor_trigger_gap_points"], 0.99
        ),
        "floor_trigger_gap_p95_dollars": _quantile(
            crossed["floor_trigger_gap_dollars"], 0.95
        ),
    }


def grouped_floor_summary(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    group_arg: str | list[str]
    group_arg = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in frame.groupby(group_arg, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append(
            floor_summary_row(
                group,
                {
                    column: value
                    for column, value in zip(group_columns, key)
                },
            )
        )
    return pd.DataFrame(rows)


def pooled_floor_summaries(
    paths: Sequence[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool exact floor rates and crossed-path quantiles without a 4x row concat."""
    groupings = [
        ("floor_ratio",),
        ("floor_ratio", "premium_band", "moneyness_band"),
    ]
    accumulators: list[dict[tuple[Any, ...], dict[str, Any]]] = [
        {} for _ in groupings
    ]
    columns = [
        "floor_ratio",
        "premium_band",
        "moneyness_band",
        "intraminute_cross",
        "completed_minute_cross",
        "hidden_cross_recover",
        "floor_trigger_gap_points",
        "floor_trigger_gap_dollars",
    ]
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        for grouping, accumulator in zip(groupings, accumulators):
            group_arg: str | list[str]
            group_arg = (
                grouping[0] if len(grouping) == 1 else list(grouping)
            )
            for key, group in frame.groupby(
                group_arg, dropna=False, sort=True
            ):
                if not isinstance(key, tuple):
                    key = (key,)
                state = accumulator.setdefault(
                    key,
                    {
                        "eligible": 0,
                        "intraminute_cross": 0,
                        "completed_minute_cross": 0,
                        "hidden_cross_recover": 0,
                        "gap_points": [],
                        "gap_dollars": [],
                    },
                )
                crossed_mask = group["intraminute_cross"].fillna(False).astype(
                    bool
                )
                state["eligible"] += int(len(group))
                state["intraminute_cross"] += int(crossed_mask.sum())
                state["completed_minute_cross"] += int(
                    group["completed_minute_cross"]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                )
                state["hidden_cross_recover"] += int(
                    group["hidden_cross_recover"]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                )
                points = _finite(
                    group.loc[crossed_mask, "floor_trigger_gap_points"]
                ).to_numpy(dtype=float)
                dollars = _finite(
                    group.loc[crossed_mask, "floor_trigger_gap_dollars"]
                ).to_numpy(dtype=float)
                if len(points):
                    state["gap_points"].append(points)
                if len(dollars):
                    state["gap_dollars"].append(dollars)
        del frame
        gc.collect()

    outputs: list[pd.DataFrame] = []
    for grouping, accumulator in zip(groupings, accumulators):
        rows: list[dict[str, Any]] = []
        for key in sorted(accumulator, key=lambda value: tuple(map(str, value))):
            state = accumulator[key]
            eligible = int(state["eligible"])
            points = (
                np.concatenate(state["gap_points"])
                if state["gap_points"]
                else np.array([], dtype=float)
            )
            dollars = (
                np.concatenate(state["gap_dollars"])
                if state["gap_dollars"]
                else np.array([], dtype=float)
            )

            def array_quantile(values: np.ndarray, q: float) -> float | None:
                return float(np.quantile(values, q)) if len(values) else None

            rows.append(
                {
                    **{
                        column: value
                        for column, value in zip(grouping, key)
                    },
                    "eligible_contract_minutes": eligible,
                    "intraminute_cross_rate": (
                        state["intraminute_cross"] / eligible
                        if eligible
                        else None
                    ),
                    "completed_minute_cross_rate": (
                        state["completed_minute_cross"] / eligible
                        if eligible
                        else None
                    ),
                    "hidden_cross_recover_rate": (
                        state["hidden_cross_recover"] / eligible
                        if eligible
                        else None
                    ),
                    "crossed_contract_minutes": int(
                        state["intraminute_cross"]
                    ),
                    "floor_trigger_gap_mean_points": (
                        float(points.mean()) if len(points) else None
                    ),
                    "floor_trigger_gap_p50_points": array_quantile(
                        points, 0.50
                    ),
                    "floor_trigger_gap_p90_points": array_quantile(
                        points, 0.90
                    ),
                    "floor_trigger_gap_p95_points": array_quantile(
                        points, 0.95
                    ),
                    "floor_trigger_gap_p99_points": array_quantile(
                        points, 0.99
                    ),
                    "floor_trigger_gap_p95_dollars": array_quantile(
                        dollars, 0.95
                    ),
                }
            )
        outputs.append(pd.DataFrame(rows))
    return outputs[0], outputs[1]


def entry_exploratory_summary(
    frame: pd.DataFrame,
    *,
    dimensions: Sequence[str] = (),
) -> pd.DataFrame:
    features = [
        "intraminute_mid_return",
        "one_second_realized_volatility",
        "intraminute_mid_range_fraction",
        "bbo_change_count",
        "spread_change",
    ]
    targets = [
        "next_minute_mid_return",
        "next_minute_bid_over_current_ask_return",
    ]
    if dimensions:
        group_arg: str | list[str]
        group_arg = dimensions[0] if len(dimensions) == 1 else list(dimensions)
        groups: Iterable[tuple[Any, pd.DataFrame]] = frame.groupby(
            group_arg, dropna=False, sort=True
        )
    else:
        groups = [((), frame)]
    rows: list[dict[str, Any]] = []
    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        dims = {
            column: value for column, value in zip(dimensions, key)
        }
        for feature in features:
            for target in targets:
                valid = group[[feature, target]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                correlation = (
                    float(valid[feature].corr(valid[target], method="spearman"))
                    if len(valid) >= 3
                    else None
                )
                top_minus_bottom = None
                if len(valid) >= 20 and valid[feature].nunique() >= 5:
                    ranks = valid[feature].rank(method="first")
                    buckets = pd.qcut(
                        ranks,
                        q=5,
                        labels=False,
                        duplicates="drop",
                    )
                    if buckets.nunique() >= 2:
                        low = valid.loc[buckets.eq(buckets.min()), target].mean()
                        high = valid.loc[buckets.eq(buckets.max()), target].mean()
                        top_minus_bottom = float(high - low)
                rows.append(
                    {
                        **dims,
                        "feature": feature,
                        "target": target,
                        "rows": int(len(valid)),
                        "spearman": correlation,
                        "top_minus_bottom_quintile_mean": top_minus_bottom,
                        "claim": "descriptive_only_no_model",
                    }
                )
    return pd.DataFrame(rows)


def decision(
    overall: dict[str, Any],
    floor_summary: pd.DataFrame,
) -> dict[str, Any]:
    threshold = TRUST_THRESHOLDS
    def number(value: Any, fallback: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return fallback
        return result if math.isfinite(result) else fallback

    gate_floor = floor_summary[
        np.isclose(
            pd.to_numeric(floor_summary["floor_ratio"], errors="coerce"),
            threshold["floor_ratio_for_gate"],
        )
    ]
    floor_row = gate_floor.iloc[0].to_dict() if len(gate_floor) else {}
    checks = {
        "one_second_coverage": (
            number(overall["executable_1s_coverage_rate"], 0.0)
            >= threshold["min_executable_1s_coverage_rate"]
        ),
        "forward_fill": (
            number(overall["forward_fill_rate"], 1.0)
            <= threshold["max_forward_fill_rate"]
        ),
        "hidden_adverse_excursion_rate": (
            number(overall["hidden_adverse_dip_rate"], 1.0)
            <= threshold["max_hidden_adverse_dip_rate"]
        ),
        "hidden_adverse_excursion_magnitude": (
            number(overall["adverse_gap_p95_points"], math.inf)
            <= threshold["max_p95_adverse_gap_points"]
        ),
        "entry_fill_distortion": (
            number(overall["entry_fill_abs_error_p95_points"], math.inf)
            <= threshold["max_p95_abs_entry_fill_error_points"]
        ),
        "exit_fill_distortion": (
            number(overall["exit_fill_abs_error_p95_points"], math.inf)
            <= threshold["max_p95_abs_exit_fill_error_points"]
        ),
        "hidden_floor_cross": (
            number(floor_row.get("hidden_cross_recover_rate"), 1.0)
            <= threshold["max_hidden_floor_cross_rate"]
        ),
        "floor_trigger_gap": (
            number(floor_row.get("floor_trigger_gap_p95_points"), math.inf)
            <= threshold["max_p95_floor_trigger_gap_points"]
        ),
    }
    trustworthy = bool(all(checks.values()))
    return {
        "minute_resolution_exit_floor_validation": (
            "TRUSTWORTHY"
            if trustworthy
            else "NOT_TRUSTWORTHY_WITHOUT_HIGHRES_VALIDATION_OR_ADJUSTMENT"
        ),
        "checks": checks,
        "all_checks_pass": trustworthy,
        "required_action": (
            "Keep minute execution/floor mechanics unchanged, with CBBO-1s as "
            "an external validation instrument."
            if trustworthy
            else "Calibrate or replace minute fill/floor assumptions using "
            "CBBO-1s evidence before the exit campaign; do not claim "
            "intraminute floor protection from minute paths."
        ),
    }


def recommendation(
    verdict: dict[str, Any],
    exploratory: pd.DataFrame,
) -> dict[str, Any]:
    correlations = pd.to_numeric(
        exploratory.get("spearman", pd.Series(dtype=float)),
        errors="coerce",
    ).abs()
    strongest = float(correlations.max()) if correlations.notna().any() else None
    strongest_row: dict[str, Any] | None = None
    if correlations.notna().any():
        strongest_row = _json_ready(
            exploratory.loc[correlations.idxmax()].to_dict()
        )
    return {
        "entry": (
            "Keep minute data as the entry training substrate and CBBO-1s as "
            "a validation/feature-discovery instrument. The exploratory "
            "statistics are not a trained or causally validated entry signal; "
            "a substrate change would require a separately owner-authorized "
            "experiment."
        ),
        "exit": (
            (
                "Keep minute data as the broad-history exit substrate and use "
                "CBBO-1s as a validation instrument."
            )
            if verdict["all_checks_pass"]
            else (
                "Use CBBO-1s to construct/calibrate exit and floor labels for "
                "the available 2025+ window, while retaining minute history "
                "for regime breadth only after applying a measured distortion "
                "model. An unadjusted minute-only exit campaign is not supported."
            )
        ),
        "shorter_denser_vs_longer_coarser_tradeoff": (
            "CBBO-1s is denser but begins 2025-02-20; minute data can reach "
            "2022 and spans more regimes. Prefer a hybrid evidence design over "
            "discarding either axis."
        ),
        "strongest_absolute_exploratory_spearman": strongest,
        "strongest_exploratory_pair": strongest_row,
        "cost_note": (
            "Pilot ~$24.12; all available CBBO-1s history roughly $280; "
            "deferred 2022-2024 minute backfill roughly $41-$347."
        ),
    }


def _fmt(value: Any, *, percent: bool = False, money: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    if percent:
        return f"{float(value) * 100:.2f}%"
    if money:
        return f"${float(value):,.2f}"
    return f"{float(value):,.4f}"


def write_report(
    path: Path,
    *,
    results: dict[str, Any],
) -> None:
    overall = results["overall"]
    verdict = results["verdict"]
    rec = results["substrate_recommendation"]
    floor_90 = next(
        (
            row
            for row in results["floor_response"]
            if math.isclose(float(row["floor_ratio"]), 0.90)
        ),
        {},
    )
    cost = results["acquisition_cost"]
    lines = [
        "# FT2-D55 Exit-Realism Pilot",
        "",
        "## Owner memo — how much do the minute photos lie?",
        "",
        f"**Verdict: `{verdict['minute_resolution_exit_floor_validation']}`.**",
        "",
        (
            f"Across {overall['contract_minutes']:,} tradable contract-minutes, "
            f"{_fmt(overall['forward_fill_rate'], percent=True)} showed no "
            "one-second BBO change. The normalized minute corpus reports "
            f"`quote_age_ms == 0` on "
            f"{_fmt(overall['minute_quote_age_zero_rate'], percent=True)} and a "
            f"null quote gap on "
            f"{_fmt(overall['minute_quote_gap_null_rate'], percent=True)}, so "
            "those minute columns could not reveal this staleness."
        ),
        "",
        (
            f"A hidden intra-minute bid dip occurred in "
            f"{_fmt(overall['hidden_adverse_dip_rate'], percent=True)} of "
            "contract-minutes with executable one-second coverage. The p95 "
            f"minute-bid-to-worst-1s-bid gap was "
            f"{_fmt(overall['adverse_gap_p95_points'])} option points "
            f"({_fmt((overall['adverse_gap_p95_points'] or 0) * 100, money=True)} "
            "per contract)."
        ),
        "",
        (
            "For fill assumptions, the p95 absolute error versus the mean "
            f"one-second interval price was "
            f"{_fmt(overall['entry_fill_abs_error_p95_points'])} points on "
            f"entry asks and {_fmt(overall['exit_fill_abs_error_p95_points'])} "
            "points on exit bids."
        ),
        "",
        (
            "At the 90%-of-prior-bid floor stress boundary, "
            f"{_fmt(floor_90.get('hidden_cross_recover_rate'), percent=True)} "
            "of eligible contract-minutes crossed and recovered before the "
            "completed-minute check; crossed paths had a p95 trigger gap of "
            f"{_fmt(floor_90.get('floor_trigger_gap_p95_points'))} points "
            f"({_fmt(floor_90.get('floor_trigger_gap_p95_dollars'), money=True)} "
            "per contract)."
        ),
        "",
        "### Distortion by decision-time premium band",
        "",
        (
            "The pooled absolute-point result is dominated by expensive "
            "contracts, so the decision must also use the governed census-v4 "
            "premium bands:"
        ),
        "",
        "| Premium band | Contract-minutes | Hidden dip | P95 adverse gap | "
        "P95 entry error | P95 exit error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results["premium_band_response"]:
        lines.append(
            f"| {row['premium_band']} | {int(row['contract_minutes']):,} | "
            f"{_fmt(row['hidden_adverse_dip_rate'], percent=True)} | "
            f"{_fmt(row['adverse_gap_p95_points'])} | "
            f"{_fmt(row['entry_fill_abs_error_p95_points'])} | "
            f"{_fmt(row['exit_fill_abs_error_p95_points'])} |"
        )
    lines.extend(
        [
        "",
        "### What should we do?",
        "",
        f"- **Entry substrate:** {rec['entry']}",
        f"- **Exit substrate:** {rec['exit']}",
        f"- **Tradeoff:** {rec['shorter_denser_vs_longer_coarser_tradeoff']}",
        (
            "- **Cost:** "
            f"30-session preflight `${cost['estimated_total_usd']:.6f}`; "
            "conservative failed-stream billable upper bound "
            f"`${cost['failed_attempt_billable_cost_upper_bound_usd']:.6f}`; "
            f"worst-case `${cost['worst_case_estimated_spend_usd']:.6f}` "
            f"under the `${cost['hard_cap_usd']:.2f}` cap. Actual vendor "
            f"invoice: `{cost['actual_vendor_invoice_cost_usd']}`. "
            f"{rec['cost_note']}"
        ),
        "",
        "## Decision checks",
        "",
        "| Check | Pass |",
        "|---|---:|",
        ]
    )
    for name, passed in verdict["checks"].items():
        lines.append(f"| {name} | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Measurement details",
            "",
            f"- Sessions: `{results['sessions']}`",
            f"- Tradable contract-minutes: `{overall['contract_minutes']}`",
            (
                "- Executable one-second coverage: "
                f"`{_fmt(overall['executable_1s_coverage_rate'], percent=True)}`"
            ),
            (
                "- Breach-and-recover rate: "
                f"`{_fmt(overall['breach_and_recover_rate'], percent=True)}`"
            ),
            (
                "- Entry minute-fill optimistic fraction: "
                f"`{_fmt(overall['entry_fill_optimistic_rate'], percent=True)}`"
            ),
            (
                "- Exit minute-fill optimistic fraction: "
                f"`{_fmt(overall['exit_fill_optimistic_rate'], percent=True)}`"
            ),
            "",
            "Detailed strata are in `per_premium_band.csv`, "
            "`per_moneyness_band.csv`, `per_premium_moneyness_band.csv`, and "
            "`floor_slippage_by_band.csv`.",
            "",
            "## Protective-floor limitation",
            "",
            (
                "The signed design defines causal completed-minute checking and "
                "an upward-only floor, but it intentionally does not yet define "
                "the forecast-to-floor equation. This pilot therefore reports a "
                "one-step response surface at floors equal to 70%, 80%, 90%, "
                "and 95% of the prior completed-minute bid. These are stress "
                "boundaries, not a fitted floor policy and not a contract change."
            ),
            "",
            "## Entry-signal exploratory",
            "",
            (
                "The one-second momentum, realized-volatility, range, quote-change, "
                "and spread-change statistics are descriptive only. No model was "
                "fit and no entry-edge claim is made. The strongest absolute "
                f"Spearman correlation observed was "
                f"`{_fmt(rec['strongest_absolute_exploratory_spearman'])}` "
                f"for `{(rec.get('strongest_exploratory_pair') or {}).get('feature')}` "
                "versus "
                f"`{(rec.get('strongest_exploratory_pair') or {}).get('target')}` "
                "within its reported stratum; this is hypothesis-generating, "
                "not evidence to change the entry substrate."
            ),
            "",
            "## Scope and route",
            "",
            (
                "No model was trained. No broker, protected data, runtime, "
                "launchd, promotion, paper-default, signed contract, graph, or "
                "census state was touched."
            ),
            "",
            "**Next: `STOP_FOR_OWNER_DECISION`.**",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def process_session(
    *,
    session: str,
    normalized_dir: Path,
    raw_root: Path,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    minute = load_minute_session(normalized_dir, session)
    one_second = load_one_second_session(raw_root, session)
    agg = aggregate_one_second(one_second)
    joined = join_contract_minutes(minute, agg)
    floor = floor_measurements(joined, one_second)
    joined["session"] = session
    floor["session"] = session
    session_dir = out_dir / "session_measurements" / session
    session_dir.mkdir(parents=True, exist_ok=True)
    distortion_path = session_dir / "contract_minute_distortion.parquet"
    floor_path = session_dir / "floor_response.parquet"
    joined.to_parquet(distortion_path, index=False)
    floor.to_parquet(floor_path, index=False)
    raw_manifest = raw_root / session / "manifest.json"
    derived = (
        normalized_dir
        / f"databento_spxw_0dte_{session}_derived_context.parquet"
    )
    manifest = {
        "schema_version": "Protocol101FT2D55MeasurementSessionManifestV1",
        "goal": GOAL,
        "session": session,
        "created_at_utc": utc_now(),
        "sources": {
            "raw_session_manifest": {
                "path": str(raw_manifest),
                "sha256": sha256_file(raw_manifest),
            },
            "normalized_derived_context": {
                "path": str(derived),
                "sha256": sha256_file(derived),
            },
        },
        "rows": {
            "minute_input": int(len(minute)),
            "one_second_input": int(len(one_second)),
            "contract_minute_distortion": int(len(joined)),
            "floor_response": int(len(floor)),
        },
        "outputs": {
            "contract_minute_distortion": {
                "path": str(distortion_path),
                "sha256": sha256_file(distortion_path),
                "bytes": distortion_path.stat().st_size,
            },
            "floor_response": {
                "path": str(floor_path),
                "sha256": sha256_file(floor_path),
                "bytes": floor_path.stat().st_size,
            },
        },
    }
    manifest_path = session_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return distortion_path, floor_path, manifest_path


def existing_session_measurement(
    *,
    session: str,
    normalized_dir: Path,
    raw_root: Path,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    session_dir = out_dir / "session_measurements" / session
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"{session}: missing measurement manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("goal") != GOAL or manifest.get("session") != session:
        raise RuntimeError(f"{session}: measurement manifest identity drift")
    raw_manifest = raw_root / session / "manifest.json"
    derived = (
        normalized_dir
        / f"databento_spxw_0dte_{session}_derived_context.parquet"
    )
    if manifest["sources"]["raw_session_manifest"]["sha256"] != sha256_file(
        raw_manifest
    ):
        raise RuntimeError(f"{session}: raw source changed after measurement")
    if manifest["sources"]["normalized_derived_context"][
        "sha256"
    ] != sha256_file(derived):
        raise RuntimeError(
            f"{session}: normalized source changed after measurement"
        )
    distortion = Path(
        manifest["outputs"]["contract_minute_distortion"]["path"]
    )
    floor = Path(manifest["outputs"]["floor_response"]["path"])
    for key, path in (
        ("contract_minute_distortion", distortion),
        ("floor_response", floor),
    ):
        if not path.exists():
            raise RuntimeError(f"{session}: missing measurement output {path}")
        if manifest["outputs"][key]["sha256"] != sha256_file(path):
            raise RuntimeError(f"{session}: measurement output hash drift: {key}")
    return distortion, floor, manifest_path


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    if sha256_file(AUTHORITY_PATH) != AUTHORITY_SHA256:
        raise SystemExit("D55 authority hash mismatch")
    acquisition_path = args.out_dir / "acquisition_summary.json"
    if not acquisition_path.exists():
        raise SystemExit(f"missing completed acquisition: {acquisition_path}")
    acquisition = json.loads(acquisition_path.read_text())
    sessions = list(acquisition["selected_dates"])
    if args.sessions is not None:
        requested = set(args.sessions)
        sessions = [session for session in sessions if session in requested]
        if set(sessions) != requested:
            raise SystemExit("requested measurement session is outside D55 acquisition")
    if not sessions:
        raise SystemExit("no D55 sessions selected for measurement")
    spec = measurement_spec()
    write_json(args.out_dir / "measurement_spec.json", spec)

    distortion_paths: list[Path] = []
    floor_paths: list[Path] = []
    session_manifests: list[Path] = []
    for session in sessions:
        if args.aggregate_only:
            distortion_path, floor_path, manifest_path = (
                existing_session_measurement(
                    session=session,
                    normalized_dir=args.normalized_dir,
                    raw_root=args.raw_root,
                    out_dir=args.out_dir,
                )
            )
        else:
            distortion_path, floor_path, manifest_path = process_session(
                session=session,
                normalized_dir=args.normalized_dir,
                raw_root=args.raw_root,
                out_dir=args.out_dir,
            )
        distortion_paths.append(distortion_path)
        floor_paths.append(floor_path)
        session_manifests.append(manifest_path)
        session_summary = grouped_distortion_summary(
            pd.read_parquet(distortion_path), []
        ).iloc[0].to_dict()
        print(
            json.dumps(
                {
                    "session": session,
                    "contract_minutes": session_summary["contract_minutes"],
                    "forward_fill_rate": session_summary["forward_fill_rate"],
                    "hidden_adverse_dip_rate": session_summary[
                        "hidden_adverse_dip_rate"
                    ],
                    "status": (
                        "reused_hash_verified"
                        if args.aggregate_only
                        else "measured"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        gc.collect()

    distortion_chunks: list[pd.DataFrame] = []
    for path in distortion_paths:
        chunk = pd.read_parquet(path, columns=DISTORTION_ANALYSIS_COLUMNS)
        for column in ("session", "premium_band", "moneyness_band"):
            chunk[column] = chunk[column].astype("category")
        distortion_chunks.append(chunk)
    distortion = pd.concat(distortion_chunks, ignore_index=True)
    del distortion_chunks
    overall_frame = grouped_distortion_summary(distortion, [])
    session_frame = grouped_distortion_summary(distortion, ["session"])
    premium_frame = grouped_distortion_summary(distortion, ["premium_band"])
    moneyness_frame = grouped_distortion_summary(
        distortion, ["moneyness_band"]
    )
    cross_frame = grouped_distortion_summary(
        distortion, ["premium_band", "moneyness_band"]
    )
    exploratory = entry_exploratory_summary(distortion)
    exploratory_premium = entry_exploratory_summary(
        distortion, dimensions=("premium_band",)
    )
    exploratory_moneyness = entry_exploratory_summary(
        distortion, dimensions=("moneyness_band",)
    )
    exploratory_all = pd.concat(
        [
            exploratory.assign(stratification="overall"),
            exploratory_premium.assign(stratification="premium_band"),
            exploratory_moneyness.assign(stratification="moneyness_band"),
        ],
        ignore_index=True,
    )
    del distortion
    gc.collect()
    floor_frame, floor_band_frame = pooled_floor_summaries(floor_paths)

    session_frame.to_csv(args.out_dir / "per_session_metrics.csv", index=False)
    premium_frame.to_csv(args.out_dir / "per_premium_band.csv", index=False)
    moneyness_frame.to_csv(
        args.out_dir / "per_moneyness_band.csv", index=False
    )
    cross_frame.to_csv(
        args.out_dir / "per_premium_moneyness_band.csv", index=False
    )
    floor_band_frame.to_csv(
        args.out_dir / "floor_slippage_by_band.csv", index=False
    )
    exploratory_all.to_csv(
        args.out_dir / "entry_signal_exploratory.csv", index=False
    )

    overall = _json_ready(overall_frame.iloc[0].to_dict())
    for key in (
        "contract_minutes",
        "one_second_stream_rows",
        "one_second_executable_rows",
    ):
        overall[key] = int(overall[key])
    floor_rows = _json_ready(floor_frame.to_dict("records"))
    premium_rows = _json_ready(premium_frame.to_dict("records"))
    verdict = decision(overall, floor_frame)
    substrate = recommendation(verdict, exploratory_all)
    results = {
        "schema_version": "Protocol101FT2D55DistortionResultsV1",
        "goal": GOAL,
        "completed_at_utc": utc_now(),
        "authority_sha256": AUTHORITY_SHA256,
        "measurement_spec": {
            "path": str(args.out_dir / "measurement_spec.json"),
            "sha256": sha256_file(args.out_dir / "measurement_spec.json"),
        },
        "acquisition_summary": {
            "path": str(acquisition_path),
            "sha256": sha256_file(acquisition_path),
        },
        "acquisition_cost": {
            "estimated_total_usd": acquisition["estimated_cost_usd"],
            "failed_attempt_billable_cost_upper_bound_usd": acquisition[
                "failed_attempt_billable_cost_upper_bound_usd"
            ],
            "worst_case_estimated_spend_usd": acquisition[
                "worst_case_estimated_spend_usd"
            ],
            "hard_cap_usd": acquisition["hard_cap_usd"],
            "actual_vendor_invoice_cost_usd": acquisition[
                "actual_vendor_invoice_cost_usd"
            ],
        },
        "sessions": len(sessions),
        "selected_dates": sessions,
        "overall": overall,
        "premium_band_response": premium_rows,
        "floor_response": floor_rows,
        "entry_signal_exploratory": {
            "rows": int(len(exploratory_all)),
            "path": str(args.out_dir / "entry_signal_exploratory.csv"),
            "claim": "descriptive_only_no_model",
        },
        "verdict": verdict,
        "substrate_recommendation": substrate,
        "outputs": {
            "per_session_metrics.csv": sha256_file(
                args.out_dir / "per_session_metrics.csv"
            ),
            "per_premium_band.csv": sha256_file(
                args.out_dir / "per_premium_band.csv"
            ),
            "per_moneyness_band.csv": sha256_file(
                args.out_dir / "per_moneyness_band.csv"
            ),
            "per_premium_moneyness_band.csv": sha256_file(
                args.out_dir / "per_premium_moneyness_band.csv"
            ),
            "floor_slippage_by_band.csv": sha256_file(
                args.out_dir / "floor_slippage_by_band.csv"
            ),
            "entry_signal_exploratory.csv": sha256_file(
                args.out_dir / "entry_signal_exploratory.csv"
            ),
        },
        "session_measurement_manifests": [
            {
                "session": session,
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for session, path in zip(sessions, session_manifests)
        ],
        "side_effects": {
            "paid_download_performed_by_measurement_runner": False,
            "broker_contacted": False,
            "model_training_or_fitting": False,
            "protected_data_accessed": False,
            "signed_contract_modified": False,
            "graph_modified": False,
            "census_modified": False,
            "runtime_or_promotion_modified": False,
        },
        "highest_allowed_claim": (
            "the minute-vs-1-second exit-realism distortion is measured; "
            "the owner can decide substrate and exit-data strategy"
        ),
        "next": "STOP_FOR_OWNER_DECISION",
    }
    write_json(args.out_dir / "distortion_results.json", _json_ready(results))
    write_report(args.out_dir / "report.md", results=results)
    print(
        json.dumps(
            {
                "outcome": "measurements_complete",
                "sessions": len(sessions),
                "verdict": verdict["minute_resolution_exit_floor_validation"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
