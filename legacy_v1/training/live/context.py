from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

from training.prepare import CACHE_DIR
from training.prepare import (
    compute_features,
    download_spx_bars,
    download_spy_bars,
    download_vix_bars,
    download_spxw_chain,
    download_spxw_full,
    normalize_features,
)
from training.live.contracts import (
    FEATURE_CONTRACT_VERSION,
    FeatureContractVersion,
    LiveContextBundle,
    bundle_path,
    load_bundle,
    save_bundle,
)


LIVE_CONTEXT_DIR = os.path.join(CACHE_DIR, "live_context")


def _merge_spx_prices(spy_df: pd.DataFrame, spx_df: pd.DataFrame | None) -> pd.DataFrame:
    if spx_df is None or len(spx_df) == 0:
        return spy_df
    merged = spy_df.merge(
        spx_df[["timestamp", "spx_open", "spx_high", "spx_low", "spx_close"]],
        on="timestamp",
        how="inner",
    )
    merged["open"] = merged["spx_open"]
    merged["high"] = merged["spx_high"]
    merged["low"] = merged["spx_low"]
    merged["close"] = merged["spx_close"]
    merged = merged.drop(columns=["spx_open", "spx_high", "spx_low", "spx_close"])
    return merged.sort_values("timestamp").reset_index(drop=True)


def _vix_to_dict(vix_df: pd.DataFrame | None) -> dict[int, dict[str, float]]:
    if vix_df is None or len(vix_df) == 0:
        return {}
    out: dict[int, dict[str, float]] = {}
    for _, row in vix_df.iterrows():
        out[int(row["timestamp"])] = {
            "vix_open": float(row["vix_open"]),
            "vix_high": float(row["vix_high"]),
            "vix_low": float(row["vix_low"]),
            "vix_close": float(row["vix_close"]),
        }
    return out


def _compute_prior_levels(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {}
    date_col = df["date"].values
    unique_dates = sorted(set(date_col))
    if len(unique_dates) < 2:
        return {}
    prev_day = unique_dates[-2]
    prev_df = df[df["date"] == prev_day]
    if prev_df.empty:
        return {}
    pv = np.sum(prev_df["close"].values * np.maximum(prev_df["volume"].values, 1.0))
    tv = np.sum(np.maximum(prev_df["volume"].values, 1.0))
    return {
        "prev_day_high": float(prev_df["high"].max()),
        "prev_day_low": float(prev_df["low"].min()),
        "prev_day_close": float(prev_df["close"].iloc[-1]),
        "prev_day_vwap": float(pv / max(tv, 1.0)),
    }


def _compute_mtf_stats(df: pd.DataFrame) -> dict[str, float]:
    if len(df) < 5:
        return {}
    close = df["close"].astype(float)
    rets = close.pct_change().dropna()
    out = {
        "mean_ret_1m": float(rets.mean()),
        "std_ret_1m": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
    }
    daily = df.groupby("date")["close"].last()
    try:
        daily.index = pd.to_datetime(daily.index)
    except Exception:
        return out
    if len(daily) > 1:
        d_rets = daily.pct_change().dropna()
        out["mean_ret_1d"] = float(d_rets.mean())
        out["std_ret_1d"] = float(d_rets.std(ddof=1)) if len(d_rets) > 1 else 0.0
    weekly = daily.resample("W-FRI").last() if len(daily) > 0 else pd.Series(dtype=float)
    if len(weekly) > 1:
        w_rets = weekly.pct_change().dropna()
        out["mean_ret_1w"] = float(w_rets.mean())
        out["std_ret_1w"] = float(w_rets.std(ddof=1)) if len(w_rets) > 1 else 0.0
    return out


def _find_latest_path(base_dir: str) -> str | None:
    if not os.path.exists(base_dir):
        return None
    files = [f for f in os.listdir(base_dir) if f.startswith("context-") and f.endswith(".pt")]
    if not files:
        return None
    return os.path.join(base_dir, sorted(files)[-1])


def load_latest_context_bundle(base_dir: str = LIVE_CONTEXT_DIR) -> LiveContextBundle | None:
    path = _find_latest_path(base_dir)
    if path is None:
        return None
    return load_bundle(path)


def refresh_context_bundle(
    as_of_date: str | None = None,
    context_days: int = 30,
    ib_port: int = 4002,
    base_dir: str = LIVE_CONTEXT_DIR,
    include_polygon_options: bool = True,
) -> tuple[LiveContextBundle, str]:
    """Refresh a rolling historical context bundle used before live market open.

    Uses Polygon for historical context where possible and fills SPX/VIX from IBKR.
    """
    if as_of_date is None:
        as_of_date = dt.date.today().strftime("%Y-%m-%d")
    as_of_dt = dt.datetime.strptime(as_of_date, "%Y-%m-%d")
    start_dt = as_of_dt - dt.timedelta(days=max(context_days * 3, 90))
    start = start_dt.strftime("%Y-%m-%d")
    end = as_of_date

    # Polygon historical bars for context base.
    try:
        spy_df = download_spy_bars(start, end)
    except SystemExit as e:  # raised by prepare._polygon_client on missing key
        raise RuntimeError("POLYGON_API_KEY is required for context_refresh") from e
    if spy_df.empty:
        raise RuntimeError("Could not build context: no SPY history from Polygon")

    # IBKR fills for true SPX + VIX context.
    os.environ["IB_PORT"] = str(ib_port)
    spx_df = download_spx_bars(start, end)
    vix_df = download_vix_bars(start, end)

    df = _merge_spx_prices(spy_df, spx_df)

    # Keep rolling context_days.
    uniq_dates = sorted(df["date"].unique())
    kept_dates = set(uniq_dates[-context_days:]) if len(uniq_dates) > context_days else set(uniq_dates)
    df = df[df["date"].isin(kept_dates)].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    vix_df_keep: pd.DataFrame | None = None
    if vix_df is not None and len(vix_df) > 0:
        vix_df_keep = vix_df[vix_df["date"].isin(kept_dates)].copy()
        vix_df_keep = vix_df_keep.sort_values("timestamp").reset_index(drop=True)

    options_data: dict[tuple[str, int], dict[str, float]] = {}
    chain_data: dict[tuple[str, int], dict[str, float]] = {}
    if include_polygon_options:
        try:
            options_data = download_spxw_full(df)
            chain_data = download_spxw_chain(df)
        except SystemExit as e:  # raised by prepare._polygon_client on missing key
            print(f"WARNING: Polygon option context unavailable, continuing without it: {e}")
        except Exception as e:  # pragma: no cover - network/env-specific
            print(f"WARNING: Polygon option context unavailable, continuing without it: {e}")

    vix_dict = _vix_to_dict(vix_df_keep)
    features, _, dates, valid, _, timestamps = compute_features(
        df,
        options_data=options_data if options_data else None,
        vix_data=vix_dict if vix_dict else None,
        chain_data=chain_data if chain_data else None,
    )
    norm_features = normalize_features(features.copy(), valid)

    norm_window = min(500, len(features))
    norm_raw_buffer = features[-norm_window:].copy()
    norm_valid_buffer = valid[-norm_window:].copy()

    contract = FeatureContractVersion()
    if not contract.validate_feature_shape(features):
        raise RuntimeError("Context feature shape mismatch with live feature contract")

    bundle = LiveContextBundle(
        as_of_date=as_of_date,
        context_start_date=min(dates) if dates else as_of_date,
        context_end_date=max(dates) if dates else as_of_date,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        raw_features=features,
        normalized_features=norm_features,
        valid_mask=valid,
        dates=list(dates),
        timestamps=[str(t) for t in timestamps],
        norm_raw_buffer=norm_raw_buffer,
        norm_valid_buffer=norm_valid_buffer,
        market_rows=df.to_dict(orient="records"),
        vix_rows=vix_dict,
        options_data=options_data,
        chain_data=chain_data,
        prior_levels=_compute_prior_levels(df),
        multi_timeframe_stats=_compute_mtf_stats(df),
        source_meta={
            "spy_source": "polygon",
            "spx_source": "ibkr",
            "vix_source": "ibkr",
            "options_source": "polygon" if include_polygon_options else "none",
            "context_days": context_days,
            "generated_at": dt.datetime.utcnow().isoformat(),
        },
    )
    path = bundle_path(base_dir, as_of_date)
    save_bundle(path, bundle)
    return bundle, path
