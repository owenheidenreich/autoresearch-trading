"""
One-time data preparation for autoresearch-trading (intraday edition).
Downloads intraday market data from Polygon.io, computes features, and provides
evaluation harness.

Usage:
    uv run prepare.py                           # full prep
    uv run prepare.py --start 2022              # custom start year
    uv run prepare.py --polygon-key YOUR_KEY    # set API key

Requires POLYGON_API_KEY env var or --polygon-key flag.
Data and features are stored in ~/.cache/autoresearch-trading/.
"""

import os
import sys
import time
import math
import argparse
import pickle
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_LOOKBACK = 78         # max lookback in hourly bars (~12 trading days worth)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
TRAIN_END = "2024-12-31"  # last date in training set
VAL_START = "2025-01-01"  # first date in validation set
VAL_END = "2025-12-31"    # last date in validation set
TRANSACTION_COST_BPS = 8  # round-trip cost in bps (higher for intraday options)
MIN_TRADES = 30           # minimum position changes for valid evaluation
ANNUAL_TRADING_HOURS = 252 * 6  # ~1512 trading hours per year (6 decision points/day)
BARS_PER_DAY = 78         # 5-min bars per RTH session (9:30-16:00)
HOURS_PER_DAY = 6         # hourly prediction points per day (10:30, ..., 15:30)
RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-trading")
DATA_DIR = os.path.join(CACHE_DIR, "data")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")

# ---------------------------------------------------------------------------
# Feature names (71 features)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # === Intraday price features (12) ===
    'return_5m', 'return_15m', 'return_30m', 'return_1h', 'return_2h', 'return_4h',
    'vwap_dist',               # distance from session VWAP (z-scored by session)
    'vwap_upper_dist',         # distance from VWAP + 1 std
    'vwap_lower_dist',         # distance from VWAP - 1 std
    'bar_range',               # (high - low) / close for current 5m bar
    'bar_range_ratio',         # current bar range vs session avg bar range
    'bar_volume_ratio',        # current bar volume vs session avg volume

    # === Session structure (8) ===
    'ib_high_dist',            # distance from initial balance high (first 30m)
    'ib_low_dist',             # distance from initial balance low
    'ib_width',                # IB width as % of price
    'overnight_high_dist',     # distance from overnight (pre-market) high
    'overnight_low_dist',      # distance from overnight low
    'session_range_pct',       # current session range as % of price
    'session_position',        # where price is within session range [0, 1]
    'gap_from_prev_close',     # gap from previous session close

    # === Intraday momentum (10) ===
    'rsi_14_5m',               # RSI-14 on 5m bars
    'rsi_5_5m',                # RSI-5 on 5m bars (fast)
    'macd_5m',                 # MACD line on 5m bars (price-normalized)
    'macd_signal_5m',          # MACD signal on 5m
    'macd_hist_5m',            # MACD histogram on 5m
    'ema8_ema21_dist',         # EMA(8) - EMA(21) on 15m bars, normalized
    'ema21_ema34_dist',        # EMA(21) - EMA(34) on 15m bars, normalized
    'ema_ribbon_aligned',      # 1 if EMA(8) > EMA(21) > EMA(34), -1 if inverted, 0 mixed
    'cci_20_5m',               # CCI(20) on 5m bars, scaled to ~[-1, 1]
    'roc_12_5m',               # Rate of change (12 bars = 1 hour)

    # === Market internals (8) ===
    'tick_level',              # NYSE TICK index level (scaled)
    'tick_ma_dist',            # TICK distance from its 20-bar MA
    'trin_level',              # TRIN (Arms Index) level
    'trin_extreme',            # 1 if TRIN < 0.5 (bullish extreme), -1 if > 2.0 (bearish)
    'ad_line_slope',           # A/D line slope (5-bar regression slope)
    'ad_volume_ratio',         # advancing volume / declining volume
    'put_call_ratio',          # put/call ratio if available, else 0
    'internals_composite',     # composite: normalized score of TICK + TRIN + A/D

    # === Multi-instrument (8) ===
    'es_spy_basis',            # ES futures - SPY cash spread
    'nq_es_ratio_change',      # NQ/ES ratio change (risk-on/risk-off)
    'vix_level_intraday',      # VIX level during session
    'vix_change_session',      # VIX change within current session
    'vix_term_spread',         # VX1 - VX2 futures spread (contango/backwardation)
    'vix_term_ratio',          # VX2/VX1 ratio (>1 = contango, <1 = backwardation)
    'tnx_change_session',      # 10yr yield change within session
    'es_nq_correlation',       # rolling 30-bar correlation between ES and NQ returns

    # === Daily context (carried forward intraday) (20) ===
    'daily_return_1d', 'daily_return_5d', 'daily_return_21d',
    'daily_rvol_5d', 'daily_rvol_21d', 'daily_rvol_ratio',
    'daily_rsi_14',
    'daily_sma_dist_20', 'daily_sma_dist_50', 'daily_sma_dist_200',
    'daily_bb_position', 'daily_bb_width',
    'daily_vix_level', 'daily_vix_zscore',
    'daily_vix_percentile', 'daily_rv_iv_spread',
    'daily_tnx_level', 'daily_tnx_change',
    'daily_volume_ratio', 'daily_gap',

    # === Time encoding (5) ===
    'time_of_day_sin',         # cyclical encoding of time within session
    'time_of_day_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'minutes_since_open',      # linear minutes since 9:30 (scaled 0-1)
]

NUM_FEATURES = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Data download via Polygon.io
# ---------------------------------------------------------------------------

def _get_polygon_client():
    """Get Polygon REST client, checking API key."""
    from polygon import RESTClient

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: Set POLYGON_API_KEY environment variable or use --polygon-key flag")
        sys.exit(1)
    return RESTClient(api_key)


def _polygon_bars(client, ticker, multiplier, timespan, start, end, limit=50000):
    """Fetch aggregated bars from Polygon, handling pagination."""
    all_bars = []
    results = client.get_aggs(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=start,
        to=end,
        limit=limit,
        sort="asc",
    )
    if results:
        for bar in results:
            all_bars.append({
                'timestamp': pd.Timestamp(bar.timestamp, unit='ms', tz='US/Eastern'),
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'vwap': getattr(bar, 'vwap', None),
                'num_trades': getattr(bar, 'transactions', None),
            })
    return pd.DataFrame(all_bars)


def _download_bars_chunked(client, ticker, multiplier, timespan, start_date, end_date):
    """Download bars in monthly chunks to handle Polygon pagination limits."""
    all_dfs = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while current < end:
        chunk_end = min(current + pd.DateOffset(months=1), end)
        print(f"  {ticker} {multiplier}{timespan}: {current.date()} -> {chunk_end.date()}")

        df = _polygon_bars(
            client, ticker, multiplier, timespan,
            current.strftime('%Y-%m-%d'),
            chunk_end.strftime('%Y-%m-%d'),
        )
        if len(df) > 0:
            all_dfs.append(df)

        current = chunk_end
        time.sleep(0.15)  # rate limit courtesy

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        result = result.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        return result
    return pd.DataFrame()


def download_intraday_data(start_date="2022-01-01", end_date=None):
    """Download all required intraday data from Polygon.io.

    Downloads:
    - SPY 5-min bars (primary)
    - VIX 5-min bars
    - QQQ 5-min bars (NQ proxy)
    - SPY daily bars (for daily context features)
    - VIX daily bars
    - TNX daily bars (10yr yield)
    - Market internals proxies (TICK, TRIN, A/D)
    - VIX futures proxy (VIXY)

    Returns dict of DataFrames.
    """
    raw_path = os.path.join(DATA_DIR, "intraday_raw.pkl")
    if os.path.exists(raw_path):
        print(f"Data: already downloaded at {raw_path}")
        with open(raw_path, 'rb') as f:
            return pickle.load(f)

    os.makedirs(DATA_DIR, exist_ok=True)

    if end_date is None:
        end_date = dt.date.today().strftime('%Y-%m-%d')

    client = _get_polygon_client()

    print(f"Downloading intraday data from {start_date} to {end_date}...")
    print("This may take a while (rate limits)...")
    print()

    data = {}

    # SPY 5-min bars (primary)
    print("Downloading SPY 5-min bars...")
    data['spy_5m'] = _download_bars_chunked(client, 'SPY', 5, 'minute', start_date, end_date)
    print(f"  -> {len(data['spy_5m'])} bars")

    # SPY daily bars (for daily context)
    print("Downloading SPY daily bars...")
    data['spy_daily'] = _download_bars_chunked(client, 'SPY', 1, 'day', '2010-01-01', end_date)
    print(f"  -> {len(data['spy_daily'])} bars")

    # VIX 5-min bars
    print("Downloading VIX 5-min bars...")
    for sym in ['I:VIX', 'VIX']:
        try:
            data['vix_5m'] = _download_bars_chunked(client, sym, 5, 'minute', start_date, end_date)
            if len(data['vix_5m']) > 0:
                print(f"  -> {len(data['vix_5m'])} bars (via {sym})")
                break
        except Exception as e:
            print(f"  {sym} failed ({e}), trying next...")
    else:
        data['vix_5m'] = pd.DataFrame()
        print("  -> VIX 5m not available, will synthesize")

    # VIX daily
    print("Downloading VIX daily bars...")
    for sym in ['I:VIX', 'VIX']:
        try:
            data['vix_daily'] = _download_bars_chunked(client, sym, 1, 'day', '2010-01-01', end_date)
            if len(data['vix_daily']) > 0:
                break
        except Exception:
            pass
    else:
        data['vix_daily'] = pd.DataFrame()
    print(f"  -> {len(data.get('vix_daily', pd.DataFrame()))} bars")

    # SPX index (ES proxy)
    print("Downloading SPX index 5-min bars (ES proxy)...")
    for sym in ['I:SPX', 'SPX']:
        try:
            data['es_5m'] = _download_bars_chunked(client, sym, 5, 'minute', start_date, end_date)
            if len(data['es_5m']) > 0:
                print(f"  -> {len(data['es_5m'])} bars (via {sym})")
                break
        except Exception:
            pass
    else:
        data['es_5m'] = pd.DataFrame()
        print("  -> SPX/ES not available")

    # QQQ (NQ proxy)
    print("Downloading QQQ 5-min bars (NQ proxy)...")
    data['nq_5m'] = _download_bars_chunked(client, 'QQQ', 5, 'minute', start_date, end_date)
    print(f"  -> {len(data['nq_5m'])} bars")

    # TNX daily (10yr yield)
    print("Downloading TNX daily bars...")
    for sym in ['I:TNX', 'TNX']:
        try:
            data['tnx_daily'] = _download_bars_chunked(client, sym, 1, 'day', '2010-01-01', end_date)
            if len(data['tnx_daily']) > 0:
                break
        except Exception:
            pass
    else:
        data['tnx_daily'] = pd.DataFrame()
    print(f"  -> {len(data.get('tnx_daily', pd.DataFrame()))} bars")

    # Market internals
    print("Downloading market internals proxies...")
    for sym, key in [('I:TICK', 'tick_5m'), ('I:TRIN', 'trin_5m')]:
        try:
            data[key] = _download_bars_chunked(client, sym, 5, 'minute', start_date, end_date)
            print(f"  {sym} -> {len(data[key])} bars")
        except Exception as e:
            print(f"  {sym} not available ({e}), will synthesize")
            data[key] = pd.DataFrame()

    for sym, key in [('I:ADV', 'adv_5m'), ('I:DECL', 'decl_5m')]:
        try:
            data[key] = _download_bars_chunked(client, sym, 5, 'minute', start_date, end_date)
            print(f"  {sym} -> {len(data[key])} bars")
        except Exception as e:
            print(f"  {sym} not available ({e}), will synthesize")
            data[key] = pd.DataFrame()

    # VIX futures proxy
    print("Downloading VIX futures proxy (VIXY)...")
    try:
        data['vixy_daily'] = _download_bars_chunked(client, 'VIXY', 1, 'day', start_date, end_date)
    except Exception:
        data['vixy_daily'] = pd.DataFrame()
    print(f"  -> {len(data.get('vixy_daily', pd.DataFrame()))} bars")

    # Save
    with open(raw_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nData saved to {raw_path}")

    return data


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _filter_rth(df, time_col='timestamp'):
    """Filter to Regular Trading Hours only (9:30-16:00 ET)."""
    if df.empty:
        return df
    ts = df[time_col]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize('US/Eastern')
    t = ts.dt.time
    mask = (t >= RTH_OPEN) & (t < RTH_CLOSE)
    return df[mask].copy()


def _rsi(series, period):
    """RSI on a series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return (100 - (100 / (1 + rs))) / 100.0


def _ema(series, span):
    """EMA."""
    return series.ewm(span=span, adjust=False).mean()


def _cci(high, low, close, period=20):
    """Commodity Channel Index."""
    typical = (high + low + close) / 3.0
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical - sma) / (0.015 * mad.replace(0, 1e-10))


def _rolling_percentile(series, window=252, min_periods=63):
    """Rolling percentile rank."""
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: np.mean(x <= x[-1]), raw=True
    )


def _align_to_spy(df, spy_timestamps, col_prefix=''):
    """Align another instrument's bars to SPY timestamps via forward-fill."""
    if df.empty:
        return pd.DataFrame(index=spy_timestamps)
    aligned = df.set_index('timestamp').reindex(spy_timestamps, method='ffill')
    if col_prefix:
        aligned.columns = [f'{col_prefix}_{c}' for c in aligned.columns]
    return aligned


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_intraday_features(raw_data):
    """Compute all 71 features from raw intraday data.

    Returns (features_df, targets_series, hourly_timestamps).
    - features_df: indexed by position in the 5m bar array, subsampled to hourly
    - targets_series: next-hour log return at each hourly point
    - hourly_timestamps: actual timestamps for each hourly bar

    All features use only past data -- no lookahead.
    """
    # ----- Filter SPY to RTH -----
    spy = raw_data['spy_5m'].copy()
    if spy.empty:
        raise ValueError("No SPY 5-min data available")

    spy = _filter_rth(spy)
    spy = spy.sort_values('timestamp').reset_index(drop=True)
    spy['date'] = spy['timestamp'].dt.date
    spy['time'] = spy['timestamp'].dt.time

    close = spy['close'].astype(float)
    high = spy['high'].astype(float)
    low = spy['low'].astype(float)
    opn = spy['open'].astype(float)
    vol = spy['volume'].astype(float).replace(0, np.nan).ffill().fillna(1)
    log_close = np.log(close)

    features = pd.DataFrame(index=spy.index)

    # =====================================================================
    # INTRADAY PRICE FEATURES (12)
    # =====================================================================

    for n, name in [(1, '5m'), (3, '15m'), (6, '30m'), (12, '1h'), (24, '2h'), (48, '4h')]:
        features[f'return_{name}'] = log_close.diff(n)

    # Session VWAP
    cum_vol_price = (spy['volume'] * spy['vwap'].fillna(close)).groupby(spy['date']).cumsum()
    cum_vol = spy['volume'].groupby(spy['date']).cumsum().replace(0, 1e-10)
    session_vwap = cum_vol_price / cum_vol

    vwap_diff = close - session_vwap
    session_vwap_std = vwap_diff.groupby(spy['date']).apply(
        lambda x: x.expanding().std().fillna(x.abs().expanding().mean())
    )
    if hasattr(session_vwap_std, 'droplevel'):
        session_vwap_std = session_vwap_std.droplevel(0)
    session_vwap_std = session_vwap_std.replace(0, 1e-10).reindex(spy.index)

    features['vwap_dist'] = vwap_diff / session_vwap_std
    features['vwap_upper_dist'] = (close - (session_vwap + session_vwap_std)) / session_vwap_std
    features['vwap_lower_dist'] = (close - (session_vwap - session_vwap_std)) / session_vwap_std

    bar_range = (high - low) / close.replace(0, 1e-10)
    features['bar_range'] = bar_range
    session_avg_range = bar_range.groupby(spy['date']).apply(lambda x: x.expanding().mean())
    if hasattr(session_avg_range, 'droplevel'):
        session_avg_range = session_avg_range.droplevel(0)
    features['bar_range_ratio'] = bar_range / session_avg_range.replace(0, 1e-10).reindex(spy.index)

    session_avg_vol = vol.groupby(spy['date']).apply(lambda x: x.expanding().mean())
    if hasattr(session_avg_vol, 'droplevel'):
        session_avg_vol = session_avg_vol.droplevel(0)
    features['bar_volume_ratio'] = vol / session_avg_vol.replace(0, 1e-10).reindex(spy.index)

    # =====================================================================
    # SESSION STRUCTURE (8)
    # =====================================================================

    ib_data = spy.groupby('date').apply(lambda g: pd.Series({
        'ib_high': g.head(6)['high'].max(),
        'ib_low': g.head(6)['low'].min(),
    }))

    ib_high = spy['date'].map(ib_data['ib_high'])
    ib_low = spy['date'].map(ib_data['ib_low'])

    features['ib_high_dist'] = (close - ib_high) / close.replace(0, 1e-10)
    features['ib_low_dist'] = (close - ib_low) / close.replace(0, 1e-10)
    features['ib_width'] = (ib_high - ib_low) / close.replace(0, 1e-10)

    # Overnight high/low approximation
    prev_close_map = {}
    dates = sorted(spy['date'].unique())
    for i, d in enumerate(dates):
        if i > 0:
            prev_mask = spy['date'] == dates[i - 1]
            prev_close_map[d] = float(spy.loc[prev_mask, 'close'].iloc[-1]) if prev_mask.any() else np.nan
        else:
            prev_close_map[d] = np.nan

    prev_close = spy['date'].map(prev_close_map)
    day_open = spy.groupby('date')['open'].transform('first')

    session_high = high.groupby(spy['date']).cummax()
    session_low = low.groupby(spy['date']).cummin()

    features['overnight_high_dist'] = (close - session_high) / close.replace(0, 1e-10)
    features['overnight_low_dist'] = (close - session_low) / close.replace(0, 1e-10)

    session_range = session_high - session_low
    features['session_range_pct'] = session_range / close.replace(0, 1e-10)
    features['session_position'] = ((close - session_low) / session_range.replace(0, 1e-10)).clip(0, 1)
    features['gap_from_prev_close'] = (day_open - prev_close) / prev_close.replace(0, 1e-10)

    # =====================================================================
    # INTRADAY MOMENTUM (10)
    # =====================================================================

    features['rsi_14_5m'] = _rsi(close, 14)
    features['rsi_5_5m'] = _rsi(close, 5)

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    features['macd_5m'] = macd_line / close.replace(0, 1e-10)
    features['macd_signal_5m'] = macd_signal / close.replace(0, 1e-10)
    features['macd_hist_5m'] = (macd_line - macd_signal) / close.replace(0, 1e-10)

    # EMA ribbon on 15m timeframe (multiply periods by 3 for 5m bars)
    ema8 = _ema(close, 8 * 3)
    ema21 = _ema(close, 21 * 3)
    ema34 = _ema(close, 34 * 3)
    features['ema8_ema21_dist'] = (ema8 - ema21) / close.replace(0, 1e-10)
    features['ema21_ema34_dist'] = (ema21 - ema34) / close.replace(0, 1e-10)
    features['ema_ribbon_aligned'] = np.where(
        (ema8 > ema21) & (ema21 > ema34), 1.0,
        np.where((ema8 < ema21) & (ema21 < ema34), -1.0, 0.0)
    )

    raw_cci = _cci(high, low, close, 20)
    features['cci_20_5m'] = (raw_cci / 200.0).clip(-2, 2)

    features['roc_12_5m'] = close.pct_change(12)

    # =====================================================================
    # MARKET INTERNALS (8)
    # =====================================================================

    tick_df = raw_data.get('tick_5m', pd.DataFrame())
    if not tick_df.empty:
        tick_df = _filter_rth(tick_df)
        tick_aligned = _align_to_spy(tick_df, spy['timestamp'])
        features['tick_level'] = tick_aligned['close'].reindex(spy.index).values / 1000.0
        tick_ma = features['tick_level'].rolling(20).mean()
        features['tick_ma_dist'] = features['tick_level'] - tick_ma
    else:
        features['tick_level'] = (close.diff() / close.shift(1).replace(0, 1e-10)).rolling(5).mean() * 100
        features['tick_ma_dist'] = features['tick_level'] - features['tick_level'].rolling(20).mean()

    trin_df = raw_data.get('trin_5m', pd.DataFrame())
    if not trin_df.empty:
        trin_df = _filter_rth(trin_df)
        trin_aligned = _align_to_spy(trin_df, spy['timestamp'])
        features['trin_level'] = trin_aligned['close'].reindex(spy.index).values
    else:
        price_up = (close.diff() > 0).astype(float)
        vol_ratio = vol.rolling(10).mean() / vol.rolling(50).mean().replace(0, 1e-10)
        features['trin_level'] = np.where(price_up, 1.0 / vol_ratio.replace(0, 1e-10), vol_ratio)
        features['trin_level'] = features['trin_level'].clip(0.3, 3.0)

    features['trin_extreme'] = np.where(
        features['trin_level'] < 0.5, 1.0,
        np.where(features['trin_level'] > 2.0, -1.0, 0.0)
    )

    adv_df = raw_data.get('adv_5m', pd.DataFrame())
    decl_df = raw_data.get('decl_5m', pd.DataFrame())
    if not adv_df.empty and not decl_df.empty:
        adv_aligned = _align_to_spy(adv_df, spy['timestamp'])
        decl_aligned = _align_to_spy(decl_df, spy['timestamp'])
        ad_line = (adv_aligned['close'] - decl_aligned['close']).reindex(spy.index).cumsum()
        ad_vol = (adv_aligned['volume'] / decl_aligned['volume'].replace(0, 1e-10)).reindex(spy.index)
    else:
        ad_raw = np.where(close.diff() > 0, vol, -vol)
        ad_line = pd.Series(ad_raw, index=spy.index).cumsum()
        ad_vol_raw = np.where(close.diff() > 0, vol.values, 1e-10) / np.maximum(
            np.where(close.diff() <= 0, vol.values, 1e-10), 1e-10
        )
        ad_vol = pd.Series(ad_vol_raw, index=spy.index)

    features['ad_line_slope'] = ad_line.rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True
    )
    ad_slope_std = features['ad_line_slope'].rolling(50).std().replace(0, 1e-10)
    features['ad_line_slope'] = features['ad_line_slope'] / ad_slope_std

    features['ad_volume_ratio'] = ad_vol.clip(0, 10)
    features['put_call_ratio'] = 0.0

    tick_z = (features['tick_level'] - features['tick_level'].rolling(50).mean()) / features['tick_level'].rolling(50).std().replace(0, 1e-10)
    trin_z = -(features['trin_level'] - features['trin_level'].rolling(50).mean()) / features['trin_level'].rolling(50).std().replace(0, 1e-10)
    ad_z = features['ad_line_slope']
    features['internals_composite'] = ((tick_z + trin_z + ad_z) / 3.0).clip(-3, 3)

    # =====================================================================
    # MULTI-INSTRUMENT (8)
    # =====================================================================

    es_df = raw_data.get('es_5m', pd.DataFrame())
    if not es_df.empty:
        es_filtered = _filter_rth(es_df)
        es_aligned = _align_to_spy(es_filtered, spy['timestamp'])
        es_close = es_aligned['close'].reindex(spy.index)
        features['es_spy_basis'] = (es_close - close) / close.replace(0, 1e-10)
    else:
        features['es_spy_basis'] = 0.0

    nq_df = raw_data.get('nq_5m', pd.DataFrame())
    if not nq_df.empty:
        nq_filtered = _filter_rth(nq_df)
        nq_aligned = _align_to_spy(nq_filtered, spy['timestamp'])
        nq_close = nq_aligned['close'].reindex(spy.index)
        nq_spy_ratio = nq_close / close.replace(0, 1e-10)
        features['nq_es_ratio_change'] = nq_spy_ratio.pct_change(12)
    else:
        features['nq_es_ratio_change'] = 0.0

    vix_5m = raw_data.get('vix_5m', pd.DataFrame())
    if not vix_5m.empty:
        vix_filtered = _filter_rth(vix_5m)
        vix_aligned = _align_to_spy(vix_filtered, spy['timestamp'])
        vix_close = vix_aligned['close'].reindex(spy.index)
        features['vix_level_intraday'] = vix_close / 100.0
        vix_session_open = vix_close.groupby(spy['date']).transform('first')
        features['vix_change_session'] = (vix_close - vix_session_open) / vix_session_open.replace(0, 1e-10)
    else:
        features['vix_level_intraday'] = 0.0
        features['vix_change_session'] = 0.0

    vixy = raw_data.get('vixy_daily', pd.DataFrame())
    vix_daily = raw_data.get('vix_daily', pd.DataFrame())
    if not vixy.empty and not vix_daily.empty:
        vixy_ts = vixy.set_index('timestamp')['close'].sort_index()
        vix_daily_ts = vix_daily.set_index('timestamp')['close'].sort_index()
        common_idx = vixy_ts.index.intersection(vix_daily_ts.index)
        if len(common_idx) > 0:
            term_spread = (vixy_ts.reindex(common_idx) - vix_daily_ts.reindex(common_idx)).ffill()
            term_ratio = (vixy_ts.reindex(common_idx) / vix_daily_ts.reindex(common_idx).replace(0, 1e-10)).ffill()
            spy_dates_pd = pd.Series(pd.to_datetime(spy['date'].values), index=spy.index)
            features['vix_term_spread'] = spy_dates_pd.apply(lambda d: term_spread.asof(d) if len(term_spread) > 0 else 0).fillna(0).values / 100.0
            features['vix_term_ratio'] = spy_dates_pd.apply(lambda d: term_ratio.asof(d) if len(term_ratio) > 0 else 1).fillna(1).values
        else:
            features['vix_term_spread'] = 0.0
            features['vix_term_ratio'] = 1.0
    else:
        features['vix_term_spread'] = 0.0
        features['vix_term_ratio'] = 1.0

    tnx_daily = raw_data.get('tnx_daily', pd.DataFrame())
    if not tnx_daily.empty:
        tnx_ts = tnx_daily.set_index('timestamp')['close'].sort_index()
        spy_dates_pd = pd.Series(pd.to_datetime(spy['date'].values), index=spy.index)
        tnx_today = spy_dates_pd.apply(lambda d: tnx_ts.asof(d) if len(tnx_ts) > 0 else np.nan)
        tnx_prev = spy_dates_pd.apply(lambda d: tnx_ts.asof(d - pd.Timedelta(days=1)) if len(tnx_ts) > 0 else np.nan)
        features['tnx_change_session'] = ((tnx_today - tnx_prev) / 100.0).fillna(0).values
    else:
        features['tnx_change_session'] = 0.0

    spy_ret = log_close.diff()
    if not nq_df.empty:
        nq_ret = np.log(nq_close.replace(0, 1e-10)).diff()
        features['es_nq_correlation'] = spy_ret.rolling(30).corr(nq_ret).fillna(0)
    else:
        features['es_nq_correlation'] = 0.0

    # =====================================================================
    # DAILY CONTEXT (20) -- carried forward from daily bars
    # =====================================================================

    spy_daily = raw_data.get('spy_daily', pd.DataFrame())
    if not spy_daily.empty:
        dc = spy_daily.set_index('timestamp').sort_index()
        dc_close = dc['close'].astype(float)
        dc_log = np.log(dc_close)
        daily_feats = pd.DataFrame(index=dc.index)

        daily_feats['daily_return_1d'] = dc_log.diff(1)
        daily_feats['daily_return_5d'] = dc_log.diff(5)
        daily_feats['daily_return_21d'] = dc_log.diff(21)

        dc_ret = dc_log.diff()
        daily_feats['daily_rvol_5d'] = dc_ret.rolling(5).std() * np.sqrt(252)
        daily_feats['daily_rvol_21d'] = dc_ret.rolling(21).std() * np.sqrt(252)
        daily_feats['daily_rvol_ratio'] = daily_feats['daily_rvol_5d'] / daily_feats['daily_rvol_21d'].replace(0, 1e-10)

        daily_feats['daily_rsi_14'] = _rsi(dc_close, 14)

        for n in [20, 50, 200]:
            sma = dc_close.rolling(n).mean()
            std = dc_close.rolling(n).std().replace(0, 1e-10)
            daily_feats[f'daily_sma_dist_{n}'] = (dc_close - sma) / std

        sma20 = dc_close.rolling(20).mean()
        std20 = dc_close.rolling(20).std().replace(0, 1e-10)
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_range = (bb_upper - bb_lower).replace(0, 1e-10)
        daily_feats['daily_bb_position'] = (dc_close - bb_lower) / bb_range
        daily_feats['daily_bb_width'] = bb_range / sma20

        if not vix_daily.empty:
            vix_dc = vix_daily.set_index('timestamp')['close'].reindex(dc.index).ffill()
            daily_feats['daily_vix_level'] = vix_dc / 100.0
            vix_mean20 = vix_dc.rolling(20).mean()
            vix_std20 = vix_dc.rolling(20).std().replace(0, 1e-10)
            daily_feats['daily_vix_zscore'] = (vix_dc - vix_mean20) / vix_std20
            daily_feats['daily_vix_percentile'] = _rolling_percentile(vix_dc, 252, 63)
            daily_feats['daily_rv_iv_spread'] = daily_feats['daily_rvol_21d'] - vix_dc / 100.0
        else:
            daily_feats['daily_vix_level'] = 0.0
            daily_feats['daily_vix_zscore'] = 0.0
            daily_feats['daily_vix_percentile'] = 0.5
            daily_feats['daily_rv_iv_spread'] = 0.0

        if not tnx_daily.empty:
            tnx_dc = tnx_daily.set_index('timestamp')['close'].reindex(dc.index).ffill()
            daily_feats['daily_tnx_level'] = tnx_dc / 100.0
            daily_feats['daily_tnx_change'] = tnx_dc.diff() / 100.0
        else:
            daily_feats['daily_tnx_level'] = 0.0
            daily_feats['daily_tnx_change'] = 0.0

        dc_vol = dc['volume'].astype(float).replace(0, np.nan).ffill()
        daily_feats['daily_volume_ratio'] = dc_vol / dc_vol.rolling(20).mean().replace(0, 1e-10)
        daily_feats['daily_gap'] = (dc['open'] - dc_close.shift(1)) / dc_close.shift(1).replace(0, 1e-10)

        # Map daily features to 5m bars
        spy_dates_pd = pd.Series(pd.to_datetime(spy['date'].values), index=spy.index)
        for col in daily_feats.columns:
            daily_series = daily_feats[col].sort_index()
            features[col] = spy_dates_pd.apply(
                lambda d, s=daily_series: s.asof(d) if len(s) > 0 else 0
            ).fillna(0).values
    else:
        for col in [c for c in FEATURE_NAMES if c.startswith('daily_')]:
            features[col] = 0.0

    # =====================================================================
    # TIME ENCODING (5)
    # =====================================================================

    minutes_since_open = (
        spy['timestamp'].dt.hour * 60 + spy['timestamp'].dt.minute - (9 * 60 + 30)
    ).clip(0, 390)
    normalized_time = minutes_since_open / 390.0

    features['time_of_day_sin'] = np.sin(2 * np.pi * normalized_time)
    features['time_of_day_cos'] = np.cos(2 * np.pi * normalized_time)

    dow = spy['timestamp'].dt.dayofweek.values.astype(float)
    features['day_of_week_sin'] = np.sin(2 * np.pi * dow / 5)
    features['day_of_week_cos'] = np.cos(2 * np.pi * dow / 5)
    features['minutes_since_open'] = normalized_time

    # =====================================================================
    # TARGETS: Next-hour log return
    # =====================================================================

    targets = log_close.diff(12).shift(-12)

    # =====================================================================
    # SUBSAMPLE TO HOURLY DECISION POINTS
    # =====================================================================

    # Decision points every hour: minutes 60, 120, 180, 240, 300, 360 into session
    # = 10:30, 11:30, 12:30, 13:30, 14:30, 15:30
    hourly_mask = minutes_since_open.isin([60, 120, 180, 240, 300, 360])

    hourly_features = features.loc[hourly_mask].copy()
    hourly_targets = targets.loc[hourly_mask].copy()
    hourly_timestamps = spy.loc[hourly_mask, 'timestamp'].copy()

    # Verify feature names
    assert list(features.columns) == FEATURE_NAMES, (
        f"Feature mismatch:\n  expected ({len(FEATURE_NAMES)}): {FEATURE_NAMES}\n"
        f"  got ({len(features.columns)}): {list(features.columns)}"
    )

    return hourly_features, hourly_targets, hourly_timestamps


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_NO_NORMALIZE = {
    'time_of_day_sin', 'time_of_day_cos', 'day_of_week_sin', 'day_of_week_cos',
    'minutes_since_open', 'ema_ribbon_aligned', 'trin_extreme', 'put_call_ratio',
    'session_position',
}


def normalize_features(features_df, window=500, min_periods=100):
    """Rolling z-score normalization for hourly features.
    Uses 500-bar window (~80 trading days at 6 bars/day).
    """
    normalized = features_df.copy()
    for col in features_df.columns:
        if col in _NO_NORMALIZE:
            continue
        rm = features_df[col].rolling(window, min_periods=min_periods).mean()
        rs = features_df[col].rolling(window, min_periods=min_periods).std().replace(0, 1e-10)
        normalized[col] = ((features_df[col] - rm) / rs).clip(-5, 5)
    return normalized


# ---------------------------------------------------------------------------
# Tensor preparation
# ---------------------------------------------------------------------------

def prepare_tensors(features_df, targets_series, timestamps):
    """Convert features and targets to tensors, compute splits, save to cache."""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    feat_np = features_df.values.astype(np.float32)
    tgt_np = targets_series.values.astype(np.float32)
    dates = timestamps

    valid_mask = ~(np.isnan(feat_np).any(axis=1) | np.isnan(tgt_np))

    date_strs = dates.dt.strftime('%Y-%m-%d').values
    train_end_idx = None
    val_start_idx = None
    val_end_idx = None

    for i, d in enumerate(date_strs):
        if d <= TRAIN_END:
            train_end_idx = i
        if val_start_idx is None and d >= VAL_START:
            val_start_idx = i
        if d <= VAL_END:
            val_end_idx = i

    if train_end_idx is None or val_start_idx is None:
        raise ValueError(f"Cannot find split boundaries. Date range: {date_strs[0]} to {date_strs[-1]}")
    if val_end_idx is None:
        val_end_idx = len(date_strs) - 1

    feat_np = np.nan_to_num(feat_np, nan=0.0)
    tgt_np = np.nan_to_num(tgt_np, nan=0.0)

    data = {
        'features': torch.tensor(feat_np, dtype=torch.float32),
        'targets': torch.tensor(tgt_np, dtype=torch.float32),
        'valid_mask': torch.tensor(valid_mask, dtype=torch.bool),
        'dates': date_strs.tolist(),
        'timestamps': dates.dt.strftime('%Y-%m-%d %H:%M').tolist(),
        'train_end_idx': int(train_end_idx),
        'val_start_idx': int(val_start_idx),
        'val_end_idx': int(val_end_idx),
        'feature_names': FEATURE_NAMES,
    }

    path = os.path.join(FEATURES_DIR, "data.pt")
    torch.save(data, path)

    n_train_valid = int(valid_mask[:train_end_idx + 1].sum())
    n_val_valid = int(valid_mask[val_start_idx:val_end_idx + 1].sum())

    print(f"Features: saved {len(date_strs)} hourly bars x {NUM_FEATURES} features")
    print(f"  Training : -> idx {train_end_idx} ({n_train_valid} valid bars)")
    print(f"  Validation: idx {val_start_idx}->{val_end_idx} ({n_val_valid} valid bars)")
    print(f"  ~{n_val_valid // HOURS_PER_DAY} trading days in validation")

    return data


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_data():
    """Load precomputed feature/target tensors (on CPU)."""
    path = os.path.join(FEATURES_DIR, "data.pt")
    if not os.path.exists(path):
        print("Feature data not found. Run `uv run prepare.py` first.")
        sys.exit(1)
    return torch.load(path, map_location="cpu", weights_only=False)


def make_dataloader(data, lookback, batch_size, split="train", device="cuda"):
    """Infinite (train) or single-pass (val) dataloader.

    Yields (x, y) where:
        x: (batch, lookback, NUM_FEATURES) -- feature windows of hourly bars
        y: (batch,)                        -- next-hour returns
    """
    features = data['features'].to(device)
    targets = data['targets'].to(device)
    valid_mask = data['valid_mask']

    if split == "train":
        end = data['train_end_idx'] + 1
    else:
        end = data['val_end_idx'] + 1

    start = max(lookback, data['val_start_idx'] if split != "train" else lookback)

    valid_indices = []
    for i in range(start, end):
        if valid_mask[i] and valid_mask[max(0, i - lookback):i].all():
            valid_indices.append(i)

    valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
    n = len(valid_indices)
    assert n > 0, f"No valid samples for split={split}, lookback={lookback}"

    offsets = torch.arange(-lookback, 0, device=device)

    if split == "train":
        while True:
            perm = torch.randperm(n, device=device)
            for i in range(0, n - batch_size + 1, batch_size):
                idx = valid_indices[perm[i:i + batch_size]]
                window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
                x = features[window_idx]
                y = targets[idx]
                yield x, y
    else:
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            idx = valid_indices[i:end_i]
            window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
            x = features[window_idx]
            y = targets[idx]
            yield x, y


def load_raw_data():
    """Load the raw intraday data dict."""
    path = os.path.join(DATA_DIR, "intraday_raw.pkl")
    if not os.path.exists(path):
        print("Raw data not found. Run `uv run prepare.py` first.")
        sys.exit(1)
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sharpe(model, data, lookback, device, batch_size=256):
    """
    Walk-forward Sharpe ratio on the held-out validation period.
    **Higher is better.** This is the single metric for autoresearch-trading.

    Now operates on HOURLY bars (6 per day).
    PnL per hour = position_t * return_{t->t+1hr} - |delta_position_t| * cost
    Sharpe = mean(hourly_pnl) / std(hourly_pnl) * sqrt(trading_hours_per_year)
    """
    model.eval()

    features = data['features'].to(device)
    targets = data['targets']
    valid_mask = data['valid_mask']

    val_start = max(lookback, data['val_start_idx'])
    val_end = data['val_end_idx'] + 1

    val_indices = [
        i for i in range(val_start, val_end)
        if valid_mask[i] and valid_mask[max(0, i - lookback):i].all()
    ]

    if len(val_indices) < MIN_TRADES:
        return {
            'val_sharpe': -999.0,
            'max_drawdown': 0.0,
            'annual_return': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'num_val_bars': len(val_indices),
            'num_val_days': len(val_indices) // HOURS_PER_DAY,
        }

    val_idx_t = torch.tensor(val_indices, dtype=torch.long, device=device)
    offsets = torch.arange(-lookback, 0, device=device)

    all_positions = []
    all_returns = []

    for i in range(0, len(val_idx_t), batch_size):
        idx = val_idx_t[i:i + batch_size]
        window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
        x = features[window_idx]

        pos = model(x)
        if pos.dim() > 1:
            pos = pos.squeeze(-1)
        pos = pos.clamp(-1.0, 1.0)

        ret = targets[idx.cpu()]
        all_positions.append(pos.cpu())
        all_returns.append(ret)

    positions = torch.cat(all_positions).numpy()
    returns = torch.cat(all_returns).numpy()

    cost_per_unit = TRANSACTION_COST_BPS / 10_000
    turnover = np.abs(np.diff(positions, prepend=0.0))
    hourly_pnl = positions * returns - turnover * cost_per_unit

    mean_pnl = float(np.mean(hourly_pnl))
    std_pnl = float(np.std(hourly_pnl, ddof=1)) if len(hourly_pnl) > 1 else 1e-10
    sharpe = (mean_pnl / max(std_pnl, 1e-10)) * math.sqrt(ANNUAL_TRADING_HOURS)

    cum_pnl = np.cumsum(hourly_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    sign_changes = np.abs(np.diff(np.sign(positions)))
    num_trades = int(np.sum(sign_changes > 0))

    wins = int(np.sum(hourly_pnl > 0))
    losses = int(np.sum(hourly_pnl < 0))
    win_rate = wins / max(wins + losses, 1)

    gross_profit = float(np.sum(hourly_pnl[hourly_pnl > 0]))
    gross_loss = float(abs(np.sum(hourly_pnl[hourly_pnl < 0])))
    profit_factor = gross_profit / max(gross_loss, 1e-10)

    annual_return = mean_pnl * ANNUAL_TRADING_HOURS

    return {
        'val_sharpe': float(sharpe),
        'max_drawdown': max_dd,
        'annual_return': annual_return,
        'num_trades': num_trades,
        'win_rate': float(win_rate),
        'profit_factor': profit_factor,
        'num_val_bars': len(val_indices),
        'num_val_days': len(val_indices) // max(HOURS_PER_DAY, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare intraday data for autoresearch-trading")
    parser.add_argument(
        "--start", type=str, default="2022-01-01",
        help="Start date for intraday data (default: 2022-01-01)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date for data (default: today)"
    )
    parser.add_argument(
        "--polygon-key", type=str, default=None,
        help="Polygon.io API key (or set POLYGON_API_KEY env var)"
    )
    args = parser.parse_args()

    if args.polygon_key:
        os.environ["POLYGON_API_KEY"] = args.polygon_key

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Num features: {NUM_FEATURES}")
    print()

    t0 = time.time()
    raw_data = download_intraday_data(args.start, args.end)
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    print("Computing intraday features...")
    t0 = time.time()
    features_df, targets, timestamps = compute_intraday_features(raw_data)
    print(f"  Hourly bars: {len(features_df)} x {NUM_FEATURES} features")
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    print("Normalizing features (rolling z-score, 500-bar window)...")
    t0 = time.time()
    norm_features = normalize_features(features_df)
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    print("Saving tensors...")
    data = prepare_tensors(norm_features, targets, timestamps)
    print()
    print("Done! Ready to train: uv run train.py")