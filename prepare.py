"""
One-time data preparation for autoresearch-trading v0.3 (options-native).
Downloads intraday market data AND options chain data from Polygon.io,
computes features (including real Greeks, IV surface, options flow),
and provides evaluation harness.

Usage:
    uv run prepare.py                           # full prep
    uv run prepare.py --start 2024              # custom start year
    uv run prepare.py --polygon-key YOUR_KEY    # set API key

Requires POLYGON_API_KEY env var or --polygon-key flag.
Polygon Options Developer plan ($29/mo+) required for Greeks/IV data.
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
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_LOOKBACK = 78         # max lookback in hourly bars (~13 trading days)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
TRAIN_END = "2024-12-31"  # last date in training set
VAL_START = "2025-01-01"  # first date in validation set
VAL_END = "2025-12-31"    # last date in validation set
TRANSACTION_COST_BPS = 12 # round-trip cost in bps (options spread + slippage)
MIN_TRADES = 30           # minimum position changes for valid evaluation
ANNUAL_TRADING_HOURS = 252 * 6  # ~1512 trading hours per year
BARS_PER_DAY = 78         # 5-min bars per RTH session (9:30-16:00)
HOURS_PER_DAY = 6         # hourly prediction points per day
RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-trading")
DATA_DIR = os.path.join(CACHE_DIR, "data")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")

# ---------------------------------------------------------------------------
# Feature names (105 features)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # === Intraday price features (12) ===
    'return_5m', 'return_15m', 'return_30m', 'return_1h', 'return_2h', 'return_4h',
    'vwap_dist',               # distance from session VWAP
    'vwap_upper_dist',         # distance from VWAP + 1 std
    'vwap_lower_dist',         # distance from VWAP - 1 std
    'bar_range',               # (high - low) / close for current 5m bar
    'bar_range_ratio',         # current bar range vs session avg
    'bar_volume_ratio',        # current bar volume vs session avg

    # === Session structure (8) ===
    'ib_high_dist',            # distance from initial balance high
    'ib_low_dist',             # distance from initial balance low
    'ib_width',                # IB width as % of price
    'overnight_high_dist',     # distance from session high so far
    'overnight_low_dist',      # distance from session low so far
    'session_range_pct',       # current session range as % of price
    'session_position',        # where price is within session range [0, 1]
    'gap_from_prev_close',     # gap from previous session close

    # === Intraday momentum (10) ===
    'rsi_14_5m',               # RSI-14 on 5m bars
    'rsi_5_5m',                # RSI-5 on 5m (fast)
    'macd_5m',                 # MACD line on 5m (price-normalized)
    'macd_signal_5m',          # MACD signal on 5m
    'macd_hist_5m',            # MACD histogram on 5m
    'ema8_ema21_dist',         # EMA8-EMA21 on simulated 15m
    'ema21_ema34_dist',        # EMA21-EMA34 on simulated 15m
    'ema_ribbon_aligned',      # 1 if bullish ribbon, -1 if bearish, 0 mixed
    'cci_20_5m',               # CCI(20) on 5m, scaled
    'roc_12_5m',               # Rate of change (12 bars = 1 hour)

    # === Market internals (8) ===
    'tick_level',              # NYSE TICK level (scaled)
    'tick_ma_dist',            # TICK distance from 20-bar MA
    'trin_level',              # TRIN (Arms Index) level
    'trin_extreme',            # 1 if TRIN < 0.5 (bullish), -1 if > 2.0 (bearish)
    'ad_line_slope',           # A/D line slope (5-bar regression)
    'ad_volume_ratio',         # advancing / declining volume
    'put_call_volume_ratio',   # actual P/C volume ratio from options data
    'internals_composite',     # composite of TICK + TRIN + A/D

    # === Options Greeks & IV surface (16) ===  [NEW in v0.3]
    'atm_iv',                  # ATM implied volatility (call+put avg)
    'atm_delta',               # ATM call delta (should be ~0.50)
    'atm_gamma',               # ATM gamma (sensitivity to price moves)
    'atm_theta',               # ATM theta (daily time decay, normalized by price)
    'atm_vega',                # ATM vega (sensitivity to IV, normalized)
    'iv_skew_25d',             # 25-delta put IV minus 25-delta call IV
    'iv_term_spread',          # near-term IV minus next-term IV
    'iv_change_session',       # IV change from session open
    'iv_percentile_20d',       # current IV percentile in trailing 20 sessions
    'gamma_exposure',          # notional gamma: gamma * OI * 100 * price, scaled
    'theta_acceleration',      # rate of theta change (accelerating decay)
    'net_premium_flow',        # call premium minus put premium (normalized)
    'oi_put_call_ratio',       # open interest put/call ratio
    'iv_rv_spread',            # implied vol minus realized vol (vol risk premium)
    'atm_bid_ask_spread',      # bid-ask spread of ATM options (liquidity proxy)
    'iv_slope_1h',             # rate of IV change over last hour

    # === Options flow (6) ===  [NEW in v0.3]
    'call_volume_surge',       # call volume vs 20-session avg
    'put_volume_surge',        # put volume vs 20-session avg
    'large_trade_bias',        # net direction of trades > 100 contracts
    'volume_weighted_delta',   # volume-weighted delta of all trades
    'near_term_oi_change',     # change in near-term open interest
    'options_volume_ratio',    # today's options volume vs 20d avg

    # === Multi-instrument (8) ===
    'es_spy_basis',            # ES-SPY spread
    'nq_es_ratio_change',      # NQ/ES ratio change (risk-on/off)
    'vix_level_intraday',      # VIX level during session
    'vix_change_session',      # VIX change within session
    'vix_term_spread',         # VIX term structure spread
    'vix_term_ratio',          # VIX term structure ratio
    'tnx_change_session',      # 10yr yield change in session
    'es_nq_correlation',       # rolling SPY-QQQ correlation

    # === Risk management state (6) ===  [NEW in v0.3]
    'session_pnl',             # cumulative hypothetical P&L this session
    'position_duration',       # how many bars current position held (normalized)
    'drawdown_from_session_peak', # drawdown from session's peak P&L
    'consecutive_losses',      # count of consecutive losing hourly bars
    'time_to_close',           # normalized time remaining in session (1->0)
    'theta_remaining',         # estimated theta burn left at current IV

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
    'time_of_day_sin',
    'time_of_day_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'minutes_since_open',

    # === Options-adjusted target context (6) ===  [NEW in v0.3]
    'current_atm_price',       # current ATM option mid-price (normalized)
    'breakeven_move',          # % move needed to break even on ATM option
    'expected_theta_cost',     # theta cost for next hour hold (normalized)
    'delta_adjusted_leverage', # effective leverage: delta * (underlying/option)
    'gamma_pnl_potential',     # gamma * expected_move^2 * 100 (convexity P&L)
    'edge_after_costs',        # model needs this much alpha to profit
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
    """Fetch aggregated bars from Polygon with retry on rate limits."""
    for attempt in range(5):
        try:
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
        except Exception as e:
            if '429' in str(e) or 'Max retries' in str(e) or 'too many' in str(e).lower():
                wait = 15 * (2 ** attempt)  # 15, 30, 60, 120, 240s
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after 5 retries for {ticker} {start}-{end}")


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
        time.sleep(13)  # Polygon Starter: 5 calls/min = 12s between calls + margin

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        result = result.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        return result
    return pd.DataFrame()


def _download_options_snapshots(client, start_date, end_date):
    """Download SPY options chain snapshots (Greeks, IV, OI, volume).

    Uses Polygon's options snapshot endpoint for ATM options near each
    trading day. We build a daily snapshot of key options metrics.

    Returns a DataFrame indexed by date with columns for Greeks, IV, flow.
    """
    options_path = os.path.join(DATA_DIR, "options_snapshots.pkl")
    if os.path.exists(options_path):
        print("Options snapshots: already downloaded")
        with open(options_path, 'rb') as f:
            return pickle.load(f)

    print("Downloading SPY options chain data...")
    print("  (This uses the Options Developer plan API)")

    all_snapshots = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while current <= end:
        # Skip weekends
        if current.weekday() >= 5:
            current += pd.Timedelta(days=1)
            continue

        date_str = current.strftime('%Y-%m-%d')

        try:
            # Get options chain snapshot for SPY
            snapshot = _fetch_options_snapshot_for_date(client, date_str)
            if snapshot is not None:
                all_snapshots.append(snapshot)
        except Exception as e:
            print(f"  {date_str}: {e}")

        current += pd.Timedelta(days=1)
        time.sleep(13)  # Polygon Starter: 5 calls/min

    if all_snapshots:
        result = pd.DataFrame(all_snapshots)
    else:
        result = pd.DataFrame()

    with open(options_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"  -> {len(result)} daily options snapshots saved")
    return result


def _fetch_options_snapshot_for_date(client, date_str):
    """Fetch options chain metrics for a single date.

    Gets the nearest ATM call and put, extracts Greeks and IV,
    and computes flow metrics from the chain.
    """
    try:
        # Use the options chain snapshot endpoint
        chain = list(client.list_snapshot_options_chain(
            "SPY",
            params={
                "strike_price.gte": 0,
                "expiration_date.gte": date_str,
                "expiration_date.lte": (pd.Timestamp(date_str) + pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                "limit": 250,
            }
        ))
    except Exception:
        return None

    if not chain:
        return None

    # Parse snapshot data
    calls = []
    puts = []

    for contract in chain:
        details = getattr(contract, 'details', None)
        greeks = getattr(contract, 'greeks', None)
        day_data = getattr(contract, 'day', None)
        underlying = getattr(contract, 'underlying_asset', None)

        if details is None:
            continue

        contract_type = getattr(details, 'contract_type', '')
        strike = getattr(details, 'strike_price', 0)
        expiration = getattr(details, 'expiration_date', '')

        iv = getattr(contract, 'implied_volatility', None)
        bid = getattr(contract, 'bid', 0) or 0
        ask = getattr(contract, 'ask', 0) or 0

        rec = {
            'strike': strike,
            'expiration': expiration,
            'iv': iv,
            'bid': bid,
            'ask': ask,
            'mid': (bid + ask) / 2 if (bid + ask) > 0 else None,
            'volume': getattr(day_data, 'volume', 0) if day_data else 0,
            'open_interest': getattr(day_data, 'open_interest', 0) if day_data else 0,
            'delta': getattr(greeks, 'delta', None) if greeks else None,
            'gamma': getattr(greeks, 'gamma', None) if greeks else None,
            'theta': getattr(greeks, 'theta', None) if greeks else None,
            'vega': getattr(greeks, 'vega', None) if greeks else None,
        }

        if contract_type == 'call':
            calls.append(rec)
        elif contract_type == 'put':
            puts.append(rec)

    if not calls or not puts:
        return None

    # Get underlying price from first contract
    underlying_price = None
    for contract in chain:
        ua = getattr(contract, 'underlying_asset', None)
        if ua:
            underlying_price = getattr(ua, 'price', None)
            if underlying_price:
                break

    if not underlying_price:
        return None

    calls_df = pd.DataFrame(calls)
    puts_df = pd.DataFrame(puts)

    # Find ATM: closest strike to underlying price
    calls_df['dist'] = (calls_df['strike'] - underlying_price).abs()
    puts_df['dist'] = (puts_df['strike'] - underlying_price).abs()

    atm_call = calls_df.loc[calls_df['dist'].idxmin()]
    atm_put = puts_df.loc[puts_df['dist'].idxmin()]

    # Find 25-delta options for skew
    calls_with_delta = calls_df.dropna(subset=['delta'])
    puts_with_delta = puts_df.dropna(subset=['delta'])

    iv_skew_25d = 0.0
    if len(puts_with_delta) > 0 and len(calls_with_delta) > 0:
        # 25-delta put (delta ~ -0.25)
        puts_with_delta = puts_with_delta.copy()
        puts_with_delta['delta_dist_25'] = (puts_with_delta['delta'].astype(float) + 0.25).abs()
        put_25d = puts_with_delta.loc[puts_with_delta['delta_dist_25'].idxmin()]

        # 25-delta call (delta ~ 0.25)
        calls_with_delta = calls_with_delta.copy()
        calls_with_delta['delta_dist_25'] = (calls_with_delta['delta'].astype(float) - 0.25).abs()
        call_25d = calls_with_delta.loc[calls_with_delta['delta_dist_25'].idxmin()]

        if put_25d['iv'] and call_25d['iv']:
            iv_skew_25d = float(put_25d['iv']) - float(call_25d['iv'])

    # Compute aggregate metrics
    total_call_vol = calls_df['volume'].sum()
    total_put_vol = puts_df['volume'].sum()
    total_call_oi = calls_df['open_interest'].sum()
    total_put_oi = puts_df['open_interest'].sum()

    # Net premium flow: sum(call_mid * call_vol) - sum(put_mid * put_vol)
    calls_df['premium_flow'] = calls_df['mid'].fillna(0) * calls_df['volume']
    puts_df['premium_flow'] = puts_df['mid'].fillna(0) * puts_df['volume']
    net_premium = calls_df['premium_flow'].sum() - puts_df['premium_flow'].sum()

    # Near-term expiration metrics
    nearest_exp = min(calls_df['expiration'].unique()) if len(calls_df) > 0 else None
    near_calls = calls_df[calls_df['expiration'] == nearest_exp] if nearest_exp else calls_df
    near_puts = puts_df[puts_df['expiration'] == nearest_exp] if nearest_exp else puts_df

    # IV term structure: nearest term ATM IV vs longer term
    iv_term_spread = 0.0
    expirations = sorted(calls_df['expiration'].unique())
    if len(expirations) >= 2:
        near_atm = near_calls.loc[near_calls['dist'].idxmin()]
        far_calls = calls_df[calls_df['expiration'] == expirations[-1]]
        if len(far_calls) > 0:
            far_calls = far_calls.copy()
            far_calls['dist'] = (far_calls['strike'] - underlying_price).abs()
            far_atm = far_calls.loc[far_calls['dist'].idxmin()]
            if near_atm['iv'] and far_atm['iv']:
                iv_term_spread = float(near_atm['iv']) - float(far_atm['iv'])

    # Gamma exposure estimate
    gamma_exposure = 0.0
    for _, row in calls_df.iterrows():
        if row['gamma'] and row['open_interest']:
            gamma_exposure += float(row['gamma']) * float(row['open_interest']) * 100 * underlying_price
    for _, row in puts_df.iterrows():
        if row['gamma'] and row['open_interest']:
            gamma_exposure -= float(row['gamma']) * float(row['open_interest']) * 100 * underlying_price

    # Normalize by a typical value
    gamma_exposure_norm = gamma_exposure / 1e9  # in billions

    # ATM bid-ask spread as % of mid
    atm_call_spread = 0.0
    if atm_call['mid'] and atm_call['mid'] > 0:
        atm_call_spread = (atm_call['ask'] - atm_call['bid']) / atm_call['mid']

    snapshot = {
        'date': date_str,
        'underlying_price': underlying_price,
        # ATM Greeks
        'atm_iv': float(atm_call.get('iv', 0) or 0 + atm_put.get('iv', 0) or 0) / 2,
        'atm_call_delta': float(atm_call.get('delta', 0) or 0),
        'atm_call_gamma': float(atm_call.get('gamma', 0) or 0),
        'atm_call_theta': float(atm_call.get('theta', 0) or 0),
        'atm_call_vega': float(atm_call.get('vega', 0) or 0),
        'atm_put_delta': float(atm_put.get('delta', 0) or 0),
        'atm_put_gamma': float(atm_put.get('gamma', 0) or 0),
        'atm_put_theta': float(atm_put.get('theta', 0) or 0),
        'atm_put_vega': float(atm_put.get('vega', 0) or 0),
        'atm_call_mid': float(atm_call.get('mid', 0) or 0),
        'atm_put_mid': float(atm_put.get('mid', 0) or 0),
        'atm_call_bid_ask': float(atm_call_spread),
        # IV surface
        'iv_skew_25d': iv_skew_25d,
        'iv_term_spread': iv_term_spread,
        # Flow
        'total_call_volume': total_call_vol,
        'total_put_volume': total_put_vol,
        'total_call_oi': total_call_oi,
        'total_put_oi': total_put_oi,
        'net_premium_flow': net_premium,
        # Gamma exposure
        'gamma_exposure_norm': gamma_exposure_norm,
    }

    return snapshot


def download_intraday_data(start_date="2024-04-01", end_date=None):
    """Download all required intraday + options data from Polygon.io.

    Downloads:
    - SPY 5-min bars (primary)
    - VIX 5-min bars
    - QQQ 5-min bars (NQ proxy)
    - SPY daily bars (daily context)
    - VIX daily bars
    - TNX daily bars (10yr yield)
    - Market internals proxies
    - VIX futures proxy (VIXY)
    - SPY options chain snapshots (Greeks, IV, OI, flow)

    Returns dict of DataFrames.
    """
    raw_path = os.path.join(DATA_DIR, "intraday_raw_v3.pkl")
    if os.path.exists(raw_path):
        print(f"Data: already downloaded at {raw_path}")
        with open(raw_path, 'rb') as f:
            return pickle.load(f)

    os.makedirs(DATA_DIR, exist_ok=True)

    if end_date is None:
        end_date = dt.date.today().strftime('%Y-%m-%d')

    client = _get_polygon_client()

    print(f"Downloading data from {start_date} to {end_date}...")
    print("This may take a while (rate limits)...")
    print()

    data = {}

    # --- Equity bars (same as v0.2) ---

    print("Downloading SPY 5-min bars...")
    data['spy_5m'] = _download_bars_chunked(client, 'SPY', 5, 'minute', start_date, end_date)
    print(f"  -> {len(data['spy_5m'])} bars")

    print("Downloading SPY daily bars...")
    try:
        data['spy_daily'] = _download_bars_chunked(client, 'SPY', 1, 'day', '2010-01-01', end_date)
    except Exception:
        print("  2010 start failed, falling back to intraday start date...")
        data['spy_daily'] = _download_bars_chunked(client, 'SPY', 1, 'day', start_date, end_date)
    print(f"  -> {len(data['spy_daily'])} bars")

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

    print("Downloading VIX daily bars...")
    for sym in ['I:VIX', 'VIX']:
        try:
            data['vix_daily'] = _download_bars_chunked(client, sym, 1, 'day', '2010-01-01', end_date)
            if len(data['vix_daily']) > 0:
                break
        except Exception:
            try:
                data['vix_daily'] = _download_bars_chunked(client, sym, 1, 'day', start_date, end_date)
                if len(data['vix_daily']) > 0:
                    break
            except Exception:
                pass
    else:
        data['vix_daily'] = pd.DataFrame()
    print(f"  -> {len(data.get('vix_daily', pd.DataFrame()))} bars")

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

    print("Downloading QQQ 5-min bars (NQ proxy)...")
    data['nq_5m'] = _download_bars_chunked(client, 'QQQ', 5, 'minute', start_date, end_date)
    print(f"  -> {len(data['nq_5m'])} bars")

    print("Downloading TNX daily bars...")
    for sym in ['I:TNX', 'TNX']:
        try:
            data['tnx_daily'] = _download_bars_chunked(client, sym, 1, 'day', '2010-01-01', end_date)
            if len(data['tnx_daily']) > 0:
                break
        except Exception:
            try:
                data['tnx_daily'] = _download_bars_chunked(client, sym, 1, 'day', start_date, end_date)
                if len(data['tnx_daily']) > 0:
                    break
            except Exception:
                pass
    else:
        data['tnx_daily'] = pd.DataFrame()
    print(f"  -> {len(data.get('tnx_daily', pd.DataFrame()))} bars")

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

    print("Downloading VIX futures proxy (VIXY)...")
    try:
        data['vixy_daily'] = _download_bars_chunked(client, 'VIXY', 1, 'day', start_date, end_date)
    except Exception:
        data['vixy_daily'] = pd.DataFrame()
    print(f"  -> {len(data.get('vixy_daily', pd.DataFrame()))} bars")

    # --- OPTIONS DATA (NEW in v0.3) ---
    print()
    print("=" * 60)
    print("Downloading SPY options chain snapshots (Greeks, IV, OI)...")
    print("=" * 60)
    data['options_snapshots'] = _download_options_snapshots(client, start_date, end_date)
    print(f"  -> {len(data['options_snapshots'])} daily snapshots")

    # Save
    with open(raw_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nAll data saved to {raw_path}")

    return data


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _filter_rth(df, time_col='timestamp'):
    """Filter to Regular Trading Hours (9:30-16:00 ET)."""
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
    """Compute all 105 features from raw intraday + options data.

    Returns (features_df, targets_series, hourly_timestamps).
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
    # MARKET INTERNALS (8) — note: put_call_ratio replaced with real data
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

    # Real put/call volume ratio from options data (replacing the 0.0 placeholder)
    options_snaps = raw_data.get('options_snapshots', pd.DataFrame())
    if not options_snaps.empty and 'total_put_volume' in options_snaps.columns:
        pc_ratio = options_snaps.set_index('date')['total_put_volume'] / \
                   options_snaps.set_index('date')['total_call_volume'].replace(0, 1e-10)
        spy_date_str = spy['date'].astype(str)
        features['put_call_volume_ratio'] = spy_date_str.map(pc_ratio).fillna(1.0).values
    else:
        features['put_call_volume_ratio'] = 1.0

    tick_z = (features['tick_level'] - features['tick_level'].rolling(50).mean()) / features['tick_level'].rolling(50).std().replace(0, 1e-10)
    trin_z = -(features['trin_level'] - features['trin_level'].rolling(50).mean()) / features['trin_level'].rolling(50).std().replace(0, 1e-10)
    ad_z = features['ad_line_slope']
    features['internals_composite'] = ((tick_z + trin_z + ad_z) / 3.0).clip(-3, 3)

    # =====================================================================
    # OPTIONS GREEKS & IV SURFACE (16) [NEW in v0.3]
    # =====================================================================

    _compute_options_greeks_features(features, spy, close, raw_data)

    # =====================================================================
    # OPTIONS FLOW (6) [NEW in v0.3]
    # =====================================================================

    _compute_options_flow_features(features, spy, raw_data)

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
    # RISK MANAGEMENT STATE (6) [NEW in v0.3]
    # =====================================================================

    _compute_risk_state_features(features, spy, close, log_close)

    # =====================================================================
    # DAILY CONTEXT (20) — carried forward from daily bars
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
    # OPTIONS-ADJUSTED TARGET CONTEXT (6) [NEW in v0.3]
    # =====================================================================

    _compute_options_target_context(features, spy, close, raw_data)

    # =====================================================================
    # TARGETS: Options-adjusted next-hour return
    # =====================================================================

    # Base target: next-hour log return (12 five-minute bars ahead)
    base_target = log_close.diff(12).shift(-12)

    # Options-adjusted target: accounts for delta leverage and theta cost
    # target = delta * base_return * leverage - theta_cost_per_hour
    # This teaches the model what the ACTUAL P&L of holding an ATM option is
    targets = _compute_options_adjusted_target(base_target, features, spy, raw_data)

    # =====================================================================
    # SUBSAMPLE TO HOURLY DECISION POINTS
    # =====================================================================

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
# Options Greeks & IV features (16)
# ---------------------------------------------------------------------------

def _compute_options_greeks_features(features, spy, close, raw_data):
    """Compute 16 options Greeks and IV surface features."""
    options_snaps = raw_data.get('options_snapshots', pd.DataFrame())
    spy_date_str = spy['date'].astype(str)

    if options_snaps.empty or len(options_snaps) == 0:
        # No options data: fill with sensible defaults
        features['atm_iv'] = 0.20  # ~20% IV typical
        features['atm_delta'] = 0.50
        features['atm_gamma'] = 0.01
        features['atm_theta'] = -0.01
        features['atm_vega'] = 0.10
        features['iv_skew_25d'] = 0.0
        features['iv_term_spread'] = 0.0
        features['iv_change_session'] = 0.0
        features['iv_percentile_20d'] = 0.5
        features['gamma_exposure'] = 0.0
        features['theta_acceleration'] = 0.0
        features['net_premium_flow'] = 0.0
        features['oi_put_call_ratio'] = 1.0
        features['iv_rv_spread'] = 0.0
        features['atm_bid_ask_spread'] = 0.01
        features['iv_slope_1h'] = 0.0
        return

    opts = options_snaps.set_index('date')

    # Map daily options snapshots to every 5m bar (constant within day)
    def _map_opt(col, default=0.0):
        if col in opts.columns:
            return spy_date_str.map(opts[col]).fillna(default).astype(float).values
        return np.full(len(spy), default)

    # ATM IV (average of call + put)
    atm_iv_raw = _map_opt('atm_iv', 0.20)
    features['atm_iv'] = atm_iv_raw

    # ATM Greeks — use call Greeks (put mirrors with sign flip)
    features['atm_delta'] = _map_opt('atm_call_delta', 0.50)
    features['atm_gamma'] = _map_opt('atm_call_gamma', 0.01)

    # Theta normalized by underlying price (so it's comparable across time)
    theta_raw = _map_opt('atm_call_theta', -0.01)
    features['atm_theta'] = theta_raw / np.maximum(close.values, 1.0)

    # Vega normalized by underlying price
    vega_raw = _map_opt('atm_call_vega', 0.10)
    features['atm_vega'] = vega_raw / np.maximum(close.values, 1.0)

    # IV skew (25-delta put - 25-delta call IV)
    features['iv_skew_25d'] = _map_opt('iv_skew_25d', 0.0)

    # IV term structure (near-term minus far-term IV)
    features['iv_term_spread'] = _map_opt('iv_term_spread', 0.0)

    # IV change within session (since this is daily snapshot, use diff)
    iv_series = pd.Series(atm_iv_raw, index=spy.index)
    iv_session_start = iv_series.groupby(spy['date']).transform('first')
    features['iv_change_session'] = iv_series - iv_session_start

    # IV percentile (rolling 20-session)
    iv_daily = opts['atm_iv'] if 'atm_iv' in opts.columns else pd.Series(dtype=float)
    if len(iv_daily) > 0:
        iv_pctile = iv_daily.rolling(20, min_periods=5).apply(
            lambda x: np.mean(x <= x.iloc[-1]), raw=False
        ).fillna(0.5)
        features['iv_percentile_20d'] = spy_date_str.map(iv_pctile).fillna(0.5).astype(float).values
    else:
        features['iv_percentile_20d'] = 0.5

    # Gamma exposure
    features['gamma_exposure'] = _map_opt('gamma_exposure_norm', 0.0)

    # Theta acceleration: rate of theta change day-over-day
    if 'atm_call_theta' in opts.columns and len(opts) > 1:
        theta_daily = opts['atm_call_theta'].astype(float)
        theta_accel = theta_daily.diff().fillna(0)
        features['theta_acceleration'] = spy_date_str.map(theta_accel).fillna(0).astype(float).values
    else:
        features['theta_acceleration'] = 0.0

    # Net premium flow (normalized by a typical daily value)
    net_prem = _map_opt('net_premium_flow', 0.0)
    net_prem_std = pd.Series(net_prem).rolling(100, min_periods=10).std().replace(0, 1e-10).values
    features['net_premium_flow'] = net_prem / net_prem_std

    # OI put/call ratio
    total_put_oi = _map_opt('total_put_oi', 1.0)
    total_call_oi = _map_opt('total_call_oi', 1.0)
    features['oi_put_call_ratio'] = total_put_oi / np.maximum(total_call_oi, 1.0)

    # IV minus realized vol spread (vol risk premium)
    # Use daily rvol from features if available, else estimate from returns
    rvol_5d = close.pct_change().rolling(30).std() * np.sqrt(252)  # 30-bar realized
    features['iv_rv_spread'] = atm_iv_raw - rvol_5d.fillna(0.15).values

    # ATM bid-ask spread (liquidity)
    features['atm_bid_ask_spread'] = _map_opt('atm_call_bid_ask', 0.01)

    # IV slope over last hour (12 bars) — approximated since we have daily snapshots
    # Within a day, IV is constant from snapshot, so this picks up day-over-day change
    features['iv_slope_1h'] = pd.Series(atm_iv_raw, index=spy.index).diff(12).fillna(0).values


# ---------------------------------------------------------------------------
# Options Flow features (6)
# ---------------------------------------------------------------------------

def _compute_options_flow_features(features, spy, raw_data):
    """Compute 6 options flow features from chain snapshots."""
    options_snaps = raw_data.get('options_snapshots', pd.DataFrame())
    spy_date_str = spy['date'].astype(str)

    if options_snaps.empty:
        features['call_volume_surge'] = 1.0
        features['put_volume_surge'] = 1.0
        features['large_trade_bias'] = 0.0
        features['volume_weighted_delta'] = 0.0
        features['near_term_oi_change'] = 0.0
        features['options_volume_ratio'] = 1.0
        return

    opts = options_snaps.set_index('date')

    # Call volume surge: today's call volume vs 20-day avg
    if 'total_call_volume' in opts.columns:
        call_vol = opts['total_call_volume'].astype(float)
        call_vol_avg = call_vol.rolling(20, min_periods=5).mean().replace(0, 1e-10)
        call_surge = (call_vol / call_vol_avg).fillna(1.0)
        features['call_volume_surge'] = spy_date_str.map(call_surge).fillna(1.0).astype(float).values
    else:
        features['call_volume_surge'] = 1.0

    # Put volume surge
    if 'total_put_volume' in opts.columns:
        put_vol = opts['total_put_volume'].astype(float)
        put_vol_avg = put_vol.rolling(20, min_periods=5).mean().replace(0, 1e-10)
        put_surge = (put_vol / put_vol_avg).fillna(1.0)
        features['put_volume_surge'] = spy_date_str.map(put_surge).fillna(1.0).astype(float).values
    else:
        features['put_volume_surge'] = 1.0

    # Large trade bias — approximated as net premium direction
    # (Full implementation would need trade-level data)
    if 'net_premium_flow' in opts.columns:
        net_flow = opts['net_premium_flow'].astype(float)
        flow_std = net_flow.rolling(20, min_periods=5).std().replace(0, 1e-10)
        bias = (net_flow / flow_std).clip(-3, 3).fillna(0)
        features['large_trade_bias'] = spy_date_str.map(bias).fillna(0).astype(float).values
    else:
        features['large_trade_bias'] = 0.0

    # Volume-weighted delta — approximated from P/C volume ratio
    if 'total_call_volume' in opts.columns and 'total_put_volume' in opts.columns:
        cv = opts['total_call_volume'].astype(float)
        pv = opts['total_put_volume'].astype(float)
        total = (cv + pv).replace(0, 1e-10)
        vw_delta = ((cv * 0.5 - pv * 0.5) / total).fillna(0)
        features['volume_weighted_delta'] = spy_date_str.map(vw_delta).fillna(0).astype(float).values
    else:
        features['volume_weighted_delta'] = 0.0

    # Near-term OI change
    if 'total_call_oi' in opts.columns and 'total_put_oi' in opts.columns:
        total_oi = opts['total_call_oi'].astype(float) + opts['total_put_oi'].astype(float)
        oi_change = total_oi.pct_change().fillna(0).clip(-1, 1)
        features['near_term_oi_change'] = spy_date_str.map(oi_change).fillna(0).astype(float).values
    else:
        features['near_term_oi_change'] = 0.0

    # Total options volume ratio vs 20d avg
    if 'total_call_volume' in opts.columns and 'total_put_volume' in opts.columns:
        total_vol = opts['total_call_volume'].astype(float) + opts['total_put_volume'].astype(float)
        avg_vol = total_vol.rolling(20, min_periods=5).mean().replace(0, 1e-10)
        vol_ratio = (total_vol / avg_vol).fillna(1.0)
        features['options_volume_ratio'] = spy_date_str.map(vol_ratio).fillna(1.0).astype(float).values
    else:
        features['options_volume_ratio'] = 1.0


# ---------------------------------------------------------------------------
# Risk Management State features (6)
# ---------------------------------------------------------------------------

def _compute_risk_state_features(features, spy, close, log_close):
    """Compute 6 risk management state features.

    These track hypothetical session P&L, drawdown, consecutive losses,
    and time remaining — teaching the model to manage risk intraday.

    Uses a simple momentum-based proxy position (since we don't have actual
    model positions during feature computation).
    """
    # Proxy position: sign of 1-hour return (what a simple trend-follower does)
    proxy_pos = np.sign(log_close.diff(12)).fillna(0)

    # Hourly returns
    hourly_ret = log_close.diff(12).fillna(0)

    # Session P&L (cumulative within each day)
    proxy_pnl = proxy_pos.shift(1) * hourly_ret  # position taken 1h ago * return
    session_pnl = proxy_pnl.groupby(spy['date']).cumsum()
    features['session_pnl'] = session_pnl

    # Position duration (how many consecutive bars same direction)
    pos_sign = proxy_pos
    pos_change = (pos_sign != pos_sign.shift(1)).astype(int)
    duration = pos_change.groupby(pos_change.cumsum()).cumcount()
    features['position_duration'] = (duration / 78).clip(0, 1)  # normalize by session length

    # Drawdown from session peak P&L
    session_peak = session_pnl.groupby(spy['date']).cummax()
    features['drawdown_from_session_peak'] = session_pnl - session_peak

    # Consecutive losing hourly bars
    is_loss = (proxy_pnl < 0).astype(int)
    # Count consecutive losses
    not_loss = (proxy_pnl >= 0).astype(int)
    loss_groups = not_loss.cumsum()
    consec = is_loss.groupby(loss_groups).cumsum()
    features['consecutive_losses'] = (consec / 6).clip(0, 1)  # normalize: 6 = full day of losses

    # Time to close: 1 at open, 0 at close
    minutes_since_open = (
        spy['timestamp'].dt.hour * 60 + spy['timestamp'].dt.minute - (9 * 60 + 30)
    ).clip(0, 390)
    features['time_to_close'] = 1.0 - (minutes_since_open / 390.0)

    # Theta remaining: how much theta burn is left today
    # Theta accelerates through the day (most burn in last 2 hours)
    # Model as sqrt(time_remaining) — options lose value slower early, faster late
    features['theta_remaining'] = np.sqrt(features['time_to_close'].clip(0, 1))


# ---------------------------------------------------------------------------
# Options-adjusted target context (6)
# ---------------------------------------------------------------------------

def _compute_options_target_context(features, spy, close, raw_data):
    """Compute 6 features that help the model understand options P&L math.

    These tell the model: "given current Greeks, how big of a move do you need
    to make money on an ATM option, and what's the cost of being wrong?"
    """
    options_snaps = raw_data.get('options_snapshots', pd.DataFrame())
    spy_date_str = spy['date'].astype(str)

    if options_snaps.empty:
        # Reasonable defaults for typical ATM SPY options
        features['current_atm_price'] = 0.005  # ~$2.50 / $500 underlying
        features['breakeven_move'] = 0.005      # ~0.5% move to break even
        features['expected_theta_cost'] = 0.001  # small hourly theta
        features['delta_adjusted_leverage'] = 100.0  # typical leverage
        features['gamma_pnl_potential'] = 0.0
        features['edge_after_costs'] = 0.003    # need ~30bps alpha
        return

    opts = options_snaps.set_index('date')

    def _map_opt(col, default=0.0):
        if col in opts.columns:
            return spy_date_str.map(opts[col]).fillna(default).astype(float).values
        return np.full(len(spy), default)

    # Current ATM option mid-price (normalized by underlying)
    atm_mid = _map_opt('atm_call_mid', 2.50)
    features['current_atm_price'] = atm_mid / np.maximum(close.values, 1.0)

    # Breakeven move: how much underlying needs to move to cover the premium
    # For ATM option: breakeven ≈ premium / (delta * 100 shares)
    delta = np.maximum(np.abs(_map_opt('atm_call_delta', 0.50)), 0.01)
    features['breakeven_move'] = (atm_mid / (delta * np.maximum(close.values, 1.0)))

    # Expected theta cost for a 1-hour hold
    # Daily theta / 6.5 trading hours
    theta_daily = np.abs(_map_opt('atm_call_theta', 0.01))
    features['expected_theta_cost'] = (theta_daily / 6.5) / np.maximum(close.values, 1.0)

    # Delta-adjusted leverage: underlying price * delta / option price
    features['delta_adjusted_leverage'] = (
        close.values * delta / np.maximum(atm_mid, 0.01)
    )

    # Gamma P&L potential: gamma * expected_move^2 * 100
    # Expected move ≈ IV * sqrt(1/252/6.5) * price (1-hour expected move)
    iv = _map_opt('atm_iv', 0.20)
    expected_move = iv * np.sqrt(1 / 252 / 6.5) * close.values
    gamma = _map_opt('atm_call_gamma', 0.01)
    features['gamma_pnl_potential'] = 0.5 * gamma * expected_move**2 * 100 / np.maximum(close.values, 1.0)

    # Edge after costs: minimum alpha needed to profit
    # = (theta_hourly + spread_cost) / (delta * price)
    spread_cost = _map_opt('atm_call_bid_ask', 0.01) * atm_mid  # dollar spread
    theta_hourly = theta_daily / 6.5
    features['edge_after_costs'] = (theta_hourly + spread_cost) / (delta * np.maximum(close.values, 1.0))


# ---------------------------------------------------------------------------
# Options-adjusted target
# ---------------------------------------------------------------------------

def _compute_options_adjusted_target(base_target, features, spy, raw_data):
    """Compute options-adjusted target.

    Instead of raw equity return, the target is:
    P&L = delta * return * leverage - hourly_theta_cost

    This teaches the model to account for:
    1. Delta leverage (ATM option amplifies returns by ~100x per contract)
    2. Theta decay (holding costs money even if direction is right)
    3. The asymmetry of options (gamma means convex payoff)

    Falls back to base_target (raw return) if no options data available.
    """
    options_snaps = raw_data.get('options_snapshots', pd.DataFrame())

    if options_snaps.empty:
        return base_target

    spy_date_str = spy['date'].astype(str)
    opts = options_snaps.set_index('date')

    def _map_opt(col, default=0.0):
        if col in opts.columns:
            return spy_date_str.map(opts[col]).fillna(default).astype(float).values
        return np.full(len(spy), default)

    delta = _map_opt('atm_call_delta', 0.50)
    theta = _map_opt('atm_call_theta', -0.01)
    gamma = _map_opt('atm_call_gamma', 0.01)
    atm_mid = np.maximum(_map_opt('atm_call_mid', 2.50), 0.01)

    base = base_target.values.copy()
    base = np.nan_to_num(base, nan=0.0)

    # Options P&L per dollar of underlying:
    # pnl = delta * return + 0.5 * gamma * return^2 * price - theta/6.5/price
    # Normalized by option price to get return on option premium
    price = np.maximum(spy['close'].astype(float).values, 1.0)

    delta_pnl = delta * base * price           # delta P&L in dollars
    gamma_pnl = 0.5 * gamma * (base * price)**2  # gamma P&L (convexity bonus)
    theta_cost = np.abs(theta) / 6.5           # hourly theta cost in dollars

    option_pnl = (delta_pnl + gamma_pnl - theta_cost) / atm_mid  # return on premium

    # Clip extreme values
    option_pnl = np.clip(option_pnl, -2.0, 2.0)

    return pd.Series(option_pnl, index=base_target.index)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_NO_NORMALIZE = {
    'time_of_day_sin', 'time_of_day_cos', 'day_of_week_sin', 'day_of_week_cos',
    'minutes_since_open', 'ema_ribbon_aligned', 'trin_extreme',
    'session_position', 'time_to_close', 'theta_remaining',
    'consecutive_losses', 'position_duration',
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
        y: (batch,)                        -- options-adjusted next-hour returns
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
    path = os.path.join(DATA_DIR, "intraday_raw_v3.pkl")
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
    Walk-forward Sharpe ratio on held-out validation period.
    **Higher is better.**

    Now evaluates on OPTIONS-ADJUSTED returns: the target already encodes
    delta leverage and theta cost, so the Sharpe reflects actual options P&L.

    PnL per hour = position_t * options_adjusted_return_{t->t+1hr} - |delta_pos| * cost
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
    parser = argparse.ArgumentParser(description="Prepare intraday + options data for autoresearch-trading v0.3")
    parser.add_argument("--start", type=str, default="2024-04-01", help="Start date for intraday data")
    parser.add_argument("--end", type=str, default=None, help="End date (default: today)")
    parser.add_argument("--polygon-key", type=str, default=None, help="Polygon.io API key")
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

    print("Computing intraday + options features...")
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
