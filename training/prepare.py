"""
Autoresearch-trading v2: data prep for SPX 0DTE options sniper model.

Data sources:
  - SPX prices:   Real SPX index via IBKR (--use-spx, recommended)
  - SPY volume:   SPY ETF via IBKR (SPX index has no volume; SPY is the
                   most liquid equity ETF — its volume is a legitimate signal)
  - SPXW options: Polygon flat files (S3 bulk data, 4-year rolling window)
  - VIX:          Real CBOE VIX index via IBKR

Date range: March 14, 2022 → present (~4 years, flat file limit).
0DTE schedule: Mon/Wed/Fri only before May 11, 2022; daily after.
Bar resolution: 1-minute (390 bars/day RTH).

Computes 60 trader-relevant features (VWAP bands, session structure,
key levels, volume profile, trend, VIX/regime, OTM/skew, Greeks, etc.)
and prepares tensors for train.py.

The model outputs 4 discrete actions: DO_NOTHING, BUY_CALL, BUY_PUT, EXIT.
Evaluation simulates actual 0DTE option trades with stops, targets,
and model-driven exits.

Usage:
  python3 prepare.py --use-spx --ib-port 4002   # Full download
  python3 prepare.py --skip-download --use-spx   # Rebuild from cache
  python3 prepare.py --quick --use-spx            # ~40 day dry run

Requires POLYGON_S3_KEY_ID + POLYGON_S3_SECRET env vars (for SPXW flat files).
Requires IB Gateway running (for SPY, SPX, VIX).
"""
from __future__ import annotations

import os, sys, time, math, argparse, pickle, gzip
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as _norm_dist
from scipy.optimize import brentq as _brentq

# ---------------------------------------------------------------------------
# Constants (fixed — define the evaluation contract)
# ---------------------------------------------------------------------------

TIME_BUDGET       = int(os.environ.get("TIME_BUDGET", 300))  # training seconds (5 min default)
BARS_PER_DAY      = 390        # 1-min bars in RTH (9:30-16:00 ET)
FORWARD_BARS      = 30         # prediction horizon: 30 x 1min = 30 min
LOOKBACK_WINDOW   = 100        # rolling window for vol/volume stats (~100 min)
MIN_TRADES        = 5          # minimum trades for valid evaluation
ANNUAL_TRADING_BARS = 252 * BARS_PER_DAY
ANNUAL_TRADING_HOURS = ANNUAL_TRADING_BARS  # compat alias

# 0DTE option trade simulation parameters
OPTION_SPREAD_BPS    = 50      # bid-ask spread on 0DTE ATM in bps of premium
STOP_LOSS_PCT        = 0.30    # 30% stop loss on premium (from journals)
MAX_HOLD_BARS        = 60      # max hold = 60 bars = 60 minutes
STOP_COOLDOWN_BARS   = 5       # 5-bar (5-min) cooldown after stop loss before re-entry
NO_TRADE_BEFORE_BAR  = 30      # first 30 bars (9:30-9:59) are hard no-trade
MAX_TRADE_RETURN     = 2.0     # cap individual trade P&L at 200% (eliminate fat-tail lottery)
STARTING_CAPITAL     = 5000.0  # starting account balance for equity curve simulation
RISK_PER_TRADE       = 0.10   # fraction of capital risked per trade (10% = $500 from $5000)
BAR_SIZE_MINUTES     = 1           # 1-minute bar resolution
ATM_DELTA            = 0.50    # ATM delta approximation
SPX_MULTIPLIER       = 100     # option multiplier
THETA_DECAY_DAILY    = 0.04    # rough daily theta as fraction of ATM premium

RTH_OPEN  = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)

# 0DTE expiration schedule
# Before May 11, 2022: SPXW 0DTE only on Mon(0), Wed(2), Fri(4)
# After May 11, 2022: daily 0DTE (Mon-Fri)
DAILY_0DTE_START = '2022-05-11'

def is_0dte_day(day_str: str) -> bool:
    """Return True if SPXW 0DTE options existed on this trading day."""
    if day_str >= DAILY_0DTE_START:
        return True  # daily 0DTE
    dow = dt.datetime.strptime(day_str, '%Y-%m-%d').weekday()
    return dow in (0, 2, 4)  # Mon, Wed, Fri only

CACHE_DIR    = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-trading")
DATA_DIR     = os.path.join(CACHE_DIR, "data")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")

# ---------------------------------------------------------------------------
# Feature names (60 features: 39 equity + 6 options + 4 VIX/regime + 6 OTM/skew + 5 Greeks)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # === Price returns (5) ===
    'ret_1',                # 1-bar (5min) return
    'ret_3',                # 3-bar (15min) return
    'ret_6',                # 6-bar (30min) return
    'ret_12',               # 12-bar (1hr) return
    'ret_24',               # 24-bar (2hr) return
    # === Volume (3) ===
    'volume_ratio',         # bar volume / 20-bar SMA
    'volume_zscore',        # (volume - mean) / std
    'volume_at_price_pctile',  # current close vs session volume profile
    # === Volatility (3) ===
    'bar_range',            # (high - low) / close
    'realized_vol',         # 20-bar rolling stdev of returns
    'range_ratio',          # current bar range / 20-bar avg range
    # === VWAP (6) ===
    'vwap_dist',            # (close - session VWAP) / close
    'vwap_slope',           # change in VWAP distance over 6 bars
    'vwap_upper1_dist',     # distance to VWAP + 1 stdev
    'vwap_lower1_dist',     # distance to VWAP - 1 stdev
    'vwap_upper2_dist',     # distance to VWAP + 2 stdev
    'vwap_lower2_dist',     # distance to VWAP - 2 stdev
    # === Session structure (5) ===
    'ib_high_dist',         # distance to initial balance high
    'ib_low_dist',          # distance to initial balance low
    'ib_width',             # IB range / close (normalized)
    'am_range_pct',         # range expansion ratio: session range / IB range
    'session_range_pct',    # full session range so far / close
    # === Key levels (6) ===
    'onh_dist',             # distance to overnight gap high (max of open, prev close)
    'onl_dist',             # distance to overnight gap low (min of open, prev close)
    'prev_high_dist',       # distance to previous day high
    'prev_low_dist',        # distance to previous day low
    'prev_close_dist',      # distance to previous day close
    'prev_vwap_dist',       # distance to previous day VWAP
    # === Trend structure (3) ===
    'trend_hh_hl',          # +1 if HH+HL pattern (uptrend), -1 if LH+LL
    'ema_cross',            # (EMA8 - EMA21) / close (momentum)
    'close_position',       # where close sits in bar range: (c-l)/(h-l)
    # === Microstructure (2) ===
    'gap',                  # overnight gap, carried all day
    'inside_bar',           # 1 if current bar inside previous bar
    # === Time (6) ===
    'minutes_to_close',     # log(minutes remaining + 1), normalized
    'time_sin',             # sin(2pi * session_progress)
    'time_cos',             # cos(2pi * session_progress)
    'day_of_week',          # 0=Mon..4=Fri, scaled to [-1,1]
    'half_hour_proximity',  # closeness to next half-hour mark (avoid entries)
    'ib_complete',          # 1 if initial balance period is done (past 10:00)
    # === Options (6) ===
    'atm_iv',               # ATM implied vol (avg of call + put IV)
    'iv_skew',              # put IV - call IV (fear/skew premium)
    'atm_premium_pct',      # ATM call premium / estimated SPX price
    'put_call_vol_ratio',   # log(put_volume / call_volume)
    'option_volume',        # log(total option volume + 1)
    'theta_rate',           # estimated theta per bar / premium
    # === VIX / Regime (4) ===
    # Uses real CBOE VIX index from IBKR when available (download_vix_bars),
    # falls back to ATM IV proxy when IBKR not connected. Both produce
    # annualized vol in decimal form (0.15 = VIX 15). Real VIX preferred
    # because it matches live broker feeds exactly — no distribution mismatch.
    'vix_level',            # CBOE VIX (or ATM IV fallback), z-score normalized
    'vix_change',           # 6-bar change in VIX (vol momentum, 30-min window)
    'vix_regime',           # regime bucket: -1=low(<15), -0.33=normal, 0.33=elevated, 1=crisis(>30)
    'vrp',                  # variance risk premium: atm_iv^2 - realized_vol^2
    # === OTM / Skew Profile (6) ===
    'otm_call_iv_5',        # IV of ATM+5 call (1 strike OTM)
    'otm_put_iv_5',         # IV of ATM-5 put (1 strike OTM)
    'iv_skew_5',            # IV(ATM-5 put) - IV(ATM+5 call): near-term skew
    'iv_term_call',         # IV(ATM+10 call) - IV(ATM+5 call): call wing steepness
    'iv_term_put',          # IV(ATM-10 put) - IV(ATM-5 put): put wing steepness
    'otm_vol_ratio',        # log(OTM volume / ATM volume): flow concentration
    # === Greeks (5) ===
    'atm_delta',            # ATM call delta (0-1, directional sensitivity)
    'atm_gamma',            # ATM call gamma (delta sensitivity to price)
    'atm_theta_per_bar',    # ATM theta per 1-min bar (time decay per bar)
    'atm_vega',             # ATM vega per 1% IV move (vol sensitivity)
    'gamma_theta_ratio',    # gamma / |theta_per_bar|: bang-for-buck (convexity vs decay)
]

NUM_FEATURES = len(FEATURE_NAMES)

# Features that should NOT be z-score normalized
_NO_NORMALIZE = {
    'time_sin', 'time_cos', 'day_of_week', 'close_position',
    'minutes_to_close', 'half_hour_proximity', 'ib_complete',
    'inside_bar', 'trend_hh_hl',
    'vix_regime',  # categorical 0-3, already scaled
}

# Action labels for the 8-class model
# Gate head: NO_TRADE / TRADE → combined with direction head for full action
# Direction head: 6 outputs [CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]
ACTION_DO_NOTHING    = 0
ACTION_BUY_CALL_ATM  = 1
ACTION_BUY_CALL_OTM5 = 2
ACTION_BUY_CALL_OTM10= 3
ACTION_BUY_PUT_ATM   = 4
ACTION_BUY_PUT_OTM5  = 5
ACTION_BUY_PUT_OTM10 = 6
ACTION_EXIT          = 7
NUM_ACTIONS          = 8

# Legacy aliases for backward compatibility with train.py imports
ACTION_BUY_CALL = ACTION_BUY_CALL_ATM
ACTION_BUY_PUT  = ACTION_BUY_PUT_ATM

# Option P&L target: round-trip spread cost as fraction of premium
SPREAD_COST_PCT   = 2 * OPTION_SPREAD_BPS / 10000.0  # 1% round-trip

# EXIT label: profit target threshold (fraction of premium)
EXIT_PROFIT_TARGET = 0.20  # exit when unrealized P&L > 20% of premium

# ---------------------------------------------------------------------------
# Black-Scholes (European, for SPXW 0DTE IV)
# ---------------------------------------------------------------------------

def _bs_d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def _bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_dist.cdf(d1) - K * math.exp(-r * T) * _norm_dist.cdf(d2)

def _bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_dist.cdf(-d2) - S * _norm_dist.cdf(-d1)

def _bs_iv(price, S, K, T, r, is_call=True):
    """Implied vol via Brent's method. Returns NaN on failure."""
    if T <= 1e-10 or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
    if price < intrinsic - 0.01:
        return np.nan
    fn = _bs_call if is_call else _bs_put
    try:
        return _brentq(lambda s: fn(S, K, T, r, s) - price, 0.01, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan

def _bs_greeks(S, K, T, r, sigma):
    """Black-Scholes Greeks for ATM call. Returns (delta, gamma, theta_per_bar, vega).
    All values are per-unit (not multiplied by contract size).
    Returns (nan, nan, nan, nan) on bad inputs."""
    if T <= 1e-10 or sigma <= 0 or S <= 0:
        return np.nan, np.nan, np.nan, np.nan
    sqrt_T = math.sqrt(T)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt_T
    nd1 = _norm_dist.cdf(d1)
    npdf_d1 = _norm_dist.pdf(d1)

    delta = nd1
    gamma = npdf_d1 / (S * sigma * sqrt_T)
    # Theta: annualized, then convert to per-bar (1 bar = 1 min = 1/(252*390) years)
    theta_annual = (-(S * npdf_d1 * sigma) / (2.0 * sqrt_T)
                    - r * K * math.exp(-r * T) * _norm_dist.cdf(d2))
    theta_per_bar = theta_annual / (252.0 * BARS_PER_DAY)
    vega = S * npdf_d1 * sqrt_T / 100.0  # per 1% IV move

    return delta, gamma, theta_per_bar, vega

# ---------------------------------------------------------------------------
# Download (unchanged)
# ---------------------------------------------------------------------------

def _polygon_client():
    key = os.environ.get('POLYGON_API_KEY')
    if not key:
        print("ERROR: Set POLYGON_API_KEY environment variable")
        sys.exit(1)
    from polygon import RESTClient
    return RESTClient(key)

def _ib_client(host: str = "127.0.0.1", port: int = None, client_id: int = 10):
    """Connect to IB Gateway. Returns IB client or exits."""
    if port is None:
        port = int(os.environ.get("IB_PORT", "4001"))
    try:
        from ib_insync import IB
    except ImportError:
        print("ERROR: pip install ib_insync")
        sys.exit(1)
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=15)
    except Exception as e:
        print(f"ERROR: Cannot connect to IB Gateway at {host}:{port}: {e}")
        print("  Run ib_probe.py to diagnose.")
        sys.exit(1)
    return ib


_s3 = None
def _s3_client():
    """Cached S3 client for Polygon flat files."""
    global _s3
    if _s3 is not None:
        return _s3
    import boto3
    key_id = os.environ.get('POLYGON_S3_KEY_ID')
    secret = os.environ.get('POLYGON_S3_SECRET')
    if not key_id or not secret:
        return None
    _s3 = boto3.client(
        's3',
        endpoint_url='https://files.massive.com',
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )
    return _s3


def download_es_bars(start: str, end: str) -> pd.DataFrame:
    """Download ES mini / SPX index 5-min bars via IBKR.

    Tries ES continuous front-month first, falls back to I:SPX cash index.
    IBKR limits: ~2 years of 5-min data, max 1-day requests for intraday.
    """
    from ib_insync import Future, Index, util

    ib = _ib_client()
    print(f"Downloading ES/SPX 5-min bars {start} -> {end} via IBKR...")

    # Try ES mini continuous first
    try:
        es = Future("ES", exchange="CME", currency="USD")
        contracts = ib.reqContractDetails(es)
        contract = contracts[0].contract if contracts else None
    except Exception:
        contract = None

    if contract is None:
        # Fall back to SPX cash index
        print("  ES not available, falling back to I:SPX")
        contract = Index("SPX", exchange="CBOE", currency="USD")
        ib.qualifyContracts(contract)

    all_bars = []
    current = dt.datetime.strptime(start, '%Y-%m-%d')
    end_dt = dt.datetime.strptime(end, '%Y-%m-%d')

    while current < end_dt:
        # IBKR requires end datetime, one day at a time for 5-min bars
        day_end = current + dt.timedelta(days=1)
        end_str = day_end.strftime('%Y%m%d 16:00:00 US/Eastern')
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=True,
            )
            if bars:
                for b in bars:
                    all_bars.append({
                        'timestamp': int(b.date.timestamp() * 1000),
                        'open': b.open, 'high': b.high,
                        'low': b.low, 'close': b.close,
                        'volume': b.volume,
                        'vwap': getattr(b, 'average', None),
                    })
                if len(all_bars) % 500 == 0:
                    print(f"  {current.strftime('%Y-%m-%d')}: {len(bars)} bars (total {len(all_bars)})")
        except Exception as e:
            print(f"  {current.strftime('%Y-%m-%d')}: FAIL {e}")
        current = day_end
        ib.sleep(0.5)  # rate limit

    ib.disconnect()

    if not all_bars:
        print("ERROR: No ES/SPX bars downloaded")
        sys.exit(1)

    df = pd.DataFrame(all_bars)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('US/Eastern')
    df = df.sort_values('datetime').reset_index(drop=True)

    df['time'] = df['datetime'].dt.time
    df = df[(df['time'] >= RTH_OPEN) & (df['time'] < RTH_CLOSE)].copy()
    df['date'] = df['datetime'].dt.date.astype(str)

    print(f"  Total RTH bars: {len(df)} across {df['date'].nunique()} days")
    return df


def _download_ibkr_index(symbol: str, start: str, end: str,
                          col_prefix: str = "") -> pd.DataFrame | None:
    """Download CBOE index 1-min bars via IBKR using weekly batches.

    Works for VIX, SPX, and any CBOE index. Uses '1 W' duration per request
    (max allowed for 1-min bars).

    IBKR quirk: index contracts reject 'US/Eastern' timezone suffix in
    endDateTime. Use bare format: '20250314 16:00:00'.

    Returns DataFrame with columns: date, timestamp, {prefix}open/high/low/close.
    Returns None if no data (caller should handle fallback).
    """
    from ib_insync import Index as IBIndex

    ib = _ib_client()
    prefix = f"{col_prefix}_" if col_prefix else ""
    print(f"Downloading {symbol} 1-min bars {start} -> {end} via IBKR (weekly batches)...")

    contract = IBIndex(symbol=symbol, exchange="CBOE", currency="USD")
    ib.qualifyContracts(contract)
    print(f"  {symbol} contract qualified: conId={contract.conId}")

    all_bars = []
    # Walk forward in 1-week steps (max duration for 1-min bars)
    current = dt.datetime.strptime(start, '%Y-%m-%d')
    end_dt = dt.datetime.strptime(end, '%Y-%m-%d')

    while current < end_dt:
        week_end = min(current + dt.timedelta(days=7), end_dt)
        end_str = week_end.strftime('%Y%m%d 16:00:00')
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr="1 W",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                timeout=30,
            )
            if bars:
                for b in bars:
                    all_bars.append({
                        'timestamp': int(b.date.timestamp() * 1000),
                        f'{prefix}open': b.open, f'{prefix}high': b.high,
                        f'{prefix}low': b.low, f'{prefix}close': b.close,
                    })
                print(f"  {current.strftime('%Y-%m-%d')}: {len(bars)} bars (total {len(all_bars)})")
            else:
                print(f"  {current.strftime('%Y-%m-%d')}: 0 bars")
        except Exception as e:
            print(f"  {current.strftime('%Y-%m-%d')}: FAIL {e}")
        current = week_end
        ib.sleep(2)  # rate limit — prevent IBKR output buffer overflow

    ib.disconnect()

    if not all_bars:
        print(f"WARNING: No {symbol} bars downloaded.")
        return None

    df = pd.DataFrame(all_bars)
    # Deduplicate — weekly windows may overlap at boundaries
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('US/Eastern')

    df['time'] = df['datetime'].dt.time
    df = df[(df['time'] >= RTH_OPEN) & (df['time'] < RTH_CLOSE)].copy()
    df['date'] = df['datetime'].dt.date.astype(str)

    print(f"  Total RTH {symbol} bars: {len(df)} across {df['date'].nunique()} days")
    return df


def download_vix_bars(start: str, end: str) -> pd.DataFrame | None:
    """Download CBOE VIX index 1-min bars via IBKR.

    This is the real CBOE VIX index — same values available from any broker
    feed in live trading, so features trained on this transfer directly.
    """
    return _download_ibkr_index("VIX", start, end, col_prefix="vix")


def download_spx_bars(start: str, end: str) -> pd.DataFrame | None:
    """Download real SPX index 1-min bars via IBKR.

    Replaces the SPY×10 proxy with actual SPX values. Since SPXW options
    are priced against SPX, using the real index eliminates tracking error.
    """
    return _download_ibkr_index("SPX", start, end, col_prefix="spx")


def download_spy_bars(start: str, end: str) -> pd.DataFrame:
    """Download SPY 5-min bars from Polygon.

    Used for SPY ETF volume (SPX index has no volume). When --use-spx is
    set, OHLC prices are replaced with real SPX from IBKR but SPY volume
    is kept as a market-participation signal.
    """
    client = _polygon_client()
    print(f"Downloading SPY 5-min bars {start} -> {end}...")

    all_bars = []
    current = dt.datetime.strptime(start, '%Y-%m-%d')
    end_dt = dt.datetime.strptime(end, '%Y-%m-%d')

    while current < end_dt:
        chunk_end = min(current + dt.timedelta(days=30), end_dt)
        try:
            bars = list(client.get_aggs(
                'SPY', 1, 'minute',
                current.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d'),
                limit=50000
            ))
            all_bars.extend(bars)
            print(f"  {current.strftime('%Y-%m-%d')}: {len(bars)} bars")
        except Exception as e:
            print(f"  {current.strftime('%Y-%m-%d')}: FAIL {e}")
        current = chunk_end
        time.sleep(0.2)  # Polygon Developer tier — no rate limit

    if not all_bars:
        print("ERROR: No SPY bars downloaded")
        sys.exit(1)

    df = pd.DataFrame([{
        'timestamp': b.timestamp,
        'open': b.open, 'high': b.high, 'low': b.low, 'close': b.close,
        'volume': b.volume, 'vwap': getattr(b, 'vwap', None),
    } for b in all_bars])

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('US/Eastern')
    df = df.sort_values('datetime').reset_index(drop=True)

    df['time'] = df['datetime'].dt.time
    df = df[(df['time'] >= RTH_OPEN) & (df['time'] < RTH_CLOSE)].copy()
    df['date'] = df['datetime'].dt.date.astype(str)

    print(f"  Total RTH bars: {len(df)} across {df['date'].nunique()} days")
    return df


def download_spy_bars_ibkr(start: str, end: str) -> pd.DataFrame:
    """Download SPY ETF 1-min bars via IBKR (replaces Polygon for full history).

    IBKR has no date restrictions for SPY. Returns the same DataFrame format
    as download_spy_bars() so it's a drop-in replacement. Volume is the key
    data — OHLC gets replaced by SPX prices when --use-spx is set.
    """
    from ib_insync import Stock

    ib = _ib_client(client_id=99)
    print(f"Downloading SPY 1-min bars {start} -> {end} via IBKR (weekly batches)...")

    spy = Stock('SPY', 'ARCA', 'USD')
    ib.qualifyContracts(spy)
    print(f"  SPY qualified: conId={spy.conId}")

    all_bars = []
    current = dt.datetime.strptime(start, '%Y-%m-%d')
    end_dt = dt.datetime.strptime(end, '%Y-%m-%d')

    while current < end_dt:
        week_end = min(current + dt.timedelta(days=7), end_dt)
        end_str = week_end.strftime('%Y%m%d 16:00:00 US/Eastern')
        try:
            bars = ib.reqHistoricalData(
                spy,
                endDateTime=end_str,
                durationStr="1 W",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                timeout=30,
            )
            if bars:
                for b in bars:
                    all_bars.append({
                        'timestamp': int(b.date.timestamp() * 1000),
                        'open': b.open, 'high': b.high,
                        'low': b.low, 'close': b.close,
                        'volume': b.volume, 'vwap': b.average,
                    })
                print(f"  {current.strftime('%Y-%m-%d')}: {len(bars)} bars (total {len(all_bars)})")
            else:
                print(f"  {current.strftime('%Y-%m-%d')}: 0 bars")
        except Exception as e:
            print(f"  {current.strftime('%Y-%m-%d')}: FAIL {e}")
        current = week_end
        ib.sleep(2)  # rate limit — prevent IBKR output buffer overflow

    ib.disconnect()

    if not all_bars:
        print("ERROR: No SPY bars from IBKR")
        sys.exit(1)

    df = pd.DataFrame(all_bars)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('US/Eastern')
    df['time'] = df['datetime'].dt.time
    df = df[(df['time'] >= RTH_OPEN) & (df['time'] < RTH_CLOSE)].copy()
    df['date'] = df['datetime'].dt.date.astype(str)

    print(f"  Total RTH bars: {len(df)} across {df['date'].nunique()} days")
    return df


def download_spxw_full(spy_df: pd.DataFrame) -> dict:
    """Download SPXW 0DTE ATM call+put 1-min bars for ALL trading days.

    Caches per-day pickles in DATA_DIR/spxw/ to avoid re-downloading.
    ATM strike: uses df open price directly when --use-spx (real SPX),
    otherwise estimates via SPY open * 10.
    Falls back to SPX (non-W) ticker for monthly expiration dates.

    Returns dict keyed by (date_str, timestamp) with option bar data.
    """
    client = _polygon_client()
    cache_dir = os.path.join(DATA_DIR, "spxw")
    os.makedirs(cache_dir, exist_ok=True)

    unique_days = sorted(spy_df['date'].unique())
    options_data = {}
    downloaded = 0
    cached = 0
    failed = 0

    print(f"Downloading SPXW 0DTE options for {len(unique_days)} trading days...")

    for day_str in unique_days:
        day_cache = os.path.join(cache_dir, f"{day_str}.pkl")

        # Use cache if available
        if os.path.exists(day_cache):
            with open(day_cache, 'rb') as f:
                day_data = pickle.load(f)
            options_data.update(day_data)
            cached += 1
            continue

        # Determine ATM strike from day's open price
        day_bars = spy_df[spy_df['date'] == day_str]
        if day_bars.empty:
            continue

        day_open = float(day_bars.iloc[0]['open'])
        # If prices are SPY-scale (<1000), multiply by 10 to get SPX-scale
        spx_est = day_open * 10.0 if day_open < 1000 else day_open
        atm_strike = round(spx_est / 5) * 5

        # Build SPXW 0DTE tickers
        day_dt = dt.datetime.strptime(day_str, '%Y-%m-%d')
        yymmdd = day_dt.strftime('%y%m%d')
        call_tk = f"O:SPXW{yymmdd}C{int(atm_strike * 1000):08d}"
        put_tk = f"O:SPXW{yymmdd}P{int(atm_strike * 1000):08d}"

        # Download call bars
        try:
            call_bars = list(client.get_aggs(call_tk, 1, 'minute', day_str, day_str, limit=50000))
        except Exception:
            call_bars = []
        time.sleep(0.12)  # Polygon Developer tier

        # Download put bars
        try:
            put_bars = list(client.get_aggs(put_tk, 1, 'minute', day_str, day_str, limit=50000))
        except Exception:
            put_bars = []
        time.sleep(0.12)  # Polygon Developer tier

        # Fallback: try SPX (non-W) for monthly expiration dates (3rd Friday)
        if not call_bars and not put_bars:
            call_tk2 = f"O:SPX{yymmdd}C{int(atm_strike * 1000):08d}"
            put_tk2 = f"O:SPX{yymmdd}P{int(atm_strike * 1000):08d}"
            try:
                call_bars = list(client.get_aggs(call_tk2, 1, 'minute', day_str, day_str, limit=50000))
            except Exception:
                pass
            time.sleep(0.12)  # Polygon Developer tier
            try:
                put_bars = list(client.get_aggs(put_tk2, 1, 'minute', day_str, day_str, limit=50000))
            except Exception:
                pass
            time.sleep(0.12)  # Polygon Developer tier

        if not call_bars and not put_bars:
            failed += 1
            # Cache empty result to avoid re-trying
            with open(day_cache, 'wb') as f:
                pickle.dump({}, f)
            continue

        # Build per-timestamp lookup
        call_by_ts = {b.timestamp: b for b in call_bars}
        put_by_ts = {b.timestamp: b for b in put_bars}
        all_ts = set(call_by_ts) | set(put_by_ts)

        day_data = {}
        for ts in all_ts:
            cb = call_by_ts.get(ts)
            pb = put_by_ts.get(ts)
            day_data[(day_str, ts)] = {
                'strike': atm_strike,
                'call_close': cb.close if cb else np.nan,
                'call_open': cb.open if cb else np.nan,
                'call_high': cb.high if cb else np.nan,
                'call_low': cb.low if cb else np.nan,
                'call_volume': cb.volume if cb else 0,
                'put_close': pb.close if pb else np.nan,
                'put_open': pb.open if pb else np.nan,
                'put_high': pb.high if pb else np.nan,
                'put_low': pb.low if pb else np.nan,
                'put_volume': pb.volume if pb else 0,
            }

        # Cache this day
        with open(day_cache, 'wb') as f:
            pickle.dump(day_data, f)

        options_data.update(day_data)
        downloaded += 1

        if downloaded % 20 == 0:
            print(f"  Progress: {downloaded} downloaded, {cached} cached, {failed} no data")

    print(f"  Total: {downloaded} downloaded, {cached} cached, {failed} no data")
    print(f"  Option bar-pairs: {len(options_data)}")
    return options_data


def download_spxw_chain(spy_df: pd.DataFrame) -> dict:
    """Download SPXW 0DTE OTM option bars at ATM±5 and ATM±10 strikes.

    Downloads 4 contracts per day (ATM+5 call, ATM+10 call, ATM-5 put, ATM-10 put).
    These provide OTM IV features and raw prices for OTM tradeable actions.
    Caches per-day pickles in DATA_DIR/spxw_chain/ to avoid re-downloading.

    Returns dict keyed by (date_str, timestamp) with OTM bar data.
    """
    client = _polygon_client()
    cache_dir = os.path.join(DATA_DIR, "spxw_chain")
    os.makedirs(cache_dir, exist_ok=True)

    unique_days = sorted(spy_df['date'].unique())
    chain_data = {}
    downloaded = 0
    cached = 0
    failed = 0

    print(f"Downloading SPXW OTM chain (ATM±5, ±10) for {len(unique_days)} trading days...")

    for day_str in unique_days:
        day_cache = os.path.join(cache_dir, f"{day_str}.pkl")

        if os.path.exists(day_cache):
            with open(day_cache, 'rb') as f:
                day_data = pickle.load(f)
            chain_data.update(day_data)
            cached += 1
            continue

        # Determine ATM strike
        day_bars = spy_df[spy_df['date'] == day_str]
        if day_bars.empty:
            continue

        day_open = float(day_bars.iloc[0]['open'])
        spx_est = day_open * 10.0 if day_open < 1000 else day_open
        atm_strike = round(spx_est / 5) * 5

        day_dt = dt.datetime.strptime(day_str, '%Y-%m-%d')
        yymmdd = day_dt.strftime('%y%m%d')

        # 4 OTM contracts: ATM+5 call, ATM+10 call, ATM-5 put, ATM-10 put
        otm_specs = [
            ('otm5_call',  'C', atm_strike + 5),
            ('otm10_call', 'C', atm_strike + 10),
            ('otm5_put',   'P', atm_strike - 5),
            ('otm10_put',  'P', atm_strike - 10),
        ]

        day_bars_data = {}
        any_data = False

        for label, cp, strike in otm_specs:
            tk = f"O:SPXW{yymmdd}{cp}{int(strike * 1000):08d}"
            try:
                bars = list(client.get_aggs(tk, 1, 'minute', day_str, day_str, limit=50000))
            except Exception:
                bars = []
            time.sleep(0.12)  # Polygon Developer tier

            # Fallback to SPX (non-W) for monthly expiration dates
            if not bars:
                tk2 = f"O:SPX{yymmdd}{cp}{int(strike * 1000):08d}"
                try:
                    bars = list(client.get_aggs(tk2, 1, 'minute', day_str, day_str, limit=50000))
                except Exception:
                    pass
                time.sleep(0.12)  # Polygon Developer tier

            if bars:
                any_data = True
                for b in bars:
                    key = (day_str, b.timestamp)
                    if key not in day_bars_data:
                        day_bars_data[key] = {'atm_strike': atm_strike}
                    day_bars_data[key][f'{label}_close'] = b.close
                    day_bars_data[key][f'{label}_volume'] = b.volume
                    day_bars_data[key][f'{label}_strike'] = strike

        if not any_data:
            failed += 1
            with open(day_cache, 'wb') as f:
                pickle.dump({}, f)
            continue

        with open(day_cache, 'wb') as f:
            pickle.dump(day_bars_data, f)

        chain_data.update(day_bars_data)
        downloaded += 1

        if downloaded % 20 == 0:
            print(f"  Progress: {downloaded} downloaded, {cached} cached, {failed} no data")

    print(f"  Total: {downloaded} downloaded, {cached} cached, {failed} no data")
    print(f"  OTM bar entries: {len(chain_data)}")
    return chain_data


def prefetch_spxw_from_flatfiles(spy_df: pd.DataFrame, api_cutoff: str = None):
    """Download SPXW 0DTE option bars from Polygon flat files (S3).

    This is the ONLY source for option data — no Polygon REST API needed.
    Downloads full-day option flat files, extracts SPXW 0DTE bars for the
    specific contracts needed (ATM call/put + OTM chain), and saves to
    per-day pickle caches.

    Only processes days that are valid 0DTE expiration days (MWF before
    May 11, 2022; daily after).

    Flat files contain 1-min bars — no aggregation needed.
    """
    s3 = _s3_client()
    if s3 is None:
        print("  Flat files: no S3 credentials (set POLYGON_S3_KEY_ID + POLYGON_S3_SECRET)")
        return

    atm_cache_dir = os.path.join(DATA_DIR, "spxw")
    chain_cache_dir = os.path.join(DATA_DIR, "spxw_chain")
    os.makedirs(atm_cache_dir, exist_ok=True)
    os.makedirs(chain_cache_dir, exist_ok=True)

    # spy_df should already be filtered to 0DTE days, but enforce it
    unique_days = sorted(d for d in spy_df['date'].unique() if is_0dte_day(d))
    if api_cutoff:
        unique_days = [d for d in unique_days if d < api_cutoff]

    # Filter to days where BOTH caches are missing
    days_needed = []
    for day_str in unique_days:
        atm_exists = os.path.exists(os.path.join(atm_cache_dir, f"{day_str}.pkl"))
        chain_exists = os.path.exists(os.path.join(chain_cache_dir, f"{day_str}.pkl"))
        if not atm_exists or not chain_exists:
            days_needed.append(day_str)

    if not days_needed:
        print(f"  Flat files: all {len(unique_days)} days already cached")
        return

    print(f"  Flat files: {len(days_needed)} days to download (of {len(unique_days)} pre-API)")
    downloaded = 0
    failed = 0

    for day_str in days_needed:
        day_dt = dt.datetime.strptime(day_str, '%Y-%m-%d')
        yymmdd = day_dt.strftime('%y%m%d')
        s3_key = f"us_options_opra/minute_aggs_v1/{day_dt.year}/{day_dt.month:02d}/{day_str}.csv.gz"

        # Determine ATM strike
        day_bars = spy_df[spy_df['date'] == day_str]
        if day_bars.empty:
            continue
        day_open = float(day_bars.iloc[0]['open'])
        spx_est = day_open * 10.0 if day_open < 1000 else day_open
        atm_strike = round(spx_est / 5) * 5

        # Target ticker prefixes (0DTE = expires today)
        target_prefix = f"O:SPXW{yymmdd}"
        target_prefix_spx = f"O:SPX{yymmdd}"

        # Build expected ticker strings for each contract
        strikes = {
            'atm_call': ('C', atm_strike),
            'atm_put': ('P', atm_strike),
            'otm5_call': ('C', atm_strike + 5),
            'otm10_call': ('C', atm_strike + 10),
            'otm5_put': ('P', atm_strike - 5),
            'otm10_put': ('P', atm_strike - 10),
        }
        target_tickers = {}
        for label, (cp, strike) in strikes.items():
            tk_w = f"O:SPXW{yymmdd}{cp}{int(strike * 1000):08d}"
            tk_spx = f"O:SPX{yymmdd}{cp}{int(strike * 1000):08d}"
            target_tickers[label] = (tk_w, tk_spx, strike)

        try:
            obj = s3.get_object(Bucket='flatfiles', Key=s3_key)
            raw = gzip.decompress(obj['Body'].read())
        except Exception as e:
            # Day not available in flat files — cache empty
            for cache_dir in [atm_cache_dir, chain_cache_dir]:
                cache_path = os.path.join(cache_dir, f"{day_str}.pkl")
                if not os.path.exists(cache_path):
                    with open(cache_path, 'wb') as f:
                        pickle.dump({}, f)
            failed += 1
            continue

        # Parse relevant lines: ticker,volume,open,close,high,low,window_start,transactions
        bars_by_ticker = defaultdict(list)
        for line in raw.decode('utf-8').split('\n'):
            if not line:
                continue
            if not (line.startswith(target_prefix) or line.startswith(target_prefix_spx)):
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            bars_by_ticker[parts[0]].append(parts)

        # Build ATM data dict (same format as download_spxw_full)
        atm_data = {}
        atm_call_tk, atm_call_spx, _ = target_tickers['atm_call']
        atm_put_tk, atm_put_spx, _ = target_tickers['atm_put']

        call_bars_raw = bars_by_ticker.get(atm_call_tk, []) or bars_by_ticker.get(atm_call_spx, [])
        put_bars_raw = bars_by_ticker.get(atm_put_tk, []) or bars_by_ticker.get(atm_put_spx, [])

        call_by_ts = {}
        for parts in call_bars_raw:
            ts_ms = int(parts[6]) // 1_000_000  # ns -> ms
            call_by_ts[ts_ms] = {
                'open': float(parts[2]), 'close': float(parts[3]),
                'high': float(parts[4]), 'low': float(parts[5]),
                'volume': int(float(parts[1])),
            }

        put_by_ts = {}
        for parts in put_bars_raw:
            ts_ms = int(parts[6]) // 1_000_000
            put_by_ts[ts_ms] = {
                'open': float(parts[2]), 'close': float(parts[3]),
                'high': float(parts[4]), 'low': float(parts[5]),
                'volume': int(float(parts[1])),
            }

        all_ts = set(call_by_ts) | set(put_by_ts)
        for ts in all_ts:
            cb = call_by_ts.get(ts)
            pb = put_by_ts.get(ts)
            atm_data[(day_str, ts)] = {
                'strike': atm_strike,
                'call_close': cb['close'] if cb else np.nan,
                'call_open': cb['open'] if cb else np.nan,
                'call_high': cb['high'] if cb else np.nan,
                'call_low': cb['low'] if cb else np.nan,
                'call_volume': cb['volume'] if cb else 0,
                'put_close': pb['close'] if pb else np.nan,
                'put_open': pb['open'] if pb else np.nan,
                'put_high': pb['high'] if pb else np.nan,
                'put_low': pb['low'] if pb else np.nan,
                'put_volume': pb['volume'] if pb else 0,
            }

        # Build OTM chain data dict (same format as download_spxw_chain)
        chain_day_data = {}
        for label in ['otm5_call', 'otm10_call', 'otm5_put', 'otm10_put']:
            tk_w, tk_spx, strike = target_tickers[label]
            raw_bars = bars_by_ticker.get(tk_w, []) or bars_by_ticker.get(tk_spx, [])
            for parts in raw_bars:
                ts_ms = int(parts[6]) // 1_000_000
                key = (day_str, ts_ms)
                if key not in chain_day_data:
                    chain_day_data[key] = {'atm_strike': atm_strike}
                chain_day_data[key][f'{label}_close'] = float(parts[3])
                chain_day_data[key][f'{label}_volume'] = int(float(parts[1]))
                chain_day_data[key][f'{label}_strike'] = strike

        # Cache both ATM and OTM data
        with open(os.path.join(atm_cache_dir, f"{day_str}.pkl"), 'wb') as f:
            pickle.dump(atm_data, f)
        with open(os.path.join(chain_cache_dir, f"{day_str}.pkl"), 'wb') as f:
            pickle.dump(chain_day_data, f)

        downloaded += 1
        if downloaded % 20 == 0:
            print(f"  Flat files: {downloaded}/{len(days_needed)} done, {failed} unavailable")

    print(f"  Flat files: {downloaded} downloaded, {failed} unavailable, "
          f"{len(unique_days) - len(days_needed)} already cached")


def download_spxw_ibkr(spy_df: pd.DataFrame, dates: list = None):
    """Download SPXW 0DTE option bars via IBKR for dates missing from cache.

    Fallback when Polygon S3 flat files aren't available (new/recent days).
    Downloads ATM call/put + OTM chain (6 contracts) via reqHistoricalData.
    Saves to same per-day pickle caches as prefetch_spxw_from_flatfiles.
    """
    from ib_insync import Option as IBOption

    atm_cache_dir = os.path.join(DATA_DIR, "spxw")
    chain_cache_dir = os.path.join(DATA_DIR, "spxw_chain")
    os.makedirs(atm_cache_dir, exist_ok=True)
    os.makedirs(chain_cache_dir, exist_ok=True)

    target_dates = dates if dates else sorted(spy_df['date'].unique())
    target_dates = [d for d in target_dates if is_0dte_day(d)]

    # Filter to dates that need downloading (both ATM and chain must be non-empty)
    days_needed = []
    for day_str in target_dates:
        atm_cache = os.path.join(atm_cache_dir, f"{day_str}.pkl")
        chain_cache = os.path.join(chain_cache_dir, f"{day_str}.pkl")
        # Skip if both caches exist and have real data (>100 bytes = not empty pickle)
        if (os.path.exists(atm_cache) and os.path.getsize(atm_cache) > 100 and
            os.path.exists(chain_cache) and os.path.getsize(chain_cache) > 100):
            continue
        days_needed.append(day_str)

    if not days_needed:
        print(f"  IBKR options: all {len(target_dates)} days already cached")
        return

    print(f"  IBKR options: downloading {len(days_needed)} days...")
    ib = _ib_client(client_id=20)  # separate client_id to avoid conflicts
    downloaded = 0
    failed = 0

    for day_str in days_needed:
        # Get ATM strike from SPX opening price
        day_bars = spy_df[spy_df['date'] == day_str]
        if day_bars.empty:
            continue
        day_open = float(day_bars.iloc[0]['open'])
        spx_est = day_open if day_open >= 1000 else day_open * 10.0
        atm_strike = round(spx_est / 5) * 5

        # Expiry for 0DTE = same day
        expiry = day_str.replace('-', '')  # YYYYMMDD

        # Define the 6 target contracts
        targets = {
            'atm_call':   ('C', atm_strike),
            'atm_put':    ('P', atm_strike),
            'otm5_call':  ('C', atm_strike + 5),
            'otm10_call': ('C', atm_strike + 10),
            'otm5_put':   ('P', atm_strike - 5),
            'otm10_put':  ('P', atm_strike - 10),
        }

        # Download 1-min bars for each contract
        bars_by_label = {}
        end_str = f"{expiry} 16:00:00"
        all_ok = True

        for label, (right, strike) in targets.items():
            contract = IBOption(
                symbol='SPX', lastTradeDateOrContractMonth=expiry,
                strike=float(strike), right=right,
                exchange='SMART', currency='USD',
                tradingClass='SPXW',
            )
            try:
                ib.qualifyContracts(contract)
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_str,
                    durationStr="1 D",
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=True,
                    timeout=30,
                )
                bar_dict = {}
                for b in (bars or []):
                    ts_ms = int(b.date.timestamp() * 1000)
                    bar_dict[ts_ms] = {
                        'open': b.open, 'close': b.close,
                        'high': b.high, 'low': b.low,
                        'volume': b.volume,
                    }
                bars_by_label[label] = bar_dict
            except Exception as e:
                print(f"    {day_str} {label} strike={strike}: FAIL {e}")
                bars_by_label[label] = {}
                all_ok = False
            ib.sleep(1)  # rate limit

        # Build ATM cache (same format as flat files)
        atm_data = {}
        call_bars = bars_by_label.get('atm_call', {})
        put_bars = bars_by_label.get('atm_put', {})
        all_ts = set(call_bars) | set(put_bars)
        for ts in all_ts:
            cb = call_bars.get(ts)
            pb = put_bars.get(ts)
            atm_data[(day_str, ts)] = {
                'strike': atm_strike,
                'call_close': cb['close'] if cb else np.nan,
                'call_open': cb['open'] if cb else np.nan,
                'call_high': cb['high'] if cb else np.nan,
                'call_low': cb['low'] if cb else np.nan,
                'call_volume': cb['volume'] if cb else 0,
                'put_close': pb['close'] if pb else np.nan,
                'put_open': pb['open'] if pb else np.nan,
                'put_high': pb['high'] if pb else np.nan,
                'put_low': pb['low'] if pb else np.nan,
                'put_volume': pb['volume'] if pb else 0,
            }

        # Build OTM chain cache (same format as flat files)
        chain_data = {}
        for label in ['otm5_call', 'otm10_call', 'otm5_put', 'otm10_put']:
            _, strike = targets[label]
            for ts, bar in bars_by_label.get(label, {}).items():
                key = (day_str, ts)
                if key not in chain_data:
                    chain_data[key] = {'atm_strike': atm_strike}
                chain_data[key][f'{label}_close'] = bar['close']
                chain_data[key][f'{label}_volume'] = bar['volume']
                chain_data[key][f'{label}_strike'] = strike

        # Save caches
        with open(os.path.join(atm_cache_dir, f"{day_str}.pkl"), 'wb') as f:
            pickle.dump(atm_data, f)
        with open(os.path.join(chain_cache_dir, f"{day_str}.pkl"), 'wb') as f:
            pickle.dump(chain_data, f)

        n_atm = len(atm_data)
        n_chain = len(chain_data)
        if n_atm > 0:
            downloaded += 1
            print(f"    {day_str}: ATM={n_atm} bars, OTM chain={n_chain} bars (strike={atm_strike})")
        else:
            failed += 1
            print(f"    {day_str}: no option data available")

    ib.disconnect()
    print(f"  IBKR options: {downloaded} downloaded, {failed} failed")


def load_spxw_caches(spy_df: pd.DataFrame) -> dict:
    """Load ATM option data from per-day pickle caches (created by prefetch_spxw_from_flatfiles)."""
    cache_dir = os.path.join(DATA_DIR, "spxw")
    unique_days = sorted(spy_df['date'].unique())
    options_data = {}
    loaded = 0
    missing = 0
    for day_str in unique_days:
        day_cache = os.path.join(cache_dir, f"{day_str}.pkl")
        if os.path.exists(day_cache):
            with open(day_cache, 'rb') as f:
                day_data = pickle.load(f)
            options_data.update(day_data)
            if day_data:
                loaded += 1
        else:
            missing += 1
    print(f"  ATM caches: {loaded} days with data, {missing} missing")
    print(f"  ATM bar-pairs: {len(options_data)}")
    return options_data


def load_spxw_chain_caches(spy_df: pd.DataFrame) -> dict:
    """Load OTM chain data from per-day pickle caches (created by prefetch_spxw_from_flatfiles)."""
    cache_dir = os.path.join(DATA_DIR, "spxw_chain")
    unique_days = sorted(spy_df['date'].unique())
    chain_data = {}
    loaded = 0
    missing = 0
    for day_str in unique_days:
        day_cache = os.path.join(cache_dir, f"{day_str}.pkl")
        if os.path.exists(day_cache):
            with open(day_cache, 'rb') as f:
                day_data = pickle.load(f)
            chain_data.update(day_data)
            if day_data:
                loaded += 1
        else:
            missing += 1
    print(f"  OTM caches: {loaded} days with data, {missing} missing")
    print(f"  OTM bar entries: {len(chain_data)}")
    return chain_data


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_session_vwap_bands(close, volume, day_mask_indices):
    """Compute session VWAP and standard deviation bands for indices within one day."""
    n = len(day_mask_indices)
    vwap_vals = np.full(n, np.nan)
    upper1 = np.full(n, np.nan)
    lower1 = np.full(n, np.nan)
    upper2 = np.full(n, np.nan)
    lower2 = np.full(n, np.nan)

    cum_pv = 0.0
    cum_vol = 0.0
    cum_pv2 = 0.0

    for k, idx in enumerate(day_mask_indices):
        c = close[idx]
        v = max(volume[idx], 1.0)
        cum_pv += c * v
        cum_vol += v
        cum_pv2 += c * c * v

        vw = cum_pv / cum_vol
        vwap_vals[k] = vw
        if cum_vol > 1 and k > 0:
            variance = (cum_pv2 / cum_vol) - vw * vw
            std = math.sqrt(max(variance, 0.0))
            upper1[k] = vw + std
            lower1[k] = vw - std
            upper2[k] = vw + 2 * std
            lower2[k] = vw - 2 * std

    return vwap_vals, upper1, lower1, upper2, lower2


def compute_features(df: pd.DataFrame, options_data: dict | None = None,
                     vix_data: dict | None = None,
                     chain_data: dict | None = None) -> tuple:
    """Compute 55 trader-relevant features from SPX 1-min bars (+ SPY volume) + SPXW options.

    Returns: (features_array, targets_array, dates_list, valid_mask, option_prices)
    option_prices is a dict with 'atm_call', 'atm_put', 'strike', 'call_pnl',
    'put_pnl', 'exit_call_label', 'exit_put_label', and OTM price arrays (per-bar).
    """
    N = len(df)
    feat = np.full((N, NUM_FEATURES), np.nan, dtype=np.float32)
    targets = np.full(N, np.nan, dtype=np.float32)
    # Option price arrays for trade simulation (not model features)
    atm_call_prices = np.full(N, np.nan, dtype=np.float32)
    atm_put_prices = np.full(N, np.nan, dtype=np.float32)
    atm_strikes = np.full(N, np.nan, dtype=np.float32)
    # OTM price arrays for Phase 5B-v2 tradeable actions
    otm5_call_prices = np.full(N, np.nan, dtype=np.float32)
    otm5_put_prices = np.full(N, np.nan, dtype=np.float32)
    otm10_call_prices = np.full(N, np.nan, dtype=np.float32)
    otm10_put_prices = np.full(N, np.nan, dtype=np.float32)
    dates = df['date'].values
    timestamps = df['datetime'].dt.strftime('%Y-%m-%d %H:%M').values if 'datetime' in df.columns else dates

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    opn = df['open'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    # Pre-compute log returns
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate([[0.0], log_ret])

    # Pre-compute EMAs
    ema8 = pd.Series(close).ewm(span=40, adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=105, adjust=False).mean().values

    # Pre-compute per-day data
    unique_dates = sorted(set(dates))
    day_indices = {d: np.where(dates == d)[0] for d in unique_dates}

    # Pre-compute overnight highs/lows, prev day stats, initial balance
    prev_day_high = {}
    prev_day_low = {}
    prev_day_close_val = {}
    prev_day_vwap_val = {}
    overnight_high = {}
    overnight_low = {}
    ib_high_map = {}
    ib_low_map = {}

    for di, day in enumerate(unique_dates):
        idx = day_indices[day]
        # Previous day stats
        if di > 0:
            prev = unique_dates[di - 1]
            pidx = day_indices[prev]
            prev_day_high[day] = np.max(high[pidx])
            prev_day_low[day] = np.min(low[pidx])
            prev_day_close_val[day] = close[pidx[-1]]
            # Prev day VWAP
            pv = np.sum(close[pidx] * np.maximum(volume[pidx], 1.0))
            tv = np.sum(np.maximum(volume[pidx], 1.0))
            prev_day_vwap_val[day] = pv / tv

        # Overnight H/L = prev day close to current open gap area
        # Use opening price ± opening bar range as overnight reference
        if di > 0:
            prev = unique_dates[di - 1]
            overnight_high[day] = max(opn[idx[0]], close[day_indices[prev][-1]])
            overnight_low[day] = min(opn[idx[0]], close[day_indices[prev][-1]])

        # Initial balance: first 30 bars = 30 minutes (9:30-10:00)
        ib_bars = idx[:min(30, len(idx))]
        ib_high_map[day] = np.max(high[ib_bars])
        ib_low_map[day] = np.min(low[ib_bars])

    # Pre-compute VWAP bands per day
    vwap_cache = {}
    for day in unique_dates:
        idx = day_indices[day]
        vw, u1, l1, u2, l2 = _compute_session_vwap_bands(close, volume, idx)
        for k, i in enumerate(idx):
            vwap_cache[i] = (vw[k], u1[k], l1[k], u2[k], l2[k])

    # -----------------------------------------------------------------------
    # Main feature loop
    # -----------------------------------------------------------------------
    for i in range(N):
        fi = 0
        day = dates[i]
        c = close[i]

        # === Returns (5) ===
        for lag in [5, 15, 30, 60, 120]:
            if i >= lag:
                feat[i, fi] = (c / close[i - lag]) - 1.0
            fi += 1

        # === Volume (3) ===
        if i >= LOOKBACK_WINDOW:
            vol_window = volume[i - LOOKBACK_WINDOW:i]
            vol_mean = np.mean(vol_window)
            vol_std = np.std(vol_window)
            feat[i, fi] = volume[i] / max(vol_mean, 1.0)
            fi += 1
            feat[i, fi] = (volume[i] - vol_mean) / max(vol_std, 1.0)
            fi += 1
        else:
            fi += 2

        # Volume at price percentile (where does current close sit in session volume profile?)
        didx = day_indices[day]
        day_pos = np.searchsorted(didx, i)
        if day_pos > 2:
            session_close = close[didx[:day_pos + 1]]
            session_vol = volume[didx[:day_pos + 1]]
            # Volume below current price / total volume
            below_mask = session_close <= c
            vol_below = np.sum(session_vol[below_mask])
            vol_total = np.sum(session_vol)
            feat[i, fi] = vol_below / max(vol_total, 1.0)
        fi += 1

        # === Volatility (3) ===
        bar_range = (high[i] - low[i]) / max(c, 1.0)
        feat[i, fi] = bar_range
        fi += 1

        if i >= LOOKBACK_WINDOW:
            feat[i, fi] = np.std(log_ret[i - LOOKBACK_WINDOW:i])
        fi += 1

        if i >= LOOKBACK_WINDOW:
            ranges = (high[i - LOOKBACK_WINDOW:i] - low[i - LOOKBACK_WINDOW:i]) / np.maximum(close[i - LOOKBACK_WINDOW:i], 1.0)
            feat[i, fi] = bar_range / max(np.mean(ranges), 1e-8)
        fi += 1

        # === VWAP (6) ===
        vw_data = vwap_cache.get(i)
        if vw_data is not None:
            vw, u1, l1, u2, l2 = vw_data
            feat[i, fi] = (c - vw) / max(c, 1.0)
            fi += 1
            # VWAP slope
            if i >= 30:
                prev_vw = vwap_cache.get(i - 30)
                if prev_vw is not None:
                    feat[i, fi] = feat[i, fi - 1] - (close[i - 30] - prev_vw[0]) / max(close[i - 30], 1.0)
            fi += 1
            # VWAP bands
            if not np.isnan(u1):
                feat[i, fi] = (c - u1) / max(c, 1.0)
                feat[i, fi + 1] = (c - l1) / max(c, 1.0)
                feat[i, fi + 2] = (c - u2) / max(c, 1.0)
                feat[i, fi + 3] = (c - l2) / max(c, 1.0)
            fi += 4
        else:
            fi += 6

        # === Session structure (5) ===
        ib_h = ib_high_map.get(day, c)
        ib_l = ib_low_map.get(day, c)
        feat[i, fi] = (c - ib_h) / max(c, 1.0)
        fi += 1
        feat[i, fi] = (c - ib_l) / max(c, 1.0)
        fi += 1
        feat[i, fi] = (ib_h - ib_l) / max(c, 1.0)
        fi += 1

        # AM range = range expansion ratio: session range so far / IB range
        session_so_far = didx[:day_pos + 1]
        session_high = np.max(high[session_so_far])
        session_low = np.min(low[session_so_far])
        ib_range = max(ib_h - ib_l, 1e-8)
        feat[i, fi] = (session_high - session_low) / ib_range  # range expansion
        fi += 1
        # Session range (absolute, normalized by price)
        feat[i, fi] = (session_high - session_low) / max(c, 1.0)
        fi += 1

        # === Key levels (6) ===
        if day in overnight_high:
            feat[i, fi] = (c - overnight_high[day]) / max(c, 1.0)
            feat[i, fi + 1] = (c - overnight_low[day]) / max(c, 1.0)
        fi += 2

        if day in prev_day_high:
            feat[i, fi] = (c - prev_day_high[day]) / max(c, 1.0)
            fi += 1
            feat[i, fi] = (c - prev_day_low[day]) / max(c, 1.0)
            fi += 1
            feat[i, fi] = (c - prev_day_close_val[day]) / max(c, 1.0)
            fi += 1
            feat[i, fi] = (c - prev_day_vwap_val[day]) / max(c, 1.0)
            fi += 1
        else:
            fi += 4

        # === Trend structure (3) ===
        # HH/HL pattern on recent 1m bars (look back 60 bars = 1 hour)
        if i >= 60:
            recent_h = high[i - 60:i + 1]
            recent_l = low[i - 60:i + 1]
            mid = len(recent_h) // 2
            first_half_h = np.max(recent_h[:mid])
            second_half_h = np.max(recent_h[mid:])
            first_half_l = np.min(recent_l[:mid])
            second_half_l = np.min(recent_l[mid:])
            if second_half_h > first_half_h and second_half_l > first_half_l:
                feat[i, fi] = 1.0   # uptrend (HH + HL)
            elif second_half_h < first_half_h and second_half_l < first_half_l:
                feat[i, fi] = -1.0  # downtrend (LH + LL)
            else:
                feat[i, fi] = 0.0   # range/chop
        fi += 1

        # EMA cross
        feat[i, fi] = (ema8[i] - ema21[i]) / max(c, 1.0)
        fi += 1

        # Close position in bar
        if high[i] > low[i]:
            feat[i, fi] = (c - low[i]) / (high[i] - low[i])
        else:
            feat[i, fi] = 0.5
        fi += 1

        # === Microstructure (2) ===
        # Gap
        day_start = didx[0]
        if day_start > 0:
            feat[i, fi] = (opn[day_start] / close[day_start - 1]) - 1.0
        else:
            feat[i, fi] = 0.0
        fi += 1

        # Inside bar
        if i > 0:
            feat[i, fi] = 1.0 if (high[i] <= high[i-1] and low[i] >= low[i-1]) else 0.0
        else:
            feat[i, fi] = 0.0
        fi += 1

        # === Time (6) ===
        bar_dt = df.iloc[i]['datetime']
        bar_time = bar_dt.time()
        minutes_into = (bar_time.hour * 60 + bar_time.minute) - (9 * 60 + 30)
        total_session = 390
        minutes_remaining = max(total_session - minutes_into, 0)
        session_progress = minutes_into / total_session

        feat[i, fi] = np.log1p(minutes_remaining) / np.log1p(total_session)
        fi += 1
        feat[i, fi] = np.sin(2 * np.pi * session_progress)
        fi += 1
        feat[i, fi] = np.cos(2 * np.pi * session_progress)
        fi += 1
        dow = bar_dt.weekday()
        feat[i, fi] = (dow - 2.0) / 2.0
        fi += 1

        # Half-hour proximity: distance to next :00 or :30 minute mark
        minute = bar_time.minute
        to_next_half = min(minute % 30, 30 - (minute % 30))
        feat[i, fi] = to_next_half / 15.0  # 0 = at half-hour, 1 = 15 min away
        fi += 1

        # IB complete (past 10:00 AM)
        feat[i, fi] = 1.0 if minutes_into >= 30 else 0.0
        fi += 1

        # === Options (6) ===
        opt_key = (dates[i], int(df.iloc[i]['timestamp']))
        opt = options_data.get(opt_key) if options_data else None
        if opt is not None and not np.isnan(opt.get('call_close', np.nan)):
            # SPX price: if close is already SPX-scale (≥1000), use directly
            spx = c if c >= 1000 else c * 10.0
            K = opt['strike']
            T = minutes_remaining / (252.0 * 390.0)
            r = 0.05

            # Store raw option prices for trade simulation
            atm_call_prices[i] = opt['call_close']
            atm_put_prices[i] = opt.get('put_close', np.nan)
            atm_strikes[i] = K

            # 39: atm_iv (average of call + put IV)
            call_iv = _bs_iv(opt['call_close'], spx, K, T, r, is_call=True)
            put_iv = np.nan
            put_close = opt.get('put_close', np.nan)
            if not np.isnan(put_close):
                put_iv = _bs_iv(put_close, spx, K, T, r, is_call=False)
            if not np.isnan(call_iv) and not np.isnan(put_iv):
                feat[i, fi] = (call_iv + put_iv) / 2.0
            elif not np.isnan(call_iv):
                feat[i, fi] = call_iv
            elif not np.isnan(put_iv):
                feat[i, fi] = put_iv
            fi += 1

            # 40: iv_skew (put IV - call IV, captures fear premium)
            if not np.isnan(call_iv) and not np.isnan(put_iv):
                feat[i, fi] = put_iv - call_iv
            fi += 1

            # 41: atm_premium_pct (call premium as fraction of SPX)
            feat[i, fi] = opt['call_close'] / max(spx, 1.0)
            fi += 1

            # 42: put_call_vol_ratio
            cv, pv = opt.get('call_volume', 0), opt.get('put_volume', 0)
            feat[i, fi] = np.log(max(pv, 1) / max(cv, 1))
            fi += 1

            # 43: option_volume (log total option activity)
            feat[i, fi] = np.log1p(cv + pv)
            fi += 1

            # 44: theta_rate (estimated theta per bar as fraction of premium)
            if T > 1e-10 and not np.isnan(call_iv) and call_iv > 0:
                theta_bar = call_iv * spx / (2.0 * math.sqrt(2.0 * math.pi * T)) / (252.0 * BARS_PER_DAY)
                feat[i, fi] = theta_bar / max(opt['call_close'], 0.01)
            fi += 1
        else:
            fi += 6  # skip all 6 options features

        # === VIX / Regime (4) ===
        # Uses real CBOE VIX index when available (from IBKR download),
        # falls back to ATM IV proxy when not. Both are annualized vol:
        # VIX 15 ≈ atm_iv 0.15. Real VIX is preferred because it matches
        # what's available from any broker feed in live trading.
        ts_ms = int(df.iloc[i]['timestamp'])
        vix_bar = vix_data.get(ts_ms) if vix_data else None
        cur_iv = feat[i, 39]   # atm_iv (already stored above)
        cur_rv = feat[i, 9]    # realized_vol (stored in volatility section)

        # Determine VIX value: real CBOE VIX or ATM IV fallback
        if vix_bar is not None:
            vix_val = vix_bar['vix_close']       # real VIX (e.g., 15.3 = VIX 15.3)
            vix_annualized = vix_val / 100.0      # convert to decimal (0.153)
        elif not np.isnan(cur_iv):
            vix_val = cur_iv * 100.0              # ATM IV as VIX proxy
            vix_annualized = cur_iv
        else:
            vix_val = np.nan
            vix_annualized = np.nan

        if not np.isnan(vix_val):
            # 45: vix_level — VIX value in decimal form (z-scored later)
            feat[i, fi] = vix_annualized
            fi += 1

            # 46: vix_change — 30-bar change in VIX level (30-min vol momentum)
            if i >= 30 and not np.isnan(feat[i - 30, 45]):
                feat[i, fi] = vix_annualized - feat[i - 30, 45]
            fi += 1

            # 47: vix_regime — bucket (scaled to [-1, 1] range)
            # <15=low(-1), 15-20=normal(-0.33), 20-30=elevated(0.33), >30=crisis(1)
            if vix_val < 15.0:
                feat[i, fi] = -1.0
            elif vix_val < 20.0:
                feat[i, fi] = -0.33
            elif vix_val < 30.0:
                feat[i, fi] = 0.33
            else:
                feat[i, fi] = 1.0
            fi += 1

            # 48: vrp — variance risk premium: IV^2 - RV^2
            # Uses atm_iv (not VIX) for IV component since VRP is about
            # the spread between option-implied vol and realized vol
            if not np.isnan(cur_iv) and not np.isnan(cur_rv) and cur_rv > 0:
                feat[i, fi] = cur_iv ** 2 - cur_rv ** 2
            fi += 1
        else:
            fi += 4  # skip all 4 VIX features

        # === OTM / Skew Profile (6) ===
        chain_bar = chain_data.get(opt_key) if chain_data else None
        if chain_bar is not None and opt is not None:
            spx_for_iv = c if c >= 1000 else c * 10.0
            T_iv = minutes_remaining / (252.0 * 390.0)
            r_iv = 0.05

            # Extract OTM close prices and store for tradeable actions
            otm5c = chain_bar.get('otm5_call_close', np.nan)
            otm5p = chain_bar.get('otm5_put_close', np.nan)
            otm10c = chain_bar.get('otm10_call_close', np.nan)
            otm10p = chain_bar.get('otm10_put_close', np.nan)
            otm5c_strike = chain_bar.get('otm5_call_strike', np.nan)
            otm5p_strike = chain_bar.get('otm5_put_strike', np.nan)
            otm10c_strike = chain_bar.get('otm10_call_strike', np.nan)
            otm10p_strike = chain_bar.get('otm10_put_strike', np.nan)

            # Store raw OTM prices for Phase 5B-v2 trade simulation
            if not np.isnan(otm5c):
                otm5_call_prices[i] = otm5c
            if not np.isnan(otm5p):
                otm5_put_prices[i] = otm5p
            if not np.isnan(otm10c):
                otm10_call_prices[i] = otm10c
            if not np.isnan(otm10p):
                otm10_put_prices[i] = otm10p

            # Compute IVs for OTM strikes
            iv_otm5c = _bs_iv(otm5c, spx_for_iv, otm5c_strike, T_iv, r_iv, is_call=True) if not np.isnan(otm5c) and not np.isnan(otm5c_strike) else np.nan
            iv_otm5p = _bs_iv(otm5p, spx_for_iv, otm5p_strike, T_iv, r_iv, is_call=False) if not np.isnan(otm5p) and not np.isnan(otm5p_strike) else np.nan
            iv_otm10c = _bs_iv(otm10c, spx_for_iv, otm10c_strike, T_iv, r_iv, is_call=True) if not np.isnan(otm10c) and not np.isnan(otm10c_strike) else np.nan
            iv_otm10p = _bs_iv(otm10p, spx_for_iv, otm10p_strike, T_iv, r_iv, is_call=False) if not np.isnan(otm10p) and not np.isnan(otm10p_strike) else np.nan

            # 49: otm_call_iv_5
            if not np.isnan(iv_otm5c):
                feat[i, fi] = iv_otm5c
            fi += 1

            # 50: otm_put_iv_5
            if not np.isnan(iv_otm5p):
                feat[i, fi] = iv_otm5p
            fi += 1

            # 51: iv_skew_5 = IV(ATM-5 put) - IV(ATM+5 call)
            if not np.isnan(iv_otm5p) and not np.isnan(iv_otm5c):
                feat[i, fi] = iv_otm5p - iv_otm5c
            fi += 1

            # 52: iv_term_call = IV(ATM+10 call) - IV(ATM+5 call)
            if not np.isnan(iv_otm10c) and not np.isnan(iv_otm5c):
                feat[i, fi] = iv_otm10c - iv_otm5c
            fi += 1

            # 53: iv_term_put = IV(ATM-10 put) - IV(ATM-5 put)
            if not np.isnan(iv_otm10p) and not np.isnan(iv_otm5p):
                feat[i, fi] = iv_otm10p - iv_otm5p
            fi += 1

            # 54: otm_vol_ratio = log(OTM volume / ATM volume)
            otm_vol = sum(chain_bar.get(f'{k}_volume', 0) or 0
                          for k in ['otm5_call', 'otm5_put', 'otm10_call', 'otm10_put'])
            atm_vol = (opt.get('call_volume', 0) or 0) + (opt.get('put_volume', 0) or 0)
            if atm_vol > 0 and otm_vol > 0:
                feat[i, fi] = np.log(otm_vol / atm_vol)
            fi += 1
        else:
            fi += 6  # skip all 6 OTM features

        # === Greeks (5) ===
        # Compute Black-Scholes Greeks from ATM option data already extracted above.
        # Uses call_iv (computed in Options section) and same S, K, T, r.
        if opt is not None and not np.isnan(opt.get('call_close', np.nan)):
            spx_g = c if c >= 1000 else c * 10.0
            K_g = opt['strike']
            T_g = minutes_remaining / (252.0 * 390.0)
            r_g = 0.05
            # Use call_iv computed earlier (variable still in scope from Options section)
            sigma_g = call_iv if not np.isnan(call_iv) else np.nan
            if not np.isnan(sigma_g) and T_g > 1e-10:
                delta, gamma, theta_bar, vega = _bs_greeks(spx_g, K_g, T_g, r_g, sigma_g)
                # 55: atm_delta
                if not np.isnan(delta):
                    feat[i, fi] = delta
                fi += 1
                # 56: atm_gamma
                if not np.isnan(gamma):
                    feat[i, fi] = gamma
                fi += 1
                # 57: atm_theta_per_bar
                if not np.isnan(theta_bar):
                    feat[i, fi] = theta_bar
                fi += 1
                # 58: atm_vega
                if not np.isnan(vega):
                    feat[i, fi] = vega
                fi += 1
                # 59: gamma_theta_ratio (convexity bang-for-buck)
                if not np.isnan(gamma) and not np.isnan(theta_bar) and abs(theta_bar) > 1e-12:
                    feat[i, fi] = gamma / abs(theta_bar)
                fi += 1
            else:
                fi += 5  # skip all 5 Greeks features (no valid IV)
        else:
            fi += 5  # skip all 5 Greeks features (no option data)

        assert fi == NUM_FEATURES, f"Feature count mismatch: {fi} != {NUM_FEATURES}"

        # Target: forward FORWARD_BARS return (still needed for loss computation)
        if i + FORWARD_BARS < N and dates[i + FORWARD_BARS] == day:
            targets[i] = (close[i + FORWARD_BARS] / close[i]) - 1.0

    # -------------------------------------------------------------------
    # Option P&L targets: actual SPXW option return per bar
    # -------------------------------------------------------------------
    # For each bar i with option prices, compute the P&L of buying an ATM
    # call or put and holding for FORWARD_BARS (6 bars = 30 min).
    call_pnl = np.full(N, np.nan, dtype=np.float32)
    put_pnl = np.full(N, np.nan, dtype=np.float32)

    for i in range(N):
        exit_bar = i + FORWARD_BARS
        if exit_bar >= N:
            continue
        # Must be same trading day
        if dates[i] != dates[exit_bar]:
            continue
        # Need actual option prices at both entry and exit
        entry_call = atm_call_prices[i]
        exit_call = atm_call_prices[exit_bar]
        entry_put = atm_put_prices[i]
        exit_put = atm_put_prices[exit_bar]

        if not np.isnan(entry_call) and not np.isnan(exit_call) and entry_call > 0:
            call_pnl[i] = (exit_call - entry_call) / entry_call - SPREAD_COST_PCT
        if not np.isnan(entry_put) and not np.isnan(exit_put) and entry_put > 0:
            put_pnl[i] = (exit_put - entry_put) / entry_put - SPREAD_COST_PCT

    # -------------------------------------------------------------------
    # OTM P&L targets: same as above but for OTM strikes (for 6-class dir head)
    # -------------------------------------------------------------------
    otm5_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm5_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_put_pnl = np.full(N, np.nan, dtype=np.float32)

    for i in range(N):
        exit_bar = i + FORWARD_BARS
        if exit_bar >= N or dates[i] != dates[exit_bar]:
            continue
        for px_arr, pnl_arr in [
            (otm5_call_prices, otm5_call_pnl),
            (otm5_put_prices, otm5_put_pnl),
            (otm10_call_prices, otm10_call_pnl),
            (otm10_put_prices, otm10_put_pnl),
        ]:
            entry_px = px_arr[i]
            exit_px = px_arr[exit_bar]
            if not np.isnan(entry_px) and not np.isnan(exit_px) and entry_px > 0:
                pnl_arr[i] = (exit_px - entry_px) / entry_px - SPREAD_COST_PCT

    # -------------------------------------------------------------------
    # EXIT labels: should an open position be closed at this bar?
    # Fixed profit-target approach: EXIT=1 if unrealized P&L > 20% of premium
    # for a position opened 1-12 bars ago.
    # -------------------------------------------------------------------
    exit_call_label = np.full(N, np.nan, dtype=np.float32)
    exit_put_label = np.full(N, np.nan, dtype=np.float32)

    for i in range(N):
        best_call_exit = 0.0
        best_put_exit = 0.0
        has_call_data = False
        has_put_data = False

        # Check if a position opened K bars ago should be exited here
        for k in range(1, MAX_HOLD_BARS + 1):
            entry = i - k
            if entry < 0 or dates[entry] != dates[i]:
                continue

            # Call position: opened at entry, check unrealized P&L at bar i
            ec = atm_call_prices[entry]
            cc = atm_call_prices[i]
            if not np.isnan(ec) and not np.isnan(cc) and ec > 0:
                unrealized = (cc - ec) / ec - SPREAD_COST_PCT
                has_call_data = True
                if unrealized > EXIT_PROFIT_TARGET:
                    best_call_exit = 1.0

            # Put position
            ep = atm_put_prices[entry]
            cp = atm_put_prices[i]
            if not np.isnan(ep) and not np.isnan(cp) and ep > 0:
                unrealized = (cp - ep) / ep - SPREAD_COST_PCT
                has_put_data = True
                if unrealized > EXIT_PROFIT_TARGET:
                    best_put_exit = 1.0

        if has_call_data:
            exit_call_label[i] = best_call_exit
        if has_put_data:
            exit_put_label[i] = best_put_exit

    # Valid mask: only equity features (0:39) must be non-NaN.
    # Options (39:45) and VIX/regime (45:49) depend on SPXW data and may be NaN.
    equity_feat_end = 39  # first 39 features are pure equity (SPX prices + SPY volume)
    equity_valid = ~np.isnan(feat[:, :equity_feat_end]).any(axis=1)
    valid = equity_valid & ~np.isnan(targets)

    option_prices = {
        'atm_call': atm_call_prices,
        'atm_put': atm_put_prices,
        'strike': atm_strikes,
        'call_pnl': call_pnl,
        'put_pnl': put_pnl,
        'exit_call_label': exit_call_label,
        'exit_put_label': exit_put_label,
        'otm5_call': otm5_call_prices,
        'otm5_put': otm5_put_prices,
        'otm10_call': otm10_call_prices,
        'otm10_put': otm10_put_prices,
        'otm5_call_pnl': otm5_call_pnl,
        'otm5_put_pnl': otm5_put_pnl,
        'otm10_call_pnl': otm10_call_pnl,
        'otm10_put_pnl': otm10_put_pnl,
    }

    return feat, targets, dates.tolist(), valid, option_prices, timestamps.tolist()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_features(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Adaptive rolling z-score. Features in _NO_NORMALIZE are skipped."""
    out = features.copy()
    window = min(len(features) // 4, 500)
    window = max(window, 50)

    for j, name in enumerate(FEATURE_NAMES):
        if name in _NO_NORMALIZE:
            continue
        col = pd.Series(out[:, j], dtype=np.float64)
        col[~valid] = np.nan
        mu = col.rolling(window, min_periods=5).mean()
        sigma = col.rolling(window, min_periods=5).std()
        normalized = (col - mu) / sigma.clip(lower=1e-10)
        out[:, j] = normalized.fillna(0.0).values
    out = np.clip(out, -5.0, 5.0)
    out = np.nan_to_num(out, nan=0.0)
    return out


# ---------------------------------------------------------------------------
# Tensor preparation
# ---------------------------------------------------------------------------

def prepare_tensors(features: np.ndarray, targets: np.ndarray,
                    dates: list, valid: np.ndarray,
                    option_prices: dict | None = None,
                    timestamps: list | None = None) -> dict:
    """Build train/val split and save tensors."""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    unique_dates = sorted(set(dates))
    split_idx = int(len(unique_dates) * 0.7)
    train_dates = set(unique_dates[:split_idx])
    val_dates = set(unique_dates[split_idx:])

    train_end_idx = max(i for i, d in enumerate(dates) if d in train_dates)
    val_start_idx = min(i for i, d in enumerate(dates) if d in val_dates)
    val_end_idx = max(i for i, d in enumerate(dates) if d in val_dates)

    print(f"  Train: {len(train_dates)} days (idx 0-{train_end_idx})")
    print(f"  Val:   {len(val_dates)} days (idx {val_start_idx}-{val_end_idx})")

    data = {
        'features': torch.tensor(features, dtype=torch.float32),
        'targets': torch.tensor(targets, dtype=torch.float32),
        'valid_mask': torch.tensor(valid, dtype=torch.bool),
        'dates': dates,
        'timestamps': timestamps or dates,
        'train_end_idx': train_end_idx,
        'val_start_idx': val_start_idx,
        'val_end_idx': val_end_idx,
    }

    # Store option prices + P&L targets for trade simulation and training
    if option_prices is not None:
        data['atm_call_prices'] = torch.tensor(option_prices['atm_call'], dtype=torch.float32)
        data['atm_put_prices'] = torch.tensor(option_prices['atm_put'], dtype=torch.float32)
        data['atm_strikes'] = torch.tensor(option_prices['strike'], dtype=torch.float32)
        data['call_pnl'] = torch.tensor(option_prices['call_pnl'], dtype=torch.float32)
        data['put_pnl'] = torch.tensor(option_prices['put_pnl'], dtype=torch.float32)
        data['exit_call_label'] = torch.tensor(option_prices['exit_call_label'], dtype=torch.float32)
        data['exit_put_label'] = torch.tensor(option_prices['exit_put_label'], dtype=torch.float32)
        # OTM prices and P&L for tradeable OTM actions
        data['otm5_call_prices'] = torch.tensor(option_prices['otm5_call'], dtype=torch.float32)
        data['otm5_put_prices'] = torch.tensor(option_prices['otm5_put'], dtype=torch.float32)
        data['otm10_call_prices'] = torch.tensor(option_prices['otm10_call'], dtype=torch.float32)
        data['otm10_put_prices'] = torch.tensor(option_prices['otm10_put'], dtype=torch.float32)
        data['otm5_call_pnl'] = torch.tensor(option_prices['otm5_call_pnl'], dtype=torch.float32)
        data['otm5_put_pnl'] = torch.tensor(option_prices['otm5_put_pnl'], dtype=torch.float32)
        data['otm10_call_pnl'] = torch.tensor(option_prices['otm10_call_pnl'], dtype=torch.float32)
        data['otm10_put_pnl'] = torch.tensor(option_prices['otm10_put_pnl'], dtype=torch.float32)

    path = os.path.join(FEATURES_DIR, "data.pt")
    torch.save(data, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved: {path} ({size_mb:.1f} MB)")
    return data


# ---------------------------------------------------------------------------
# Public API (imported by train.py)
# ---------------------------------------------------------------------------

def load_data():
    """Load precomputed tensors.

    Checks multiple paths: the standard cache dir, the script directory,
    and /root/data.pt (for Akash containers where data is uploaded manually).
    """
    search_paths = [
        os.path.join(FEATURES_DIR, "data.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.pt"),
        "/root/data.pt",
    ]
    for path in search_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            return torch.load(path, map_location="cpu", weights_only=False)
    print(f"Data not found. Searched: {search_paths}")
    print("Run `python3 prepare.py` first.")
    sys.exit(1)


def make_dataloader(data, lookback, batch_size, split="train", device="cuda"):
    """Infinite (train) or single-pass (val) dataloader.

    Yields (x, y):
        x: (batch, lookback, NUM_FEATURES)
        y: tuple of (fwd_ret, call_pnl, put_pnl, exit_call, exit_put,
                     otm5_call_pnl, otm5_put_pnl, otm10_call_pnl, otm10_put_pnl)
           each (batch,). NaN where option data is unavailable.
    """
    features = data['features'].to(device)
    targets = data['targets'].to(device)
    valid_mask = data['valid_mask']

    # Option P&L targets (may be absent in old data.pt files)
    has_pnl = 'call_pnl' in data
    n_total = len(targets)
    if has_pnl:
        call_pnl_all = data['call_pnl'].to(device)
        put_pnl_all = data['put_pnl'].to(device)
        exit_call_all = data['exit_call_label'].to(device)
        exit_put_all = data['exit_put_label'].to(device)
    else:
        call_pnl_all = torch.full((n_total,), float('nan'), device=device)
        put_pnl_all = torch.full((n_total,), float('nan'), device=device)
        exit_call_all = torch.full((n_total,), float('nan'), device=device)
        exit_put_all = torch.full((n_total,), float('nan'), device=device)

    # OTM P&L targets (may be absent in old data.pt files)
    def _get_or_nan(key):
        if key in data:
            return data[key].to(device)
        return torch.full((n_total,), float('nan'), device=device)

    otm5_call_pnl_all = _get_or_nan('otm5_call_pnl')
    otm5_put_pnl_all = _get_or_nan('otm5_put_pnl')
    otm10_call_pnl_all = _get_or_nan('otm10_call_pnl')
    otm10_put_pnl_all = _get_or_nan('otm10_put_pnl')

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
                y = (targets[idx], call_pnl_all[idx], put_pnl_all[idx],
                     exit_call_all[idx], exit_put_all[idx],
                     otm5_call_pnl_all[idx], otm5_put_pnl_all[idx],
                     otm10_call_pnl_all[idx], otm10_put_pnl_all[idx])
                yield x, y
    else:
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            idx = valid_indices[i:end_i]
            window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
            x = features[window_idx]
            y = (targets[idx], call_pnl_all[idx], put_pnl_all[idx],
                 exit_call_all[idx], exit_put_all[idx],
                 otm5_call_pnl_all[idx], otm5_put_pnl_all[idx],
                 otm10_call_pnl_all[idx], otm10_put_pnl_all[idx])
            yield x, y


# ---------------------------------------------------------------------------
# Evaluation: Trade Simulation (the new primary metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_trades(model, data, lookback, device, batch_size=256,
                    stop_loss_pct=None, max_hold_bars=None,
                    max_trade_return=None,
                    starting_capital=None, risk_per_trade=None):
    """Simulate 0DTE option trades on validation set.

    Supports two model output formats:
      - Two-head: model(x) returns (gate_logits, dir_logits)
        gate_logits: (batch, 2) [NO_TRADE, TRADE]
        dir_logits:  (batch, 2) [CALL, PUT]
      - Legacy single-head: model(x) returns (batch, 3 or 4) logits

    Actions: DO_NOTHING=0, BUY_CALL=1, BUY_PUT=2, EXIT=3
    EXIT while in trade → close position (model-driven exit).
    EXIT while not in trade → treated as DO_NOTHING.

    Optional overrides (defaults from module constants):
      stop_loss_pct: Stop loss as fraction of premium (default 0.30)
      max_hold_bars: Max bars to hold a position (default 60)
      max_trade_return: Cap individual trade P&L (default 2.0 = 200%)
      starting_capital: Starting account balance for equity curve (default 5000.0)
      risk_per_trade: Fraction of capital risked per trade (default 0.10)

    Returns dict with trader + quant metrics and composite score.
    """
    # Allow train.py to override strategy parameters
    _stop_loss = stop_loss_pct if stop_loss_pct is not None else STOP_LOSS_PCT
    _max_hold = max_hold_bars if max_hold_bars is not None else MAX_HOLD_BARS
    _max_return = max_trade_return if max_trade_return is not None else MAX_TRADE_RETURN
    _starting_capital = starting_capital if starting_capital is not None else STARTING_CAPITAL
    _risk_per_trade = risk_per_trade if risk_per_trade is not None else RISK_PER_TRADE
    model.eval()

    features = data['features'].to(device)
    targets = data['targets']
    valid_mask = data['valid_mask']
    dates = data['dates']
    timestamps = data.get('timestamps', dates)
    atm_strikes = data.get('atm_strikes')

    val_start = max(lookback, data['val_start_idx'])
    val_end = data['val_end_idx'] + 1

    val_indices = [
        i for i in range(val_start, val_end)
        if valid_mask[i] and valid_mask[max(0, i - lookback):i].all()
    ]

    if len(val_indices) < 10:
        return _empty_metrics(len(val_indices))

    val_idx_t = torch.tensor(val_indices, dtype=torch.long, device=device)
    offsets = torch.arange(-lookback, 0, device=device)

    # Get model predictions for all val bars
    all_actions = []
    for i in range(0, len(val_idx_t), batch_size):
        idx = val_idx_t[i:i + batch_size]
        window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
        x = features[window_idx]
        out = model(x)

        if isinstance(out, tuple) and len(out) == 2:
            # Two-head model: (gate_logits, dir_logits)
            gate_logits, dir_logits = out
            gate_action = torch.argmax(gate_logits, dim=-1)   # 0=no_trade, 1=trade
            dir_action = torch.argmax(dir_logits, dim=-1)     # 0-5 for 6 dir classes
            n_dir = dir_logits.shape[-1]

            if n_dir == 6:
                # 6-class direction: [CALL_ATM, CALL_OTM5, CALL_OTM10,
                #                      PUT_ATM, PUT_OTM5, PUT_OTM10]
                # Map dir_action (0-5) → action constants (1-6)
                batch_actions = torch.where(
                    gate_action == 1,
                    dir_action + 1,  # ACTION_BUY_CALL_ATM=1 through ACTION_BUY_PUT_OTM10=6
                    torch.full_like(gate_action, ACTION_EXIT),  # gate=no_trade → EXIT candidate
                )
            else:
                # Legacy 2-class direction: [CALL, PUT]
                batch_actions = torch.where(
                    gate_action == 1,
                    torch.where(dir_action == 0,
                                torch.full_like(gate_action, ACTION_BUY_CALL_ATM),
                                torch.full_like(gate_action, ACTION_BUY_PUT_ATM)),
                    torch.full_like(gate_action, ACTION_EXIT),
                )
            all_actions.append(batch_actions.cpu())
        else:
            # Legacy single-head: (batch, NUM_ACTIONS) logits
            logits = out
            probs = torch.softmax(logits, dim=-1)
            actions = torch.argmax(probs, dim=-1)
            all_actions.append(actions.cpu())

    actions = torch.cat(all_actions).numpy()

    # Count unique val dates
    val_dates_list = [dates[i] for i in val_indices]
    num_val_days = len(set(val_dates_list))

    # Pre-compute bar_of_day for each validation index (0=9:30, 29=9:59, 30=10:00)
    _bar_of_day = {}
    _prev_date = None
    _bod = 0
    for gi in val_indices:
        d = dates[gi]
        if d != _prev_date:
            _bod = 0
            _prev_date = d
        else:
            _bod += 1
        _bar_of_day[gi] = _bod

    # -------------------------------------------------------------------
    # Simulate trades (actual option prices when available, delta fallback)
    # -------------------------------------------------------------------
    # Map action → price array for each strike/direction
    _CALL_ACTIONS = {ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5, ACTION_BUY_CALL_OTM10}
    _ENTRY_ACTIONS = {ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5, ACTION_BUY_CALL_OTM10,
                      ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5, ACTION_BUY_PUT_OTM10}

    _ACTION_NAMES = {
        ACTION_BUY_CALL_ATM: 'CALL_ATM', ACTION_BUY_CALL_OTM5: 'CALL_OTM5',
        ACTION_BUY_CALL_OTM10: 'CALL_OTM10', ACTION_BUY_PUT_ATM: 'PUT_ATM',
        ACTION_BUY_PUT_OTM5: 'PUT_OTM5', ACTION_BUY_PUT_OTM10: 'PUT_OTM10',
    }

    def _get_px_array(action, data_dict):
        """Get the price array for a given action."""
        mapping = {
            ACTION_BUY_CALL_ATM:   data_dict.get('atm_call_prices'),
            ACTION_BUY_CALL_OTM5:  data_dict.get('otm5_call_prices'),
            ACTION_BUY_CALL_OTM10: data_dict.get('otm10_call_prices'),
            ACTION_BUY_PUT_ATM:    data_dict.get('atm_put_prices'),
            ACTION_BUY_PUT_OTM5:   data_dict.get('otm5_put_prices'),
            ACTION_BUY_PUT_OTM10:  data_dict.get('otm10_put_prices'),
        }
        return mapping.get(action)

    atm_call_px = data.get('atm_call_prices')
    atm_put_px = data.get('atm_put_prices')
    has_option_prices = atm_call_px is not None and atm_put_px is not None

    trade_pnls = []
    trade_details = []
    model_exit_count = 0
    in_trade = False
    trade_entry_bar = 0
    trade_action = 0
    trade_entry_price = 0.0
    trade_use_actual = False
    trade_px_array = None
    last_stop_bar = -STOP_COOLDOWN_BARS  # initialize so first entry isn't blocked
    cooldown_blocked_count = 0
    pre_10am_blocked_count = 0

    for k, global_idx in enumerate(val_indices):
        if in_trade:
            bars_held = k - trade_entry_bar
            entry_global = val_indices[trade_entry_bar]

            # --- P&L computation ---
            if trade_use_actual and trade_px_array is not None:
                current_px = float(trade_px_array[global_idx]) if not torch.isnan(trade_px_array[global_idx]) else np.nan

                if not np.isnan(current_px) and trade_entry_price > 0:
                    net_pnl_pct = (current_px - trade_entry_price) / trade_entry_price
                else:
                    direction = 1.0 if trade_action in _CALL_ACTIONS else -1.0
                    cum_ret = 0.0
                    for j in range(trade_entry_bar + 1, k + 1):
                        if j < len(val_indices):
                            gidx = val_indices[j]
                            bar_ret = features[gidx, 0].item() if features.dim() == 2 else 0
                            cum_ret += bar_ret
                    theta_per_bar = THETA_DECAY_DAILY / BARS_PER_DAY
                    net_pnl_pct = direction * cum_ret / ATM_DELTA - theta_per_bar * bars_held
            else:
                direction = 1.0 if trade_action in _CALL_ACTIONS else -1.0
                cum_ret = 0.0
                for j in range(trade_entry_bar + 1, k + 1):
                    if j < len(val_indices):
                        gidx = val_indices[j]
                        bar_ret = features[gidx, 0].item() if features.dim() == 2 else 0
                        cum_ret += bar_ret
                theta_per_bar = THETA_DECAY_DAILY / BARS_PER_DAY
                net_pnl_pct = direction * cum_ret / ATM_DELTA - theta_per_bar * bars_held

            hit_stop = net_pnl_pct <= -_stop_loss
            hit_max_hold = bars_held >= _max_hold
            eod = dates[global_idx] != dates[entry_global]
            model_exit = (actions[k] == ACTION_EXIT)

            if hit_stop or hit_max_hold or eod or model_exit or k == len(val_indices) - 1:
                if hit_stop:
                    final_pnl = -_stop_loss
                    last_stop_bar = k
                else:
                    final_pnl = net_pnl_pct

                spread_cost = OPTION_SPREAD_BPS / 10000.0 * 2
                final_pnl -= spread_cost

                # Cap individual trade P&L to eliminate fat-tail lottery dependency
                final_pnl = max(-_stop_loss, min(final_pnl, _max_return))

                if model_exit:
                    model_exit_count += 1

                # Exit reason
                if hit_stop:
                    exit_reason = 'stop_loss'
                elif model_exit:
                    exit_reason = 'model_exit'
                elif eod:
                    exit_reason = 'end_of_day'
                elif hit_max_hold:
                    exit_reason = 'max_hold'
                else:
                    exit_reason = 'end_of_data'

                # Strike info
                strike_val = float(atm_strikes[entry_global]) if atm_strikes is not None and not torch.isnan(atm_strikes[entry_global]) else None

                trade_pnls.append(final_pnl)
                trade_details.append({
                    'trade_num': len(trade_details) + 1,
                    'date': dates[entry_global],
                    'entry_time': timestamps[entry_global],
                    'exit_time': timestamps[global_idx],
                    'direction': _ACTION_NAMES.get(trade_action, 'UNKNOWN'),
                    'strike': strike_val,
                    'entry_price': trade_entry_price if trade_use_actual else None,
                    'bars_held': bars_held,
                    'hold_minutes': bars_held * BAR_SIZE_MINUTES,
                    'pnl_pct': round(final_pnl * 100, 4),
                    'exit_reason': exit_reason,
                    'actual_prices': trade_use_actual,
                    'result': 'WIN' if final_pnl > 0 else 'LOSS',
                })
                in_trade = False

        # Entry: any BUY action can open a position
        if not in_trade and actions[k] in _ENTRY_ACTIONS:
            if (k - last_stop_bar) < STOP_COOLDOWN_BARS:
                cooldown_blocked_count += 1
                continue  # cooldown after stop loss
            if _bar_of_day.get(global_idx, 999) < NO_TRADE_BEFORE_BAR:
                pre_10am_blocked_count += 1
                continue  # no entries before 10:00 AM
            in_trade = True
            trade_entry_bar = k
            trade_action = actions[k]
            trade_use_actual = False
            trade_entry_price = 0.0
            trade_px_array = _get_px_array(trade_action, data)

            if trade_px_array is not None and not torch.isnan(trade_px_array[global_idx]):
                entry_px = float(trade_px_array[global_idx])
                if entry_px > 0:
                    trade_entry_price = entry_px
                    trade_use_actual = True

    # -------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------
    pnls = np.array(trade_pnls) if trade_pnls else np.array([0.0])
    num_trades = len(trade_pnls)

    if num_trades < MIN_TRADES:
        return _empty_metrics(len(val_indices), num_val_days, num_trades)

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / max(num_trades, 1)
    avg_winner = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loser = float(np.mean(losses)) if len(losses) > 0 else 0.0
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / max(gross_loss, 1e-10)
    trades_per_day = num_trades / max(num_val_days, 1)

    max_consec_loss = 0
    cur_consec = 0
    for p in pnls:
        if p <= 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls, ddof=1)) if num_trades > 1 else 1e-10
    trades_per_year = trades_per_day * 252
    trade_sharpe = (mean_pnl / max(std_pnl, 1e-10)) * math.sqrt(max(trades_per_year, 1))

    downside_returns = pnls[pnls < 0]
    downside_std = float(np.std(downside_returns, ddof=1)) if len(downside_returns) > 1 else 1e-10
    sortino = (mean_pnl / max(downside_std, 1e-10)) * math.sqrt(max(trades_per_year, 1))

    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    max_drawdown = float(np.min(dd)) if len(dd) > 0 else 0.0

    total_return = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0
    calmar = total_return / max(abs(max_drawdown), 1e-10)

    # Composite score with bell-curve trade frequency
    # Sweet spot: 2-6 trades/day. Penalize both under- AND over-trading.
    # Under-trading: ramps up from 0 at tpd=0.5 to 1.0 at tpd=2
    # Over-trading: QUADRATIC decay above 6 tpd (tpd=10→0.36x, tpd=12→0.25x, tpd=15→0.16x)
    MAX_TRADES_PER_DAY = 6.0  # above this, score decays quadratically
    if trades_per_day < 0.5:
        score = -10.0
    elif trades_per_day < 1.5:
        trade_freq_mult = min(1.0, trades_per_day / 2.0)
        if trades_per_day <= MAX_TRADES_PER_DAY:
            freq_mult = trade_freq_mult
        else:
            freq_mult = max(0.1, (MAX_TRADES_PER_DAY / trades_per_day) ** 2)
        raw_score = profit_factor * max(trade_sharpe, 0.0) * freq_mult
        ramp = (trades_per_day - 0.5) / 1.0
        score = -5.0 * (1.0 - ramp) + raw_score * ramp
    else:
        if trades_per_day <= MAX_TRADES_PER_DAY:
            freq_mult = min(1.0, trades_per_day / 2.0)
        else:
            freq_mult = max(0.1, (MAX_TRADES_PER_DAY / trades_per_day) ** 2)
        score = profit_factor * max(trade_sharpe, 0.0) * freq_mult

    # Consecutive loss penalty: incremental decay above 3 consecutive losses
    if max_consec_loss > 3 and score > 0:
        consec_penalty = max(0.5, 1.0 - 0.05 * (max_consec_loss - 3))
        score *= consec_penalty

    # Short-hold penalty: penalize excessive 1-bar noise scalping
    short_hold_pct = 0.0
    if num_trades > 10:
        short_holds = sum(1 for d in trade_details if d['bars_held'] <= 1)
        short_hold_pct = short_holds / num_trades
        if short_hold_pct > 0.30 and score > 0:
            noise_penalty = max(0.7, 1.0 - (short_hold_pct - 0.30))
            score *= noise_penalty

    # Stop-loss rate penalty: penalize poor entry selection
    # 30% stop rate = no penalty, 50% = 0.80x, 70% = 0.60x, 80% = 0.50x floor
    stop_loss_rate = 0.0
    if num_trades > 10:
        stop_loss_rate = sum(1 for d in trade_details if d.get('exit_reason') == 'stop_loss') / num_trades
        if stop_loss_rate > 0.30 and score > 0:
            sl_penalty = max(0.5, 1.0 - (stop_loss_rate - 0.30))
            score *= sl_penalty

    total_bars = len(actions)
    do_nothing_pct = float(np.sum(actions == ACTION_DO_NOTHING)) / max(total_bars, 1)
    exit_pct = float(np.sum(actions == ACTION_EXIT)) / max(total_bars, 1)

    # --- Equity curve (dollar-denominated, informational) ---
    capital = _starting_capital
    equity_curve = [capital]
    for pnl_pct in trade_pnls:
        position_size = capital * _risk_per_trade
        dollar_pnl = position_size * pnl_pct
        capital += dollar_pnl
        equity_curve.append(capital)

    equity_arr = np.array(equity_curve)
    final_capital = float(equity_arr[-1])
    total_dollar_return = (final_capital - _starting_capital) / _starting_capital

    equity_peak = np.maximum.accumulate(equity_arr)
    equity_dd = (equity_arr - equity_peak) / np.maximum(equity_peak, 1e-10)
    max_equity_dd = float(np.min(equity_dd)) if len(equity_dd) > 0 else 0.0

    if num_trades > 1:
        equity_returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], 1e-10)
        eq_mean = float(np.mean(equity_returns))
        eq_std = float(np.std(equity_returns, ddof=1))
        equity_sharpe = (eq_mean / max(eq_std, 1e-10)) * math.sqrt(max(trades_per_year, 1))
    else:
        equity_sharpe = 0.0

    return {
        'score': float(score),
        'profit_factor': float(profit_factor),
        'win_rate': float(win_rate),
        'avg_winner': float(avg_winner),
        'avg_loser': float(avg_loser),
        'trades_per_day': float(trades_per_day),
        'max_consec_loss': int(max_consec_loss),
        'num_trades': int(num_trades),
        'trade_sharpe': float(trade_sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_drawdown),
        'calmar': float(calmar),
        'ev_per_trade': float(mean_pnl),
        'do_nothing_pct': float(do_nothing_pct),
        'exit_pct': float(exit_pct),
        'model_exit_count': int(model_exit_count),
        'num_val_bars': len(val_indices),
        'num_val_days': int(num_val_days),
        'total_return': float(total_return),
        'cooldown_blocked': int(cooldown_blocked_count),
        'pre_10am_blocked': int(pre_10am_blocked_count),
        'short_hold_pct': round(float(short_hold_pct), 3),
        'stop_loss_rate': round(float(stop_loss_rate), 3),
        'final_capital': round(final_capital, 2),
        'equity_sharpe': round(float(equity_sharpe), 4),
        'max_equity_dd': round(float(max_equity_dd), 4),
        'total_dollar_return': round(float(total_dollar_return), 4),
        'trade_log': trade_details,
    }


def _empty_metrics(num_val_bars=0, num_val_days=0, num_trades=0):
    return {
        'score': -10.0,
        'profit_factor': 0.0, 'win_rate': 0.0,
        'avg_winner': 0.0, 'avg_loser': 0.0,
        'trades_per_day': 0.0, 'max_consec_loss': 0,
        'num_trades': num_trades,
        'trade_sharpe': 0.0, 'sortino': 0.0,
        'max_drawdown': 0.0, 'calmar': 0.0,
        'ev_per_trade': 0.0, 'do_nothing_pct': 1.0,
        'exit_pct': 0.0, 'model_exit_count': 0,
        'num_val_bars': num_val_bars, 'num_val_days': num_val_days,
        'total_return': 0.0,
        'final_capital': STARTING_CAPITAL,
        'equity_sharpe': 0.0,
        'max_equity_dd': 0.0,
        'total_dollar_return': 0.0,
    }


# ---------------------------------------------------------------------------
# Legacy: evaluate_sharpe (kept for quant comparison)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sharpe(model, data, lookback, device, batch_size=256,
                    confidence_threshold=0.0):
    """Walk-forward Sharpe on validation set (legacy continuous metric)."""
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

    if len(val_indices) < 20:
        return {'val_sharpe': -999.0, 'max_drawdown': 0.0, 'annual_return': 0.0,
                'num_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'num_val_bars': len(val_indices), 'num_val_days': 0}

    val_idx_t = torch.tensor(val_indices, dtype=torch.long, device=device)
    offsets = torch.arange(-lookback, 0, device=device)

    all_positions, all_returns = [], []
    for i in range(0, len(val_idx_t), batch_size):
        idx = val_idx_t[i:i + batch_size]
        window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
        x = features[window_idx]
        out = model(x)
        # Handle multiple model output formats
        if isinstance(out, tuple) and len(out) == 2:
            # Two-head model: (gate_logits, dir_logits)
            gate_logits, dir_logits = out
            gate_probs = torch.softmax(gate_logits, dim=-1)   # [no_trade, trade]
            dir_probs = torch.softmax(dir_logits, dim=-1)     # [call_atm..put_otm10]
            trade_prob = gate_probs[:, 1]
            # Sum call probs (first 3) vs put probs (last 3) for position sizing
            n_dir = dir_probs.shape[-1]
            if n_dir == 6:
                call_prob = dir_probs[:, :3].sum(dim=-1)
                put_prob = dir_probs[:, 3:].sum(dim=-1)
            else:
                call_prob = dir_probs[:, 0]
                put_prob = dir_probs[:, 1]
            pos = trade_prob * (call_prob - put_prob)
        elif out.dim() == 1 or (out.dim() == 2 and out.shape[-1] == 1):
            pos = out.squeeze(-1).clamp(-1.0, 1.0)
        else:
            # Single-head N-class model: map to position
            probs = torch.softmax(out, dim=-1)
            pos = probs[:, ACTION_BUY_CALL] - probs[:, ACTION_BUY_PUT]
        if confidence_threshold > 0:
            pos = torch.where(pos.abs() < confidence_threshold,
                              torch.zeros_like(pos), pos)
        all_positions.append(pos.cpu())
        all_returns.append(targets[idx.cpu()])

    positions = torch.cat(all_positions).numpy()
    returns = torch.cat(all_returns).numpy()

    cost = 12 / 10_000  # 12 bps
    turnover = np.abs(np.diff(positions, prepend=0.0))
    pnl = positions * returns - turnover * cost

    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 1e-10
    sharpe = (mean_pnl / max(std_pnl, 1e-10)) * math.sqrt(ANNUAL_TRADING_BARS)

    cum = np.cumsum(pnl)
    dd = cum - np.maximum.accumulate(cum)
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    trades = int(np.sum(np.abs(np.diff(np.sign(positions))) > 0))
    wins = int(np.sum(pnl > 0))
    losses_count = int(np.sum(pnl < 0))
    win_rate = wins / max(wins + losses_count, 1)
    gross_profit = float(np.sum(pnl[pnl > 0]))
    gross_loss = float(abs(np.sum(pnl[pnl < 0])))
    profit_factor = gross_profit / max(gross_loss, 1e-10)

    return {
        'val_sharpe': float(sharpe),
        'max_drawdown': max_dd,
        'annual_return': mean_pnl * ANNUAL_TRADING_BARS,
        'num_trades': trades,
        'win_rate': float(win_rate),
        'profit_factor': profit_factor,
        'num_val_bars': len(val_indices),
        'num_val_days': len(val_indices) // max(BARS_PER_DAY, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SPX 0DTE trading data")
    parser.add_argument("--start", type=str, default="2022-03-14")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--polygon-key", type=str, default=None)
    parser.add_argument("--s3-key-id", type=str,
                        default=os.environ.get("POLYGON_S3_KEY_ID"),
                        help="Polygon flat files S3 key ID (or env POLYGON_S3_KEY_ID)")
    parser.add_argument("--s3-secret", type=str,
                        default=os.environ.get("POLYGON_S3_SECRET"),
                        help="Polygon flat files S3 secret (or env POLYGON_S3_SECRET)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="~40 trading days for fast dry run")
    parser.add_argument("--skip-options", action="store_true",
                        help="Skip SPXW options download (use cached or no options)")
    parser.add_argument("--skip-chain", action="store_true",
                        help="Skip SPXW OTM chain download (use cached or no OTM data)")
    parser.add_argument("--skip-vix", action="store_true",
                        help="Skip VIX download via IBKR (use cached or ATM IV fallback)")
    parser.add_argument("--use-spx", action="store_true",
                        help="Use real SPX index prices from IBKR (volume still from SPY ETF)")
    parser.add_argument("--spy-source", type=str, default="ibkr",
                        choices=["ibkr", "polygon"],
                        help="SPY data source: ibkr (full history) or polygon (2yr limit)")
    parser.add_argument("--ib-port", type=int, default=None,
                        help="IB Gateway port (default: env IB_PORT or 4001; paper=4002)")
    args = parser.parse_args()

    if args.ib_port is not None:
        os.environ["IB_PORT"] = str(args.ib_port)

    if args.polygon_key:
        os.environ["POLYGON_API_KEY"] = args.polygon_key

    if args.s3_key_id:
        os.environ["POLYGON_S3_KEY_ID"] = args.s3_key_id
    if args.s3_secret:
        os.environ["POLYGON_S3_SECRET"] = args.s3_secret

    if args.end is None:
        args.end = (dt.date.today() - dt.timedelta(days=1)).strftime('%Y-%m-%d')

    if args.quick:
        args.start = (dt.date.today() - dt.timedelta(days=70)).strftime('%Y-%m-%d')
        print(f"Quick mode: {args.start} -> {args.end} (~40 trading days)")

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Features: {NUM_FEATURES}, Bar size: {BAR_SIZE_MINUTES}min")
    print(f"Cache: {CACHE_DIR}")
    print()

    # --- Download SPY (volume source) ---
    cache_path = os.path.join(DATA_DIR, "spy_1min.pkl")
    if args.skip_download or os.path.exists(cache_path):
        if not os.path.exists(cache_path):
            old_cache = os.path.join(DATA_DIR, "spy_5min.pkl")
            if os.path.exists(old_cache):
                cache_path = old_cache
                print("  (using old 5-min SPY cache — rebuild for 1-min)")
            else:
                print(f"ERROR: Cache not found: {cache_path}")
                sys.exit(1)
        print(f"Loading cached SPY data...")
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
        print(f"  {len(df)} bars, {df['date'].nunique()} days")
    else:
        if args.spy_source == "ibkr":
            df = download_spy_bars_ibkr(args.start, args.end)
        else:
            df = download_spy_bars(args.start, args.end)
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"  Cached to {cache_path}")
    print()

    # --- Filter to 0DTE days only ---
    # Before May 11, 2022: Mon/Wed/Fri only. After: daily.
    all_days_before = df['date'].nunique()
    df = df[df['date'].apply(is_0dte_day)].copy()
    df = df.reset_index(drop=True)
    skipped = all_days_before - df['date'].nunique()
    print(f"0DTE filter: {df['date'].nunique()} valid days ({skipped} non-0DTE days removed)")
    print()

    # --- Options (ATM + OTM) from Polygon flat files (S3) ---
    options_data = None
    chain_data = None
    if not args.skip_options and not args.skip_download:
        print(f"Downloading SPXW options from flat files (all {df['date'].nunique()} days)...")
        prefetch_spxw_from_flatfiles(df, api_cutoff=None)  # no cutoff — flat files for ALL days
        print()

    # Load ATM option caches
    if not args.skip_options:
        opt_cache = os.path.join(DATA_DIR, "spxw_full.pkl")
        if args.skip_download and os.path.exists(opt_cache):
            print("Loading cached options data...")
            with open(opt_cache, 'rb') as f:
                options_data = pickle.load(f)
        else:
            options_data = load_spxw_caches(df)
            with open(opt_cache, 'wb') as f:
                pickle.dump(options_data, f)
            print(f"  Cached {len(options_data)} ATM bar-pairs to {opt_cache}")
        print()

    # Load OTM chain caches
    if not args.skip_chain:
        chain_cache = os.path.join(DATA_DIR, "spxw_chain_full.pkl")
        if args.skip_download and os.path.exists(chain_cache):
            print("Loading cached OTM chain data...")
            with open(chain_cache, 'rb') as f:
                chain_data = pickle.load(f)
        else:
            chain_data = load_spxw_chain_caches(df)
            with open(chain_cache, 'wb') as f:
                pickle.dump(chain_data, f)
            print(f"  Cached {len(chain_data)} OTM bar entries to {chain_cache}")
        if chain_data:
            print(f"  OTM chain: {len(chain_data)} bar entries")
        print()

    # --- VIX (via IBKR) ---
    vix_data = None
    if not args.skip_vix:
        vix_cache = os.path.join(DATA_DIR, "vix_1min.pkl")
        if os.path.exists(vix_cache):
            print("Loading cached VIX data...")
            with open(vix_cache, 'rb') as f:
                vix_data = pickle.load(f)
        elif not args.skip_download:
            vix_df = download_vix_bars(args.start, args.end)
            if vix_df is not None and len(vix_df) > 0:
                # Build lookup: timestamp (ms) → vix_close
                vix_data = {}
                for _, row in vix_df.iterrows():
                    vix_data[row['timestamp']] = {
                        'vix_open': row['vix_open'],
                        'vix_high': row['vix_high'],
                        'vix_low': row['vix_low'],
                        'vix_close': row['vix_close'],
                    }
                with open(vix_cache, 'wb') as f:
                    pickle.dump(vix_data, f)
                print(f"  Cached {len(vix_data)} VIX bars to {vix_cache}")
            else:
                print("  VIX download returned no data — will fall back to ATM IV proxy")
        if vix_data:
            print(f"  VIX data: {len(vix_data)} bars")
        print()

    # --- SPX price replacement (optional) ---
    if args.use_spx:
        spx_cache = os.path.join(DATA_DIR, "spx_1min.pkl")
        if os.path.exists(spx_cache):
            print("Loading cached SPX data...")
            with open(spx_cache, 'rb') as f:
                spx_df = pickle.load(f)
        elif not args.skip_download:
            spx_df = download_spx_bars(args.start, args.end)
            if spx_df is not None:
                with open(spx_cache, 'wb') as f:
                    pickle.dump(spx_df, f)
                print(f"  Cached SPX data to {spx_cache}")

        if spx_df is not None and len(spx_df) > 0:
            pre_count = len(df)
            df = df.merge(
                spx_df[['timestamp', 'spx_open', 'spx_high', 'spx_low', 'spx_close']],
                on='timestamp', how='inner'
            )
            df['open'] = df['spx_open']
            df['high'] = df['spx_high']
            df['low'] = df['spx_low']
            df['close'] = df['spx_close']
            df = df.drop(columns=['spx_open', 'spx_high', 'spx_low', 'spx_close'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            print(f"  Using real SPX prices ({pre_count} -> {len(df)} bars, SPY volume kept)")
        else:
            print("  WARNING: No SPX data — falling back to SPY prices")
        print()

    # --- Alignment validation ---
    import random as _rng
    print("=== DATA ALIGNMENT REPORT ===")
    print(f"  SPY bars: {len(df)} across {df['date'].nunique()} days")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    spy_dates = set(df['date'].unique())

    if options_data:
        opt_dates = set(k[0] for k in options_data.keys())
        opt_only = opt_dates - spy_dates
        spy_only = spy_dates - opt_dates
        matched_dates = opt_dates & spy_dates
        spy_timestamps = set(zip(df['date'], df['timestamp'].astype(int)))
        opt_timestamps = set(options_data.keys())
        ts_matched = spy_timestamps & opt_timestamps
        print(f"  Options: {len(opt_dates)} days with data, {len(matched_dates)} match SPY dates")
        print(f"    Timestamp-level matches: {len(ts_matched)}/{len(spy_timestamps)} SPY bars ({100*len(ts_matched)/len(spy_timestamps):.1f}%)")
        if opt_only:
            print(f"    WARNING: {len(opt_only)} option days not in SPY data")
        if spy_only:
            print(f"    {len(spy_only)} SPY days without options (expected for data gaps)")

    if vix_data:
        spy_ts_set = set(df['timestamp'].astype(int))
        vix_ts_set = set(vix_data.keys())
        vix_matched = spy_ts_set & vix_ts_set
        print(f"  VIX: {len(vix_data)} bars, {len(vix_matched)}/{len(spy_ts_set)} match SPY timestamps ({100*len(vix_matched)/len(spy_ts_set):.1f}%)")

    if chain_data:
        chain_dates = set(k[0] for k in chain_data.keys())
        chain_matched = chain_dates & spy_dates
        print(f"  OTM chain: {len(chain_dates)} days with data, {len(chain_matched)} match SPY dates")

    sample_date = _rng.choice(sorted(df['date'].unique()))
    sample_bars = df[df['date'] == sample_date]
    sample_ts = int(sample_bars.iloc[0]['timestamp'])
    print(f"\n  Spot check ({sample_date}, first bar ts={sample_ts}):")
    print(f"    SPY bar: open={sample_bars.iloc[0]['open']:.2f} vol={int(sample_bars.iloc[0]['volume'])}")
    if options_data:
        opt = options_data.get((sample_date, sample_ts))
        print(f"    Options: {'FOUND' if opt else 'MISSING'}" + (f" strike={opt['strike']}" if opt else ""))
    if vix_data:
        vix = vix_data.get(sample_ts)
        print(f"    VIX: {'FOUND' if vix else 'MISSING'}" + (f" close={vix['vix_close']:.2f}" if vix else ""))
    if chain_data:
        ch = chain_data.get((sample_date, sample_ts))
        print(f"    OTM chain: {'FOUND' if ch else 'MISSING'}")

    print("=== END ALIGNMENT REPORT ===\n")

    # --- Features ---
    print(f"Computing {NUM_FEATURES} features from {len(df)} bars...")
    t0 = time.time()
    features, targets, dates, valid, option_prices, timestamps = compute_features(df, options_data, vix_data, chain_data)
    valid_count = int(np.sum(valid))
    opt_count = int(np.sum(~np.isnan(option_prices['atm_call']))) if option_prices else 0
    pnl_count = int(np.sum(~np.isnan(option_prices['call_pnl']))) if option_prices else 0
    exit_count = int(np.sum(option_prices['exit_call_label'] == 1.0)) if option_prices else 0
    otm_count = int(np.sum(~np.isnan(option_prices['otm5_call']))) if option_prices else 0
    print(f"  Valid bars: {valid_count}/{len(df)} ({100*valid_count/len(df):.0f}%)")
    print(f"  Bars with option prices: {opt_count}/{len(df)} ({100*opt_count/len(df):.0f}%)")
    print(f"  Bars with OTM prices: {otm_count}/{len(df)} ({100*otm_count/len(df):.0f}%)")
    print(f"  Bars with option P&L: {pnl_count}/{len(df)} ({100*pnl_count/len(df):.0f}%)")
    print(f"  Bars with EXIT=1 (call): {exit_count}")
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # --- Normalize ---
    print("Normalizing (adaptive rolling z-score)...")
    t0 = time.time()
    features = normalize_features(features, valid)
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # --- Tensors ---
    print("Preparing tensors...")
    data = prepare_tensors(features, targets, dates, valid, option_prices, timestamps)
    print()

    print(f"Done! Features: {NUM_FEATURES}, Actions: {NUM_ACTIONS}")
    print("Run: python3 train.py")
