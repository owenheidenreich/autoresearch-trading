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

Computes 37 trader-relevant features (v3: returns, volume, VWAP, session,
options, VIX/regime, Greeks, Bollinger, market structure) — reduced from 70 for
signal density and training speed — and prepares tensors for train.py.

The model uses a two-head action contract:
  - Gate head: NO_TRADE / TRADE
  - Direction head: CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10
This yields 8 effective actions (DO_NOTHING, 6 entries, EXIT). Evaluation
simulates actual 0DTE option trades with stops, targets, and model-driven exits.

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
OPTION_SPREAD_BPS    = 150     # bid-ask spread on 0DTE ATM in bps of premium (150 bps one-way = 3% round-trip; realistic for ATM SPX 0DTE)
STOP_LOSS_PCT        = 0.30    # legacy constant — kept for backward compat; active code uses compute_dynamic_stop()
DYNAMIC_STOP_BASE    = float(os.environ.get("DYNAMIC_STOP_BASE", 0.35))
DYNAMIC_STOP_MIN     = 0.15    # minimum stop-loss (floor)
DYNAMIC_STOP_MAX     = 0.60    # maximum stop-loss (ceiling)
MAX_HOLD_BARS        = BARS_PER_DAY  # hold until stop/profit/EOD (0DTE closes at EOD)
STOP_COOLDOWN_BARS   = 5       # 5-bar (5-min) cooldown after stop loss before re-entry
NO_TRADE_BEFORE_BAR  = 30      # first 30 bars (9:30-9:59) are hard no-trade
MAX_TRADE_RETURN     = 5.0     # cap individual trade P&L at 500% (allow large winners with learned exits)
STARTING_CAPITAL     = 10_000.0  # starting account balance ($10k paper trading account)
POSITION_RISK_TARGET = 0.05     # target 5% of account per trade; scales whole contracts with account growth
SPX_MULTIPLIER       = 100      # SPX option contract multiplier (premium * 100 = cost)
BAR_SIZE_MINUTES     = 1           # 1-minute bar resolution
SPX_MULTIPLIER       = 100     # option multiplier

# Keep core model actions on ±5/±10, but expand historical sidecar ladders to ±15/±20.
OTM_STRIKE_STEPS      = (5, 10, 15, 20)
SIDE_ACTION_ORDER     = (
    "call_atm",
    "call_otm5",
    "call_otm10",
    "put_atm",
    "put_otm5",
    "put_otm10",
)
SIDE_ACTION_TO_CHAIN_KEY = {
    "call_otm5": "otm5_call",
    "call_otm10": "otm10_call",
    "put_otm5": "otm5_put",
    "put_otm10": "otm10_put",
}
SIDE_ACTION_TO_IDX = {name: i for i, name in enumerate(SIDE_ACTION_ORDER)}
SIDE_ALL_LEGS = SIDE_ACTION_ORDER + (
    "call_otm15",
    "call_otm20",
    "put_otm15",
    "put_otm20",
)

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
# Feature names (37 features: 32 core + 5 market structure from first-principles review)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # === Price returns (2) ===
    'ret_6',                # 30-bar (30min) return
    'ret_12',               # 60-bar (1hr) return
    # === Volume (3) ===
    'volume_ratio',         # bar volume / 20-bar SMA
    'volume_zscore',        # (volume - mean) / std
    'volume_at_price_pctile',  # current close vs session volume profile
    # === Volatility (3) ===
    'bar_range',            # (high - low) / close
    'realized_vol',         # 20-bar rolling stdev of returns
    'range_ratio',          # current bar range / 20-bar avg range
    # === VWAP (2) ===
    'vwap_dist',            # (close - session VWAP) / close
    'vwap_slope',           # change in VWAP distance over 6 bars
    # === Session structure (2) ===
    'ib_width',             # IB range / close (normalized)
    'session_range_pct',    # full session range so far / close
    # === Key levels (2) ===
    'prev_high_dist',       # distance to previous day high
    'prev_low_dist',        # distance to previous day low
    # === Trend (3) ===
    'ema_cross',            # (EMA8 - EMA21) / close (momentum)
    'consec_direction',     # consecutive same-direction bars: +N for up, -N for down
    'speed_estimate',       # |5-bar return| / realized_vol: normalized speed of move
    # === Microstructure (2) ===
    'gap',                  # overnight gap, carried all day
    'inside_bar',           # 1 if current bar inside previous bar
    # === Time (3) ===
    'minutes_to_close',     # log(minutes remaining + 1), normalized
    'time_sin',             # sin(2pi * session_progress)
    'time_cos',             # cos(2pi * session_progress)
    # === Options (2) ===
    'atm_iv',               # ATM implied vol (avg of call + put IV)
    'iv_skew',              # put IV - call IV (fear/skew premium)
    # === VIX / Regime (2) ===
    'vix_regime',           # regime bucket: -1=low(<15), -0.33=normal, 0.33=elevated, 1=crisis(>30)
    'vrp',                  # variance risk premium: atm_iv^2 - realized_vol^2
    # === Greeks (3) ===
    'atm_gamma',            # ATM call gamma (delta sensitivity to price)
    'atm_theta_per_bar',    # ATM theta per 1-min bar (time decay per bar)
    'charm_estimate',       # estimated dDelta/dT: delta sensitivity to time decay
    # === Bollinger (1) ===
    'bollinger_position',   # (close - BB_mid) / (BB_upper - BB_lower): position within bands
    # === Range extras (2) ===
    'rsi_14',               # 14-period RSI (0-1 scale)
    'session_range_position',  # (close - session_low) / (session_high - session_low)
    # === Market structure (5) ===
    'poc_dist',             # (close - session POC) / close: distance to Point of Control
    'va_position',          # position within Value Area: 0=VAL, 1=VAH, <0/>1 = outside
    'vwap_band_sigma',      # distance from VWAP in σ units (±1σ, ±2σ bands)
    'ib_break',             # IB break state: -1=below IB low, 0=inside, +1=above IB high
    'theta_pressure',       # afternoon theta pressure: ramps 0→1 from bar 120 (11:30am) to close
]

NUM_FEATURES = len(FEATURE_NAMES)

# Named index lookup for cross-references within compute_features
_FEAT_IDX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

# Features that should NOT be z-score normalized
_NO_NORMALIZE = {
    'time_sin', 'time_cos',
    'minutes_to_close',
    'inside_bar',
    'vix_regime',           # categorical, already scaled
    'bollinger_position',   # already normalized to [-1, 1]-ish range
    'session_range_position',  # already 0-1
    'rsi_14',               # already 0-1
    'va_position',          # already ~0-1 (can exceed but bounded)
    'ib_break',             # categorical: -1, 0, +1
    'theta_pressure',       # already 0-1
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

# Option P&L target: round-trip spread cost as fraction of premium
SPREAD_COST_PCT   = 2 * OPTION_SPREAD_BPS / 10000.0  # 3% round-trip (at 150 bps one-way)

# EXIT label: profit target threshold (fraction of premium)
EXIT_PROFIT_TARGET = 0.20  # exit when unrealized P&L > 20% of premium


def compute_dynamic_pnl(entry_bar, prices, dates, cost_pct,
                        stop_loss=STOP_LOSS_PCT, profit_target=None,
                        max_hold=BARS_PER_DAY):
    """Simulate a trade from entry_bar using dynamic exits matching evaluation.

    Returns (exit_bar, pnl_after_cost, exit_reason).
    Exit reasons: 'stop_loss', 'profit_target', 'eod', 'max_hold'.
    profit_target=None means no hardcoded TP (model decides when to exit).
    """
    N = len(prices)
    entry_px = prices[entry_bar]
    if np.isnan(entry_px) or entry_px <= 0:
        return (entry_bar, np.nan, 'invalid')
    day = dates[entry_bar]
    last_valid_px = entry_px
    for k in range(1, max_hold + 1):
        check = entry_bar + k
        if check >= N or dates[check] != day:
            # End of day
            pnl = (last_valid_px - entry_px) / entry_px - cost_pct
            return (min(check - 1, N - 1), pnl, 'eod')
        px = prices[check]
        if np.isnan(px):
            continue
        last_valid_px = px
        unrealized = (px - entry_px) / entry_px
        if unrealized <= -stop_loss:
            return (check, -stop_loss - cost_pct, 'stop_loss')
        if profit_target is not None and unrealized >= profit_target:
            return (check, unrealized - cost_pct, 'profit_target')
    # Max hold
    pnl = (last_valid_px - entry_px) / entry_px - cost_pct
    return (entry_bar + max_hold, pnl, 'max_hold')


def compute_dynamic_stop(gate_confidence, iv_feature, vix_feature):
    """Dynamic stop-loss: wider for uncertain/volatile, tighter for confident/calm.

    Args:
        gate_confidence: softmax probability of TRADE action (0.5-1.0)
        iv_feature: atm_iv feature value (z-scored or raw)
        vix_feature: vix_regime feature value (-1 to 1)

    Returns:
        stop_pct: stop-loss as fraction of premium, clamped to [DYNAMIC_STOP_MIN, DYNAMIC_STOP_MAX]
    """
    # Higher confidence → tighter stop (if the model is sure, cut fast if wrong)
    confidence_factor = 1.0 - (gate_confidence - 0.5) * 0.8  # 1.0 at 0.5, 0.6 at 1.0
    # Higher IV → wider stop (options swing more)
    iv_factor = 1.0 + max(0.0, float(iv_feature)) * 0.15
    # Higher VIX → wider stop (crisis regime)
    vix_factor = 1.0 + max(0.0, float(vix_feature)) * 0.10
    stop = DYNAMIC_STOP_BASE * confidence_factor * iv_factor * vix_factor
    return max(DYNAMIC_STOP_MIN, min(DYNAMIC_STOP_MAX, stop))


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


def _otm_chain_specs(atm_strike: float) -> list[tuple[str, str, float]]:
    specs: list[tuple[str, str, float]] = []
    for step in OTM_STRIKE_STEPS:
        specs.append((f"otm{step}_call", "C", float(atm_strike + step)))
    for step in OTM_STRIKE_STEPS:
        specs.append((f"otm{step}_put", "P", float(atm_strike - step)))
    return specs


def _safe_float(v, default: float = float("nan")) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _spread_bps_proxy(mid_px: float, high_px: float, low_px: float) -> float:
    """Spread estimate from option premium level (not bar range).

    Bar high-low conflates bid-ask spread with directional price movement —
    a $5 ATM option routinely moves $0.50 in a minute (1000 bps range) even
    though the actual spread is ~$0.15. Use premium-tier lookup based on
    known SPX 0DTE market microstructure instead.
    """
    mid = _safe_float(mid_px)
    if not np.isfinite(mid) or mid <= 0:
        return float("nan")
    # SPX 0DTE typical bid-ask spreads by premium tier:
    #   Premium >= $5:  ~$0.15 spread (liquid ATM)
    #   Premium $2-5:   ~$0.10 spread
    #   Premium $0.50-2:~$0.10 spread (still liquid near-money)
    #   Premium < $0.50:~$0.05 spread (wide relative to premium)
    if mid >= 5.0:
        spread_dollar = 0.15
    elif mid >= 2.0:
        spread_dollar = 0.10
    elif mid >= 0.50:
        spread_dollar = 0.10
    else:
        spread_dollar = 0.05
    return float(np.clip((spread_dollar / mid) * 10000.0, 1.0, 1500.0))


def _quote_age_proxy_seconds(volume: float) -> float:
    vol = _safe_float(volume, default=float("nan"))
    if not np.isfinite(vol):
        return float("nan")
    if vol > 0:
        return 0.0
    return 60.0


def _quality_score_proxy(spread_bps: float, quote_age_s: float, size: float) -> float:
    s = _safe_float(spread_bps)
    a = _safe_float(quote_age_s)
    q = _safe_float(size, default=0.0)
    if not np.isfinite(s):
        return float("nan")
    spread_score = math.exp(-max(s, 0.0) / 200.0)
    age_score = math.exp(-max(a, 0.0) / 45.0) if np.isfinite(a) else 0.0
    size_score = min(math.log1p(max(q, 0.0)) / 4.0, 1.0)
    return float(np.clip(0.60 * spread_score + 0.25 * size_score + 0.15 * age_score, 0.0, 1.0))


def _slippage_bps_proxy(spread_bps: float, quote_age_s: float, size: float) -> float:
    s = _safe_float(spread_bps)
    a = _safe_float(quote_age_s, default=60.0)
    q = _safe_float(size, default=0.0)
    if not np.isfinite(s):
        return float("nan")
    vol_penalty = 30.0 if q <= 0 else min(30.0, 12.0 / math.sqrt(q + 1.0))
    age_penalty = min(30.0, max(a, 0.0) / 2.0)
    return float(np.clip(0.35 * s + vol_penalty + age_penalty, 2.0, 600.0))

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

_ib_next_client_id = 10

def _ib_client(host: str = "127.0.0.1", port: int = None, client_id: int = None):
    """Connect to IB Gateway with unique client ID. Returns IB client or exits."""
    global _ib_next_client_id
    if port is None:
        port = int(os.environ.get("IB_PORT", "4001"))
    if client_id is None:
        client_id = _ib_next_client_id
        _ib_next_client_id += 1
    try:
        from ib_insync import IB
    except ImportError:
        print("ERROR: pip install ib_insync")
        sys.exit(1)
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=30)
    except Exception as e:
        print(f"ERROR: Cannot connect to IB Gateway at {host}:{port} (clientId={client_id}): {e}")
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


def _incremental_update(cache_path: str, download_fn, start: str, end: str,
                        date_col: str = 'date') -> pd.DataFrame:
    """Load existing cache, download only new data, merge and save.

    This is the core incremental rebuild mechanism. Instead of nuking the
    entire cache and re-downloading 3+ years of history, we:
    1. Load the existing pkl cache (if any)
    2. Find the last date already cached
    3. Download only from (last_date - 1 day overlap) to end
    4. Concat, deduplicate, filter to [start, end], save

    The 1-day overlap ensures we don't miss partial days at the boundary.
    Deduplication by timestamp handles the overlap cleanly.

    Args:
        cache_path: Path to the monolithic pkl file
        download_fn: Callable(start, end) -> DataFrame
        start: Requested start date (YYYY-MM-DD)
        end: Requested end date (YYYY-MM-DD)
        date_col: Column name containing date strings
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached_df = pickle.load(f)

        if date_col not in cached_df.columns:
            print(f"  Cache {os.path.basename(cache_path)}: missing '{date_col}' column, full re-download")
            cached_df = None
        else:
            cache_max = cached_df[date_col].max()
            cache_min = cached_df[date_col].min()
            cache_days = cached_df[date_col].nunique()

            # Check if cache already covers the full range
            if cache_max >= end and cache_min <= start:
                print(f"  Cache {os.path.basename(cache_path)}: already covers {start}..{end} "
                      f"({cache_days} days, {len(cached_df)} bars)")
                return cached_df

            # Check if we need significantly earlier data (start moved backward)
            # Tolerate up to 7 days gap — the requested start may be a weekend
            # or holiday where no trading data exists anyway.
            start_gap_days = (dt.datetime.strptime(cache_min, '%Y-%m-%d')
                              - dt.datetime.strptime(start, '%Y-%m-%d')).days
            if start_gap_days > 7:
                print(f"  Cache starts at {cache_min} but need {start} "
                      f"({start_gap_days} days gap) — full re-download")
                cached_df = None
            elif cache_max >= end:
                # Cache covers everything we need
                print(f"  Cache {os.path.basename(cache_path)}: covers through {cache_max} "
                      f"({cache_days} days, {len(cached_df)} bars)")
                return cached_df
            else:
                # Incremental: download from (cache_max - 1 day) for overlap safety
                overlap_start = (dt.datetime.strptime(cache_max, '%Y-%m-%d')
                                 - dt.timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"  Cache {os.path.basename(cache_path)}: has {start}..{cache_max} "
                      f"({cache_days} days). Fetching {overlap_start}..{end}...")
                new_df = download_fn(overlap_start, end)
                if new_df is not None and len(new_df) > 0:
                    merged = pd.concat([cached_df, new_df], ignore_index=True)
                    merged = merged.drop_duplicates(subset='timestamp').sort_values('timestamp')
                    merged = merged.reset_index(drop=True)
                    new_days = merged[date_col].nunique() - cache_days
                    print(f"  Incremental: +{new_days} days, {len(merged)} total bars")
                    with open(cache_path, 'wb') as f:
                        pickle.dump(merged, f)
                    print(f"  Updated cache: {os.path.basename(cache_path)}")
                    return merged
                else:
                    print(f"  No new data fetched — using existing cache")
                    return cached_df
    else:
        cached_df = None

    # Full download (no cache or cache invalidated)
    print(f"  Full download: {start}..{end}")
    df = download_fn(start, end)
    if df is not None and len(df) > 0:
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"  Saved cache: {os.path.basename(cache_path)}")
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
        bars = None
        for attempt, backoff in enumerate([10, 30, 60], 1):
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_str,
                    durationStr="1 W",
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=True,
                    timeout=60,
                )
                if bars is not None and len(bars) > 0:
                    break  # success — got actual data
                elif bars is not None and len(bars) == 0:
                    # Empty list often means timeout/rate-limit, not truly empty
                    print(f"  {current.strftime('%Y-%m-%d')}: attempt {attempt}/3 got 0 bars, retry in {backoff}s")
                    ib.sleep(backoff)
                    bars = None  # reset so we retry
            except Exception as e:
                print(f"  {current.strftime('%Y-%m-%d')}: attempt {attempt}/3 FAIL ({e}), retry in {backoff}s")
                ib.sleep(backoff)
        if bars:
            for b in bars:
                all_bars.append({
                    'timestamp': int(b.date.timestamp() * 1000),
                    f'{prefix}open': b.open, f'{prefix}high': b.high,
                    f'{prefix}low': b.low, f'{prefix}close': b.close,
                })
            print(f"  {current.strftime('%Y-%m-%d')}: {len(bars)} bars (total {len(all_bars)})")
        else:
            print(f"  {current.strftime('%Y-%m-%d')}: 0 bars after 3 attempts")
        current = week_end
        ib.sleep(5)  # rate limit — prevent IBKR pacing violations

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


def _vix_df_to_dict(vix_df: pd.DataFrame) -> dict:
    """Convert VIX DataFrame to timestamp→OHLC lookup dict."""
    vix_data = {}
    for _, row in vix_df.iterrows():
        vix_data[row['timestamp']] = {
            'vix_open': row['vix_open'],
            'vix_high': row['vix_high'],
            'vix_low': row['vix_low'],
            'vix_close': row['vix_close'],
        }
    return vix_data


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
        ib.sleep(5)  # rate limit — prevent IBKR pacing violations

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
        if day_open < 1000:
            raise RuntimeError(f"SPY-scale prices detected ({day_open:.2f}) — use --use-spx for real SPX prices")
        spx_est = day_open
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
    """Download SPXW 0DTE OTM option bars at ATM±5/±10/±15/±20 strikes.

    Core model features still use ±5/±10. ±15/±20 are stored as sidecar labels/data.
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

    print(f"Downloading SPXW OTM chain (ATM±5, ±10, ±15, ±20) for {len(unique_days)} trading days...")

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
        if day_open < 1000:
            raise RuntimeError(f"SPY-scale prices detected ({day_open:.2f}) — use --use-spx for real SPX prices")
        spx_est = day_open
        atm_strike = round(spx_est / 5) * 5

        day_dt = dt.datetime.strptime(day_str, '%Y-%m-%d')
        yymmdd = day_dt.strftime('%y%m%d')

        otm_specs = _otm_chain_specs(atm_strike)

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
                    day_bars_data[key][f'{label}_open'] = b.open
                    day_bars_data[key][f'{label}_close'] = b.close
                    day_bars_data[key][f'{label}_high'] = b.high
                    day_bars_data[key][f'{label}_low'] = b.low
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
        if day_open < 1000:
            raise RuntimeError(f"SPY-scale prices detected ({day_open:.2f}) — use --use-spx for real SPX prices")
        spx_est = day_open
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
            'otm15_call': ('C', atm_strike + 15),
            'otm20_call': ('C', atm_strike + 20),
            'otm5_put': ('P', atm_strike - 5),
            'otm10_put': ('P', atm_strike - 10),
            'otm15_put': ('P', atm_strike - 15),
            'otm20_put': ('P', atm_strike - 20),
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
        for label in [
            'otm5_call', 'otm10_call', 'otm15_call', 'otm20_call',
            'otm5_put', 'otm10_put', 'otm15_put', 'otm20_put',
        ]:
            tk_w, tk_spx, strike = target_tickers[label]
            raw_bars = bars_by_ticker.get(tk_w, []) or bars_by_ticker.get(tk_spx, [])
            for parts in raw_bars:
                ts_ms = int(parts[6]) // 1_000_000
                key = (day_str, ts_ms)
                if key not in chain_day_data:
                    chain_day_data[key] = {'atm_strike': atm_strike}
                chain_day_data[key][f'{label}_open'] = float(parts[2])
                chain_day_data[key][f'{label}_close'] = float(parts[3])
                chain_day_data[key][f'{label}_high'] = float(parts[4])
                chain_day_data[key][f'{label}_low'] = float(parts[5])
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
    Downloads ATM call/put + OTM chain (10 contracts) via reqHistoricalData.
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
        if day_open < 1000:
            raise RuntimeError(f"SPY-scale prices detected ({day_open:.2f}) — use --use-spx for real SPX prices")
        spx_est = day_open
        atm_strike = round(spx_est / 5) * 5

        # Expiry for 0DTE = same day
        expiry = day_str.replace('-', '')  # YYYYMMDD

        # Define the 10 target contracts (core ±5/±10 plus sidecar ±15/±20)
        targets = {
            'atm_call':   ('C', atm_strike),
            'atm_put':    ('P', atm_strike),
            'otm5_call':  ('C', atm_strike + 5),
            'otm10_call': ('C', atm_strike + 10),
            'otm15_call': ('C', atm_strike + 15),
            'otm20_call': ('C', atm_strike + 20),
            'otm5_put':   ('P', atm_strike - 5),
            'otm10_put':  ('P', atm_strike - 10),
            'otm15_put':  ('P', atm_strike - 15),
            'otm20_put':  ('P', atm_strike - 20),
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
        for label in [
            'otm5_call', 'otm10_call', 'otm15_call', 'otm20_call',
            'otm5_put', 'otm10_put', 'otm15_put', 'otm20_put',
        ]:
            _, strike = targets[label]
            for ts, bar in bars_by_label.get(label, {}).items():
                key = (day_str, ts)
                if key not in chain_data:
                    chain_data[key] = {'atm_strike': atm_strike}
                chain_data[key][f'{label}_open'] = bar['open']
                chain_data[key][f'{label}_close'] = bar['close']
                chain_data[key][f'{label}_high'] = bar['high']
                chain_data[key][f'{label}_low'] = bar['low']
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
    """Compute 37 trader-relevant features from SPX 1-min bars (+ SPY volume) + SPXW options.

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
    # Deeper sidecar ladders (kept out of the 60-feature model contract).
    otm15_call_prices = np.full(N, np.nan, dtype=np.float32)
    otm15_put_prices = np.full(N, np.nan, dtype=np.float32)
    otm20_call_prices = np.full(N, np.nan, dtype=np.float32)
    otm20_put_prices = np.full(N, np.nan, dtype=np.float32)
    # Dynamic remap sidecar labels (per-bar ATM and remapped strike ladders).
    dynamic_atm_strikes = np.full(N, np.nan, dtype=np.float32)
    remap_call_strikes = np.full((N, len(OTM_STRIKE_STEPS)), np.nan, dtype=np.float32)
    remap_put_strikes = np.full((N, len(OTM_STRIKE_STEPS)), np.nan, dtype=np.float32)
    static_call_strikes = np.full((N, len(OTM_STRIKE_STEPS)), np.nan, dtype=np.float32)
    static_put_strikes = np.full((N, len(OTM_STRIKE_STEPS)), np.nan, dtype=np.float32)
    call_strike_drift = np.full((N, len(OTM_STRIKE_STEPS)), np.nan, dtype=np.float32)
    put_strike_drift = np.full((N, len(OTM_STRIKE_STEPS)), np.nan, dtype=np.float32)
    # Sidecar quote-quality + cost realism labels for 6 tradeable legs.
    action_spread_bps = np.full((N, len(SIDE_ACTION_ORDER)), np.nan, dtype=np.float32)
    action_quote_age_s = np.full((N, len(SIDE_ACTION_ORDER)), np.nan, dtype=np.float32)
    action_size = np.full((N, len(SIDE_ACTION_ORDER)), np.nan, dtype=np.float32)
    action_quality_score = np.full((N, len(SIDE_ACTION_ORDER)), np.nan, dtype=np.float32)
    action_slippage_bps = np.full((N, len(SIDE_ACTION_ORDER)), np.nan, dtype=np.float32)
    action_cost_bps = np.full((N, len(SIDE_ACTION_ORDER)), np.nan, dtype=np.float32)
    actionable_mask = np.zeros(N, dtype=np.float32)
    risk_state_mask = np.ones(N, dtype=np.float32)
    supervision_weight = np.full(N, np.nan, dtype=np.float32)
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

    # Pre-compute rolling volume profile (POC, VAH, VAL) per bar
    # POC = price level with highest traded volume in session so far
    # Value Area = range containing 70% of session volume
    # Uses 50-bin histogram of close prices weighted by volume
    vp_cache = {}  # i -> (poc, vah, val)
    for day in unique_dates:
        idx = day_indices[day]
        for k, bar_i in enumerate(idx):
            if k < 5:  # need at least 5 bars for meaningful VP
                continue
            session_bars = idx[:k + 1]
            sc = close[session_bars]
            sv = np.maximum(volume[session_bars], 1.0)
            price_min, price_max = sc.min(), sc.max()
            if price_max - price_min < 0.01:
                continue
            n_bins = min(50, max(10, k))
            bin_edges = np.linspace(price_min, price_max, n_bins + 1)
            bin_vol = np.zeros(n_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            for j_vp in range(len(sc)):
                b = min(int((sc[j_vp] - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
                bin_vol[b] += sv[j_vp]
            poc_idx = np.argmax(bin_vol)
            poc = bin_centers[poc_idx]
            # Value Area: expand from POC until 70% of volume captured
            total_vol = bin_vol.sum()
            va_vol = bin_vol[poc_idx]
            lo_b, hi_b = poc_idx, poc_idx
            while va_vol / total_vol < 0.70 and (lo_b > 0 or hi_b < n_bins - 1):
                expand_lo = bin_vol[lo_b - 1] if lo_b > 0 else 0
                expand_hi = bin_vol[hi_b + 1] if hi_b < n_bins - 1 else 0
                if expand_lo >= expand_hi and lo_b > 0:
                    lo_b -= 1
                    va_vol += bin_vol[lo_b]
                elif hi_b < n_bins - 1:
                    hi_b += 1
                    va_vol += bin_vol[hi_b]
                else:
                    lo_b -= 1
                    va_vol += bin_vol[lo_b]
            val_price = bin_edges[lo_b]
            vah_price = bin_edges[hi_b + 1]
            vp_cache[bar_i] = (poc, vah_price, val_price)

    # -----------------------------------------------------------------------
    # Main feature loop (37 features — v2 core + 5 market structure)
    # -----------------------------------------------------------------------
    for i in range(N):
        fi = 0
        day = dates[i]
        c = close[i]

        # === Returns (2): ret_6, ret_12 ===
        for lag in [30, 60]:
            if i >= lag and dates[i - lag] == day:
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

        # Volume at price percentile
        didx = day_indices[day]
        day_pos = np.searchsorted(didx, i)
        if day_pos > 2:
            session_close = close[didx[:day_pos + 1]]
            session_vol = volume[didx[:day_pos + 1]]
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

        # === VWAP (2): vwap_dist, vwap_slope ===
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
        else:
            fi += 2

        # === Session structure (2): ib_width, session_range_pct ===
        ib_h = ib_high_map.get(day, c)
        ib_l = ib_low_map.get(day, c)
        feat[i, fi] = (ib_h - ib_l) / max(c, 1.0)  # ib_width
        fi += 1

        session_so_far = didx[:day_pos + 1]
        session_high = np.max(high[session_so_far])
        session_low = np.min(low[session_so_far])
        feat[i, fi] = (session_high - session_low) / max(c, 1.0)  # session_range_pct
        fi += 1

        # === Key levels (2): prev_high_dist, prev_low_dist ===
        if day in prev_day_high:
            feat[i, fi] = (c - prev_day_high[day]) / max(c, 1.0)
            fi += 1
            feat[i, fi] = (c - prev_day_low[day]) / max(c, 1.0)
            fi += 1
        else:
            fi += 2

        # === Trend (3): ema_cross, consec_direction, speed_estimate ===
        feat[i, fi] = (ema8[i] - ema21[i]) / max(c, 1.0)  # ema_cross
        fi += 1

        # consec_direction
        if i > 0:
            consec = 0
            direction = 1 if close[i] >= close[i - 1] else -1
            for j_c in range(i, max(i - 20, 0) - 1, -1):
                if j_c == 0:
                    break
                bar_dir = 1 if close[j_c] >= close[j_c - 1] else -1
                if bar_dir == direction:
                    consec += 1
                else:
                    break
            feat[i, fi] = direction * min(consec, 10) / 10.0
        fi += 1

        # speed_estimate
        if i >= 5:
            ret5 = abs((c / close[i - 5]) - 1.0)
            rv = feat[i, _FEAT_IDX['realized_vol']]
            if not np.isnan(rv) and rv > 1e-8:
                feat[i, fi] = ret5 / rv
        fi += 1

        # === Microstructure (2): gap, inside_bar ===
        day_start = didx[0]
        if day_start > 0:
            feat[i, fi] = (opn[day_start] / close[day_start - 1]) - 1.0
        else:
            feat[i, fi] = 0.0
        fi += 1

        if i > 0:
            feat[i, fi] = 1.0 if (high[i] <= high[i-1] and low[i] >= low[i-1]) else 0.0
        else:
            feat[i, fi] = 0.0
        fi += 1

        # === Time (3): minutes_to_close, time_sin, time_cos ===
        bar_dt = df.iloc[i]['datetime']
        bar_time = bar_dt.time()
        minutes_into = (bar_time.hour * 60 + bar_time.minute) - (9 * 60 + 30)
        total_session = 390
        minutes_remaining = max(total_session - minutes_into, 0)
        session_progress = minutes_into / total_session
        spx_for_remap = c  # SPX-scale required (--use-spx default)
        dyn_atm = round(spx_for_remap / 5.0) * 5.0
        dynamic_atm_strikes[i] = dyn_atm
        for j, step in enumerate(OTM_STRIKE_STEPS):
            remap_call_strikes[i, j] = dyn_atm + step
            remap_put_strikes[i, j] = dyn_atm - step

        feat[i, fi] = np.log1p(minutes_remaining) / np.log1p(total_session)
        fi += 1
        feat[i, fi] = np.sin(2 * np.pi * session_progress)
        fi += 1
        feat[i, fi] = np.cos(2 * np.pi * session_progress)
        fi += 1

        # === Options (2): atm_iv, iv_skew ===
        opt_key = (dates[i], int(df.iloc[i]['timestamp']))
        opt = options_data.get(opt_key) if options_data else None
        call_iv = np.nan  # initialize for use in Greeks/VIX sections
        if opt is not None and not np.isnan(opt.get('call_close', np.nan)):
            spx = c  # SPX-scale required (--use-spx default)
            K = opt['strike']
            T = minutes_remaining / (252.0 * 390.0)
            r = 0.05

            # Store raw option prices for trade simulation
            atm_call_prices[i] = opt['call_close']
            atm_put_prices[i] = opt.get('put_close', np.nan)
            atm_strikes[i] = K
            call_close = _safe_float(opt.get('call_close', np.nan))
            call_high = _safe_float(opt.get('call_high', np.nan))
            call_low = _safe_float(opt.get('call_low', np.nan))
            call_vol = _safe_float(opt.get('call_volume', 0.0), default=0.0)
            put_close = _safe_float(opt.get('put_close', np.nan))
            put_high = _safe_float(opt.get('put_high', np.nan))
            put_low = _safe_float(opt.get('put_low', np.nan))
            put_vol = _safe_float(opt.get('put_volume', 0.0), default=0.0)

            # Sidecar quote-quality + cost labels for ATM legs.
            for leg_name, px, hi, lo, vol in [
                ("call_atm", call_close, call_high, call_low, call_vol),
                ("put_atm", put_close, put_high, put_low, put_vol),
            ]:
                leg_idx = SIDE_ACTION_TO_IDX.get(leg_name)
                if leg_idx is None or not np.isfinite(px) or px <= 0:
                    continue
                spread_bps = _spread_bps_proxy(px, hi, lo)
                age_s = _quote_age_proxy_seconds(vol)
                quality = _quality_score_proxy(spread_bps, age_s, vol)
                slip_bps = _slippage_bps_proxy(spread_bps, age_s, vol)
                action_spread_bps[i, leg_idx] = spread_bps
                action_quote_age_s[i, leg_idx] = age_s
                action_size[i, leg_idx] = vol
                action_quality_score[i, leg_idx] = quality
                action_slippage_bps[i, leg_idx] = slip_bps
                action_cost_bps[i, leg_idx] = spread_bps + 2.0 * slip_bps

            # atm_iv (average of call + put IV)
            call_iv = _bs_iv(opt['call_close'], spx, K, T, r, is_call=True)
            put_iv = np.nan
            if not np.isnan(put_close):
                put_iv = _bs_iv(put_close, spx, K, T, r, is_call=False)
            if not np.isnan(call_iv) and not np.isnan(put_iv):
                feat[i, fi] = (call_iv + put_iv) / 2.0
            elif not np.isnan(call_iv):
                feat[i, fi] = call_iv
            elif not np.isnan(put_iv):
                feat[i, fi] = put_iv
            fi += 1

            # iv_skew (put IV - call IV)
            if not np.isnan(call_iv) and not np.isnan(put_iv):
                feat[i, fi] = put_iv - call_iv
            fi += 1
        else:
            fi += 2  # skip options features

        # === VIX / Regime (2): vix_regime, vrp ===
        ts_ms = int(df.iloc[i]['timestamp'])
        vix_bar = vix_data.get(ts_ms) if vix_data else None
        cur_iv = feat[i, _FEAT_IDX['atm_iv']]
        cur_rv = feat[i, _FEAT_IDX['realized_vol']]

        # Determine VIX value: real CBOE VIX ONLY (no fallback)
        if vix_bar is not None:
            vix_val = vix_bar['vix_close']
            vix_annualized = vix_val / 100.0
        else:
            # VIX data missing for this bar — leave as NaN (validated at end)
            vix_val = np.nan
            vix_annualized = np.nan

        if not np.isnan(vix_val):
            # vix_regime
            if vix_val < 15.0:
                feat[i, fi] = -1.0
            elif vix_val < 20.0:
                feat[i, fi] = -0.33
            elif vix_val < 30.0:
                feat[i, fi] = 0.33
            else:
                feat[i, fi] = 1.0
            fi += 1

            # vrp
            if not np.isnan(cur_iv) and not np.isnan(cur_rv) and cur_rv > 0:
                feat[i, fi] = cur_iv ** 2 - cur_rv ** 2
            fi += 1
        else:
            fi += 2  # skip VIX features

        # === OTM chain data (prices for trade simulation only, no features) ===
        chain_bar = chain_data.get(opt_key) if chain_data else None
        if chain_bar is not None and opt is not None:
            spx_for_iv = c  # SPX-scale required (--use-spx default)
            T_iv = minutes_remaining / (252.0 * 390.0)
            r_iv = 0.05
            chain_close_map: dict[str, float] = {}
            chain_high_map: dict[str, float] = {}
            chain_low_map: dict[str, float] = {}
            chain_volume_map: dict[str, float] = {}
            chain_strike_map: dict[str, float] = {}
            for j, step in enumerate(OTM_STRIKE_STEPS):
                for side in ("call", "put"):
                    k = f"otm{step}_{side}"
                    chain_close_map[k] = _safe_float(chain_bar.get(f"{k}_close", np.nan))
                    chain_high_map[k] = _safe_float(chain_bar.get(f"{k}_high", chain_close_map[k]))
                    chain_low_map[k] = _safe_float(chain_bar.get(f"{k}_low", chain_close_map[k]))
                    chain_volume_map[k] = _safe_float(chain_bar.get(f"{k}_volume", 0.0), default=0.0)
                    chain_strike_map[k] = _safe_float(chain_bar.get(f"{k}_strike", np.nan))
                    if side == "call":
                        static_call_strikes[i, j] = chain_strike_map[k]
                        if np.isfinite(chain_strike_map[k]):
                            call_strike_drift[i, j] = chain_strike_map[k] - remap_call_strikes[i, j]
                    else:
                        static_put_strikes[i, j] = chain_strike_map[k]
                        if np.isfinite(chain_strike_map[k]):
                            put_strike_drift[i, j] = chain_strike_map[k] - remap_put_strikes[i, j]

            # Store raw OTM prices for trade simulation + deeper sidecar ladders.
            otm5c = chain_close_map.get("otm5_call", np.nan)
            otm5p = chain_close_map.get("otm5_put", np.nan)
            otm10c = chain_close_map.get("otm10_call", np.nan)
            otm10p = chain_close_map.get("otm10_put", np.nan)
            otm15c = chain_close_map.get("otm15_call", np.nan)
            otm15p = chain_close_map.get("otm15_put", np.nan)
            otm20c = chain_close_map.get("otm20_call", np.nan)
            otm20p = chain_close_map.get("otm20_put", np.nan)

            if not np.isnan(otm5c):
                otm5_call_prices[i] = otm5c
            if not np.isnan(otm5p):
                otm5_put_prices[i] = otm5p
            if not np.isnan(otm10c):
                otm10_call_prices[i] = otm10c
            if not np.isnan(otm10p):
                otm10_put_prices[i] = otm10p
            if not np.isnan(otm15c):
                otm15_call_prices[i] = otm15c
            if not np.isnan(otm15p):
                otm15_put_prices[i] = otm15p
            if not np.isnan(otm20c):
                otm20_call_prices[i] = otm20c
            if not np.isnan(otm20p):
                otm20_put_prices[i] = otm20p

            # Sidecar quote-quality + cost labels for tradeable OTM legs (±5, ±10).
            for leg_name, chain_key in SIDE_ACTION_TO_CHAIN_KEY.items():
                leg_idx = SIDE_ACTION_TO_IDX[leg_name]
                px = chain_close_map.get(chain_key, np.nan)
                if not np.isfinite(px) or px <= 0:
                    continue
                spread_bps = _spread_bps_proxy(
                    px,
                    chain_high_map.get(chain_key, px),
                    chain_low_map.get(chain_key, px),
                )
                age_s = _quote_age_proxy_seconds(chain_volume_map.get(chain_key, 0.0))
                qsize = chain_volume_map.get(chain_key, 0.0)
                quality = _quality_score_proxy(spread_bps, age_s, qsize)
                slip_bps = _slippage_bps_proxy(spread_bps, age_s, qsize)
                action_spread_bps[i, leg_idx] = spread_bps
                action_quote_age_s[i, leg_idx] = age_s
                action_size[i, leg_idx] = qsize
                action_quality_score[i, leg_idx] = quality
                action_slippage_bps[i, leg_idx] = slip_bps
                action_cost_bps[i, leg_idx] = spread_bps + 2.0 * slip_bps

        # === Greeks (3): atm_gamma, atm_theta_per_bar, charm_estimate ===
        if opt is not None and not np.isnan(opt.get('call_close', np.nan)):
            spx_g = c  # SPX-scale required (--use-spx default)
            K_g = opt['strike']
            T_g = minutes_remaining / (252.0 * 390.0)
            r_g = 0.05
            sigma_g = call_iv if not np.isnan(call_iv) else np.nan
            if not np.isnan(sigma_g) and T_g > 1e-10:
                delta, gamma, theta_bar, vega = _bs_greeks(spx_g, K_g, T_g, r_g, sigma_g)
                # atm_gamma
                if not np.isnan(gamma):
                    feat[i, fi] = gamma
                fi += 1
                # atm_theta_per_bar
                if not np.isnan(theta_bar):
                    feat[i, fi] = theta_bar
                fi += 1
                # charm_estimate
                if sigma_g > 0:
                    d1 = (math.log(spx_g / K_g) + (0.05 + 0.5 * sigma_g**2) * T_g) / (sigma_g * math.sqrt(T_g))
                    nd1_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
                    charm_val = -nd1_pdf * (2.0 * 0.05 * T_g - d1 * sigma_g * math.sqrt(T_g)) / (2.0 * T_g * sigma_g * math.sqrt(T_g))
                    if abs(charm_val) < 100:
                        feat[i, fi] = charm_val
                fi += 1
            else:
                fi += 3  # skip Greeks (no valid IV)
        else:
            fi += 3  # skip Greeks (no option data)

        # === Bollinger (1): bollinger_position ===
        if i >= 20:
            bb_window = close[max(i - 20, 0):i + 1]
            bb_mid = np.mean(bb_window)
            bb_std = np.std(bb_window)
            bb_width = (bb_mid + 2.0 * bb_std) - (bb_mid - 2.0 * bb_std)
            if bb_width > 1e-8:
                feat[i, fi] = (c - bb_mid) / (bb_width / 2.0)
        fi += 1

        # === Range extras (2): rsi_14, session_range_position ===
        # rsi_14
        if i >= 14:
            rsi_window = close[i - 14:i + 1]
            rsi_changes = np.diff(rsi_window)
            gains = np.maximum(rsi_changes, 0)
            losses = np.maximum(-rsi_changes, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 1e-10:
                rs = avg_gain / avg_loss
                feat[i, fi] = rs / (1.0 + rs)
            else:
                feat[i, fi] = 1.0
        fi += 1

        # session_range_position
        if session_high > session_low:
            feat[i, fi] = (c - session_low) / (session_high - session_low)
        else:
            feat[i, fi] = 0.5
        fi += 1

        # === Market structure (5): poc_dist, va_position, vwap_band_sigma, ib_break, theta_pressure ===

        # poc_dist: (close - POC) / close
        vp_data = vp_cache.get(i)
        if vp_data is not None:
            poc, vah, val = vp_data
            feat[i, fi] = (c - poc) / max(c, 1.0)
        fi += 1

        # va_position: where price sits in Value Area (0=VAL, 1=VAH)
        if vp_data is not None:
            poc, vah, val = vp_data
            va_range = vah - val
            if va_range > 0.01:
                feat[i, fi] = (c - val) / va_range
            else:
                feat[i, fi] = 0.5
        fi += 1

        # vwap_band_sigma: distance from VWAP in σ units
        vw_data_ms = vwap_cache.get(i)
        if vw_data_ms is not None:
            vw_ms, u1_ms, l1_ms, u2_ms, l2_ms = vw_data_ms
            if np.isfinite(u1_ms) and np.isfinite(l1_ms):
                sigma = (u1_ms - vw_ms)  # 1σ distance
                if sigma > 1e-8:
                    feat[i, fi] = (c - vw_ms) / sigma
        fi += 1

        # ib_break: -1 if below IB low, 0 if inside, +1 if above IB high
        ib_h_ms = ib_high_map.get(day, c)
        ib_l_ms = ib_low_map.get(day, c)
        if c > ib_h_ms:
            feat[i, fi] = 1.0
        elif c < ib_l_ms:
            feat[i, fi] = -1.0
        else:
            feat[i, fi] = 0.0
        fi += 1

        # theta_pressure: ramps 0→1 from bar 120 (11:30am) to close (bar 390)
        bar_in_day = day_pos  # 0-indexed position in session
        feat[i, fi] = max(0.0, (bar_in_day - 120) / 270.0) if bar_in_day > 120 else 0.0
        fi += 1

        # Sidecar quality/risk masks for supervision weighting.
        row_quality = action_quality_score[i]
        row_cost = action_cost_bps[i]
        valid_q = np.isfinite(row_quality) & np.isfinite(row_cost)
        vix_reg = feat[i, _FEAT_IDX['vix_regime']]
        risk_off = (
            minutes_remaining <= 5
            or (np.isfinite(vix_reg) and vix_reg >= 1.0)
        )
        risk_state_mask[i] = 0.0 if risk_off else 1.0
        if np.any(valid_q):
            q_mean = float(np.nanmean(row_quality[valid_q]))
            c_mean = float(np.nanmean(row_cost[valid_q]))
            leg_coverage = float(np.mean(valid_q.astype(np.float32)))
            is_actionable = (q_mean >= 0.30) and (c_mean <= 500.0) and (leg_coverage >= 0.34)
            actionable_mask[i] = 1.0 if is_actionable else 0.0
            base_w = float(np.clip(q_mean * (1.0 - min(c_mean, 700.0) / 700.0), 0.0, 1.0))
            if risk_off:
                base_w *= 0.50
            if not is_actionable:
                base_w *= 0.20
            supervision_weight[i] = max(base_w, 0.0)
        else:
            actionable_mask[i] = 0.0
            supervision_weight[i] = 0.0

        assert fi == NUM_FEATURES, f"Feature count mismatch: {fi} != {NUM_FEATURES}"

        # Target: forward FORWARD_BARS return (still used for equity-level loss)
        if i + FORWARD_BARS < N and dates[i + FORWARD_BARS] == day:
            targets[i] = (close[i + FORWARD_BARS] / close[i]) - 1.0

    # -------------------------------------------------------------------
    # Option P&L targets: EOD/max-hold exits (no hardcoded stop)
    # -------------------------------------------------------------------
    # For each bar i, simulate a trade to EOD or max_hold.
    # No stop-loss in training labels — the model learns exits via gate head,
    # and dynamic stops are applied only during evaluation.
    call_pnl = np.full(N, np.nan, dtype=np.float32)
    put_pnl = np.full(N, np.nan, dtype=np.float32)
    call_pnl_realistic = np.full(N, np.nan, dtype=np.float32)
    put_pnl_realistic = np.full(N, np.nan, dtype=np.float32)

    _price_arrays = {
        'call_atm': atm_call_prices,
        'put_atm': atm_put_prices,
    }

    for i in range(N):
        for leg_name, pnl_arr, pnl_real_arr in [
            ('call_atm', call_pnl, call_pnl_realistic),
            ('put_atm', put_pnl, put_pnl_realistic),
        ]:
            px_arr = _price_arrays[leg_name]
            if np.isnan(px_arr[i]) or px_arr[i] <= 0:
                continue
            cost_bps = action_cost_bps[i, SIDE_ACTION_TO_IDX[leg_name]]
            if not np.isfinite(cost_bps):
                cost_bps = 2.0 * OPTION_SPREAD_BPS
            cost_pct = cost_bps / 10000.0
            _, pnl_val, _ = compute_dynamic_pnl(i, px_arr, dates, SPREAD_COST_PCT, stop_loss=1.0)
            if not np.isnan(pnl_val):
                pnl_arr[i] = pnl_val
            _, pnl_real_val, _ = compute_dynamic_pnl(i, px_arr, dates, cost_pct, stop_loss=1.0)
            if not np.isnan(pnl_real_val):
                pnl_real_arr[i] = pnl_real_val

    # -------------------------------------------------------------------
    # OTM P&L targets: EOD/max-hold exits for OTM strikes (6-class dir head)
    # -------------------------------------------------------------------
    otm5_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm5_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm15_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm15_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm20_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm20_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm5_call_pnl_realistic = np.full(N, np.nan, dtype=np.float32)
    otm5_put_pnl_realistic = np.full(N, np.nan, dtype=np.float32)
    otm10_call_pnl_realistic = np.full(N, np.nan, dtype=np.float32)
    otm10_put_pnl_realistic = np.full(N, np.nan, dtype=np.float32)

    _otm_legs = [
        (otm5_call_prices, otm5_call_pnl, otm5_call_pnl_realistic, "call_otm5"),
        (otm5_put_prices, otm5_put_pnl, otm5_put_pnl_realistic, "put_otm5"),
        (otm10_call_prices, otm10_call_pnl, otm10_call_pnl_realistic, "call_otm10"),
        (otm10_put_prices, otm10_put_pnl, otm10_put_pnl_realistic, "put_otm10"),
        (otm15_call_prices, otm15_call_pnl, None, "call_otm15"),
        (otm15_put_prices, otm15_put_pnl, None, "put_otm15"),
        (otm20_call_prices, otm20_call_pnl, None, "call_otm20"),
        (otm20_put_prices, otm20_put_pnl, None, "put_otm20"),
    ]

    for i in range(N):
        for px_arr, pnl_arr, pnl_real_arr, leg_name in _otm_legs:
            if np.isnan(px_arr[i]) or px_arr[i] <= 0:
                continue
            _, pnl_val, _ = compute_dynamic_pnl(i, px_arr, dates, SPREAD_COST_PCT, stop_loss=1.0)
            if not np.isnan(pnl_val):
                pnl_arr[i] = pnl_val
            if pnl_real_arr is not None and leg_name in SIDE_ACTION_TO_IDX:
                leg_idx = SIDE_ACTION_TO_IDX[leg_name]
                leg_cost_bps = action_cost_bps[i, leg_idx]
                if not np.isfinite(leg_cost_bps):
                    leg_cost_bps = 2.0 * OPTION_SPREAD_BPS
                _, pnl_real_val, _ = compute_dynamic_pnl(i, px_arr, dates, leg_cost_bps / 10000.0, stop_loss=1.0)
                if not np.isnan(pnl_real_val):
                    pnl_real_arr[i] = pnl_real_val

    # -------------------------------------------------------------------
    # Stopped P&L targets: same as above but WITH dynamic stop applied.
    # Computed at 3 stop levels (tight/medium/wide) to align with dynamic
    # stops at eval time. Train.py selects the closest level based on
    # predicted gate confidence.
    # -------------------------------------------------------------------
    STOP_LEVELS = [0.20, 0.35, 0.50]  # tight, medium, wide
    STOP_LEVEL_NAMES = ['tight', 'med', 'wide']

    # ATM stopped P&L at default level (backward compat)
    call_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    put_stopped_pnl = np.full(N, np.nan, dtype=np.float32)

    # Multi-level stopped P&L for ATM
    call_stopped_pnl_tight = np.full(N, np.nan, dtype=np.float32)
    call_stopped_pnl_wide = np.full(N, np.nan, dtype=np.float32)
    put_stopped_pnl_tight = np.full(N, np.nan, dtype=np.float32)
    put_stopped_pnl_wide = np.full(N, np.nan, dtype=np.float32)

    for i in range(N):
        for leg_name, pnl_med, pnl_tight, pnl_wide in [
            ('call_atm', call_stopped_pnl, call_stopped_pnl_tight, call_stopped_pnl_wide),
            ('put_atm', put_stopped_pnl, put_stopped_pnl_tight, put_stopped_pnl_wide),
        ]:
            px_arr = _price_arrays[leg_name]
            if np.isnan(px_arr[i]) or px_arr[i] <= 0:
                continue
            for stop_level, pnl_arr in zip(STOP_LEVELS, [pnl_tight, pnl_med, pnl_wide]):
                _, pnl_val, _ = compute_dynamic_pnl(
                    i, px_arr, dates, SPREAD_COST_PCT,
                    stop_loss=stop_level, max_hold=MAX_HOLD_BARS)
                if not np.isnan(pnl_val):
                    pnl_arr[i] = pnl_val

    # OTM stopped P&L (default level only — less critical for OTM)
    otm5_call_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm5_put_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_call_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_put_stopped_pnl = np.full(N, np.nan, dtype=np.float32)

    _otm_stopped_legs = [
        (otm5_call_prices, otm5_call_stopped_pnl, "call_otm5"),
        (otm5_put_prices, otm5_put_stopped_pnl, "put_otm5"),
        (otm10_call_prices, otm10_call_stopped_pnl, "call_otm10"),
        (otm10_put_prices, otm10_put_stopped_pnl, "put_otm10"),
    ]

    for i in range(N):
        for px_arr, pnl_arr, leg_name in _otm_stopped_legs:
            if np.isnan(px_arr[i]) or px_arr[i] <= 0:
                continue
            _, pnl_val, _ = compute_dynamic_pnl(
                i, px_arr, dates, SPREAD_COST_PCT,
                stop_loss=DYNAMIC_STOP_BASE, max_hold=MAX_HOLD_BARS)
            if not np.isnan(pnl_val):
                pnl_arr[i] = pnl_val

    # -------------------------------------------------------------------
    # EXIT labels: hindsight-optimal exit timing.
    # For each bar i, look at hypothetical entries from past N bars.
    # EXIT=1 if current bar is at or near the peak P&L for any entry,
    # or if P&L has dropped >50% from its peak (trailing-stop signal).
    # This teaches the model WHEN to exit — cut losers, let winners run.
    # -------------------------------------------------------------------
    exit_call_label = np.full(N, np.nan, dtype=np.float32)
    exit_put_label = np.full(N, np.nan, dtype=np.float32)

    # Max lookback for exit labels (limit to 120 bars for performance)
    _exit_lookback = min(MAX_HOLD_BARS, 120)
    # Trailing stop threshold: exit if P&L drops this fraction from high-water mark
    _trail_drop_frac = 0.50
    # Minimum HWM before trailing stop activates (avoid noisy exits on tiny gains)
    _trail_min_hwm = 0.05
    # Momentum stall: exit if P&L hasn't improved in this many bars
    _stall_bars = 10
    # Minimum P&L to trigger stall-based exit (don't exit stalls on losing trades)
    _stall_min_pnl = 0.02

    for i in range(N):
        best_call_exit = 0.0
        best_put_exit = 0.0
        has_call_data = False
        has_put_data = False

        for k in range(1, _exit_lookback + 1):
            entry = i - k
            if entry < 0 or dates[entry] != dates[i]:
                continue

            # --- Call: causal exit signals (no future data) ---
            ec = atm_call_prices[entry]
            cc = atm_call_prices[i]
            if not np.isnan(ec) and not np.isnan(cc) and ec > 0:
                unrealized_now = (cc - ec) / ec - SPREAD_COST_PCT
                has_call_data = True

                # Find high-water mark from entry to current bar (backward only)
                hwm = unrealized_now
                for back in range(1, k + 1):
                    past = i - back
                    if past < entry:
                        break
                    pc = atm_call_prices[past]
                    if not np.isnan(pc) and ec > 0:
                        hwm = max(hwm, (pc - ec) / ec - SPREAD_COST_PCT)

                # Signal 1: Trailing stop — P&L dropped >50% from HWM
                if hwm > _trail_min_hwm and unrealized_now < hwm * (1.0 - _trail_drop_frac):
                    best_call_exit = 1.0

                # Signal 2: Momentum stall — P&L positive but hasn't improved recently
                if unrealized_now > _stall_min_pnl and k >= _stall_bars:
                    stall_start = i - _stall_bars
                    if stall_start >= entry:
                        sc = atm_call_prices[stall_start]
                        if not np.isnan(sc) and ec > 0:
                            pnl_at_stall = (sc - ec) / ec - SPREAD_COST_PCT
                            if unrealized_now <= pnl_at_stall:
                                best_call_exit = 1.0

            # --- Put: same causal logic ---
            ep = atm_put_prices[entry]
            cp = atm_put_prices[i]
            if not np.isnan(ep) and not np.isnan(cp) and ep > 0:
                unrealized_now = (cp - ep) / ep - SPREAD_COST_PCT
                has_put_data = True

                hwm = unrealized_now
                for back in range(1, k + 1):
                    past = i - back
                    if past < entry:
                        break
                    pp = atm_put_prices[past]
                    if not np.isnan(pp) and ep > 0:
                        hwm = max(hwm, (pp - ep) / ep - SPREAD_COST_PCT)

                if hwm > _trail_min_hwm and unrealized_now < hwm * (1.0 - _trail_drop_frac):
                    best_put_exit = 1.0

                if unrealized_now > _stall_min_pnl and k >= _stall_bars:
                    stall_start = i - _stall_bars
                    if stall_start >= entry:
                        sp = atm_put_prices[stall_start]
                        if not np.isnan(sp) and ep > 0:
                            pnl_at_stall = (sp - ep) / ep - SPREAD_COST_PCT
                            if unrealized_now <= pnl_at_stall:
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
        'dynamic_atm_strike': dynamic_atm_strikes,
        'remap_call_strikes': remap_call_strikes,
        'remap_put_strikes': remap_put_strikes,
        'static_call_strikes': static_call_strikes,
        'static_put_strikes': static_put_strikes,
        'call_strike_drift': call_strike_drift,
        'put_strike_drift': put_strike_drift,
        'call_pnl': call_pnl,
        'put_pnl': put_pnl,
        'call_pnl_realistic': call_pnl_realistic,
        'put_pnl_realistic': put_pnl_realistic,
        'exit_call_label': exit_call_label,
        'exit_put_label': exit_put_label,
        'otm5_call': otm5_call_prices,
        'otm5_put': otm5_put_prices,
        'otm10_call': otm10_call_prices,
        'otm10_put': otm10_put_prices,
        'otm15_call': otm15_call_prices,
        'otm15_put': otm15_put_prices,
        'otm20_call': otm20_call_prices,
        'otm20_put': otm20_put_prices,
        'otm5_call_pnl': otm5_call_pnl,
        'otm5_put_pnl': otm5_put_pnl,
        'otm10_call_pnl': otm10_call_pnl,
        'otm10_put_pnl': otm10_put_pnl,
        'otm15_call_pnl': otm15_call_pnl,
        'otm15_put_pnl': otm15_put_pnl,
        'otm20_call_pnl': otm20_call_pnl,
        'otm20_put_pnl': otm20_put_pnl,
        'otm5_call_pnl_realistic': otm5_call_pnl_realistic,
        'otm5_put_pnl_realistic': otm5_put_pnl_realistic,
        'otm10_call_pnl_realistic': otm10_call_pnl_realistic,
        'otm10_put_pnl_realistic': otm10_put_pnl_realistic,
        'call_stopped_pnl': call_stopped_pnl,
        'put_stopped_pnl': put_stopped_pnl,
        'call_stopped_pnl_tight': call_stopped_pnl_tight,
        'call_stopped_pnl_wide': call_stopped_pnl_wide,
        'put_stopped_pnl_tight': put_stopped_pnl_tight,
        'put_stopped_pnl_wide': put_stopped_pnl_wide,
        'otm5_call_stopped_pnl': otm5_call_stopped_pnl,
        'otm5_put_stopped_pnl': otm5_put_stopped_pnl,
        'otm10_call_stopped_pnl': otm10_call_stopped_pnl,
        'otm10_put_stopped_pnl': otm10_put_stopped_pnl,
        'action_leg_names': list(SIDE_ACTION_ORDER),
        'action_spread_bps': action_spread_bps,
        'action_quote_age_s': action_quote_age_s,
        'action_size': action_size,
        'action_quality_score': action_quality_score,
        'action_slippage_bps': action_slippage_bps,
        'action_cost_bps': action_cost_bps,
        'actionable_mask': actionable_mask,
        'risk_state_mask': risk_state_mask,
        'supervision_weight': supervision_weight,
    }

    return feat, targets, dates.tolist(), valid, option_prices, timestamps.tolist()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _rolling_zscore(features: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    out = features.copy()
    for j in range(out.shape[1]):
        name = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"feature_{j}"
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


def _per_day_zscore(features: np.ndarray, valid: np.ndarray, dates: list) -> np.ndarray:
    """Per-day mean, global std normalization.

    Removes day-specific feature fingerprints that cause the model to memorize
    individual dates (e.g. the Sep 17 attractor) while preserving cross-day
    regime information via the global standard deviation.
    """
    out = features.copy().astype(np.float64)

    # Build day boundaries from dates
    day_starts = [0] + [i for i in range(1, len(dates)) if dates[i] != dates[i - 1]]
    day_ends = day_starts[1:] + [len(dates)]

    for j in range(out.shape[1]):
        name = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"feature_{j}"
        if name in _NO_NORMALIZE:
            continue

        # Global std across all valid bars (preserves regime-level differences)
        all_vals = out[:, j][valid]
        if len(all_vals) < 10:
            out[:, j] = 0.0
            continue
        global_std = np.std(all_vals)
        if global_std < 1e-10:
            out[:, j] = 0.0
            continue

        # Per-day mean subtraction (removes day-specific fingerprints)
        for ds, de in zip(day_starts, day_ends):
            chunk = out[ds:de, j]
            mask = valid[ds:de]
            day_vals = chunk[mask]
            if len(day_vals) < 3:
                out[ds:de, j] = 0.0
            else:
                day_mean = np.mean(day_vals)
                out[ds:de, j] = (chunk - day_mean) / global_std

    out = np.clip(out, -5.0, 5.0)
    out = np.nan_to_num(out, nan=0.0)
    return out


def normalize_features(features: np.ndarray, valid: np.ndarray,
                        dates: list | None = None) -> np.ndarray:
    """Per-day z-score normalization (if dates provided) or rolling z-score fallback.

    Per-day mode uses per-day mean + global std to prevent cross-day fingerprinting
    while preserving regime-level information. Falls back to rolling z-score for
    replay/live (no dates available).
    """
    if len(features) == 0:
        return features.copy()
    if dates is not None and len(dates) == len(features):
        return _per_day_zscore(features, valid.astype(bool), dates)
    # Fallback: rolling z-score (for replay/live where dates aren't available)
    window = min(len(features) // 4, 500)
    window = max(window, 50)
    return _rolling_zscore(features, valid.astype(bool), int(window))


def normalize_features_with_context(
    features: np.ndarray,
    valid: np.ndarray,
    context_raw: np.ndarray | None,
    context_valid: np.ndarray | None,
) -> np.ndarray:
    """Normalize features with rolling history seeded by training-era context.

    This preserves continuity of rolling statistics for replay/live bars that
    come after the training date range.
    """
    if (
        context_raw is None
        or context_valid is None
        or len(context_raw) == 0
        or len(context_valid) == 0
    ):
        return normalize_features(features, valid)

    if context_raw.ndim != 2 or features.ndim != 2:
        return normalize_features(features, valid)
    if context_raw.shape[1] != features.shape[1]:
        return normalize_features(features, valid)
    if len(context_valid) != len(context_raw):
        return normalize_features(features, valid)

    ctx_raw = np.asarray(context_raw, dtype=np.float64)
    cur_raw = np.asarray(features, dtype=np.float64)
    merged_raw = np.vstack([ctx_raw, cur_raw])
    merged_valid = np.concatenate(
        [np.asarray(context_valid, dtype=bool), np.asarray(valid, dtype=bool)]
    )
    window = min(len(merged_raw) // 4, 500)
    window = max(window, 50)
    merged_norm = _rolling_zscore(merged_raw, merged_valid, int(window))
    return merged_norm[len(ctx_raw):]


# ---------------------------------------------------------------------------
# Tensor preparation
# ---------------------------------------------------------------------------

def prepare_tensors(features: np.ndarray, targets: np.ndarray,
                    dates: list, valid: np.ndarray,
                    option_prices: dict | None = None,
                    timestamps: list | None = None,
                    raw_features: np.ndarray | None = None,
                    norm_window: int = 500) -> dict:
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

    raw_source = raw_features if raw_features is not None else features
    if len(raw_source):
        ctx_window = max(1, min(int(norm_window), len(raw_source)))
        data['norm_window'] = int(ctx_window)
        data['norm_raw_buffer'] = torch.tensor(raw_source[-ctx_window:], dtype=torch.float32)
        data['norm_valid_buffer'] = torch.tensor(valid[-ctx_window:], dtype=torch.bool)

    # Store option prices + P&L targets for trade simulation and training
    if option_prices is not None:
        data['atm_call_prices'] = torch.tensor(option_prices['atm_call'], dtype=torch.float32)
        data['atm_put_prices'] = torch.tensor(option_prices['atm_put'], dtype=torch.float32)
        data['atm_strikes'] = torch.tensor(option_prices['strike'], dtype=torch.float32)
        if 'dynamic_atm_strike' in option_prices:
            data['dynamic_atm_strike'] = torch.tensor(option_prices['dynamic_atm_strike'], dtype=torch.float32)
        for k in ('remap_call_strikes', 'remap_put_strikes', 'static_call_strikes', 'static_put_strikes',
                  'call_strike_drift', 'put_strike_drift'):
            if k in option_prices:
                data[k] = torch.tensor(option_prices[k], dtype=torch.float32)
        data['call_pnl'] = torch.tensor(option_prices['call_pnl'], dtype=torch.float32)
        data['put_pnl'] = torch.tensor(option_prices['put_pnl'], dtype=torch.float32)
        if 'call_pnl_realistic' in option_prices:
            data['call_pnl_realistic'] = torch.tensor(option_prices['call_pnl_realistic'], dtype=torch.float32)
        if 'put_pnl_realistic' in option_prices:
            data['put_pnl_realistic'] = torch.tensor(option_prices['put_pnl_realistic'], dtype=torch.float32)
        data['exit_call_label'] = torch.tensor(option_prices['exit_call_label'], dtype=torch.float32)
        data['exit_put_label'] = torch.tensor(option_prices['exit_put_label'], dtype=torch.float32)
        # OTM prices and P&L for tradeable OTM actions
        data['otm5_call_prices'] = torch.tensor(option_prices['otm5_call'], dtype=torch.float32)
        data['otm5_put_prices'] = torch.tensor(option_prices['otm5_put'], dtype=torch.float32)
        data['otm10_call_prices'] = torch.tensor(option_prices['otm10_call'], dtype=torch.float32)
        data['otm10_put_prices'] = torch.tensor(option_prices['otm10_put'], dtype=torch.float32)
        for k in ('otm15_call', 'otm15_put', 'otm20_call', 'otm20_put'):
            if k in option_prices:
                data[f'{k}_prices'] = torch.tensor(option_prices[k], dtype=torch.float32)
        data['otm5_call_pnl'] = torch.tensor(option_prices['otm5_call_pnl'], dtype=torch.float32)
        data['otm5_put_pnl'] = torch.tensor(option_prices['otm5_put_pnl'], dtype=torch.float32)
        data['otm10_call_pnl'] = torch.tensor(option_prices['otm10_call_pnl'], dtype=torch.float32)
        data['otm10_put_pnl'] = torch.tensor(option_prices['otm10_put_pnl'], dtype=torch.float32)
        for k in (
            'otm15_call_pnl', 'otm15_put_pnl', 'otm20_call_pnl', 'otm20_put_pnl',
            'otm5_call_pnl_realistic', 'otm5_put_pnl_realistic',
            'otm10_call_pnl_realistic', 'otm10_put_pnl_realistic',
            'call_stopped_pnl', 'put_stopped_pnl',
            'call_stopped_pnl_tight', 'call_stopped_pnl_wide',
            'put_stopped_pnl_tight', 'put_stopped_pnl_wide',
            'otm5_call_stopped_pnl', 'otm5_put_stopped_pnl',
            'otm10_call_stopped_pnl', 'otm10_put_stopped_pnl',
        ):
            if k in option_prices:
                data[k] = torch.tensor(option_prices[k], dtype=torch.float32)
        for k in (
            'action_spread_bps', 'action_quote_age_s', 'action_size',
            'action_quality_score', 'action_slippage_bps', 'action_cost_bps',
            'actionable_mask', 'risk_state_mask', 'supervision_weight',
        ):
            if k in option_prices:
                data[k] = torch.tensor(option_prices[k], dtype=torch.float32)
        if 'action_leg_names' in option_prices:
            data['action_leg_names'] = list(option_prices['action_leg_names'])

    # Day boundaries for sequential batching (Phase 3)
    date_list = data['dates'].tolist() if isinstance(data['dates'], torch.Tensor) else list(data['dates'])
    day_boundaries = [0] + [i for i in range(1, len(date_list)) if date_list[i] != date_list[i - 1]]
    data['day_boundaries'] = torch.tensor(day_boundaries, dtype=torch.long)

    # --- Data quality guardrails (HARD FAIL) ---
    if 'actionable_mask' in data:
        am = data['actionable_mask'].numpy() if isinstance(data['actionable_mask'], torch.Tensor) else data['actionable_mask']
        dt_arr = data['dates'].numpy() if isinstance(data['dates'], torch.Tensor) else data['dates']
        from collections import Counter
        date_act_counts = Counter()
        total_act = 0
        for idx_g in range(len(am)):
            if am[idx_g] > 0.5:
                date_act_counts[str(dt_arr[idx_g])] += 1
                total_act += 1
        n_act_dates = len(date_act_counts)
        print(f"  Actionable bars: {total_act}/{len(am)} ({100*total_act/max(len(am),1):.1f}%)")
        print(f"  Actionable dates: {n_act_dates}")
        for d_g, cnt_g in date_act_counts.most_common(5):
            print(f"    Date {d_g}: {cnt_g} bars ({100*cnt_g/max(total_act,1):.1f}%)")
        if n_act_dates < 5:
            raise RuntimeError(
                f"HARD FAIL: Only {n_act_dates} dates have actionable bars (need >=5). "
                f"Check _spread_bps_proxy calibration and actionable_mask threshold."
            )
        if total_act > 0:
            max_d, max_cnt = date_act_counts.most_common(1)[0]
            if max_cnt / total_act > 0.30:
                raise RuntimeError(
                    f"HARD FAIL: Date {max_d} has {max_cnt}/{total_act} "
                    f"({100*max_cnt/total_act:.0f}%) of actionable bars — data quality inconsistency."
                )

    # --- Data completeness validation (HARD FAIL) ---
    feat_tensor = data['features']
    n_bars = feat_tensor.shape[0]
    n_feat = feat_tensor.shape[-1]
    assert n_feat == NUM_FEATURES, f"Feature count mismatch: {n_feat} != {NUM_FEATURES}"

    # Check per-feature NaN rate
    feat_np = feat_tensor.numpy() if isinstance(feat_tensor, torch.Tensor) else feat_tensor
    for fi_check in range(n_feat):
        nan_rate = np.isnan(feat_np[:, fi_check]).mean() if feat_np.ndim == 2 else np.isnan(feat_np[:, :, fi_check]).mean()
        if nan_rate > 0.05:
            fname = FEATURE_NAMES[fi_check] if fi_check < len(FEATURE_NAMES) else f"feature_{fi_check}"
            raise RuntimeError(
                f"HARD FAIL: Feature '{fname}' (idx {fi_check}) has {nan_rate*100:.1f}% NaN rate (max 5%)")

    # Check required stopped P&L fields exist
    for req_key in ('call_stopped_pnl', 'put_stopped_pnl',
                    'otm5_call_stopped_pnl', 'otm5_put_stopped_pnl',
                    'otm10_call_stopped_pnl', 'otm10_put_stopped_pnl',
                    'call_stopped_pnl_tight', 'call_stopped_pnl_wide',
                    'put_stopped_pnl_tight', 'put_stopped_pnl_wide'):
        if req_key not in data:
            raise RuntimeError(f"HARD FAIL: Missing required field '{req_key}' in data.pt")

    print(f"  Data completeness: PASSED ({n_bars} bars, {n_feat} features, all NaN rates <5%)")

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


def _load_dataloader_arrays(data, device):
    """Load all target arrays needed by dataloaders. Returns a dict of tensors."""
    required_targets = (
        'call_pnl', 'put_pnl', 'exit_call_label', 'exit_put_label',
        'otm5_call_pnl', 'otm5_put_pnl', 'otm10_call_pnl', 'otm10_put_pnl',
    )
    missing = [k for k in required_targets if k not in data]
    if missing:
        raise KeyError(
            "data.pt missing required targets: " + ", ".join(missing)
            + ". Rebuild data.pt with the current prepare.py."
        )

    call_pnl_all = data['call_pnl'].to(device)
    arrays = {
        'features': data['features'].to(device),
        'targets': data['targets'].to(device),
        'call_pnl': call_pnl_all,
        'put_pnl': data['put_pnl'].to(device),
        'exit_call': data['exit_call_label'].to(device),
        'exit_put': data['exit_put_label'].to(device),
        'otm5_call_pnl': data['otm5_call_pnl'].to(device),
        'otm5_put_pnl': data['otm5_put_pnl'].to(device),
        'otm10_call_pnl': data['otm10_call_pnl'].to(device),
        'otm10_put_pnl': data['otm10_put_pnl'].to(device),
    }
    for k in ('supervision_weight', 'actionable_mask', 'risk_state_mask'):
        v = data.get(k)
        arrays[k] = v.to(device) if v is not None else torch.ones_like(call_pnl_all)

    # Stopped P&L arrays (v5) — REQUIRED (no fallback)
    for k in ('call_stopped_pnl', 'put_stopped_pnl',
              'otm5_call_stopped_pnl', 'otm5_put_stopped_pnl',
              'otm10_call_stopped_pnl', 'otm10_put_stopped_pnl'):
        v = data.get(k)
        if v is not None:
            arrays[k] = v.to(device)
        else:
            raise RuntimeError(f"Missing required field in data.pt: {k} — rebuild data.pt")

    # Multi-level stopped P&L (tight=0.20, wide=0.50) — REQUIRED (no fallback)
    for k in ('call_stopped_pnl_tight', 'call_stopped_pnl_wide',
              'put_stopped_pnl_tight', 'put_stopped_pnl_wide'):
        v = data.get(k)
        if v is not None:
            arrays[k] = v.to(device)
        else:
            raise RuntimeError(f"Missing required field in data.pt: {k} — rebuild data.pt")

    return arrays


def _build_y_tuple(arrays, idx):
    """Build the y tuple for a batch of indices."""
    return (arrays['targets'][idx],
            arrays['call_pnl'][idx], arrays['put_pnl'][idx],
            arrays['exit_call'][idx], arrays['exit_put'][idx],
            arrays['otm5_call_pnl'][idx], arrays['otm5_put_pnl'][idx],
            arrays['otm10_call_pnl'][idx], arrays['otm10_put_pnl'][idx],
            arrays['supervision_weight'][idx], arrays['actionable_mask'][idx],
            arrays['risk_state_mask'][idx],
            # v5: stopped P&L at med level (positions 12-17)
            arrays['call_stopped_pnl'][idx], arrays['put_stopped_pnl'][idx],
            arrays['otm5_call_stopped_pnl'][idx], arrays['otm5_put_stopped_pnl'][idx],
            arrays['otm10_call_stopped_pnl'][idx], arrays['otm10_put_stopped_pnl'][idx],
            # v6: multi-level stopped P&L tight/wide (positions 18-21)
            arrays['call_stopped_pnl_tight'][idx], arrays['call_stopped_pnl_wide'][idx],
            arrays['put_stopped_pnl_tight'][idx], arrays['put_stopped_pnl_wide'][idx])


def make_dataloader(data, lookback, batch_size, split="train", device="cuda", target_mask=None):
    """Infinite (train) or single-pass (val) dataloader.

    Yields (x, y):
        x: (batch, lookback, NUM_FEATURES)
        y: tuple of (fwd_ret, call_pnl, put_pnl, exit_call, exit_put,
                     otm5_call_pnl, otm5_put_pnl, otm10_call_pnl, otm10_put_pnl,
                     supervision_weight, actionable_mask, risk_state_mask,
                     call_stopped_pnl, put_stopped_pnl,
                     otm5_call_stopped_pnl, otm5_put_stopped_pnl,
                     otm10_call_stopped_pnl, otm10_put_stopped_pnl,
                     call_stopped_pnl_tight, call_stopped_pnl_wide,
                     put_stopped_pnl_tight, put_stopped_pnl_wide)
           each (batch,). NaN where option data is unavailable.

    Args:
        target_mask: optional bool tensor/array same length as features. When provided,
                     only indices where target_mask[i] is True are used as training targets.
                     Lookback context still uses the original valid_mask.
    """
    arrays = _load_dataloader_arrays(data, device)
    features = arrays['features']
    valid_mask = data['valid_mask']

    if split == "train":
        end = data['train_end_idx'] + 1
    else:
        end = data['val_end_idx'] + 1

    start = max(lookback, data['val_start_idx'] if split != "train" else lookback)

    valid_indices = []
    for i in range(start, end):
        if not valid_mask[i]:
            continue
        if target_mask is not None:
            # Specialist mode: target_mask selects which bars to train on.
            # Lookback validity relaxed — invalid lookback bars are zero-filled.
            if target_mask[i]:
                valid_indices.append(i)
        else:
            # Standard mode: require full lookback validity
            if valid_mask[max(0, i - lookback):i].all():
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
                yield x, _build_y_tuple(arrays, idx)
    else:
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            idx = valid_indices[i:end_i]
            window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
            x = features[window_idx]
            yield x, _build_y_tuple(arrays, idx)


def make_day_sequential_loader(data, lookback, batch_size, device="cuda", split="train"):
    """Day-sequential dataloader: yields bars sequentially within sampled days.

    For each epoch, samples `batch_size` random days from the split, then
    iterates through bars of each day sequentially. This allows carrying
    position state forward across bars within a day.

    Yields (x, y, bar_in_day):
        x: (actual_batch, lookback, NUM_FEATURES)
        y: same tuple as make_dataloader
        bar_in_day: int, position within the trading day (0 = first bar)
    """
    arrays = _load_dataloader_arrays(data, device)
    features = arrays['features']
    valid_mask = data['valid_mask']

    if split == "train":
        end = data['train_end_idx'] + 1
    else:
        end = data['val_end_idx'] + 1
    start = max(lookback, data['val_start_idx'] if split != "train" else lookback)

    # Build day boundaries within the split
    day_boundaries = data.get('day_boundaries')
    if day_boundaries is None:
        raise KeyError("data.pt missing 'day_boundaries'. Rebuild with current prepare.py.")
    day_boundaries = day_boundaries.tolist()

    # Find days that fall within the split range
    split_days = []  # list of (day_start, day_end) tuples
    for d in range(len(day_boundaries)):
        day_start = day_boundaries[d]
        day_end = day_boundaries[d + 1] if d + 1 < len(day_boundaries) else len(valid_mask)
        # Day must overlap with split range
        if day_end <= start or day_start >= end:
            continue
        effective_start = max(day_start, start)
        effective_end = min(day_end, end)
        if effective_end - effective_start >= lookback + 1:
            split_days.append((effective_start, effective_end))

    assert len(split_days) > 0, f"No valid days for split={split}"

    offsets = torch.arange(-lookback, 0, device=device)

    while True:
        # Sample batch_size random days
        day_indices = torch.randint(0, len(split_days), (min(batch_size, len(split_days)),))
        selected_days = [split_days[di] for di in day_indices]

        # Truncate to shortest selected day so all days contribute equally
        # to every bar position. Prevents long days from dominating late offsets.
        min_bars = min(de - ds for ds, de in selected_days)

        for bar_offset in range(lookback, min_bars):
            # Every selected day contributes to every bar position
            batch_indices = []
            for ds, de in selected_days:
                bar_idx = ds + bar_offset
                if bar_idx < de and bar_idx >= start:
                    if valid_mask[bar_idx] and valid_mask[max(0, bar_idx - lookback):bar_idx].all():
                        batch_indices.append(bar_idx)

            if len(batch_indices) == 0:
                continue

            idx = torch.tensor(batch_indices, dtype=torch.long, device=device)
            window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
            x = features[window_idx]
            yield x, _build_y_tuple(arrays, idx), bar_offset - lookback


# ---------------------------------------------------------------------------
# Evaluation: Trade Simulation (the new primary metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_trades(model, data, lookback, device, batch_size=1024,
                    stop_loss_pct=None, max_hold_bars=None,
                    max_trade_return=None,
                    starting_capital=None, position_risk_target=None,
                    score_config=None):
    """Simulate 0DTE option trades on validation set.

    Required model output format (strict foundation contract):
      - Two-head: model(x) returns (gate_logits, dir_logits)
        gate_logits: (batch, 2) [NO_TRADE, TRADE]
        dir_logits:  (batch, 6) [CALL_ATM, CALL_OTM5, CALL_OTM10,
                                 PUT_ATM, PUT_OTM5, PUT_OTM10]

    Effective semantics:
      - Gate=TRADE + direction head -> one of 6 BUY actions.
      - Gate=NO_TRADE while in position -> model EXIT.
      - Gate=NO_TRADE while flat -> DO_NOTHING.

    Optional overrides (defaults from module constants):
      stop_loss_pct: Stop loss as fraction of premium (default 0.30)
      max_hold_bars: Max bars to hold a position (default BARS_PER_DAY)
      max_trade_return: Cap individual trade P&L (default 5.0 = 500%)
      starting_capital: Starting account balance for equity curve (default 10000.0)
      position_risk_target: Target risk per trade as fraction of account (default 0.05 = 5%).
          Determines whole contract count: n = max(1, floor(balance * target / cost)).
      score_config: Dict of score tuning params (all default to neutral/0.0):
        - win_rate_bonus: Reward high win rates (0.0-1.0)
        - rr_bonus: Reward good R:R ratio (0.0-2.0)
        - drawdown_penalty: Penalize deep drawdowns (0.0-1.0)
        - hold_bonus: Reward appropriate hold times (0.0-1.0)
        - freq_center: Ideal trades per day (1.0-8.0, default 3.0)
        - freq_width: How tight the freq band is (1.0-6.0, default 3.0)
        - consec_loss_threshold: Max consecutive losses before penalty (2-8, default 3)
        - short_hold_threshold: Short hold % penalty trigger (0.10-0.60, default 0.30)
        - stop_rate_threshold: Stop loss rate penalty trigger (0.10-0.60, default 0.30)
        - ruin_penalty: Severity of account ruin penalty (0.0-1.0, default 1.0)
        - ruin_threshold: Equity fraction that triggers ruin (0.05-0.50, default 0.25 = 75% loss)

    Returns dict with trader + quant metrics and composite score.
    """
    # Allow train.py to override strategy parameters
    _stop_loss = stop_loss_pct if stop_loss_pct is not None else STOP_LOSS_PCT  # legacy; dynamic stop used instead
    _max_hold = max_hold_bars if max_hold_bars is not None else MAX_HOLD_BARS
    _max_return = max_trade_return if max_trade_return is not None else MAX_TRADE_RETURN
    _starting_capital = starting_capital if starting_capital is not None else STARTING_CAPITAL
    _position_risk_target = position_risk_target if position_risk_target is not None else POSITION_RISK_TARGET

    # Score tuning config (all defaults produce neutral/unchanged score)
    _sc = score_config or {}
    _sc_wr_bonus = float(_sc.get('win_rate_bonus', 0.0))
    _sc_rr_bonus = float(_sc.get('rr_bonus', 0.0))
    _sc_dd_penalty = float(_sc.get('drawdown_penalty', 0.5))
    _sc_hold_bonus = float(_sc.get('hold_bonus', 0.0))
    _sc_freq_center = float(_sc.get('freq_center', 3.0))
    _sc_freq_width = float(_sc.get('freq_width', 3.0))
    _sc_consec_thresh = int(_sc.get('consec_loss_threshold', 3))
    _sc_short_thresh = float(_sc.get('short_hold_threshold', 0.30))
    _sc_stop_thresh = float(_sc.get('stop_rate_threshold', 0.30))
    _sc_ruin_penalty = float(_sc.get('ruin_penalty', 1.0))
    _sc_ruin_thresh = float(_sc.get('ruin_threshold', 0.25))
    _sc_risk_frac_penalty = float(_sc.get('risk_fraction_penalty', 0.5))

    model.eval()

    features = data['features'].to(device)
    targets = data['targets']
    valid_mask = data['valid_mask']
    dates = data['dates']
    timestamps = data.get('timestamps', dates)
    atm_strikes = data.get('atm_strikes')
    action_cost_matrix = data.get('action_cost_bps')
    action_quality_matrix = data.get('action_quality_score')
    actionable_series = data.get('actionable_mask')
    risk_state_series = data.get('risk_state_mask')

    val_start = max(lookback, data['val_start_idx'])
    val_end = data['val_end_idx'] + 1

    val_indices = [
        i for i in range(val_start, val_end)
        if valid_mask[i] and valid_mask[max(0, i - lookback):i].all()
    ]

    if len(val_indices) < 10:
        return _empty_metrics(len(val_indices))

    actionable_bar_rate = 0.0
    risk_off_bar_rate = 0.0
    if actionable_series is not None:
        actionable_vals = np.array([float(actionable_series[i]) for i in val_indices], dtype=np.float64)
        actionable_bar_rate = float(np.mean(actionable_vals))
    if risk_state_series is not None:
        risk_vals = np.array([float(risk_state_series[i]) for i in val_indices], dtype=np.float64)
        risk_off_bar_rate = float(np.mean(1.0 - risk_vals))

    val_idx_t = torch.tensor(val_indices, dtype=torch.long, device=device)
    offsets = torch.arange(-lookback, 0, device=device)

    # Phase 1: Batch inference for direction logits (position-independent)
    # Phase 2: Sequential inference for gate decisions with position state
    all_dir_actions = []
    all_dir_logits_list = []
    for i in range(0, len(val_idx_t), batch_size):
        idx = val_idx_t[i:i + batch_size]
        window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
        x = features[window_idx]
        out = model(x)  # no position_state → gate_input = last (backward compat)
        if not isinstance(out, tuple) or len(out) < 2:
            raise ValueError(
                "evaluate_trades requires model output tuple with at least "
                "(gate_logits, dir_logits). Got: " + str(type(out))
            )
        gate_logits, dir_logits = out[0], out[1]
        if gate_logits.ndim != 2 or gate_logits.shape[-1] != 2:
            raise ValueError(
                f"Invalid gate head shape: expected (batch, 2), got {tuple(gate_logits.shape)}"
            )
        if dir_logits.ndim != 2 or dir_logits.shape[-1] != 6:
            raise ValueError(
                f"Invalid direction head shape: expected (batch, 6), got {tuple(dir_logits.shape)}"
            )
        dir_action = torch.argmax(dir_logits, dim=-1)
        all_dir_actions.append(dir_action.cpu())

    dir_actions = torch.cat(all_dir_actions).numpy()

    # Phase 2: Position-aware sequential gate inference
    # Build position state for each bar based on trade simulation state
    _has_position_proj = hasattr(model, 'position_proj')
    actions = np.empty(len(val_indices), dtype=np.int64)
    gate_no_trade = np.empty(len(val_indices), dtype=bool)
    gate_confidence = np.empty(len(val_indices), dtype=np.float64)

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

    # Helper: map action → option price array (needed by both position tracking and trade sim)
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

    # Feature indices for dynamic stop computation
    _idx_atm_iv = _FEAT_IDX['atm_iv']
    _idx_vix_regime = _FEAT_IDX['vix_regime']

    # Track position state for gate decisions
    _pos_in_trade = False
    _pos_bars_held = 0
    _pos_unrealized_pnl = 0.0
    _pos_entry_price = 0.0
    _pos_px_array = None
    _pos_last_stop_bar = -STOP_COOLDOWN_BARS
    _pos_dynamic_stop = DYNAMIC_STOP_BASE  # per-trade dynamic stop (set at entry)
    # Account state tracking (shadow simulation for position_state input)
    _pos_account_balance = _starting_capital
    _pos_consecutive_losses = 0
    _pos_n_contracts = 1

    _pos_best_pnl = 0.0           # Phase D: best P&L since entry
    _pos_bars_since_high = 0       # Phase D: bars since P&L peak

    for k, global_idx in enumerate(val_indices):
        # Build position state tensor (7 dims: holding, bars_held, unrealized_pnl,
        #   account_health, loss_streak, best_pnl, bars_since_high)
        if _has_position_proj:
            _ps_dim = getattr(model, 'POSITION_STATE_DIM', 7)
            pos_state = torch.zeros(1, _ps_dim, device=device)
            if _pos_in_trade:
                pos_state[0, 0] = 1.0
                pos_state[0, 1] = min(_pos_bars_held / BARS_PER_DAY, 1.0)
                pos_state[0, 2] = float(np.tanh(_pos_unrealized_pnl * 5.0))
                if _ps_dim >= 7:
                    pos_state[0, 5] = float(np.tanh(_pos_best_pnl * 2.0))
                    pos_state[0, 6] = min(_pos_bars_since_high / BARS_PER_DAY, 1.0)
            pos_state[0, 3] = _pos_account_balance / _starting_capital  # account_health
            pos_state[0, 4] = min(_pos_consecutive_losses / max(_sc_consec_thresh, 1), 1.0)  # loss_streak_frac
        else:
            pos_state = None

        # Run gate inference with position state
        idx_t = val_idx_t[k:k+1]
        window_idx = idx_t.unsqueeze(1) + offsets.unsqueeze(0)
        x = features[window_idx]
        _out = model(x, position_state=pos_state)
        gate_logits = _out[0]
        gate_action = int(torch.argmax(gate_logits, dim=-1).item())
        gate_conf = float(torch.softmax(gate_logits, dim=-1)[0, 1].item())  # P(TRADE)

        gate_no_trade[k] = (gate_action == 0)
        gate_confidence[k] = gate_conf
        if gate_action == 1:
            actions[k] = int(dir_actions[k]) + 1  # BUY_CALL_ATM=1 .. BUY_PUT_OTM10=6
        else:
            actions[k] = ACTION_DO_NOTHING

        # Update position tracking for next bar's position state
        if _pos_in_trade:
            _pos_bars_held += 1
            if _pos_px_array is not None:
                px_now = _pos_px_array[global_idx]
                if not torch.isnan(px_now) and _pos_entry_price > 0:
                    _pos_unrealized_pnl = (float(px_now) - _pos_entry_price) / _pos_entry_price
            # Phase D: track best P&L for value head context
            if _pos_unrealized_pnl > _pos_best_pnl:
                _pos_best_pnl = _pos_unrealized_pnl
                _pos_bars_since_high = 0
            else:
                _pos_bars_since_high += 1

            # Phase D: value-based exit
            _value_exit = False
            _has_value_head = hasattr(model, 'value_head')
            if _has_value_head and _pos_bars_held >= 2:
                with torch.no_grad():
                    _vout = model(x, position_state=pos_state, return_value=True)
                    _vp = float(_vout[2][0].item())
                if _vp < 0.02:  # VALUE_EXIT_THRESHOLD
                    _value_exit = True

            # Check exit conditions (mirrors trade loop below)
            hit_stop = _pos_unrealized_pnl <= -_pos_dynamic_stop
            hit_max_hold = _pos_bars_held >= _max_hold
            entry_date = dates[val_indices[k - _pos_bars_held]] if k >= _pos_bars_held else None
            eod = dates[global_idx] != entry_date if entry_date else False
            model_exit = gate_no_trade[k]
            if hit_stop or hit_max_hold or eod or model_exit or _value_exit:
                # Approximate dollar P&L for account tracking
                _exit_pnl = -_pos_dynamic_stop if hit_stop else _pos_unrealized_pnl
                _dollar_pnl = _exit_pnl * _pos_entry_price * SPX_MULTIPLIER * _pos_n_contracts
                _pos_account_balance += _dollar_pnl
                _pos_account_balance = max(_pos_account_balance, 0.0)
                if _exit_pnl <= 0:
                    _pos_consecutive_losses += 1
                else:
                    _pos_consecutive_losses = 0
                _pos_in_trade = False
                _pos_best_pnl = 0.0
                _pos_bars_since_high = 0
                if hit_stop:
                    _pos_last_stop_bar = k
        elif actions[k] in {ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5, ACTION_BUY_CALL_OTM10,
                            ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5, ACTION_BUY_PUT_OTM10}:
            if (k - _pos_last_stop_bar) >= STOP_COOLDOWN_BARS:
                if _bar_of_day.get(global_idx, 999) >= NO_TRADE_BEFORE_BAR:
                    candidate_px_array = _get_px_array(actions[k], data)
                    if candidate_px_array is not None and not torch.isnan(candidate_px_array[global_idx]):
                        entry_px = float(candidate_px_array[global_idx])
                        if entry_px > 0:
                            # Affordability check: can't buy what you can't afford
                            contract_cost = entry_px * SPX_MULTIPLIER
                            if contract_cost > _pos_account_balance:
                                actions[k] = ACTION_DO_NOTHING
                            else:
                                _pos_n_contracts = max(1, int(_pos_account_balance * _position_risk_target / contract_cost))
                                _pos_in_trade = True
                                _pos_bars_held = 0
                                _pos_entry_price = entry_px
                                _pos_px_array = candidate_px_array
                                _pos_unrealized_pnl = 0.0
                                _pos_best_pnl = 0.0
                                _pos_bars_since_high = 0
                                # Compute dynamic stop at entry from confidence + market state
                                _pos_dynamic_stop = compute_dynamic_stop(
                                    gate_conf,
                                    float(features[global_idx, _idx_atm_iv]),
                                    float(features[global_idx, _idx_vix_regime]),
                                )

    # Count unique val dates
    val_dates_list = [dates[i] for i in val_indices]
    num_val_days = len(set(val_dates_list))

    # -------------------------------------------------------------------
    # Simulate trades (strict option-price-based P&L)
    # -------------------------------------------------------------------
    # Map action → price array for each strike/direction
    _ENTRY_ACTIONS = {ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5, ACTION_BUY_CALL_OTM10,
                      ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5, ACTION_BUY_PUT_OTM10}

    _ACTION_NAMES = {
        ACTION_BUY_CALL_ATM: 'CALL_ATM', ACTION_BUY_CALL_OTM5: 'CALL_OTM5',
        ACTION_BUY_CALL_OTM10: 'CALL_OTM10', ACTION_BUY_PUT_ATM: 'PUT_ATM',
        ACTION_BUY_PUT_OTM5: 'PUT_OTM5', ACTION_BUY_PUT_OTM10: 'PUT_OTM10',
    }

    required_price_keys = (
        'atm_call_prices',
        'atm_put_prices',
        'otm5_call_prices',
        'otm5_put_prices',
        'otm10_call_prices',
        'otm10_put_prices',
    )
    missing_prices = [k for k in required_price_keys if k not in data or data.get(k) is None]
    if missing_prices:
        raise KeyError(
            "data.pt missing required option price arrays: "
            + ", ".join(missing_prices)
            + ". Rebuild data.pt with the current prepare.py."
        )

    trade_pnls = []
    trade_details = []
    entry_cost_bps_samples = []
    entry_quality_samples = []
    entry_cost_known_count = 0
    high_cost_entries = 0
    low_quality_entries = 0
    model_exit_count = 0
    in_trade = False
    trade_entry_bar = 0
    trade_action = 0
    trade_entry_price = 0.0
    trade_entry_cost_bps = 2.0 * OPTION_SPREAD_BPS
    trade_entry_quality = float("nan")
    trade_entry_actionable = 0.0
    trade_last_price = 0.0
    trade_use_actual = False
    trade_px_array = None
    trade_dynamic_stop = DYNAMIC_STOP_BASE  # per-trade dynamic stop (set at entry)
    last_stop_bar = -STOP_COOLDOWN_BARS  # initialize so first entry isn't blocked
    cooldown_blocked_count = 0
    pre_10am_blocked_count = 0
    do_nothing_count = 0
    exit_signal_count = 0
    # Inline account tracking (replaces post-hoc equity curve)
    account_balance = _starting_capital
    equity_history = [_starting_capital]
    trade_risk_fractions = []
    trades_blocked_by_balance = 0
    trade_n_contracts = 1

    for k, global_idx in enumerate(val_indices):
        gate_flat_signal = bool(gate_no_trade[k])
        if in_trade:
            bars_held = k - trade_entry_bar
            entry_global = val_indices[trade_entry_bar]

            # --- P&L computation ---
            if not trade_use_actual or trade_px_array is None or trade_entry_price <= 0:
                raise RuntimeError("Invalid trade state: active position without usable option prices.")

            px_now = trade_px_array[global_idx]
            if not torch.isnan(px_now):
                trade_last_price = float(px_now)
            current_px = trade_last_price
            net_pnl_pct = (current_px - trade_entry_price) / trade_entry_price

            hit_stop = net_pnl_pct <= -trade_dynamic_stop
            hit_max_hold = bars_held >= _max_hold
            eod = dates[global_idx] != dates[entry_global]
            model_exit = gate_flat_signal

            # Phase D: value-based exit
            _trade_value_exit = False
            _has_value_head = hasattr(model, 'value_head')
            if _has_value_head and bars_held >= 2 and not hit_stop:
                with torch.no_grad():
                    _vout = model(x, position_state=pos_state, return_value=True)
                    _vp = float(_vout[2][0].item())
                if _vp < 0.02:  # VALUE_EXIT_THRESHOLD
                    _trade_value_exit = True

            if hit_stop or hit_max_hold or eod or model_exit or _trade_value_exit or k == len(val_indices) - 1:
                if hit_stop:
                    final_pnl = -trade_dynamic_stop
                    last_stop_bar = k
                else:
                    final_pnl = net_pnl_pct

                final_pnl -= float(trade_entry_cost_bps) / 10000.0

                # Cap individual trade P&L to eliminate fat-tail lottery dependency
                final_pnl = max(-trade_dynamic_stop, min(final_pnl, _max_return))

                if model_exit:
                    model_exit_count += 1
                    exit_signal_count += 1

                # Exit reason (priority: stop > model > value > max_hold > eod)
                if hit_stop:
                    exit_reason = 'stop_loss'
                elif model_exit:
                    exit_reason = 'model_exit'
                elif _trade_value_exit:
                    exit_reason = 'value_exit'
                elif eod:
                    exit_reason = 'end_of_day'
                elif hit_max_hold:
                    exit_reason = 'max_hold'
                else:
                    exit_reason = 'end_of_data'

                # Strike info
                strike_val = float(atm_strikes[entry_global]) if atm_strikes is not None and not torch.isnan(atm_strikes[entry_global]) else None

                # Inline account tracking: compute dollar P&L and update balance
                dollar_pnl = final_pnl * trade_entry_price * SPX_MULTIPLIER * trade_n_contracts
                account_balance += dollar_pnl
                account_balance = max(account_balance, 0.0)
                equity_history.append(account_balance)

                trade_pnls.append(final_pnl)
                # Extract diagnostic context for trade-level analysis
                _entry_bod = _bar_of_day.get(entry_global, -1)
                _vix_idx = _FEAT_IDX.get('vix_regime')
                _entry_vix = float(features[entry_global, _vix_idx].cpu()) if _vix_idx is not None else 0.0

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
                    'entry_cost_bps': round(float(trade_entry_cost_bps), 4),
                    'entry_quality': None if np.isnan(trade_entry_quality) else round(float(trade_entry_quality), 4),
                    'entry_actionable': int(trade_entry_actionable > 0.5),
                    'pnl_pct': round(final_pnl * 100, 4),
                    'dollar_pnl': round(dollar_pnl, 2),
                    'n_contracts': trade_n_contracts,
                    'exit_reason': exit_reason,
                    'dynamic_stop_pct': round(trade_dynamic_stop * 100, 2),
                    'actual_prices': trade_use_actual,
                    'result': 'WIN' if final_pnl > 0 else 'LOSS',
                    'bar_of_day': _entry_bod,
                    'vix_regime': round(_entry_vix, 2),
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
            candidate_action = actions[k]
            candidate_px_array = _get_px_array(candidate_action, data)
            if candidate_px_array is None:
                continue
            if torch.isnan(candidate_px_array[global_idx]):
                continue
            entry_px = float(candidate_px_array[global_idx])
            if entry_px <= 0:
                continue
            # Affordability check: can't buy what you can't afford
            contract_cost = entry_px * SPX_MULTIPLIER
            if contract_cost > account_balance:
                trades_blocked_by_balance += 1
                continue
            trade_n_contracts = max(1, int(account_balance * _position_risk_target / contract_cost))
            total_position_cost = contract_cost * trade_n_contracts
            trade_risk_fractions.append(total_position_cost / max(account_balance, 1e-10))
            action_idx = int(candidate_action - 1)
            entry_cost_bps = 2.0 * OPTION_SPREAD_BPS
            entry_quality = float("nan")
            if action_cost_matrix is not None and action_idx >= 0:
                try:
                    c_bps = float(action_cost_matrix[global_idx, action_idx])
                    if np.isfinite(c_bps):
                        entry_cost_bps = c_bps
                        entry_cost_known_count += 1
                except Exception:
                    pass
            if action_quality_matrix is not None and action_idx >= 0:
                try:
                    q_val = float(action_quality_matrix[global_idx, action_idx])
                    if np.isfinite(q_val):
                        entry_quality = q_val
                except Exception:
                    pass
            entry_actionable = 0.0
            if actionable_series is not None:
                try:
                    entry_actionable = float(actionable_series[global_idx])
                except Exception:
                    entry_actionable = 0.0
            entry_cost_bps_samples.append(float(entry_cost_bps))
            if np.isfinite(entry_quality):
                entry_quality_samples.append(float(entry_quality))
            if entry_cost_bps > 250.0:
                high_cost_entries += 1
            if np.isfinite(entry_quality) and entry_quality < 0.30:
                low_quality_entries += 1

            in_trade = True
            trade_entry_bar = k
            trade_action = candidate_action
            trade_px_array = candidate_px_array
            trade_entry_price = entry_px
            trade_entry_cost_bps = float(entry_cost_bps)
            trade_entry_quality = float(entry_quality)
            trade_entry_actionable = float(entry_actionable)
            trade_last_price = entry_px
            trade_use_actual = True
            # Compute dynamic stop at entry from gate confidence + market state
            trade_dynamic_stop = compute_dynamic_stop(
                gate_confidence[k],
                float(features[global_idx, _idx_atm_iv]),
                float(features[global_idx, _idx_vix_regime]),
            )
        elif gate_flat_signal:
            do_nothing_count += 1

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

    # --- Additional metrics for score tuning ---
    avg_hold_bars = float(np.mean([d['bars_held'] for d in trade_details])) if trade_details else 0.0
    rr_ratio = abs(avg_winner) / max(abs(avg_loser), 1e-10) if avg_loser != 0 else 10.0

    # Composite score with configurable trade frequency band
    # freq_center and freq_width define the sweet spot
    _freq_hi = _sc_freq_center + _sc_freq_width   # upper bound of sweet spot
    _freq_lo = max(0.5, _sc_freq_center - _sc_freq_width)  # lower bound
    if trades_per_day < 0.5:
        score = -10.0
    elif trades_per_day < _freq_lo:
        # Ramp from -5 to raw_score as tpd approaches sweet spot
        if trades_per_day <= _freq_hi:
            freq_mult = min(1.0, trades_per_day / max(_sc_freq_center, 1.0))
        else:
            freq_mult = max(0.1, (_freq_hi / trades_per_day) ** 2)
        raw_score = profit_factor * trade_sharpe * freq_mult
        ramp = (trades_per_day - 0.5) / max(_freq_lo - 0.5, 0.5)
        ramp = min(1.0, ramp)
        score = -5.0 * (1.0 - ramp) + raw_score * ramp
    else:
        if trades_per_day <= _freq_hi:
            freq_mult = min(1.0, trades_per_day / max(_sc_freq_center, 1.0))
        else:
            freq_mult = max(0.1, (_freq_hi / trades_per_day) ** 2)
        score = profit_factor * trade_sharpe * freq_mult

    # --- Configurable penalties (thresholds tunable by agent) ---

    # Consecutive loss penalty (strengthened: 15% per loss beyond threshold, floor 0.2)
    if max_consec_loss > _sc_consec_thresh and score > 0:
        consec_penalty = max(0.2, 1.0 - 0.15 * (max_consec_loss - _sc_consec_thresh))
        score *= consec_penalty

    # Short-hold penalty
    short_hold_pct = 0.0
    if num_trades > 10:
        short_holds = sum(1 for d in trade_details if d['bars_held'] <= 1)
        short_hold_pct = short_holds / num_trades
        if short_hold_pct > _sc_short_thresh and score > 0:
            noise_penalty = max(0.7, 1.0 - (short_hold_pct - _sc_short_thresh))
            score *= noise_penalty

    # Stop-loss rate penalty
    stop_loss_rate = 0.0
    if num_trades > 10:
        stop_loss_rate = sum(1 for d in trade_details if d.get('exit_reason') == 'stop_loss') / num_trades
        if stop_loss_rate > _sc_stop_thresh and score > 0:
            sl_penalty = max(0.5, 1.0 - (stop_loss_rate - _sc_stop_thresh))
            score *= sl_penalty

    # --- New tunable bonuses/penalties (all default to neutral at 0.0) ---

    # Win rate bonus: reward consistent winners
    if _sc_wr_bonus > 0 and score > 0:
        wr_mult = 1.0 + _sc_wr_bonus * max(0.0, (win_rate - 0.40)) / 0.60
        score *= wr_mult

    # R:R ratio bonus: reward strategies where avg_win > avg_loss
    if _sc_rr_bonus > 0 and score > 0:
        rr_mult = 1.0 + _sc_rr_bonus * max(0.0, (rr_ratio - 1.0)) / 2.0
        score *= rr_mult

    # Max drawdown penalty: punish deep cumulative drawdowns
    if _sc_dd_penalty > 0 and score > 0:
        dd_mult = max(0.3, 1.0 - _sc_dd_penalty * max(0.0, abs(max_drawdown) - 0.10))
        score *= dd_mult

    # Hold time quality bonus: reward avg hold in sweet spot (5-60 bars)
    if _sc_hold_bonus > 0 and score > 0:
        # Bell curve centered at 30 bars, width 25
        hold_z = (avg_hold_bars - 30.0) / 25.0
        hold_bell = math.exp(-0.5 * hold_z * hold_z)
        hold_mult = 1.0 + _sc_hold_bonus * hold_bell
        score *= hold_mult

    # Risk fraction penalty: penalize models that risk too much per trade
    avg_risk_fraction = float(np.mean(trade_risk_fractions)) if trade_risk_fractions else 0.0
    max_risk_fraction = float(np.max(trade_risk_fractions)) if trade_risk_fractions else 0.0
    if _sc_risk_frac_penalty > 0 and trade_risk_fractions and score > 0:
        if avg_risk_fraction > 0.30:
            risk_penalty = max(0.3, 1.0 - _sc_risk_frac_penalty * (avg_risk_fraction - 0.30))
            score *= risk_penalty

    total_bars = len(actions)
    do_nothing_pct = float(do_nothing_count) / max(total_bars, 1)
    exit_pct = float(exit_signal_count) / max(total_bars, 1)
    avg_entry_cost_bps = float(np.mean(entry_cost_bps_samples)) if entry_cost_bps_samples else float(2.0 * OPTION_SPREAD_BPS)
    avg_entry_quality = float(np.mean(entry_quality_samples)) if entry_quality_samples else 0.0
    cost_realism_coverage = float(entry_cost_known_count) / max(num_trades, 1)
    high_cost_entry_rate = float(high_cost_entries) / max(num_trades, 1)
    low_quality_entry_rate = float(low_quality_entries) / max(num_trades, 1)

    # --- Equity curve (inline-tracked from simulation, dollar-denominated) ---
    equity_arr = np.array(equity_history)
    final_capital = float(equity_arr[-1])
    total_dollar_return = (final_capital - _starting_capital) / _starting_capital

    equity_peak = np.maximum.accumulate(equity_arr)
    equity_dd = (equity_arr - equity_peak) / np.maximum(equity_peak, 1e-10)
    max_equity_dd = float(np.min(equity_dd)) if len(equity_dd) > 0 else 0.0

    # --- Account ruin detection ---
    # Check if equity ever dropped below ruin threshold (fraction of starting capital)
    min_equity = float(np.min(equity_arr))
    ruin_floor = _starting_capital * _sc_ruin_thresh
    hit_ruin = min_equity < ruin_floor
    min_equity_frac = min_equity / _starting_capital  # 0.0 = total wipeout, 1.0 = never lost

    if hit_ruin and _sc_ruin_penalty > 0:
        # How far past ruin did we go? Scale penalty by severity
        # At ruin_thresh (e.g. 25% of capital left), penalty starts
        # At 0% capital, penalty is maximum
        ruin_severity = max(0.0, 1.0 - min_equity_frac / max(_sc_ruin_thresh, 1e-10))
        # ruin_severity: 0.0 = just touched ruin line, 1.0 = total wipeout
        # Apply: score gets hammered — floor at -5.0 for total ruin
        ruin_mult = max(0.0, 1.0 - _sc_ruin_penalty * ruin_severity)
        if ruin_mult < 0.05:
            # Near-total or total ruin → hard negative score
            score = min(score, -5.0)
        else:
            score *= ruin_mult

    if num_trades > 1:
        equity_returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], 1e-10)
        eq_mean = float(np.mean(equity_returns))
        eq_std = float(np.std(equity_returns, ddof=1))
        equity_sharpe = (eq_mean / max(eq_std, 1e-10)) * math.sqrt(max(trades_per_year, 1))
    else:
        equity_sharpe = 0.0

    # --- worst_chunk_pf: split trades into 5 date-chunks, report min PF ---
    worst_chunk_pf = 1.0  # default if not enough trades
    chunk_details = []
    if num_trades >= 5:
        trade_dates = [d.get('date', '') for d in trade_details]
        unique_dates = sorted(set(trade_dates))
        n_chunks = min(5, len(unique_dates))
        if n_chunks >= 2:
            chunk_size = max(1, len(unique_dates) // n_chunks)
            chunk_pfs = []
            for ci in range(n_chunks):
                start_i = ci * chunk_size
                end_i = start_i + chunk_size if ci < n_chunks - 1 else len(unique_dates)
                chunk_dates = set(unique_dates[start_i:end_i])
                chunk_trades = [d for d in trade_details if d.get('date', '') in chunk_dates]
                if not chunk_trades:
                    continue
                chunk_wins = sum(d['pnl_pct'] for d in chunk_trades if d['pnl_pct'] > 0)
                chunk_losses = abs(sum(d['pnl_pct'] for d in chunk_trades if d['pnl_pct'] <= 0))
                if chunk_losses > 0:
                    chunk_pfs.append(chunk_wins / chunk_losses)
                elif chunk_wins > 0:
                    chunk_pfs.append(100.0)  # all winners
                # else: 0 wins 0 losses in chunk, skip
            if chunk_pfs:
                worst_chunk_pf = min(chunk_pfs)
            # Build per-chunk detail for regime analysis
            chunk_details = []
            for ci in range(n_chunks):
                start_i = ci * chunk_size
                end_i = start_i + chunk_size if ci < n_chunks - 1 else len(unique_dates)
                chunk_dates_set = set(unique_dates[start_i:end_i])
                chunk_trades_ci = [d for d in trade_details if d.get('date', '') in chunk_dates_set]
                if not chunk_trades_ci:
                    continue
                c_wins = sum(d['pnl_pct'] for d in chunk_trades_ci if d['pnl_pct'] > 0)
                c_losses = abs(sum(d['pnl_pct'] for d in chunk_trades_ci if d['pnl_pct'] <= 0))
                c_pf = c_wins / c_losses if c_losses > 0 else (100.0 if c_wins > 0 else 0.0)
                c_wr = sum(1 for d in chunk_trades_ci if d['pnl_pct'] > 0) / len(chunk_trades_ci)
                chunk_details.append({
                    'chunk': ci + 1,
                    'dates': f"{unique_dates[start_i]}..{unique_dates[min(end_i - 1, len(unique_dates) - 1)]}",
                    'trades': len(chunk_trades_ci),
                    'profit_factor': round(c_pf, 2),
                    'win_rate': round(c_wr, 3),
                })

    # --- direction_collapse_pct: fraction choosing single most common action ---
    direction_collapse_pct = 0.0
    if num_trades >= 1:
        direction_counts: dict[str, int] = {}
        for d in trade_details:
            act = d.get('direction', 'UNKNOWN')
            direction_counts[act] = direction_counts.get(act, 0) + 1
        max_count = max(direction_counts.values())
        direction_collapse_pct = float(max_count) / num_trades

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
        'avg_entry_cost_bps': round(float(avg_entry_cost_bps), 4),
        'avg_entry_quality': round(float(avg_entry_quality), 4),
        'cost_realism_coverage': round(float(cost_realism_coverage), 4),
        'high_cost_entry_rate': round(float(high_cost_entry_rate), 4),
        'low_quality_entry_rate': round(float(low_quality_entry_rate), 4),
        'actionable_bar_rate': round(float(actionable_bar_rate), 4),
        'risk_off_bar_rate': round(float(risk_off_bar_rate), 4),
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
        'worst_chunk_pf': round(float(worst_chunk_pf), 4),
        'direction_collapse_pct': round(float(direction_collapse_pct), 4),
        'rr_ratio': round(float(rr_ratio), 4),
        'avg_hold_bars': round(float(avg_hold_bars), 2),
        'model_exit_rate': round(float(model_exit_count) / max(num_trades, 1), 4),
        'hit_ruin': bool(hit_ruin),
        'min_equity_frac': round(float(min_equity_frac), 4),
        'avg_risk_fraction': round(float(avg_risk_fraction), 4),
        'max_risk_fraction': round(float(max_risk_fraction), 4),
        'trades_blocked_by_balance': int(trades_blocked_by_balance),
        'avg_n_contracts': round(float(np.mean([t['n_contracts'] for t in trade_details])) if trade_details else 1.0, 2),
        'chunk_details': chunk_details if num_trades >= 5 else [],
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
        'avg_entry_cost_bps': float(2.0 * OPTION_SPREAD_BPS),
        'avg_entry_quality': 0.0,
        'cost_realism_coverage': 0.0,
        'high_cost_entry_rate': 1.0,
        'low_quality_entry_rate': 1.0,
        'actionable_bar_rate': 0.0,
        'risk_off_bar_rate': 0.0,
        'num_val_bars': num_val_bars, 'num_val_days': num_val_days,
        'total_return': 0.0,
        'final_capital': STARTING_CAPITAL,
        'equity_sharpe': 0.0,
        'max_equity_dd': 0.0,
        'total_dollar_return': 0.0,
        'worst_chunk_pf': 1.0,
        'direction_collapse_pct': 0.0,
        'rr_ratio': 0.0,
        'avg_hold_bars': 0.0,
        'model_exit_rate': 0.0,
        'hit_ruin': False,
        'min_equity_frac': 1.0,
        'avg_risk_fraction': 0.0,
        'max_risk_fraction': 0.0,
        'trades_blocked_by_balance': 0,
        'chunk_details': [],
    }


# ---------------------------------------------------------------------------
# Secondary metric: evaluate_sharpe (strict two-head contract)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sharpe(model, data, lookback, device, batch_size=1024,
                    confidence_threshold=0.0):
    """Walk-forward Sharpe on validation set (secondary continuous metric)."""
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
        if not isinstance(out, tuple) or len(out) < 2:
            raise ValueError(
                "evaluate_sharpe requires model output tuple with at least "
                "(gate_logits, dir_logits). Got: " + str(type(out))
            )
        gate_logits, dir_logits = out[0], out[1]
        if gate_logits.ndim != 2 or gate_logits.shape[-1] != 2:
            raise ValueError(
                f"Invalid gate head shape: expected (batch, 2), got {tuple(gate_logits.shape)}"
            )
        if dir_logits.ndim != 2 or dir_logits.shape[-1] != 6:
            raise ValueError(
                f"Invalid direction head shape: expected (batch, 6), got {tuple(dir_logits.shape)}"
            )

        gate_probs = torch.softmax(gate_logits, dim=-1)   # [no_trade, trade]
        dir_probs = torch.softmax(dir_logits, dim=-1)     # [call_atm..put_otm10]
        trade_prob = gate_probs[:, 1]
        call_prob = dir_probs[:, :3].sum(dim=-1)
        put_prob = dir_probs[:, 3:].sum(dim=-1)
        pos = trade_prob * (call_prob - put_prob)
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
    # REMOVED: --skip-options, --skip-chain, --skip-vix
    # Hard-fail policy: all data sources are REQUIRED. No fallbacks.
    parser.add_argument("--use-spx", action="store_true", default=True,
                        help="Use real SPX index prices from IBKR (default: True, volume still from SPY ETF)")
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

    # --- Download SPY (volume source) — incremental ---
    cache_path = os.path.join(DATA_DIR, "spy_1min.pkl")
    if args.skip_download:
        if not os.path.exists(cache_path):
            old_cache = os.path.join(DATA_DIR, "spy_5min.pkl")
            if os.path.exists(old_cache):
                cache_path = old_cache
                print("  (using old 5-min SPY cache — rebuild for 1-min)")
            else:
                print(f"ERROR: Cache not found: {cache_path}")
                sys.exit(1)
        print(f"Loading cached SPY data (--skip-download)...")
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
        print(f"  {len(df)} bars, {df['date'].nunique()} days")
    else:
        spy_download_fn = download_spy_bars_ibkr if args.spy_source == "ibkr" else download_spy_bars
        df = _incremental_update(cache_path, spy_download_fn, args.start, args.end)
        if df is None or len(df) == 0:
            print("ERROR: No SPY bars available")
            sys.exit(1)
    print()

    # --- Filter to 0DTE days only ---
    # Before May 11, 2022: Mon/Wed/Fri only. After: daily.
    all_days_before = df['date'].nunique()
    df = df[df['date'].apply(is_0dte_day)].copy()
    df = df.reset_index(drop=True)
    skipped = all_days_before - df['date'].nunique()
    print(f"0DTE filter: {df['date'].nunique()} valid days ({skipped} non-0DTE days removed)")
    print()

    # --- Options (ATM + OTM) from Polygon flat files (S3) --- REQUIRED (no fallback)
    options_data = None
    chain_data = None
    if not args.skip_download:
        print(f"Downloading SPXW options from flat files (all {df['date'].nunique()} days)...")
        prefetch_spxw_from_flatfiles(df, api_cutoff=None)  # no cutoff — flat files for ALL days
        print()

    # Load ATM option caches — REQUIRED
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
    if not options_data:
        raise RuntimeError("Options data is REQUIRED — no ATM option data loaded")
    print()

    # Load OTM chain caches — REQUIRED
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
    if not chain_data:
        raise RuntimeError("OTM chain data is REQUIRED — no chain data loaded")
    if chain_data:
        print(f"  OTM chain: {len(chain_data)} bar entries")
        print()

    # --- VIX (via IBKR) — incremental --- REQUIRED (no fallback)
    if not args.skip_download:
        print("Cooldown before VIX download (IBKR rate limiting — 60s)...")
        time.sleep(60)
    vix_data = None
    vix_cache = os.path.join(DATA_DIR, "vix_1min.pkl")
    if args.skip_download:
        if os.path.exists(vix_cache):
            print("Loading cached VIX data (--skip-download)...")
            with open(vix_cache, 'rb') as f:
                _vix_raw = pickle.load(f)
            if isinstance(_vix_raw, dict):
                vix_data = _vix_raw
            else:
                vix_data = _vix_df_to_dict(_vix_raw)
        else:
            raise RuntimeError(f"VIX cache not found: {vix_cache} — VIX data is REQUIRED (no fallback)")
    else:
        # Migrate legacy dict cache → DataFrame for incremental support
        if os.path.exists(vix_cache):
            with open(vix_cache, 'rb') as f:
                _check = pickle.load(f)
            if isinstance(_check, dict):
                print("  VIX cache is legacy dict format — re-downloading for incremental support")
                os.remove(vix_cache)
        vix_df = _incremental_update(vix_cache, download_vix_bars, args.start, args.end)
        if vix_df is not None and len(vix_df) > 0:
            vix_data = _vix_df_to_dict(vix_df)
        else:
            raise RuntimeError("VIX download failed — VIX data is REQUIRED (no fallback)")
        if vix_data:
            print(f"  VIX data: {len(vix_data)} bars")
        print()

    # --- SPX price replacement (optional) — incremental ---
    if args.use_spx:
        spx_cache = os.path.join(DATA_DIR, "spx_1min.pkl")
        if args.skip_download:
            if os.path.exists(spx_cache):
                print("Loading cached SPX data (--skip-download)...")
                with open(spx_cache, 'rb') as f:
                    spx_df = pickle.load(f)
            else:
                spx_df = None
        else:
            spx_df = _incremental_update(spx_cache, download_spx_bars, args.start, args.end)

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
    raw_features = features.copy()
    valid_count = int(np.sum(valid))
    opt_count = int(np.sum(~np.isnan(option_prices['atm_call']))) if option_prices else 0
    pnl_count = int(np.sum(~np.isnan(option_prices['call_pnl']))) if option_prices else 0
    exit_count = int(np.sum(option_prices['exit_call_label'] == 1.0)) if option_prices else 0
    otm_count = int(np.sum(~np.isnan(option_prices['otm5_call']))) if option_prices else 0
    otm_deep_count = int(np.sum(~np.isnan(option_prices.get('otm20_call', np.array([]))))) if option_prices else 0
    actionable_count = int(np.sum(option_prices.get('actionable_mask', np.zeros(len(df))) > 0.5)) if option_prices else 0
    print(f"  Valid bars: {valid_count}/{len(df)} ({100*valid_count/len(df):.0f}%)")
    print(f"  Bars with option prices: {opt_count}/{len(df)} ({100*opt_count/len(df):.0f}%)")
    print(f"  Bars with OTM prices: {otm_count}/{len(df)} ({100*otm_count/len(df):.0f}%)")
    print(f"  Bars with deep OTM (+/-20) prices: {otm_deep_count}/{len(df)} ({100*otm_deep_count/len(df):.0f}%)")
    print(f"  Bars with option P&L: {pnl_count}/{len(df)} ({100*pnl_count/len(df):.0f}%)")
    print(f"  Bars flagged actionable (quality/risk mask): {actionable_count}/{len(df)} ({100*actionable_count/len(df):.0f}%)")
    print(f"  Bars with EXIT=1 (call): {exit_count}")
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # --- Normalize ---
    print("Normalizing (per-day mean, global std — anti-fingerprint)...")
    t0 = time.time()
    features = normalize_features(features, valid, dates=dates)
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # --- Tensors ---
    print("Preparing tensors...")
    data = prepare_tensors(
        features,
        targets,
        dates,
        valid,
        option_prices,
        timestamps,
        raw_features=raw_features,
    )
    print()

    print(f"Done! Features: {NUM_FEATURES}, Actions: {NUM_ACTIONS}")
    print("Run: python3 train.py")
