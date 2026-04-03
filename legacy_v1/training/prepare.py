"""
Autoresearch-trading v18: data prep for SPX 0DTE options trading model.

Data sources:
  - SPX prices:   Real SPX index via IBKR
  - SPY volume:   SPY ETF via IBKR (SPX index has no volume)
  - SPXW options: Polygon flat files (S3 bulk data, 4-year rolling window)
  - VIX:          Real CBOE VIX index via IBKR

Date range: March 14, 2022 to present (~4 years).
Bar resolution: 1-minute (390 bars/day RTH).

Computes 39 features and prediction labels for train.py.

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
LOOKBACK_WINDOW   = 60         # rolling window for vol/volume stats (~60 min, v11: reduced from 100 to unlock morning bars)
MIN_TRADES        = 5          # minimum trades for valid evaluation
ANNUAL_TRADING_BARS = 252 * BARS_PER_DAY
ANNUAL_TRADING_HOURS = ANNUAL_TRADING_BARS  # compat alias

# 0DTE option trade simulation parameters
OPTION_SPREAD_BPS    = 150     # bid-ask spread on 0DTE ATM in bps of premium (150 bps one-way = 3% round-trip; realistic for ATM SPX 0DTE)
STOP_LOSS_PCT        = 0.30    # legacy constant — kept for backward compat; active code uses compute_dynamic_stop()
DYNAMIC_STOP_BASE    = float(os.environ.get("DYNAMIC_STOP_BASE", 0.45))
DYNAMIC_STOP_MIN     = 0.15    # minimum stop-loss (floor)
DYNAMIC_STOP_MAX     = 0.60    # maximum stop-loss (ceiling)
MAX_HOLD_BARS        = BARS_PER_DAY  # hold until stop/profit/EOD (0DTE closes at EOD)
STOP_COOLDOWN_BARS   = 5       # 5-bar (5-min) cooldown after stop loss before re-entry

# Position state tanh scaling — shared across train.py, prepare.py, decision.py, replay.py
# Agents can tune via env var (e.g. PNL_TANH_SCALE=3.0). All files import from here.
PNL_TANH_SCALE       = float(os.environ.get("PNL_TANH_SCALE", 5.0))
BEST_PNL_TANH_SCALE  = float(os.environ.get("BEST_PNL_TANH_SCALE", 2.0))
NO_TRADE_BEFORE_BAR  = 30      # first 30 bars (9:30-9:59) are hard no-trade
NO_TRADE_LUNCH_START = 60      # bar 60 = 10:30 AM (start of lunch chop suppression)
NO_TRADE_LUNCH_END   = 240     # bar 240 = 13:30 PM (end of lunch chop suppression)
MIN_HOLD_BARS        = 2       # minimum bars to hold before model_exit is allowed
EXIT_GATE_THRESHOLD  = 0.60    # gate NO_TRADE prob must exceed this to exit (argmax = 0.50, higher = more patient)
MAX_TRADE_RETURN     = 5.0     # cap individual trade P&L at 500% (allow large winners with learned exits)
STARTING_CAPITAL     = 10_000.0  # starting account balance ($10k paper trading account)
POSITION_RISK_TARGET = 0.05     # target 5% of account per trade; scales whole contracts with account growth
SPX_MULTIPLIER       = 100      # SPX option contract multiplier (premium * 100 = cost)
BAR_SIZE_MINUTES     = 1           # 1-minute bar resolution
SPX_MULTIPLIER       = 100     # option multiplier

# Full OTM ladder: ±5/±10/±15/±20/±25/±30 (Pickles trades up to +30 OTM)
OTM_STRIKE_STEPS      = (5, 10, 15, 20, 25, 30)
SIDE_ACTION_ORDER     = (
    "call_atm",
    "call_otm5",
    "call_otm10",
    "call_otm15",
    "call_otm20",
    "call_otm25",
    "call_otm30",
    "put_atm",
    "put_otm5",
    "put_otm10",
    "put_otm15",
    "put_otm20",
    "put_otm25",
    "put_otm30",
)
SIDE_ACTION_TO_CHAIN_KEY = {
    "call_otm5": "otm5_call",
    "call_otm10": "otm10_call",
    "call_otm15": "otm15_call",
    "call_otm20": "otm20_call",
    "call_otm25": "otm25_call",
    "call_otm30": "otm30_call",
    "put_otm5": "otm5_put",
    "put_otm10": "otm10_put",
    "put_otm15": "otm15_put",
    "put_otm20": "otm20_put",
    "put_otm25": "otm25_put",
    "put_otm30": "otm30_put",
}
SIDE_ACTION_TO_IDX = {name: i for i, name in enumerate(SIDE_ACTION_ORDER)}
SIDE_ALL_LEGS = SIDE_ACTION_ORDER  # v10: all legs are now tradeable (no sidecar)

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
# Feature names (v18: 39 features)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # === Price returns (2) ===
    'ret_6',                # 0: 30-bar (30min) return
    'ret_12',               # 1: 60-bar (1hr) return
    # === Volume (1) ===
    'volume_ratio',         # 2: bar volume / 20-bar SMA
    # === v18: replaced volume_at_price_pctile (redundant w/ va_position) ===
    'gamma_pressure',       # 3: sum(gamma_i * volume_i * sign_i) across chain (GEX proxy)
    # === Volatility (3) ===
    'bar_range',            # 4: (high - low) / close
    'realized_vol',         # 5: 20-bar rolling stdev of returns
    'range_ratio',          # 6: current bar range / 20-bar avg range
    # === VWAP (1) ===
    'vwap_dist',            # 7: (close - session VWAP) / close
    # === Session structure (1) ===
    'session_range_pct',    # 8: full session range so far / close
    # === Key levels (1) ===
    'prev_high_dist',       # 9: distance to previous day high
    # === Trend (3) ===
    'ema_cross',            # 10: (EMA8 - EMA21) / close (momentum)
    'consec_direction',     # 11: consecutive same-direction bars
    'speed_estimate',       # 12: |5-bar return| / realized_vol
    # === v18: replaced inside_bar (30-40% of bars, no signal at 1-min) ===
    'vix_roc',              # 13: VIX 10-bar rate of change (domain: "vol RoC > vol level")
    # === Time (1) ===
    'minutes_to_close',     # 14: log(minutes remaining + 1), normalized
    # === v18: replaced time_cos, then event_day (hardcoded dates unreliable) ===
    'iv_percentile',        # 15: current IV rank vs rolling 60-day history [0,1] (captures event days implicitly)
    # === Options (2) ===
    'atm_iv',               # 16: ATM implied vol (avg of call + put IV)
    'iv_skew',              # 17: put IV - call IV (fear/skew premium)
    # === VIX / Regime (2) ===
    'vix_regime',           # 18: regime bucket: -1=low(<15), -0.33=normal, 0.33=elevated, 1=crisis(>30)
    'vrp',                  # 19: variance risk premium: atm_iv^2 - realized_vol^2
    # === Greeks (3) ===
    'atm_gamma',            # 20: ATM call gamma (delta sensitivity to price)
    'atm_theta_per_bar',    # 21: ATM theta per 1-min bar (time decay per bar)
    'charm_estimate',       # 22: estimated dDelta/dT: delta sensitivity to time decay
    # === Bollinger (1) -- reworked to 5-min in v18 ===
    'bollinger_position',   # 23: 5-min Bollinger band position
    # === Range extras (2) ===
    'rsi_7',                # 24: 5-min RSI (Elder: "7-9 bars for intraday" on 5-min charts)
    'session_range_position',  # 25: (close - session_low) / (session_high - session_low)
    # === Market structure (3) ===
    'poc_dist',             # 26: (close - session POC) / close
    'va_position',          # 27: position within Value Area
    'ib_break',             # 28: IB break state: -1/0/+1
    # === v9 features (3) ===
    'atr_14',               # 29: 14-bar ATR / close
    'bar_delta',            # 30: (close - open) / (high - low) intrabar pressure
    'session_cum_delta',    # 31: cumulative bar deltas
    # === v18: replaced top_of_hour_min (calendar artifact, not market signal) ===
    'option_spread_width',  # 32: ATM option (high-low)/close (bid-ask proxy, cost awareness)
    # === v10 features reworked to 5-min in v18 (3) ===
    'macdh_slope',          # 33: 5-min MACD-H direction (Elder's #1 signal, correct timeframe)
    'force_index_2',        # 34: 5-min Force Index (Elder, correct timeframe)
    # === v18: replaced vol_price_diverg (too noisy at 1-min) ===
    'prev_close_dist',      # 35: (close - prev_day_close) / close
    # === v10 continued ===
    'effort_vs_result',     # 36: 5-min effort vs result (Coulling, correct timeframe)
    'trend_5min',           # 37: 5-min EMA(13) slope (Elder Triple Screen)
    # === v17 promoted (1) ===
    'overnight_gap',        # 38: (day open - prev close) / prev close
]

NUM_FEATURES = len(FEATURE_NAMES)

# Named index lookup for cross-references within compute_features
_FEAT_IDX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

# Features that should NOT be z-score normalized
_NO_NORMALIZE = {
    'minutes_to_close',     # already log-normalized
    'iv_percentile',        # already 0-1
    'vix_regime',           # categorical, already scaled
    'bollinger_position',   # already normalized to [-1, 1]-ish range
    'session_range_position',  # already 0-1
    'rsi_7',                # already 0-1
    'va_position',          # already ~0-1 (can exceed but bounded)
    'ib_break',             # categorical: -1, 0, +1
    'bar_delta',            # already -1 to +1
    'macdh_slope',          # already -1/0/+1
    'effort_vs_result',     # already bounded [-3, 3]
}

# Action labels
# Gate head: NO_TRADE / TRADE → combined with direction head for full action
# Direction head: 14 outputs [CALL_ATM, CALL_OTM5..OTM30, PUT_ATM, PUT_OTM5..OTM30]
ACTION_DO_NOTHING      = 0
ACTION_BUY_CALL_ATM    = 1
ACTION_BUY_CALL_OTM5   = 2
ACTION_BUY_CALL_OTM10  = 3
ACTION_BUY_CALL_OTM15  = 4
ACTION_BUY_CALL_OTM20  = 5
ACTION_BUY_CALL_OTM25  = 6
ACTION_BUY_CALL_OTM30  = 7
ACTION_BUY_PUT_ATM     = 8
ACTION_BUY_PUT_OTM5    = 9
ACTION_BUY_PUT_OTM10   = 10
ACTION_BUY_PUT_OTM15   = 11
ACTION_BUY_PUT_OTM20   = 12
ACTION_BUY_PUT_OTM25   = 13
ACTION_BUY_PUT_OTM30   = 14
ACTION_EXIT            = 15
NUM_ACTIONS            = 16

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

_ib_next_client_id = 20

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

    def _safe_sleep(seconds: float) -> None:
        """Sleep without hanging on dead IB connection."""
        if ib.isConnected():
            ib.sleep(seconds)
        else:
            time.sleep(seconds)

    def _ensure_connected() -> None:
        """Reconnect if IB connection dropped."""
        if not ib.isConnected():
            print(f"  {symbol}: reconnecting to IB Gateway...")
            try:
                ib.connect(
                    os.environ.get("IB_HOST", "127.0.0.1"),
                    int(os.environ.get("IB_PORT", "4001")),
                    clientId=ib.client.clientId,
                    timeout=30,
                )
                ib.qualifyContracts(contract)
                print(f"  {symbol}: reconnected OK")
            except Exception as e:
                print(f"  {symbol}: reconnect failed: {e}")

    while current < end_dt:
        week_end = min(current + dt.timedelta(days=7), end_dt)
        end_str = week_end.strftime('%Y%m%d 16:00:00')
        bars = None
        for attempt, backoff in enumerate([10, 30, 60], 1):
            _ensure_connected()
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
                    _safe_sleep(backoff)
                    bars = None  # reset so we retry
            except Exception as e:
                print(f"  {current.strftime('%Y-%m-%d')}: attempt {attempt}/3 FAIL ({e}), retry in {backoff}s")
                _safe_sleep(backoff)
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
        _safe_sleep(5)  # rate limit — prevent IBKR pacing violations

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
            'otm25_call': ('C', atm_strike + 25),
            'otm30_call': ('C', atm_strike + 30),
            'otm5_put': ('P', atm_strike - 5),
            'otm10_put': ('P', atm_strike - 10),
            'otm15_put': ('P', atm_strike - 15),
            'otm20_put': ('P', atm_strike - 20),
            'otm25_put': ('P', atm_strike - 25),
            'otm30_put': ('P', atm_strike - 30),
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
            'otm5_call', 'otm10_call', 'otm15_call', 'otm20_call', 'otm25_call', 'otm30_call',
            'otm5_put', 'otm10_put', 'otm15_put', 'otm20_put', 'otm25_put', 'otm30_put',
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

        # Define the 14 target contracts (ATM + OTM ±5/±10/±15/±20/±25/±30)
        targets = {
            'atm_call':   ('C', atm_strike),
            'atm_put':    ('P', atm_strike),
            'otm5_call':  ('C', atm_strike + 5),
            'otm10_call': ('C', atm_strike + 10),
            'otm15_call': ('C', atm_strike + 15),
            'otm20_call': ('C', atm_strike + 20),
            'otm25_call': ('C', atm_strike + 25),
            'otm30_call': ('C', atm_strike + 30),
            'otm5_put':   ('P', atm_strike - 5),
            'otm10_put':  ('P', atm_strike - 10),
            'otm15_put':  ('P', atm_strike - 15),
            'otm20_put':  ('P', atm_strike - 20),
            'otm25_put':  ('P', atm_strike - 25),
            'otm30_put':  ('P', atm_strike - 30),
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
            'otm5_call', 'otm10_call', 'otm15_call', 'otm20_call', 'otm25_call', 'otm30_call',
            'otm5_put', 'otm10_put', 'otm15_put', 'otm20_put', 'otm25_put', 'otm30_put',
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
    """Compute 39 features from SPX 1-min bars + SPY volume + SPXW options + VIX.

    Returns: (features_array, targets_array, dates_list, valid_mask, option_prices_dict)
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
    # Deep OTM ladders (v10: all tradeable)
    otm15_call_prices = np.full(N, np.nan, dtype=np.float32)
    otm15_put_prices = np.full(N, np.nan, dtype=np.float32)
    otm20_call_prices = np.full(N, np.nan, dtype=np.float32)
    otm20_put_prices = np.full(N, np.nan, dtype=np.float32)
    otm25_call_prices = np.full(N, np.nan, dtype=np.float32)
    otm25_put_prices = np.full(N, np.nan, dtype=np.float32)
    otm30_call_prices = np.full(N, np.nan, dtype=np.float32)
    otm30_put_prices = np.full(N, np.nan, dtype=np.float32)
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

    # Day index lookup (needed by all per-day pre-computations below)
    unique_dates = sorted(set(dates))
    day_indices = {d: np.where(dates == d)[0] for d in unique_dates}

    # Pre-compute log returns (zeroed at day boundaries to prevent overnight contamination)
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.concatenate([[0.0], log_ret])
    # Zero out the first bar of each day (overnight return is not intraday data)
    _day_starts = set()
    for _d, _didx in day_indices.items():
        if len(_didx) > 0:
            _day_starts.add(_didx[0])
    for _ds in _day_starts:
        log_ret[_ds] = 0.0

    # Pre-compute EMAs (per-day to prevent cross-day state bleeding)
    ema8 = np.zeros(N, dtype=np.float64)
    ema21 = np.zeros(N, dtype=np.float64)
    for _d, _didx in day_indices.items():
        if len(_didx) < 2:
            continue
        _day_close = pd.Series(close[_didx])
        ema8[_didx] = _day_close.ewm(span=40, adjust=False).mean().values
        ema21[_didx] = _day_close.ewm(span=105, adjust=False).mean().values

    # Pre-compute True Range and ATR-14 (day-boundary safe)
    true_range = np.full(N, np.nan, dtype=np.float64)
    true_range[0] = high[0] - low[0]
    for i_tr in range(1, N):
        tr1 = high[i_tr] - low[i_tr]
        if i_tr in _day_starts:
            # First bar of day: use high-low only (no previous day's close)
            true_range[i_tr] = tr1
        else:
            tr2 = abs(high[i_tr] - close[i_tr - 1])
            tr3 = abs(low[i_tr] - close[i_tr - 1])
            true_range[i_tr] = max(tr1, tr2, tr3)
    # Per-day ATR to prevent cross-day rolling
    atr_14 = np.full(N, np.nan, dtype=np.float64)
    for _d, _didx in day_indices.items():
        if len(_didx) < 2:
            continue
        _day_tr = pd.Series(true_range[_didx])
        atr_14[_didx] = _day_tr.rolling(14, min_periods=1).mean().values

    # Pre-compute bar delta: (close - open) / (high - low), clipped to [-1, +1]
    bar_range_raw = high - low
    bar_delta = np.where(bar_range_raw > 1e-8, (close - opn) / bar_range_raw, 0.0)
    bar_delta = np.clip(bar_delta, -1.0, 1.0)

    # Pre-compute session cumulative delta (vectorized per-day cumsum)
    session_cum_delta = np.zeros(N, dtype=np.float64)
    for _day_str, _didx in day_indices.items():
        if len(_didx) > 0:
            session_cum_delta[_didx] = np.cumsum(bar_delta[_didx])

    # === v18: 5-min aggregated features ===
    # All Elder/Coulling indicators reworked to 5-min bars (correct timeframe for intraday).
    # Pattern: aggregate 1-min to 5-min per day, compute indicator, expand back.

    bollinger_5min = np.full(N, np.nan, dtype=np.float64)
    rsi_5min = np.full(N, np.nan, dtype=np.float64)
    macdh_slope = np.zeros(N, dtype=np.float64)
    force_index_2 = np.zeros(N, dtype=np.float64)
    effort_vs_result = np.zeros(N, dtype=np.float64)
    trend_5min = np.zeros(N, dtype=np.float64)
    for _day_str, _didx in day_indices.items():
        if len(_didx) < 5:
            continue
        day_close = close[_didx]
        day_high = high[_didx]
        day_low = low[_didx]
        day_open = opn[_didx]
        day_vol = volume[_didx].astype(np.float64)
        n_5min = len(day_close) // 5
        if n_5min < 2:
            continue

        # Build 5-min OHLCV arrays
        close_5m = np.array([day_close[(j + 1) * 5 - 1] for j in range(n_5min)])
        high_5m = np.array([np.max(day_high[j * 5:(j + 1) * 5]) for j in range(n_5min)])
        low_5m = np.array([np.min(day_low[j * 5:(j + 1) * 5]) for j in range(n_5min)])
        open_5m = np.array([day_open[j * 5] for j in range(n_5min)])
        vol_5m = np.array([np.sum(day_vol[j * 5:(j + 1) * 5]) for j in range(n_5min)])

        def _expand_to_1min(arr_5m):
            """Expand 5-min array back to 1-min indices."""
            for j in range(len(arr_5m)):
                s = j * 5
                e = min((j + 1) * 5, len(_didx))
                yield j, _didx[s:e], arr_5m[j]

        # --- Bollinger (20-bar = 100-min on 5-min chart) ---
        bb_mid = pd.Series(close_5m).rolling(20, min_periods=5).mean().values
        bb_std = pd.Series(close_5m).rolling(20, min_periods=5).std().values
        for j, idx_1m, _ in _expand_to_1min(close_5m):
            if not np.isnan(bb_mid[j]) and bb_std[j] > 1e-8:
                bw = 2.0 * bb_std[j]
                bollinger_5min[idx_1m] = (close_5m[j] - bb_mid[j]) / bw

        # --- RSI 7-period on 5-min ---
        if n_5min >= 8:
            deltas_5m = np.diff(close_5m)
            gains_5m = np.where(deltas_5m > 0, deltas_5m, 0.0)
            losses_5m = np.where(deltas_5m < 0, -deltas_5m, 0.0)
            period = 7
            avg_g = np.mean(gains_5m[:period])
            avg_l = np.mean(losses_5m[:period])
            rsi_arr = np.full(n_5min, np.nan)
            for j in range(period, n_5min):
                if j > period:
                    avg_g = (avg_g * (period - 1) + gains_5m[j - 1]) / period
                    avg_l = (avg_l * (period - 1) + losses_5m[j - 1]) / period
                if avg_l > 1e-10:
                    rs = avg_g / avg_l
                    rsi_arr[j] = rs / (1.0 + rs)
                else:
                    rsi_arr[j] = 1.0
            for j, idx_1m, _ in _expand_to_1min(rsi_arr):
                if not np.isnan(rsi_arr[j]):
                    rsi_5min[idx_1m] = rsi_arr[j]

        # --- MACD-H slope on 5-min ---
        ema12_5m = pd.Series(close_5m).ewm(span=12, adjust=False).mean().values
        ema26_5m = pd.Series(close_5m).ewm(span=26, adjust=False).mean().values
        macd_5m = ema12_5m - ema26_5m
        signal_5m = pd.Series(macd_5m).ewm(span=9, adjust=False).mean().values
        macdh_5m = macd_5m - signal_5m
        slope_macdh = np.zeros(n_5min)
        slope_macdh[1:] = np.sign(macdh_5m[1:] - macdh_5m[:-1])
        for j, idx_1m, _ in _expand_to_1min(slope_macdh):
            macdh_slope[idx_1m] = slope_macdh[j]

        # --- Force Index 2-bar on 5-min ---
        pc_5m = np.zeros(n_5min)
        pc_5m[1:] = close_5m[1:] - close_5m[:-1]
        force_raw_5m = vol_5m * pc_5m
        force_ema2_5m = pd.Series(force_raw_5m).ewm(span=2, adjust=False).mean().values
        fabs_std_5m = pd.Series(np.abs(force_ema2_5m)).rolling(20, min_periods=1).std().values
        fi_5m = np.where(fabs_std_5m > 1e-8, force_ema2_5m / fabs_std_5m, 0.0)
        fi_5m = np.clip(fi_5m, -5.0, 5.0)
        for j, idx_1m, _ in _expand_to_1min(fi_5m):
            force_index_2[idx_1m] = fi_5m[j]

        # --- Effort vs Result on 5-min ---
        body_5m = np.abs(close_5m - open_5m)
        avg_body_5m = pd.Series(body_5m).rolling(20, min_periods=1).mean().values
        avg_vol_5m = pd.Series(vol_5m).rolling(20, min_periods=1).mean().values
        br_5m = np.where(avg_body_5m > 1e-8, body_5m / avg_body_5m, 1.0)
        vr_5m = np.where(avg_vol_5m > 1e-8, vol_5m / avg_vol_5m, 1.0)
        evr_5m = np.where(vr_5m > 0.1, br_5m / vr_5m, 0.0)
        evr_5m = np.clip(evr_5m, -3.0, 3.0)
        for j, idx_1m, _ in _expand_to_1min(evr_5m):
            effort_vs_result[idx_1m] = evr_5m[j]

        # --- Trend 5-min (EMA-13 slope) ---
        ema13_5m = pd.Series(close_5m).ewm(span=13, adjust=False).mean().values
        slope_5m = np.zeros(n_5min, dtype=np.float64)
        slope_5m[1:] = (ema13_5m[1:] - ema13_5m[:-1]) / np.maximum(close_5m[1:], 1.0)
        for j in range(n_5min):
            start_bar = j * 5
            end_bar = min((j + 1) * 5, len(_didx))
            trend_5min[_didx[start_bar:end_bar]] = slope_5m[j]

    trend_5min = np.clip(trend_5min, -0.01, 0.01)

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
    # Pre-compute iv_percentile cache (rolling 60-day IV history per day)
    # -----------------------------------------------------------------------
    _iv_history_by_day = {}
    _daily_close_iv = {}
    for _day in unique_dates:
        _didx = day_indices[_day]
        for _bi in reversed(_didx):
            _okey = (_day, int(df.iloc[_bi]['timestamp']))
            _odata = options_data.get(_okey) if options_data else None
            if _odata is not None and not np.isnan(_odata.get('call_close', np.nan)):
                _spx = close[_bi]
                _K = _odata['strike']
                _min_rem = max(BARS_PER_DAY - (_bi - _didx[0]), 1)
                _T = _min_rem / (252.0 * 390.0)
                _civ = _bs_iv(_odata['call_close'], _spx, _K, _T, 0.05, is_call=True)
                if np.isfinite(_civ):
                    _daily_close_iv[_day] = _civ
                break
    _day_list = list(unique_dates)
    for _di, _day in enumerate(_day_list):
        _start = max(0, _di - 60)
        _hist = [_daily_close_iv[_day_list[j]] for j in range(_start, _di) if _day_list[j] in _daily_close_iv]
        if _hist:
            _iv_history_by_day[_day] = _hist

    # -----------------------------------------------------------------------
    # Main feature loop (39 features)
    # -----------------------------------------------------------------------
    for i in range(N):
        fi = 0
        day = dates[i]
        c = close[i]

        # --- Pre-compute lookups needed by multiple feature blocks ---
        didx = day_indices[day]
        day_pos = np.searchsorted(didx, i)
        bar_dt = df.iloc[i]['datetime']
        bar_time = bar_dt.time()
        minutes_into = (bar_time.hour * 60 + bar_time.minute) - (9 * 60 + 30)
        total_session = 390
        minutes_remaining = max(total_session - minutes_into, 0)
        session_progress = minutes_into / total_session

        opt_key = (dates[i], int(df.iloc[i]['timestamp']))
        opt = options_data.get(opt_key) if options_data else None
        ts_ms = int(df.iloc[i]['timestamp'])
        vix_bar = vix_data.get(ts_ms) if vix_data else None
        chain_bar = chain_data.get(opt_key) if chain_data else None

        # === Returns (2): ret_6, ret_12 ===
        for lag in [30, 60]:
            if i >= lag and dates[i - lag] == day:
                feat[i, fi] = (c / close[i - lag]) - 1.0
            fi += 1

        # === Volume (2) — pruned volume_zscore ===
        if i >= LOOKBACK_WINDOW:
            vol_window = volume[i - LOOKBACK_WINDOW:i]
            vol_mean = np.mean(vol_window)
            feat[i, fi] = volume[i] / max(vol_mean, 1.0)
        else:
            # v11 fallback: use available history or neutral value
            if i > 0 and dates[i - 1] == day:
                vol_window = volume[max(0, i - LOOKBACK_WINDOW):i]
                vol_mean = np.mean(vol_window) if len(vol_window) > 0 else 1.0
                feat[i, fi] = volume[i] / max(vol_mean, 1.0)
            else:
                feat[i, fi] = 1.0  # neutral: current volume = average
        fi += 1

        # gamma_pressure: sum(gamma_i * volume_i * sign_i) across OTM chain (GEX proxy)
        _T_gp = minutes_remaining / (252.0 * 390.0)
        if chain_bar is not None and opt is not None and _T_gp > 1e-10:
            gp = 0.0
            gp_valid = False
            spx_gp = c
            r_gp = 0.05
            for _step in OTM_STRIKE_STEPS:
                for _side, _sign in [('call', 1.0), ('put', -1.0)]:
                    _k_key = f'otm{_step}_{_side}'
                    _px = _safe_float(chain_bar.get(f'{_k_key}_close', np.nan))
                    _vol = _safe_float(chain_bar.get(f'{_k_key}_volume', 0.0), default=0.0)
                    _strike = _safe_float(chain_bar.get(f'{_k_key}_strike', np.nan))
                    if np.isfinite(_px) and np.isfinite(_strike) and _px > 0 and _vol > 0:
                        _iv = _bs_iv(_px, spx_gp, _strike, _T_gp, r_gp, is_call=(_side == 'call'))
                        if np.isfinite(_iv) and _iv > 0:
                            _, _gamma, _, _ = _bs_greeks(spx_gp, _strike, _T_gp, r_gp, _iv)
                            if np.isfinite(_gamma):
                                gp += _gamma * _vol * _sign
                                gp_valid = True
            if gp_valid:
                feat[i, fi] = gp
        fi += 1

        # === Volatility (3) ===
        bar_range = (high[i] - low[i]) / max(c, 1.0)
        feat[i, fi] = bar_range
        fi += 1

        # realized_vol: same-day lookback only (no cross-day contamination)
        _day_start_idx = didx[0]
        _same_day_lookback = min(LOOKBACK_WINDOW, i - _day_start_idx)
        if _same_day_lookback >= 5:
            feat[i, fi] = np.std(log_ret[i - _same_day_lookback:i])
        else:
            feat[i, fi] = 0.0
        fi += 1

        # range_ratio: same-day lookback only
        if _same_day_lookback >= 5:
            _rb = i - _same_day_lookback
            ranges = (high[_rb:i] - low[_rb:i]) / np.maximum(close[_rb:i], 1.0)
            feat[i, fi] = bar_range / max(np.mean(ranges), 1e-8)
        else:
            feat[i, fi] = 1.0
        fi += 1

        # === VWAP (1) — pruned vwap_slope ===
        vw_data = vwap_cache.get(i)
        if vw_data is not None:
            vw, u1, l1, u2, l2 = vw_data
            feat[i, fi] = (c - vw) / max(c, 1.0)
        fi += 1

        # === Session structure (1) — pruned ib_width ===
        session_so_far = didx[:day_pos + 1]
        session_high = np.max(high[session_so_far])
        session_low = np.min(low[session_so_far])
        feat[i, fi] = (session_high - session_low) / max(c, 1.0)  # session_range_pct
        fi += 1

        # === Key levels (1) — pruned prev_low_dist ===
        if day in prev_day_high:
            feat[i, fi] = (c - prev_day_high[day]) / max(c, 1.0)
        fi += 1

        # === Trend (3): ema_cross, consec_direction, speed_estimate ===
        feat[i, fi] = (ema8[i] - ema21[i]) / max(c, 1.0)  # ema_cross
        fi += 1

        # consec_direction (same-day only)
        if i > _day_start_idx:
            consec = 0
            direction = 1 if close[i] >= close[i - 1] else -1
            for j_c in range(i, max(i - 20, _day_start_idx) - 1, -1):
                if j_c <= _day_start_idx:
                    break
                bar_dir = 1 if close[j_c] >= close[j_c - 1] else -1
                if bar_dir == direction:
                    consec += 1
                else:
                    break
            feat[i, fi] = direction * min(consec, 10) / 10.0
        fi += 1

        # speed_estimate (same-day only)
        if i - _day_start_idx >= 5:
            ret5 = abs((c / close[i - 5]) - 1.0)
            rv = feat[i, _FEAT_IDX['realized_vol']]
            if not np.isnan(rv) and rv > 1e-8:
                feat[i, fi] = ret5 / rv
        fi += 1

        # vix_roc: VIX 10-bar rate of change (domain: "vol RoC > vol level")
        if vix_bar is not None and not np.isnan(vix_bar['vix_close']):
            vix_now = vix_bar['vix_close']
            vix_prev_val = np.nan
            if i >= 10 and dates[i - 10] == day:
                ts_prev = int(df.iloc[i - 10]['timestamp'])
                vix_prev_bar = vix_data.get(ts_prev) if vix_data else None
                if vix_prev_bar is not None:
                    vix_prev_val = vix_prev_bar['vix_close']
            if not np.isnan(vix_prev_val) and vix_prev_val > 0:
                feat[i, fi] = (vix_now - vix_prev_val) / vix_prev_val
        fi += 1

        # === Time (2): minutes_to_close, iv_percentile ===
        # (bar_dt, bar_time, minutes_remaining, session_progress computed at loop top)
        spx_for_remap = c  # SPX-scale required (--use-spx default)
        dyn_atm = round(spx_for_remap / 5.0) * 5.0
        dynamic_atm_strikes[i] = dyn_atm
        for j, step in enumerate(OTM_STRIKE_STEPS):
            remap_call_strikes[i, j] = dyn_atm + step
            remap_put_strikes[i, j] = dyn_atm - step

        feat[i, fi] = np.log1p(minutes_remaining) / np.log1p(total_session)
        fi += 1
        # iv_percentile: current IV rank vs rolling 60-day history [0,1]
        # Captures elevated-IV days (FOMC, CPI, NFP) implicitly without hardcoded dates
        # Compute IV directly from option data (atm_iv feature not yet filled at this slot)
        _iv_pctile_val = np.nan
        if opt is not None and not np.isnan(opt.get('call_close', np.nan)):
            _T_ivp = minutes_remaining / (252.0 * 390.0)
            if _T_ivp > 1e-10:
                _cur_iv = _bs_iv(opt['call_close'], c, opt['strike'], _T_ivp, 0.05, is_call=True)
                if np.isfinite(_cur_iv) and day in _iv_history_by_day:
                    hist = _iv_history_by_day[day]
                    if len(hist) >= 5:
                        _iv_pctile_val = float(np.sum(np.array(hist) <= _cur_iv)) / len(hist)
        feat[i, fi] = _iv_pctile_val
        fi += 1

        # === Options (2): atm_iv, iv_skew ===
        # (opt_key, opt computed at loop top)
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
        # (ts_ms, vix_bar computed at loop top)
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
        # (chain_bar computed at loop top)
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
            otm25c = chain_close_map.get("otm25_call", np.nan)
            otm25p = chain_close_map.get("otm25_put", np.nan)
            otm30c = chain_close_map.get("otm30_call", np.nan)
            otm30p = chain_close_map.get("otm30_put", np.nan)

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
            if not np.isnan(otm25c):
                otm25_call_prices[i] = otm25c
            if not np.isnan(otm25p):
                otm25_put_prices[i] = otm25p
            if not np.isnan(otm30c):
                otm30_call_prices[i] = otm30c
            if not np.isnan(otm30p):
                otm30_put_prices[i] = otm30p

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

        # === Bollinger (1): 5-min bollinger_position (pre-computed) ===
        if not np.isnan(bollinger_5min[i]):
            feat[i, fi] = bollinger_5min[i]
        fi += 1

        # === Range extras (2): rsi_7 (5-min), session_range_position ===
        # rsi_7 on 5-min bars (pre-computed)
        if not np.isnan(rsi_5min[i]):
            feat[i, fi] = rsi_5min[i]
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

        # pruned vwap_band_sigma (0.96 corr w/ volume_at_price_pctile)

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

        # pruned theta_pressure (-0.98 corr w/ minutes_to_close)

        # === v9 features (4) — pruned econ_calendar ===

        # atr_14: 14-bar Average True Range / close (volatility context for stops)
        if not np.isnan(atr_14[i]):
            feat[i, fi] = atr_14[i] / max(c, 1.0)
        fi += 1

        # bar_delta: (close - open) / (high - low) — intrabar buy/sell pressure
        feat[i, fi] = bar_delta[i]
        fi += 1

        # session_cum_delta: cumulative bar deltas since session open
        feat[i, fi] = session_cum_delta[i]
        fi += 1

        # pruned econ_calendar (95.6% zeros)

        # option_spread_width: ATM option (high-low)/close as bid-ask proxy
        if opt is not None:
            call_h_sw = _safe_float(opt.get('call_high', np.nan))
            call_l_sw = _safe_float(opt.get('call_low', np.nan))
            call_c_sw = _safe_float(opt.get('call_close', np.nan))
            if np.isfinite(call_h_sw) and np.isfinite(call_l_sw) and np.isfinite(call_c_sw) and call_c_sw > 0:
                feat[i, fi] = (call_h_sw - call_l_sw) / call_c_sw
        fi += 1

        # === v10 new features (5) ===

        # macdh_slope: MACD-H tick direction (Elder's #1 signal)
        feat[i, fi] = macdh_slope[i]
        fi += 1

        # force_index_2: 2-bar EMA of Force Index, normalized
        feat[i, fi] = force_index_2[i]
        fi += 1

        # prev_close_dist: (close - prev_day_close) / close
        prev_cl_dist = prev_day_close_val.get(day)
        if prev_cl_dist is not None and prev_cl_dist > 0:
            feat[i, fi] = (c - prev_cl_dist) / c
        fi += 1

        # effort_vs_result: body/avg_body / vol/avg_vol anomaly
        feat[i, fi] = effort_vs_result[i]
        fi += 1

        # trend_5min: 5-min aggregated EMA(13) slope (Triple Screen)
        feat[i, fi] = trend_5min[i]
        fi += 1

        # overnight_gap: (day open - prev close) / prev close (promoted from tournament in v17)
        prev_cl = prev_day_close_val.get(day)
        if prev_cl is not None and prev_cl > 0:
            day_open_px = opn[day_indices[day][0]]
            feat[i, fi] = (day_open_px - prev_cl) / prev_cl
        fi += 1

        # (tournament feature system removed in v18)

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
    # OTM P&L targets: EOD/max-hold exits for OTM strikes (14-class dir head)
    # -------------------------------------------------------------------
    otm5_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm5_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm15_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm15_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm20_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm20_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm25_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm25_put_pnl = np.full(N, np.nan, dtype=np.float32)
    otm30_call_pnl = np.full(N, np.nan, dtype=np.float32)
    otm30_put_pnl = np.full(N, np.nan, dtype=np.float32)
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
        (otm25_call_prices, otm25_call_pnl, None, "call_otm25"),
        (otm25_put_prices, otm25_put_pnl, None, "put_otm25"),
        (otm30_call_prices, otm30_call_pnl, None, "call_otm30"),
        (otm30_put_prices, otm30_put_pnl, None, "put_otm30"),
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

    # OTM stopped P&L (default level only)
    otm5_call_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm5_put_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_call_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm10_put_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm15_call_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm15_put_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm20_call_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm20_put_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm25_call_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm25_put_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm30_call_stopped_pnl = np.full(N, np.nan, dtype=np.float32)
    otm30_put_stopped_pnl = np.full(N, np.nan, dtype=np.float32)

    _otm_stopped_legs = [
        (otm5_call_prices, otm5_call_stopped_pnl, "call_otm5"),
        (otm5_put_prices, otm5_put_stopped_pnl, "put_otm5"),
        (otm10_call_prices, otm10_call_stopped_pnl, "call_otm10"),
        (otm10_put_prices, otm10_put_stopped_pnl, "put_otm10"),
        (otm15_call_prices, otm15_call_stopped_pnl, "call_otm15"),
        (otm15_put_prices, otm15_put_stopped_pnl, "put_otm15"),
        (otm20_call_prices, otm20_call_stopped_pnl, "call_otm20"),
        (otm20_put_prices, otm20_put_stopped_pnl, "put_otm20"),
        (otm25_call_prices, otm25_call_stopped_pnl, "call_otm25"),
        (otm25_put_prices, otm25_put_stopped_pnl, "put_otm25"),
        (otm30_call_prices, otm30_call_stopped_pnl, "call_otm30"),
        (otm30_put_prices, otm30_put_stopped_pnl, "put_otm30"),
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

    # Exit label parameters (v7: hindsight peak detection for profitable entries only)
    _exit_lookback = int(os.environ.get("EXIT_LOOKBACK", 15))
    _trail_drop_frac = float(os.environ.get("EXIT_TRAIL_DROP", 0.40))
    _trail_min_hwm = float(os.environ.get("EXIT_TRAIL_MIN_HWM", 0.03))
    _stall_bars = int(os.environ.get("EXIT_STALL_BARS", 6))
    _stall_min_pnl = float(os.environ.get("EXIT_STALL_MIN_PNL", 0.02))
    _tp_threshold = float(os.environ.get("EXIT_TP_THRESHOLD", 0.20))
    # Minimum entry P&L to generate exit labels (skip deeply underwater entries)
    # -0.05 means skip entries that are >5% underwater — these are stop-loss territory, not exit signals
    _min_entry_pnl = float(os.environ.get("EXIT_MIN_ENTRY_PNL", -0.05))

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

                # v7: Skip underwater entries — only generate exit labels for profitable trades
                if unrealized_now < _min_entry_pnl:
                    continue

                # Find high-water mark from entry to current bar (backward only)
                hwm = unrealized_now
                for back in range(1, k + 1):
                    past = i - back
                    if past < entry:
                        break
                    pc = atm_call_prices[past]
                    if not np.isnan(pc) and ec > 0:
                        hwm = max(hwm, (pc - ec) / ec - SPREAD_COST_PCT)

                # Signal 1: Trailing stop — P&L dropped significantly from HWM
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

                # Signal 3: Take profit — P&L exceeds threshold
                if unrealized_now >= _tp_threshold:
                    best_call_exit = 1.0

            # --- Put: same causal logic ---
            ep = atm_put_prices[entry]
            cp = atm_put_prices[i]
            if not np.isnan(ep) and not np.isnan(cp) and ep > 0:
                unrealized_now = (cp - ep) / ep - SPREAD_COST_PCT
                has_put_data = True

                # v7: Skip underwater entries
                if unrealized_now < _min_entry_pnl:
                    continue

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

                # Signal 3: Take profit
                if unrealized_now >= _tp_threshold:
                    best_put_exit = 1.0

        if has_call_data:
            exit_call_label[i] = best_call_exit
        if has_put_data:
            exit_put_label[i] = best_put_exit

    # Valid mask: only equity features (0:16) must be non-NaN.
    # Options (16+) depend on SPXW data and may be NaN.
    equity_feat_end = 16  # first 16 features are pure equity (price, volume, vol, vwap, session, levels, trend, micro, time)
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
        'otm25_call': otm25_call_prices,
        'otm25_put': otm25_put_prices,
        'otm30_call': otm30_call_prices,
        'otm30_put': otm30_put_prices,
        'otm5_call_pnl': otm5_call_pnl,
        'otm5_put_pnl': otm5_put_pnl,
        'otm10_call_pnl': otm10_call_pnl,
        'otm10_put_pnl': otm10_put_pnl,
        'otm15_call_pnl': otm15_call_pnl,
        'otm15_put_pnl': otm15_put_pnl,
        'otm20_call_pnl': otm20_call_pnl,
        'otm20_put_pnl': otm20_put_pnl,
        'otm25_call_pnl': otm25_call_pnl,
        'otm25_put_pnl': otm25_put_pnl,
        'otm30_call_pnl': otm30_call_pnl,
        'otm30_put_pnl': otm30_put_pnl,
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
        'otm15_call_stopped_pnl': otm15_call_stopped_pnl,
        'otm15_put_stopped_pnl': otm15_put_stopped_pnl,
        'otm20_call_stopped_pnl': otm20_call_stopped_pnl,
        'otm20_put_stopped_pnl': otm20_put_stopped_pnl,
        'otm25_call_stopped_pnl': otm25_call_stopped_pnl,
        'otm25_put_stopped_pnl': otm25_put_stopped_pnl,
        'otm30_call_stopped_pnl': otm30_call_stopped_pnl,
        'otm30_put_stopped_pnl': otm30_put_stopped_pnl,
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
# v11: Setup Recognition + Regime Classification
# ---------------------------------------------------------------------------

def detect_setups(features: np.ndarray, valid: np.ndarray, dates) -> np.ndarray:
    """Detect structural trading setups from feature values (pre-normalization).

    Returns a boolean array (N,) where True = a recognizable setup is present.
    Operates on RAW (unnormalized) features so thresholds are interpretable.

    Four setups from domain knowledge:
    1. VWAP Pullback — price extended from VWAP, now reverting (mean reversion)
    2. IB Breakout — Initial Balance break with directional follow-through
    3. Magic Time Reversal — 10:00-10:30 ET counter-trend at extremes
    4. Trend Continuation — strong trend with volume confirmation
    """
    N = features.shape[0]
    setup_mask = np.zeros(N, dtype=bool)
    if N == 0:
        return setup_mask

    # Feature indices (raw, pre-normalization)
    idx_vwap_dist = _FEAT_IDX['vwap_dist']           # 7
    idx_bar_delta = _FEAT_IDX['bar_delta']            # 30
    idx_ib_break = _FEAT_IDX['ib_break']              # 28
    idx_consec = _FEAT_IDX['consec_direction']         # 11
    idx_vol_ratio = _FEAT_IDX['volume_ratio']          # 2
    idx_min_close = _FEAT_IDX['minutes_to_close']      # 14
    idx_srp = _FEAT_IDX['session_range_position']      # 25
    idx_trend5 = _FEAT_IDX['trend_5min']               # 37
    idx_speed = _FEAT_IDX['speed_estimate']            # 12

    vwap_dist = features[:, idx_vwap_dist]
    bar_delta = features[:, idx_bar_delta]
    ib_break = features[:, idx_ib_break]
    consec = features[:, idx_consec]
    vol_ratio = features[:, idx_vol_ratio]
    min_close_raw = features[:, idx_min_close]
    srp = features[:, idx_srp]
    trend5 = features[:, idx_trend5]
    speed = features[:, idx_speed]

    # minutes_to_close is stored as log1p(minutes)/log1p(390), invert to get raw minutes
    minutes_remaining = np.expm1(min_close_raw * np.log1p(390))

    # --- Setup 1: VWAP Pullback (mean reversion) ---
    # Price at VWAP ±1.5σ equivalent (vwap_dist > ~0.002 is ~1.5σ for SPX)
    # AND bar delta shows reversal toward VWAP
    extended_up = vwap_dist > 0.002
    extended_down = vwap_dist < -0.002
    reverting_down = bar_delta < -0.2  # bearish bar
    reverting_up = bar_delta > 0.2     # bullish bar
    vwap_pullback = (extended_up & reverting_down) | (extended_down & reverting_up)

    # --- Setup 2: IB Breakout (trend day) ---
    # IB break detected AND directional follow-through AND volume confirmation
    ib_active = np.abs(ib_break) > 0.5
    directional = np.abs(consec) >= 0.2  # ≥2 consecutive bars (scaled /10)
    vol_confirm = vol_ratio > 1.0
    ib_breakout = ib_active & directional & vol_confirm

    # --- Setup 3: Magic Time Reversal (10:00-10:30 ET = 330-360 min to close) ---
    # Counter-trend at session extremes during Magic Time window
    magic_window = (minutes_remaining >= 330) & (minutes_remaining <= 360)
    at_extreme = (srp > 0.85) | (srp < 0.15)
    # Reversal: bar delta opposes the extreme
    reversal_at_high = (srp > 0.85) & (bar_delta < -0.1)
    reversal_at_low = (srp < 0.15) & (bar_delta > 0.1)
    magic_time = magic_window & at_extreme & (reversal_at_high | reversal_at_low)

    # --- Setup 4: Trend Continuation ---
    # Strong trend + speed + volume, not at session extremes
    strong_trend = np.abs(trend5) > 0.3
    has_speed = speed > 0.3
    has_volume = vol_ratio > 0.8
    not_extreme = (srp > 0.2) & (srp < 0.8)
    trend_continuation = strong_trend & has_speed & has_volume & not_extreme

    # Combine all setups
    setup_mask = vwap_pullback | ib_breakout | magic_time | trend_continuation
    # Only count valid bars
    setup_mask = setup_mask & valid

    return setup_mask


def compute_regime_labels(features: np.ndarray, valid: np.ndarray, dates) -> np.ndarray:
    """Classify each bar as TRENDING (True) or CHOP (False).

    Operates on RAW (unnormalized) features.
    TRENDING requires multiple confirming signals — conservative by default.
    """
    N = features.shape[0]
    regime_mask = np.zeros(N, dtype=bool)
    if N == 0:
        return regime_mask

    idx_atr = _FEAT_IDX['atr_14']                     # 29
    idx_consec = _FEAT_IDX['consec_direction']         # 11
    idx_vol_ratio = _FEAT_IDX['volume_ratio']          # 2
    idx_ib_break = _FEAT_IDX['ib_break']               # 28
    idx_trend5 = _FEAT_IDX['trend_5min']               # 37
    idx_speed = _FEAT_IDX['speed_estimate']            # 12

    atr = features[:, idx_atr]
    consec = features[:, idx_consec]
    vol_ratio = features[:, idx_vol_ratio]
    ib_break = features[:, idx_ib_break]
    trend5 = features[:, idx_trend5]
    speed = features[:, idx_speed]

    # ATR expanding: compare to rolling mean
    # Use a simple approach: ATR above its own 30-bar moving average
    atr_expanding = np.zeros(N, dtype=bool)
    date_arr = np.array(dates) if not isinstance(dates, np.ndarray) else dates
    for i in range(N):
        if not valid[i] or np.isnan(atr[i]):
            continue
        lookback_start = max(0, i - 30)
        window_atr = atr[lookback_start:i]
        window_valid = ~np.isnan(window_atr)
        if window_valid.sum() >= 5:
            atr_expanding[i] = atr[i] > np.mean(window_atr[window_valid])

    # Directional consistency: ≥2 consecutive bars
    directional = np.abs(consec) >= 0.2  # consec is scaled /10, so 0.2 = 2 bars

    # Volume confirming
    vol_confirm = vol_ratio > 0.8

    # Catalyst: IB break OR strong 5-min trend
    ib_active = np.abs(ib_break) > 0.5
    strong_trend = np.abs(trend5) > 0.3

    # TRENDING = ATR expanding + directional + volume + catalyst
    regime_mask = atr_expanding & directional & vol_confirm & (ib_active | strong_trend)
    regime_mask = regime_mask & valid

    return regime_mask


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


def _per_day_zscore(features: np.ndarray, valid: np.ndarray, dates: list,
                    walk_forward: bool = True) -> np.ndarray:
    """Per-day mean, expanding-window std normalization (walk-forward).

    Removes day-specific feature fingerprints that cause the model to memorize
    individual dates while preserving regime-level information.

    Walk-forward mode (default): std computed using only data up to each day,
    preventing future information leakage in normalization statistics.
    """
    out = features.copy().astype(np.float64)

    # Build day boundaries from dates
    day_starts = [0] + [i for i in range(1, len(dates)) if dates[i] != dates[i - 1]]
    day_ends = day_starts[1:] + [len(dates)]

    for j in range(out.shape[1]):
        name = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"feature_{j}"
        if name in _NO_NORMALIZE:
            continue

        if not walk_forward:
            # Legacy: global std across all valid bars
            all_vals = out[:, j][valid]
            if len(all_vals) < 10:
                out[:, j] = 0.0
                continue
            global_std = float(np.std(all_vals))
            if global_std < 1e-10:
                out[:, j] = 0.0
                continue
            for ds, de in zip(day_starts, day_ends):
                chunk = out[ds:de, j]
                mask = valid[ds:de]
                day_vals = chunk[mask]
                if len(day_vals) < 3:
                    out[ds:de, j] = 0.0
                else:
                    day_mean = np.mean(day_vals)
                    out[ds:de, j] = (chunk - day_mean) / global_std
        else:
            # Walk-forward: expanding-window std (no future data leakage)
            # Use a minimum of 20 days of data for std estimation
            min_warmup_days = 20
            for day_i, (ds, de) in enumerate(zip(day_starts, day_ends)):
                chunk = out[ds:de, j]
                mask = valid[ds:de]
                day_vals = chunk[mask]
                if len(day_vals) < 3:
                    out[ds:de, j] = 0.0
                    continue
                day_mean = np.mean(day_vals)
                # Expanding std: use all valid data up to and including this day
                expanding_end = ds  # strict walk-forward: exclude current day from std
                expanding_vals = out[:expanding_end, j][valid[:expanding_end]]
                if len(expanding_vals) < 50 or day_i < min_warmup_days:
                    # Not enough history: use day-level std as fallback
                    day_std = float(np.std(day_vals))
                    if day_std < 1e-10:
                        out[ds:de, j] = 0.0
                    else:
                        out[ds:de, j] = (chunk - day_mean) / day_std
                else:
                    expanding_std = float(np.std(expanding_vals))
                    if expanding_std < 1e-10:
                        out[ds:de, j] = 0.0
                    else:
                        out[ds:de, j] = (chunk - day_mean) / expanding_std

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
    # Purge gap: skip 1 day at the boundary to prevent label/feature leakage
    purge_days = 1
    train_dates = set(unique_dates[:split_idx])
    val_dates = set(unique_dates[split_idx + purge_days:])

    train_end_idx = max(i for i, d in enumerate(dates) if d in train_dates)
    val_start_idx = min(i for i, d in enumerate(dates) if d in val_dates)
    val_end_idx = max(i for i, d in enumerate(dates) if d in val_dates)

    print(f"  Train: {len(train_dates)} days (idx 0-{train_end_idx})")
    print(f"  Purge: {purge_days} day(s) at boundary")
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
        '_provenance': {
            'created_at': __import__('datetime').datetime.now().isoformat(),
            'feature_names': list(FEATURE_NAMES),
            'num_features': NUM_FEATURES,
            'num_train_days': len(train_dates),
            'num_val_days': len(val_dates),
            'train_date_range': f"{min(train_dates)}..{max(train_dates)}",
            'val_date_range': f"{min(val_dates)}..{max(val_dates)}",
            'total_bars': len(features),
            'split_ratio': 0.7,
        },
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
        for k in ('otm15_call', 'otm15_put', 'otm20_call', 'otm20_put',
                  'otm25_call', 'otm25_put', 'otm30_call', 'otm30_put'):
            if k in option_prices:
                data[f'{k}_prices'] = torch.tensor(option_prices[k], dtype=torch.float32)
        data['otm5_call_pnl'] = torch.tensor(option_prices['otm5_call_pnl'], dtype=torch.float32)
        data['otm5_put_pnl'] = torch.tensor(option_prices['otm5_put_pnl'], dtype=torch.float32)
        data['otm10_call_pnl'] = torch.tensor(option_prices['otm10_call_pnl'], dtype=torch.float32)
        data['otm10_put_pnl'] = torch.tensor(option_prices['otm10_put_pnl'], dtype=torch.float32)
        for k in (
            'otm15_call_pnl', 'otm15_put_pnl', 'otm20_call_pnl', 'otm20_put_pnl',
            'otm25_call_pnl', 'otm25_put_pnl', 'otm30_call_pnl', 'otm30_put_pnl',
            'otm5_call_pnl_realistic', 'otm5_put_pnl_realistic',
            'otm10_call_pnl_realistic', 'otm10_put_pnl_realistic',
            'call_stopped_pnl', 'put_stopped_pnl',
            'call_stopped_pnl_tight', 'call_stopped_pnl_wide',
            'put_stopped_pnl_tight', 'put_stopped_pnl_wide',
            'otm5_call_stopped_pnl', 'otm5_put_stopped_pnl',
            'otm10_call_stopped_pnl', 'otm10_put_stopped_pnl',
            'otm15_call_stopped_pnl', 'otm15_put_stopped_pnl',
            'otm20_call_stopped_pnl', 'otm20_put_stopped_pnl',
            'otm25_call_stopped_pnl', 'otm25_put_stopped_pnl',
            'otm30_call_stopped_pnl', 'otm30_put_stopped_pnl',
        ):
            if k in option_prices:
                data[k] = torch.tensor(option_prices[k], dtype=torch.float32)
        for k in (
            'action_spread_bps', 'action_quote_age_s', 'action_size',
            'action_quality_score', 'action_slippage_bps', 'action_cost_bps',
            'actionable_mask', 'risk_state_mask', 'supervision_weight',
            'setup_mask', 'regime_mask',
        ):
            if k in option_prices:
                data[k] = torch.tensor(option_prices[k], dtype=torch.float32)
        if 'action_leg_names' in option_prices:
            data['action_leg_names'] = list(option_prices['action_leg_names'])

        # v17: Prediction labels (forward returns + volatility + action targets)
        for k in ('pred_return_15', 'pred_return_30', 'pred_return_60',
                  'pred_volatility_30', 'pred_action_target'):
            if k in option_prices:
                data[k] = torch.tensor(option_prices[k], dtype=torch.float32)

        # v18: Path-quality labels
        for k in ('v18_mfe', 'v18_mae', 'v18_entry_gate',
                  'v18_risk_stop_distance', 'v18_risk_target_distance',
                  'v18_risk_conviction', 'v18_exit_label',
                  'v18_direction_label', 'v18_bar_weight'):
            if k in option_prices:
                data[k] = torch.tensor(option_prices[k], dtype=torch.float32)

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

    # Write SHA256 sidecar for upload verification
    import hashlib as _hashlib
    with open(path, "rb") as _f:
        _content_hash = _hashlib.sha256(_f.read()).hexdigest()
    with open(path + ".sha256", "w") as _f:
        _f.write(_content_hash + "\n")
    print(f"  Saved: {path} ({size_mb:.1f} MB, hash: {_content_hash[:16]})")
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
            _data = torch.load(path, map_location="cpu", weights_only=False)
            # Print provenance for auditability
            _prov = _data.get('_provenance', {})
            if _prov:
                print(f"  Provenance: {_prov.get('num_features', '?')} features, "
                      f"{_prov.get('num_val_days', '?')} val days, "
                      f"val range {_prov.get('val_date_range', '?')}, "
                      f"created {_prov.get('created_at', '?')[:19]}")
            else:
                print(f"  WARNING: data.pt has no provenance metadata (pre-v14 format)")
            return _data
    print(f"Data not found. Searched: {search_paths}")
    print("Run `python3 prepare.py` first.")
    sys.exit(1)


def make_prediction_dataloader(data, lookback, batch_size, split="train", device="cuda"):
    """Simple dataloader for v17 prediction model.

    Yields (x, y):
        x: (batch, lookback, NUM_FEATURES)
        y: (batch, 5) — return_15, return_30, return_60, vol_30, action_target
    """
    features = data['features'].to(device)
    valid_mask = data['valid_mask']

    # Build prediction label tensor
    pred_keys = ['pred_return_15', 'pred_return_30', 'pred_return_60',
                 'pred_volatility_30', 'pred_action_target']
    pred_arrays = []
    for k in pred_keys:
        v = data.get(k)
        if v is None:
            raise KeyError(f"data.pt missing '{k}'. Rebuild with current prepare.py.")
        pred_arrays.append(v)
    pred_labels = torch.stack(pred_arrays, dim=-1).to(device)  # (N, 5)

    if split == "train":
        end = data['train_end_idx'] + 1
        start = max(lookback, 0)
    else:
        end = data['val_end_idx'] + 1
        start = max(lookback, data['val_start_idx'])

    # Build day start index lookup for cross-day prevention
    _dates = data.get('dates', [])
    _day_start_map = {}  # bar_index -> first bar of that day
    if _dates:
        _prev_d = None
        for _bi in range(len(_dates)):
            if _dates[_bi] != _prev_d:
                _cur_day_start = _bi
                _prev_d = _dates[_bi]
            _day_start_map[_bi] = _cur_day_start

    # Find valid indices (same-day lookback, valid prediction labels)
    valid_indices = []
    for i in range(start, end):
        if not valid_mask[i]:
            continue
        if torch.isnan(pred_labels[i, 1]):
            continue
        # Require full lookback within the same day (no cross-day window)
        _ds = _day_start_map.get(i, 0)
        if i - _ds < lookback:
            continue  # not enough same-day bars for full lookback
        window = valid_mask[i - lookback:i]
        if len(window) > 0 and window.sum() >= 0.9 * len(window):
            valid_indices.append(i)

    valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
    n = len(valid_indices)
    assert n > 0, f"No valid samples for split={split}"
    print(f"  Prediction dataloader ({split}): {n} valid bars")

    offsets = torch.arange(-lookback, 0, device=device)

    if split == "train":
        while True:
            perm = torch.randperm(n, device=device)
            for i in range(0, n - batch_size + 1, batch_size):
                idx = valid_indices[perm[i:i + batch_size]]
                window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
                x = features[window_idx]
                y = pred_labels[idx]
                # Replace NaN with 0 in labels (for bars near EOD with missing horizons)
                y = torch.nan_to_num(y, nan=0.0)
                yield x, y
    else:
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            idx = valid_indices[i:end_i]
            window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
            x = features[window_idx]
            y = pred_labels[idx]
            y = torch.nan_to_num(y, nan=0.0)
            yield x, y


def make_v18_dataloader(data, lookback, batch_size, split="train", device="cuda"):
    """Dataloader for v18 5-head model.

    Yields (x, y_pred, y_v18, bar_weight):
        x: (batch, lookback, NUM_FEATURES)
        y_pred: (batch, 5) -- return_15, return_30, return_60, vol_30, action_target (v17 compat)
        y_v18: (batch, 8) -- entry_gate, risk_stop, risk_target, risk_conviction,
                              exit_label, direction_label, mfe, mae
        bar_weight: (batch,) -- time-of-day loss weighting
    """
    features = data['features'].to(device)
    valid_mask = data['valid_mask']

    # v17 prediction labels (still used for return head)
    pred_keys = ['pred_return_15', 'pred_return_30', 'pred_return_60',
                 'pred_volatility_30', 'pred_action_target']
    pred_arrays = []
    for k in pred_keys:
        v = data.get(k)
        if v is None:
            raise KeyError(f"data.pt missing '{k}'. Rebuild with current prepare.py.")
        pred_arrays.append(v)
    pred_labels = torch.stack(pred_arrays, dim=-1).to(device)  # (N, 5)

    # v18 path-quality labels
    v18_keys = ['v18_entry_gate', 'v18_risk_stop_distance', 'v18_risk_target_distance',
                'v18_risk_conviction', 'v18_exit_label', 'v18_direction_label',
                'v18_mfe', 'v18_mae']
    v18_arrays = []
    for k in v18_keys:
        v = data.get(k)
        if v is None:
            raise KeyError(f"data.pt missing '{k}'. Rebuild with current prepare.py (v18).")
        v18_arrays.append(v)
    v18_labels = torch.stack(v18_arrays, dim=-1).to(device)  # (N, 8)

    # bar_weight
    bw = data.get('v18_bar_weight')
    if bw is None:
        raise KeyError("data.pt missing 'v18_bar_weight'. Rebuild with current prepare.py (v18).")
    bar_weight = bw.to(device)  # (N,)

    if split == "train":
        end = data['train_end_idx'] + 1
        start = max(lookback, 0)
    else:
        end = data['val_end_idx'] + 1
        start = max(lookback, data['val_start_idx'])

    # Build day start index lookup for cross-day prevention
    _dates = data.get('dates', [])
    _day_start_map = {}
    if _dates:
        _prev_d = None
        for _bi in range(len(_dates)):
            if _dates[_bi] != _prev_d:
                _cur_day_start = _bi
                _prev_d = _dates[_bi]
            _day_start_map[_bi] = _cur_day_start

    # Find valid indices (same-day lookback, valid v18 labels)
    valid_indices = []
    for i in range(start, end):
        if not valid_mask[i]:
            continue
        # Require valid v18 entry_gate (index 0 in v18_labels)
        if torch.isnan(v18_labels[i, 0]):
            continue
        # Require valid 30-bar return (for return head)
        if torch.isnan(pred_labels[i, 1]):
            continue
        # Require full lookback within the same day
        _ds = _day_start_map.get(i, 0)
        if i - _ds < lookback:
            continue
        window = valid_mask[i - lookback:i]
        if len(window) > 0 and window.sum() >= 0.9 * len(window):
            valid_indices.append(i)

    valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
    n = len(valid_indices)
    assert n > 0, f"No valid v18 samples for split={split}"
    print(f"  v18 dataloader ({split}): {n} valid bars")

    offsets = torch.arange(-lookback, 0, device=device)

    if split == "train":
        while True:
            perm = torch.randperm(n, device=device)
            for i in range(0, n - batch_size + 1, batch_size):
                idx = valid_indices[perm[i:i + batch_size]]
                window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
                x = features[window_idx]
                y_pred = torch.nan_to_num(pred_labels[idx], nan=0.0)
                y_v18 = torch.nan_to_num(v18_labels[idx], nan=0.0)
                bw = torch.nan_to_num(bar_weight[idx], nan=0.0)
                yield x, y_pred, y_v18, bw
    else:
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            idx = valid_indices[i:end_i]
            window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
            x = features[window_idx]
            y_pred = torch.nan_to_num(pred_labels[idx], nan=0.0)
            y_v18 = torch.nan_to_num(v18_labels[idx], nan=0.0)
            bw = torch.nan_to_num(bar_weight[idx], nan=0.0)
            yield x, y_pred, y_v18, bw


# ---------------------------------------------------------------------------
# Prediction label computation (v17)
# ---------------------------------------------------------------------------

PREDICTION_ENTRY_THRESHOLD = 0.002  # trade when abs(return_30) > this

def compute_prediction_labels_from_prices(close_prices, dates, valid):
    """Compute forward return labels and action targets from raw close prices.

    Returns dict with keys: return_15, return_30, return_60, volatility_30, action_target
    All arrays are (N,) float32, NaN where not computable.
    """
    N = len(close_prices)
    labels = {
        'return_15': np.full(N, np.nan, dtype=np.float32),
        'return_30': np.full(N, np.nan, dtype=np.float32),
        'return_60': np.full(N, np.nan, dtype=np.float32),
        'volatility_30': np.full(N, np.nan, dtype=np.float32),
        'action_target': np.full(N, np.nan, dtype=np.float32),
    }

    for i in range(N):
        if not valid[i]:
            continue
        day = dates[i]

        # Forward returns at 15/30/60 bar horizons
        for horizon, key in [(15, 'return_15'), (30, 'return_30'), (60, 'return_60')]:
            end = i + horizon
            if end < N and dates[end] == day and close_prices[i] > 0:
                labels[key][i] = (close_prices[end] - close_prices[i]) / close_prices[i]

        # Realized volatility over next 30 bars
        end_vol = min(i + 30, N)
        if end_vol > i + 5:
            window_prices = close_prices[i:end_vol].copy()
            window_dates = dates[i:end_vol]
            same_day = np.array([d == day for d in window_dates])
            window_prices = window_prices[same_day]
            if len(window_prices) > 5 and not np.any(np.isnan(window_prices)):
                bar_returns = np.diff(window_prices) / window_prices[:-1]
                labels['volatility_30'][i] = np.std(bar_returns).astype(np.float32)

        # Action target: graded signal based on return magnitude
        r30 = labels['return_30'][i]
        if not np.isnan(r30):
            abs_r30 = abs(r30)
            if abs_r30 > PREDICTION_ENTRY_THRESHOLD * 2:
                labels['action_target'][i] = 1.0
            elif abs_r30 > PREDICTION_ENTRY_THRESHOLD:
                labels['action_target'][i] = 0.5
            else:
                labels['action_target'][i] = 0.0

    return labels


# ---------------------------------------------------------------------------
# v18 label computation: path-quality labels (MFE/MAE, entry gate, risk, exit, direction)
# ---------------------------------------------------------------------------

# Direction label classes (6-class: call/put x ATM/OTM5/OTM10)
V18_DIRECTION_CLASSES = [
    'call_atm', 'call_otm5', 'call_otm10',
    'put_atm', 'put_otm5', 'put_otm10',
]
V18_NUM_DIRECTIONS = len(V18_DIRECTION_CLASSES)

# Time-of-day bar_weight for loss weighting
V18_MORNING_WEIGHT = 1.0    # bar 30-60: prime morning window
V18_MIDDAY_WEIGHT = 0.5     # bar 60-240: midday (includes lunch)
V18_LUNCH_WEIGHT = 0.1      # override for bars 120-210 (core lunch chop)
V18_AFTERNOON_WEIGHT = 0.8  # bar 240-330: afternoon
V18_POWER_HOUR_WEIGHT = 0.0 # bar 330+: too dangerous for 0DTE longs

# MFE/MAE forward window (bars)
V18_FORWARD_WINDOW = 30


def compute_v18_labels(close_prices, high_prices, low_prices, dates, valid,
                       atr_values, option_stopped_pnl):
    """Compute v18 path-quality labels for 5-head model.

    Args:
        close_prices: (N,) float64 array of SPX close prices
        high_prices: (N,) float64 array of SPX high prices
        low_prices: (N,) float64 array of SPX low prices
        dates: (N,) array of date strings
        valid: (N,) bool array
        atr_values: (N,) float64 array of ATR-14 values (raw, not normalized)
        option_stopped_pnl: dict mapping strike key -> (N,) float32 array
            Keys: 'call_stopped_pnl', 'put_stopped_pnl',
                  'otm5_call_stopped_pnl', 'otm5_put_stopped_pnl',
                  'otm10_call_stopped_pnl', 'otm10_put_stopped_pnl'

    Returns:
        dict with keys:
            'mfe': (N,) max favorable excursion (% of entry price)
            'mae': (N,) max adverse excursion (% of entry price, negative)
            'entry_gate': (N,) sigmoid((MFE - MAE) / ATR - 1.0), zeroed lunch/power
            'risk_stop_distance': (N,) MAE / ATR (how far to set stop)
            'risk_target_distance': (N,) MFE / ATR (how far the target is)
            'risk_conviction': (N,) MFE / (MFE + |MAE|) (0-1, quality of path)
            'exit_label': (N,) 1.0 when remaining path quality flips unfavorable
            'direction_label': (N,) int class index (0-5) for best strike type
            'bar_weight': (N,) time-of-day loss weighting
    """
    N = len(close_prices)

    labels = {
        'mfe': np.full(N, np.nan, dtype=np.float32),
        'mae': np.full(N, np.nan, dtype=np.float32),
        'entry_gate': np.full(N, np.nan, dtype=np.float32),
        'risk_stop_distance': np.full(N, np.nan, dtype=np.float32),
        'risk_target_distance': np.full(N, np.nan, dtype=np.float32),
        'risk_conviction': np.full(N, np.nan, dtype=np.float32),
        'exit_label': np.full(N, np.nan, dtype=np.float32),
        'direction_label': np.full(N, np.nan, dtype=np.float32),
        'bar_weight': np.full(N, np.nan, dtype=np.float32),
    }

    # Build day start/end index map for same-day windowing
    unique_dates = sorted(set(dates))
    day_indices = {d: np.where(np.array([x == d for x in dates]))[0] for d in unique_dates}
    bar_of_day_map = np.zeros(N, dtype=np.int32)
    for d, idx_arr in day_indices.items():
        for k, gi in enumerate(idx_arr):
            bar_of_day_map[gi] = k

    # Map from stopped_pnl dict keys to our 6-class labels
    pnl_keys = [
        'call_stopped_pnl', 'otm5_call_stopped_pnl', 'otm10_call_stopped_pnl',
        'put_stopped_pnl', 'otm5_put_stopped_pnl', 'otm10_put_stopped_pnl',
    ]

    for i in range(N):
        if not valid[i]:
            continue
        day = dates[i]
        bar_of_day = bar_of_day_map[i]
        price = close_prices[i]

        if price <= 0 or np.isnan(price):
            continue

        # --- bar_weight (always computable) ---
        if bar_of_day < NO_TRADE_BEFORE_BAR:
            labels['bar_weight'][i] = 0.0  # pre-market warmup
        elif bar_of_day < MORNING_END_BAR:
            labels['bar_weight'][i] = V18_MORNING_WEIGHT
        elif bar_of_day < 120:
            labels['bar_weight'][i] = V18_MIDDAY_WEIGHT
        elif bar_of_day < 210:
            labels['bar_weight'][i] = V18_LUNCH_WEIGHT  # core lunch
        elif bar_of_day < NO_TRADE_LUNCH_END:
            labels['bar_weight'][i] = V18_MIDDAY_WEIGHT
        elif bar_of_day < 330:
            labels['bar_weight'][i] = V18_AFTERNOON_WEIGHT
        else:
            labels['bar_weight'][i] = V18_POWER_HOUR_WEIGHT

        # --- MFE / MAE over forward window (same day only) ---
        end_bar = min(i + V18_FORWARD_WINDOW, N)
        # Clip to same day
        same_day_end = end_bar
        for j in range(i + 1, end_bar):
            if dates[j] != day:
                same_day_end = j
                break

        if same_day_end <= i + 1:
            continue  # not enough forward bars

        fwd_highs = high_prices[i + 1:same_day_end]
        fwd_lows = low_prices[i + 1:same_day_end]

        if len(fwd_highs) == 0:
            continue

        # MFE: max upside from entry (best high - entry) / entry
        mfe = (np.nanmax(fwd_highs) - price) / price
        # MAE: max downside from entry (worst low - entry) / entry (negative)
        mae = (np.nanmin(fwd_lows) - price) / price

        labels['mfe'][i] = mfe
        labels['mae'][i] = mae

        # --- ATR-normalized risk labels ---
        atr = atr_values[i]
        if np.isnan(atr) or atr <= 0:
            atr = price * 0.001  # fallback: 0.1% of price

        atr_pct = atr / price  # ATR as % of price

        labels['risk_stop_distance'][i] = abs(mae) / atr_pct
        labels['risk_target_distance'][i] = mfe / atr_pct
        denominator = mfe + abs(mae)
        labels['risk_conviction'][i] = (mfe / denominator) if denominator > 0 else 0.5

        # --- Entry gate: sigmoid((MFE - |MAE|) / atr_pct - 1.0) ---
        edge = (mfe - abs(mae)) / atr_pct
        gate_raw = 1.0 / (1.0 + math.exp(-(edge - 1.0)))
        # Zero during lunch core and power hour
        if 120 <= bar_of_day < 210 or bar_of_day >= 330:
            gate_raw = 0.0
        # Zero pre-market
        if bar_of_day < NO_TRADE_BEFORE_BAR:
            gate_raw = 0.0
        labels['entry_gate'][i] = gate_raw

        # --- Exit label: bar-by-bar remaining path quality ---
        # For each bar j in [i+1, same_day_end), compute remaining MFE/MAE
        # Exit = 1.0 when remaining edge goes negative
        # We compute the exit label at bar i as: the first bar where
        # remaining_mfe < remaining_mae (path turns bad)
        # Actually, exit_label[i] is whether THIS bar is a good time to exit
        # if we entered at some earlier bar. We set it to 1.0 when
        # forward path quality is negative (remaining MFE < |remaining MAE|).
        remaining_mfe = (np.nanmax(fwd_highs) - price) / price  # same as mfe
        remaining_mae = (np.nanmin(fwd_lows) - price) / price   # same as mae
        if remaining_mfe < abs(remaining_mae):
            labels['exit_label'][i] = 1.0  # path is unfavorable, exit now
        else:
            labels['exit_label'][i] = 0.0

        # --- Direction label: 6-class argmax by stopped P&L ---
        pnl_values = np.full(V18_NUM_DIRECTIONS, np.nan, dtype=np.float32)
        for cls_idx, key in enumerate(pnl_keys):
            arr = option_stopped_pnl.get(key)
            if arr is not None and i < len(arr):
                pnl_values[cls_idx] = arr[i]

        # If we have at least one valid P&L, pick the best
        valid_mask = ~np.isnan(pnl_values)
        if valid_mask.any():
            # Among valid P&L, pick best. If all negative, still pick least bad.
            best_cls = np.nanargmax(pnl_values)
            labels['direction_label'][i] = float(best_cls)

    # Second pass: refine exit_label using sliding window
    # For bar i, exit_label should be 1.0 when remaining path from i is bad.
    # We need to compute this more carefully: for each bar, look at what
    # happens from THIS bar forward (not from some hypothetical entry).
    for i in range(N):
        if not valid[i] or np.isnan(labels['mfe'][i]):
            continue
        day = dates[i]
        price = close_prices[i]
        if price <= 0:
            continue

        end_bar = min(i + V18_FORWARD_WINDOW, N)
        same_day_end = end_bar
        for j in range(i + 1, end_bar):
            if dates[j] != day:
                same_day_end = j
                break

        if same_day_end <= i + 1:
            labels['exit_label'][i] = 1.0  # no more bars, exit
            continue

        # Look at next 5 bars vs full window
        short_end = min(i + 5, same_day_end)
        short_highs = high_prices[i + 1:short_end]
        short_lows = low_prices[i + 1:short_end]

        if len(short_highs) == 0:
            labels['exit_label'][i] = 1.0
            continue

        short_mfe = (np.nanmax(short_highs) - price) / price
        short_mae = (np.nanmin(short_lows) - price) / price

        # Exit when near-term (5 bar) risk exceeds near-term reward
        if short_mfe < abs(short_mae) * 0.5:
            labels['exit_label'][i] = 1.0
        else:
            labels['exit_label'][i] = 0.0

    return labels


# Bar-of-day boundary for morning end (used by v18 bar_weight)
MORNING_END_BAR = 60  # bar 60 = 10:30 AM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SPX 0DTE trading data")
    parser.add_argument("--start", type=str, default="2022-05-11")
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
    otm_deep_count = int(np.sum(~np.isnan(option_prices.get('otm30_call', np.array([]))))) if option_prices else 0
    actionable_count = int(np.sum(option_prices.get('actionable_mask', np.zeros(len(df))) > 0.5)) if option_prices else 0
    print(f"  Valid bars: {valid_count}/{len(df)} ({100*valid_count/len(df):.0f}%)")
    print(f"  Bars with option prices: {opt_count}/{len(df)} ({100*opt_count/len(df):.0f}%)")
    print(f"  Bars with OTM prices: {otm_count}/{len(df)} ({100*otm_count/len(df):.0f}%)")
    print(f"  Bars with deep OTM (+/-30) prices: {otm_deep_count}/{len(df)} ({100*otm_deep_count/len(df):.0f}%)")
    print(f"  Bars with option P&L: {pnl_count}/{len(df)} ({100*pnl_count/len(df):.0f}%)")
    print(f"  Bars flagged actionable (quality/risk mask): {actionable_count}/{len(df)} ({100*actionable_count/len(df):.0f}%)")
    print(f"  Bars with EXIT=1 (call): {exit_count}")
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # --- v17: Prediction labels (forward SPX returns + volatility) ---
    print("Computing prediction labels (forward returns, volatility, action targets)...")
    t0 = time.time()
    # Extract close prices for prediction label computation
    # Feature index 0 is 'ret_6' — we need actual close prices
    # The close prices are stored in the dataframe
    close_prices = df['close'].values.astype(np.float32) if 'close' in df.columns else None
    if close_prices is not None:
        pred_labels = compute_prediction_labels_from_prices(close_prices, dates, valid)
        for k, v in pred_labels.items():
            option_prices[f'pred_{k}'] = v
        n_valid_r30 = int(np.sum(~np.isnan(pred_labels['return_30'])))
        n_trade = int(np.sum(pred_labels['action_target'] > 0.5))
        print(f"  Valid 30-bar returns: {n_valid_r30}/{valid_count}")
        print(f"  Action target (trade): {n_trade}/{n_valid_r30} ({100*n_trade/max(n_valid_r30,1):.1f}%)")
        print(f"  Mean abs return (30-bar): {np.nanmean(np.abs(pred_labels['return_30'])):.6f}")
        print(f"  Mean volatility (30-bar): {np.nanmean(pred_labels['volatility_30']):.6f}")
    else:
        print("  WARNING: No close prices available, skipping prediction labels")
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # --- v18: Path-quality labels (MFE/MAE, entry gate, risk, exit, direction) ---
    print("Computing v18 path-quality labels...")
    t0 = time.time()
    if close_prices is not None and option_prices is not None:
        high_prices = df['high'].values.astype(np.float64)
        low_prices = df['low'].values.astype(np.float64)
        # Extract raw ATR-14 values (feature index 29, pre-normalization)
        # ATR in raw_features is atr_14 / close, so multiply back by close
        idx_atr = _FEAT_IDX['atr_14']
        atr_raw = raw_features[:, idx_atr].astype(np.float64)
        atr_abs = atr_raw * close_prices.astype(np.float64)  # convert back to absolute ATR
        # Provide stopped P&L arrays for direction labeling
        stopped_pnl_dict = {}
        for k in ('call_stopped_pnl', 'put_stopped_pnl',
                  'otm5_call_stopped_pnl', 'otm5_put_stopped_pnl',
                  'otm10_call_stopped_pnl', 'otm10_put_stopped_pnl'):
            if k in option_prices:
                stopped_pnl_dict[k] = option_prices[k]
        v18_labels = compute_v18_labels(
            close_prices.astype(np.float64), high_prices, low_prices,
            dates, valid, atr_abs, stopped_pnl_dict,
        )
        for k, v in v18_labels.items():
            option_prices[f'v18_{k}'] = v
        n_valid_gate = int(np.sum(~np.isnan(v18_labels['entry_gate'])))
        n_good_gate = int(np.sum(v18_labels['entry_gate'] > 0.5))
        n_valid_dir = int(np.sum(~np.isnan(v18_labels['direction_label'])))
        n_exit = int(np.sum(v18_labels['exit_label'] == 1.0))
        print(f"  Valid entry gates: {n_valid_gate}/{valid_count}")
        print(f"  Good gates (>0.5): {n_good_gate}/{n_valid_gate} ({100*n_good_gate/max(n_valid_gate,1):.1f}%)")
        print(f"  Valid direction labels: {n_valid_dir}/{valid_count}")
        print(f"  Exit=1 bars: {n_exit}/{n_valid_gate}")
        print(f"  Mean MFE: {np.nanmean(v18_labels['mfe']):.6f}")
        print(f"  Mean MAE: {np.nanmean(v18_labels['mae']):.6f}")
        print(f"  Mean conviction: {np.nanmean(v18_labels['risk_conviction']):.4f}")
    else:
        print("  WARNING: Missing close/option data, skipping v18 labels")
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # --- v11: Setup Recognition + Regime Classification (pre-normalization) ---
    # (Legacy — kept for backward compatibility with old train.py versions)
    date_arr = np.array(dates)
    if option_prices is not None:
        option_prices['setup_mask'] = np.zeros(len(dates), dtype=np.float32)
        option_prices['regime_mask'] = np.zeros(len(dates), dtype=np.float32)

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
