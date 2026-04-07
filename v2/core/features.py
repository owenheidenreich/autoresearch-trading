"""Feature engineering: 39-feature vector from market data.

Constants, normalization, and feature contract validation.
The actual compute_features() function (1,222 lines in v1 prepare.py)
is not ported yet - v2 training uses pre-computed features from the
dataset pipeline. Full port is a future task for live trading.

v1 origin: training/prepare.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BARS_PER_DAY = 390
FORWARD_BARS = 30
LOOKBACK_WINDOW = 60
NUM_FEATURES = 55  # 39 original + 16 enriched (removed dupes, volume_zero_flag, vix features)

OPTION_SPREAD_BPS = 150
SPREAD_COST_PCT = 2 * OPTION_SPREAD_BPS / 10000.0  # 0.03 = 3% round-trip

STOP_LOSS_PCT = 0.30
DYNAMIC_STOP_BASE = 0.45
DYNAMIC_STOP_MIN = 0.15
DYNAMIC_STOP_MAX = 0.60
MAX_HOLD_BARS = BARS_PER_DAY
STOP_COOLDOWN_BARS = 5
MIN_HOLD_BARS = 2
EXIT_GATE_THRESHOLD = 0.60

NO_TRADE_BEFORE_BAR = 30
NO_TRADE_LUNCH_START = 60
NO_TRADE_LUNCH_END = 240
NO_TRADE_AFTER_BAR = 330

STARTING_CAPITAL = 10_000
POSITION_RISK_TARGET = 0.05
SPX_MULTIPLIER = 100

# Feature names in canonical order
FEATURE_NAMES = [
    # === Price returns (2) ===
    'ret_6',                    # 0: 30-bar (30min) return
    'ret_12',                   # 1: 60-bar (1hr) return
    # === Volume (1) ===
    'volume_ratio',             # 2: bar volume / 20-bar SMA
    # === Gamma (1) ===
    'gamma_pressure',           # 3: GEX proxy
    # === Volatility (3) ===
    'bar_range',                # 4: (high - low) / close
    'realized_vol',             # 5: 20-bar rolling stdev of returns
    'range_ratio',              # 6: current bar range / 20-bar avg range
    # === VWAP (1) ===
    'vwap_dist',                # 7: (close - session VWAP) / close
    # === Session structure (1) ===
    'session_range_pct',        # 8: session range / close
    # === Key levels (1) ===
    'prev_high_dist',           # 9: distance to previous day high
    # === Trend (3) ===
    'ema_cross',                # 10: (EMA8 - EMA21) / close
    'consec_direction',         # 11: consecutive same-direction bars
    'speed_estimate',           # 12: |5-bar return| / realized_vol
    # === VIX (1) ===
    'vix_roc',                  # 13: VIX 10-bar rate of change
    # === Time (1) ===
    'minutes_to_close',         # 14: log(minutes remaining + 1)
    # === IV Percentile (1) ===
    'iv_percentile',            # 15: current IV rank vs 60-day [0,1]
    # === Options (2) ===
    'atm_iv',                   # 16: ATM implied vol
    'iv_skew',                  # 17: put IV - call IV
    # === VIX / Regime (2) ===
    'vix_regime',               # 18: regime bucket
    'vrp',                      # 19: variance risk premium
    # === Greeks (3) ===
    'atm_gamma',                # 20: ATM call gamma
    'atm_theta_per_bar',        # 21: ATM theta per 1-min bar
    'charm_estimate',           # 22: dDelta/dT
    # === Bollinger (1) ===
    'bollinger_position',       # 23: 5-min Bollinger band position
    # === Range extras (2) ===
    'rsi_7',                    # 24: 5-min RSI(7)
    'session_range_position',   # 25: position within session range
    # === Market structure (3) ===
    'poc_dist',                 # 26: distance to session POC
    'va_position',              # 27: position within Value Area
    'ib_break',                 # 28: Initial Balance break state
    # === v9 features (3) ===
    'atr_14',                   # 29: 14-bar ATR / close
    'bar_delta',                # 30: (close - open) / (high - low)
    'session_cum_delta',        # 31: cumulative bar deltas
    # === Spread width (1) ===
    'option_spread_width',      # 32: ATM option bid-ask proxy
    # === v10 features (3) ===
    'macdh_slope',              # 33: 5-min MACD-H direction
    'force_index_2',            # 34: 5-min Force Index
    'prev_close_dist',          # 35: distance to prev day close
    # === v10 continued (2) ===
    'effort_vs_result',         # 36: 5-min effort vs result
    'trend_5min',               # 37: 5-min EMA(13) slope
    # === v17 promoted (1) ===
    'overnight_gap',            # 38: (day open - prev close) / prev close
    # === v2 enriched (16) -- from wide-grid option data ===
    'current_moneyness_pct',    # 39: how far ATM strike is from current SPX
    'intraday_drift_pct',       # 40: (current SPX - opening ATM) / ATM
    'near_atm_moneyness_pct',   # 41: moneyness of nearest-ATM strike
    'near_atm_call_volume',     # 42: call volume at nearest ATM strike
    'near_atm_put_volume',      # 43: put volume at nearest ATM strike
    'near_atm_total_volume',    # 44: total volume at nearest ATM
    'call_put_flow_ratio',      # 45: call_vol / total_vol (flow direction)
    'log_total_volume',         # 47: log(1 + total volume)
    'chain_call_put_ratio',     # 48: call/total across entire chain
    'log_chain_volume',         # 49: log(1 + chain total volume)
    'call_hl_range_pct',        # 50: (high-low)/mid for nearest ATM call
    'near_atm_call_price_norm', # 51: call_close / SPX * 100 (normalized)
    'near_atm_put_price_norm',  # 52: put_close / SPX * 100 (normalized)
    'theta_acceleration',       # 53: 1/sqrt(minutes_to_close) (0DTE specific)
    'near_atm_transactions',    # 53: transactions at nearest ATM
    'put_call_txn_ratio',       # 54: put_txn / (call_txn + put_txn) (order flow)
]

assert len(FEATURE_NAMES) == NUM_FEATURES

# Fast name -> index lookup
_FEAT_IDX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

# Features excluded from z-score normalization (already bounded/categorical)
_NO_NORMALIZE = {
    'minutes_to_close',
    'iv_percentile',
    'vix_regime',
    'bollinger_position',
    'session_range_position',
    'rsi_7',
    'va_position',
    'ib_break',
    'bar_delta',
    'macdh_slope',
    'effort_vs_result',
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _rolling_zscore(features: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score normalization. Used for replay/live."""
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


def _per_day_zscore(features: np.ndarray, valid: np.ndarray,
                    dates: list, walk_forward: bool = True) -> np.ndarray:
    """Per-day mean, expanding-window std normalization (walk-forward).

    Removes day-specific feature fingerprints that cause the model to memorize
    individual dates while preserving regime-level information.
    """
    out = features.copy().astype(np.float64)

    day_starts = [0] + [i for i in range(1, len(dates)) if dates[i] != dates[i - 1]]
    day_ends = day_starts[1:] + [len(dates)]

    for j in range(out.shape[1]):
        name = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"feature_{j}"
        if name in _NO_NORMALIZE:
            continue

        if not walk_forward:
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
            min_warmup_days = 20
            for day_i, (ds, de) in enumerate(zip(day_starts, day_ends)):
                chunk = out[ds:de, j]
                mask = valid[ds:de]
                day_vals = chunk[mask]
                if len(day_vals) < 3:
                    out[ds:de, j] = 0.0
                    continue
                day_mean = np.mean(day_vals)
                expanding_end = ds
                expanding_vals = out[:expanding_end, j][valid[:expanding_end]]
                if len(expanding_vals) < 50 or day_i < min_warmup_days:
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
    """Normalize features. Per-day z-score if dates provided, rolling otherwise."""
    if len(features) == 0:
        return features.copy()
    if dates is not None and len(dates) == len(features):
        return _per_day_zscore(features, valid.astype(bool), dates)
    window = min(len(features) // 4, 500)
    window = max(window, 50)
    return _rolling_zscore(features, valid.astype(bool), int(window))


# ---------------------------------------------------------------------------
# Feature Contract
# ---------------------------------------------------------------------------

FEATURE_CONTRACT_VERSION = "v2.0"


def validate_feature_shape(arr: np.ndarray) -> bool:
    """Check feature array has correct shape."""
    return arr.ndim == 2 and arr.shape[1] == NUM_FEATURES


def validate_feature_names(names: list[str]) -> bool:
    """Check feature names match canonical order."""
    return list(names) == list(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Spread model (used by simulator)
# ---------------------------------------------------------------------------

def compute_adaptive_spread_bps(minutes_remaining: float, vix_regime: float,
                                is_otm: bool = False) -> float:
    """Adaptive spread in basis points by time-of-day and VIX regime.

    Ported from v1 training/replay.py lines 84-124.
    """
    if minutes_remaining > 330:
        base = 40.0
    elif minutes_remaining > 270:
        base = 30.0
    elif minutes_remaining > 210:
        base = 50.0
    elif minutes_remaining > 150:
        base = 80.0
    elif minutes_remaining > 90:
        base = 50.0
    elif minutes_remaining > 30:
        base = 60.0
    else:
        base = 150.0

    if is_otm:
        base *= 2.0

    vix_mult = 1.0 + max(0.0, vix_regime - 0.3) * 2.0
    return min(base * vix_mult, 500.0)
