"""Feature constants, normalization, and contract validation.

v2 rebuild: all features computed from raw data in v2/pipeline/compute_features.py.
Normalization uses rolling z-score (not per-day) to preserve regime information.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from v2.pipeline.compute_features import (
    ALL_FEATURE_NAMES,
    FLOW_FEATURE_NAMES,
    OPTION_FEATURE_NAMES,
    PRICE_FEATURE_NAMES,
    SESSION_FEATURE_NAMES,
    SURFACE_FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BARS_PER_DAY = 390
FORWARD_BARS = 30
LOOKBACK_WINDOW = 60

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
NO_TRADE_AFTER_BAR = 270  # Match label supervision window (labels exist for bars 30-269)

STARTING_CAPITAL = 10_000
POSITION_RISK_TARGET = 0.05
SPX_MULTIPLIER = 100

# Canonical feature names (single source of truth lives in compute_features.py)
FEATURE_NAMES = list(ALL_FEATURE_NAMES)
NUM_FEATURES = len(FEATURE_NAMES)
LEGACY_52_FEATURE_NAMES = list(PRICE_FEATURE_NAMES) + list(OPTION_FEATURE_NAMES) + list(FLOW_FEATURE_NAMES)

# Fast name -> index lookup
_FEAT_IDX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

# Features excluded from z-score normalization (already bounded/categorical).
# These are either naturally in [0, 1], [-1, 1], or small discrete sets.
_NO_NORMALIZE = {
    # Bounded time features
    'minutes_to_close',       # log(mtc) / log(390), in [0, 1]
    'theta_acceleration',     # 1/sqrt(mtc), in [0, 1]
    # Bounded percentile/rank
    'iv_percentile',          # [0, 1] percentile rank
    # Categorical/discrete
    'vix_regime',             # {-1, -0.33, 0.33, 1}
    'ib_break',               # {-1, 0, 1}
    'macdh_slope',            # {-1, 0, 1}
    # Already bounded indicators
    'bollinger_position',     # ~[-2, 2] but typically [-1, 1]
    'rsi_7',                  # [0, 1]
    'session_range_position', # [0, 1]
    'va_position',            # [0, 1]
    'bar_delta',              # [-1, 1]
    'consec_direction',       # [-1, 1]
    'effort_vs_result',       # [0, 3]
    # Bounded flow ratios
    'call_put_flow_ratio',    # [0, 1]
    'chain_call_put_ratio',   # [0, 1]
    'put_call_txn_ratio',     # [0, 1]
    # Intraday phase features (already bounded)
    'intraday_sin',           # [-1, 1]
    'intraday_cos',           # [-1, 1]
    'intraday_phase',         # [0, 1] (normalized phase / 6)
    # Session-structure bounded states
    'first15_close_position', # [0, 1]
    'first15_acceptance',     # [-1, 1]
    'vwap_reclaim_state',     # {-1, 0, 1}
    'marker_10am',            # [0, 1]
    'marker_11am',            # [0, 1]
    'marker_1130am',          # [0, 1]
    'lunch_flag',             # {0, 1}
    'power_hour_flag',        # {0, 1}
    'volume_climax_signal',   # [0, 3]
    'breakout_confirmation',  # [-2, 2]
    # v3 W2a additions — bounded / discrete by construction.
    'sigma_pos',                # clipped to [-5, 5] in market_structure
    'inside_first15',           # {0, 1}
    'late_window_40_120_flag',  # {0, 1}
    'omar_mid_pos_units',       # clipped to [-10, 10] by construction
    'last10_break_state',       # {-1, 0, +1}
}


# ---------------------------------------------------------------------------
# Normalization: Rolling z-score (preserves regime information)
# ---------------------------------------------------------------------------

def _rolling_zscore(features: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score normalization with walk-forward expanding window.

    Unlike per-day z-score, this preserves inter-day regime variation.
    A high-VIX day will have high realized_vol z-scores because it's
    unusual relative to the last N days.

    For the first `window` bars, uses expanding window (cold start).
    """
    out = features.copy().astype(np.float64)
    for j in range(out.shape[1]):
        name = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"feature_{j}"
        if name in _NO_NORMALIZE:
            continue
        col = pd.Series(out[:, j], dtype=np.float64)
        col[~valid] = np.nan
        mu = col.expanding(min_periods=5).mean()
        sigma = col.expanding(min_periods=5).std()
        # Switch to rolling after warmup
        rolling_mu = col.rolling(window, min_periods=5).mean()
        rolling_sigma = col.rolling(window, min_periods=5).std()
        # Use expanding for first `window` bars, rolling after
        switch_idx = min(window, len(col))
        mu.iloc[switch_idx:] = rolling_mu.iloc[switch_idx:]
        sigma.iloc[switch_idx:] = rolling_sigma.iloc[switch_idx:]
        normalized = (col - mu) / sigma.clip(lower=1e-10)
        out[:, j] = normalized.fillna(0.0).values
    out = np.clip(out, -5.0, 5.0)
    out = np.nan_to_num(out, nan=0.0)
    return out


def normalize_features(features: np.ndarray, valid: np.ndarray,
                       dates: list | None = None,
                       window: int = 390 * 60) -> np.ndarray:
    """Normalize features using rolling z-score.

    Args:
        features: (n_bars, n_features) raw feature array
        valid: (n_bars,) boolean mask of valid bars
        dates: unused in v2 (kept for API compatibility)
        window: rolling window size in bars (default: 60 days * 390 bars)
    """
    if len(features) == 0:
        return features.copy()
    return _rolling_zscore(features, valid.astype(bool), window)


# ---------------------------------------------------------------------------
# Feature Contract
# ---------------------------------------------------------------------------

FEATURE_CONTRACT_VERSION = "v2.2"  # W2a: +8 late-session market-structure features (81 -> 89)


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
    """Adaptive spread in basis points by time-of-day and VIX regime."""
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
