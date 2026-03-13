"""
One-time data preparation for autoresearch-trading.
Downloads market data, computes features, and provides evaluation harness.

Usage:
    uv run prepare.py                  # full prep (download + features)
    uv run prepare.py --start 2010     # custom start year

Data and features are stored in ~/.cache/autoresearch-trading/.
"""

import os
import sys
import time
import math
import argparse
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_LOOKBACK = 252       # maximum lookback window (1 year of trading days)
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
TRAIN_END = "2024-12-31" # last date in training set
VAL_START = "2025-01-01" # first date in validation set
VAL_END = "2025-12-31"   # last date in validation set
TRANSACTION_COST_BPS = 5 # round-trip transaction cost in basis points
MIN_TRADES = 20          # minimum position changes for valid evaluation
ANNUAL_TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-trading")
DATA_DIR = os.path.join(CACHE_DIR, "data")
FEATURES_DIR = os.path.join(CACHE_DIR, "features")

# ---------------------------------------------------------------------------
# Feature names (39 features)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Returns (6)
    'return_1d', 'return_2d', 'return_5d', 'return_10d', 'return_21d', 'return_63d',
    # Realized volatility (5)
    'rvol_5d', 'rvol_10d', 'rvol_21d', 'rvol_63d', 'rvol_ratio_5_21',
    # Momentum indicators (8)
    'rsi_14', 'rsi_5', 'macd', 'macd_signal', 'macd_hist',
    'sma_dist_20', 'sma_dist_50', 'sma_dist_200',
    # Mean reversion (2)
    'bb_position', 'bb_width',
    # Volume (2)
    'volume_sma_ratio', 'volume_change',
    # Price range (3)
    'daily_range', 'daily_range_ratio', 'gap',
    # VIX / volatility regime (6)
    'vix_level', 'vix_change_1d', 'vix_change_5d',
    'vix_zscore_20', 'vix_percentile_252', 'rv_iv_spread',
    # Interest rates (3)
    'tnx_level', 'tnx_change_1d', 'tnx_change_5d',
    # Cyclical time encoding (4)
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
]

NUM_FEATURES = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _flatten_yf_columns(df):
    """Handle yfinance MultiIndex columns (varies by version)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_market_data(start_date="2010-01-01"):
    """Download SPY, VIX, TNX daily data via yfinance. Returns merged DataFrame."""
    import yfinance as yf

    raw_path = os.path.join(DATA_DIR, "raw_data.parquet")
    if os.path.exists(raw_path):
        print(f"Data: already downloaded at {raw_path}")
        return pd.read_parquet(raw_path)

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading market data from {start_date}...")

    spy = _flatten_yf_columns(
        yf.download("SPY", start=start_date, auto_adjust=True, progress=False)
    )
    vix = _flatten_yf_columns(
        yf.download("^VIX", start=start_date, auto_adjust=True, progress=False)
    )
    tnx = _flatten_yf_columns(
        yf.download("^TNX", start=start_date, auto_adjust=True, progress=False)
    )

    # Merge on trading days (SPY as the anchor)
    df = pd.DataFrame(index=spy.index)
    df['open'] = spy['Open']
    df['high'] = spy['High']
    df['low'] = spy['Low']
    df['close'] = spy['Close']
    df['volume'] = spy['Volume']
    df['vix_close'] = vix['Close'].reindex(spy.index).ffill()
    df['tnx_close'] = tnx['Close'].reindex(spy.index).ffill()

    df = df.dropna(subset=['close'])
    df = df.ffill().bfill()

    df.to_parquet(raw_path)
    print(f"Data: saved {len(df)} trading days to {raw_path}")
    return df

# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _rsi(series, period):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return (100 - (100 / (1 + rs))) / 100.0  # normalized to [0, 1]


def _ema(series, span):
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _rolling_percentile(series, window=252, min_periods=63):
    """Rolling percentile rank (fraction of window below current value)."""
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: np.mean(x <= x[-1]), raw=True
    )


def compute_features(df):
    """Compute all features from raw OHLCV + VIX + TNX data.

    Returns (features_df, targets_series).
    All features use only past data — no lookahead.
    Targets are 1-day forward log returns (close-to-close).
    """
    close = df['close'].astype(float)
    log_close = np.log(close)

    features = pd.DataFrame(index=df.index)

    # ---- Returns at multiple horizons ----
    for n in [1, 2, 5, 10, 21, 63]:
        features[f'return_{n}d'] = log_close.diff(n)

    # ---- Realized volatility ----
    daily_ret = log_close.diff()
    for n in [5, 10, 21, 63]:
        features[f'rvol_{n}d'] = daily_ret.rolling(n).std() * np.sqrt(ANNUAL_TRADING_DAYS)
    features['rvol_ratio_5_21'] = (
        features['rvol_5d'] / features['rvol_21d'].replace(0, 1e-10)
    )

    # ---- Momentum / Trend indicators ----
    features['rsi_14'] = _rsi(close, 14)
    features['rsi_5'] = _rsi(close, 5)

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    features['macd'] = macd_line / close          # price-normalized
    features['macd_signal'] = macd_signal / close
    features['macd_hist'] = (macd_line - macd_signal) / close

    for n in [20, 50, 200]:
        sma = close.rolling(n).mean()
        roll_std = close.rolling(n).std().replace(0, 1e-10)
        features[f'sma_dist_{n}'] = (close - sma) / roll_std

    # ---- Mean reversion (Bollinger Bands) ----
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std().replace(0, 1e-10)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = (bb_upper - bb_lower).replace(0, 1e-10)
    features['bb_position'] = (close - bb_lower) / bb_range
    features['bb_width'] = bb_range / sma20

    # ---- Volume ----
    vol = df['volume'].astype(float).replace(0, np.nan).ffill()
    vol_sma20 = vol.rolling(20).mean().replace(0, 1e-10)
    features['volume_sma_ratio'] = vol / vol_sma20
    features['volume_change'] = vol.pct_change()

    # ---- Daily range / gaps ----
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    opn = df['open'].astype(float)
    features['daily_range'] = (high - low) / close
    range_sma20 = features['daily_range'].rolling(20).mean().replace(0, 1e-10)
    features['daily_range_ratio'] = features['daily_range'] / range_sma20
    features['gap'] = (opn - close.shift(1)) / close.shift(1)

    # ---- VIX / Volatility regime ----
    vix = df['vix_close'].astype(float)
    features['vix_level'] = vix / 100.0                          # scaled
    features['vix_change_1d'] = vix.pct_change()
    features['vix_change_5d'] = vix.pct_change(5)
    vix_mean20 = vix.rolling(20).mean()
    vix_std20 = vix.rolling(20).std().replace(0, 1e-10)
    features['vix_zscore_20'] = (vix - vix_mean20) / vix_std20
    features['vix_percentile_252'] = _rolling_percentile(vix, 252, 63)
    features['rv_iv_spread'] = features['rvol_21d'] - vix / 100.0

    # ---- Interest rates ----
    tnx = df['tnx_close'].astype(float)
    features['tnx_level'] = tnx / 100.0
    features['tnx_change_1d'] = tnx.diff() / 100.0
    features['tnx_change_5d'] = tnx.diff(5) / 100.0

    # ---- Cyclical time encoding ----
    dow = df.index.dayofweek.values.astype(float)
    month = df.index.month.values.astype(float)
    features['day_of_week_sin'] = np.sin(2 * np.pi * dow / 5)
    features['day_of_week_cos'] = np.cos(2 * np.pi * dow / 5)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)

    # ---- Target: 1-day forward log return ----
    targets = log_close.diff().shift(-1)

    # Verify feature names match
    assert list(features.columns) == FEATURE_NAMES, (
        f"Feature mismatch:\n  expected: {FEATURE_NAMES}\n  got: {list(features.columns)}"
    )

    return features, targets

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Features that should NOT be z-scored (already bounded / cyclical)
_NO_NORMALIZE = {'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'}


def normalize_features(features_df, window=252, min_periods=63):
    """Rolling z-score normalization. Uses only past data — no lookahead.
    Cyclical time features are left untouched.
    Extreme values clipped to [-5, 5].
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

def prepare_tensors(features_df, targets_series):
    """Convert features and targets to tensors, compute splits, save to cache."""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    feat_np = features_df.values.astype(np.float32)
    tgt_np = targets_series.values.astype(np.float32)
    dates = features_df.index

    # Valid rows: no NaN in features AND target exists
    valid_mask = ~(np.isnan(feat_np).any(axis=1) | np.isnan(tgt_np))

    # Find split boundaries
    train_end_idx = dates.get_indexer([pd.Timestamp(TRAIN_END)], method='ffill')[0]
    val_start_idx = dates.get_indexer([pd.Timestamp(VAL_START)], method='bfill')[0]
    # Clamp val_end to actual data range
    val_end_ts = pd.Timestamp(VAL_END)
    if val_end_ts > dates[-1]:
        val_end_idx = len(dates) - 1
    else:
        val_end_idx = dates.get_indexer([val_end_ts], method='ffill')[0]

    # Fill remaining NaN with 0 (valid_mask already excludes them)
    feat_np = np.nan_to_num(feat_np, nan=0.0)
    tgt_np = np.nan_to_num(tgt_np, nan=0.0)

    data = {
        'features': torch.tensor(feat_np, dtype=torch.float32),
        'targets': torch.tensor(tgt_np, dtype=torch.float32),
        'valid_mask': torch.tensor(valid_mask, dtype=torch.bool),
        'dates': dates.strftime('%Y-%m-%d').tolist(),
        'train_end_idx': int(train_end_idx),
        'val_start_idx': int(val_start_idx),
        'val_end_idx': int(val_end_idx),
        'feature_names': FEATURE_NAMES,
    }

    path = os.path.join(FEATURES_DIR, "data.pt")
    torch.save(data, path)

    # Stats
    n_train_valid = int(valid_mask[:train_end_idx + 1].sum())
    n_val_valid = int(valid_mask[val_start_idx:val_end_idx + 1].sum())

    print(f"Features: saved {len(dates)} days × {NUM_FEATURES} features")
    print(f"  Training : {dates[0].strftime('%Y-%m-%d')} → "
          f"{dates[train_end_idx].strftime('%Y-%m-%d')} ({n_train_valid} valid days)")
    print(f"  Validation: {dates[val_start_idx].strftime('%Y-%m-%d')} → "
          f"{dates[val_end_idx].strftime('%Y-%m-%d')} ({n_val_valid} valid days)")

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
        x: (batch, lookback, NUM_FEATURES) — feature windows
        y: (batch,)                        — 1-day forward returns
    """
    features = data['features'].to(device)
    targets = data['targets'].to(device)
    valid_mask = data['valid_mask']  # keep on CPU for indexing

    if split == "train":
        end = data['train_end_idx'] + 1
    else:
        end = data['val_end_idx'] + 1

    start = max(lookback, data['val_start_idx'] if split != "train" else lookback)

    # Precompute valid indices
    valid_indices = []
    for i in range(start, end):
        if valid_mask[i] and valid_mask[max(0, i - lookback):i].all():
            valid_indices.append(i)

    valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
    n = len(valid_indices)
    assert n > 0, f"No valid samples for split={split}, lookback={lookback}"

    offsets = torch.arange(-lookback, 0, device=device)  # (lookback,)

    if split == "train":
        while True:
            perm = torch.randperm(n, device=device)
            for i in range(0, n - batch_size + 1, batch_size):
                idx = valid_indices[perm[i:i + batch_size]]           # (B,)
                window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)  # (B, lookback)
                x = features[window_idx]                              # (B, lookback, F)
                y = targets[idx]                                      # (B,)
                yield x, y
    else:
        # Single sequential pass
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            idx = valid_indices[i:end_i]
            window_idx = idx.unsqueeze(1) + offsets.unsqueeze(0)
            x = features[window_idx]
            y = targets[idx]
            yield x, y


def load_raw_data():
    """Load the raw OHLCV + VIX + TNX DataFrame (for custom feature engineering in train.py)."""
    path = os.path.join(DATA_DIR, "raw_data.parquet")
    if not os.path.exists(path):
        print("Raw data not found. Run `uv run prepare.py` first.")
        sys.exit(1)
    return pd.read_parquet(path)

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sharpe(model, data, lookback, device, batch_size=256):
    """
    Walk-forward Sharpe ratio on the held-out validation period.
    **Higher is better.** This is the single metric for autoresearch-trading.

    Interface contract:
        model(x)  where x is (batch, lookback, NUM_FEATURES)
        returns    (batch,) tensor of position signals.
        Positions are clamped to [-1, 1].

    PnL per day = position_t × return_{t+1} − |Δposition_t| × cost
    Sharpe = mean(daily_pnl) / std(daily_pnl) × √252

    Returns a dict with val_sharpe and supplementary metrics.
    """
    model.eval()

    features = data['features'].to(device)
    targets = data['targets']       # stay on CPU for numpy conversion
    valid_mask = data['valid_mask']

    val_start = max(lookback, data['val_start_idx'])
    val_end = data['val_end_idx'] + 1

    # Collect valid validation-day indices
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
            'num_val_days': len(val_indices),
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

    # --- Compute PnL with transaction costs ---
    cost_per_unit = TRANSACTION_COST_BPS / 10_000
    turnover = np.abs(np.diff(positions, prepend=0.0))
    daily_pnl = positions * returns - turnover * cost_per_unit

    # --- Sharpe ratio ---
    mean_pnl = float(np.mean(daily_pnl))
    std_pnl = float(np.std(daily_pnl, ddof=1)) if len(daily_pnl) > 1 else 1e-10
    sharpe = (mean_pnl / max(std_pnl, 1e-10)) * math.sqrt(ANNUAL_TRADING_DAYS)

    # --- Max drawdown ---
    cum_pnl = np.cumsum(daily_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # --- Trade count (position sign changes) ---
    sign_changes = np.abs(np.diff(np.sign(positions)))
    num_trades = int(np.sum(sign_changes > 0))

    # --- Win rate ---
    wins = int(np.sum(daily_pnl > 0))
    losses = int(np.sum(daily_pnl < 0))
    win_rate = wins / max(wins + losses, 1)

    # --- Profit factor ---
    gross_profit = float(np.sum(daily_pnl[daily_pnl > 0]))
    gross_loss = float(abs(np.sum(daily_pnl[daily_pnl < 0])))
    profit_factor = gross_profit / max(gross_loss, 1e-10)

    # --- Annualized return ---
    annual_return = mean_pnl * ANNUAL_TRADING_DAYS

    return {
        'val_sharpe': float(sharpe),
        'max_drawdown': max_dd,
        'annual_return': annual_return,
        'num_trades': num_trades,
        'win_rate': float(win_rate),
        'profit_factor': profit_factor,
        'num_val_days': len(val_indices),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch-trading")
    parser.add_argument(
        "--start", type=str, default="2010-01-01",
        help="Start date for data download (default: 2010-01-01)"
    )
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    t0 = time.time()
    raw_df = download_market_data(args.start)
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # Step 2: Compute features
    print("Computing features...")
    t0 = time.time()
    features_df, targets = compute_features(raw_df)
    print(f"  Raw features: {len(features_df)} days × {NUM_FEATURES} features")
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # Step 3: Normalize (rolling z-score, no lookahead)
    print("Normalizing features (rolling z-score, 252-day window)...")
    t0 = time.time()
    norm_features = normalize_features(features_df)
    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # Step 4: Save tensors
    print("Saving tensors...")
    data = prepare_tensors(norm_features, targets)
    print()
    print("Done! Ready to train: uv run train.py")
