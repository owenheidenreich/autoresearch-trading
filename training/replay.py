"""
Replay mode: run trained model against a market day for paper trading validation.

Downloads data for a specific trading day, computes features, and runs the
trained model bar-by-bar to simulate trades. Useful for:
  - Forward-testing on unseen days
  - Validating model behavior before live paper trading
  - Speed-adjustable replay (instant to real-time)

Usage:
  python3 replay.py --date 2026-03-17                    # instant replay
  python3 replay.py --date 2026-03-17 --speed 10         # 10x speed
  python3 replay.py --date 2026-03-17 --verbose           # every bar
  python3 replay.py --date 2026-03-17 --output trades.csv # save log
"""
from __future__ import annotations

import os, sys, time, argparse, pickle, json
import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from prepare.py (safe — guarded by __main__)
from prepare import (
    compute_features, normalize_features, normalize_features_with_context,
    _ib_client, _download_ibkr_index, download_spy_bars_ibkr, download_vix_bars,
    download_spxw_ibkr, load_spxw_caches, load_spxw_chain_caches,
    is_0dte_day, DATA_DIR, CACHE_DIR, NUM_FEATURES, BARS_PER_DAY,
    BAR_SIZE_MINUTES, STOP_LOSS_PCT, MAX_HOLD_BARS, OPTION_SPREAD_BPS,
    STOP_COOLDOWN_BARS, NO_TRADE_BEFORE_BAR,
    THETA_DECAY_DAILY, ATM_DELTA,
    ACTION_DO_NOTHING, ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5,
    ACTION_BUY_CALL_OTM10, ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5,
    ACTION_BUY_PUT_OTM10, ACTION_EXIT, NUM_ACTIONS,
)

# Path to pre-computed features (matches training exactly)
DATA_PT_PATH = os.path.join(CACHE_DIR, "features", "data.pt")

# ---------------------------------------------------------------------------
# Feature groups (must match train.py)
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    'returns':   (0, 5),
    'volume':    (5, 8),
    'vol':       (8, 11),
    'vwap':      (11, 17),
    'session':   (17, 22),
    'levels':    (22, 28),
    'trend':     (28, 31),
    'micro':     (31, 33),
    'time':      (33, 39),
    'options':   (39, 45),
    'vix':       (45, 49),
    'otm':       (49, 55),
    'greeks':    (55, 60),
}

# ---------------------------------------------------------------------------
# Model architecture (copied from train.py — can't import due to module-level execution)
# ---------------------------------------------------------------------------

class FeatureGroupGating(nn.Module):
    """Learn which feature groups matter per timestep."""

    def __init__(self, num_features, d_model, groups):
        super().__init__()
        self.groups = groups
        self.n_groups = len(groups)
        self.gate_net = nn.Sequential(
            nn.Linear(num_features, self.n_groups * 2),
            nn.GELU(),
            nn.Linear(self.n_groups * 2, self.n_groups),
            nn.Sigmoid(),
        )
        self.group_projs = nn.ModuleDict()
        for name, (start, end) in groups.items():
            self.group_projs[name] = nn.Linear(end - start, d_model)
        self.mix = nn.Linear(d_model * self.n_groups, d_model)

    def forward(self, x):
        gates = self.gate_net(x)
        projected = []
        for i, (name, (start, end)) in enumerate(self.groups.items()):
            proj = self.group_projs[name](x[:, :, start:end])
            proj = proj * gates[:, :, i:i+1]
            projected.append(proj)
        return self.mix(torch.cat(projected, dim=-1))


class TradingModel(nn.Module):
    """Two-head sniper model for SPX 0DTE options."""

    def __init__(self, num_features=NUM_FEATURES, lookback=120,
                 d_model=64, n_heads=4, n_layers=4,
                 ff_mult=4, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.feature_gate = FeatureGroupGating(num_features, d_model, FEATURE_GROUPS)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult, dropout=dropout,
            batch_first=True, activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        mask = nn.Transformer.generate_square_subsequent_mask(lookback)
        self.register_buffer('causal_mask', mask)

        self.gate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self.dir_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6),
        )
        with torch.no_grad():
            self.gate_head[-1].bias[0] = -0.5

    def forward(self, x):
        x = self.feature_gate(x)
        x = self.input_norm(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x, mask=self.causal_mask[:x.size(1), :x.size(1)],
                              is_causal=True)
        last = x[:, -1, :]
        return self.gate_head(last), self.dir_head(last)


# ---------------------------------------------------------------------------
# Action constants and names
# ---------------------------------------------------------------------------

_CALL_ACTIONS = {ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5, ACTION_BUY_CALL_OTM10}
_ENTRY_ACTIONS = {ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5, ACTION_BUY_CALL_OTM10,
                  ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5, ACTION_BUY_PUT_OTM10}

ACTION_NAMES = {
    ACTION_DO_NOTHING: 'DO_NOTHING',
    ACTION_BUY_CALL_ATM: 'BUY_CALL_ATM',
    ACTION_BUY_CALL_OTM5: 'BUY_CALL_OTM5',
    ACTION_BUY_CALL_OTM10: 'BUY_CALL_OTM10',
    ACTION_BUY_PUT_ATM: 'BUY_PUT_ATM',
    ACTION_BUY_PUT_OTM5: 'BUY_PUT_OTM5',
    ACTION_BUY_PUT_OTM10: 'BUY_PUT_OTM10',
    ACTION_EXIT: 'EXIT',
}

DIR_NAMES = ['C_ATM', 'C_OTM5', 'C_OTM10', 'P_ATM', 'P_OTM5', 'P_OTM10']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_class_from_train_py(train_py_path: str):
    """Dynamically load model class from a train.py file.

    The autoresearch loop saves the winning architecture as best_train.py.
    This function extracts class definitions from that file so we can
    instantiate the exact architecture that was trained.
    """
    import ast as _ast

    with open(train_py_path, 'r') as f:
        source = f.read()

    # Parse AST and extract only imports + class definitions
    tree = _ast.parse(source)
    class_lines = set()
    for node in _ast.walk(tree):
        if isinstance(node, (
            _ast.ClassDef, _ast.Import, _ast.ImportFrom, _ast.FunctionDef
        )):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                # Skip indented imports (inside if blocks, functions, etc.)
                if isinstance(node, (_ast.Import, _ast.ImportFrom)) and node.col_offset > 0:
                    continue
                for ln in range(node.lineno, node.end_lineno + 1):
                    class_lines.add(ln)

    lines = source.split('\n')

    # Also grab top-level assignments (constants like D_MODEL, FEATURE_GROUPS)
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.Assign, _ast.AugAssign)):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                for ln in range(node.lineno, node.end_lineno + 1):
                    class_lines.add(ln)

    # Build the extractable source (imports + constants + classes)
    extracted = []
    for i, line in enumerate(lines, 1):
        if i in class_lines:
            extracted.append(line)
        else:
            extracted.append('')  # preserve line numbers for error messages

    exec_source = '\n'.join(extracted)

    # Execute in a namespace with required modules
    namespace = {
        'torch': torch, 'nn': nn, 'np': np,
        'F': torch.nn.functional,
        'math': __import__('math'),
        'os': os,
        '__file__': train_py_path,
        'NUM_FEATURES': NUM_FEATURES,
        'BARS_PER_DAY': BARS_PER_DAY,
    }

    # Add prepare imports that train.py typically uses
    try:
        from prepare import (
            NUM_FEATURES as _nf, FEATURE_NAMES, ANNUAL_TRADING_BARS,
            BARS_PER_DAY as _bpd, NUM_ACTIONS,
            ACTION_DO_NOTHING as _a0, ACTION_BUY_CALL_ATM as _a1,
            ACTION_BUY_CALL_OTM5 as _a2, ACTION_BUY_CALL_OTM10 as _a3,
            ACTION_BUY_PUT_ATM as _a4, ACTION_BUY_PUT_OTM5 as _a5,
            ACTION_BUY_PUT_OTM10 as _a6, ACTION_EXIT as _a7,
            ACTION_BUY_CALL as _ac, ACTION_BUY_PUT as _ap,
        )
        namespace.update({
            'FEATURE_NAMES': FEATURE_NAMES,
            'ANNUAL_TRADING_BARS': ANNUAL_TRADING_BARS,
            'NUM_ACTIONS': NUM_ACTIONS,
            'ACTION_DO_NOTHING': _a0, 'ACTION_BUY_CALL_ATM': _a1,
            'ACTION_BUY_CALL_OTM5': _a2, 'ACTION_BUY_CALL_OTM10': _a3,
            'ACTION_BUY_PUT_ATM': _a4, 'ACTION_BUY_PUT_OTM5': _a5,
            'ACTION_BUY_PUT_OTM10': _a6, 'ACTION_EXIT': _a7,
            'ACTION_BUY_CALL': _ac, 'ACTION_BUY_PUT': _ap,
        })
    except ImportError:
        pass

    try:
        exec(exec_source, namespace)
    except Exception as e:
        print(f"WARNING: Could not load classes from {train_py_path}: {e}")
        return None

    # Find the model class (look for TradingModel or any nn.Module subclass)
    model_class = namespace.get('TradingModel')
    if model_class is None:
        # Find any nn.Module subclass that isn't FeatureGroupGating
        for name, obj in namespace.items():
            if (isinstance(obj, type) and issubclass(obj, nn.Module)
                    and obj is not nn.Module
                    and 'Model' in name):
                model_class = obj
                break

    return model_class


def load_model(path: str, device: str = 'cpu', train_py_path: str = None):
    """Load trained model from checkpoint.

    If train_py_path is provided, loads the model class from that file
    (for custom architectures from autoresearch runs).
    """
    if not os.path.exists(path):
        print(f"ERROR: Model not found: {path}")
        sys.exit(1)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']
    metrics = ckpt.get('metrics', {})

    # Try loading custom model class from train.py
    model_cls = None
    if train_py_path and os.path.exists(train_py_path):
        print(f"Loading model architecture from: {train_py_path}")
        model_cls = _load_model_class_from_train_py(train_py_path)
        if model_cls:
            print(f"  Found model class: {model_cls.__name__}")

    if model_cls is None:
        # Also try best_train.py next to the model checkpoint
        best_train = os.path.join(os.path.dirname(path), 'best_train.py')
        if os.path.exists(best_train) and train_py_path != best_train:
            print(f"Loading model architecture from: {best_train}")
            model_cls = _load_model_class_from_train_py(best_train)
            if model_cls:
                print(f"  Found model class: {model_cls.__name__}")

    if model_cls is None:
        model_cls = TradingModel
        print("  Using default TradingModel architecture")

    # Instantiate model
    try:
        model = model_cls(
            num_features=config.get('num_features', NUM_FEATURES),
            lookback=config.get('lookback', 120),
            d_model=config.get('d_model', 64),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('depth', 4),
            ff_mult=config.get('ff_mult', 4),
            dropout=config.get('dropout', 0.1),
        )
    except TypeError:
        # Custom model may have different constructor args — try minimal
        model = model_cls()

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    lookback = config.get('lookback', 120)
    score = metrics.get('score', 'N/A')
    print(f"Model loaded: {path}")
    print(f"  d_model={config.get('d_model')}, depth={config.get('depth')}, "
          f"lookback={lookback}, score={score}")

    return model, lookback, config, metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_replay_data(replay_date: str, warmup_days: int = 5,
                         ib_port: int = 4002, no_download: bool = False):
    """Load warm-up data from cache + download replay day data.

    Returns (df, options_data, vix_data, chain_data) — same format as prepare.py main.
    """
    replay_dt = dt.datetime.strptime(replay_date, '%Y-%m-%d')

    # --- Load warm-up from SPY cache ---
    spy_cache = os.path.join(DATA_DIR, "spy_1min.pkl")
    spx_cache = os.path.join(DATA_DIR, "spx_1min.pkl")
    vix_cache = os.path.join(DATA_DIR, "vix_1min.pkl")

    # Determine warm-up date range (go back extra calendar days to get enough trading days)
    warmup_start = (replay_dt - dt.timedelta(days=warmup_days * 2 + 5)).strftime('%Y-%m-%d')

    # Try loading warm-up from cached training data
    spy_warmup = None
    if os.path.exists(spy_cache):
        with open(spy_cache, 'rb') as f:
            cached_spy = pickle.load(f)
        spy_warmup = cached_spy[
            (cached_spy['date'] >= warmup_start) & (cached_spy['date'] < replay_date)
        ].copy()
        if len(spy_warmup) > 0:
            # Keep only last N trading days for warm-up
            warmup_dates = sorted(spy_warmup['date'].unique())[-warmup_days:]
            spy_warmup = spy_warmup[spy_warmup['date'].isin(warmup_dates)].copy()
            print(f"Warm-up: {len(spy_warmup)} bars from cache "
                  f"({spy_warmup['date'].min()} to {spy_warmup['date'].max()})")
        else:
            spy_warmup = None
            print("Warm-up: no cached data in range, will download")

    # --- Download replay day ---
    if no_download:
        print("No-download mode: using cached data only")
        if spy_warmup is None:
            print("ERROR: No cached data available and --no-download set")
            sys.exit(1)
        # Check if replay date is in cache
        if os.path.exists(spy_cache):
            with open(spy_cache, 'rb') as f:
                cached_spy = pickle.load(f)
            replay_bars = cached_spy[cached_spy['date'] == replay_date].copy()
            if len(replay_bars) > 0:
                df = pd.concat([spy_warmup, replay_bars], ignore_index=True)
            else:
                print(f"ERROR: Replay date {replay_date} not in cache")
                sys.exit(1)
        else:
            print("ERROR: No SPY cache found")
            sys.exit(1)
    else:
        os.environ["IB_PORT"] = str(ib_port)

        # Download replay day from IBKR
        # Use a 2-week window to get the replay day (IBKR requires 1W min)
        dl_start = (replay_dt - dt.timedelta(days=3)).strftime('%Y-%m-%d')
        dl_end = (replay_dt + dt.timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"\nDownloading replay day data from IBKR...")
        spy_new = download_spy_bars_ibkr(dl_start, dl_end)
        spy_replay = spy_new[spy_new['date'] == replay_date].copy()

        if len(spy_replay) == 0:
            print(f"ERROR: No SPY data for {replay_date} — is it a trading day?")
            sys.exit(1)

        print(f"  Replay day: {len(spy_replay)} SPY bars")

        if spy_warmup is not None:
            df = pd.concat([spy_warmup, spy_replay], ignore_index=True)
        else:
            # Use downloaded data as both warmup and replay
            df = spy_new.copy()

    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    # --- SPX price replacement ---
    spx_df = None
    if not no_download:
        dl_start = (replay_dt - dt.timedelta(days=3)).strftime('%Y-%m-%d')
        dl_end = (replay_dt + dt.timedelta(days=1)).strftime('%Y-%m-%d')
        spx_df = _download_ibkr_index("SPX", dl_start, dl_end, col_prefix="spx")

    # Also try SPX cache for warm-up days
    if os.path.exists(spx_cache):
        with open(spx_cache, 'rb') as f:
            cached_spx = pickle.load(f)
        if spx_df is not None:
            spx_df = pd.concat([cached_spx, spx_df], ignore_index=True)
            spx_df = spx_df.drop_duplicates(subset='timestamp').sort_values('timestamp')
        else:
            spx_df = cached_spx

    if spx_df is not None and len(spx_df) > 0:
        pre = len(df)
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
        print(f"  SPX prices merged ({pre} -> {len(df)} bars)")
    else:
        print("  WARNING: No SPX data — using SPY prices")

    # --- VIX data ---
    vix_data = None
    if os.path.exists(vix_cache):
        with open(vix_cache, 'rb') as f:
            vix_data = pickle.load(f)
        print(f"  VIX: loaded {len(vix_data)} cached bars")

    if not no_download:
        # Download VIX for replay day
        dl_start = (replay_dt - dt.timedelta(days=3)).strftime('%Y-%m-%d')
        dl_end = (replay_dt + dt.timedelta(days=1)).strftime('%Y-%m-%d')
        vix_df = download_vix_bars(dl_start, dl_end)
        if vix_df is not None and len(vix_df) > 0:
            if vix_data is None:
                vix_data = {}
            for _, row in vix_df.iterrows():
                vix_data[row['timestamp']] = {
                    'vix_open': row['vix_open'], 'vix_high': row['vix_high'],
                    'vix_low': row['vix_low'], 'vix_close': row['vix_close'],
                }
            print(f"  VIX: added replay day bars (total {len(vix_data)})")

    # --- SPXW options via IBKR (same data source as live trading) ---
    options_data = None
    chain_data = None

    if not no_download:
        print(f"  Downloading SPXW options via IBKR...")
        try:
            download_spxw_ibkr(df, dates=[replay_date])
        except Exception as e:
            print(f"  IBKR option download failed: {e}")
            print(f"  Replay will use delta-estimated P&L")

    options_data = load_spxw_caches(df)
    chain_data = load_spxw_chain_caches(df)

    print(f"\nData ready: {len(df)} bars, {df['date'].nunique()} days "
          f"({replay_date} = {len(df[df['date'] == replay_date])} bars)")

    return df, options_data, vix_data, chain_data


# ---------------------------------------------------------------------------
# Training feature loader (ensures replay sees same features as training)
# ---------------------------------------------------------------------------

def load_training_features(replay_date):
    """Load pre-computed features from data.pt for a date that's in the training set.

    Returns (features_t, raw_features, dates, valid, option_prices, timestamps)
    matching the same interface as compute_features + normalize_features,
    or None if the date is not found in data.pt.
    """
    if not os.path.exists(DATA_PT_PATH):
        return None

    print(f"  Checking data.pt for {replay_date}...")
    data = torch.load(DATA_PT_PATH, map_location='cpu', weights_only=False)

    all_dates = data['dates']  # list of date strings, length = total bars

    # Check if replay_date exists in data.pt
    date_indices = [i for i, d in enumerate(all_dates) if d == replay_date]
    if not date_indices:
        print(f"  {replay_date} not in data.pt — using live-computed features")
        return None

    print(f"  Found {len(date_indices)} bars for {replay_date} in data.pt")

    # We need enough context before the replay day for the lookback window.
    # Load ALL data up to and including the replay day to preserve
    # the global indexing that the model was trained with.
    last_idx = max(date_indices)
    # We need bars from the start for proper lookback context
    features_np = data['features'].numpy()  # already normalized
    valid_np = data['valid_mask'].numpy()
    dates_list = list(all_dates)
    timestamps_list = list(data['timestamps'])

    # Truncate to end at last bar of replay day (no future data)
    end = last_idx + 1
    features_np = features_np[:end]
    valid_np = valid_np[:end]
    dates_list = dates_list[:end]
    timestamps_list = timestamps_list[:end]

    # Build option_prices dict from data.pt
    option_prices = {}
    for key in ['atm_call_prices', 'atm_put_prices',
                'otm5_call_prices', 'otm5_put_prices',
                'otm10_call_prices', 'otm10_put_prices',
                'atm_strikes']:
        if key in data:
            arr = data[key].numpy()[:end]
            option_prices[key.replace('_prices', '').replace('_', '_')] = arr

    # Remap to match the naming convention replay expects
    px_remap = {
        'atm_call': option_prices.get('atm_call'),
        'atm_put': option_prices.get('atm_put'),
        'otm5_call': option_prices.get('otm5_call'),
        'otm5_put': option_prices.get('otm5_put'),
        'otm10_call': option_prices.get('otm10_call'),
        'otm10_put': option_prices.get('otm10_put'),
        'atm_strikes': option_prices.get('atm_strikes'),
    }

    features_t = torch.tensor(features_np, dtype=torch.float32)

    print(f"  Loaded {len(features_np)} bars from data.pt (pre-normalized, matches training)")

    return features_t, features_np, dates_list, valid_np, px_remap, timestamps_list


def _load_norm_context():
    """Load raw feature buffer from data.pt for normalization continuity.

    Returns (context_raw, context_valid) or None if not available.
    """
    if not os.path.exists(DATA_PT_PATH):
        return None
    data = torch.load(DATA_PT_PATH, map_location='cpu', weights_only=False)
    if 'norm_raw_buffer' not in data:
        return None
    ctx_raw = data['norm_raw_buffer'].numpy()
    ctx_valid = data['norm_valid_buffer'].numpy()
    window = data.get('norm_window', len(ctx_raw))
    print(f"  Loaded normalization context: {len(ctx_raw)} bars (window={window})")
    return ctx_raw, ctx_valid


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

def _format_time(ts):
    """Format a timestamp (ms epoch or string) to HH:MM ET."""
    if isinstance(ts, (int, float, np.integer, np.floating)):
        bar_dt = dt.datetime.fromtimestamp(int(ts) / 1000, tz=dt.timezone.utc)
        bar_dt = bar_dt.astimezone(dt.timezone(dt.timedelta(hours=-4)))
        return bar_dt.strftime('%H:%M')
    return str(ts)[:5]


def _safe_float(arr, idx):
    """Safely extract a float from a numpy array, returning None on NaN."""
    if arr is None:
        return None
    val = float(arr[idx])
    return val if not np.isnan(val) else None


def run_replay(model, features_t, raw_features, dates, valid, option_prices,
               timestamps, replay_date, lookback, raw_df,
               speed=0, verbose=False, device='cpu'):
    """Run bar-by-bar inference and trade simulation for the replay day.

    Args:
        raw_features: un-normalized features array (for reading raw market values)
        raw_df: original DataFrame with close/high/low/volume columns

    Returns list of trade dicts with full market context.
    """
    model.eval()
    features = features_t.to(device)

    # Raw market arrays from DataFrame
    raw_close = raw_df['close'].values.astype(np.float64)
    raw_high = raw_df['high'].values.astype(np.float64)
    raw_low = raw_df['low'].values.astype(np.float64)
    raw_volume = raw_df['volume'].values.astype(np.float64)

    # Find replay day indices
    replay_indices = [i for i in range(len(dates)) if dates[i] == replay_date]
    if not replay_indices:
        print(f"ERROR: No bars for {replay_date} in feature data")
        return []

    # Filter to valid bars with enough lookback
    valid_indices = [
        i for i in replay_indices
        if i >= lookback and valid[i] and valid[max(0, i - lookback):i].all()
    ]

    if not valid_indices:
        print(f"No valid bars with sufficient lookback for {replay_date}")
        return []

    # Session open/high/low for the replay day
    day_open = raw_close[replay_indices[0]] if replay_indices else None

    print(f"\n{'='*60}")
    print(f"  REPLAY: {replay_date} ({dt.datetime.strptime(replay_date, '%Y-%m-%d').strftime('%A')})")
    print(f"  Bars: {len(valid_indices)} valid (of {len(replay_indices)} total)")
    print(f"  Speed: {'instant' if speed == 0 else f'{speed}x'}")
    if day_open:
        print(f"  Open: {day_open:.2f}")
    print(f"{'='*60}\n")

    # Price arrays for P&L computation
    atm_call_px = option_prices.get('atm_call') if option_prices else None
    atm_put_px = option_prices.get('atm_put') if option_prices else None
    atm_strikes = option_prices.get('atm_strikes') if option_prices else None

    def get_px_array(action):
        if option_prices is None:
            return None
        mapping = {
            ACTION_BUY_CALL_ATM: option_prices.get('atm_call'),
            ACTION_BUY_CALL_OTM5: option_prices.get('otm5_call'),
            ACTION_BUY_CALL_OTM10: option_prices.get('otm10_call'),
            ACTION_BUY_PUT_ATM: option_prices.get('atm_put'),
            ACTION_BUY_PUT_OTM5: option_prices.get('otm5_put'),
            ACTION_BUY_PUT_OTM10: option_prices.get('otm10_put'),
        }
        return mapping.get(action)

    # VIX feature index = 45 (vix_level in raw features)
    VIX_FEAT_IDX = 45

    # Pre-compute bar_of_day for replay indices (0=9:30, 29=9:59, 30=10:00)
    _bar_of_day = {}
    _prev_date = None
    _bod = 0
    for gi in replay_indices:
        d = dates[gi]
        if d != _prev_date:
            _bod = 0
            _prev_date = d
        else:
            _bod += 1
        _bar_of_day[gi] = _bod

    # Trade state
    trades = []
    bar_log = []  # per-bar decisions log
    in_trade = False
    trade_entry_k = 0
    trade_action = 0
    trade_entry_price = 0.0
    trade_use_actual = False
    trade_px_array = None
    trade_entry_gate_prob = 0.0
    trade_entry_dir_probs = None
    cum_pnl = 0.0
    last_stop_k = -STOP_COOLDOWN_BARS  # initialize so first entry isn't blocked
    cooldown_blocked = 0
    pre_10am_blocked = 0

    offsets = torch.arange(-lookback, 0, device=device)

    for k_pos, global_idx in enumerate(valid_indices):
        # --- Get model prediction ---
        idx_t = torch.tensor([global_idx], dtype=torch.long, device=device)
        window_idx = idx_t.unsqueeze(1) + offsets.unsqueeze(0)
        x = features[window_idx]

        with torch.no_grad():
            gate_logits, dir_logits = model(x)

        gate_probs = torch.softmax(gate_logits, dim=-1)[0].cpu().numpy()
        dir_probs = torch.softmax(dir_logits, dim=-1)[0].cpu().numpy()
        gate_action = int(torch.argmax(gate_logits, dim=-1)[0].item())
        dir_action = int(torch.argmax(dir_logits, dim=-1)[0].item())

        # Decode action
        if gate_action == 1:  # TRADE
            action = dir_action + 1  # ACTION_BUY_CALL_ATM=1 through ACTION_BUY_PUT_OTM10=6
        else:  # NO_TRADE
            action = ACTION_EXIT if in_trade else ACTION_DO_NOTHING

        # Market snapshot at this bar
        time_str = _format_time(timestamps[global_idx] if timestamps is not None else '')
        spx_now = float(raw_close[global_idx])
        spx_high = float(raw_high[global_idx])
        spx_low = float(raw_low[global_idx])
        vol_now = float(raw_volume[global_idx])
        strike_now = _safe_float(atm_strikes, global_idx)

        # VIX from raw (un-normalized) features
        vix_now = float(raw_features[global_idx, VIX_FEAT_IDX]) if not np.isnan(raw_features[global_idx, VIX_FEAT_IDX]) else None

        # Session high/low up to this bar
        session_bars_so_far = [j for j in replay_indices if j <= global_idx]
        session_high = float(np.max(raw_high[session_bars_so_far]))
        session_low = float(np.min(raw_low[session_bars_so_far]))

        # 5-min return (from 5 bars ago)
        ret_5 = float(raw_features[global_idx, 0]) if not np.isnan(raw_features[global_idx, 0]) else None
        # 30-min return
        ret_30 = float(raw_features[global_idx, 2]) if not np.isnan(raw_features[global_idx, 2]) else None
        # Trend structure (HH/HL)
        trend = float(raw_features[global_idx, 28]) if not np.isnan(raw_features[global_idx, 28]) else None
        # Realized vol
        rvol = float(raw_features[global_idx, 9]) if not np.isnan(raw_features[global_idx, 9]) else None
        # Volume z-score
        vol_z = float(raw_features[global_idx, 6]) if not np.isnan(raw_features[global_idx, 6]) else None

        # Log every bar decision
        bar_entry = {
            'time': time_str,
            'spx': spx_now,
            'volume': vol_now,
            'gate_trade_prob': float(gate_probs[1]),
            'top_direction': DIR_NAMES[dir_action],
            'top_dir_prob': float(dir_probs[dir_action]),
            'action': ACTION_NAMES.get(action, '?'),
            'position': 'IN_TRADE' if in_trade else 'FLAT',
        }
        bar_log.append(bar_entry)

        # --- Handle trade exit ---
        if in_trade:
            bars_held = k_pos - trade_entry_k
            entry_global = valid_indices[trade_entry_k]

            # P&L computation
            if trade_use_actual and trade_px_array is not None:
                current_px = float(trade_px_array[global_idx]) if not np.isnan(float(trade_px_array[global_idx])) else np.nan
                if not np.isnan(current_px) and trade_entry_price > 0:
                    net_pnl_pct = (current_px - trade_entry_price) / trade_entry_price
                else:
                    direction = 1.0 if trade_action in _CALL_ACTIONS else -1.0
                    cum_ret = sum(
                        features[valid_indices[j], 0].item()
                        for j in range(trade_entry_k + 1, k_pos + 1)
                        if j < len(valid_indices) and features.dim() == 2
                    )
                    theta = THETA_DECAY_DAILY * BAR_SIZE_MINUTES / (BARS_PER_DAY * BAR_SIZE_MINUTES)
                    net_pnl_pct = direction * cum_ret / ATM_DELTA - theta * bars_held
            else:
                direction = 1.0 if trade_action in _CALL_ACTIONS else -1.0
                cum_ret = sum(
                    features[valid_indices[j], 0].item()
                    for j in range(trade_entry_k + 1, k_pos + 1)
                    if j < len(valid_indices) and features.dim() == 2
                )
                theta = THETA_DECAY_DAILY * BAR_SIZE_MINUTES / (BARS_PER_DAY * BAR_SIZE_MINUTES)
                net_pnl_pct = direction * cum_ret / ATM_DELTA - theta * bars_held

            hit_stop = net_pnl_pct <= -STOP_LOSS_PCT
            hit_max_hold = bars_held >= MAX_HOLD_BARS
            model_exit = (action == ACTION_EXIT)
            is_last = (k_pos == len(valid_indices) - 1)

            if hit_stop or hit_max_hold or model_exit or is_last:
                final_pnl = -STOP_LOSS_PCT if hit_stop else net_pnl_pct
                final_pnl -= OPTION_SPREAD_BPS / 10000.0 * 2  # spread cost

                if hit_stop:
                    reason = 'STOP_LOSS'
                elif model_exit:
                    reason = 'MODEL_EXIT'
                elif hit_max_hold:
                    reason = 'MAX_HOLD'
                else:
                    reason = 'EOD'

                cum_pnl += final_pnl
                result = 'WIN' if final_pnl > 0 else 'LOSS'

                entry_time_str = _format_time(timestamps[entry_global])
                entry_strike = _safe_float(atm_strikes, entry_global)

                # SPX at entry vs exit
                spx_entry = float(raw_close[entry_global])
                spx_exit = spx_now
                spx_move = spx_exit - spx_entry
                spx_move_pct = (spx_exit / spx_entry - 1.0) * 100

                # VIX at entry
                vix_entry = float(raw_features[entry_global, VIX_FEAT_IDX]) if not np.isnan(raw_features[entry_global, VIX_FEAT_IDX]) else None

                # Volume during trade
                trade_bar_indices = valid_indices[trade_entry_k:k_pos + 1]
                trade_volume = float(np.sum(raw_volume[trade_bar_indices]))
                avg_bar_volume = float(np.mean(raw_volume[trade_bar_indices]))

                # SPX high/low during trade
                trade_spx_high = float(np.max(raw_high[trade_bar_indices]))
                trade_spx_low = float(np.min(raw_low[trade_bar_indices]))

                # Max favorable / max adverse excursion (SPX points from entry)
                if trade_action in _CALL_ACTIONS:
                    mfe = trade_spx_high - spx_entry
                    mae = spx_entry - trade_spx_low
                else:
                    mfe = spx_entry - trade_spx_low
                    mae = trade_spx_high - spx_entry

                # Option price at exit (if actual)
                exit_option_px = None
                if trade_use_actual and trade_px_array is not None:
                    exit_option_px = _safe_float(trade_px_array, global_idx)

                # Exit gate probability (model's confidence in exiting)
                exit_gate_notrade_prob = float(gate_probs[0])

                trade = {
                    # Identity
                    'num': len(trades) + 1,
                    'date': replay_date,
                    'result': result,
                    # Timing
                    'entry_time': entry_time_str,
                    'exit_time': time_str,
                    'bars_held': bars_held,
                    'hold_min': bars_held * BAR_SIZE_MINUTES,
                    'reason': reason,
                    # Direction & strike
                    'direction': ACTION_NAMES.get(trade_action, '?'),
                    'strike': entry_strike,
                    # Option prices
                    'entry_option_px': trade_entry_price if trade_use_actual else None,
                    'exit_option_px': exit_option_px,
                    'actual_px': trade_use_actual,
                    # P&L
                    'pnl_pct': round(final_pnl * 100, 2),
                    'cum_pnl_pct': round(cum_pnl * 100, 2),
                    # SPX context
                    'spx_entry': round(spx_entry, 2),
                    'spx_exit': round(spx_exit, 2),
                    'spx_move': round(spx_move, 2),
                    'spx_move_pct': round(spx_move_pct, 3),
                    'spx_high_during': round(trade_spx_high, 2),
                    'spx_low_during': round(trade_spx_low, 2),
                    'mfe_points': round(mfe, 2),
                    'mae_points': round(mae, 2),
                    # Volume context
                    'total_volume_during': int(trade_volume),
                    'avg_bar_volume': int(avg_bar_volume),
                    # VIX context
                    'vix_entry': round(vix_entry, 2) if vix_entry else None,
                    'vix_exit': round(vix_now, 2) if vix_now else None,
                    # Session context
                    'session_high': round(session_high, 2),
                    'session_low': round(session_low, 2),
                    'day_open': round(day_open, 2) if day_open else None,
                    # Model confidence
                    'entry_gate_prob': round(trade_entry_gate_prob, 4),
                    'entry_dir_probs': {DIR_NAMES[i]: round(float(trade_entry_dir_probs[i]), 4)
                                        for i in range(len(DIR_NAMES))} if trade_entry_dir_probs is not None else None,
                    'exit_gate_notrade_prob': round(exit_gate_notrade_prob, 4),
                    # Market regime at entry
                    'entry_trend': trade_entry_trend,
                    'entry_rvol': trade_entry_rvol,
                    'entry_vol_z': trade_entry_vol_z,
                    'entry_ret_5': trade_entry_ret_5,
                    'entry_ret_30': trade_entry_ret_30,
                }
                trades.append(trade)

                # Print trade exit
                pnl_sign = '+' if final_pnl >= 0 else ''
                cum_sign = '+' if cum_pnl >= 0 else ''
                print(f"  {time_str}  {reason:<12} P&L={pnl_sign}{final_pnl*100:.1f}%  "
                      f"held={bars_held}min  cumP&L={cum_sign}{cum_pnl*100:.1f}%  [{result}]")
                print(f"         SPX {spx_entry:.2f} -> {spx_exit:.2f} ({spx_move:+.2f}pts)  "
                      f"MFE={mfe:+.1f}  MAE={mae:.1f}")
                if not trade_use_actual:
                    print(f"         (delta-estimated P&L — no option prices available)")

                if reason == 'STOP_LOSS':
                    last_stop_k = k_pos
                in_trade = False

        # --- Handle trade entry ---
        if not in_trade and action in _ENTRY_ACTIONS:
            # Cooldown after stop loss (matches evaluate_trades)
            if (k_pos - last_stop_k) < STOP_COOLDOWN_BARS:
                cooldown_blocked += 1
                if verbose:
                    print(f"  {time_str}  BLOCKED (cooldown {k_pos - last_stop_k}/{STOP_COOLDOWN_BARS} bars after stop)")
                continue
            # No entries before 10:00 AM (matches evaluate_trades)
            if _bar_of_day.get(global_idx, 999) < NO_TRADE_BEFORE_BAR:
                pre_10am_blocked += 1
                if verbose:
                    print(f"  {time_str}  BLOCKED (pre-10am, bar {_bar_of_day.get(global_idx, 0)} < {NO_TRADE_BEFORE_BAR})")
                continue
            in_trade = True
            trade_entry_k = k_pos
            trade_action = action
            trade_use_actual = False
            trade_entry_price = 0.0
            trade_px_array = get_px_array(trade_action)
            trade_entry_gate_prob = float(gate_probs[1])
            trade_entry_dir_probs = dir_probs.copy()
            # Capture market regime at entry
            trade_entry_trend = trend
            trade_entry_rvol = round(rvol, 6) if rvol is not None else None
            trade_entry_vol_z = round(vol_z, 2) if vol_z is not None else None
            trade_entry_ret_5 = round(ret_5, 6) if ret_5 is not None else None
            trade_entry_ret_30 = round(ret_30, 6) if ret_30 is not None else None

            if trade_px_array is not None and not np.isnan(float(trade_px_array[global_idx])):
                entry_px = float(trade_px_array[global_idx])
                if entry_px > 0:
                    trade_entry_price = entry_px
                    trade_use_actual = True

            strike_info = f"strike={strike_now}" if strike_now else ""
            px_info = f"premium=${trade_entry_price:.2f}" if trade_use_actual else "(no option px)"
            gate_info = f"conf={gate_probs[1]:.0%}"
            vix_info = f"VIX={vix_now:.1f}" if vix_now else ""
            print(f"  {time_str}  {ACTION_NAMES[action]:<16}  SPX={spx_now:.2f}  "
                  f"{strike_info}  {px_info}  {gate_info}  {vix_info}")

        # --- Verbose output ---
        if verbose and action == ACTION_DO_NOTHING:
            gate_str = f"gate={gate_probs[1]:.2f}"
            dir_str = ' '.join(f"{DIR_NAMES[i]}:{dir_probs[i]:.2f}" for i in range(len(dir_probs)))
            state = 'IN_TRADE' if in_trade else 'FLAT'
            print(f"  {time_str}  {gate_str} NO_TRADE  [{dir_str}]  {state}  SPX={spx_now:.2f}")

        # Speed control
        if speed > 0:
            time.sleep(1.0 / speed)

    if cooldown_blocked or pre_10am_blocked:
        print(f"\n  Entries blocked: {cooldown_blocked} cooldown, {pre_10am_blocked} pre-10am")

    return trades, bar_log


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(trades, bar_log, replay_date):
    """Print formatted trade summary with full analysis."""
    print(f"\n{'='*70}")
    print(f"  TRADE SUMMARY — {replay_date}")
    print(f"{'='*70}")

    if not trades:
        print("  No trades taken.")
        _print_session_stats(bar_log)
        return

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    print(f"  Trades: {len(trades)}  |  Win rate: {len(wins)}/{len(trades)} "
          f"({100*len(wins)/len(trades):.0f}%)")

    if wins:
        print(f"  Avg winner: +{np.mean(wins):.1f}%  |  Best: +{max(wins):.1f}%")
    if losses:
        print(f"  Avg loser:  {np.mean(losses):.1f}%  |  Worst: {min(losses):.1f}%")

    total = sum(pnls)
    print(f"  Total P&L: {'+' if total >= 0 else ''}{total:.1f}%")

    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.01
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    print(f"  Profit factor: {pf:.2f}")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t['reason']
        reasons[r] = reasons.get(r, 0) + 1
    reason_str = ' | '.join(f"{k}: {v}" for k, v in sorted(reasons.items()))
    print(f"  Exit reasons: {reason_str}")

    # Avg hold time
    avg_hold = np.mean([t['hold_min'] for t in trades])
    print(f"  Avg hold time: {avg_hold:.0f} min")

    # --- Quick trade table ---
    print(f"\n  {'#':<3} {'Time':<12} {'Direction':<16} {'Strike':<7} "
          f"{'Entry':>7} {'P&L':>8} {'SPX Move':>10} {'MFE':>6} {'MAE':>6} {'Reason':<12}")
    print(f"  {'-'*90}")

    for t in trades:
        strike_str = f"{int(t['strike'])}" if t['strike'] else '?'
        px_str = f"${t['entry_option_px']:.2f}" if t['entry_option_px'] else 'est.'
        pnl_str = f"{'+' if t['pnl_pct'] >= 0 else ''}{t['pnl_pct']:.1f}%"
        time_range = f"{t['entry_time']}-{t['exit_time']}"
        spx_str = f"{t['spx_move']:+.1f}pts"
        mfe_str = f"{t['mfe_points']:+.1f}"
        mae_str = f"{t['mae_points']:.1f}"
        print(f"  {t['num']:<3} {time_range:<12} {t['direction']:<16} {strike_str:<7} "
              f"{px_str:>7} {pnl_str:>8} {spx_str:>10} {mfe_str:>6} {mae_str:>6} {t['reason']:<12}")

    # --- Detailed trade journal ---
    print(f"\n{'='*70}")
    print(f"  DETAILED TRADE JOURNAL")
    print(f"{'='*70}")

    for t in trades:
        print(f"\n  --- Trade #{t['num']}: {t['direction']} ---")
        print(f"  Entry: {t['entry_time']}  |  Exit: {t['exit_time']}  |  Held: {t['hold_min']}min  |  {t['reason']}")
        print(f"  Result: {t['result']}  |  P&L: {'+' if t['pnl_pct'] >= 0 else ''}{t['pnl_pct']:.2f}%")

        # SPX context
        print(f"\n  Market at entry:")
        print(f"    SPX:  {t['spx_entry']:.2f}  (day open: {t.get('day_open', '?')})")
        if t.get('vix_entry'):
            print(f"    VIX:  {t['vix_entry']:.1f}", end='')
            if t.get('vix_exit'):
                vix_chg = t['vix_exit'] - t['vix_entry']
                print(f"  ->  {t['vix_exit']:.1f}  ({vix_chg:+.1f} during trade)")
            else:
                print()

        # Trend/regime
        trend_map = {1.0: 'UPTREND (HH+HL)', -1.0: 'DOWNTREND (LH+LL)', 0.0: 'RANGE/CHOP'}
        trend_str = trend_map.get(t.get('entry_trend'), 'N/A')
        print(f"    Trend: {trend_str}")
        if t.get('entry_rvol') is not None:
            print(f"    Realized vol: {t['entry_rvol']:.6f}  |  Volume z-score: {t.get('entry_vol_z', 'N/A')}")
        if t.get('entry_ret_5') is not None:
            print(f"    Recent returns: 5-bar={t['entry_ret_5']*100:.3f}%  30-bar={t.get('entry_ret_30', 0)*100:.3f}%")

        # Price action during trade
        print(f"\n  Price action during trade:")
        print(f"    SPX: {t['spx_entry']:.2f} -> {t['spx_exit']:.2f}  ({t['spx_move']:+.2f}pts / {t['spx_move_pct']:+.3f}%)")
        print(f"    High: {t['spx_high_during']:.2f}  |  Low: {t['spx_low_during']:.2f}  |  Range: {t['spx_high_during'] - t['spx_low_during']:.2f}pts")
        print(f"    MFE: {t['mfe_points']:+.2f}pts  |  MAE: {t['mae_points']:.2f}pts  |  MFE/MAE: {t['mfe_points'] / max(t['mae_points'], 0.01):.2f}")

        # Option prices
        if t.get('actual_px'):
            print(f"\n  Option pricing:")
            print(f"    Strike: {int(t['strike']) if t['strike'] else '?'}  |  Entry: ${t['entry_option_px']:.2f}", end='')
            if t.get('exit_option_px'):
                print(f"  |  Exit: ${t['exit_option_px']:.2f}")
            else:
                print()
        else:
            print(f"\n  Option pricing: delta-estimated (no option bar data)")

        # Volume
        print(f"\n  Volume:")
        print(f"    Total during trade: {t['total_volume_during']:,}  |  Avg/bar: {t['avg_bar_volume']:,}")

        # Model confidence
        print(f"\n  Model confidence:")
        print(f"    Entry gate (TRADE) prob: {t['entry_gate_prob']:.1%}")
        if t.get('entry_dir_probs'):
            probs_sorted = sorted(t['entry_dir_probs'].items(), key=lambda x: x[1], reverse=True)
            probs_str = '  '.join(f"{k}:{v:.1%}" for k, v in probs_sorted)
            print(f"    Direction probs: {probs_str}")
        if t['reason'] == 'MODEL_EXIT':
            print(f"    Exit gate (NO_TRADE) prob: {t['exit_gate_notrade_prob']:.1%}")

        # Session context
        print(f"\n  Session context:")
        print(f"    Session high: {t['session_high']:.2f}  |  Session low: {t['session_low']:.2f}")

    # --- Session activity stats ---
    _print_session_stats(bar_log)


def _print_session_stats(bar_log):
    """Print stats about the model's behavior across the full session."""
    if not bar_log:
        return

    print(f"\n{'='*70}")
    print(f"  SESSION ACTIVITY")
    print(f"{'='*70}")

    gate_probs = [b['gate_trade_prob'] for b in bar_log]
    print(f"  Total bars evaluated: {len(bar_log)}")
    print(f"  Gate (TRADE) probability: avg={np.mean(gate_probs):.1%}  "
          f"max={np.max(gate_probs):.1%}  min={np.min(gate_probs):.1%}")

    # When was the model most tempted to trade but didn't?
    flat_bars = [(b['time'], b['gate_trade_prob'], b['top_direction'], b['spx'])
                 for b in bar_log if b['action'] == 'DO_NOTHING']
    if flat_bars:
        top_near_misses = sorted(flat_bars, key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Near-misses (highest TRADE prob without acting):")
        for t, p, d, spx in top_near_misses:
            print(f"    {t}  gate={p:.1%}  top_dir={d}  SPX={spx:.2f}")

    # Direction preference distribution
    dir_counts = {}
    for b in bar_log:
        d = b['top_direction']
        dir_counts[d] = dir_counts.get(d, 0) + 1
    print(f"\n  Preferred direction across session:")
    for d, c in sorted(dir_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {d}: {c} bars ({100*c/len(bar_log):.0f}%)")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay trading day with trained model")
    parser.add_argument("--date", type=str, required=True,
                        help="Trading day to replay (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (default: best_model.pt)")
    parser.add_argument("--speed", type=int, default=0,
                        help="Replay speed: 0=instant, 1=realtime, 10=10x (default: 0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every bar decision (default: only trades)")
    parser.add_argument("--ib-port", type=int, default=4002,
                        help="IB Gateway port (default: 4002 paper)")
    parser.add_argument("--warmup-days", type=int, default=5,
                        help="Previous trading days for warm-up (default: 5)")
    parser.add_argument("--no-download", action="store_true",
                        help="Use cached data only (no IBKR/S3 downloads)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save trade log to CSV file")
    parser.add_argument("--train-py", type=str, default=None,
                        help="Path to train.py/best_train.py for custom model architecture")
    args = parser.parse_args()

    replay_date = args.date

    # Validate date
    if not is_0dte_day(replay_date):
        dow_name = dt.datetime.strptime(replay_date, '%Y-%m-%d').strftime('%A')
        print(f"WARNING: {replay_date} ({dow_name}) is NOT a 0DTE expiration day")
        print(f"  Pre-May 2022: only Mon/Wed/Fri have 0DTE. Results may be meaningless.")

    # Check if date is in training set
    spy_cache = os.path.join(DATA_DIR, "spy_1min.pkl")
    if os.path.exists(spy_cache):
        with open(spy_cache, 'rb') as f:
            cached = pickle.load(f)
        if replay_date in cached['date'].values:
            print(f"NOTE: {replay_date} is in the training set (not out-of-sample)")

    # Find model
    model_path = args.model
    if model_path is None:
        # Search common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), 'best_model.pt'),
            os.path.join(os.path.dirname(__file__), '..', 'best_model.pt'),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            print("ERROR: No model found. Train first or specify --model path")
            sys.exit(1)

    # Load model
    device = 'cpu'
    model, lookback, config, metrics = load_model(model_path, device,
                                                   train_py_path=args.train_py)

    # Try to load pre-computed features from data.pt (matches training exactly)
    training_data = load_training_features(replay_date)

    # Always download raw data for trade journal context (SPX prices, volume, etc.)
    df, options_data, vix_data, chain_data = download_replay_data(
        replay_date, args.warmup_days, args.ib_port, args.no_download
    )

    if training_data is not None:
        # Use data.pt features for model inference (identical to training)
        features_t, _, dates, valid, option_prices, timestamps = training_data
        print(f"  Using data.pt features for model inference (normalization-matched)")

        # Still compute raw features for trade journal market context
        print(f"\nComputing raw features for trade journal...")
        raw_feat, _, _, _, _, _ = compute_features(df, options_data, vix_data, chain_data)
        raw_features = raw_feat

        # Raw df needs to align with data.pt indexing. We build a minimal
        # raw_df that covers the replay day bars within data.pt's date range.
        # The raw_close/high/low/volume arrays in run_replay come from raw_df,
        # but data.pt features span the full training history (much larger).
        # We need a raw_df that matches data.pt length for global_idx access.
        # Solution: pad raw arrays to match data.pt length, filling non-replay
        # bars with NaN (they won't be accessed for trade journal).
        n_training = len(features_t)
        n_raw = len(df)

        # Find where replay day bars sit in data.pt
        replay_day_indices_dt = [i for i, d in enumerate(dates) if d == replay_date]
        # Find where replay day bars sit in raw df
        replay_day_indices_raw = [i for i in range(len(df)) if df.iloc[i]['date'] == replay_date]

        if len(replay_day_indices_dt) == len(replay_day_indices_raw):
            # Build padded arrays for raw market data aligned to data.pt indices
            padded_close = np.full(n_training, np.nan)
            padded_high = np.full(n_training, np.nan)
            padded_low = np.full(n_training, np.nan)
            padded_volume = np.full(n_training, np.nan)
            padded_raw_features = np.zeros((n_training, raw_features.shape[1]))

            for dt_idx, raw_idx in zip(replay_day_indices_dt, replay_day_indices_raw):
                padded_close[dt_idx] = df.iloc[raw_idx]['close']
                padded_high[dt_idx] = df.iloc[raw_idx]['high']
                padded_low[dt_idx] = df.iloc[raw_idx]['low']
                padded_volume[dt_idx] = df.iloc[raw_idx]['volume']
                if raw_idx < len(raw_features):
                    padded_raw_features[dt_idx] = raw_features[raw_idx]

            # Create a synthetic df with padded arrays
            raw_df = pd.DataFrame({
                'close': padded_close,
                'high': padded_high,
                'low': padded_low,
                'volume': padded_volume,
                'date': dates,
            })
            raw_features = padded_raw_features
        else:
            print(f"  WARNING: Bar count mismatch (data.pt={len(replay_day_indices_dt)}, "
                  f"raw={len(replay_day_indices_raw)}). Falling back to live features.")
            # Fallback to live-computed features
            training_data = None

    if training_data is None:
        # Out-of-sample day or fallback: compute features from downloaded data
        print(f"\nComputing features...")
        features, targets, dates, valid, option_prices, timestamps = compute_features(
            df, options_data, vix_data, chain_data
        )
        raw_features = features.copy()

        # Use normalization context from data.pt if available (ensures continuity
        # with training-era rolling statistics for OOS/live days)
        norm_context = _load_norm_context()
        if norm_context is not None:
            ctx_raw, ctx_valid = norm_context
            print(f"Normalizing with training context ({len(ctx_raw)} bars)...")
            features = normalize_features_with_context(
                features, valid, ctx_raw, ctx_valid
            )
        else:
            print(f"WARNING: No normalization context in data.pt — using standalone normalization")
            print(f"  (Regenerate data.pt with latest prepare.py to fix this)")
            features = normalize_features(features, valid)

        features_t = torch.tensor(features, dtype=torch.float32)
        raw_df = df

    # Run replay
    trades, bar_log = run_replay(
        model, features_t, raw_features, dates, valid, option_prices, timestamps,
        replay_date, lookback, raw_df=raw_df,
        speed=args.speed, verbose=args.verbose, device=device
    )

    # Summary + detailed journal
    print_summary(trades, bar_log, replay_date)

    # Save to CSV
    if args.output and trades:
        import csv, json
        # Flatten entry_dir_probs dict for CSV
        flat_trades = []
        for t in trades:
            ft = dict(t)
            if ft.get('entry_dir_probs'):
                for k, v in ft['entry_dir_probs'].items():
                    ft[f'dir_prob_{k}'] = v
                del ft['entry_dir_probs']
            flat_trades.append(ft)

        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_trades[0].keys())
            writer.writeheader()
            writer.writerows(flat_trades)
        print(f"Trade log saved to {args.output}")

    # Save full trade journal as JSON (richer than CSV)
    if args.output and trades:
        json_path = args.output.replace('.csv', '') + '_journal.json'
        journal = {
            'replay_date': replay_date,
            'model_path': model_path,
            'model_config': {k: v for k, v in config.items() if not isinstance(v, (torch.Tensor,))},
            'model_score': metrics.get('score', None),
            'trades': trades,
            'session_stats': {
                'total_bars': len(bar_log),
                'avg_gate_prob': round(float(np.mean([b['gate_trade_prob'] for b in bar_log])), 4),
                'max_gate_prob': round(float(np.max([b['gate_trade_prob'] for b in bar_log])), 4),
            },
            'bar_log': bar_log,
        }
        with open(json_path, 'w') as f:
            json.dump(journal, f, indent=2, default=str)
        print(f"Full journal saved to {json_path}")


if __name__ == "__main__":
    main()
