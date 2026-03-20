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

import os, sys, time, argparse, pickle, json, csv, re, hashlib
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from prepare.py (safe — guarded by __main__)
try:
    from prepare import (
        compute_features, normalize_features, normalize_features_with_context,
        _ib_client, _download_ibkr_index, download_spy_bars_ibkr, download_vix_bars,
        download_spxw_ibkr, load_spxw_caches, load_spxw_chain_caches,
        is_0dte_day, DATA_DIR, CACHE_DIR, NUM_FEATURES, BARS_PER_DAY,
        BAR_SIZE_MINUTES, STOP_LOSS_PCT, MAX_HOLD_BARS,
        OPTION_SPREAD_BPS, STOP_COOLDOWN_BARS, NO_TRADE_BEFORE_BAR,
        STARTING_CAPITAL, SPX_MULTIPLIER,
        ACTION_DO_NOTHING, ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5,
        ACTION_BUY_CALL_OTM10, ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5,
        ACTION_BUY_PUT_OTM10, ACTION_EXIT, NUM_ACTIONS,
    )
except ModuleNotFoundError as e:
    if e.name != "prepare":
        raise
    from training.prepare import (
        compute_features, normalize_features, normalize_features_with_context,
        _ib_client, _download_ibkr_index, download_spy_bars_ibkr, download_vix_bars,
        download_spxw_ibkr, load_spxw_caches, load_spxw_chain_caches,
        is_0dte_day, DATA_DIR, CACHE_DIR, NUM_FEATURES, BARS_PER_DAY,
        BAR_SIZE_MINUTES, STOP_LOSS_PCT, MAX_HOLD_BARS,
        OPTION_SPREAD_BPS, STOP_COOLDOWN_BARS, NO_TRADE_BEFORE_BAR,
        STARTING_CAPITAL, SPX_MULTIPLIER,
        ACTION_DO_NOTHING, ACTION_BUY_CALL_ATM, ACTION_BUY_CALL_OTM5,
        ACTION_BUY_CALL_OTM10, ACTION_BUY_PUT_ATM, ACTION_BUY_PUT_OTM5,
        ACTION_BUY_PUT_OTM10, ACTION_EXIT, NUM_ACTIONS,
    )

# Path to pre-computed features (matches training exactly)
DATA_PT_PATH = os.path.join(CACHE_DIR, "features", "data.pt")
ET_TZ = ZoneInfo("America/New_York")

DEFAULT_TRADE_CSV_COLUMNS = [
    "num",
    "date",
    "result",
    "entry_time",
    "exit_time",
    "bars_held",
    "hold_min",
    "reason",
    "direction",
    "strike",
    "entry_option_px",
    "exit_option_px",
    "actual_px",
    "pnl_pct",
    "cum_pnl_pct",
    "spx_entry",
    "spx_exit",
    "spx_move",
    "spx_move_pct",
    "spx_high_during",
    "spx_low_during",
    "mfe_points",
    "mae_points",
    "total_volume_during",
    "avg_bar_volume",
    "vix_entry",
    "vix_exit",
    "session_high",
    "session_low",
    "day_open",
    "entry_gate_prob",
    "exit_gate_notrade_prob",
    "entry_trend",
    "entry_rvol",
    "entry_vol_z",
    "entry_ret_5",
    "entry_ret_30",
    "entry_reason_codes",
    "exit_reason_codes",
]

LEDGER_TRADE_COLUMNS = [
    "ledger_version",
    "replay_run_id",
    "replay_date",
    "trade_id",
    "trade_num",
    "entry_time",
    "exit_time",
    "entry_timestamp_ms",
    "exit_timestamp_ms",
    "entry_timestamp_utc",
    "exit_timestamp_utc",
    "bars_held",
    "hold_min",
    "direction",
    "strike",
    "entry_option_px",
    "exit_option_px",
    "actual_px",
    "pnl_pct",
    "cum_pnl_pct",
    "reason",
    "entry_reason_codes_json",
    "exit_reason_codes_json",
    "entry_gate_prob",
    "exit_gate_notrade_prob",
    "entry_confidence",
    "risk_mode",
    "min_trade_prob",
    "entry_stop_price",
    "entry_take_profit_price",
    "exit_stop_price",
    "exit_take_profit_price",
    "risk_updates",
    "spx_entry",
    "spx_exit",
    "spx_move",
    "spx_move_pct",
    "mfe_points",
    "mae_points",
]

LEDGER_BAR_COLUMNS = [
    "ledger_version",
    "replay_run_id",
    "replay_date",
    "bar_seq",
    "global_idx",
    "bar_of_day",
    "time",
    "timestamp_raw",
    "timestamp_ms",
    "timestamp_utc",
    "position",
    "proposed_action",
    "executed_action",
    "gate_trade_prob",
    "gate_notrade_prob",
    "top_direction",
    "top_dir_prob",
    "dir_prob_C_ATM",
    "dir_prob_C_OTM5",
    "dir_prob_C_OTM10",
    "dir_prob_P_ATM",
    "dir_prob_P_OTM5",
    "dir_prob_P_OTM10",
    "spx",
    "volume",
    "policy_gate_reason_codes_json",
    "policy_gate_payload_json",
]

LEDGER_DAY_COLUMNS = [
    "ledger_version",
    "replay_run_id",
    "replay_date",
    "model_path",
    "model_score",
    "risk_mode",
    "min_trade_prob",
    "total_bars",
    "num_trades",
    "wins",
    "losses",
    "win_rate",
    "total_pnl_pct",
    "profit_factor",
    "avg_gate_prob",
    "max_gate_prob",
    "cooldown_blocked",
    "pre_10am_blocked",
    "missing_option_array_blocked",
    "missing_option_price_blocked",
    "exit_reason_counts_json",
    "entry_reason_counts_json",
    "qa_passed",
    "qa_critical_count",
    "qa_warning_count",
]

# ---------------------------------------------------------------------------
# Feature groups (must match train.py)
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    'returns':   (0, 2),
    'volume':    (2, 5),
    'vol':       (5, 8),
    'vwap':      (8, 10),
    'session':   (10, 12),
    'levels':    (12, 14),
    'trend':     (14, 17),
    'micro':     (17, 19),
    'time':      (19, 22),
    'options':   (22, 24),
    'vix':       (24, 26),
    'greeks':    (26, 29),
    'bollinger': (29, 30),
    'range_ext': (30, 32),
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
        self.d_model = d_model
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

        # Position state: [is_holding, bars_held_norm, unrealized_pnl_norm, account_health, loss_streak_frac]
        self.position_proj = nn.Linear(5, d_model // 4)
        self.position_gate_proj = nn.Linear(d_model + d_model // 4, d_model)

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

    def forward(self, x, position_state=None):
        x = self.feature_gate(x)
        x = self.input_norm(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x, mask=self.causal_mask[:x.size(1), :x.size(1)],
                              is_causal=True)
        last = x[:, -1, :]

        if position_state is not None:
            pos_emb = torch.relu(self.position_proj(position_state))
            gate_input = self.position_gate_proj(torch.cat([last, pos_emb], dim=-1))
        else:
            gate_input = last

        return self.gate_head(gate_input), self.dir_head(last)


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

def _load_model_class_from_train_py(train_py_path: str, config: dict | None = None):
    """Dynamically load model class from a train.py file.

    The autoresearch loop saves the winning architecture as best_train.py.
    This function extracts class definitions from that file so we can
    instantiate the exact architecture that was trained.
    """
    import ast as _ast

    with open(train_py_path, 'r') as f:
        source = f.read()

    safe_names: set[str] = {
        "NUM_FEATURES",
        "BARS_PER_DAY",
        "NUM_ACTIONS",
        "ANNUAL_TRADING_BARS",
        "FEATURE_NAMES",
        "ACTION_DO_NOTHING",
        "ACTION_BUY_CALL_ATM",
        "ACTION_BUY_CALL_OTM5",
        "ACTION_BUY_CALL_OTM10",
        "ACTION_BUY_PUT_ATM",
        "ACTION_BUY_PUT_OTM5",
        "ACTION_BUY_PUT_OTM10",
        "ACTION_EXIT",
        # Hyperparameters injected from checkpoint config (defined via _env_int/_env_float)
        "LOOKBACK", "D_MODEL", "N_HEADS", "DEPTH", "FF_MULT", "DROPOUT",
        "USE_RMSNORM", "USE_NO_BIAS", "QUALITY_GATE_STRENGTH",
        "POSITION_STATE_WEIGHT", "DYNAMIC_STOP_STRENGTH",
    }

    def _is_safe_constant_expr(node: _ast.AST) -> bool:
        if isinstance(node, _ast.Constant):
            return True
        if isinstance(node, _ast.Name):
            return node.id in safe_names
        if isinstance(node, (_ast.Tuple, _ast.List, _ast.Set)):
            return all(_is_safe_constant_expr(x) for x in node.elts)
        if isinstance(node, _ast.Dict):
            return all(
                (k is None or _is_safe_constant_expr(k)) and _is_safe_constant_expr(v)
                for k, v in zip(node.keys, node.values)
            )
        if isinstance(node, _ast.UnaryOp):
            return isinstance(node.op, (_ast.UAdd, _ast.USub, _ast.Not)) and _is_safe_constant_expr(node.operand)
        if isinstance(node, _ast.BinOp):
            return (
                isinstance(node.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.FloorDiv, _ast.Mod, _ast.Pow))
                and _is_safe_constant_expr(node.left)
                and _is_safe_constant_expr(node.right)
            )
        if isinstance(node, _ast.BoolOp):
            return all(_is_safe_constant_expr(v) for v in node.values)
        if isinstance(node, _ast.Compare):
            return _is_safe_constant_expr(node.left) and all(
                _is_safe_constant_expr(c) for c in node.comparators
            )
        if isinstance(node, _ast.IfExp):
            return (
                _is_safe_constant_expr(node.test)
                and _is_safe_constant_expr(node.body)
                and _is_safe_constant_expr(node.orelse)
            )
        return False

    def _is_safe_top_level_assign(node: _ast.AST) -> bool:
        if isinstance(node, _ast.Assign):
            if not all(isinstance(t, _ast.Name) for t in node.targets):
                return False
            return _is_safe_constant_expr(node.value)
        if isinstance(node, _ast.AnnAssign):
            if not isinstance(node.target, _ast.Name):
                return False
            return node.value is None or _is_safe_constant_expr(node.value)
        return False

    # Parse AST and extract imports + classes/functions + safe constant assignments.
    # We intentionally skip side-effect assignments such as os.environ[...] writes.
    tree = _ast.parse(source)
    class_lines = set()
    for node in _ast.iter_child_nodes(tree):
        include = False
        if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.Import, _ast.ImportFrom)):
            include = True
            # Skip indented imports (inside if blocks/functions/etc.)
            if isinstance(node, (_ast.Import, _ast.ImportFrom)) and node.col_offset > 0:
                include = False
        elif _is_safe_top_level_assign(node):
            include = True
            if isinstance(node, _ast.Assign):
                for target in node.targets:
                    if isinstance(target, _ast.Name):
                        safe_names.add(target.id)
            elif isinstance(node, _ast.AnnAssign) and isinstance(node.target, _ast.Name):
                safe_names.add(node.target.id)
        if include and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            for ln in range(node.lineno, node.end_lineno + 1):
                class_lines.add(ln)

    lines = source.split("\n")

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
        )
        namespace.update({
            'FEATURE_NAMES': FEATURE_NAMES,
            'ANNUAL_TRADING_BARS': ANNUAL_TRADING_BARS,
            'NUM_ACTIONS': NUM_ACTIONS,
            'ACTION_DO_NOTHING': _a0, 'ACTION_BUY_CALL_ATM': _a1,
            'ACTION_BUY_CALL_OTM5': _a2, 'ACTION_BUY_CALL_OTM10': _a3,
            'ACTION_BUY_PUT_ATM': _a4, 'ACTION_BUY_PUT_OTM5': _a5,
            'ACTION_BUY_PUT_OTM10': _a6, 'ACTION_EXIT': _a7,
        })
    except ImportError:
        pass

    # Inject hyperparameters from checkpoint config so that class default
    # parameter values like `lookback=LOOKBACK` resolve during exec.
    # These are normally set by _env_int/_env_float calls which are
    # excluded by the AST safe-expression filter.
    _hyper_map = {
            'LOOKBACK': ('lookback', 120),
            'D_MODEL': ('d_model', 96),
            'N_HEADS': ('n_heads', 4),
            'DEPTH': ('depth', 6),
            'FF_MULT': ('ff_mult', 4),
            'DROPOUT': ('dropout', 0.1),
            'USE_RMSNORM': ('use_rmsnorm', 0),
            'USE_NO_BIAS': ('use_no_bias', 0),
            'QUALITY_GATE_STRENGTH': ('quality_gate_strength', 0.5),
            'POSITION_STATE_WEIGHT': ('position_state_weight', 1.0),
            'DYNAMIC_STOP_STRENGTH': ('dynamic_stop_strength', 0.5),
            'CHARM_FLOW_STRENGTH': ('charm_flow_strength', 0.3),
    }
    if config:
        for var_name, (config_key, default) in _hyper_map.items():
            if var_name not in namespace:
                namespace[var_name] = config.get(config_key, default)

    # Derived aliases that the AST filter can't extract (they use attribute
    # access like nn.LayerNorm which isn't recognized as a safe expression).
    # NormLayer depends on USE_RMSNORM which is injected above; RMSNorm is
    # defined as a class in the extracted source and will be available after exec.
    # We pre-set NormLayer=LayerNorm, then re-resolve after exec if RMSNorm exists.
    if 'NormLayer' not in namespace:
        namespace['NormLayer'] = nn.LayerNorm

    try:
        exec(exec_source, namespace)
    except Exception as e:
        print(f"WARNING: Could not load classes from {train_py_path}: {e}")
        return None

    # Re-resolve NormLayer now that RMSNorm class may have been defined by exec
    if namespace.get('USE_RMSNORM') and 'RMSNorm' in namespace:
        namespace['NormLayer'] = namespace['RMSNorm']

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
        model_cls = _load_model_class_from_train_py(train_py_path, config=config)
        if model_cls:
            print(f"  Found model class: {model_cls.__name__}")

    if model_cls is None:
        # Also try best_train.py next to the model checkpoint
        best_train = os.path.join(os.path.dirname(path), 'best_train.py')
        if os.path.exists(best_train) and train_py_path != best_train:
            print(f"Loading model architecture from: {best_train}")
            model_cls = _load_model_class_from_train_py(best_train, config=config)
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

    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  New layers (will use random init): {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    model.to(device)
    model.eval()

    lookback = config.get('lookback', 120)
    score = metrics.get('score', 'N/A')
    num_features = _infer_model_num_features(model, config)
    print(f"Model loaded: {path}")
    print(f"  d_model={config.get('d_model')}, depth={config.get('depth')}, "
          f"lookback={lookback}, num_features={num_features}, score={score}")

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
            print("  Replay will require cached option bar data for strict contract mode.")

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


def _infer_model_num_features(model, config):
    cfg_features = config.get("num_features")
    if cfg_features is not None:
        try:
            return int(cfg_features)
        except Exception:
            pass
    try:
        return int(model.feature_gate.gate_net[0].in_features)
    except Exception:
        return int(NUM_FEATURES)


def _align_features_to_model_width(features: np.ndarray, expected_width: int, source_name: str) -> np.ndarray:
    if features.ndim != 2:
        raise RuntimeError(f"{source_name} features must be 2D, got shape={features.shape}")
    width = int(features.shape[1])
    if width < expected_width:
        raise RuntimeError(
            f"{source_name} feature width {width} is smaller than model width {expected_width}"
        )
    if width > expected_width:
        print(
            f"  NOTE: {source_name} feature width {width} > model width {expected_width}; "
            f"truncating to first {expected_width} features"
        )
        return features[:, :expected_width]
    return features


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

def _format_time(ts):
    """Format a timestamp (ms epoch or string) to HH:MM ET."""
    if isinstance(ts, (int, float, np.integer, np.floating)):
        bar_dt = dt.datetime.fromtimestamp(int(ts) / 1000, tz=dt.timezone.utc)
        bar_dt = bar_dt.astimezone(ET_TZ)
        return bar_dt.strftime('%H:%M')
    s = str(ts).strip()
    if not s:
        return ""
    if s.isdigit():
        raw = int(s)
        if raw > 10_000_000_000:  # likely ms epoch
            bar_dt = dt.datetime.fromtimestamp(raw / 1000, tz=dt.timezone.utc)
            return bar_dt.astimezone(ET_TZ).strftime("%H:%M")
    try:
        parsed = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(ET_TZ).strftime("%H:%M")
    except Exception:
        pass
    m = re.search(r"(\d{2}):(\d{2})", s)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return s[:5]


def _safe_float(arr, idx):
    """Safely extract a float from a numpy array, returning None on NaN."""
    if arr is None:
        return None
    val = float(arr[idx])
    return val if not np.isnan(val) else None


def _timestamp_ms(ts, replay_date: str | None = None) -> int | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float, np.integer, np.floating)):
        raw = int(ts)
        if raw > 10_000_000_000:
            return raw
        if raw > 1_000_000_000:
            return raw * 1000
    s = str(ts).strip()
    if not s:
        return None
    if s.isdigit():
        raw = int(s)
        if raw > 10_000_000_000:
            return raw
        if raw > 1_000_000_000:
            return raw * 1000
    try:
        parsed = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ET_TZ)
        return int(parsed.astimezone(dt.timezone.utc).timestamp() * 1000)
    except Exception:
        pass
    try:
        if replay_date and re.fullmatch(r"\d{2}:\d{2}", s):
            parsed = dt.datetime.strptime(f"{replay_date} {s}", "%Y-%m-%d %H:%M").replace(tzinfo=ET_TZ)
            return int(parsed.astimezone(dt.timezone.utc).timestamp() * 1000)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", s):
            parsed = dt.datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=ET_TZ)
            return int(parsed.astimezone(dt.timezone.utc).timestamp() * 1000)
    except Exception:
        return None
    return None


def _timestamp_iso_utc(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    try:
        return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).isoformat()
    except Exception:
        return None


def _canonical_replay_run_id(*, replay_date: str, model_path: str, risk_mode: str, min_trade_prob: float) -> str:
    payload = f"{replay_date}|{os.path.abspath(model_path)}|{risk_mode}|{min_trade_prob:.4f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _json_dump(obj: object) -> str:
    return json.dumps(obj, sort_keys=True)


def _build_ledger_tables(
    *,
    replay_date: str,
    replay_run_id: str,
    trades: list[dict[str, object]],
    bar_log: list[dict[str, object]],
    session_stats: dict[str, object],
    model_path: str,
    model_score: float | None,
    risk_mode: str,
    min_trade_prob: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_rows: list[dict[str, object]] = []
    for t in trades:
        entry_ts_ms = _timestamp_ms(t.get("entry_timestamp_ms") or t.get("entry_time"), replay_date=replay_date)
        exit_ts_ms = _timestamp_ms(t.get("exit_timestamp_ms") or t.get("exit_time"), replay_date=replay_date)
        row = {
            "ledger_version": "replay_ledger_v1",
            "replay_run_id": replay_run_id,
            "replay_date": replay_date,
            "trade_id": str(t.get("trade_id") or f"{replay_run_id}-T{int(t.get('num', 0)):04d}"),
            "trade_num": int(t.get("num", 0)),
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
            "entry_timestamp_ms": entry_ts_ms,
            "exit_timestamp_ms": exit_ts_ms,
            "entry_timestamp_utc": _timestamp_iso_utc(entry_ts_ms),
            "exit_timestamp_utc": _timestamp_iso_utc(exit_ts_ms),
            "bars_held": int(t.get("bars_held", 0)),
            "hold_min": int(t.get("hold_min", 0)),
            "direction": t.get("direction"),
            "strike": t.get("strike"),
            "entry_option_px": t.get("entry_option_px"),
            "exit_option_px": t.get("exit_option_px"),
            "actual_px": bool(t.get("actual_px", False)),
            "pnl_pct": t.get("pnl_pct"),
            "cum_pnl_pct": t.get("cum_pnl_pct"),
            "reason": t.get("reason"),
            "entry_reason_codes_json": _json_dump(t.get("entry_reason_codes", [])),
            "exit_reason_codes_json": _json_dump(t.get("exit_reason_codes", [])),
            "entry_gate_prob": t.get("entry_gate_prob"),
            "exit_gate_notrade_prob": t.get("exit_gate_notrade_prob"),
            "entry_confidence": t.get("entry_confidence"),
            "risk_mode": t.get("risk_mode", risk_mode),
            "min_trade_prob": t.get("min_trade_prob", min_trade_prob),
            "entry_stop_price": t.get("entry_stop_price"),
            "entry_take_profit_price": t.get("entry_take_profit_price"),
            "exit_stop_price": t.get("exit_stop_price"),
            "exit_take_profit_price": t.get("exit_take_profit_price"),
            "risk_updates": int(t.get("risk_updates", 0)),
            "spx_entry": t.get("spx_entry"),
            "spx_exit": t.get("spx_exit"),
            "spx_move": t.get("spx_move"),
            "spx_move_pct": t.get("spx_move_pct"),
            "mfe_points": t.get("mfe_points"),
            "mae_points": t.get("mae_points"),
        }
        trade_rows.append(row)

    bar_rows: list[dict[str, object]] = []
    for i, b in enumerate(bar_log, start=1):
        ts_ms = _timestamp_ms(b.get("timestamp_ms") or b.get("timestamp_raw"), replay_date=replay_date)
        dir_probs = b.get("dir_probs")
        dir_map = dict(dir_probs) if isinstance(dir_probs, dict) else {}
        row = {
            "ledger_version": "replay_ledger_v1",
            "replay_run_id": replay_run_id,
            "replay_date": replay_date,
            "bar_seq": i,
            "global_idx": b.get("global_idx"),
            "bar_of_day": b.get("bar_of_day"),
            "time": b.get("time"),
            "timestamp_raw": b.get("timestamp_raw"),
            "timestamp_ms": ts_ms,
            "timestamp_utc": _timestamp_iso_utc(ts_ms),
            "position": b.get("position"),
            "proposed_action": b.get("action"),
            "executed_action": b.get("executed_action"),
            "gate_trade_prob": b.get("gate_trade_prob"),
            "gate_notrade_prob": b.get("gate_notrade_prob"),
            "top_direction": b.get("top_direction"),
            "top_dir_prob": b.get("top_dir_prob"),
            "dir_prob_C_ATM": dir_map.get("C_ATM"),
            "dir_prob_C_OTM5": dir_map.get("C_OTM5"),
            "dir_prob_C_OTM10": dir_map.get("C_OTM10"),
            "dir_prob_P_ATM": dir_map.get("P_ATM"),
            "dir_prob_P_OTM5": dir_map.get("P_OTM5"),
            "dir_prob_P_OTM10": dir_map.get("P_OTM10"),
            "spx": b.get("spx"),
            "volume": b.get("volume"),
            "policy_gate_reason_codes_json": _json_dump(b.get("policy_gate_reason_codes", [])),
            "policy_gate_payload_json": _json_dump(b.get("policy_gate_payload", {})),
        }
        bar_rows.append(row)

    trades_df = pd.DataFrame(trade_rows, columns=LEDGER_TRADE_COLUMNS)
    bars_df = pd.DataFrame(bar_rows, columns=LEDGER_BAR_COLUMNS)

    wins = int((trades_df["pnl_pct"] > 0).sum()) if not trades_df.empty else 0
    losses = int((trades_df["pnl_pct"] <= 0).sum()) if not trades_df.empty else 0
    total_pnl = float(trades_df["pnl_pct"].sum()) if not trades_df.empty else 0.0
    gross_win = float(trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].sum()) if not trades_df.empty else 0.0
    gross_loss = abs(float(trades_df.loc[trades_df["pnl_pct"] <= 0, "pnl_pct"].sum())) if not trades_df.empty else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    exit_counts: dict[str, int] = {}
    for reason in trades_df["reason"].dropna().astype(str).tolist() if not trades_df.empty else []:
        exit_counts[reason] = exit_counts.get(reason, 0) + 1
    entry_counts: dict[str, int] = {}
    for payload in trades_df["entry_reason_codes_json"].dropna().astype(str).tolist() if not trades_df.empty else []:
        try:
            codes = json.loads(payload)
        except Exception:
            codes = []
        if isinstance(codes, list):
            for c in codes:
                key = str(c)
                entry_counts[key] = entry_counts.get(key, 0) + 1

    day_row = {
        "ledger_version": "replay_ledger_v1",
        "replay_run_id": replay_run_id,
        "replay_date": replay_date,
        "model_path": os.path.abspath(model_path),
        "model_score": model_score,
        "risk_mode": risk_mode,
        "min_trade_prob": min_trade_prob,
        "total_bars": int(session_stats.get("total_bars", len(bars_df))),
        "num_trades": int(len(trades_df)),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / max(len(trades_df), 1)) if len(trades_df) else 0.0,
        "total_pnl_pct": total_pnl,
        "profit_factor": profit_factor,
        "avg_gate_prob": session_stats.get("avg_gate_prob"),
        "max_gate_prob": session_stats.get("max_gate_prob"),
        "cooldown_blocked": int(session_stats.get("cooldown_blocked", 0)),
        "pre_10am_blocked": int(session_stats.get("pre_10am_blocked", 0)),
        "missing_option_array_blocked": int(session_stats.get("missing_option_array_blocked", 0)),
        "missing_option_price_blocked": int(session_stats.get("missing_option_price_blocked", 0)),
        "exit_reason_counts_json": _json_dump(exit_counts),
        "entry_reason_counts_json": _json_dump(entry_counts),
        "qa_passed": True,
        "qa_critical_count": 0,
        "qa_warning_count": 0,
    }
    days_df = pd.DataFrame([day_row], columns=LEDGER_DAY_COLUMNS)
    return trades_df, bars_df, days_df


def _run_replay_qa(
    *,
    trades_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    days_df: pd.DataFrame,
    strategy_params: dict[str, object],
) -> dict[str, object]:
    anomalies: list[dict[str, object]] = []

    def add(severity: str, code: str, message: str, details: dict[str, object] | None = None) -> None:
        anomalies.append(
            {
                "severity": severity,
                "code": code,
                "message": message,
                "details": details or {},
            }
        )

    day = days_df.iloc[0].to_dict() if not days_df.empty else {}
    if int(day.get("num_trades", 0)) != int(len(trades_df)):
        add(
            "critical",
            "day_trade_count_mismatch",
            "Day-level trade count does not match trades ledger rows.",
            {"day_num_trades": int(day.get("num_trades", 0)), "trade_rows": int(len(trades_df))},
        )
    if int(day.get("total_bars", 0)) != int(len(bars_df)):
        add(
            "critical",
            "day_bar_count_mismatch",
            "Day-level bar count does not match bars ledger rows.",
            {"day_total_bars": int(day.get("total_bars", 0)), "bar_rows": int(len(bars_df))},
        )

    if not trades_df.empty:
        if trades_df["trade_id"].duplicated().any():
            add("critical", "duplicate_trade_id", "Duplicate trade_id detected in ledger.")
        nums = trades_df["trade_num"].tolist()
        if nums != sorted(nums):
            add("warning", "non_monotonic_trade_num", "Trade numbers are not monotonic ascending.")
        bad_times = trades_df[
            ~trades_df["entry_time"].astype(str).str.match(r"^\d{2}:\d{2}$")
            | ~trades_df["exit_time"].astype(str).str.match(r"^\d{2}:\d{2}$")
        ]
        if not bad_times.empty:
            add(
                "critical",
                "bad_trade_time_format",
                "Trade rows contain non-HH:MM entry/exit time values.",
                {"rows": bad_times["trade_id"].tolist()},
            )
        bad_ts = trades_df[trades_df["entry_timestamp_ms"].isna() | trades_df["exit_timestamp_ms"].isna()]
        if not bad_ts.empty:
            add(
                "critical",
                "missing_trade_timestamp_ms",
                "Trade rows missing timestamp_ms fields.",
                {"rows": bad_ts["trade_id"].tolist()},
            )
        order_bad = trades_df[trades_df["exit_timestamp_ms"] < trades_df["entry_timestamp_ms"]]
        if not order_bad.empty:
            add(
                "critical",
                "trade_exit_before_entry",
                "One or more trades have exit before entry.",
                {"rows": order_bad["trade_id"].tolist()},
            )

        spread_cost_pct = float(strategy_params.get("option_spread_bps", OPTION_SPREAD_BPS)) / 10000.0 * 2 * 100.0
        stop_loss_pct = float(strategy_params.get("stop_loss_pct", STOP_LOSS_PCT)) * 100.0
        for _, row in trades_df.iterrows():
            entry_px = row.get("entry_option_px")
            exit_px = row.get("exit_option_px")
            if not isinstance(entry_px, (float, int)) or not isinstance(exit_px, (float, int)):
                continue
            if float(entry_px) <= 0:
                continue
            reason = str(row.get("reason") or "")
            risk_mode = str(row.get("risk_mode") or "training")
            if reason == "STOP_LOSS" and risk_mode == "training":
                expected = -(stop_loss_pct + spread_cost_pct)
            else:
                expected = ((float(exit_px) - float(entry_px)) / float(entry_px)) * 100.0 - spread_cost_pct
            actual = float(row.get("pnl_pct", 0.0))
            if abs(expected - actual) > 0.75:
                add(
                    "warning",
                    "trade_pnl_mismatch",
                    "Trade pnl_pct does not match recomputed value within tolerance.",
                    {"trade_id": row.get("trade_id"), "expected": round(expected, 4), "actual": round(actual, 4)},
                )
            hold_min = row.get("hold_min")
            bars_held = row.get("bars_held")
            if isinstance(hold_min, (float, int)) and isinstance(bars_held, (float, int)):
                expected_hold = int(round(float(bars_held) * BAR_SIZE_MINUTES))
                if abs(float(hold_min) - expected_hold) >= 1.0:
                    add(
                        "warning",
                        "trade_hold_duration_mismatch",
                        "hold_min is inconsistent with bars_held * bar_size_minutes.",
                        {
                            "trade_id": row.get("trade_id"),
                            "bars_held": int(float(bars_held)),
                            "hold_min": float(hold_min),
                            "expected_hold_min": expected_hold,
                        },
                    )

        cum = 0.0
        for _, row in trades_df.sort_values("trade_num").iterrows():
            cum += float(row.get("pnl_pct", 0.0))
            actual_cum = float(row.get("cum_pnl_pct", 0.0))
            if abs(cum - actual_cum) > 0.75:
                add(
                    "warning",
                    "cum_pnl_mismatch",
                    "cum_pnl_pct is inconsistent with running sum of pnl_pct.",
                    {"trade_id": row.get("trade_id"), "expected_cum": round(cum, 4), "actual_cum": round(actual_cum, 4)},
                )
        if "total_pnl_pct" in day:
            try:
                expected_total = float(trades_df["pnl_pct"].sum())
                actual_total = float(day.get("total_pnl_pct", 0.0))
                if abs(expected_total - actual_total) > 0.75:
                    add(
                        "warning",
                        "day_total_pnl_mismatch",
                        "Day total_pnl_pct does not match sum of trade pnl_pct.",
                        {"expected_total": round(expected_total, 4), "actual_total": round(actual_total, 4)},
                    )
            except Exception:
                pass

    if not bars_df.empty:
        missing_ts = bars_df[bars_df["timestamp_ms"].isna()]
        if not missing_ts.empty:
            add(
                "critical",
                "missing_bar_timestamp_ms",
                "Bar ledger contains rows without timestamp_ms.",
                {"bar_seq": missing_ts["bar_seq"].tolist()[:20]},
            )
        bad_times = bars_df[~bars_df["time"].astype(str).str.match(r"^\d{2}:\d{2}$")]
        if not bad_times.empty:
            add(
                "critical",
                "bad_bar_time_format",
                "Bar ledger contains non-HH:MM time values.",
                {"bar_seq": bad_times["bar_seq"].tolist()[:20]},
            )
        if bars_df["bar_seq"].duplicated().any():
            add("critical", "duplicate_bar_seq", "Bar ledger contains duplicate bar_seq values.")
        expected_seq = np.arange(1, len(bars_df) + 1)
        got_seq = bars_df["bar_seq"].fillna(-1).astype(int).to_numpy()
        if got_seq.shape[0] == expected_seq.shape[0] and not np.array_equal(got_seq, expected_seq):
            add("warning", "bar_seq_not_contiguous", "bar_seq is not contiguous 1..N.")
        ts_values = bars_df["timestamp_ms"].dropna().astype("int64").to_numpy()
        if ts_values.size >= 2 and np.any(np.diff(ts_values) < 0):
            add("critical", "bar_timestamp_non_monotonic", "Bar timestamps are not monotonic non-decreasing.")
        if bars_df["global_idx"].notna().any() and bars_df["global_idx"].duplicated().any():
            add("warning", "duplicate_global_idx", "Bar ledger contains duplicate global_idx values.")
        for col in ("gate_trade_prob", "gate_notrade_prob"):
            bad_prob = bars_df[(bars_df[col].notna()) & ((bars_df[col] < -1e-6) | (bars_df[col] > 1 + 1e-6))]
            if not bad_prob.empty:
                add(
                    "critical",
                    f"{col}_out_of_range",
                    f"{col} must be within [0, 1].",
                    {"bar_seq": bad_prob["bar_seq"].tolist()[:20]},
                )

    critical_count = sum(1 for a in anomalies if a["severity"] == "critical")
    warning_count = sum(1 for a in anomalies if a["severity"] == "warning")
    passed = critical_count == 0
    return {
        "schema_version": "replay_qa_v1",
        "passed": bool(passed),
        "critical_count": int(critical_count),
        "warning_count": int(warning_count),
        "anomalies": anomalies,
    }


def _write_dataframe_csv_parquet(df: pd.DataFrame, csv_path: str, parquet_path: str) -> None:
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"WARNING: Could not write parquet {parquet_path}: {e}")


def _write_replay_ledger_and_qa(
    *,
    output_csv: str,
    trades_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    days_df: pd.DataFrame,
    qa: dict[str, object],
) -> dict[str, str]:
    base = output_csv[:-4] if output_csv.lower().endswith(".csv") else output_csv
    paths = {
        "ledger_trades_csv": f"{base}_ledger_trades.csv",
        "ledger_trades_parquet": f"{base}_ledger_trades.parquet",
        "ledger_bars_csv": f"{base}_ledger_bars.csv",
        "ledger_bars_parquet": f"{base}_ledger_bars.parquet",
        "ledger_days_csv": f"{base}_ledger_days.csv",
        "ledger_days_parquet": f"{base}_ledger_days.parquet",
        "qa_json": f"{base}_qa.json",
        "qa_anomalies_jsonl": f"{base}_qa_anomalies.jsonl",
    }
    _write_dataframe_csv_parquet(trades_df, paths["ledger_trades_csv"], paths["ledger_trades_parquet"])
    _write_dataframe_csv_parquet(bars_df, paths["ledger_bars_csv"], paths["ledger_bars_parquet"])
    _write_dataframe_csv_parquet(days_df, paths["ledger_days_csv"], paths["ledger_days_parquet"])
    with open(paths["qa_json"], "w") as f:
        json.dump(qa, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(paths["qa_anomalies_jsonl"], "w") as f:
        for row in qa.get("anomalies", []):
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Replay ledger saved: {paths['ledger_trades_csv']}, {paths['ledger_bars_csv']}, {paths['ledger_days_csv']}")
    print(f"Replay QA saved: {paths['qa_json']}")
    return paths


def run_replay(model, features_t, raw_features, dates, valid, option_prices,
               timestamps, replay_date, lookback, raw_df,
               speed=0, verbose=False, device='cpu',
               quiet=False):
    """Run bar-by-bar inference and trade simulation for the replay day.

    Plays the EXACT SAME GAME as training evaluate_trades() and live service:
    - Argmax gate (no threshold)
    - Position state fed to gate head
    - Fixed stop-loss (30%) and take-profit (20%) matching training
    - Cooldown after stop, no entries before 10:00 AM
    - Trades can hold all day (0DTE closes at EOD)

    Returns (trades, bar_log, session_stats).
    """
    model.eval()
    features = features_t.to(device)
    has_position_proj = hasattr(model, 'position_proj')

    raw_close = raw_df['close'].values.astype(np.float64)
    raw_high = raw_df['high'].values.astype(np.float64)
    raw_low = raw_df['low'].values.astype(np.float64)
    raw_volume = raw_df['volume'].values.astype(np.float64)

    replay_indices = [i for i in range(len(dates)) if dates[i] == replay_date]
    empty_stats = {
        "total_bars": 0, "avg_gate_prob": 0.0, "max_gate_prob": 0.0,
        "cooldown_blocked": 0, "pre_10am_blocked": 0,
        "missing_option_array_blocked": 0, "missing_option_price_blocked": 0,
        "num_trades": 0,
    }
    if not replay_indices:
        if not quiet:
            print(f"ERROR: No bars for {replay_date} in feature data")
        return [], [], empty_stats

    valid_indices = [
        i for i in replay_indices
        if i >= lookback and valid[i] and valid[max(0, i - lookback):i].all()
    ]
    if not valid_indices:
        if not quiet:
            print(f"No valid bars with sufficient lookback for {replay_date}")
        return [], [], empty_stats

    day_open = raw_close[replay_indices[0]] if replay_indices else None

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  REPLAY: {replay_date} ({dt.datetime.strptime(replay_date, '%Y-%m-%d').strftime('%A')})")
        print(f"  Bars: {len(valid_indices)} valid (of {len(replay_indices)} total)")
        if day_open:
            print(f"  Open: {day_open:.2f}")
        print(f"{'='*60}\n")

    # Option price arrays (strict contract)
    required_price_keys = ('atm_call', 'atm_put', 'otm5_call', 'otm5_put', 'otm10_call', 'otm10_put')
    if option_prices is None:
        raise ValueError("Replay requires option_prices from data.pt.")
    missing_px = [k for k in required_price_keys if option_prices.get(k) is None]
    if missing_px:
        raise KeyError(f"Replay missing option price arrays: {', '.join(missing_px)}")

    atm_strikes = option_prices.get('atm_strikes')

    def get_px_array(action):
        return {
            ACTION_BUY_CALL_ATM: option_prices.get('atm_call'),
            ACTION_BUY_CALL_OTM5: option_prices.get('otm5_call'),
            ACTION_BUY_CALL_OTM10: option_prices.get('otm10_call'),
            ACTION_BUY_PUT_ATM: option_prices.get('atm_put'),
            ACTION_BUY_PUT_OTM5: option_prices.get('otm5_put'),
            ACTION_BUY_PUT_OTM10: option_prices.get('otm10_put'),
        }.get(action)

    VIX_FEAT_IDX = 45

    # Pre-compute bar_of_day
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

    # Position + trade state
    trades = []
    bar_log = []
    in_trade = False
    trade_entry_k = 0
    trade_action = 0
    trade_entry_price = 0.0
    trade_last_price = 0.0
    trade_px_array = None
    trade_entry_gate_prob = 0.0
    trade_entry_confidence = 0.0
    trade_entry_dir_probs = None
    trade_entry_reason_codes: list[str] = []
    trade_stop_price = 0.0
    trade_take_profit_price = 0.0
    cum_pnl = 0.0
    bars_held = 0
    unrealized_pnl = 0.0
    _trade_entry_trend = None
    _trade_entry_rvol = None
    _trade_entry_vol_z = None
    _trade_entry_ret_5 = None
    _trade_entry_ret_30 = None
    last_stop_k = -STOP_COOLDOWN_BARS
    cooldown_blocked = 0
    pre_10am_blocked = 0
    missing_option_array_blocked = 0
    missing_option_price_blocked = 0
    # Account state tracking for position_state + affordability
    account_balance = STARTING_CAPITAL
    consecutive_losses = 0
    trades_blocked_by_balance = 0

    offsets = torch.arange(-lookback, 0, device=device)

    for k_pos, global_idx in enumerate(valid_indices):
        # --- Build position state tensor (same as training eval + live decision.py) ---
        pos_state = None
        if has_position_proj:
            pos_state = torch.zeros(1, 5, device=device)
            if in_trade:
                pos_state[0, 0] = 1.0
                pos_state[0, 1] = min(bars_held / BARS_PER_DAY, 1.0)
                pos_state[0, 2] = float(np.tanh(unrealized_pnl * 5.0))
            pos_state[0, 3] = account_balance / STARTING_CAPITAL  # account_health
            pos_state[0, 4] = min(consecutive_losses / 3.0, 1.0)  # loss_streak_frac

        # --- Get model prediction ---
        idx_t = torch.tensor([global_idx], dtype=torch.long, device=device)
        window_idx = idx_t.unsqueeze(1) + offsets.unsqueeze(0)
        x = features[window_idx]

        with torch.no_grad():
            if has_position_proj:
                gate_logits, dir_logits = model(x, position_state=pos_state)
            else:
                gate_logits, dir_logits = model(x)

        gate_probs_t = torch.softmax(gate_logits, dim=-1)[0].cpu().numpy()
        dir_probs = torch.softmax(dir_logits, dim=-1)[0].cpu().numpy()
        gate_action = int(torch.argmax(gate_logits, dim=-1)[0].item())  # 0=NO_TRADE, 1=TRADE
        dir_action = int(np.argmax(dir_probs))
        gate_trade_prob = float(gate_probs_t[1])
        gate_notrade_prob = float(gate_probs_t[0])

        # Argmax gate — same as training evaluate_trades(). No hardcoded threshold.
        if gate_action == 1:  # TRADE
            action = dir_action + 1
        else:  # NO_TRADE
            action = ACTION_DO_NOTHING

        # Market snapshot
        ts_raw = timestamps[global_idx] if timestamps is not None else None
        time_str = _format_time(ts_raw if ts_raw is not None else '')
        ts_ms = _timestamp_ms(ts_raw, replay_date=replay_date)
        spx_now = float(raw_close[global_idx])
        spx_high = float(raw_high[global_idx])
        spx_low = float(raw_low[global_idx])
        vol_now = float(raw_volume[global_idx])
        strike_now = _safe_float(atm_strikes, global_idx)
        vix_now = float(raw_features[global_idx, VIX_FEAT_IDX]) if raw_features.shape[1] > VIX_FEAT_IDX and not np.isnan(raw_features[global_idx, VIX_FEAT_IDX]) else None

        # Session high/low up to this bar
        session_bars_so_far = [j for j in replay_indices if j <= global_idx]
        session_high = float(np.max(raw_high[session_bars_so_far]))
        session_low = float(np.min(raw_low[session_bars_so_far]))

        # Raw feature snapshots for trade context
        ret_5 = float(raw_features[global_idx, 0]) if not np.isnan(raw_features[global_idx, 0]) else None
        ret_30 = float(raw_features[global_idx, 2]) if raw_features.shape[1] > 2 and not np.isnan(raw_features[global_idx, 2]) else None
        trend = float(raw_features[global_idx, 28]) if raw_features.shape[1] > 28 and not np.isnan(raw_features[global_idx, 28]) else None
        rvol = float(raw_features[global_idx, 9]) if raw_features.shape[1] > 9 and not np.isnan(raw_features[global_idx, 9]) else None
        vol_z = float(raw_features[global_idx, 6]) if raw_features.shape[1] > 6 and not np.isnan(raw_features[global_idx, 6]) else None

        confidence = gate_trade_prob * float(dir_probs[dir_action])

        reason_codes: list[str] = []
        if gate_action == 1:
            reason_codes.append("trade_signal")
        else:
            reason_codes.append("gate_no_trade")

        # Bar log entry
        bar_entry = {
            'time': time_str,
            'timestamp_raw': str(ts_raw) if ts_raw is not None else None,
            'timestamp_ms': ts_ms,
            'global_idx': int(global_idx),
            'bar_of_day': int(_bar_of_day.get(global_idx, -1)),
            'spx': spx_now,
            'volume': vol_now,
            'gate_trade_prob': gate_trade_prob,
            'gate_notrade_prob': gate_notrade_prob,
            'top_direction': DIR_NAMES[dir_action],
            'top_dir_prob': float(dir_probs[dir_action]),
            'dir_probs': {DIR_NAMES[i]: float(dir_probs[i]) for i in range(len(dir_probs))},
            'action': ACTION_NAMES.get(action, '?'),
            'executed_action': ACTION_NAMES.get(action, '?'),
            'position': 'IN_TRADE' if in_trade else 'FLAT',
            'policy_gate_reason_codes': list(reason_codes),
        }
        if in_trade and action in _ENTRY_ACTIONS:
            bar_entry['executed_action'] = 'HOLD_IN_TRADE'
            bar_entry['policy_gate_reason_codes'].append('entry_signal_ignored_in_trade')
        bar_log.append(bar_entry)

        # --- Handle trade exit ---
        if in_trade:
            bars_held = k_pos - trade_entry_k
            entry_global = valid_indices[trade_entry_k]

            px_now = _safe_float(trade_px_array, global_idx)
            if px_now is not None:
                trade_last_price = float(px_now)
            current_px = trade_last_price
            net_pnl_pct = (current_px - trade_entry_price) / trade_entry_price
            unrealized_pnl = net_pnl_pct

            # Exit conditions — matching prepare.py evaluate_trades()
            hit_stop = net_pnl_pct <= -STOP_LOSS_PCT
            hit_max_hold = bars_held >= MAX_HOLD_BARS
            model_exit = (action == ACTION_DO_NOTHING)  # gate=NO_TRADE while holding
            is_last = (k_pos == len(valid_indices) - 1)

            if hit_stop or hit_max_hold or model_exit or is_last:
                if hit_stop:
                    final_pnl = -STOP_LOSS_PCT
                    reason = 'STOP_LOSS'
                elif model_exit:
                    final_pnl = net_pnl_pct
                    reason = 'MODEL_EXIT'
                elif hit_max_hold:
                    final_pnl = net_pnl_pct
                    reason = 'MAX_HOLD'
                else:
                    final_pnl = net_pnl_pct
                    reason = 'EOD'

                final_pnl -= OPTION_SPREAD_BPS / 10000.0 * 2

                # Update account balance (dollar P&L)
                dollar_pnl = final_pnl * trade_entry_price * SPX_MULTIPLIER
                account_balance += dollar_pnl
                account_balance = max(account_balance, 0.0)
                if final_pnl <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                cum_pnl += final_pnl
                result = 'WIN' if final_pnl > 0 else 'LOSS'

                entry_time_str = _format_time(timestamps[entry_global])
                entry_strike = _safe_float(atm_strikes, entry_global)
                spx_entry = float(raw_close[entry_global])
                spx_exit = spx_now
                spx_move = spx_exit - spx_entry
                spx_move_pct = (spx_exit / spx_entry - 1.0) * 100

                vix_entry = float(raw_features[entry_global, VIX_FEAT_IDX]) if raw_features.shape[1] > VIX_FEAT_IDX and not np.isnan(raw_features[entry_global, VIX_FEAT_IDX]) else None

                trade_bar_indices = valid_indices[trade_entry_k:k_pos + 1]
                trade_volume = float(np.sum(raw_volume[trade_bar_indices]))
                avg_bar_volume = float(np.mean(raw_volume[trade_bar_indices]))
                trade_spx_high = float(np.max(raw_high[trade_bar_indices]))
                trade_spx_low = float(np.min(raw_low[trade_bar_indices]))

                if trade_action in _CALL_ACTIONS:
                    mfe = trade_spx_high - spx_entry
                    mae = spx_entry - trade_spx_low
                else:
                    mfe = spx_entry - trade_spx_low
                    mae = trade_spx_high - spx_entry

                trade = {
                    'num': len(trades) + 1,
                    'trade_id': f"{replay_date}-T{len(trades) + 1:04d}",
                    'date': replay_date,
                    'result': result,
                    'entry_time': entry_time_str,
                    'exit_time': time_str,
                    'entry_timestamp_ms': _timestamp_ms(timestamps[entry_global] if timestamps is not None else None, replay_date=replay_date),
                    'exit_timestamp_ms': ts_ms,
                    'bars_held': bars_held,
                    'hold_min': bars_held * BAR_SIZE_MINUTES,
                    'reason': reason,
                    'direction': ACTION_NAMES.get(trade_action, '?'),
                    'strike': entry_strike,
                    'entry_option_px': trade_entry_price,
                    'exit_option_px': trade_last_price,
                    'pnl_pct': round(final_pnl * 100, 2),
                    'cum_pnl_pct': round(cum_pnl * 100, 2),
                    'spx_entry': round(spx_entry, 2),
                    'spx_exit': round(spx_exit, 2),
                    'spx_move': round(spx_move, 2),
                    'spx_move_pct': round(spx_move_pct, 3),
                    'spx_high_during': round(trade_spx_high, 2),
                    'spx_low_during': round(trade_spx_low, 2),
                    'mfe_points': round(mfe, 2),
                    'mae_points': round(mae, 2),
                    'total_volume_during': int(trade_volume),
                    'avg_bar_volume': int(avg_bar_volume),
                    'vix_entry': round(vix_entry, 2) if vix_entry else None,
                    'vix_exit': round(vix_now, 2) if vix_now else None,
                    'session_high': round(session_high, 2),
                    'session_low': round(session_low, 2),
                    'day_open': round(day_open, 2) if day_open else None,
                    'entry_gate_prob': round(trade_entry_gate_prob, 4),
                    'entry_dir_probs': {DIR_NAMES[i]: round(float(trade_entry_dir_probs[i]), 4)
                                        for i in range(len(DIR_NAMES))} if trade_entry_dir_probs is not None else None,
                    'exit_gate_notrade_prob': round(gate_notrade_prob, 4),
                    'entry_confidence': round(float(trade_entry_confidence), 6),
                    'entry_stop_price': round(float(trade_stop_price), 4),
                    'entry_take_profit_price': round(float(trade_take_profit_price), 4),
                    'entry_trend': getattr(trade, 'entry_trend', None) if False else _trade_entry_trend,
                    'entry_rvol': _trade_entry_rvol,
                    'entry_vol_z': _trade_entry_vol_z,
                    'entry_ret_5': _trade_entry_ret_5,
                    'entry_ret_30': _trade_entry_ret_30,
                    'entry_reason_codes': list(trade_entry_reason_codes) if trade_entry_reason_codes else ['trade_signal'],
                    'exit_reason_codes': [reason.lower()],
                }
                trades.append(trade)

                if not quiet:
                    pnl_sign = '+' if final_pnl >= 0 else ''
                    cum_sign = '+' if cum_pnl >= 0 else ''
                    print(f"  {time_str}  {reason:<12} P&L={pnl_sign}{final_pnl*100:.1f}%  "
                          f"held={bars_held}min  cumP&L={cum_sign}{cum_pnl*100:.1f}%  [{result}]")

                if reason == 'STOP_LOSS':
                    last_stop_k = k_pos
                in_trade = False
                bars_held = 0
                unrealized_pnl = 0.0

        # --- Handle trade entry ---
        if not in_trade and action in _ENTRY_ACTIONS:
            if (k_pos - last_stop_k) < STOP_COOLDOWN_BARS:
                cooldown_blocked += 1
                bar_entry['executed_action'] = 'DO_NOTHING'
                bar_entry['policy_gate_reason_codes'].append('blocked_cooldown')
                if verbose and not quiet:
                    print(f"  {time_str}  BLOCKED (cooldown {k_pos - last_stop_k}/{STOP_COOLDOWN_BARS})")
                continue
            if _bar_of_day.get(global_idx, 999) < NO_TRADE_BEFORE_BAR:
                pre_10am_blocked += 1
                bar_entry['executed_action'] = 'DO_NOTHING'
                bar_entry['policy_gate_reason_codes'].append('blocked_pre_10am')
                if verbose and not quiet:
                    print(f"  {time_str}  BLOCKED (pre-10am)")
                continue
            candidate_px_array = get_px_array(action)
            if candidate_px_array is None:
                missing_option_array_blocked += 1
                bar_entry['executed_action'] = 'DO_NOTHING'
                bar_entry['policy_gate_reason_codes'].append('blocked_missing_option_array')
                continue
            entry_px_raw = _safe_float(candidate_px_array, global_idx)
            if entry_px_raw is None or entry_px_raw <= 0:
                missing_option_price_blocked += 1
                bar_entry['executed_action'] = 'DO_NOTHING'
                bar_entry['policy_gate_reason_codes'].append('blocked_missing_option_price')
                continue
            # Affordability check: can't buy what you can't afford
            contract_cost = float(entry_px_raw) * SPX_MULTIPLIER
            if contract_cost > account_balance:
                trades_blocked_by_balance += 1
                bar_entry['executed_action'] = 'DO_NOTHING'
                bar_entry['policy_gate_reason_codes'].append('blocked_insufficient_balance')
                if verbose and not quiet:
                    print(f"  {time_str}  BLOCKED (insufficient balance: ${account_balance:.0f} < ${contract_cost:.0f})")
                continue

            # Execute entry — fixed stop/TP matching training
            in_trade = True
            trade_entry_k = k_pos
            trade_action = action
            trade_entry_price = float(entry_px_raw)
            trade_last_price = float(entry_px_raw)
            trade_px_array = candidate_px_array
            trade_entry_gate_prob = float(gate_trade_prob)
            trade_entry_dir_probs = dir_probs.copy()
            trade_entry_confidence = confidence
            trade_stop_price = float(trade_entry_price * (1.0 - STOP_LOSS_PCT))
            trade_take_profit_price = float(trade_entry_price * 6.0)  # effectively no TP — model decides
            bars_held = 0
            unrealized_pnl = 0.0
            _trade_entry_trend = trend
            _trade_entry_rvol = round(rvol, 6) if rvol is not None else None
            _trade_entry_vol_z = round(vol_z, 2) if vol_z is not None else None
            _trade_entry_ret_5 = round(ret_5, 6) if ret_5 is not None else None
            _trade_entry_ret_30 = round(ret_30, 6) if ret_30 is not None else None
            trade_entry_reason_codes = [x for x in bar_entry['policy_gate_reason_codes'] if x]
            bar_entry['policy_gate_reason_codes'].append('entry_executed')

            if not quiet:
                strike_info = f"strike={strike_now}" if strike_now else ""
                print(f"  {time_str}  {ACTION_NAMES[action]:<16}  SPX={spx_now:.2f}  "
                      f"{strike_info}  premium=${trade_entry_price:.2f}  conf={gate_trade_prob:.0%}")

        if verbose and not quiet and action == ACTION_DO_NOTHING and not in_trade:
            print(f"  {time_str}  gate={gate_trade_prob:.2f} NO_TRADE  SPX={spx_now:.2f}")

        if speed > 0:
            time.sleep(1.0 / speed)

    gate_prob_list = [b['gate_trade_prob'] for b in bar_log]
    session_stats = {
        "total_bars": len(bar_log),
        "avg_gate_prob": round(float(np.mean(gate_prob_list)), 4) if gate_prob_list else 0.0,
        "max_gate_prob": round(float(np.max(gate_prob_list)), 4) if gate_prob_list else 0.0,
        "cooldown_blocked": int(cooldown_blocked),
        "pre_10am_blocked": int(pre_10am_blocked),
        "missing_option_array_blocked": int(missing_option_array_blocked),
        "missing_option_price_blocked": int(missing_option_price_blocked),
        "trades_blocked_by_balance": int(trades_blocked_by_balance),
        "account_balance": round(float(account_balance), 2),
        "num_trades": int(len(trades)),
    }

    return trades, bar_log, session_stats


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(trades, bar_log, replay_date, session_stats=None):
    """Print formatted trade summary with full analysis."""
    print(f"\n{'='*70}")
    print(f"  TRADE SUMMARY — {replay_date}")
    print(f"{'='*70}")

    if not trades:
        print("  No trades taken.")
        _print_session_stats(bar_log, session_stats=session_stats)
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
        print(f"\n  Option pricing:")
        print(f"    Strike: {int(t['strike']) if t['strike'] else '?'}  |  Entry: ${t['entry_option_px']:.2f}", end='')
        if t.get('exit_option_px'):
            print(f"  |  Exit: ${t['exit_option_px']:.2f}")
        else:
            print()

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
    _print_session_stats(bar_log, session_stats=session_stats)


def _print_session_stats(bar_log, session_stats=None):
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
    if isinstance(session_stats, dict):
        print(
            "  Entry blocks: "
            f"cooldown={session_stats.get('cooldown_blocked', 0)}, "
            f"pre_10am={session_stats.get('pre_10am_blocked', 0)}, "
            f"missing_array={session_stats.get('missing_option_array_blocked', 0)}, "
            f"missing_price={session_stats.get('missing_option_price_blocked', 0)}"
        )

    # When was the model most tempted to trade but didn't?
    flat_bars = [(b['time'], b['gate_trade_prob'], b['top_direction'], b['spx'])
                 for b in bar_log if b.get('executed_action', b.get('action')) == 'DO_NOTHING']
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
# Trade database CSV
# ---------------------------------------------------------------------------

BACKTEST_CSV_COLUMNS = [
    # Identity
    "trade_id", "date", "num", "result",
    # Timing
    "entry_time", "exit_time", "bars_held", "hold_min",
    # Contract
    "direction", "strike", "entry_option_px", "exit_option_px",
    # P&L
    "pnl_pct", "cum_pnl_pct", "reason",
    # SPX context
    "spx_entry", "spx_exit", "spx_move", "spx_move_pct",
    "spx_high_during", "spx_low_during", "mfe_points", "mae_points",
    # Volume
    "total_volume_during", "avg_bar_volume",
    # VIX
    "vix_entry", "vix_exit",
    # Session
    "session_high", "session_low", "day_open",
    # Model signals
    "entry_gate_prob", "entry_confidence", "exit_gate_notrade_prob",
    "entry_stop_price", "entry_take_profit_price",
    # Market regime at entry
    "entry_trend", "entry_rvol", "entry_vol_z", "entry_ret_5", "entry_ret_30",
    # Reason codes
    "entry_reason_codes", "exit_reason_codes",
    # Direction probabilities
    "dir_prob_C_ATM", "dir_prob_C_OTM5", "dir_prob_C_OTM10",
    "dir_prob_P_ATM", "dir_prob_P_OTM5", "dir_prob_P_OTM10",
]


def write_trade_csv(trades: list[dict], path: str) -> None:
    """Write comprehensive trade database CSV."""
    flat_trades = []
    for t in trades:
        ft = dict(t)
        # Flatten direction probs
        if ft.get('entry_dir_probs'):
            for k, v in ft['entry_dir_probs'].items():
                ft[f'dir_prob_{k}'] = v
            del ft['entry_dir_probs']
        # Flatten reason codes to strings
        if isinstance(ft.get('entry_reason_codes'), list):
            ft['entry_reason_codes'] = '|'.join(ft['entry_reason_codes'])
        if isinstance(ft.get('exit_reason_codes'), list):
            ft['exit_reason_codes'] = '|'.join(ft['exit_reason_codes'])
        flat_trades.append(ft)

    fieldnames = list(BACKTEST_CSV_COLUMNS)
    for row in flat_trades:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        if flat_trades:
            writer.writerows(flat_trades)
    print(f"Trade database saved: {path} ({len(flat_trades)} trades)")


# ---------------------------------------------------------------------------
# Equity curve chart
# ---------------------------------------------------------------------------

def generate_equity_curve(trades: list[dict], output_path: str,
                          starting_cash: float = 10_000.0,
                          title_suffix: str = "") -> None:
    """Generate an interactive Plotly HTML equity curve (portfolio value over time).

    Args:
        trades: list of trade dicts from run_replay (must have pnl_pct, date, exit_time)
        output_path: path to write HTML
        starting_cash: initial portfolio value in dollars
        title_suffix: appended to chart title
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("WARNING: plotly not installed. Skipping equity curve generation.")
        return

    if not trades:
        print("No trades to chart equity curve.")
        return

    # Build equity series: one point per trade exit
    cash = starting_cash
    times = []       # datetime objects for x-axis
    time_labels = []  # string labels for hover
    values = []
    colors = []
    hover_texts = []

    # Starting point
    first_trade = trades[0]
    start_label = f"{first_trade['date']} {first_trade.get('entry_time', '09:30')}"
    try:
        from datetime import datetime as _dt
        times.append(_dt.strptime(start_label, "%Y-%m-%d %H:%M"))
    except (ValueError, TypeError):
        times.append(start_label)
    time_labels.append(start_label)
    values.append(cash)
    colors.append('gray')
    hover_texts.append(f"Starting balance: ${cash:,.0f}")

    for t in trades:
        pnl_pct = t['pnl_pct'] / 100.0  # convert from percent to decimal
        entry_px = t.get('entry_option_px', 0)
        if entry_px and entry_px > 0:
            # Realistic: 1 contract, dollar P&L from option premium change
            contract_cost = entry_px * SPX_MULTIPLIER
            if contract_cost > cash:
                continue  # couldn't have afforded this trade
            pnl_dollar = pnl_pct * entry_px * SPX_MULTIPLIER
        else:
            pnl_dollar = cash * pnl_pct
        cash += pnl_dollar
        cash = max(cash, 0.0)
        label = f"{t['date']} {t.get('exit_time', '')}"
        try:
            times.append(_dt.strptime(label.strip(), "%Y-%m-%d %H:%M"))
        except (ValueError, TypeError):
            times.append(label)
        time_labels.append(label)
        values.append(cash)
        result = t.get('result', '')
        colors.append('#2ecc71' if result == 'WIN' else '#e74c3c')
        hover_texts.append(
            f"Trade #{t['num']}: {t.get('direction', '?')}<br>"
            f"P&L: {t['pnl_pct']:+.2f}% (${pnl_dollar:+,.0f})<br>"
            f"Result: {result}<br>"
            f"Balance: ${cash:,.0f}"
        )

    ending_cash = cash
    total_return = (ending_cash / starting_cash - 1) * 100

    # Peak / drawdown series
    peak = starting_cash
    drawdowns = []
    for v in values:
        peak = max(peak, v)
        dd = (v - peak) / peak * 100
        drawdowns.append(dd)

    max_dd = min(drawdowns)

    # Build chart
    fig = go.Figure()

    # Equity line
    fig.add_trace(go.Scatter(
        x=times,
        y=values,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#3498db', width=2.5),
        marker=dict(size=7, color=colors, line=dict(width=1, color='white')),
        text=hover_texts,
        hoverinfo='text',
        customdata=time_labels,
        hovertemplate='%{text}<extra>%{customdata}</extra>',
    ))

    # Starting cash reference line
    fig.add_hline(
        y=starting_cash, line_dash="dash", line_color="gray",
        annotation_text=f"Start: ${starting_cash:,.0f}",
        annotation_position="bottom left",
    )

    # High-water mark line
    peaks = []
    peak = starting_cash
    for v in values:
        peak = max(peak, v)
        peaks.append(peak)
    fig.add_trace(go.Scatter(
        x=times,
        y=peaks,
        mode='lines',
        name='High Water Mark',
        line=dict(color='rgba(46, 204, 113, 0.3)', width=1, dash='dot'),
        hoverinfo='skip',
    ))

    title = f"Equity Curve — ${starting_cash:,.0f} → ${ending_cash:,.0f} ({total_return:+.1f}%)"
    if title_suffix:
        title += title_suffix

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(
            title="Date",
            type='date',
            tickangle=-45,
            showgrid=True, gridcolor='rgba(128,128,128,0.2)',
            dtick="M1",
            tickformat="%b %Y",
        ),
        yaxis=dict(
            title="Portfolio Value ($)",
            tickformat="$,.0f",
            showgrid=True, gridcolor='rgba(128,128,128,0.2)',
        ),
        template='plotly_dark',
        hovermode='x unified',
        height=600,
        annotations=[
            dict(
                x=times[-1], y=ending_cash,
                text=f"${ending_cash:,.0f}<br>({total_return:+.1f}%)<br>Max DD: {max_dd:.1f}%",
                showarrow=True, arrowhead=2, ax=40, ay=-40,
                font=dict(size=12, color='white'),
                bgcolor='rgba(0,0,0,0.7)', bordercolor='white',
            ),
        ],
    )

    fig.write_html(output_path, include_plotlyjs=True)
    print(f"Equity curve saved: {output_path}")

    # Write companion CSV
    csv_path = output_path.replace('.html', '.csv')
    peak = starting_cash
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trade_num', 'date', 'time', 'direction', 'result',
                          'pnl_pct', 'pnl_dollar', 'balance', 'high_water_mark',
                          'drawdown_pct'])
        # Row 0: starting balance
        writer.writerow([0, trades[0]['date'], trades[0].get('entry_time', ''),
                          '', '', '', '', f"{starting_cash:.2f}",
                          f"{starting_cash:.2f}", '0.00'])
        bal = starting_cash
        for t in trades:
            pnl_pct = t['pnl_pct'] / 100.0
            entry_px = t.get('entry_option_px', 0)
            if entry_px and entry_px > 0:
                contract_cost = entry_px * SPX_MULTIPLIER
                if contract_cost > bal:
                    continue  # couldn't afford
                pnl_dollar = pnl_pct * entry_px * SPX_MULTIPLIER
            else:
                pnl_dollar = bal * pnl_pct
            bal += pnl_dollar
            bal = max(bal, 0.0)
            peak = max(peak, bal)
            dd = (bal - peak) / peak * 100
            writer.writerow([
                t['num'], t['date'], t.get('exit_time', ''),
                t.get('direction', ''), t.get('result', ''),
                f"{t['pnl_pct']:.2f}", f"{pnl_dollar:.2f}",
                f"{bal:.2f}", f"{peak:.2f}", f"{dd:.2f}",
            ])
    print(f"Equity curve CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# Plotly chart
# ---------------------------------------------------------------------------

def generate_plotly_chart(trades: list[dict], bar_log: list[dict],
                          replay_date: str, output_path: str,
                          title_suffix: str = "",
                          ohlcv_df: pd.DataFrame | None = None) -> None:
    """Generate interactive Plotly HTML chart with candlesticks/line, multi-timeframe, and trade markers.

    Args:
        trades: list of trade dicts from run_replay
        bar_log: per-bar decision log (used for gate probability subplot)
        replay_date: date string or 'backtest' for multi-day
        output_path: path to write HTML
        title_suffix: appended to chart title
        ohlcv_df: DataFrame with columns [datetime, open, high, low, close, volume]
                  indexed by datetime. If None, falls back to bar_log SPX prices.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("WARNING: plotly not installed. Skipping chart generation.")
        return

    if ohlcv_df is None and not bar_log:
        print("No data to chart.")
        return

    # --- Build 1-min OHLCV base ---
    if ohlcv_df is not None and not ohlcv_df.empty:
        df_1m = ohlcv_df.copy()
        if not isinstance(df_1m.index, pd.DatetimeIndex):
            if 'datetime' in df_1m.columns:
                df_1m.index = pd.to_datetime(df_1m['datetime'])
            else:
                df_1m.index = pd.RangeIndex(len(df_1m))
        df_1m = df_1m.sort_index()
    else:
        # Fallback: build from bar_log (line chart only, no real OHLC)
        times_raw = []
        for b in bar_log:
            d = b.get('date', replay_date)
            t = b.get('time', '09:30')
            try:
                times_raw.append(pd.Timestamp(f"{d} {t}"))
            except Exception:
                times_raw.append(pd.Timestamp(f"{replay_date} {t}"))
        df_1m = pd.DataFrame({
            'open': [b['spx'] for b in bar_log],
            'high': [b['spx'] for b in bar_log],
            'low': [b['spx'] for b in bar_log],
            'close': [b['spx'] for b in bar_log],
            'volume': [b.get('volume', 0) for b in bar_log],
        }, index=pd.DatetimeIndex(times_raw))

    # --- Resample to multiple timeframes ---
    timeframes = {
        '1min': None,  # already have it
        '5min': '5min',
        '30min': '30min',
        '1H': '1h',
        '1D': '1D',
        '1W': '1W',
    }

    resampled = {'1min': df_1m}
    for label, rule in timeframes.items():
        if rule is None:
            continue
        rs = df_1m.resample(rule)
        resampled[label] = pd.DataFrame({
            'open': rs['open'].first(),
            'high': rs['high'].max(),
            'low': rs['low'].min(),
            'close': rs['close'].last(),
            'volume': rs['volume'].sum(),
        }).dropna(subset=['close'])

    tf_labels = list(resampled.keys())

    # --- Build trade entry/exit datetime mapping ---
    def _trade_dt(trade, field_time, field_date='date'):
        d = trade.get(field_date, replay_date)
        t = trade.get(field_time, '09:30')
        try:
            return pd.Timestamp(f"{d} {t}")
        except Exception:
            return None

    entry_dts = []
    exit_dts = []
    for t in trades:
        entry_dts.append(_trade_dt(t, 'entry_time'))
        exit_dts.append(_trade_dt(t, 'exit_time'))

    # --- Create figure ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.03,
    )

    # Trace index tracking for visibility toggling
    trace_groups = {}  # {(tf, chart_type): [trace_indices]}
    trace_idx = 0

    for tf_label, df_tf in resampled.items():
        x = df_tf.index

        # Candlestick trace
        fig.add_trace(go.Candlestick(
            x=x, open=df_tf['open'], high=df_tf['high'],
            low=df_tf['low'], close=df_tf['close'],
            name=f'SPX {tf_label}',
            increasing_line_color='#26A69A', decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A', decreasing_fillcolor='#EF5350',
            visible=(tf_label == '1min'),
            showlegend=False,
        ), row=1, col=1)
        trace_groups[(tf_label, 'candle')] = [trace_idx]
        trace_idx += 1

        # Line trace
        fig.add_trace(go.Scatter(
            x=x, y=df_tf['close'], mode='lines',
            name=f'SPX {tf_label}',
            line=dict(color='#2196F3', width=1.5),
            visible=False,
            showlegend=False,
        ), row=1, col=1)
        trace_groups[(tf_label, 'line')] = [trace_idx]
        trace_idx += 1

        # Volume bars
        colors = ['#26A69A' if c >= o else '#EF5350'
                  for o, c in zip(df_tf['open'], df_tf['close'])]
        fig.add_trace(go.Bar(
            x=x, y=df_tf['volume'], name=f'Vol {tf_label}',
            marker_color=colors, opacity=0.5,
            visible=(tf_label == '1min'),
            showlegend=False,
        ), row=2, col=1)
        trace_groups[(tf_label, 'vol_candle')] = [trace_idx]
        trace_idx += 1

        # Volume bars for line mode (same data, just separate visibility)
        fig.add_trace(go.Bar(
            x=x, y=df_tf['volume'], name=f'Vol {tf_label}',
            marker_color='rgba(100,100,100,0.4)', opacity=0.5,
            visible=False,
            showlegend=False,
        ), row=2, col=1)
        trace_groups[(tf_label, 'vol_line')] = [trace_idx]
        trace_idx += 1

    # --- Trade markers (batched into few traces for performance) ---
    entry_trace_start = trace_idx

    # Batch entries/exits by category to minimize trace count
    # Categories: win_call_entry, win_put_entry, loss_call_entry, loss_put_entry,
    #             win_exit, loss_exit, win_connector, loss_connector
    batches = {
        'win_call_entry': {'x': [], 'y': [], 'text': [], 'hover': []},
        'win_put_entry': {'x': [], 'y': [], 'text': [], 'hover': []},
        'loss_call_entry': {'x': [], 'y': [], 'text': [], 'hover': []},
        'loss_put_entry': {'x': [], 'y': [], 'text': [], 'hover': []},
        'win_exit': {'x': [], 'y': [], 'hover': []},
        'loss_exit': {'x': [], 'y': [], 'hover': []},
    }
    connector_x = []  # interleaved with None for line breaks
    connector_y = []
    connector_colors = []

    for i, t in enumerate(trades):
        edt = entry_dts[i]
        xdt = exit_dts[i]
        if edt is None:
            continue

        is_win = t['result'] == 'WIN'
        is_call = 'CALL' in t.get('direction', '')
        prefix = 'win' if is_win else 'loss'
        direction = 'call' if is_call else 'put'

        entry_key = f'{prefix}_{direction}_entry'
        batches[entry_key]['x'].append(edt)
        batches[entry_key]['y'].append(t['spx_entry'])
        batches[entry_key]['text'].append(f"#{t['num']}")
        batches[entry_key]['hover'].append(
            f"<b>ENTRY #{t['num']}</b><br>"
            f"{t['date']} {t.get('entry_time','')}<br>"
            f"Direction: {t['direction']}<br>"
            f"SPX: {t['spx_entry']}<br>"
            f"Option: ${t['entry_option_px']:.2f}<br>"
            f"Gate: {t['entry_gate_prob']:.1%}<br>"
            f"<extra></extra>"
        )

        if xdt is not None:
            exit_key = f'{prefix}_exit'
            batches[exit_key]['x'].append(xdt)
            batches[exit_key]['y'].append(t['spx_exit'])
            batches[exit_key]['hover'].append(
                f"<b>EXIT #{t['num']}</b><br>"
                f"{t['date']} {t.get('exit_time','')}<br>"
                f"Reason: {t['reason']}<br>"
                f"P&L: {t['pnl_pct']:+.1f}%<br>"
                f"Held: {t['hold_min']}min<br>"
                f"<extra></extra>"
            )
            # Connector: entry->exit with None break
            connector_x.extend([edt, xdt, None])
            connector_y.extend([t['spx_entry'], t['spx_exit'], None])

    entry_style = {
        'win_call_entry':  {'color': '#4CAF50', 'symbol': 'triangle-up'},
        'win_put_entry':   {'color': '#4CAF50', 'symbol': 'triangle-down'},
        'loss_call_entry': {'color': '#F44336', 'symbol': 'triangle-up'},
        'loss_put_entry':  {'color': '#F44336', 'symbol': 'triangle-down'},
    }
    for key, style in entry_style.items():
        b = batches[key]
        if not b['x']:
            continue
        fig.add_trace(go.Scatter(
            x=b['x'], y=b['y'],
            mode='markers+text',
            marker=dict(size=8, color=style['color'], symbol=style['symbol'],
                       line=dict(width=1, color='white')),
            text=b['text'], textposition='top center',
            textfont=dict(size=7, color=style['color']),
            hovertemplate=b['hover'],
            showlegend=False,
        ), row=1, col=1)
        trace_idx += 1

    exit_style = {
        'win_exit':  '#4CAF50',
        'loss_exit': '#F44336',
    }
    for key, color in exit_style.items():
        b = batches[key]
        if not b['x']:
            continue
        fig.add_trace(go.Scatter(
            x=b['x'], y=b['y'],
            mode='markers',
            marker=dict(size=7, color=color, symbol='x',
                       line=dict(width=2, color=color)),
            hovertemplate=b['hover'],
            showlegend=False,
        ), row=1, col=1)
        trace_idx += 1

    # Single connector trace with None-breaks between segments
    if connector_x:
        fig.add_trace(go.Scatter(
            x=connector_x, y=connector_y,
            mode='lines',
            line=dict(color='rgba(180,180,180,0.3)', width=0.8, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)
        trace_idx += 1

    trade_trace_count = trace_idx - entry_trace_start

    # --- Build visibility buttons ---
    total_traces = trace_idx

    def make_visibility(active_tf, chart_type):
        vis = [False] * total_traces
        # Show price trace for this tf + chart type
        candle_key = (active_tf, 'candle')
        line_key = (active_tf, 'line')
        vol_candle_key = (active_tf, 'vol_candle')
        vol_line_key = (active_tf, 'vol_line')

        if chart_type == 'candle':
            for idx in trace_groups.get(candle_key, []):
                vis[idx] = True
            for idx in trace_groups.get(vol_candle_key, []):
                vis[idx] = True
        else:
            for idx in trace_groups.get(line_key, []):
                vis[idx] = True
            for idx in trace_groups.get(vol_line_key, []):
                vis[idx] = True

        # Trade markers always visible
        for idx in range(entry_trace_start, total_traces):
            vis[idx] = True
        return vis

    # Timeframe buttons
    tf_buttons = []
    for tf in tf_labels:
        tf_buttons.append(dict(
            label=tf,
            method='update',
            args=[{'visible': make_visibility(tf, 'candle')},
                  {'title': f'SPX {tf} Candles — {replay_date}{title_suffix}'}],
        ))

    # Chart type buttons
    type_buttons_candle = []
    type_buttons_line = []
    for tf in tf_labels:
        type_buttons_candle.append(dict(
            label=tf,
            method='update',
            args=[{'visible': make_visibility(tf, 'candle')},
                  {'title': f'SPX {tf} Candles — {replay_date}{title_suffix}'}],
        ))
        type_buttons_line.append(dict(
            label=tf,
            method='update',
            args=[{'visible': make_visibility(tf, 'line')},
                  {'title': f'SPX {tf} Line — {replay_date}{title_suffix}'}],
        ))

    # Summary stats — use dollar-compounded return, not naive sum of percentages
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    wr = f"{100*wins/len(trades):.0f}%" if trades else "N/A"
    # Compute real compounded return using dollar P&L (same logic as equity curve)
    _cash = 10_000.0
    for t in trades:
        _pnl_pct = t['pnl_pct'] / 100.0
        _entry_px = t.get('entry_option_px', 0)
        if _entry_px and _entry_px > 0:
            _cost = _entry_px * SPX_MULTIPLIER
            if _cost > _cash:
                continue
            _dollar_pnl = _pnl_pct * _entry_px * SPX_MULTIPLIER
        else:
            _dollar_pnl = _cash * _pnl_pct
        _cash += _dollar_pnl
        _cash = max(_cash, 0.0)
    total_return_pct = (_cash / 10_000.0 - 1) * 100
    summary_text = f"Trades: {len(trades)} | Win rate: {wr} | Return: {total_return_pct:+.1f}%"

    fig.update_layout(
        title=dict(text=f'SPX 1min Candles — {replay_date}{title_suffix}', x=0.5),
        height=900,
        template='plotly_dark',
        hovermode='closest',
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        margin=dict(l=60, r=20, t=100, b=40),
        updatemenus=[
            # Timeframe selector (candle)
            dict(
                type='buttons',
                direction='right',
                x=0.0, xanchor='left',
                y=1.12, yanchor='top',
                buttons=type_buttons_candle,
                showactive=True,
                bgcolor='rgba(50,50,50,0.8)',
                font=dict(color='white', size=11),
                bordercolor='#555',
            ),
            # Chart type toggle
            dict(
                type='buttons',
                direction='right',
                x=0.7, xanchor='left',
                y=1.12, yanchor='top',
                buttons=[
                    dict(label='Candles',
                         method='update',
                         args=[{'visible': make_visibility('1min', 'candle')},
                               {'title': f'SPX 1min Candles — {replay_date}{title_suffix}'}]),
                    dict(label='Line',
                         method='update',
                         args=[{'visible': make_visibility('1min', 'line')},
                               {'title': f'SPX 1min Line — {replay_date}{title_suffix}'}]),
                ],
                showactive=True,
                bgcolor='rgba(50,50,50,0.8)',
                font=dict(color='white', size=11),
                bordercolor='#555',
            ),
        ],
        annotations=[
            dict(text="Timeframe:", x=0.0, xref="paper", y=1.16, yref="paper",
                 showarrow=False, font=dict(size=12, color='#aaa')),
            dict(text="Chart:", x=0.7, xref="paper", y=1.16, yref="paper",
                 showarrow=False, font=dict(size=12, color='#aaa')),
            dict(text=summary_text, x=0.5, xref="paper", y=1.04, yref="paper",
                 showarrow=False, font=dict(size=13, color='#ddd'),
                 bgcolor='rgba(30,30,30,0.7)'),
        ],
    )

    fig.update_yaxes(title_text="SPX", row=1, col=1, gridcolor='rgba(80,80,80,0.3)')
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor='rgba(80,80,80,0.3)')

    # Remove overnight gaps and weekends — stitch trading sessions together
    rangebreaks = [
        dict(bounds=["sat", "mon"]),                # hide weekends
        dict(bounds=[16, 9.5], pattern="hour"),      # hide 4PM-9:30AM
    ]
    fig.update_xaxes(
        gridcolor='rgba(80,80,80,0.3)',
        rangebreaks=rangebreaks,
        row=1, col=1,
    )
    fig.update_xaxes(
        gridcolor='rgba(80,80,80,0.3)',
        rangebreaks=rangebreaks,
        row=2, col=1,
    )

    fig.write_html(output_path, include_plotlyjs=True)
    print(f"Chart saved: {output_path}")


# ---------------------------------------------------------------------------
# Full validation set backtest
# ---------------------------------------------------------------------------

def run_backtest(model, data_pt_path: str, device: str = 'cpu',
                 output_dir: str = 'backtest_output',
                 all_dates_mode: bool = False) -> None:
    """Run replay across the validation set (or full dataset with all_dates_mode) from data.pt.

    Outputs:
    - backtest_trades.csv — comprehensive trade database
    - backtest_chart_{date}.html — per-day interactive Plotly charts
    - backtest_summary.txt — aggregate statistics
    """
    if not os.path.exists(data_pt_path):
        print(f"ERROR: data.pt not found at {data_pt_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data.pt...")
    data = torch.load(data_pt_path, map_location='cpu', weights_only=False)

    all_dates = data['dates']
    features_np = data['features'].numpy()
    valid_np = data['valid_mask'].numpy()
    timestamps_list = list(data['timestamps'])

    # Build option prices dict
    option_prices = {}
    for key in ['atm_call_prices', 'atm_put_prices',
                'otm5_call_prices', 'otm5_put_prices',
                'otm10_call_prices', 'otm10_put_prices',
                'atm_strikes']:
        if key in data:
            arr = data[key].numpy()
            option_prices[key.replace('_prices', '').replace('_', '_')] = arr
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

    # Date selection: --all-dates runs the entire dataset, otherwise validation only
    unique_dates = sorted(set(all_dates))
    if all_dates_mode:
        val_dates = unique_dates
        print(f"Full dataset: {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")
    else:
        n_train = int(len(unique_dates) * 0.8)
        val_dates = unique_dates[n_train:]
        print(f"Validation set: {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})")

    # Build aligned SPX market data using date+bar_of_day mapping.
    # data.pt indices do NOT match spy_df/spx_df indices (different start dates,
    # different bar counts). We align by matching each date+bar_of_day pair.
    n_feat = len(features_np)

    spx_cache = os.path.join(DATA_DIR, "spx_1min.pkl")
    spy_cache = os.path.join(DATA_DIR, "spy_1min.pkl")

    if os.path.exists(spx_cache):
        with open(spx_cache, 'rb') as f:
            spx_df = pickle.load(f)
        # SPX has spx_open/spx_high/spx_low/spx_close columns
        src_close = spx_df['spx_close'].values.astype(np.float64)
        src_high = spx_df['spx_high'].values.astype(np.float64)
        src_low = spx_df['spx_low'].values.astype(np.float64)
        src_open = spx_df['spx_open'].values.astype(np.float64)
        # Volume comes from spy_df (SPX index has no volume)
        if os.path.exists(spy_cache):
            with open(spy_cache, 'rb') as f:
                spy_df = pickle.load(f)
            src_volume = spy_df['volume'].values.astype(np.float64)
            src_dates_for_idx = spy_df['date'].astype(str).values
        else:
            src_volume = np.zeros(len(src_close))
            src_dates_for_idx = spx_df['date'].astype(str).values
        print(f"Using SPX data ({len(src_close)} bars) with date+bar_of_day alignment")
    elif os.path.exists(spy_cache):
        with open(spy_cache, 'rb') as f:
            spy_df = pickle.load(f)
        src_close = spy_df['close'].values.astype(np.float64)
        src_high = spy_df['high'].values.astype(np.float64)
        src_low = spy_df['low'].values.astype(np.float64)
        src_open = spy_df['open'].values.astype(np.float64)
        src_volume = spy_df['volume'].values.astype(np.float64)
        src_dates_for_idx = spy_df['date'].astype(str).values
        print("WARNING: No spx_1min.pkl — falling back to SPY prices")
    else:
        print("WARNING: No market data cache — using NaN for raw market data")
        raw_df = pd.DataFrame({
            'close': np.full(n_feat, np.nan),
            'high': np.full(n_feat, np.nan),
            'low': np.full(n_feat, np.nan),
            'open': np.full(n_feat, np.nan),
            'volume': np.zeros(n_feat),
            'date': all_dates,
        })
        # Skip alignment — no source data
        src_close = None

    if src_close is not None:
        # Build date -> first index mapping for source data
        src_date_start = {}
        for i, d in enumerate(src_dates_for_idx):
            if d not in src_date_start:
                src_date_start[d] = i

        # Build bar_of_day for data.pt
        aligned_close = np.full(n_feat, np.nan)
        aligned_high = np.full(n_feat, np.nan)
        aligned_low = np.full(n_feat, np.nan)
        aligned_open = np.full(n_feat, np.nan)
        aligned_volume = np.zeros(n_feat)

        prev_date = None
        bar_in_day = 0
        n_aligned = 0
        for i in range(n_feat):
            d = all_dates[i]
            if d != prev_date:
                bar_in_day = 0
                prev_date = d
            else:
                bar_in_day += 1
            if d in src_date_start:
                src_idx = src_date_start[d] + bar_in_day
                if src_idx < len(src_close):
                    aligned_close[i] = src_close[src_idx]
                    aligned_high[i] = src_high[src_idx]
                    aligned_low[i] = src_low[src_idx]
                    aligned_open[i] = src_open[src_idx]
                    if src_idx < len(src_volume):
                        aligned_volume[i] = src_volume[src_idx]
                    n_aligned += 1

        pct_aligned = 100 * n_aligned / n_feat
        print(f"Aligned {n_aligned}/{n_feat} bars ({pct_aligned:.1f}%) to SPX prices")

        raw_df = pd.DataFrame({
            'close': aligned_close,
            'high': aligned_high,
            'low': aligned_low,
            'open': aligned_open,
            'volume': aligned_volume,
            'date': all_dates,
        })

    lookback = int(data.get('config', {}).get('lookback', 120)) if isinstance(data.get('config'), dict) else 120
    # Try to get lookback from model
    if hasattr(model, 'lookback'):
        lookback = model.lookback

    # Helper to build OHLCV DataFrame for a given day from aligned raw_df
    def _build_day_ohlcv(day):
        day_mask = raw_df['date'] == day
        day_raw = raw_df[day_mask]
        if len(day_raw) == 0:
            return None
        day_indices = day_raw.index.tolist()
        day_times = []
        for di in day_indices:
            ts_raw = timestamps_list[di] if di < len(timestamps_list) else None
            t_str = _format_time(ts_raw if ts_raw is not None else '')
            try:
                day_times.append(pd.Timestamp(f"{day} {t_str}"))
            except Exception:
                day_times.append(pd.Timestamp(f"{day} 09:30"))
        return pd.DataFrame({
            'open': day_raw['open'].values if 'open' in day_raw.columns else day_raw['close'].values,
            'high': day_raw['high'].values,
            'low': day_raw['low'].values,
            'close': day_raw['close'].values,
            'volume': day_raw['volume'].values,
        }, index=pd.DatetimeIndex(day_times))

    all_trades = []
    all_bar_logs = []  # Collected for all-trades chart
    day_stats = []
    account_balance_running = STARTING_CAPITAL  # Cross-day equity tracking

    for day_idx, day in enumerate(val_dates):
        trades, bar_log, stats = run_replay(
            model, features_t, features_np, list(all_dates), valid_np,
            px_remap, timestamps_list, day, lookback, raw_df,
            device=device, quiet=True,
        )

        # Track cross-day equity (dollar-based, matches generate_equity_curve)
        day_dollar_pnl = 0.0
        for t in trades:
            entry_px = t.get('entry_option_px', 0)
            pnl_pct = t['pnl_pct'] / 100.0
            if entry_px and entry_px > 0:
                contract_cost = entry_px * SPX_MULTIPLIER
                if contract_cost > account_balance_running:
                    t['cum_pnl_pct'] = round((account_balance_running / STARTING_CAPITAL - 1) * 100, 2)
                    continue  # blocked by affordability
                dollar_pnl = pnl_pct * entry_px * SPX_MULTIPLIER
            else:
                dollar_pnl = account_balance_running * pnl_pct
            account_balance_running += dollar_pnl
            account_balance_running = max(account_balance_running, 0.0)
            day_dollar_pnl += dollar_pnl
            t['cum_pnl_pct'] = round((account_balance_running / STARTING_CAPITAL - 1) * 100, 2)

        all_trades.extend(trades)

        # Tag bar_log entries with date and collect for all-trades chart
        for b in bar_log:
            b['date'] = day
        all_bar_logs.extend(bar_log)

        n_wins = sum(1 for t in trades if t['result'] == 'WIN')
        n_losses = len(trades) - n_wins

        day_stats.append({
            'date': day, 'trades': len(trades), 'wins': n_wins,
            'losses': n_losses, 'pnl_dollar': round(day_dollar_pnl, 2),
        })

        day_return_pct = (day_dollar_pnl / max(account_balance_running - day_dollar_pnl, 1.0)) * 100
        status = f"  [{day_idx+1}/{len(val_dates)}] {day}: {len(trades)} trades, P&L=${day_dollar_pnl:+,.0f} ({day_return_pct:+.1f}%)"
        if trades:
            status += f"  (W{n_wins}/L{n_losses})"
        print(status)

        # Generate per-day chart if there were trades
        if trades and bar_log:
            ohlcv_day = _build_day_ohlcv(day)
            charts_dir = os.path.join(output_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            chart_path = os.path.join(charts_dir, f"chart_{day}.html")
            generate_plotly_chart(trades, bar_log, day, chart_path, ohlcv_df=ohlcv_day)

    # Renumber trades across full backtest
    for i, t in enumerate(all_trades):
        t['num'] = i + 1
        t['trade_id'] = f"BT-{t['date']}-T{i+1:04d}"

    # Write comprehensive CSV
    csv_path = os.path.join(output_dir, "backtest_trades.csv")
    write_trade_csv(all_trades, csv_path)

    # Generate all-trades chart across all days with trades
    if all_trades:
        trade_dates = sorted(set(t['date'] for t in all_trades))
        ohlcv_parts = [_build_day_ohlcv(d) for d in trade_dates]
        ohlcv_parts = [p for p in ohlcv_parts if p is not None]
        if ohlcv_parts:
            ohlcv_all = pd.concat(ohlcv_parts).sort_index()
            charts_dir = os.path.join(output_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            chart_all_path = os.path.join(charts_dir, "chart_all_trades.html")
            generate_plotly_chart(
                all_trades, all_bar_logs, 'backtest', chart_all_path,
                title_suffix=f" ({len(trade_dates)} days, {len(all_trades)} trades)",
                ohlcv_df=ohlcv_all,
            )

    # Generate equity curve
    if all_trades:
        equity_path = os.path.join(output_dir, "equity_curve.html")
        generate_equity_curve(
            all_trades, equity_path,
            title_suffix=f" ({len(val_dates)} days, {len(all_trades)} trades)",
        )

    # Write summary
    _write_backtest_summary(all_trades, day_stats, val_dates, output_dir)


def _write_backtest_summary(trades: list[dict], day_stats: list[dict],
                             val_dates: list[str], output_dir: str) -> None:
    """Write aggregate backtest statistics."""
    summary_path = os.path.join(output_dir, "backtest_summary.txt")
    lines = []
    lines.append(f"BACKTEST SUMMARY")
    lines.append(f"{'='*60}")
    lines.append(f"Period: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")
    lines.append(f"Total trades: {len(trades)}")

    if not trades:
        lines.append("No trades taken.")
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Summary saved: {summary_path}")
        return

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Reconstruct equity curve (matches generate_equity_curve and prepare.py)
    cash = STARTING_CAPITAL
    peak_cash = cash
    for t in trades:
        pnl_pct = t['pnl_pct'] / 100.0
        entry_px = t.get('entry_option_px', 0)
        if entry_px and entry_px > 0:
            contract_cost = entry_px * SPX_MULTIPLIER
            if contract_cost > cash:
                continue
            pnl_dollar = pnl_pct * entry_px * SPX_MULTIPLIER
        else:
            pnl_dollar = cash * pnl_pct
        cash += pnl_dollar
        cash = max(cash, 0.0)
        peak_cash = max(peak_cash, cash)
    total_return = (cash / STARTING_CAPITAL - 1) * 100
    max_dd = ((cash - peak_cash) / peak_cash * 100) if peak_cash > 0 else 0.0
    # Recompute max DD properly over full equity path
    _eq_cash = STARTING_CAPITAL
    _eq_peak = _eq_cash
    _eq_max_dd = 0.0
    for t in trades:
        pnl_pct = t['pnl_pct'] / 100.0
        entry_px = t.get('entry_option_px', 0)
        if entry_px and entry_px > 0:
            if entry_px * SPX_MULTIPLIER > _eq_cash:
                continue
            _eq_cash += pnl_pct * entry_px * SPX_MULTIPLIER
        else:
            _eq_cash += _eq_cash * pnl_pct
        _eq_cash = max(_eq_cash, 0.0)
        _eq_peak = max(_eq_peak, _eq_cash)
        dd = (_eq_cash - _eq_peak) / _eq_peak * 100 if _eq_peak > 0 else 0.0
        _eq_max_dd = min(_eq_max_dd, dd)

    lines.append(f"Win rate: {len(wins)}/{len(trades)} ({100*len(wins)/len(trades):.0f}%)")
    lines.append(f"Total P&L: {total_return:+.1f}% (${STARTING_CAPITAL:,.0f} → ${cash:,.0f})")
    lines.append(f"Max drawdown: {_eq_max_dd:.1f}%")

    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.01
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    lines.append(f"Profit factor: {pf:.2f}")

    if wins:
        lines.append(f"Avg winner: +{np.mean(wins):.1f}%  |  Best: +{max(wins):.1f}%")
    if losses:
        lines.append(f"Avg loser: {np.mean(losses):.1f}%  |  Worst: {min(losses):.1f}%")

    avg_hold = np.mean([t['bars_held'] for t in trades])
    lines.append(f"Avg hold: {avg_hold:.0f} bars ({avg_hold:.0f} min)")

    trades_per_day = len(trades) / len(val_dates)
    lines.append(f"Trades/day: {trades_per_day:.1f}")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t['reason']
        reasons[r] = reasons.get(r, 0) + 1
    lines.append(f"\nExit reasons:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {r}: {c} ({100*c/len(trades):.0f}%)")

    # Direction breakdown
    dirs = {}
    for t in trades:
        d = t['direction']
        dirs[d] = dirs.get(d, 0) + 1
    lines.append(f"\nDirection breakdown:")
    for d, c in sorted(dirs.items(), key=lambda x: -x[1]):
        d_trades = [t for t in trades if t['direction'] == d]
        d_wins = sum(1 for t in d_trades if t['result'] == 'WIN')
        d_dollar_pnl = sum(
            (t['pnl_pct'] / 100.0) * t.get('entry_option_px', 0) * SPX_MULTIPLIER
            for t in d_trades if t.get('entry_option_px', 0) > 0
        )
        lines.append(f"  {d}: {c} trades, {d_wins}W, P&L=${d_dollar_pnl:+,.0f}")

    # Best/worst days
    lines.append(f"\nBest days:")
    sorted_days = sorted(day_stats, key=lambda x: x['pnl_dollar'], reverse=True)
    for ds in sorted_days[:5]:
        if ds['trades'] > 0:
            lines.append(f"  {ds['date']}: ${ds['pnl_dollar']:+,.0f}  ({ds['trades']} trades, W{ds['wins']}/L{ds['losses']})")

    lines.append(f"\nWorst days:")
    for ds in sorted_days[-5:]:
        if ds['trades'] > 0:
            lines.append(f"  {ds['date']}: ${ds['pnl_dollar']:+,.0f}  ({ds['trades']} trades, W{ds['wins']}/L{ds['losses']})")

    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSummary saved: {summary_path}")
    print('\n'.join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_model_path(args_model):
    """Find model checkpoint path."""
    if args_model is not None:
        return args_model
    candidates = [
        os.path.join(os.path.dirname(__file__), 'best_model.pt'),
        os.path.join(os.path.dirname(__file__), '..', 'best_model.pt'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    print("ERROR: No model found. Train first or specify --model path")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Replay/backtest trading model")
    parser.add_argument("--date", type=str, default=None,
                        help="Single trading day to replay (YYYY-MM-DD)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run full validation set backtest (outputs CSV + charts)")
    parser.add_argument("--all-dates", action="store_true",
                        help="Run backtest on ALL dates in data.pt, not just the validation split")
    parser.add_argument("--output-dir", type=str, default="backtest_output",
                        help="Output directory for backtest results (default: backtest_output)")
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
                        help="Save trade log to CSV file (single-day mode)")
    parser.add_argument("--train-py", type=str, default=None,
                        help="Path to train.py/best_train.py for custom model architecture")
    args = parser.parse_args()

    if not args.date and not args.backtest:
        print("ERROR: Specify --date YYYY-MM-DD for single-day replay or --backtest for full validation set")
        sys.exit(1)

    model_path = _find_model_path(args.model)
    device = 'cpu'
    model, lookback, config, metrics = load_model(model_path, device,
                                                   train_py_path=args.train_py)

    # --- Full backtest mode ---
    if args.backtest:
        run_backtest(model, DATA_PT_PATH, device=device, output_dir=args.output_dir,
                     all_dates_mode=args.all_dates)
        return

    # --- Single-day replay mode ---
    replay_date = args.date
    if not is_0dte_day(replay_date):
        dow_name = dt.datetime.strptime(replay_date, '%Y-%m-%d').strftime('%A')
        print(f"WARNING: {replay_date} ({dow_name}) is NOT a 0DTE expiration day")

    model_num_features = _infer_model_num_features(model, config)
    training_data = load_training_features(replay_date)

    df, options_data, vix_data, chain_data = download_replay_data(
        replay_date, args.warmup_days, args.ib_port, args.no_download
    )

    if training_data is not None:
        features_t, _, dates, valid, option_prices, timestamps = training_data
        features_np = _align_features_to_model_width(
            features_t.numpy(), model_num_features, source_name="data.pt"
        )
        features_t = torch.tensor(features_np, dtype=torch.float32)
        print(f"  Using data.pt features for model inference (normalization-matched)")

        # Use raw market data from downloaded df, padded to data.pt length.
        # Raw features for market context come from data.pt's un-normalized
        # features (features_np IS normalized, so we use it as-is for context
        # — the raw feature indices like VIX/trend/vol still carry meaning).
        n_training = len(features_t)
        raw_features = features_np  # normalized features used for context snapshots
        replay_day_indices_dt = [i for i, d in enumerate(dates) if d == replay_date]
        replay_day_indices_raw = [i for i in range(len(df)) if df.iloc[i]['date'] == replay_date]

        if len(replay_day_indices_dt) == len(replay_day_indices_raw):
            padded_close = np.full(n_training, np.nan)
            padded_high = np.full(n_training, np.nan)
            padded_low = np.full(n_training, np.nan)
            padded_volume = np.full(n_training, np.nan)

            for dt_idx, raw_idx in zip(replay_day_indices_dt, replay_day_indices_raw):
                padded_close[dt_idx] = df.iloc[raw_idx]['close']
                padded_high[dt_idx] = df.iloc[raw_idx]['high']
                padded_low[dt_idx] = df.iloc[raw_idx]['low']
                padded_volume[dt_idx] = df.iloc[raw_idx]['volume']

            raw_df = pd.DataFrame({
                'close': padded_close, 'high': padded_high,
                'low': padded_low, 'volume': padded_volume, 'date': dates,
            })
        else:
            print(f"  WARNING: Bar count mismatch. Falling back to live features.")
            training_data = None

    if training_data is None:
        print(f"\nComputing features...")
        features, targets, dates, valid, option_prices, timestamps = compute_features(
            df, options_data, vix_data, chain_data
        )
        raw_features = features.copy()
        norm_context = _load_norm_context()
        if norm_context is not None:
            ctx_raw, ctx_valid = norm_context
            features = normalize_features_with_context(features, valid, ctx_raw, ctx_valid)
        else:
            features = normalize_features(features, valid)
        features = _align_features_to_model_width(features, model_num_features, source_name="replay-computed")
        features_t = torch.tensor(features, dtype=torch.float32)
        raw_df = df

    trades, bar_log, session_stats = run_replay(
        model, features_t, raw_features, dates, valid, option_prices, timestamps,
        replay_date, lookback, raw_df=raw_df,
        speed=args.speed, verbose=args.verbose, device=device,
    )

    print_summary(trades, bar_log, replay_date, session_stats=session_stats)

    # Save outputs
    output_base = args.output or f"replay_{replay_date}"
    if not output_base.endswith('.csv'):
        output_base += '.csv'

    write_trade_csv(trades, output_base)
    chart_path = output_base.replace('.csv', '_chart.html')
    generate_plotly_chart(trades, bar_log, replay_date, chart_path)
    equity_path = output_base.replace('.csv', '_equity.html')
    generate_equity_curve(trades, equity_path)


if __name__ == "__main__":
    main()
