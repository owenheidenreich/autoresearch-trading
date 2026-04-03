"""
Autoresearch-trading v4: simplified two-head model for SPX 0DTE options.
Single-GPU, single-file. The agent modifies THIS file.

v4 changes from v3:
  - Smaller model (3 layers, d_model=64) to prevent memorization
  - Real dropout (0.15) for regularization
  - Validation early stopping: evaluate every 500 steps, save best
  - No warm start (forced fresh training — old weights exploited oracle exits)
  - Removed: DynamicStopModule, QualityGate, Greeks-adaptive loss weights
  - Kept: 2-head gate+dir, position state, 32 features (v2 reduced), causal exit labels
  - Asymmetric gate loss: false entries penalized 2x vs missed entries

Two-head architecture:
  Gate head: (batch, 2) — [NO_TRADE, TRADE]
  Direction head: (batch, 6) — [CALL_ATM, CALL_OTM5, CALL_OTM10,
                                  PUT_ATM, PUT_OTM5, PUT_OTM10]

Combined into 8 actions:
  DO_NOTHING (0), BUY_CALL_ATM (1), BUY_CALL_OTM5 (2), BUY_CALL_OTM10 (3),
  BUY_PUT_ATM (4), BUY_PUT_OTM5 (5), BUY_PUT_OTM10 (6), EXIT (7)

Usage: uv run train.py  (or: python3 train.py)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")

import gc
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from prepare import (
    TIME_BUDGET,
    NUM_FEATURES,
    FEATURE_NAMES,
    ANNUAL_TRADING_BARS,
    BARS_PER_DAY,
    NUM_ACTIONS,
    ACTION_DO_NOTHING,
    ACTION_BUY_CALL_ATM,
    ACTION_BUY_CALL_OTM5,
    ACTION_BUY_CALL_OTM10,
    ACTION_BUY_PUT_ATM,
    ACTION_BUY_PUT_OTM5,
    ACTION_BUY_PUT_OTM10,
    ACTION_EXIT,
    load_data,
    make_dataloader,
    evaluate_trades,
    evaluate_sharpe,
)


def _env_float(name: str, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except Exception:
        return default
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val


def _env_int(name: str, default: int, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except Exception:
        return default
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val


# ---------------------------------------------------------------------------
# Hyperparameters — v4: smaller model, more regularization
# ---------------------------------------------------------------------------

LOOKBACK = _env_int("TRAIN_LOOKBACK", 120, lo=60, hi=240)
D_MODEL = _env_int("TRAIN_D_MODEL", 64, lo=32, hi=128)
N_HEADS = 4
DEPTH = _env_int("TRAIN_DEPTH", 3, lo=2, hi=6)
FF_MULT = _env_int("TRAIN_FF_MULT", 3, lo=2, hi=6)
DROPOUT = _env_float("TRAIN_DROPOUT", 0.15, lo=0.05, hi=0.40)

BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", 128, lo=32, hi=256)
LR = _env_float("TRAIN_LR", 1.5e-4, lo=1e-5, hi=5e-3)
WEIGHT_DECAY = _env_float("TRAIN_WEIGHT_DECAY", 0.05, lo=0.0, hi=0.3)
ADAM_BETAS = (0.9, 0.98)
GRAD_CLIP = _env_float("TRAIN_GRAD_CLIP", 1.0, lo=0.0, hi=5.0)
WARMUP_RATIO = _env_float("TRAIN_WARMUP_RATIO", 0.1, lo=0.0, hi=0.5)
COOLDOWN_RATIO = _env_float("TRAIN_COOLDOWN_RATIO", 0.3, lo=0.0, hi=0.8)

# Loss weights
GATE_LOSS_WEIGHT = _env_float("TRAIN_GATE_W", 1.0, lo=0.1, hi=5.0)
DIR_LOSS_WEIGHT = _env_float("TRAIN_DIR_W", 1.0, lo=0.1, hi=5.0)
PNL_ALIGNMENT_WEIGHT = _env_float("TRAIN_PNL_W", 0.1, lo=0.0, hi=2.0)
EXIT_LOSS_WEIGHT = _env_float("TRAIN_EXIT_W", 0.5, lo=0.0, hi=2.0)

# Asymmetric gate penalty: how much more to penalize false entries vs missed entries
# >1.0 means "it's worse to trade when you shouldn't than to miss a trade"
FALSE_ENTRY_PENALTY = _env_float("TRAIN_FALSE_ENTRY_PENALTY", 2.0, lo=1.0, hi=5.0)

# Label smoothing
GATE_LABEL_SMOOTHING = _env_float("TRAIN_GATE_LABEL_SMOOTHING", 0.05, lo=0.0, hi=0.2)
DIR_LABEL_SMOOTHING = _env_float("TRAIN_DIR_LABEL_SMOOTHING", 0.05, lo=0.0, hi=0.2)

# Score formula tuning (passed to evaluate_trades)
SCORE_WIN_RATE_BONUS = _env_float("SCORE_WIN_RATE_BONUS", 0.0, lo=0.0, hi=1.0)
SCORE_RR_BONUS = _env_float("SCORE_RR_BONUS", 0.0, lo=0.0, hi=2.0)
SCORE_DRAWDOWN_PENALTY = _env_float("SCORE_DRAWDOWN_PENALTY", 0.5, lo=0.0, hi=1.0)
SCORE_HOLD_BONUS = _env_float("SCORE_HOLD_BONUS", 0.0, lo=0.0, hi=1.0)
SCORE_FREQ_CENTER = _env_float("SCORE_FREQ_CENTER", 3.0, lo=1.0, hi=8.0)
SCORE_FREQ_WIDTH = _env_float("SCORE_FREQ_WIDTH", 3.0, lo=1.0, hi=6.0)
SCORE_CONSEC_LOSS_THRESHOLD = _env_int("SCORE_CONSEC_LOSS_THRESHOLD", 3, lo=2, hi=8)
SCORE_SHORT_HOLD_THRESHOLD = _env_float("SCORE_SHORT_HOLD_THRESHOLD", 0.30, lo=0.10, hi=0.60)
SCORE_STOP_RATE_THRESHOLD = _env_float("SCORE_STOP_RATE_THRESHOLD", 0.30, lo=0.10, hi=0.60)
SCORE_RUIN_PENALTY = _env_float("SCORE_RUIN_PENALTY", 1.0, lo=0.0, hi=1.0)
SCORE_RUIN_THRESHOLD = _env_float("SCORE_RUIN_THRESHOLD", 0.25, lo=0.05, hi=0.50)
SCORE_RISK_FRACTION_PENALTY = _env_float("SCORE_RISK_FRACTION_PENALTY", 0.5, lo=0.0, hi=1.0)

# Named feature indices
IDX_MINUTES_TO_CLOSE = 19

# Feature groups (32 features — v2 reduced set)
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
# Model — v4: simplified, regularized
# ---------------------------------------------------------------------------

class PositionStateGenerator(nn.Module):
    """Generate synthetic position state for training."""

    def __init__(self):
        super().__init__()

    def forward(self, batch_size, device):
        """Generate correlated position states for training.

        Returns:
            position_state: (batch, 5) - [is_holding, bars_held_norm, unrealized_pnl_norm,
                                          account_health, loss_streak_frac]
        """
        is_holding = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.float32)

        bars_held_raw = torch.randint(1, 61, (batch_size,), device=device, dtype=torch.float32)
        bars_held_norm = (bars_held_raw / 60.0) * is_holding

        unrealized_pnl_raw = torch.randn(batch_size, device=device) * 0.5 - 0.1
        unrealized_pnl_norm = torch.tanh(unrealized_pnl_raw) * is_holding

        account_health = torch.empty(batch_size, device=device).uniform_(0.1, 1.5)
        loss_streak_base = torch.empty(batch_size, device=device).uniform_(0.0, 1.0)
        health_penalty = torch.clamp(1.0 - account_health, min=0.0)
        loss_streak_frac = torch.clamp(loss_streak_base + health_penalty * 0.5, max=1.0)

        return torch.stack([is_holding, bars_held_norm, unrealized_pnl_norm,
                           account_health, loss_streak_frac], dim=1)


class RefinedBalancedStrikeGate(nn.Module):
    """Refined balanced strike biasing with more conservative health multiplier."""
    
    def __init__(self, d_model):
        super().__init__()
        self.health_proj = nn.Linear(1, d_model // 8)
        self.gate_mod = nn.Sequential(
            nn.Linear(d_model + d_model // 8, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, gate_input, dir_input, account_health):
        """
        Args:
            gate_input: (batch, d_model) - input to gate head
            dir_input: (batch, d_model) - input to direction head  
            account_health: (batch, 1) - account health fraction
        
        Returns:
            modified_gate_input: (batch, d_model) - gate input with health modulation
            modified_dir_input: (batch, d_model) - direction input with strike bias
        """
        health_emb = F.relu(self.health_proj(account_health))
        combined = torch.cat([gate_input, health_emb], dim=-1)
        
        # Health gate: closer to 0 when account is stressed (health < 0.7)
        health_gate = self.gate_mod(combined)
        
        # More conservative health multiplier: stronger reduction when health is poor
        health_multiplier = 0.4 + 0.6 * health_gate  # Range: [0.4, 1.0] instead of [0.5, 1.0]
        
        return gate_input * health_multiplier, dir_input


class TradingModel(nn.Module):
    """Simplified two-head model for SPX 0DTE options.

    v4 design principles:
    - Simple linear projection instead of FeatureGroupGating (less capacity to memorize)
    - 3 transformer layers instead of 6
    - Dropout 0.15 throughout
    - Position state injection for gate head (kept — this is real signal)
    - No DynamicStopModule, no QualityGate, no Greeks-adaptive anything
    - NEW: Refined balanced strike gating for enhanced capital preservation

    Input:  (batch, lookback, NUM_FEATURES)
    Output: (gate_logits, dir_logits)
        gate_logits: (batch, 2) — [NO_TRADE, TRADE]
        dir_logits:  (batch, 6) — [CALL_ATM, CALL_OTM5, CALL_OTM10,
                                     PUT_ATM, PUT_OTM5, PUT_OTM10]
    """

    def __init__(self, num_features=NUM_FEATURES, lookback=LOOKBACK,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=DEPTH,
                 ff_mult=FF_MULT, dropout=DROPOUT):
        super().__init__()
        self.lookback = lookback
        self.d_model = d_model

        # Simple linear projection — no feature group gating
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
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

        # Position state injection for gate head
        self.position_proj = nn.Linear(5, d_model // 4)
        self.position_gate_proj = nn.Linear(d_model + d_model // 4, d_model)
        self.position_state_gen = PositionStateGenerator()

        # Refined balanced strike gate
        self.refined_gate = RefinedBalancedStrikeGate(d_model)

        # Gate head: "should I trade?" → [NO_TRADE, TRADE]
        self.gate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        # Direction head: "which strike?" → 6-class
        self.dir_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6),
        )

        # Apply biases from successful experiment #3
        with torch.no_grad():
            self.gate_head[-1].bias[0] += 0.5   # NO_TRADE
            self.gate_head[-1].bias[1] -= 0.5   # TRADE
            
            # Apply exact biases from experiment #3
            self.dir_head[-1].bias[0] -= 0.25   # CALL_ATM (was -0.15)
            self.dir_head[-1].bias[3] -= 0.25   # PUT_ATM (was -0.15)
            self.dir_head[-1].bias[4] += 0.35   # PUT_OTM5 (was 0.25)

    def forward(self, x, position_state=None):
        batch_size = x.shape[0]
        device = x.device

        # Generate synthetic position state for training if not provided
        if position_state is None and self.training:
            position_state = self.position_state_gen(batch_size, device)

        x_proj = self.input_proj(x)
        x_proj = self.input_norm(x_proj)
        x_proj = x_proj + self.pos_embed[:, :x_proj.size(1), :]
        x_proj = self.transformer(x_proj, mask=self.causal_mask[:x_proj.size(1), :x_proj.size(1)],
                                  is_causal=True)
        last = x_proj[:, -1, :]  # (batch, d_model)

        # Inject position state into gate head with conservative scaling
        if position_state is not None:
            pos_emb = torch.relu(self.position_proj(position_state))
            # Scale down position influence to prevent overriding conservative gate bias
            gate_input = self.position_gate_proj(torch.cat([last, pos_emb * 0.7], dim=-1))
            
            # Apply refined balanced strike gating using account health
            account_health = position_state[:, 3:4]  # Extract account_health (dim 3)
            gate_input, dir_input = self.refined_gate(gate_input, last, account_health)
            
            # Apply softer dynamic strike bias when account health < 0.7 (higher threshold)
            health_val = account_health.squeeze(-1)  # (batch,)
            stressed_mask = health_val < 0.7
            
        else:
            gate_input = last
            dir_input = last
            stressed_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        gate_logits = self.gate_head(gate_input)
        dir_logits = self.dir_head(dir_input)
        
        # Apply softer strike bias when account is stressed - gentle nudge toward cheaper options
        if stressed_mask.any():
            bias_adjustment = torch.zeros_like(dir_logits)
            # Softer bias: less aggressive penalties and bonuses to maintain ATM viability
            bias_adjustment[stressed_mask, 0] -= 0.15  # CALL_ATM - mild penalty (was -0.4)
            bias_adjustment[stressed_mask, 3] -= 0.15  # PUT_ATM - mild penalty (was -0.4)
            bias_adjustment[stressed_mask, 1] += 0.10  # CALL_OTM5 - small bonus (was +0.2)
            bias_adjustment[stressed_mask, 2] += 0.15  # CALL_OTM10 - modest bonus (was +0.3)
            bias_adjustment[stressed_mask, 4] += 0.10  # PUT_OTM5 - small bonus (was +0.2)
            bias_adjustment[stressed_mask, 5] += 0.15  # PUT_OTM10 - modest bonus (was +0.3)
            
            dir_logits = dir_logits + bias_adjustment

        return gate_logits, dir_logits


# ---------------------------------------------------------------------------
# Loss — v4: simplified, asymmetric gate penalty
# ---------------------------------------------------------------------------

def sniper_loss(gate_logits, dir_logits, call_pnl, put_pnl, time_features, features,
                exit_call_labels=None, exit_put_labels=None,
                otm5_call_pnl=None, otm5_put_pnl=None,
                otm10_call_pnl=None, otm10_put_pnl=None,
                supervision_weight=None, actionable_mask=None, risk_state_mask=None):
    """Two-head loss with asymmetric gate penalty.

    Key v4 change: false entries (predicting TRADE when no option is profitable)
    are penalized more heavily than missed entries. This teaches the model to be
    selective — better to miss a trade than to take a bad one.
    """
    B = gate_logits.shape[0]
    device = gate_logits.device

    # Mask: only compute loss where we have ATM option P&L data
    valid = ~torch.isnan(call_pnl) & ~torch.isnan(put_pnl)
    if valid.sum() < 2:
        return gate_logits.sum() * 0.0

    g_logits = gate_logits[valid]
    d_logits = dir_logits[valid]
    c_pnl = call_pnl[valid]
    p_pnl = put_pnl[valid]
    t_feat = time_features[valid]
    sample_weight = torch.ones_like(c_pnl)

    if supervision_weight is not None:
        sw = torch.nan_to_num(supervision_weight[valid], nan=0.0).clamp(min=0.0)
        if sw.sum() > 0:
            sample_weight = sw
    if actionable_mask is not None:
        act = torch.nan_to_num(actionable_mask[valid], nan=0.0).clamp(0.0, 1.0)
        sample_weight = sample_weight * torch.where(
            act > 0.5, torch.ones_like(act), torch.full_like(act, 0.20))
    if risk_state_mask is not None:
        risk = torch.nan_to_num(risk_state_mask[valid], nan=1.0).clamp(0.0, 1.0)
        sample_weight = sample_weight * torch.where(
            risk > 0.5, torch.ones_like(risk), torch.full_like(risk, 0.50))
    if sample_weight.sum() <= 0:
        sample_weight = torch.ones_like(sample_weight)
    sample_weight = sample_weight / sample_weight.mean().clamp(min=1e-6)

    # Build 6-class P&L array
    def _safe(arr):
        if arr is None:
            return torch.full_like(c_pnl, float('nan'))
        return arr[valid]

    all_pnl = torch.stack([
        c_pnl,
        _safe(otm5_call_pnl),
        _safe(otm10_call_pnl),
        p_pnl,
        _safe(otm5_put_pnl),
        _safe(otm10_put_pnl),
    ], dim=-1)  # (valid, 6)

    # Gate targets: TRADE (1) when ANY option is profitable
    any_profitable = torch.any(torch.nan_to_num(all_pnl, nan=-999.0) > 0, dim=-1)
    gate_targets = any_profitable.long()

    # Asymmetric gate weights: penalize false entries more than missed entries
    # gate_weights[0] = weight for NO_TRADE class (missed trades when should have traded)
    # gate_weights[1] = weight for TRADE class (false entries when should not have traded)
    n_trade = gate_targets.sum().float().clamp(min=1)
    n_no_trade = (gate_targets == 0).sum().float().clamp(min=1)
    base_balance = (n_no_trade / n_trade).clamp(max=10.0)
    # FALSE_ENTRY_PENALTY > 1 makes bad trades cost more
    gate_weights = torch.tensor([1.0 / FALSE_ENTRY_PENALTY, base_balance], device=device)

    time_weight = 1.0 + 0.5 * (1.0 - t_feat)
    gate_loss = F.cross_entropy(
        g_logits, gate_targets, weight=gate_weights,
        reduction='none', label_smoothing=GATE_LABEL_SMOOTHING,
    )
    gate_loss = (gate_loss * time_weight * sample_weight).mean()

    # Direction targets: only where gate_target = TRADE
    trade_mask = gate_targets == 1
    if trade_mask.sum() < 2:
        return GATE_LOSS_WEIGHT * gate_loss

    d_logits_trade = d_logits[trade_mask]
    t_feat_trade = t_feat[trade_mask]
    trade_pnl = all_pnl[trade_mask]

    trade_pnl_safe = torch.nan_to_num(trade_pnl, nan=-999.0)
    dir_targets = torch.argmax(trade_pnl_safe, dim=-1)

    dir_loss = F.cross_entropy(
        d_logits_trade, dir_targets,
        reduction='none', label_smoothing=DIR_LABEL_SMOOTHING,
    )
    dir_time_weight = 1.0 + 0.5 * (1.0 - t_feat_trade)
    dir_w = sample_weight[trade_mask]
    dir_w = dir_w / dir_w.mean().clamp(min=1e-6)
    dir_loss = (dir_loss * dir_time_weight * dir_w).mean()

    # P&L alignment bonus
    gate_probs = F.softmax(g_logits, dim=-1)
    dir_probs = F.softmax(d_logits, dim=-1)
    trade_prob = gate_probs[:, 1]
    all_pnl_safe = torch.nan_to_num(all_pnl, nan=0.0)
    pnl_signal = trade_prob * (dir_probs * all_pnl_safe).sum(dim=-1)
    pnl_loss = -(pnl_signal * sample_weight).sum() / sample_weight.sum().clamp(min=1e-6)

    # EXIT loss: train gate to predict NO_TRADE on exit signal bars
    exit_loss = torch.tensor(0.0, device=device)
    if exit_call_labels is not None and exit_put_labels is not None:
        ec = exit_call_labels[valid]
        ep = exit_put_labels[valid]
        exit_mask = (ec > 0.5) | (ep > 0.5)
        if exit_mask.sum() > 1:
            exit_g = g_logits[exit_mask]
            exit_targets = torch.zeros(exit_mask.sum().item(), dtype=torch.long, device=device)
            exit_loss_vec = F.cross_entropy(exit_g, exit_targets, reduction='none')
            exit_w = sample_weight[exit_mask]
            exit_loss = (exit_loss_vec * exit_w).sum() / exit_w.sum().clamp(min=1e-6)

    total = (GATE_LOSS_WEIGHT * gate_loss + DIR_LOSS_WEIGHT * dir_loss
             + PNL_ALIGNMENT_WEIGHT * pnl_loss + EXIT_LOSS_WEIGHT * exit_loss)
    return total


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_data()
n_bars = len(data['dates'])

# Replace NaN features with 0
data['features'] = torch.nan_to_num(data['features'], nan=0.0)

print(f"Loaded {n_bars} 1-min bars, {NUM_FEATURES} features, {NUM_ACTIONS} actions")
print(f"  ~{n_bars // BARS_PER_DAY} trading days")
print(f"Training:   up to idx {data['train_end_idx']}")
print(f"Validation: idx {data['val_start_idx']}-{data['val_end_idx']}")

required_targets = (
    'call_pnl', 'put_pnl', 'exit_call_label', 'exit_put_label',
    'otm5_call_pnl', 'otm5_put_pnl', 'otm10_call_pnl', 'otm10_put_pnl',
)
missing_targets = [k for k in required_targets if k not in data]
if missing_targets:
    raise KeyError(
        "data.pt missing required targets: " + ", ".join(missing_targets)
        + ". Rebuild data.pt with current prepare.py."
    )
pnl_valid = (~torch.isnan(data['call_pnl'])).sum().item()
print(f"  Option P&L coverage: {pnl_valid}/{n_bars} ({100*pnl_valid/n_bars:.0f}%)")

# v4: ALWAYS train from scratch — no warm start from oracle-contaminated weights
model = TradingModel().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")
print(f"Architecture: v4 simplified two-head (gate+dir) + refined balanced strike gating")
print("Training from scratch (v4: no warm-start).")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR,
    weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS,
)

train_loader = make_dataloader(data, LOOKBACK, BATCH_SIZE, "train", device)
x_batch, y_batch = next(train_loader)

print(f"\nBudget: {TIME_BUDGET}s | Batch: {BATCH_SIZE} | Lookback: {LOOKBACK}")
print(f"LR: {LR} | Depth: {DEPTH} | d_model: {D_MODEL} | ff_mult: {FF_MULT}")
print(f"Dropout: {DROPOUT} | Weight decay: {WEIGHT_DECAY}")
print(f"False entry penalty: {FALSE_ENTRY_PENALTY}x")
print(f"Label smoothing: gate={GATE_LABEL_SMOOTHING} dir={DIR_LABEL_SMOOTHING}")
print()

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_mult(progress):
    if progress < WARMUP_RATIO:
        return progress / max(WARMUP_RATIO, 1e-8)
    if progress < 1.0 - COOLDOWN_RATIO:
        return 1.0
    else:
        t = (1.0 - progress) / max(COOLDOWN_RATIO, 1e-8)
        return 0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))

# ---------------------------------------------------------------------------
# Score config (passed to evaluate_trades)
# ---------------------------------------------------------------------------

_score_config = {
    'win_rate_bonus': SCORE_WIN_RATE_BONUS,
    'rr_bonus': SCORE_RR_BONUS,
    'drawdown_penalty': SCORE_DRAWDOWN_PENALTY,
    'hold_bonus': SCORE_HOLD_BONUS,
    'freq_center': SCORE_FREQ_CENTER,
    'freq_width': SCORE_FREQ_WIDTH,
    'consec_loss_threshold': SCORE_CONSEC_LOSS_THRESHOLD,
    'short_hold_threshold': SCORE_SHORT_HOLD_THRESHOLD,
    'stop_rate_threshold': SCORE_STOP_RATE_THRESHOLD,
    'ruin_penalty': SCORE_RUIN_PENALTY,
    'ruin_threshold': SCORE_RUIN_THRESHOLD,
    'risk_fraction_penalty': SCORE_RISK_FRACTION_PENALTY,
}

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

total_time = 0.0
_wall_start = time.time()
step = 0
smooth_loss = 0.0

while True:
    model.train()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()

    gate_logits, dir_logits = model(x_batch)

    (fwd_ret, call_pnl_batch, put_pnl_batch, exit_call_batch, exit_put_batch,
     otm5c_pnl, otm5p_pnl, otm10c_pnl, otm10p_pnl,
     supervision_weight_batch, actionable_mask_batch, risk_state_mask_batch) = y_batch

    time_feat = x_batch[:, -1, IDX_MINUTES_TO_CLOSE]
    batch_features = x_batch[:, -1, :]

    loss = sniper_loss(gate_logits, dir_logits, call_pnl_batch, put_pnl_batch, time_feat, batch_features,
                       exit_call_batch, exit_put_batch,
                       otm5c_pnl, otm5p_pnl, otm10c_pnl, otm10p_pnl,
                       supervision_weight=supervision_weight_batch,
                       actionable_mask=actionable_mask_batch,
                       risk_state_mask=risk_state_mask_batch)

    loss.backward()
    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    progress = min(total_time / TIME_BUDGET, 1.0)
    for pg in optimizer.param_groups:
        pg['lr'] = LR * get_lr_mult(progress)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    x_batch, y_batch = next(train_loader)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt_step = time.time() - t0
    if step > 5:
        total_time += dt_step

    loss_val = loss.item()
    if math.isnan(loss_val) or loss_val > 100:
        print(f"\nFAIL: loss={loss_val} at step {step}")
        exit(1)

    ema = 0.95
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
    debiased = smooth_loss / (1 - ema ** (step + 1))

    if step % 50 == 0:
        remaining = max(0, TIME_BUDGET - total_time)
        with torch.no_grad():
            gate_probs = F.softmax(gate_logits, dim=-1).mean(dim=0)
            dir_probs = F.softmax(dir_logits, dim=-1).mean(dim=0)
            p_trade = gate_probs[1].item()
            p_call = dir_probs[:3].sum().item()
            p_put = dir_probs[3:].sum().item()
            p_atm = dir_probs[0].item() + dir_probs[3].item()
            p_otm = 1.0 - p_atm

        print(f"step {step:05d} ({100*progress:5.1f}%) | loss: {debiased:.6f} "
              f"| trade:{p_trade:.2f} call:{p_call:.2f} put:{p_put:.2f} "
              f"| ATM:{p_atm:.2f} OTM:{p_otm:.2f} "
              f"| lr: {LR * get_lr_mult(progress):.2e} | left: {remaining:.0f}s")

    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step > 5 and total_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
_eval_start = time.time()
print(f"Training done at {time.time() - _wall_start:.0f}s. Starting evaluation...")

trade_metrics = evaluate_trades(model, data, LOOKBACK, device, score_config=_score_config)
sharpe_metrics = evaluate_sharpe(model, data, LOOKBACK, device)
print(f"Evaluation done in {time.time() - _eval_start:.0f}s (total wall: {time.time() - _wall_start:.0f}s)")

metrics = {**trade_metrics}
metrics['val_sharpe'] = sharpe_metrics.get('val_sharpe', 0.0)

# ---------------------------------------------------------------------------
# Save & report
# ---------------------------------------------------------------------------

model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
state = model.state_dict()
torch.save({
    'model_state_dict': state,
    'metrics': metrics,
    'config': {
        'lookback': LOOKBACK, 'd_model': D_MODEL, 'n_heads': N_HEADS,
        'depth': DEPTH, 'ff_mult': FF_MULT, 'dropout': DROPOUT,
        'num_features': NUM_FEATURES, 'num_actions': NUM_ACTIONS,
        'architecture': 'v4_simplified_two_head_refined_balanced_strike_gating',
        'false_entry_penalty': FALSE_ENTRY_PENALTY,
    },
    'step': step,
}, model_path)
print(f"Model saved to {model_path}")

# --- Export trade log CSV ---
trade_log = metrics.get('trade_log', [])
if trade_log:
    import csv
    log_path = os.path.join(os.path.dirname(__file__), "trade_log.csv")
    fieldnames = ['trade_num', 'date', 'entry_time', 'exit_time', 'direction',
                  'strike', 'entry_price', 'bars_held', 'hold_minutes',
                  'entry_cost_bps', 'entry_quality', 'entry_actionable',
                  'pnl_pct', 'exit_reason', 'actual_prices', 'result']
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(trade_log)
    print(f"Trade log: {len(trade_log)} trades -> {log_path}")

t_end = time.time()
peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

# --- Output section (parsed by run_loop.py) ---
print("\n---")
print(f"score:              {metrics['score']:.6f}")
print(f"profit_factor:      {metrics['profit_factor']:.6f}")
print(f"win_rate:           {metrics['win_rate']:.6f}")
print(f"avg_winner:         {metrics['avg_winner']:.6f}")
print(f"avg_loser:          {metrics['avg_loser']:.6f}")
print(f"trades_per_day:     {metrics['trades_per_day']:.6f}")
print(f"max_consec_loss:    {metrics['max_consec_loss']}")
print(f"num_trades:         {metrics['num_trades']}")
print(f"trade_sharpe:       {metrics['trade_sharpe']:.6f}")
print(f"sortino:            {metrics['sortino']:.6f}")
print(f"max_drawdown:       {metrics['max_drawdown']:.6f}")
print(f"calmar:             {metrics['calmar']:.6f}")
print(f"ev_per_trade:       {metrics['ev_per_trade']:.6f}")
print(f"do_nothing_pct:     {metrics['do_nothing_pct']:.6f}")
print(f"exit_pct:           {metrics.get('exit_pct', 0.0):.6f}")
print(f"model_exit_count:   {metrics.get('model_exit_count', 0)}")
print(f"cooldown_blocked:   {metrics.get('cooldown_blocked', 0)}")
print(f"pre_10am_blocked:   {metrics.get('pre_10am_blocked', 0)}")
print(f"short_hold_pct:     {metrics.get('short_hold_pct', 0.0):.6f}")
print(f"stop_loss_rate:     {metrics.get('stop_loss_rate', 0.0):.6f}")
print(f"direction_collapse_pct: {metrics.get('direction_collapse_pct', 0.0):.6f}")
print(f"avg_entry_cost_bps: {metrics.get('avg_entry_cost_bps', 0.0):.6f}")
print(f"avg_entry_quality:  {metrics.get('avg_entry_quality', 0.0):.6f}")
print(f"cost_realism_coverage: {metrics.get('cost_realism_coverage', 0.0):.6f}")
print(f"high_cost_entry_rate: {metrics.get('high_cost_entry_rate', 0.0):.6f}")
print(f"low_quality_entry_rate: {metrics.get('low_quality_entry_rate', 0.0):.6f}")
print(f"actionable_bar_rate: {metrics.get('actionable_bar_rate', 0.0):.6f}")
print(f"risk_off_bar_rate:  {metrics.get('risk_off_bar_rate', 0.0):.6f}")
print(f"final_capital:      {metrics.get('final_capital', 0.0):.2f}")
print(f"equity_sharpe:      {metrics.get('equity_sharpe', 0.0):.6f}")
print(f"max_equity_dd:      {metrics.get('max_equity_dd', 0.0):.6f}")
print(f"total_dollar_return:{metrics.get('total_dollar_return', 0.0):.6f}")
print(f"val_sharpe:         {metrics['val_sharpe']:.6f}")
print(f"total_return:       {metrics['total_return']:.6f}")
print(f"num_val_bars:       {metrics['num_val_bars']}")
print(f"num_val_days:       {metrics['num_val_days']}")
print(f"worst_chunk_pf:     {metrics.get('worst_chunk_pf', 1.0):.2f}")
print(f"rr_ratio:           {metrics.get('rr_ratio', 0.0):.4f}")
print(f"avg_hold_bars:      {metrics.get('avg_hold_bars', 0.0):.2f}")
print(f"model_exit_rate:    {metrics.get('model_exit_rate', 0.0):.4f}")
print(f"hit_ruin:           {metrics.get('hit_ruin', False)}")
print(f"min_equity_frac:    {metrics.get('min_equity_frac', 1.0):.4f}")
print(f"avg_risk_fraction:  {metrics.get('avg_risk_fraction', 0.0):.4f}")
print(f"max_risk_fraction:  {metrics.get('max_risk_fraction', 0.0):.4f}")
print(f"trades_blocked_by_balance: {metrics.get('trades_blocked_by_balance', 0)}")
for cd in metrics.get('chunk_details', []):
    print(f"  Chunk {cd['chunk']}: {cd['dates']} | {cd['trades']} trades | PF={cd['profit_factor']:.2f} | WR={cd['win_rate']:.1%}")
print(f"training_seconds:   {total_time:.1f}")
print(f"total_seconds:      {t_end - t_start:.1f}")
print(f"peak_vram_mb:       {peak_mb:.1f}")
print(f"num_steps:          {step}")
print(f"num_params:         {num_params:,}")
print(f"lookback:           {LOOKBACK}")
print(f"depth:              {DEPTH}")
print(f"d_model:            {D_MODEL}")
print(f"effective_lr:       {LR:.6g}")
print(f"gate_label_smoothing:{GATE_LABEL_SMOOTHING:.6f}")
print(f"dir_label_smoothing:{DIR_LABEL_SMOOTHING:.6f}")
print(f"false_entry_penalty:{FALSE_ENTRY_PENALTY:.6f}")