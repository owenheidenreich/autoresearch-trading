"""
Autoresearch-trading v3: two-head sniper model for SPX 0DTE options.
Single-GPU, single-file. The agent modifies THIS file.

Two-head architecture:
  Gate head: (batch, 2) — [NO_TRADE, TRADE]
  Direction head: (batch, 6) — [CALL_ATM, CALL_OTM5, CALL_OTM10,
                                  PUT_ATM, PUT_OTM5, PUT_OTM10]

Combined into 8 actions:
  DO_NOTHING (0), BUY_CALL_ATM (1), BUY_CALL_OTM5 (2), BUY_CALL_OTM10 (3),
  BUY_PUT_ATM (4), BUY_PUT_OTM5 (5), BUY_PUT_OTM10 (6), EXIT (7)

  - Gate=TRADE + Dir=i → BUY action i+1
  - Gate=NO_TRADE while in position → EXIT (handled at inference)
  - Gate=NO_TRADE while flat → DO_NOTHING

Loss: trained on actual SPXW option P&L (ATM + OTM at ±5 and ±10 strikes).
Optimizes composite score = ProfitFactor * TradeSharpe * freq_mult, where:
  - freq_mult uses a bell-curve over trades_per_day (tpd):
    - tpd < 0.5: hard floor score = -10
    - 0.5 <= tpd < 1.5: linear ramp from -5 to raw_score
    - 1.5 <= tpd <= 6.0: min(1, tpd/2)
    - tpd > 6.0: quadratic decay max(0.1, (6/tpd)^2)
  - Three multiplicative penalties (applied only when score > 0):
    - Consecutive loss penalty: -5% per loss beyond 3 (floor 0.5x)
    - Short-hold penalty: penalizes >30% 1-bar trades (floor 0.7x)
    - Stop-loss rate penalty: penalizes >30% stop-outs (floor 0.5x)
  - Negative scores ARE possible and informative (PF<1 yields negative Sharpe)

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
# Hyperparameters
# ---------------------------------------------------------------------------

LOOKBACK = _env_int("TRAIN_LOOKBACK", 120, lo=60, hi=240)  # 1-min bars of context
D_MODEL = 96             # embedding dim — LOCKED (must match best_model.pt)
N_HEADS = 4              # attention heads — LOCKED
DEPTH = 6                # transformer layers — LOCKED
FF_MULT = _env_int("TRAIN_FF_MULT", 4, lo=2, hi=8)
DROPOUT = _env_float("TRAIN_DROPOUT", 0.03, lo=0.0, hi=0.2)

BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", 128, lo=32, hi=256)
LR = _env_float("TRAIN_LR", 3e-4, lo=1e-5, hi=5e-3)
WEIGHT_DECAY = _env_float("TRAIN_WEIGHT_DECAY", 0.01, lo=0.0, hi=0.2)
ADAM_BETAS = (0.9, 0.98)
GRAD_CLIP = _env_float("TRAIN_GRAD_CLIP", 1.0, lo=0.0, hi=5.0)
WARMUP_RATIO = _env_float("TRAIN_WARMUP_RATIO", 0.1, lo=0.0, hi=0.5)
COOLDOWN_RATIO = _env_float("TRAIN_COOLDOWN_RATIO", 0.3, lo=0.0, hi=0.8)

# Base loss mixing (will be dynamically adjusted by Greeks)
BASE_GATE_LOSS_WEIGHT = _env_float("TRAIN_BASE_GATE_W", 1.0, lo=0.1, hi=5.0)
BASE_DIR_LOSS_WEIGHT = _env_float("TRAIN_BASE_DIR_W", 1.0, lo=0.1, hi=5.0)
BASE_PNL_ALIGNMENT_WEIGHT = _env_float("TRAIN_BASE_PNL_W", 0.1, lo=0.0, hi=2.0)
BASE_EXIT_LOSS_WEIGHT = _env_float("TRAIN_BASE_EXIT_W", 0.3, lo=0.0, hi=2.0)

# Greeks-adaptive loss parameters
GREEKS_ADAPTATION_STRENGTH = _env_float("TRAIN_GREEKS_ADAPT", 0.5, lo=0.0, hi=1.5)

# Quality gate parameters
QUALITY_GATE_STRENGTH = _env_float("TRAIN_QUALITY_GATE", 0.6, lo=0.0, hi=2.0)

# Position state parameters
POSITION_STATE_WEIGHT = _env_float("TRAIN_POSITION_STATE_WEIGHT", 1.0, lo=0.0, hi=3.0)

# Extra loop controls (foundation-safe; defaults preserve current behavior)
GATE_LABEL_SMOOTHING = _env_float("TRAIN_GATE_LABEL_SMOOTHING", 0.0, lo=0.0, hi=0.2)
DIR_LABEL_SMOOTHING = _env_float("TRAIN_DIR_LABEL_SMOOTHING", 0.0, lo=0.0, hi=0.2)
TRADE_RATE_REG_WEIGHT = _env_float("TRAIN_TRADE_RATE_REG_WEIGHT", 0.0, lo=0.0, hi=2.0)
TARGET_TRADE_RATE = _env_float("TRAIN_TARGET_TRADE_RATE", 0.40, lo=0.05, hi=0.95)

# Score formula tuning (all defaults = neutral/unchanged; agent activates as needed)
SCORE_WIN_RATE_BONUS = _env_float("SCORE_WIN_RATE_BONUS", 0.0, lo=0.0, hi=1.0)
SCORE_RR_BONUS = _env_float("SCORE_RR_BONUS", 0.0, lo=0.0, hi=2.0)
SCORE_DRAWDOWN_PENALTY = _env_float("SCORE_DRAWDOWN_PENALTY", 0.0, lo=0.0, hi=1.0)
SCORE_HOLD_BONUS = _env_float("SCORE_HOLD_BONUS", 0.0, lo=0.0, hi=1.0)
SCORE_FREQ_CENTER = _env_float("SCORE_FREQ_CENTER", 3.0, lo=1.0, hi=8.0)
SCORE_FREQ_WIDTH = _env_float("SCORE_FREQ_WIDTH", 3.0, lo=1.0, hi=6.0)
SCORE_CONSEC_LOSS_THRESHOLD = _env_int("SCORE_CONSEC_LOSS_THRESHOLD", 3, lo=2, hi=8)
SCORE_SHORT_HOLD_THRESHOLD = _env_float("SCORE_SHORT_HOLD_THRESHOLD", 0.30, lo=0.10, hi=0.60)
SCORE_STOP_RATE_THRESHOLD = _env_float("SCORE_STOP_RATE_THRESHOLD", 0.30, lo=0.10, hi=0.60)

# microgpt-inspired architecture options (all default to current behavior)
USE_RMSNORM = _env_int("TRAIN_USE_RMSNORM", 0, lo=0, hi=1)    # 1 = RMSNorm instead of LayerNorm
USE_NO_BIAS = _env_int("TRAIN_NO_BIAS", 0, lo=0, hi=1)        # 1 = no biases in linear layers
LR_SCHEDULE = os.environ.get("TRAIN_LR_SCHEDULE", "cosine")    # "cosine" or "linear"

# Feature groups for gating (70 features: 39 equity + 6 options + 4 VIX/regime + 6 OTM/skew + 5 Greeks + 10 extended)
FEATURE_GROUPS = {
    'returns':   (0, 5),     # ret_1..ret_24
    'volume':    (5, 8),     # volume_ratio, volume_zscore, vol_at_price
    'vol':       (8, 11),    # bar_range, realized_vol, range_ratio
    'vwap':      (11, 17),   # vwap_dist, slope, upper1, lower1, upper2, lower2
    'session':   (17, 22),   # ib_high_dist, ib_low_dist, ib_width, am_range, session_range
    'levels':    (22, 28),   # onh, onl, prev_h, prev_l, prev_c, prev_vwap
    'trend':     (28, 31),   # trend_hh_hl, ema_cross, close_position
    'micro':     (31, 33),   # gap, inside_bar
    'time':      (33, 39),   # minutes_to_close, sin, cos, dow, half_hour, ib_complete
    'options':   (39, 45),   # atm_iv, iv_skew, atm_premium_pct, put_call_vol, opt_vol, theta_rate
    'vix':       (45, 49),   # vix_level, vix_change, vix_regime, vrp (regime awareness)
    'otm':       (49, 55),   # otm_call_iv, otm_put_iv, iv_skew_5, iv_term_call, iv_term_put, otm_vol_ratio
    'greeks':    (55, 60),   # delta, gamma, theta, vega, gamma_theta_ratio
    'ext_greeks': (60, 62),  # charm_estimate, vanna_estimate
    'bollinger': (62, 64),   # bollinger_position, bollinger_width
    'momentum':  (64, 66),   # consec_direction, speed_estimate
    'vwap_ext':  (66, 67),   # vwap_crosses
    'range_ext': (67, 70),   # session_range_position, rsi_14, atr_ratio
}

# Named feature indices (derived from FEATURE_GROUPS, avoids magic numbers)
IDX_MINUTES_TO_CLOSE = 33   # time group: minutes_to_close
IDX_GAMMA_THETA_RATIO = 59  # greeks group: gamma/|theta| ratio


class RMSNorm(nn.Module):
    """RMSNorm (from microgpt/LLaMA) — cheaper than LayerNorm, no mean centering."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(ms + self.eps)
        return self.weight * x


# Choose normalization based on config
NormLayer = RMSNorm if USE_RMSNORM else nn.LayerNorm

# ---------------------------------------------------------------------------
# Model
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


class QualityGate(nn.Module):
    """Quality gate that conditions TRADE probability on gamma_theta_ratio."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, gate_logits, gamma_theta_ratio):
        """Apply quality scaling to TRADE logit based on gamma_theta_ratio.
        
        Args:
            gate_logits: (batch, 2) - [NO_TRADE, TRADE]
            gamma_theta_ratio: (batch,) - gamma/|theta| ratio from features
            
        Returns:
            Adjusted gate_logits with TRADE logit scaled by quality
        """
        # Handle NaN values - replace with neutral value of 1.0
        gtr = torch.nan_to_num(gamma_theta_ratio, nan=1.0)
        
        # Quality score: sigmoid(gamma_theta_ratio - 1.0) 
        # >1 = good (approaches 1.0), <1 = bad (approaches 0.0)
        quality_score = torch.sigmoid(gtr - 1.0)
        
        # Scale quality score: 0.2 to 1.0 range to avoid completely killing trades
        quality_score = 0.2 + 0.8 * quality_score
        
        # Apply quality gate: multiply TRADE logit by quality score
        adjusted_logits = gate_logits.clone()
        adjusted_logits[:, 1] = gate_logits[:, 1] * (1.0 + QUALITY_GATE_STRENGTH * (quality_score - 1.0))
        
        return adjusted_logits


class PositionStateGenerator(nn.Module):
    """Generate synthetic position state for training."""
    
    def __init__(self):
        super().__init__()
        # Random position probabilities for training
        self.register_buffer('_dummy', torch.tensor(0.0))  # For device tracking
        
    def forward(self, batch_size, device):
        """Generate random position states for training.
        
        Returns:
            position_state: (batch, 3) - [is_holding, bars_held_norm, unrealized_pnl_norm]
        """
        # Sample random position states
        is_holding = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.float32)
        
        # Only generate hold time and PnL for positions that are holding
        bars_held_raw = torch.randint(1, 61, (batch_size,), device=device, dtype=torch.float32)  # 1-60 bars
        bars_held_norm = bars_held_raw / 60.0  # Normalize to [0, 1]
        bars_held_norm = bars_held_norm * is_holding  # Zero out when not holding
        
        # Realistic PnL distribution: biased negative (most options lose money)
        unrealized_pnl_raw = torch.randn(batch_size, device=device) * 0.5 - 0.1  # Mean -10%, std 50%
        unrealized_pnl_norm = torch.tanh(unrealized_pnl_raw)  # Clamp to [-1, 1]
        unrealized_pnl_norm = unrealized_pnl_norm * is_holding  # Zero out when not holding
        
        return torch.stack([is_holding, bars_held_norm, unrealized_pnl_norm], dim=1)


class TradingModel(nn.Module):
    """Two-head sniper model for SPX 0DTE options with position state awareness.

    Input:  (batch, lookback, NUM_FEATURES)
    Output: (gate_logits, dir_logits)
        gate_logits: (batch, 2) — [NO_TRADE, TRADE]
        dir_logits:  (batch, 6) — [CALL_ATM, CALL_OTM5, CALL_OTM10,
                                     PUT_ATM, PUT_OTM5, PUT_OTM10]

    The gate head decides "should I be in a trade right now?"
    The direction head decides "which strike and direction?"
    At inference, gate=NO_TRADE while in a position → EXIT signal.

    Position state tensor (batch, 3):
        [is_holding, bars_held_norm, unrealized_pnl_norm]
    The gate head sees position context for smarter entry/exit decisions.
    """

    def __init__(self, num_features=NUM_FEATURES, lookback=LOOKBACK,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=DEPTH,
                 ff_mult=FF_MULT, dropout=DROPOUT):
        super().__init__()
        self.lookback = lookback
        self.d_model = d_model

        _bias = not bool(USE_NO_BIAS)  # microgpt-inspired: no biases option
        self.feature_gate = FeatureGroupGating(num_features, d_model, FEATURE_GROUPS)
        self.input_norm = NormLayer(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult, dropout=dropout,
            batch_first=True, activation='gelu', norm_first=True,
        )
        # Apply no-bias option to transformer sublayers
        if not _bias:
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias = None
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        mask = nn.Transformer.generate_square_subsequent_mask(lookback)
        self.register_buffer('causal_mask', mask)

        # Position state injection for gate head
        self.position_proj = nn.Linear(3, d_model // 4)
        self.position_gate_proj = nn.Linear(d_model + d_model // 4, d_model)
        self.position_state_gen = PositionStateGenerator()

        # Gate head: "should I trade?" → [NO_TRADE, TRADE]
        self.gate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        # Direction head: "which strike?" → [CALL_ATM, CALL_OTM5, CALL_OTM10,
        #                                     PUT_ATM, PUT_OTM5, PUT_OTM10]
        self.dir_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6),
        )

        # Quality gate module
        self.quality_gate = QualityGate()

        # Further reduced bias toward NO_TRADE to encourage trading
        with torch.no_grad():
            self.gate_head[-1].bias[0] = -0.1   # NO_TRADE bias (reduced from -0.2)

    def forward(self, x, position_state=None):
        batch_size = x.shape[0]
        device = x.device
        
        # Generate synthetic position state for training if not provided
        if position_state is None and self.training:
            position_state = self.position_state_gen(batch_size, device)
        
        x_transformed = self.feature_gate(x)
        x_transformed = self.input_norm(x_transformed)
        x_transformed = x_transformed + self.pos_embed[:, :x_transformed.size(1), :]
        x_transformed = self.transformer(x_transformed, mask=self.causal_mask[:x_transformed.size(1), :x_transformed.size(1)],
                              is_causal=True)
        last = x_transformed[:, -1, :]  # (batch, d_model)

        # Inject position state into gate head input
        if position_state is not None:
            pos_emb = torch.relu(self.position_proj(position_state))
            gate_input = self.position_gate_proj(torch.cat([last, pos_emb], dim=-1))
        else:
            gate_input = last

        # Get raw gate and direction logits
        raw_gate_logits = self.gate_head(gate_input)
        dir_logits = self.dir_head(last)  # direction is position-independent

        # Apply quality gate using gamma_theta_ratio from input features
        gamma_theta_ratio = x[:, -1, IDX_GAMMA_THETA_RATIO]  # gamma/|theta| ratio
        gate_logits = self.quality_gate(raw_gate_logits, gamma_theta_ratio)

        return gate_logits, dir_logits


# ---------------------------------------------------------------------------
# Greeks-Adaptive Loss Functions
# ---------------------------------------------------------------------------

def compute_greeks_loss_weights(features):
    """Dynamically adjust loss weights based on Greeks and market conditions.
    
    Args:
        features: (batch, NUM_FEATURES) - current bar features
        
    Returns:
        dict with adjusted loss weights
    """
    device = features.device
    batch_size = features.shape[0]
    
    # Extract Greeks features (indices 55-59)
    atm_delta = features[:, 55]          # 55: ATM delta (0-1)
    atm_gamma = features[:, 56]          # 56: ATM gamma  
    atm_theta_per_bar = features[:, 57]  # 57: theta per 1-min bar (negative)
    atm_vega = features[:, 58]           # 58: vega sensitivity
    gamma_theta_ratio = features[:, IDX_GAMMA_THETA_RATIO]  # gamma/|theta| ratio
    
    # Extract time feature for theta acceleration awareness
    minutes_to_close = features[:, 33]   # 33: minutes to close
    
    # Handle NaN values by replacing with neutral values
    atm_delta = torch.nan_to_num(atm_delta, nan=0.5)  # neutral at 0.5
    atm_gamma = torch.nan_to_num(atm_gamma, nan=0.0)
    atm_theta_per_bar = torch.nan_to_num(atm_theta_per_bar, nan=-0.01)
    atm_vega = torch.nan_to_num(atm_vega, nan=0.1)
    gamma_theta_ratio = torch.nan_to_num(gamma_theta_ratio, nan=1.0)
    minutes_to_close = torch.nan_to_num(minutes_to_close, nan=0.5)
    
    # Normalize features for weight computation
    gtr_norm = torch.sigmoid(gamma_theta_ratio - 1.0)  # >1 = good, <1 = bad
    delta_uncertainty = 4 * atm_delta * (1 - atm_delta)  # max at delta=0.5 (most uncertain)
    theta_accel = torch.sigmoid(5 * (1 - minutes_to_close))  # higher near close
    high_vol_env = torch.sigmoid(atm_vega - 0.2)  # higher when vega > 0.2
    
    # Compute adaptive weights
    # 1. PNL_ALIGNMENT: higher when gamma_theta_ratio is favorable
    pnl_weight = BASE_PNL_ALIGNMENT_WEIGHT * (1 + GREEKS_ADAPTATION_STRENGTH * gtr_norm)
    
    # 2. EXIT_LOSS: higher when theta is accelerating (afternoon)
    exit_weight = BASE_EXIT_LOSS_WEIGHT * (1 + GREEKS_ADAPTATION_STRENGTH * theta_accel)
    
    # 3. DIR_LOSS: lower when delta near 0.5 (direction less predictable)
    dir_weight = BASE_DIR_LOSS_WEIGHT * (1 - GREEKS_ADAPTATION_STRENGTH * delta_uncertainty)
    
    # 4. GATE_LOSS: higher in high-vol environments (be more selective)
    gate_weight = BASE_GATE_LOSS_WEIGHT * (1 + GREEKS_ADAPTATION_STRENGTH * high_vol_env)
    
    # Return batch-averaged weights
    return {
        'gate_weight': gate_weight.mean().item(),
        'dir_weight': dir_weight.mean().item(), 
        'pnl_weight': pnl_weight.mean().item(),
        'exit_weight': exit_weight.mean().item(),
    }


def sniper_loss(gate_logits, dir_logits, call_pnl, put_pnl, time_features, features,
                exit_call_labels=None, exit_put_labels=None,
                otm5_call_pnl=None, otm5_put_pnl=None,
                otm10_call_pnl=None, otm10_put_pnl=None,
                supervision_weight=None, actionable_mask=None, risk_state_mask=None):
    """Two-head loss trained on actual SPXW option P&L with Greeks-adaptive weights.

    gate_logits: (batch, 2) — [NO_TRADE, TRADE]
    dir_logits:  (batch, 6) — [CALL_ATM, CALL_OTM5, CALL_OTM10,
                                PUT_ATM, PUT_OTM5, PUT_OTM10]
    call_pnl:    (batch,) — ATM call option P&L (NaN where unavailable)
    put_pnl:     (batch,) — ATM put option P&L (NaN where unavailable)
    time_features: (batch,) — minutes_to_close (feature index 33), normalized
    features:    (batch, NUM_FEATURES) — all features for Greeks adaptation
    exit_call_labels: (batch,) — 1.0 when call profit target hit (optional)
    exit_put_labels:  (batch,) — 1.0 when put profit target hit (optional)
    otm5_call_pnl:  (batch,) — OTM+5 call P&L (optional, NaN where unavailable)
    otm5_put_pnl:   (batch,) — OTM-5 put P&L (optional, NaN where unavailable)
    otm10_call_pnl: (batch,) — OTM+10 call P&L (optional, NaN where unavailable)
    otm10_put_pnl:  (batch,) — OTM-10 put P&L (optional, NaN where unavailable)

    Gate target: TRADE (1) when any option P&L > 0 (profitable trade exists)
    Direction target: argmax of [call_atm, call_otm5, call_otm10, put_atm, put_otm5, put_otm10] P&L
    EXIT target: gate=NO_TRADE (0) on bars where exit labels fire
    """
    B = gate_logits.shape[0]
    device = gate_logits.device
    if dir_logits.shape[-1] != 6:
        raise ValueError(
            f"sniper_loss requires 6-direction head, got shape {tuple(dir_logits.shape)}"
        )

    # Mask: only compute loss where we have ATM option P&L data
    valid = ~torch.isnan(call_pnl) & ~torch.isnan(put_pnl)
    if valid.sum() < 2:
        return gate_logits.sum() * 0.0

    g_logits = gate_logits[valid]
    d_logits = dir_logits[valid]
    c_pnl = call_pnl[valid]
    p_pnl = put_pnl[valid]
    t_feat = time_features[valid]
    valid_features = features[valid]
    sample_weight = torch.ones_like(c_pnl)
    if supervision_weight is not None:
        sw = torch.nan_to_num(supervision_weight[valid], nan=0.0).clamp(min=0.0)
        if sw.sum() > 0:
            sample_weight = sw
    if actionable_mask is not None:
        act = torch.nan_to_num(actionable_mask[valid], nan=0.0).clamp(0.0, 1.0)
        sample_weight = sample_weight * torch.where(
            act > 0.5,
            torch.ones_like(act),
            torch.full_like(act, 0.20),
        )
    if risk_state_mask is not None:
        risk = torch.nan_to_num(risk_state_mask[valid], nan=1.0).clamp(0.0, 1.0)
        sample_weight = sample_weight * torch.where(
            risk > 0.5,
            torch.ones_like(risk),
            torch.full_like(risk, 0.50),
        )
    if sample_weight.sum() <= 0:
        sample_weight = torch.ones_like(sample_weight)
    # Keep average weight near 1 to avoid LR retuning.
    sample_weight = sample_weight / sample_weight.mean().clamp(min=1e-6)

    # Compute Greeks-adaptive loss weights
    loss_weights = compute_greeks_loss_weights(valid_features)
    GATE_LOSS_WEIGHT = loss_weights['gate_weight']
    DIR_LOSS_WEIGHT = loss_weights['dir_weight']
    PNL_ALIGNMENT_WEIGHT = loss_weights['pnl_weight']
    EXIT_LOSS_WEIGHT = loss_weights['exit_weight']

    # Build 6-class P&L array: [call_atm, call_otm5, call_otm10, put_atm, put_otm5, put_otm10]
    # Missing OTM values stay NaN and are excluded from argmax via nan_to_num(-999).
    def _safe(arr):
        if arr is None:
            return torch.full_like(c_pnl, float('nan'))
        return arr[valid]

    all_pnl = torch.stack([
        c_pnl,             # CALL_ATM
        _safe(otm5_call_pnl),   # CALL_OTM5
        _safe(otm10_call_pnl),  # CALL_OTM10
        p_pnl,             # PUT_ATM
        _safe(otm5_put_pnl),    # PUT_OTM5
        _safe(otm10_put_pnl),   # PUT_OTM10
    ], dim=-1)  # (valid, 6)

    # Gate targets: TRADE (1) when ANY option is profitable
    any_profitable = torch.any(torch.nan_to_num(all_pnl, nan=-999.0) > 0, dim=-1)
    gate_targets = any_profitable.long()

    # Gate class weights
    n_trade = gate_targets.sum().float().clamp(min=1)
    n_no_trade = (gate_targets == 0).sum().float().clamp(min=1)
    gate_weights = torch.tensor([1.0, (n_no_trade / n_trade).clamp(max=10.0)], device=device)

    time_weight = 1.0 + 0.5 * (1.0 - t_feat)
    gate_loss = F.cross_entropy(
        g_logits,
        gate_targets,
        weight=gate_weights,
        reduction='none',
        label_smoothing=GATE_LABEL_SMOOTHING,
    )
    gate_loss = (gate_loss * time_weight * sample_weight).mean()

    # Direction targets: only where gate_target = TRADE
    trade_mask = gate_targets == 1
    if trade_mask.sum() < 2:
        return GATE_LOSS_WEIGHT * gate_loss

    d_logits_trade = d_logits[trade_mask]
    t_feat_trade = t_feat[trade_mask]
    trade_pnl = all_pnl[trade_mask]  # (trade, 6)

    # 6-class: argmax of P&L across all strikes/directions.
    # Replace NaN with -inf so they never win the argmax.
    trade_pnl_safe = torch.nan_to_num(trade_pnl, nan=-999.0)
    dir_targets = torch.argmax(trade_pnl_safe, dim=-1)

    dir_loss = F.cross_entropy(
        d_logits_trade,
        dir_targets,
        reduction='none',
        label_smoothing=DIR_LABEL_SMOOTHING,
    )
    dir_time_weight = 1.0 + 0.5 * (1.0 - t_feat_trade)
    dir_w = sample_weight[trade_mask]
    dir_w = dir_w / dir_w.mean().clamp(min=1e-6)
    dir_loss = (dir_loss * dir_time_weight * dir_w).mean()

    # P&L alignment bonus: weighted sum of dir_probs × actual P&L
    gate_probs = F.softmax(g_logits, dim=-1)
    dir_probs = F.softmax(d_logits, dim=-1)
    trade_prob = gate_probs[:, 1]
    all_pnl_safe = torch.nan_to_num(all_pnl, nan=0.0)
    pnl_signal = trade_prob * (dir_probs * all_pnl_safe).sum(dim=-1)
    pnl_loss = -(pnl_signal * sample_weight).sum() / sample_weight.sum().clamp(min=1e-6)

    # EXIT loss
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

    trade_rate_loss = torch.tensor(0.0, device=device)
    if TRADE_RATE_REG_WEIGHT > 0:
        pred_trade_rate = trade_prob.mean()
        trade_rate_loss = (pred_trade_rate - TARGET_TRADE_RATE) ** 2

    total = (GATE_LOSS_WEIGHT * gate_loss + DIR_LOSS_WEIGHT * dir_loss
             + PNL_ALIGNMENT_WEIGHT * pnl_loss + EXIT_LOSS_WEIGHT * exit_loss
             + TRADE_RATE_REG_WEIGHT * trade_rate_loss)
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

# Replace NaN features with 0 so they don't poison gradients
data['features'] = torch.nan_to_num(data['features'], nan=0.0)

print(f"Loaded {n_bars} 1-min bars, {NUM_FEATURES} features, {NUM_ACTIONS} actions")
print(f"  ~{n_bars // BARS_PER_DAY} trading days")
print(f"Training:   up to idx {data['train_end_idx']}")
print(f"Validation: idx {data['val_start_idx']}-{data['val_end_idx']}")

required_targets = (
    'call_pnl',
    'put_pnl',
    'exit_call_label',
    'exit_put_label',
    'otm5_call_pnl',
    'otm5_put_pnl',
    'otm10_call_pnl',
    'otm10_put_pnl',
)
missing_targets = [k for k in required_targets if k not in data]
if missing_targets:
    raise KeyError(
        "data.pt missing required two-head/OTM targets: "
        + ", ".join(missing_targets)
        + ". Rebuild data.pt with current prepare.py."
    )
pnl_valid = (~torch.isnan(data['call_pnl'])).sum().item()
print(f"  Option P&L coverage: {pnl_valid}/{n_bars} ({100*pnl_valid/n_bars:.0f}%)")

model = TradingModel().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")
print(f"Architecture: two-head (gate: NO_TRADE/TRADE, dir: 6-class ATM+OTM) + Greeks-adaptive loss + Quality gate + Position state")

# Warm-start from previous best if architecture matches
WARM_START = int(os.environ.get("WARM_START", "1"))
_warm_started = False
if WARM_START:
    _best_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    if os.path.exists(_best_path):
        try:
            ckpt = torch.load(_best_path, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if missing:
                print(f"  New layers (will train from scratch): {missing}")
            _warm_started = True
            print(f"Warm-start: loaded weights from best_model.pt (step {ckpt.get('step', '?')})")
            del ckpt
        except Exception as e:
            print(f"Warm-start failed (architecture mismatch?): {e}")
            print("Training from scratch.")

if not _warm_started:
    print("Training from scratch (no warm-start).")

_effective_lr = LR * 0.3 if _warm_started else LR
optimizer = torch.optim.AdamW(
    model.parameters(), lr=_effective_lr,
    weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS,
)

# NOTE: torch.compile disabled — causes OOM on 64Gi Akash containers.
# Model is tiny (< 1M params); compile overhead >> benefit.  Do NOT re-enable.
# SAFETY: BATCH_SIZE must be ≤ 256, D_MODEL ≤ 128, DEPTH ≤ 8.
# Violating these limits WILL crash the container.
if os.environ.get("TORCH_COMPILE", "0") == "1":
    pass  # torch.compile removed
train_loader = make_dataloader(data, LOOKBACK, BATCH_SIZE, "train", device)
x_batch, y_batch = next(train_loader)

print(f"\nBudget: {TIME_BUDGET}s | Batch: {BATCH_SIZE} | Lookback: {LOOKBACK}")
print(f"LR: {LR} (effective: {_effective_lr}) | Depth: {DEPTH} | d_model: {D_MODEL}")
print(f"Warm-start: {'yes' if _warm_started else 'no'}")
print(f"Greeks adaptation strength: {GREEKS_ADAPTATION_STRENGTH}")
print(f"Quality gate strength: {QUALITY_GATE_STRENGTH}")
print(f"Position state weight: {POSITION_STATE_WEIGHT}")
print(f"Label smoothing: gate={GATE_LABEL_SMOOTHING} dir={DIR_LABEL_SMOOTHING}")
print(f"Trade-rate regularizer: weight={TRADE_RATE_REG_WEIGHT} target={TARGET_TRADE_RATE}")
print()

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_mult(progress):
    if progress < WARMUP_RATIO:
        return progress / max(WARMUP_RATIO, 1e-8)
    if LR_SCHEDULE == "linear":
        # Linear decay from 1.0 to 0.0 after warmup (microgpt-style)
        return max(0.0, 1.0 - (progress - WARMUP_RATIO) / max(1.0 - WARMUP_RATIO, 1e-8))
    # Cosine cooldown (default)
    if progress < 1.0 - COOLDOWN_RATIO:
        return 1.0
    else:
        t = (1.0 - progress) / max(COOLDOWN_RATIO, 1e-8)
        return 0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

total_time = 0.0
step = 0
smooth_loss = 0.0

while True:
    model.train()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()

    gate_logits, dir_logits = model(x_batch)

    # Unpack multi-target: (fwd_ret, call_pnl, put_pnl, exit_call, exit_put,
    #   otm5_call_pnl, otm5_put_pnl, otm10_call_pnl, otm10_put_pnl,
    #   supervision_weight, actionable_mask, risk_state_mask)
    (fwd_ret, call_pnl_batch, put_pnl_batch, exit_call_batch, exit_put_batch,
     otm5c_pnl, otm5p_pnl, otm10c_pnl, otm10p_pnl,
     supervision_weight_batch, actionable_mask_batch, risk_state_mask_batch) = y_batch

    # Extract time feature (minutes_to_close) from last bar in lookback window
    time_feat = x_batch[:, -1, IDX_MINUTES_TO_CLOSE]  # minutes_to_close
    
    # Extract full features for Greeks adaptation
    batch_features = x_batch[:, -1, :]  # (batch, NUM_FEATURES)

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
        pg['lr'] = _effective_lr * get_lr_mult(progress)

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
            # Sum call vs put probabilities (ATM + OTM5 + OTM10)
            p_call = dir_probs[:3].sum().item()
            p_put = dir_probs[3:].sum().item()
            
            # Show current Greeks-adapted loss weights
            sample_weights = compute_greeks_loss_weights(batch_features)
            
            # Show average quality gate effect
            gtr_current = batch_features[:, IDX_GAMMA_THETA_RATIO]  # gamma_theta_ratio
            gtr_clean = torch.nan_to_num(gtr_current, nan=1.0)
            avg_quality = torch.sigmoid(gtr_clean - 1.0).mean().item()
            
        print(f"step {step:05d} ({100*progress:5.1f}%) | loss: {debiased:.6f} "
              f"| trade:{p_trade:.2f} call:{p_call:.2f} put:{p_put:.2f} "
              f"| gate_w:{sample_weights['gate_weight']:.2f} pnl_w:{sample_weights['pnl_weight']:.2f} "
              f"| quality:{avg_quality:.2f} | lr: {_effective_lr * get_lr_mult(progress):.2e} | left: {remaining:.0f}s")

    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step > 5 and total_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Evaluate (primary: trade simulation; secondary: Sharpe)
# ---------------------------------------------------------------------------

model.eval()
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
}
trade_metrics = evaluate_trades(model, data, LOOKBACK, device, score_config=_score_config)
sharpe_metrics = evaluate_sharpe(model, data, LOOKBACK, device)

# Merge — trade_metrics is primary, sharpe_metrics only adds val_sharpe
metrics = {**trade_metrics}
metrics['val_sharpe'] = sharpe_metrics.get('val_sharpe', 0.0)

# ---------------------------------------------------------------------------
# Save & report
# ---------------------------------------------------------------------------

model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
state = model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict()
torch.save({
    'model_state_dict': state,
    'metrics': metrics,
    'config': {
        'lookback': LOOKBACK, 'd_model': D_MODEL, 'n_heads': N_HEADS,
        'depth': DEPTH, 'ff_mult': FF_MULT, 'dropout': DROPOUT,
        'num_features': NUM_FEATURES, 'num_actions': NUM_ACTIONS,
        'architecture': 'two_head_greeks_adaptive_quality_gate_position_state',
        'greeks_adaptation_strength': GREEKS_ADAPTATION_STRENGTH,
        'quality_gate_strength': QUALITY_GATE_STRENGTH,
        'position_state_weight': POSITION_STATE_WEIGHT,
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
# Primary optimization target
print(f"score:              {metrics['score']:.6f}")
# Trader metrics
print(f"profit_factor:      {metrics['profit_factor']:.6f}")
print(f"win_rate:           {metrics['win_rate']:.6f}")
print(f"avg_winner:         {metrics['avg_winner']:.6f}")
print(f"avg_loser:          {metrics['avg_loser']:.6f}")
print(f"trades_per_day:     {metrics['trades_per_day']:.6f}")
print(f"max_consec_loss:    {metrics['max_consec_loss']}")
print(f"num_trades:         {metrics['num_trades']}")
# Quant metrics
print(f"trade_sharpe:       {metrics['trade_sharpe']:.6f}")
print(f"sortino:            {metrics['sortino']:.6f}")
print(f"max_drawdown:       {metrics['max_drawdown']:.6f}")
print(f"calmar:             {metrics['calmar']:.6f}")
print(f"ev_per_trade:       {metrics['ev_per_trade']:.6f}")
# Distribution
print(f"do_nothing_pct:     {metrics['do_nothing_pct']:.6f}")
print(f"exit_pct:           {metrics.get('exit_pct', 0.0):.6f}")
print(f"model_exit_count:   {metrics.get('model_exit_count', 0)}")
# Trade quality diagnostics
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
# Equity curve (informational)
print(f"final_capital:      {metrics.get('final_capital', 0.0):.2f}")
print(f"equity_sharpe:      {metrics.get('equity_sharpe', 0.0):.6f}")
print(f"max_equity_dd:      {metrics.get('max_equity_dd', 0.0):.6f}")
print(f"total_dollar_return:{metrics.get('total_dollar_return', 0.0):.6f}")
# Legacy Sharpe
print(f"val_sharpe:         {metrics['val_sharpe']:.6f}")
# Meta
print(f"total_return:       {metrics['total_return']:.6f}")
print(f"num_val_bars:       {metrics['num_val_bars']}")
print(f"num_val_days:       {metrics['num_val_days']}")
print(f"worst_chunk_pf:     {metrics.get('worst_chunk_pf', 1.0):.2f}")
print(f"rr_ratio:           {metrics.get('rr_ratio', 0.0):.4f}")
print(f"avg_hold_bars:      {metrics.get('avg_hold_bars', 0.0):.2f}")
print(f"model_exit_rate:    {metrics.get('model_exit_rate', 0.0):.4f}")
# Walk-forward chunk breakdown
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
print(f"effective_lr:       {_effective_lr:.6g}")
print(f"gate_label_smoothing:{GATE_LABEL_SMOOTHING:.6f}")
print(f"dir_label_smoothing:{DIR_LABEL_SMOOTHING:.6f}")
print(f"trade_rate_reg_weight:{TRADE_RATE_REG_WEIGHT:.6f}")
print(f"target_trade_rate:  {TARGET_TRADE_RATE:.6f}")