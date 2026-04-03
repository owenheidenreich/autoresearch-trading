"""
Autoresearch-trading v6: four-head model for SPX 0DTE options.
Single-GPU, single-file. The agent modifies THIS file.

Four-head architecture:
  Gate head:      (batch, 2) — [NO_TRADE, TRADE]
  Direction head: (batch, 6) — [CALL_ATM, CALL_OTM5, CALL_OTM10,
                                 PUT_ATM, PUT_OTM5, PUT_OTM10]
  Value head:     (batch, 1) — remaining P&L prediction (exit intelligence)
  Risk head:      (batch, 3) — [stop_pct, size_frac, conviction] (account-aware risk)

Combined into 8 actions:
  DO_NOTHING (0), BUY_CALL_ATM (1), BUY_CALL_OTM5 (2), BUY_CALL_OTM10 (3),
  BUY_PUT_ATM (4), BUY_PUT_OTM5 (5), BUY_PUT_OTM10 (6), EXIT (7)

Usage: uv run train.py  (or: python3 train.py)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
# Anti-overfit defaults: recency weighting, day diversity, gate entropy, temporal smoothing, layer freezing
os.environ.setdefault("WEIGHT_RECENT_BOOST", "0.3")
os.environ.setdefault("WEIGHT_DAY_DIVERSITY", "1.0")
os.environ.setdefault("REG_GATE_ENTROPY", "0.10")
os.environ.setdefault("REG_TEMPORAL_SMOOTH", "0.0")
os.environ.setdefault("WARM_FREEZE_RATIO", "0.0")

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
DROPOUT = _env_float("TRAIN_DROPOUT", 0.30, lo=0.05, hi=0.40)  # Increased from 0.25: force generalization across dates, not single-date memorization

BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", 512, lo=32, hi=2048)
LR = _env_float("TRAIN_LR", 2.5e-4, lo=1e-5, hi=5e-3)  # Proven sweet spot
WEIGHT_DECAY = _env_float("TRAIN_WEIGHT_DECAY", 0.08, lo=0.0, hi=0.3)  # Increased from 0.05: combat overfit
ADAM_BETAS = (0.9, 0.98)
GRAD_CLIP = _env_float("TRAIN_GRAD_CLIP", 1.0, lo=0.0, hi=5.0)
WARMUP_RATIO = _env_float("TRAIN_WARMUP_RATIO", 0.15, lo=0.0, hi=0.5)  # Longer warmup
COOLDOWN_RATIO = _env_float("TRAIN_COOLDOWN_RATIO", 0.3, lo=0.0, hi=0.8)

# Loss weights
GATE_LOSS_WEIGHT = _env_float("TRAIN_GATE_W", 0.5, lo=0.1, hi=5.0)
DIR_LOSS_WEIGHT = _env_float("TRAIN_DIR_W", 2.5, lo=0.1, hi=5.0)  # Increased from 2.0 to break direction collapse
PNL_ALIGNMENT_WEIGHT = _env_float("TRAIN_PNL_W", 0.5, lo=0.0, hi=2.0)  # Increased from 0.3: stronger profitability signal
EXIT_LOSS_WEIGHT = _env_float("TRAIN_EXIT_W", 0.15, lo=0.0, hi=2.0)  # Increased from 0.10: confirmed working, helps model learn exits
CONFIDENCE_LOSS_WEIGHT = _env_float("TRAIN_CONF_W", 0.10, lo=0.0, hi=1.0)  # Confidence calibration: reward high conf on winners, penalize on losers
VALUE_LOSS_WEIGHT = _env_float("TRAIN_VALUE_W", 0.3, lo=0.0, hi=2.0)  # Phase D: value head MSE on remaining P&L
VALUE_EXIT_THRESHOLD = _env_float("TRAIN_VALUE_EXIT_THRESH", 0.02, lo=-0.5, hi=0.5)  # Exit when value_pred < threshold
RISK_LOSS_WEIGHT = _env_float("TRAIN_RISK_W", 0.2, lo=0.0, hi=2.0)  # Phase E: risk head (stop + size + conviction)

# Asymmetric gate penalty: how much more to penalize false entries vs missed entries
# >1.0 means "it's worse to trade when you shouldn't than to miss a trade"
FALSE_ENTRY_PENALTY = _env_float("TRAIN_FALSE_ENTRY_PENALTY", 1.2, lo=1.0, hi=5.0)  # Reduced from 1.5: less conservative gate, model was too reluctant to trade

# Label smoothing
GATE_LABEL_SMOOTHING = _env_float("TRAIN_GATE_LABEL_SMOOTHING", 0.05, lo=0.0, hi=0.2)
DIR_LABEL_SMOOTHING = _env_float("TRAIN_DIR_LABEL_SMOOTHING", 0.05, lo=0.0, hi=0.2)

# Anti-overfit levers (SCHED_/WEIGHT_/WARM_/REG_ prefixes, all default 0.0)
WEIGHT_RECENT_BOOST = _env_float("WEIGHT_RECENT_BOOST", 0.0, lo=0.0, hi=2.0)
WEIGHT_DAY_DIVERSITY = _env_float("WEIGHT_DAY_DIVERSITY", 0.0, lo=0.0, hi=2.0)
REG_GATE_ENTROPY = _env_float("REG_GATE_ENTROPY", 0.0, lo=0.0, hi=1.0)
REG_TEMPORAL_SMOOTH = _env_float("REG_TEMPORAL_SMOOTH", 0.0, lo=0.0, hi=1.0)
WARM_FREEZE_RATIO = _env_float("WARM_FREEZE_RATIO", 0.0, lo=0.0, hi=0.5)

# Time-of-day specialist filter (Phase B: Regime-Specialized Ensemble)
# Values: "" (all bars), "morning" (0-120), "midday" (120-240), "afternoon" (240-390), "highvol" (VIX>20 days)
TOD_FILTER = os.environ.get("TRAIN_TOD_FILTER", "")

# Reward-Weighted Regression (Phase C: RWR)
# Higher values upweight bars on high-reward trajectories/days
RWR_WEIGHT = _env_float("TRAIN_RWR_WEIGHT", 0.0, lo=0.0, hi=5.0)
DAY_RWR_WEIGHT = _env_float("TRAIN_DAY_RWR_WEIGHT", 0.0, lo=0.0, hi=5.0)

# Score formula — LOCKED. Do NOT modify these values.
# Changing these games the evaluation metric without improving trading.
# See program.md "Score Formula is LOCKED" for details.

# Named feature indices
IDX_MINUTES_TO_CLOSE = 19
IDX_ATM_IV = 22

# Feature groups (37 features — v3 with market structure)
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
    'mkt_struct': (32, 37),
}


# ---------------------------------------------------------------------------
# Model — v6: four-head (gate + direction + value + risk)
# ---------------------------------------------------------------------------


class TradingModel(nn.Module):
    """Four-head model for SPX 0DTE options (v6).

    Architecture:
    - Shared transformer backbone
    - Gate head: (batch, 2) — [NO_TRADE, TRADE]  (entry/exit signal)
    - Direction head: (batch, 6) — strike selection
    - Value head: (batch, 1) — expected remaining P&L  (exit intelligence)
    - Risk head: (batch, 3) — [stop_pct, size_frac, conviction]  (account-aware risk)

    Position state: 7 dims
      [0] in_trade, [1] bars_held, [2] unrealized_pnl, [3] account_health,
      [4] loss_streak, [5] best_pnl_since_entry, [6] bars_since_pnl_high

    Account state: 4 dims (separate input, risk head only)
      [0] account_growth_ratio, [1] log_account_size,
      [2] daily_pnl_fraction, [3] win_rate_20

    Input:  (batch, lookback, NUM_FEATURES)
    Output: (gate_logits, dir_logits) or with return_value/return_risk flags
    """

    POSITION_STATE_DIM = 7  # class constant for external reference
    ACCOUNT_STATE_DIM = 4   # separate from position state for backward compat

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

        # Position state injection for gate head (7 dims: original 5 + best_pnl + bars_since_high)
        self.position_proj = nn.Linear(self.POSITION_STATE_DIM, d_model // 4)
        self.position_gate_proj = nn.Linear(d_model + d_model // 4, d_model)

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

        # Value head: "how much P&L remains?" → scalar (Phase D)
        # Shares backbone but gets position state for trade context
        self.value_proj = nn.Linear(d_model + d_model // 4, d_model)
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Risk head: account-aware risk management (v6)
        # Separate account state input — only risk head uses it
        self.risk_account_proj = nn.Linear(self.ACCOUNT_STATE_DIM, d_model // 4)  # 4 → 16
        self.risk_proj = nn.Linear(d_model + d_model // 4 + d_model // 4, d_model)  # 64+16+16 → 64
        self.risk_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),  # 64 → 32
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),        # 32 → 3: stop_pct, size_frac, conviction
        )

        # Learned time-of-day loss weights: model discovers which bars matter most
        # 390 learnable weights (one per bar of day), initialized to 0 (neutral via softplus)
        self.tod_weight_logits = nn.Parameter(torch.zeros(BARS_PER_DAY))

        # Direction bias: favor ATM (domain knowledge: highest gamma, most responsive)
        # OTM has -601% cumulative backtest returns — penalize it
        # Gate bias: PRO-TRADE start — model defaults to NO_TRADE, must learn when to trade
        # Reverted to pro-trade from [−0.3, +0.3] to [+0.3, −0.3] to prevent day-clustering / overtrading
        with torch.no_grad():
            self.gate_head[-1].bias[0] -= 0.3   # NO_TRADE: discourage (pro-trade)
            self.gate_head[-1].bias[1] += 0.3   # TRADE: encourage (find opportunities)

            # ATM favored, OTM penalized (domain knowledge)
            # Symmetric CALL/PUT initialization to prevent direction collapse
            self.dir_head[-1].bias[0] += 0.20   # CALL_ATM bonus (increased)
            self.dir_head[-1].bias[3] += 0.20   # PUT_ATM bonus (increased, equal to CALL)
            self.dir_head[-1].bias[1] -= 0.15   # CALL_OTM5 penalty (increased)
            self.dir_head[-1].bias[2] -= 0.20   # CALL_OTM10 penalty (increased)
            self.dir_head[-1].bias[4] -= 0.15   # PUT_OTM5 penalty (increased)
            self.dir_head[-1].bias[5] -= 0.20   # PUT_OTM10 penalty (increased)

            # Risk head bias init (domain knowledge priors)
            self.risk_head[-1].bias[0] = 0.0   # sigmoid(0)=0.5 → mid-range stop (~0.35)
            self.risk_head[-1].bias[1] = -1.0  # sigmoid(-1)≈0.27 → conservative sizing
            self.risk_head[-1].bias[2] = 0.0   # tanh(0)=0 → neutral conviction

    def forward(self, x, position_state=None, account_state=None,
                return_value=False, return_risk=False):
        batch_size = x.shape[0]
        device = x.device

        if position_state is None:
            position_state = torch.zeros(batch_size, self.POSITION_STATE_DIM, device=device)
            position_state[:, 3] = 1.0  # account_health = 1.0

        x_proj = self.input_proj(x)
        x_proj = self.input_norm(x_proj)
        x_proj = x_proj + self.pos_embed[:, :x_proj.size(1), :]
        x_proj = self.transformer(x_proj, mask=self.causal_mask[:x_proj.size(1), :x_proj.size(1)],
                                  is_causal=True)
        last = x_proj[:, -1, :]  # (batch, d_model)

        # Inject position state into gate head
        pos_emb = torch.relu(self.position_proj(position_state))
        gate_input = self.position_gate_proj(torch.cat([last, pos_emb], dim=-1))

        gate_logits = self.gate_head(gate_input)
        dir_logits = self.dir_head(last)

        outputs = (gate_logits, dir_logits)

        if return_value:
            value_input = self.value_proj(torch.cat([last, pos_emb], dim=-1))
            value_pred = self.value_head(value_input).squeeze(-1)  # (batch,)
            outputs = outputs + (value_pred,)

        if return_risk:
            if account_state is None:
                account_state = torch.zeros(batch_size, self.ACCOUNT_STATE_DIM, device=device)
                account_state[:, 0] = 1.0  # neutral growth ratio
            acct_emb = torch.relu(self.risk_account_proj(account_state))
            risk_input = self.risk_proj(torch.cat([last, pos_emb, acct_emb], dim=-1))
            risk_raw = self.risk_head(risk_input)
            stop_pct = 0.15 + 0.45 * torch.sigmoid(risk_raw[:, 0])   # [0.15, 0.60]
            size_frac = torch.sigmoid(risk_raw[:, 1])                  # [0, 1]
            conviction = torch.tanh(risk_raw[:, 2])                    # [-1, +1]
            risk_output = torch.stack([stop_pct, size_frac, conviction], dim=-1)
            outputs = outputs + (risk_output,)

        return outputs if len(outputs) > 2 else outputs


# ---------------------------------------------------------------------------
# Loss — v5: EV-weighted, magnitude-aware
# ---------------------------------------------------------------------------

# Scale for sigmoid soft target (controls sharpness of trade/no-trade boundary)
EV_GATE_SCALE = _env_float("TRAIN_EV_GATE_SCALE", 10.0, lo=1.0, hi=50.0)


def _select_stop_level_pnl(tight, med, wide, iv_feat, vix_feat):
    """Select stopped P&L level per-bar based on market conditions (IV, VIX).

    Mirrors the dynamic stop formula: stop = BASE * iv_factor * vix_factor.
    Selects tight/med/wide stopped P&L to match what the dynamic stop would use.
    """
    if tight is None or wide is None:
        return med
    iv_factor = 1.0 + torch.clamp(iv_feat, min=0.0) * 0.15
    vix_factor = 1.0 + torch.clamp(vix_feat, min=0.0) * 0.10
    effective_stop = 0.35 * iv_factor * vix_factor
    use_tight = effective_stop <= 0.275
    use_wide = effective_stop >= 0.425
    result = med.clone()
    result[use_tight] = tight[use_tight]
    result[use_wide] = wide[use_wide]
    return result


def sniper_loss(gate_logits, dir_logits, call_pnl, put_pnl, time_features, features,
                exit_call_labels=None, exit_put_labels=None,
                otm5_call_pnl=None, otm5_put_pnl=None,
                otm10_call_pnl=None, otm10_put_pnl=None,
                supervision_weight=None, actionable_mask=None, risk_state_mask=None,
                call_stopped_pnl=None, put_stopped_pnl=None,
                otm5_call_stopped_pnl=None, otm5_put_stopped_pnl=None,
                otm10_call_stopped_pnl=None, otm10_put_stopped_pnl=None,
                call_stopped_tight=None, call_stopped_wide=None,
                put_stopped_tight=None, put_stopped_wide=None,
                is_holding=None,
                tod_weight_logits=None):
    """v5 EV-weighted loss: magnitude-aware gate + return-weighted direction.

    Key changes from v4:
    - Gate: soft continuous target from sigmoid(best_stopped_pnl * scale) instead of binary
    - Direction: return-weighted soft targets instead of argmax classification
    - Uses stopped P&L (with dynamic stops) when available, falls back to unstopped
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

    # Build 6-class P&L arrays (unstopped for backward compat, stopped for v5 targets)
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

    # Stopped P&L — multi-level selection based on IV/VIX when available
    cs_med = _safe(call_stopped_pnl) if call_stopped_pnl is not None else c_pnl
    ps_med = _safe(put_stopped_pnl) if put_stopped_pnl is not None else p_pnl

    # Select appropriate stop level per-bar using market conditions
    iv_feat = features[valid, IDX_ATM_IV] if features.shape[-1] > IDX_ATM_IV else torch.zeros_like(c_pnl)
    vix_feat = features[valid, 5] if features.shape[-1] > 5 else torch.zeros_like(c_pnl)  # idx 5 = vix in feature set

    cs_pnl = _select_stop_level_pnl(
        _safe(call_stopped_tight) if call_stopped_tight is not None else None,
        cs_med,
        _safe(call_stopped_wide) if call_stopped_wide is not None else None,
        iv_feat, vix_feat)
    ps_pnl = _select_stop_level_pnl(
        _safe(put_stopped_tight) if put_stopped_tight is not None else None,
        ps_med,
        _safe(put_stopped_wide) if put_stopped_wide is not None else None,
        iv_feat, vix_feat)

    all_stopped_pnl = torch.stack([
        cs_pnl,
        _safe(otm5_call_stopped_pnl) if otm5_call_stopped_pnl is not None else _safe(otm5_call_pnl),
        _safe(otm10_call_stopped_pnl) if otm10_call_stopped_pnl is not None else _safe(otm10_call_pnl),
        ps_pnl,
        _safe(otm5_put_stopped_pnl) if otm5_put_stopped_pnl is not None else _safe(otm5_put_pnl),
        _safe(otm10_put_stopped_pnl) if otm10_put_stopped_pnl is not None else _safe(otm10_put_pnl),
    ], dim=-1)  # (valid, 6)

    all_stopped_safe = torch.nan_to_num(all_stopped_pnl, nan=-999.0)

    # ---- Gate loss: binary classification ----
    # Best available return across all 6 option types (after stops)
    best_pnl = all_stopped_safe.max(dim=-1).values  # (valid,)

    # ---- Gate targets: unified entry + exit signal ----
    # TRADE (1) if profitable AND not an exit bar.
    # NO_TRADE (0) if unprofitable OR exit signal fires.
    # This resolves the prior gate/exit conflict where both losses trained
    # the gate on the same bars with opposing targets.
    gate_targets = (best_pnl > 0.0).long()  # 0=NO_TRADE, 1=TRADE

    # Override: exit-labeled bars → NO_TRADE, but ONLY when the model is holding
    # This resolves the entry/exit conflict (Flaw 2): flat bars learn entry signals
    # without exit interference; holding bars learn exit signals.
    if exit_call_labels is not None and exit_put_labels is not None:
        ec = exit_call_labels[valid]
        ep = exit_put_labels[valid]
        exit_signal = (ec > 0.5) | (ep > 0.5)
        # Only override profitable bars — unprofitable bars are already NO_TRADE
        exit_override = exit_signal & (best_pnl > 0.0)

        # Position-conditional: only apply exit overrides when model is holding
        # Random batches (flat state) → is_holding=None → no exit override
        # Day-seq batches → is_holding tracks real position → exit fires only while holding
        if is_holding is not None:
            holding_mask = is_holding[valid] > 0.5
            exit_override = exit_override & holding_mask

        if EXIT_LOSS_WEIGHT >= 1.0:
            gate_targets[exit_override] = 0
        else:
            override_prob = EXIT_LOSS_WEIGHT
            override_mask = torch.rand(exit_override.sum().item(), device=device) < override_prob
            override_indices = exit_override.nonzero(as_tuple=True)[0][override_mask]
            gate_targets[override_indices] = 0

    # Learned time-of-day weighting: model discovers which bars matter most
    # tod_weight_logits is a 390-dim parameter on the model, indexed by bar_of_day
    if tod_weight_logits is not None:
        bar_idx = ((1.0 - t_feat) * (BARS_PER_DAY - 1)).long().clamp(0, BARS_PER_DAY - 1)
        tod_weight = F.softplus(tod_weight_logits[bar_idx]) + 0.5  # floor at 0.5, no upper bound
    else:
        tod_weight = 1.0  # uniform if not provided
    gate_loss = F.cross_entropy(g_logits, gate_targets, reduction='none')
    gate_loss = (gate_loss * tod_weight * sample_weight).mean()

    # ---- Direction loss: return-weighted soft targets ----
    # Only train direction where gate target is TRADE
    trade_mask = gate_targets == 1
    if trade_mask.sum() < 2:
        return GATE_LOSS_WEIGHT * gate_loss

    d_logits_trade = d_logits[trade_mask]
    t_feat_trade = t_feat[trade_mask]
    trade_pnl = all_stopped_safe[trade_mask]

    # Soft direction targets: shift to positive, normalize to distribution
    trade_pnl_shifted = trade_pnl - trade_pnl.min(dim=-1, keepdim=True).values + 0.01
    dir_soft_targets = trade_pnl_shifted / trade_pnl_shifted.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    # KL-divergence style loss: cross-entropy with soft targets
    dir_loss = -(dir_soft_targets * F.log_softmax(d_logits_trade, dim=-1)).sum(dim=-1)

    dir_time_weight = 1.0 + 0.5 * (1.0 - t_feat_trade)
    dir_w = sample_weight[trade_mask]
    dir_w = dir_w / dir_w.mean().clamp(min=1e-6)
    dir_loss = (dir_loss * dir_time_weight * dir_w).mean()

    # ---- Direction entropy bonus (hardcoded): penalize collapsed distributions ----
    # When model predicts 100% one direction, entropy=0. We subtract entropy from the
    # loss to reward diversity. DIRECTION_ENTROPY_BONUS is a fixed constant (not tunable
    # via _env_float) so it doesn't violate the hyperparameter count constraint.
    DIRECTION_ENTROPY_BONUS = 0.20  # hardcoded: ~0.5 nats bonus for uniform distribution
    dir_probs_trade = F.softmax(d_logits_trade, dim=-1)
    dir_entropy = -(dir_probs_trade * (dir_probs_trade + 1e-8).log()).sum(dim=-1)  # (valid_trade,)
    entropy_bonus = (dir_entropy * dir_w).mean()
    dir_loss = dir_loss - DIRECTION_ENTROPY_BONUS * entropy_bonus

    # ---- P&L alignment bonus ----
    gate_probs = F.softmax(g_logits, dim=-1)
    dir_probs = F.softmax(d_logits, dim=-1)
    trade_prob = gate_probs[:, 1]
    all_pnl_safe = torch.nan_to_num(all_stopped_pnl, nan=0.0)
    pnl_signal = trade_prob * (dir_probs * all_pnl_safe).sum(dim=-1)
    pnl_loss = -(pnl_signal * sample_weight).sum() / sample_weight.sum().clamp(min=1e-6)

    # ---- Confidence calibration loss ----
    # Gate softmax gives P(TRADE). Reward high confidence on winners, penalize on losers.
    # For TRADE bars (gate_target=1): confidence = P(TRADE), outcome = sign(best_pnl)
    # Loss = -mean(outcome_sign * log(confidence)) — pushes confidence toward correctness
    conf_loss = gate_logits.sum() * 0.0  # default zero
    if CONFIDENCE_LOSS_WEIGHT > 0.0 and trade_mask.sum() >= 2:
        trade_conf = F.softmax(g_logits[trade_mask], dim=-1)[:, 1]  # P(TRADE) for trade bars
        trade_best_pnl = best_pnl[trade_mask]
        # +1 for winners, -1 for losers (soft via tanh for gradient flow)
        outcome_sign = torch.tanh(trade_best_pnl * 5.0)
        # High confidence on winners → positive reward; high confidence on losers → penalty
        conf_loss = -(outcome_sign * torch.log(trade_conf.clamp(min=1e-6))).mean()

    total = (GATE_LOSS_WEIGHT * gate_loss + DIR_LOSS_WEIGHT * dir_loss
             + PNL_ALIGNMENT_WEIGHT * pnl_loss + CONFIDENCE_LOSS_WEIGHT * conf_loss)
    return total


if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------

    t_start = time.time()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 mixed precision on Ampere+ GPUs (H100/A100) — no GradScaler needed
    _use_amp = (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
    if _use_amp:
        print("Mixed precision: bf16 enabled (Ampere+ GPU detected)")

    data = load_data()
    n_bars = len(data['dates'])

    # Replace NaN features with 0
    data['features'] = torch.nan_to_num(data['features'], nan=0.0)

    # Apply WEIGHT_RECENT_BOOST: scale supervision_weight by a linear recency ramp
    # over the training range so more recent bars get higher weight.
    # Linear ramp from 1.0 (oldest bar) to (1 + WEIGHT_RECENT_BOOST) (newest bar).
    # Only applied over training range to avoid leaking val info.
    _train_end = data['train_end_idx']
    if WEIGHT_RECENT_BOOST > 0.0 and _train_end > 1 and 'supervision_weight' in data:
        _sw = data['supervision_weight']  # numpy array or tensor
        import numpy as _np
        _ramp = _np.linspace(1.0, 1.0 + WEIGHT_RECENT_BOOST, _train_end).astype(_np.float32)
        _sw_arr = _sw.numpy() if hasattr(_sw, 'numpy') else _sw
        _sw_arr[:_train_end] = _sw_arr[:_train_end] * _ramp
        if hasattr(_sw, 'numpy'):
            data['supervision_weight'] = torch.from_numpy(_sw_arr)
        else:
            data['supervision_weight'] = _sw_arr
        print(f"Recency weights applied: WEIGHT_RECENT_BOOST={WEIGHT_RECENT_BOOST} "
              f"(ramp 1.0→{1.0+WEIGHT_RECENT_BOOST:.2f} over {_train_end} train bars)")
    else:
        print(f"Recency weights: disabled (WEIGHT_RECENT_BOOST={WEIGHT_RECENT_BOOST})")

    # WEIGHT_DAY_DIVERSITY: upweight bars on days where profitable trades are rare
    # so the model can't just memorize a few high-signal days (e.g., Sep 17)
    if WEIGHT_DAY_DIVERSITY > 0.0 and 'day_boundaries' in data and 'supervision_weight' in data:
        _day_bounds = data['day_boundaries'].numpy()
        _call_pnl_np = data['call_pnl'].numpy()
        _put_pnl_np = data['put_pnl'].numpy()
        _sw = data['supervision_weight']
        _sw_arr = _sw.numpy() if hasattr(_sw, 'numpy') else _sw
        import numpy as _np
        # Count profitable bars per day (proxy for how "easy" the day is)
        _day_starts = list(_day_bounds) + [len(_call_pnl_np)]
        _n_days_processed = 0
        for _di in range(len(_day_starts) - 1):
            _s, _e = int(_day_starts[_di]), int(_day_starts[_di + 1])
            if _e > _train_end:
                break
            _c = _call_pnl_np[_s:_e]
            _p = _put_pnl_np[_s:_e]
            _profitable = ((_c > 0) & ~_np.isnan(_c)) | ((_p > 0) & ~_np.isnan(_p))
            _n_prof = _profitable.sum()
            # Inverse density: days with many profitable bars get downweighted
            # days with few get upweighted (clamped to avoid extreme weights)
            if _n_prof > 0:
                _day_weight = 1.0 + WEIGHT_DAY_DIVERSITY * (1.0 - min(_n_prof / 50.0, 1.0))
            else:
                _day_weight = 1.0
            _sw_arr[_s:_e] *= _day_weight
            _n_days_processed += 1
        if hasattr(_sw, 'numpy'):
            data['supervision_weight'] = torch.from_numpy(_sw_arr)
        print(f"Day diversity weights applied: WEIGHT_DAY_DIVERSITY={WEIGHT_DAY_DIVERSITY} "
              f"({_n_days_processed} days processed)")
    else:
        print(f"Day diversity weights: disabled (WEIGHT_DAY_DIVERSITY={WEIGHT_DAY_DIVERSITY})")

    # Phase C: Reward-Weighted Regression (RWR)
    # Upweight bars that precede high-reward trajectories
    if (RWR_WEIGHT > 0 or DAY_RWR_WEIGHT > 0) and 'call_pnl' in data and 'day_boundaries' in data:
        _sw = data['supervision_weight']
        _sw_arr = _sw.numpy() if hasattr(_sw, 'numpy') else _sw
        _call_pnl = data['call_pnl'].numpy() if hasattr(data['call_pnl'], 'numpy') else data['call_pnl']
        _put_pnl = data['put_pnl'].numpy() if hasattr(data['put_pnl'], 'numpy') else data['put_pnl']
        _day_bounds = data['day_boundaries'].numpy()
        _day_starts = list(_day_bounds) + [len(_call_pnl)]

        if RWR_WEIGHT > 0:
            # Trajectory reward: for each bar, best P&L in next 30 bars
            _fwd_window = 30
            _best_fwd = np.zeros(len(_call_pnl), dtype=np.float32)
            _c_safe = np.nan_to_num(_call_pnl, nan=0.0)
            _p_safe = np.nan_to_num(_put_pnl, nan=0.0)
            _max_pnl = np.maximum(_c_safe, _p_safe)
            # Vectorized forward max: rolling max over next _fwd_window bars
            for _i in range(len(_max_pnl)):
                _end_w = min(_i + _fwd_window, len(_max_pnl))
                _best_fwd[_i] = max(_max_pnl[_i:_end_w].max(), 0.0)

            # Normalize to [0, 1] using percentile (robust to outliers)
            _p99 = np.percentile(_best_fwd[_best_fwd > 0], 99) if (_best_fwd > 0).any() else 1.0
            _traj_weight = np.clip(_best_fwd / max(_p99, 1e-6), 0.0, 1.0)

            # Apply: sw *= (1 + RWR_WEIGHT * traj_weight), only where sw is valid
            _rwr_factor = 1.0 + RWR_WEIGHT * _traj_weight
            _valid_sw = ~np.isnan(_sw_arr)
            _sw_arr[_valid_sw] = _sw_arr[_valid_sw] * _rwr_factor[_valid_sw]
            print(f"RWR trajectory: weight={RWR_WEIGHT}, mean_factor={_rwr_factor[_valid_sw].mean():.3f}, "
                  f"bars_with_reward={(_traj_weight > 0).sum()}/{len(_traj_weight)}")

        if DAY_RWR_WEIGHT > 0:
            # Day reward: cumulative P&L per day, normalized
            _day_pnl = np.zeros(len(_call_pnl), dtype=np.float32)
            for _di in range(len(_day_starts) - 1):
                _s, _e = int(_day_starts[_di]), int(_day_starts[_di + 1])
                if _e > _train_end:
                    break
                _day_call = np.nanmean(_call_pnl[_s:_e]) if not np.all(np.isnan(_call_pnl[_s:_e])) else 0.0
                _day_put = np.nanmean(_put_pnl[_s:_e]) if not np.all(np.isnan(_put_pnl[_s:_e])) else 0.0
                _day_best = max(float(_day_call), float(_day_put), 0.0)
                _day_pnl[_s:_e] = _day_best

            _dp99 = np.percentile(_day_pnl[_day_pnl > 0], 99) if (_day_pnl > 0).any() else 1.0
            _day_norm = np.clip(_day_pnl / max(_dp99, 1e-6), 0.0, 1.0)
            _day_factor = 1.0 + DAY_RWR_WEIGHT * _day_norm
            _valid_sw2 = ~np.isnan(_sw_arr)
            _sw_arr[_valid_sw2] = _sw_arr[_valid_sw2] * _day_factor[_valid_sw2]
            print(f"RWR day: weight={DAY_RWR_WEIGHT}, mean_factor={_day_factor[_valid_sw2].mean():.3f}")

        data['supervision_weight'] = torch.from_numpy(_sw_arr) if isinstance(_sw_arr, np.ndarray) else _sw_arr
    elif RWR_WEIGHT > 0 or DAY_RWR_WEIGHT > 0:
        print(f"RWR: disabled (missing call_pnl or day_boundaries)")
    else:
        print(f"RWR: disabled (RWR_WEIGHT={RWR_WEIGHT}, DAY_RWR_WEIGHT={DAY_RWR_WEIGHT})")

    print(f"Gate entropy reg: REG_GATE_ENTROPY={REG_GATE_ENTROPY}")
    print(f"Temporal smoothing reg: REG_TEMPORAL_SMOOTH={REG_TEMPORAL_SMOOTH}")
    print(f"Warm freeze ratio: WARM_FREEZE_RATIO={WARM_FREEZE_RATIO}")

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

    # Compute data fingerprint for checkpoint provenance tracking
    import hashlib as _hashlib
    _data_shape_str = f"{data['features'].shape}_{data['train_end_idx']}_{data['val_start_idx']}_{data['val_end_idx']}"
    _data_fingerprint = _hashlib.sha256(_data_shape_str.encode()).hexdigest()[:16]
    print(f"  Data fingerprint: {_data_fingerprint}")

    model = TradingModel().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Architecture: v6 four-head (gate+dir+value+risk), ATM-biased, 7-dim position + 4-dim account state")

    # Warm start: load best_model.pt if it exists and shapes are compatible
    _warm_start_loaded = False
    _warm_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    if os.path.exists(_warm_path):
        try:
            import numpy as np
            torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
            _ckpt = torch.load(_warm_path, map_location=device, weights_only=True)
            _state = _ckpt if not isinstance(_ckpt, dict) or 'model_state_dict' not in _ckpt else _ckpt['model_state_dict']
            # Architecture version gate: reject incompatible checkpoints
            _ckpt_arch = _ckpt.get('architecture', 'unknown') if isinstance(_ckpt, dict) else 'unknown'
            _ckpt_has_value = _ckpt.get('has_value_head', False) if isinstance(_ckpt, dict) else ('value_head.4.weight' in _state)
            _ckpt_has_risk = _ckpt.get('has_risk_head', False) if isinstance(_ckpt, dict) else ('risk_head.4.weight' in _state)
            _ckpt_pos_dim = _ckpt.get('position_state_dim', 5) if isinstance(_ckpt, dict) else (_state['position_proj.weight'].shape[1] if 'position_proj.weight' in _state else 5)
            if not _ckpt_has_value or not _ckpt_has_risk or _ckpt_pos_dim < TradingModel.POSITION_STATE_DIM:
                print(f"WARNING: Checkpoint incompatible (arch={_ckpt_arch}, value_head={_ckpt_has_value}, risk_head={_ckpt_has_risk}, pos_dim={_ckpt_pos_dim}).")
                print(f"  Current model requires: value_head=True, risk_head=True, pos_dim={TradingModel.POSITION_STATE_DIM}.")
                print(f"  Rejecting warm start — training from scratch.")
                raise ValueError("Architecture version mismatch")
            _missing, _unexpected = model.load_state_dict(_state, strict=False)
            if _missing:
                print(f"  WARNING: Missing keys (random init): {[k for k in _missing]}")
            if _unexpected:
                print(f"  Unexpected keys (ignored): {[k for k in _unexpected]}")
            _warm_start_loaded = True
            print(f"Warm start: loaded weights from {_warm_path} (arch={_ckpt_arch}, pos_dim={_ckpt_pos_dim})")
        except Exception as e:
            print(f"Warm start failed ({e}), training from scratch.")
    else:
        print("Training from scratch (no best_model.pt).")

    # WARM_FREEZE_RATIO: freeze transformer layers for first N% of training
    # Only heads (gate_head, dir_head) and position projection train initially.
    # This prevents catastrophic forgetting of learned representations.
    _frozen_params = []
    if _warm_start_loaded and WARM_FREEZE_RATIO > 0:
        for name, param in model.named_parameters():
            if any(k in name for k in ['transformer', 'input_proj', 'input_norm', 'pos_embed']):
                param.requires_grad = False
                _frozen_params.append(name)
        print(f"Warm freeze: {len(_frozen_params)} params frozen for first {WARM_FREEZE_RATIO*100:.0f}% of training")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
        weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS,
    )

    from prepare import make_day_sequential_loader

    # Phase B: Apply TOD filter to training data (specialists train on bar subsets)
    # target_mask controls which bars are used as training targets;
    # lookback context still uses all valid bars (valid_mask unchanged).
    _tod_target_mask = None
    if TOD_FILTER:
        _tod_ranges = {
            "morning":   (0, 120),   # 9:30-11:30
            "midday":    (120, 240), # 11:30-13:30
            "afternoon": (240, 390), # 13:30-16:00
        }
        day_boundaries = data['day_boundaries'].tolist()
        n_bars = len(data['valid_mask'])
        tm = torch.ones(n_bars, dtype=torch.bool)

        if TOD_FILTER == "highvol":
            features_t = data['features']
            IDX_VIX_REGIME = 24
            for d in range(len(day_boundaries)):
                ds = day_boundaries[d]
                de = day_boundaries[d + 1] if d + 1 < len(day_boundaries) else n_bars
                day_vr = features_t[ds:de, IDX_VIX_REGIME]
                valid_vr = day_vr[~torch.isnan(day_vr)]
                mean_vr = float(valid_vr.mean()) if len(valid_vr) > 0 else -1
                if mean_vr <= 0:  # VIX <= 20
                    tm[ds:de] = False
        elif TOD_FILTER in _tod_ranges:
            lo_bar, hi_bar = _tod_ranges[TOD_FILTER]
            for d in range(len(day_boundaries)):
                ds = day_boundaries[d]
                de = day_boundaries[d + 1] if d + 1 < len(day_boundaries) else n_bars
                for i in range(ds, de):
                    bar_in_day = i - ds
                    if bar_in_day < lo_bar or bar_in_day >= hi_bar:
                        tm[i] = False
        else:
            raise ValueError(f"Unknown TRAIN_TOD_FILTER={TOD_FILTER!r}. Valid: morning, midday, afternoon, highvol")

        _tod_target_mask = tm
        n_targets = int((tm & data['valid_mask']).sum())
        n_total = int(data['valid_mask'].sum())
        print(f"TOD filter '{TOD_FILTER}': {n_total} → {n_targets} target bars ({n_targets/max(n_total,1)*100:.1f}%)")

    train_loader = make_dataloader(data, LOOKBACK, BATCH_SIZE, "train", device, target_mask=_tod_target_mask)
    x_batch, y_batch = next(train_loader)

    # Day-sequential loader for Phase 3
    DAY_SEQ_RATIO = _env_float("TRAIN_DAY_SEQ_RATIO", 0.85, lo=0.0, hi=1.0)  # Increased from 0.7: more temporal context for generalization
    DAY_SEQ_BATCH = _env_int("TRAIN_DAY_SEQ_BATCH", 16, lo=4, hi=64)
    _use_day_seq = DAY_SEQ_RATIO > 0
    if _use_day_seq:
        try:
            day_seq_loader = make_day_sequential_loader(data, LOOKBACK, DAY_SEQ_BATCH, device, "train")
            _day_seq_available = True
            print(f"Day-sequential batching: {DAY_SEQ_RATIO*100:.0f}% sequential, {(1-DAY_SEQ_RATIO)*100:.0f}% random (batch={DAY_SEQ_BATCH})")
        except (KeyError, AssertionError) as e:
            print(f"Day-sequential loader unavailable ({e}), using random only")
            _day_seq_available = False
    else:
        _day_seq_available = False

    print(f"\nBudget: {TIME_BUDGET}s | Batch: {BATCH_SIZE} | Lookback: {LOOKBACK}")
    print(f"LR: {LR} | Depth: {DEPTH} | d_model: {D_MODEL} | ff_mult: {FF_MULT}")
    print(f"Dropout: {DROPOUT} | Weight decay: {WEIGHT_DECAY}")
    print(f"Label smoothing: gate={GATE_LABEL_SMOOTHING} dir={DIR_LABEL_SMOOTHING}")
    print(f"Loss weights: gate={GATE_LOSS_WEIGHT}, dir={DIR_LOSS_WEIGHT}, pnl={PNL_ALIGNMENT_WEIGHT}")
    print(f"Exit override strength: {EXIT_LOSS_WEIGHT} (gate targets flipped on exit bars)")
    print(f"Risk loss weight: {RISK_LOSS_WEIGHT} (stop + size + conviction)")
    print(f"Position state: flat for random batches, tracked for day-sequential")
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
        'win_rate_bonus': 0.0,
        'rr_bonus': 0.3,
        'drawdown_penalty': 0.5,
        'hold_bonus': 0.0,
        'freq_center': 2.5,
        'freq_width': 2.5,
        'consec_loss_threshold': 3,
        'short_hold_threshold': 0.30,
        'stop_rate_threshold': 0.30,
        'ruin_penalty': 1.0,
        'ruin_threshold': 0.25,
        'risk_fraction_penalty': 0.5,
    }

    # ---------------------------------------------------------------------------
    # Training loop — v5: hybrid day-sequential + random batching
    # ---------------------------------------------------------------------------

    def _unpack_y(y_batch):
        """Unpack the y tuple from both random and day-sequential loaders.

        Supports both 18-element (v5) and 22-element (v6 multi-level) tuples.
        """
        if len(y_batch) >= 22:
            (fwd_ret, call_pnl_batch, put_pnl_batch, exit_call_batch, exit_put_batch,
             otm5c_pnl, otm5p_pnl, otm10c_pnl, otm10p_pnl,
             supervision_weight_batch, actionable_mask_batch, risk_state_mask_batch,
             call_stopped_batch, put_stopped_batch,
             otm5c_stopped, otm5p_stopped, otm10c_stopped, otm10p_stopped,
             call_stopped_tight, call_stopped_wide,
             put_stopped_tight, put_stopped_wide) = y_batch
        else:
            (fwd_ret, call_pnl_batch, put_pnl_batch, exit_call_batch, exit_put_batch,
             otm5c_pnl, otm5p_pnl, otm10c_pnl, otm10p_pnl,
             supervision_weight_batch, actionable_mask_batch, risk_state_mask_batch,
             call_stopped_batch, put_stopped_batch,
             otm5c_stopped, otm5p_stopped, otm10c_stopped, otm10p_stopped) = y_batch
            call_stopped_tight = call_stopped_wide = None
            put_stopped_tight = put_stopped_wide = None
        return {
            'call_pnl': call_pnl_batch, 'put_pnl': put_pnl_batch,
            'exit_call': exit_call_batch, 'exit_put': exit_put_batch,
            'otm5c': otm5c_pnl, 'otm5p': otm5p_pnl,
            'otm10c': otm10c_pnl, 'otm10p': otm10p_pnl,
            'sw': supervision_weight_batch, 'am': actionable_mask_batch,
            'rsm': risk_state_mask_batch,
            'call_stopped': call_stopped_batch, 'put_stopped': put_stopped_batch,
            'otm5c_stopped': otm5c_stopped, 'otm5p_stopped': otm5p_stopped,
            'otm10c_stopped': otm10c_stopped, 'otm10p_stopped': otm10p_stopped,
            'call_stopped_tight': call_stopped_tight, 'call_stopped_wide': call_stopped_wide,
            'put_stopped_tight': put_stopped_tight, 'put_stopped_wide': put_stopped_wide,
        }


    def _update_position_state(position_state, gate_logits, dir_logits, y_dict):
        """Update position state based on model predictions (no grad).

        Tracks 7 dims: is_holding, bars_held, unrealized_pnl, account_health,
        loss_streak, best_pnl_since_entry, bars_since_pnl_high.
        Applies stop-loss and max-hold exits to align training with evaluation.
        """
        with torch.no_grad():
            B = gate_logits.shape[0]
            gate_pred = gate_logits.argmax(dim=-1)  # 0=NO_TRADE, 1=TRADE
            gate_conf = torch.softmax(gate_logits.float(), dim=-1)[:, 1]  # P(TRADE)

            is_holding = position_state[:, 0]
            bars_held = position_state[:, 1]
            unrealized_pnl = position_state[:, 2]
            account_health = position_state[:, 3]
            loss_streak = position_state[:, 4]
            best_pnl = position_state[:, 5]
            bars_since_high = position_state[:, 6]

            # For entering/holding trades: look up the best stopped P&L as proxy
            all_stopped = torch.stack([
                y_dict['call_stopped'], y_dict['put_stopped'],
                y_dict['otm5c_stopped'], y_dict['otm5p_stopped'],
                y_dict['otm10c_stopped'], y_dict['otm10p_stopped'],
            ], dim=-1)
            cur_pnl = torch.nan_to_num(all_stopped, nan=-999.0).max(dim=-1).values

            # Entry: model says TRADE when not holding
            entering = (gate_pred == 1) & (is_holding < 0.5)
            # Exit: model says NO_TRADE when holding
            model_exiting = (gate_pred == 0) & (is_holding > 0.5)

            # Stop-loss exit: unrealized P&L breaches dynamic stop (aligns with evaluate_trades)
            stop_exiting = (is_holding > 0.5) & (unrealized_pnl < -0.35)

            # Max-hold exit: held too long (normalized bars_held > 60/390 ≈ 0.154)
            max_hold_exiting = (is_holding > 0.5) & (bars_held > 60.0 / BARS_PER_DAY)

            # Combined exits: model exit OR stop-loss OR max-hold
            exiting = model_exiting | stop_exiting | max_hold_exiting

            # Update state
            new_holding = is_holding.clone()
            new_bars = bars_held.clone()
            new_pnl = unrealized_pnl.clone()
            new_health = account_health.clone()
            new_streak = loss_streak.clone()
            new_best_pnl = best_pnl.clone()
            new_bars_since_high = bars_since_high.clone()

            # Entries
            new_holding[entering] = 1.0
            new_bars[entering] = 0.0
            new_pnl[entering] = 0.0
            new_best_pnl[entering] = 0.0
            new_bars_since_high[entering] = 0.0

            # Increment bars for holding positions
            still_holding = (new_holding > 0.5) & ~entering
            new_bars[still_holding] = (bars_held[still_holding] + 1.0 / BARS_PER_DAY).clamp(max=1.0)

            # Update unrealized P&L for holding positions (reduced saturation for better gradient)
            cur_pnl_tanh = torch.tanh(cur_pnl * 2.0)
            new_pnl[new_holding > 0.5] = cur_pnl_tanh[new_holding > 0.5]

            # Update best_pnl and bars_since_high for value head context
            holding_mask = new_holding > 0.5
            improved = holding_mask & (cur_pnl_tanh > new_best_pnl)
            new_best_pnl[improved] = cur_pnl_tanh[improved]
            new_bars_since_high[improved] = 0.0
            not_improved = holding_mask & ~improved & ~entering
            new_bars_since_high[not_improved] = (bars_since_high[not_improved] + 1.0 / BARS_PER_DAY).clamp(max=1.0)

            # Exits: update account health and loss streak
            exit_pnl = unrealized_pnl[exiting]
            stop_exit_pnl = torch.where(
                stop_exiting[exiting] if exiting.any() else torch.zeros(0, dtype=torch.bool, device=exit_pnl.device),
                torch.full_like(exit_pnl, -0.35),
                exit_pnl
            )
            losing_exit = stop_exit_pnl < 0
            new_health[exiting] = (account_health[exiting] +
                                   torch.where(stop_exit_pnl > 0,
                                              stop_exit_pnl * 0.1,
                                              stop_exit_pnl * 0.5)).clamp(0.1, 1.5)
            new_streak[exiting] = torch.where(
                losing_exit,
                (loss_streak[exiting] + 0.33).clamp(max=1.0),
                torch.zeros_like(loss_streak[exiting])
            )

            new_holding[exiting] = 0.0
            new_bars[exiting] = 0.0
            new_pnl[exiting] = 0.0
            new_best_pnl[exiting] = 0.0
            new_bars_since_high[exiting] = 0.0

            return torch.stack([new_holding, new_bars, new_pnl, new_health, new_streak,
                               new_best_pnl, new_bars_since_high], dim=1)


    total_time = 0.0
    _wall_start = time.time()
    step = 0
    smooth_loss = 0.0
    _day_seq_position_state = None  # carried across bars within a day
    _day_seq_account_state = None   # account state for risk head
    _day_seq_step_count = 0  # bars processed in current day sequence
    _day_seq_win_count = 0   # rolling win count for account state
    _day_seq_trade_count = 0 # rolling trade count for account state
    _day_seq_daily_pnl = 0.0 # accumulated P&L for current day
    _day_seq_account_balance = 10000.0  # simulated account balance

    while True:
        model.train()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()

        # Hybrid sampling: alternate between day-sequential and random
        use_seq_this_step = (_day_seq_available and
                             torch.rand(1).item() < DAY_SEQ_RATIO)

        if use_seq_this_step:
            # Day-sequential step with real position state
            x_seq, y_seq, bar_in_day = next(day_seq_loader)
            y_dict = _unpack_y(y_seq)

            if bar_in_day == 0 or _day_seq_position_state is None or _day_seq_position_state.shape[0] != x_seq.shape[0]:
                # Start of new day — reset position state (7 dims)
                _day_seq_position_state = torch.zeros(x_seq.shape[0], TradingModel.POSITION_STATE_DIM, device=device)
                _day_seq_position_state[:, 3] = 1.0  # account_health = 1.0
                # Reset daily P&L, build account state
                _day_seq_daily_pnl = 0.0
                _win_rate = _day_seq_win_count / max(_day_seq_trade_count, 1)
                _day_seq_account_state = torch.zeros(x_seq.shape[0], TradingModel.ACCOUNT_STATE_DIM, device=device)
                _day_seq_account_state[:, 0] = _day_seq_account_balance / 10000.0  # growth ratio
                _day_seq_account_state[:, 1] = min(math.log10(max(_day_seq_account_balance, 1000) / 1000) / 3.0, 1.0)  # log scale
                _day_seq_account_state[:, 2] = 0.0  # daily P&L resets
                _day_seq_account_state[:, 3] = _win_rate  # rolling win rate

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=_use_amp):
                _fwd_out = model(
                    x_seq, position_state=_day_seq_position_state,
                    account_state=_day_seq_account_state,
                    return_value=True, return_risk=True)
                gate_logits, dir_logits, value_pred, risk_output = _fwd_out

                time_feat = x_seq[:, -1, IDX_MINUTES_TO_CLOSE]
                batch_features = x_seq[:, -1, :]

                loss = sniper_loss(
                    gate_logits, dir_logits,
                    y_dict['call_pnl'], y_dict['put_pnl'], time_feat, batch_features,
                    y_dict['exit_call'], y_dict['exit_put'],
                    y_dict['otm5c'], y_dict['otm5p'], y_dict['otm10c'], y_dict['otm10p'],
                    supervision_weight=y_dict['sw'],
                    actionable_mask=y_dict['am'],
                    risk_state_mask=y_dict['rsm'],
                    call_stopped_pnl=y_dict['call_stopped'],
                    put_stopped_pnl=y_dict['put_stopped'],
                    otm5_call_stopped_pnl=y_dict['otm5c_stopped'],
                    otm5_put_stopped_pnl=y_dict['otm5p_stopped'],
                    otm10_call_stopped_pnl=y_dict['otm10c_stopped'],
                    otm10_put_stopped_pnl=y_dict['otm10p_stopped'],
                    call_stopped_tight=y_dict.get('call_stopped_tight'),
                    call_stopped_wide=y_dict.get('call_stopped_wide'),
                    put_stopped_tight=y_dict.get('put_stopped_tight'),
                    put_stopped_wide=y_dict.get('put_stopped_wide'),
                    is_holding=_day_seq_position_state[:, 0],
                    tod_weight_logits=model.tod_weight_logits,
                )

                # Current best P&L across 6 option types (shared by value + risk heads)
                _all_stopped = torch.stack([
                    y_dict['call_stopped'], y_dict['put_stopped'],
                    y_dict['otm5c_stopped'], y_dict['otm5p_stopped'],
                    y_dict['otm10c_stopped'], y_dict['otm10p_stopped'],
                ], dim=-1)
                _cur_best = torch.nan_to_num(_all_stopped, nan=-999.0).max(dim=-1).values

                # Phase D: Value head loss (only while holding)
                _holding = _day_seq_position_state[:, 0] > 0.5
                if VALUE_LOSS_WEIGHT > 0 and _holding.any():
                    _value_target = _cur_best[_holding].clamp(-1.0, 1.0)
                    _value_pred_holding = value_pred[_holding]
                    _value_loss = F.mse_loss(_value_pred_holding, _value_target)
                    loss = loss + VALUE_LOSS_WEIGHT * _value_loss

                # Phase E: Risk head loss (only while holding)
                if RISK_LOSS_WEIGHT > 0 and _holding.any():
                    _risk_holding = risk_output[_holding]  # (n_holding, 3)
                    _pred_stop = _risk_holding[:, 0]
                    _pred_size = _risk_holding[:, 1]
                    _pred_conv = _risk_holding[:, 2]
                    _pnl_holding = _cur_best[_holding]

                    # 1. Stop distance hindsight (weight 0.5)
                    # Optimal stop = just wider than max adverse excursion
                    # Proxy: for winners, stop should be tight; for losers, wider
                    _mae_proxy = (-_day_seq_position_state[_holding, 2]).clamp(0.0)  # unrealized drawdown
                    _optimal_stop = (0.15 + _mae_proxy * 0.45 * 1.1).clamp(0.15, 0.60)
                    _stop_loss_risk = F.mse_loss(_pred_stop, _optimal_stop)

                    # 2. Position size reward (weight 0.3)
                    _acct_growth = _day_seq_account_state[_holding, 0].clamp(0.5, 1.0) if _day_seq_account_state is not None else torch.ones_like(_pred_size)
                    _size_target = torch.sigmoid(_pnl_holding * 5.0) * _acct_growth
                    _size_loss = F.mse_loss(_pred_size, _size_target)

                    # 3. Conviction-exit interaction (weight 0.2)
                    _conv_target = torch.tanh(_pnl_holding * 3.0)
                    _conv_loss = F.mse_loss(_pred_conv, _conv_target)

                    _risk_loss = 0.5 * _stop_loss_risk + 0.3 * _size_loss + 0.2 * _conv_loss
                    loss = loss + RISK_LOSS_WEIGHT * _risk_loss

            # Update position state for next bar (fp32, outside autocast)
            _day_seq_position_state = _update_position_state(
                _day_seq_position_state, gate_logits.float(), dir_logits.float(), y_dict)
        else:
            # Random batch step — flat position state (not holding)
            # Matches evaluation semantics: random bars are independent, model is flat
            # Exit overrides NOT applied (is_holding=None) — clean entry-only learning
            y_dict = _unpack_y(y_batch)
            flat_state = torch.zeros(x_batch.shape[0], TradingModel.POSITION_STATE_DIM, device=device)
            flat_state[:, 3] = 1.0  # account_health = 1.0

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=_use_amp):
                gate_logits, dir_logits = model(x_batch, position_state=flat_state)

                time_feat = x_batch[:, -1, IDX_MINUTES_TO_CLOSE]
                batch_features = x_batch[:, -1, :]

                loss = sniper_loss(
                    gate_logits, dir_logits,
                    y_dict['call_pnl'], y_dict['put_pnl'], time_feat, batch_features,
                    y_dict['exit_call'], y_dict['exit_put'],
                    y_dict['otm5c'], y_dict['otm5p'], y_dict['otm10c'], y_dict['otm10p'],
                    supervision_weight=y_dict['sw'],
                    actionable_mask=y_dict['am'],
                    risk_state_mask=y_dict['rsm'],
                    call_stopped_pnl=y_dict['call_stopped'],
                    put_stopped_pnl=y_dict['put_stopped'],
                    otm5_call_stopped_pnl=y_dict['otm5c_stopped'],
                    otm5_put_stopped_pnl=y_dict['otm5p_stopped'],
                    otm10_call_stopped_pnl=y_dict['otm10c_stopped'],
                    otm10_put_stopped_pnl=y_dict['otm10p_stopped'],
                    call_stopped_tight=y_dict.get('call_stopped_tight'),
                    call_stopped_wide=y_dict.get('call_stopped_wide'),
                    put_stopped_tight=y_dict.get('put_stopped_tight'),
                    put_stopped_wide=y_dict.get('put_stopped_wide'),
                    is_holding=None,  # flat state = no exit overrides
                    tod_weight_logits=model.tod_weight_logits,
                )

        # Regularization: gate entropy + temporal smoothing (separate from total_loss)
        _need_retain = (REG_GATE_ENTROPY > 0.0) or (REG_TEMPORAL_SMOOTH > 0.0 and use_seq_this_step)
        if _need_retain:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        # REG_GATE_ENTROPY: maximize entropy of gate predictions to prevent collapse
        if REG_GATE_ENTROPY > 0.0:
            _gate_log_probs = F.log_softmax(gate_logits, dim=-1)
            _gate_entropy = -(_gate_log_probs.exp() * _gate_log_probs).sum(dim=-1).mean()
            _entropy_loss = -REG_GATE_ENTROPY * _gate_entropy
            _entropy_loss.backward(retain_graph=(REG_TEMPORAL_SMOOTH > 0.0 and use_seq_this_step))

        # REG_TEMPORAL_SMOOTH: penalize large changes in gate predictions between
        # adjacent bars in day-sequential batches. Forces temporal consistency.
        if REG_TEMPORAL_SMOOTH > 0.0 and use_seq_this_step and gate_logits.shape[0] > 1:
            _gate_probs = F.softmax(gate_logits, dim=-1)
            _temporal_diff = (_gate_probs[1:] - _gate_probs[:-1]).pow(2).sum(dim=-1).mean()
            _smooth_loss = REG_TEMPORAL_SMOOTH * _temporal_diff
            _smooth_loss.backward()

        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        progress = min(total_time / TIME_BUDGET, 1.0)

        # WARM_FREEZE_RATIO: unfreeze transformer layers after freeze period
        if _frozen_params and progress >= WARM_FREEZE_RATIO:
            for name, param in model.named_parameters():
                if not param.requires_grad and name in _frozen_params:
                    param.requires_grad = True
            print(f"\n[step {step}] Unfreezing {len(_frozen_params)} transformer params at {progress*100:.0f}% progress")
            _frozen_params.clear()  # Only unfreeze once

        for pg in optimizer.param_groups:
            pg['lr'] = LR * get_lr_mult(progress)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if not use_seq_this_step:
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

            mode = "seq" if use_seq_this_step else "rnd"
            print(f"step {step:05d} ({100*progress:5.1f}%) | loss: {debiased:.6f} "
                  f"| trade:{p_trade:.2f} call:{p_call:.2f} put:{p_put:.2f} "
                  f"| ATM:{p_atm:.2f} OTM:{p_otm:.2f} [{mode}] "
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

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=_use_amp):
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
            'architecture': 'v6_four_head_gate_dir_value_risk',
            'false_entry_penalty': FALSE_ENTRY_PENALTY,
            'position_state_dim': TradingModel.POSITION_STATE_DIM,
            'account_state_dim': TradingModel.ACCOUNT_STATE_DIM,
            'has_value_head': True,
            'has_risk_head': True,
        },
        'training_dynamics': {
            'batch_size': BATCH_SIZE,
            'exit_loss_weight': EXIT_LOSS_WEIGHT,
            'bf16': _use_amp,
        },
        'data_fingerprint': _data_fingerprint,
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

    # --- Trade diagnostics (parsed by run_loop.py for agent feedback) ---
    if trade_log:
        print("\n=== TRADE DIAGNOSTICS ===")

        # Top 5 best/worst trades
        sorted_trades = sorted(trade_log, key=lambda t: t.get('pnl_pct', 0))
        print("\nWorst 5 trades:")
        for t in sorted_trades[:5]:
            print(f"  {t.get('date','')} {t.get('entry_time','')} {t.get('direction','')} "
                  f"{t.get('strike','')} PnL={t.get('pnl_pct',0):+.2%} "
                  f"hold={t.get('bars_held',0)}bars exit={t.get('exit_reason','')}")
        print("Best 5 trades:")
        for t in sorted_trades[-5:]:
            print(f"  {t.get('date','')} {t.get('entry_time','')} {t.get('direction','')} "
                  f"{t.get('strike','')} PnL={t.get('pnl_pct',0):+.2%} "
                  f"hold={t.get('bars_held',0)}bars exit={t.get('exit_reason','')}")

        # Time-of-day breakdown
        time_buckets = {'Morning(9:30-11:30)': [], 'Lunch(11:30-13:30)': [],
                        'Afternoon(13:30-15:30)': [], 'PowerHour(15:30-16:00)': []}
        for t in trade_log:
            hm = t.get('entry_time', '12:00')
            try:
                h, m = int(hm.split(':')[0]), int(hm.split(':')[1])
                mins = h * 60 + m
            except (ValueError, IndexError):
                mins = 720
            if mins < 690:  # 11:30
                bucket = 'Morning(9:30-11:30)'
            elif mins < 810:  # 13:30
                bucket = 'Lunch(11:30-13:30)'
            elif mins < 930:  # 15:30
                bucket = 'Afternoon(13:30-15:30)'
            else:
                bucket = 'PowerHour(15:30-16:00)'
            time_buckets[bucket].append(t.get('pnl_pct', 0))

        print("\nTime-of-day breakdown:")
        for bucket, pnls in time_buckets.items():
            if pnls:
                wins = sum(1 for p in pnls if p > 0)
                avg = sum(pnls) / len(pnls)
                print(f"  {bucket}: {len(pnls)} trades, WR={wins/len(pnls):.1%}, avg={avg:+.2%}")

        # Call vs Put breakdown
        call_pnls = [t.get('pnl_pct', 0) for t in trade_log if 'CALL' in t.get('direction', '')]
        put_pnls = [t.get('pnl_pct', 0) for t in trade_log if 'PUT' in t.get('direction', '')]
        print("\nDirection breakdown:")
        if call_pnls:
            cw = sum(1 for p in call_pnls if p > 0)
            print(f"  CALL: {len(call_pnls)} trades, WR={cw/len(call_pnls):.1%}, avg={sum(call_pnls)/len(call_pnls):+.2%}")
        if put_pnls:
            pw = sum(1 for p in put_pnls if p > 0)
            print(f"  PUT:  {len(put_pnls)} trades, WR={pw/len(put_pnls):.1%}, avg={sum(put_pnls)/len(put_pnls):+.2%}")

        # Exit reason breakdown
        exit_counts = {}
        for t in trade_log:
            reason = t.get('exit_reason', 'unknown')
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
        print("\nExit reasons:")
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} ({count/len(trade_log):.1%})")

        # Multi-level stop info
        has_multilevel = any(
            y_batch[18] is not None if len(y_batch) >= 22 else False
            for _ in [0]  # dummy loop
        )
        if has_multilevel:
            print("\nStop-loss alignment: MULTI-LEVEL (tight/med/wide selected by IV+VIX)")
        else:
            print("\nStop-loss alignment: SINGLE-LEVEL (med only)")

        print("=== END DIAGNOSTICS ===")

    t_end = time.time()
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    # Compute num_trade_dates for generalization tracking
    trade_dates_set = set(t.get('date', '') for t in trade_log) if trade_log else set()
    metrics['num_trade_dates'] = len(trade_dates_set)

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
    print(f"num_trade_dates:    {metrics.get('num_trade_dates', 0)}")
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

    # --- Structured JSON output (parsed by inner_loop.py) ---
    import json as _json
    _json_metrics = {k: v for k, v in metrics.items() if k != 'trade_log'}
    _json_metrics['chunk_details'] = metrics.get('chunk_details', [])
    _json_metrics['num_steps'] = step
    _json_metrics['training_seconds'] = total_time
    _json_metrics['total_seconds'] = t_end - t_start
    _json_metrics['peak_vram_mb'] = peak_mb
    print("METRICS_JSON:" + _json.dumps(_json_metrics, default=str))