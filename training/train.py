"""
Autoresearch-trading v18: SPX 0DTE trading model.

Five-head architecture:
  Market head:    (batch, 3) -- predicted SPX % change at 15/30/60 bar horizons
  Entry gate:     (batch, 1) -- sigmoid: should we enter a trade?
  Risk head:      (batch, 3) -- [stop_distance, target_distance, conviction]
  Exit head:      (batch, 1) -- sigmoid: should we exit current trade?
  Direction head: (batch, 6) -- 6-class softmax: call/put x ATM/OTM5/OTM10

Labels from path-quality analysis (MFE/MAE), not hindsight P&L.
Loss weighted by time-of-day bar_weight (morning=1.0, lunch=0.1, power_hour=0.0).

Usage: python3 train.py  (or deploy to GPU via inner_loop.py)
"""

import os
import sys
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
    BARS_PER_DAY,
    LOOKBACK_WINDOW,
    V18_NUM_DIRECTIONS,
    load_data,
    make_v18_dataloader,
)


# ---------------------------------------------------------------------------
# Hyperparameters -- the agent mutates these
# ---------------------------------------------------------------------------

def _env_float(name, default, lo=None, hi=None):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except Exception:
        return default
    if lo is not None: val = max(lo, val)
    if hi is not None: val = min(hi, val)
    return val

def _env_int(name, default, lo=None, hi=None):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except Exception:
        return default
    if lo is not None: val = max(lo, val)
    if hi is not None: val = min(hi, val)
    return val


# Architecture
LOOKBACK     = _env_int("TRAIN_LOOKBACK", 120, lo=60, hi=240)
D_MODEL      = _env_int("TRAIN_D_MODEL", 64, lo=32, hi=128)
N_HEADS      = 4
DEPTH        = _env_int("TRAIN_DEPTH", 3, lo=2, hi=6)
FF_MULT      = _env_int("TRAIN_FF_MULT", 3, lo=2, hi=6)
DROPOUT      = _env_float("TRAIN_DROPOUT", 0.30, lo=0.05, hi=0.40)

# Training
BATCH_SIZE   = _env_int("TRAIN_BATCH_SIZE", 1024, lo=32, hi=2048)
LR           = _env_float("TRAIN_LR", 2.5e-4, lo=1e-5, hi=5e-3)
WEIGHT_DECAY = _env_float("TRAIN_WEIGHT_DECAY", 0.08, lo=0.0, hi=0.3)
ADAM_BETAS   = (0.9, 0.98)
GRAD_CLIP    = _env_float("TRAIN_GRAD_CLIP", 1.0, lo=0.0, hi=5.0)
WARMUP_RATIO = _env_float("TRAIN_WARMUP_RATIO", 0.15, lo=0.0, hi=0.5)
COOLDOWN_RATIO = _env_float("TRAIN_COOLDOWN_RATIO", 0.4, lo=0.0, hi=0.8)

# v18 loss weights (relative to market prediction head)
GATE_W       = _env_float("TRAIN_GATE_W", 1.0, lo=0.0, hi=10.0)
RISK_W       = _env_float("TRAIN_RISK_W", 0.5, lo=0.0, hi=5.0)
EXIT_W       = _env_float("TRAIN_EXIT_W", 1.0, lo=0.0, hi=10.0)
DIR_W        = _env_float("TRAIN_DIR_W", 1.0, lo=0.0, hi=10.0)

# Feature groups (required by run_loop.py safety check)
FEATURE_GROUPS = {
    'returns':    (0, 2),    # ret_6, ret_12
    'volume':     (2, 3),    # volume_ratio
    'gamma':      (3, 4),    # gamma_pressure (v18)
    'vol':        (4, 7),    # bar_range, realized_vol, range_ratio
    'vwap':       (7, 8),    # vwap_dist
    'session':    (8, 9),    # session_range_pct
    'levels':     (9, 10),   # prev_high_dist
    'trend':      (10, 13),  # ema_cross, consec_direction, speed_estimate
    'vix_roc':    (13, 14),  # vix_roc (v18, was inside_bar)
    'time':       (14, 16),
    'options':    (16, 18),  # atm_iv, iv_skew
    'vix':        (18, 20),  # vix_regime, vrp
    'greeks':     (20, 23),  # atm_gamma, atm_theta_per_bar, charm_estimate
    'bollinger':  (23, 24),  # bollinger_position (5-min, v18)
    'range_ext':  (24, 26),  # rsi_7 (5-min, v18), session_range_position
    'mkt_struct': (26, 29),  # poc_dist, va_position, ib_break
    'pressure':   (29, 33),  # atr_14, bar_delta, session_cum_delta, option_spread_width (v18)
    'elder_5m':   (33, 37),  # macdh_slope, force_index_2, prev_close_dist, effort_vs_result (5-min, v18)
    'trend_5m':   (37, 38),  # trend_5min (5-min)
    'overnight':  (38, 39),  # overnight_gap
}

# Score formula -- LOCKED. Do NOT modify.
_score_config = {
    'win_rate_bonus': 0.5,
    'rr_bonus': 0.1,
    'drawdown_penalty': 0.5,
    'hold_bonus': 0.0,
    'freq_center': 1.5,
    'freq_width': 2.5,
    'consec_loss_threshold': 3,
    'short_hold_threshold': 0.30,
    'stop_rate_threshold': 0.30,
    'ruin_penalty': 1.0,
    'ruin_threshold': 0.25,
    'risk_fraction_penalty': 0.5,
}

# Return prediction horizons (in bars = minutes)
HORIZONS = [15, 30, 60]

# Number of direction classes
NUM_DIRECTIONS = V18_NUM_DIRECTIONS  # 6: call/put x ATM/OTM5/OTM10


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TradingModel(nn.Module):
    """v18 5-head SPX 0DTE trading model.

    Shared transformer backbone with 5 specialized heads:
    1. Market head: predict SPX returns at 15/30/60 bar horizons
    2. Entry gate: sigmoid -- probability this is a good entry
    3. Risk head: stop_distance, target_distance, conviction
    4. Exit head: sigmoid -- probability we should exit now
    5. Direction head: 6-class softmax (call/put x ATM/OTM5/OTM10)
    """

    def __init__(self, num_features=NUM_FEATURES, lookback=LOOKBACK,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=DEPTH,
                 ff_mult=FF_MULT, dropout=DROPOUT):
        super().__init__()
        self.lookback = lookback
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)

        # Transformer backbone (causal, Pre-LN)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult, dropout=dropout,
            batch_first=True, activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        mask = nn.Transformer.generate_square_subsequent_mask(lookback)
        self.register_buffer('causal_mask', mask)

        # Head 1: Market prediction (pure SPX returns, no trading context)
        self.market_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, len(HORIZONS)),  # 3 outputs
        )

        # Head 2: Entry gate (should we enter?)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Head 3: Risk parameters (stop, target, conviction)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )

        # Head 4: Exit signal (should we exit?)
        self.exit_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Head 5: Direction (6-class: call/put x ATM/OTM5/OTM10)
        self.direction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, NUM_DIRECTIONS),
        )

    def forward(self, x):
        # Shared backbone
        x_proj = self.input_proj(x)
        x_proj = self.input_norm(x_proj)
        x_proj = x_proj + self.pos_embed[:, :x_proj.size(1), :]
        x_proj = self.transformer(x_proj, mask=self.causal_mask[:x_proj.size(1), :x_proj.size(1)],
                                  is_causal=True)
        last = x_proj[:, -1, :]  # (batch, d_model)

        # Head outputs
        market_pred = self.market_head(last)                 # (batch, 3)
        entry_gate = torch.sigmoid(self.gate_head(last).squeeze(-1))   # (batch,)
        risk_params = self.risk_head(last)                   # (batch, 3)
        risk_params = torch.relu(risk_params)                # risk params are non-negative
        exit_signal = torch.sigmoid(self.exit_head(last).squeeze(-1))  # (batch,)
        direction_logits = self.direction_head(last)          # (batch, 6)

        return market_pred, entry_gate, risk_params, exit_signal, direction_logits


# ---------------------------------------------------------------------------
# Loss -- v18 multi-term with time-of-day weighting
# ---------------------------------------------------------------------------

def v18_loss(market_pred, entry_gate, risk_params, exit_signal, direction_logits,
             y_pred, y_v18, bar_weight):
    """Five-term loss for v18 trading model.

    Args:
        market_pred: (batch, 3) predicted SPX returns
        entry_gate: (batch,) predicted entry probability
        risk_params: (batch, 3) [stop_dist, target_dist, conviction]
        exit_signal: (batch,) predicted exit probability
        direction_logits: (batch, 6) direction class logits
        y_pred: (batch, 5) [ret_15, ret_30, ret_60, vol_30, action_target]
        y_v18: (batch, 8) [entry_gate, risk_stop, risk_target, risk_conviction,
                           exit_label, direction_label, mfe, mae]
        bar_weight: (batch,) time-of-day loss weight
    """
    # Unpack v18 labels
    gate_target = y_v18[:, 0]       # entry gate (0-1)
    risk_target = y_v18[:, 1:4]     # stop_dist, target_dist, conviction
    exit_target = y_v18[:, 4]       # exit label (0 or 1)
    dir_target = y_v18[:, 5].long() # direction class (0-5)

    # Unpack v17 labels for market head
    ret_target = y_pred[:, :3]      # return_15, return_30, return_60

    # Clamp direction target to valid range
    dir_target = dir_target.clamp(0, NUM_DIRECTIONS - 1)

    # 1. Market prediction loss (Huber, delta=0.01 for v18)
    market_loss = F.huber_loss(market_pred, ret_target, delta=0.01)

    # 2. Entry gate loss (BCE)
    gate_loss = F.binary_cross_entropy(entry_gate, gate_target)

    # 3. Risk parameter loss (Huber on stop/target/conviction)
    risk_loss = F.huber_loss(risk_params, risk_target, delta=0.5)

    # 4. Exit signal loss (BCE)
    exit_loss = F.binary_cross_entropy(exit_signal, exit_target)

    # 5. Direction loss (cross-entropy, weighted by bar_weight)
    dir_loss = F.cross_entropy(direction_logits, dir_target, reduction='none')
    dir_loss = (dir_loss * bar_weight).mean()

    # Combine with weights and bar_weight
    # Market loss always active (no bar_weight -- predict everywhere)
    # Gate, risk, exit, direction weighted by time-of-day
    weighted_gate = (gate_loss * bar_weight).mean() if bar_weight.sum() > 0 else gate_loss
    weighted_risk = (risk_loss)  # risk is already per-sample mean
    weighted_exit = (exit_loss * bar_weight).mean() if bar_weight.sum() > 0 else exit_loss

    total = (market_loss
             + GATE_W * weighted_gate
             + RISK_W * weighted_risk
             + EXIT_W * weighted_exit
             + DIR_W * dir_loss)

    return total, {
        'market': market_loss.item(),
        'gate': gate_loss.item(),
        'risk': risk_loss.item(),
        'exit': exit_loss.item(),
        'direction': dir_loss.item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Architecture: d_model={D_MODEL}, heads={N_HEADS}, depth={DEPTH}, "
          f"ff_mult={FF_MULT}, dropout={DROPOUT}")
    print(f"Training: lr={LR}, batch={BATCH_SIZE}, weight_decay={WEIGHT_DECAY}")
    print(f"Loss weights: GATE_W={GATE_W}, RISK_W={RISK_W}, EXIT_W={EXIT_W}, DIR_W={DIR_W}")
    print(f"Horizons: {HORIZONS} bars, directions: {NUM_DIRECTIONS}")

    # Load data
    data = load_data()
    n_feat = data['features'].shape[-1]
    n_bars = len(data['features'])
    print(f"Loaded {n_bars} bars, {n_feat} features")

    lookback = min(LOOKBACK, LOOKBACK_WINDOW * 2)

    # Create v18 dataloaders
    train_loader = make_v18_dataloader(data, lookback, BATCH_SIZE, split="train", device=device)
    val_loader = make_v18_dataloader(data, lookback, 2048, split="val", device=device)

    # Create model (FRESH START -- v18 architecture incompatible with v17)
    model = TradingModel(num_features=n_feat, lookback=lookback).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Warm start if best_model.pt exists AND is v18-compatible
    best_model_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')
    if os.path.exists(best_model_path):
        try:
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            config = ckpt.get('config', {})
            if config.get('model_version') == 'v18':
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state, strict=False)
                print(f"Warm start from {best_model_path} (v18)")
            else:
                print(f"Skipping warm start: model is {config.get('model_version', 'v17')}, need v18 (fresh start)")
        except Exception as e:
            print(f"Warm start failed ({e}), training from scratch")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
    )

    # LR schedule: warmup -> steady -> cooldown
    est_steps = 5000
    warmup_steps = int(est_steps * WARMUP_RATIO)
    cooldown_start = int(est_steps * (1 - COOLDOWN_RATIO))

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step >= cooldown_start:
            progress = (step - cooldown_start) / max(est_steps - cooldown_start, 1)
            return max(0.1, 1.0 - 0.9 * progress)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    start_time = time.time()
    step = 0
    batch_in_epoch = 0
    epoch_loss = 0.0
    epoch_components = {'market': 0.0, 'gate': 0.0, 'risk': 0.0, 'exit': 0.0, 'direction': 0.0}

    model.train()
    for xb, y_pred, y_v18, bw in train_loader:
        if time.time() - start_time >= TIME_BUDGET:
            break

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            market_pred, entry_gate, risk_params, exit_signal, dir_logits = model(xb)
            loss, components = v18_loss(
                market_pred, entry_gate, risk_params, exit_signal, dir_logits,
                y_pred, y_v18, bw,
            )

        optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        for k, v in components.items():
            epoch_components[k] += v
        batch_in_epoch += 1
        step += 1

        if step % 100 == 0:
            avg = epoch_loss / batch_in_epoch
            elapsed = time.time() - start_time
            comp_str = " ".join(f"{k}={v/batch_in_epoch:.4f}" for k, v in epoch_components.items())
            print(f"Step {step}: loss={avg:.6f}, lr={scheduler.get_last_lr()[0]:.2e}, "
                  f"time={elapsed:.0f}s/{TIME_BUDGET}s  [{comp_str}]")
            epoch_loss = 0.0
            batch_in_epoch = 0
            for k in epoch_components:
                epoch_components[k] = 0.0

    # Validation
    model.eval()
    val_losses = []
    pred_all = []
    actual_all = []
    gate_all = []
    gate_target_all = []
    exit_all = []
    exit_target_all = []
    dir_pred_all = []
    dir_target_all = []
    vix_all = []
    time_all = []

    IDX_VIX = 18
    IDX_MTC = 14

    with torch.no_grad():
        for xb, y_pred, y_v18, bw in val_loader:
            market_pred, entry_gate, risk_params, exit_signal, dir_logits = model(xb)
            vloss, _ = v18_loss(market_pred, entry_gate, risk_params, exit_signal,
                                dir_logits, y_pred, y_v18, bw)
            val_losses.append(vloss.item())
            pred_all.append(market_pred.cpu())
            actual_all.append(y_pred[:, :3].cpu())
            gate_all.append(entry_gate.cpu())
            gate_target_all.append(y_v18[:, 0].cpu())
            exit_all.append(exit_signal.cpu())
            exit_target_all.append(y_v18[:, 4].cpu())
            dir_pred_all.append(dir_logits.argmax(dim=-1).cpu())
            dir_target_all.append(y_v18[:, 5].long().cpu())
            if xb.shape[-1] > IDX_VIX:
                vix_all.append(xb[:, -1, IDX_VIX].cpu())
            if xb.shape[-1] > IDX_MTC:
                time_all.append(xb[:, -1, IDX_MTC].cpu())

    val_loss = np.mean(val_losses) if val_losses else float('inf')
    preds = torch.cat(pred_all, dim=0) if pred_all else torch.zeros(0, 3)
    actuals = torch.cat(actual_all, dim=0) if actual_all else torch.zeros(0, 3)
    gates = torch.cat(gate_all, dim=0) if gate_all else torch.zeros(0)
    gate_targets = torch.cat(gate_target_all, dim=0) if gate_target_all else torch.zeros(0)
    exits = torch.cat(exit_all, dim=0) if exit_all else torch.zeros(0)
    exit_targets = torch.cat(exit_target_all, dim=0) if exit_target_all else torch.zeros(0)
    dir_preds = torch.cat(dir_pred_all, dim=0) if dir_pred_all else torch.zeros(0, dtype=torch.long)
    dir_targets = torch.cat(dir_target_all, dim=0) if dir_target_all else torch.zeros(0, dtype=torch.long)
    vix_data = torch.cat(vix_all, dim=0) if vix_all else torch.zeros(0)
    time_data = torch.cat(time_all, dim=0) if time_all else torch.zeros(0)

    # --- Metrics ---
    if len(preds) > 0:
        pred_30 = preds[:, 1]
        actual_30 = actuals[:, 1]

        # Direction accuracy (sign match on 30-bar horizon)
        significant = actual_30.abs() > 0.0005
        dir_correct = ((pred_30 > 0) == (actual_30 > 0)).float()
        dir_accuracy = dir_correct[significant].mean().item() if significant.any() else 0.5
        n_significant = int(significant.sum().item())

        # Return MAE
        return_mae = (pred_30 - actual_30).abs().mean().item()

        # Rank correlation
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(pred_30.numpy(), actual_30.numpy())
            rank_corr = rho if not np.isnan(rho) else 0.0
        except ImportError:
            rank_corr = 0.0

        # Win rate with margin of error
        wr_stderr = math.sqrt(dir_accuracy * (1 - dir_accuracy) / max(n_significant, 1))
        wr_lower_bound = dir_accuracy - 1.96 * wr_stderr

        # Gate accuracy (how well does gate predict good entries?)
        gate_acc = ((gates > 0.5) == (gate_targets > 0.5)).float().mean().item()
        gate_precision = 0.0
        gate_selected = gates > 0.5
        if gate_selected.any():
            gate_precision = gate_targets[gate_selected].mean().item()

        # Exit accuracy
        exit_acc = ((exits > 0.5) == (exit_targets > 0.5)).float().mean().item()

        # Direction accuracy (6-class)
        dir_class_acc = (dir_preds == dir_targets).float().mean().item()

        # Regime decomposition
        wr_high_vol = 0.5
        wr_low_vol = 0.5
        if len(vix_data) == len(dir_correct) and len(vix_data) > 0:
            high_vol = (vix_data > 0) & significant
            low_vol = (vix_data <= 0) & significant
            if high_vol.any():
                wr_high_vol = dir_correct[high_vol].mean().item()
            if low_vol.any():
                wr_low_vol = dir_correct[low_vol].mean().item()

        # Session decomposition
        wr_morning = 0.5
        wr_afternoon = 0.5
        if len(time_data) == len(dir_correct) and len(time_data) > 0:
            morning = (time_data > 4.5) & significant
            afternoon = (time_data <= 4.5) & significant
            if morning.any():
                wr_morning = dir_correct[morning].mean().item()
            if afternoon.any():
                wr_afternoon = dir_correct[afternoon].mean().item()

        # Consecutive loss tracking
        max_consec_wrong = 0
        current_streak = 0
        for c in dir_correct[significant].tolist():
            if c < 0.5:
                current_streak += 1
                max_consec_wrong = max(max_consec_wrong, current_streak)
            else:
                current_streak = 0
    else:
        dir_accuracy = 0.5
        return_mae = 1.0
        rank_corr = 0.0
        wr_lower_bound = 0.0
        wr_high_vol = 0.5
        wr_low_vol = 0.5
        wr_morning = 0.5
        wr_afternoon = 0.5
        max_consec_wrong = 0
        n_significant = 0
        gate_acc = 0.5
        gate_precision = 0.0
        exit_acc = 0.5
        dir_class_acc = 1.0 / NUM_DIRECTIONS

    print(f"\n--- Validation ---")
    print(f"Val loss: {val_loss:.6f}")
    print(f"Direction accuracy (30-bar): {dir_accuracy:.1%} (n={n_significant})")
    print(f"  WR lower bound (95%): {wr_lower_bound:.1%}")
    print(f"  WR high-vol: {wr_high_vol:.1%} | WR low-vol: {wr_low_vol:.1%}")
    print(f"  WR morning:  {wr_morning:.1%} | WR afternoon: {wr_afternoon:.1%}")
    print(f"  Max consecutive wrong: {max_consec_wrong}")
    print(f"Return MAE (30-bar): {return_mae:.6f}")
    print(f"Rank correlation: {rank_corr:.4f}")
    print(f"Gate accuracy: {gate_acc:.1%} | Gate precision: {gate_precision:.1%}")
    print(f"Exit accuracy: {exit_acc:.1%}")
    print(f"Direction class accuracy: {dir_class_acc:.1%} (chance={1/NUM_DIRECTIONS:.1%})")

    # Score: direction accuracy * (1 + rank_correlation)
    score = dir_accuracy * (1.0 + max(0, rank_corr))
    print(f"Score: {score:.4f}")

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': {
            'score': score,
            'val_loss': val_loss,
            'dir_accuracy': dir_accuracy,
            'return_mae': return_mae,
            'rank_corr': rank_corr,
            'wr_lower_bound': wr_lower_bound,
            'wr_high_vol': wr_high_vol,
            'wr_low_vol': wr_low_vol,
            'wr_morning': wr_morning,
            'wr_afternoon': wr_afternoon,
            'max_consec_wrong': max_consec_wrong,
            'gate_accuracy': gate_acc,
            'gate_precision': gate_precision,
            'exit_accuracy': exit_acc,
            'direction_class_accuracy': dir_class_acc,
        },
        'config': {
            'model_version': 'v18',
            'num_features': n_feat,
            'lookback': lookback,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'depth': DEPTH,
            'ff_mult': FF_MULT,
            'dropout': DROPOUT,
            'horizons': HORIZONS,
            'num_directions': NUM_DIRECTIONS,
        },
        '_score_config': _score_config,
    }, save_path)
    print(f"Saved to {save_path}")

    # Print summary for inner_loop.py parsing
    print(f"\n=== METRICS ===")
    print(f"score: {score:.6f}")
    print(f"val_loss: {val_loss:.6f}")
    print(f"dir_accuracy: {dir_accuracy:.4f}")
    print(f"return_mae: {return_mae:.6f}")
    print(f"rank_corr: {rank_corr:.4f}")
    print(f"wr_lower_bound: {wr_lower_bound:.4f}")
    print(f"wr_high_vol: {wr_high_vol:.4f}")
    print(f"wr_low_vol: {wr_low_vol:.4f}")
    print(f"wr_morning: {wr_morning:.4f}")
    print(f"wr_afternoon: {wr_afternoon:.4f}")
    print(f"max_consec_wrong: {max_consec_wrong}")
    print(f"gate_accuracy: {gate_acc:.4f}")
    print(f"gate_precision: {gate_precision:.4f}")
    print(f"exit_accuracy: {exit_acc:.4f}")
    print(f"direction_class_accuracy: {dir_class_acc:.4f}")
    print(f"num_steps: {step}")
    print(f"training_seconds: {time.time() - start_time:.0f}")


if __name__ == '__main__':
    main()
