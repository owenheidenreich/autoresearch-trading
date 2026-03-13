"""
Autoresearch-trading training script (intraday edition).
Single-GPU, single-file. The agent modifies THIS file.

Predicts next-hour SPY direction for intraday options trading.
71 features including intraday bars, session structure, market internals,
multi-instrument data, and daily context.

Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from prepare import (
    TIME_BUDGET,
    NUM_FEATURES,
    FEATURE_NAMES,
    ANNUAL_TRADING_HOURS,
    HOURS_PER_DAY,
    load_data,
    make_dataloader,
    evaluate_sharpe,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags)
# ---------------------------------------------------------------------------

# Model architecture
LOOKBACK = 36            # hourly bars of history (36 = 6 trading days)
D_MODEL = 128            # model embedding dimension
N_HEADS = 4              # number of attention heads
DEPTH = 4                # number of transformer layers
FF_MULT = 4              # feedforward expansion factor
DROPOUT = 0.1            # dropout rate

# Optimization
BATCH_SIZE = 64          # batch size
LR = 3e-4                # peak learning rate (AdamW)
WEIGHT_DECAY = 0.01      # weight decay
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0          # gradient norm clipping
WARMUP_RATIO = 0.1       # fraction of time budget for LR warmup
COOLDOWN_RATIO = 0.3     # fraction of time budget for LR cooldown

# Loss: "directional" | "mse" | "sharpe" | "combined"
LOSS_TYPE = "directional"
SHARPE_ALPHA = 0.3       # weight for sharpe component in "combined" loss

# Position sizing
CONFIDENCE_THRESHOLD = 0.0  # flat if |prediction| < threshold (0 = always trade)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IntradayTradingModel(nn.Module):
    """Causal temporal transformer for intraday market prediction.

    Input:  (batch, lookback, num_features)  -- hourly bars with 71 features
    Output: (batch,) position signals in [-1, 1]
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        lookback: int = LOOKBACK,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = DEPTH,
        ff_mult: int = FF_MULT,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lookback = lookback

        # Feature projection
        self.input_proj = nn.Linear(num_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)

        # Transformer encoder (causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(lookback)
        self.register_buffer('causal_mask', mask)

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        B, T, F = x.shape

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x + self.pos_embed[:, :T, :]

        x = self.transformer(
            x,
            mask=self.causal_mask[:T, :T],
            is_causal=True,
        )

        x = x[:, -1, :]
        out = self.output_head(x).squeeze(-1)

        # Apply confidence threshold
        if CONFIDENCE_THRESHOLD > 0:
            out = torch.where(out.abs() < CONFIDENCE_THRESHOLD, torch.zeros_like(out), out)

        return out


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

# Load data
data = load_data()
n_bars = len(data['dates'])
print(f"Loaded {n_bars} hourly bars, {NUM_FEATURES} features")
print(f"  ~{n_bars // HOURS_PER_DAY} trading days")
print(f"Training:   up to idx {data['train_end_idx']}")
print(f"Validation: idx {data['val_start_idx']}-{data['val_end_idx']}")

# Build model
model = IntradayTradingModel().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=ADAM_BETAS,
)

# Compile
model = torch.compile(model)

# Dataloader
train_loader = make_dataloader(data, LOOKBACK, BATCH_SIZE, "train", device)
x_batch, y_batch = next(train_loader)

print(f"\nTime budget: {TIME_BUDGET}s")
print(f"Batch size: {BATCH_SIZE}  |  Lookback: {LOOKBACK}  |  Loss: {LOSS_TYPE}")
print(f"LR: {LR}  |  Depth: {DEPTH}  |  d_model: {D_MODEL}")
print()

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    """Warmup -> constant -> cosine cooldown."""
    if progress < WARMUP_RATIO:
        return progress / max(WARMUP_RATIO, 1e-8)
    elif progress < 1.0 - COOLDOWN_RATIO:
        return 1.0
    else:
        t = (1.0 - progress) / max(COOLDOWN_RATIO, 1e-8)
        return 0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0
smooth_loss = 0.0

while True:
    model.train()
    torch.cuda.synchronize()
    t0 = time.time()

    x, y = x_batch, y_batch
    pred = model(x)

    # --- Loss ---
    if LOSS_TYPE == "directional":
        loss = -(pred * y).mean()
    elif LOSS_TYPE == "mse":
        loss = F.mse_loss(pred, y.sign())
    elif LOSS_TYPE == "sharpe":
        pnl = pred * y
        loss = -(pnl.mean() / (pnl.std() + 1e-8))
    elif LOSS_TYPE == "combined":
        dir_loss = -(pred * y).mean()
        pnl = pred * y
        sharpe_loss = -(pnl.mean() / (pnl.std() + 1e-8))
        loss = (1 - SHARPE_ALPHA) * dir_loss + SHARPE_ALPHA * sharpe_loss
    else:
        loss = -(pred * y).mean()

    loss.backward()

    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for pg in optimizer.param_groups:
        pg['lr'] = LR * lrm

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    x_batch, y_batch = next(train_loader)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 5:
        total_training_time += dt

    loss_val = loss.item()

    if math.isnan(loss_val) or loss_val > 100:
        print(f"\nFAIL: loss={loss_val} at step {step}")
        exit(1)

    ema = 0.95
    smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
    debiased = smooth_loss / (1 - ema ** (step + 1))
    pct = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    if step % 50 == 0:
        print(f"step {step:05d} ({pct:5.1f}%) | loss: {debiased:.6f} "
              f"| lr: {LR * lrm:.2e} | dt: {dt*1000:.0f}ms "
              f"| remaining: {remaining:.0f}s")

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
metrics = evaluate_sharpe(model, data, LOOKBACK, device)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_sharpe:       {metrics['val_sharpe']:.6f}")
print(f"max_drawdown:     {metrics['max_drawdown']:.6f}")
print(f"annual_return:    {metrics['annual_return']:.6f}")
print(f"num_trades:       {metrics['num_trades']}")
print(f"win_rate:         {metrics['win_rate']:.6f}")
print(f"profit_factor:    {metrics['profit_factor']:.6f}")
print(f"num_val_bars:     {metrics['num_val_bars']}")
print(f"num_val_days:     {metrics['num_val_days']}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params:,}")
print(f"lookback:         {LOOKBACK}")
print(f"depth:            {DEPTH}")