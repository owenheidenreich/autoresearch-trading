"""ART² v2 Training Loop

The mutable research file. The autoresearch loop mutates this file.

Model architecture: time-series encoder + trade decision heads.
Input: (batch, lookback, 39) normalized features
Output: TradeIntent fields (trade, direction, strike_offset, stop, target, hold, confidence)

Loss: supervised from oracle labels (core/labels.py).
Evaluation: replay simulation (core/simulator.py + core/metrics.py).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from v2.core.features import (
    NUM_FEATURES, BARS_PER_DAY,
    NO_TRADE_BEFORE_BAR, NO_TRADE_AFTER_BAR,
)
from v2.core.metrics import score_config_fingerprint


# ---------------------------------------------------------------------------
# Hyperparameters (tunable by autoresearch)
# ---------------------------------------------------------------------------

LOOKBACK = int(os.environ.get("TRAIN_LOOKBACK", 60))
D_MODEL = int(os.environ.get("TRAIN_D_MODEL", 64))
N_HEADS = 4
DEPTH = int(os.environ.get("TRAIN_DEPTH", 3))
DROPOUT = float(os.environ.get("TRAIN_DROPOUT", 0.1))

BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 512))
LR = float(os.environ.get("TRAIN_LR", 3e-4))
WEIGHT_DECAY = float(os.environ.get("TRAIN_WEIGHT_DECAY", 0.05))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 30))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 300))  # seconds

# Loss weights
GATE_W = float(os.environ.get("WEIGHT_GATE", 1.0))
DIR_W = float(os.environ.get("WEIGHT_DIR", 1.0))
STRIKE_W = float(os.environ.get("WEIGHT_STRIKE", 0.5))
RISK_W = float(os.environ.get("WEIGHT_RISK", 0.3))

# Gate selectivity: pos_weight < 1.0 makes the model more conservative
# (penalizes false positives more than false negatives)
# Tier 3 has 73.7% positive rate. pos_weight=0.1 means 10x penalty for
# false positives -- forces model to be highly selective.
GATE_POS_WEIGHT = float(os.environ.get("WEIGHT_GATE_POS", 0.1))

# Number of strike offset classes: 13 (ATM + 6 call offsets + 6 put offsets)
NUM_STRIKE_CLASSES = 13
# Strike offsets: -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30
STRIKE_OFFSETS = list(range(-30, 31, 5))
STRIKE_OFFSET_TO_IDX = {off: i for i, off in enumerate(STRIKE_OFFSETS)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TradingModel(nn.Module):
    """v2 Trading Model.

    Input: (batch, lookback, NUM_FEATURES)
    Outputs:
        gate: (batch, 1) - sigmoid probability of entering a trade
        direction: (batch, 2) - softmax over call/put
        strike: (batch, NUM_STRIKE_CLASSES) - softmax over strike offsets
        risk: (batch, 3) - [stop_pct, target_pct, max_hold_frac]
        confidence: (batch, 1) - sigmoid confidence score
    """

    def __init__(
        self,
        d_model: int = None,
        depth: int = None,
        n_heads: int = None,
        dropout: float = None,
    ):
        super().__init__()
        # Use provided args or fall back to module-level globals
        d = d_model or D_MODEL
        dep = depth or DEPTH
        nh = n_heads or N_HEADS
        dr = dropout if dropout is not None else DROPOUT

        self.input_proj = nn.Linear(NUM_FEATURES, d)
        self.input_norm = nn.LayerNorm(d)
        self.pos_enc = PositionalEncoding(d, max_len=LOOKBACK + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nh,
            dim_feedforward=d * 4,
            dropout=dr,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=dep)

        # Causal mask
        self.register_buffer(
            'causal_mask',
            nn.Transformer.generate_square_subsequent_mask(LOOKBACK),
        )

        # Heads
        self.gate_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, 2),  # call, put
        )
        self.strike_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, NUM_STRIKE_CLASSES),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, 3),  # stop_pct, target_pct, max_hold_frac
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, lookback, NUM_FEATURES)
        Returns:
            dict with gate, direction, strike, risk, confidence
        """
        B, T, F = x.shape

        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.pos_enc(h)

        # Causal transformer
        mask = self.causal_mask[:T, :T] if T <= self.causal_mask.size(0) else None
        h = self.encoder(h, mask=mask)

        # Use last token
        last = h[:, -1, :]  # (B, D_MODEL)

        return {
            'gate': self.gate_head(last),               # (B, 1) logits
            'direction': self.direction_head(last),      # (B, 2) logits
            'strike': self.strike_head(last),            # (B, 13) logits
            'risk': self.risk_head(last),                # (B, 3) raw
            'confidence': self.confidence_head(last),    # (B, 1) logits
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TradeDataset(Dataset):
    """Windowed dataset for training."""

    def __init__(self, features: torch.Tensor, labels: dict[str, torch.Tensor],
                 mask: torch.Tensor, lookback: int = LOOKBACK):
        self.features = features
        self.labels = labels
        self.lookback = lookback

        # Valid indices: must have full lookback window and be in mask
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        all_indices = np.arange(lookback, len(features))
        self.indices = all_indices[mask_np[lookback:]].copy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        i = self.indices[idx]
        window = self.features[i - self.lookback:i]  # (lookback, 39)

        target = {}
        for key, tensor in self.labels.items():
            target[key] = tensor[i]

        return window, target


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    bar_of_day: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute training loss from model outputs and oracle labels.

    Returns (total_loss, loss_dict) for logging.
    """
    device = outputs['gate'].device

    # Oracle labels
    oracle_trade = targets['oracle_trade'].float().to(device)
    oracle_right = targets['oracle_right'].long().to(device)
    oracle_offset = targets['oracle_strike_offset'].long().to(device)
    oracle_stop = targets['oracle_stop_pct'].float().to(device)
    oracle_target = targets['oracle_target_pct'].float().to(device)
    oracle_max_hold = targets['oracle_max_hold'].float().to(device)
    oracle_confidence = targets['oracle_confidence'].float().to(device)

    # 1. Gate loss: BCE with pos_weight to control selectivity
    # pos_weight < 1.0 penalizes false positives (saying "trade" when shouldn't)
    # more than false negatives (missing a trade opportunity)
    pw = torch.tensor([GATE_POS_WEIGHT], device=device)
    gate_loss = F.binary_cross_entropy_with_logits(
        outputs['gate'].squeeze(-1), oracle_trade,
        pos_weight=pw,
    )

    # 2. Direction loss: cross-entropy on call/put (only for trade=True bars)
    trade_mask = oracle_trade > 0.5
    if trade_mask.any():
        dir_targets = oracle_right[trade_mask]
        # Filter valid direction targets (0=call, 1=put)
        valid_dir = (dir_targets >= 0) & (dir_targets <= 1)
        if valid_dir.any():
            # Balance direction loss: equal weight to calls and puts
            dir_weight = torch.ones(2, device=device)
            n_calls = (dir_targets[valid_dir] == 0).sum().float()
            n_puts = (dir_targets[valid_dir] == 1).sum().float()
            if n_calls > 0 and n_puts > 0:
                dir_weight[0] = n_puts / (n_calls + n_puts)  # upweight minority
                dir_weight[1] = n_calls / (n_calls + n_puts)
            dir_loss = F.cross_entropy(
                outputs['direction'][trade_mask][valid_dir],
                dir_targets[valid_dir],
                weight=dir_weight,
            )
        else:
            dir_loss = torch.tensor(0.0, device=device)

        # 3. Strike loss: cross-entropy on strike offset class
        # Convert offset to class index
        strike_targets = torch.zeros_like(oracle_offset[trade_mask])
        for i, off in enumerate(oracle_offset[trade_mask]):
            off_val = int(off.item())
            strike_targets[i] = STRIKE_OFFSET_TO_IDX.get(off_val, NUM_STRIKE_CLASSES // 2)
        strike_loss = F.cross_entropy(
            outputs['strike'][trade_mask],
            strike_targets,
        )

        # 4. Risk loss: Huber on stop, target, max_hold (only for trades)
        risk_out = outputs['risk'][trade_mask]  # (n_trades, 3)
        risk_targets = torch.stack([
            oracle_stop[trade_mask],
            oracle_target[trade_mask],
            oracle_max_hold[trade_mask] / BARS_PER_DAY,  # normalize to [0,1]
        ], dim=-1)
        risk_loss = F.huber_loss(risk_out, risk_targets, delta=0.5)
    else:
        dir_loss = torch.tensor(0.0, device=device)
        strike_loss = torch.tensor(0.0, device=device)
        risk_loss = torch.tensor(0.0, device=device)

    # 5. Confidence loss: BCE
    conf_loss = F.binary_cross_entropy_with_logits(
        outputs['confidence'].squeeze(-1), oracle_confidence,
    )

    # Total weighted loss
    total = (
        GATE_W * gate_loss
        + DIR_W * dir_loss
        + STRIKE_W * strike_loss
        + RISK_W * risk_loss
        + 0.5 * conf_loss
    )

    loss_dict = {
        'gate': gate_loss.item(),
        'direction': dir_loss.item(),
        'strike': strike_loss.item(),
        'risk': risk_loss.item(),
        'confidence': conf_loss.item(),
        'total': total.item(),
    }

    return total, loss_dict


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def load_dataset(path: str = "v2/data.pt") -> dict:
    """Load v2 dataset."""
    print(f"Loading dataset from {path}...")
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d


def train(data_path: str = "v2/data.pt", model_path: str = "v2/model.pt"):
    """Main training function."""
    t_start = time.time()

    # Load data
    data = load_dataset(data_path)
    features = data['X']
    bar_of_day = data.get('bar_of_day', torch.zeros(len(features), dtype=torch.long))

    labels = {
        'oracle_trade': data['oracle_trade'],
        'oracle_right': data['oracle_right'],
        'oracle_strike_offset': data['oracle_strike_offset'],
        'oracle_stop_pct': data['oracle_stop_pct'],
        'oracle_target_pct': data['oracle_target_pct'],
        'oracle_max_hold': data['oracle_max_hold'],
        'oracle_confidence': data['oracle_confidence'],
    }

    # Train/val split
    train_mask = data['train_mask']
    val_mask = data['val_mask']

    train_ds = TradeDataset(features, labels, train_mask, lookback=LOOKBACK)
    val_ds = TradeDataset(features, labels, val_mask, lookback=LOOKBACK)

    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters, device={device}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        # Time budget check
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET:
            print(f"Time budget ({TIME_BUDGET}s) reached at epoch {epoch}")
            break

        # Train
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = {k: v.to(device) for k, v in batch_y.items()}

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss, loss_dict = compute_loss(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss_dict)

        scheduler.step()

        # Aggregate train losses
        avg_train = {}
        if train_losses:
            for key in train_losses[0]:
                avg_train[key] = np.mean([d[key] for d in train_losses])

        # Validate
        model.eval()
        val_losses = []
        val_gate_correct = 0
        val_gate_total = 0
        val_dir_correct = 0
        val_dir_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = {k: v.to(device) for k, v in batch_y.items()}

                outputs = model(batch_x)
                loss, loss_dict = compute_loss(outputs, batch_y)
                val_losses.append(loss_dict)

                # Gate accuracy
                gate_pred = (torch.sigmoid(outputs['gate'].squeeze(-1)) > 0.5).float()
                gate_true = batch_y['oracle_trade'].float()
                val_gate_correct += (gate_pred == gate_true).sum().item()
                val_gate_total += len(gate_true)

                # Direction accuracy (only for trade bars)
                trade_mask = batch_y['oracle_trade'] > 0.5
                if trade_mask.any():
                    dir_pred = outputs['direction'][trade_mask].argmax(dim=-1)
                    dir_true = batch_y['oracle_right'][trade_mask]
                    valid = (dir_true >= 0) & (dir_true <= 1)
                    if valid.any():
                        val_dir_correct += (dir_pred[valid] == dir_true[valid]).sum().item()
                        val_dir_total += valid.sum().item()

        avg_val = {}
        if val_losses:
            for key in val_losses[0]:
                avg_val[key] = np.mean([d[key] for d in val_losses])

        gate_acc = val_gate_correct / val_gate_total if val_gate_total > 0 else 0
        dir_acc = val_dir_correct / val_dir_total if val_dir_total > 0 else 0

        print(f"Epoch {epoch:3d} | "
              f"train_loss={avg_train.get('total', 0):.4f} | "
              f"val_loss={avg_val.get('total', 0):.4f} | "
              f"gate_acc={gate_acc:.3f} | "
              f"dir_acc={dir_acc:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best
        val_total = avg_val.get('total', float('inf'))
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            dataset_fp = data.get('metadata', {}).get('fingerprint', 'unknown')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_total,
                'gate_acc': gate_acc,
                'dir_acc': dir_acc,
                'hyperparams': {
                    'lookback': LOOKBACK, 'd_model': D_MODEL, 'depth': DEPTH,
                    'n_heads': N_HEADS, 'dropout': DROPOUT,
                    'batch_size': BATCH_SIZE, 'lr': LR,
                },
                'score_config_fingerprint': score_config_fingerprint(),
                'dataset_fingerprint': dataset_fp,
            }, model_path)

    # Final metrics output (for inner_loop parsing)
    metrics = {
        'val_loss': best_val_loss,
        'gate_accuracy': gate_acc,
        'dir_accuracy': dir_acc,
        'best_epoch': best_epoch,
        'epochs_run': min(epoch, EPOCHS),
        'score_config_fingerprint': score_config_fingerprint(),
    }
    print(f"\nMETRICS_JSON:{json.dumps(metrics)}")
    print(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")

    return model, metrics


if __name__ == "__main__":
    _data = os.environ.get("TRAIN_DATA_PATH", "v2/data.pt")
    _model = os.environ.get("TRAIN_MODEL_PATH", "v2/model.pt")
    train(data_path=_data, model_path=_model)
