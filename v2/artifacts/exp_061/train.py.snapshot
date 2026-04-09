"""ART² v2 Training Loop -- P&L Prediction Model

The model predicts expected P&L for BOTH call and put directions.
Trading decisions derive from predictions at inference time:
  - gate = True if max(pred_call_pnl, pred_put_pnl) > threshold
  - direction = argmax(pred_call_pnl, pred_put_pnl)
  - confidence = |pred_call_pnl - pred_put_pnl|

This replaces the old classification approach (gate/direction as separate heads)
with regression: predict the outcome, then decide whether to trade.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from v2.core.features import (
    BARS_PER_DAY,
    NO_TRADE_BEFORE_BAR, NO_TRADE_AFTER_BAR,
)

# Feature count determined at runtime from data.pt shape
NUM_FEATURES = int(os.environ.get("NUM_FEATURES", 47))
from v2.core.metrics import score_config_fingerprint


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LOOKBACK = int(os.environ.get("TRAIN_LOOKBACK", 30))
D_MODEL = int(os.environ.get("TRAIN_D_MODEL", 64))
N_HEADS = 4
DEPTH = int(os.environ.get("TRAIN_DEPTH", 3))
DROPOUT = float(os.environ.get("TRAIN_DROPOUT", 0.05))

BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 2048))
LR = float(os.environ.get("TRAIN_LR", 5e-4))
WEIGHT_DECAY = float(os.environ.get("TRAIN_WEIGHT_DECAY", 0.03))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 30))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 300))

# Loss weights
PNL_W = float(os.environ.get("WEIGHT_PNL", 1.0))
RISK_W = float(os.environ.get("WEIGHT_RISK", 0.3))

# For replay compatibility
NUM_STRIKE_CLASSES = 13
STRIKE_OFFSETS = list(range(-30, 31, 5))
STRIKE_OFFSET_TO_IDX = {off: i for i, off in enumerate(STRIKE_OFFSETS)}

REGIME_DIM = 16


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


class FiLMLayer(nn.Module):
    def __init__(self, regime_dim: int, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(regime_dim, feature_dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        with torch.no_grad():
            self.fc.bias[:feature_dim] = 1.0

    def forward(self, regime: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gb = self.fc(regime)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma * x + beta


class TradingModel(nn.Module):
    """P&L prediction model.

    Instead of classifying gate/direction, predicts expected P&L for
    both call and put. Trading decisions derive from predictions:
      gate = max(call_pnl, put_pnl) > threshold
      direction = argmax(call_pnl, put_pnl)

    Outputs:
        call_pnl: (batch, 1) - predicted P&L if buying ATM call
        put_pnl: (batch, 1) - predicted P&L if buying ATM put
        risk: (batch, 3) - [stop_pct, target_pct, max_hold_frac]

    For replay compatibility, also outputs gate, direction, strike, confidence
    derived from the P&L predictions.
    """

    def __init__(
        self,
        d_model: int = None,
        depth: int = None,
        n_heads: int = None,
        dropout: float = None,
    ):
        super().__init__()
        d = d_model or D_MODEL
        dep = depth or DEPTH
        nh = n_heads or N_HEADS
        dr = dropout if dropout is not None else DROPOUT

        self.input_proj = nn.Linear(NUM_FEATURES, d)
        self.input_norm = nn.LayerNorm(d)
        self.pos_enc = PositionalEncoding(d, max_len=LOOKBACK + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nh, dim_feedforward=d * 4,
            dropout=dr, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=dep)

        self.register_buffer(
            'causal_mask',
            nn.Transformer.generate_square_subsequent_mask(LOOKBACK),
        )

        # Regime encoder
        self.regime_encoder = nn.Sequential(
            nn.Linear(NUM_FEATURES, 32), nn.GELU(), nn.Linear(32, REGIME_DIM),
        )

        # FiLM layers for P&L heads
        self.film_call = FiLMLayer(REGIME_DIM, d)
        self.film_put = FiLMLayer(REGIME_DIM, d)
        self.film_risk = FiLMLayer(REGIME_DIM, d)

        # P&L prediction heads (regression, not classification)
        self.call_pnl_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, 1),
        )
        self.put_pnl_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T, F = x.shape

        regime = self.regime_encoder(x[:, -1, :])

        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.pos_enc(h)

        mask = self.causal_mask[:T, :T] if T <= self.causal_mask.size(0) else None
        h = self.encoder(h, mask=mask)

        last = h[:, -1, :]

        call_pnl = self.call_pnl_head(self.film_call(regime, last))  # (B, 1)
        put_pnl = self.put_pnl_head(self.film_put(regime, last))    # (B, 1)
        risk = self.risk_head(self.film_risk(regime, last))           # (B, 3)

        # Derive gate/direction/strike/confidence for replay compatibility
        call_v = call_pnl.squeeze(-1)  # (B,)
        put_v = put_pnl.squeeze(-1)    # (B,)

        # Gate: raw max predicted P&L as logit (no scaling)
        # sigmoid(0) = 0.5, so gate_threshold=0.5 means "trade when best P&L > 0"
        max_pnl = torch.max(call_v, put_v)
        gate_logit = max_pnl

        # Direction: [call_logit, put_logit] from P&L predictions
        direction = torch.stack([call_v, put_v], dim=-1)  # (B, 2)

        # Strike: mild ATM prior (model can learn to override, but defaults to ATM)
        strike = torch.zeros(B, NUM_STRIKE_CLASSES, device=x.device)
        strike[:, NUM_STRIKE_CLASSES // 2] = 1.0

        # Confidence: margin between directions
        confidence = torch.abs(call_v - put_v).unsqueeze(-1) * 3.0  # (B, 1)

        return {
            'call_pnl': call_pnl,
            'put_pnl': put_pnl,
            'gate': gate_logit.unsqueeze(-1),
            'direction': direction,
            'strike': strike,
            'risk': risk,
            'confidence': confidence,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TradeDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: dict[str, torch.Tensor],
                 mask: torch.Tensor, lookback: int = LOOKBACK):
        self.features = features
        self.labels = labels
        self.lookback = lookback

        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        all_indices = np.arange(lookback, len(features))
        self.indices = all_indices[mask_np[lookback:]].copy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        i = self.indices[idx]
        window = self.features[i - self.lookback:i]
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
    """P&L regression loss.

    The model predicts call_pnl and put_pnl. Loss = Huber on both predictions
    against actual forward P&L for each direction.

    Only bars with valid signal (label_direction >= 0) are used.
    """
    device = outputs['call_pnl'].device

    lab_call_pnl = targets['label_call_pnl'].float().to(device)
    lab_put_pnl = targets['label_put_pnl'].float().to(device)
    lab_direction = targets['label_direction'].long().to(device)

    # Valid signal mask: bars where we have P&L data
    valid = lab_direction >= 0

    if not valid.any():
        zero = torch.tensor(0.0, device=device)
        return zero, {'call_pnl': 0.0, 'put_pnl': 0.0, 'total': 0.0}

    # P&L regression: predict both directions
    pred_call = outputs['call_pnl'].squeeze(-1)[valid]
    pred_put = outputs['put_pnl'].squeeze(-1)[valid]
    true_call = lab_call_pnl[valid]
    true_put = lab_put_pnl[valid]

    # Direction-asymmetric loss: penalize optimistic errors more for puts than calls.
    # Trade data shows puts WR 66% vs calls 82%. Put P&L predictions are noisier.
    # Calls: 4x penalty for predicting profit on actual loss.
    # Puts: 6x penalty -- stricter because put predictions less reliable.
    call_err = pred_call - true_call
    put_err = pred_put - true_put
    call_weight = torch.where(
        (call_err > 0) & (true_call < 0), 4.0, 1.0
    )
    put_weight = torch.where(
        (put_err > 0) & (true_put < 0), 6.0, 1.0
    )

    # Sample weighting: bars with large |P&L| carry more signal
    # Bars near zero P&L are noisy coin flips; large moves are learnable
    max_abs_pnl = torch.max(true_call.abs(), true_put.abs())
    sample_weight = 1.0 + max_abs_pnl  # baseline 1.0, up-weight high-signal bars

    call_loss = (sample_weight * call_weight * F.huber_loss(pred_call, true_call, delta=0.5, reduction='none')).mean()
    put_loss = (sample_weight * put_weight * F.huber_loss(pred_put, true_put, delta=0.5, reduction='none')).mean()

    pnl_loss = call_loss + put_loss

    # Risk loss: per-bar targets from labels if available, else fixed defaults
    # hold_frac normalized by MAX_HOLD_BARS to match replay decoding:
    #   replay does: max_hold = max(hold_lo, int(hold_raw * hold_hi))
    #   so training target must be: label_max_hold / hold_hi
    from v2.core.policy import DEFAULT_POLICY
    _hold_hi = DEFAULT_POLICY.max_hold_range[1]  # 250
    risk_out = outputs['risk'][valid]
    if 'label_stop_pct' in targets and 'label_target_pct' in targets and 'label_max_hold' in targets:
        t_stop = targets['label_stop_pct'].float().to(device)[valid]
        t_target = targets['label_target_pct'].float().to(device)[valid]
        t_hold = targets['label_max_hold'].float().to(device)[valid] / _hold_hi
        risk_target = torch.stack([t_stop, t_target, t_hold], dim=-1)
    else:
        risk_target = torch.tensor([0.30, 0.50, 30.0 / _hold_hi], device=device)
        risk_target = risk_target.unsqueeze(0).expand_as(risk_out)
    risk_loss = F.huber_loss(risk_out, risk_target, delta=0.5)

    total = PNL_W * pnl_loss + RISK_W * risk_loss

    # Metrics for logging
    with torch.no_grad():
        # Direction accuracy: did we predict the right side?
        pred_dir = (pred_put > pred_call).long()
        true_dir = lab_direction[valid]
        dir_valid = (true_dir >= 0) & (true_dir <= 1)
        dir_acc = (pred_dir[dir_valid] == true_dir[dir_valid]).float().mean().item() if dir_valid.any() else 0.0

        # Gate accuracy: does max(pred) > 0 match label_trade?
        lab_trade = targets['label_trade'].float().to(device)
        pred_trade = (torch.max(pred_call, pred_put) > 0).float()
        true_trade = lab_trade[valid]
        gate_acc = (pred_trade == true_trade).float().mean().item()

        # Conditional metrics: direction accuracy only on gated bars
        pred_gated = pred_trade.bool()
        dir_acc_gated = 0.0
        if pred_gated.any() and dir_valid.any():
            gated_and_valid = pred_gated & dir_valid
            if gated_and_valid.any():
                dir_acc_gated = (pred_dir[gated_and_valid] == true_dir[gated_and_valid]).float().mean().item()

        # Avg predicted P&L for gated vs ungated (measures gate value)
        max_pred = torch.max(pred_call, pred_put)
        avg_pnl_gated = max_pred[pred_gated].mean().item() if pred_gated.any() else 0.0
        avg_pnl_ungated = max_pred[~pred_gated].mean().item() if (~pred_gated).any() else 0.0

    loss_dict = {
        'call_pnl': call_loss.item(),
        'put_pnl': put_loss.item(),
        'risk': risk_loss.item(),
        'total': total.item(),
        'dir_acc': dir_acc,
        'gate_acc': gate_acc,
        'dir_acc_gated': dir_acc_gated,
        'avg_pnl_gated': avg_pnl_gated,
        'avg_pnl_ungated': avg_pnl_ungated,
    }

    return total, loss_dict


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def load_dataset(path: str = "v2/data.pt") -> dict:
    print(f"Loading dataset from {path}...")
    return torch.load(path, map_location="cpu", weights_only=False)


SEED = int(os.environ.get("TRAIN_SEED", 123))


def train(data_path: str = "v2/data.pt", model_path: str = "v2/model.pt",
          train_mask_override=None, val_mask_override=None):
    t_start = time.time()

    # Reproducible training (re-read env var so walk-forward can set per-fold seeds)
    seed = int(os.environ.get("TRAIN_SEED", SEED))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")

    data = load_dataset(data_path)
    features = data['X']

    labels = {
        'label_trade': data['label_trade'],
        'label_direction': data['label_direction'],
        'label_call_pnl': data['label_call_pnl'],
        'label_put_pnl': data['label_put_pnl'],
        'label_stop_pct': data['label_stop_pct'],
        'label_target_pct': data['label_target_pct'],
        'label_max_hold': data['label_max_hold'],
        'label_confidence': data['label_confidence'],
    }

    train_mask = train_mask_override if train_mask_override is not None else data['train_mask']
    val_mask = val_mask_override if val_mask_override is not None else data['val_mask']

    train_ds = TradeDataset(features, labels, train_mask, lookback=LOOKBACK)
    val_ds = TradeDataset(features, labels, val_mask, lookback=LOOKBACK)

    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    best_epoch = 0
    gate_acc = 0.0
    dir_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET:
            print(f"Time budget ({TIME_BUDGET}s) reached at epoch {epoch}")
            break

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

        avg_train = {}
        if train_losses:
            for key in train_losses[0]:
                avg_train[key] = np.mean([d[key] for d in train_losses])

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = {k: v.to(device) for k, v in batch_y.items()}

                outputs = model(batch_x)
                loss, loss_dict = compute_loss(outputs, batch_y)
                val_losses.append(loss_dict)

        avg_val = {}
        if val_losses:
            for key in val_losses[0]:
                avg_val[key] = np.mean([d[key] for d in val_losses])

        gate_acc = avg_val.get('gate_acc', 0)
        dir_acc = avg_val.get('dir_acc', 0)

        dir_acc_gated = avg_val.get('dir_acc_gated', 0)
        avg_pnl_g = avg_val.get('avg_pnl_gated', 0)
        avg_pnl_u = avg_val.get('avg_pnl_ungated', 0)

        print(f"Epoch {epoch:3d} | "
              f"train={avg_train.get('total', 0):.4f} | "
              f"val={avg_val.get('total', 0):.4f} | "
              f"call={avg_val.get('call_pnl', 0):.4f} put={avg_val.get('put_pnl', 0):.4f} | "
              f"gate={gate_acc:.3f} dir={dir_acc:.3f} dir_g={dir_acc_gated:.3f} | "
              f"pnl_g={avg_pnl_g:+.4f} pnl_u={avg_pnl_u:+.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

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
