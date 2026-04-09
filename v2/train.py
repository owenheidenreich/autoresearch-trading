"""ART² v2 Training Loop -- MFE Prediction Model

The model predicts Maximum Favorable Excursion (MFE) for BOTH call and put
directions. MFE = how far the option price goes in our favor within a forward
window, uncapped by any target. This teaches the model TRADE QUALITY:

  - Runner bars (MFE > 100%): wide target, long hold
  - Scalp bars (MFE 20-100%): tight target, short hold
  - Skip bars (MFE < 20%): gate says no trade

Trading decisions derive from MFE predictions at inference time:
  - gate = True if max(pred_call_mfe, pred_put_mfe) > threshold
  - direction = argmax(pred_call_mfe, pred_put_mfe)
  - risk params = conditioned on predicted MFE magnitude
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
MFE_W = float(os.environ.get("WEIGHT_MFE", 1.0))
RISK_W = float(os.environ.get("WEIGHT_RISK", 0.3))

# MFE computation
MFE_WINDOW = int(os.environ.get("MFE_WINDOW", 120))  # bars forward for MFE

# Gate scaling: MFE predictions are in log1p space (0-4.4 range).
# We need aggressive gating to stay selective.
# gate_logit = (max_mfe - GATE_CENTER) * GATE_SCALE
# sigmoid(0) = 0.5, so GATE_CENTER = the MFE level where we're 50/50 on trading.
# log1p(0.5) = 0.405, so GATE_CENTER=0.6 means "trade when predicted MFE > ~82%"
GATE_CENTER = float(os.environ.get("GATE_CENTER", 0.6))
GATE_SCALE = float(os.environ.get("GATE_SCALE", 3.0))

# For replay compatibility
NUM_STRIKE_CLASSES = 13
STRIKE_OFFSETS = list(range(-30, 31, 5))
STRIKE_OFFSET_TO_IDX = {off: i for i, off in enumerate(STRIKE_OFFSETS)}

REGIME_DIM = 16


# ---------------------------------------------------------------------------
# MFE Computation
# ---------------------------------------------------------------------------

def compute_mfe_mae(prices: torch.Tensor, bar_of_day: torch.Tensor,
                    dates: list, window: int = MFE_WINDOW) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Maximum Favorable Excursion and Maximum Adverse Excursion.

    For each bar, looks forward `window` bars (same day only) and computes:
      MFE = max(future_prices) / entry_price - 1
      MAE = min(future_prices) / entry_price - 1
    """
    N = len(prices)
    mfe = torch.full((N,), float('nan'))
    mae = torch.full((N,), float('nan'))
    prices_np = prices.numpy() if isinstance(prices, torch.Tensor) else prices

    # Build day boundaries for efficient same-day checking
    day_end = {}
    for i, d in enumerate(dates):
        day_end[d] = i

    for i in range(N):
        px = prices_np[i]
        if px <= 0.5 or np.isnan(px):
            continue

        d = dates[i]
        end = min(i + window, day_end.get(d, i) + 1)
        if end <= i + 1:
            continue

        fwd = prices_np[i + 1:end]
        valid_mask = ~np.isnan(fwd) & (fwd > 0)
        if valid_mask.sum() < 3:
            continue

        fwd_valid = fwd[valid_mask]
        mfe[i] = float((fwd_valid.max() / px) - 1.0)
        mae[i] = float((fwd_valid.min() / px) - 1.0)

    return mfe, mae


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
    """MFE prediction model with trade-quality-aware gating.

    Predicts Maximum Favorable Excursion (uncapped upside potential) for
    both call and put directions. Uses aggressive gate scaling to stay
    selective: only trades bars with high predicted MFE.

    Outputs use 'call_pnl'/'put_pnl' keys for replay compatibility,
    but they actually represent MFE predictions in log1p space.
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

        # FiLM layers
        self.film_call = FiLMLayer(REGIME_DIM, d)
        self.film_put = FiLMLayer(REGIME_DIM, d)
        self.film_risk = FiLMLayer(REGIME_DIM, d)

        # MFE prediction heads
        self.call_pnl_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, 1),
        )
        self.put_pnl_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(d // 2, 1),
        )
        # Risk head: sees backbone + detached MFE predictions
        self.risk_head = nn.Sequential(
            nn.Linear(d + 2, d // 2), nn.GELU(), nn.Dropout(dr),
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

        call_mfe = self.call_pnl_head(self.film_call(regime, last))  # (B, 1)
        put_mfe = self.put_pnl_head(self.film_put(regime, last))    # (B, 1)

        # Risk head sees backbone + MFE predictions
        mfe_context = torch.cat([call_mfe.detach(), put_mfe.detach()], dim=-1)
        risk_input = torch.cat([self.film_risk(regime, last), mfe_context], dim=-1)
        risk = self.risk_head(risk_input)

        # Derive gate/direction for replay compatibility
        call_v = call_mfe.squeeze(-1)
        put_v = put_mfe.squeeze(-1)

        # Gate: centered and scaled so sigmoid threshold is meaningful
        # GATE_CENTER=0.6 -> trade when predicted log1p(MFE) > 0.6 -> MFE > 82%
        # GATE_SCALE=3.0 -> sharp sigmoid transition
        max_mfe = torch.max(call_v, put_v)
        gate_logit = (max_mfe - GATE_CENTER) * GATE_SCALE

        direction = torch.stack([call_v, put_v], dim=-1)

        strike = torch.zeros(B, NUM_STRIKE_CLASSES, device=x.device)
        strike[:, NUM_STRIKE_CLASSES // 2] = 1.0

        confidence = torch.abs(call_v - put_v).unsqueeze(-1) * 3.0

        return {
            'call_pnl': call_mfe,
            'put_pnl': put_mfe,
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
    """MFE regression loss with runner weighting and direction asymmetry."""
    device = outputs['call_pnl'].device

    mfe_call = targets['mfe_call'].float().to(device)
    mfe_put = targets['mfe_put'].float().to(device)
    mae_call = targets['mae_call'].float().to(device)
    mae_put = targets['mae_put'].float().to(device)
    lab_direction = targets['label_direction'].long().to(device)

    valid = ~torch.isnan(mfe_call) & ~torch.isnan(mfe_put) & (lab_direction >= 0)

    if not valid.any():
        zero = torch.tensor(0.0, device=device)
        return zero, {'call_mfe': 0.0, 'put_mfe': 0.0, 'total': 0.0}

    # MFE regression in log1p space
    pred_call = outputs['call_pnl'].squeeze(-1)[valid]
    pred_put = outputs['put_pnl'].squeeze(-1)[valid]
    true_call = torch.log1p(torch.clamp(mfe_call[valid], min=0))
    true_put = torch.log1p(torch.clamp(mfe_put[valid], min=0))

    # Direction-asymmetric: puts 6x penalty for over-prediction
    call_err = pred_call - true_call
    put_err = pred_put - true_put
    call_weight = torch.where(
        (call_err > 0) & (true_call < 0.2), 4.0, 1.0
    )
    put_weight = torch.where(
        (put_err > 0) & (true_put < 0.2), 6.0, 1.0
    )

    # Runner-weighted sampling
    best_mfe = torch.max(mfe_call[valid], mfe_put[valid])
    best_mfe_clamped = torch.clamp(best_mfe, 0, 5.0)
    sample_weight = 1.0 + 3.0 * best_mfe_clamped

    call_loss = (sample_weight * call_weight * F.huber_loss(
        pred_call, true_call, delta=1.0, reduction='none')).mean()
    put_loss = (sample_weight * put_weight * F.huber_loss(
        pred_put, true_put, delta=1.0, reduction='none')).mean()

    mfe_loss = call_loss + put_loss

    # Risk loss: MFE-derived targets
    from v2.core.policy import DEFAULT_POLICY
    _hold_hi = DEFAULT_POLICY.max_hold_range[1]

    risk_out = outputs['risk'][valid]
    best_mfe_v = torch.max(mfe_call[valid], mfe_put[valid])
    best_mae_v = torch.min(mae_call[valid], mae_put[valid])

    t_stop = torch.clamp(best_mae_v.abs() * 0.5, 0.10, 0.50)
    t_target = torch.clamp(best_mfe_v * 0.7, 0.15, 5.0)
    t_hold = torch.clamp(best_mfe_v * 80 + 30, 30, 350) / _hold_hi

    is_put = (lab_direction[valid] == 1).float()
    t_hold = t_hold * (1.0 - 0.25 * is_put)

    risk_target = torch.stack([t_stop, t_target, t_hold], dim=-1)
    risk_loss = F.huber_loss(risk_out, risk_target, delta=0.5)

    total = MFE_W * mfe_loss + RISK_W * risk_loss

    # Metrics
    with torch.no_grad():
        pred_dir = (pred_put > pred_call).long()
        true_dir = lab_direction[valid]
        dir_valid = (true_dir >= 0) & (true_dir <= 1)
        dir_acc = (pred_dir[dir_valid] == true_dir[dir_valid]).float().mean().item() if dir_valid.any() else 0.0

        lab_trade = targets['label_trade'].float().to(device)
        # Gate fires when max_mfe > GATE_CENTER (matches inference logic)
        max_pred = torch.max(pred_call, pred_put)
        pred_trade = (max_pred > GATE_CENTER).float()
        true_trade = lab_trade[valid]
        gate_acc = (pred_trade == true_trade).float().mean().item()

        pred_gated = pred_trade.bool()
        avg_pred_gated = max_pred[pred_gated].mean().item() if pred_gated.any() else 0.0
        avg_true_gated = torch.log1p(torch.clamp(best_mfe_v[pred_gated], min=0)).mean().item() if pred_gated.any() else 0.0

        is_runner = best_mfe_v > 1.0
        runner_recall = 0.0
        if is_runner.any() and pred_gated.any():
            runner_recall = (pred_gated & is_runner).float().sum().item() / max(is_runner.float().sum().item(), 1)

        gate_rate = pred_gated.float().mean().item()
        avg_target = risk_out[:, 1].mean().item()
        avg_hold = risk_out[:, 2].mean().item() * _hold_hi

    loss_dict = {
        'call_mfe': call_loss.item(),
        'put_mfe': put_loss.item(),
        'risk': risk_loss.item(),
        'total': total.item(),
        'dir_acc': dir_acc,
        'gate_acc': gate_acc,
        'gate_rate': gate_rate,
        'avg_pred_gated': avg_pred_gated,
        'avg_true_gated': avg_true_gated,
        'runner_recall': runner_recall,
        'avg_target': avg_target,
        'avg_hold': avg_hold,
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

    seed = int(os.environ.get("TRAIN_SEED", SEED))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    print(f"Seed: {seed}")

    data = load_dataset(data_path)
    features = data['X']

    # --- Compute MFE/MAE labels from raw option prices ---
    print(f"Computing MFE/MAE labels (window={MFE_WINDOW} bars)...")
    t_mfe_start = time.time()

    mfe_call, mae_call = compute_mfe_mae(
        data['atm_call_prices'], data['bar_of_day'], data['dates'], window=MFE_WINDOW)
    mfe_put, mae_put = compute_mfe_mae(
        data['atm_put_prices'], data['bar_of_day'], data['dates'], window=MFE_WINDOW)

    valid_mfe = ~torch.isnan(mfe_call) & ~torch.isnan(mfe_put)
    n_valid = valid_mfe.sum().item()
    if n_valid > 0:
        best_mfe = torch.where(valid_mfe, torch.max(mfe_call, mfe_put), torch.tensor(float('nan')))
        best_valid = best_mfe[~torch.isnan(best_mfe)]
        runners = (best_valid > 1.0).sum().item()
        print(f"MFE computed: {n_valid:,} valid bars, "
              f"mean={best_valid.mean():.3f}, runners(>100%): {runners:,} ({100*runners/len(best_valid):.1f}%), "
              f"max={best_valid.max():.1f}x, took {time.time()-t_mfe_start:.1f}s")

    labels = {
        'label_trade': data['label_trade'],
        'label_direction': data['label_direction'],
        'label_call_pnl': data['label_call_pnl'],
        'label_put_pnl': data['label_put_pnl'],
        'label_stop_pct': data['label_stop_pct'],
        'label_target_pct': data['label_target_pct'],
        'label_max_hold': data['label_max_hold'],
        'label_confidence': data['label_confidence'],
        'mfe_call': mfe_call,
        'mfe_put': mfe_put,
        'mae_call': mae_call,
        'mae_put': mae_put,
    }

    train_mask = train_mask_override if train_mask_override is not None else data['train_mask']
    val_mask = val_mask_override if val_mask_override is not None else data['val_mask']

    train_ds = TradeDataset(features, labels, train_mask, lookback=LOOKBACK)
    val_ds = TradeDataset(features, labels, val_mask, lookback=LOOKBACK)

    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    print(f"Gate: center={GATE_CENTER}, scale={GATE_SCALE} -> trade when MFE > {(math.exp(GATE_CENTER)-1)*100:.0f}%")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} params, device={device}")

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
        gate_rate = avg_val.get('gate_rate', 0)
        runner_recall = avg_val.get('runner_recall', 0)
        avg_target = avg_val.get('avg_target', 0)
        avg_hold = avg_val.get('avg_hold', 0)

        print(f"Epoch {epoch:3d} | "
              f"train={avg_train.get('total', 0):.4f} | "
              f"val={avg_val.get('total', 0):.4f} | "
              f"call={avg_val.get('call_mfe', 0):.4f} put={avg_val.get('put_mfe', 0):.4f} | "
              f"gate={gate_acc:.3f} dir={dir_acc:.3f} grate={gate_rate:.3f} runner={runner_recall:.3f} | "
              f"tgt={avg_target:.2f} hold={avg_hold:.0f} | "
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
