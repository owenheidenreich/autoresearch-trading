"""H3c — Causal TCN (Temporal Convolutional Network) oracle.

Replaces the L3 oracle's HistGradientBoostingClassifier (point-in-time scalar
features) with a causal sequence model that processes the full per-bar
trajectory of a trade.

Architecture: stack of causal Conv1d layers with exponentially expanding
dilations. Each layer uses LEFT-padding only, so output[t] depends strictly
on input[≤t]. Receptive field grows geometrically with depth.

Causality: verified by mutate-future-bars test in
v3/tests/test_tcn_causality.py.

Per-bar target: same as HGB oracle — `int(current_pnl >= suffix_max[i])`.
Per-bar BCE loss with a mask for variable-length trades.

Discipline anchor: 2026-04-25 oracle-gate label-leakage retraction.
The look-ahead audit is non-negotiable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D conv with LEFT-only padding so output[t] depends strictly on input[≤t].

    Standard pattern: F.pad(x, (left, 0)) then nn.Conv1d with padding=0.
    Output length equals input length.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = F.pad(x, (self.padding, 0))  # left-only padding
        return self.conv(x)


class TCNBlock(nn.Module):
    """Two causal convs + GELU activations + residual connection.

    No normalization (initial smoke version). Add CausalGroupNorm later if
    training is unstable.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = x
        out = F.gelu(self.conv1(x))
        out = F.gelu(self.conv2(out))
        return out + residual


class TCNOracle(nn.Module):
    """Causal TCN for per-bar exit-probability prediction.

    Input:  (batch, n_bars, n_features)
    Output: (batch, n_bars) — per-bar exit logit (sigmoid for prob)
    """

    def __init__(
        self,
        n_features: int = 99,
        hidden: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.n_features = n_features
        self.input_proj = nn.Conv1d(n_features, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden, kernel_size, dilation=2 ** i)
            for i in range(num_blocks)
        ])
        self.output_head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        # (B, 1, T) → (B, T)
        return self.output_head(x).squeeze(1)


@dataclass
class TCNTrainConfig:
    n_features: int = 99
    hidden: int = 64
    num_blocks: int = 4
    kernel_size: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 30
    early_stop_patience: int = 5
    max_bars: int = 200
    val_frac: float = 0.15
    device: str = "cpu"
    seed: int = 42


def trade_data_to_tensors(
    trade_data: list[dict[str, Any]],
    max_bars: int,
    n_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert list-of-trade-dicts (from _build_trade_data) into padded tensors.

    Each trade has a per_bar list with state_features + trade_state + target.
    We concatenate state + trade_state per bar (matches HGB feature ordering).

    Returns:
        x: (n_trades, max_bars, n_features) — padded with zeros
        y: (n_trades, max_bars) — target (0/1), padded with zeros
        mask: (n_trades, max_bars) — 1 for real bars, 0 for padding
    """
    n = len(trade_data)
    x = np.zeros((n, max_bars, n_features), dtype=np.float32)
    y = np.zeros((n, max_bars), dtype=np.float32)
    mask = np.zeros((n, max_bars), dtype=np.float32)
    for i, td in enumerate(trade_data):
        bars = td["per_bar"]
        n_b = min(len(bars), max_bars)
        for t in range(n_b):
            payload = bars[t]
            feats = np.concatenate([payload["state_features"], payload["trade_state"]])
            x[i, t, :feats.shape[0]] = feats
            y[i, t] = float(payload["target"])
            mask[i, t] = 1.0
    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(mask),
    )


def fit_standardizer(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-feature mean/std using only valid (mask==1) bars."""
    # x: (N, T, F); mask: (N, T)
    flat_x = x.view(-1, x.shape[-1])
    flat_m = mask.view(-1).bool()
    valid = flat_x[flat_m]
    mean = valid.mean(dim=0)
    std = valid.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def standardize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply z-score; zero out padded positions to keep them as 0."""
    out = (x - mean) / std
    return out * mask.unsqueeze(-1)


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-bar BCE, averaged only over valid bars."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def train_tcn(
    trade_data: list[dict[str, Any]],
    cfg: TCNTrainConfig,
    *,
    val_data: list[dict[str, Any]] | None = None,
) -> tuple[TCNOracle, dict[str, torch.Tensor], dict[str, list]]:
    """Train a TCN on the provided trade_data.

    If val_data is provided, use it for early stopping; otherwise carve out
    a random val_frac fraction of trade_data.

    Returns:
        model: trained TCNOracle (on cfg.device)
        standardizer: dict with 'mean' and 'std' tensors
        history: dict with per-epoch train/val loss
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = torch.device(cfg.device)
    x_full, y_full, m_full = trade_data_to_tensors(trade_data, cfg.max_bars, cfg.n_features)

    if val_data is None:
        # Random split
        n = len(trade_data)
        idx = rng.permutation(n)
        n_val = max(int(cfg.val_frac * n), 5)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        x_train, y_train, m_train = x_full[train_idx], y_full[train_idx], m_full[train_idx]
        x_val, y_val, m_val = x_full[val_idx], y_full[val_idx], m_full[val_idx]
    else:
        x_train, y_train, m_train = x_full, y_full, m_full
        x_val, y_val, m_val = trade_data_to_tensors(val_data, cfg.max_bars, cfg.n_features)

    # Fit standardizer on train only
    mean, std = fit_standardizer(x_train, m_train)
    x_train = standardize(x_train, mean, std, m_train)
    x_val = standardize(x_val, mean, std, m_val)

    model = TCNOracle(
        n_features=cfg.n_features,
        hidden=cfg.hidden,
        num_blocks=cfg.num_blocks,
        kernel_size=cfg.kernel_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    n_train = len(x_train)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state: dict | None = None
    patience = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_idx = rng.permutation(n_train)
        train_losses = []
        for i in range(0, n_train, cfg.batch_size):
            batch_idx = epoch_idx[i : i + cfg.batch_size]
            xb = x_train[batch_idx].to(device)
            yb = y_train[batch_idx].to(device)
            mb = m_train[batch_idx].to(device)
            logits = model(xb)
            loss = masked_bce_loss(logits, yb, mb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        model.eval()
        with torch.no_grad():
            xv = x_val.to(device)
            yv = y_val.to(device)
            mv = m_val.to(device)
            val_logits = model(xv)
            val_loss = float(masked_bce_loss(val_logits, yv, mv).item())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"mean": mean, "std": std}, history


def predict_per_bar(
    model: TCNOracle,
    trade_data: list[dict[str, Any]],
    standardizer: dict[str, torch.Tensor],
    cfg: TCNTrainConfig,
) -> list[np.ndarray]:
    """Run inference on trade_data; return per-trade per-bar exit probability arrays."""
    device = torch.device(cfg.device)
    x, _, m = trade_data_to_tensors(trade_data, cfg.max_bars, cfg.n_features)
    x = standardize(x, standardizer["mean"], standardizer["std"], m)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(trade_data), cfg.batch_size):
            xb = x[i : i + cfg.batch_size].to(device)
            mb = m[i : i + cfg.batch_size]
            logits = model(xb).cpu()
            probs = torch.sigmoid(logits)
            for j in range(len(probs)):
                n_bars = int(mb[j].sum().item())
                out.append(probs[j, :n_bars].numpy())
    return out
