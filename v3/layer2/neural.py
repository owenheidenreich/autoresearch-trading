from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Layer2MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.10):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Layer2SharedEncoder(nn.Module):
    """Shared trunk + two regression heads.

    Replaces the two independent Layer2MLP instances with a single
    bar-state encoder that both entry and side heads must use. The
    hypothesis: trees implicitly share splits across outputs; separate
    MLPs relearn that structure per-head and overfit the smaller
    per-head signal. Forcing a shared representation should recover
    what the tree's implicit sharing provides.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        trunk_layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(depth, 1)):
            trunk_layers.append(nn.Linear(dim, hidden_dim))
            trunk_layers.append(nn.GELU())
            trunk_layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.entry_head = nn.Linear(hidden_dim, 1)
        self.side_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        entry = self.entry_head(h).squeeze(-1)
        side = self.side_head(h).squeeze(-1)
        return entry, side


@dataclass
class TorchTabularPredictor:
    """Pickle-friendly wrapper so replay.py can call .predict() like sklearn."""

    state_dict: dict
    mean: np.ndarray
    std: np.ndarray
    in_dim: int
    hidden_dim: int
    depth: int
    dropout: float
    device_hint: str = "cpu"

    def _build_model(self, device: str = "cpu") -> Layer2MLP:
        model = Layer2MLP(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
        )
        model.load_state_dict(self.state_dict)
        model.to(device)
        model.eval()
        return model

    def predict(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape={X.shape}")
        Xn = ((X.astype(np.float32) - self.mean) / self.std).astype(np.float32, copy=False)
        device = "cuda" if torch.cuda.is_available() and self.device_hint == "cuda" else "cpu"
        model = self._build_model(device=device)
        out = np.zeros(Xn.shape[0], dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(Xn), batch_size):
                xb = torch.from_numpy(Xn[start:start + batch_size]).to(device)
                pred = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
                out[start:start + len(pred)] = pred
        return out


def make_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0).astype(np.float32)
    std = np.nanstd(X, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    return mean, std


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xn = (X.astype(np.float32) - mean) / std
    return np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_t = torch.from_numpy(X.astype(np.float32, copy=False))
    y_t = torch.from_numpy(y.astype(np.float32, copy=False))
    if weights is None:
        w_t = torch.ones_like(y_t)
    else:
        w_t = torch.from_numpy(weights.astype(np.float32, copy=False))
    ds = TensorDataset(X_t, y_t, w_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    err = pred - target
    abs_err = torch.abs(err)
    quadratic = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    denom = torch.clamp(weight.sum(), min=1.0)
    return (loss * weight).sum() / denom


def train_regressor(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weight: Optional[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_weight: Optional[np.ndarray],
    device: str,
    seed: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[TorchTabularPredictor, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mean, std = make_standardizer(X_train)
    X_train_n = standardize(X_train, mean, std)
    X_val_n = standardize(X_val, mean, std)

    model = Layer2MLP(
        in_dim=X_train_n.shape[1],
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_loader = make_loader(X_train_n, y_train, train_weight, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val_n, y_val, val_weight, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = weighted_huber_loss(pred, yb, wb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.inference_mode():
            for xb, yb, wb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                pred = model(xb)
                val_losses.append(weighted_huber_loss(pred, yb, wb).item())
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val - 1e-6:
            best_val = mean_val
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    predictor = TorchTabularPredictor(
        state_dict=best_state,
        mean=mean,
        std=std,
        in_dim=X_train_n.shape[1],
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        device_hint="cuda" if device == "cuda" else "cpu",
    )
    info = {
        "best_val_loss": float(best_val),
        "best_epoch": float(best_epoch),
    }
    return predictor, info


@dataclass
class SharedEncoderPredictor:
    """Pickle-friendly wrapper exposing .predict(X) for ONE head of a shared encoder.

    Two instances (head='entry', head='side') are saved per fold so replay.py
    can load them with the same interface as the separate-head MLP or the
    HistGBM tree baseline.
    """

    state_dict: dict
    mean: np.ndarray
    std: np.ndarray
    in_dim: int
    hidden_dim: int
    depth: int
    dropout: float
    head: str  # "entry" or "side"
    device_hint: str = "cpu"

    def _build_model(self, device: str = "cpu") -> Layer2SharedEncoder:
        model = Layer2SharedEncoder(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
        )
        model.load_state_dict(self.state_dict)
        model.to(device)
        model.eval()
        return model

    def predict(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape={X.shape}")
        Xn = ((X.astype(np.float32) - self.mean) / self.std).astype(np.float32, copy=False)
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        device = "cuda" if torch.cuda.is_available() and self.device_hint == "cuda" else "cpu"
        model = self._build_model(device=device)
        out = np.zeros(Xn.shape[0], dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(Xn), batch_size):
                xb = torch.from_numpy(Xn[start:start + batch_size]).to(device)
                entry_pred, side_pred = model(xb)
                pred = entry_pred if self.head == "entry" else side_pred
                arr = pred.detach().cpu().numpy().astype(np.float32, copy=False)
                out[start:start + len(arr)] = arr
        return out


def _multitask_huber(
    entry_pred: torch.Tensor,
    side_pred: torch.Tensor,
    y_entry: torch.Tensor,
    y_side: torch.Tensor,
    w_entry: torch.Tensor,
    w_side: torch.Tensor,
    w_entry_loss: float,
    w_side_loss: float,
    delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted Huber on each head, summed with task-level weights.

    Samples with per-head weight 0 contribute nothing to that head's loss;
    weighted_huber_loss_internal clamps the denominator so all-zero weights
    yield loss 0 instead of NaN.
    """
    entry_loss = _weighted_huber(entry_pred, y_entry, w_entry, delta=delta)
    side_loss = _weighted_huber(side_pred, y_side, w_side, delta=delta)
    total = w_entry_loss * entry_loss + w_side_loss * side_loss
    return total, entry_loss.detach(), side_loss.detach()


def _weighted_huber(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    err = pred - target
    abs_err = torch.abs(err)
    quadratic = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    denom = torch.clamp(weight.sum(), min=1.0)
    return (loss * weight).sum() / denom


def train_multitask(
    *,
    X_train: np.ndarray,
    y_entry_train: np.ndarray,
    y_side_train: np.ndarray,
    entry_mask_train: np.ndarray,
    side_mask_train: np.ndarray,
    entry_weight_train: np.ndarray,
    side_weight_train: np.ndarray,
    X_val: np.ndarray,
    y_entry_val: np.ndarray,
    y_side_val: np.ndarray,
    entry_mask_val: np.ndarray,
    side_mask_val: np.ndarray,
    entry_weight_val: np.ndarray,
    side_weight_val: np.ndarray,
    w_entry_loss: float,
    w_side_loss: float,
    device: str,
    seed: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[SharedEncoderPredictor, SharedEncoderPredictor, dict[str, float]]:
    """Train a Layer2SharedEncoder on entry + side targets simultaneously.

    Input arrays are ALIGNED on the SAME row order; per-head validity is
    expressed via boolean masks. Samples invalid for a head contribute
    zero weight to that head's loss.

    Returns (entry_predictor, side_predictor, info). Both predictors share
    the same encoder state; only their `head` field differs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mean, std = make_standardizer(X_train)
    X_train_n = standardize(X_train, mean, std)
    X_val_n = standardize(X_val, mean, std)

    # Per-sample per-head effective weight: 0 when target is invalid.
    wE_train = entry_weight_train.astype(np.float32) * entry_mask_train.astype(np.float32)
    wS_train = side_weight_train.astype(np.float32) * side_mask_train.astype(np.float32)
    wE_val = entry_weight_val.astype(np.float32) * entry_mask_val.astype(np.float32)
    wS_val = side_weight_val.astype(np.float32) * side_mask_val.astype(np.float32)

    # Targets: replace invalid with 0 (masked out by zero weight above).
    yE_train = np.where(entry_mask_train, y_entry_train, 0.0).astype(np.float32)
    yS_train = np.where(side_mask_train, y_side_train, 0.0).astype(np.float32)
    yE_val = np.where(entry_mask_val, y_entry_val, 0.0).astype(np.float32)
    yS_val = np.where(side_mask_val, y_side_val, 0.0).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(yE_train),
        torch.from_numpy(yS_train),
        torch.from_numpy(wE_train),
        torch.from_numpy(wS_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_n),
        torch.from_numpy(yE_val),
        torch.from_numpy(yS_val),
        torch.from_numpy(wE_val),
        torch.from_numpy(wS_val),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = Layer2SharedEncoder(
        in_dim=X_train_n.shape[1],
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_state = None
    best_val = float("inf")
    best_entry_val = float("inf")
    best_side_val = float("inf")
    best_epoch = -1
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yeb, ysb, web, wsb in train_loader:
            xb = xb.to(device)
            yeb = yeb.to(device)
            ysb = ysb.to(device)
            web = web.to(device)
            wsb = wsb.to(device)
            optimizer.zero_grad()
            ep, sp = model(xb)
            total, _, _ = _multitask_huber(
                ep, sp, yeb, ysb, web, wsb, w_entry_loss, w_side_loss
            )
            total.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        v_total = []
        v_entry = []
        v_side = []
        with torch.inference_mode():
            for xb, yeb, ysb, web, wsb in val_loader:
                xb = xb.to(device)
                yeb = yeb.to(device)
                ysb = ysb.to(device)
                web = web.to(device)
                wsb = wsb.to(device)
                ep, sp = model(xb)
                total, el, sl = _multitask_huber(
                    ep, sp, yeb, ysb, web, wsb, w_entry_loss, w_side_loss
                )
                v_total.append(total.item())
                v_entry.append(el.item())
                v_side.append(sl.item())
        mean_val = float(np.mean(v_total)) if v_total else float("inf")
        mean_entry = float(np.mean(v_entry)) if v_entry else float("inf")
        mean_side = float(np.mean(v_side)) if v_side else float("inf")
        if mean_val < best_val - 1e-6:
            best_val = mean_val
            best_entry_val = mean_entry
            best_side_val = mean_side
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    predictor_entry = SharedEncoderPredictor(
        state_dict=best_state,
        mean=mean,
        std=std,
        in_dim=X_train_n.shape[1],
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        head="entry",
        device_hint="cuda" if device == "cuda" else "cpu",
    )
    predictor_side = SharedEncoderPredictor(
        state_dict=best_state,
        mean=mean,
        std=std,
        in_dim=X_train_n.shape[1],
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        head="side",
        device_hint="cuda" if device == "cuda" else "cpu",
    )
    info = {
        "best_val_loss": float(best_val),
        "best_entry_val_loss": float(best_entry_val),
        "best_side_val_loss": float(best_side_val),
        "best_epoch": float(best_epoch),
    }
    return predictor_entry, predictor_side, info
