from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _safe_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


@dataclass
class SurfaceStandardizer:
    scalar_mean: np.ndarray
    scalar_std: np.ndarray
    seq_mean: np.ndarray
    seq_std: np.ndarray
    contract_mean: np.ndarray
    contract_std: np.ndarray

    @classmethod
    def fit(
        cls,
        scalar: np.ndarray,
        seq: np.ndarray,
        seq_mask: np.ndarray,
        call_contracts: np.ndarray,
        call_mask: np.ndarray,
        put_contracts: np.ndarray,
        put_mask: np.ndarray,
    ) -> "SurfaceStandardizer":
        scalar_mean, scalar_std = _safe_mean_std(scalar)

        seq_flat = seq[seq_mask.astype(bool)]
        if len(seq_flat) == 0:
            seq_mean = np.zeros(seq.shape[-1], dtype=np.float32)
            seq_std = np.ones(seq.shape[-1], dtype=np.float32)
        else:
            seq_mean, seq_std = _safe_mean_std(seq_flat)

        contract_flat = np.concatenate(
            [
                call_contracts[call_mask.astype(bool)],
                put_contracts[put_mask.astype(bool)],
            ],
            axis=0,
        )
        if len(contract_flat) == 0:
            contract_mean = np.zeros(call_contracts.shape[-1], dtype=np.float32)
            contract_std = np.ones(call_contracts.shape[-1], dtype=np.float32)
        else:
            contract_mean, contract_std = _safe_mean_std(contract_flat)

        return cls(
            scalar_mean=scalar_mean,
            scalar_std=scalar_std,
            seq_mean=seq_mean,
            seq_std=seq_std,
            contract_mean=contract_mean,
            contract_std=contract_std,
        )

    def transform(
        self,
        scalar: np.ndarray,
        seq: np.ndarray,
        seq_mask: np.ndarray,
        call_contracts: np.ndarray,
        call_mask: np.ndarray,
        put_contracts: np.ndarray,
        put_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scalar_n = ((scalar.astype(np.float32) - self.scalar_mean) / self.scalar_std).astype(np.float32, copy=False)
        scalar_n = np.nan_to_num(scalar_n, nan=0.0, posinf=0.0, neginf=0.0)

        seq_n = ((seq.astype(np.float32) - self.seq_mean) / self.seq_std).astype(np.float32, copy=False)
        seq_n = np.nan_to_num(seq_n, nan=0.0, posinf=0.0, neginf=0.0)
        seq_n *= seq_mask[..., None].astype(np.float32)

        call_n = ((call_contracts.astype(np.float32) - self.contract_mean) / self.contract_std).astype(np.float32, copy=False)
        put_n = ((put_contracts.astype(np.float32) - self.contract_mean) / self.contract_std).astype(np.float32, copy=False)
        call_n = np.nan_to_num(call_n, nan=0.0, posinf=0.0, neginf=0.0)
        put_n = np.nan_to_num(put_n, nan=0.0, posinf=0.0, neginf=0.0)
        call_n *= call_mask[..., None].astype(np.float32)
        put_n *= put_mask[..., None].astype(np.float32)
        return scalar_n, seq_n, call_n, put_n


class Layer2SurfaceSharedEncoder(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        seq_dim: int,
        contract_dim: int,
        hidden_dim: int = 128,
        seq_hidden_dim: int = 64,
        contract_hidden_dim: int = 48,
        depth: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.seq_gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=seq_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.contract_encoder = nn.Sequential(
            nn.Linear(contract_dim, contract_hidden_dim),
            nn.GELU(),
            nn.Linear(contract_hidden_dim, contract_hidden_dim),
            nn.GELU(),
        )

        trunk_in_dim = hidden_dim + seq_hidden_dim + contract_hidden_dim * 6
        trunk_layers: list[nn.Module] = []
        dim = trunk_in_dim
        for _ in range(max(depth, 1)):
            trunk_layers.append(nn.Linear(dim, hidden_dim))
            trunk_layers.append(nn.GELU())
            trunk_layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.entry_head = nn.Linear(hidden_dim, 1)
        self.side_head = nn.Linear(hidden_dim, 1)

    def _encode_sequence(self, seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        lengths = seq_mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.seq_gru(packed)
        return hidden[-1]

    def _encode_contracts(self, contracts: torch.Tensor, contract_mask: torch.Tensor) -> torch.Tensor:
        token_h = self.contract_encoder(contracts)
        mask = contract_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = (token_h * mask).sum(dim=1) / denom
        neg_inf = torch.full_like(token_h, -1e9)
        max_pool = torch.where(mask > 0.0, token_h, neg_inf).max(dim=1).values
        has_any = contract_mask.sum(dim=1, keepdim=True) > 0.0
        max_pool = torch.where(has_any, max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(
        self,
        scalar: torch.Tensor,
        seq: torch.Tensor,
        seq_mask: torch.Tensor,
        call_contracts: torch.Tensor,
        call_mask: torch.Tensor,
        put_contracts: torch.Tensor,
        put_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scalar_h = self.scalar_proj(scalar)
        seq_h = self._encode_sequence(seq, seq_mask)
        call_h = self._encode_contracts(call_contracts, call_mask)
        put_h = self._encode_contracts(put_contracts, put_mask)
        joint = torch.cat([scalar_h, seq_h, call_h, put_h, call_h - put_h], dim=-1)
        hidden = self.trunk(joint)
        return self.entry_head(hidden).squeeze(-1), self.side_head(hidden).squeeze(-1)


@dataclass
class SurfaceSharedEncoderPredictor:
    state_dict: dict
    standardizer: SurfaceStandardizer
    scalar_dim: int
    seq_dim: int
    contract_dim: int
    hidden_dim: int
    seq_hidden_dim: int
    contract_hidden_dim: int
    depth: int
    dropout: float
    head: str
    device_hint: str = "cpu"

    def _build_model(self, device: str) -> Layer2SurfaceSharedEncoder:
        model = Layer2SurfaceSharedEncoder(
            scalar_dim=self.scalar_dim,
            seq_dim=self.seq_dim,
            contract_dim=self.contract_dim,
            hidden_dim=self.hidden_dim,
            seq_hidden_dim=self.seq_hidden_dim,
            contract_hidden_dim=self.contract_hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
        )
        model.load_state_dict(self.state_dict)
        model.to(device)
        model.eval()
        return model

    def predict(
        self,
        scalar: np.ndarray,
        seq: np.ndarray,
        seq_mask: np.ndarray,
        call_contracts: np.ndarray,
        call_mask: np.ndarray,
        put_contracts: np.ndarray,
        put_mask: np.ndarray,
        batch_size: int = 2048,
    ) -> np.ndarray:
        scalar_n, seq_n, call_n, put_n = self.standardizer.transform(
            scalar,
            seq,
            seq_mask,
            call_contracts,
            call_mask,
            put_contracts,
            put_mask,
        )
        device = "cuda" if torch.cuda.is_available() and self.device_hint == "cuda" else "cpu"
        model = self._build_model(device=device)
        out = np.zeros(len(scalar_n), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(scalar_n), batch_size):
                end = start + batch_size
                scalar_b = torch.from_numpy(scalar_n[start:end]).to(device)
                seq_b = torch.from_numpy(seq_n[start:end]).to(device)
                seq_mask_b = torch.from_numpy(seq_mask[start:end].astype(np.float32)).to(device)
                call_b = torch.from_numpy(call_n[start:end]).to(device)
                call_mask_b = torch.from_numpy(call_mask[start:end].astype(np.float32)).to(device)
                put_b = torch.from_numpy(put_n[start:end]).to(device)
                put_mask_b = torch.from_numpy(put_mask[start:end].astype(np.float32)).to(device)
                entry_pred, side_pred = model(
                    scalar_b,
                    seq_b,
                    seq_mask_b,
                    call_b,
                    call_mask_b,
                    put_b,
                    put_mask_b,
                )
                pred = entry_pred if self.head == "entry" else side_pred
                out[start:end] = pred.detach().cpu().numpy().astype(np.float32, copy=False)
        return out


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


def _multitask_huber(
    entry_pred: torch.Tensor,
    side_pred: torch.Tensor,
    entry_target: torch.Tensor,
    side_target: torch.Tensor,
    entry_weight: torch.Tensor,
    side_weight: torch.Tensor,
    w_entry_loss: float,
    w_side_loss: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entry_loss = weighted_huber_loss(entry_pred, entry_target, entry_weight)
    side_loss = weighted_huber_loss(side_pred, side_target, side_weight)
    total = w_entry_loss * entry_loss + w_side_loss * side_loss
    return total, entry_loss, side_loss


def train_surface_multitask(
    *,
    scalar_train: np.ndarray,
    seq_train: np.ndarray,
    seq_mask_train: np.ndarray,
    call_train: np.ndarray,
    call_mask_train: np.ndarray,
    put_train: np.ndarray,
    put_mask_train: np.ndarray,
    y_entry_train: np.ndarray,
    y_side_train: np.ndarray,
    entry_mask_train: np.ndarray,
    side_mask_train: np.ndarray,
    entry_weight_train: np.ndarray,
    side_weight_train: np.ndarray,
    scalar_val: np.ndarray,
    seq_val: np.ndarray,
    seq_mask_val: np.ndarray,
    call_val: np.ndarray,
    call_mask_val: np.ndarray,
    put_val: np.ndarray,
    put_mask_val: np.ndarray,
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
    seq_hidden_dim: int,
    contract_hidden_dim: int,
    depth: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[SurfaceSharedEncoderPredictor, SurfaceSharedEncoderPredictor, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    standardizer = SurfaceStandardizer.fit(
        scalar_train,
        seq_train,
        seq_mask_train,
        call_train,
        call_mask_train,
        put_train,
        put_mask_train,
    )
    scalar_train_n, seq_train_n, call_train_n, put_train_n = standardizer.transform(
        scalar_train,
        seq_train,
        seq_mask_train,
        call_train,
        call_mask_train,
        put_train,
        put_mask_train,
    )
    scalar_val_n, seq_val_n, call_val_n, put_val_n = standardizer.transform(
        scalar_val,
        seq_val,
        seq_mask_val,
        call_val,
        call_mask_val,
        put_val,
        put_mask_val,
    )

    wE_train = entry_weight_train.astype(np.float32) * entry_mask_train.astype(np.float32)
    wS_train = side_weight_train.astype(np.float32) * side_mask_train.astype(np.float32)
    wE_val = entry_weight_val.astype(np.float32) * entry_mask_val.astype(np.float32)
    wS_val = side_weight_val.astype(np.float32) * side_mask_val.astype(np.float32)
    yE_train = np.where(entry_mask_train, y_entry_train, 0.0).astype(np.float32)
    yS_train = np.where(side_mask_train, y_side_train, 0.0).astype(np.float32)
    yE_val = np.where(entry_mask_val, y_entry_val, 0.0).astype(np.float32)
    yS_val = np.where(side_mask_val, y_side_val, 0.0).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(scalar_train_n),
        torch.from_numpy(seq_train_n),
        torch.from_numpy(seq_mask_train.astype(np.float32)),
        torch.from_numpy(call_train_n),
        torch.from_numpy(call_mask_train.astype(np.float32)),
        torch.from_numpy(put_train_n),
        torch.from_numpy(put_mask_train.astype(np.float32)),
        torch.from_numpy(yE_train),
        torch.from_numpy(yS_train),
        torch.from_numpy(wE_train),
        torch.from_numpy(wS_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(scalar_val_n),
        torch.from_numpy(seq_val_n),
        torch.from_numpy(seq_mask_val.astype(np.float32)),
        torch.from_numpy(call_val_n),
        torch.from_numpy(call_mask_val.astype(np.float32)),
        torch.from_numpy(put_val_n),
        torch.from_numpy(put_mask_val.astype(np.float32)),
        torch.from_numpy(yE_val),
        torch.from_numpy(yS_val),
        torch.from_numpy(wE_val),
        torch.from_numpy(wS_val),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = Layer2SurfaceSharedEncoder(
        scalar_dim=scalar_train_n.shape[1],
        seq_dim=seq_train_n.shape[2],
        contract_dim=call_train_n.shape[2],
        hidden_dim=hidden_dim,
        seq_hidden_dim=seq_hidden_dim,
        contract_hidden_dim=contract_hidden_dim,
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
        for scalar_b, seq_b, seq_mask_b, call_b, call_mask_b, put_b, put_mask_b, yeb, ysb, web, wsb in train_loader:
            scalar_b = scalar_b.to(device)
            seq_b = seq_b.to(device)
            seq_mask_b = seq_mask_b.to(device)
            call_b = call_b.to(device)
            call_mask_b = call_mask_b.to(device)
            put_b = put_b.to(device)
            put_mask_b = put_mask_b.to(device)
            yeb = yeb.to(device)
            ysb = ysb.to(device)
            web = web.to(device)
            wsb = wsb.to(device)

            optimizer.zero_grad()
            entry_pred, side_pred = model(
                scalar_b,
                seq_b,
                seq_mask_b,
                call_b,
                call_mask_b,
                put_b,
                put_mask_b,
            )
            total, _, _ = _multitask_huber(
                entry_pred,
                side_pred,
                yeb,
                ysb,
                web,
                wsb,
                w_entry_loss,
                w_side_loss,
            )
            total.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_total = []
        val_entry = []
        val_side = []
        with torch.inference_mode():
            for scalar_b, seq_b, seq_mask_b, call_b, call_mask_b, put_b, put_mask_b, yeb, ysb, web, wsb in val_loader:
                scalar_b = scalar_b.to(device)
                seq_b = seq_b.to(device)
                seq_mask_b = seq_mask_b.to(device)
                call_b = call_b.to(device)
                call_mask_b = call_mask_b.to(device)
                put_b = put_b.to(device)
                put_mask_b = put_mask_b.to(device)
                yeb = yeb.to(device)
                ysb = ysb.to(device)
                web = web.to(device)
                wsb = wsb.to(device)

                entry_pred, side_pred = model(
                    scalar_b,
                    seq_b,
                    seq_mask_b,
                    call_b,
                    call_mask_b,
                    put_b,
                    put_mask_b,
                )
                total, entry_loss, side_loss = _multitask_huber(
                    entry_pred,
                    side_pred,
                    yeb,
                    ysb,
                    web,
                    wsb,
                    w_entry_loss,
                    w_side_loss,
                )
                val_total.append(total.item())
                val_entry.append(entry_loss.item())
                val_side.append(side_loss.item())

        mean_val = float(np.mean(val_total)) if val_total else float("inf")
        mean_entry = float(np.mean(val_entry)) if val_entry else float("inf")
        mean_side = float(np.mean(val_side)) if val_side else float("inf")
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

    common = {
        "state_dict": best_state,
        "standardizer": standardizer,
        "scalar_dim": scalar_train_n.shape[1],
        "seq_dim": seq_train_n.shape[2],
        "contract_dim": call_train_n.shape[2],
        "hidden_dim": hidden_dim,
        "seq_hidden_dim": seq_hidden_dim,
        "contract_hidden_dim": contract_hidden_dim,
        "depth": depth,
        "dropout": dropout,
        "device_hint": "cuda" if device == "cuda" else "cpu",
    }
    entry_predictor = SurfaceSharedEncoderPredictor(head="entry", **common)
    side_predictor = SurfaceSharedEncoderPredictor(head="side", **common)
    info = {
        "best_val_loss": float(best_val),
        "best_entry_val_loss": float(best_entry_val),
        "best_side_val_loss": float(best_side_val),
        "best_epoch": float(best_epoch),
    }
    return entry_predictor, side_predictor, info

