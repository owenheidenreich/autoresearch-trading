"""ART² v4 Training Loop -- hierarchical: gate → direction → strike selection."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from v2.core.chain_data import NUM_CONTRACT_FEATURES, padded_snapshot
from v2.core.metrics import score_config_fingerprint


NUM_FEATURES = int(os.environ.get("NUM_FEATURES", 47))
LOOKBACK = int(os.environ.get("TRAIN_LOOKBACK", 30))
D_MODEL = int(os.environ.get("TRAIN_D_MODEL", 96))
N_HEADS = 4
DEPTH = int(os.environ.get("TRAIN_DEPTH", 3))
DROPOUT = float(os.environ.get("TRAIN_DROPOUT", 0.05))
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 1024))
LR = float(os.environ.get("TRAIN_LR", 3e-4))
WEIGHT_DECAY = float(os.environ.get("TRAIN_WEIGHT_DECAY", 0.03))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 24))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 300))
SEL_W = float(os.environ.get("WEIGHT_SEL", 1.0))
GATE_W = float(os.environ.get("WEIGHT_GATE", 1.0))
DIR_W = float(os.environ.get("WEIGHT_DIR", 1.0))
SEED = int(os.environ.get("TRAIN_SEED", 123))
SOFT_TEMP = float(os.environ.get("SOFT_TEMP", 0.20))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TradingModel(nn.Module):
    """Hierarchical trading model: gate → direction → strike selection."""

    def __init__(self, d_model: int | None = None, depth: int | None = None, n_heads: int | None = None, dropout: float | None = None):
        super().__init__()
        d = d_model or D_MODEL
        dep = depth or DEPTH
        nh = n_heads or N_HEADS
        dr = DROPOUT if dropout is None else dropout

        # Shared context encoder
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
        self.register_buffer("causal_mask", nn.Transformer.generate_square_subsequent_mask(LOOKBACK))

        # Contract embedding
        self.contract_proj = nn.Sequential(
            nn.Linear(NUM_CONTRACT_FEATURES, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Head 1: Gate (trade / no-trade) from context
        self.gate_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(d // 2, 1),
        )

        # Head 2: Direction (call=0 / put=1) from context
        # Uses detached context to prevent direction overfitting from corrupting encoder
        self.direction_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d // 4, 1),
        )

        # Head 3: Strike selection (context + contract → score)
        self.strike_head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, x: torch.Tensor, contracts: torch.Tensor) -> dict[str, torch.Tensor]:
        _, seq_len, _ = x.shape
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.pos_enc(h)
        mask = self.causal_mask[:seq_len, :seq_len] if seq_len <= self.causal_mask.size(0) else None
        h = self.encoder(h, mask=mask)
        context = h[:, -1, :]

        # Gate from context (gradient flows to encoder)
        gate_logit = self.gate_head(context).squeeze(-1)
        # Direction from detached context (prevents overfitting from corrupting encoder)
        direction_logit = self.direction_head(context.detach()).squeeze(-1)  # >0 = put

        # Strike scores from context + contract features
        contract_emb = self.contract_proj(contracts)
        context_exp = context.unsqueeze(1).expand(-1, contract_emb.size(1), -1)
        combined = torch.cat([context_exp, contract_emb], dim=-1)
        contract_scores = self.strike_head(combined).squeeze(-1)

        valid_mask = contracts[:, :, 0] > 0.5
        return {
            "gate_logit": gate_logit,
            "direction_logit": direction_logit,
            "contract_scores": contract_scores,
            "valid_mask": valid_mask,
        }


class TradeDataset(Dataset):
    def __init__(self, data: dict, mask: torch.Tensor, lookback: int = LOOKBACK):
        self.features = data["X"]
        self.lookback = lookback
        dates = data["dates"]
        bar_of_day = data["bar_of_day"]
        meta = data.get("metadata", {})
        sidecar_dir = meta["chain_sidecar_dir"]
        max_contracts = int(meta["max_contracts_per_bar"])

        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        all_indices = np.arange(lookback, len(self.features))
        self.indices = all_indices[mask_np[lookback:]].copy()

        n = len(self.indices)
        self.all_contracts = torch.zeros(n, max_contracts, NUM_CONTRACT_FEATURES, dtype=torch.float32)
        self.all_labels = torch.full((n, max_contracts), float("nan"), dtype=torch.float32)
        self.all_best_idx = torch.zeros(n, dtype=torch.long)
        self.all_label_trade = torch.zeros(n, dtype=torch.bool)
        self.all_label_trade_valid = torch.zeros(n, dtype=torch.bool)

        t0 = time.time()
        sidecar_cache: dict[str, dict] = {}
        for j in range(n):
            i = int(self.indices[j])
            day = dates[i]
            local_bar = int(bar_of_day[i])
            if day not in sidecar_cache:
                sidecar_cache[day] = torch.load(
                    os.path.join(sidecar_dir, f"{day}.pt"),
                    map_location="cpu",
                    weights_only=False,
                )
            sc = sidecar_cache[day]
            contracts, labels, _ = padded_snapshot(sc, local_bar, max_contracts)
            self.all_contracts[j] = torch.from_numpy(contracts)
            self.all_labels[j] = torch.from_numpy(labels)
            self.all_best_idx[j] = int(sc["bar_best_contract_idx"][local_bar])
            self.all_label_trade[j] = bool(sc["bar_label_trade"][local_bar])
            self.all_label_trade_valid[j] = bool(sc["bar_labelable"][local_bar])
        print(f"  Dataset materialized: {n:,} samples, {len(sidecar_cache)} days, {time.time() - t0:.1f}s")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        window = self.features[i - self.lookback : i]
        target = {
            "contract_labels": self.all_labels[idx],
            "best_idx": self.all_best_idx[idx],
            "label_trade": self.all_label_trade[idx],
            "label_trade_valid": self.all_label_trade_valid[idx],
        }
        return window, self.all_contracts[idx], target


def compute_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["contract_scores"].device
    scores = outputs["contract_scores"]
    gate_logit = outputs["gate_logit"]
    dir_logit = outputs["direction_logit"]
    valid_mask = outputs["valid_mask"]
    labels = targets["contract_labels"].to(device)
    best_idx = targets["best_idx"].to(device)
    label_trade = targets["label_trade"].to(device)
    label_trade_valid = targets["label_trade_valid"].to(device)
    contracts_full = targets["contracts_full"].to(device)

    supervised_rows = label_trade_valid
    trade_rows = supervised_rows & label_trade & (best_idx >= 0)

    # --- A. Gate loss: balanced BCE on trade/no-trade ---
    gate_loss = torch.tensor(0.0, device=device)
    if supervised_rows.any():
        gate_target = label_trade.float()
        sup_idx = supervised_rows.nonzero(as_tuple=True)[0]
        sup_trade = label_trade[sup_idx]
        n_pos = sup_trade.sum().item()
        n_neg = len(sup_trade) - n_pos
        n_min = int(min(n_pos, n_neg))
        if n_min > 0:
            pos_idx = sup_idx[sup_trade.bool()]
            neg_idx = sup_idx[~sup_trade.bool()]
            pos_sel = pos_idx[torch.randperm(len(pos_idx), device=device)[:n_min]]
            neg_sel = neg_idx[torch.randperm(len(neg_idx), device=device)[:n_min]]
            balanced_idx = torch.cat([pos_sel, neg_sel])
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logit[balanced_idx],
                gate_target[balanced_idx],
                reduction="mean",
            )
        else:
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logit[sup_idx],
                gate_target[sup_idx],
                reduction="mean",
            )

    # --- B. Direction loss: BCE on oracle direction (call=0, put=1) ---
    dir_loss = torch.tensor(0.0, device=device)
    dir_acc = 0.0
    if trade_rows.any():
        tr_contracts = contracts_full[trade_rows]
        tr_best_idx = best_idx[trade_rows]
        rows_range = torch.arange(tr_contracts.size(0), device=device)
        oracle_is_put = (tr_contracts[rows_range, tr_best_idx, 2] > 0.5).float()
        dir_loss = F.binary_cross_entropy_with_logits(
            dir_logit[trade_rows],
            oracle_is_put,
            reduction="mean",
        )
        pred_put = (dir_logit[trade_rows] > 0).float()
        dir_acc = (pred_put == oracle_is_put).float().mean().item()

    # --- C. Strike selection: KL only over contracts matching oracle direction ---
    sel_loss = torch.tensor(0.0, device=device)
    if trade_rows.any():
        tr_scores = scores[trade_rows]
        tr_valid = valid_mask[trade_rows]
        tr_labels = labels[trade_rows]
        tr_contracts = contracts_full[trade_rows]
        tr_best_idx = best_idx[trade_rows]

        rows_range = torch.arange(tr_contracts.size(0), device=device)
        oracle_is_put = tr_contracts[rows_range, tr_best_idx, 2] > 0.5
        contract_is_put = tr_contracts[:, :, 2] > 0.5
        dir_match = contract_is_put == oracle_is_put.unsqueeze(1)

        pnl_for_target = tr_labels.clone()
        pnl_for_target[~tr_valid] = -1e9
        pnl_for_target[~torch.isfinite(pnl_for_target)] = -1e9
        pnl_for_target[~dir_match] = -1e9
        soft_target = F.softmax(pnl_for_target / SOFT_TEMP, dim=-1)

        logits_for_sel = tr_scores.clone()
        logits_for_sel[~tr_valid] = -1e9
        logits_for_sel[~dir_match] = -1e9
        log_probs = F.log_softmax(logits_for_sel, dim=-1)
        sel_loss = F.kl_div(log_probs, soft_target, reduction="batchmean")

    total = GATE_W * gate_loss + DIR_W * dir_loss + SEL_W * sel_loss

    # --- Metrics ---
    pred_trade = (gate_logit > 0).detach()
    gate_acc = (
        (pred_trade[supervised_rows] == label_trade[supervised_rows]).float().mean().item()
        if supervised_rows.any() else 0.0
    )
    trade_rate = pred_trade[supervised_rows].float().mean().item() if supervised_rows.any() else 0.0

    return total, {
        "gate": float(gate_loss.item()),
        "dir": float(dir_loss.item()),
        "sel": float(sel_loss.item()),
        "total": float(total.item()),
        "gate_acc": gate_acc,
        "dir_acc": dir_acc,
        "trade_rate": trade_rate,
    }


def load_dataset(path: str = "v2/data.pt") -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def train(data_path: str = "v2/data.pt", model_path: str = "v2/models/model.pt", train_mask_override=None, val_mask_override=None):
    t_start = time.time()
    seed = int(os.environ.get("TRAIN_SEED", SEED))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    data = load_dataset(data_path)
    train_mask = train_mask_override if train_mask_override is not None else data["train_mask"]
    val_mask = val_mask_override if val_mask_override is not None else data["val_mask"]

    train_ds = TradeDataset(data, train_mask, lookback=LOOKBACK)
    val_ds = TradeDataset(data, val_mask, lookback=LOOKBACK)
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    def collate_fn(batch):
        windows, contracts, targets_list = zip(*batch)
        windows = torch.stack(windows)
        contracts = torch.stack(contracts)
        targets = {
            "contracts_full": contracts,
            "contract_labels": torch.stack([t["contract_labels"] for t in targets_list]),
            "best_idx": torch.stack([t["best_idx"] for t in targets_list]),
            "label_trade": torch.stack([t["label_trade"] for t in targets_list]),
            "label_trade_valid": torch.stack([t["label_trade_valid"] for t in targets_list]),
        }
        return windows, contracts, targets

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = {"gate_accuracy": 0.0, "direction_accuracy": 0.0}

    for epoch in range(1, EPOCHS + 1):
        if time.time() - t_start > TIME_BUDGET:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        train_stats = []
        for batch_x, batch_c, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_c = batch_c.to(device)
            batch_y = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_y.items()}

            optimizer.zero_grad()
            outputs = model(batch_x, batch_c)
            loss, metrics = compute_loss(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_stats.append(metrics)

        scheduler.step()

        model.eval()
        val_stats = []
        with torch.no_grad():
            for batch_x, batch_c, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_c = batch_c.to(device)
                batch_y = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_y.items()}
                outputs = model(batch_x, batch_c)
                _, metrics = compute_loss(outputs, batch_y)
                val_stats.append(metrics)

        avg_train = {k: float(np.mean([d[k] for d in train_stats])) for k in train_stats[0]} if train_stats else {}
        avg_val = {k: float(np.mean([d[k] for d in val_stats])) for k in val_stats[0]} if val_stats else {}
        print(
            f"Epoch {epoch:3d} | train={avg_train.get('total', 0):.4f} | "
            f"val={avg_val.get('total', 0):.4f} | "
            f"gate={avg_val.get('gate', 0):.4f} dir={avg_val.get('dir', 0):.4f} sel={avg_val.get('sel', 0):.4f} | "
            f"g_acc={avg_val.get('gate_acc', 0):.3f} "
            f"d_acc={avg_val.get('dir_acc', 0):.3f} trd={avg_val.get('trade_rate', 0):.3f}"
        )

        val_total = avg_val.get("total", float("inf"))
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            best_metrics = {
                "gate_accuracy": avg_val.get("gate_acc", 0.0),
                "direction_accuracy": avg_val.get("dir_acc", 0.0),
            }
            dataset_fp = data.get("metadata", {}).get("fingerprint", "unknown")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_class": "TradingModel",
                    "epoch": epoch,
                    "val_loss": val_total,
                    "hyperparams": {
                        "lookback": LOOKBACK,
                        "d_model": D_MODEL,
                        "depth": DEPTH,
                        "n_heads": N_HEADS,
                        "dropout": DROPOUT,
                        "contract_features": NUM_CONTRACT_FEATURES,
                        "max_contracts_per_bar": int(data["metadata"]["max_contracts_per_bar"]),
                    },
                    "score_config_fingerprint": score_config_fingerprint(),
                    "dataset_fingerprint": dataset_fp,
                },
                model_path,
            )

    metrics = {
        "val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_run": min(epoch, EPOCHS),
        "score_config_fingerprint": score_config_fingerprint(),
        **best_metrics,
    }
    print(f"\nMETRICS_JSON:{json.dumps(metrics)}")
    print(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    return model, metrics


if __name__ == "__main__":
    _data = os.environ.get("TRAIN_DATA_PATH", "v2/data.pt")
    _model = os.environ.get("TRAIN_MODEL_PATH", "v2/models/model.pt")
    train(data_path=_data, model_path=_model)
