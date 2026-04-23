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
class UnifiedSurfaceStandardizer:
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
        contracts: np.ndarray,
        contract_mask: np.ndarray,
    ) -> "UnifiedSurfaceStandardizer":
        scalar_mean, scalar_std = _safe_mean_std(scalar)

        seq_flat = seq[seq_mask.astype(bool)]
        if len(seq_flat) == 0:
            seq_mean = np.zeros(seq.shape[-1], dtype=np.float32)
            seq_std = np.ones(seq.shape[-1], dtype=np.float32)
        else:
            seq_mean, seq_std = _safe_mean_std(seq_flat)

        contract_flat = contracts[contract_mask.astype(bool)]
        if len(contract_flat) == 0:
            contract_mean = np.zeros(contracts.shape[-1], dtype=np.float32)
            contract_std = np.ones(contracts.shape[-1], dtype=np.float32)
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
        contracts: np.ndarray,
        contract_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scalar_n = ((scalar.astype(np.float32) - self.scalar_mean) / self.scalar_std).astype(np.float32, copy=False)
        scalar_n = np.nan_to_num(scalar_n, nan=0.0, posinf=0.0, neginf=0.0)

        seq_n = ((seq.astype(np.float32) - self.seq_mean) / self.seq_std).astype(np.float32, copy=False)
        seq_n = np.nan_to_num(seq_n, nan=0.0, posinf=0.0, neginf=0.0)
        seq_n *= seq_mask[..., None].astype(np.float32)

        contracts_n = ((contracts.astype(np.float32) - self.contract_mean) / self.contract_std).astype(np.float32, copy=False)
        contracts_n = np.nan_to_num(contracts_n, nan=0.0, posinf=0.0, neginf=0.0)
        contracts_n *= contract_mask[..., None].astype(np.float32)
        return scalar_n, seq_n, contracts_n


class Layer2UnifiedActionModel(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        seq_dim: int,
        contract_dim: int,
        hidden_dim: int = 160,
        seq_hidden_dim: int = 96,
        contract_hidden_dim: int = 64,
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
        state_in_dim = hidden_dim + seq_hidden_dim
        state_layers: list[nn.Module] = []
        dim = state_in_dim
        for _ in range(max(depth, 1)):
            state_layers.append(nn.Linear(dim, hidden_dim))
            state_layers.append(nn.GELU())
            state_layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.state_trunk = nn.Sequential(*state_layers)

        self.contract_encoder = nn.Sequential(
            nn.Linear(contract_dim, contract_hidden_dim),
            nn.GELU(),
            nn.Linear(contract_hidden_dim, contract_hidden_dim),
            nn.GELU(),
        )
        action_in_dim = hidden_dim + contract_hidden_dim
        self.contract_action_trunk = nn.Sequential(
            nn.Linear(action_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.flat_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.clean_head = nn.Linear(hidden_dim, 1)
        self.stopout_head = nn.Linear(hidden_dim, 1)

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

    def forward(
        self,
        scalar: torch.Tensor,
        seq: torch.Tensor,
        seq_mask: torch.Tensor,
        contracts: torch.Tensor,
        contract_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del contract_mask  # mask is applied upstream during normalization and lossing.
        scalar_h = self.scalar_proj(scalar)
        seq_h = self._encode_sequence(seq, seq_mask)
        state_h = self.state_trunk(torch.cat([scalar_h, seq_h], dim=-1))

        token_h = self.contract_encoder(contracts)
        state_expanded = state_h.unsqueeze(1).expand(-1, contracts.shape[1], -1)
        action_hidden = self.contract_action_trunk(torch.cat([state_expanded, token_h], dim=-1))

        flat_hidden = self.flat_trunk(state_h)
        flat_utility = self.utility_head(flat_hidden).squeeze(-1)
        flat_clean = self.clean_head(flat_hidden).squeeze(-1)
        flat_stopout = self.stopout_head(flat_hidden).squeeze(-1)

        token_utility = self.utility_head(action_hidden).squeeze(-1)
        token_clean = self.clean_head(action_hidden).squeeze(-1)
        token_stopout = self.stopout_head(action_hidden).squeeze(-1)

        utility = torch.cat([flat_utility.unsqueeze(1), token_utility], dim=1)
        clean = torch.cat([flat_clean.unsqueeze(1), token_clean], dim=1)
        stopout = torch.cat([flat_stopout.unsqueeze(1), token_stopout], dim=1)
        return utility, clean, stopout


@dataclass
class UnifiedActionPredictor:
    state_dict: dict
    standardizer: UnifiedSurfaceStandardizer
    scalar_dim: int
    seq_dim: int
    contract_dim: int
    hidden_dim: int
    seq_hidden_dim: int
    contract_hidden_dim: int
    depth: int
    dropout: float
    device_hint: str = "cpu"

    def _build_model(self, device: str) -> Layer2UnifiedActionModel:
        model = Layer2UnifiedActionModel(
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
        contracts: np.ndarray,
        contract_mask: np.ndarray,
        batch_size: int = 1024,
    ) -> dict[str, np.ndarray]:
        scalar_n, seq_n, contracts_n = self.standardizer.transform(
            scalar,
            seq,
            seq_mask,
            contracts,
            contract_mask,
        )
        device = "cuda" if torch.cuda.is_available() and self.device_hint == "cuda" else "cpu"
        model = self._build_model(device=device)
        utility = np.zeros((len(scalar_n), contracts.shape[1] + 1), dtype=np.float32)
        clean_prob = np.zeros_like(utility)
        stopout_prob = np.zeros_like(utility)
        with torch.inference_mode():
            for start in range(0, len(scalar_n), batch_size):
                end = start + batch_size
                scalar_b = torch.from_numpy(scalar_n[start:end]).to(device)
                seq_b = torch.from_numpy(seq_n[start:end]).to(device)
                seq_mask_b = torch.from_numpy(seq_mask[start:end].astype(np.float32)).to(device)
                contract_b = torch.from_numpy(contracts_n[start:end]).to(device)
                contract_mask_b = torch.from_numpy(contract_mask[start:end].astype(np.float32)).to(device)

                util_b, clean_b, stopout_b = model(
                    scalar_b,
                    seq_b,
                    seq_mask_b,
                    contract_b,
                    contract_mask_b,
                )
                utility[start:end] = util_b.detach().cpu().numpy().astype(np.float32, copy=False)
                clean_prob[start:end] = torch.sigmoid(clean_b).detach().cpu().numpy().astype(np.float32, copy=False)
                stopout_prob[start:end] = torch.sigmoid(stopout_b).detach().cpu().numpy().astype(np.float32, copy=False)
        return {
            "utility": utility,
            "clean_prob": clean_prob,
            "stopout_prob": stopout_prob,
        }


def _masked_weighted_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    valid = mask > 0.0
    if not valid.any():
        return pred.new_tensor(0.0)
    err = pred - target
    abs_err = torch.abs(err)
    quadratic = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    denom = torch.clamp((weight * mask).sum(), min=1.0)
    return (loss * weight * mask).sum() / denom


def _masked_bce(
    logit: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask > 0.0
    if not valid.any():
        return logit.new_tensor(0.0)
    loss = nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    denom = torch.clamp(mask.sum(), min=1.0)
    return (loss * mask).sum() / denom


def _pairwise_ranking_loss(
    pred_utility: torch.Tensor,
    target_utility_raw: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.20,
) -> torch.Tensor:
    """Best-vs-rest hinge over valid actions.

    The smoke model picked calls 91% of the time and the calibrator rode that
    with a negative decision margin. A 0.05 arcsinh hinge (~$5 of PnL) was not
    enough to force real discrimination. 0.20 corresponds to ~$20 of PnL and
    matches the scale at which a 0DTE trader can actually tell apart strikes.
    """
    valid = valid_mask > 0.0
    if not valid.any():
        return pred_utility.new_tensor(0.0)

    target_masked = torch.where(valid, target_utility_raw, torch.full_like(target_utility_raw, float("-inf")))
    best_idx = target_masked.argmax(dim=1)
    row_ids = torch.arange(pred_utility.shape[0], device=pred_utility.device)
    best_pred = pred_utility[row_ids, best_idx].unsqueeze(1)
    pair_mask = valid.clone()
    pair_mask[row_ids, best_idx] = False
    if not pair_mask.any():
        return pred_utility.new_tensor(0.0)
    loss = torch.relu(margin - (best_pred - pred_utility))
    denom = torch.clamp(pair_mask.sum(), min=1)
    return (loss * pair_mask.float()).sum() / denom.float()


def _flat_ranking_loss(
    pred_utility: torch.Tensor,
    target_utility_raw: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.20,
    utility_eps: float = 10.0,
) -> torch.Tensor:
    """Bidirectional pressure against flat (action id 0).

    The best-vs-rest hinge only enforces ONE anchor per bar. That lets the
    model's flat prediction drift while contract predictions rearrange, which
    is exactly the failure mode that gave calibration a negative margin. We
    explicitly push flat above every losing tradeable contract and every
    winning tradeable contract above flat, using the same arcsinh-scale hinge.
    """
    if valid_mask.shape[1] < 2:
        return pred_utility.new_tensor(0.0)
    valid_contracts = valid_mask[:, 1:] > 0.0
    if not valid_contracts.any():
        return pred_utility.new_tensor(0.0)
    flat_pred = pred_utility[:, :1]
    contract_pred = pred_utility[:, 1:]
    contract_target = target_utility_raw[:, 1:]
    # "positive" = utility > utility_eps dollars; "negative" = utility < -utility_eps.
    pos_mask = valid_contracts & (contract_target > utility_eps)
    neg_mask = valid_contracts & (contract_target < -utility_eps)

    pos_loss = torch.relu(margin - (contract_pred - flat_pred))
    neg_loss = torch.relu(margin - (flat_pred - contract_pred))

    pos_denom = torch.clamp(pos_mask.sum(), min=1).float()
    neg_denom = torch.clamp(neg_mask.sum(), min=1).float()
    pos_term = (pos_loss * pos_mask.float()).sum() / pos_denom
    neg_term = (neg_loss * neg_mask.float()).sum() / neg_denom
    return 0.5 * (pos_term + neg_term)


def _side_contrastive_loss(
    pred_utility: torch.Tensor,
    target_utility_raw: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.20,
    utility_eps: float = 10.0,
) -> torch.Tensor:
    """Force direct same-bar call-vs-put discrimination.

    Best-vs-rest ranking can be satisfied by a global side prior if the model
    already likes one side and only needs to sort contracts within it. This
    term asks a simpler question the previous objective never asked directly:
    on bars where the best call and best put meaningfully disagree, did we
    score the better side above the worse side?
    """
    n_actions = pred_utility.shape[1]
    n_contracts = n_actions - 1
    if n_contracts < 2 or (n_contracts % 2) != 0:
        return pred_utility.new_tensor(0.0)

    top_k = n_contracts // 2
    call_valid = valid_mask[:, 1 : 1 + top_k] > 0.0
    put_valid = valid_mask[:, 1 + top_k :] > 0.0
    if not call_valid.any() or not put_valid.any():
        return pred_utility.new_tensor(0.0)

    call_target = torch.where(
        call_valid,
        target_utility_raw[:, 1 : 1 + top_k],
        torch.full_like(target_utility_raw[:, 1 : 1 + top_k], float("-inf")),
    )
    put_target = torch.where(
        put_valid,
        target_utility_raw[:, 1 + top_k :],
        torch.full_like(target_utility_raw[:, 1 + top_k :], float("-inf")),
    )
    row_ids = torch.arange(pred_utility.shape[0], device=pred_utility.device)
    best_call_idx = call_target.argmax(dim=1)
    best_put_idx = put_target.argmax(dim=1)

    best_call_target = call_target[row_ids, best_call_idx]
    best_put_target = put_target[row_ids, best_put_idx]
    best_call_pred = pred_utility[:, 1 : 1 + top_k][row_ids, best_call_idx]
    best_put_pred = pred_utility[:, 1 + top_k :][row_ids, best_put_idx]

    have_both = (
        call_valid.any(dim=1)
        & put_valid.any(dim=1)
        & torch.isfinite(best_call_target)
        & torch.isfinite(best_put_target)
    )
    call_better = have_both & (best_call_target >= best_put_target + utility_eps)
    put_better = have_both & (best_put_target >= best_call_target + utility_eps)
    if not call_better.any() and not put_better.any():
        return pred_utility.new_tensor(0.0)

    call_loss = torch.relu(margin - (best_call_pred - best_put_pred))
    put_loss = torch.relu(margin - (best_put_pred - best_call_pred))
    total = call_loss * call_better.float() + put_loss * put_better.float()
    denom = torch.clamp(call_better.sum() + put_better.sum(), min=1).float()
    return total.sum() / denom


def train_unified_action_model(
    *,
    scalar_train: np.ndarray,
    seq_train: np.ndarray,
    seq_mask_train: np.ndarray,
    contracts_train: np.ndarray,
    contract_mask_train: np.ndarray,
    utility_train: np.ndarray,
    utility_raw_train: np.ndarray,
    clean_train: np.ndarray,
    stopout_train: np.ndarray,
    available_mask_train: np.ndarray,
    tradeable_mask_train: np.ndarray,
    scalar_val: np.ndarray,
    seq_val: np.ndarray,
    seq_mask_val: np.ndarray,
    contracts_val: np.ndarray,
    contract_mask_val: np.ndarray,
    utility_val: np.ndarray,
    utility_raw_val: np.ndarray,
    clean_val: np.ndarray,
    stopout_val: np.ndarray,
    available_mask_val: np.ndarray,
    tradeable_mask_val: np.ndarray,
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
    w_regression: float,
    w_ranking: float,
    w_side_contrastive: float,
    w_clean: float,
    w_stopout: float,
) -> tuple[UnifiedActionPredictor, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    standardizer = UnifiedSurfaceStandardizer.fit(
        scalar_train,
        seq_train,
        seq_mask_train,
        contracts_train,
        contract_mask_train,
    )
    scalar_train_n, seq_train_n, contracts_train_n = standardizer.transform(
        scalar_train,
        seq_train,
        seq_mask_train,
        contracts_train,
        contract_mask_train,
    )
    scalar_val_n, seq_val_n, contracts_val_n = standardizer.transform(
        scalar_val,
        seq_val,
        seq_mask_val,
        contracts_val,
        contract_mask_val,
    )

    reg_mask_train = np.isfinite(utility_train).astype(np.float32) * tradeable_mask_train.astype(np.float32)
    reg_mask_val = np.isfinite(utility_val).astype(np.float32) * tradeable_mask_val.astype(np.float32)
    reg_mask_train = np.nan_to_num(reg_mask_train, nan=0.0).astype(np.float32)
    reg_mask_val = np.nan_to_num(reg_mask_val, nan=0.0).astype(np.float32)
    utility_train_filled = np.nan_to_num(utility_train, nan=0.0).astype(np.float32)
    utility_val_filled = np.nan_to_num(utility_val, nan=0.0).astype(np.float32)
    utility_raw_train_filled = np.nan_to_num(utility_raw_train, nan=-1e9).astype(np.float32)
    utility_raw_val_filled = np.nan_to_num(utility_raw_val, nan=-1e9).astype(np.float32)

    abs_utility = np.abs(np.nan_to_num(utility_raw_train, nan=0.0))
    reg_weight_train = (0.5 + np.clip(abs_utility / 250.0, 0.0, 3.0)).astype(np.float32)
    # Flat is the anchor — its true utility is exactly $0 every bar. If we
    # downweight the anchor, the model's predicted flat value drifts, and a
    # drifting flat score is the denominator of `best_nonflat_score -
    # flat_score`. Upweighting it (1.5) pins the margin distribution in a
    # regime where calibration can reliably pick a positive threshold.
    reg_weight_train[:, 0] = 1.5
    reg_weight_val = np.ones_like(utility_val_filled, dtype=np.float32)

    clean_mask_train = np.isfinite(clean_train).astype(np.float32) * available_mask_train.astype(np.float32)
    clean_mask_val = np.isfinite(clean_val).astype(np.float32) * available_mask_val.astype(np.float32)
    stopout_mask_train = np.isfinite(stopout_train).astype(np.float32) * available_mask_train.astype(np.float32)
    stopout_mask_val = np.isfinite(stopout_val).astype(np.float32) * available_mask_val.astype(np.float32)
    clean_mask_train = np.nan_to_num(clean_mask_train, nan=0.0).astype(np.float32)
    clean_mask_val = np.nan_to_num(clean_mask_val, nan=0.0).astype(np.float32)
    stopout_mask_train = np.nan_to_num(stopout_mask_train, nan=0.0).astype(np.float32)
    stopout_mask_val = np.nan_to_num(stopout_mask_val, nan=0.0).astype(np.float32)

    clean_train_filled = np.nan_to_num(clean_train, nan=0.0).astype(np.float32)
    clean_val_filled = np.nan_to_num(clean_val, nan=0.0).astype(np.float32)
    stopout_train_filled = np.nan_to_num(stopout_train, nan=0.0).astype(np.float32)
    stopout_val_filled = np.nan_to_num(stopout_val, nan=0.0).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(scalar_train_n),
        torch.from_numpy(seq_train_n),
        torch.from_numpy(seq_mask_train.astype(np.float32)),
        torch.from_numpy(contracts_train_n),
        torch.from_numpy(contract_mask_train.astype(np.float32)),
        torch.from_numpy(utility_train_filled),
        torch.from_numpy(utility_raw_train_filled),
        torch.from_numpy(reg_mask_train),
        torch.from_numpy(reg_weight_train),
        torch.from_numpy(clean_train_filled),
        torch.from_numpy(clean_mask_train),
        torch.from_numpy(stopout_train_filled),
        torch.from_numpy(stopout_mask_train),
        torch.from_numpy(np.nan_to_num(tradeable_mask_train, nan=0.0).astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(scalar_val_n),
        torch.from_numpy(seq_val_n),
        torch.from_numpy(seq_mask_val.astype(np.float32)),
        torch.from_numpy(contracts_val_n),
        torch.from_numpy(contract_mask_val.astype(np.float32)),
        torch.from_numpy(utility_val_filled),
        torch.from_numpy(utility_raw_val_filled),
        torch.from_numpy(reg_mask_val),
        torch.from_numpy(reg_weight_val),
        torch.from_numpy(clean_val_filled),
        torch.from_numpy(clean_mask_val),
        torch.from_numpy(stopout_val_filled),
        torch.from_numpy(stopout_mask_val),
        torch.from_numpy(np.nan_to_num(tradeable_mask_val, nan=0.0).astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = Layer2UnifiedActionModel(
        scalar_dim=scalar_train_n.shape[1],
        seq_dim=seq_train_n.shape[2],
        contract_dim=contracts_train_n.shape[2],
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
    best_epoch = -1
    stale = 0
    best_parts: dict[str, float] = {}

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            (
                scalar_b,
                seq_b,
                seq_mask_b,
                contracts_b,
                contract_mask_b,
                utility_b,
                utility_raw_b,
                reg_mask_b,
                reg_weight_b,
                clean_b,
                clean_mask_b,
                stopout_b,
                stopout_mask_b,
                tradeable_b,
            ) = [x.to(device) for x in batch]

            optimizer.zero_grad()
            utility_pred, clean_pred, stopout_pred = model(
                scalar_b,
                seq_b,
                seq_mask_b,
                contracts_b,
                contract_mask_b,
            )
            reg_loss = _masked_weighted_huber(utility_pred, utility_b, reg_mask_b, reg_weight_b)
            rank_loss = _pairwise_ranking_loss(utility_pred, utility_raw_b, tradeable_b)
            flat_rank_loss = _flat_ranking_loss(utility_pred, utility_raw_b, tradeable_b)
            side_loss = _side_contrastive_loss(utility_pred, utility_raw_b, tradeable_b)
            clean_loss = _masked_bce(clean_pred, clean_b, clean_mask_b)
            stopout_loss = _masked_bce(stopout_pred, stopout_b, stopout_mask_b)
            total_loss = (
                w_regression * reg_loss
                + w_ranking * rank_loss
                + w_ranking * flat_rank_loss
                + w_side_contrastive * side_loss
                + w_clean * clean_loss
                + w_stopout * stopout_loss
            )
            total_loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_totals: list[float] = []
        val_regs: list[float] = []
        val_ranks: list[float] = []
        val_sides: list[float] = []
        val_cleans: list[float] = []
        val_stopouts: list[float] = []
        with torch.inference_mode():
            for batch in val_loader:
                (
                    scalar_b,
                    seq_b,
                    seq_mask_b,
                    contracts_b,
                    contract_mask_b,
                    utility_b,
                    utility_raw_b,
                    reg_mask_b,
                    reg_weight_b,
                    clean_b,
                    clean_mask_b,
                    stopout_b,
                    stopout_mask_b,
                    tradeable_b,
                ) = [x.to(device) for x in batch]
                utility_pred, clean_pred, stopout_pred = model(
                    scalar_b,
                    seq_b,
                    seq_mask_b,
                    contracts_b,
                    contract_mask_b,
                )
                reg_loss = _masked_weighted_huber(utility_pred, utility_b, reg_mask_b, reg_weight_b)
                rank_loss = _pairwise_ranking_loss(utility_pred, utility_raw_b, tradeable_b)
                flat_rank_loss = _flat_ranking_loss(utility_pred, utility_raw_b, tradeable_b)
                side_loss = _side_contrastive_loss(utility_pred, utility_raw_b, tradeable_b)
                clean_loss = _masked_bce(clean_pred, clean_b, clean_mask_b)
                stopout_loss = _masked_bce(stopout_pred, stopout_b, stopout_mask_b)
                total_loss = (
                    w_regression * reg_loss
                    + w_ranking * rank_loss
                    + w_ranking * flat_rank_loss
                    + w_side_contrastive * side_loss
                    + w_clean * clean_loss
                    + w_stopout * stopout_loss
                )
                val_totals.append(float(total_loss.item()))
                val_regs.append(float(reg_loss.item()))
                val_ranks.append(float(rank_loss.item()) + float(flat_rank_loss.item()))
                val_sides.append(float(side_loss.item()))
                val_cleans.append(float(clean_loss.item()))
                val_stopouts.append(float(stopout_loss.item()))

        mean_val = float(np.mean(val_totals)) if val_totals else float("inf")
        if mean_val < best_val - 1e-6:
            best_val = mean_val
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_parts = {
                "best_regression_val_loss": float(np.mean(val_regs)) if val_regs else float("inf"),
                "best_ranking_val_loss": float(np.mean(val_ranks)) if val_ranks else float("inf"),
                "best_side_contrastive_val_loss": float(np.mean(val_sides)) if val_sides else float("inf"),
                "best_clean_val_loss": float(np.mean(val_cleans)) if val_cleans else float("inf"),
                "best_stopout_val_loss": float(np.mean(val_stopouts)) if val_stopouts else float("inf"),
            }
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    predictor = UnifiedActionPredictor(
        state_dict=best_state,
        standardizer=standardizer,
        scalar_dim=scalar_train_n.shape[1],
        seq_dim=seq_train_n.shape[2],
        contract_dim=contracts_train_n.shape[2],
        hidden_dim=hidden_dim,
        seq_hidden_dim=seq_hidden_dim,
        contract_hidden_dim=contract_hidden_dim,
        depth=depth,
        dropout=dropout,
        device_hint="cuda" if device == "cuda" else "cpu",
    )
    training_info = {
        "best_val_loss": float(best_val),
        "best_epoch": float(best_epoch),
        "train_rows": float(len(scalar_train)),
        "val_rows": float(len(scalar_val)),
        "w_side_contrastive": float(w_side_contrastive),
        **best_parts,
    }
    return predictor, training_info
