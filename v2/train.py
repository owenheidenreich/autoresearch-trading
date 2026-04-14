"""ART² v4 Training Loop -- balanced gate + soft KL contract selection."""
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


NUM_FEATURES = int(os.environ.get("NUM_FEATURES", 52))
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
SEED = int(os.environ.get("TRAIN_SEED", 123))
SOFT_TEMP = float(os.environ.get("SOFT_TEMP", 0.25))
NOISE_MARGIN = float(os.environ.get("NOISE_MARGIN", 0.01))
AMBIG_WEIGHT = float(os.environ.get("AMBIG_WEIGHT", 0.3))
SIDE_SEL_W = float(os.environ.get("SIDE_SEL_W", 0.0))
EXACT_W = float(os.environ.get("EXACT_W", 0.0))
OPP_W = float(os.environ.get("OPP_W", 0.5))
SIDE_W = float(os.environ.get("SIDE_W", 0.0))
AGG_W = float(os.environ.get("AGG_W", 0.0))
# Moneyness bucket boundaries for aggression head
AGG_ATM_THRESH = 0.5   # |moneyness_pct| < 0.5% = ATM
AGG_NEAR_THRESH = 1.5  # 0.5-1.5% = near-OTM, >1.5% = far-OTM


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
    """Contract scorer with balanced gate + soft KL selection training."""

    def __init__(self, d_model: int | None = None, depth: int | None = None, n_heads: int | None = None, dropout: float | None = None):
        super().__init__()
        d = d_model or D_MODEL
        dep = depth or DEPTH
        nh = n_heads or N_HEADS
        dr = DROPOUT if dropout is None else dropout

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

        self.contract_proj = nn.Sequential(
            nn.Linear(NUM_CONTRACT_FEATURES, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.no_trade_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(d // 2, 1),
        )
        self.call_score_head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.put_score_head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.put_bias = nn.Parameter(torch.tensor(0.0))

        # Independent opportunity-quality head: "should I trade this bar?"
        # Decides from context alone, before seeing any contracts.
        self.opportunity_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(d // 2, 1),
        )

        # Side prediction head: P(call side is better) from context alone
        self.side_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

        # Aggression head: predict moneyness bucket (ATM / near-OTM / far-OTM)
        self.aggression_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 3),
        )

    def forward(self, x: torch.Tensor, contracts: torch.Tensor) -> dict[str, torch.Tensor]:
        _, seq_len, _ = x.shape
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.pos_enc(h)
        mask = self.causal_mask[:seq_len, :seq_len] if seq_len <= self.causal_mask.size(0) else None
        h = self.encoder(h, mask=mask)
        context = h[:, -1, :]

        # --- Contract feature normalization ---
        c = contracts.clone()
        valid_mask = contracts[:, :, 0] > 0.5
        is_put = contracts[:, :, 2] > 0.5

        # Step 1: Greek sign normalization — align puts with calls
        # delta(8), moneyness_pct(11), distance_points(12), charm(16) flip sign for puts
        put_flip = is_put.float()
        for fidx in (8, 11, 12, 16):
            c[:, :, fidx] = c[:, :, fidx] * (1.0 - 2.0 * put_flip)

        # Step 2: Per-bar z-score on continuous features
        # Skip: 0 (contract_valid), 2 (right_is_put), 13 (minutes_to_close), 14 (quality)
        valid_f = valid_mask.float()
        count = valid_f.sum(dim=1, keepdim=True).clamp(min=1)
        for fidx in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21):
            feat = c[:, :, fidx]
            masked = feat * valid_f
            mean = masked.sum(dim=1, keepdim=True) / count
            diff = (feat - mean) * valid_f
            std = (diff.pow(2).sum(dim=1, keepdim=True) / count).sqrt().clamp(min=1e-6)
            c[:, :, fidx] = (feat - mean) / std

        # Step 3: Zero out invalid contracts
        c = c * valid_f.unsqueeze(-1)

        contract_emb = self.contract_proj(c)
        context_exp = context.unsqueeze(1).expand(-1, contract_emb.size(1), -1)
        combined = torch.cat([context_exp, contract_emb], dim=-1)

        # Route each contract through its side-specific score head
        call_scores = self.call_score_head(combined).squeeze(-1)
        put_scores = self.put_score_head(combined).squeeze(-1)

        # Per-bar mean centering with learned put bias
        call_valid = (~is_put) & valid_mask
        put_valid = is_put & valid_mask
        call_count = call_valid.float().sum(dim=-1, keepdim=True).clamp(min=1)
        put_count = put_valid.float().sum(dim=-1, keepdim=True).clamp(min=1)
        call_mean = (call_scores * call_valid.float()).sum(dim=-1, keepdim=True) / call_count
        put_mean = (put_scores * put_valid.float()).sum(dim=-1, keepdim=True) / put_count
        call_scores_centered = call_scores - call_mean
        put_scores_centered = put_scores - put_mean

        contract_scores = torch.where(is_put, put_scores_centered + self.put_bias, call_scores_centered)

        no_trade_score = self.no_trade_head(context).squeeze(-1)
        opportunity_logit = self.opportunity_head(context).squeeze(-1)
        side_logit = self.side_head(context).squeeze(-1)
        aggression_logits = self.aggression_head(context)  # (B, 3)
        return {
            "contract_scores": contract_scores,
            "no_trade_score": no_trade_score,
            "valid_mask": valid_mask,
            "is_put": is_put,
            "call_scores_raw": call_scores,
            "put_scores_raw": put_scores,
            "opportunity_logit": opportunity_logit,
            "side_logit": side_logit,
            "aggression_logits": aggression_logits,
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
    no_trade = outputs["no_trade_score"]
    valid_mask = outputs["valid_mask"]
    labels = targets["contract_labels"].to(device)
    best_idx = targets["best_idx"].to(device)
    label_trade = targets["label_trade"].to(device)
    label_trade_valid = targets["label_trade_valid"].to(device)
    contracts_full = targets["contracts_full"].to(device)

    supervised_rows = label_trade_valid
    trade_rows = supervised_rows & label_trade & (best_idx >= 0)
    opportunity_logit = outputs["opportunity_logit"]
    side_logit = outputs["side_logit"]

    # --- A. Gate loss: balanced BCE (uses max(scores)-no_trade for backward compat) ---
    gate_loss = torch.tensor(0.0, device=device)
    if supervised_rows.any():
        masked_scores = scores.clone()
        masked_scores[~valid_mask] = -1e9
        best_contract_score, _ = masked_scores.max(dim=-1)
        gate_logit = best_contract_score - no_trade
        gate_target = label_trade.float()
        # Balanced sampling: match minority class count
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

    # --- A2. Opportunity loss: independent gate from context alone ---
    opp_loss = torch.tensor(0.0, device=device)
    if OPP_W > 0 and supervised_rows.any():
        sup_idx = supervised_rows.nonzero(as_tuple=True)[0]
        opp_target = label_trade[sup_idx].float()
        n_pos = opp_target.sum().item()
        n_neg = len(opp_target) - n_pos
        n_min = int(min(n_pos, n_neg))
        if n_min > 0:
            pos_idx = sup_idx[opp_target.bool()]
            neg_idx = sup_idx[~opp_target.bool()]
            pos_sel = pos_idx[torch.randperm(len(pos_idx), device=device)[:n_min]]
            neg_sel = neg_idx[torch.randperm(len(neg_idx), device=device)[:n_min]]
            balanced_opp = torch.cat([pos_sel, neg_sel])
            opp_loss = F.binary_cross_entropy_with_logits(
                opportunity_logit[balanced_opp],
                label_trade[balanced_opp].float(),
                reduction="mean",
            )

    # --- A3. Side loss: predict call vs put from context alone (trade rows only) ---
    side_loss = torch.tensor(0.0, device=device)
    if SIDE_W > 0 and trade_rows.any():
        tr_best = best_idx[trade_rows]
        tr_contracts = contracts_full[trade_rows]
        rows_arange = torch.arange(tr_best.size(0), device=device)
        oracle_is_put = (tr_contracts[rows_arange, tr_best, 2] > 0.5).float()
        # side_logit > 0 means "call is better" (sigmoid=1), put -> target=0
        side_target = 1.0 - oracle_is_put  # 1.0 = call oracle, 0.0 = put oracle
        side_loss = F.binary_cross_entropy_with_logits(
            side_logit[trade_rows],
            side_target,
            reduction="mean",
        )

    # --- A4. Aggression loss: predict moneyness bucket from context (trade rows only) ---
    agg_loss = torch.tensor(0.0, device=device)
    aggression_logits = outputs["aggression_logits"]
    if AGG_W > 0 and trade_rows.any():
        tr_best_a = best_idx[trade_rows]
        tr_contracts_a = contracts_full[trade_rows]
        rows_a = torch.arange(tr_best_a.size(0), device=device)
        # moneyness_pct is at index 11 in contract features
        oracle_moneyness = tr_contracts_a[rows_a, tr_best_a, 11].abs()
        # Bucket: 0=ATM, 1=near-OTM, 2=far-OTM
        agg_target = torch.where(
            oracle_moneyness < AGG_ATM_THRESH, torch.tensor(0, device=device),
            torch.where(oracle_moneyness < AGG_NEAR_THRESH, torch.tensor(1, device=device),
                        torch.tensor(2, device=device))
        )
        agg_loss = F.cross_entropy(aggression_logits[trade_rows], agg_target, reduction="mean")

    # --- B. Selection loss: KL over valid contracts, skip noise bars ---
    sel_loss = torch.tensor(0.0, device=device)
    if trade_rows.any():
        tr_scores = scores[trade_rows]
        tr_valid = valid_mask[trade_rows]
        tr_labels = labels[trade_rows]

        pnl_for_target = tr_labels.clone()
        pnl_for_target[~tr_valid] = -1e9
        pnl_for_target[~torch.isfinite(pnl_for_target)] = -1e9

        if tr_scores.size(0) > 0:
            # Soft ambiguous bar handling: instead of dropping ambiguous bars,
            # weight them down and use uniform target for unclear cases.
            top2_vals, _ = pnl_for_target.topk(min(2, pnl_for_target.size(-1)), dim=-1)
            if top2_vals.size(-1) >= 2:
                margin = top2_vals[:, 0] - top2_vals[:, 1]
                clear_bars = margin > NOISE_MARGIN
            else:
                clear_bars = torch.ones(tr_scores.size(0), dtype=torch.bool, device=device)

            # Clear bars: PnL-derived soft target
            soft_target = F.softmax(pnl_for_target / SOFT_TEMP, dim=-1)

            # Ambiguous bars: uniform target over valid contracts
            n_valid_per_bar = tr_valid.float().sum(dim=-1, keepdim=True).clamp(min=1)
            uniform_target = tr_valid.float() / n_valid_per_bar

            # Blend: clear bars use soft_target, ambiguous use uniform
            target = torch.where(
                clear_bars.unsqueeze(-1).expand_as(soft_target),
                soft_target,
                uniform_target,
            )
            # Per-bar weight: clear=1.0, ambiguous=AMBIG_WEIGHT
            bar_weight = torch.where(clear_bars, 1.0, AMBIG_WEIGHT)

            logits_for_sel = tr_scores.clone()
            logits_for_sel[~tr_valid] = -1e9
            log_probs = F.log_softmax(logits_for_sel, dim=-1)
            per_bar_kl = F.kl_div(log_probs, target, reduction="none").sum(dim=-1)
            sel_loss = (per_bar_kl * bar_weight).mean()

    # --- B2. Exact-oracle cross-entropy: push exact oracle contract above neighbors ---
    exact_loss = torch.tensor(0.0, device=device)
    if EXACT_W > 0 and trade_rows.any():
        ex_scores = scores[trade_rows]
        ex_valid = valid_mask[trade_rows]
        ex_best = best_idx[trade_rows]

        # Mask invalid contracts
        ex_logits = ex_scores.clone()
        ex_logits[~ex_valid] = -1e9

        # Cross-entropy with oracle contract as target class
        exact_loss = F.cross_entropy(ex_logits, ex_best, reduction="mean")

    # --- C. Per-side ranking KL: teach each head to rank within its own side ---
    side_sel_loss = torch.tensor(0.0, device=device)
    if SIDE_SEL_W > 0 and trade_rows.any():
        is_put = outputs["is_put"]
        call_raw = outputs["call_scores_raw"]
        put_raw = outputs["put_scores_raw"]

        tr_is_put = is_put[trade_rows]
        tr_call_raw = call_raw[trade_rows]
        tr_put_raw = put_raw[trade_rows]
        tr_valid_side = valid_mask[trade_rows]
        tr_labels_side = labels[trade_rows]
        tr_best_idx = best_idx[trade_rows]

        # Determine oracle side per row from the oracle contract
        rows_arange = torch.arange(tr_best_idx.size(0), device=device)
        oracle_is_put = tr_is_put[rows_arange, tr_best_idx]

        # Filter noise bars same as section B
        pnl_side = tr_labels_side.clone()
        pnl_side[~tr_valid_side] = -1e9
        pnl_side[~torch.isfinite(pnl_side)] = -1e9
        if NOISE_MARGIN > 0:
            top2_side, _ = pnl_side.topk(2, dim=-1)
            clear_side = (top2_side[:, 0] - top2_side[:, 1]) > NOISE_MARGIN
        else:
            clear_side = torch.ones(pnl_side.size(0), dtype=torch.bool, device=device)

        # Call-oracle rows: KL over call contracts only
        call_oracle_mask = (~oracle_is_put) & clear_side
        if call_oracle_mask.any():
            co_scores = tr_call_raw[call_oracle_mask]
            co_valid = tr_valid_side[call_oracle_mask] & (~tr_is_put[call_oracle_mask])
            co_pnl = pnl_side[call_oracle_mask].clone()
            co_pnl[tr_is_put[call_oracle_mask]] = -1e9  # mask puts from target
            co_pnl[~co_valid] = -1e9
            co_target = F.softmax(co_pnl / SOFT_TEMP, dim=-1)
            co_logits = co_scores.clone()
            co_logits[~co_valid] = -1e9
            co_log_probs = F.log_softmax(co_logits, dim=-1)
            side_sel_loss = side_sel_loss + F.kl_div(co_log_probs, co_target, reduction="batchmean")

        # Put-oracle rows: KL over put contracts only
        put_oracle_mask = oracle_is_put & clear_side
        if put_oracle_mask.any():
            po_scores = tr_put_raw[put_oracle_mask]
            po_valid = tr_valid_side[put_oracle_mask] & tr_is_put[put_oracle_mask]
            po_pnl = pnl_side[put_oracle_mask].clone()
            po_pnl[~tr_is_put[put_oracle_mask]] = -1e9  # mask calls from target
            po_pnl[~po_valid] = -1e9
            po_target = F.softmax(po_pnl / SOFT_TEMP, dim=-1)
            po_logits = po_scores.clone()
            po_logits[~po_valid] = -1e9
            po_log_probs = F.log_softmax(po_logits, dim=-1)
            side_sel_loss = side_sel_loss + F.kl_div(po_log_probs, po_target, reduction="batchmean")

    total = (GATE_W * gate_loss + SEL_W * sel_loss + SIDE_SEL_W * side_sel_loss
             + EXACT_W * exact_loss + OPP_W * opp_loss + SIDE_W * side_loss
             + AGG_W * agg_loss)

    # --- Metrics ---
    masked_scores_eval = scores.detach().clone()
    masked_scores_eval[~valid_mask] = -1e9
    best_eval, pred_contract = masked_scores_eval.max(dim=-1)
    # Use opportunity_logit as primary gate (independent of contract ranking)
    pred_trade = opportunity_logit.detach() > 0

    gate_acc = (
        (pred_trade[supervised_rows] == label_trade[supervised_rows]).float().mean().item()
        if supervised_rows.any() else 0.0
    )
    trade_rate = pred_trade[supervised_rows].float().mean().item() if supervised_rows.any() else 0.0

    dir_acc = 0.0
    if trade_rows.any():
        trade_contracts = contracts_full[trade_rows]
        pred_idx = pred_contract[trade_rows]
        true_idx = best_idx[trade_rows]
        rows = torch.arange(trade_contracts.size(0), device=device)
        pred_put = (trade_contracts[rows, pred_idx, 2] > 0.5).long()
        true_put = (trade_contracts[rows, true_idx, 2] > 0.5).long()
        dir_acc = (pred_put == true_put).float().mean().item() if len(pred_put) else 0.0

    # Aggression bucket accuracy
    agg_acc = 0.0
    if trade_rows.any():
        with torch.no_grad():
            pred_bucket = aggression_logits[trade_rows].detach().argmax(dim=-1)
            tr_best_aa = best_idx[trade_rows]
            tr_contracts_aa = contracts_full[trade_rows]
            rows_aa = torch.arange(tr_best_aa.size(0), device=device)
            oracle_m = tr_contracts_aa[rows_aa, tr_best_aa, 11].abs()
            true_bucket = torch.where(
                oracle_m < AGG_ATM_THRESH, torch.tensor(0, device=device),
                torch.where(oracle_m < AGG_NEAR_THRESH, torch.tensor(1, device=device),
                            torch.tensor(2, device=device))
            )
            agg_acc = (pred_bucket == true_bucket).float().mean().item()

    # Side prediction accuracy
    side_acc = 0.0
    if trade_rows.any():
        with torch.no_grad():
            tr_best_s = best_idx[trade_rows]
            tr_contracts_s = contracts_full[trade_rows]
            rows_s = torch.arange(tr_best_s.size(0), device=device)
            true_is_put = tr_contracts_s[rows_s, tr_best_s, 2] > 0.5
            pred_is_call = side_logit[trade_rows].detach() > 0
            true_is_call = ~true_is_put
            side_acc = (pred_is_call == true_is_call).float().mean().item()

    return total, {
        "gate": float(gate_loss.item()),
        "sel": float(sel_loss.item()),
        "side_sel": float(side_sel_loss.item()),
        "exact": float(exact_loss.item()),
        "opp": float(opp_loss.item()),
        "side": float(side_loss.item()),
        "agg": float(agg_loss.item()),
        "total": float(total.item()),
        "gate_acc": gate_acc,
        "dir_acc": dir_acc,
        "side_acc": side_acc,
        "agg_acc": agg_acc,
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
            f"gate_l={avg_val.get('gate', 0):.4f} sel={avg_val.get('sel', 0):.4f} opp={avg_val.get('opp', 0):.4f} side_l={avg_val.get('side', 0):.4f} | "
            f"gate={avg_val.get('gate_acc', 0):.3f} "
            f"dir={avg_val.get('dir_acc', 0):.3f} side={avg_val.get('side_acc', 0):.3f} trd_rate={avg_val.get('trade_rate', 0):.3f}"
        )

        val_total = avg_val.get("total", float("inf"))
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            best_metrics = {
                "gate_accuracy": avg_val.get("gate_acc", 0.0),
                "direction_accuracy": avg_val.get("dir_acc", 0.0),
                "side_accuracy": avg_val.get("side_acc", 0.0),
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
