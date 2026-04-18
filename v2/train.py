"""ART² v4 Training Loop -- balanced gate + soft KL contract selection."""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass

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
LR = float(os.environ.get("TRAIN_LR", 1e-4))
WEIGHT_DECAY = float(os.environ.get("TRAIN_WEIGHT_DECAY", 0.03))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 24))
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 300))
SEL_W = float(os.environ.get("WEIGHT_SEL", 1.0))
GATE_W = float(os.environ.get("WEIGHT_GATE", 1.0))
SEED = int(os.environ.get("TRAIN_SEED", 123))
OPP_LABEL = os.environ.get("OPP_LABEL", "strict")  # "old", "strict", "consensus", "high_threshold"
SOFT_TEMP = float(os.environ.get("SOFT_TEMP", 0.08))
NOISE_MARGIN = float(os.environ.get("NOISE_MARGIN", 0.01))
AMBIG_WEIGHT = float(os.environ.get("AMBIG_WEIGHT", 0.3))
SIDE_SEL_W = float(os.environ.get("SIDE_SEL_W", 0.0))
EXACT_W = float(os.environ.get("EXACT_W", 0.0))
SIDE_W = float(os.environ.get("SIDE_W", 0.0))
AGG_W = float(os.environ.get("AGG_W", 0.0))
QUALITY_SEL = int(os.environ.get("QUALITY_SEL", 0))  # 1 = weight selection loss by frac_profitable
COMP_W = float(os.environ.get("COMP_W", 0.0))       # competence head weight on opportunity_logit
COMP_MODE = os.environ.get("COMP_MODE", "frozen")    # "frozen", "live", or "quality"
CKPT_SELECTION_MODE = os.environ.get("CKPT_SELECTION_MODE", "loss_proxy")
SEL_TARGET_MODE = os.environ.get("SEL_TARGET_MODE", "default")  # "default", "strict_mask", or "soft_pnl"
GATE_TARGET_MODE = os.environ.get("GATE_TARGET_MODE", "binary")   # "binary" (BCE) or "max_pnl" (MSE on bar max row_labels)
# Threshold shift for max_pnl gate: subtract from target so inference threshold 0 naturally gates
# profitable (>threshold) vs marginal bars. Without this, all bars have positive target → all pass.
GATE_PNL_THRESHOLD = float(os.environ.get("GATE_PNL_THRESHOLD", 0.2))
# Scale multiplier for max_pnl MSE loss — natural MSE scale ~0.05 is 10× smaller than BCE ~0.65,
# so without scaling the gate head gets negligible gradient vs selection/gate BCE paths.
GATE_PNL_LOSS_SCALE = float(os.environ.get("GATE_PNL_LOSS_SCALE", 10.0))
# Moneyness bucket boundaries for aggression head
AGG_ATM_THRESH = 0.5   # |moneyness_pct| < 0.5% = ATM
AGG_NEAR_THRESH = 1.5  # 0.5-1.5% = near-OTM, >1.5% = far-OTM

# All env vars that shape training — captured in checkpoint for provenance
_TRAINING_ENV_VARS = [
    "NUM_FEATURES", "TRAIN_LOOKBACK", "TRAIN_D_MODEL", "TRAIN_DEPTH",
    "TRAIN_DROPOUT", "TRAIN_BATCH_SIZE", "TRAIN_LR", "TRAIN_WEIGHT_DECAY",
    "TRAIN_EPOCHS", "TIME_BUDGET", "WEIGHT_SEL", "WEIGHT_GATE", "TRAIN_SEED",
    "OPP_LABEL", "SOFT_TEMP", "NOISE_MARGIN", "AMBIG_WEIGHT",
    "SIDE_SEL_W", "EXACT_W", "SIDE_W", "AGG_W", "QUALITY_SEL",
    "COMP_W", "COMP_MODE", "COMP_TEACHER", "SIDE_MODE", "ALPHA_SIDE",
    "CKPT_SELECTION_MODE", "SEL_TARGET_MODE", "GATE_TARGET_MODE",
    "GATE_PNL_THRESHOLD", "GATE_PNL_LOSS_SCALE",
    "SESSION_HISTORY_K", "ANTI_LOCKIN",
    "ENV_DECAY_COEFF", "ENV_LATE_ENTRY_BAR", "ENV_LATE_EXIT_BAR",
]


def _capture_env_overrides() -> dict:
    """Capture all training-relevant env vars that are set."""
    return {k: os.environ[k] for k in _TRAINING_ENV_VARS if k in os.environ}


def _get_config_fingerprint() -> str:
    try:
        from v2.core.config import RUNTIME_CONFIG
        return RUNTIME_CONFIG.fingerprint()
    except Exception:
        return "unknown"


@dataclass
class CheckpointCandidate:
    epoch: int
    proxy_criterion: float
    val_total: float
    val_metrics: dict[str, float]
    checkpoint_path: str


def _compute_checkpoint_proxy(avg_val: dict[str, float], best_sel_so_far: float) -> tuple[float, float]:
    """Current loss-based proxy used to decide which epochs are replay-worthy."""
    val_total = avg_val.get("total", float("inf"))
    if COMP_W > 0:
        comp_val = avg_val.get("comp", float("inf"))
        sel_val = avg_val.get("sel", float("inf"))
        next_best_sel = min(best_sel_so_far, sel_val)
        sel_ok = sel_val <= next_best_sel * 1.2  # sel must not regress >20%
        return (comp_val if sel_ok else float("inf")), next_best_sel
    if GATE_W > 0:
        return avg_val.get("gate", float("inf")), best_sel_so_far
    return val_total, best_sel_so_far


def _save_training_checkpoint(
    *,
    model: nn.Module,
    model_path: str,
    epoch: int,
    val_total: float,
    data: dict,
) -> None:
    """Serialize a training checkpoint with full provenance."""
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
            "config_fingerprint": _get_config_fingerprint(),
            "env_overrides": _capture_env_overrides(),
            "checkpoint_selection_mode": CKPT_SELECTION_MODE,
        },
        model_path,
    )


def _replay_selection_key(metrics) -> tuple[float, ...]:
    """Order validation-replay candidates by economic usefulness, not loss."""
    gate_pass = 1.0 if metrics.gate_failure is None else 0.0
    return (
        gate_pass,
        float(metrics.score) if gate_pass else 0.0,
        float(metrics.profit_factor),
        -float(metrics.max_account_drawdown),
        float(metrics.win_rate),
        float(metrics.positive_day_rate),
        -float(metrics.trades_per_day),
    )


def _build_replay_selection_policy():
    """Mirror the inference policy used by the current training env."""
    import dataclasses as _dc
    from v2.core.policy import DEFAULT_POLICY

    side_mode = os.environ.get("SIDE_MODE", DEFAULT_POLICY.side_mode)
    alpha_side = float(os.environ.get("ALPHA_SIDE", str(DEFAULT_POLICY.alpha_side)))
    if side_mode == DEFAULT_POLICY.side_mode and alpha_side == DEFAULT_POLICY.alpha_side:
        return DEFAULT_POLICY
    return _dc.replace(DEFAULT_POLICY, side_mode=side_mode, alpha_side=alpha_side)


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
        # Dual score heads: independent call and put scoring.
        # exp_170e: with SIDE_SEL_W=0.2 for within-side auxiliary supervision.
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

        opportunity_logit = self.opportunity_head(context).squeeze(-1)
        side_logit = self.side_head(context).squeeze(-1)
        aggression_logits = self.aggression_head(context)  # (B, 3)
        return {
            "contract_scores": contract_scores,
            "valid_mask": valid_mask,
            "is_put": is_put,
            "call_scores_raw": call_scores,
            "put_scores_raw": put_scores,
            "opportunity_logit": opportunity_logit,
            "side_logit": side_logit,
            "aggression_logits": aggression_logits,
            "context": context,
        }


def _compute_strict_opportunity(sc: dict, local_bar: int) -> bool:
    """Strict opportunity: at least one contract has clean, survivable path.

    Requires ALL of:
      raw_return_10bar > 0.12   (clears friction with margin)
      mae_10bar > -0.08         (bounded adverse excursion)
      bars_to_breakeven < 5     (quick confirmation)
      mfe_5bar > 0.03           (early momentum)
    """
    bar_ptrs = sc["bar_ptrs"]
    start = int(bar_ptrs[local_bar])
    end = int(bar_ptrs[local_bar + 1])
    if end <= start:
        return False
    raw_returns = sc.get("row_raw_returns")
    mfe = sc.get("row_mfe")
    mae = sc.get("row_mae")
    btbe = sc.get("row_bars_to_breakeven")
    if raw_returns is None or mfe is None or mae is None or btbe is None:
        return bool(sc["bar_label_trade"][local_bar])  # fallback to old
    IDX_5, IDX_10 = 0, 1
    for ro in range(start, end):
        r10 = float(raw_returns[ro, IDX_10])
        m10 = float(mae[ro, IDX_10])
        m5 = float(mfe[ro, IDX_5])
        bb = float(btbe[ro])
        if (np.isfinite(r10) and r10 > 0.12 and
            np.isfinite(m10) and m10 > -0.08 and
            np.isfinite(bb) and bb < 5 and
            np.isfinite(m5) and m5 > 0.03):
            return True
    return False


def _compute_contract_strict_mask(sc: dict, local_bar: int, max_contracts: int) -> np.ndarray:
    """Per-contract strict tradability mask for one bar snapshot.

    This mirrors `_compute_strict_opportunity`, but preserves *which* contracts
    satisfy the fast/clean path constraints so the scorer can rank within that
    subset instead of ranking all contracts by eventual long-hold PnL.
    """
    mask = np.zeros(max_contracts, dtype=bool)
    bar_ptrs = sc["bar_ptrs"]
    start = int(bar_ptrs[local_bar])
    end = int(bar_ptrs[local_bar + 1])
    if end <= start:
        return mask
    raw_returns = sc.get("row_raw_returns")
    mfe = sc.get("row_mfe")
    mae = sc.get("row_mae")
    btbe = sc.get("row_bars_to_breakeven")
    if raw_returns is None or mfe is None or mae is None or btbe is None:
        return mask
    idx_5, idx_10 = 0, 1
    use = min(end - start, max_contracts)
    for local_row, ro in enumerate(range(start, start + use)):
        r10 = float(raw_returns[ro, idx_10])
        m10 = float(mae[ro, idx_10])
        m5 = float(mfe[ro, idx_5])
        bb = float(btbe[ro])
        if (
            np.isfinite(r10) and r10 > 0.12 and
            np.isfinite(m10) and m10 > -0.08 and
            np.isfinite(bb) and bb < 5 and
            np.isfinite(m5) and m5 > 0.03
        ):
            mask[local_row] = True
    return mask


def _selection_target_pool(
    valid_mask: torch.Tensor,
    strict_mask: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """Which contracts should receive target mass for selection supervision.

    `strict_mask` narrows the target to fast/clean contracts on bars where they
    exist, but gracefully falls back to the full valid set otherwise. Logits are
    still normalized over all valid contracts so non-strict contracts are
    penalized if they steal probability mass.
    """
    if mode != "strict_mask" or strict_mask is None:
        return valid_mask
    strict_pool = valid_mask & strict_mask
    any_strict = strict_pool.any(dim=-1, keepdim=True)
    return torch.where(any_strict, strict_pool, valid_mask)


def _compute_consensus_opportunity(sc: dict, local_bar: int) -> bool:
    """Consensus opportunity: at least one contract profitable under all 3 policies.

    Requires the same contract to have net_pnl > 4% under DEFAULT, SHORT, and EOD
    exit policies simultaneously. This filters for bars where opportunity is
    robust to exit timing — a stronger signal that correlates with market
    conditions (high vol, trending) rather than path-dependent luck.
    """
    bar_ptrs = sc["bar_ptrs"]
    start = int(bar_ptrs[local_bar])
    end = int(bar_ptrs[local_bar + 1])
    if end <= start:
        return False
    labels_default = sc.get("row_labels")
    labels_short = sc.get("row_labels_short")
    labels_eod = sc.get("row_labels_eod")
    if labels_default is None or labels_short is None or labels_eod is None:
        return bool(sc["bar_label_trade"][local_bar])  # fallback
    for ro in range(start, end):
        d = float(labels_default[ro])
        s = float(labels_short[ro])
        e = float(labels_eod[ro])
        if (np.isfinite(d) and d > 0.04 and
            np.isfinite(s) and s > 0.04 and
            np.isfinite(e) and e > 0.04):
            return True
    return False


def _compute_frac_profitable(sc: dict, local_bar: int) -> float:
    """Fraction of executable contracts with positive PnL at this bar.

    Continuous [0, 1] target. Correlates with context features at r≈0.12
    (atm_iv, vrp, atm_gamma) — 2x stronger than any binary label variant.
    """
    bar_ptrs = sc["bar_ptrs"]
    start = int(bar_ptrs[local_bar])
    end = int(bar_ptrs[local_bar + 1])
    if end <= start:
        return 0.0
    labels = sc["row_labels"][start:end]
    valid = []
    for l in labels:
        lf = float(l)
        if np.isfinite(lf):
            valid.append(lf)
    if not valid:
        return 0.0
    return sum(1 for v in valid if v > 0) / len(valid)


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
        self.all_label_quality = torch.zeros(n, dtype=torch.float32)
        self.all_contract_strict_mask = (
            torch.zeros(n, max_contracts, dtype=torch.bool)
            if SEL_TARGET_MODE == "strict_mask" else None
        )

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
            if OPP_LABEL == "strict":
                self.all_label_trade[j] = _compute_strict_opportunity(sc, local_bar)
            elif OPP_LABEL == "consensus":
                self.all_label_trade[j] = _compute_consensus_opportunity(sc, local_bar)
            elif OPP_LABEL == "high_threshold":
                bp = sc.get("bar_best_pnl")
                self.all_label_trade[j] = bp is not None and float(bp[local_bar]) > 0.20
            else:
                self.all_label_trade[j] = bool(sc["bar_label_trade"][local_bar])
            self.all_label_trade_valid[j] = bool(sc["bar_labelable"][local_bar])
            self.all_label_quality[j] = _compute_frac_profitable(sc, local_bar)
            if self.all_contract_strict_mask is not None:
                self.all_contract_strict_mask[j] = torch.from_numpy(
                    _compute_contract_strict_mask(sc, local_bar, max_contracts)
                )
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
            "label_quality": self.all_label_quality[idx],
        }
        if self.all_contract_strict_mask is not None:
            target["contract_strict_mask"] = self.all_contract_strict_mask[idx]
        return window, self.all_contracts[idx], target


def _compute_competence_label(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    teacher_outputs: dict | None = None,
    comp_mode: str = "frozen",
    label_quality: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute competence label: can the ranker land in the top 5 contracts?

    Target: rank<=5 (absolute rank, not percentile).
    AUC 0.697, 32.8% positive rate, lift@10 2.0 — the most learnable
    competence signal available from context features.

    Returns (label_binary, label_continuous, comp_valid) where:
      label_binary: 1 if chosen contract rank <= 5, else 0
      label_continuous: normalized rank (1 = best, 0 = worst) for diagnostics
      comp_valid: mask of bars with finite chosen PnL
    """
    B = labels.size(0)

    if comp_mode == "quality" and label_quality is not None:
        label_cont = label_quality.clamp(0, 1)
        label_bin = (label_cont > 0.5).float()
        comp_valid = torch.ones(B, dtype=torch.bool, device=device)
        return label_bin, label_cont, comp_valid

    with torch.no_grad():
        if comp_mode == "frozen" and teacher_outputs is not None:
            det_scores = teacher_outputs["contract_scores"].detach().clone()
            det_valid = teacher_outputs["valid_mask"]
            det_scores[~det_valid] = -1e9
        else:  # live
            det_scores = scores.detach().clone()
            det_scores[~valid_mask] = -1e9

        model_chosen = det_scores.argmax(dim=-1)
        rows = torch.arange(B, device=device)
        chosen_pnl = labels[rows, model_chosen]

        # Rank of chosen contract among valid contracts
        valid_pnl = labels.clone()
        valid_pnl[~valid_mask] = -1e9
        valid_pnl[~torch.isfinite(valid_pnl)] = -1e9
        chosen_rank = (valid_pnl > chosen_pnl.unsqueeze(-1)).sum(dim=-1) + 1
        n_valid = valid_mask.sum(dim=-1).float().clamp(min=1)

        # Primary target: absolute top-5 rank
        label_bin = (chosen_rank <= 5).float()
        # Continuous diagnostic: normalized rank
        label_cont = (1.0 - chosen_rank.float() / n_valid).clamp(0, 1)

        comp_valid = torch.isfinite(chosen_pnl)
        label_bin[~comp_valid] = 0.0

    return label_bin, label_cont, comp_valid


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    teacher_outputs: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["contract_scores"].device
    scores = outputs["contract_scores"]
    valid_mask = outputs["valid_mask"]
    labels = targets["contract_labels"].to(device)
    best_idx = targets["best_idx"].to(device)
    label_trade = targets["label_trade"].to(device)
    label_trade_valid = targets["label_trade_valid"].to(device)
    contracts_full = targets["contracts_full"].to(device)
    strict_contract_mask = targets.get("contract_strict_mask")
    if strict_contract_mask is not None:
        strict_contract_mask = strict_contract_mask.to(device)

    label_quality = targets["label_quality"].to(device)
    supervised_rows = label_trade_valid
    trade_rows = supervised_rows & label_trade & (best_idx >= 0)
    opportunity_logit = outputs["opportunity_logit"]
    side_logit = outputs["side_logit"]

    # --- A. Gate loss: the live gate is opportunity_logit ---
    gate_loss = torch.tensor(0.0, device=device)
    if GATE_TARGET_MODE == "max_pnl" and supervised_rows.any():
        pnl_for_max = labels.clone()
        pnl_for_max = torch.where(valid_mask, pnl_for_max, torch.full_like(pnl_for_max, float("-inf")))
        pnl_for_max[~torch.isfinite(pnl_for_max) & (pnl_for_max != float("-inf"))] = float("-inf")
        bar_max_pnl, _ = pnl_for_max.max(dim=-1)
        bar_max_pnl = torch.where(torch.isfinite(bar_max_pnl), bar_max_pnl, torch.zeros_like(bar_max_pnl))
        gate_target_value = bar_max_pnl - GATE_PNL_THRESHOLD
        sup_idx = supervised_rows.nonzero(as_tuple=True)[0]
        gate_loss = GATE_PNL_LOSS_SCALE * F.mse_loss(
            opportunity_logit[sup_idx],
            gate_target_value[sup_idx],
            reduction="mean",
        )
    elif supervised_rows.any():
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
                opportunity_logit[balanced_idx],
                gate_target[balanced_idx],
                reduction="mean",
            )
        else:
            gate_loss = F.binary_cross_entropy_with_logits(
                opportunity_logit[sup_idx],
                gate_target[sup_idx],
                reduction="mean",
            )

    # --- A2. Competence loss: predict model's ranking quality ---
    comp_loss = torch.tensor(0.0, device=device)
    if COMP_W > 0 and GATE_TARGET_MODE != "max_pnl" and supervised_rows.any():
        label_comp_bin, label_comp_cont, comp_valid_mask = _compute_competence_label(
            scores, labels, valid_mask, device,
            teacher_outputs=teacher_outputs,
            comp_mode=COMP_MODE, label_quality=label_quality,
        )
        comp_eligible = supervised_rows & comp_valid_mask
        if comp_eligible.any():
            comp_idx = comp_eligible.nonzero(as_tuple=True)[0]
            comp_target = label_comp_bin[comp_idx]
            n_pos = comp_target.sum().item()
            n_neg = len(comp_target) - n_pos
            n_min = int(min(n_pos, n_neg))
            if n_min > 0:
                pos_idx = comp_idx[comp_target.bool()]
                neg_idx = comp_idx[~comp_target.bool()]
                pos_sel = pos_idx[torch.randperm(len(pos_idx), device=device)[:n_min]]
                neg_sel = neg_idx[torch.randperm(len(neg_idx), device=device)[:n_min]]
                balanced_comp = torch.cat([pos_sel, neg_sel])
                comp_loss = F.binary_cross_entropy_with_logits(
                    opportunity_logit[balanced_comp],
                    label_comp_bin[balanced_comp],
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
        tr_strict = strict_contract_mask[trade_rows] if strict_contract_mask is not None else None

        pnl_for_target = tr_labels.clone()
        target_pool = _selection_target_pool(tr_valid, tr_strict, SEL_TARGET_MODE)
        pnl_for_target[~target_pool] = -1e9
        pnl_for_target[~torch.isfinite(pnl_for_target)] = -1e9

        if tr_scores.size(0) > 0:
            if SEL_TARGET_MODE == "soft_pnl":
                # exp_174: spread target mass across all valid contracts proportional
                # to PnL at a higher effective temperature. No ambiguity filter, no
                # uniform-target blend. Relies on SOFT_TEMP being raised via env
                # (default 0.08 is effectively one-hot; use 0.5 or higher).
                target = F.softmax(pnl_for_target / SOFT_TEMP, dim=-1)
                bar_weight = torch.ones(tr_scores.size(0), device=device)
            else:
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

                # Ambiguous bars: uniform target over the active target pool
                n_valid_per_bar = target_pool.float().sum(dim=-1, keepdim=True).clamp(min=1)
                uniform_target = target_pool.float() / n_valid_per_bar

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
            # Quality weighting: emphasize bars with many profitable contracts
            if QUALITY_SEL:
                q_weight = label_quality[trade_rows].clamp(min=0.05)
                bar_weight = bar_weight * q_weight
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
    # Requires separate call/put heads; skip when using unified scorer.
    side_sel_loss = torch.tensor(0.0, device=device)
    _unified_scorer = outputs["call_scores_raw"] is outputs["put_scores_raw"]
    if SIDE_SEL_W > 0 and trade_rows.any() and not _unified_scorer:
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
             + EXACT_W * exact_loss
             + (COMP_W * comp_loss if COMP_W > 0 else 0.0)
             + SIDE_W * side_loss + AGG_W * agg_loss)

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

    # Competence accuracy + opportunity_logit distribution diagnostics
    comp_acc = 0.0
    gate_pass_rate = 0.0
    gate_mean = 0.0
    gate_std = 0.0
    if supervised_rows.any():
        with torch.no_grad():
            gate_vals = opportunity_logit[supervised_rows].detach()
            gate_pass_rate = (gate_vals > 0).float().mean().item()
            gate_mean = gate_vals.mean().item()
            gate_std = gate_vals.std().item() if gate_vals.numel() > 1 else 0.0
    if COMP_W > 0 and supervised_rows.any():
        with torch.no_grad():
            label_cb, _, cv = _compute_competence_label(
                scores, labels, valid_mask, device,
                teacher_outputs=teacher_outputs,
                comp_mode=COMP_MODE, label_quality=label_quality,
            )
            finite = supervised_rows & cv
            if finite.any():
                actual = label_cb[finite]
                pred = (opportunity_logit[finite].detach() > 0).float()
                comp_acc = (pred == actual).float().mean().item()

    return total, {
        "gate": float(gate_loss.item()),
        "sel": float(sel_loss.item()),
        "side_sel": float(side_sel_loss.item()),
        "exact": float(exact_loss.item()),
        "comp": float(comp_loss.item()),
        "side": float(side_loss.item()),
        "agg": float(agg_loss.item()),
        "total": float(total.item()),
        "gate_acc": gate_acc,
        "dir_acc": dir_acc,
        "side_acc": side_acc,
        "agg_acc": agg_acc,
        "comp_acc": comp_acc,
        "trade_rate": trade_rate,
        "gate_pass_rate": gate_pass_rate,
        "gate_mean": gate_mean,
        "gate_std": gate_std,
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

    # --- Load-time validation: check dataset matches RuntimeConfig ---
    from v2.core.config import RUNTIME_CONFIG
    meta = data.get("metadata", {})
    config_errors = RUNTIME_CONFIG.validate_dataset_metadata(meta)
    if config_errors:
        raise RuntimeError(
            f"Dataset metadata does not match RuntimeConfig:\n" +
            "\n".join(f"  - {e}" for e in config_errors)
        )

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
            "label_quality": torch.stack([t["label_quality"] for t in targets_list]),
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

    # --- Frozen teacher for competence label (COMP_MODE=frozen) ---
    global COMP_MODE  # may be reassigned to "live" if teacher not found
    teacher = None
    if COMP_W > 0 and COMP_MODE == "frozen":
        teacher_path = os.environ.get("COMP_TEACHER", "v2/models/model.pt")
        if os.path.exists(teacher_path):
            teacher_ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
            hp = teacher_ckpt.get("hyperparams", {})
            teacher = TradingModel(
                d_model=hp.get("d_model", D_MODEL),
                depth=hp.get("depth", DEPTH),
            ).to(device)
            teacher.load_state_dict(teacher_ckpt["model_state_dict"])
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            print(f"  Frozen teacher loaded from {teacher_path}")
        else:
            print(f"  WARNING: teacher {teacher_path} not found, falling back to live mode")
            COMP_MODE = "live"

    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = {"gate_accuracy": 0.0, "direction_accuracy": 0.0}
    _best_sel = float("inf")  # track best selection loss for checkpoint guardrail
    _candidate_dir = None
    _replay_candidates: list[CheckpointCandidate] = []

    # --- Training log (observability: survives lease death) ---
    from v2.core.observability import (
        generate_run_id, ensure_run_dir, identity_block,
        write_jsonl_header, append_jsonl, write_jsonl_summary,
    )
    _run_id = generate_run_id(
        experiment_id=os.environ.get("EXPERIMENT_ID"),
        stage="train",
    )
    _run_dir = ensure_run_dir(_run_id)
    _log_path = _run_dir / "training_log.jsonl"
    _identity = identity_block(
        run_id=_run_id,
        stage="train",
        dataset_fingerprint=meta.get("fingerprint", "unknown"),
    )
    write_jsonl_header(
        _log_path,
        schema_name="training_log",
        schema_version="1.0",
        producer="v2.train",
        identity=_identity,
        epochs_planned=EPOCHS,
    )
    if CKPT_SELECTION_MODE == "val_replay":
        _candidate_dir = _run_dir / "checkpoint_candidates"
        _candidate_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        if time.time() - t_start > TIME_BUDGET:
            print(f"Time budget reached at epoch {epoch}")
            break

        model.train()
        train_stats = []
        _pre_clip_norm = None
        for batch_x, batch_c, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_c = batch_c.to(device)
            batch_y = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_y.items()}

            optimizer.zero_grad()
            teacher_outputs = None
            if teacher is not None and COMP_W > 0:
                with torch.no_grad():
                    teacher_outputs = teacher(batch_x, batch_c)
            outputs = model(batch_x, batch_c)
            loss, metrics = compute_loss(outputs, batch_y, teacher_outputs=teacher_outputs)
            loss.backward()
            _pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                teacher_outputs_v = None
                if teacher is not None and COMP_W > 0:
                    teacher_outputs_v = teacher(batch_x, batch_c)
                outputs = model(batch_x, batch_c)
                _, metrics = compute_loss(outputs, batch_y, teacher_outputs=teacher_outputs_v)
                val_stats.append(metrics)

        avg_train = {k: float(np.mean([d[k] for d in train_stats])) for k in train_stats[0]} if train_stats else {}
        avg_val = {k: float(np.mean([d[k] for d in val_stats])) for k in val_stats[0]} if val_stats else {}
        _gate_line = (f" gate_pass={avg_val.get('gate_pass_rate', 0):.3f}"
                      f" gate_μ={avg_val.get('gate_mean', 0):.3f}"
                      f" gate_σ={avg_val.get('gate_std', 0):.3f}")
        _comp_line = ""
        if COMP_W > 0:
            _comp_line = f" | comp={avg_val.get('comp', 0):.4f} comp_acc={avg_val.get('comp_acc', 0):.3f}"
        print(
            f"Epoch {epoch:3d} | train={avg_train.get('total', 0):.4f} | "
            f"val={avg_val.get('total', 0):.4f} | "
            f"gate_l={avg_val.get('gate', 0):.4f} sel={avg_val.get('sel', 0):.4f} side_l={avg_val.get('side', 0):.4f} | "
            f"gate={avg_val.get('gate_acc', 0):.3f} "
            f"dir={avg_val.get('dir_acc', 0):.3f} side={avg_val.get('side_acc', 0):.3f} trd_rate={avg_val.get('trade_rate', 0):.3f}"
            f"{_gate_line}{_comp_line}"
        )

        # --- Append training log line (flush immediately) ---
        _epoch_wall = time.time() - t_start
        append_jsonl(_log_path, {
            "record_type": "epoch",
            "epoch": epoch,
            "train_loss": avg_train.get("total", 0),
            "val_loss": avg_val.get("total", 0),
            "gate_loss": avg_val.get("gate", 0),
            "sel_loss": avg_val.get("sel", 0),
            "side_loss": avg_val.get("side", 0),
            "gate_acc": avg_val.get("gate_acc", 0),
            "dir_acc": avg_val.get("dir_acc", 0),
            "side_acc": avg_val.get("side_acc", 0),
            "trade_rate": avg_val.get("trade_rate", 0),
            "comp_loss": avg_val.get("comp", 0),
            "comp_acc": avg_val.get("comp_acc", 0),
            "gate_pass_rate": avg_val.get("gate_pass_rate", 0),
            "gate_mean": avg_val.get("gate_mean", 0),
            "gate_std": avg_val.get("gate_std", 0),
            "pre_clip_grad_norm": float(_pre_clip_norm) if _pre_clip_norm is not None else None,
            "lr": optimizer.param_groups[0]["lr"],
            "wall_seconds": round(_epoch_wall, 1),
            "is_best": False,  # updated below if this becomes best
        })

        val_total = avg_val.get("total", float("inf"))
        val_criterion, _best_sel = _compute_checkpoint_proxy(avg_val, _best_sel)
        _is_proxy_best = val_criterion < best_val_loss

        if CKPT_SELECTION_MODE == "loss_proxy":
            if _is_proxy_best:
                best_val_loss = val_criterion
                best_epoch = epoch
                best_metrics = {
                    "gate_accuracy": avg_val.get("gate_acc", 0.0),
                    "direction_accuracy": avg_val.get("dir_acc", 0.0),
                    "side_accuracy": avg_val.get("side_acc", 0.0),
                }
                _save_training_checkpoint(
                    model=model,
                    model_path=model_path,
                    epoch=epoch,
                    val_total=val_total,
                    data=data,
                )
        elif CKPT_SELECTION_MODE == "val_replay":
            if _is_proxy_best:
                best_val_loss = val_criterion
                candidate_path = str(_candidate_dir / f"epoch_{epoch:03d}.pt")
                _save_training_checkpoint(
                    model=model,
                    model_path=candidate_path,
                    epoch=epoch,
                    val_total=val_total,
                    data=data,
                )
                _replay_candidates.append(CheckpointCandidate(
                    epoch=epoch,
                    proxy_criterion=val_criterion,
                    val_total=val_total,
                    val_metrics=dict(avg_val),
                    checkpoint_path=candidate_path,
                ))
        else:
            raise ValueError(
                f"Unknown CKPT_SELECTION_MODE={CKPT_SELECTION_MODE!r}. "
                f"Expected 'loss_proxy' or 'val_replay'."
            )

    if CKPT_SELECTION_MODE == "val_replay":
        if not _replay_candidates:
            raise RuntimeError("No replay-aligned checkpoint candidates were captured during training.")

        final_candidate = _replay_candidates[-1]
        if final_candidate.epoch != epoch:
            final_candidate_path = str(_candidate_dir / f"epoch_{epoch:03d}.pt")
            _save_training_checkpoint(
                model=model,
                model_path=final_candidate_path,
                epoch=epoch,
                val_total=avg_val.get("total", float("inf")),
                data=data,
            )
            _replay_candidates.append(CheckpointCandidate(
                epoch=epoch,
                proxy_criterion=float("inf"),
                val_total=avg_val.get("total", float("inf")),
                val_metrics=dict(avg_val),
                checkpoint_path=final_candidate_path,
            ))

        print(f"\n--- VALIDATION REPLAY CHECKPOINT SELECTION ({len(_replay_candidates)} candidates) ---")
        from v2.replay import load_model_from_path, replay_validation

        _val_replay_key = "_val_replay_ckpt_select"
        data[_val_replay_key] = val_mask
        _selection_policy = _build_replay_selection_policy()
        _best_candidate = None
        _best_replay_metrics = None
        try:
            for cand in _replay_candidates:
                cand_model = load_model_from_path(cand.checkpoint_path, device=device)
                replay_metrics, _, _ = replay_validation(
                    cand_model,
                    data,
                    mask_key=_val_replay_key,
                    policy=_selection_policy,
                    device=device,
                )
                print(
                    f"  epoch={cand.epoch:03d} proxy={cand.proxy_criterion:.4f} "
                    f"PF={replay_metrics.profit_factor:.3f} DD={replay_metrics.max_account_drawdown:.1%} "
                    f"score={replay_metrics.score:.4f} gate={replay_metrics.gate_failure or 'pass'}"
                )
                if (
                    _best_candidate is None
                    or _replay_selection_key(replay_metrics) > _replay_selection_key(_best_replay_metrics)
                ):
                    _best_candidate = cand
                    _best_replay_metrics = replay_metrics
        finally:
            del data[_val_replay_key]

        if _best_candidate is None or _best_replay_metrics is None:
            raise RuntimeError("Validation replay checkpoint selection produced no valid candidate.")

        shutil.copy2(_best_candidate.checkpoint_path, model_path)
        best_epoch = _best_candidate.epoch
        best_val_loss = _best_candidate.val_total
        best_metrics = {
            "gate_accuracy": _best_candidate.val_metrics.get("gate_acc", 0.0),
            "direction_accuracy": _best_candidate.val_metrics.get("dir_acc", 0.0),
            "side_accuracy": _best_candidate.val_metrics.get("side_acc", 0.0),
            "val_replay_score": _best_replay_metrics.score,
            "val_replay_profit_factor": _best_replay_metrics.profit_factor,
            "val_replay_drawdown": _best_replay_metrics.max_account_drawdown,
            "val_replay_win_rate": _best_replay_metrics.win_rate,
            "val_replay_gate_failure": _best_replay_metrics.gate_failure or "",
        }
        print(
            f"Selected epoch {best_epoch} by validation replay: "
            f"PF={_best_replay_metrics.profit_factor:.3f} "
            f"DD={_best_replay_metrics.max_account_drawdown:.1%} "
            f"score={_best_replay_metrics.score:.4f} "
            f"gate={_best_replay_metrics.gate_failure or 'pass'}"
        )

    metrics = {
        "val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_run": min(epoch, EPOCHS),
        "score_config_fingerprint": score_config_fingerprint(),
        "checkpoint_selection_mode": CKPT_SELECTION_MODE,
        **best_metrics,
    }
    print(f"\nMETRICS_JSON:{json.dumps(metrics)}")
    print(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")

    # --- Training log summary ---
    _convergence_flag = "early_best" if best_epoch <= max(1, EPOCHS * 0.2) else "normal"
    write_jsonl_summary(
        _log_path,
        status="completed",
        best_epoch=best_epoch,
        total_epochs=min(epoch, EPOCHS),
        best_val_loss=best_val_loss,
        checkpoint_selection_mode=CKPT_SELECTION_MODE,
        convergence_flag=_convergence_flag,
        run_dir=str(_run_dir),
    )
    print(f"Training log: {_log_path}")

    return model, metrics


if __name__ == "__main__":
    _data = os.environ.get("TRAIN_DATA_PATH", "v2/data.pt")
    _model = os.environ.get("TRAIN_MODEL_PATH", "v2/models/model.pt")
    train(data_path=_data, model_path=_model)
