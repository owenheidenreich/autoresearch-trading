"""Policy/value network for v3 PPO trading."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from shared.chain_data import NUM_CONTRACT_FEATURES
from shared.features import NUM_FEATURES
from v3.core.market_state import FIVE_MINUTE_FEATURE_NAMES
from v3.core.schema import (
    EXIT_STYLES,
    MANAGE_MODES,
    LOOKBACK_1M,
    LOOKBACK_5M,
    SESSION_STATE_FEATURE_NAMES,
    TRADE_STATE_FEATURE_NAMES,
    AgentAction,
)


LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0
ENTRY_CONT_DIM = 4
ADJUST_CONT_DIM = 4
TOTAL_CONT_DIM = ENTRY_CONT_DIM + ADJUST_CONT_DIM


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _tanh_normal_log_prob(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    std = log_std.exp()
    pre_tanh = _atanh(action)
    base = Normal(mean, std).log_prob(pre_tanh)
    correction = torch.log(1 - action.pow(2) + 1e-6)
    return base - correction


def observation_to_tensors(obs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    """Convert PolicyObservation into a batched tensor dict."""

    return {
        "context_1m": torch.as_tensor(obs.context_1m, dtype=torch.float32, device=device).unsqueeze(0),
        "context_5m": torch.as_tensor(obs.context_5m, dtype=torch.float32, device=device).unsqueeze(0),
        "session_state": torch.as_tensor(obs.session_state, dtype=torch.float32, device=device).unsqueeze(0),
        "trade_state": torch.as_tensor(obs.trade_state, dtype=torch.float32, device=device).unsqueeze(0),
        "contracts": torch.as_tensor(obs.contract_snapshot, dtype=torch.float32, device=device).unsqueeze(0),
        "contract_mask": torch.as_tensor(obs.valid_mask, dtype=torch.bool, device=device).unsqueeze(0),
        "account": torch.as_tensor(obs.account_vector(), dtype=torch.float32, device=device).unsqueeze(0),
        "position_contract": torch.as_tensor(obs.position_contract_features, dtype=torch.float32, device=device).unsqueeze(0),
        "in_position": torch.tensor([[1.0 if obs.position.in_position else 0.0]], dtype=torch.float32, device=device),
    }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, *, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class RLPolicy(nn.Module):
    """Multi-stream transformer encoder + pointer contract policy + PPO value head."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        depth: int = 3,
        n_heads: int = 4,
        dropout: float = 0.10,
        account_dim: int = 6,
        context_5m_dim: int = len(FIVE_MINUTE_FEATURE_NAMES),
        session_dim: int = len(SESSION_STATE_FEATURE_NAMES),
        trade_dim: int = len(TRADE_STATE_FEATURE_NAMES),
    ) -> None:
        super().__init__()
        self.context_1m_proj = nn.Linear(NUM_FEATURES, d_model)
        self.context_1m_norm = nn.LayerNorm(d_model)
        self.context_1m_pos = PositionalEncoding(d_model, max_len=LOOKBACK_1M + 10)
        self.context_5m_proj = nn.Linear(context_5m_dim, d_model)
        self.context_5m_norm = nn.LayerNorm(d_model)
        self.context_5m_pos = PositionalEncoding(d_model, max_len=LOOKBACK_5M + 10)

        layer_1m = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        layer_5m = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, n_heads // 2),
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder_1m = nn.TransformerEncoder(layer_1m, num_layers=depth)
        self.encoder_5m = nn.TransformerEncoder(layer_5m, num_layers=max(1, depth - 1))
        self.register_buffer("causal_mask_1m", nn.Transformer.generate_square_subsequent_mask(LOOKBACK_1M))
        self.register_buffer("causal_mask_5m", nn.Transformer.generate_square_subsequent_mask(LOOKBACK_5M))

        self.state_proj = nn.Sequential(
            nn.Linear(account_dim + session_dim + trade_dim + NUM_CONTRACT_FEATURES, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.contract_proj = nn.Sequential(
            nn.Linear(NUM_CONTRACT_FEATURES, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.shared = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.contract_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.flat_head = nn.Linear(d_model, 2)
        self.manage_head = nn.Linear(d_model, len(MANAGE_MODES))
        self.exit_style_head = nn.Linear(d_model, len(EXIT_STYLES))
        self.cont_mean_head = nn.Linear(d_model, TOTAL_CONT_DIM)
        self.cont_logstd_head = nn.Linear(d_model, TOTAL_CONT_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context_1m = self.context_1m_proj(obs["context_1m"])
        context_1m = self.context_1m_norm(context_1m)
        context_1m = self.context_1m_pos(context_1m)
        seq_1m = context_1m.shape[1]
        enc_1m = self.encoder_1m(context_1m, mask=self.causal_mask_1m[:seq_1m, :seq_1m])
        ctx_1m = enc_1m[:, -1, :]

        context_5m = self.context_5m_proj(obs["context_5m"])
        context_5m = self.context_5m_norm(context_5m)
        context_5m = self.context_5m_pos(context_5m)
        seq_5m = context_5m.shape[1]
        enc_5m = self.encoder_5m(context_5m, mask=self.causal_mask_5m[:seq_5m, :seq_5m])
        ctx_5m = enc_5m[:, -1, :]

        state_in = torch.cat(
            [obs["account"], obs["session_state"], obs["trade_state"], obs["position_contract"]],
            dim=-1,
        )
        state = self.state_proj(state_in)
        shared = self.shared(torch.cat([ctx_1m, ctx_5m, state], dim=-1))

        contract_emb = self.contract_proj(obs["contracts"])
        shared_exp = shared.unsqueeze(1).expand(-1, contract_emb.shape[1], -1)
        contract_scores = self.contract_head(torch.cat([shared_exp, contract_emb], dim=-1)).squeeze(-1)

        log_std = self.cont_logstd_head(shared).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return {
            "contract_scores": contract_scores,
            "flat_logits": self.flat_head(shared),
            "manage_logits": self.manage_head(shared),
            "exit_style_logits": self.exit_style_head(shared),
            "cont_mean": self.cont_mean_head(shared),
            "cont_log_std": log_std,
            "value": self.value_head(shared).squeeze(-1),
        }

    def _masked_contract_logits(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = scores.clone()
        logits[~mask] = -1e9
        all_invalid = ~mask.any(dim=-1)
        if all_invalid.any():
            logits[all_invalid, 0] = 0.0
        return logits

    def _sample_continuous(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        if deterministic:
            pre = mean
        else:
            pre = mean + std * torch.randn_like(std)
        action = torch.tanh(pre)
        log_prob = _tanh_normal_log_prob(action, mean, log_std).sum(dim=-1)
        entropy = Normal(mean, std).entropy().sum(dim=-1)
        return action, log_prob, entropy

    def act(
        self,
        obs: dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        out = self.forward(obs)
        in_position = obs["in_position"].squeeze(-1) > 0.5

        flat_dist = Categorical(logits=out["flat_logits"])
        manage_dist = Categorical(logits=out["manage_logits"])
        exit_style_dist = Categorical(logits=out["exit_style_logits"])
        contract_logits = self._masked_contract_logits(out["contract_scores"], obs["contract_mask"])
        contract_dist = Categorical(logits=contract_logits)
        cont_action, cont_log_prob, cont_entropy = self._sample_continuous(
            out["cont_mean"],
            out["cont_log_std"],
            deterministic=deterministic,
        )

        if deterministic:
            flat_action = flat_dist.probs.argmax(dim=-1)
            manage_action = manage_dist.probs.argmax(dim=-1)
            exit_style = exit_style_dist.probs.argmax(dim=-1)
            contract_idx = contract_logits.argmax(dim=-1)
        else:
            flat_action = flat_dist.sample()
            manage_action = manage_dist.sample()
            exit_style = exit_style_dist.sample()
            contract_idx = contract_dist.sample()

        total_log_prob = torch.zeros_like(out["value"])
        total_entropy = torch.zeros_like(out["value"])

        flat_rows = ~in_position
        if flat_rows.any():
            total_log_prob[flat_rows] += flat_dist.log_prob(flat_action)[flat_rows]
            total_entropy[flat_rows] += flat_dist.entropy()[flat_rows]
            open_rows = flat_rows & (flat_action == 1)
            if open_rows.any():
                total_log_prob[open_rows] += contract_dist.log_prob(contract_idx)[open_rows]
                total_log_prob[open_rows] += exit_style_dist.log_prob(exit_style)[open_rows]
                total_log_prob[open_rows] += cont_log_prob[open_rows]
                total_entropy[open_rows] += contract_dist.entropy()[open_rows]
                total_entropy[open_rows] += exit_style_dist.entropy()[open_rows]
                total_entropy[open_rows] += cont_entropy[open_rows]

        live_rows = in_position
        if live_rows.any():
            total_log_prob[live_rows] += manage_dist.log_prob(manage_action)[live_rows]
            total_entropy[live_rows] += manage_dist.entropy()[live_rows]
            adjust_rows = live_rows & (manage_action == 2)
            if adjust_rows.any():
                total_log_prob[adjust_rows] += cont_log_prob[adjust_rows]
                total_entropy[adjust_rows] += cont_entropy[adjust_rows]

        action = {
            "flat_action": flat_action.unsqueeze(-1),
            "manage_action": manage_action.unsqueeze(-1),
            "contract_idx": contract_idx.unsqueeze(-1),
            "exit_style": exit_style.unsqueeze(-1),
            "continuous": cont_action,
        }
        aux = {
            "value": out["value"],
            "flat_probs": flat_dist.probs,
            "manage_probs": manage_dist.probs,
            "exit_style_probs": exit_style_dist.probs,
            "contract_logits": contract_logits,
            "entropy": total_entropy,
        }
        return action, total_log_prob, out["value"], aux

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(obs)
        in_position = obs["in_position"].squeeze(-1) > 0.5
        flat_action = actions["flat_action"].squeeze(-1)
        manage_action = actions["manage_action"].squeeze(-1)
        contract_idx = actions["contract_idx"].squeeze(-1)
        exit_style = actions["exit_style"].squeeze(-1)
        cont_action = actions["continuous"]

        flat_dist = Categorical(logits=out["flat_logits"])
        manage_dist = Categorical(logits=out["manage_logits"])
        exit_style_dist = Categorical(logits=out["exit_style_logits"])
        contract_logits = self._masked_contract_logits(out["contract_scores"], obs["contract_mask"])
        contract_dist = Categorical(logits=contract_logits)
        cont_log_prob = _tanh_normal_log_prob(cont_action, out["cont_mean"], out["cont_log_std"]).sum(dim=-1)
        cont_entropy = Normal(out["cont_mean"], out["cont_log_std"].exp()).entropy().sum(dim=-1)

        total_log_prob = torch.zeros_like(out["value"])
        total_entropy = torch.zeros_like(out["value"])

        flat_rows = ~in_position
        if flat_rows.any():
            total_log_prob[flat_rows] += flat_dist.log_prob(flat_action)[flat_rows]
            total_entropy[flat_rows] += flat_dist.entropy()[flat_rows]
            open_rows = flat_rows & (flat_action == 1)
            if open_rows.any():
                total_log_prob[open_rows] += contract_dist.log_prob(contract_idx)[open_rows]
                total_log_prob[open_rows] += exit_style_dist.log_prob(exit_style)[open_rows]
                total_log_prob[open_rows] += cont_log_prob[open_rows]
                total_entropy[open_rows] += contract_dist.entropy()[open_rows]
                total_entropy[open_rows] += exit_style_dist.entropy()[open_rows]
                total_entropy[open_rows] += cont_entropy[open_rows]

        live_rows = in_position
        if live_rows.any():
            total_log_prob[live_rows] += manage_dist.log_prob(manage_action)[live_rows]
            total_entropy[live_rows] += manage_dist.entropy()[live_rows]
            adjust_rows = live_rows & (manage_action == 2)
            if adjust_rows.any():
                total_log_prob[adjust_rows] += cont_log_prob[adjust_rows]
                total_entropy[adjust_rows] += cont_entropy[adjust_rows]

        return total_log_prob, total_entropy, out["value"]

    def to_agent_action(
        self,
        obs: Any,
        action: dict[str, torch.Tensor],
        aux: dict[str, torch.Tensor],
    ) -> AgentAction:
        in_position = bool(obs.position.in_position)
        cont = action["continuous"][0].detach().cpu().tolist()
        flat_choice = int(action["flat_action"][0].item())
        manage_choice = int(action["manage_action"][0].item())
        contract_row = int(action["contract_idx"][0].item())
        exit_style_idx = int(action["exit_style"][0].item())
        exit_style = EXIT_STYLES[exit_style_idx]

        if not in_position:
            open_prob = float(aux["flat_probs"][0, flat_choice].item())
            if flat_choice == 0:
                return AgentAction.noop()
            contract_prob = float(torch.softmax(aux["contract_logits"][0], dim=-1)[contract_row].item())
            confidence = max(0.0, min(1.0, open_prob * contract_prob))
            return AgentAction(
                action_type="OPEN",
                contract_row=contract_row,
                exit_style=exit_style,
                risk_budget_frac=cont[0],
                stop_frac=cont[1],
                target_frac=cont[2],
                time_stop_frac=cont[3],
                updated_stop_frac=cont[4],
                updated_target_frac=cont[5],
                updated_time_stop_frac=cont[6],
                size_delta_frac=cont[7],
                confidence=confidence,
            )

        manage_mode = MANAGE_MODES[manage_choice]
        manage_prob = float(aux["manage_probs"][0, manage_choice].item())
        if manage_mode == "HOLD":
            return AgentAction(action_type="HOLD", manage_mode=manage_mode, confidence=manage_prob)
        if manage_mode == "CLOSE":
            return AgentAction(action_type="CLOSE", manage_mode=manage_mode, confidence=manage_prob)
        return AgentAction(
            action_type="ADJUST",
            manage_mode=manage_mode,
            exit_style=obs.position.exit_style,
            updated_stop_frac=cont[4],
            updated_target_frac=cont[5],
            updated_time_stop_frac=cont[6],
            size_delta_frac=cont[7],
            confidence=manage_prob,
        )
