"""Sequential session-level trading agent.

Uses a frozen TradingModel as feature extractor and adds a policy/value
head that conditions on session state (position, P&L, trade history).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.core.env import SESSION_STATE_DIM, NUM_ACTIONS


class SequentialAgent(nn.Module):
    """Actor-critic agent for session-level 0DTE trading.

    The frozen encoder (TradingModel) produces a 96-dim context vector
    and per-contract scores. The agent adds session state awareness and
    outputs action probabilities + value estimate.
    """

    def __init__(
        self,
        encoder: nn.Module,
        context_dim: int = 96,
        session_dim: int = SESSION_STATE_DIM,
        hidden_dim: int = 64,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.session_proj = nn.Sequential(
            nn.Linear(session_dim, 32),
            nn.GELU(),
        )

        combined_dim = context_dim + 32
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, NUM_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        context_window: torch.Tensor,
        contracts: torch.Tensor,
        session_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            context_window: (B, lookback, num_features)
            contracts: (B, max_contracts, num_contract_features)
            session_state: (B, SESSION_STATE_DIM)

        Returns:
            dict with action_logits, action_probs, value, contract_scores, context
        """
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            enc_out = self.encoder(context_window, contracts)

        context = enc_out["context"]  # (B, 96)
        contract_scores = enc_out["contract_scores"]  # (B, max_contracts)

        session_emb = self.session_proj(session_state)  # (B, 32)
        combined = torch.cat([context, session_emb], dim=-1)  # (B, 128)

        action_logits = self.policy_head(combined)  # (B, 4)
        value = self.value_head(combined).squeeze(-1)  # (B,)

        return {
            "action_logits": action_logits,
            "action_probs": F.softmax(action_logits, dim=-1),
            "value": value,
            "contract_scores": contract_scores,
            "context": context,
            # Pass through diagnostics from encoder
            "opportunity_logit": enc_out.get("opportunity_logit"),
            "side_logit": enc_out.get("side_logit"),
            "aggression_logits": enc_out.get("aggression_logits"),
        }

    def act(
        self,
        context_window: torch.Tensor,
        contracts: torch.Tensor,
        session_state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Select an action for a single observation.

        Returns: (action, log_prob, value)
        """
        out = self.forward(
            context_window.unsqueeze(0),
            contracts.unsqueeze(0),
            session_state.unsqueeze(0),
        )
        logits = out["action_logits"][0]
        value = out["value"][0].item()

        if deterministic:
            action = int(logits.argmax().item())
            log_prob = F.log_softmax(logits, dim=-1)[action].item()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item())
            log_prob = dist.log_prob(action_t).item()

        return action, log_prob, value
