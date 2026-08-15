"""Unfitted shared lifecycle policy for one long SPXW 0DTE option.

The module is intentionally not registered in the fit gate.  It exists to make
the same-game data decision concrete: a built entry-and-exit representation
must fit the projected conservative evidence budget before quote acquisition
can claim it enables a model.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from v5.research.causal_day_architectures import (
    ArchitectureInputError,
    ArchitectureScores,
    CausalPolicyBatch,
    trainable_parameter_count,
)
from v5.research.causal_day_compact_interaction import (
    CONTRACT_BASE_INDICES,
    MONEYNESS_INDEX,
    SIDE_INDEX,
    STATE_CANDLE_INDICES,
)
from v5.research.causal_day_tensorizer import (
    ACCOUNT_FEATURES,
    CANDLE_FEATURES,
    CLOCK_FEATURES,
    LADDER_FEATURES,
    POSITION_FEATURES,
)


ARCHITECTURE_NAME = "compact_shared_lifecycle"
HIDDEN_SIZE = 3
MAX_PROJECTED_CONSERVATIVE_PARAMETERS = 122

# Mean and maximum of these five fields preserve a cheap, permutation-invariant
# view of the whole live ladder, including contracts that are context-only.
LADDER_CONTEXT_FEATURES = (
    "spread",
    "self_iv",
    "self_gamma",
    "self_theta_per_minute",
    "moneyness_itm_points",
)
LADDER_CONTEXT_INDICES = tuple(
    LADDER_FEATURES.index(name) for name in LADDER_CONTEXT_FEATURES
)
ACCOUNT_INDICES = (0, 3)  # cash fraction and remaining trade-cap fraction
POSITION_INDICES = (1, 2, 3, 4, 5, 6, 7, 9)


def _masked_mean_max(values: Tensor, mask: Tensor) -> Tensor:
    visible = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
    count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(values.dtype)
    mean = visible.sum(dim=1) / count
    negative = torch.finfo(values.dtype).min
    maximum = torch.where(mask.unsqueeze(-1), values, negative).max(dim=1).values
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=1)


class CompactSharedLifecyclePolicy(nn.Module):
    """One shared entry/exit model with explicit time and origin state."""

    def __init__(self) -> None:
        super().__init__()
        state_width = (
            len(STATE_CANDLE_INDICES)
            + CLOCK_FEATURES
            + 2 * len(LADDER_CONTEXT_INDICES)
            + len(ACCOUNT_INDICES)
        )
        self.shared_state = nn.Linear(state_width, HIDDEN_SIZE)
        self.contract_base = nn.Linear(len(CONTRACT_BASE_INDICES), 1)
        self.direction = nn.Linear(HIDDEN_SIZE, 1)
        self.depth = nn.Linear(HIDDEN_SIZE, 1)
        self.wait = nn.Linear(HIDDEN_SIZE, 1)
        self.exit = nn.Linear(HIDDEN_SIZE + len(POSITION_INDICES), 2)

    def _validate(self, batch: CausalPolicyBatch) -> None:
        expected = (
            len(CANDLE_FEATURES),
            len(LADDER_FEATURES),
            ACCOUNT_FEATURES,
            POSITION_FEATURES,
            CLOCK_FEATURES,
        )
        actual = (
            batch.candles.shape[-1],
            batch.ladder.shape[-1],
            batch.account.shape[-1],
            batch.position.shape[-1],
            batch.clock.shape[-1],
        )
        if actual != expected:
            raise ArchitectureInputError(
                f"{ARCHITECTURE_NAME} requires canonical dimensions {expected}, got {actual}"
            )
        if not batch.candle_mask.any(dim=1).all():
            raise ArchitectureInputError("every row requires a completed candle")
        if (batch.entry_action_mask & ~batch.ladder_mask).any():
            raise ArchitectureInputError("entry actions must be visible ladder nodes")
        if len(batch.roles) != batch.batch_size:
            raise ArchitectureInputError("every row requires one routed role")

    def _state(self, batch: CausalPolicyBatch) -> Tensor:
        self._validate(batch)
        last = batch.candle_mask.sum(dim=1) - 1
        candle = batch.candles[
            torch.arange(batch.batch_size, device=last.device), last
        ][:, STATE_CANDLE_INDICES]
        ladder = batch.ladder[:, :, LADDER_CONTEXT_INDICES]
        ladder_context = _masked_mean_max(ladder, batch.ladder_mask)
        state = torch.cat(
            (
                candle,
                batch.clock,
                ladder_context,
                batch.account[:, ACCOUNT_INDICES],
            ),
            dim=1,
        )
        return torch.tanh(self.shared_state(state))

    def forward(self, batch: CausalPolicyBatch) -> ArchitectureScores:
        state = self._state(batch)
        contract = batch.ladder[:, :, CONTRACT_BASE_INDICES]
        base = self.contract_base(contract).squeeze(-1)
        side = batch.ladder[:, :, SIDE_INDEX]
        moneyness = batch.ladder[:, :, MONEYNESS_INDEX]
        contract_logits = (
            base + self.direction(state) * side + self.depth(state) * moneyness
        ).masked_fill(~batch.entry_action_mask, -torch.inf)
        abstain = self.wait(state).squeeze(-1)
        exit_state = torch.cat((state, batch.position[:, POSITION_INDICES]), dim=1)
        exits = self.exit(exit_state)
        return ArchitectureScores(
            contract_logits,
            abstain,
            exits,
            batch.roles,
            ARCHITECTURE_NAME,
        )


def computed_parameter_count() -> int:
    """Build and count the proposed member; no transcribed count."""

    return trainable_parameter_count(CompactSharedLifecyclePolicy())
