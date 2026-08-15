"""Unfitted compact entry policy with explicit state-by-contract interactions.

This is a design artifact, not fit authorization.  Job 39's ``ContractHead``
adds one state-dependent offset to every contract in a minute, so chart state
cannot change the preferred side or strike.  This policy makes that interaction
explicit while staying inside the conservative 29--50 parameter budget.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from v5.research.causal_day_architectures import (
    ArchitectureInputError,
    ArchitectureScores,
    CausalPolicyBatch,
    ENTRY_ROLES,
    trainable_parameter_count,
)
from v5.research.causal_day_tensorizer import CANDLE_FEATURES, LADDER_FEATURES


ARCHITECTURE_NAME = "compact_interaction_entry"

# Eight last-completed-candle fields describe the candle and the day's visible
# structure.  The five already-causal clock fields retain the morning/afternoon
# distinction, including the 12:46 router, without creating specialist models.
STATE_CANDLE_FEATURES = (
    "body_points",
    "upper_wick_points",
    "lower_wick_points",
    "close_from_session_open_points",
    "close_from_running_high_points",
    "close_from_running_low_points",
    "range_position",
    "return_1m",
)
STATE_CANDLE_INDICES = tuple(CANDLE_FEATURES.index(name) for name in STATE_CANDLE_FEATURES)
STATE_CLOCK_INDICES = tuple(range(5))

# These are contemporaneous option terms, not forward outcomes.  The base score
# ranks ordinary tradeability/geometry; the two explicit products below allow
# visible chart state to change side and OTM-depth ordering.
CONTRACT_BASE_FEATURES = (
    "ask",
    "spread",
    "self_iv",
    "self_gamma",
    "self_theta_per_minute",
)
CONTRACT_BASE_INDICES = tuple(LADDER_FEATURES.index(name) for name in CONTRACT_BASE_FEATURES)
SIDE_INDEX = LADDER_FEATURES.index("is_call")
MONEYNESS_INDEX = LADDER_FEATURES.index("moneyness_itm_points")


class CompactInteractionEntryPolicy(nn.Module):
    """A 48-parameter wait/call/put selector for one causal minute.

    ``direction_state * is_call`` means changing the tape can reverse call/put
    ordering. ``depth_state * moneyness`` means the same state can change the
    desired OTM depth.  ``wait`` is an explicit trainable action; no external
    prediction cutoff is part of this architecture.

    The module intentionally supports entry roles only.  Exit learning remains
    a later, separately gated stage owned by the regime that opened the trade.
    """

    def __init__(self) -> None:
        super().__init__()
        state_width = len(STATE_CANDLE_INDICES) + len(STATE_CLOCK_INDICES)
        self.direction_state = nn.Linear(state_width, 1)
        self.depth_state = nn.Linear(state_width, 1)
        self.wait = nn.Linear(state_width, 1)
        self.contract_base = nn.Linear(len(CONTRACT_BASE_INDICES), 1)

    def _state(self, batch: CausalPolicyBatch) -> Tensor:
        if any(role not in ENTRY_ROLES for role in batch.roles):
            raise ArchitectureInputError(f"{ARCHITECTURE_NAME} accepts entry roles only")
        if batch.candles.shape[-1] != len(CANDLE_FEATURES):
            raise ArchitectureInputError("compact policy requires the declared candle tensor")
        if batch.ladder.shape[-1] != len(LADDER_FEATURES):
            raise ArchitectureInputError("compact policy requires the declared ladder tensor")
        if batch.clock.shape[-1] != len(STATE_CLOCK_INDICES):
            raise ArchitectureInputError("compact policy requires all five declared clock fields")
        if not batch.candle_mask.any(dim=1).all():
            raise ArchitectureInputError("every row requires a completed candle")
        last = batch.candle_mask.sum(dim=1) - 1
        candle = batch.candles[
            torch.arange(batch.batch_size, device=last.device), last
        ][:, STATE_CANDLE_INDICES]
        return torch.cat((candle, batch.clock[:, STATE_CLOCK_INDICES]), dim=1)

    def forward(self, batch: CausalPolicyBatch) -> ArchitectureScores:
        state = self._state(batch)
        contract = batch.ladder[:, :, CONTRACT_BASE_INDICES]
        base = self.contract_base(contract).squeeze(-1)
        direction = self.direction_state(state)
        depth = self.depth_state(state)
        side = batch.ladder[:, :, SIDE_INDEX]
        moneyness = batch.ladder[:, :, MONEYNESS_INDEX]
        logits = base + direction * side + depth * moneyness
        logits = logits.masked_fill(~batch.entry_action_mask, -torch.inf)
        abstain = self.wait(state).squeeze(-1)
        exits = state.new_full((batch.batch_size, 2), -torch.inf)
        return ArchitectureScores(logits, abstain, exits, batch.roles, ARCHITECTURE_NAME)


def computed_parameter_count() -> int:
    """Build and count; never transcribe the evidence-budget claim."""

    return trainable_parameter_count(CompactInteractionEntryPolicy())
