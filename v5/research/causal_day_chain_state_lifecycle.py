"""Declared lifecycle member carrying the chain-internal feature contract.

**A new member, never an edit.** `causal_day_compact_shared_lifecycle.py` is
hash-pinned by `PREACQUISITION_SEMANTIC_FREEZE_V1`, whose post-contact rule
forbids changing a listed source and directs development iterations through the
alpha ledger instead. This is that iteration: the frozen baseline is left byte
-identical and this sits beside it.

**What differs from the baseline, and why.** The baseline spends its state width
on eight chart channels and ten ladder-context aggregates. The 375-cell census
(ledger row 338) measured chart state to be uninformative about buying premium,
so carrying eight collinear copies of a dead signal is capacity spent for
nothing. This member trades them for the one information family this corpus owns
and no fitted model has ever read as *state*: the option chain's own internals.

The exchange is capacity-neutral by measurement — 118 parameters against the
baseline's 120 — so it cannot be refused on budget grounds relative to the member
already declared.

**Two fields are barred deliberately, and one bar was corrected by measurement.**
`self_gamma` is excluded as geometry: for a 0DTE contract it is near-deterministic
in moneyness, IV and clock, and it is the family V5 became a sensor for. Raw
whole-chain extent fields are excluded as era detectors. The pre-fit review of
2026-08-18 then found that bar aimed at the wrong hazard — the corpus ladder is a
fixed +/-25-point band, so nothing here is monotone in ladder size, while an
index-normalised field walked straight past it. The generalised rule is therefore
**no field monotone in any slowly-varying calendar quantity**, verified by the
era probe rather than asserted; `implied_spot_dispersion_ratio` is the repaired
form of the field that failed it.

Nothing here is fitted. Building the module is how its parameter count is
established -- the 2026-08-14 ruling bars transcribed counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from v5.research.causal_day_architectures import (
    ArchitectureInputError,
    ArchitectureScores,
    CausalPolicyBatch,
    trainable_parameter_count,
)
from v5.research.causal_day_tensorizer import (
    ACCOUNT_FEATURES,
    CANDLE_FEATURES,
    CLOCK_FEATURES,
    LADDER_FEATURES,
    POSITION_FEATURES,
)
from v5.research.chain_internal_features import (
    CHAIN_STATE_FEATURES,
    CONTRACT_CHAIN_FEATURES,
)

@dataclass(frozen=True)
class ChainPolicyBatch(CausalPolicyBatch):
    """A batch carrying minute-common chain-internal state.

    Declared here rather than by widening `CausalPolicyBatch`, because that class
    lives in `causal_day_architectures.py`, which is **hash-pinned by the sealed
    V3 action-value declarations**. Adding a field there — even an optional one
    with a default — changes the file's digest and breaks the reseal guard that
    proves those declarations still describe the code that ran. Subclassing keeps
    the seal byte-intact and gives this member the field it needs.
    """

    chain: Tensor | None = None


ARCHITECTURE_NAME = "chain_state_lifecycle"
HIDDEN_SIZE = 3

# Four tape channels, down from eight. One lookback survives from a collinear
# family of seven; candle anatomy and the redundant running-extreme distances go.
STATE_CANDLE_FEATURES = (
    "close_from_session_open_points",
    "range_position",
    "return_1m",
    "realised_vol_15m",
)
STATE_CANDLE_INDICES = tuple(CANDLE_FEATURES.index(name) for name in STATE_CANDLE_FEATURES)

# Mean and max of three fields, down from five. `self_gamma` is barred as
# geometry; `moneyness_itm_points` aggregates are barred because their max is the
# ladder's edge.
LADDER_CONTEXT_FEATURES = ("spread", "self_iv", "self_theta_per_minute")
LADDER_CONTEXT_INDICES = tuple(
    LADDER_FEATURES.index(name) for name in LADDER_CONTEXT_FEATURES
)

# Contract base widens 5 -> 6: `self_gamma` out, two informational terms in.
# `smile_residual` is the market disagreeing with its own surface about one
# strike; `contract_depth_imbalance` is who is leaning on it. Both are
# per-contract, so both reorder the ladder directly rather than through the state.
CONTRACT_BASE_FEATURES = ("ask", "spread", "self_iv", "self_theta_per_minute")
CONTRACT_BASE_INDICES = tuple(
    LADDER_FEATURES.index(name) for name in CONTRACT_BASE_FEATURES
)
SIDE_INDEX = LADDER_FEATURES.index("is_call")
MONEYNESS_INDEX = LADDER_FEATURES.index("moneyness_itm_points")

ACCOUNT_INDICES = (0, 3)
POSITION_INDICES = (1, 2, 3, 4, 5, 6, 7, 9)

STATE_WIDTH = (
    len(STATE_CANDLE_INDICES)
    + CLOCK_FEATURES
    + len(CHAIN_STATE_FEATURES)
    + 2 * len(LADDER_CONTEXT_INDICES)
    + len(ACCOUNT_INDICES)
)
CONTRACT_WIDTH = len(CONTRACT_BASE_INDICES) + len(CONTRACT_CHAIN_FEATURES)


def _masked_mean_max(values: Tensor, mask: Tensor) -> Tensor:
    visible = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
    count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(values.dtype)
    mean = visible.sum(dim=1) / count
    negative = torch.finfo(values.dtype).min
    maximum = torch.where(mask.unsqueeze(-1), values, negative).max(dim=1).values
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=1)


class ChainStateLifecyclePolicy(nn.Module):
    """One shared entry/exit model whose state carries chain internals."""

    def __init__(self) -> None:
        super().__init__()
        self.shared_state = nn.Linear(STATE_WIDTH, HIDDEN_SIZE)
        self.contract_base = nn.Linear(CONTRACT_WIDTH, 1)
        self.direction = nn.Linear(HIDDEN_SIZE, 1)
        self.depth = nn.Linear(HIDDEN_SIZE, 1)
        self.wait = nn.Linear(HIDDEN_SIZE, 1)
        self.exit = nn.Linear(HIDDEN_SIZE + len(POSITION_INDICES), 2)

    def _validate(self, batch: CausalPolicyBatch) -> None:
        expected = (
            len(CANDLE_FEATURES),
            len(LADDER_FEATURES) + len(CONTRACT_CHAIN_FEATURES),
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
        # A batch without chain state is refused rather than zero-filled: training
        # this member on silently absent state is the failure it exists to avoid.
        if batch.chain is None:
            raise ArchitectureInputError(
                f"{ARCHITECTURE_NAME} requires chain-internal state; batch.chain is None"
            )
        if batch.chain.shape != (batch.batch_size, len(CHAIN_STATE_FEATURES)):
            raise ArchitectureInputError(
                f"chain state must be ({batch.batch_size}, {len(CHAIN_STATE_FEATURES)}), "
                f"got {tuple(batch.chain.shape)}"
            )
        if not torch.isfinite(batch.chain).all():
            raise ArchitectureInputError(
                "chain state contains non-finite values; fold-fitted imputation is required"
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
        ladder_context = _masked_mean_max(
            batch.ladder[:, :, LADDER_CONTEXT_INDICES], batch.ladder_mask
        )
        state = torch.cat(
            (
                candle,
                batch.clock,
                batch.chain,
                ladder_context,
                batch.account[:, ACCOUNT_INDICES],
            ),
            dim=1,
        )
        return torch.tanh(self.shared_state(state))

    def forward(self, batch: CausalPolicyBatch) -> ArchitectureScores:
        state = self._state(batch)
        base_columns = list(CONTRACT_BASE_INDICES) + [
            len(LADDER_FEATURES) + offset for offset in range(len(CONTRACT_CHAIN_FEATURES))
        ]
        contract = batch.ladder[:, :, base_columns]
        base = self.contract_base(contract).squeeze(-1)
        side = batch.ladder[:, :, SIDE_INDEX]
        moneyness = batch.ladder[:, :, MONEYNESS_INDEX]
        contract_logits = (
            base + self.direction(state) * side + self.depth(state) * moneyness
        ).masked_fill(~batch.entry_action_mask, -torch.inf)
        abstain = self.wait(state).squeeze(-1)
        exit_state = torch.cat((state, batch.position[:, POSITION_INDICES]), dim=1)
        return ArchitectureScores(
            contract_logits,
            abstain,
            self.exit(exit_state),
            batch.roles,
            ARCHITECTURE_NAME,
        )


def computed_parameter_count() -> int:
    """Build and count the declared member; never a transcribed count."""

    return trainable_parameter_count(ChainStateLifecyclePolicy())
