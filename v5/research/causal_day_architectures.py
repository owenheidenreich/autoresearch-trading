"""Unfitted architecture interfaces for the causal 0DTE day trader.

These modules define the three predeclared neural representations and their
shallow controls.  They intentionally provide forward passes only: fitting is
owned by the fail-closed gate in :mod:`v5.research.causal_day_policy_gate` and
is not authorized merely because a module can be instantiated.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from v5.research.causal_day_policy_gate import ROLES


ENTRY_ROLES = ("morning_entry", "afternoon_entry")
EXIT_ROLES = ("morning_exit", "afternoon_exit")


class ArchitectureInputError(ValueError):
    """A batch violates the causal sequence/ladder interface contract."""


@dataclass(frozen=True)
class ArchitectureDimensions:
    candle_features: int
    ladder_features: int
    account_features: int
    position_features: int
    clock_features: int
    # No default. A silent default of 16 caused a false defect report on
    # 2026-08-14: every recorded artifact was built at width 8, a review
    # recomputed at the default 16, and the resulting 2.4x mismatch was read as
    # five wrong parameter counts. Callers must now say what width they mean.
    hidden_size: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or value <= 0:
                raise ArchitectureInputError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CausalPolicyBatch:
    """Right-padded session prefixes and contemporaneous ladder snapshots."""

    candles: Tensor
    candle_mask: Tensor
    ladder: Tensor
    ladder_mask: Tensor
    entry_action_mask: Tensor
    account: Tensor
    position: Tensor
    clock: Tensor
    roles: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.candles.shape[0])

    def validate(self, dimensions: ArchitectureDimensions) -> None:
        b = self.batch_size
        expected = {
            "candles": (b, None, dimensions.candle_features),
            "candle_mask": (b, self.candles.shape[1]),
            "ladder": (b, None, dimensions.ladder_features),
            "ladder_mask": (b, self.ladder.shape[1]),
            "entry_action_mask": (b, self.ladder.shape[1]),
            "account": (b, dimensions.account_features),
            "position": (b, dimensions.position_features),
            "clock": (b, dimensions.clock_features),
        }
        actual = {
            "candles": tuple(self.candles.shape),
            "candle_mask": tuple(self.candle_mask.shape),
            "ladder": tuple(self.ladder.shape),
            "ladder_mask": tuple(self.ladder_mask.shape),
            "entry_action_mask": tuple(self.entry_action_mask.shape),
            "account": tuple(self.account.shape),
            "position": tuple(self.position.shape),
            "clock": tuple(self.clock.shape),
        }
        for name, shape in expected.items():
            got = actual[name]
            if len(got) != len(shape) or any(
                want is not None and got_value != want
                for got_value, want in zip(got, shape, strict=True)
            ):
                raise ArchitectureInputError(f"{name} shape {got} does not match {shape}")
        if (
            self.candle_mask.dtype is not torch.bool
            or self.ladder_mask.dtype is not torch.bool
            or self.entry_action_mask.dtype is not torch.bool
        ):
            raise ArchitectureInputError("candle, ladder and entry-action masks must be boolean")
        if len(self.roles) != b or any(role not in ROLES for role in self.roles):
            raise ArchitectureInputError("roles must contain one declared routed role per batch row")
        if not self.candle_mask.any(dim=1).all():
            raise ArchitectureInputError("every row requires at least one completed candle")
        if ((~self.candle_mask[:, :-1]) & self.candle_mask[:, 1:]).any():
            raise ArchitectureInputError("candle masks must be causal right-padded prefixes")
        if (self.entry_action_mask & ~self.ladder_mask).any():
            raise ArchitectureInputError("entry actions must be a subset of the visible ladder")
        for row, role in enumerate(self.roles):
            if role in EXIT_ROLES and self.entry_action_mask[row].any():
                raise ArchitectureInputError("exit roles may not expose entry actions")
        visible_candles = self.candles[self.candle_mask]
        visible_ladder = self.ladder[self.ladder_mask]
        if not torch.isfinite(visible_candles).all():
            raise ArchitectureInputError("visible candle values must be finite")
        if visible_ladder.numel() and not torch.isfinite(visible_ladder).all():
            raise ArchitectureInputError("visible ladder values must be finite")
        for name, tensor in (
            ("account", self.account),
            ("position", self.position),
            ("clock", self.clock),
        ):
            if not torch.isfinite(tensor).all():
                raise ArchitectureInputError(f"{name} values must be finite")


@dataclass(frozen=True)
class ArchitectureScores:
    """Common output contract for joint and specialist architectures."""

    contract_logits: Tensor
    abstain_logits: Tensor
    exit_logits: Tensor
    roles: tuple[str, ...]
    architecture: str

    def active_logits(self, row: int) -> Tensor:
        role = self.roles[row]
        if role in ENTRY_ROLES:
            return torch.cat((self.abstain_logits[row : row + 1], self.contract_logits[row]))
        return self.exit_logits[row]


def _masked_ladder_summary(nodes: Tensor, mask: Tensor) -> Tensor:
    visible = torch.where(mask.unsqueeze(-1), nodes, torch.zeros_like(nodes))
    count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(nodes.dtype)
    mean = visible.sum(dim=1) / count
    negative = torch.finfo(nodes.dtype).min
    maximum = torch.where(mask.unsqueeze(-1), nodes, negative).max(dim=1).values
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


class SequenceSurfaceEncoder(nn.Module):
    """Causal GRU over completed candles plus a permutation-invariant ladder set."""

    def __init__(self, dimensions: ArchitectureDimensions):
        super().__init__()
        h = dimensions.hidden_size
        self.dimensions = dimensions
        self.candle_projection = nn.Linear(dimensions.candle_features, h)
        self.candle_gru = nn.GRU(h, h, batch_first=True)
        self.ladder_node = nn.Sequential(
            nn.Linear(dimensions.ladder_features, h), nn.Tanh(), nn.Linear(h, h)
        )
        fusion_width = h * 3 + dimensions.account_features + dimensions.position_features + dimensions.clock_features
        self.fusion = nn.Sequential(nn.Linear(fusion_width, h), nn.Tanh())

    def forward(self, batch: CausalPolicyBatch) -> tuple[Tensor, Tensor]:
        batch.validate(self.dimensions)
        candle_values = torch.where(
            batch.candle_mask.unsqueeze(-1), batch.candles, torch.zeros_like(batch.candles)
        )
        projected = torch.tanh(self.candle_projection(candle_values))
        sequence, _ = self.candle_gru(projected)
        last = batch.candle_mask.sum(dim=1) - 1
        candle_state = sequence[torch.arange(batch.batch_size, device=last.device), last]

        ladder_values = torch.where(
            batch.ladder_mask.unsqueeze(-1), batch.ladder, torch.zeros_like(batch.ladder)
        )
        nodes = self.ladder_node(ladder_values)
        ladder_state = _masked_ladder_summary(nodes, batch.ladder_mask)
        fused = torch.cat(
            (candle_state, ladder_state, batch.account, batch.position, batch.clock), dim=-1
        )
        return self.fusion(fused), nodes


class ShallowSurfaceEncoder(nn.Module):
    """Non-sequential control using the last completed candle and current set summary."""

    def __init__(self, dimensions: ArchitectureDimensions):
        super().__init__()
        h = dimensions.hidden_size
        self.dimensions = dimensions
        self.ladder_node = nn.Sequential(nn.Linear(dimensions.ladder_features, h), nn.Tanh())
        width = dimensions.candle_features + h * 2 + dimensions.account_features + dimensions.position_features + dimensions.clock_features
        self.fusion = nn.Sequential(nn.Linear(width, h), nn.Tanh())

    def forward(self, batch: CausalPolicyBatch) -> tuple[Tensor, Tensor]:
        batch.validate(self.dimensions)
        last = batch.candle_mask.sum(dim=1) - 1
        candle_state = batch.candles[
            torch.arange(batch.batch_size, device=last.device), last
        ]
        ladder_values = torch.where(
            batch.ladder_mask.unsqueeze(-1), batch.ladder, torch.zeros_like(batch.ladder)
        )
        nodes = self.ladder_node(ladder_values)
        ladder_state = _masked_ladder_summary(nodes, batch.ladder_mask)
        fused = torch.cat(
            (candle_state, ladder_state, batch.account, batch.position, batch.clock), dim=-1
        )
        return self.fusion(fused), nodes


class ContractHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.abstain = nn.Linear(hidden_size, 1)
        self.contract = nn.Linear(hidden_size * 2, 1)

    def forward(self, state: Tensor, nodes: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        expanded = state.unsqueeze(1).expand(-1, nodes.shape[1], -1)
        logits = self.contract(torch.cat((expanded, nodes), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        return logits, self.abstain(state).squeeze(-1)


class JointPolicy(nn.Module):
    def __init__(self, dimensions: ArchitectureDimensions, *, sequence: bool, name: str):
        super().__init__()
        self.name = name
        self.encoder = (
            SequenceSurfaceEncoder(dimensions)
            if sequence
            else ShallowSurfaceEncoder(dimensions)
        )
        self.entry_head = ContractHead(dimensions.hidden_size)
        self.exit_head = nn.Linear(dimensions.hidden_size, 2)

    def forward(self, batch: CausalPolicyBatch) -> ArchitectureScores:
        state, nodes = self.encoder(batch)
        contracts, abstain = self.entry_head(state, nodes, batch.entry_action_mask)
        return ArchitectureScores(
            contracts, abstain, self.exit_head(state), batch.roles, self.name
        )


class SharedFourHeadPolicy(nn.Module):
    def __init__(self, dimensions: ArchitectureDimensions, *, sequence: bool, name: str):
        super().__init__()
        self.name = name
        self.encoder = (
            SequenceSurfaceEncoder(dimensions)
            if sequence
            else ShallowSurfaceEncoder(dimensions)
        )
        self.entry_heads = nn.ModuleDict(
            {role: ContractHead(dimensions.hidden_size) for role in ENTRY_ROLES}
        )
        self.exit_heads = nn.ModuleDict(
            {role: nn.Linear(dimensions.hidden_size, 2) for role in EXIT_ROLES}
        )

    def forward(self, batch: CausalPolicyBatch) -> ArchitectureScores:
        state, nodes = self.encoder(batch)
        b, n = batch.batch_size, batch.ladder.shape[1]
        contracts = state.new_full((b, n), -torch.inf)
        abstain = state.new_full((b,), -torch.inf)
        exits = state.new_full((b, 2), -torch.inf)
        for role, head in self.entry_heads.items():
            selected = torch.tensor(
                [value == role for value in batch.roles], device=state.device
            )
            if selected.any():
                role_contracts, role_abstain = head(state, nodes, batch.entry_action_mask)
                contracts[selected] = role_contracts[selected]
                abstain[selected] = role_abstain[selected]
        for role, head in self.exit_heads.items():
            selected = torch.tensor(
                [value == role for value in batch.roles], device=state.device
            )
            if selected.any():
                exits[selected] = head(state[selected])
        return ArchitectureScores(contracts, abstain, exits, batch.roles, self.name)


class IndependentSpecialistsPolicy(nn.Module):
    """Conditional control: four encoders, one for each routed role."""

    def __init__(self, dimensions: ArchitectureDimensions):
        super().__init__()
        self.name = "four_independent"
        self.encoders = nn.ModuleDict(
            {role: SequenceSurfaceEncoder(dimensions) for role in ROLES}
        )
        self.entry_heads = nn.ModuleDict(
            {role: ContractHead(dimensions.hidden_size) for role in ENTRY_ROLES}
        )
        self.exit_heads = nn.ModuleDict(
            {role: nn.Linear(dimensions.hidden_size, 2) for role in EXIT_ROLES}
        )

    def forward(self, batch: CausalPolicyBatch) -> ArchitectureScores:
        batch.validate(next(iter(self.encoders.values())).dimensions)
        b, n = batch.batch_size, batch.ladder.shape[1]
        contracts = batch.candles.new_full((b, n), -torch.inf)
        abstain = batch.candles.new_full((b,), -torch.inf)
        exits = batch.candles.new_full((b, 2), -torch.inf)
        for role, encoder in self.encoders.items():
            selected = torch.tensor(
                [value == role for value in batch.roles], device=batch.candles.device
            )
            if not selected.any():
                continue
            state, nodes = encoder(batch)
            if role in ENTRY_ROLES:
                role_contracts, role_abstain = self.entry_heads[role](
                    state, nodes, batch.entry_action_mask
                )
                contracts[selected] = role_contracts[selected]
                abstain[selected] = role_abstain[selected]
            else:
                exits[selected] = self.exit_heads[role](state[selected])
        return ArchitectureScores(contracts, abstain, exits, batch.roles, self.name)


def build_architecture(name: str, dimensions: ArchitectureDimensions) -> nn.Module:
    if name == "compact_interaction_entry":
        # Imported lazily because the compact policy uses the common contracts
        # in this module. The canonical registry may count it only against the
        # real tensorizer dimensions; unlike the width-based comparison family,
        # its 48 parameters do not depend on hidden_size.
        from v5.research import causal_day_tensorizer as tz
        from v5.research.causal_day_compact_interaction import (
            CompactInteractionEntryPolicy,
        )

        expected = (
            len(tz.CANDLE_FEATURES),
            len(tz.LADDER_FEATURES),
            tz.ACCOUNT_FEATURES,
            tz.POSITION_FEATURES,
            tz.CLOCK_FEATURES,
        )
        actual = (
            dimensions.candle_features,
            dimensions.ladder_features,
            dimensions.account_features,
            dimensions.position_features,
            dimensions.clock_features,
        )
        if actual != expected:
            raise ArchitectureInputError(
                f"compact architecture requires canonical tensorizer dimensions {expected}, got {actual}"
            )
        return CompactInteractionEntryPolicy()
    if name == "shallow_joint":
        return JointPolicy(dimensions, sequence=False, name=name)
    if name == "shallow_four_head":
        return SharedFourHeadPolicy(dimensions, sequence=False, name=name)
    if name == "neural_joint":
        return JointPolicy(dimensions, sequence=True, name=name)
    if name == "neural_four_head":
        return SharedFourHeadPolicy(dimensions, sequence=True, name=name)
    if name == "four_independent":
        return IndependentSpecialistsPolicy(dimensions)
    raise ArchitectureInputError(f"unknown architecture: {name}")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def declared_dimensions(hidden_size: int | None = None) -> ArchitectureDimensions:
    """The real tensorizer widths, derived rather than transcribed.

    A governance ruling was signed on 2026-08-14 against hand-copied parameter
    counts that were wrong by roughly 2.4x on every architecture. The fix is
    that nothing may state a count it did not compute, and nothing may compute
    one from dimensions it did not read from the tensorizer.
    """

    from v5.research import causal_day_tensorizer as tz
    from v5.research import knobs

    return ArchitectureDimensions(
        candle_features=len(tz.CANDLE_FEATURES),
        ladder_features=len(tz.LADDER_FEATURES),
        account_features=tz.ACCOUNT_FEATURES,
        position_features=tz.POSITION_FEATURES,
        clock_features=tz.CLOCK_FEATURES,
        hidden_size=(
            hidden_size
            if hidden_size is not None
            else int(knobs.frozen_value("causal_day_hidden_size"))
        ),
    )


def computed_parameter_counts(
    hidden_size: int | None = None,
) -> dict[str, int]:
    """Build every declared architecture and count it. No literals anywhere."""

    dimensions = declared_dimensions(hidden_size)
    return {
        name: trainable_parameter_count(build_architecture(name, dimensions))
        for name in ARCHITECTURE_NAMES
    }


ARCHITECTURE_NAMES = (
    "shallow_joint",
    "shallow_four_head",
    "neural_joint",
    "neural_four_head",
    "four_independent",
    "compact_interaction_entry",
)
