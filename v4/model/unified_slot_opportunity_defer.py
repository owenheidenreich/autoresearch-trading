"""Slot opportunity-cost defer gate for unified conservative overlays."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


ROLE_LABEL = "FOUNDATION_UNIFIED_SLOT_OPPORTUNITY_DEFER_OVERLAY_V1"


@dataclass(frozen=True)
class SlotOpportunityDeferConfig:
    min_net_advantage_margin: float = 250.0
    blocked_cost_uncertainty_weight: float = 1.0
    max_blocked_protocol101_entries: int = 1
    require_nonnegative_q1_q3_stress: bool = True
    baseline: str = "PAPER_DEFAULT_PROTOCOL101"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def slot_opportunity_defer_decision(
    *,
    predicted_challenger_advantage: float,
    estimated_blocked_protocol101_cost: float,
    blocked_cost_uncertainty: float = 0.0,
    estimated_blocked_entries: int = 0,
    config: SlotOpportunityDeferConfig = SlotOpportunityDeferConfig(),
) -> dict[str, Any]:
    """Allow an override only after charging the baseline slot opportunity cost."""

    adjusted = (
        float(predicted_challenger_advantage)
        - max(0.0, float(estimated_blocked_protocol101_cost))
        - float(config.blocked_cost_uncertainty_weight) * max(0.0, float(blocked_cost_uncertainty))
    )
    too_many_blocked_entries = int(estimated_blocked_entries) > int(config.max_blocked_protocol101_entries)
    allowed = adjusted >= float(config.min_net_advantage_margin) and not too_many_blocked_entries
    return {
        "allowed": bool(allowed),
        "decision": "allow_challenger_override" if allowed else "defer_to_protocol101",
        "adjusted_advantage": float(adjusted),
        "required_margin": float(config.min_net_advantage_margin),
        "estimated_blocked_protocol101_cost": float(estimated_blocked_protocol101_cost),
        "blocked_cost_uncertainty": float(blocked_cost_uncertainty),
        "estimated_blocked_entries": int(estimated_blocked_entries),
        "max_blocked_protocol101_entries": int(config.max_blocked_protocol101_entries),
        "baseline": config.baseline,
    }


def oracle_net_vs_blocked_protocol101(*, challenger_pnl: float, blocked_protocol101_pnl: float) -> float:
    """Diagnostic-only realized net contribution of an override versus restored baseline slot use."""

    return float(challenger_pnl) - max(0.0, float(blocked_protocol101_pnl))


def blocked_protocol101_cost_for_interval(
    entry_times: np.ndarray,
    pnl_values: np.ndarray,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    slippage_per_side: float = 0.0,
    contract_multiplier: float = 100.0,
) -> dict[str, float | int]:
    """Label-only sum of Protocol101 entries that would be blocked by an interval."""

    if len(entry_times) != len(pnl_values):
        raise ValueError("entry_times and pnl_values must have the same length")
    if pd.isna(start) or pd.isna(end) or pd.Timestamp(end) <= pd.Timestamp(start):
        return {"blocked_entries": 0, "blocked_pnl": 0.0}
    left = int(np.searchsorted(entry_times, np.datetime64(pd.Timestamp(start).to_datetime64()), side="left"))
    right = int(np.searchsorted(entry_times, np.datetime64(pd.Timestamp(end).to_datetime64()), side="left"))
    if right <= left:
        return {"blocked_entries": 0, "blocked_pnl": 0.0}
    stress = 2.0 * float(slippage_per_side) * float(contract_multiplier)
    pnl = np.asarray(pnl_values[left:right], dtype=float) - stress
    return {"blocked_entries": int(right - left), "blocked_pnl": float(np.nansum(pnl))}
