"""Unified conservative offline policy foundation.

This module freezes the next ML direction as a constrained sequential decision
contract rather than a Protocol276 tweak. It intentionally contains no model
training code. The primitives here define the live-like state/action/execution
game, action-advantage label wrappers, causal feature guardrails, and the
conservative defer-to-Protocol101 policy gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from v4.model.unified_serial_game import (
    CONTRACT_MULTIPLIER,
    UnifiedSerialGameConfig,
    build_flat_action_advantage_labels,
    candidate_dict_from_row,
    hold_exit_opportunity_advantage_path,
    validate_unified_candidate,
)


ROLE_LABEL = "FOUNDATION_UNIFIED_CONSERVATIVE_OFFLINE_POLICY_V1"
UNIFIED_DECISION_STATE_CONTRACT = "UnifiedDecisionStateV1"
EXECUTION_MODEL_CONTRACT = "ExecutionModelV1"
ACTION_ADVANTAGE_LABEL_CONTRACT = "ActionAdvantageLabelV1"
CONSERVATIVE_POLICY_GATE_CONTRACT = "ConservativePolicyImprovementGateV1"
PAPER_DEFAULT_BASELINE = "PAPER_DEFAULT_PROTOCOL101"

FLAT_ACTIONS = ("wait", "enter")
HOLDING_ACTIONS = ("hold", "exit")
POSITION_STATES = ("flat", "holding")
FILL_MODEL_DETERMINISTIC = "deterministic_ask_bid"
FILL_MODEL_STOCHASTIC = "stochastic_calibrated"

MODEL_FORBIDDEN_FEATURE_TOKENS = (
    "candidate_pnl",
    "candidate_exit",
    "future_",
    "oracle_",
    "label",
    "target",
    "q_wait",
    "q_enter",
    "q_exit",
    "q_hold",
    "a_enter",
    "a_hold",
    "a_exit",
    "a_switch",
    "best_path",
    "best_future",
    "future_best",
)


@dataclass(frozen=True)
class CandidateSnapshotV1:
    candidate_uid: str
    contract_id: str
    right: str
    entry_bid: float
    entry_ask: float
    entry_premium: float
    quote_age_ms: float
    context_age_ms: float = 0.0
    root: str = "SPXW"
    settlement_style: str = "PM"
    offset: float = 0.0
    entry_mid: float | None = None
    entry_spread: float | None = None
    entry_bid_size: float = 1.0
    entry_ask_size: float = 1.0
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_theta: float = 0.0
    entry_iv: float = 0.0

    def to_candidate_dict(self, decision_time: str) -> dict[str, Any]:
        bid = finite(self.entry_bid)
        ask = finite(self.entry_ask)
        mid = finite(self.entry_mid, (bid + ask) / 2.0 if math.isfinite(bid) and math.isfinite(ask) else math.nan)
        spread = finite(self.entry_spread, ask - bid if math.isfinite(bid) and math.isfinite(ask) else math.nan)
        return {
            "decision_time": decision_time,
            "candidate_uid": self.candidate_uid,
            "contract_id": self.contract_id,
            "root": self.root,
            "settlement_style": self.settlement_style,
            "right": self.right,
            "offset": self.offset,
            "entry_bid": bid,
            "entry_ask": ask,
            "entry_mid": mid,
            "entry_spread": spread,
            "entry_bid_size": self.entry_bid_size,
            "entry_ask_size": self.entry_ask_size,
            "entry_premium": self.entry_premium,
            "entry_delta": self.entry_delta,
            "entry_gamma": self.entry_gamma,
            "entry_theta": self.entry_theta,
            "entry_iv": self.entry_iv,
            "quote_age_ms": self.quote_age_ms,
            "context_age_ms": self.context_age_ms,
        }


@dataclass(frozen=True)
class PositionSnapshotV1:
    contract_id: str
    right: str
    entry_time: str
    entry_ask: float
    current_bid: float
    current_ask: float
    current_pnl: float
    mfe_to_now: float
    mae_to_now: float
    giveback_from_mfe: float
    minutes_since_entry: float
    minutes_to_forced_flat: float
    quantity: int = 1


@dataclass(frozen=True)
class UnifiedDecisionStateV1:
    split: str
    session: str
    decision_time: str
    position_state: str
    account_equity: float
    cash_available: float
    candidates: tuple[CandidateSnapshotV1, ...] = ()
    position: PositionSnapshotV1 | None = None
    recent_history_rows: int = 0

    def account_state(self, config: UnifiedSerialGameConfig) -> dict[str, Any]:
        return {
            "account_equity": self.account_equity,
            "cash_available": self.cash_available,
            "open_position_count": 1 if self.position_state == "holding" else 0,
            "max_concurrent_positions": config.max_concurrent_positions,
            "max_contracts": config.max_contracts,
        }


@dataclass(frozen=True)
class ExecutionModelV1Config:
    starting_cash: float = 10_000.0
    max_contracts: int = 1
    max_concurrent_positions: int = 1
    entry_price: str = "ask"
    exit_price: str = "bid"
    slippage_per_side: float = 0.0
    fill_model: str = FILL_MODEL_DETERMINISTIC
    fill_observations: int = 0
    required_fill_observations: int = 30
    allow_midpoint_labels: bool = False


@dataclass(frozen=True)
class ExecutionModelV1:
    config: ExecutionModelV1Config = field(default_factory=ExecutionModelV1Config)

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        if self.config.max_contracts != 1:
            errors.append("max_contracts_must_equal_one")
        if self.config.max_concurrent_positions != 1:
            errors.append("max_concurrent_positions_must_equal_one")
        if self.config.entry_price != "ask":
            errors.append("entry_price_must_be_ask")
        if self.config.exit_price != "bid":
            errors.append("exit_price_must_be_bid")
        if self.config.allow_midpoint_labels:
            errors.append("midpoint_labels_disallowed")
        if self.config.fill_model == FILL_MODEL_STOCHASTIC and self.config.fill_observations < self.config.required_fill_observations:
            errors.append("stochastic_fill_model_insufficient_observations")
        if self.config.fill_model not in {FILL_MODEL_DETERMINISTIC, FILL_MODEL_STOCHASTIC}:
            errors.append("unknown_fill_model")
        return {"status": "pass" if not errors else "fail", "errors": errors}

    def entry_debit(self, candidate: CandidateSnapshotV1) -> float:
        return float((candidate.entry_ask + self.config.slippage_per_side) * CONTRACT_MULTIPLIER)

    def exit_credit(self, position: PositionSnapshotV1) -> float:
        return float((position.current_bid - self.config.slippage_per_side) * CONTRACT_MULTIPLIER * position.quantity)


@dataclass(frozen=True)
class ActionMaskV1:
    wait: bool
    enter_candidates: tuple[bool, ...]
    hold: bool
    exit: bool
    forced_exit_required: bool
    reasons_by_candidate: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConservativePolicyGateConfig:
    min_advantage_margin: float = 250.0
    uncertainty_penalty_weight: float = 1.0
    ood_penalty_weight: float = 1.0
    baseline_action: str = PAPER_DEFAULT_BASELINE


def action_mask_for_decision_state(
    state: UnifiedDecisionStateV1,
    *,
    game_config: UnifiedSerialGameConfig = UnifiedSerialGameConfig(),
) -> ActionMaskV1:
    if state.position_state not in POSITION_STATES:
        raise ValueError(f"invalid position_state: {state.position_state}")
    forced = is_forced_flat_time(state.decision_time, game_config)
    if state.position_state == "holding":
        return ActionMaskV1(
            wait=False,
            enter_candidates=tuple(False for _ in state.candidates),
            hold=not forced,
            exit=True,
            forced_exit_required=forced,
            reasons_by_candidate=tuple(("holding_state_disallows_new_entry",) for _ in state.candidates),
        )
    account = state.account_state(game_config)
    enter_mask: list[bool] = []
    reasons: list[tuple[str, ...]] = []
    for candidate in state.candidates:
        result = validate_unified_candidate(candidate.to_candidate_dict(state.decision_time), account, config=game_config)
        enter_mask.append(result["status"] == "pass")
        reasons.append(tuple(result.get("errors", [])))
    return ActionMaskV1(
        wait=True,
        enter_candidates=tuple(enter_mask),
        hold=False,
        exit=False,
        forced_exit_required=False,
        reasons_by_candidate=tuple(reasons),
    )


def build_flat_action_advantage_label_v1(
    frame: pd.DataFrame,
    *,
    config: UnifiedSerialGameConfig = UnifiedSerialGameConfig(),
    slippage_per_side: float = 0.0,
) -> pd.DataFrame:
    labels = build_flat_action_advantage_labels(frame, config=config, slippage_per_side=slippage_per_side)
    if labels.empty:
        return labels
    out = labels.copy()
    out["label_contract"] = ACTION_ADVANTAGE_LABEL_CONTRACT
    out["a_wait"] = out["q_wait"] - np.maximum(out["q_wait"], out["best_enter_value_at_decision"])
    out["action_space"] = "flat_wait_or_enter_candidate"
    out["future_path_columns_used_as_features"] = False
    return out


def build_holding_action_advantage_label_v1(
    path_bid: Iterable[float],
    *,
    entry_ask: float,
    future_flat_values: Iterable[float] | None = None,
    slippage_per_side: float = 0.0,
) -> pd.DataFrame:
    adjusted_bid = [float(value) - float(slippage_per_side) for value in path_bid]
    labels = hold_exit_opportunity_advantage_path(adjusted_bid, entry_ask, future_flat_values)
    if labels.empty:
        return labels
    out = labels.copy()
    out["label_contract"] = ACTION_ADVANTAGE_LABEL_CONTRACT
    out["a_exit"] = out["q_exit"] - out["q_hold"]
    out["action_space"] = "holding_hold_or_exit"
    out["future_path_columns_used_as_features"] = False
    return out


def validate_no_future_feature_columns(columns: Iterable[str]) -> None:
    bad: list[str] = []
    for column in columns:
        name = str(column).lower()
        if any(token in name for token in MODEL_FORBIDDEN_FEATURE_TOKENS):
            bad.append(str(column))
    if bad:
        raise ValueError(f"future/label columns are not allowed as model features: {bad}")


def conservative_policy_improvement_decision(
    *,
    challenger_advantage: float,
    baseline_advantage: float = 0.0,
    uncertainty: float = 0.0,
    ood_penalty: float = 0.0,
    config: ConservativePolicyGateConfig = ConservativePolicyGateConfig(),
) -> dict[str, Any]:
    adjusted = (
        finite(challenger_advantage, 0.0)
        - finite(baseline_advantage, 0.0)
        - config.uncertainty_penalty_weight * max(0.0, finite(uncertainty, 0.0))
        - config.ood_penalty_weight * max(0.0, finite(ood_penalty, 0.0))
    )
    allowed = adjusted >= config.min_advantage_margin
    return {
        "decision": "allow_challenger_action" if allowed else "defer_to_protocol101",
        "allowed": bool(allowed),
        "adjusted_advantage": float(adjusted),
        "required_margin": float(config.min_advantage_margin),
        "baseline_action": config.baseline_action,
    }


def unified_conservative_policy_contract() -> dict[str, Any]:
    execution = ExecutionModelV1()
    return {
        "role_label": ROLE_LABEL,
        "paper_default_baseline": PAPER_DEFAULT_BASELINE,
        "state_contract": {
            "name": UNIFIED_DECISION_STATE_CONTRACT,
            "position_states": POSITION_STATES,
            "flat_actions": FLAT_ACTIONS,
            "holding_actions": HOLDING_ACTIONS,
            "required_flat_state": [
                "market context",
                "account state",
                "full live-feasible SPXW PM 0DTE candidate surface",
                "candidate masks",
                "quote freshness",
                "Greeks",
                "time to close",
                "recent causal session history",
            ],
            "required_holding_state": [
                "current position",
                "entry context",
                "bid/ask path",
                "MFE/MAE",
                "giveback",
                "theta/gamma burden",
                "future flat-slot opportunity stream",
                "account state",
            ],
        },
        "execution_model": {
            "name": EXECUTION_MODEL_CONTRACT,
            **asdict(execution.config),
            "validation": execution.validate(),
        },
        "label_contract": {
            "name": ACTION_ADVANTAGE_LABEL_CONTRACT,
            "flat": ["A_enter = Q(enter candidate) - Q(wait)", "A_wait = Q(wait) - max(Q(wait), best Q(enter))"],
            "holding": ["A_hold = Q(hold) - Q(exit now)", "A_exit = Q(exit now) - Q(hold)"],
            "constraints": [
                "one account",
                "one open position",
                "one contract",
                "ask-entry",
                "bid-exit",
                "affordability",
                "forced flat",
                "live-feasible candidates",
                "no future/path/label columns in model inputs",
            ],
        },
        "conservative_gate": {
            "name": CONSERVATIVE_POLICY_GATE_CONTRACT,
            **asdict(ConservativePolicyGateConfig()),
        },
    }


def flat_decision_state_from_candidates(
    candidates: pd.DataFrame,
    *,
    split: str,
    session: str,
    decision_time: str,
    account_equity: float = 10_000.0,
    cash_available: float = 10_000.0,
) -> UnifiedDecisionStateV1:
    snapshots = []
    for _, row in candidates.iterrows():
        candidate = candidate_dict_from_row(row)
        snapshots.append(
            CandidateSnapshotV1(
                candidate_uid=str(row.get("candidate_uid", "")),
                contract_id=str(candidate.get("contract_id", "")),
                right=str(candidate.get("right", "")),
                entry_bid=finite(candidate.get("entry_bid")),
                entry_ask=finite(candidate.get("entry_ask")),
                entry_premium=finite(candidate.get("entry_premium")),
                quote_age_ms=finite(candidate.get("quote_age_ms")),
                context_age_ms=finite(candidate.get("context_age_ms"), 0.0),
                root=str(candidate.get("root", "SPXW")),
                settlement_style=str(candidate.get("settlement_style", "PM")),
                offset=finite(candidate.get("offset"), 0.0),
                entry_mid=finite(candidate.get("entry_mid")),
                entry_spread=finite(candidate.get("entry_spread")),
                entry_bid_size=finite(candidate.get("entry_bid_size"), 1.0),
                entry_ask_size=finite(candidate.get("entry_ask_size"), 1.0),
                entry_delta=finite(candidate.get("entry_delta"), 0.0),
                entry_gamma=finite(candidate.get("entry_gamma"), 0.0),
                entry_theta=finite(candidate.get("entry_theta"), 0.0),
                entry_iv=finite(candidate.get("entry_iv"), 0.0),
            )
        )
    return UnifiedDecisionStateV1(
        split=split,
        session=session,
        decision_time=decision_time,
        position_state="flat",
        account_equity=float(account_equity),
        cash_available=float(cash_available),
        candidates=tuple(snapshots),
    )


def is_forced_flat_time(decision_time: str, config: UnifiedSerialGameConfig) -> bool:
    ts = pd.to_datetime(decision_time, utc=True, errors="coerce")
    if pd.isna(ts):
        return False
    local = pd.Timestamp(ts).tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    return minute >= int(config.forced_flat_minute_et)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default
