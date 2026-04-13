"""Runtime contracts for the v3 pure-RL trading system."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


LOOKBACK_1M = 90
LOOKBACK_5M = 78
CONTRACT_MULTIPLIER = 100
BARS_PER_DAY = 390
FIVE_MINUTE_BUCKET = 5
MODEL_SCHEMA_VERSION = "v3_multiscale_v1"

ACTION_TYPES = ("NOOP", "OPEN", "HOLD", "CLOSE", "ADJUST")
EXIT_STYLES = ("STATIC", "MODEL_EXIT", "TIME_ONLY")
MANAGE_MODES = ("HOLD", "CLOSE", "ADJUST")

SESSION_STATE_FEATURE_NAMES = (
    "minutes_from_open_norm",
    "minutes_to_close_norm",
    "spot_return_from_open",
    "spot_return_from_vwap",
    "dist_to_session_high_pct",
    "dist_to_session_low_pct",
    "dist_to_opening_range_high_pct",
    "dist_to_opening_range_low_pct",
    "dist_to_initial_balance_high_pct",
    "dist_to_initial_balance_low_pct",
    "session_range_expansion_pct",
    "realized_vol_since_open",
    "bars_since_session_high_norm",
    "bars_since_session_low_norm",
    "cum_delta_since_open_norm",
    "cum_volume_vs_time_of_day_norm",
)

TRADE_STATE_FEATURE_NAMES = (
    "in_position",
    "bars_held_norm",
    "bars_since_last_adjust_norm",
    "qty_norm",
    "risk_budget_frac",
    "avg_entry_price_norm",
    "current_mid_price_norm",
    "stop_gap_frac",
    "target_gap_frac",
    "time_to_forced_exit_norm",
    "unrealized_pnl_frac_equity",
    "realized_trade_pnl_frac_equity",
    "mfe_frac_equity",
    "mae_frac_equity",
    "peak_unrealized_frac_equity",
    "trough_unrealized_frac_equity",
    "entry_bar_norm",
    "add_count_norm",
    "reduce_count_norm",
    "entry_vwap_dist",
    "entry_session_range_position",
    "entry_call_put_flow_ratio",
)


@dataclass
class PositionState:
    """Current live-position state."""

    in_position: bool = False
    contract_index: int = -1
    snapshot_row: int = -1
    expiry: str = ""
    strike: float = 0.0
    right: str = ""
    qty: int = 0
    avg_entry_price: float = 0.0
    current_mid_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    max_hold_bar: int = 0
    exit_style: str = "STATIC"
    opened_bar: int = -1
    last_update_bar: int = -1
    last_adjust_bar: int = -1
    cumulative_realized_pnl: float = 0.0
    risk_budget_frac: float = 0.0
    mfe_dollars: float = 0.0
    mae_dollars: float = 0.0
    peak_unrealized_pnl: float = 0.0
    trough_unrealized_pnl: float = 0.0
    add_count: int = 0
    reduce_count: int = 0
    entry_vwap_dist: float = 0.0
    entry_session_range_position: float = 0.0
    entry_call_put_flow_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def bars_held(self) -> int:
        if not self.in_position or self.opened_bar < 0 or self.last_update_bar < 0:
            return 0
        return max(0, self.last_update_bar - self.opened_bar)


@dataclass
class PolicyObservation:
    """One policy observation for the current decision bar."""

    date: str
    local_bar: int
    global_bar: int
    spot_price: float
    context_1m: np.ndarray
    context_5m: np.ndarray
    session_state: np.ndarray
    trade_state: np.ndarray
    contract_snapshot: np.ndarray
    contract_indices: np.ndarray
    valid_mask: np.ndarray
    position: PositionState
    realized_day_pnl: float
    cash: float
    equity: float
    peak_equity: float
    drawdown: float
    position_contract_features: np.ndarray

    def account_vector(self) -> np.ndarray:
        denom = max(1.0, self.equity)
        vec = np.asarray(
            [
                self.realized_day_pnl / denom,
                self.cash / denom,
                self.equity / max(self.peak_equity, 1.0),
                self.peak_equity / 10_000.0,
                self.drawdown,
                self.spot_price / 10000.0,
            ],
            dtype=np.float32,
        )
        return vec

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["context_1m"] = self.context_1m.tolist()
        out["context_5m"] = self.context_5m.tolist()
        out["session_state"] = self.session_state.tolist()
        out["trade_state"] = self.trade_state.tolist()
        out["contract_snapshot"] = self.contract_snapshot.tolist()
        out["contract_indices"] = self.contract_indices.tolist()
        out["valid_mask"] = self.valid_mask.tolist()
        out["position_contract_features"] = self.position_contract_features.tolist()
        return out


@dataclass
class AgentAction:
    """Environment-facing action."""

    action_type: str = "NOOP"
    contract_row: int = -1
    exit_style: str = "STATIC"
    manage_mode: str = "HOLD"
    risk_budget_frac: float = 0.0
    stop_frac: float = 0.0
    target_frac: float = 0.0
    time_stop_frac: float = 0.0
    updated_stop_frac: float = 0.0
    updated_target_frac: float = 0.0
    updated_time_stop_frac: float = 0.0
    size_delta_frac: float = 0.0
    confidence: float = 0.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.action_type not in ACTION_TYPES:
            errors.append(f"unknown action_type={self.action_type}")
        if self.exit_style not in EXIT_STYLES:
            errors.append(f"unknown exit_style={self.exit_style}")
        if self.manage_mode not in MANAGE_MODES:
            errors.append(f"unknown manage_mode={self.manage_mode}")
        if not -1.0 <= self.size_delta_frac <= 1.0:
            errors.append(f"size_delta_frac must be in [-1, 1], got {self.size_delta_frac}")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append(f"confidence must be in [0, 1], got {self.confidence}")
        return errors

    @classmethod
    def noop(cls) -> "AgentAction":
        return cls(action_type="NOOP", manage_mode="HOLD")


@dataclass
class ExecutionResult:
    """One transition result for tracing and debugging."""

    date: str
    local_bar: int
    action_type: str
    executed: bool
    event: str
    reason_codes: tuple[str, ...] = ()
    contract_index: int = -1
    snapshot_row: int = -1
    qty_delta: int = 0
    resulting_qty: int = 0
    fill_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    max_hold_bar: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    drawdown: float = 0.0
    reward: float = 0.0
    confidence: float = 0.0
    validation_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class StepTrace:
    """Detailed replay trace for one bar transition."""

    date: str
    local_bar: int
    global_bar: int
    in_position: bool
    action_type: str
    manage_mode: str
    exit_style: str
    selected_contract_row: int
    selected_contract_index: int
    pointer_scores: list[float]
    valid_mask: list[int]
    context_summary: dict[str, float]
    session_memory: dict[str, float]
    trade_memory: dict[str, float]
    selected_contract_features: dict[str, float]
    sampled_action: dict[str, Any]
    execution: dict[str, Any]
    realized_day_pnl: float
    equity: float
    drawdown: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class TradeRecord:
    """Closed trade lifecycle summary."""

    date: str
    expiry: str
    strike: float
    right: str
    contract_index: int
    entry_bar: int
    exit_bar: int
    max_qty: int
    entry_price: float
    exit_price: float
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str
    adjustments: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class EpisodeSummary:
    """Per-day evaluation summary."""

    date: str
    steps: int
    total_reward: float
    starting_equity: float
    ending_equity: float
    realized_pnl: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    turnover_contracts: int
    exposure_bars: int
    avg_action_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RLArtifactManifest:
    """Saved experiment manifest for v3 artifacts."""

    experiment_id: str
    created_at: str
    dataset_fingerprint: str
    checkpoint_path: str
    training_config: dict[str, Any]
    reward_config: dict[str, Any]
    execution_config: dict[str, Any]
    evaluation_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    trace_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def zero_contract_features(n_features: int) -> np.ndarray:
    return np.zeros((n_features,), dtype=np.float32)


def zero_trade_state() -> np.ndarray:
    return np.zeros((len(TRADE_STATE_FEATURE_NAMES),), dtype=np.float32)
