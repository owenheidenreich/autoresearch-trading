from __future__ import annotations

import dataclasses
import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from training.prepare import FEATURE_NAMES


FEATURE_CONTRACT_VERSION = "ibkr_live_v1"


@dataclass(frozen=True)
class FeatureContractVersion:
    version_id: str = FEATURE_CONTRACT_VERSION
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    source: str = "ibkr_live"
    created_at: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat())

    def validate_feature_shape(self, arr: np.ndarray) -> bool:
        return arr.ndim == 2 and arr.shape[1] == len(self.feature_names)

    def validate_feature_names(self, names: list[str]) -> bool:
        return list(names) == list(self.feature_names)


@dataclass
class LiveContextBundle:
    as_of_date: str
    context_start_date: str
    context_end_date: str
    feature_contract_version: str
    raw_features: np.ndarray
    normalized_features: np.ndarray
    valid_mask: np.ndarray
    dates: list[str]
    timestamps: list[str]
    norm_raw_buffer: np.ndarray
    norm_valid_buffer: np.ndarray
    market_rows: list[dict[str, Any]]
    vix_rows: dict[int, dict[str, float]]
    options_data: dict[tuple[str, int], dict[str, float]]
    chain_data: dict[tuple[str, int], dict[str, float]]
    prior_levels: dict[str, float]
    multi_timeframe_stats: dict[str, float]
    source_meta: dict[str, Any] = field(default_factory=dict)

    def num_bars(self) -> int:
        return int(self.raw_features.shape[0])


@dataclass
class DecisionIntent:
    action: int
    contract: Any
    qty: int
    entry_order: str
    stop_price: float
    take_profit_price: float
    confidence: float
    reason_codes: list[str]
    entry_limit_price: float | None = None
    reference_price: float | None = None
    decision_id: str | None = None
    intent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskUpdateIntent:
    position_id: str
    new_stop_price: float | None = None
    new_take_profit_price: float | None = None
    reason_codes: list[str] = field(default_factory=list)
    decision_id: str | None = None
    intent_id: str | None = None


@dataclass
class ExecutionState:
    position_id: str
    contract: Any
    qty: int
    status: str
    entry_order_id: int
    stop_order_id: int
    take_profit_order_id: int
    current_stop: float
    current_take_profit: float
    created_at: str
    updated_at: str
    entry_price_reference: float | None = None
    fill_status: str = "PENDING"
    fill_price: float | None = None
    fill_time: str | None = None
    slippage_bps: float | None = None
    session_id: str | None = None
    decision_id: str | None = None
    intent_id: str | None = None
    ib_perm_id_entry: int | None = None
    last_exec_id: str | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def bundle_path(base_dir: str, as_of_date: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"context-{as_of_date}.pt")


def save_bundle(path: str, bundle: LiveContextBundle) -> None:
    payload = dataclasses.asdict(bundle)
    torch.save(payload, path)


def load_bundle(path: str) -> LiveContextBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return LiveContextBundle(**payload)
