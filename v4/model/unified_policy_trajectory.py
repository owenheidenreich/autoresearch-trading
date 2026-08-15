"""Trajectory dataset primitives for the unified conservative policy.

The trajectory dataset is the bridge between the frozen policy contract and a
future neural model. This module does not train a model. It extracts flat and
holding decision states from existing action-advantage artifacts, validates
that model features are causal, and summarizes coverage gaps before a full
offline policy dataset is promoted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import pandas as pd

from v4.model.unified_conservative_policy import (
    ACTION_ADVANTAGE_LABEL_CONTRACT,
    EXECUTION_MODEL_CONTRACT,
    ROLE_LABEL as FOUNDATION_ROLE_LABEL,
    UNIFIED_DECISION_STATE_CONTRACT,
    UnifiedDecisionStateV1,
    PositionSnapshotV1,
    flat_decision_state_from_candidates,
    validate_no_future_feature_columns,
)
from v4.scripts.run_protocol164_full_action_space_dataset import FULL_ACTION_FEATURE_COLUMNS
from v4.scripts.run_protocol270_full_surface_action_advantage_dataset import OPTIONAL_HISTORY_COLUMNS
from v4.scripts.run_protocol275_position_state_lifecycle_policy import FEATURE_COLUMNS as HOLDING_FEATURE_COLUMNS


ROLE_LABEL = "DATASET_UNIFIED_POLICY_TRAJECTORY_FOUNDATION_V1"
TRAJECTORY_DATASET_CONTRACT = "UnifiedPolicyTrajectoryDatasetV1"
FLAT_STATE_SOURCE = "DATASET_FULL_SURFACE_ACTION_ADVANTAGE_V1 / Protocol270"
HOLDING_STATE_SOURCE = "DATASET_POSITION_STATE_ACTION_ADVANTAGE_V1 / Protocol274"
PROTOCOL276_EVIDENCE_SOURCE = "AUDIT_PROTOCOL276_INTEGRATED_LIFECYCLE_FAILURE_ATTRIBUTION_V1"

FLAT_TRAJECTORY_FEATURE_COLUMNS = tuple(dict.fromkeys([*FULL_ACTION_FEATURE_COLUMNS, *OPTIONAL_HISTORY_COLUMNS]))
HOLDING_LABEL_DERIVED_COLUMNS = {"entry_a_enter", "entry_q_wait", "entry_q_enter"}
HOLDING_TRAJECTORY_FEATURE_COLUMNS = tuple(column for column in HOLDING_FEATURE_COLUMNS if column not in HOLDING_LABEL_DERIVED_COLUMNS)


@dataclass(frozen=True)
class TrajectoryDatasetContractV1:
    name: str = TRAJECTORY_DATASET_CONTRACT
    foundation_role_label: str = FOUNDATION_ROLE_LABEL
    state_contract: str = UNIFIED_DECISION_STATE_CONTRACT
    execution_contract: str = EXECUTION_MODEL_CONTRACT
    label_contract: str = ACTION_ADVANTAGE_LABEL_CONTRACT
    flat_state_source: str = FLAT_STATE_SOURCE
    holding_state_source: str = HOLDING_STATE_SOURCE
    protocol276_evidence_source: str = PROTOCOL276_EVIDENCE_SOURCE
    paper_default_baseline: str = "PAPER_DEFAULT_PROTOCOL101"
    status: str = "foundation_manifest_only_training_blocked"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_trajectory_feature_contract() -> dict[str, Any]:
    validate_no_future_feature_columns(FLAT_TRAJECTORY_FEATURE_COLUMNS)
    validate_no_future_feature_columns(HOLDING_TRAJECTORY_FEATURE_COLUMNS)
    return {
        "status": "pass",
        "flat_feature_count": len(FLAT_TRAJECTORY_FEATURE_COLUMNS),
        "holding_feature_count": len(HOLDING_TRAJECTORY_FEATURE_COLUMNS),
        "future_or_label_columns_in_model_features": 0,
    }


def flat_decision_states_from_action_rows(
    frame: pd.DataFrame,
    *,
    max_events: int = 0,
    starting_cash: float = 10_000.0,
) -> list[UnifiedDecisionStateV1]:
    required = {"split", "session", "decision_time", "decision_dt", "candidate_uid", "contract_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing flat trajectory columns: {missing}")
    working = frame.copy()
    working["decision_dt"] = pd.to_datetime(working["decision_dt"], utc=True, errors="coerce")
    working = working[working["decision_dt"].notna()].sort_values(["split", "session", "decision_dt", "candidate_uid"])
    states: list[UnifiedDecisionStateV1] = []
    for (_, _, _), group in working.groupby(["split", "session", "decision_dt"], sort=True):
        first = group.iloc[0]
        states.append(
            flat_decision_state_from_candidates(
                group,
                split=str(first["split"]),
                session=str(first["session"]),
                decision_time=str(first["decision_time"]),
                account_equity=float(starting_cash),
                cash_available=float(starting_cash),
            )
        )
        if max_events > 0 and len(states) >= max_events:
            break
    return states


def holding_decision_state_from_row(row: pd.Series, *, account_equity: float = 10_000.0) -> UnifiedDecisionStateV1:
    position = PositionSnapshotV1(
        contract_id=str(row.get("contract_id", "")),
        right=str(row.get("right", "")),
        entry_time=str(row.get("entry_time", "")),
        entry_ask=float(pd.to_numeric(pd.Series([row.get("entry_ask")]), errors="coerce").iloc[0]),
        current_bid=float(pd.to_numeric(pd.Series([row.get("bid")]), errors="coerce").iloc[0]),
        current_ask=float(pd.to_numeric(pd.Series([row.get("ask")]), errors="coerce").iloc[0]),
        current_pnl=float(pd.to_numeric(pd.Series([row.get("current_pnl")]), errors="coerce").fillna(0.0).iloc[0]),
        mfe_to_now=float(pd.to_numeric(pd.Series([row.get("mfe_to_now")]), errors="coerce").fillna(0.0).iloc[0]),
        mae_to_now=float(pd.to_numeric(pd.Series([row.get("mae_to_now")]), errors="coerce").fillna(0.0).iloc[0]),
        giveback_from_mfe=float(pd.to_numeric(pd.Series([row.get("giveback_from_mfe")]), errors="coerce").fillna(0.0).iloc[0]),
        minutes_since_entry=float(pd.to_numeric(pd.Series([row.get("minutes_since_entry")]), errors="coerce").fillna(0.0).iloc[0]),
        minutes_to_forced_flat=float(pd.to_numeric(pd.Series([row.get("minutes_to_forced_flat")]), errors="coerce").fillna(0.0).iloc[0]),
    )
    return UnifiedDecisionStateV1(
        split=str(row.get("split", "")),
        session=str(row.get("session", "")),
        decision_time=str(row.get("state_time", "")),
        position_state="holding",
        account_equity=float(account_equity),
        cash_available=max(0.0, float(account_equity) - float(position.entry_ask) * 100.0),
        position=position,
    )


def holding_decision_states_from_rows(frame: pd.DataFrame, *, max_rows: int = 0, account_equity: float = 10_000.0) -> list[UnifiedDecisionStateV1]:
    required = {"split", "session", "state_time", "contract_id", "entry_ask", "bid", "ask", "current_pnl"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing holding trajectory columns: {missing}")
    working = frame.copy()
    working["state_dt"] = pd.to_datetime(working["state_time"], utc=True, errors="coerce")
    working = working[working["state_dt"].notna()].sort_values(["split", "session", "state_dt", "contract_id"])
    if max_rows > 0:
        working = working.head(max_rows)
    return [holding_decision_state_from_row(row, account_equity=account_equity) for _, row in working.iterrows()]


def summarize_flat_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["split", "candidate_rows", "decision_events", "sessions", "median_candidates_per_event"])
    working = frame.copy()
    working["decision_dt"] = pd.to_datetime(working["decision_dt"], utc=True, errors="coerce")
    rows = []
    for split, group in working.groupby("split", sort=True):
        events = group[["session", "decision_dt"]].drop_duplicates()
        per_event = group.groupby(["session", "decision_dt"], sort=False).size()
        rows.append(
            {
                "split": str(split),
                "candidate_rows": int(len(group)),
                "decision_events": int(len(events)),
                "sessions": int(group["session"].nunique()),
                "median_candidates_per_event": float(per_event.median()) if len(per_event) else 0.0,
                "oracle_enter_fraction": float(group["oracle_action"].astype(str).eq("enter").mean()) if "oracle_action" in group.columns else 0.0,
                "negative_a_enter_fraction": float((pd.to_numeric(group.get("a_enter"), errors="coerce") < 0.0).mean()) if "a_enter" in group.columns else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_holding_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["split", "state_rows", "trades", "sessions", "oracle_hold_fraction"])
    rows = []
    for split, group in frame.groupby("split", sort=True):
        rows.append(
            {
                "split": str(split),
                "state_rows": int(len(group)),
                "trades": int(group["candidate_uid"].astype(str).nunique()) if "candidate_uid" in group.columns else 0,
                "sessions": int(group["session"].nunique()),
                "oracle_hold_fraction": float(group["oracle_holding_action"].astype(str).eq("hold").mean())
                if "oracle_holding_action" in group.columns
                else 0.0,
                "median_minutes_since_entry": float(pd.to_numeric(group.get("minutes_since_entry"), errors="coerce").median())
                if "minutes_since_entry" in group.columns
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def trajectory_foundation_decision(
    *,
    flat_rows: int,
    holding_rows: int,
    feature_contract_status: str,
    fill_model_ready: bool,
    untouched_holdout_ready: bool,
    live_parity_ready: bool,
) -> str:
    if flat_rows <= 0 or holding_rows <= 0 or feature_contract_status != "pass":
        return "blocked_unified_trajectory_foundation_missing_required_rows_or_features"
    if not (fill_model_ready and untouched_holdout_ready and live_parity_ready):
        return "unified_trajectory_foundation_ready_training_blocked_by_foundation_gates"
    return "unified_trajectory_foundation_ready_for_preregistered_training"


def trajectory_contract_payload() -> dict[str, Any]:
    return {
        **TrajectoryDatasetContractV1().to_dict(),
        "flat_feature_columns": list(FLAT_TRAJECTORY_FEATURE_COLUMNS),
        "holding_feature_columns": list(HOLDING_TRAJECTORY_FEATURE_COLUMNS),
        "excluded_holding_label_derived_columns": sorted(HOLDING_LABEL_DERIVED_COLUMNS),
        "feature_contract": validate_trajectory_feature_contract(),
    }
