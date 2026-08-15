"""Unified serial DP oracle manifest for conservative policy training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import pandas as pd


ROLE_LABEL = "DATASET_UNIFIED_SERIAL_DP_ORACLE_V1"
READY_DECISION = "unified_serial_dp_oracle_ready_for_baseline_aligned_training_scope"
BLOCKED_DECISION = "unified_serial_dp_oracle_blocked_missing_required_alignment"


@dataclass(frozen=True)
class SerialDpOracleManifestV1:
    training_splits: tuple[str, ...]
    excluded_sessions_with_missing_holding_paths: int
    included_sessions: int
    flat_candidate_rows: int
    flat_decision_events: int
    flat_oracle_enter_events: int
    holding_state_rows: int
    holding_oracle_trades: int
    baseline_event_seed_rows: int
    baseline_seed_count: int
    holding_coverage_of_oracle_entries: float
    baseline_coverage_of_flat_events: float
    disallowed_baseline_actions: dict[str, int]
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_serial_dp_oracle_scope(
    flat: pd.DataFrame,
    holding: pd.DataFrame,
    baseline_actions: pd.DataFrame,
    path_skips: pd.DataFrame,
    *,
    training_splits: Iterable[str],
) -> dict[str, Any]:
    splits = tuple(str(item) for item in training_splits)
    flat_scope = normalize_flat(flat)
    holding_scope = normalize_holding(holding)
    baseline_scope = normalize_baseline_actions(baseline_actions)
    skip_scope = normalize_path_skips(path_skips)

    flat_scope = flat_scope[flat_scope["split"].isin(splits)].copy()
    holding_scope = holding_scope[holding_scope["split"].isin(splits)].copy()
    baseline_scope = baseline_scope[baseline_scope["split"].isin(splits)].copy()
    skip_scope = skip_scope[skip_scope["split"].isin(splits)].copy()

    skipped_sessions = set(zip(skip_scope["split"].astype(str), skip_scope["session"].astype(str)))
    flat_sessions = flat_scope[["split", "session"]].drop_duplicates()
    included_sessions_frame = flat_sessions[
        ~flat_sessions.apply(lambda row: (str(row["split"]), str(row["session"])) in skipped_sessions, axis=1)
    ].copy()
    included_sessions = set(zip(included_sessions_frame["split"].astype(str), included_sessions_frame["session"].astype(str)))

    flat_included = keep_sessions(flat_scope, included_sessions)
    holding_included = keep_sessions(holding_scope, included_sessions)
    baseline_included = keep_sessions(baseline_scope, included_sessions)

    flat_events = build_flat_event_oracle(flat_included)
    holding_manifest = build_holding_trade_manifest(holding_included)
    session_manifest = build_session_manifest(flat_sessions, included_sessions, skipped_sessions)
    manifest = summarize_scope(splits, flat_included, flat_events, holding_included, holding_manifest, baseline_included, session_manifest)
    return {
        "manifest": manifest,
        "flat_events": flat_events,
        "holding_trade_manifest": holding_manifest,
        "session_manifest": session_manifest,
    }


def normalize_flat(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = out["split"].astype(str)
    out["session"] = out["session"].astype(str)
    out["decision_dt"] = pd.to_datetime(out["decision_dt"], utc=True, errors="coerce")
    for column in ["q_wait", "q_enter", "a_enter", "session_oracle_value", "best_enter_value_at_decision", "entry_premium"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[out["decision_dt"].notna()].copy()


def normalize_holding(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = out["split"].astype(str)
    out["session"] = out["session"].astype(str)
    out["state_time"] = pd.to_datetime(out["state_time"], utc=True, errors="coerce")
    for column in ["q_exit", "q_hold", "a_hold", "a_switch", "current_pnl"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out[out["state_time"].notna()].copy()


def normalize_baseline_actions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = out["split"].astype(str)
    out["session"] = out["session"].astype(str)
    out["decision_dt"] = pd.to_datetime(out["decision_dt"], utc=True, errors="coerce")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(0).astype(int)
    return out[out["decision_dt"].notna()].copy()


def normalize_path_skips(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["split", "session", "skip_reason"])
    out = frame.copy()
    out["split"] = out["split"].astype(str)
    out["session"] = out["session"].astype(str)
    return out


def keep_sessions(frame: pd.DataFrame, sessions: set[tuple[str, str]]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame.apply(lambda row: (str(row["split"]), str(row["session"])) in sessions, axis=1)
    return frame[mask].copy()


def build_flat_event_oracle(flat: pd.DataFrame) -> pd.DataFrame:
    if flat.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "session",
                "decision_dt",
                "decision_time",
                "q_wait",
                "session_oracle_value",
                "best_enter_value_at_decision",
                "oracle_action",
                "oracle_action_uid",
                "oracle_contract_id",
                "oracle_a_enter",
                "candidate_count",
            ]
        )
    rows: list[dict[str, Any]] = []
    for keys, group in flat.groupby(["split", "session", "decision_dt"], sort=True):
        split, session, decision_dt = keys
        first = group.iloc[0]
        oracle_uid = str(first.get("oracle_action_uid", "wait"))
        oracle_rows = group[group["candidate_uid"].astype(str).eq(oracle_uid)]
        oracle = oracle_rows.iloc[0] if not oracle_rows.empty else None
        rows.append(
            {
                "split": str(split),
                "session": str(session),
                "decision_dt": pd.Timestamp(decision_dt),
                "decision_time": str(first.get("decision_time", "")),
                "q_wait": float(first.get("q_wait", 0.0)),
                "session_oracle_value": float(first.get("session_oracle_value", 0.0)),
                "best_enter_value_at_decision": float(first.get("best_enter_value_at_decision", 0.0)),
                "oracle_action": "enter" if oracle is not None else "wait",
                "oracle_action_uid": oracle_uid,
                "oracle_contract_id": str(oracle.get("contract_id", "")) if oracle is not None else "",
                "oracle_a_enter": float(oracle.get("a_enter", 0.0)) if oracle is not None else 0.0,
                "candidate_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_holding_trade_manifest(holding: pd.DataFrame) -> pd.DataFrame:
    if holding.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "session",
                "candidate_uid",
                "state_rows",
                "entry_time",
                "first_state_time",
                "last_state_time",
                "hold_rows",
                "exit_rows",
                "median_a_hold",
            ]
        )
    rows: list[dict[str, Any]] = []
    for keys, group in holding.groupby(["split", "session", "candidate_uid"], sort=True):
        split, session, candidate_uid = keys
        rows.append(
            {
                "split": str(split),
                "session": str(session),
                "candidate_uid": str(candidate_uid),
                "state_rows": int(len(group)),
                "entry_time": str(group["entry_time"].iloc[0]) if "entry_time" in group.columns else "",
                "first_state_time": pd.Timestamp(group["state_time"].min()).isoformat(),
                "last_state_time": pd.Timestamp(group["state_time"].max()).isoformat(),
                "hold_rows": int(group["oracle_holding_action"].astype(str).eq("hold").sum()),
                "exit_rows": int(group["oracle_holding_action"].astype(str).eq("exit").sum()),
                "median_a_hold": float(pd.to_numeric(group["a_hold"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def build_session_manifest(
    flat_sessions: pd.DataFrame,
    included_sessions: set[tuple[str, str]],
    skipped_sessions: set[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for _, row in flat_sessions.sort_values(["split", "session"]).iterrows():
        key = (str(row["split"]), str(row["session"]))
        rows.append(
            {
                "split": key[0],
                "session": key[1],
                "included": key in included_sessions,
                "excluded_reason": "missing_holding_path" if key in skipped_sessions else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_scope(
    splits: tuple[str, ...],
    flat: pd.DataFrame,
    flat_events: pd.DataFrame,
    holding: pd.DataFrame,
    holding_manifest: pd.DataFrame,
    baseline_actions: pd.DataFrame,
    session_manifest: pd.DataFrame,
) -> SerialDpOracleManifestV1:
    oracle_enter_uids = set(flat_events.loc[flat_events["oracle_action"].eq("enter"), "oracle_action_uid"].astype(str))
    holding_uids = set(holding_manifest["candidate_uid"].astype(str)) if not holding_manifest.empty else set()
    holding_coverage = len(oracle_enter_uids & holding_uids) / len(oracle_enter_uids) if oracle_enter_uids else 0.0
    baseline_event_count = baseline_actions[["split", "session", "decision_dt"]].drop_duplicates().shape[0] if not baseline_actions.empty else 0
    flat_event_count = int(len(flat_events))
    baseline_coverage = baseline_event_count / flat_event_count if flat_event_count else 0.0
    disallowed = {}
    if not baseline_actions.empty:
        bad = baseline_actions[baseline_actions["protocol101_action"].astype(str).eq("baseline_not_available_for_split")]
        disallowed = {str(key): int(value) for key, value in bad["protocol101_action"].value_counts().items()}
    decision = READY_DECISION
    if (
        flat.empty
        or holding.empty
        or flat_event_count <= 0
        or holding_coverage < 1.0
        or baseline_coverage < 1.0
        or disallowed
    ):
        decision = BLOCKED_DECISION
    return SerialDpOracleManifestV1(
        training_splits=splits,
        excluded_sessions_with_missing_holding_paths=int((~session_manifest["included"].astype(bool)).sum())
        if not session_manifest.empty
        else 0,
        included_sessions=int(session_manifest["included"].astype(bool).sum()) if not session_manifest.empty else 0,
        flat_candidate_rows=int(len(flat)),
        flat_decision_events=flat_event_count,
        flat_oracle_enter_events=int(len(oracle_enter_uids)),
        holding_state_rows=int(len(holding)),
        holding_oracle_trades=int(len(holding_uids)),
        baseline_event_seed_rows=int(len(baseline_actions)),
        baseline_seed_count=int(baseline_actions["seed"].nunique()) if not baseline_actions.empty else 0,
        holding_coverage_of_oracle_entries=float(holding_coverage),
        baseline_coverage_of_flat_events=float(baseline_coverage),
        disallowed_baseline_actions=disallowed,
        decision=decision,
    )
