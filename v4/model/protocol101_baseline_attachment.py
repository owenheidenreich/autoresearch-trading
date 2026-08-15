"""Protocol101 baseline-action attachment for unified policy trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


ROLE_LABEL = "DATASET_PROTOCOL101_BASELINE_ACTION_ATTACHMENT_V1"
BASELINE_ACTION_SOURCE = "PAPER_DEFAULT_PROTOCOL101"


@dataclass(frozen=True)
class BaselineAttachmentSummary:
    flat_events: int
    event_seed_rows: int
    baseline_trades: int
    matched_entry_trades: int
    unmatched_entry_trades: int
    baseline_trades_without_event: int
    splits_with_flat_events: tuple[str, ...]
    splits_with_baseline: tuple[str, ...]
    splits_without_baseline: tuple[str, ...]
    baseline_splits_without_flat_events: tuple[str, ...]
    action_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "flat_events": self.flat_events,
            "event_seed_rows": self.event_seed_rows,
            "baseline_trades": self.baseline_trades,
            "matched_entry_trades": self.matched_entry_trades,
            "unmatched_entry_trades": self.unmatched_entry_trades,
            "baseline_trades_without_event": self.baseline_trades_without_event,
            "splits_with_flat_events": list(self.splits_with_flat_events),
            "splits_with_baseline": list(self.splits_with_baseline),
            "splits_without_baseline": list(self.splits_without_baseline),
            "baseline_splits_without_flat_events": list(self.baseline_splits_without_flat_events),
            "action_counts": self.action_counts,
        }


def build_protocol101_baseline_event_actions(flat_candidates: pd.DataFrame, baseline_trades: pd.DataFrame) -> pd.DataFrame:
    """Attach seed-specific Protocol101 actions to flat decision events.

    The output is event-level rather than candidate-level. It records whether
    Protocol101 would wait, enter a matched surface candidate, be holding a
    prior position, or be outside baseline coverage at each decision timestamp.
    """

    required_flat = {"split", "session", "decision_dt", "decision_time", "candidate_uid", "contract_id"}
    required_trades = {"reported_split", "seed", "session", "decision_time", "exit_time", "contract_id", "candidate_uid", "pnl"}
    missing_flat = sorted(required_flat - set(flat_candidates.columns))
    missing_trades = sorted(required_trades - set(baseline_trades.columns))
    if missing_flat:
        raise ValueError(f"missing flat candidate columns: {missing_flat}")
    if missing_trades:
        raise ValueError(f"missing baseline trade columns: {missing_trades}")

    flat = flat_candidates.copy()
    flat["decision_dt"] = pd.to_datetime(flat["decision_dt"], utc=True, errors="coerce")
    flat = flat[flat["decision_dt"].notna()].copy()
    events = (
        flat[["split", "session", "decision_dt", "decision_time"]]
        .drop_duplicates(["split", "session", "decision_dt"])
        .sort_values(["split", "session", "decision_dt"])
        .reset_index(drop=True)
    )
    candidates = (
        flat[["split", "session", "decision_dt", "contract_id", "candidate_uid"]]
        .drop_duplicates(["split", "session", "decision_dt", "contract_id"])
        .rename(columns={"candidate_uid": "surface_candidate_uid"})
    )
    trades = normalize_protocol101_trades(baseline_trades)
    rows: list[dict[str, Any]] = []
    baseline_splits = set(trades["split"].astype(str).unique())
    baseline_seeds_by_split = {
        str(split): sorted(group["seed"].astype(int).unique())
        for split, group in trades.groupby("split", sort=True)
    }
    for (split, session), session_events in events.groupby(["split", "session"], sort=True):
        split = str(split)
        session = str(session)
        if split not in baseline_splits:
            for event in session_events.itertuples(index=False):
                rows.append(base_event_row(event, seed=0, action="baseline_not_available_for_split"))
            continue
        session_trades = trades[(trades["split"].eq(split)) & (trades["session"].eq(session))]
        seeds = baseline_seeds_by_split.get(split, [])
        for seed in seeds:
            seed_trades = session_trades[session_trades["seed"].eq(seed)]
            if seed_trades.empty:
                for event in session_events.itertuples(index=False):
                    rows.append(base_event_row(event, seed=int(seed), action="wait"))
                continue
            ordered_trades = seed_trades.sort_values(["entry_dt", "exit_dt"]).reset_index(drop=True)
            rows.extend(event_actions_for_seed(session_events, int(seed), ordered_trades))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(
        candidates,
        on=["split", "session", "decision_dt", "contract_id"],
        how="left",
    )
    out["surface_candidate_uid"] = out["surface_candidate_uid"].fillna("")
    out["baseline_action_source"] = BASELINE_ACTION_SOURCE
    return out.sort_values(["split", "session", "seed", "decision_dt"]).reset_index(drop=True)


def normalize_protocol101_trades(frame: pd.DataFrame) -> pd.DataFrame:
    trades = frame.copy()
    trades["split"] = trades["reported_split"].astype(str)
    trades["session"] = trades["session"].astype(str)
    trades["seed"] = pd.to_numeric(trades["seed"], errors="coerce").fillna(0).astype(int)
    trades["entry_dt"] = pd.to_datetime(trades["decision_time"], utc=True, errors="coerce")
    trades["exit_dt"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    return trades[trades["entry_dt"].notna() & trades["exit_dt"].notna()].reset_index(drop=True)


def event_actions_for_seed(events: pd.DataFrame, seed: int, trades: pd.DataFrame) -> list[dict[str, Any]]:
    ordered_events = events.sort_values("decision_dt")
    trade_rows = list(trades.sort_values(["entry_dt", "exit_dt"]).to_dict("records"))
    rows: list[dict[str, Any]] = []
    trade_idx = 0
    for event in ordered_events.itertuples(index=False):
        event_dt = pd.Timestamp(event.decision_dt)
        while trade_idx < len(trade_rows) and pd.Timestamp(trade_rows[trade_idx]["exit_dt"]) < event_dt:
            trade_idx += 1
        if trade_idx >= len(trade_rows):
            rows.append(base_event_row(event, seed=seed, action="wait"))
            continue
        same_time_entries = [row for row in trade_rows if pd.Timestamp(row["entry_dt"]) == event_dt]
        if same_time_entries:
            rows.append(_trade_event_row(event, seed, "enter", same_time_entries[0]))
            continue
        trade = trade_rows[trade_idx]
        entry_dt = pd.Timestamp(trade["entry_dt"])
        exit_dt = pd.Timestamp(trade["exit_dt"])
        if entry_dt < event_dt < exit_dt:
            rows.append(_trade_event_row(event, seed, "holding", trade))
        elif exit_dt == event_dt:
            rows.append(_trade_event_row(event, seed, "exit_then_wait", trade))
        else:
            rows.append(base_event_row(event, seed=seed, action="wait"))
    return rows


def event_action_for_seed(event: Any, seed: int, trades: pd.DataFrame) -> dict[str, Any]:
    """Compatibility helper for tests and narrow callers."""

    return event_actions_for_seed(pd.DataFrame([event._asdict() if hasattr(event, "_asdict") else dict(event)]), seed, trades)[0]


def _trade_event_row(event: Any, seed: int, action: str, trade: dict[str, Any] | pd.Series) -> dict[str, Any]:
    return base_event_row(
        event,
        seed=seed,
        action=action,
        contract_id=str(trade["contract_id"]),
        baseline_trade_candidate_uid=str(trade["candidate_uid"]),
        baseline_exit_time=str(trade["exit_time"]),
        baseline_trade_pnl=float(trade["pnl"]),
    )


def base_event_row(
    event: Any,
    *,
    seed: int,
    action: str,
    contract_id: str = "",
    baseline_trade_candidate_uid: str = "",
    baseline_exit_time: str = "",
    baseline_trade_pnl: float = 0.0,
) -> dict[str, Any]:
    return {
        "split": str(event.split),
        "session": str(event.session),
        "decision_time": str(event.decision_time),
        "decision_dt": pd.Timestamp(event.decision_dt),
        "seed": int(seed),
        "protocol101_action": action,
        "contract_id": contract_id,
        "baseline_trade_candidate_uid": baseline_trade_candidate_uid,
        "baseline_exit_time": baseline_exit_time,
        "baseline_trade_pnl": float(baseline_trade_pnl),
    }


def summarize_protocol101_baseline_attachment(actions: pd.DataFrame, baseline_trades: pd.DataFrame) -> BaselineAttachmentSummary:
    trades = normalize_protocol101_trades(baseline_trades) if not baseline_trades.empty else pd.DataFrame()
    if actions.empty:
        return BaselineAttachmentSummary(0, 0, len(trades), 0, 0, len(trades), (), (), (), (), {})
    entry_actions = actions[actions["protocol101_action"].eq("enter")]
    matched = int(entry_actions["surface_candidate_uid"].astype(str).ne("").sum()) if "surface_candidate_uid" in entry_actions.columns else 0
    unmatched = int(len(entry_actions) - matched)
    splits_with_flat = tuple(sorted(actions["split"].astype(str).unique()))
    splits_with_baseline = tuple(sorted(trades["split"].astype(str).unique())) if not trades.empty else ()
    splits_without_baseline = tuple(sorted(set(splits_with_flat) - set(splits_with_baseline)))
    baseline_splits_without_flat = tuple(sorted(set(splits_with_baseline) - set(splits_with_flat)))
    return BaselineAttachmentSummary(
        flat_events=int(actions[["split", "session", "decision_dt"]].drop_duplicates().shape[0]),
        event_seed_rows=int(len(actions)),
        baseline_trades=int(len(trades)),
        matched_entry_trades=matched,
        unmatched_entry_trades=unmatched,
        baseline_trades_without_event=max(0, int(len(trades) - len(entry_actions))),
        splits_with_flat_events=splits_with_flat,
        splits_with_baseline=splits_with_baseline,
        splits_without_baseline=splits_without_baseline,
        baseline_splits_without_flat_events=baseline_splits_without_flat,
        action_counts={str(key): int(value) for key, value in actions["protocol101_action"].value_counts().sort_index().items()},
    )
