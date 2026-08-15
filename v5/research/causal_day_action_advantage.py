"""One-trade serial action-value labels for the causal SPX day trader.

These are supervised outcomes, never policy-visible features.  A flat policy
with one trade available may either buy one currently eligible contract or
preserve the slot for a later minute.  The fixed 120-minute payoff comes from
the settlement-complete causal-day candidate table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.audit_causal_day_coverage import ENTRY_MINUTES
from v5.research.causal_day_attribution import time_band


LABEL_NAME = "serial_action_advantage_120m"
HORIZON_MINUTES = 120
TRADE_CAP = 1
ALLOWED_EXIT_TYPES = frozenset(("executable_bid", "validated_cash_settlement"))

REQUIRED_COLUMNS = (
    "session",
    "entry_minute",
    "entry_regime",
    "trade_id",
    "contract_id",
    "right",
    "self_delta",
    "entry_ask_usd",
    "entry_mid_usd",
    "net_bid_120m_usd",
    "net_mid_120m_usd",
    "clock_exit_minute_120m",
    "clock_exit_type_120m",
    "realised_hold_120m",
)


class ActionAdvantageError(RuntimeError):
    """The immutable payoff table cannot satisfy the declared label law."""


@dataclass(frozen=True)
class LabelledSession:
    candidates: pd.DataFrame
    minutes: pd.DataFrame
    summary: dict[str, Any]


def _future_wait(best_enter: np.ndarray) -> np.ndarray:
    """Best value strictly after each minute, including the no-trade value 0."""

    values = np.asarray(best_enter, dtype=float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ActionAdvantageError("minute-best enter values must be one finite-or-missing vector")
    wait = np.zeros(len(values), dtype=float)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        wait[index] = running
        if np.isfinite(values[index]):
            running = max(running, float(values[index]))
    return wait


def _best_by_minute(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        frame.sort_values(
            ["entry_minute", value_column, "contract_id", "trade_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates("entry_minute", keep="first")
        .sort_values("entry_minute", kind="mergesort")
        .reset_index(drop=True)
    )


def label_one_session(frame: pd.DataFrame) -> LabelledSession:
    """Attach Q(enter), Q(wait), and action advantage to one complete session.

    ``Q_wait(t)`` excludes minute ``t``.  If the same maximum occurs at several
    times, the primary oracle uses the earliest minute and then the
    lexicographically first contract.  The oracle is diagnostic only; the
    future fit will regress the dense Q values.
    """

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ActionAdvantageError(f"candidate input columns missing: {missing}")
    source = frame.loc[:, REQUIRED_COLUMNS].copy()
    sessions = source["session"].astype(str).unique()
    if len(sessions) != 1:
        raise ActionAdvantageError("label_one_session requires exactly one session")
    session = str(sessions[0])
    source["session"] = source["session"].astype(str)
    source["entry_minute"] = source["entry_minute"].astype(str)
    source["contract_id"] = source["contract_id"].astype(str)
    source["trade_id"] = source["trade_id"].astype(str)
    if source["trade_id"].duplicated().any():
        raise ActionAdvantageError(f"{session}: duplicate trade_id")
    observed_minutes = set(source["entry_minute"].unique())
    outside = sorted(observed_minutes - set(ENTRY_MINUTES))
    if outside:
        raise ActionAdvantageError(f"{session}: candidate minute outside 09:35--15:00: {outside}")
    if not set(source["clock_exit_type_120m"].astype(str)).issubset(ALLOWED_EXIT_TYPES):
        raise ActionAdvantageError(f"{session}: blocked or unknown 120-minute exit type")
    for column in (
        "self_delta",
        "entry_ask_usd",
        "entry_mid_usd",
        "net_bid_120m_usd",
        "net_mid_120m_usd",
        "realised_hold_120m",
    ):
        source[column] = pd.to_numeric(source[column], errors="coerce")
        if not np.isfinite(source[column].to_numpy(float)).all():
            raise ActionAdvantageError(f"{session}: {column} is missing or non-finite")
    if not source["entry_ask_usd"].between(0.0, 10_000.0, inclusive="right").all():
        raise ActionAdvantageError(f"{session}: action is not affordable by the $10,000 account")

    source["q_enter_bid_120m_usd"] = source["net_bid_120m_usd"]
    source["q_enter_mid_120m_usd"] = source["net_mid_120m_usd"]
    best_bid = _best_by_minute(source, "q_enter_bid_120m_usd")
    best_mid = _best_by_minute(source, "q_enter_mid_120m_usd")
    best_bid = best_bid.set_index("entry_minute").reindex(ENTRY_MINUTES)
    best_mid = best_mid.set_index("entry_minute").reindex(ENTRY_MINUTES)
    q_wait_bid = _future_wait(best_bid["q_enter_bid_120m_usd"].to_numpy(float))
    q_wait_mid = _future_wait(best_mid["q_enter_mid_120m_usd"].to_numpy(float))
    if (np.diff(q_wait_bid) > 1e-12).any() or (np.diff(q_wait_mid) > 1e-12).any():
        raise ActionAdvantageError(f"{session}: wait value is not non-increasing")

    minute = pd.DataFrame(
        {
            "session": session,
            "entry_minute": list(ENTRY_MINUTES),
            "q_wait_bid_120m_usd": q_wait_bid,
            "q_wait_mid_120m_usd": q_wait_mid,
            "best_q_enter_bid_120m_usd": best_bid["q_enter_bid_120m_usd"].to_numpy(float),
            "best_q_enter_mid_120m_usd": best_mid["q_enter_mid_120m_usd"].to_numpy(float),
            "best_bid_contract_id": best_bid["contract_id"].where(best_bid["contract_id"].notna(), None).to_numpy(),
            "best_mid_contract_id": best_mid["contract_id"].where(best_mid["contract_id"].notna(), None).to_numpy(),
        }
    )
    minute["best_a_enter_bid_120m_usd"] = (
        minute["best_q_enter_bid_120m_usd"] - minute["q_wait_bid_120m_usd"]
    )
    minute["best_a_enter_mid_120m_usd"] = (
        minute["best_q_enter_mid_120m_usd"] - minute["q_wait_mid_120m_usd"]
    )
    minute["time_band"] = minute["entry_minute"].map(time_band)

    wait_bid = minute.set_index("entry_minute")["q_wait_bid_120m_usd"]
    wait_mid = minute.set_index("entry_minute")["q_wait_mid_120m_usd"]
    source["q_wait_bid_120m_usd"] = source["entry_minute"].map(wait_bid)
    source["q_wait_mid_120m_usd"] = source["entry_minute"].map(wait_mid)
    source["a_enter_bid_120m_usd"] = (
        source["q_enter_bid_120m_usd"] - source["q_wait_bid_120m_usd"]
    )
    source["a_enter_mid_120m_usd"] = (
        source["q_enter_mid_120m_usd"] - source["q_wait_mid_120m_usd"]
    )

    global_bid = max(0.0, float(best_bid["q_enter_bid_120m_usd"].max(skipna=True)))
    global_mid = max(0.0, float(best_mid["q_enter_mid_120m_usd"].max(skipna=True)))
    selected_trade: str | None = None
    selected_minute: str | None = None
    selected_contract: str | None = None
    selected_right: str | None = None
    if global_bid > 0.0:
        winning = best_bid[
            np.isclose(best_bid["q_enter_bid_120m_usd"], global_bid)
        ].reset_index()
        chosen = winning.sort_values(
            ["entry_minute", "contract_id", "trade_id"], kind="mergesort"
        ).iloc[0]
        selected_trade = str(chosen["trade_id"])
        selected_minute = str(chosen["entry_minute"])
        selected_contract = str(chosen["contract_id"])
        selected_right = str(chosen["right"])
    source["is_primary_oracle_action"] = (
        source["trade_id"].eq(selected_trade) if selected_trade is not None else False
    )
    minute["is_primary_oracle_minute"] = (
        minute["entry_minute"].eq(selected_minute) if selected_minute is not None else False
    )
    expected_actions = int(global_bid > 0.0)
    if int(source["is_primary_oracle_action"].sum()) != expected_actions:
        raise ActionAdvantageError(f"{session}: primary oracle action is not unique")

    source = source.sort_values(
        ["entry_minute", "contract_id", "trade_id"], kind="mergesort"
    ).reset_index(drop=True)
    summary = {
        "session": session,
        "candidate_rows": int(len(source)),
        "decision_minutes": int(len(minute)),
        "minutes_without_eligible_action": int(
            minute["best_q_enter_bid_120m_usd"].isna().sum()
        ),
        "global_oracle_bid_120m_usd": global_bid,
        "global_oracle_mid_120m_usd": global_mid,
        "oracle_entry_minute": selected_minute,
        "oracle_contract_id": selected_contract,
        "oracle_right": selected_right,
        "oracle_time_band": time_band(selected_minute) if selected_minute else None,
        "positive_q_enter_bid_rows": int(source["q_enter_bid_120m_usd"].gt(0.0).sum()),
        "positive_a_enter_bid_rows": int(source["a_enter_bid_120m_usd"].gt(0.0).sum()),
        "positive_best_advantage_minutes": int(
            minute["best_a_enter_bid_120m_usd"].gt(0.0).sum()
        ),
    }
    return LabelledSession(source, minute, summary)
