"""Pure causal account-state adapter used for Phase-0b offline receipts."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class AccountEvent:
    event_time_ns: int
    kind: str
    pnl_dollars: float = 0.0


@dataclass(frozen=True)
class CausalAccountState:
    entries_so_far: int = 0
    realized_session_pnl_dollars: float = 0.0
    last_exit_time_ns: int | None = None
    loss_streak_so_far: int = 0
    position_occupancy: int = 0
    initial_entry_budget_dollars: float = 500.0


def apply_account_event(state: CausalAccountState, event: AccountEvent) -> CausalAccountState:
    if event.event_time_ns < 0:
        raise ValueError("account event time must be nonnegative")
    if event.kind == "ENTRY_FILL":
        if state.position_occupancy != 0:
            raise ValueError("entry fill while occupied")
        return replace(state, entries_so_far=state.entries_so_far + 1, position_occupancy=1)
    if event.kind == "EXIT_FILL":
        if state.position_occupancy != 1:
            raise ValueError("exit fill while flat")
        realized = state.realized_session_pnl_dollars + float(event.pnl_dollars)
        streak = state.loss_streak_so_far + 1 if event.pnl_dollars < 0.0 else 0
        return replace(
            state,
            realized_session_pnl_dollars=realized,
            last_exit_time_ns=event.event_time_ns,
            loss_streak_so_far=streak,
            position_occupancy=0,
        )
    raise ValueError(f"unknown account event kind: {event.kind}")


def replay_account_events(events: Iterable[AccountEvent]) -> CausalAccountState:
    state = CausalAccountState()
    prior_time = -1
    for event in events:
        if event.event_time_ns < prior_time:
            raise ValueError("account events are not causal-order sorted")
        state = apply_account_event(state, event)
        prior_time = event.event_time_ns
    return state


def account_feature_snapshot(state: CausalAccountState, *, decision_time_ns: int) -> dict[str, float]:
    if state.last_exit_time_ns is not None and decision_time_ns < state.last_exit_time_ns:
        raise ValueError("decision precedes last exit")
    seconds_since = -1.0 if state.last_exit_time_ns is None else (decision_time_ns - state.last_exit_time_ns) / 1_000_000_000.0
    return {
        "entries_so_far": float(state.entries_so_far),
        "realized_session_pnl_dollars": float(state.realized_session_pnl_dollars),
        "seconds_since_last_exit": float(seconds_since),
        "loss_streak_so_far": float(state.loss_streak_so_far),
        "position_occupancy": float(state.position_occupancy),
        "remaining_entry_budget_dollars": float(
            max(0.0, state.initial_entry_budget_dollars + state.realized_session_pnl_dollars)
        ),
    }


def historical_account_feature_snapshot(state: CausalAccountState, *, decision_time_ns: int) -> dict[str, float]:
    return account_feature_snapshot(state, decision_time_ns=decision_time_ns)


def live_account_feature_snapshot(state: CausalAccountState, *, decision_time_ns: int) -> dict[str, float]:
    return account_feature_snapshot(state, decision_time_ns=decision_time_ns)
