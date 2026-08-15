"""Deterministic, minute-boundary $10,000 SPXW 0DTE replay simulator.

The simulator contains no strategy.  A supplied policy receives only the
current boundary state and returns ``ABSTAIN``, ``BUY``, ``HOLD``, or ``SELL``.
The same engine can later host a shallow control, a sequence policy, or a
human-authored action ledger without changing fill/account laws.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Protocol

import numpy as np
import pandas as pd

from v5.ops.audit_causal_day_coverage import (
    ENTRY_MINUTES,
    FIRST_DECISION_MINUTE,
    LAST_ENTRY_MINUTE,
    LAST_QUOTE_MINUTE,
    QUOTE_MINUTES,
    live_two_sided,
)
from v5.ops.build_causal_day_dataset import (
    full_ladder_state,
    minute_number,
    prepare_quotes,
    regime_for_minute,
)
from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD


ActionKind = Literal["ABSTAIN", "BUY", "HOLD", "SELL"]
RiskMode = Literal["ticket_only", "ticket_and_breaker"]

STARTING_EQUITY_USD = 10_000.0
DAILY_BREAKER_SHARE = 0.05
MAX_HOLD_MINUTES = 120
CONTRACT_MULTIPLIER = 100.0


class SimulatorError(RuntimeError):
    """An action or input violates a deterministic simulator law."""


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    contract_id: str | None = None
    reason: str = ""
    probability: float | None = None
    candidate_probabilities: Mapping[str, float] | None = None
    diagnostics: Mapping[str, Any] | None = None


@dataclass
class Position:
    contract_id: str
    raw_symbol: str
    right: str
    strike: float
    origin_regime: str
    entry_minute: str
    entry_ask: float
    entry_ask_usd: float
    entry_mid: float
    entry_mid_usd: float
    entry_underlying_price: float
    entry_moneyness_itm_points: float
    maximum_unrealised_usd: float = -np.inf
    minimum_unrealised_usd: float = np.inf
    maximum_itm_depth_points: float = -np.inf
    maximum_underlying_favourable_points: float = 0.0
    minimum_underlying_favourable_points: float = 0.0
    first_cross_minute: str | None = None
    pending_exit_reason: str | None = None
    requested_exit_minute: str | None = None


@dataclass(frozen=True)
class DecisionState:
    session: str
    minute: str
    role: str
    cash_usd: float
    realised_pnl_usd: float
    trades_opened: int
    trade_cap: int
    breaker_triggered: bool
    position: Position | None
    position_quote: pd.Series | None
    entry_candidates: pd.DataFrame
    ladder_snapshot: pd.DataFrame
    causal_features: pd.Series | None


class Policy(Protocol):
    def __call__(self, state: DecisionState) -> Action: ...


@dataclass
class SimulationResult:
    session: str
    risk_mode: RiskMode
    trade_cap: int
    starting_equity_usd: float
    ending_cash_usd: float | None
    realised_pnl_usd: float
    trades: pd.DataFrame
    events: pd.DataFrame
    considered_ladder: pd.DataFrame
    blocked_terminal_position: bool
    unresolved_position: Position | None = field(repr=False)


def role_for_state(minute: str, position: Position | None) -> str:
    if position is not None:
        return f"{position.origin_regime}_exit"
    return f"{regime_for_minute(minute)}_entry"


def _snapshot(quotes: pd.DataFrame, minute: str) -> pd.DataFrame:
    return quotes[quotes["minute"].eq(minute)].copy()


def _executable_exit_row(snapshot: pd.DataFrame, contract_id: str) -> pd.Series | None:
    match = snapshot[snapshot["contract_id"].eq(contract_id)]
    if match.empty:
        return None
    valid = live_two_sided(match) & match["bid_size"].ge(1.0)
    if not valid.any():
        return None
    return match[valid].iloc[-1]


def _mark_unrealised(position: Position, snapshot: pd.DataFrame) -> None:
    row = _executable_exit_row(snapshot, position.contract_id)
    if row is None:
        return
    value = (
        float(row["bid"]) * CONTRACT_MULTIPLIER
        - position.entry_ask_usd
        - FEES_PER_ROUND_TRIP_USD
    )
    position.maximum_unrealised_usd = max(position.maximum_unrealised_usd, value)
    position.minimum_unrealised_usd = min(position.minimum_unrealised_usd, value)


def _snapshot_spx(snapshot: pd.DataFrame) -> float | None:
    values = pd.to_numeric(snapshot["underlying_price"], errors="coerce").dropna().unique()
    return float(values[0]) if len(values) == 1 else None


def _itm_points(position: Position, spx: float) -> float:
    return spx - position.strike if position.right == "C" else position.strike - spx


def _mark_underlying_path(position: Position, snapshot: pd.DataFrame, minute: str) -> None:
    spx = _snapshot_spx(snapshot)
    if spx is None:
        return
    favourable = (
        spx - position.entry_underlying_price
        if position.right == "C"
        else position.entry_underlying_price - spx
    )
    itm = _itm_points(position, spx)
    position.maximum_underlying_favourable_points = max(
        position.maximum_underlying_favourable_points, favourable
    )
    position.minimum_underlying_favourable_points = min(
        position.minimum_underlying_favourable_points, favourable
    )
    position.maximum_itm_depth_points = max(position.maximum_itm_depth_points, itm)
    if position.first_cross_minute is None and itm >= 0.0:
        position.first_cross_minute = minute


def _path_fields(position: Position, minute: str, final_spx: float | None) -> dict[str, object]:
    final_itm = _itm_points(position, final_spx) if final_spx is not None else np.nan
    return {
        "entry_underlying_price": position.entry_underlying_price,
        "entry_moneyness_itm_points": position.entry_moneyness_itm_points,
        "entry_otm_depth_points": max(0.0, -position.entry_moneyness_itm_points),
        "maximum_itm_depth_points": position.maximum_itm_depth_points,
        "final_itm_depth_points": final_itm,
        "otm_to_itm_conversion": position.first_cross_minute is not None,
        "first_cross_minute": position.first_cross_minute,
        "time_to_cross_minutes": (
            minute_number(position.first_cross_minute) - minute_number(position.entry_minute)
            if position.first_cross_minute is not None
            else np.nan
        ),
        "underlying_mfe_points": position.maximum_underlying_favourable_points,
        "underlying_mae_points": position.minimum_underlying_favourable_points,
        "final_underlying_price": final_spx,
        "path_last_minute": minute,
    }


def _position_minutes(position: Position, minute: str) -> int:
    return minute_number(minute) - minute_number(position.entry_minute)


def _validate_action_metadata(action: Action, eligible_ids: set[str]) -> None:
    if action.probability is not None and (
        not np.isfinite(action.probability) or not 0.0 <= action.probability <= 1.0
    ):
        raise SimulatorError("action probability must be finite and inside [0,1]")
    if action.candidate_probabilities is None:
        return
    unknown = set(action.candidate_probabilities) - eligible_ids
    if unknown:
        raise SimulatorError(f"candidate probabilities name ineligible contracts: {sorted(unknown)}")
    for contract_id, value in action.candidate_probabilities.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SimulatorError(
                f"candidate probability for {contract_id} must be finite and inside [0,1]"
            )


class ActionLedger:
    """Known-answer policy keyed by minute and routed role."""

    def __init__(self, actions: dict[tuple[str, str], Action]):
        self.actions = dict(actions)
        self.carries_session_state = False

    def __call__(self, state: DecisionState) -> Action:
        default = Action("HOLD") if state.position is not None else Action("ABSTAIN")
        return self.actions.get((state.minute, state.role), default)


def simulate_session(
    raw_quotes: pd.DataFrame,
    session: str,
    policy: Policy,
    *,
    trade_cap: int,
    risk_mode: RiskMode = "ticket_only",
    validated_settlement_spx: float | None = None,
    terminal_zero_recovery: bool = False,
    minute_features: pd.DataFrame | None = None,
    capture_replay_state: bool = False,
) -> SimulationResult:
    """Replay one session under the frozen minute/fill/account laws."""

    if trade_cap not in (1, 2, 3):
        raise SimulatorError("trade_cap must be one of the declared 1/2/3 family")
    if risk_mode not in ("ticket_only", "ticket_and_breaker"):
        raise SimulatorError(f"unknown risk mode: {risk_mode}")
    if bool(getattr(policy, "carries_session_state", False)):
        reset = getattr(policy, "reset_session", None)
        if not callable(reset):
            raise SimulatorError(
                "a stateful policy must implement reset_session(session) for episode isolation"
            )
        reset(session)
    quotes = prepare_quotes(raw_quotes, session)
    available_minutes = set(quotes["minute"])
    if set(QUOTE_MINUTES) - available_minutes:
        raise SimulatorError(f"{session}: simulator requires the complete declared quote clock")
    snapshots = {minute: group.copy() for minute, group in quotes.groupby("minute", sort=False)}
    policy_ladder, _ = full_ladder_state(quotes, session)
    ladder_snapshots = {
        minute: group.copy()
        for minute, group in policy_ladder.groupby("minute", sort=False)
    }
    feature_rows: dict[str, pd.Series] = {}
    if minute_features is not None:
        scoped = minute_features[minute_features["session"].astype(str).eq(session)].copy()
        if scoped["minute"].duplicated().any():
            raise SimulatorError(f"{session}: duplicate causal feature minute")
        feature_rows = {str(row["minute"]): row for _, row in scoped.iterrows()}

    cash = STARTING_EQUITY_USD
    realised = 0.0
    position: Position | None = None
    trades_opened = 0
    breaker_triggered = False
    blocked_terminal = False
    event_rows: list[dict] = []
    trade_rows: list[dict] = []
    considered_rows: list[dict[str, Any]] = []

    def event(minute: str, role: str, action: Action, status: str, **extra: object) -> None:
        event_rows.append(
            {
                "session": session,
                "minute": minute,
                "role": role,
                "action": action.kind,
                "requested_contract_id": action.contract_id,
                "reason": action.reason,
                "action_probability": action.probability,
                "policy_diagnostics_json": (
                    json.dumps(action.diagnostics, sort_keys=True, default=str)
                    if action.diagnostics is not None
                    else None
                ),
                "status": status,
                "cash_usd": cash,
                "realised_pnl_usd": realised,
                "trades_opened": trades_opened,
                "breaker_triggered": breaker_triggered,
                **extra,
            }
        )

    def close_at_bid(minute: str, row: pd.Series, reason: str, role: str) -> None:
        nonlocal cash, realised, position, breaker_triggered
        assert position is not None
        exit_bid = float(row["bid"])
        exit_mid = float(row["mid"])
        final_spx = float(row["underlying_price"])
        proceeds = exit_bid * CONTRACT_MULTIPLIER - FEES_PER_ROUND_TRIP_USD
        pnl = proceeds - position.entry_ask_usd
        cash += proceeds
        realised += pnl
        trade_rows.append(
            {
                "session": session,
                "contract_id": position.contract_id,
                "raw_symbol": position.raw_symbol,
                "right": position.right,
                "strike": position.strike,
                "origin_regime": position.origin_regime,
                "entry_minute": position.entry_minute,
                "exit_minute": minute,
                "entry_ask": position.entry_ask,
                "entry_mid": position.entry_mid,
                "exit_bid": exit_bid,
                "exit_mid": exit_mid,
                "entry_ask_usd": position.entry_ask_usd,
                "entry_mid_usd": position.entry_mid_usd,
                "exit_proceeds_after_fee_usd": proceeds,
                "entry_half_spread_usd": (
                    position.entry_ask - position.entry_mid
                )
                * CONTRACT_MULTIPLIER,
                "exit_half_spread_usd": (exit_mid - exit_bid) * CONTRACT_MULTIPLIER,
                "total_spread_paid_usd": (
                    position.entry_ask - position.entry_mid + exit_mid - exit_bid
                )
                * CONTRACT_MULTIPLIER,
                "fees_usd": FEES_PER_ROUND_TRIP_USD,
                "net_pnl_usd": pnl,
                "minutes_held": _position_minutes(position, minute),
                "maximum_unrealised_usd": position.maximum_unrealised_usd,
                "minimum_unrealised_usd": position.minimum_unrealised_usd,
                "exit_reason": reason,
                "exit_type": "executable_bid",
                "requested_exit_minute": position.requested_exit_minute,
                **_path_fields(position, minute, final_spx),
            }
        )
        event(minute, role, Action("SELL", position.contract_id, reason), "filled_bid", fill_price=exit_bid)
        position = None
        if risk_mode == "ticket_and_breaker" and realised <= -STARTING_EQUITY_USD * DAILY_BREAKER_SHARE:
            breaker_triggered = True

    def settle_terminal(minute: str, settlement_spx: float, role: str) -> None:
        nonlocal cash, realised, position, breaker_triggered
        assert position is not None
        intrinsic = (
            max(0.0, settlement_spx - position.strike)
            if position.right == "C"
            else max(0.0, position.strike - settlement_spx)
        )
        proceeds = intrinsic * CONTRACT_MULTIPLIER - FEES_PER_ROUND_TRIP_USD
        pnl = proceeds - position.entry_ask_usd
        cash += proceeds
        realised += pnl
        trade_rows.append(
            {
                "session": session,
                "contract_id": position.contract_id,
                "raw_symbol": position.raw_symbol,
                "right": position.right,
                "strike": position.strike,
                "origin_regime": position.origin_regime,
                "entry_minute": position.entry_minute,
                "exit_minute": minute,
                "entry_ask": position.entry_ask,
                "entry_mid": position.entry_mid,
                "exit_bid": np.nan,
                "exit_mid": np.nan,
                "entry_ask_usd": position.entry_ask_usd,
                "entry_mid_usd": position.entry_mid_usd,
                "exit_proceeds_after_fee_usd": proceeds,
                "entry_half_spread_usd": (
                    position.entry_ask - position.entry_mid
                )
                * CONTRACT_MULTIPLIER,
                "exit_half_spread_usd": np.nan,
                "total_spread_paid_usd": np.nan,
                "fees_usd": FEES_PER_ROUND_TRIP_USD,
                "net_pnl_usd": pnl,
                "minutes_held": _position_minutes(position, minute),
                "maximum_unrealised_usd": position.maximum_unrealised_usd,
                "minimum_unrealised_usd": position.minimum_unrealised_usd,
                "exit_reason": "session_close_no_executable_bid",
                "exit_type": "validated_cash_settlement",
                "settlement_spx": settlement_spx,
                "requested_exit_minute": position.requested_exit_minute,
                **_path_fields(position, minute, settlement_spx),
            }
        )
        event(
            minute,
            role,
            Action("SELL", position.contract_id, "terminal_cash_settlement"),
            "cash_settled",
            settlement_spx=settlement_spx,
            intrinsic=intrinsic,
        )
        position = None
        if risk_mode == "ticket_and_breaker" and realised <= -STARTING_EQUITY_USD * DAILY_BREAKER_SHARE:
            breaker_triggered = True

    def settle_zero_recovery(minute: str, role: str, terminal_spx: float | None) -> None:
        """Declared conservative sensitivity, never an executable fill claim."""

        nonlocal cash, realised, position, breaker_triggered
        assert position is not None
        proceeds = -FEES_PER_ROUND_TRIP_USD
        pnl = proceeds - position.entry_ask_usd
        cash += proceeds
        realised += pnl
        trade_rows.append(
            {
                "session": session,
                "contract_id": position.contract_id,
                "raw_symbol": position.raw_symbol,
                "right": position.right,
                "strike": position.strike,
                "origin_regime": position.origin_regime,
                "entry_minute": position.entry_minute,
                "exit_minute": minute,
                "entry_ask": position.entry_ask,
                "entry_mid": position.entry_mid,
                "exit_bid": np.nan,
                "exit_mid": np.nan,
                "entry_ask_usd": position.entry_ask_usd,
                "entry_mid_usd": position.entry_mid_usd,
                "exit_proceeds_after_fee_usd": proceeds,
                "entry_half_spread_usd": (
                    position.entry_ask - position.entry_mid
                )
                * CONTRACT_MULTIPLIER,
                "exit_half_spread_usd": np.nan,
                "total_spread_paid_usd": np.nan,
                "fees_usd": FEES_PER_ROUND_TRIP_USD,
                "net_pnl_usd": pnl,
                "minutes_held": _position_minutes(position, minute),
                "maximum_unrealised_usd": position.maximum_unrealised_usd,
                "minimum_unrealised_usd": position.minimum_unrealised_usd,
                "exit_reason": "session_close_no_executable_bid",
                "exit_type": "zero_recovery_sensitivity",
                "requested_exit_minute": position.requested_exit_minute,
                **_path_fields(position, minute, terminal_spx),
            }
        )
        event(
            minute,
            role,
            Action("SELL", position.contract_id, "terminal_zero_recovery"),
            "zero_recovery_sensitivity",
        )
        position = None
        if risk_mode == "ticket_and_breaker" and realised <= -STARTING_EQUITY_USD * DAILY_BREAKER_SHARE:
            breaker_triggered = True

    for minute in QUOTE_MINUTES:
        if minute < FIRST_DECISION_MINUTE:
            continue
        snapshot = snapshots[minute]
        ladder_snapshot = ladder_snapshots.get(
            minute, policy_ladder.iloc[0:0].copy()
        )
        if position is not None:
            _mark_unrealised(position, snapshot)
            _mark_underlying_path(position, snapshot, minute)
        role = role_for_state(minute, position)

        # A previously submitted marketable exit owns the next action. It fills
        # at the first later executable bid and cannot be cancelled for a better
        # price in this deterministic simulator.
        if position is not None and position.pending_exit_reason is not None:
            row = _executable_exit_row(snapshot, position.contract_id)
            if row is not None:
                close_at_bid(minute, row, position.pending_exit_reason, role)
                continue
            if minute == LAST_QUOTE_MINUTE:
                if validated_settlement_spx is not None:
                    settle_terminal(minute, validated_settlement_spx, role)
                elif terminal_zero_recovery:
                    settle_zero_recovery(minute, role, _snapshot_spx(snapshot))
                else:
                    blocked_terminal = True
                    event(
                        minute,
                        role,
                        Action("SELL", position.contract_id, position.pending_exit_reason),
                        "blocked_no_bid_and_unvalidated_settlement",
                    )
                break
            event(
                minute,
                role,
                Action("SELL", position.contract_id, position.pending_exit_reason),
                "pending_no_executable_bid",
            )
            continue

        # The hold clock or close submits an exit before consulting a policy.
        if position is not None and (
            _position_minutes(position, minute) >= MAX_HOLD_MINUTES
            or minute == LAST_QUOTE_MINUTE
        ):
            reason = "maximum_hold" if minute != LAST_QUOTE_MINUTE else "session_close"
            position.pending_exit_reason = reason
            position.requested_exit_minute = minute
            row = _executable_exit_row(snapshot, position.contract_id)
            if row is not None:
                close_at_bid(minute, row, reason, role)
                continue
            if minute == LAST_QUOTE_MINUTE:
                if validated_settlement_spx is not None:
                    settle_terminal(minute, validated_settlement_spx, role)
                elif terminal_zero_recovery:
                    settle_zero_recovery(minute, role, _snapshot_spx(snapshot))
                else:
                    blocked_terminal = True
                    event(
                        minute,
                        role,
                        Action("SELL", position.contract_id, reason),
                        "blocked_no_bid_and_unvalidated_settlement",
                    )
                break
            event(minute, role, Action("SELL", position.contract_id, reason), "pending_no_executable_bid")
            continue

        entry_candidates = ladder_snapshot[
            ladder_snapshot["entry_eligible"]
            & ladder_snapshot["minute"].isin(ENTRY_MINUTES)
            & (ladder_snapshot["ask"] * CONTRACT_MULTIPLIER <= cash)
        ].reset_index(drop=True)
        position_quote: pd.Series | None = None
        if position is not None:
            held = snapshot[snapshot["contract_id"].eq(position.contract_id)]
            if not held.empty:
                position_quote = held.iloc[-1].copy(deep=True)
        state = DecisionState(
            session=session,
            minute=minute,
            role=role,
            cash_usd=cash,
            realised_pnl_usd=realised,
            trades_opened=trades_opened,
            trade_cap=trade_cap,
            breaker_triggered=breaker_triggered,
            position=replace(position) if position is not None else None,
            position_quote=position_quote,
            entry_candidates=entry_candidates.copy(deep=True),
            ladder_snapshot=ladder_snapshot.copy(deep=True),
            causal_features=(
                feature_rows[minute].copy(deep=True)
                if minute in feature_rows
                else None
            ),
        )
        action = policy(state)
        if action.kind not in ("ABSTAIN", "BUY", "HOLD", "SELL"):
            raise SimulatorError(f"policy returned unknown action: {action.kind}")
        eligible_ids = set(entry_candidates["contract_id"].astype(str))
        _validate_action_metadata(action, eligible_ids)
        if capture_replay_state:
            probabilities = action.candidate_probabilities or {}
            for candidate in ladder_snapshot.to_dict("records"):
                contract_id = str(candidate["contract_id"])
                considered_rows.append(
                    {
                        **candidate,
                        "decision_role": role,
                        "selected": bool(
                            action.kind == "BUY" and action.contract_id == contract_id
                        ),
                        "policy_probability": probabilities.get(contract_id),
                        "emitted_action": action.kind,
                        "action_probability": action.probability,
                        "action_reason": action.reason,
                    }
                )

        if position is None:
            if action.kind in ("ABSTAIN", "HOLD"):
                event(minute, role, action, "flat_no_action", eligible_contracts=len(entry_candidates))
                continue
            if action.kind == "SELL":
                event(minute, role, action, "rejected_sell_while_flat")
                continue
            if minute > LAST_ENTRY_MINUTE:
                event(minute, role, action, "rejected_outside_entry_clock")
                continue
            if trades_opened >= trade_cap:
                event(minute, role, action, "rejected_trade_cap")
                continue
            if risk_mode == "ticket_and_breaker" and breaker_triggered:
                event(minute, role, action, "rejected_daily_breaker")
                continue
            if action.contract_id is None:
                event(minute, role, action, "rejected_missing_contract")
                continue
            selected = entry_candidates[entry_candidates["contract_id"].eq(action.contract_id)]
            if selected.empty:
                event(minute, role, action, "rejected_ineligible_contract")
                continue
            row = selected.iloc[-1]
            debit = float(row["ask"]) * CONTRACT_MULTIPLIER
            if debit > cash:
                event(minute, role, action, "rejected_buying_power", required_cash_usd=debit)
                continue
            cash -= debit
            position = Position(
                contract_id=str(row["contract_id"]),
                raw_symbol=str(row["raw_symbol"]),
                right=str(row["right"]),
                strike=float(row["strike"]),
                origin_regime=regime_for_minute(minute),
                entry_minute=minute,
                entry_ask=float(row["ask"]),
                entry_ask_usd=debit,
                entry_mid=float(row["mid"]),
                entry_mid_usd=float(row["mid"]) * CONTRACT_MULTIPLIER,
                entry_underlying_price=float(row["underlying_price"]),
                entry_moneyness_itm_points=float(row["moneyness_itm_points"]),
                maximum_itm_depth_points=float(row["moneyness_itm_points"]),
            )
            trades_opened += 1
            _mark_unrealised(position, snapshot)
            _mark_underlying_path(position, snapshot, minute)
            event(
                minute,
                role,
                action,
                "filled_ask",
                fill_price=float(row["ask"]),
                eligible_contracts=len(entry_candidates),
            )
            continue

        # Holding branch.
        if action.kind in ("HOLD", "ABSTAIN"):
            event(minute, role, action, "held")
            continue
        if action.kind == "BUY":
            event(minute, role, action, "rejected_overlapping_position")
            continue
        position.pending_exit_reason = action.reason or "policy_sell"
        position.requested_exit_minute = minute
        row = _executable_exit_row(snapshot, position.contract_id)
        if row is None:
            event(minute, role, action, "pending_no_executable_bid")
            continue
        close_at_bid(minute, row, position.pending_exit_reason, role)

    ending = None if blocked_terminal else cash
    return SimulationResult(
        session=session,
        risk_mode=risk_mode,
        trade_cap=trade_cap,
        starting_equity_usd=STARTING_EQUITY_USD,
        ending_cash_usd=ending,
        realised_pnl_usd=realised,
        trades=pd.DataFrame(trade_rows),
        events=pd.DataFrame(event_rows),
        considered_ladder=pd.DataFrame(considered_rows),
        blocked_terminal_position=blocked_terminal,
        unresolved_position=position,
    )
