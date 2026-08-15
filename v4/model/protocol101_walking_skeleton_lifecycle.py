"""Quarantined 1-second lifecycle plumbing for the Protocol101 walking skeleton.

This module is deliberately not a production policy.  It consumes only the
owner-authorized official ``cbbo-1s`` slice, fits a small HGB HOLD/EXIT baseline,
and preserves the signed floor ordering: check the floor committed at t-1,
evaluate the model, then ratchet the floor for t+1 only after HOLD.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


NY = ZoneInfo("America/New_York")
QUARANTINE_LABELS = (
    "walking_skeleton",
    "throwaway",
    "paper-only",
    "forced_buy_plumbing_harness",
)
QUARANTINE_TEXT = "|".join(QUARANTINE_LABELS)

FEATURE_NAMES = (
    "elapsed_seconds",
    "minutes_to_forced_flat",
    "entry_ask",
    "current_bid",
    "current_ask",
    "current_mid",
    "spread",
    "net_pnl_dollars",
    "net_pnl_return",
    "mfe_dollars",
    "mae_dollars",
    "giveback_dollars",
    "seconds_since_mfe",
    "bid_velocity_1s",
    "bid_velocity_5s",
    "bid_velocity_30s",
    "committed_floor_input",
    "right_is_call",
    "market_phase_code",
)

_CONTRACT = re.compile(
    r"^SPXW-(?P<expiry>\d{8})-(?P<strike>\d+(?:\.\d{3}))-(?P<right>[CP])$"
)


@dataclass(frozen=True)
class FloorSpec:
    initial_fraction: float = 0.60
    canary_initial_fraction: float = 0.99
    minimum_age_seconds: int = 30
    profit_activation_fraction: float = 0.05
    minimum_breathing_points: float = 0.25
    spread_breathing_multiple: float = 2.0
    peak_breathing_fraction: float = 0.15


def contract_id_to_raw_symbol(contract_id: str) -> str:
    """Convert the normalized Protocol101 identity to the OPRA raw symbol."""

    match = _CONTRACT.fullmatch(str(contract_id))
    if match is None:
        raise ValueError(f"unsupported SPXW contract identity: {contract_id}")
    expiry = match.group("expiry")
    strike_milli = int(round(float(match.group("strike")) * 1_000.0))
    return f"SPXW  {expiry[2:]}{match.group('right')}{strike_milli:08d}"


def _phase_code(timestamp: pd.Timestamp) -> int:
    local = timestamp.tz_convert(NY)
    minute = local.hour * 60 + local.minute
    if minute < 11 * 60:
        return 0
    if minute < 13 * 60 + 30:
        return 1
    return 2


def _reverse_window(values: pd.Series, window: int, operation: str) -> pd.Series:
    reversed_values = values.iloc[::-1]
    rolling = reversed_values.rolling(window=window, min_periods=1)
    result = rolling.max() if operation == "max" else rolling.min()
    return result.iloc[::-1].set_axis(values.index)


def build_one_second_trajectory(
    intent: dict[str, Any],
    *,
    parquet_path: Path,
    forced_flat_et: str = "15:55",
    round_trip_fee_dollars: float = 3.0,
) -> pd.DataFrame:
    """Build a causal official-cbbo-1s trajectory for one forced entry intent."""

    raw_symbol = contract_id_to_raw_symbol(str(intent["contract_id"]))
    columns = [
        "ts_event",
        "symbol",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
    ]
    quotes = pd.read_parquet(
        parquet_path,
        columns=columns,
        filters=[("symbol", "==", raw_symbol)],
    ).reset_index()
    if "ts_recv" not in quotes.columns:
        raise ValueError(f"official cbbo-1s file lacks ts_recv: {parquet_path}")
    quotes["sample_time"] = pd.to_datetime(quotes["ts_recv"], utc=True).dt.floor("s")
    quotes["event_time"] = pd.to_datetime(quotes["ts_event"], utc=True)
    for column in ("bid_px_00", "ask_px_00"):
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    quotes = quotes[
        quotes["event_time"].notna()
        & quotes["bid_px_00"].notna()
        & quotes["ask_px_00"].notna()
        & (quotes["bid_px_00"] >= 0.0)
        & (quotes["ask_px_00"] > 0.0)
        & (quotes["ask_px_00"] >= quotes["bid_px_00"])
    ].copy()
    quotes.sort_values(["sample_time", "event_time"], inplace=True)
    quotes = quotes.drop_duplicates("sample_time", keep="last")

    entry_time = pd.Timestamp(int(intent["entry_fill_time_ns"]), unit="ns", tz="UTC").ceil("s")
    session = str(intent["session"])
    hour, minute = [int(value) for value in forced_flat_et.split(":")]
    deadline = pd.Timestamp(session, tz=NY).replace(hour=hour, minute=minute).tz_convert("UTC")
    quotes = quotes[(quotes["sample_time"] >= entry_time) & (quotes["sample_time"] <= deadline)]
    if quotes.empty:
        raise ValueError(f"no official cbbo-1s quotes after entry for {session} {raw_symbol}")

    grid = pd.DataFrame({"sample_time": pd.date_range(entry_time, deadline, freq="1s")})
    frame = grid.merge(
        quotes[
            [
                "sample_time",
                "event_time",
                "bid_px_00",
                "ask_px_00",
                "bid_sz_00",
                "ask_sz_00",
            ]
        ],
        on="sample_time",
        how="left",
        validate="one_to_one",
    )
    quote_columns = [
        "event_time",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
    ]
    frame[quote_columns] = frame[quote_columns].ffill()
    frame = frame.dropna(subset=["event_time", "bid_px_00", "ask_px_00"]).copy()
    if frame.empty or frame["sample_time"].iloc[-1] != deadline:
        raise ValueError(f"official cbbo-1s trajectory does not reach 15:55 ET: {session} {raw_symbol}")

    entry_ask = float(intent["entry_ask"])
    bid = frame["bid_px_00"].astype(float)
    ask = frame["ask_px_00"].astype(float)
    pnl = (bid - entry_ask) * 100.0 - float(round_trip_fee_dollars)
    mfe = pnl.cummax()
    mae = pnl.cummin()
    maximum_bid = bid.cummax()
    maximum_index = bid.expanding().apply(lambda values: float(np.argmax(values)), raw=True)
    elapsed = (frame["sample_time"] - entry_time).dt.total_seconds().astype(int)
    seconds_since_mfe = elapsed.to_numpy() - maximum_index.to_numpy(dtype=int)
    future_max_300 = _reverse_window(bid, 301, "max")
    future_min_300 = _reverse_window(bid, 301, "min")
    future_drawdown = bid - future_min_300
    future_headroom = future_max_300 - bid

    frame["session"] = session
    frame["split"] = str(intent["split"])
    frame["contract_id"] = str(intent["contract_id"])
    frame["raw_symbol"] = raw_symbol
    frame["decision_time_ns"] = int(intent["decision_time_ns"])
    frame["entry_fill_time_ns"] = int(intent["entry_fill_time_ns"])
    frame["sample_time_ns"] = frame["sample_time"].astype("int64")
    frame["source_event_time_ns"] = frame["event_time"].astype("int64")
    frame["market_quote_age_ms"] = (
        (frame["sample_time"] - frame["event_time"]).dt.total_seconds() * 1_000.0
    ).clip(lower=0.0)
    frame["entry_ask"] = entry_ask
    frame["current_bid"] = bid
    frame["current_ask"] = ask
    frame["current_mid"] = (bid + ask) / 2.0
    frame["spread"] = ask - bid
    frame["net_pnl_dollars"] = pnl
    frame["net_pnl_return"] = pnl / max(entry_ask * 100.0 + round_trip_fee_dollars, 1e-12)
    frame["mfe_dollars"] = mfe
    frame["mae_dollars"] = mae
    frame["giveback_dollars"] = mfe - pnl
    frame["seconds_since_mfe"] = seconds_since_mfe
    frame["elapsed_seconds"] = elapsed
    frame["minutes_to_forced_flat"] = (
        (deadline - frame["sample_time"]).dt.total_seconds() / 60.0
    ).clip(lower=0.0)
    frame["bid_velocity_1s"] = bid.diff(1).fillna(0.0)
    frame["bid_velocity_5s"] = bid.diff(5).fillna(0.0)
    frame["bid_velocity_30s"] = bid.diff(30).fillna(0.0)
    frame["right_is_call"] = 1.0 if str(intent["right"]) == "C" else 0.0
    frame["market_phase_code"] = [_phase_code(value) for value in frame["sample_time"]]
    # This feature is the floor committed by the preceding second.  It is
    # derived causally from quotes through t-1 and is never recomputed from a
    # future path during model scoring.
    floor_spec = FloorSpec()
    committed = entry_ask * floor_spec.initial_fraction
    running_peak = -math.inf
    committed_inputs: list[float] = []
    for current_bid, current_ask in zip(bid.to_numpy(), ask.to_numpy()):
        committed_inputs.append(float(committed))
        running_peak = max(running_peak, float(current_bid))
        committed = next_floor(
            committed_floor=committed,
            entry_ask=entry_ask,
            maximum_bid=running_peak,
            current_spread=float(current_ask - current_bid),
            spec=floor_spec,
        )
    frame["committed_floor_input"] = committed_inputs
    frame["future_max_bid_300s_label"] = future_max_300
    frame["future_min_bid_300s_label"] = future_min_300
    frame["exit_target"] = (
        (frame["elapsed_seconds"] >= 120)
        & (frame["minutes_to_forced_flat"] > 1.0)
        & (frame["net_pnl_dollars"] > 0.0)
        & (future_headroom <= 0.05)
        & (future_drawdown >= np.maximum(0.15, 0.10 * bid))
    ).astype(np.int8)
    frame["label_definition"] = "near_next_300s_peak_with_positive_net_pnl_and_future_drawdown"
    frame["quarantine_labels"] = QUARANTINE_TEXT
    return frame


def fit_lifecycle_baseline(
    tensor: pd.DataFrame,
    *,
    fit_sessions: list[str],
    calibration_sessions: list[str],
    random_state: int = 101,
) -> tuple[HistGradientBoostingClassifier, float, dict[str, Any]]:
    """Fit the bounded HGB baseline and derive one calibration-only threshold."""

    fit = tensor[tensor["session"].isin(fit_sessions)].copy()
    calibration = tensor[tensor["session"].isin(calibration_sessions)].copy()
    fit = fit[(fit["elapsed_seconds"] % 5 == 0) | fit["exit_target"].astype(bool)]
    if fit.empty or fit["exit_target"].nunique() != 2:
        raise ValueError("fit tensor does not contain both HOLD and EXIT targets")
    x_fit = fit.loc[:, FEATURE_NAMES].astype(float).fillna(0.0)
    y_fit = fit["exit_target"].astype(int).to_numpy()
    positives = max(1, int(y_fit.sum()))
    negatives = max(1, int((y_fit == 0).sum()))
    weights = np.where(y_fit == 1, negatives / positives, 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=100,
        max_depth=3,
        min_samples_leaf=80,
        l2_regularization=1.0,
        random_state=random_state,
    )
    model.fit(x_fit, y_fit, sample_weight=weights)

    x_cal = calibration.loc[:, FEATURE_NAMES].astype(float).fillna(0.0)
    y_cal = calibration["exit_target"].astype(int).to_numpy()
    probability = model.predict_proba(x_cal)[:, 1]
    positive_probability = probability[y_cal == 1]
    threshold = 0.65
    if len(positive_probability):
        threshold = float(np.clip(np.quantile(positive_probability, 0.25), 0.45, 0.85))
    metrics: dict[str, Any] = {
        "family": "sklearn HistGradientBoostingClassifier",
        "fit_rows": int(len(fit)),
        "fit_sessions": list(fit_sessions),
        "calibration_rows": int(len(calibration)),
        "calibration_sessions": list(calibration_sessions),
        "fit_hold_targets": int((y_fit == 0).sum()),
        "fit_exit_targets": int((y_fit == 1).sum()),
        "calibration_hold_targets": int((y_cal == 0).sum()),
        "calibration_exit_targets": int((y_cal == 1).sum()),
        "threshold": threshold,
        "threshold_rule": "calibration_exit_probability_q25_clipped_0_45_0_85",
        "formal_model_quality_claim": False,
        "quarantine_labels": list(QUARANTINE_LABELS),
    }
    if len(np.unique(y_cal)) == 2:
        metrics["calibration_roc_auc_descriptive"] = float(roc_auc_score(y_cal, probability))
        metrics["calibration_average_precision_descriptive"] = float(
            average_precision_score(y_cal, probability)
        )
    else:
        metrics["calibration_roc_auc_descriptive"] = None
        metrics["calibration_average_precision_descriptive"] = None
    return model, threshold, metrics


def attach_exit_probabilities(
    tensor: pd.DataFrame,
    model: HistGradientBoostingClassifier,
) -> pd.DataFrame:
    out = tensor.copy()
    out["exit_probability"] = model.predict_proba(
        out.loc[:, FEATURE_NAMES].astype(float).fillna(0.0)
    )[:, 1]
    return out


def next_floor(
    *,
    committed_floor: float,
    entry_ask: float,
    maximum_bid: float,
    current_spread: float,
    spec: FloorSpec,
) -> float:
    """Return the D52 breathing-room floor; it may stay or rise, never fall."""

    candidate = float(entry_ask) * float(spec.initial_fraction)
    if maximum_bid >= entry_ask * (1.0 + spec.profit_activation_fraction):
        breathing = max(
            spec.minimum_breathing_points,
            spec.spread_breathing_multiple * max(0.0, current_spread),
            spec.peak_breathing_fraction * maximum_bid,
        )
        candidate = max(candidate, maximum_bid - breathing)
    return max(float(committed_floor), float(candidate))


def apply_lifecycle_policy(
    trajectory: pd.DataFrame,
    *,
    threshold: float,
    route: str,
    floor_spec: FloorSpec | None = None,
    minimum_model_age_seconds: int = 120,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply floor-before-model arbitration and return one terminal exit."""

    if route not in {
        "full_policy",
        "learned_exit_path_canary",
        "floor_path_canary",
        "forced_flat_path_canary",
    }:
        raise ValueError(f"unknown lifecycle route: {route}")
    spec = floor_spec or FloorSpec()
    ordered = trajectory.sort_values("sample_time_ns").reset_index(drop=True)
    entry_ask = float(ordered["entry_ask"].iloc[0])
    canary_floor = route == "floor_path_canary"
    floor_enabled = route != "forced_flat_path_canary"
    model_enabled = route in {"full_policy", "learned_exit_path_canary"}
    model_threshold = 0.0 if route == "learned_exit_path_canary" else float(threshold)
    committed_floor = entry_ask * (
        spec.canary_initial_fraction if canary_floor else spec.initial_fraction
    )
    maximum_bid = -math.inf
    actions: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None

    for index, row in ordered.iterrows():
        current_bid = float(row["current_bid"])
        current_ask = float(row["current_ask"])
        maximum_bid = max(maximum_bid, current_bid)
        elapsed = int(row["elapsed_seconds"])
        is_deadline = float(row["minutes_to_forced_flat"]) <= 0.0
        action = "HOLD"
        trigger = "model_and_floor_hold"
        floor_used = committed_floor

        if (
            floor_enabled
            and elapsed >= spec.minimum_age_seconds
            and current_bid <= committed_floor
        ):
            action = "EXIT"
            trigger = "floor_trigger"
        elif (
            model_enabled
            and elapsed >= minimum_model_age_seconds
            and float(row["exit_probability"]) >= model_threshold
        ):
            action = "EXIT"
            trigger = "learned_exit"
        elif is_deadline:
            action = "EXIT"
            trigger = "forced_flat"

        event = {
            "session": str(row["session"]),
            "split": str(row["split"]),
            "contract_id": str(row["contract_id"]),
            "decision_time_ns": int(row["decision_time_ns"]),
            "lifecycle_time_ns": int(row["sample_time_ns"]),
            "action": action,
            "trigger": trigger,
            "route": route,
            "current_bid": current_bid,
            "current_ask": current_ask,
            "exit_probability": float(row["exit_probability"]),
            "model_threshold": model_threshold,
            "committed_floor_from_previous_second": float(floor_used),
            "floor_gap_points": float(current_bid - floor_used),
            "quarantine_labels": QUARANTINE_TEXT,
        }
        actions.append(event)
        if action == "EXIT":
            exit_index = index if is_deadline else min(index + 1, len(ordered) - 1)
            exit_row = ordered.iloc[exit_index]
            exit_bid = float(exit_row["current_bid"])
            terminal = {
                **event,
                "action_time_ns": int(row["sample_time_ns"]),
                "exit_time_ns": int(exit_row["sample_time_ns"]),
                "exit_source_event_time_ns": int(exit_row["source_event_time_ns"]),
                "exit_bid": exit_bid,
                "exit_market_quote_age_ms": float(exit_row["market_quote_age_ms"]),
                "entry_ask": entry_ask,
                "pnl_after_fee": (exit_bid - entry_ask) * 100.0 - 3.0,
                "hold_seconds": int(
                    (int(exit_row["sample_time_ns"]) - int(row["entry_fill_time_ns"]))
                    / 1_000_000_000
                ),
                "peak_bid": float(maximum_bid),
                "floor_slippage_points": (
                    float(floor_used - exit_bid) if trigger == "floor_trigger" else None
                ),
            }
            break

        if floor_enabled:
            committed_floor = next_floor(
                committed_floor=committed_floor,
                entry_ask=entry_ask,
                maximum_bid=maximum_bid,
                current_spread=current_ask - current_bid,
                spec=spec,
            )
        actions[-1]["floor_committed_for_next_second"] = float(committed_floor)

    if terminal is None:
        raise ValueError("trajectory ended without a terminal lifecycle action")
    return terminal, actions


def apply_entry_safety_prefilter(
    candidates: list[dict[str, Any]],
    *,
    starting_cash: float = 10_000.0,
    fee_dollars: float = 3.0,
    daily_budget_fraction: float = 0.05,
    max_trades_per_session: int = 6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply D48/D49, affordability, cutoff, and one-position safety masks."""

    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: (
            str(item["session"]),
            int(item["decision_time_ns"]),
            str(item["contract_id"]),
        ),
    )
    cash = float(starting_cash)
    active_session: str | None = None
    session_start = cash
    realized_session_pnl = 0.0
    session_trades = 0
    pending: dict[str, Any] | None = None
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    session_summaries: dict[str, dict[str, Any]] = {}

    def realize_if_due(decision_ns: int) -> None:
        nonlocal cash, realized_session_pnl, pending
        if pending is not None and int(pending["exit_time_ns"]) <= decision_ns:
            pnl = float(pending["pnl_after_fee"])
            cash += pnl
            realized_session_pnl += pnl
            pending = None

    for item in ordered:
        session = str(item["session"])
        decision_ns = int(item["decision_time_ns"])
        if active_session != session:
            if pending is not None:
                cash += float(pending["pnl_after_fee"])
                pending = None
            active_session = session
            session_start = cash
            realized_session_pnl = 0.0
            session_trades = 0
            session_summaries[session] = {
                "session_start_equity": session_start,
                "accepted": 0,
                "rejected": 0,
                "quarantine_labels": list(QUARANTINE_LABELS),
            }
        realize_if_due(decision_ns)
        entry_ask = float(item["entry_ask"])
        premium_plus_fee = entry_ask * 100.0 + fee_dollars
        daily_budget = daily_budget_fraction * session_start
        realized_loss = max(0.0, -realized_session_pnl)
        remaining_budget = daily_budget - realized_loss
        local = pd.Timestamp(decision_ns, unit="ns", tz="UTC").tz_convert(NY)
        reason: str | None = None
        if pending is not None:
            reason = "one_open_position_overlap"
        elif (local.hour, local.minute) >= (15, 30):
            reason = "at_or_after_15_30_entry_cutoff"
        elif session_trades >= max_trades_per_session:
            reason = "six_trade_session_cap"
        elif not math.isfinite(entry_ask) or entry_ask < 1.0:
            reason = "d49_sub_one_dollar_soft_close_exclusion"
        elif remaining_budget + 1e-9 < 103.0:
            reason = "d49_session_soft_closed_no_one_dollar_contract_fits"
        elif premium_plus_fee > daily_budget + 1e-9:
            reason = "d48_premium_cap_mask"
        elif realized_loss + premium_plus_fee > daily_budget + 1e-9:
            reason = "d49_remaining_budget_mask"
        elif premium_plus_fee > cash + 1e-9:
            reason = "unaffordable"

        audit = {
            **item,
            "cash_before": cash,
            "session_start_equity": session_start,
            "realized_session_pnl_before": realized_session_pnl,
            "realized_session_loss_before": realized_loss,
            "daily_budget": daily_budget,
            "remaining_daily_budget": remaining_budget,
            "premium_plus_fee_required": premium_plus_fee,
            "d48_pass": premium_plus_fee <= daily_budget + 1e-9,
            "d49_pass": realized_loss + premium_plus_fee <= daily_budget + 1e-9,
            "soft_close_active": remaining_budget + 1e-9 < 103.0,
            "quarantine_labels": QUARANTINE_TEXT,
        }
        if reason is not None:
            audit["entry_safety_outcome"] = "WAIT"
            audit["entry_safety_reason"] = reason
            rejected.append(audit)
            session_summaries[session]["rejected"] += 1
            continue

        audit["entry_safety_outcome"] = "BUY"
        audit["entry_safety_reason"] = None
        accepted.append(audit)
        pending = audit
        session_trades += 1
        session_summaries[session]["accepted"] += 1

    if pending is not None:
        cash += float(pending["pnl_after_fee"])
    for session, summary in session_summaries.items():
        summary["accepted_between_one_and_six"] = 1 <= int(summary["accepted"]) <= 6
    aggregate = {
        "starting_cash": float(starting_cash),
        "ending_cash": cash,
        "sessions": session_summaries,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "all_accepted_d48_pass": all(bool(item["d48_pass"]) for item in accepted),
        "all_accepted_d49_pass": all(bool(item["d49_pass"]) for item in accepted),
        "all_sessions_one_to_six": all(
            bool(item["accepted_between_one_and_six"])
            for item in session_summaries.values()
        ),
        "quarantine_labels": list(QUARANTINE_LABELS),
    }
    return accepted, rejected, aggregate
