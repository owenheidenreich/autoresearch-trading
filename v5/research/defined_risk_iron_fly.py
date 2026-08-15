"""One preregistered $10,000-compatible defined-risk short-premium structure."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from v5.ops.audit_causal_day_coverage import live_two_sided
from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD


ENTRY_MINUTE = "15:00"
TARGET_EXIT_MINUTE = "15:15"
LAST_MINUTE = "16:00"
WING_WIDTH_POINTS = 5.0
CONTRACT_MULTIPLIER = 100.0
LEGS = 4
STARTING_EQUITY_USD = 10_000.0
MAX_LOSS_SHARE = 0.05


class IronFlyError(RuntimeError):
    """The quote session violates the frozen defined-risk measurement law."""


@dataclass(frozen=True)
class IronFlyResult:
    row: dict[str, object]


def _live(frame: pd.DataFrame) -> pd.DataFrame:
    liquid = (
        pd.to_numeric(frame["bid_size"], errors="coerce").ge(1.0)
        & pd.to_numeric(frame["ask_size"], errors="coerce").ge(1.0)
    )
    return frame[live_two_sided(frame) & liquid]


def _entry_structure(entry: pd.DataFrame, spot: float) -> dict[str, pd.Series] | None:
    live = _live(entry)
    pairs = live.groupby(["strike", "right"], sort=True).size()
    strikes = sorted(float(value) for value in live["strike"].unique())
    candidates = []
    for strike in strikes:
        required = (
            (strike, "C"),
            (strike, "P"),
            (strike + WING_WIDTH_POINTS, "C"),
            (strike - WING_WIDTH_POINTS, "P"),
        )
        if all(key in pairs.index for key in required):
            candidates.append(strike)
    if not candidates:
        return None
    # Uses only the entry spot and entry-visible chain. A future quote cannot
    # make a different strike become the selected structure.
    center = sorted(candidates, key=lambda value: (abs(value - spot), value))[0]
    roles = {
        "short_call": (center, "C"),
        "short_put": (center, "P"),
        "long_call": (center + WING_WIDTH_POINTS, "C"),
        "long_put": (center - WING_WIDTH_POINTS, "P"),
    }
    selected: dict[str, pd.Series] = {}
    for name, (strike, right) in roles.items():
        rows = live[live["strike"].eq(strike) & live["right"].astype(str).eq(right)]
        if rows.empty:
            return None
        selected[name] = rows.sort_values("contract_id", kind="mergesort").iloc[0]
    return selected


def _entry_credit(legs: dict[str, pd.Series], price: str) -> float:
    return float(
        legs["short_call"][price]
        + legs["short_put"][price]
        - legs["long_call"][price]
        - legs["long_put"][price]
    )


def _exit_debit(legs: dict[str, pd.Series], price_short: str, price_long: str) -> float:
    return float(
        legs["short_call"][price_short]
        + legs["short_put"][price_short]
        - legs["long_call"][price_long]
        - legs["long_put"][price_long]
    )


def _settlement_debit(center: float, settlement_spx: float) -> float:
    short = abs(settlement_spx - center)
    wings = max(0.0, settlement_spx - center - WING_WIDTH_POINTS) + max(
        0.0, center - WING_WIDTH_POINTS - settlement_spx
    )
    return float(short - wings)


def evaluate_session(
    quotes: pd.DataFrame,
    *,
    session: str,
    settlement_spx: float,
) -> IronFlyResult:
    """Evaluate one causal 15:00 iron fly, including abstention as a zero session."""

    required = {
        "minute",
        "contract_id",
        "strike",
        "right",
        "bid",
        "ask",
        "mid",
        "bid_size",
        "ask_size",
        "quote_age_ms",
        "underlying_price",
    }
    missing = sorted(required - set(quotes.columns))
    if missing:
        raise IronFlyError(f"quote columns missing: {missing}")
    entry = quotes[quotes["minute"].astype(str).eq(ENTRY_MINUTE)]
    base = {
        "session": session,
        "entry_minute": ENTRY_MINUTE,
        "target_exit_minute": TARGET_EXIT_MINUTE,
        "wing_width_points": WING_WIDTH_POINTS,
        "status": "abstain_no_entry_structure",
        "traded": False,
        "net_touch_usd": 0.0,
        "gross_mid_usd": 0.0,
        "naked_short_net_touch_usd": 0.0,
        "long_straddle_net_touch_usd": 0.0,
        "declared_max_loss_usd": 0.0,
    }
    if entry.empty:
        return IronFlyResult(base)
    spot = float(pd.to_numeric(entry["underlying_price"], errors="coerce").median())
    if not np.isfinite(spot):
        raise IronFlyError(f"{session}: 15:00 spot is unavailable")
    legs = _entry_structure(entry, spot)
    if legs is None:
        return IronFlyResult(base)
    center = float(legs["short_call"]["strike"])
    # Short legs transact at the bid and long legs at the ask.
    touch_credit = float(
        legs["short_call"]["bid"]
        + legs["short_put"]["bid"]
        - legs["long_call"]["ask"]
        - legs["long_put"]["ask"]
    )
    mid_credit = _entry_credit(legs, "mid")
    fee = LEGS * FEES_PER_ROUND_TRIP_USD
    max_loss = WING_WIDTH_POINTS * CONTRACT_MULTIPLIER - touch_credit * CONTRACT_MULTIPLIER + fee
    if (
        touch_credit <= 0.0
        or touch_credit > WING_WIDTH_POINTS + 1e-9
        or max_loss > STARTING_EQUITY_USD * MAX_LOSS_SHARE
    ):
        base.update(
            {
                "status": "abstain_entry_credit_or_risk",
                "center_strike": center,
                "entry_touch_credit_usd": touch_credit * CONTRACT_MULTIPLIER,
                "declared_max_loss_usd": max(0.0, max_loss),
            }
        )
        return IronFlyResult(base)

    ids = {name: str(row["contract_id"]) for name, row in legs.items()}
    exit_minute = LAST_MINUTE
    exit_type = "validated_cash_settlement"
    touch_debit = _settlement_debit(center, settlement_spx)
    mid_debit = touch_debit
    for minute in sorted(
        value
        for value in quotes["minute"].astype(str).unique()
        if TARGET_EXIT_MINUTE <= value <= LAST_MINUTE
    ):
        snapshot = _live(quotes[quotes["minute"].astype(str).eq(minute)])
        exit_legs: dict[str, pd.Series] = {}
        for name, contract_id in ids.items():
            row = snapshot[snapshot["contract_id"].astype(str).eq(contract_id)]
            if row.empty:
                break
            exit_legs[name] = row.iloc[-1]
        if len(exit_legs) != LEGS:
            continue
        candidate_debit = _exit_debit(exit_legs, "ask", "bid")
        # A combo limit refuses to pay more than its expiry-defined width.
        if candidate_debit > WING_WIDTH_POINTS + 1e-9:
            continue
        touch_debit = candidate_debit
        mid_debit = _exit_debit(exit_legs, "mid", "mid")
        exit_minute = minute
        exit_type = "executable_combo_touch"
        break

    net_touch = (touch_credit - touch_debit) * CONTRACT_MULTIPLIER - fee
    gross_mid = (mid_credit - mid_debit) * CONTRACT_MULTIPLIER
    if net_touch < -max_loss - 1e-6:
        raise IronFlyError(f"{session}: realised loss exceeds entry-defined maximum")

    # Same short ATM legs as a naked control; same exit minute/settlement.
    short_entry = float(legs["short_call"]["bid"] + legs["short_put"]["bid"])
    if exit_type == "validated_cash_settlement":
        short_exit = abs(settlement_spx - center)
    else:
        snapshot = _live(quotes[quotes["minute"].astype(str).eq(exit_minute)])
        short_exit = sum(
            float(snapshot[snapshot["contract_id"].astype(str).eq(ids[name])].iloc[-1]["ask"])
            for name in ("short_call", "short_put")
        )
    naked_short = (short_entry - short_exit) * CONTRACT_MULTIPLIER - 2 * FEES_PER_ROUND_TRIP_USD
    long_straddle = -naked_short - 4 * FEES_PER_ROUND_TRIP_USD
    return IronFlyResult(
        {
            **base,
            "status": "traded",
            "traded": True,
            "center_strike": center,
            "entry_spot": spot,
            "entry_touch_credit_usd": touch_credit * CONTRACT_MULTIPLIER,
            "entry_mid_credit_usd": mid_credit * CONTRACT_MULTIPLIER,
            "exit_minute": exit_minute,
            "exit_type": exit_type,
            "exit_touch_debit_usd": touch_debit * CONTRACT_MULTIPLIER,
            "exit_mid_debit_usd": mid_debit * CONTRACT_MULTIPLIER,
            "fees_usd": fee,
            "net_touch_usd": net_touch,
            "gross_mid_usd": gross_mid,
            "naked_short_net_touch_usd": naked_short,
            "long_straddle_net_touch_usd": long_straddle,
            "declared_max_loss_usd": max_loss,
            "max_loss_share_of_equity": max_loss / STARTING_EQUITY_USD,
        }
    )
