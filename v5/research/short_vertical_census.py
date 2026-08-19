"""Preregistered defined-risk short-vertical census on the owned quote corpus.

Owner-authorized 2026-08-15: one $0 census, no purchase, no broker contact, no
trading, with a pre-committed STOP if no cell clears fee-only execution. The
measurement is rule-based with zero fitted parameters. It mirrors the causal
laws of `defined_risk_iron_fly.py`: strikes are chosen from the entry snapshot
only, short legs price at the bid and long legs at the ask for touch, and the
position holds to validated cash settlement so the spread toll is paid exactly
once, at entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from v5.ops.audit_causal_day_coverage import live_two_sided
from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD


SIDES = ("call_credit", "put_credit")
WIDTH_POINTS = (5.0, 10.0)
OTM_DISTANCE_POINTS = (0.0, 10.0, 20.0)
ENTRY_MINUTES = ("13:00", "14:00", "15:00")
CONTRACT_MULTIPLIER = 100.0
LEGS = 2
STARTING_EQUITY_USD = 10_000.0
MAX_LOSS_USD = 500.0  # the 5% daily breaker on the $10,000 account
PASSIVE_ADVERSE_SELECTION_USD = 86.0  # job 30's conservative per-order bound

FAMILY = tuple(
    {
        "cell": f"{side}_w{int(width)}_d{int(distance)}_{minute.replace(':', '')}",
        "side": side,
        "width_points": width,
        "otm_distance_points": distance,
        "entry_minute": minute,
    }
    for side, width, distance, minute in product(
        SIDES, WIDTH_POINTS, OTM_DISTANCE_POINTS, ENTRY_MINUTES
    )
)
FAMILY_SIZE = len(FAMILY)


class VerticalCensusError(RuntimeError):
    """The quote session violates the frozen census measurement law."""


@dataclass(frozen=True)
class SessionCells:
    rows: list[dict[str, object]]


def _live(frame: pd.DataFrame) -> pd.DataFrame:
    liquid = (
        pd.to_numeric(frame["bid_size"], errors="coerce").ge(1.0)
        & pd.to_numeric(frame["ask_size"], errors="coerce").ge(1.0)
    )
    return frame[live_two_sided(frame) & liquid]


def _select_legs(
    live: pd.DataFrame, *, side: str, width: float, distance: float, spot: float
) -> dict[str, pd.Series] | None:
    """Choose short and long legs from the entry snapshot only.

    Call credit: short call at the strike nearest (spot + distance), long call
    `width` points further OTM. Put credit mirrors below spot. Among strikes
    where both legs are live, the one nearest the target wins; ties take the
    lower strike. A later quote can never change the selection.
    """

    right = "C" if side == "call_credit" else "P"
    sign = 1.0 if side == "call_credit" else -1.0
    target = spot + sign * distance
    rows = live[live["right"].astype(str).eq(right)]
    strikes = sorted(float(value) for value in rows["strike"].unique())
    candidates = []
    for strike in strikes:
        if side == "call_credit" and strike < spot:
            continue
        if side == "put_credit" and strike > spot:
            continue
        long_strike = strike + sign * width
        if long_strike in strikes or any(
            abs(long_strike - other) < 1e-9 for other in strikes
        ):
            candidates.append(strike)
    if not candidates:
        return None
    short_strike = sorted(candidates, key=lambda value: (abs(value - target), value))[0]
    long_strike = short_strike + sign * width
    legs = {}
    for name, strike in (("short", short_strike), ("long", long_strike)):
        match = rows[rows["strike"].astype(float).sub(strike).abs().lt(1e-9)]
        if match.empty:
            return None
        legs[name] = match.sort_values("contract_id", kind="mergesort").iloc[0]
    return legs


def _settlement_debit(
    *, side: str, short_strike: float, long_strike: float, settlement_spx: float
) -> float:
    if side == "call_credit":
        short = max(0.0, settlement_spx - short_strike)
        long = max(0.0, settlement_spx - long_strike)
    else:
        short = max(0.0, short_strike - settlement_spx)
        long = max(0.0, long_strike - settlement_spx)
    return float(short - long)


def evaluate_session(
    quotes: pd.DataFrame, *, session: str, settlement_spx: float
) -> SessionCells:
    """Evaluate all 36 census cells for one session; abstentions score zero."""

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
        raise VerticalCensusError(f"quote columns missing: {missing}")

    snapshots: dict[str, tuple[pd.DataFrame, float]] = {}
    for minute in ENTRY_MINUTES:
        entry = quotes[quotes["minute"].astype(str).eq(minute)]
        if entry.empty:
            continue
        spot = float(pd.to_numeric(entry["underlying_price"], errors="coerce").median())
        if not np.isfinite(spot):
            continue
        snapshots[minute] = (_live(entry), spot)

    fee = LEGS * FEES_PER_ROUND_TRIP_USD
    rows = []
    for cell in FAMILY:
        base = {
            "session": session,
            **cell,
            "status": "abstain_no_snapshot",
            "traded": False,
            "net_touch_usd": 0.0,
            "net_fee_only_usd": 0.0,
            "net_passive_diag_usd": 0.0,
            "declared_max_loss_usd": 0.0,
        }
        snapshot = snapshots.get(str(cell["entry_minute"]))
        if snapshot is None:
            rows.append(base)
            continue
        live, spot = snapshot
        legs = _select_legs(
            live,
            side=str(cell["side"]),
            width=float(cell["width_points"]),
            distance=float(cell["otm_distance_points"]),
            spot=spot,
        )
        if legs is None:
            base["status"] = "abstain_no_leg_pair"
            rows.append(base)
            continue
        touch_credit = float(legs["short"]["bid"] - legs["long"]["ask"])
        mid_credit = float(legs["short"]["mid"] - legs["long"]["mid"])
        width = float(cell["width_points"])
        max_loss = (width - touch_credit) * CONTRACT_MULTIPLIER + fee
        if touch_credit <= 0.0 or touch_credit >= width or max_loss > MAX_LOSS_USD:
            base.update(
                {
                    "status": "abstain_credit_or_risk",
                    "entry_touch_credit_usd": touch_credit * CONTRACT_MULTIPLIER,
                    "declared_max_loss_usd": max(0.0, max_loss),
                }
            )
            rows.append(base)
            continue
        debit = _settlement_debit(
            side=str(cell["side"]),
            short_strike=float(legs["short"]["strike"]),
            long_strike=float(legs["long"]["strike"]),
            settlement_spx=settlement_spx,
        )
        net_touch = (touch_credit - debit) * CONTRACT_MULTIPLIER - fee
        net_fee_only = (mid_credit - debit) * CONTRACT_MULTIPLIER - fee
        if net_touch < -max_loss - 1e-6:
            raise VerticalCensusError(
                f"{session} {cell['cell']}: loss exceeds entry-defined maximum"
            )
        rows.append(
            {
                **base,
                "status": "traded",
                "traded": True,
                "entry_spot": spot,
                "short_strike": float(legs["short"]["strike"]),
                "long_strike": float(legs["long"]["strike"]),
                "entry_touch_credit_usd": touch_credit * CONTRACT_MULTIPLIER,
                "entry_mid_credit_usd": mid_credit * CONTRACT_MULTIPLIER,
                "settlement_debit_usd": debit * CONTRACT_MULTIPLIER,
                "fees_usd": fee,
                "net_touch_usd": net_touch,
                "net_fee_only_usd": net_fee_only,
                "net_passive_diag_usd": net_fee_only - PASSIVE_ADVERSE_SELECTION_USD,
                "declared_max_loss_usd": max_loss,
            }
        )
    return SessionCells(rows)
