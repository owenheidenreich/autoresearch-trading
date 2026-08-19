"""Measure the account survival profile of candidate per-trade risk caps.

Written for the 2026-08-16 amendment request on job 46's position-sizing law.
It answers, from measured outcomes rather than assumption:

- what share of the model's action space each cap admits;
- what the round trip costs as a share of premium at each ticket size;
- how often a long 0DTE actually loses its whole premium, against the current
  law's assumption that every ticket does; and
- what each cap does to a $10,000 serial account's survival, using the existing
  `check_occupancy_risk.simulate` machinery, which is parameterised on the
  breaker and premium ceiling precisely so a proposed amendment can be measured
  rather than argued.

Read-only: no fit, no vendor contact, no spend, no promotion. It informs a
drafted amendment that the owner must sign before anything is fitted under it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json
from v5.ops.check_occupancy_risk import SESSIONS_PER_YEAR, simulate
from v5.ops.measure_hold_occupancy import strata_means

LADDER = Path("/Volumes/AR_TRADING_DATA/derived/causal_day_trader_v2/ladder.parquet")
PATHS = Path("/Volumes/AR_TRADING_DATA/derived/quoted_exit_paths.parquet")
FEES_USD = 3.08
OTM_BAND = (-25.0, 0.0)
CAPS_USD = (500.0, 1000.0, 1500.0, 2000.0, 2500.0)
# Declared before measuring: a breaker that admits at least two maximum-size
# losses, plus the signed 5% for comparison.
BREAKERS = (0.05, 0.10, 0.15, 0.20)
STOP_FRACTIONS = (None, 0.50)
TRADES_PER_SESSION = 2


def eligibility(ladder: pd.DataFrame) -> dict[str, Any]:
    low, high = OTM_BAND
    otm = ladder[
        (ladder.moneyness_itm_points >= low)
        & (ladder.moneyness_itm_points < high)
        & (ladder.bid > 0)
        & (ladder.ask > 0)
    ]
    ticket = otm.ask.to_numpy() * 100.0
    whole = ladder[(ladder.bid > 0) & (ladder.ask > 0)]
    whole_ticket = whole.ask.to_numpy() * 100.0
    return {
        "otm_band_candidates": int(len(otm)),
        "whole_chain_candidates": int(len(whole)),
        "share_eligible_otm_band": {
            f"{int(c)}": round(float(((ticket + FEES_USD) <= c).mean()), 4) for c in CAPS_USD
        },
        "share_eligible_whole_chain": {
            f"{int(c)}": round(float(((whole_ticket + FEES_USD) <= c).mean()), 4)
            for c in CAPS_USD
        },
        "share_in_1000_2000_band_otm": round(
            float(((ticket >= 1000) & (ticket <= 2000)).mean()), 4
        ),
        "round_trip_share_of_premium_by_ticket": _friction(otm),
    }


def _friction(otm: pd.DataFrame) -> dict[str, Any]:
    ticket = otm.ask.to_numpy() * 100.0
    rt = (otm.spread_usd.to_numpy() + FEES_USD) / ticket
    out: dict[str, Any] = {}
    for low, high, label in (
        (0, 200, "under_200"),
        (200, 500, "200_to_500"),
        (500, 1000, "500_to_1000"),
        (1000, 2000, "1000_to_2000"),
        (2000, np.inf, "over_2000"),
    ):
        sel = (ticket >= low) & (ticket < high)
        if sel.any():
            out[label] = {
                "candidates": int(sel.sum()),
                "median_round_trip_share": round(float(np.median(rt[sel])), 4),
            }
    return out


def trade_outcomes(stop_fraction: float | None) -> pd.DataFrame:
    """Per-trade dollar outcomes under a declared stop, or a plain 60m hold."""

    frame = pd.read_parquet(
        PATHS,
        columns=["trade_id", "minute_in_trade", "return_from_entry", "bid", "entry_ask_usd"],
    ).sort_values(["trade_id", "minute_in_trade"])
    live = frame[frame.minute_in_trade > 0]
    rows: list[dict[str, float]] = []
    for trade_id, group in live.groupby("trade_id", sort=False):
        ask = float(group.entry_ask_usd.iloc[0])
        returns = group.return_from_entry.to_numpy()
        bids = group.bid.to_numpy() * 100.0
        exit_value = float(bids[-1])
        if stop_fraction is not None:
            hit = np.flatnonzero(returns <= -stop_fraction)
            if hit.size:
                exit_value = float(bids[hit[0]])
        rows.append(
            {
                "trade_id": trade_id,
                "ticket_usd": ask,
                "net_usd": exit_value - ask - FEES_USD,
            }
        )
    return pd.DataFrame(rows)


def loss_profile() -> dict[str, Any]:
    frame = pd.read_parquet(
        PATHS, columns=["trade_id", "minute_in_trade", "return_from_entry"]
    )
    live = frame[frame.minute_in_trade > 0]
    mae = live.groupby("trade_id", sort=False).return_from_entry.min().to_numpy()
    return {
        "trades": int(len(mae)),
        "share_drawdown_at_or_beyond": {
            f"{int(t * 100)}pct": round(float((mae <= -t).mean()), 4)
            for t in (0.99, 0.95, 0.90, 0.75, 0.50)
        },
    }


def survival(outcomes: pd.DataFrame, *, cap: float, breaker: float, rng) -> dict[str, Any]:
    eligible = outcomes[(outcomes.ticket_usd + FEES_USD) <= cap]
    if len(eligible) < 100:
        return {"cap_usd": cap, "daily_breaker": breaker, "status": "TOO_FEW_TRADES"}
    wins = eligible.net_usd[eligible.net_usd > 0].to_numpy()
    losses = eligible.net_usd[eligible.net_usd <= 0].to_numpy()
    if wins.size < 50 or losses.size < 50:
        return {"cap_usd": cap, "daily_breaker": breaker, "status": "TOO_FEW_TRADES"}
    accuracy = float((eligible.net_usd > 0).mean())
    mean_premium = float(eligible.ticket_usd.mean())
    result = simulate(
        trades_per_session=TRADES_PER_SESSION,
        win_quantiles_usd=np.array(strata_means(wins)),
        loss_quantiles_usd=np.array(strata_means(losses)),
        mean_premium_usd=mean_premium,
        accuracy=accuracy,
        account_usd=10_000.0,
        sessions=SESSIONS_PER_YEAR,
        paths=20_000,
        rng=rng,
        daily_breaker=breaker,
        premium_ceiling_usd=cap,
    )
    result.update(
        {
            "cap_usd": cap,
            "measured_accuracy": round(accuracy, 4),
            "mean_premium_usd": round(mean_premium, 2),
            "eligible_trades": int(len(eligible)),
            "worst_measured_loss_usd": round(float(eligible.net_usd.min()), 2),
            "mean_loss_usd": round(float(losses.mean()), 2),
            "mean_loss_share_of_premium": round(
                float((-losses).mean() / mean_premium), 4
            ),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")

    ladder = pd.read_parquet(
        LADDER, columns=["ask", "bid", "spread_usd", "moneyness_itm_points"]
    )
    receipt: dict[str, Any] = {
        "schema_version": "v5.ticket-risk-law-measurement.v1",
        "purpose": "measure candidate per-trade risk caps for the drafted amendment",
        "fees_usd": FEES_USD,
        "otm_band": list(OTM_BAND),
        "trades_per_session": TRADES_PER_SESSION,
        "eligibility": eligibility(ladder),
        "loss_profile": loss_profile(),
        "survival": [],
        "integrity": {
            "model_fitted": False,
            "money_spent": False,
            "vendor_contacted": False,
            "self_adopted": False,
        },
    }
    for stop in STOP_FRACTIONS:
        outcomes = trade_outcomes(stop)
        label = "hold_60m_no_stop" if stop is None else f"stop_at_minus_{int(stop * 100)}pct"
        for cap in CAPS_USD:
            for breaker in BREAKERS:
                rng = np.random.default_rng(
                    int.from_bytes(
                        hashlib.sha256(f"{label}|{cap}|{breaker}".encode()).digest()[:4],
                        "little",
                    )
                )
                row = survival(outcomes, cap=cap, breaker=breaker, rng=rng)
                row["exit_law"] = label
                receipt["survival"].append(row)
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True))

    el = receipt["eligibility"]
    print("OTM-band eligibility by cap:", el["share_eligible_otm_band"])
    print("loss profile:", receipt["loss_profile"]["share_drawdown_at_or_beyond"])
    print(f"\n{'exit law':22s} {'cap':>6s} {'brk':>5s} {'acc':>6s} {'breaker/yr':>10s} {'ruin':>6s} {'median x':>9s}")
    for row in receipt["survival"]:
        if row.get("status") == "TOO_FEW_TRADES":
            continue
        print(
            f"{row['exit_law']:22s} {row['cap_usd']:6.0f} {row['daily_breaker']:5.2f}"
            f" {row['measured_accuracy']:6.3f}"
            f" {row['share_of_sessions_hitting_the_breaker']:10.4f}"
            f" {row['share_of_years_breaching_the_survival_floor']:6.3f}"
            f" {row['median_year_end_equity_multiple']:9.3f}"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
