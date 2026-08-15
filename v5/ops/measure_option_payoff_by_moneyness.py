"""Conditional 60-minute option payoff by moneyness, on the full owned corpus.

Replicates the gate-chain audit's frozen conditional-outcome study as closely as
the owned data allows, and extends it across the strike ladder.

The frozen figures (+$308.17 / -$425.33, break-even 57.99%) were measured on the
**1,031 Phase-1 four-box trajectories** -- one retired entry policy's selected
contracts -- in a file that lives on an unmounted volume. That is a different
population from "every contract on the ladder", so a disagreement between the
two is expected and is not by itself evidence that either is wrong. What this
module settles is the *shape*: how the bar moves with moneyness, measured the
same way, with the same friction charged.

Definitions taken verbatim from the audit:
  correct     = call with positive underlying movement, or put with negative
  break-even  = -mean(wrong) / (mean(correct) - mean(wrong))
  friction    = the binding measured $3.08 round trip

Computes no policy, no signal and no threshold. It characterises the instrument.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5.research import knobs

CONTRACT_MULTIPLIER = 100.0  # SPX index options. The corpus field is an int32 sentinel.
ENTRY_MINUTE = "09:35"
EXIT_MINUTE = "10:35"
BUCKETS = (
    (-1e9, -100.0, "deep OTM"),
    (-100.0, -25.0, "OTM"),
    (-25.0, 25.0, "near ATM"),
    (25.0, 100.0, "ITM"),
    (100.0, 300.0, "deep ITM"),
    (300.0, 1e9, "very deep ITM"),
)


def _spot(frame: pd.DataFrame) -> float | None:
    """Put/call parity proxy: the strike where call and put mids are closest."""

    piv = frame.pivot_table(
        index="strike", columns="right", values="mid", aggfunc="last"
    ).dropna()
    if piv.empty or not {"C", "P"}.issubset(piv.columns):
        return None
    return float((piv["C"] - piv["P"]).abs().idxmin())


def session_rows(path: Path) -> pd.DataFrame | None:
    d = pd.read_parquet(
        path, columns=["event_time", "strike", "right", "mid", "bid", "ask"]
    )
    for col in ("strike", "mid", "bid", "ask"):
        d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
    d = d.dropna(subset=["strike", "mid", "bid", "ask"])
    d = d[(d["mid"] > 0) & (d["ask"] > d["bid"]) & (d["bid"] > 0)]
    if d.empty:
        return None
    minute = (
        pd.to_datetime(d["event_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    entry, exit_ = d[minute == ENTRY_MINUTE], d[minute == EXIT_MINUTE]
    if entry.empty or exit_.empty:
        return None
    s0, s1 = _spot(entry), _spot(exit_)
    if s0 is None or s1 is None or s0 == s1:
        return None

    up = bool(s1 > s0)
    m = entry.merge(exit_, on=["strike", "right"], suffixes=("_0", "_1"))
    is_call = m["right"].eq("C").to_numpy()
    m["moneyness"] = np.where(is_call, s0 - m["strike"], m["strike"] - s0)
    m["premium"] = m["mid_0"] * CONTRACT_MULTIPLIER
    m["gross"] = (m["mid_1"] - m["mid_0"]) * CONTRACT_MULTIPLIER
    m["correct"] = is_call == up
    m["session"] = path.name.split("_")[-1].replace(".parquet", "")
    m["underlying_move"] = s1 - s0
    # Measured aggressive crossing: half the quoted spread on entry and again on
    # exit. A flat dollar friction is wrong across the ladder, because a
    # near-ATM contract's spread is far wider in dollars than a cheap one's even
    # when it is far narrower as a share of premium.
    m["spread_cost"] = (
        0.5 * (m["ask_0"] - m["bid_0"]) + 0.5 * (m["ask_1"] - m["bid_1"])
    ) * CONTRACT_MULTIPLIER
    return m[
        [
            "session", "moneyness", "premium", "gross", "correct",
            "underlying_move", "spread_cost",
        ]
    ]


def summarise(frame: pd.DataFrame, fee: float, label: str) -> dict:
    """Break-even at fees only, and at fees plus the measured spread crossing."""

    gross = frame["gross"].to_numpy(float)
    spread = frame["spread_cost"].to_numpy(float)
    ok = frame["correct"].to_numpy(bool)
    if ok.sum() < 30 or (~ok).sum() < 30:
        return {}

    def breakeven(net: np.ndarray) -> tuple[float, float, float | None]:
        w, l = float(net[ok].mean()), float(-net[~ok].mean())
        denom = w + l
        return w, -l, (l / denom if denom > 0 and w > 0 else None)

    fee_only = breakeven(gross - fee)
    realistic = breakeven(gross - fee - spread)
    premium = float(frame["premium"].mean())
    mean_spread = float(spread.mean())
    return {
        "bucket": label,
        "observations": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "mean_premium_usd": round(premium, 2),
        "mean_spread_cost_usd": round(mean_spread, 2),
        "spread_share_of_premium": round(mean_spread / premium, 4) if premium else None,
        "fee_only": {
            "mean_net_when_correct_usd": round(fee_only[0], 2),
            "mean_net_when_wrong_usd": round(fee_only[1], 2),
            "breakeven_accuracy": round(fee_only[2], 6) if fee_only[2] else None,
        },
        "with_measured_spread": {
            "mean_net_when_correct_usd": round(realistic[0], 2),
            "mean_net_when_wrong_usd": round(realistic[1], 2),
            "breakeven_accuracy": round(realistic[2], 6) if realistic[2] else None,
            "round_trip_cost_usd": round(fee + mean_spread, 2),
        },
        "affordable_at_10k": bool(premium < 10_000.0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path.home()
        / ".autoresearch-trading/pathd_2025-08-01_2026-07-31/aligned/normalized",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    friction = float(knobs.frozen_value("option_round_trip_fee_dollars"))
    files = sorted(
        f for f in args.corpus.glob("databento_spxw_0dte_*.parquet")
        if "official_context" not in f.name
    )
    if args.limit:
        files = files[: args.limit]

    parts, skipped = [], 0
    for i, path in enumerate(files, 1):
        rows = session_rows(path)
        if rows is None:
            skipped += 1
        else:
            parts.append(rows)
        if i % 25 == 0:
            print(f"  {i}/{len(files)} sessions", flush=True)
    table = pd.concat(parts, ignore_index=True)

    by_bucket = []
    for lo, hi, label in BUCKETS:
        sub = table[(table["moneyness"] > lo) & (table["moneyness"] <= hi)]
        got = summarise(sub, friction, label)
        if got:
            by_bucket.append(got)

    payload = {
        "schema_version": "v5.option-payoff-by-moneyness.v1",
        "entry_minute": ENTRY_MINUTE,
        "exit_minute": EXIT_MINUTE,
        "contract_multiplier": CONTRACT_MULTIPLIER,
        "friction_usd_round_trip": friction,
        "sessions_available": len(files),
        "sessions_used": int(table["session"].nunique()),
        "sessions_skipped": skipped,
        "observations": int(len(table)),
        "share_of_sessions_up": round(
            float(
                table.groupby("session")["underlying_move"].first().gt(0).mean()
            ),
            4,
        ),
        "definitions": {
            "correct": "call with positive underlying movement, or put with negative",
            "breakeven": "-mean(wrong) / (mean(correct) - mean(wrong))",
            "source": "v5/research/findings/GATE_CHAIN_AUDIT_2026_08_05.md",
        },
        "frozen_reference": {
            "population": "1,031 Phase-1 four-box trajectories, 852 reaching 60m",
            "mean_net_when_correct_usd": 308.17,
            "mean_net_when_wrong_usd": -425.33,
            "breakeven_accuracy": 0.5799,
            "note": (
                "A different population from this study: one retired entry policy's "
                "selected contracts, not the whole ladder. Its source file is on an "
                "unmounted volume and could not be re-read."
            ),
        },
        "by_moneyness": by_bucket,
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"\n{payload['sessions_used']} sessions, {payload['observations']:,} obs, "
        f"${friction} charged, {100 * payload['share_of_sessions_up']:.1f}% up\n"
    )
    head = (
        f"{'bucket':<14} {'n':>7} {'premium':>9} {'spread':>8} {'%prem':>7} "
        f"{'BE fees':>9} {'BE +spread':>11} {'@$10k':>6}"
    )
    print(head)
    print("-" * len(head))
    for row in by_bucket:
        fee_be = row["fee_only"]["breakeven_accuracy"]
        real_be = row["with_measured_spread"]["breakeven_accuracy"]
        print(
            f"{row['bucket']:<14} {row['observations']:>7,} "
            f"{row['mean_premium_usd']:>9,.0f} {row['mean_spread_cost_usd']:>8,.0f} "
            f"{100 * (row['spread_share_of_premium'] or 0):>6.1f}% "
            f"{(f'{100 * fee_be:.2f}%' if fee_be else 'n/a'):>9} "
            f"{(f'{100 * real_be:.2f}%' if real_be else 'IMPOSSIBLE'):>11} "
            f"{'yes' if row['affordable_at_10k'] else 'NO':>6}"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
