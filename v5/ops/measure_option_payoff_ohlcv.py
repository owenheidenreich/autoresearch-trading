"""Conditional option payoff by moneyness, on the OHLCV 0DTE corpus.

This is the same measurement as
:mod:`v5.ops.measure_option_payoff_by_moneyness` against a materially different
source, and the differences are the point rather than an inconvenience:

**Trades, not quotes.** ``ohlcv-1m`` records a bar only when a contract actually
traded in that minute, so a contract present at both the entry and the exit
minute is by construction one a real order could have worked. That is closer to
what the bot could trade than a quote-based ladder, and it is also a *liquidity
selection*: illiquid strikes drop out rather than appearing with a wide spread.
Both statements are true and the receipt records the selection rather than
hiding it.

**Prices are last trades, not mids.** A trade print bounces between bid and ask,
which adds noise a mid does not have. Measured dispersion is therefore mildly
inflated relative to the quote-based study.

**The spread is carried, not measured.** OHLCV has no bid or ask, so round-trip
cost uses the frozen per-band constants measured on the owned quote corpus. That
is an assumption about 2022-2025 spreads taken from 2025-2026 evidence, and it is
recorded in the receipt as such rather than presented as measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5.research import knobs

ENTRY_MINUTE = "09:35"
EXIT_MINUTE = "10:35"
CONTRACT_MULTIPLIER = 100.0

# Round-trip cost per band: the $3.08 measured fee plus the spread crossing
# measured on the owned quote corpus 2026-08-13. Carried forward, not measured
# here -- see the module docstring.
BAND_ROUND_TRIP_USD = {
    "deep OTM": 9.0,
    "OTM": 12.0,
    "near ATM": 25.0,
    "ITM": 84.0,
    "deep ITM": 369.0,
    "very deep ITM": 822.0,
}
BUCKETS = (
    (-1e9, -100.0, "deep OTM"),
    (-100.0, -25.0, "OTM"),
    (-25.0, 25.0, "near ATM"),
    (25.0, 100.0, "ITM"),
    (100.0, 300.0, "deep ITM"),
    (300.0, 1e9, "very deep ITM"),
)


def _spot(frame: pd.DataFrame) -> float | None:
    """Put/call parity proxy from traded prices at one minute."""

    piv = frame.pivot_table(
        index="strike", columns="right", values="close", aggfunc="last"
    ).dropna()
    if piv.empty or not {"C", "P"}.issubset(piv.columns):
        return None
    return float((piv["C"] - piv["P"]).abs().idxmin())


def session_rows(path: Path) -> pd.DataFrame | None:
    frame = pd.read_parquet(path)
    if frame.empty:
        return None
    for col in ("strike", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    frame = frame.dropna(subset=["strike", "close"])
    frame = frame[frame["close"] > 0]
    if frame.empty:
        return None
    frame["hm"] = (
        pd.to_datetime(frame["ts_event"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    entry, exit_ = frame[frame["hm"] == ENTRY_MINUTE], frame[frame["hm"] == EXIT_MINUTE]
    if entry.empty or exit_.empty:
        return None
    s0, s1 = _spot(entry), _spot(exit_)
    if s0 is None or s1 is None or s0 == s1:
        return None

    merged = entry.merge(exit_, on=["strike", "right"], suffixes=("_0", "_1"))
    if merged.empty:
        return None
    is_call = merged["right"].eq("C").to_numpy()
    up = bool(s1 > s0)
    merged["moneyness"] = np.where(
        is_call, s0 - merged["strike"], merged["strike"] - s0
    )
    merged["premium"] = merged["close_0"] * CONTRACT_MULTIPLIER
    merged["gross"] = (merged["close_1"] - merged["close_0"]) * CONTRACT_MULTIPLIER
    merged["correct"] = is_call == up
    merged["session"] = path.name[:10]
    return merged[["session", "moneyness", "premium", "gross", "correct"]]


def summarise(frame: pd.DataFrame, label: str) -> dict:
    cost = BAND_ROUND_TRIP_USD[label]
    net = frame["gross"].to_numpy(float) - cost
    ok = frame["correct"].to_numpy(bool)
    if ok.sum() < 50 or (~ok).sum() < 50:
        return {}
    win, loss = float(net[ok].mean()), float(-net[~ok].mean())
    denom = win + loss
    return {
        "bucket": label,
        "observations": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "mean_premium_usd": round(float(frame["premium"].mean()), 2),
        "round_trip_cost_usd": cost,
        "round_trip_cost_provenance": "CARRIED from the owned quote corpus, not measured here",
        "mean_net_when_correct_usd": round(win, 2),
        "mean_net_when_wrong_usd": round(-loss, 2),
        "breakeven_accuracy": round(loss / denom, 6) if denom > 0 and win > 0 else None,
        "affordable_at_10k": bool(frame["premium"].mean() < 10_000.0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/spxw_0dte_2022-06-01_2026-07-31/raw"),
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.corpus.glob("*.parquet"))
    parts, skipped = [], 0
    for i, path in enumerate(files, 1):
        got = session_rows(path)
        if got is None:
            skipped += 1
        else:
            parts.append(got)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    table = pd.concat(parts, ignore_index=True)

    by_bucket = [
        row
        for lo, hi, label in BUCKETS
        if (
            row := summarise(
                table[(table["moneyness"] > lo) & (table["moneyness"] <= hi)], label
            )
        )
    ]
    payload = {
        "schema_version": "v5.option-payoff-ohlcv.v1",
        "source_schema": "ohlcv-1m",
        "entry_minute": ENTRY_MINUTE,
        "exit_minute": EXIT_MINUTE,
        "sessions_available": len(files),
        "sessions_used": int(table["session"].nunique()),
        "sessions_skipped": skipped,
        "observations": int(len(table)),
        "known_limitations": [
            "bars exist only where a contract traded, so this is a liquidity-selected population",
            "prices are last trades rather than mids, which adds bounce noise",
            "the spread half of round-trip cost is carried from the owned quote corpus",
        ],
        "reference_quote_study": (
            "v4/audit/autoresearch/option_payoff_by_moneyness_2026_08_13/receipt.json"
        ),
        "by_moneyness": by_bucket,
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\n{payload['sessions_used']} sessions, {payload['observations']:,} obs\n")
    head = f"{'bucket':<14} {'n':>8} {'premium':>9} {'correct':>9} {'wrong':>9} {'break-even':>11}"
    print(head)
    print("-" * len(head))
    for row in by_bucket:
        be = row["breakeven_accuracy"]
        print(
            f"{row['bucket']:<14} {row['observations']:>8,} "
            f"{row['mean_premium_usd']:>9,.0f} {row['mean_net_when_correct_usd']:>9,.0f} "
            f"{row['mean_net_when_wrong_usd']:>9,.0f} "
            f"{(f'{100 * be:.2f}%' if be else 'impossible'):>11}"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
