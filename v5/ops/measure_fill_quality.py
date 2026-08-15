"""What does a round trip actually cost, and could a patient order pay less?

Every economic conclusion this project has reached is a ratio: the option is
priced above delivery by about 0.55% of premium over fifteen minutes, and the
round trip costs 2.83%. No signal search changes a ratio. The **only** input
that moves it is what an order actually pays, and that has never been measured —
the $25 aggressive round trip is a single carried constant applied to four years
of sessions.

The owned quote corpus can answer it. It carries bid, ask and size for every
contract every minute across 251 sessions, so the spread is observable rather
than assumed, and so is the question that decides everything: **if an order
rests at the midpoint instead of crossing, how often does the market come to
it?**

## What is measured, and what it approximates

**The spread** is exact: ask minus bid on near-the-money contracts, by hour and
by era. The aggressive round trip is that spread plus the measured $3.08 of fees.

**The passive fill** is a proxy and is labelled as one. Minute bars cannot show
an order resting inside a minute, so a passive buy posted at minute ``t``'s
midpoint is counted as filled if the **ask** at some minute within the patience
window drops to or below that price — the market came down to meet it. This is
conservative in one direction and optimistic in another, and both are stated in
the receipt rather than buried: it ignores queue position, which makes it
optimistic, and it ignores fills that happen and reverse inside a single minute,
which makes it pessimistic.

**Adverse selection** is the reason a fill is not automatically good news. When a
passive buy fills, the market has usually moved down, so the position starts
underwater. This measures that directly: the midpoint at the end of the patience
window, against the price paid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_MULTIPLIER = 100.0
FEES_PER_LEG_USD = 3.08
NEAR_ATM_POINTS = 25.0
# How long an order is willing to rest before giving up and crossing.
PATIENCE_MINUTES = (1, 2, 5)
QUOTE_CORPUS = (
    Path.home() / ".autoresearch-trading/pathd_2025-08-01_2026-07-31/aligned/normalized"
)
RTH_START, RTH_END = "09:35", "16:00"


def session_quotes(path: Path) -> pd.DataFrame | None:
    """Near-the-money quotes per minute for one session."""

    try:
        d = pd.read_parquet(path, columns=["event_time", "strike", "right", "bid", "ask", "mid"])
    except Exception:
        return None
    if d.empty:
        return None
    for col in ("strike", "bid", "ask", "mid"):
        d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
    d = d.dropna(subset=["strike", "bid", "ask", "mid"])
    d = d[(d["bid"] > 0) & (d["ask"] > d["bid"])]
    if d.empty:
        return None
    d["minute"] = (
        pd.to_datetime(d["event_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    d = d[(d["minute"] >= RTH_START) & (d["minute"] <= RTH_END)]
    if d.empty:
        return None

    # Spot by put/call parity on mids, per minute, then keep the near-ATM band.
    piv = d.pivot_table(index="minute", columns=["strike", "right"], values="mid")
    calls = piv.loc[:, piv.columns.get_level_values("right") == "C"]
    puts = piv.loc[:, piv.columns.get_level_values("right") == "P"]
    calls.columns = calls.columns.get_level_values("strike")
    puts.columns = puts.columns.get_level_values("strike")
    shared = calls.columns.intersection(puts.columns)
    if len(shared) < 5:
        return None
    implied = (calls[shared] - puts[shared]).add(pd.Series(shared, index=shared))
    spot = implied.median(axis=1)

    d = d.join(spot.rename("spot"), on="minute").dropna(subset=["spot"])
    d = d[(d["strike"] - d["spot"]).abs() <= NEAR_ATM_POINTS]
    if d.empty:
        return None
    d["session"] = path.name.replace("databento_spxw_0dte_", "")[:10]
    d["hour"] = d["minute"].str[:2]
    d["spread"] = d["ask"] - d["bid"]
    d["spread_usd"] = d["spread"] * CONTRACT_MULTIPLIER
    d["spread_share_of_mid"] = d["spread"] / d["mid"]
    return d[
        ["session", "minute", "hour", "strike", "right", "bid", "ask", "mid",
         "spread", "spread_usd", "spread_share_of_mid"]
    ]


def passive_fills(frame: pd.DataFrame) -> pd.DataFrame:
    """Would a buy resting at this minute's midpoint have been met?

    One row per contract-minute per patience window. ``filled`` is true when the
    ask within the window reaches the posted midpoint. ``adverse`` is the
    midpoint at the end of the window less the price paid: negative means the
    fill happened because the market was leaving.
    """

    rows = []
    for (_, strike, right), part in frame.groupby(["session", "strike", "right"], sort=False):
        part = part.sort_values("minute")
        ask = part["ask"].to_numpy(float)
        bid = part["bid"].to_numpy(float)
        mid = part["mid"].to_numpy(float)
        spread = part["spread_usd"].to_numpy(float)
        session = part["session"].to_numpy()
        hour = part["hour"].to_numpy()
        n = len(part)
        for patience in PATIENCE_MINUTES:
            for i in range(n - patience):
                move = mid[i + patience] - mid[i]
                # A passive BUY fills when the ask comes down to the posted mid;
                # a passive SELL fills when the bid comes up to it. The two are
                # adversely selected in opposite directions, which is why both
                # are measured: the short side is where the edge was found.
                for side, filled in (
                    ("buy", bool(np.any(ask[i + 1 : i + 1 + patience] <= mid[i]))),
                    ("sell", bool(np.any(bid[i + 1 : i + 1 + patience] >= mid[i]))),
                ):
                    rows.append(
                        {
                            "session": session[i],
                            "hour": hour[i],
                            "patience": patience,
                            "side": side,
                            "filled": filled,
                            # Signed so that negative always means the fill went
                            # against the resting order.
                            "adverse": (move if side == "buy" else -move) if filled else np.nan,
                            "spread_usd": spread[i],
                        }
                    )
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=QUOTE_CORPUS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    files = sorted(
        f for f in args.corpus.glob("databento_spxw_0dte_*.parquet")
        if "official_context" not in f.name
    )
    if args.limit:
        files = files[: args.limit]

    quotes, fills = [], []
    for i, path in enumerate(files, 1):
        got = session_quotes(path)
        if got is not None:
            quotes.append(got)
            fills.append(passive_fills(got))
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    q = pd.concat(quotes, ignore_index=True)
    f = pd.concat(fills, ignore_index=True)

    def spread_block(part: pd.DataFrame, label: str) -> dict:
        return {
            "population": label,
            "observations": int(len(part)),
            "mean_spread_usd": round(float(part["spread_usd"].mean()), 2),
            "median_spread_usd": round(float(part["spread_usd"].median()), 2),
            "mean_spread_share_of_mid": round(
                float(part["spread_share_of_mid"].mean()), 6
            ),
            "mean_mid_usd": round(float(part["mid"].mean() * CONTRACT_MULTIPLIER), 2),
            "aggressive_round_trip_usd": round(
                float(part["spread_usd"].mean() + FEES_PER_LEG_USD), 2
            ),
        }

    q["year"] = q["session"].str[:4]
    spreads = [spread_block(q, "all near-ATM quotes")]
    spreads += [spread_block(c, f"entered {h}:00") for h, c in q.groupby("hour") if len(c) > 5_000]
    spreads += [spread_block(c, f"year {y}") for y, c in q.groupby("year") if len(c) > 5_000]

    fill_blocks = []
    for (patience, side), part in f.groupby(["patience", "side"]):
        got = part[part["filled"]]
        fill_blocks.append(
            {
                "patience_minutes": int(patience),
                "side": side,
                "orders": int(len(part)),
                "fill_rate": round(float(part["filled"].mean()), 4),
                "mean_adverse_usd": round(
                    float(got["adverse"].mean() * CONTRACT_MULTIPLIER), 2
                )
                if len(got)
                else None,
                "mean_spread_saved_usd": round(
                    float(got["spread_usd"].mean() / 2.0), 2
                )
                if len(got)
                else None,
                "net_advantage_usd": round(
                    float(
                        got["spread_usd"].mean() / 2.0
                        + got["adverse"].mean() * CONTRACT_MULTIPLIER
                    ),
                    2,
                )
                if len(got)
                else None,
            }
        )

    payload = {
        "schema_version": "v5.fill-quality.v1",
        "question": (
            "The whole conclusion is a ratio of spread to mispricing. What is "
            "the spread, and would a patient order pay less than it?"
        ),
        "fees_per_leg_usd": FEES_PER_LEG_USD,
        "moneyness_band_points": NEAR_ATM_POINTS,
        "sessions": int(q["session"].nunique()),
        "spread_by_population": spreads,
        "passive_fill_proxy": fill_blocks,
        "proxy_limitations": [
            "minute bars cannot show an order resting inside a minute",
            "queue position is ignored, which makes the fill rate optimistic",
            "fills that occur and reverse inside one minute are missed, which "
            "makes it pessimistic",
            "adverse selection is measured at the end of the patience window, "
            "not at the moment of fill",
        ],
        "computes_no_policy": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\n{payload['sessions']} sessions, {len(q):,} near-ATM contract-minutes\n")
    head = f"  {'population':>22} {'obs':>9} {'mid':>8} {'spread':>8} {'as %':>7} {'round trip':>11}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for b in spreads:
        print(
            f"  {b['population']:>22} {b['observations']:>9,} {b['mean_mid_usd']:>8,.0f} "
            f"{b['mean_spread_usd']:>8,.2f} {100 * b['mean_spread_share_of_mid']:>6.2f}% "
            f"{b['aggressive_round_trip_usd']:>11,.2f}"
        )
    print(f"\n  passive order resting at the midpoint:")
    print(f"  {'patience':>9} {'side':>5} {'orders':>10} {'fill rate':>10} "
          f"{'spread saved':>13} {'adverse':>9} {'net':>8}")
    print("  " + "-" * 70)
    for b in fill_blocks:
        print(
            f"  {b['patience_minutes']:>8}m {b['side']:>5} {b['orders']:>10,} "
            f"{100 * b['fill_rate']:>9.1f}% {b['mean_spread_saved_usd']:>13,.2f} "
            f"{b['mean_adverse_usd']:>9,.2f} {b['net_advantage_usd']:>8,.2f}"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
