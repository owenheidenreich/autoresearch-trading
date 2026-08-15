"""Trade paths priced at the touch, so an exit cannot sell a print.

The exit was declared solved on 2026-08-13: optimal stopping by fitted value
iteration drove gross profit from −$7.3 to −$0.2, and that number is the whole
basis for "the exit is solved, so every dollar must come from the entry."

It was measured on last-trade minute bars, and the value iteration is
``V = max(price, C)`` — an explicit **maximum over the price series**. A maximum
over a noisy series is inflated by the noise, and on this corpus the noise is
large: the put/call-parity residual has a standard deviation of **$102.60**
against a **$19.79** round trip. Selling at a high print is, mechanically,
selling a print that landed on the ask. The exit had exactly the freedom the
entry had, and the entry's version of it did not survive quote pricing.

So this rebuilds the same trade paths on the owned quote corpus, carrying three
price series per minute so the question can be decomposed rather than argued:

* **bid** — what a seller actually receives, so an honest exit stops here;
* **mid** — a clean price with no spread and no bounce, which isolates how much
  of the print result was noise rather than cost;
* **ask** — what the entry paid.

Entries are deliberately dumb: every five minutes, the nearest-to-the-money call
and put. The exit is the object of study, so the entry must not be, and a random
entry is what the original study used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.build_decision_dataset import DECISION_EVERY
from v5.ops.build_quoted_dataset import quote_frames
from v5.ops.measure_fill_quality import QUOTE_CORPUS
from v5.ops.measure_hold_occupancy import _index, _label
from v5.ops.resolve_exit_price_convention import CONTRACT_MULTIPLIER
from v5.research import greeks as gk

FIRST_ENTRY = _index("09:35")
LAST_ENTRY = _index("15:00")
MAX_HOLD_MINUTES = 60
# The contract each slot is represented by, matching the original exit study:
# one position, one contract, chosen at the entry minute by moneyness alone.
NEAR_ATM_POINTS = 25.0


def session_paths(path: Path) -> pd.DataFrame | None:
    got = quote_frames(path)
    if got is None:
        return None
    frames, spot = got
    bid, ask, mid = frames["bid"], frames["ask"], frames["mid"]
    session = path.name.replace("databento_spxw_0dte_", "")[:10]

    bid_f, ask_f, mid_f = bid.ffill(), ask.ffill(), mid.ffill()
    spot_f = spot.ffill()
    strikes = mid.columns.get_level_values("strike").to_numpy(float)
    is_call = mid.columns.get_level_values("right").to_numpy() == "C"
    minutes = list(mid.index)
    position = {m: i for i, m in enumerate(minutes)}

    rows = []
    for start in range(FIRST_ENTRY, LAST_ENTRY + 1, DECISION_EVERY):
        entry_minute = _label(start)
        if entry_minute not in position:
            continue
        here = float(spot_f.get(entry_minute, np.nan))
        if not np.isfinite(here):
            continue
        entry_ask = ask.loc[entry_minute].to_numpy(float)
        entry_bid = bid.loc[entry_minute].to_numpy(float)
        moneyness = np.where(is_call, here - strikes, strikes - here)
        live = (
            np.isfinite(entry_ask) & np.isfinite(entry_bid)
            & (entry_bid > 0.0) & (entry_ask > entry_bid)
            & (np.abs(moneyness) <= NEAR_ATM_POINTS)
        )
        for want_call in (True, False):
            side = np.flatnonzero(live & (is_call == want_call))
            if not side.size:
                continue
            column = int(side[np.argmin(np.abs(moneyness[side]))])
            paid = float(entry_ask[column])
            entry_mid = float(mid.loc[entry_minute].to_numpy(float)[column])
            strike = float(strikes[column])

            first = position[entry_minute]
            window = minutes[first : first + MAX_HOLD_MINUTES + 1]
            if len(window) < 2:
                continue
            path_bid = bid_f[mid.columns[column]].reindex(window).to_numpy(float)
            path_mid = mid_f[mid.columns[column]].reindex(window).to_numpy(float)
            path_spot = spot_f.reindex(window).to_numpy(float)
            usable = np.isfinite(path_bid) & np.isfinite(path_mid) & np.isfinite(path_spot)
            if usable.sum() < 5:
                continue

            # The state the policy sees is built from the mid, which is what a
            # screen shows. What it receives on exit is the bid.
            running_max = np.maximum.accumulate(np.where(usable, path_mid, -np.inf))
            running_min = np.minimum.accumulate(np.where(usable, path_mid, np.inf))
            back_1 = np.concatenate([[np.nan], path_mid[:-1]])
            back_3 = np.concatenate([[np.nan] * 3, path_mid[:-3]]) if len(path_mid) > 3 else np.full(len(path_mid), np.nan)
            spot_3 = np.concatenate([[np.nan] * 3, path_spot[:-3]]) if len(path_spot) > 3 else np.full(len(path_spot), np.nan)
            left = np.array([float(_index("16:00") - _index(m)) for m in window])

            rows.append(
                pd.DataFrame(
                    {
                        "session": session,
                        "trade_id": f"{session}|{entry_minute}|{'C' if want_call else 'P'}",
                        "entry_minute": entry_minute,
                        "minute": window,
                        "minute_in_trade": np.arange(len(window), dtype=float),
                        "minutes_left": float(len(window) - 1) - np.arange(len(window)),
                        "minutes_to_expiry": left,
                        "entry_ask_usd": paid * CONTRACT_MULTIPLIER,
                        "entry_mid_usd": entry_mid * CONTRACT_MULTIPLIER,
                        "spread_usd": (paid - float(entry_bid[column])) * CONTRACT_MULTIPLIER,
                        "bid": path_bid,
                        "mid": path_mid,
                        "spot": path_spot,
                        "strike": strike,
                        "is_call": want_call,
                        "return_from_entry": path_mid / entry_mid - 1.0,
                        "peak_so_far": running_max / entry_mid - 1.0,
                        "trough_so_far": running_min / entry_mid - 1.0,
                        "drawdown_from_peak": path_mid / np.where(
                            np.isfinite(running_max) & (running_max > 0), running_max, np.nan
                        ) - 1.0,
                        "return_1m": path_mid / back_1 - 1.0,
                        "return_3m": path_mid / back_3 - 1.0,
                        "underlying_return_from_entry": path_spot / here - 1.0,
                        "underlying_return_3m": path_spot / spot_3 - 1.0,
                        "moneyness_now": np.where(
                            want_call, path_spot - strike, strike - path_spot
                        ),
                    }
                )[usable]
            )

    if not rows:
        return None
    table = pd.concat(rows, ignore_index=True)
    solved = gk.greeks_batch(
        table["mid"].to_numpy(float),
        table["spot"].to_numpy(float),
        table["strike"].to_numpy(float),
        table["minutes_to_expiry"].to_numpy(float),
        table["is_call"].to_numpy(bool),
    )
    table["iv"] = solved["iv"]
    table["delta"] = solved["delta"]
    table["gamma_dollars"] = solved["gamma"] * table["spot"] ** 2 / 100.0
    table["theta_share_of_price"] = solved["theta_per_minute"] / table["mid"]
    table["vega"] = solved["vega"]
    return table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quotes", type=Path, default=QUOTE_CORPUS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    files = [
        f for f in sorted(args.quotes.glob("*.parquet"))
        if "official_context" not in f.name
    ]
    if args.limit:
        files = files[: args.limit]

    parts = []
    for i, path in enumerate(files, 1):
        got = session_paths(path)
        if got is not None:
            parts.append(got)
        if i % 50 == 0:
            print(f"  {i}/{len(files)} sessions, "
                  f"{sum(len(x) for x in parts):,} rows", flush=True)

    table = pd.concat(parts, ignore_index=True)
    table = table.sort_values(["session", "trade_id", "minute_in_trade"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out)

    trades = table["trade_id"].nunique()
    print(
        f"\n{len(table):,} minute-rows, {trades:,} trades, "
        f"{table['session'].nunique()} sessions"
    )
    print(f"  mean entry ask ${table.groupby('trade_id')['entry_ask_usd'].first().mean():,.0f}")
    print(f"  mean spread   ${table.groupby('trade_id')['spread_usd'].first().mean():,.2f}")
    print(f"  mean hold available {len(table) / trades:.1f} minutes")
    print(f"\ntable: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
