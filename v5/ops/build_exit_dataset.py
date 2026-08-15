"""One row per minute of every open trade, for training an exit model.

The excursion work established the prize: a near-ATM 0DTE contract reaches a
median +28.2% within thirty minutes, holding to a clock earns -$32 a trade, and
selling at the best minute close would earn +$321. Every declared exit rule
captures none of that gap. The open question is what a **trained** exit captures,
and answering it needs the thing no dataset in this project has ever held: the
state of a trade at each minute while it is still open.

Each row is a decision point. The features are what the bot would know standing
in that minute; the label is what waiting is worth.

## The label

``remaining_max_gain`` is the best price still to come in the window, relative to
the price right now. It is positive when holding still has something to offer and
zero or negative when the trade has already seen its best. A model that predicts
it turns the exit into an ordinary supervised problem: **leave when nothing
better is coming.**

The label reads the future, which is what a label is for. Every *feature* is
strictly causal, and the model that uses them is validated chronologically, which
is where leakage would actually bite.

## What is deliberately not here

No entry signal. Trades are opened unconditionally every fifteen minutes on both
sides, so the exit model is measured in isolation rather than flattered by an
entry that already knew something.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.measure_hold_occupancy import FIRST_INDEX, LAST_INDEX, _label
from v5.ops.measure_scalp_exits import (
    ENTRY_EVERY_MINUTES,
    NEAR_ATM_POINTS,
    WINDOW_MINUTES,
)
from v5.ops.resolve_exit_price_convention import CONTRACT_MULTIPLIER, TRADE_CORPUS
from v5.ops.screen_intraday_direction import parity_spot
from v5.research import greeks as gk


def session_rows(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_parquet(
            path, columns=["ts_event", "open", "high", "low", "close", "strike", "right"]
        )
    except Exception:
        return None
    if frame.empty:
        return None
    for col in ("strike", "open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    frame = frame.dropna(subset=["strike", "close"])
    frame = frame[frame["close"] > 0]
    if frame.empty:
        return None
    frame["minute"] = (
        pd.to_datetime(frame["ts_event"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    grids = {
        field: frame.pivot_table(
            index="minute", columns=["strike", "right"], values=field, aggfunc="last"
        ).sort_index()
        for field in ("high", "low", "close")
    }
    close_grid = grids["close"]
    if close_grid.empty:
        return None
    spot = parity_spot(close_grid)
    if spot is None:
        return None
    spot = spot.ffill()
    strikes = close_grid.columns.get_level_values("strike").to_numpy(float)
    is_call = close_grid.columns.get_level_values("right").to_numpy() == "C"
    position = {m: i for i, m in enumerate(close_grid.index)}
    arr = {k: v.reindex(columns=close_grid.columns).to_numpy(float) for k, v in grids.items()}
    spot_arr = spot.reindex(close_grid.index).to_numpy(float)

    rows = []
    entry_index = FIRST_INDEX
    while entry_index + WINDOW_MINUTES <= LAST_INDEX:
        entry_label = _label(entry_index)
        entry_index += ENTRY_EVERY_MINUTES
        if entry_label not in position:
            continue
        i0 = position[entry_label]
        i1 = i0 + WINDOW_MINUTES
        if i1 >= len(close_grid.index):
            continue
        s0 = spot_arr[i0]
        if not np.isfinite(s0):
            continue
        entry_prices = arr["close"][i0]
        moneyness = np.where(is_call, s0 - strikes, strikes - s0)
        eligible = np.isfinite(entry_prices) & (np.abs(moneyness) <= NEAR_ATM_POINTS)
        if not eligible.any():
            continue

        for want_call in (True, False):
            on_side = np.flatnonzero(eligible & (is_call == want_call))
            if not on_side.size:
                continue
            j = int(on_side[np.argmin(np.abs(moneyness[on_side]))])
            entry = float(entry_prices[j])
            hi = arr["high"][i0 + 1 : i1 + 1, j]
            lo = arr["low"][i0 + 1 : i1 + 1, j]
            cl = arr["close"][i0 + 1 : i1 + 1, j]
            und = spot_arr[i0 + 1 : i1 + 1]
            if not np.isfinite(cl).any():
                continue
            trade_id = f"{path.name[:10]}|{entry_label}|{'C' if want_call else 'P'}"
            peak, trough = entry, entry
            for m in range(len(cl)):
                price = cl[m]
                if not np.isfinite(price) or price <= 0:
                    continue
                if np.isfinite(hi[m]):
                    peak = max(peak, hi[m])
                if np.isfinite(lo[m]):
                    trough = min(trough, lo[m])
                future = hi[m + 1 :]
                future = future[np.isfinite(future)]
                # Everything below this line is known standing in minute m.
                prev = cl[m - 1] if m >= 1 and np.isfinite(cl[m - 1]) else entry
                prev3 = cl[m - 3] if m >= 3 and np.isfinite(cl[m - 3]) else entry
                spot_now = und[m] if np.isfinite(und[m]) else s0
                spot_prev3 = (
                    und[m - 3] if m >= 3 and np.isfinite(und[m - 3]) else s0
                )
                rows.append(
                    {
                        "session": path.name[:10],
                        "trade_id": trade_id,
                        "minute_in_trade": m + 1,
                        "minutes_left": WINDOW_MINUTES - (m + 1),
                        "minutes_to_close": LAST_INDEX - (_minute_index(entry_label) + m + 1),
                        "price": price,
                        "entry_premium": entry * CONTRACT_MULTIPLIER,
                        "return_from_entry": price / entry - 1.0,
                        "peak_so_far": peak / entry - 1.0,
                        "drawdown_from_peak": price / peak - 1.0,
                        "trough_so_far": trough / entry - 1.0,
                        "return_1m": price / prev - 1.0,
                        "return_3m": price / prev3 - 1.0,
                        "strike": float(strikes[j]),
                        "is_call": bool(want_call),
                        "spot_now": float(spot_now),
                        "minutes_to_expiry": float(
                            LAST_INDEX - (_minute_index(entry_label) + m + 1)
                        ),
                        "moneyness_now": float(
                            (spot_now - strikes[j]) if want_call
                            else (strikes[j] - spot_now)
                        ),
                        "moneyness_at_entry": float(moneyness[j]),
                        "underlying_return_from_entry": spot_now / s0 - 1.0,
                        "underlying_return_3m": spot_now / spot_prev3 - 1.0,
                        # Label: what waiting is still worth, from here.
                        "remaining_max_gain": (
                            float(future.max()) / price - 1.0 if future.size else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows) if rows else None


def _minute_index(label: str) -> int:
    hours, minutes = (int(part) for part in label.split(":"))
    return hours * 60 + minutes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=TRADE_CORPUS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.corpus.glob("*.parquet"))
    if args.limit:
        files = files[: args.limit]
    parts = []
    for i, path in enumerate(files, 1):
        got = session_rows(path)
        if got is not None:
            parts.append(got)
        if i % 200 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    table = pd.concat(parts, ignore_index=True)
    # Greeks recomputed from price, spot, strike and the clock — the same
    # function a live decision would call, so nothing here can diverge from what
    # the bot would see. See v5/research/greeks.py.
    print("  computing greeks...", flush=True)
    got = gk.greeks_batch(
        table["price"].to_numpy(float),
        table["spot_now"].to_numpy(float),
        table["strike"].to_numpy(float),
        table["minutes_to_expiry"].to_numpy(float),
        table["is_call"].to_numpy(bool),
    )
    for name, values in got.items():
        table[name] = values
    # Theta as a share of the premium being paid: what the clock costs this
    # trade, in the units the decision is made in.
    table["theta_share_of_price"] = table["theta_per_minute"] / table["price"]
    table["gamma_dollars"] = (
        table["gamma"] * table["spot_now"] ** 2 / 100.0
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out)

    print(
        f"\n{len(table):,} decision points across {table['trade_id'].nunique():,} trades "
        f"and {table['session'].nunique()} sessions"
    )
    solved = table["iv"].notna().mean()
    print(
        f"  greeks solved on {100 * solved:.1f}% of rows; "
        f"median IV {table['iv'].median():.3f}, "
        f"median |theta|/price per minute {table['theta_share_of_price'].abs().median():.5f}"
    )
    print(
        f"  label: mean remaining gain {100 * table['remaining_max_gain'].mean():+.2f}%, "
        f"median {100 * table['remaining_max_gain'].median():+.2f}%, "
        f"share with nothing left {100 * (table['remaining_max_gain'] <= 0).mean():.1f}%"
    )
    print(f"\ntable: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
