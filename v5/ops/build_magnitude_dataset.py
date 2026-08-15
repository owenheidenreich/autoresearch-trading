"""Cache one row per serial slot, with causal features and exact payoffs.

The directional screen closed with a negative and a mechanism: long 0DTE premium
pays only when the underlying travels far enough to outrun what the option
charged. That makes **magnitude**, not direction, the open question, and testing
it needs many hypotheses against the same slots.

Scanning the corpus takes about a quarter of an hour, so this module does it
once and writes a table. Everything downstream reads the table, which also means
every hypothesis is scored against **identical** slots rather than against a
population that quietly shifts with each rewrite.

Two disciplines are wired in here rather than left to the caller:

**Every feature is causal.** Each is computed from the parity spot at or before
the entry minute. Nothing reads the exit.

**Nothing is filtered on an outcome.** A slot is dropped only when the chain is
too thin to price at all. The 2026-08-13 defect — skipping slots whose
strike-grid spot had not moved, which discarded 18.7% of slots averaging
-$186.90 — came from exactly the kind of convenience filter this module refuses.

The straddle premium is recorded because it is the market's own forecast of the
move. A magnitude signal is only worth anything if it beats that, and a table
that omits it would let a downstream test mistake volatility clustering for edge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.measure_hold_occupancy import (
    FIRST_INDEX,
    LAST_INDEX,
    _index,
    _label,
    session_pivot,
)
from v5.ops.resolve_exit_price_convention import (
    CONTRACT_MULTIPLIER,
    NEAR_ATM_POINTS,
    TRADE_CORPUS,
)

HOLDS_MINUTES = (15, 60)
# Lookback windows for the causal range features, in minutes.
LOOKBACKS = (15, 30, 60)


def session_rows(path: Path, hold: int) -> pd.DataFrame | None:
    got = session_pivot(path)
    if got is None:
        return None
    pivot, spot = got
    spot_f = spot.ffill()
    filled = pivot.ffill()
    strikes = pivot.columns.get_level_values("strike").to_numpy(float)
    is_call = pivot.columns.get_level_values("right").to_numpy() == "C"

    rows = []
    entry_index = FIRST_INDEX
    while entry_index + hold <= LAST_INDEX:
        entry_label, exit_label = _label(entry_index), _label(entry_index + hold)
        entry_index += hold
        if entry_label not in pivot.index or exit_label not in pivot.index:
            continue
        s0, s1 = spot_f.get(entry_label), spot_f.get(exit_label)
        if s0 is None or s1 is None or not np.isfinite(s0) or not np.isfinite(s1):
            continue

        history = spot_f.loc[:entry_label].dropna()
        if len(history) < max(LOOKBACKS) + 1:
            continue
        feats = {}
        for back in LOOKBACKS:
            window = spot_f.loc[_label(_index(entry_label) - back) : entry_label].dropna()
            if len(window) < back // 3:
                feats = {}
                break
            feats[f"range_{back}m"] = float(window.max() - window.min())
            feats[f"move_{back}m"] = float(s0 - window.iloc[0])
        if not feats:
            continue
        low, high = float(history.min()), float(history.max())
        feats["range_position"] = (s0 - low) / (high - low) if high > low else 0.5
        feats["session_range"] = high - low

        entry_price = pivot.loc[entry_label].to_numpy(float)
        exit_present = pivot.loc[exit_label].to_numpy(float)
        exit_price = np.where(
            np.isfinite(exit_present), exit_present, filled.loc[exit_label].to_numpy(float)
        )
        moneyness = np.where(is_call, s0 - strikes, strikes - s0)
        eligible = (
            np.isfinite(entry_price)
            & np.isfinite(exit_price)
            & (np.abs(moneyness) <= NEAR_ATM_POINTS)
        )
        legs = {}
        for side, want_call in (("call", True), ("put", False)):
            on_side = np.flatnonzero(eligible & (is_call == want_call))
            if not on_side.size:
                break
            i = int(on_side[np.argmin(np.abs(moneyness[on_side]))])
            legs[f"{side}_premium"] = entry_price[i] * CONTRACT_MULTIPLIER
            legs[f"{side}_gross"] = (
                exit_price[i] - entry_price[i]
            ) * CONTRACT_MULTIPLIER
        if len(legs) < 4:
            continue

        rows.append(
            {
                "session": path.name[:10],
                "entry_minute": entry_label,
                "minutes_to_close": LAST_INDEX - _index(entry_label),
                "hold": hold,
                "spot": s0,
                "move": s1 - s0,
                "abs_move": abs(s1 - s0),
                **feats,
                **legs,
            }
        )
    return pd.DataFrame(rows) if rows else None


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
    for hold in HOLDS_MINUTES:
        for i, path in enumerate(files, 1):
            got = session_rows(path, hold)
            if got is not None:
                parts.append(got)
            if i % 200 == 0:
                print(f"  {hold}m: {i}/{len(files)}", flush=True)
    table = pd.concat(parts, ignore_index=True)
    # The straddle premium is the market's forecast of the move; the pair's
    # gross is what the move actually paid.
    table["straddle_premium"] = table["call_premium"] + table["put_premium"]
    table["straddle_gross"] = table["call_gross"] + table["put_gross"]
    table = table.sort_values(["hold", "session", "entry_minute"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out)
    print(f"\n{len(table):,} slots, {table['session'].nunique()} sessions")
    for hold, part in table.groupby("hold"):
        print(
            f"  {hold}m: {len(part):,} slots, "
            f"{len(part) / part['session'].nunique():.1f} per session, "
            f"median |move| {part['abs_move'].median():.2f} pts, "
            f"median straddle ${part['straddle_premium'].median():,.0f}"
        )
    print(f"\ntable: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
