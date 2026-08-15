"""Is the exit still solved when it has to sell at the bid?

The 2026-08-13 result — optimal stopping drives gross from −$7.3 to −$0.2, so
the exit is solved and every dollar must come from the entry — was measured on
last-trade minute bars. The entry model measured the same way turned out to be
harvesting print noise, and the exit's own formulation is
``V = max(price, C)``: an explicit maximum over the price series. A maximum over
a noisy series is inflated by the noise, and selling at a high print *is*
selling a print that landed on the ask.

This re-runs the identical formulation on quote-priced paths. It imports
`fit_continuation` from [`train_optimal_exit`](train_optimal_exit.py) rather than
reimplementing it, so the value iteration, feature list, sweep count and model
hyperparameters are provably the same and **only the price source changes**.

Three price sources, run separately, which is what turns one number into a
decomposition:

* **bid** — an honest exit: bought the offer, sold the bid.
* **mid** — no spread and almost no bounce. This is the important one: it says
  what the policy is worth in a world where trading is free but prices are real.
  If the print result was noise rather than cost, this is where it vanishes.
* **entry at ask, settle at bid** is what the net column charges; only the
  measured $3.08 of fees is added, because crossing is already in the prices.

Four policies on each, so the fitted rule is never read against zero alone:
hold to the clock, optimal stopping, a random exit minute, and perfect foresight
as the ceiling — plus the same machinery on shuffled labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.build_quoted_exit_dataset import MAX_HOLD_MINUTES
from v5.ops.train_optimal_exit import FEATURES, fit_continuation
from v5.ops.resolve_exit_price_convention import CONTRACT_MULTIPLIER

FEES_PER_ROUND_TRIP_USD = 3.08
CLOCK_MINUTES = 30
# The ladder the fitted rule must beat to have earned its complexity.
FIXED_CLOCKS = (5, 8, 15, 30, 60)
# Trailing stops: exit when the price falls this far from its own running peak.
#
# Every exit family tested so far can only leave EARLY -- a clock, and a fitted
# stopping rule that holds 7.9 minutes and has no trimming ability. None of them
# can ride a sustained move, so a trend-catching entry is unmeasurable through
# them by construction. A trail is the declared family that CAN hold: it stays in
# while the move runs and leaves only after it turns.
#
# Not the same as the fixed -30% stop already closed on the trade corpus. That
# measured from the ENTRY and fired inside normal noise; this measures from the
# running peak, so it gives back a fraction of a gain rather than a fraction of
# the ticket.
TRAIL_LEVELS = (0.15, 0.25, 0.35, 0.50)
WARMUP_SESSIONS = 100
FOLD_SESSIONS = 30
BOOTSTRAP_DRAWS = 2_000
SEED = 20260814


def bootstrap_ci(values: np.ndarray, sessions: np.ndarray) -> tuple[float, float]:
    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        pick = rng.choice(unique, size=len(unique), replace=True)
        draws[b] = values[np.concatenate([index[s] for s in pick])].mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def walk_forward(table: pd.DataFrame, *, shuffle: bool, seed: int) -> pd.DataFrame:
    sessions = sorted(table["session"].unique())
    out = []
    start = WARMUP_SESSIONS
    while start < len(sessions):
        train = table[table["session"].isin(set(sessions[:start]))]
        test = table[table["session"].isin(set(sessions[start : start + FOLD_SESSIONS]))]
        start += FOLD_SESSIONS
        if len(train) < 5_000 or test.empty:
            continue
        model = fit_continuation(train, seed=seed, shuffle=shuffle)
        if model is None:
            continue
        got = test.copy()
        got["continuation"] = model.predict(test[list(FEATURES)].to_numpy(float))
        out.append(got)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score(pred: pd.DataFrame, *, mode: str, entry_column: str, seed: int,
          clock: int = CLOCK_MINUTES, trail: float = 0.25) -> dict:
    """Run one exit policy over every trade and report what it kept."""

    rng = np.random.default_rng(seed)
    gross, sessions, held, ids = [], [], [], []
    for _, trade in pred.groupby("trade_id", sort=False):
        trade = trade.sort_values("minute_in_trade")
        price = trade["price"].to_numpy(float)
        entry = float(trade[entry_column].iloc[0]) / CONTRACT_MULTIPLIER
        if price.size < 2:
            continue
        # A trade cannot be opened and closed inside the same minute bar. The
        # original print study's paths began one minute after entry, so every
        # policy here decides from minute one onwards; without this the stopping
        # rule can "exit" at the fill and the holding times are not comparable.
        if mode == "hold":
            row = min(clock, price.size - 1)
        elif mode == "optimal_stop":
            fires = np.flatnonzero(price >= trade["continuation"].to_numpy(float))
            fires = fires[fires >= 1]
            row = int(fires[0]) if fires.size else price.size - 1
        elif mode == "trail":
            peak = np.maximum.accumulate(price)
            hit = np.flatnonzero(price <= peak * (1.0 - trail))
            hit = hit[hit >= 1]
            row = int(hit[0]) if hit.size else price.size - 1
        elif mode == "random":
            row = int(rng.integers(1, price.size))
        elif mode == "oracle":
            row = int(np.argmax(price[1:])) + 1
        else:
            raise ValueError(mode)
        row = max(row, 1)
        gross.append((price[row] - entry) * CONTRACT_MULTIPLIER)
        sessions.append(trade["session"].iloc[0])
        ids.append(trade["trade_id"].iloc[0])
        held.append(row)

    values = np.asarray(gross)
    lo, hi = bootstrap_ci(values, np.asarray(sessions))
    return {
        "_per_trade": values,
        "_sessions": np.asarray(sessions),
        "_ids": np.asarray(ids),
        "mode": (f"hold {clock}m" if mode == "hold"
                 else f"trail {100 * trail:.0f}%" if mode == "trail" else mode),
        "trades": int(len(values)),
        "mean_gross_usd": round(float(values.mean()), 2),
        "mean_net_usd": round(float(values.mean() - FEES_PER_ROUND_TRIP_USD), 2),
        "gross_ci95_usd": [round(lo, 2), round(hi, 2)],
        "net_clears_zero": bool(lo - FEES_PER_ROUND_TRIP_USD > 0.0),
        "mean_minutes_held": round(float(np.mean(held)), 2),
        "share_profitable": round(float((values > 0).mean()), 4),
    }


def paired_difference(a: dict, b: dict) -> dict:
    """Trade-by-trade difference between two policies on the same trades.

    Reading two overlapping confidence intervals is the weak form of this
    comparison and can hide a real difference or invent one. The policies are
    run over identical trades, so the difference is paired and its interval is
    far tighter than either policy's own.
    """

    order_a = np.argsort(a["_ids"])
    order_b = np.argsort(b["_ids"])
    assert np.array_equal(a["_ids"][order_a], b["_ids"][order_b]), "policies differ in trades"
    diff = a["_per_trade"][order_a] - b["_per_trade"][order_b]
    lo, hi = bootstrap_ci(diff, a["_sessions"][order_a])
    return {
        "comparison": f"{a['mode']} minus {b['mode']}",
        "mean_difference_usd": round(float(diff.mean()), 2),
        "ci95_usd": [round(lo, 2), round(hi, 2)],
        "better_with_confidence": bool(lo > 0.0),
        "worse_with_confidence": bool(hi < 0.0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(args.table)
    raw = raw.dropna(subset=[f for f in FEATURES if f in raw])
    raw = raw[raw["minute_in_trade"] <= MAX_HOLD_MINUTES]
    # Faithful to the print study, whose paths began one minute after entry.
    # The entry price is carried separately, and the path features at minute
    # one were already built using minute zero, so nothing is lost.
    raw = raw[raw["minute_in_trade"] >= 1]

    blocks = {}
    for source, entry_column in (("bid", "entry_ask_usd"), ("mid", "entry_mid_usd")):
        # The one change between the two runs. `fit_continuation` reads a column
        # literally named `price`, so pointing it at a different series is the
        # entire difference; the formulation is byte-identical either way.
        table = raw.assign(price=raw[source])
        scored = walk_forward(table, shuffle=False, seed=SEED)
        null = walk_forward(table, shuffle=True, seed=SEED + 1)
        if scored.empty:
            continue
        # A stopping rule that exits after eight minutes on average must be
        # read against simply *holding* for eight minutes. Without that control
        # "the policy beats a thirty-minute clock" says only that shorter is
        # better, which is a fact about theta and not a fitted skill.
        runs = [
            score(scored, mode="hold", entry_column=entry_column, seed=SEED + 3, clock=c)
            for c in FIXED_CLOCKS
        ]
        runs += [
            score(scored, mode="trail", entry_column=entry_column, seed=SEED + 3, trail=t)
            for t in TRAIL_LEVELS
        ]
        runs += [
            score(scored, mode=mode, entry_column=entry_column, seed=SEED + 3)
            for mode in ("optimal_stop", "random", "oracle")
        ]
        if not null.empty:
            got = score(null, mode="optimal_stop", entry_column=entry_column, seed=SEED + 5)
            got["mode"] = "optimal_stop (shuffled null)"
            runs.append(got)
        fitted = next(r for r in runs if r["mode"] == "optimal_stop")
        pairs = [
            paired_difference(fitted, r) for r in runs
            if r["mode"].startswith(("hold ", "trail ")) or r["mode"] == "random"
        ]
        blocks[source] = {
            "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
            "paired_vs_fitted": pairs,
        }

    payload = {
        "schema_version": "v5.quoted-exit.v1",
        "question": (
            "The exit was declared solved on last-trade prints, where the value "
            "iteration takes a maximum over a series whose parity residual has a "
            "standard deviation of $102.60. Does it survive selling at the bid, "
            "and does it survive clean prices even with no spread at all?"
        ),
        "formulation": (
            "fitted value iteration imported unchanged from train_optimal_exit; "
            "only the price series differs"
        ),
        "fees_per_round_trip_usd": FEES_PER_ROUND_TRIP_USD,
        "clock_minutes": CLOCK_MINUTES,
        "validation": (
            f"chronological walk-forward, {WARMUP_SESSIONS} warm-up sessions, "
            f"{FOLD_SESSIONS}-session folds"
        ),
        "print_result_being_retested": {
            "hold_gross_usd": -7.3,
            "optimal_stop_gross_usd": -0.2,
            "improvement_usd": 7.1,
            "source": "v4/audit/autoresearch/optimal_exit_2026_08_13/receipt.json",
        },
        "source_table": str(args.table),
        "by_price_source": blocks,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for source, block in blocks.items():
        runs = block["runs"]
        base = next((r for r in runs if r["mode"] == f"hold {CLOCK_MINUTES}m"), None)
        print(f"\nexit settled at the {source.upper()}"
              f"{'  (entry at ask - what a taker really gets)' if source == 'bid' else '  (no spread, clean prices)'}")
        head = (
            f"  {'policy':>28} {'trades':>7} {'gross$':>9} {'ci lo':>9} {'ci hi':>9} "
            f"{'net$':>9} {'held':>6} {'vs hold':>9}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for r in runs:
            lift = r["mean_gross_usd"] - base["mean_gross_usd"] if base else float("nan")
            print(
                f"  {r['mode']:>28} {r['trades']:>7,} {r['mean_gross_usd']:>9,.1f} "
                f"{r['gross_ci95_usd'][0]:>9,.1f} {r['gross_ci95_usd'][1]:>9,.1f} "
                f"{r['mean_net_usd']:>9,.1f} {r['mean_minutes_held']:>6.1f} "
                f"{lift:>+9,.1f}"
            )
        print(f"\n  paired, trade by trade: does the fitted rule beat each simple one?")
        for pair in block["paired_vs_fitted"]:
            verdict = ("BETTER" if pair["better_with_confidence"]
                       else "WORSE" if pair["worse_with_confidence"] else "indistinguishable")
            print(f"    {pair['comparison']:>44}: {pair['mean_difference_usd']:>+7.1f} "
                  f"CI [{pair['ci95_usd'][0]:>+7.1f}, {pair['ci95_usd'][1]:>+7.1f}]  {verdict}")
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
