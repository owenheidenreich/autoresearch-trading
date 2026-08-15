"""The whole design: an entry that picks the contract, an exit that works it.

The exit model on its own recovered the option's decay and nothing more. That is
not a verdict on the design, because it was tested with **random entries** —
every fifteen minutes, both sides, no selection at all. A scalp does not trade
that way. The entry is supposed to find the contract about to make the largest
premium move in the shortest time, and only then does the exit have something
worth working.

This module tests the two halves together.

**The entry label is the excursion, not the direction.** How far the contract's
price travels in its favour, as a fraction of the premium paid. Every screen this
project has ever run predicted direction; none has predicted this, and it is what
the described design is actually asking for.

**Both models are chronological.** Entry and exit are each fitted only on
sessions strictly earlier than the ones they act on, and the entry's selection
cut comes from its own training fold.

**The reference is a random entry taking the same number of trades.** Selecting
the most promising tenth of trades and comparing against *all* trades would
credit selection with the effect of trading less. The comparison is against a
random tenth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from v5.ops.train_exit_model import (
    COST_PER_TRADE_USD,
    CONTRACT_MULTIPLIER,
    FEATURES as EXIT_FEATURES,
    FOLD_SESSIONS,
    INITIAL_TRAIN_SESSIONS,
    LABEL as EXIT_LABEL,
    SEED,
    realise,
    walk_forward as exit_walk_forward,
)

# Entry-time features, all known at the moment the contract is bought.
ENTRY_FEATURES = (
    "range_15m_rel",
    "range_30m_rel",
    "range_60m_rel",
    "move_15m_rel",
    "move_30m_rel",
    "move_60m_rel",
    "range_position",
    "session_range_rel",
    "minutes_to_close",
    "straddle_share_of_spot",
    "moneyness_at_entry",
    "entry_premium_rel",
)
# Two candidate entry labels. The first is the design as stated — "the largest
# premium-changing move" read as a percentage. The second is the same idea in
# the units that actually pay: a 74% excursion on a $200 contract is $148, while
# a 39% excursion on a $1,000 contract is $390, and the fixed round trip falls
# on both alike. Which of these is the right objective is exactly the question.
ENTRY_LABELS = ("trade_excursion", "trade_excursion_usd")
# Fractions of the available trades the entry model is allowed to take.
SELECTION_RATES = (0.05, 0.10, 0.25, 0.50)
BOOTSTRAP_DRAWS = 2_000


def build(exit_table: pd.DataFrame, magnitude_table: pd.DataFrame) -> pd.DataFrame:
    """Attach entry-time features to every decision point of every trade."""

    parts = exit_table["trade_id"].str.split("|", expand=True)
    exit_table = exit_table.copy()
    exit_table["entry_minute"] = parts[1]

    entry = magnitude_table[magnitude_table["hold"] == 15].copy()
    spot = entry["spot"].to_numpy(float)
    for name in ("range_15m", "range_30m", "range_60m", "move_15m", "move_30m",
                 "move_60m", "session_range"):
        entry[f"{name}_rel"] = entry[name] / spot
    entry["straddle_share_of_spot"] = entry["straddle_premium"] / (spot * 100.0)
    entry["spot_at_entry"] = spot
    keep = ["session", "entry_minute", "spot_at_entry", "range_position",
            "straddle_share_of_spot"] + [
        f"{n}_rel" for n in ("range_15m", "range_30m", "range_60m", "move_15m",
                             "move_30m", "move_60m", "session_range")
    ]
    joined = exit_table.merge(
        entry[keep].drop_duplicates(subset=["session", "entry_minute"]),
        on=["session", "entry_minute"],
        how="inner",
    )
    joined["entry_premium_rel"] = joined["entry_premium"] / (
        joined["spot_at_entry"] * 100.0
    )
    # The entry label: how far this trade's price travelled in its favour,
    # measured from the entry price. Known only after the fact, which is what a
    # label is.
    excursion = (
        joined.assign(gain=joined["price"] / (joined["entry_premium"] / CONTRACT_MULTIPLIER) - 1.0)
        .groupby("trade_id")["gain"]
        .max()
        .rename(ENTRY_LABELS[0])
    )
    joined = joined.merge(excursion, on="trade_id", how="left")
    # The same excursion in dollars, net of the round trip it must pay.
    peak_usd = (
        joined.assign(
            gain_usd=(joined["price"] - joined["entry_premium"] / CONTRACT_MULTIPLIER)
            * CONTRACT_MULTIPLIER
        )
        .groupby("trade_id")["gain_usd"]
        .max()
        .rename("trade_excursion_usd")
    ) - COST_PER_TRADE_USD
    return joined.merge(peak_usd, on="trade_id", how="left")


def entry_walk_forward(
    table: pd.DataFrame, *, shuffle: bool, seed: int, label: str
) -> pd.DataFrame:
    """One prediction per trade, from a model fitted only on earlier sessions."""

    per_trade = (
        table.sort_values("minute_in_trade")
        .groupby("trade_id", as_index=False)
        .first()
    )
    rng = np.random.default_rng(seed)
    sessions = sorted(per_trade["session"].unique())
    out = []
    start = INITIAL_TRAIN_SESSIONS
    while start < len(sessions):
        train = per_trade[per_trade["session"].isin(set(sessions[:start]))]
        test = per_trade[
            per_trade["session"].isin(set(sessions[start : start + FOLD_SESSIONS]))
        ]
        start += FOLD_SESSIONS
        if len(train) < 2_000 or test.empty:
            continue
        y = train[label].to_numpy(float)
        if shuffle:
            y = rng.permutation(y)
        model = HistGradientBoostingRegressor(
            max_depth=4, max_iter=300, learning_rate=0.05,
            min_samples_leaf=100, l2_regularization=1.0, random_state=seed,
        )
        model.fit(train[list(ENTRY_FEATURES)].to_numpy(float), y)
        got = test.copy()
        got["predicted_excursion"] = model.predict(
            test[list(ENTRY_FEATURES)].to_numpy(float)
        )
        in_sample = model.predict(train[list(ENTRY_FEATURES)].to_numpy(float))
        for rate in SELECTION_RATES:
            got[f"cut_{rate}"] = float(np.quantile(in_sample, 1.0 - rate))
        out.append(got)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(
    exits: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    rate: float,
    exit_quantile: float,
    random_entry_seed: int | None,
) -> dict:
    """Take the selected trades, run the exit on them, and report the result."""

    if random_entry_seed is None:
        chosen = entries[
            entries["predicted_excursion"] >= entries[f"cut_{rate}"]
        ]["trade_id"]
    else:
        rng = np.random.default_rng(random_entry_seed)
        take = max(1, int(round(rate * len(entries))))
        chosen = entries["trade_id"].sample(
            n=min(take, len(entries)), random_state=int(rng.integers(0, 2**31))
        )
    chosen = set(chosen)
    part = exits[exits["trade_id"].isin(chosen)]
    if part.empty:
        return {}

    nets, sessions, excursions = [], [], []
    for _, trade in part.groupby("trade_id", sort=False):
        trade = trade.sort_values("minute_in_trade")
        cut = trade[f"cut_{exit_quantile}"].to_numpy(float)
        fires = np.flatnonzero(trade["predicted_gain"].to_numpy(float) < cut)
        row = int(fires[0]) if fires.size else None
        nets.append(realise(trade, row))
        sessions.append(trade["session"].iloc[0])
        entry_price = float(trade["entry_premium"].iloc[0]) / CONTRACT_MULTIPLIER
        excursions.append(float(trade["price"].max()) / entry_price - 1.0)

    net = np.asarray(nets)
    sess = np.asarray(sessions)
    unique = np.unique(sess)
    index = {s: np.flatnonzero(sess == s) for s in unique}
    boot = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        pick = boot.choice(unique, size=len(unique), replace=True)
        draws[b] = net[np.concatenate([index[s] for s in pick])].mean()
    return {
        "selection_rate": rate,
        "exit_quantile": exit_quantile,
        "entry": "random" if random_entry_seed is not None else "model",
        "trades": int(len(net)),
        "mean_realised_excursion_pct": round(100 * float(np.mean(excursions)), 2),
        "mean_net_usd": round(float(net.mean()), 2),
        "ci95_usd": [
            round(float(np.quantile(draws, 0.025)), 2),
            round(float(np.quantile(draws, 0.975)), 2),
        ],
        "clears_zero": bool(np.quantile(draws, 0.025) > 0.0),
        "share_profitable": round(float((net > 0).mean()), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exit-table", type=Path, required=True)
    p.add_argument("--magnitude-table", type=Path, required=True)
    p.add_argument("--exit-quantile", type=float, default=0.6)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    joined = build(
        pd.read_parquet(args.exit_table), pd.read_parquet(args.magnitude_table)
    ).sort_values(["session", "trade_id", "minute_in_trade"])

    exits = exit_walk_forward(joined, shuffle_labels=False, seed=SEED)
    if exits.empty:
        raise SystemExit("no folds were scored")

    runs, correlations = [], {}
    for label in ENTRY_LABELS:
        entries = entry_walk_forward(joined, shuffle=False, seed=SEED, label=label)
        null_entries = entry_walk_forward(joined, shuffle=True, seed=SEED + 1, label=label)
        if entries.empty:
            continue
        shared = set(exits["trade_id"]) & set(entries["trade_id"])
        entries = entries[entries["trade_id"].isin(shared)]
        null_entries = null_entries[null_entries["trade_id"].isin(shared)]
        correlations[label] = round(
            float(np.corrcoef(entries["predicted_excursion"], entries[label])[0, 1]), 6
        )
        for rate in SELECTION_RATES:
            for tag, frame, rnd in (
                (f"model:{label}", entries, None),
                ("random", entries, SEED + 7),
                (f"null:{label}", null_entries, None),
            ):
                got = evaluate(exits, frame, rate=rate,
                               exit_quantile=args.exit_quantile, random_entry_seed=rnd)
                if got:
                    got["entry"] = tag
                    got["entry_label"] = label
                    runs.append(got)
    entries_available = int(exits["trade_id"].nunique())

    payload = {
        "schema_version": "v5.entry-and-exit.v1",
        "question": "does an entry that predicts the excursion make the exit worth having?",
        "entry_features": list(ENTRY_FEATURES),
        "entry_label": "the trade's realised excursion, as a fraction of premium paid",
        "exit_label": EXIT_LABEL,
        "exit_quantile_used": args.exit_quantile,
        "selection_rates": list(SELECTION_RATES),
        "cost_per_trade_usd": COST_PER_TRADE_USD,
        "entry_prediction_correlation_by_label": correlations,
        "trades_available": entries_available,
        "sessions": int(exits["session"].nunique()),
        "runs": runs,
        "reference": (
            "a random entry taking the same number of trades, so selection is "
            "not credited with the effect of trading less"
        ),
        "not_a_validated_result": (
            "a positive here is a reason to run a known-answer campaign, not a "
            "candidate for promotion."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\n{payload['trades_available']:,} trades over {payload['sessions']} sessions")
    for label, corr in correlations.items():
        print(f"  entry model on {label}: prediction/label correlation {corr:+.4f}")
    print()
    head = (
        f"  {'take':>6} {'entry':>26} {'trades':>8} {'excursion':>10} "
        f"{'net/trade':>10} {'95% CI':>22} {'winners':>8}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for row in runs:
        lo, hi = row["ci95_usd"]
        print(
            f"  {100 * row['selection_rate']:>5.0f}% {row['entry']:>26} "
            f"{row['trades']:>8,} {row['mean_realised_excursion_pct']:>9.1f}% "
            f"{row['mean_net_usd']:>10,.1f} {f'[{lo:,.1f}, {hi:,.1f}]':>22} "
            f"{100 * row['share_profitable']:>7.1f}%"
            f"{'  CLEARS' if row['clears_zero'] else ''}"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
