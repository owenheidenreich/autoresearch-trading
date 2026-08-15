"""Both models, both fixed: greeks in the entry, optimal stopping in the exit.

Three results set this up.

**The exit can make a trade free and no more.** Formulated as optimal stopping,
it drives the gross profit and loss of a randomly-entered trade from -$7.3 to
**-$0.2** — it removes the decay exactly and stops there. That is the ceiling on
an exit working a random entry, and it means every dollar of profit has to come
from the entry.

**The entry's label was wrong and is now right.** Selecting on *percentage*
excursion doubles the excursion and loses more than random, because percentage
selects cheap contracts against a fixed round trip. In dollars net of cost the
same machinery beats random at every selection rate.

**The entry was blind.** It read the underlying's path and the contract's price.
It now reads implied volatility, delta, gamma in dollars, theta as a share of
premium, and vega — all recomputed from price by :mod:`v5.research.greeks`, so
the same function runs offline and live and nothing can diverge.

Entry-time greeks are reconstructed from the entry premium, the strike, the
parity spot at entry and the clock — never from the first minute *after* entry,
which would be a minute of look-ahead on the one decision that must be causal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from v5.ops.train_exit_model import (
    BOOTSTRAP_DRAWS,
    CONTRACT_MULTIPLIER,
    COST_PER_TRADE_USD,
    FOLD_SESSIONS,
    INITIAL_TRAIN_SESSIONS,
    SEED,
)
from v5.ops.train_optimal_exit import walk_forward as exit_walk_forward
from v5.research import greeks as gk

ENTRY_FEATURES = (
    "minutes_to_expiry_at_entry",
    "moneyness_at_entry",
    "entry_premium",
    "entry_premium_rel",
    "entry_iv",
    "entry_delta",
    "entry_gamma_dollars",
    "entry_theta_share",
    "entry_vega_rel",
    # Path features from the magnitude table, all computed from history ending
    # AT the entry minute. Audited against the same timestamp rule that caught
    # the leak below.
    "range_15m_rel",
    "range_30m_rel",
    "range_60m_rel",
    "move_15m_rel",
    "move_30m_rel",
    "move_60m_rel",
    "range_position",
)
# REMOVED 2026-08-13 after a feature audit: ``underlying_return_3m`` on the
# first recorded row is spot(entry+1)/spot(entry) - 1, the underlying's move in
# the minute *after* the entry decision. It correlates +0.19 with the trade's
# outcome and lifted the top-5% cell from break-even to +$119 a trade. The
# shuffled-label null did not catch it and could not: permuting the label
# removes the thing a leak would leak to. Only a timestamp audit of each
# feature finds this class of defect.
LEAKED_AND_REMOVED = ("underlying_return_3m_at_entry",)
ENTRY_LABEL = "trade_excursion_usd"
SELECTION_RATES = (0.05, 0.10, 0.25, 0.50)


def add_path_features(first: pd.DataFrame, magnitude: pd.DataFrame) -> pd.DataFrame:
    """Underlying path as of the entry minute, from the magnitude table.

    Every column here is built from a window ending at the entry minute, never
    after it.
    """

    entry = magnitude[magnitude["hold"] == 15].copy()
    spot = entry["spot"].to_numpy(float)
    for name in ("range_15m", "range_30m", "range_60m",
                 "move_15m", "move_30m", "move_60m"):
        entry[f"{name}_rel"] = entry[name] / spot
    keep = ["session", "entry_minute", "range_position"] + [
        f"{n}_rel" for n in ("range_15m", "range_30m", "range_60m",
                             "move_15m", "move_30m", "move_60m")
    ]
    first = first.copy()
    first["entry_minute"] = first["trade_id"].str.split("|").str[1]
    return first.merge(
        entry[keep].drop_duplicates(subset=["session", "entry_minute"]),
        on=["session", "entry_minute"], how="inner",
    )


def add_entry_features(table: pd.DataFrame, magnitude: pd.DataFrame) -> pd.DataFrame:
    """Greeks as they stood at the entry minute, from entry-minute inputs only."""

    first = (
        table.sort_values("minute_in_trade")
        .groupby("trade_id", as_index=False)
        .first()
    )
    entry_price = first["entry_premium"].to_numpy(float) / CONTRACT_MULTIPLIER
    strike = first["strike"].to_numpy(float)
    is_call = first["is_call"].to_numpy(bool)
    moneyness = first["moneyness_at_entry"].to_numpy(float)
    # moneyness is signed so positive means in the money, so the spot at entry
    # is recoverable from the strike without touching any later minute.
    spot_at_entry = np.where(is_call, strike + moneyness, strike - moneyness)
    # The first recorded minute is one after entry, so entry had one more minute.
    minutes = first["minutes_to_expiry"].to_numpy(float) + 1.0

    got = gk.greeks_batch(entry_price, spot_at_entry, strike, minutes, is_call)
    first["minutes_to_expiry_at_entry"] = minutes
    first["spot_at_entry"] = spot_at_entry
    first["entry_premium_rel"] = first["entry_premium"] / (spot_at_entry * 100.0)
    first["entry_iv"] = got["iv"]
    first["entry_delta"] = got["delta"]
    first["entry_gamma_dollars"] = got["gamma"] * spot_at_entry**2 / 100.0
    first["entry_theta_share"] = got["theta_per_minute"] / entry_price
    first["entry_vega_rel"] = got["vega"] / (entry_price * CONTRACT_MULTIPLIER)
    # The label: the trade's best dollar gain, net of the round trip it pays.
    peak = (
        table.assign(
            gain=(table["price"] - table["entry_premium"] / CONTRACT_MULTIPLIER)
            * CONTRACT_MULTIPLIER
        )
        .groupby("trade_id")["gain"]
        .max()
        .rename(ENTRY_LABEL)
    ) - COST_PER_TRADE_USD
    first = add_path_features(first, magnitude)
    keep = ["trade_id", "session", ENTRY_LABEL] + list(ENTRY_FEATURES)
    return first.merge(peak, on="trade_id", how="left")[keep]


def entry_walk_forward(per_trade: pd.DataFrame, *, shuffle: bool, seed: int) -> pd.DataFrame:
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
        y = train[ENTRY_LABEL].to_numpy(float)
        if shuffle:
            y = rng.permutation(y)
        model = HistGradientBoostingRegressor(
            max_depth=4, max_iter=300, learning_rate=0.05,
            min_samples_leaf=100, l2_regularization=1.0, random_state=seed,
        )
        model.fit(train[list(ENTRY_FEATURES)].to_numpy(float), y)
        got = test.copy()
        got["predicted"] = model.predict(test[list(ENTRY_FEATURES)].to_numpy(float))
        in_sample = model.predict(train[list(ENTRY_FEATURES)].to_numpy(float))
        for rate in SELECTION_RATES:
            got[f"cut_{rate}"] = float(np.quantile(in_sample, 1.0 - rate))
        out.append(got)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def combine(exits: pd.DataFrame, entries: pd.DataFrame, *, rate: float,
            random_seed: int | None, label: str) -> dict:
    """Selected trades, closed by the optimal-stopping exit."""

    if random_seed is None:
        chosen = set(entries[entries["predicted"] >= entries[f"cut_{rate}"]]["trade_id"])
    else:
        take = max(1, int(round(rate * len(entries))))
        chosen = set(entries["trade_id"].sample(n=min(take, len(entries)),
                                                random_state=random_seed))
    part = exits[exits["trade_id"].isin(chosen)]
    if part.empty:
        return {}
    nets, sessions, held = [], [], []
    for _, trade in part.groupby("trade_id", sort=False):
        trade = trade.sort_values("minute_in_trade")
        price = trade["price"].to_numpy(float)
        entry = float(trade["entry_premium"].iloc[0]) / CONTRACT_MULTIPLIER
        fires = np.flatnonzero(price >= trade["continuation"].to_numpy(float))
        row = int(fires[0]) if fires.size else len(price) - 1
        nets.append((price[row] - entry) * CONTRACT_MULTIPLIER - COST_PER_TRADE_USD)
        sessions.append(trade["session"].iloc[0])
        held.append(row + 1)

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
        "entry": label,
        "selection_rate": rate,
        "trades": int(len(net)),
        "mean_minutes_held": round(float(np.mean(held)), 2),
        "mean_gross_usd": round(float(net.mean() + COST_PER_TRADE_USD), 2),
        "mean_net_usd": round(float(net.mean()), 2),
        "ci95_usd": [
            round(float(np.quantile(draws, 0.025)), 2),
            round(float(np.quantile(draws, 0.975)), 2),
        ],
        "clears_zero": bool(np.quantile(draws, 0.025) > 0.0),
        "share_profitable": round(float((net > 0).mean()), 4),
        "trades_per_session": round(float(len(net) / len(unique)), 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--magnitude-table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from v5.ops.train_optimal_exit import FEATURES as EXIT_FEATURES

    table = pd.read_parquet(args.table)
    table = table.dropna(subset=[f for f in EXIT_FEATURES if f in table]).sort_values(
        ["session", "trade_id", "minute_in_trade"]
    )
    exits = exit_walk_forward(table, shuffle=False, seed=SEED)
    if exits.empty:
        raise SystemExit("no exit folds were scored")

    per_trade = add_entry_features(
        table, pd.read_parquet(args.magnitude_table)
    ).dropna(subset=list(ENTRY_FEATURES))
    entries = entry_walk_forward(per_trade, shuffle=False, seed=SEED)
    nulls = entry_walk_forward(per_trade, shuffle=True, seed=SEED + 1)
    shared = set(exits["trade_id"]) & set(entries["trade_id"])
    exits = exits[exits["trade_id"].isin(shared)]
    entries = entries[entries["trade_id"].isin(shared)]
    nulls = nulls[nulls["trade_id"].isin(shared)]

    correlation = float(np.corrcoef(entries["predicted"], entries[ENTRY_LABEL])[0, 1])
    runs = []
    for rate in SELECTION_RATES:
        for label, frame, seed in (
            ("model", entries, None),
            ("random", entries, SEED + 7),
            ("null (shuffled)", nulls, None),
        ):
            got = combine(exits, frame, rate=rate, random_seed=seed, label=label)
            if got:
                runs.append(got)

    payload = {
        "schema_version": "v5.full-system.v1",
        "question": "with both fixes, does the combined system clear the cost of trading?",
        "entry_features": list(ENTRY_FEATURES),
        "entry_label": "the trade's best dollar gain, net of the round trip",
        "exit": "optimal stopping by fitted value iteration",
        "greeks_provenance": (
            "recomputed from price, spot, strike and clock; entry greeks use the "
            "entry minute only, never the minute after it"
        ),
        "cost_per_trade_usd": COST_PER_TRADE_USD,
        "entry_prediction_correlation": round(correlation, 6),
        "trades_available": int(entries["trade_id"].nunique()),
        "sessions": int(entries["session"].nunique()),
        "runs": runs,
        "not_a_validated_result": (
            "a positive here is a reason to freeze a hypothesis and run a "
            "known-answer campaign, not a candidate for promotion."
        ),
        "source_table": str(args.table),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"\n{payload['trades_available']:,} trades over {payload['sessions']} sessions; "
        f"entry prediction/label correlation {correlation:+.4f}\n"
    )
    head = (
        f"  {'take':>6} {'entry':>17} {'trades':>8} {'per sess':>9} {'held':>6} "
        f"{'gross':>9} {'net/trade':>10} {'95% CI':>22} {'winners':>8}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for row in runs:
        lo, hi = row["ci95_usd"]
        print(
            f"  {100 * row['selection_rate']:>5.0f}% {row['entry']:>17} "
            f"{row['trades']:>8,} {row['trades_per_session']:>9.2f} "
            f"{row['mean_minutes_held']:>6.1f} {row['mean_gross_usd']:>9,.1f} "
            f"{row['mean_net_usd']:>10,.1f} {f'[{lo:,.1f}, {hi:,.1f}]':>22} "
            f"{100 * row['share_profitable']:>7.1f}%"
            f"{'  CLEARS' if row['clears_zero'] else ''}"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
