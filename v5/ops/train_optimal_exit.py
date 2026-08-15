"""The exit as optimal stopping: hold only when holding is worth more.

The first exit model predicted ``remaining_max_gain`` — the best price still to
come — and the policy left when that fell below a threshold. It was the wrong
objective and the symptom was visible: the label is positive at 90% of decision
points, because prices wobble and *something* better almost always arrives, so a
rule keyed to it barely fires and captured 2% of the available prize.

The right question is not "will a better price appear?" It is **"is this price
better than what following my own policy from here would get me?"** That is an
optimal-stopping problem and it has a standard solution.

## Fitted value iteration

Let ``V`` be the value of standing in a trade at some minute, and ``C`` the value
of continuing rather than exiting now:

* at the last minute the position must close, so ``V = price``;
* otherwise ``V = max(price, C)`` — take the better of leaving and staying;
* and ``C`` is what the *next* minute is worth, which is what the model learns.

Starting from "exit immediately everywhere" and sweeping this a few times drives
``V`` up to the value of the best stopping rule the features can express. The
policy that falls out needs no threshold at all: **exit the first minute the
price is at least the predicted continuation value.**

Two properties matter for what comes later.

**The round trip does not appear.** It is paid once whatever minute the trade
closes, so it cannot change the stopping decision — only whether to open at all.
It is subtracted at the end, where it belongs.

**Every input is available live.** The features are the contract's own price and
path, its greeks recomputed from that price, and the clock. A live decision runs
the same fitted function on the same quantities.
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

# The contract's own state, its greeks, and the clock. Nothing a live feed lacks.
FEATURES = (
    "minute_in_trade",
    "minutes_left",
    "minutes_to_expiry",
    "return_from_entry",
    "peak_so_far",
    "drawdown_from_peak",
    "trough_so_far",
    "return_1m",
    "return_3m",
    "underlying_return_from_entry",
    "underlying_return_3m",
    "moneyness_now",
    "iv",
    "delta",
    "gamma_dollars",
    "theta_share_of_price",
    "vega",
)
# How many times the value function is swept. Three is enough for a
# thirty-minute horizon; the value stops moving well before that.
VALUE_ITERATIONS = 3


def _next_within_trade(frame: pd.DataFrame, column: str) -> np.ndarray:
    """The value of ``column`` one minute later, NaN at the end of a trade."""

    shifted = frame.groupby("trade_id", sort=False)[column].shift(-1)
    return shifted.to_numpy(float)


def fit_continuation(train: pd.DataFrame, *, seed: int, shuffle: bool):
    """Sweep the value function and return the fitted continuation model."""

    rng = np.random.default_rng(seed)
    price = train["price"].to_numpy(float)
    is_last = np.isnan(_next_within_trade(train, "price"))
    # Start from "exit immediately": the value of standing anywhere is today's
    # price. Each sweep lets the option of waiting add whatever it is worth.
    value = price.copy()
    model = None
    features = train[list(FEATURES)].to_numpy(float)
    for _ in range(VALUE_ITERATIONS):
        target = _next_within_trade(train.assign(_v=value), "_v")
        usable = ~is_last & np.isfinite(target)
        if usable.sum() < 1_000:
            break
        y = target[usable]
        if shuffle:
            y = rng.permutation(y)
        model = HistGradientBoostingRegressor(
            max_depth=4, max_iter=300, learning_rate=0.05,
            min_samples_leaf=200, l2_regularization=1.0, random_state=seed,
        )
        model.fit(features[usable], y)
        continuation = model.predict(features)
        value = np.where(is_last, price, np.maximum(price, continuation))
    return model


def walk_forward(table: pd.DataFrame, *, shuffle: bool, seed: int) -> pd.DataFrame:
    """Continuation values for each fold, from models fitted on earlier sessions."""

    sessions = sorted(table["session"].unique())
    out = []
    start = INITIAL_TRAIN_SESSIONS
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


def score(pred: pd.DataFrame, *, mode: str, seed: int) -> dict:
    """Run a policy over every trade and report what it kept."""

    rng = np.random.default_rng(seed)
    nets, sessions, held = [], [], []
    for _, trade in pred.groupby("trade_id", sort=False):
        trade = trade.sort_values("minute_in_trade")
        price = trade["price"].to_numpy(float)
        entry = float(trade["entry_premium"].iloc[0]) / CONTRACT_MULTIPLIER
        if mode == "optimal_stop":
            # Leave the first minute the price is worth at least what waiting is.
            fires = np.flatnonzero(price >= trade["continuation"].to_numpy(float))
            row = int(fires[0]) if fires.size else len(price) - 1
        elif mode == "hold":
            row = len(price) - 1
        elif mode == "random":
            row = int(rng.integers(0, len(price)))
        elif mode == "oracle":
            row = int(np.argmax(price))
        else:
            raise ValueError(mode)
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
        "mode": mode,
        "trades": int(len(net)),
        "mean_minutes_held": round(float(np.mean(held)), 2),
        "mean_net_usd": round(float(net.mean()), 2),
        "mean_gross_usd": round(float(net.mean() + COST_PER_TRADE_USD), 2),
        "ci95_usd": [
            round(float(np.quantile(draws, 0.025)), 2),
            round(float(np.quantile(draws, 0.975)), 2),
        ],
        "clears_zero": bool(np.quantile(draws, 0.025) > 0.0),
        "share_profitable": round(float((net > 0).mean()), 4),
        "worst_trade_usd": round(float(net.min()), 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    table = pd.read_parquet(args.table)
    table = table.dropna(subset=[f for f in FEATURES if f in table]).sort_values(
        ["session", "trade_id", "minute_in_trade"]
    )
    real = walk_forward(table, shuffle=False, seed=SEED)
    null = walk_forward(table, shuffle=True, seed=SEED + 1)
    if real.empty:
        raise SystemExit("no folds were scored")

    runs = [score(real, mode=m, seed=SEED + i) for i, m in enumerate(
        ("optimal_stop", "hold", "random", "oracle")
    )]
    null_runs = [score(null, mode="optimal_stop", seed=SEED)] if not null.empty else []

    payload = {
        "schema_version": "v5.optimal-exit.v1",
        "question": "does optimal stopping capture more of the exit prize than a threshold rule?",
        "formulation": (
            "fitted value iteration: V = max(price, C) with C the learned value "
            "of continuing, swept "
            f"{VALUE_ITERATIONS} times; exit the first minute price >= C"
        ),
        "features": list(FEATURES),
        "greeks_provenance": (
            "recomputed from price, spot, strike and the clock by "
            "v5/research/greeks.py, so the same function runs offline and live"
        ),
        "cost_per_trade_usd": COST_PER_TRADE_USD,
        "cost_note": (
            "paid once whenever the trade closes, so it cannot affect the "
            "stopping decision; subtracted after"
        ),
        "validation": (
            f"chronological walk-forward, {INITIAL_TRAIN_SESSIONS} warm-up "
            f"sessions, {FOLD_SESSIONS}-session folds"
        ),
        "trades_scored": int(real["trade_id"].nunique()),
        "sessions_scored": int(real["session"].nunique()),
        "runs": runs,
        "null_shuffled_labels": null_runs,
        "source_table": str(args.table),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"\n{payload['trades_scored']:,} trades over {payload['sessions_scored']} "
        f"sessions, scored out of fold\n"
    )
    head = (
        f"  {'policy':>18} {'held':>6} {'gross':>9} {'net/trade':>10} "
        f"{'95% CI':>22} {'winners':>8}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for label, rows in (("", runs), ("null ", null_runs)):
        for row in rows:
            lo, hi = row["ci95_usd"]
            print(
                f"  {label + row['mode']:>18} {row['mean_minutes_held']:>6.1f} "
                f"{row['mean_gross_usd']:>9,.1f} {row['mean_net_usd']:>10,.1f} "
                f"{f'[{lo:,.1f}, {hi:,.1f}]':>22} "
                f"{100 * row['share_profitable']:>7.1f}%"
                f"{'  CLEARS' if row['clears_zero'] else ''}"
            )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
