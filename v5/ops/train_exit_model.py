"""Train the exit: leave when nothing better is coming, and see what it captures.

The excursion measurement set the terms. Holding to a clock earns **-$32** a
trade; selling at the best minute close would earn **+$321**; every declared exit
rule captures none of the gap and most make the break-even worse. The question
this module answers is the only one left: **how much of that $348 does a trained
exit actually get?**

At every minute of an open trade the model predicts ``remaining_max_gain`` — the
best price still to come, relative to the price right now — and the policy exits
the first time that prediction falls below a declared threshold. Nothing about
the entry is learned. Trades are opened unconditionally on both sides every
fifteen minutes, so whatever the exit earns, it earned alone.

## Why the comparisons matter more than the number

A profitable exit policy is easy to produce by accident. Exiting early shortens
the holding period, which on a decaying asset looks like skill. Three references
separate skill from that:

* **hold to horizon** — what the trade was worth with no exit at all;
* **random exit** — exit at a uniformly random minute, matching the policy's
  *average holding time* but carrying no information. This is the reference that
  catches "shorter is better" masquerading as a model;
* **oracle** — sell at the best minute close, the unattainable ceiling.

Plus a **shuffled-label null**, which retrains on permuted labels. A pipeline
that still profits with the labels destroyed is leaking, and the gap between the
real run and the nulls is the result — not the real run alone.

Validation is chronological throughout: every prediction comes from a model
fitted only on strictly earlier sessions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = (
    "minute_in_trade",
    "minutes_left",
    "minutes_to_close",
    "return_from_entry",
    "peak_so_far",
    "drawdown_from_peak",
    "trough_so_far",
    "return_1m",
    "return_3m",
    "moneyness_at_entry",
    "underlying_return_from_entry",
    "underlying_return_3m",
    "entry_premium",
)
LABEL = "remaining_max_gain"

INITIAL_TRAIN_SESSIONS = 300
FOLD_SESSIONS = 100
COST_PER_TRADE_USD = 23.0
CONTRACT_MULTIPLIER = 100.0
# Exit when the model expects less than this much still to come. Declared, not
# tuned: zero means "nothing better is coming", and the two others bracket it.
EXIT_THRESHOLDS = (0.0, 0.02, 0.05)
# Absolute thresholds barely fire: the best price still to come is above the
# current price at 90% of decision points, so "exit when nothing is left" is a
# rule that almost never triggers. These are quantiles of the model's own
# predictions **on its training fold**, which is causal and makes the policy
# actually choose. A rule that fires at the 40th percentile leaves roughly the
# least promising 40% of minutes.
EXIT_QUANTILES = (0.2, 0.4, 0.6, 0.8)
BOOTSTRAP_DRAWS = 2_000
SEED = 20260813


def walk_forward(table: pd.DataFrame, *, shuffle_labels: bool, seed: int) -> pd.DataFrame:
    """Predict each fold from a model fitted only on earlier sessions."""

    rng = np.random.default_rng(seed)
    sessions = sorted(table["session"].unique())
    out = []
    start = INITIAL_TRAIN_SESSIONS
    while start < len(sessions):
        train = table[table["session"].isin(set(sessions[:start]))]
        test = table[table["session"].isin(set(sessions[start : start + FOLD_SESSIONS]))]
        start += FOLD_SESSIONS
        if len(train) < 5_000 or test.empty:
            continue
        y = train[LABEL].to_numpy(float)
        if shuffle_labels:
            y = rng.permutation(y)
        model = HistGradientBoostingRegressor(
            max_depth=4,
            max_iter=300,
            learning_rate=0.05,
            min_samples_leaf=200,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(train[list(FEATURES)].to_numpy(float), y)
        got = test.copy()
        got["predicted_gain"] = model.predict(test[list(FEATURES)].to_numpy(float))
        # Thresholds from the training fold's own predictions, so the policy is
        # calibrated without seeing the sessions it will trade.
        in_sample = model.predict(train[list(FEATURES)].to_numpy(float))
        for q in EXIT_QUANTILES:
            got[f"cut_{q}"] = float(np.quantile(in_sample, q))
        out.append(got)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _entry_price(part: pd.DataFrame) -> float:
    return float(part["entry_premium"].iloc[0]) / CONTRACT_MULTIPLIER


def realise(part: pd.DataFrame, exit_row: int | None) -> float:
    """Net dollars for one trade exiting at ``exit_row``, or at the horizon."""

    entry = _entry_price(part)
    price = float(
        part["price"].iloc[exit_row] if exit_row is not None else part["price"].iloc[-1]
    )
    return (price - entry) * CONTRACT_MULTIPLIER - COST_PER_TRADE_USD


def score(pred: pd.DataFrame, threshold: float, *, mode: str, seed: int) -> dict:
    """Run one exit policy over every trade and report what it kept."""

    rng = np.random.default_rng(seed)
    nets, holds, sessions, held_minutes = [], [], [], []
    for _, part in pred.groupby("trade_id", sort=False):
        part = part.sort_values("minute_in_trade")
        if mode == "model":
            fires = np.flatnonzero(part["predicted_gain"].to_numpy(float) < threshold)
            row = int(fires[0]) if fires.size else None
        elif mode == "model_quantile":
            cut = part[f"cut_{threshold}"].to_numpy(float)
            fires = np.flatnonzero(part["predicted_gain"].to_numpy(float) < cut)
            row = int(fires[0]) if fires.size else None
        elif mode == "hold":
            row = None
        elif mode == "random":
            row = int(rng.integers(0, len(part)))
        elif mode == "oracle":
            row = int(np.argmax(part["price"].to_numpy(float)))
        else:
            raise ValueError(mode)
        nets.append(realise(part, row))
        holds.append(realise(part, None))
        sessions.append(part["session"].iloc[0])
        held_minutes.append(len(part) if row is None else row + 1)

    net = np.asarray(nets)
    sess = np.asarray(sessions)
    unique = np.unique(sess)
    index = {s: np.flatnonzero(sess == s) for s in unique}
    boot = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        chosen = boot.choice(unique, size=len(unique), replace=True)
        draws[b] = net[np.concatenate([index[s] for s in chosen])].mean()
    return {
        "mode": mode,
        "threshold": threshold if mode == "model" else None,
        "trades": int(len(net)),
        "mean_net_usd": round(float(net.mean()), 2),
        "ci95_usd": [
            round(float(np.quantile(draws, 0.025)), 2),
            round(float(np.quantile(draws, 0.975)), 2),
        ],
        "clears_zero": bool(np.quantile(draws, 0.025) > 0.0),
        "beats_hold_usd": round(float(net.mean() - np.mean(holds)), 2),
        "mean_minutes_held": round(float(np.mean(held_minutes)), 2),
        "share_profitable": round(float((net > 0).mean()), 4),
        "worst_trade_usd": round(float(net.min()), 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    table = pd.read_parquet(args.table).sort_values(["session", "trade_id", "minute_in_trade"])
    real = walk_forward(table, shuffle_labels=False, seed=SEED)
    null = walk_forward(table, shuffle_labels=True, seed=SEED + 1)
    if real.empty:
        raise SystemExit("no folds were scored")

    runs = [score(real, t, mode="model", seed=SEED) for t in EXIT_THRESHOLDS]
    runs += [score(real, q, mode="model_quantile", seed=SEED) for q in EXIT_QUANTILES]
    runs.append(score(real, 0.0, mode="hold", seed=SEED))
    runs.append(score(real, 0.0, mode="oracle", seed=SEED))
    # The random exit is matched to the model's own average holding time by
    # construction: both draw from the same set of minutes.
    runs.append(score(real, 0.0, mode="random", seed=SEED + 5))
    null_runs = (
        [score(null, q, mode="model_quantile", seed=SEED) for q in EXIT_QUANTILES]
        if not null.empty
        else []
    )

    correlation = float(np.corrcoef(real["predicted_gain"], real[LABEL])[0, 1])
    payload = {
        "schema_version": "v5.exit-model.v1",
        "question": "how much of the exit prize does a trained exit actually capture?",
        "features": list(FEATURES),
        "label": "best price still to come in the window, relative to the price now",
        "model": "HistGradientBoostingRegressor, depth 4, 300 iterations, lr 0.05",
        "validation": (
            f"chronological walk-forward, {INITIAL_TRAIN_SESSIONS} warm-up sessions, "
            f"{FOLD_SESSIONS}-session folds, each predicted by a model fitted only "
            "on strictly earlier sessions"
        ),
        "policy": (
            "exit at the first minute the predicted remaining gain falls below "
            "the threshold; quantile variants take the cut from the model's own "
            "predictions on its training fold"
        ),
        "exit_quantiles": list(EXIT_QUANTILES),
        "cost_per_trade_usd": COST_PER_TRADE_USD,
        "decision_points_scored": int(len(real)),
        "trades_scored": int(real["trade_id"].nunique()),
        "sessions_scored": int(real["session"].nunique()),
        "prediction_label_correlation": round(correlation, 6),
        "runs": runs,
        "null_shuffled_labels": null_runs,
        "references": {
            "hold": "no exit at all",
            "random": "exit at a uniformly random minute, carrying no information",
            "oracle": "sell at the best minute close; unattainable ceiling",
        },
        "not_a_validated_result": (
            "a positive here is a reason to run a known-answer campaign, not a "
            "candidate for promotion. The entry is unconditional and no gate has "
            "been passed."
        ),
        "source_table": str(args.table),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"\n{payload['trades_scored']:,} trades over {payload['sessions_scored']} "
        f"sessions scored out of fold; prediction/label correlation "
        f"{correlation:+.4f}\n"
    )
    head = (
        f"  {'run':>22} {'trades':>8} {'held':>6} {'net/trade':>10} "
        f"{'95% CI':>22} {'vs hold':>9} {'winners':>8}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for label, rows in (("", runs), ("null ", null_runs)):
        for row in rows:
            name = f"{label}{row['mode']}"
            if row["threshold"] is not None:
                name += f" @{row['threshold']:.2f}"
            name = name.replace("model_quantile", "model q")
            lo, hi = row["ci95_usd"]
            print(
                f"  {name:>22} {row['trades']:>8,} {row['mean_minutes_held']:>6.1f} "
                f"{row['mean_net_usd']:>10,.1f} {f'[{lo:,.1f}, {hi:,.1f}]':>22} "
                f"{row['beats_hold_usd']:>+9,.1f} "
                f"{100 * row['share_profitable']:>7.1f}%"
                f"{'  CLEARS' if row['clears_zero'] else ''}"
            )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
