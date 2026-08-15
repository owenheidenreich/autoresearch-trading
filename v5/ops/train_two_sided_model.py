"""A model that chooses to buy, to sell, or to stand aside — and its honest score.

Every screen this project has run tested **buying**. That was an omission, not a
conclusion: an unconditional average being negative does not foreclose a
conditional strategy, and scoring both sides of the same declared rules shows it
plainly — at sixty minutes a short in the last ninety minutes keeps **+$32.3 a
trade after the full aggressive round trip**, which long-only scoring could never
have seen.

So this fits a model to the quantity that actually decides a trade: the expected
profit and loss of the straddle, in units of its own premium. Given that, the
policy writes itself — sell when the option looks dear, buy when it looks cheap,
stand aside when the expected edge does not clear the cost of trading.

## What keeps this honest

**Chronological folds only.** Every prediction is made by a model that saw only
strictly earlier sessions. There is no random split anywhere, because a random
split lets tomorrow train the model that trades today.

**The cost is charged before the trade is taken, not after.** The policy demands
an expected edge larger than the round trip it is about to pay, so the threshold
is an economic one rather than a fitted one.

**Two null controls run alongside.** The labels are shuffled within the training
window, and separately the sign of the prediction is randomised. A pipeline that
scores well on shuffled labels is leaking, and the number to look at is the gap
between the real run and the nulls, not the real run alone.

**The tail is reported, not averaged away.** A short straddle can lose several
times the premium it collected. Mean profit with an unreported tail is how
accounts die.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = (
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
    "expansion",
    "cheapness",
)
LABEL = "straddle_return"

# Sessions in the first training block; every later fold trains on everything
# before it.
INITIAL_TRAIN_SESSIONS = 300
FOLD_SESSIONS = 100

COSTS_PER_LEG_USD = (3.08, 14.0, 25.0)
# The policy must expect to clear this multiple of the round trip before it
# trades. Declared, not tuned.
EDGE_MARGIN = 1.5

BOOTSTRAP_DRAWS = 2_000
SEED = 20260813


def prepare(table: pd.DataFrame) -> pd.DataFrame:
    """Causal features, all scale-free so a 2022 session is comparable to 2026."""

    out = table.copy()
    spot = out["spot"].to_numpy(float)
    for name in ("range_15m", "range_30m", "range_60m", "move_15m", "move_30m",
                 "move_60m", "session_range"):
        out[f"{name}_rel"] = out[name] / spot
    out["straddle_share_of_spot"] = out["straddle_premium"] / (spot * 100.0)
    out["expansion"] = out["range_15m"] / out["range_60m"].replace(0.0, np.nan)
    out["cheapness"] = out["range_30m"] / (
        out["straddle_premium"].replace(0.0, np.nan) / 100.0
    )
    # The label is a return on the straddle, which is stationary across eras in
    # a way the dollar profit and loss is not.
    out[LABEL] = out["straddle_gross"] / out["straddle_premium"]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURES, LABEL])


def walk_forward(table: pd.DataFrame, *, shuffle_labels: bool, seed: int) -> pd.DataFrame:
    """Predict each fold from a model that saw only earlier sessions."""

    rng = np.random.default_rng(seed)
    sessions = sorted(table["session"].unique())
    predictions = []
    start = INITIAL_TRAIN_SESSIONS
    while start < len(sessions):
        train_ids = set(sessions[:start])
        test_ids = set(sessions[start : start + FOLD_SESSIONS])
        start += FOLD_SESSIONS
        train = table[table["session"].isin(train_ids)]
        test = table[table["session"].isin(test_ids)]
        if len(train) < 500 or test.empty:
            continue
        y = train[LABEL].to_numpy(float)
        if shuffle_labels:
            y = rng.permutation(y)
        model = HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=200,
            learning_rate=0.05,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(train[list(FEATURES)].to_numpy(float), y)
        got = test.copy()
        got["predicted_return"] = model.predict(test[list(FEATURES)].to_numpy(float))
        predictions.append(got)
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def score_policy(pred: pd.DataFrame, cost_per_leg: float, *, random_side_seed=None) -> dict:
    """Take the side the model expects to pay, when it expects to pay enough."""

    premium = pred["straddle_premium"].to_numpy(float)
    edge_usd = pred["predicted_return"].to_numpy(float) * premium
    round_trip = 2 * cost_per_leg
    take = np.abs(edge_usd) > EDGE_MARGIN * round_trip
    side = np.sign(edge_usd)
    if random_side_seed is not None:
        side = np.random.default_rng(random_side_seed).choice([-1.0, 1.0], size=len(side))
    net = side * pred["straddle_gross"].to_numpy(float) - round_trip
    if take.sum() < 100:
        return {"cost_per_leg_usd": cost_per_leg, "trades": int(take.sum()),
                "verdict": "too few trades"}

    taken_net = net[take]
    sessions = pred["session"].to_numpy()[take]
    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        draws[b] = taken_net[np.concatenate([index[s] for s in chosen])].mean()
    per_session = pd.Series(taken_net).groupby(pd.Series(sessions)).sum()
    return {
        "cost_per_leg_usd": cost_per_leg,
        "trades": int(take.sum()),
        "share_of_slots_traded": round(float(take.mean()), 4),
        "share_short": round(float((side[take] < 0).mean()), 4),
        "sessions_traded": int(len(unique)),
        "mean_net_usd": round(float(taken_net.mean()), 2),
        "ci95_usd": [
            round(float(np.quantile(draws, 0.025)), 2),
            round(float(np.quantile(draws, 0.975)), 2),
        ],
        "clears_zero": bool(np.quantile(draws, 0.025) > 0.0),
        "total_net_usd": round(float(taken_net.sum()), 2),
        "mean_per_session_usd": round(float(per_session.mean()), 2),
        "worst_trade_usd": round(float(taken_net.min()), 2),
        "worst_session_usd": round(float(per_session.min()), 2),
        "p01_trade_usd": round(float(np.quantile(taken_net, 0.01)), 2),
        "share_of_trades_losing": round(float((taken_net < 0).mean()), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    raw = pd.read_parquet(args.table)
    results = {}
    for hold, part in raw.groupby("hold"):
        prepared = prepare(part).sort_values(["session", "entry_minute"])
        real = walk_forward(prepared, shuffle_labels=False, seed=SEED)
        null = walk_forward(prepared, shuffle_labels=True, seed=SEED + 1)
        if real.empty:
            continue
        block = {
            "slots_scored_out_of_fold": int(len(real)),
            "sessions_scored": int(real["session"].nunique()),
            "predicted_vs_actual_correlation": round(
                float(np.corrcoef(real["predicted_return"], real[LABEL])[0, 1]), 6
            ),
            "policy": [score_policy(real, c) for c in COSTS_PER_LEG_USD],
            "null_shuffled_labels": [
                score_policy(null, c) for c in COSTS_PER_LEG_USD
            ] if not null.empty else [],
            "null_random_side": [
                score_policy(real, c, random_side_seed=SEED + 2)
                for c in COSTS_PER_LEG_USD
            ],
        }
        results[f"{hold}m"] = block

    payload = {
        "schema_version": "v5.two-sided-model.v1",
        "question": (
            "Given a model that may buy, sell or stand aside, is there an "
            "out-of-fold profit after the cost of trading?"
        ),
        "features": list(FEATURES),
        "label": "straddle gross profit and loss as a fraction of straddle premium",
        "model": "HistGradientBoostingRegressor, depth 3, 200 iterations, lr 0.05",
        "validation": (
            f"chronological walk-forward: first {INITIAL_TRAIN_SESSIONS} sessions "
            f"train the first model, each later fold of {FOLD_SESSIONS} sessions is "
            "predicted by a model fitted only on strictly earlier sessions"
        ),
        "policy": (
            f"trade the side of the predicted edge when |edge| exceeds "
            f"{EDGE_MARGIN}x the round trip it will pay; otherwise stand aside"
        ),
        "costs_per_leg_usd": list(COSTS_PER_LEG_USD),
        "controls": [
            "labels shuffled inside each training window",
            "side randomised while keeping the model's trade selection",
        ],
        "source_table": str(args.table),
        "by_hold": results,
        "not_a_validated_result": (
            "an out-of-fold profit here is a reason to run a known-answer "
            "campaign, not a candidate for promotion. Short straddles are barred "
            "by the charter and carry a tail this scoring reports but does not "
            "size."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for hold, block in results.items():
        print(
            f"\n{hold} hold — {block['sessions_scored']} sessions scored out of fold, "
            f"prediction/actual correlation {block['predicted_vs_actual_correlation']:+.4f}"
        )
        head = (
            f"  {'run':>16} {'$/leg':>6} {'trades':>7} {'short':>6} {'net/trade':>10} "
            f"{'95% CI':>20} {'per session':>12} {'worst trade':>12}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for label, rows in (
            ("model", block["policy"]),
            ("null: shuffled", block["null_shuffled_labels"]),
            ("null: random side", block["null_random_side"]),
        ):
            for row in rows:
                if "mean_net_usd" not in row:
                    print(f"  {label:>16} {row['cost_per_leg_usd']:>6.2f} "
                          f"{row['trades']:>7,}   {row['verdict']}")
                    continue
                lo, hi = row["ci95_usd"]
                print(
                    f"  {label:>16} {row['cost_per_leg_usd']:>6.2f} {row['trades']:>7,} "
                    f"{100 * row['share_short']:>5.0f}% {row['mean_net_usd']:>10,.1f} "
                    f"{f'[{lo:,.1f}, {hi:,.1f}]':>20} "
                    f"{row['mean_per_session_usd']:>12,.1f} "
                    f"{row['worst_trade_usd']:>12,.0f}"
                    f"{'  CLEARS' if row['clears_zero'] else ''}"
                )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
