"""Train where the data is plentiful, score where the price is honest.

Scoring the selective policy on the quote corpus alone leaves a fair objection:
251 sessions minus a warm-up is far less training data than the 1,045-session
trade corpus, so a real signal might simply be unlearnable there. This removes
that objection instead of arguing about it.

The trade corpus runs 2022-06-01 to 2026-07-31. The quote corpus runs
2025-08-01 to 2026-07-31. So the model is fitted on every trade-corpus session
**strictly before the quote corpus begins**, and scored on the quote corpus with
entry charged at the ask and exit paid at the bid.

That is a clean out-of-time test in both senses at once: the training window
ends before the scoring window starts, and the scoring window prices trades at
the touch rather than off the tape. If the timing signal found on prints is real,
it transfers. If it was the tape, it cannot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v5.ops.train_selective_policy import (
    FEATURES,
    SEED,
    TARGET_HIT_RATES,
    bootstrap_ci,
    by_year,
    matched_reference,
    prepare,
    threshold_for_precision,
    walk_serially,
)


def summarise(trades: pd.DataFrame, label: str, target: float) -> dict:
    if trades.empty or len(trades) < 100:
        return {}
    net = trades["net_label"].to_numpy(float)
    lo, hi = bootstrap_ci(net, trades["session"].to_numpy())
    mid = trades["net_label_fair"].to_numpy(float)
    keep = np.isfinite(mid)
    mid_lo, mid_hi = (
        bootstrap_ci(mid[keep], trades["session"].to_numpy()[keep])
        if keep.sum() > 10 else (float("nan"), float("nan"))
    )
    return {
        "label": label,
        "target_hit_rate": target,
        "trades": int(len(net)),
        "hit_rate": round(float((net > 0).mean()), 4),
        "mean_net_usd": round(float(net.mean()), 2),
        "ci95_usd": [round(lo, 2), round(hi, 2)],
        "clears_zero": bool(lo > 0.0),
        "mean_net_mid_usd": round(float(np.nanmean(mid)), 2),
        "ci95_mid_usd": [round(mid_lo, 2), round(mid_hi, 2)],
        "clears_zero_mid": bool(mid_lo > 0.0),
        "mean_premium_usd": round(float(trades["entry_premium"].mean()), 2),
        "mean_spread_usd": round(float(trades["spread_usd"].mean()), 2)
        if "spread_usd" in trades else None,
        "mean_delta": round(float(trades["delta"].mean()), 4),
        "call_share": round(float(trades["is_call_int"].mean()), 4),
        "by_year": by_year(trades),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-table", type=Path, required=True)
    p.add_argument("--score-table", type=Path, required=True)
    p.add_argument("--label", default="profitable",
                   choices=("profitable", "profitable_fair"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    score = prepare(pd.read_parquet(args.score_table), label_column=args.label)
    train = prepare(pd.read_parquet(args.train_table), label_column=args.label)
    cutoff = score["session"].min()
    # Strictly before. No session may appear in both windows.
    train = train[train["session"] < cutoff]
    if train.empty:
        raise SystemExit("no training sessions strictly before the scoring window")

    y = train[args.label].to_numpy(int)
    model = HistGradientBoostingClassifier(
        max_depth=5, max_iter=300, learning_rate=0.05,
        min_samples_leaf=200, l2_regularization=1.0, random_state=SEED,
    )
    model.fit(train[list(FEATURES)].to_numpy(float), y)
    in_sample = model.predict_proba(train[list(FEATURES)].to_numpy(float))[:, 1]
    score = score.copy()
    score["score"] = model.predict_proba(score[list(FEATURES)].to_numpy(float))[:, 1]

    results, kept = [], {}
    for target in TARGET_HIT_RATES:
        cut = threshold_for_precision(in_sample, y, target)
        taken = walk_serially(score, (score["score"] >= cut).to_numpy())
        kept[target] = taken
        got = summarise(taken, "model", target)
        if got:
            results.append(got)
            control = matched_reference(score, taken, SEED + 11)
            hit = summarise(control, "matched contract", target)
            if hit:
                results.append(hit)

    payload = {
        "schema_version": "v5.out-of-time-policy.v1",
        "question": (
            "Does the timing signal learned on last-trade prints survive when "
            "the same policy is charged the ask and paid the bid?"
        ),
        "train_table": str(args.train_table),
        "score_table": str(args.score_table),
        "train_sessions": int(train["session"].nunique()),
        "train_window": [str(train["session"].min()), str(train["session"].max())],
        "score_sessions": int(score["session"].nunique()),
        "score_window": [str(score["session"].min()), str(score["session"].max())],
        "label_column": args.label,
        "pricing": "entry at the ask, exit at the bid, fees only on top",
        "results": results,
        "not_a_validated_result": (
            "a positive here is a reason to freeze and run a known-answer "
            "campaign, not a candidate for promotion."
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        f"\ntrained on {payload['train_sessions']} sessions "
        f"({payload['train_window'][0]} to {payload['train_window'][1]}), "
        f"prints\nscored on {payload['score_sessions']} sessions "
        f"({payload['score_window'][0]} to {payload['score_window'][1]}), "
        f"ask/bid\n"
    )
    head = (
        f"  {'target':>6} {'who':>17} {'trades':>7} {'hit':>6} {'net$':>9} "
        f"{'ci lo':>9} {'ci hi':>9} | {'mid-mid$':>9} {'premium':>8} {'spread':>7}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in results:
        star = " *" if r["clears_zero"] else ""
        print(
            f"  {r['target_hit_rate']:>6.2f} {r['label']:>17} {r['trades']:>7,} "
            f"{100 * r['hit_rate']:>5.1f}% {r['mean_net_usd']:>9,.1f} "
            f"{r['ci95_usd'][0]:>9,.1f} {r['ci95_usd'][1]:>9,.1f} | "
            f"{r['mean_net_mid_usd']:>9,.1f} {r['mean_premium_usd']:>8,.0f} "
            f"{r['mean_spread_usd'] or 0:>7,.1f}{star}"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
