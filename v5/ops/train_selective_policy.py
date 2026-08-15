"""Learn when to trade, not how to rank everything.

Every entry model this project has fitted was asked to **rank** candidates, and
was then made to take a fixed share of them — the top 5%, 10%, 25%. That is the
wrong question for a policy whose most important move is standing aside. Ranking
forces a trade on the worst day of the year; the interesting object is the
*operating point*, and it was never studied.

So this module asks the question the other way round. The model predicts whether
one specific candidate trade clears its own measured round trip. Then, for a
range of target hit rates, the threshold that achieves that hit rate **on the
training fold** is carried forward to the next fold unchanged, and we record
what it actually bought: how often it traded, how often it was right, and what
it kept. The output is a curve — how selective must the bot be to reach 55%, and
is anything left when it gets there — rather than a single number.

**One position at a time.** A candidate is taken only if the bot is idle, and it
occupies the bot for the label horizon. Without that the equity curve is a
fiction that holds forty overlapping contracts, and the validation packet
rejects overlapping positions for exactly this reason.

**Three references, every run.** The current configuration against a random
selection matched on trade count, and against the same machinery retrained on
shuffled labels. A shuffled-label null cannot see feature leakage — that is what
`test_decision_dataset.py` is for — but it does catch a model that is only
exploiting the shape of the selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v5.ops.build_decision_dataset import LABEL_HORIZON
from v5.ops.measure_hold_occupancy import _index

FEATURES = (
    "is_call_int", "moneyness", "moneyness_rel", "entry_premium", "entry_premium_rel",
    "minutes_to_expiry", "round_trip_usd",
    "contract_volume_5m", "contract_volume_15m", "contract_share_of_chain",
    "chain_volume_15m", "chain_call_share_15m",
    "move_5m_rel", "move_15m_rel", "move_30m_rel", "move_60m_rel",
    "range_5m_rel", "range_15m_rel", "range_30m_rel", "range_60m_rel",
    "realised_vol_30m", "range_position", "session_range_rel",
    "minutes_since_open", "minutes_to_close",
    "iv", "delta", "gamma_dollars", "theta_share", "vega_rel",
)

# How choosy to ask the model to be. Each is a hit rate demanded of the training
# fold; the threshold that delivers it is what the test fold must live with.
TARGET_HIT_RATES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)

WARMUP_SESSIONS = 300
FOLD_SESSIONS = 100
BOOTSTRAP_DRAWS = 2_000
SEED = 20260814
# Precision measured on fifty training rows is noise; the shuffled null latched
# a threshold off exactly such a spike and posted +$113/trade on 224 trades.
MIN_PRECISION_ROWS = 2_000


def prepare(table: pd.DataFrame, label_column: str = "profitable") -> pd.DataFrame:
    table = table.copy()
    table["is_call_int"] = table["is_call"].astype(int)
    table["minute_index"] = table["entry_minute"].map(_index)
    return table.dropna(subset=[*FEATURES, "net_label", label_column])


def threshold_for_precision(
    scores: np.ndarray, correct: np.ndarray, target: float
) -> float:
    """Lowest score whose selected set reaches ``target`` hit rate in training.

    Lowest rather than highest, because a policy that trades more at the same
    precision is strictly better, and because taking the highest would pick a
    threshold supported by a handful of training rows.
    """

    order = np.argsort(scores)[::-1]
    hits = np.cumsum(correct[order])
    counts = np.arange(1, len(order) + 1)
    precision = hits / counts
    # Ignore the first few, where precision is one trade wide and meaningless.
    usable = counts >= MIN_PRECISION_ROWS
    ok = usable & (precision >= target)
    if not ok.any():
        return np.inf
    return float(scores[order][np.flatnonzero(ok)[-1]])


def walk_serially(part: pd.DataFrame, taken: np.ndarray) -> pd.DataFrame:
    """Honour one-position-at-a-time inside each session.

    Candidates arrive in decision-minute order. The best scoring candidate at a
    minute is taken if the bot is idle; it is then busy for the label horizon.
    """

    part = part.assign(_take=taken)
    rows = []
    for _, session in part.groupby("session", sort=False):
        free_at = -1
        session = session.sort_values(["minute_index", "score"], ascending=[True, False])
        for minute, block in session.groupby("minute_index", sort=True):
            if minute < free_at:
                continue
            hit = block[block["_take"]]
            if hit.empty:
                continue
            rows.append(hit.iloc[0])
            free_at = minute + LABEL_HORIZON
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=part.columns)


def bootstrap_ci(values: np.ndarray, sessions: np.ndarray) -> tuple[float, float]:
    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        pick = rng.choice(unique, size=len(unique), replace=True)
        draws[b] = values[np.concatenate([index[s] for s in pick])].mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def matched_reference(
    table: pd.DataFrame, model_trades: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Random timing, but the same *kind* of contract the model chose.

    The plain random reference draws from the whole chain, so it holds delta
    0.02 paper at $1,109 while the model holds delta 0.66 paper at $3,287.
    Comparing those two credits the model with having picked an instrument
    rather than a moment. This control removes that: it reproduces the model's
    joint distribution over side, delta and premium, and then picks the minutes
    at random. Whatever survives it is timing.
    """

    if model_trades.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    pool = table[table["session"].isin(set(model_trades["session"].unique()))].copy()
    delta_edges = np.quantile(model_trades["delta"], np.linspace(0, 1, 6))
    premium_edges = np.quantile(model_trades["entry_premium"], np.linspace(0, 1, 6))
    key = lambda f: (  # noqa: E731 - a local addressing helper, not a policy
        f["is_call_int"].astype(int).astype(str)
        + "|" + np.digitize(f["delta"], delta_edges).astype(str)
        + "|" + np.digitize(f["entry_premium"], premium_edges).astype(str)
    )
    pool["_cell"] = key(pool)
    wanted = pd.Series(key(model_trades)).value_counts()
    picked = []
    for cell, count in wanted.items():
        here = pool[pool["_cell"] == cell]
        if here.empty:
            continue
        picked.append(here.sample(n=min(count, len(here)), random_state=seed))
    return pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()


def by_year(trades: pd.DataFrame) -> dict:
    """A policy that only worked in one regime is a regime, not a policy."""

    if trades.empty:
        return {}
    years = trades["session"].str[:4]
    out = {}
    for year, part in trades.groupby(years):
        net = part["net_label"].to_numpy(float)
        out[str(year)] = {
            "trades": int(len(net)),
            "hit_rate": round(float((net > 0).mean()), 4),
            "mean_net_usd": round(float(net.mean()), 2),
        }
    return out


def summarise(trades: pd.DataFrame, label: str, target: float, sessions: int) -> dict:
    if trades.empty:
        return {"label": label, "target_hit_rate": target, "trades": 0}
    net = trades["net_label"].to_numpy(float)
    lo, hi = bootstrap_ci(net, trades["session"].to_numpy())
    # The identical policy scored with bid/ask bounce cancelled. A policy that
    # is profitable on prints and not on parity-averaged value was timing the
    # spread, which nobody who has to pay the spread can do.
    fair = trades["net_label_fair"].to_numpy(float)
    keep = np.isfinite(fair)
    fair_lo, fair_hi = (
        bootstrap_ci(fair[keep], trades["session"].to_numpy()[keep])
        if keep.sum() > 10 else (float("nan"), float("nan"))
    )
    return {
        "by_year": by_year(trades),
        "mean_net_fair_usd": round(float(np.nanmean(fair)), 2),
        "hit_rate_fair": round(float((fair[keep] > 0).mean()), 4) if keep.any() else None,
        "ci95_fair_usd": [round(fair_lo, 2), round(fair_hi, 2)],
        "clears_zero_fair": bool(fair_lo > 0.0),
        "mean_parity_residual_usd": round(float(trades["parity_residual"].mean()), 2),
        "label": label,
        "target_hit_rate": target,
        "trades": int(len(net)),
        "sessions_traded": int(trades["session"].nunique()),
        "trades_per_traded_session": round(len(net) / trades["session"].nunique(), 2),
        "trades_per_available_session": round(len(net) / sessions, 3),
        "hit_rate": round(float((net > 0).mean()), 4),
        "mean_net_usd": round(float(net.mean()), 2),
        "median_net_usd": round(float(np.median(net)), 2),
        "total_net_usd": round(float(net.sum()), 2),
        "mean_premium_usd": round(float(trades["entry_premium"].mean()), 2),
        "mean_delta": round(float(trades["delta"].mean()), 4),
        "call_share": round(float(trades["is_call_int"].mean()), 4),
        "ci95_usd": [round(lo, 2), round(hi, 2)],
        "clears_zero": bool(lo > 0.0),
    }


def run(table: pd.DataFrame, *, shuffle: bool, seed: int,
        label_column: str = "profitable",
        warmup: int = WARMUP_SESSIONS, fold: int = FOLD_SESSIONS) -> dict[float, pd.DataFrame]:
    """Chronological walk-forward; returns the taken trades per target."""

    rng = np.random.default_rng(seed)
    sessions = sorted(table["session"].unique())
    picked: dict[float, list[pd.DataFrame]] = {t: [] for t in TARGET_HIT_RATES}

    start = warmup
    while start < len(sessions):
        train = table[table["session"].isin(set(sessions[:start]))]
        test = table[table["session"].isin(set(sessions[start : start + fold]))]
        start += fold
        if len(train) < 5_000 or test.empty:
            continue

        y = train[label_column].to_numpy(int)
        if shuffle:
            y = rng.permutation(y)
        model = HistGradientBoostingClassifier(
            max_depth=5, max_iter=300, learning_rate=0.05,
            min_samples_leaf=200, l2_regularization=1.0, random_state=seed,
        )
        model.fit(train[list(FEATURES)].to_numpy(float), y)

        in_sample = model.predict_proba(train[list(FEATURES)].to_numpy(float))[:, 1]
        scored = test.copy()
        scored["score"] = model.predict_proba(test[list(FEATURES)].to_numpy(float))[:, 1]
        for target in TARGET_HIT_RATES:
            cut = threshold_for_precision(in_sample, y, target)
            got = walk_serially(scored, (scored["score"] >= cut).to_numpy())
            if not got.empty:
                picked[target].append(got)

    return {
        t: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
        for t, v in picked.items()
    }


def random_reference(table: pd.DataFrame, counts: dict[float, int], seed: int,
                     warmup: int = WARMUP_SESSIONS) -> dict:
    """Same number of trades, chosen at random, same occupancy rule."""

    rng = np.random.default_rng(seed)
    sessions = sorted(table["session"].unique())
    out = {}
    pool = table[table["session"].isin(set(sessions[warmup:]))].copy()
    for target, want in counts.items():
        if want <= 0:
            out[target] = pd.DataFrame()
            continue
        pool["score"] = rng.random(len(pool))
        # Choose a score cut that yields roughly the wanted number of trades
        # after the occupancy walk, so the reference is matched on count.
        cut = np.quantile(pool["score"], max(0.0, 1.0 - 3.0 * want / len(pool)))
        got = walk_serially(pool, (pool["score"] >= cut).to_numpy())
        if len(got) > want:
            got = got.sample(n=want, random_state=seed)
        out[target] = got
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=WARMUP_SESSIONS,
                   help="sessions reserved before the first fit; the quote corpus\n                        has 251 sessions and cannot spare 300")
    p.add_argument("--fold", type=int, default=FOLD_SESSIONS)
    p.add_argument("--label", default="profitable",
                   choices=("profitable", "profitable_fair"),
                   help="profitable_fair trains on parity-averaged value, so the "
                        "model is not rewarded for timing bid/ask bounce")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = prepare(pd.read_parquet(args.table), label_column=args.label)
    total_sessions = table["session"].nunique()
    scored_sessions = max(0, total_sessions - args.warmup)

    model_trades = run(table, shuffle=False, seed=SEED, label_column=args.label,
                       warmup=args.warmup, fold=args.fold)
    null_trades = run(table, shuffle=True, seed=SEED + 1, label_column=args.label,
                      warmup=args.warmup, fold=args.fold)
    counts = {t: len(v) for t, v in model_trades.items()}
    random_trades = random_reference(table, counts, SEED + 7, warmup=args.warmup)
    matched_trades = {
        t: matched_reference(table, v, SEED + 11) for t, v in model_trades.items()
    }

    results = []
    for target in TARGET_HIT_RATES:
        for label, source in (
            ("model", model_trades),
            ("matched contract", matched_trades),
            ("random", random_trades),
            ("null (shuffled)", null_trades),
        ):
            got = summarise(source[target], label, target, scored_sessions)
            # A cell of a handful of trades is noise wearing a result's clothes;
            # the shuffled null produces these whenever its target is unreachable.
            if got.get("trades", 0) >= 100:
                results.append(got)

    best = max(
        (r for r in results if r["label"] == "model" and r["trades"] >= 100),
        key=lambda r: r["mean_net_usd"],
        default=None,
    )
    equity = []
    if best is not None:
        trades = model_trades[best["target_hit_rate"]].sort_values(["session", "minute_index"])
        curve = trades.groupby("session")["net_label"].sum().cumsum()
        equity = [{"session": s, "cumulative_net_usd": round(float(v), 2)}
                  for s, v in curve.items()]
        # Persist the trades themselves so the next question does not need a refit.
        trades.to_parquet(args.out.parent / "model_trades.parquet")

    payload = {
        "schema_version": "v5.selective-policy.v1",
        "question": (
            "How choosy must the bot be to reach a given hit rate, and is "
            "anything left when it gets there?"
        ),
        "label": f"trade clears its own measured round trip at {LABEL_HORIZON} minutes",
        "label_column": args.label,
        "features": list(FEATURES),
        "target_hit_rates": list(TARGET_HIT_RATES),
        "validation": (
            f"chronological walk-forward, {args.warmup} warm-up sessions, "
            f"{args.fold}-session folds; thresholds from the training fold only"
        ),
        "occupancy": "one position at a time, held for the label horizon",
        "sessions_total": int(total_sessions),
        "sessions_scored": int(scored_sessions),
        "source_table": str(args.table),
        "results": results,
        "best_model_cell": best,
        "equity_curve": equity,
        "not_a_validated_result": (
            "a positive here is a reason to freeze and run a known-answer "
            "campaign, not a candidate for promotion."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    head = (
        f"  {'target':>6} {'who':>17} {'trades':>7} {'hit':>6} "
        f"{'net$':>8} {'ci lo':>8} {'ci hi':>8} | {'FAIR net$':>9} {'ci lo':>8} "
        f"{'ci hi':>8} {'resid':>7} {'premium':>8}"
    )
    print(f"\n{scored_sessions} scored sessions, one position at a time")
    print("FAIR = the same trades priced at parity-averaged value, bounce cancelled\n")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in results:
        star = " *" if r.get("clears_zero_fair") else ""
        print(
            f"  {r['target_hit_rate']:>6.2f} {r['label']:>17} {r['trades']:>7,} "
            f"{100 * r['hit_rate']:>5.1f}% {r['mean_net_usd']:>8,.1f} "
            f"{r['ci95_usd'][0]:>8,.1f} {r['ci95_usd'][1]:>8,.1f} | "
            f"{r['mean_net_fair_usd']:>9,.1f} {r['ci95_fair_usd'][0]:>8,.1f} "
            f"{r['ci95_fair_usd'][1]:>8,.1f} {r['mean_parity_residual_usd']:>7,.1f} "
            f"{r['mean_premium_usd']:>8,.0f}{star}"
        )
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
