"""Nine cells, two main effects, one interaction.

Every disappointing result this project has produced was a combined number with
no way to blame a half. An entry scored through a fitted exit and an exit scored
on a fitted entry stream cannot be told apart from one another.

This runs **{random, model, oracle} entry x {clock, model, oracle} exit** over one
trade population at one pricing, so the two contributions and their interference
are measured rather than inferred:

* **entry effect** = (model, clock) - (random, clock). A clock has no capacity to
  absorb or destroy entry information, so this is the entry's contribution alone.
* **exit effect** = (random, model) - (random, clock). Random entries carry no
  information, so this is the exit's contribution alone.
* **interaction** = (model, model) - (random, clock) - entry effect - exit effect.

The interaction is the point. A negative one is the diagnosis that the halves are
fighting — an exit closing before the entry's thesis plays out — which is
currently invisible and is the likeliest reason a combination disappoints.

**One currency.** Every cell reports `n` trades, hit rate `p`, mean winner `W` and
mean loser `L`, because that is what makes the halves comparable: the entry moves
`p` and composition, the exit moves `W` and `L`, and the combined cell can be
*predicted* from the parts and checked against what it actually did.

**Trade counts are matched across entry arms.** Random and oracle each take the
same number of trades as the arm they are compared against, so an effect is never
an artefact of trading more or less.

Phase 0 (`--phase0`) fills only the oracle row and column and fits nothing. It is
a gate: if a half's ceiling is small, no model of that half is worth building.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v5.ops.build_factorial_population import MAX_HOLD_MINUTES
from v5.ops.train_optimal_exit import FEATURES as EXIT_FEATURES, fit_continuation
from v5.ops.train_selective_policy import FEATURES as ENTRY_FEATURES
from v5.ops.resolve_exit_price_convention import CONTRACT_MULTIPLIER

CLOCK_MINUTES = 5
FEES_USD = 3.08
WARMUP_SESSIONS = 100
FOLD_SESSIONS = 30
BOOTSTRAP_DRAWS = 2_000
SEED = 20260814

ENTRY_ARMS = ("random", "model", "oracle")
EXIT_ARMS = ("clock", "model", "oracle")


# ---------------------------------------------------------------- exit policies

def exit_rows(paths: pd.DataFrame, arm: str, rng: np.random.Generator,
              price_column: str) -> pd.DataFrame:
    """The minute each trade closes under one exit arm, and what it fetches.

    A trade cannot be opened and closed in the same minute bar, so every arm
    decides from minute one onward. Without that the stopping rule can 'exit' at
    its own fill and holding times stop being comparable.
    """

    out = []
    for trade_id, trade in paths.groupby("trade_id", sort=False):
        trade = trade.sort_values("minute_in_trade")
        price = trade[price_column].to_numpy(float)
        elapsed = trade["minute_in_trade"].to_numpy(float)
        if price.size < 1:
            continue
        # Address the path by elapsed minute, never by position. The table starts
        # at minute one because a trade cannot open and close in the same bar, so
        # a positional index silently holds one minute longer than it claims.
        if arm == "clock":
            row = int(np.argmin(np.abs(elapsed - CLOCK_MINUTES)))
        elif arm == "oracle":
            row = int(np.argmax(price))
        elif arm == "model":
            fires = np.flatnonzero(price >= trade["continuation"].to_numpy(float))
            row = int(fires[0]) if fires.size else price.size - 1
        else:
            raise ValueError(arm)
        if elapsed[row] < 1.0:
            continue
        out.append((trade_id, float(price[row]), float(elapsed[row])))
    return pd.DataFrame(out, columns=["trade_id", "exit_price", "minutes_held"])


def fit_exit(paths: pd.DataFrame, price_column: str, *, seed: int,
             shuffle: bool = False) -> pd.DataFrame:
    """Continuation values by chronological walk-forward, fitted on random entries."""

    table = paths.assign(price=paths[price_column])
    sessions = sorted(table["session"].unique())
    out, start = [], WARMUP_SESSIONS
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
        got["continuation"] = model.predict(test[list(EXIT_FEATURES)].to_numpy(float))
        out.append(got)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------- entry policies

def entry_scores(candidates: pd.DataFrame, *, seed: int, shuffle: bool) -> pd.DataFrame:
    """Out-of-fold probability that a candidate clears its own round trip."""

    sessions = sorted(candidates["session"].unique())
    out, start = [], WARMUP_SESSIONS
    rng = np.random.default_rng(seed)
    while start < len(sessions):
        train = candidates[candidates["session"].isin(set(sessions[:start]))]
        test = candidates[candidates["session"].isin(set(sessions[start : start + FOLD_SESSIONS]))]
        start += FOLD_SESSIONS
        if len(train) < 5_000 or test.empty:
            continue
        y = train["profitable"].to_numpy(int)
        if shuffle:
            y = rng.permutation(y)
        model = HistGradientBoostingClassifier(
            max_depth=5, max_iter=300, learning_rate=0.05,
            min_samples_leaf=200, l2_regularization=1.0, random_state=seed,
        )
        model.fit(train[list(ENTRY_FEATURES)].to_numpy(float), y)
        got = test.copy()
        got["score"] = model.predict_proba(test[list(ENTRY_FEATURES)].to_numpy(float))[:, 1]
        out.append(got)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def select(frame: pd.DataFrame, arm: str, rng: np.random.Generator,
           hold: int) -> pd.DataFrame:
    """One position at a time; at each idle decision minute, take one candidate.

    `oracle` sees the realised outcome and is a ceiling, never a target. It picks
    the best contract available at a minute it is already committed to trading —
    perfect contract choice with mechanical timing — so it stays comparable to
    `random`, which takes the same number of trades at the same minutes.
    """

    if arm == "random":
        key = rng.random(len(frame))
    elif arm == "oracle":
        key = frame["net"].to_numpy(float)
    elif arm == "model":
        key = frame["score"].to_numpy(float)
    else:
        raise ValueError(arm)

    frame = frame.assign(_key=key).sort_values(
        ["session", "minute_index", "_key"], ascending=[True, True, False]
    )
    taken = []
    for _, session in frame.groupby("session", sort=False):
        free_at = -1.0
        for minute, block in session.groupby("minute_index", sort=True):
            if minute < free_at:
                continue
            taken.append(block.iloc[0])
            free_at = minute + hold
    return pd.DataFrame(taken)


# ------------------------------------------------------------------- reporting

def bootstrap_ci(values: np.ndarray, sessions: np.ndarray) -> tuple[float, float]:
    unique = np.unique(sessions)
    index = {s: np.flatnonzero(sessions == s) for s in unique}
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        pick = rng.choice(unique, size=len(unique), replace=True)
        draws[b] = values[np.concatenate([index[s] for s in pick])].mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def currency(trades: pd.DataFrame, entry_arm: str, exit_arm: str) -> dict:
    """The four numbers everything else is derived from."""

    net = trades["net"].to_numpy(float)
    wins, losses = net[net > 0], net[net <= 0]
    lo, hi = bootstrap_ci(net, trades["session"].to_numpy())
    p = float((net > 0).mean())
    W = float(wins.mean()) if wins.size else 0.0
    L = float(-losses.mean()) if losses.size else 0.0
    return {
        "entry": entry_arm,
        "exit": exit_arm,
        "n": int(len(net)),
        "p": round(p, 4),
        "W": round(W, 2),
        "L": round(L, 2),
        "mean_net_usd": round(float(net.mean()), 2),
        "ci95_usd": [round(lo, 2), round(hi, 2)],
        "clears_zero": bool(lo > 0.0),
        "breakeven_p": round((L + FEES_USD) / (W + L), 4) if (W + L) > 0 else None,
        "mean_minutes_held": round(float(trades["minutes_held"].mean()), 2),
        "mean_premium_usd": round(float(trades["entry_ask_usd"].mean()), 2),
        "mean_delta": round(float(trades["delta"].mean()), 4),
        "call_share": round(float(trades["is_call"].mean()), 4),
    }


def predicted_from_parts(base: dict, entry_only: dict, exit_only: dict) -> dict:
    """What the combination should be worth if the halves do not interfere.

    Take the hit rate the entry earned alone, and apply the exit's effect on the
    win and loss magnitudes to the contracts the entry actually chose.

    **The exit's effect must be applied as a ratio, not a level.** The entry
    changes which contracts are held, so the exit's absolute `W` and `L` — earned
    on a random contract mix — do not transfer to the entry's mix. Splicing the
    levels reports a composition shift as interference; the first run of this
    driver did exactly that and disagreed with its own factorial by $18.
    """

    p = entry_only["p"]
    w_ratio = exit_only["W"] / base["W"] if base["W"] else 1.0
    l_ratio = exit_only["L"] / base["L"] if base["L"] else 1.0
    W = entry_only["W"] * w_ratio
    L = entry_only["L"] * l_ratio
    return {
        "p_from_entry": round(p, 4),
        "exit_effect_on_W": round(w_ratio, 4),
        "exit_effect_on_L": round(l_ratio, 4),
        "W": round(W, 2),
        "L": round(L, 2),
        "predicted_net_usd": round(p * W - (1.0 - p) * L - FEES_USD, 2),
    }


def component_shifts(base: dict, entry_only: dict, exit_only: dict,
                     combined: dict) -> dict:
    """Which of the three numbers each half moved, and where they disagree.

    This is the diagnosis the currency exists for: an entry that lifts the hit
    rate while enlarging the losers, or an exit that stretches winners without
    trimming losses, are different problems with different repairs.
    """

    def shift(a: dict, b: dict) -> dict:
        return {
            "p": round(b["p"] - a["p"], 4),
            "W": round(b["W"] - a["W"], 2),
            "L": round(b["L"] - a["L"], 2),
        }

    return {
        "entry_moved": shift(base, entry_only),
        "exit_moved": shift(base, exit_only),
        "combined_moved": shift(base, combined),
        "reading": (
            "the entry is expected to move p; the exit is expected to move W and "
            "L. An entry that also enlarges L, or an exit that enlarges L while "
            "stretching W, is a named defect rather than a disappointing total."
        ),
    }


def build_cell(candidates: pd.DataFrame, paths: pd.DataFrame, *, entry_arm: str,
               exit_arm: str, price_column: str, entry_price_column: str,
               fitted_exit: pd.DataFrame, rng: np.random.Generator,
               match_count: int | None) -> pd.DataFrame | None:
    """Trades taken and closed under one (entry, exit) pair."""

    source = fitted_exit if exit_arm == "model" else paths
    if source.empty:
        return None
    closes = exit_rows(source, exit_arm, rng, price_column)
    if closes.empty:
        return None

    frame = candidates.merge(closes, on="trade_id", how="inner")
    if frame.empty:
        return None
    frame["net"] = (
        frame["exit_price"] * CONTRACT_MULTIPLIER - frame[entry_price_column] - FEES_USD
    )
    taken = select(frame, entry_arm, rng, MAX_HOLD_MINUTES)
    if taken.empty:
        return None
    if match_count is not None and len(taken) > match_count:
        # Trade counts are matched across entry arms so an effect is never an
        # artefact of trading more or less than the arm it is compared to.
        taken = taken.sort_values("_key", ascending=False).head(match_count) \
            if entry_arm != "random" else taken.sample(n=match_count, random_state=SEED)
    return taken


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--paths", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase0", action="store_true",
                   help="oracle row and column only; fits nothing")
    p.add_argument("--pricing", choices=("bid", "mid"), default="bid")
    p.add_argument("--label", default="profitable")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(args.candidates)
    # The entry feature list names the integer form; the population stores the
    # boolean. Derive it here rather than in the table so both stay one source.
    candidates["is_call_int"] = candidates["is_call"].astype(int)
    paths = pd.read_parquet(args.paths)
    paths = paths.dropna(subset=[f for f in EXIT_FEATURES if f in paths])
    paths = paths[paths["minute_in_trade"] >= 1]

    price_column = "bid" if args.pricing == "bid" else "mid"
    entry_price_column = "entry_ask_usd" if args.pricing == "bid" else "entry_mid_usd"

    # Score every candidate under a clock exit so the entry label exists.
    clock = exit_rows(paths, "clock", np.random.default_rng(SEED), price_column)
    labelled = candidates.merge(clock, on="trade_id", how="inner")
    labelled["net"] = (
        labelled["exit_price"] * CONTRACT_MULTIPLIER
        - labelled[entry_price_column] - FEES_USD
    )
    labelled["profitable"] = (labelled["net"] > 0).astype(int)
    candidates = candidates.merge(
        labelled[["trade_id", "profitable"]], on="trade_id", how="inner"
    )

    entry_arms = ["random", "oracle"] if args.phase0 else list(ENTRY_ARMS)
    exit_arms = ["clock", "oracle"] if args.phase0 else list(EXIT_ARMS)

    fitted_exit = pd.DataFrame()
    if "model" in exit_arms:
        fitted_exit = fit_exit(paths, price_column, seed=SEED)
    if not args.phase0:
        scored = entry_scores(candidates, seed=SEED, shuffle=False)
        candidates = scored if not scored.empty else candidates.assign(score=0.0)
    else:
        candidates = candidates.assign(score=0.0)

    rng = np.random.default_rng(SEED)
    # The baseline fixes the trade count every other cell is matched to.
    baseline = build_cell(candidates, paths, entry_arm="random", exit_arm="clock",
                          price_column=price_column,
                          entry_price_column=entry_price_column,
                          fitted_exit=fitted_exit, rng=rng, match_count=None)
    if baseline is None:
        raise SystemExit("baseline cell is empty")
    match_count = len(baseline)

    grid = {}
    for entry_arm in entry_arms:
        for exit_arm in exit_arms:
            if entry_arm == "oracle" and exit_arm == "model":
                continue
            if entry_arm == "model" and exit_arm == "oracle":
                continue
            taken = (baseline if (entry_arm, exit_arm) == ("random", "clock")
                     else build_cell(candidates, paths, entry_arm=entry_arm,
                                     exit_arm=exit_arm, price_column=price_column,
                                     entry_price_column=entry_price_column,
                                     fitted_exit=fitted_exit, rng=rng,
                                     match_count=match_count))
            if taken is not None and len(taken) >= 50:
                grid[(entry_arm, exit_arm)] = currency(taken, entry_arm, exit_arm)

    effects = {}
    base = grid.get(("random", "clock"))
    if base and not args.phase0:
        entry_only = grid.get(("model", "clock"))
        exit_only = grid.get(("random", "model"))
        combined = grid.get(("model", "model"))
        if entry_only and exit_only and combined:
            entry_effect = entry_only["mean_net_usd"] - base["mean_net_usd"]
            exit_effect = exit_only["mean_net_usd"] - base["mean_net_usd"]
            interaction = (
                combined["mean_net_usd"] - base["mean_net_usd"]
                - entry_effect - exit_effect
            )
            predicted = predicted_from_parts(base, entry_only, exit_only)
            shifts = component_shifts(base, entry_only, exit_only, combined)
            effects = {
                "entry_effect_usd": round(entry_effect, 2),
                "exit_effect_usd": round(exit_effect, 2),
                "interaction_usd": round(interaction, 2),
                "combined_measured_usd": combined["mean_net_usd"],
                "combined_predicted_from_parts": predicted,
                "component_shifts": shifts,
                "reading": (
                    "negative interaction means the halves fight: the exit closes "
                    "before the entry's thesis plays out, or the entry picks paths "
                    "the exit was not fitted on"
                ),
            }

    ceilings = {}
    if base:
        for name, key in (("entry_ceiling", ("oracle", "clock")),
                          ("exit_ceiling", ("random", "oracle")),
                          ("joint_ceiling", ("oracle", "oracle"))):
            if key in grid:
                ceilings[name] = {
                    "cell_net_usd": grid[key]["mean_net_usd"],
                    "above_baseline_usd": round(
                        grid[key]["mean_net_usd"] - base["mean_net_usd"], 2),
                }

    payload = {
        "schema_version": "v5.factorial.v1",
        "phase": "0 (ceiling map, no fitting)" if args.phase0 else "full factorial",
        "pricing": (
            "entry at the ask, exit at the bid, fees only"
            if args.pricing == "bid" else "mid to mid, fees only, no spread"
        ),
        "clock_minutes": CLOCK_MINUTES,
        "max_hold_minutes": MAX_HOLD_MINUTES,
        "fees_usd": FEES_USD,
        "validation": (
            f"chronological walk-forward, {WARMUP_SESSIONS} warm-up sessions, "
            f"{FOLD_SESSIONS}-session folds"
        ),
        "trade_count_matched_to": match_count,
        "sessions": int(candidates["session"].nunique()),
        "oracle_note": (
            "oracles see the outcome and bound the prize; they are never a target. "
            "The exit oracle is a running maximum no non-anticipating rule reaches."
        ),
        "cells": [v for v in grid.values()],
        "ceilings": ceilings,
        "effects": effects,
        "not_a_validated_result": (
            "a positive here is a reason to freeze and run a known-answer "
            "campaign, not a candidate for promotion."
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\n{payload['phase']} | {payload['pricing']} | "
          f"{payload['sessions']} sessions | counts matched at {match_count}\n")
    head = (f"  {'entry':>8} {'exit':>8} {'n':>6} {'p':>7} {'W':>9} {'L':>9} "
            f"{'net$':>9} {'ci lo':>9} {'ci hi':>9} {'held':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for cell in grid.values():
        star = " *" if cell["clears_zero"] else ""
        print(f"  {cell['entry']:>8} {cell['exit']:>8} {cell['n']:>6,} "
              f"{100 * cell['p']:>6.1f}% {cell['W']:>9,.1f} {cell['L']:>9,.1f} "
              f"{cell['mean_net_usd']:>9,.1f} {cell['ci95_usd'][0]:>9,.1f} "
              f"{cell['ci95_usd'][1]:>9,.1f} {cell['mean_minutes_held']:>6.1f}{star}")

    if ceilings:
        print("\n  headroom above the baseline:")
        for name, got in ceilings.items():
            print(f"    {name:>15}: {got['above_baseline_usd']:>+9,.1f}")
    if effects:
        print(f"\n  entry effect {effects['entry_effect_usd']:>+8,.1f}"
              f"   exit effect {effects['exit_effect_usd']:>+8,.1f}"
              f"   INTERACTION {effects['interaction_usd']:>+8,.1f}")
        pred = effects["combined_predicted_from_parts"]
        print(f"  combined predicted from parts {pred['predicted_net_usd']:>+8,.1f}"
              f"  measured {effects['combined_measured_usd']:>+8,.1f}")
        moved = effects["component_shifts"]
        print(f"\n  what each half moved, against the baseline:")
        print(f"    {'':>10} {'p':>9} {'W':>9} {'L':>9}")
        for name in ("entry_moved", "exit_moved", "combined_moved"):
            got = moved[name]
            print(f"    {name.replace('_moved', ''):>10} "
                  f"{100 * got['p']:>+8.1f}pp {got['W']:>+9,.1f} {got['L']:>+9,.1f}")
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
