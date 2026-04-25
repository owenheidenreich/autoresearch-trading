"""Ensemble consensus filtering across the 5-seed champion stack.

Hypothesis: bars where multiple seeds independently arrive at the same
side are higher-confidence than single-seed picks. Filtering to K-of-5
side consensus should tighten the trade portfolio and lift PF, at the
cost of trade count.

For the SPX combined-fix champion (`spx_combined_3seed_001`, seeds
42-46), per-seed agg PF spans 1.65-2.20 and the 5-seed mean is 1.881 —
short of the (retired) V1+L3 floor. If consensus filtering on the
existing per-seed chosen_trades produces a portfolio with PF >= 2.0 at
trade count >= 150, the deployment recipe gains a "vote" filter without
any additional training.

Two execution modes are reported:

1. **Per-seed K-consensus**: keep seed S's trades only on bars where >=K
   of the 5 seeds (including S) agree on the same side as S. PF is
   computed on seed S's chosen_objective_pnl across the kept bars.
   This represents deploying ONE seed's model with a runtime gate that
   asks the other 4 seeds whether they would have agreed.

2. **Ensemble-mean K-consensus**: at each consensus bar (>=K seeds
   agreed on a side), the realized "trade" pnl is the mean of the K
   agreeing seeds' chosen_objective_pnl on that bar. PF is then
   computed on the resulting set of one trade per consensus bar. This
   represents deploying all 5 seeds in parallel and averaging fills.

Output JSON has both views per K and a comparison row for the 5-seed
naive mean (no consensus filter).

Usage:

    .venv/bin/python -m v3.analysis.ensemble_consensus \
        --champion-dir v3/artifacts \
        --champion-name spx_combined_3seed_001 \
        --seeds 42 43 44 45 46 \
        --out v3/artifacts/ensemble_consensus/spx_combined_3seed_001.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd


def _profit_factor(pnl: np.ndarray) -> float:
    """PF = sum(positive) / |sum(negative)|. Returns inf if no losers, 0 if no winners."""
    pnl = np.asarray(pnl, dtype=float)
    pos = pnl[pnl > 0].sum()
    neg = pnl[pnl < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def _max_drawdown_pct(pnl: np.ndarray, capital_base: float = 25_000.0) -> float:
    """Max drawdown as percent of running peak equity.

    Matches `replay_metrics_from_pnls` in v3/layer2/common.py: starts at
    capital_base, walks pnls, tracks peak, computes (peak-equity)/peak*100,
    returns max. Returns positive number (e.g. 12.62 for 12.62%).
    """
    if len(pnl) == 0:
        return 0.0
    equity = capital_base
    peak = capital_base
    max_dd = 0.0
    for p in pnl:
        equity += float(p)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return float(max_dd)


def _summarize(pnl: np.ndarray) -> dict[str, Any]:
    pnl = np.asarray(pnl, dtype=float)
    n_raw = len(pnl)
    finite_mask = np.isfinite(pnl)
    pnl = pnl[finite_mask]
    n = len(pnl)
    n_dropped = n_raw - n
    if n == 0:
        return {
            "n_trades": 0,
            "n_dropped_nan": int(n_dropped),
            "pf": 0.0,
            "mean_dd_pct": 0.0,
            "sum_pnl": 0.0,
            "mean_pnl": 0.0,
            "win_rate": 0.0,
        }
    return {
        "n_trades": int(n),
        "n_dropped_nan": int(n_dropped),
        "pf": _profit_factor(pnl),
        "mean_dd_pct": _max_drawdown_pct(pnl),
        "sum_pnl": float(pnl.sum()),
        "mean_pnl": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
    }


def _load_seed(champion_dir: str, name: str, seed: int) -> pd.DataFrame:
    path = os.path.join(
        champion_dir,
        f"layer2_unified_policy_{name}_seed{seed}",
        f"seed_{seed}",
        "chosen_trades.pkl",
    )
    with open(path, "rb") as f:
        df = pickle.load(f)
    df = df.copy()
    df["seed"] = seed
    df = df.sort_values(["day", "bar_index"]).reset_index(drop=True)
    return df


def _build_consensus_view(per_bar: pd.DataFrame) -> pd.DataFrame:
    """One row per (day, bar_index): consensus counts and side decision.

    Columns: day, bar_index, n_seeds, n_call, n_put, max_side, max_count
    """
    grp = per_bar.groupby(["day", "bar_index"])
    summary = grp.agg(
        n_seeds=("seed", "nunique"),
        n_call=("chosen_side", lambda s: int((s == "call").sum())),
        n_put=("chosen_side", lambda s: int((s == "put").sum())),
    ).reset_index()
    summary["max_count"] = summary[["n_call", "n_put"]].max(axis=1)
    summary["max_side"] = np.where(summary["n_call"] >= summary["n_put"], "call", "put")
    return summary


def per_seed_consensus(
    seed_dfs: dict[int, pd.DataFrame], consensus_summary: pd.DataFrame, K: int
) -> dict[int, dict[str, Any]]:
    """For each seed, restrict its trades to bars where >=K seeds agree on
    seed's chosen side."""
    # Build a (day, bar_index, side) -> count map for fast lookup
    cs = consensus_summary.set_index(["day", "bar_index"])
    out: dict[int, dict[str, Any]] = {}
    for seed, df in seed_dfs.items():
        keep_mask = []
        for _, row in df.iterrows():
            key = (row["day"], row["bar_index"])
            if key not in cs.index:
                keep_mask.append(False)
                continue
            agg = cs.loc[key]
            side = row["chosen_side"]
            count = agg["n_call"] if side == "call" else agg["n_put"]
            keep_mask.append(count >= K)
        kept = df[pd.Series(keep_mask, index=df.index)]
        out[seed] = _summarize(kept["chosen_objective_pnl"].values)
        out[seed]["n_call"] = int((kept["chosen_side"] == "call").sum())
        out[seed]["n_put"] = int((kept["chosen_side"] == "put").sum())
    return out


def ensemble_mean_consensus(
    per_bar: pd.DataFrame, consensus_summary: pd.DataFrame, K: int
) -> dict[str, Any]:
    """At each consensus bar (>=K seeds agreed on max_side), realized pnl =
    mean of those K agreeing seeds' chosen_objective_pnl on that bar.

    Per-bar pnl is the deployment-style realization for "vote then take
    that side". PF over the resulting one-trade-per-bar series.
    """
    qualifying = consensus_summary[consensus_summary["max_count"] >= K].copy()
    qualifying = qualifying.sort_values(["day", "bar_index"]).reset_index(drop=True)
    pnl_list: list[float] = []
    side_list: list[str] = []
    for _, row in qualifying.iterrows():
        day = row["day"]
        bar_idx = row["bar_index"]
        side = row["max_side"]
        agreeing = per_bar[
            (per_bar["day"] == day)
            & (per_bar["bar_index"] == bar_idx)
            & (per_bar["chosen_side"] == side)
        ]
        if len(agreeing) == 0:
            continue
        seed_pnls = agreeing["chosen_objective_pnl"].values.astype(float)
        finite = np.isfinite(seed_pnls)
        if not finite.any():
            continue
        pnl_list.append(float(np.mean(seed_pnls[finite])))
        side_list.append(side)
    pnl = np.asarray(pnl_list, dtype=float)
    summary = _summarize(pnl)
    summary["n_call"] = sum(1 for s in side_list if s == "call")
    summary["n_put"] = sum(1 for s in side_list if s == "put")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-dir", default="v3/artifacts")
    parser.add_argument("--champion-name", default="spx_combined_3seed_001")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--out",
        default="v3/artifacts/ensemble_consensus/spx_combined_3seed_001.json",
    )
    parser.add_argument("--capital-base", type=float, default=25_000.0)
    args = parser.parse_args()

    seed_dfs = {s: _load_seed(args.champion_dir, args.champion_name, s) for s in args.seeds}
    per_bar = pd.concat(seed_dfs.values(), ignore_index=True)
    print(f"Loaded {len(seed_dfs)} seeds, {len(per_bar)} total trade rows")

    consensus_summary = _build_consensus_view(per_bar)
    print(f"Unique (day, bar_index) tuples: {len(consensus_summary)}")
    for K in [1, 2, 3, 4, 5]:
        n_qualifying = (consensus_summary["max_count"] >= K).sum()
        print(f"  K={K}: {n_qualifying} bars where >={K} seeds agree on a side")

    # Baselines: per-seed unfiltered, and 5-seed mean
    per_seed_baseline = {
        s: _summarize(seed_dfs[s]["chosen_objective_pnl"].values) for s in args.seeds
    }
    pf_values = [per_seed_baseline[s]["pf"] for s in args.seeds]
    dd_values = [per_seed_baseline[s]["mean_dd_pct"] for s in args.seeds]
    five_seed_mean = {
        "mean_pf": float(np.mean(pf_values)),
        "min_pf": float(np.min(pf_values)),
        "mean_dd_pct": float(np.mean(dd_values)),
        "max_dd_pct": float(np.max(dd_values)),
        "mean_n_trades": float(np.mean([per_seed_baseline[s]["n_trades"] for s in args.seeds])),
    }

    # Consensus views
    per_seed_K = {}
    ensemble_K = {}
    for K in [2, 3, 4, 5]:
        per_seed_K[K] = per_seed_consensus(seed_dfs, consensus_summary, K)
        ensemble_K[K] = ensemble_mean_consensus(per_bar, consensus_summary, K)

    # Aggregate per_seed_K results into mean/min PF for direct comparison
    per_seed_K_agg: dict[int, dict[str, float]] = {}
    for K, seed_results in per_seed_K.items():
        pfs = [seed_results[s]["pf"] for s in args.seeds]
        dds = [seed_results[s]["mean_dd_pct"] for s in args.seeds]
        ns = [seed_results[s]["n_trades"] for s in args.seeds]
        per_seed_K_agg[K] = {
            "mean_pf": float(np.mean(pfs)),
            "min_pf": float(np.min(pfs)),
            "mean_dd_pct": float(np.mean(dds)),
            "max_dd_pct": float(np.max(dds)),
            "mean_n_trades": float(np.mean(ns)),
            "min_n_trades": int(np.min(ns)),
        }

    out = {
        "champion_name": args.champion_name,
        "seeds": args.seeds,
        "capital_base": args.capital_base,
        "n_unique_bars": int(len(consensus_summary)),
        "qualifying_bars_by_K": {
            int(K): int((consensus_summary["max_count"] >= K).sum())
            for K in [1, 2, 3, 4, 5]
        },
        "per_seed_baseline": per_seed_baseline,
        "five_seed_mean_baseline": five_seed_mean,
        "per_seed_K_consensus": {
            str(K): {str(s): per_seed_K[K][s] for s in args.seeds} for K in [2, 3, 4, 5]
        },
        "per_seed_K_aggregated": {str(K): per_seed_K_agg[K] for K in [2, 3, 4, 5]},
        "ensemble_mean_K_consensus": {str(K): ensemble_K[K] for K in [2, 3, 4, 5]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")

    print("\n=== Five-seed mean baseline (no consensus filter) ===")
    print(f"  mean PF: {five_seed_mean['mean_pf']:.3f}")
    print(f"  min PF:  {five_seed_mean['min_pf']:.3f}")
    print(f"  mean DD: {five_seed_mean['mean_dd_pct']:.2f}%")
    print(f"  mean trades/seed: {five_seed_mean['mean_n_trades']:.0f}")

    print("\n=== Per-seed K-consensus (5-seed aggregated) ===")
    print(f"{'K':>3} {'mean_pf':>10} {'min_pf':>10} {'mean_dd':>10} {'max_dd':>10} {'mean_n':>10} {'min_n':>10}")
    for K in [2, 3, 4, 5]:
        a = per_seed_K_agg[K]
        print(
            f"{K:>3} {a['mean_pf']:>10.3f} {a['min_pf']:>10.3f} "
            f"{a['mean_dd_pct']:>10.2f} {a['max_dd_pct']:>10.2f} "
            f"{a['mean_n_trades']:>10.1f} {a['min_n_trades']:>10}"
        )

    print("\n=== Ensemble-mean K-consensus (one trade per consensus bar) ===")
    print(f"{'K':>3} {'pf':>10} {'dd_pct':>10} {'n_trades':>10} {'mean_pnl':>10} {'win_rate':>10}")
    for K in [2, 3, 4, 5]:
        e = ensemble_K[K]
        print(
            f"{K:>3} {e['pf']:>10.3f} {e['mean_dd_pct']:>10.2f} "
            f"{e['n_trades']:>10} {e['mean_pnl']:>10.2f} {e['win_rate']:>10.3f}"
        )


if __name__ == "__main__":
    main()
