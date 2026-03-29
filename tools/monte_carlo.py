#!/usr/bin/env python3
"""Monte Carlo strategy stress-testing.

Addresses hindsight bias in backtesting: "backtests have advantage of hindsight,
where something you would have never taken in the moment actually worked out on paper.
Rather than back-testing PnL, configure for win-rate with considerable margin of error
to emulate that human factor." — Pickles

Usage:
    python3 tools/monte_carlo.py --trades results/art2/cycle-008/replay/backtest_trades.csv
    python3 tools/monte_carlo.py --trades <path> --simulations 10000 --hindsight-remove 5,10,15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Core Monte Carlo Engine
# ---------------------------------------------------------------------------

def load_trades(path: str) -> np.ndarray:
    """Load per-trade P&L from backtest_trades.csv. Returns array of pnl_pct values."""
    import csv
    pnls = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("pnl_pct", "")
            if val == "" or val is None:
                continue
            pnls.append(float(val))
    if not pnls:
        print(f"ERROR: No trades found in {path}", file=sys.stderr)
        sys.exit(1)
    return np.array(pnls, dtype=np.float64)


def _equity_curve(pnls: np.ndarray) -> np.ndarray:
    """Build equity curve from pnl_pct array using ADDITIVE returns.
    Each trade risks a fixed fraction, not compounding. This matches
    how the replay actually sizes positions (fixed % of account).
    Starts at 100. Each trade adds pnl_pct * position_risk (5% of equity)."""
    RISK_PER_TRADE = 0.05  # matches POSITION_RISK_TARGET in replay
    equity = np.empty(len(pnls) + 1)
    equity[0] = 100.0
    for i, p in enumerate(pnls):
        # Dollar P&L = equity * risk_fraction * return_on_risk
        dollar_pnl = equity[i] * RISK_PER_TRADE * (p / 100.0)
        equity[i + 1] = max(equity[i] + dollar_pnl, 0.0)
    return equity


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Max drawdown as percentage of peak."""
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd) * 100.0)


def _max_consecutive_losses(pnls: np.ndarray) -> int:
    """Longest streak of consecutive losing trades."""
    max_streak = 0
    current = 0
    for p in pnls:
        if p < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _profit_factor(pnls: np.ndarray) -> float:
    """Gross profit / gross loss."""
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses < 1e-10:
        return 999.0
    return float(wins / losses)


def _basic_metrics(pnls: np.ndarray) -> dict:
    """Compute basic metrics from a P&L array."""
    eq = _equity_curve(pnls)
    n = len(pnls)
    wins = np.sum(pnls > 0)
    return {
        "n_trades": n,
        "win_rate": float(wins / n) if n > 0 else 0.0,
        "profit_factor": _profit_factor(pnls),
        "total_return_pct": float(eq[-1] - 100.0),
        "max_drawdown_pct": _max_drawdown_pct(eq),
        "avg_winner_pct": float(pnls[pnls > 0].mean()) if wins > 0 else 0.0,
        "avg_loser_pct": float(pnls[pnls < 0].mean()) if (n - wins) > 0 else 0.0,
        "max_consecutive_losses": _max_consecutive_losses(pnls),
    }


# ---------------------------------------------------------------------------
# Simulation 1: Bootstrap Resampling
# ---------------------------------------------------------------------------

def bootstrap_resample(pnls: np.ndarray, n_sims: int = 10000, seed: int = 42) -> dict:
    """Resample trades with replacement to build confidence intervals."""
    rng = np.random.default_rng(seed)
    n = len(pnls)

    returns = np.empty(n_sims)
    drawdowns = np.empty(n_sims)
    win_rates = np.empty(n_sims)
    consec_losses = np.empty(n_sims, dtype=int)

    for i in range(n_sims):
        sample = rng.choice(pnls, size=n, replace=True)
        eq = _equity_curve(sample)
        returns[i] = eq[-1] - 100.0
        drawdowns[i] = _max_drawdown_pct(eq)
        win_rates[i] = np.sum(sample > 0) / n
        consec_losses[i] = _max_consecutive_losses(sample)

    ruin_count = np.sum(returns <= -95.0)  # lost 95%+ = ruin

    return {
        "median_return_pct": float(np.median(returns)),
        "p5_return_pct": float(np.percentile(returns, 5)),
        "p25_return_pct": float(np.percentile(returns, 25)),
        "p75_return_pct": float(np.percentile(returns, 75)),
        "p95_return_pct": float(np.percentile(returns, 95)),
        "median_max_drawdown_pct": float(np.median(drawdowns)),
        "p95_max_drawdown_pct": float(np.percentile(drawdowns, 95)),
        "median_win_rate": float(np.median(win_rates)),
        "p5_win_rate": float(np.percentile(win_rates, 5)),
        "ruin_probability": float(ruin_count / n_sims),
        "median_consec_losses": int(np.median(consec_losses)),
        "p95_consec_losses": int(np.percentile(consec_losses, 95)),
        "p99_consec_losses": int(np.percentile(consec_losses, 99)),
    }


# ---------------------------------------------------------------------------
# Simulation 2: Hindsight Removal
# ---------------------------------------------------------------------------

def hindsight_removal(pnls: np.ndarray, remove_pcts: list[int],
                      n_sims: int = 10000, seed: int = 42) -> dict:
    """Remove top X% of winners and re-run bootstrap.
    Models: 'Would this strategy survive if the best trades were hindsight bias?'"""
    results = {}
    sorted_pnls = np.sort(pnls)  # ascending

    for pct in remove_pcts:
        n_remove = max(1, int(len(pnls) * pct / 100.0))
        # Remove the top N winners (highest P&L trades)
        trimmed = sorted_pnls[:-n_remove]
        metrics = _basic_metrics(trimmed)
        boot = bootstrap_resample(trimmed, n_sims=n_sims, seed=seed)

        results[f"top_{pct}pct_removed"] = {
            "trades_removed": n_remove,
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "total_return_pct": metrics["total_return_pct"],
            "median_return_pct": boot["median_return_pct"],
            "p5_return_pct": boot["p5_return_pct"],
            "ruin_probability": boot["ruin_probability"],
            "still_profitable": metrics["total_return_pct"] > 0,
            "bootstrap_profitable": boot["median_return_pct"] > 0,
        }

    return results


# ---------------------------------------------------------------------------
# Simulation 3: Win Rate Sensitivity
# ---------------------------------------------------------------------------

def win_rate_sensitivity(pnls: np.ndarray, degrade_pcts: list[int],
                         n_sims: int = 10000, seed: int = 42) -> dict:
    """Degrade win rate by converting random winners to losers.
    Models: 'At what win rate does this strategy break?'"""
    rng = np.random.default_rng(seed)
    results = {}
    actual_wr = np.sum(pnls > 0) / len(pnls)

    for degrade in degrade_pcts:
        target_wr = actual_wr - (degrade / 100.0)
        if target_wr <= 0:
            results[f"at_{int(target_wr*100)}pct"] = {
                "target_win_rate": target_wr,
                "median_return_pct": -100.0,
                "ruin_probability": 1.0,
                "broken": True,
            }
            continue

        # For each sim, randomly flip some winners to losers
        winners_idx = np.where(pnls > 0)[0]
        n_to_flip = int(len(pnls) * (degrade / 100.0))

        sim_returns = np.empty(n_sims)
        for i in range(n_sims):
            degraded = pnls.copy()
            if n_to_flip > 0 and len(winners_idx) > 0:
                flip_idx = rng.choice(winners_idx, size=min(n_to_flip, len(winners_idx)), replace=False)
                # Convert winners to average loser
                avg_loss = pnls[pnls < 0].mean() if np.any(pnls < 0) else -5.0
                degraded[flip_idx] = avg_loss
            sample = rng.choice(degraded, size=len(degraded), replace=True)
            eq = _equity_curve(sample)
            sim_returns[i] = eq[-1] - 100.0

        ruin_count = np.sum(sim_returns <= -95.0)
        results[f"degraded_{degrade}pct"] = {
            "target_win_rate": round(target_wr, 4),
            "median_return_pct": float(np.median(sim_returns)),
            "p5_return_pct": float(np.percentile(sim_returns, 5)),
            "ruin_probability": float(ruin_count / n_sims),
            "broken": float(np.median(sim_returns)) <= 0,
        }

    # Find breakeven win rate via binary search
    lo, hi = 0.0, actual_wr
    for _ in range(20):
        mid = (lo + hi) / 2.0
        degrade_amount = actual_wr - mid
        n_to_flip = int(len(pnls) * degrade_amount)
        winners_idx_local = np.where(pnls > 0)[0]

        test_returns = np.empty(1000)
        for i in range(1000):
            degraded = pnls.copy()
            if n_to_flip > 0 and len(winners_idx_local) > 0:
                flip_idx = rng.choice(winners_idx_local, size=min(n_to_flip, len(winners_idx_local)), replace=False)
                avg_loss = pnls[pnls < 0].mean() if np.any(pnls < 0) else -5.0
                degraded[flip_idx] = avg_loss
            sample = rng.choice(degraded, size=len(degraded), replace=True)
            eq = _equity_curve(sample)
            test_returns[i] = eq[-1] - 100.0

        if np.median(test_returns) > 0:
            hi = mid
        else:
            lo = mid

    results["breakeven_win_rate"] = round((lo + hi) / 2.0, 4)

    return results


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def compute_verdict(actual: dict, bootstrap: dict, hindsight: dict, sensitivity: dict) -> tuple[str, str]:
    """Determine overall robustness verdict."""
    reasons = []

    # Check hindsight dependency
    top10 = hindsight.get("top_10pct_removed", {})
    if top10 and not top10.get("bootstrap_profitable", True):
        return "HINDSIGHT_DEPENDENT", (
            f"Removing top 10% of winners makes strategy unprofitable "
            f"(median return {top10.get('median_return_pct', 0):.1f}%). "
            f"Strategy depends on outlier trades that may be hindsight bias."
        )

    top5 = hindsight.get("top_5pct_removed", {})
    if top5 and not top5.get("bootstrap_profitable", True):
        return "HINDSIGHT_DEPENDENT", (
            f"Removing top 5% of winners makes strategy unprofitable "
            f"(median return {top5.get('median_return_pct', 0):.1f}%). "
            f"Strategy heavily depends on a few big winners."
        )

    # Check ruin probability
    if bootstrap.get("ruin_probability", 0) > 0.01:
        reasons.append(f"ruin probability {bootstrap['ruin_probability']:.1%}")

    # Check drawdown severity
    if bootstrap.get("p95_max_drawdown_pct", 0) > 80:
        reasons.append(f"95th percentile drawdown {bootstrap['p95_max_drawdown_pct']:.1f}%")

    # Check win rate sensitivity
    breakeven = sensitivity.get("breakeven_win_rate", 0)
    actual_wr = actual.get("win_rate", 0)
    margin = actual_wr - breakeven
    if margin < 0.03:
        reasons.append(f"only {margin:.1%} win rate margin above breakeven ({breakeven:.1%})")

    if reasons:
        return "FRAGILE", "Strategy is profitable but fragile: " + "; ".join(reasons) + "."

    return "ROBUST", (
        f"Strategy survives hindsight removal (top 10% winners removed: still profitable), "
        f"has {margin:.1%} win rate margin above breakeven ({breakeven:.1%}), "
        f"and ruin probability is {bootstrap.get('ruin_probability', 0):.2%}."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(trades_path: str, n_sims: int = 10000,
                 hindsight_pcts: Optional[list] = None,
                 degrade_pcts: Optional[list] = None) -> dict:
    """Run full Monte Carlo analysis. Returns structured results dict."""
    if hindsight_pcts is None:
        hindsight_pcts = [5, 10, 15]
    if degrade_pcts is None:
        degrade_pcts = [5, 10, 15]

    pnls = load_trades(trades_path)
    actual = _basic_metrics(pnls)

    print(f"Loaded {len(pnls)} trades from {trades_path}")
    print(f"  Win rate: {actual['win_rate']:.1%}, PF: {actual['profit_factor']:.2f}, "
          f"Return: {actual['total_return_pct']:+.1f}%")
    print(f"  Max DD: {actual['max_drawdown_pct']:.1f}%, "
          f"Max consec losses: {actual['max_consecutive_losses']}")
    print()

    print(f"Running {n_sims:,} bootstrap simulations...")
    boot = bootstrap_resample(pnls, n_sims=n_sims)
    print(f"  Median return: {boot['median_return_pct']:+.1f}%  "
          f"[5th: {boot['p5_return_pct']:+.1f}%, 95th: {boot['p95_return_pct']:+.1f}%]")
    print(f"  Median max DD: {boot['median_max_drawdown_pct']:.1f}%  "
          f"[95th: {boot['p95_max_drawdown_pct']:.1f}%]")
    print(f"  Ruin probability: {boot['ruin_probability']:.2%}")
    print(f"  Consec losses [median: {boot['median_consec_losses']}, "
          f"95th: {boot['p95_consec_losses']}, 99th: {boot['p99_consec_losses']}]")
    print()

    print(f"Hindsight removal (top {hindsight_pcts}% winners removed)...")
    hs = hindsight_removal(pnls, hindsight_pcts, n_sims=n_sims)
    for key, val in hs.items():
        status = "PROFITABLE" if val.get("bootstrap_profitable") else "UNPROFITABLE"
        print(f"  {key}: WR {val.get('win_rate', 0):.1%}, "
              f"PF {val.get('profit_factor', 0):.2f}, "
              f"median return {val.get('median_return_pct', 0):+.1f}% — {status}")
    print()

    print(f"Win rate sensitivity (degrade by {degrade_pcts}%)...")
    sens = win_rate_sensitivity(pnls, degrade_pcts, n_sims=n_sims)
    for key, val in sens.items():
        if key == "breakeven_win_rate":
            print(f"  Breakeven win rate: {val:.1%}")
        else:
            status = "BROKEN" if val.get("broken") else "OK"
            print(f"  {key}: target WR {val.get('target_win_rate', 0):.1%}, "
                  f"median return {val.get('median_return_pct', 0):+.1f}%, "
                  f"ruin {val.get('ruin_probability', 0):.1%} — {status}")
    print()

    verdict, reason = compute_verdict(actual, boot, hs, sens)
    print(f"VERDICT: {verdict}")
    print(f"  {reason}")

    return {
        "n_trades": len(pnls),
        "actual_metrics": actual,
        "bootstrap": boot,
        "hindsight_removal": hs,
        "win_rate_sensitivity": sens,
        "verdict": verdict,
        "verdict_reason": reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo strategy stress-testing")
    parser.add_argument("--trades", required=True, help="Path to backtest_trades.csv")
    parser.add_argument("--simulations", type=int, default=10000, help="Number of simulations")
    parser.add_argument("--hindsight-remove", type=str, default="5,10,15",
                        help="Comma-separated percentages of top winners to remove")
    parser.add_argument("--win-rate-degrade", type=str, default="5,10,15",
                        help="Comma-separated percentages to degrade win rate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    hs_pcts = [int(x) for x in args.hindsight_remove.split(",")]
    wr_pcts = [int(x) for x in args.win_rate_degrade.split(",")]

    results = run_analysis(args.trades, n_sims=args.simulations,
                           hindsight_pcts=hs_pcts, degrade_pcts=wr_pcts)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        # Print JSON to stdout
        print(f"\n{'='*60}")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
