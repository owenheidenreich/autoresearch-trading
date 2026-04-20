"""V2-pruned-gated integer-contract realism audit.

Takes the frozen V2-pruned-gated trade stream at the accepted deployment
floor `f = 0.25%` and translates it into discrete SPX contracts across a
grid of account sizes. No model, no gate, no features, no sizing
fraction, no exits changed. The simulator is additive-dollar:

  entry_cost_per_contract = entry_mid * 100
  contracts = floor((equity * f) / entry_cost_per_contract)
  if contracts == 0: skip (unaffordable)
  equity += contracts * net_pnl_dollars_per_contract

The per-contract `net_pnl_dollars` already accounts for spread cost via
the adaptive spread model used everywhere else in the baseline; no new
fill model.

Account grid: $100k, $250k, $500k, $1.0M, $2.0M.

Reports per account:
- participation rate (trades that afford >= 1 contract)
- average contracts per executed trade
- total executed, skipped-due-to-affordability
- CAGR, max DD, Calmar, worst 20-trade DD
- final equity

Min-viable diagnostics:
- account sizes needed for cheapest / p25 / median / p75 / p90 trade
- account sizes needed for 25% / 50% / 75% participation

Reads: v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_gated/trades.csv
(V2-pruned-gated commit, 2026-04-20). Writes: summary.json + per-account
CSVs under v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_gated_integer_contract/.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from v2.analysis.mechanical_baseline_opening_reversion import write_json, write_csv


# ---------------------------------------------------------------------------
# Locked parameters
# ---------------------------------------------------------------------------

SIZING_FRACTION = 0.0025                  # 0.25% — accepted deployment floor
SPX_MULTIPLIER = 100                      # $100 per point
ACCOUNT_GRID = (100_000, 250_000, 500_000, 1_000_000, 2_000_000)
TRADING_DAYS_PER_YEAR = 252
TEST_CALENDAR_DAYS = 300                   # 5 folds × 60 test days
TRADES_SOURCE = "v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_gated/trades.csv"
OUT_DIR_DEFAULT = "v2/artifacts/mechanical_baseline_opening_reversion_v2_pruned_gated_integer_contract"


# ---------------------------------------------------------------------------
# Source trade stream
# ---------------------------------------------------------------------------

@dataclass
class BaselineTradeLite:
    date: str
    fold: int
    bar_entry: int
    side: str
    entry_mid: float
    net_pct: float
    net_pnl_dollars: float                # per-contract net PnL (with spread cost)


def load_strategy_trades(path: str) -> list[BaselineTradeLite]:
    trades: list[BaselineTradeLite] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("strategy_label", "") != "strategy":
                continue
            trades.append(BaselineTradeLite(
                date=row["date"],
                fold=int(row["fold"]),
                bar_entry=int(row["bar_entry"]),
                side=row["side"],
                entry_mid=float(row["entry_mid"]),
                net_pct=float(row["net_pct"]),
                net_pnl_dollars=float(row["net_pnl_dollars"]),
            ))
    trades.sort(key=lambda t: (t.date, t.bar_entry))
    return trades


# ---------------------------------------------------------------------------
# Integer-contract simulator
# ---------------------------------------------------------------------------

@dataclass
class ExecutedTrade:
    date: str
    fold: int
    side: str
    entry_mid: float
    entry_cost_per_contract: float         # entry_mid * SPX_MULTIPLIER
    contracts: int
    equity_before: float
    net_pnl_dollars_per_contract: float
    trade_pnl_dollars: float
    equity_after: float


@dataclass
class SkippedTrade:
    date: str
    fold: int
    side: str
    entry_mid: float
    entry_cost_per_contract: float
    equity_at_bar: float
    required_equity_for_1_contract: float


def simulate_integer_contract(
    trades: list[BaselineTradeLite],
    account_size: float,
    f: float = SIZING_FRACTION,
) -> tuple[list[ExecutedTrade], list[SkippedTrade], np.ndarray]:
    equity = float(account_size)
    equity_curve = [equity]
    executed: list[ExecutedTrade] = []
    skipped: list[SkippedTrade] = []
    for t in trades:
        entry_cost = t.entry_mid * SPX_MULTIPLIER
        budget = equity * f
        contracts = int(np.floor(budget / entry_cost)) if entry_cost > 0 else 0
        if contracts == 0:
            required_eq = entry_cost / f
            skipped.append(SkippedTrade(
                date=t.date, fold=t.fold, side=t.side,
                entry_mid=t.entry_mid, entry_cost_per_contract=entry_cost,
                equity_at_bar=equity, required_equity_for_1_contract=required_eq,
            ))
            equity_curve.append(equity)           # equity unchanged
            continue
        trade_pnl = contracts * t.net_pnl_dollars
        equity_before = equity
        equity += trade_pnl
        executed.append(ExecutedTrade(
            date=t.date, fold=t.fold, side=t.side,
            entry_mid=t.entry_mid, entry_cost_per_contract=entry_cost,
            contracts=contracts, equity_before=equity_before,
            net_pnl_dollars_per_contract=t.net_pnl_dollars,
            trade_pnl_dollars=trade_pnl, equity_after=equity,
        ))
        equity_curve.append(equity)
    return executed, skipped, np.asarray(equity_curve, dtype=np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(equity)
    return np.where(running_max > 0, (running_max - equity) / running_max, 0.0)


def _worst_20_trade_dd(equity: np.ndarray, window: int = 20) -> float:
    if len(equity) <= window:
        return float(_drawdown_series(equity).max() if len(equity) else 0.0)
    worst = 0.0
    for start in range(0, len(equity) - window):
        w = equity[start:start + window + 1]
        rm = np.maximum.accumulate(w)
        wdd = np.where(rm > 0, (rm - w) / rm, 0.0)
        worst = max(worst, float(wdd.max()))
    return worst


def account_metrics(
    *,
    equity_curve: np.ndarray,
    executed: list[ExecutedTrade],
    skipped: list[SkippedTrade],
    account_size: float,
    n_baseline_trades: int,
    n_calendar_days: int,
) -> dict[str, Any]:
    final = float(equity_curve[-1])
    total_return = (final - account_size) / account_size
    years = max(n_calendar_days / TRADING_DAYS_PER_YEAR, 1e-6)
    if final > 0 and account_size > 0:
        cagr = (final / account_size) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0
    dd = _drawdown_series(equity_curve)
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    ulcer = float(np.sqrt(np.mean(dd ** 2))) if len(dd) > 0 else 0.0
    if max_dd > 1e-9:
        calmar = cagr / max_dd
    else:
        calmar = float("inf") if cagr > 0 else 0.0
    worst20 = _worst_20_trade_dd(equity_curve)
    n_executed = len(executed)
    participation = n_executed / max(n_baseline_trades, 1)
    contracts_arr = np.array([t.contracts for t in executed], dtype=np.float64)
    return {
        "account_size": account_size,
        "n_baseline_trades": n_baseline_trades,
        "n_executed": n_executed,
        "n_skipped_affordability": len(skipped),
        "participation_rate": participation,
        "avg_contracts_per_executed": float(contracts_arr.mean()) if len(contracts_arr) else 0.0,
        "max_contracts_per_trade": int(contracts_arr.max()) if len(contracts_arr) else 0,
        "median_contracts_per_trade": float(np.median(contracts_arr)) if len(contracts_arr) else 0.0,
        "final_equity": final,
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "ulcer_index": ulcer,
        "calmar": calmar,
        "worst_20_trade_dd": worst20,
    }


# ---------------------------------------------------------------------------
# Min-viable-account diagnostics
# ---------------------------------------------------------------------------

def min_viable_diagnostics(trades: list[BaselineTradeLite]) -> dict[str, Any]:
    entry_costs = np.array([t.entry_mid * SPX_MULTIPLIER for t in trades], dtype=np.float64)
    sorted_costs = np.sort(entry_costs)
    percentiles_pct = [0, 10, 25, 50, 75, 90, 95, 100]
    cost_pct = {f"p{p}": float(np.percentile(sorted_costs, p)) for p in percentiles_pct}
    min_account = {k: v / SIZING_FRACTION for k, v in cost_pct.items()}

    # Account sizes needed for participation targets
    # Static estimate using starting equity only (dynamic drifts from
    # simulation will be tiny at f = 0.25% and small DDs)
    participation_targets = [0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
    # If we want fraction p of trades to be affordable, we need account_size
    # such that the p-th percentile of entry_costs <= account_size * f.
    # i.e., account_size = percentile(entry_costs, p * 100) / f
    part_accounts = {}
    for p in participation_targets:
        # fraction p of trades have cost <= percentile(costs, p*100)
        # We need account_size * f >= that percentile cost
        p_cost = float(np.percentile(sorted_costs, p * 100))
        part_accounts[f"{int(p * 100)}%"] = p_cost / SIZING_FRACTION
    return {
        "entry_cost_per_contract": cost_pct,
        "min_account_to_afford_that_percentile_trade": min_account,
        "account_size_for_static_participation_rate": part_accounts,
        "n_baseline_trades": len(trades),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=TRADES_SOURCE)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[int-contract] Sizing fraction: {SIZING_FRACTION:.4f} (0.25% floor)")
    print(f"[int-contract] SPX multiplier: {SPX_MULTIPLIER}")
    print(f"[int-contract] Account grid: {[f'${a:,}' for a in ACCOUNT_GRID]}")
    print(f"[int-contract] Loading trades from: {args.trades}")

    trades = load_strategy_trades(args.trades)
    print(f"[int-contract] Loaded {len(trades)} strategy trades (sorted chronologically)")

    # Sanity: reproduce user's back-of-envelope numbers
    entry_costs = np.array([t.entry_mid * SPX_MULTIPLIER for t in trades])
    print(f"\n=== Entry-cost distribution (per 1 contract) ===")
    for p in [0, 10, 25, 50, 75, 90, 95, 100]:
        print(f"  p{p:>3d}: ${float(np.percentile(entry_costs, p)):>10,.2f}")
    print(f"  mean: ${entry_costs.mean():>10,.2f}")
    print(f"  std:  ${entry_costs.std():>10,.2f}")

    diag = min_viable_diagnostics(trades)
    print(f"\n=== Min-viable account size to afford each percentile trade "
          f"(static, at f={SIZING_FRACTION:.4f}) ===")
    for k, v in diag["min_account_to_afford_that_percentile_trade"].items():
        print(f"  {k:<5}: ${v:>12,.0f}")
    print(f"\n=== Static account size for participation target "
          f"(exactly that percentile of trades affordable) ===")
    for k, v in diag["account_size_for_static_participation_rate"].items():
        print(f"  {k:<6}: ${v:>12,.0f}")

    # Full per-account simulation
    os.makedirs(args.out_dir, exist_ok=True)
    per_account: list[dict] = []
    print(f"\n=== Per-account integer-contract simulation ===")
    print(f"{'account':>12} {'exec':>5} {'skip':>5} {'part%':>7} "
          f"{'avg_ct':>8} {'max_ct':>7} "
          f"{'total_ret':>10} {'cagr':>9} {'max_dd':>9} {'calmar':>8} "
          f"{'worst20':>9} {'final_eq':>14}")
    for acct in ACCOUNT_GRID:
        executed, skipped, equity_curve = simulate_integer_contract(
            trades, account_size=acct, f=SIZING_FRACTION,
        )
        m = account_metrics(
            equity_curve=equity_curve,
            executed=executed, skipped=skipped,
            account_size=acct, n_baseline_trades=len(trades),
            n_calendar_days=TEST_CALENDAR_DAYS,
        )
        per_account.append({
            "account_size": acct,
            "metrics": m,
            "executed": [asdict(t) for t in executed],
            "skipped": [asdict(t) for t in skipped],
        })
        print(f"  ${acct:>10,} {m['n_executed']:>5d} {m['n_skipped_affordability']:>5d} "
              f"{m['participation_rate']*100:>6.1f}% {m['avg_contracts_per_executed']:>8.2f} "
              f"{m['max_contracts_per_trade']:>7d} "
              f"{m['total_return']:>+10.4f} {m['cagr']:>+9.4f} {m['max_dd']:>9.4f} "
              f"{m['calmar']:>+8.2f} {m['worst_20_trade_dd']:>9.4f} "
              f"${m['final_equity']:>12,.0f}")

        # Dump per-account executed trades
        if executed:
            write_csv(
                os.path.join(args.out_dir, f"executed_${acct:,}.csv".replace(",", "_")),
                executed,
                list(ExecutedTrade.__dataclass_fields__.keys()),
            )

    # Answer to the question: what account size hits "meaningful participation"?
    # Report first account size >= 50% participation and >= 75%
    print(f"\n=== Decision-question summary ===")
    first_50 = None
    first_75 = None
    first_any = None
    for row in per_account:
        pr = row["metrics"]["participation_rate"]
        if first_any is None and row["metrics"]["n_executed"] > 0:
            first_any = row["account_size"]
        if first_50 is None and pr >= 0.50:
            first_50 = row["account_size"]
        if first_75 is None and pr >= 0.75:
            first_75 = row["account_size"]
    print(f"  smallest account in grid with any execution: "
          f"{'$' + format(first_any, ',') if first_any else 'none'}")
    print(f"  smallest account in grid with >=50% participation: "
          f"{'$' + format(first_50, ',') if first_50 else 'none'}")
    print(f"  smallest account in grid with >=75% participation: "
          f"{'$' + format(first_75, ',') if first_75 else 'none'}")

    # Dump summary
    write_json(os.path.join(args.out_dir, "summary.json"), {
        "experiment_id": "mechbase_opening_reversion_v2_pruned_gated_integer_contract",
        "sizing_fraction": SIZING_FRACTION,
        "spx_multiplier": SPX_MULTIPLIER,
        "account_grid_dollars": list(ACCOUNT_GRID),
        "n_baseline_trades": len(trades),
        "test_calendar_days": TEST_CALENDAR_DAYS,
        "entry_cost_distribution": {
            f"p{p}": float(np.percentile(entry_costs, p))
            for p in [0, 10, 25, 50, 75, 90, 95, 100]
        },
        "entry_cost_mean": float(entry_costs.mean()),
        "entry_cost_std": float(entry_costs.std()),
        "min_viable_diagnostics": diag,
        "per_account": [
            {"account_size": r["account_size"], "metrics": r["metrics"]}
            for r in per_account
        ],
        "decision_summary": {
            "smallest_account_any_execution": first_any,
            "smallest_account_50pct_participation": first_50,
            "smallest_account_75pct_participation": first_75,
        },
        "source_trades": args.trades,
        "note": "Additive-dollar integer-contract simulation. Each trade's "
                "per-contract net_pnl_dollars (including spread cost) is "
                "scaled by floor((equity*f)/(entry_mid*100)) integer contracts. "
                "Unaffordable trades (contracts==0) are skipped, equity unchanged.",
    })
    print(f"\nWrote summary + per-account executed CSVs to {args.out_dir}")
    print(f"Total elapsed: {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
