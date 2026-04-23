"""Layer-2 slippage / spread stress test (Check 3 from the reality-checks plan).

Question: does the Layer-2 PF (1.455) survive realistic SPX 0DTE
execution costs? The current fill model in v3/oracles/exit_headroom.py
uses a static half-spread on both legs plus $1 commission. It ignores
the 2-5 bps of realistic late-day slippage on 0DTE exits (spreads widen
3-8x in the final 15 minutes; mid may not be fillable).

Approach: apply an additional fixed-dollar-per-round-trip slippage
charge to each trade's PnL, report PF/DD/TPD sensitivity across a
grid. Same grid applied to the teacher baseline (from teacher_baseline_
trades.csv) so the comparison stays honest — what matters for
deployment is not absolute PF but the EDGE OVER THE BASELINE after
costs.

Stress grid (additional dollars per round trip):
  $0    — optimistic baseline (matches replay.py today)
  $10   — mild slippage, typical for liquid 0DTE in the body of the day
  $25   — moderate, typical for late-day exits under normal vol
  $50   — heavy, vol-event or illiquid contract at exit

Disqualifying signal (per plan §2c):
- PF drops below 1.0 at the $25 stress level → edge is vulnerable to
  realistic execution costs; needs exit timing work (Layer-3).
- PF stays above teacher baseline across the grid → edge survives
  realistic cost; acceptable for paper trading.

Run:
    python -m v3.analysis.layer2_slippage_stress \
        --run-dir v3/artifacts/layer2_shared_enc_fixedq_detach
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from v3.layer2.common import replay_metrics_from_pnls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-2 slippage / spread stress test.")
    p.add_argument("--run-dir", default="v3/artifacts/layer2_shared_enc_fixedq_detach")
    p.add_argument("--equity", type=float, default=25_000.0)
    return p.parse_args()


def _load_trades(run_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _stress(trades: pd.DataFrame, equity: float, slip_dollars: float) -> dict:
    """Apply fixed-dollar additional slippage per trade; recompute PF/DD/TPD."""
    if trades.empty:
        return {"pf": float("nan"), "max_dd_pct": float("nan"), "trades": 0,
                "trades_per_day": 0.0, "mean_pnl": 0.0, "net_pnl": 0.0}
    adj = trades["pnl"].astype(float) - float(slip_dollars)
    pnls = adj.tolist()
    m = replay_metrics_from_pnls(pnls, equity)
    n_days = trades["day"].nunique() if "day" in trades.columns else len(trades)
    m["trades"] = len(pnls)
    m["trades_per_day"] = float(len(pnls) / max(n_days, 1))
    m["mean_pnl"] = float(adj.mean())
    m["net_pnl"] = float(adj.sum())
    return m


def main() -> int:
    args = parse_args()
    layer2 = _load_trades(args.run_dir, "layer2_trades.csv")
    teacher = _load_trades(args.run_dir, "teacher_baseline_trades.csv")
    print(f"Loaded {len(layer2)} Layer-2 trades, {len(teacher)} teacher baseline trades")

    grid = [0.0, 10.0, 25.0, 50.0]
    rows = []
    for slip in grid:
        l = _stress(layer2, args.equity, slip)
        t = _stress(teacher, args.equity, slip)
        rows.append({
            "slip_$_per_rt": slip,
            "layer2_pf": l["pf"],
            "layer2_dd": l["max_dd_pct"],
            "layer2_mean": l["mean_pnl"],
            "layer2_net": l["net_pnl"],
            "teacher_pf": t["pf"],
            "teacher_dd": t["max_dd_pct"],
            "teacher_mean": t["mean_pnl"],
            "edge_pf": l["pf"] - t["pf"],
            "edge_net": l["net_pnl"] - t["net_pnl"],
        })

    print()
    print("=" * 100)
    print("Slippage stress grid (additional $ per round-trip applied uniformly to every trade)")
    print("=" * 100)
    print(
        f"{'slip $/rt':>10}{'l2_PF':>8}{'l2_DD%':>8}{'l2_mean':>9}{'l2_net':>10}"
        f"{'teach_PF':>10}{'teach_DD%':>10}{'teach_mean':>11}"
        f"{'edge_PF':>9}{'edge_net':>10}"
    )
    for r in rows:
        print(
            f"{r['slip_$_per_rt']:>10.0f}"
            f"{r['layer2_pf']:>8.3f}{r['layer2_dd']:>8.1f}{r['layer2_mean']:>9.1f}{r['layer2_net']:>10.0f}"
            f"{r['teacher_pf']:>10.3f}{r['teacher_dd']:>10.1f}{r['teacher_mean']:>11.1f}"
            f"{r['edge_pf']:>9.3f}{r['edge_net']:>10.0f}"
        )

    # --- Verdict ------------------------------------------------------------
    print()
    print("=" * 100)
    print("Verdict criteria (plan §2c)")
    print("=" * 100)
    at_25 = [r for r in rows if r["slip_$_per_rt"] == 25.0][0]
    l2_below_1 = at_25["layer2_pf"] < 1.0
    edge_holds = all(r["edge_pf"] > 0 for r in rows)

    if l2_below_1:
        print(f"  ⚠ HARD STOP: at $25 slippage per RT, Layer-2 PF = {at_25['layer2_pf']:.3f} < 1.0.")
        print("     Edge is vulnerable to realistic SPX 0DTE execution costs.")
        print("     Fix: exit-timing discipline (Layer-3), not paper trading.")
    elif edge_holds:
        print(f"  ✓ Edge over teacher baseline HOLDS across the full grid.")
        print(f"    Smallest edge_PF observed: {min(r['edge_pf'] for r in rows):+.3f}")
        print(f"    Layer-2 PF at $25 slip: {at_25['layer2_pf']:.3f} (>= 1.0)")
        print(f"    Realistic cost does not eat the edge.")
    else:
        print(f"  ~ Mixed: Layer-2 stays profitable but edge over teacher erodes somewhere.")
        print(f"    Edges by slip level: " + ", ".join(f"${r['slip_$_per_rt']:.0f}→{r['edge_pf']:+.3f}" for r in rows))

    return 0


if __name__ == "__main__":
    sys.exit(main())
