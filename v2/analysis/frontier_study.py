"""Frontier study: discipline vs profitability vs regime robustness.

Runs both key agents (RL v1 profitable, stabilized disciplined) across
all 5 walk-forward folds and breaks down by fold, regime, and side.

Purpose: map the Pareto frontier between overtrading chaos and
over-defensive undertrading before deciding the next move.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from v2.core.chain_data import padded_snapshot
from v2.core.features import _FEAT_IDX
from v2.core.metrics import compute_metrics
from v2.core.policy import DEFAULT_POLICY
from v2.core.walkforward import generate_folds
from v2.replay import replay_sequential
from v2.seq_agent import SequentialAgent
from v2.train import TradingModel, D_MODEL, LOOKBACK

IDX_VIX = _FEAT_IDX["vix_regime"]
IDX_TREND = _FEAT_IDX["trend_5min"]


def _load_agent(encoder_path: str, agent_path: str, device: str = "cpu"):
    encoder = TradingModel()
    if os.path.exists(encoder_path):
        ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    encoder.eval()

    agent = SequentialAgent(encoder, context_dim=D_MODEL, freeze_encoder=True)
    if os.path.exists(agent_path):
        seq_ckpt = torch.load(agent_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(seq_ckpt["agent_state_dict"], strict=False)

    return agent


def _day_regime(data: dict, day: str) -> tuple[str, str]:
    """Classify a day by average VIX regime and trend."""
    dates = data["dates"]
    features = data["X"]
    vix_vals = []
    trend_vals = []
    for i, d in enumerate(dates):
        if d == day:
            vix_vals.append(float(features[i, IDX_VIX]))
            trend_vals.append(float(features[i, IDX_TREND]))
    if not vix_vals:
        return "unknown", "unknown"
    avg_vix = np.mean(vix_vals)
    avg_trend = np.mean(trend_vals)

    if avg_vix <= -0.33:
        vix_label = "low"
    elif avg_vix >= 0.33:
        vix_label = "high"
    else:
        vix_label = "med"

    if avg_trend < -0.3:
        trend_label = "bear"
    elif avg_trend > 0.3:
        trend_label = "bull"
    else:
        trend_label = "flat"

    return vix_label, trend_label


def _side_pf(trades: list) -> tuple[float, float, int, int]:
    """Compute PF for calls and puts separately. Returns (call_pf, put_pf, n_call, n_put)."""
    call_wins = call_losses = 0.0
    put_wins = put_losses = 0.0
    n_call = n_put = 0

    for t in trades:
        # Determine side from intent or entry action
        # SimulatedTrade has trade_date and net_pnl_pct
        # We need side info — check if intent has right
        pnl = t.net_pnl_pct
        right = getattr(t.intent, 'right', None) if hasattr(t, 'intent') else None

        # Fallback: we don't have side in SimulatedTrade from sequential replay
        # Will use entry_sides from episode summaries instead
        if pnl >= 0:
            call_wins += pnl  # placeholder — actual side breakdown below
        else:
            call_losses += abs(pnl)

    # Can't reliably split by side from SimulatedTrade alone in sequential replay
    # Return totals — per-side analysis uses episode summaries
    total_wins = sum(t.net_pnl_pct for t in trades if t.net_pnl_pct >= 0)
    total_losses = sum(abs(t.net_pnl_pct) for t in trades if t.net_pnl_pct < 0)
    return total_wins, total_losses, len(trades), 0


def _print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_row(label: str, values: dict):
    parts = [f"{label:<20s}"]
    for k, v in values.items():
        if isinstance(v, float):
            parts.append(f"{v:>8.3f}")
        elif isinstance(v, int):
            parts.append(f"{v:>8d}")
        elif isinstance(v, str):
            parts.append(f"{v:>8s}")
    print("  " + "  ".join(parts))


def run_frontier_study(
    data_path: str = "v2/data.pt",
    encoder_path: str = "v2/models/model_trained_encoder.pt",
    agent_paths: dict[str, str] | None = None,
):
    if agent_paths is None:
        agent_paths = {
            "RL-v1 (profitable)": "v2/models/seq_agent_rl.pt",
            "Stable (disciplined)": "v2/models/seq_agent_stable.pt",
        }
        # Add balanced agent if it exists
        if os.path.exists("v2/models/seq_agent_balanced.pt"):
            agent_paths["Balanced (side-fix)"] = "v2/models/seq_agent_balanced.pt"
        if os.path.exists("v2/models/seq_agent_exitfix.pt"):
            agent_paths["Exit-fix (decay)"] = "v2/models/seq_agent_exitfix.pt"

    print("Loading data...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    all_days = sorted(set(data["dates"]))

    # Generate folds
    folds = generate_folds(all_days)
    print(f"  {len(folds)} folds, {sum(len(f.test_days) for f in folds)} total test days")

    # Pre-compute day regimes
    print("Computing day regimes...")
    day_regimes = {}
    for fold in folds:
        for day in fold.test_days:
            if day not in day_regimes:
                day_regimes[day] = _day_regime(data, day)

    # Run each agent across all folds
    results = {}
    for agent_name, agent_path in agent_paths.items():
        print(f"\n--- Running: {agent_name} ---")
        agent = _load_agent(encoder_path, agent_path)

        agent_results = {
            "fold_metrics": {},
            "fold_episodes": {},
            "fold_trades": {},
            "all_trades": [],
            "all_episodes": [],
        }

        for fold in folds:
            test_days = fold.test_days
            print(f"  Fold {fold.fold_idx}: {len(test_days)} days "
                  f"({test_days[0]} to {test_days[-1]})")

            metrics, trades, episodes = replay_sequential(
                agent, data, test_days, deterministic=True,
            )
            agent_results["fold_metrics"][fold.fold_idx] = metrics
            agent_results["fold_episodes"][fold.fold_idx] = episodes
            agent_results["fold_trades"][fold.fold_idx] = trades
            agent_results["all_trades"].extend(trades)
            agent_results["all_episodes"].extend(episodes)

        results[agent_name] = agent_results

    # === PRINT RESULTS ===

    # --- 1. Per-Fold Comparison ---
    _print_header("PER-FOLD COMPARISON")
    print(f"\n  {'Agent':<22s} {'Fold':>4s} {'TPD':>6s} {'PF':>6s} {'Flips':>6s} "
          f"{'0day':>5s} {'1day':>5s} {'C/P':>8s} {'Days':>5s}")
    print(f"  {'-' * 75}")

    for agent_name in agent_paths:
        ar = results[agent_name]
        for fi in sorted(ar["fold_metrics"].keys()):
            m = ar["fold_metrics"][fi]
            eps = ar["fold_episodes"][fi]
            n = len(eps)
            if n == 0:
                continue
            tpd = m.trades_per_day
            pf = m.profit_factor
            flips = np.mean([e["side_flips"] for e in eps])
            zero_d = sum(1 for e in eps if e["trades"] == 0)
            one_d = sum(1 for e in eps if e["trades"] == 1)
            calls = sum(e["actions"].get(1, 0) for e in eps)
            puts = sum(e["actions"].get(2, 0) for e in eps)
            total_entries = calls + puts
            cp = f"{100*calls/max(total_entries,1):.0f}/{100*puts/max(total_entries,1):.0f}" if total_entries > 0 else "n/a"

            print(f"  {agent_name:<22s} {fi:4d} {tpd:6.2f} {pf:6.3f} {flips:6.2f} "
                  f"{zero_d:5d} {one_d:5d} {cp:>8s} {n:5d}")
        print()

    # --- 2. Per-Regime Breakdown ---
    _print_header("PER-REGIME BREAKDOWN (VIX × Trend)")

    for agent_name in agent_paths:
        ar = results[agent_name]
        print(f"\n  {agent_name}")
        print(f"  {'Regime':<12s} {'Days':>5s} {'TPD':>6s} {'C/P':>8s} {'Flips':>6s} "
              f"{'0day':>5s} {'1day':>5s}")
        print(f"  {'-' * 55}")

        # Group episodes by regime
        regime_episodes = defaultdict(list)
        for ep in ar["all_episodes"]:
            vix, trend = day_regimes.get(ep["day"], ("unk", "unk"))
            regime_episodes[f"{vix}/{trend}"].append(ep)

        for regime in sorted(regime_episodes.keys()):
            eps = regime_episodes[regime]
            n = len(eps)
            tpd = np.mean([e["trades"] for e in eps])
            flips = np.mean([e["side_flips"] for e in eps])
            zero_d = sum(1 for e in eps if e["trades"] == 0)
            one_d = sum(1 for e in eps if e["trades"] == 1)
            calls = sum(e["actions"].get(1, 0) for e in eps)
            puts = sum(e["actions"].get(2, 0) for e in eps)
            total = calls + puts
            cp = f"{100*calls/max(total,1):.0f}/{100*puts/max(total,1):.0f}" if total > 0 else "n/a"

            print(f"  {regime:<12s} {n:5d} {tpd:6.2f} {cp:>8s} {flips:6.2f} "
                  f"{zero_d:5d} {one_d:5d}")

    # --- 3. Call vs Put Performance ---
    _print_header("CALL vs PUT PERFORMANCE")

    for agent_name in agent_paths:
        ar = results[agent_name]
        print(f"\n  {agent_name}")

        # Reconstruct per-trade side from episode entry_sides
        all_entry_sides = []
        for ep in ar["all_episodes"]:
            all_entry_sides.extend(ep.get("entry_sides", []))

        all_trades = ar["all_trades"]

        # Match trades to sides (trades and entry_sides should be in same order)
        call_pnls = []
        put_pnls = []
        for i, t in enumerate(all_trades):
            if i < len(all_entry_sides):
                side = all_entry_sides[i]
                if side == "call":
                    call_pnls.append(t.net_pnl_pct)
                else:
                    put_pnls.append(t.net_pnl_pct)

        for side_name, pnls in [("Call", call_pnls), ("Put", put_pnls)]:
            if not pnls:
                print(f"    {side_name}: no trades")
                continue
            wins = [p for p in pnls if p >= 0]
            losses = [p for p in pnls if p < 0]
            gross_win = sum(wins) if wins else 0
            gross_loss = sum(abs(l) for l in losses) if losses else 0
            pf = gross_win / max(gross_loss, 1e-6)
            wr = len(wins) / len(pnls)
            avg_w = np.mean(wins) if wins else 0
            avg_l = np.mean(losses) if losses else 0
            print(f"    {side_name}: n={len(pnls):4d}  PF={pf:.3f}  WR={100*wr:.1f}%  "
                  f"avgW={avg_w:.4f}  avgL={avg_l:.4f}")

    # --- 4. Frontier Summary ---
    _print_header("FRONTIER SUMMARY")

    headers = list(agent_paths.keys())
    metrics_names = [
        "Total trades", "Trades/day", "PF", "Win rate",
        "Zero-trade days", "Single-entry days", "Flips/day",
        "Call %", "Early-phase %",
    ]

    # Compute per-agent aggregates across all folds
    agent_aggs = {}
    for agent_name in headers:
        ar = results[agent_name]
        all_eps = ar["all_episodes"]
        all_trades = ar["all_trades"]
        n_days = len(all_eps)

        total_trades = len(all_trades)
        tpd = total_trades / max(n_days, 1)

        wins = sum(t.net_pnl_pct for t in all_trades if t.net_pnl_pct >= 0)
        losses = sum(abs(t.net_pnl_pct) for t in all_trades if t.net_pnl_pct < 0)
        pf = wins / max(losses, 1e-6)
        wr = sum(1 for t in all_trades if t.net_pnl_pct >= 0) / max(total_trades, 1)

        zero_d = sum(1 for e in all_eps if e["trades"] == 0)
        one_d = sum(1 for e in all_eps if e["trades"] == 1)
        flips = np.mean([e["side_flips"] for e in all_eps])

        calls = sum(e["actions"].get(1, 0) for e in all_eps)
        puts = sum(e["actions"].get(2, 0) for e in all_eps)
        total_entries = calls + puts
        call_pct = 100 * calls / max(total_entries, 1)

        # Early-phase entries
        early_entries = 0
        total_act_entries = 0
        for ep in all_eps:
            actions = ep.get("actions_sequence", [])
            bars = ep.get("bars", [])
            for a, b in zip(actions, bars):
                if a in (1, 2):
                    total_act_entries += 1
                    if b <= 74:
                        early_entries += 1
        early_pct = 100 * early_entries / max(total_act_entries, 1)

        agent_aggs[agent_name] = {
            "Total trades": total_trades,
            "Trades/day": tpd,
            "PF": pf,
            "Win rate": wr,
            "Zero-trade days": f"{zero_d}/{n_days}",
            "Single-entry days": f"{one_d}/{n_days}",
            "Flips/day": flips,
            "Call %": f"{call_pct:.1f}%",
            "Early-phase %": f"{early_pct:.1f}%",
        }

    # Print side by side
    col_w = 24
    print(f"\n  {'Metric':<22s}" + "".join(f"{h:>{col_w}s}" for h in headers))
    print(f"  {'-' * (22 + col_w * len(headers))}")
    for metric in metrics_names:
        row = f"  {metric:<22s}"
        for h in headers:
            v = agent_aggs[h][metric]
            if isinstance(v, float):
                row += f"{v:>{col_w}.3f}"
            elif isinstance(v, int):
                row += f"{v:>{col_w}d}"
            else:
                row += f"{v:>{col_w}s}"
        print(row)

    # Per-fold TPD comparison
    print(f"\n  {'Fold TPD':<22s}" + "".join(f"{h:>{col_w}s}" for h in headers))
    print(f"  {'-' * (22 + col_w * len(headers))}")
    for fi in sorted(folds, key=lambda f: f.fold_idx):
        row = f"  Fold {fi.fold_idx:<16d}"
        for h in headers:
            m = results[h]["fold_metrics"].get(fi.fold_idx)
            if m:
                row += f"{m.trades_per_day:>{col_w}.2f}"
            else:
                row += f"{'n/a':>{col_w}s}"
        print(row)

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Frontier study: discipline vs profitability")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--encoder", default="v2/models/model_trained_encoder.pt")
    args = parser.parse_args()

    run_frontier_study(data_path=args.data, encoder_path=args.encoder)


if __name__ == "__main__":
    main()
