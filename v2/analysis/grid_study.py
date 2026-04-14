"""Grid study: side-imbalance × decay-penalty interaction.

Trains multiple RL configs sequentially, then evaluates all on 5 folds.
Designed to run on GPU in one session.

Usage (on GPU):
    python3 -m v2.analysis.grid_study --data v2/data.pt --model v2/models/model.pt --bc v2/models/seq_agent_bc.pt

Usage (local eval only, after downloading checkpoints):
    python3 -m v2.analysis.grid_study --eval-only --data v2/data.pt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

# Grid configurations: (side_coeff, decay_coeff, label)
GRID = [
    (0.03, 0.001, "s03_d001"),
    (0.06, 0.0,   "s06_d000"),
    (0.06, 0.001, "s06_d001"),
    (0.06, 0.002, "s06_d002"),
]

# Pre-existing checkpoints (already trained, just evaluate)
EXISTING = {
    "s03_d000": "v2/models/seq_agent_balanced.pt",     # side=0.03, decay=0
    "s03_d002": "v2/models/seq_agent_exitfix.pt",      # side=0.03, decay=0.002
}


def train_grid(data_path: str, model_path: str, bc_path: str):
    """Train all grid configs sequentially."""
    for side_coeff, decay_coeff, label in GRID:
        output = f"v2/models/grid_{label}.pt"
        if os.path.exists(output):
            print(f"\n--- SKIP {label} (already exists) ---")
            continue

        print(f"\n{'='*60}")
        print(f"  GRID: side={side_coeff}, decay={decay_coeff} → {label}")
        print(f"{'='*60}")

        env = os.environ.copy()
        env["SEQ_TIME_BUDGET"] = "480"
        env["RL_EPOCHS"] = "30"
        env["RL_ENTRY_COST"] = "0.015"
        env["RL_ENTRY_ESCALATION"] = "0.015"
        env["RL_KL_COEFF"] = "0.03"
        env["RL_SIDE_IMBALANCE"] = str(side_coeff)
        env["ENV_DECAY_COEFF"] = str(decay_coeff)

        cmd = [
            sys.executable, "-m", "v2.train_seq",
            "--data", data_path,
            "--model", model_path,
            "--bc-checkpoint", bc_path,
            "--output", output,
            "--mode", "rl",
        ]
        result = subprocess.run(cmd, env=env, capture_output=False)
        if result.returncode != 0:
            print(f"  WARNING: {label} training failed")


def eval_grid(data_path: str, encoder_path: str):
    """Evaluate all grid configs + existing checkpoints on 5 folds."""
    from v2.core.walkforward import generate_folds
    from v2.replay import replay_sequential
    from v2.seq_agent import SequentialAgent
    from v2.train import TradingModel, D_MODEL

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    all_days = sorted(set(data["dates"]))
    folds = generate_folds(all_days)

    # Collect all checkpoints
    all_configs = {}
    for label, path in EXISTING.items():
        if os.path.exists(path):
            all_configs[label] = path
    for side_coeff, decay_coeff, label in GRID:
        path = f"v2/models/grid_{label}.pt"
        if os.path.exists(path):
            all_configs[label] = path

    if not all_configs:
        print("No checkpoints found. Run training first.")
        return

    # Load encoder once
    encoder = TradingModel()
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    encoder.eval()

    # Evaluate each config
    results = {}
    for label, path in sorted(all_configs.items()):
        print(f"\n--- Evaluating: {label} ---")
        agent = SequentialAgent(encoder, context_dim=D_MODEL, freeze_encoder=True)
        seq_ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent.load_state_dict(seq_ckpt["agent_state_dict"], strict=False)

        fold_data = {}
        all_trades = []
        all_episodes = []

        for fold in folds:
            metrics, trades, episodes = replay_sequential(
                agent, data, fold.test_days, deterministic=True,
            )
            fold_data[fold.fold_idx] = {"metrics": metrics, "trades": trades, "episodes": episodes}
            all_trades.extend(trades)
            all_episodes.extend(episodes)

        results[label] = {"folds": fold_data, "all_trades": all_trades, "all_episodes": all_episodes}

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"  GRID STUDY RESULTS")
    print(f"{'='*100}")

    # Decode label to coefficients
    def decode(label):
        parts = label.split("_")
        s = int(parts[0][1:]) / 100
        d = int(parts[1][1:]) / 1000
        return s, d

    labels = sorted(results.keys())

    # Header
    col = 14
    header = f"  {'Metric':<22s}"
    for label in labels:
        s, d = decode(label)
        header += f"{'s='+str(s)+' d='+str(d):>{col}s}"
    print(header)
    print(f"  {'-' * (22 + col * len(labels))}")

    # Compute per-config metrics
    for metric_name in [
        "Overall PF", "Fold 0 PF", "Fold 1 PF", "Fold 2 PF", "Fold 3 PF", "Fold 4 PF",
        "Call %", "Call PF", "Put PF",
        "TPD", "Flips/day", "0-trade days", "1-entry days",
        "EOD exits", "Avg EOD PnL", "Early %",
    ]:
        row = f"  {metric_name:<22s}"
        for label in labels:
            r = results[label]
            eps = r["all_episodes"]
            trades = r["all_trades"]
            n_days = len(eps)

            if metric_name == "Overall PF":
                wins = sum(t.net_pnl_pct for t in trades if t.net_pnl_pct >= 0)
                losses = sum(abs(t.net_pnl_pct) for t in trades if t.net_pnl_pct < 0)
                val = wins / max(losses, 1e-6)
                row += f"{val:>{col}.3f}"
            elif metric_name.startswith("Fold") and metric_name.endswith("PF"):
                fi = int(metric_name.split()[1])
                fd = r["folds"].get(fi)
                if fd:
                    row += f"{fd['metrics'].profit_factor:>{col}.3f}"
                else:
                    row += f"{'n/a':>{col}s}"
            elif metric_name == "Call %":
                calls = sum(e["actions"].get(1, 0) for e in eps)
                puts = sum(e["actions"].get(2, 0) for e in eps)
                total = calls + puts
                val = 100 * calls / max(total, 1)
                row += f"{val:>{col}.1f}"
            elif metric_name in ("Call PF", "Put PF"):
                side = "call" if "Call" in metric_name else "put"
                # Match trades to sides from episodes
                entry_sides = []
                for ep in eps:
                    entry_sides.extend(ep.get("entry_sides", []))
                side_pnls = []
                for i, t in enumerate(trades):
                    if i < len(entry_sides) and entry_sides[i] == side:
                        side_pnls.append(t.net_pnl_pct)
                if side_pnls:
                    w = sum(p for p in side_pnls if p >= 0)
                    l = sum(abs(p) for p in side_pnls if p < 0)
                    val = w / max(l, 1e-6)
                    row += f"{val:>{col}.3f}"
                else:
                    row += f"{'n/a':>{col}s}"
            elif metric_name == "TPD":
                val = len(trades) / max(n_days, 1)
                row += f"{val:>{col}.2f}"
            elif metric_name == "Flips/day":
                val = np.mean([e["side_flips"] for e in eps])
                row += f"{val:>{col}.3f}"
            elif metric_name == "0-trade days":
                val = sum(1 for e in eps if e["trades"] == 0)
                row += f"{val:>{col}d}"
            elif metric_name == "1-entry days":
                val = sum(1 for e in eps if e["trades"] == 1)
                row += f"{val:>{col}d}"
            elif metric_name == "EOD exits":
                val = sum(e.get("exit_reasons", {}).get("EOD", 0) for e in eps)
                row += f"{val:>{col}d}"
            elif metric_name == "Avg EOD PnL":
                eod_pnls = []
                for t in trades:
                    if t.exit_reason == "EOD":
                        eod_pnls.append(t.net_pnl_pct)
                if eod_pnls:
                    row += f"{np.mean(eod_pnls):>{col}.3f}"
                else:
                    row += f"{'n/a':>{col}s}"
            elif metric_name == "Early %":
                early = 0
                total_e = 0
                for ep in eps:
                    actions = ep.get("actions_sequence", [])
                    bars = ep.get("bars", [])
                    for a, b in zip(actions, bars):
                        if a in (1, 2):
                            total_e += 1
                            if b <= 74:
                                early += 1
                val = 100 * early / max(total_e, 1)
                row += f"{val:>{col}.1f}"
        print(row)

    print(f"\n{'='*100}")


def main():
    parser = argparse.ArgumentParser(description="Grid study: side × decay interaction")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--model", default="v2/models/model.pt",
                        help="Encoder checkpoint (for training and eval)")
    parser.add_argument("--bc", default="v2/models/seq_agent_bc.pt",
                        help="BC checkpoint to fine-tune from")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    args = parser.parse_args()

    encoder_path = args.model

    if not args.eval_only:
        train_grid(args.data, args.model, args.bc)

    eval_grid(args.data, encoder_path)


if __name__ == "__main__":
    main()
