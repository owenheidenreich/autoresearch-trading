"""Evaluate user-guided time filters on the time-aware action pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v4.model.action_pilot import (
    ActionMLP,
    load_action_decisions,
    predict_actions,
    simulate_action_policy,
)
from v4.model.environment_diagnostics import time_bucket
from v4.model.supervised_pilot import (
    FeatureScaler,
    PilotConfig,
    Trade,
    metrics_for_trades,
    session_from_path,
    split_name,
)


FILTERS = {
    "all_times": ("first_30", "post_open_morning", "midday", "late_afternoon"),
    "skip_first_30": ("post_open_morning", "midday", "late_afternoon"),
    "post_open_and_late": ("post_open_morning", "late_afternoon"),
    "late_afternoon_only": ("late_afternoon",),
    "post_open_only": ("post_open_morning",),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("data/models/v4_spxw_action_pilot_policy2_timeaware.pt"))
    p.add_argument("--report", type=Path, default=Path("v4/audit/action_pilot_policy2_timeaware/report.json"))
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/action_pilot_policy2_timeaware"))
    return p.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _load_model(path: Path) -> tuple[ActionMLP, FeatureScaler, PilotConfig]:
    checkpoint = torch.load(path, weights_only=False)
    config = PilotConfig(**checkpoint["config"])
    model = ActionMLP(input_dim=checkpoint["input_dim"], hidden_dim=max(config.hidden_dim, 128))
    model.load_state_dict(checkpoint["model_state"])
    scaler = FeatureScaler(
        fill=np.asarray(checkpoint["scaler"]["fill"], dtype=np.float32),
        mean=np.asarray(checkpoint["scaler"]["mean"], dtype=np.float32),
        std=np.asarray(checkpoint["scaler"]["std"], dtype=np.float32),
    )
    return model, scaler, config


def _filter_trades(trades: list[Trade], allowed: tuple[str, ...]) -> list[Trade]:
    out = []
    for trade in trades:
        bucket = time_bucket(__import__("pandas").Timestamp(trade.decision_time).to_pydatetime())
        if bucket in allowed:
            out.append(trade)
    return out


def main() -> int:
    args = parse_args()
    model, scaler, config = _load_model(args.model)
    threshold = json.loads(args.report.read_text())["chosen_threshold"]
    paths = _paths_by_split(args.data_dir)
    payload = {
        "source_model": str(args.model),
        "threshold": threshold,
        "policy": config.policy_name,
        "environment_framing": (
            "These are user-guided time filters evaluated on the existing pilot data. "
            "They are not proof of broad-market generalization, but they test whether "
            "the model can use time-of-day structure that appears repeatedly in the data."
        ),
        "filters": {},
    }
    for split, split_paths in paths.items():
        decisions = load_action_decisions(split_paths, policy_index=config.policy_index)
        predictions = predict_actions(model, scaler, decisions, target_scale=config.target_scale)
        trades = simulate_action_policy(
            decisions,
            predictions,
            threshold=threshold,
            cooldown_minutes=config.cooldown_minutes,
            strategy="action_neural_timeaware",
        )
        payload["filters"][split] = {
            name: metrics_for_trades(_filter_trades(trades, allowed))
            for name, allowed in FILTERS.items()
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "time_filter_report.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    md_path = args.out_dir / "time_filter_report.md"
    lines = [
        "# Time-Aware Action Filter Report",
        "",
        payload["environment_framing"],
        "",
        "| Filter | Train PnL | Train PF | Val PnL | Val PF | Test Trades | Test PnL | Test PF | Test DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in FILTERS:
        tr = payload["filters"]["train"][name]
        va = payload["filters"]["validation"][name]
        te = payload["filters"]["test"][name]
        lines.append(
            f"| {name} | {tr['total_pnl']:.0f} | {tr['profit_factor']:.3f} | "
            f"{va['total_pnl']:.0f} | {va['profit_factor']:.3f} | "
            f"{te['trades']} | {te['total_pnl']:.0f} | {te['profit_factor']:.3f} | {te['max_drawdown']:.0f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)
    print(json.dumps(payload["filters"]["test"], indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
