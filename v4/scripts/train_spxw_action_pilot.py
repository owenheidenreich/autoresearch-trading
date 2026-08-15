"""Train and evaluate the SPXW 0DTE decision-level action pilot."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from v4.model.action_pilot import (
    action_model_state_dict,
    choose_action_threshold,
    load_action_decisions,
    metrics_for_trades,
    predict_actions,
    session_from_path,
    simulate_action_baseline,
    simulate_action_policy,
    split_name,
    summarize_random_action_baseline,
    train_action_model,
    write_json,
)
from v4.model.supervised_pilot import PilotConfig
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/spxw_0dte_neural_derived"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/action_pilot_policy2"))
    p.add_argument("--model-out", type=Path, default=Path("data/models/v4_spxw_action_pilot_policy2.pt"))
    p.add_argument("--policy-index", type=int, choices=sorted(POLICY_META), default=2)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _baseline_metrics(decisions, config: PilotConfig) -> dict:
    out = {
        "no_trade": metrics_for_trades([]),
        "random_atm_side": summarize_random_action_baseline(decisions, config=config),
    }
    for kind in ("atm_call", "atm_put", "vwap_omar"):
        trades = simulate_action_baseline(
            decisions,
            kind=kind,
            cooldown_minutes=config.cooldown_minutes,
            seed=config.seed,
        )
        out[kind] = metrics_for_trades(trades)
    return out


def main() -> int:
    args = parse_args()
    policy_name, cooldown = POLICY_META[args.policy_index]
    config = replace(
        PilotConfig(),
        policy_index=args.policy_index,
        policy_name=policy_name,
        cooldown_minutes=cooldown,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=128,
        seed=args.seed,
    )
    paths = _paths_by_split(args.data_dir)
    print("loading action decisions", {k: len(v) for k, v in paths.items()}, flush=True)
    train_decisions = load_action_decisions(paths["train"], policy_index=config.policy_index)
    val_decisions = load_action_decisions(paths["validation"], policy_index=config.policy_index)
    test_decisions = load_action_decisions(paths["test"], policy_index=config.policy_index)
    print(
        "decision counts",
        {"train": len(train_decisions), "validation": len(val_decisions), "test": len(test_decisions)},
        flush=True,
    )

    model, scaler, history = train_action_model(train_decisions, val_decisions, config=config)
    print("training history", json.dumps(history[-4:], indent=2), flush=True)

    val_pred = predict_actions(model, scaler, val_decisions, target_scale=config.target_scale)
    threshold, sweep = choose_action_threshold(val_decisions, val_pred, config=config)
    print("chosen threshold", threshold, flush=True)

    predictions = {
        "train": predict_actions(model, scaler, train_decisions, target_scale=config.target_scale),
        "validation": val_pred,
        "test": predict_actions(model, scaler, test_decisions, target_scale=config.target_scale),
    }
    decisions_by_split = {
        "train": train_decisions,
        "validation": val_decisions,
        "test": test_decisions,
    }
    neural = {}
    for split, decisions in decisions_by_split.items():
        trades = simulate_action_policy(
            decisions,
            predictions[split],
            threshold=threshold,
            cooldown_minutes=config.cooldown_minutes,
            strategy="action_neural",
        )
        neural[split] = {
            "threshold": threshold,
            "metrics": metrics_for_trades(trades),
            "sample_trades": [trade.__dict__ for trade in trades[:10]],
        }

    baselines = {
        "validation": _baseline_metrics(val_decisions, config),
        "test": _baseline_metrics(test_decisions, config),
    }
    report = {
        "config": asdict(config),
        "splits": {
            split: {
                "sessions": [session_from_path(p) for p in paths[split]],
                "decision_rows": len(decisions_by_split[split]),
            }
            for split in ("train", "validation", "test")
        },
        "training_history": history,
        "chosen_threshold": threshold,
        "threshold_sweep_validation": sweep,
        "neural": neural,
        "baselines": baselines,
        "interpretation": {
            "clears_next_data_purchase_gate": bool(
                neural["test"]["metrics"]["total_pnl"] > 0
                and neural["test"]["metrics"]["trades"] >= 20
                and neural["test"]["metrics"]["profit_factor"] > 1.05
                and neural["test"]["metrics"]["max_drawdown"] > -5000
            ),
            "regime_caveat": (
                "The Jan-Mar 2026 pilot window is real market truth, but only one "
                "environment slice. Passing this harness would not prove generalization "
                "across calmer, trending, shock-driven, and liquidity-fragmented markets; "
                "failing it does not fully falsify ideas that may require broader "
                "environment coverage."
            ),
            "notes": (
                "Decision-level action pilot for no-trade/ATM-call/ATM-put. Passing this "
                "would justify model iteration, not live trading or broad data purchases."
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.json"
    write_json(report_path, report)
    write_json(Path("v4/audit/action_pilot_latest.json"), report)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        action_model_state_dict(
            model=model,
            scaler=scaler,
            config=config,
            history=history,
            input_dim=model.net[0].in_features,
        ),
        args.model_out,
    )
    print(f"WROTE {report_path}")
    print(f"WROTE {args.model_out}")
    print(json.dumps(report["interpretation"], indent=2), flush=True)
    print(
        json.dumps(
            {"test_neural": neural["test"]["metrics"], "test_baselines": baselines["test"]},
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
