"""Train and evaluate the first SPXW 0DTE supervised neural pilot."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from v4.model.supervised_pilot import (
    PilotConfig,
    choose_threshold,
    collect_examples,
    load_decisions,
    metrics_for_trades,
    model_state_dict,
    predict_decisions,
    session_from_path,
    simulate_baseline,
    simulate_model_policy,
    split_name,
    summarize_random_baseline,
    train_model,
    write_json,
)


POLICY_META = {
    0: ("ask_to_bid_stop35_target60_hold10m", 10),
    1: ("ask_to_bid_stop50_target100_hold25m", 25),
    2: ("ask_to_bid_stop65_target150_hold45m", 45),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/spxw_0dte_neural_derived"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/supervised_pilot"))
    p.add_argument("--model-out", type=Path, default=Path("data/models/v4_spxw_supervised_pilot.pt"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max-train-examples", type=int, default=350_000)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--policy-index", type=int, choices=sorted(POLICY_META), default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        split = split_name(session_from_path(path))
        out[split].append(path)
    return out


def _baseline_metrics(decisions, config: PilotConfig) -> dict:
    out = {
        "no_trade": metrics_for_trades([]),
        "random_valid": summarize_random_baseline(decisions, config=config),
    }
    for kind in ("atm_call", "atm_put", "vwap_omar"):
        trades = simulate_baseline(
            decisions,
            kind=kind,
            cooldown_minutes=config.cooldown_minutes,
            seed=config.seed,
        )
        out[kind] = metrics_for_trades(trades)
    return out


def main() -> int:
    args = parse_args()
    policy_name, cooldown_minutes = POLICY_META[args.policy_index]
    config = replace(
        PilotConfig(),
        policy_index=args.policy_index,
        policy_name=policy_name,
        cooldown_minutes=cooldown_minutes,
        epochs=args.epochs,
        max_train_examples=args.max_train_examples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    paths = _paths_by_split(args.data_dir)
    if not all(paths.values()):
        raise SystemExit(f"missing split files under {args.data_dir}: { {k: len(v) for k, v in paths.items()} }")

    print("loading decisions", {k: len(v) for k, v in paths.items()}, flush=True)
    train_decisions = load_decisions(paths["train"], policy_index=config.policy_index)
    val_decisions = load_decisions(paths["validation"], policy_index=config.policy_index)
    test_decisions = load_decisions(paths["test"], policy_index=config.policy_index)
    print(
        "decision counts",
        {
            "train": len(train_decisions),
            "validation": len(val_decisions),
            "test": len(test_decisions),
        },
        flush=True,
    )

    x_train_preview, y_train_preview = collect_examples(train_decisions, max_examples=None)
    print(
        "train examples",
        {
            "candidates": len(y_train_preview),
            "input_dim": x_train_preview.shape[1],
            "label_mean": float(y_train_preview.mean()),
            "label_median": float(__import__("numpy").median(y_train_preview)),
        },
        flush=True,
    )
    del x_train_preview, y_train_preview

    model, scaler, history = train_model(train_decisions, val_decisions, config=config)
    print("training history", json.dumps(history[-3:], indent=2), flush=True)

    val_predictions = predict_decisions(
        model,
        scaler,
        val_decisions,
        target_scale=config.target_scale,
    )
    threshold, threshold_sweep = choose_threshold(
        val_decisions,
        val_predictions,
        config=config,
    )
    print("chosen threshold", threshold, flush=True)

    test_predictions = predict_decisions(
        model,
        scaler,
        test_decisions,
        target_scale=config.target_scale,
    )
    train_predictions = predict_decisions(
        model,
        scaler,
        train_decisions,
        target_scale=config.target_scale,
    )

    neural = {}
    for name, decisions, preds in (
        ("train", train_decisions, train_predictions),
        ("validation", val_decisions, val_predictions),
        ("test", test_decisions, test_predictions),
    ):
        trades = simulate_model_policy(
            decisions,
            preds,
            threshold=threshold,
            cooldown_minutes=config.cooldown_minutes,
            strategy="neural_threshold",
        )
        neural[name] = {
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
            name: {
                "sessions": [session_from_path(p) for p in paths[name]],
                "decision_rows": len(
                    {
                        "train": train_decisions,
                        "validation": val_decisions,
                        "test": test_decisions,
                    }[name]
                ),
            }
            for name in ("train", "validation", "test")
        },
        "training_history": history,
        "chosen_threshold": threshold,
        "threshold_sweep_validation": threshold_sweep,
        "neural": neural,
        "baselines": baselines,
        "interpretation": {
            "clears_next_data_purchase_gate": bool(
                neural["test"]["metrics"]["total_pnl"] > 0
                and neural["test"]["metrics"]["trades"] >= 20
                and neural["test"]["metrics"]["profit_factor"] > 1.05
            ),
            "regime_caveat": (
                "The Jan-Mar 2026 pilot window is real market truth, but only one "
                "environment slice. The result is useful for data-stack falsification "
                "and model design, but not enough to claim generalization across calmer, "
                "trending, shock-driven, and liquidity-fragmented markets."
            ),
            "notes": (
                "This is a first supervised falsification harness. Passing it would justify "
                "more modeling work, not live trading. Failing it argues against buying more data yet."
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.json"
    write_json(report_path, report)
    write_json(Path("v4/audit/supervised_pilot_latest.json"), report)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        model_state_dict(
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
    print(json.dumps({"test_neural": neural["test"]["metrics"], "test_baselines": baselines["test"]}, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
