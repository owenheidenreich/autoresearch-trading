"""Evaluate whether the current pilot earns a broader data purchase.

The broad purchase gate is deliberately stricter than the model-iteration gate:
it uses the existing pilot only, chooses thresholds and time filters on February
validation, and scores March holdout across multiple random seeds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from v4.model.action_pilot import (
    load_action_decisions,
    metrics_for_trades,
    predict_actions,
    simulate_action_policy,
    train_action_model,
)
from v4.model.environment_diagnostics import time_bucket
from v4.model.supervised_pilot import PilotConfig, Trade, session_from_path, split_name
from v4.scripts.evaluate_timeaware_action_filters import FILTERS
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


@dataclass(frozen=True)
class PurchaseGate:
    """Evidence threshold for authorizing broader historical data spend."""

    min_test_profit_factor_median: float = 1.20
    min_test_pnl_median: float = 5_000.0
    min_positive_seed_fraction: float = 0.70
    min_positive_day_fraction_median: float = 0.55
    max_drawdown_floor_median: float = -6_000.0
    min_test_trades_median: float = 40.0
    max_top_day_profit_share_median: float = 0.45


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/broad_data_purchase_signal"))
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    p.add_argument("--policy-indexes", nargs="*", type=int, default=[0, 1, 2], choices=sorted(POLICY_META))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4096)
    return p.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _filter_trades(trades: list[Trade], allowed: tuple[str, ...]) -> list[Trade]:
    out = []
    for trade in trades:
        bucket = time_bucket(__import__("pandas").Timestamp(trade.decision_time).to_pydatetime())
        if bucket in allowed:
            out.append(trade)
    return out


def _top_day_profit_share(trades: list[Trade]) -> float:
    by_day: dict[str, float] = {}
    for trade in trades:
        by_day[trade.session] = by_day.get(trade.session, 0.0) + trade.pnl
    positives = np.asarray([v for v in by_day.values() if v > 0], dtype=float)
    if positives.sum() <= 0:
        return 1.0
    return float(positives.max() / positives.sum())


def _filter_metrics(trades: list[Trade], allowed: tuple[str, ...]) -> dict:
    filtered = _filter_trades(trades, allowed)
    metrics = metrics_for_trades(filtered)
    metrics["top_day_profit_share"] = _top_day_profit_share(filtered)
    return metrics


def _select_filter(validation_filter_metrics: dict[str, dict]) -> str:
    """Select the time filter using validation only."""
    eligible = {
        name: metrics
        for name, metrics in validation_filter_metrics.items()
        if metrics["trades"] >= 20
        and metrics["total_pnl"] > 0
        and metrics["profit_factor"] > 1.10
        and metrics["max_drawdown"] > -7_500
    }
    pool = eligible if eligible else validation_filter_metrics
    return max(
        pool,
        key=lambda name: (
            pool[name]["profit_factor"],
            pool[name]["total_pnl"],
            -abs(pool[name]["max_drawdown"]),
        ),
    )


def _run_one(
    *,
    paths: dict[str, list[Path]],
    policy_index: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> dict:
    policy_name, cooldown = POLICY_META[policy_index]
    config = replace(
        PilotConfig(),
        policy_index=policy_index,
        policy_name=policy_name,
        cooldown_minutes=cooldown,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=128,
        seed=seed,
    )
    decisions = {
        split: load_action_decisions(files, policy_index=policy_index)
        for split, files in paths.items()
    }
    model, scaler, history = train_action_model(
        decisions["train"],
        decisions["validation"],
        config=config,
    )

    predictions = {
        split: predict_actions(model, scaler, split_decisions, target_scale=config.target_scale)
        for split, split_decisions in decisions.items()
    }
    from v4.model.action_pilot import choose_action_threshold

    threshold, _ = choose_action_threshold(
        decisions["validation"],
        predictions["validation"],
        config=config,
    )
    raw_trades = {
        split: simulate_action_policy(
            split_decisions,
            predictions[split],
            threshold=threshold,
            cooldown_minutes=cooldown,
            strategy="action_neural_timeaware",
        )
        for split, split_decisions in decisions.items()
    }
    filter_metrics = {
        split: {
            name: _filter_metrics(raw_trades[split], allowed)
            for name, allowed in FILTERS.items()
        }
        for split in ("train", "validation", "test")
    }
    selected_filter = _select_filter(filter_metrics["validation"])
    return {
        "policy_index": policy_index,
        "policy_name": policy_name,
        "seed": seed,
        "threshold": threshold,
        "selected_filter": selected_filter,
        "best_epoch": next((x["epoch"] for x in history if x["is_best"]), None),
        "filter_metrics": filter_metrics,
        "selected": {
            split: filter_metrics[split][selected_filter]
            for split in ("train", "validation", "test")
        },
    }


def _aggregate(results: list[dict], gate: PurchaseGate) -> dict:
    by_policy = {}
    for policy_index in sorted({r["policy_index"] for r in results}):
        rows = [r for r in results if r["policy_index"] == policy_index]
        test = [r["selected"]["test"] for r in rows]
        selected_filters = [r["selected_filter"] for r in rows]
        arr = {
            "total_pnl": np.asarray([m["total_pnl"] for m in test], dtype=float),
            "profit_factor": np.asarray([m["profit_factor"] for m in test], dtype=float),
            "max_drawdown": np.asarray([m["max_drawdown"] for m in test], dtype=float),
            "trades": np.asarray([m["trades"] for m in test], dtype=float),
            "positive_day_fraction": np.asarray([m["positive_day_fraction"] for m in test], dtype=float),
            "top_day_profit_share": np.asarray([m["top_day_profit_share"] for m in test], dtype=float),
        }
        summary = {
            "policy_name": POLICY_META[policy_index][0],
            "runs": len(rows),
            "selected_filter_counts": {
                name: selected_filters.count(name)
                for name in sorted(set(selected_filters))
            },
            "test_pnl_median": float(np.median(arr["total_pnl"])),
            "test_pnl_mean": float(arr["total_pnl"].mean()),
            "test_profit_factor_median": float(np.median(arr["profit_factor"])),
            "test_max_drawdown_median": float(np.median(arr["max_drawdown"])),
            "test_trades_median": float(np.median(arr["trades"])),
            "positive_seed_fraction": float((arr["total_pnl"] > 0).mean()),
            "positive_day_fraction_median": float(np.median(arr["positive_day_fraction"])),
            "top_day_profit_share_median": float(np.median(arr["top_day_profit_share"])),
        }
        summary["passes_broad_purchase_gate"] = bool(
            summary["test_profit_factor_median"] >= gate.min_test_profit_factor_median
            and summary["test_pnl_median"] >= gate.min_test_pnl_median
            and summary["positive_seed_fraction"] >= gate.min_positive_seed_fraction
            and summary["positive_day_fraction_median"] >= gate.min_positive_day_fraction_median
            and summary["test_max_drawdown_median"] >= gate.max_drawdown_floor_median
            and summary["test_trades_median"] >= gate.min_test_trades_median
            and summary["top_day_profit_share_median"] <= gate.max_top_day_profit_share_median
        )
        by_policy[str(policy_index)] = summary

    any_pass = any(x["passes_broad_purchase_gate"] for x in by_policy.values())
    recommendation = (
        "broad_purchase_supported"
        if any_pass
        else "no_broad_purchase_yet_continue_existing_data_iteration"
    )
    return {
        "gate": asdict(gate),
        "recommendation": recommendation,
        "by_policy": by_policy,
    }


def main() -> int:
    args = parse_args()
    paths = _paths_by_split(args.data_dir)
    gate = PurchaseGate()
    results = []
    for policy_index in args.policy_indexes:
        for seed in args.seeds:
            print(f"running policy={policy_index} seed={seed}", flush=True)
            results.append(
                _run_one(
                    paths=paths,
                    policy_index=policy_index,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                )
            )
    aggregate = _aggregate(results, gate)
    payload = {
        "framing": (
            "Broad data purchase requires repeated evidence on the existing pilot, "
            "with validation-only threshold/filter selection and March holdout scoring. "
            "This does not use or purchase new data."
        ),
        "seeds": args.seeds,
        "policy_indexes": args.policy_indexes,
        "aggregate": aggregate,
        "runs": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    md_path = args.out_dir / "report.md"
    lines = [
        "# Broad Data Purchase Signal",
        "",
        payload["framing"],
        "",
        f"Recommendation: **{aggregate['recommendation']}**",
        "",
        "## Gate",
        "",
        "| Criterion | Value |",
        "|---|---:|",
    ]
    for key, value in aggregate["gate"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Policy Robustness",
        "",
        "| Policy | Runs | Filters | Median PnL | Median PF | Median DD | Positive Seeds | Positive Days | Top-Day Share | Pass |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for policy_index, summary in aggregate["by_policy"].items():
        lines.append(
            f"| {summary['policy_name']} | {summary['runs']} | "
            f"{summary['selected_filter_counts']} | {summary['test_pnl_median']:.0f} | "
            f"{summary['test_profit_factor_median']:.3f} | {summary['test_max_drawdown_median']:.0f} | "
            f"{summary['positive_seed_fraction']:.2f} | {summary['positive_day_fraction_median']:.2f} | "
            f"{summary['top_day_profit_share_median']:.2f} | {summary['passes_broad_purchase_gate']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)
    print(json.dumps(aggregate, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
