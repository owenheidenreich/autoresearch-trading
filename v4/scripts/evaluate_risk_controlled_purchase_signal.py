"""Risk-controlled evidence gate for a broader SPXW 0DTE data purchase.

This is stricter than the first broad-purchase pass. It chooses threshold,
time window, and simple daily risk controls on February validation only, then
scores the locked policy on March holdout across multiple random seeds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from v4.model.action_pilot import (
    ActionDecision,
    load_action_decisions,
    predict_actions,
    train_action_model,
)
from v4.model.environment_diagnostics import time_bucket
from v4.model.supervised_pilot import PilotConfig, Trade, metrics_for_trades, session_from_path, split_name
from v4.scripts.evaluate_broad_data_purchase_signal import PurchaseGate
from v4.scripts.evaluate_timeaware_action_filters import FILTERS
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


MAX_TRADES_PER_DAY = (2, 3, 4, 6, 99)
DAILY_LOSS_STOPS = (None, -500.0, -1_000.0, -1_500.0, -2_500.0)


@dataclass(frozen=True)
class RiskConfig:
    """Executable policy controls selected before holdout scoring."""

    threshold: float
    time_filter: str
    allowed_buckets: tuple[str, ...]
    max_trades_per_day: int
    daily_loss_stop: float | None

    @property
    def name(self) -> str:
        stop = "none" if self.daily_loss_stop is None else f"{self.daily_loss_stop:.0f}"
        max_trades = "unlimited" if self.max_trades_per_day >= 99 else str(self.max_trades_per_day)
        return (
            f"{self.time_filter}|threshold={self.threshold:.4f}|"
            f"max_trades={max_trades}|daily_stop={stop}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/risk_controlled_purchase_signal"))
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    p.add_argument("--policy-indexes", nargs="*", type=int, default=[0, 1, 2], choices=sorted(POLICY_META))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--min-validation-trades", type=int, default=25)
    return p.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _candidate_thresholds(predictions: np.ndarray) -> list[float]:
    if len(predictions) == 0:
        return [float("inf")]
    top = np.max(predictions[:, 1:], axis=1)
    top = top[np.isfinite(top)]
    if len(top) == 0:
        return [float("inf")]
    quantiles = [0.0, 0.20, 0.35, 0.50, 0.65, 0.75, 0.85, 0.90, 0.94, 0.97, 0.985]
    return sorted(set(np.quantile(top, quantiles).round(4).tolist() + [0.0]))


def _parse_trade_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _top_day_profit_share(trades: Sequence[Trade]) -> float:
    by_day: dict[str, float] = {}
    for trade in trades:
        by_day[trade.session] = by_day.get(trade.session, 0.0) + trade.pnl
    positives = np.asarray([v for v in by_day.values() if v > 0], dtype=float)
    if positives.sum() <= 0:
        return 1.0
    return float(positives.max() / positives.sum())


def metrics_with_concentration(trades: Sequence[Trade]) -> dict:
    metrics = metrics_for_trades(trades)
    metrics["top_day_profit_share"] = _top_day_profit_share(trades)
    return metrics


def simulate_risk_controlled_policy(
    decisions: Sequence[ActionDecision],
    predictions: np.ndarray,
    *,
    config: RiskConfig,
    cooldown_minutes: int,
    strategy: str,
) -> list[Trade]:
    """Simulate one-contract action policy with time and daily controls.

    Time eligibility is checked before cooldown so disallowed windows cannot
    consume the one-contract-at-a-time slot.
    """
    trades: list[Trade] = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    allowed = set(config.allowed_buckets)

    for decision, pred in zip(decisions, predictions):
        if time_bucket(decision.decision_time) not in allowed:
            continue
        if decision.session in halted_sessions:
            continue
        if trades_by_session.get(decision.session, 0) >= config.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue

        action = int(np.argmax(pred[1:]) + 1)
        score = float(pred[action])
        if not np.isfinite(score) or score < config.threshold:
            continue
        pnl = float(decision.labels[action])
        if not np.isfinite(pnl):
            continue

        trades.append(
            Trade(
                session=decision.session,
                decision_time=decision.decision_time.isoformat(),
                pnl=pnl,
                score=score,
                right="C" if action == 1 else "P",
                offset=float(decision.offsets[action]),
                strategy=strategy,
            )
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        pnl_by_session[decision.session] = pnl_by_session.get(decision.session, 0.0) + pnl
        next_time_by_session[decision.session] = decision.decision_time + timedelta(
            minutes=cooldown_minutes
        )
        if (
            config.daily_loss_stop is not None
            and pnl_by_session[decision.session] <= config.daily_loss_stop
        ):
            halted_sessions.add(decision.session)

    return trades


def _risk_config_grid(thresholds: Sequence[float]) -> list[RiskConfig]:
    configs = []
    for threshold in thresholds:
        for filter_name, allowed_buckets in FILTERS.items():
            for max_trades in MAX_TRADES_PER_DAY:
                for daily_stop in DAILY_LOSS_STOPS:
                    configs.append(
                        RiskConfig(
                            threshold=float(threshold),
                            time_filter=filter_name,
                            allowed_buckets=tuple(allowed_buckets),
                            max_trades_per_day=max_trades,
                            daily_loss_stop=daily_stop,
                        )
                    )
    return configs


def _selection_key(row: dict) -> tuple[float, float, float, float, float, float]:
    metrics = row["metrics"]
    profit_factor = metrics["profit_factor"]
    if not np.isfinite(profit_factor):
        profit_factor = 999.0
    return (
        float(metrics["total_pnl"]),
        float(profit_factor),
        float(metrics["positive_day_fraction"]),
        -float(metrics["top_day_profit_share"]),
        float(metrics["max_drawdown"]),
        float(metrics["trades"]),
    )


def select_risk_config(
    decisions: Sequence[ActionDecision],
    predictions: np.ndarray,
    *,
    cooldown_minutes: int,
    min_validation_trades: int,
) -> tuple[RiskConfig, list[dict]]:
    """Select controls from validation data only."""
    sweep = []
    for config in _risk_config_grid(_candidate_thresholds(predictions)):
        trades = simulate_risk_controlled_policy(
            decisions,
            predictions,
            config=config,
            cooldown_minutes=cooldown_minutes,
            strategy="validation_risk_sweep",
        )
        sweep.append(
            {
                "config": asdict(config) | {"name": config.name},
                "metrics": metrics_with_concentration(trades),
            }
        )

    eligible = [
        row
        for row in sweep
        if row["metrics"]["trades"] >= min_validation_trades
        and row["metrics"]["total_pnl"] > 0
        and row["metrics"]["profit_factor"] >= 1.15
        and row["metrics"]["positive_day_fraction"] >= 0.50
        and row["metrics"]["max_drawdown"] >= -6_000
        and row["metrics"]["top_day_profit_share"] <= 0.65
    ]
    pool = eligible if eligible else sweep
    best = max(pool, key=_selection_key)
    cfg = best["config"]
    selected = RiskConfig(
        threshold=float(cfg["threshold"]),
        time_filter=str(cfg["time_filter"]),
        allowed_buckets=tuple(cfg["allowed_buckets"]),
        max_trades_per_day=int(cfg["max_trades_per_day"]),
        daily_loss_stop=cfg["daily_loss_stop"],
    )
    return selected, sweep


def _run_one(
    *,
    paths: dict[str, list[Path]],
    policy_index: int,
    seed: int,
    epochs: int,
    batch_size: int,
    min_validation_trades: int,
) -> dict:
    policy_name, cooldown = POLICY_META[policy_index]
    config = PilotConfig(
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
    selected_config, validation_sweep = select_risk_config(
        decisions["validation"],
        predictions["validation"],
        cooldown_minutes=cooldown,
        min_validation_trades=min_validation_trades,
    )
    trades_by_split = {
        split: simulate_risk_controlled_policy(
            split_decisions,
            predictions[split],
            config=selected_config,
            cooldown_minutes=cooldown,
            strategy="action_neural_timeaware_risk_controlled",
        )
        for split, split_decisions in decisions.items()
    }
    return {
        "policy_index": policy_index,
        "policy_name": policy_name,
        "seed": seed,
        "selected_config": asdict(selected_config) | {"name": selected_config.name},
        "best_epoch": next((x["epoch"] for x in history if x["is_best"]), None),
        "selected": {
            split: metrics_with_concentration(trades)
            for split, trades in trades_by_split.items()
        },
        "validation_sweep_top10": sorted(validation_sweep, key=_selection_key, reverse=True)[:10],
    }


def _aggregate(results: list[dict], gate: PurchaseGate) -> dict:
    by_policy = {}
    for policy_index in sorted({r["policy_index"] for r in results}):
        rows = [r for r in results if r["policy_index"] == policy_index]
        test = [r["selected"]["test"] for r in rows]
        configs = [r["selected_config"]["name"] for r in rows]
        filters = [r["selected_config"]["time_filter"] for r in rows]
        max_trades = [r["selected_config"]["max_trades_per_day"] for r in rows]
        daily_stops = [r["selected_config"]["daily_loss_stop"] for r in rows]
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
            "selected_config_counts": {
                name: configs.count(name)
                for name in sorted(set(configs))
            },
            "selected_filter_counts": {
                name: filters.count(name)
                for name in sorted(set(filters))
            },
            "selected_max_trades_counts": {
                str(name): max_trades.count(name)
                for name in sorted(set(max_trades))
            },
            "selected_daily_stop_counts": {
                str(name): daily_stops.count(name)
                for name in sorted(set(daily_stops), key=lambda x: (x is not None, x or 0))
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


def _write_report(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    aggregate = payload["aggregate"]
    lines = [
        "# Risk-Controlled Broad Data Purchase Signal",
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
        "| Policy | Runs | Filters | Max Trades | Daily Stops | Median PnL | Median PF | Median DD | Trades | Positive Seeds | Positive Days | Top-Day Share | Pass |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, summary in aggregate["by_policy"].items():
        lines.append(
            f"| {summary['policy_name']} | {summary['runs']} | "
            f"{summary['selected_filter_counts']} | {summary['selected_max_trades_counts']} | "
            f"{summary['selected_daily_stop_counts']} | {summary['test_pnl_median']:.0f} | "
            f"{summary['test_profit_factor_median']:.3f} | {summary['test_max_drawdown_median']:.0f} | "
            f"{summary['test_trades_median']:.0f} | {summary['positive_seed_fraction']:.2f} | "
            f"{summary['positive_day_fraction_median']:.2f} | "
            f"{summary['top_day_profit_share_median']:.2f} | "
            f"{summary['passes_broad_purchase_gate']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "A pass means the current pilot produced a broad historical-data purchase signal "
            "after validation-only selection of model threshold, time window, and basic daily "
            "risk controls. A fail does not reject the project; it means the next work should "
            "happen on the existing data before buying a much larger history."
        ),
    ]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)


def main() -> int:
    args = parse_args()
    paths = _paths_by_split(args.data_dir)
    gate = PurchaseGate()
    results = []
    for policy_index in args.policy_indexes:
        for seed in args.seeds:
            print(f"running risk policy={policy_index} seed={seed}", flush=True)
            results.append(
                _run_one(
                    paths=paths,
                    policy_index=policy_index,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    min_validation_trades=args.min_validation_trades,
                )
            )

    aggregate = _aggregate(results, gate)
    payload = {
        "framing": (
            "This evaluator uses only the existing January-March pilot data. It selects "
            "the action threshold, time window, max trades per day, and daily loss stop "
            "from February validation, then scores March holdout across random seeds. "
            "No new paid data is used or purchased."
        ),
        "seeds": args.seeds,
        "policy_indexes": args.policy_indexes,
        "risk_grid": {
            "time_filters": FILTERS,
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "daily_loss_stops": DAILY_LOSS_STOPS,
        },
        "aggregate": aggregate,
        "runs": results,
    }
    _write_report(args.out_dir, payload)
    print(json.dumps(aggregate, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
