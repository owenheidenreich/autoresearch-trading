"""Bounded autoresearch loop for SPXW 0DTE neural option trading.

The loop is inspired by the small, fixed-budget autoresearch pattern, but the
trading version is intentionally more adversarial about reward hacking:

* train on January
* use early-February validation for early stopping/calibration only
* select trials on late-February validation only
* report March as audit-only holdout

No trial is allowed to choose parameters from March metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from v4.model.action_pilot import (
    ACTION_DECISION_LOSS_CONFIG,
    ACTION_LOSS_MODES,
    ActionDecision,
    action_feature_version,
    load_action_decisions,
    predict_actions,
    train_action_model,
)
from v4.model.environment_diagnostics import time_bucket
from v4.model.supervised_pilot import PilotConfig, Trade, metrics_for_trades, session_from_path, split_name
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.evaluate_timeaware_action_filters import FILTERS
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_autoresearch_001"
MIN_SELECTION_TRADES = 12


@dataclass(frozen=True)
class AutoresearchTrial:
    """One pre-registered neural trading trial."""

    name: str
    time_filter: str
    allowed_buckets: tuple[str, ...]
    min_edge_vs_no_trade: float
    max_trades_per_day: int
    daily_loss_stop: float | None

    @property
    def config_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def registered_trials() -> tuple[AutoresearchTrial, ...]:
    """Small fixed trial surface for the first loop."""
    specs = [
        ("all_times_edge0_max4", "all_times", 0.0, 4, None),
        ("skip_first30_edge0_max4_stop1000", "skip_first_30", 0.0, 4, -1_000.0),
        ("post_open_late_edge0_max4_stop1000", "post_open_and_late", 0.0, 4, -1_000.0),
        ("late_afternoon_edge0_max2", "late_afternoon_only", 0.0, 2, None),
        ("post_open_late_edge25_max4", "post_open_and_late", 25.0, 4, None),
        ("late_afternoon_edge25_max2", "late_afternoon_only", 25.0, 2, None),
    ]
    return tuple(
        AutoresearchTrial(
            name=name,
            time_filter=time_filter,
            allowed_buckets=tuple(FILTERS[time_filter]),
            min_edge_vs_no_trade=float(edge),
            max_trades_per_day=int(max_trades),
            daily_loss_stop=daily_stop,
        )
        for name, time_filter, edge, max_trades, daily_stop in specs
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_autoresearch_001"))
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    p.add_argument("--policy-indexes", nargs="*", type=int, default=[0, 1, 2], choices=sorted(POLICY_META))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--loss-mode", choices=ACTION_LOSS_MODES, default="huber")
    return p.parse_args()


def _paths_by_split(data_dir: Path) -> dict[str, list[Path]]:
    out = {"train": [], "validation": [], "test": []}
    for path in sorted(data_dir.glob("*.pkl")):
        out[split_name(session_from_path(path))].append(path)
    return out


def _top_day_profit_share(trades: Sequence[Trade]) -> float:
    by_day: dict[str, float] = {}
    for trade in trades:
        by_day[trade.session] = by_day.get(trade.session, 0.0) + trade.pnl
    positives = np.asarray([v for v in by_day.values() if v > 0], dtype=float)
    if positives.sum() <= 0:
        return 1.0
    return float(positives.max() / positives.sum())


def _selection_reward(metrics: dict) -> float:
    """Validation-only score that resists common backtest hacks."""
    trades = float(metrics["trades"])
    if trades < MIN_SELECTION_TRADES:
        return -1_000_000.0 + trades
    profit_factor = float(metrics["profit_factor"])
    if not np.isfinite(profit_factor):
        profit_factor = 5.0
    profit_factor = min(profit_factor, 5.0)
    top_day_share = float(metrics.get("top_day_profit_share", 1.0))
    max_drawdown = float(metrics["max_drawdown"])
    positive_days = float(metrics["positive_day_fraction"])
    total_pnl = float(metrics["total_pnl"])
    return (
        total_pnl
        + 1_000.0 * (profit_factor - 1.0)
        + 1_500.0 * (positive_days - 0.50)
        + 0.15 * max_drawdown
        - 1_500.0 * max(0.0, top_day_share - 0.45)
    )


def selection_key(row: dict) -> tuple[float, float, float, float, float]:
    metrics = row["selection_metrics"]
    profit_factor = float(metrics["profit_factor"])
    if not np.isfinite(profit_factor):
        profit_factor = 5.0
    return (
        float(row["selection_reward"]),
        float(metrics["total_pnl"]),
        min(profit_factor, 5.0),
        float(metrics["positive_day_fraction"]),
        -float(metrics.get("top_day_profit_share", 1.0)),
    )


def simulate_true_no_trade_policy(
    decisions: Sequence[ActionDecision],
    predictions: np.ndarray,
    *,
    trial: AutoresearchTrial,
    cooldown_minutes: int,
    strategy: str,
) -> list[Trade]:
    """Let the neural no-trade output compete directly with call/put actions."""
    trades: list[Trade] = []
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    pnl_by_session: dict[str, float] = {}
    halted_sessions: set[str] = set()
    allowed = set(trial.allowed_buckets)

    for decision, pred in zip(decisions, predictions):
        if time_bucket(decision.decision_time) not in allowed:
            continue
        if decision.session in halted_sessions:
            continue
        if trades_by_session.get(decision.session, 0) >= trial.max_trades_per_day:
            continue
        next_time = next_time_by_session.get(decision.session)
        if next_time is not None and decision.decision_time < next_time:
            continue
        if len(pred) < 3 or not np.isfinite(pred).all():
            continue

        action = int(np.argmax(pred))
        if action == 0:
            continue
        edge_vs_no_trade = float(pred[action] - pred[0])
        if edge_vs_no_trade < trial.min_edge_vs_no_trade:
            continue
        pnl = float(decision.labels[action])
        if not np.isfinite(pnl):
            continue

        trades.append(
            Trade(
                session=decision.session,
                decision_time=decision.decision_time.isoformat(),
                pnl=pnl,
                score=edge_vs_no_trade,
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
        if trial.daily_loss_stop is not None and pnl_by_session[decision.session] <= trial.daily_loss_stop:
            halted_sessions.add(decision.session)

    return trades


def _metrics(trades: Sequence[Trade]) -> dict:
    metrics = metrics_with_concentration(trades)
    metrics["selection_reward"] = _selection_reward(metrics)
    return metrics


def _run_one_model(
    *,
    paths: dict[str, list[Path]],
    policy_index: int,
    seed: int,
    epochs: int,
    batch_size: int,
    loss_mode: str,
    trials: Sequence[AutoresearchTrial],
) -> list[dict]:
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
    split_decisions = {
        split: load_action_decisions(files, policy_index=policy_index)
        for split, files in paths.items()
    }
    calibration_decisions, selection_decisions = split_validation_by_session(
        split_decisions["validation"]
    )
    model, scaler, history = train_action_model(
        split_decisions["train"],
        calibration_decisions,
        config=config,
        loss_mode=loss_mode,
    )
    decision_sets = {
        "train": split_decisions["train"],
        "calibration": calibration_decisions,
        "validation_selection": selection_decisions,
        "test": split_decisions["test"],
    }
    predictions = {
        split: predict_actions(model, scaler, decisions, target_scale=config.target_scale)
        for split, decisions in decision_sets.items()
    }

    rows = []
    for trial in trials:
        metrics_by_split = {}
        for split, decisions in decision_sets.items():
            trades = simulate_true_no_trade_policy(
                decisions,
                predictions[split],
                trial=trial,
                cooldown_minutes=cooldown,
                strategy=f"{LOOP_ID}:{trial.name}",
            )
            metrics_by_split[split] = _metrics(trades)
        rows.append(
            {
                "loop_id": LOOP_ID,
                "policy_index": policy_index,
                "policy_name": policy_name,
                "seed": seed,
                "loss_mode": loss_mode,
                "feature_version": action_feature_version(loss_mode),
                "trial": asdict(trial) | {"config_id": trial.config_id},
                "best_epoch": next((x["epoch"] for x in history if x["is_best"]), None),
                "split_counts": {
                    split: len(decisions)
                    for split, decisions in decision_sets.items()
                },
                "selection_metrics": metrics_by_split["validation_selection"],
                "audit_test_metrics": metrics_by_split["test"],
                "metrics_by_split": metrics_by_split,
                "selection_reward": _selection_reward(metrics_by_split["validation_selection"]),
            }
        )
    return rows


def aggregate_results(results: Sequence[dict]) -> dict:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in results:
        key = (int(row["policy_index"]), str(row["trial"]["name"]))
        groups.setdefault(key, []).append(row)

    by_trial = {}
    for (policy_index, trial_name), rows in sorted(groups.items()):
        selection = [r["selection_metrics"] for r in rows]
        audit = [r["audit_test_metrics"] for r in rows]
        rewards = np.asarray([r["selection_reward"] for r in rows], dtype=float)
        row = {
            "policy_index": policy_index,
            "policy_name": rows[0]["policy_name"],
            "trial_name": trial_name,
            "runs": len(rows),
            "trial": rows[0]["trial"],
            "selection_reward_median": float(np.median(rewards)),
            "selection_pnl_median": float(np.median([m["total_pnl"] for m in selection])),
            "selection_profit_factor_median": float(np.median([m["profit_factor"] for m in selection])),
            "selection_positive_day_fraction_median": float(
                np.median([m["positive_day_fraction"] for m in selection])
            ),
            "selection_trades_median": float(np.median([m["trades"] for m in selection])),
            "selection_top_day_profit_share_median": float(
                np.median([m["top_day_profit_share"] for m in selection])
            ),
            "audit_test_pnl_median": float(np.median([m["total_pnl"] for m in audit])),
            "audit_test_profit_factor_median": float(np.median([m["profit_factor"] for m in audit])),
            "audit_test_max_drawdown_median": float(np.median([m["max_drawdown"] for m in audit])),
            "audit_test_trades_median": float(np.median([m["trades"] for m in audit])),
            "audit_test_positive_seed_fraction": float(
                np.mean([m["total_pnl"] > 0 for m in audit])
            ),
            "audit_test_positive_day_fraction_median": float(
                np.median([m["positive_day_fraction"] for m in audit])
            ),
            "audit_test_top_day_profit_share_median": float(
                np.median([m["top_day_profit_share"] for m in audit])
            ),
        }
        row["passes_research_selection_floor"] = bool(
            row["selection_trades_median"] >= MIN_SELECTION_TRADES
            and row["selection_pnl_median"] > 0
            and row["selection_profit_factor_median"] >= 1.05
            and row["selection_positive_day_fraction_median"] >= 0.45
            and row["selection_top_day_profit_share_median"] <= 0.75
        )
        by_trial[f"policy{policy_index}:{trial_name}"] = row

    ranked = sorted(by_trial.values(), key=lambda row: row["selection_reward_median"], reverse=True)
    eligible = [row for row in ranked if row["passes_research_selection_floor"]]
    champion = eligible[0] if eligible else (ranked[0] if ranked else None)
    return {
        "selection_basis": (
            "champion selected from February validation-selection floor-eligible trials "
            "by validation reward only; if none are eligible, the top ineligible trial is reported"
        ),
        "eligible_trial_count": len(eligible),
        "champion": champion,
        "ranked_trials": ranked,
        "by_trial": by_trial,
    }


def _write_report(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    aggregate = payload["aggregate"]
    champion = aggregate["champion"]
    lines = [
        f"# {LOOP_ID} Autoresearch Report",
        "",
        payload["framing"],
        "",
        f"Action model objective: `{payload['training_objective']['feature_version']}`",
        "",
        "## Anti-Reward-Hacking Contract",
        "",
    ]
    for item in payload["anti_reward_hacking_contract"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Champion",
        "",
    ]
    if champion is None:
        lines.append("No champion was selected.")
    else:
        floor_status = (
            "floor-cleared"
            if champion["passes_research_selection_floor"]
            else "top ineligible"
        )
        lines += [
            f"Selection-only champion: **policy{champion['policy_index']} / {champion['trial_name']}** ({floor_status})",
            "",
            "| Metric | Selection | March Audit |",
            "|---|---:|---:|",
            f"| Median PnL | {champion['selection_pnl_median']:.0f} | {champion['audit_test_pnl_median']:.0f} |",
            f"| Median PF | {champion['selection_profit_factor_median']:.3f} | {champion['audit_test_profit_factor_median']:.3f} |",
            f"| Median Trades | {champion['selection_trades_median']:.0f} | {champion['audit_test_trades_median']:.0f} |",
            f"| Positive Day Fraction | {champion['selection_positive_day_fraction_median']:.2f} | {champion['audit_test_positive_day_fraction_median']:.2f} |",
            f"| Top-Day Share | {champion['selection_top_day_profit_share_median']:.2f} | {champion['audit_test_top_day_profit_share_median']:.2f} |",
        ]
    lines += [
        "",
        "## Ranked Trials",
        "",
        "| Rank | Policy | Trial | Sel Reward | Sel PnL | Sel PF | Sel Trades | Audit PnL | Audit PF | Audit DD | Audit Positive Seeds |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(aggregate["ranked_trials"], start=1):
        lines.append(
            f"| {idx} | {row['policy_name']} | {row['trial_name']} | "
            f"{row['selection_reward_median']:.0f} | {row['selection_pnl_median']:.0f} | "
            f"{row['selection_profit_factor_median']:.3f} | {row['selection_trades_median']:.0f} | "
            f"{row['audit_test_pnl_median']:.0f} | {row['audit_test_profit_factor_median']:.3f} | "
            f"{row['audit_test_max_drawdown_median']:.0f} | "
            f"{row['audit_test_positive_seed_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "This is a research-loop result, not a live-trading approval and not a broad "
            "data-purchase approval. March is shown as an audit result only; the champion "
            "above was selected without using March."
        ),
    ]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(json_path)
    print(md_path)


def main() -> int:
    args = parse_args()
    paths = _paths_by_split(args.data_dir)
    trials = registered_trials()
    all_results = []
    for policy_index in args.policy_indexes:
        for seed in args.seeds:
            print(f"autoresearch loop={LOOP_ID} policy={policy_index} seed={seed}", flush=True)
            all_results.extend(
                _run_one_model(
                    paths=paths,
                    policy_index=policy_index,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    loss_mode=args.loss_mode,
                    trials=trials,
                )
            )

    aggregate = aggregate_results(all_results)
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Bounded autoresearch loop for a neural SPXW 0DTE options action model. "
            "The neural net chooses no-trade/call/put, and every executed trade is one "
            "long option contract with existing ask-entry/bid-exit labels. No paid data "
            "is downloaded by this loop."
        ),
        "program": "v4/autoresearch/program.md",
        "training_objective": {
            "feature_version": action_feature_version(args.loss_mode),
            "loss_mode": args.loss_mode,
            "loss_config": ACTION_DECISION_LOSS_CONFIG if args.loss_mode == "decision_aware" else None,
        },
        "anti_reward_hacking_contract": [
            "Trials are pre-registered before execution.",
            "Every trial is logged; failures are not deleted.",
            "Selection reward uses only the second half of February validation sessions.",
            "March metrics are audit-only and cannot select a champion.",
            "Too-few-trade trials receive a large selection penalty.",
            "Top-day profit concentration is penalized.",
            "v4 CBBO ask-entry/bid-exit labels remain the executable PnL truth.",
            "No paid data download is allowed inside the loop.",
        ],
        "split_protocol": {
            "train": "January 2026",
            "calibration": "first half of February 2026 sessions",
            "selection": "second half of February 2026 sessions",
            "audit_test": "March 2026",
        },
        "args": {
            "data_dir": str(args.data_dir),
            "out_dir": str(args.out_dir),
            "seeds": args.seeds,
            "policy_indexes": args.policy_indexes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss_mode": args.loss_mode,
        },
        "registered_trials": [asdict(trial) | {"config_id": trial.config_id} for trial in trials],
        "aggregate": aggregate,
        "all_results": all_results,
    }
    _write_report(args.out_dir, payload)
    print(json.dumps(aggregate["champion"], indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
