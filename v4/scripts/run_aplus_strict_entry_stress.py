"""Run a fixed one-trade-per-day stress test for the Protocol 003 A+ champion.

This script does not search a new trial grid. It takes the Protocol 003 winning
model family and applies one fixed, stricter execution rule:

* same A+ surface model
* same train/calibration/selection/March/Q4 split
* one trade maximum per session
* no extra score threshold

The point is to test whether the A+ neural lead survives when it must be more
selective, not to tune another set of knobs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from v4.model.environment_diagnostics import time_bucket
from v4.model.hypothesis_protocol import (
    MarketStructureCache,
    ProtocolTrial,
    SurfaceVariant,
    bootstrap_trade_pnl,
    predict_surface_actions,
    stress_trades,
    summarize_random_baseline,
    token_feature_names,
    train_surface_model,
    window_seed,
)
from v4.model.supervised_pilot import PilotConfig, Trade
from v4.scripts.evaluate_calibrated_abstention_signal import split_validation_by_session
from v4.scripts.evaluate_risk_controlled_purchase_signal import metrics_with_concentration
from v4.scripts.run_aplus_neural_protocol import (
    _load_surface_decisions_cached,
    _paths_by_split,
    _protocol_window,
)
from v4.scripts.train_spxw_supervised_pilot import POLICY_META


LOOP_ID = "v4_aplus_strict_entry_stress_003b"
STRICT_TRIAL = ProtocolTrial(
    name="fixed_all_times_edge0_max1",
    allowed_buckets=("first_30", "post_open_morning", "midday", "late_afternoon"),
    min_edge_vs_no_trade=0.0,
    max_trades_per_day=1,
    daily_loss_stop=None,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    parser.add_argument("--q4-data-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_q4_2025"))
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/v4_aplus_strict_entry_stress_003b"))
    parser.add_argument("--decision-cache-dir", type=Path, default=Path("data/cache/v4_aplus_surface_decisions"))
    parser.add_argument("--no-decision-cache", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--policy-index", type=int, default=1, choices=sorted(POLICY_META))
    parser.add_argument("--variant-name", default="surface_structure_aplus_huber")
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--min-edge-vs-no-trade", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def _variant_from_name(name: str) -> SurfaceVariant:
    variants = {
        "surface_structure_aplus_huber": SurfaceVariant(
            name="surface_structure_aplus_huber",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="huber",
        ),
        "surface_structure_aplus_multitask": SurfaceVariant(
            name="surface_structure_aplus_multitask",
            action_space="surface",
            market_mode="structure",
            token_mode="aplus",
            loss_mode="aplus_multitask",
        ),
    }
    if name not in variants:
        raise SystemExit(f"unknown strict stress variant: {name}")
    return variants[name]


def _trial_from_args(args: argparse.Namespace) -> ProtocolTrial:
    return ProtocolTrial(
        name=f"fixed_all_times_edge{int(args.min_edge_vs_no_trade)}_max{args.max_trades_per_day}",
        allowed_buckets=STRICT_TRIAL.allowed_buckets,
        min_edge_vs_no_trade=float(args.min_edge_vs_no_trade),
        max_trades_per_day=int(args.max_trades_per_day),
        daily_loss_stop=None,
    )


def _simulate_with_contracts(
    decisions: Sequence,
    predictions: np.ndarray,
    *,
    trial: ProtocolTrial,
    cooldown_minutes: int,
    strategy: str,
    feature_names: Sequence[str],
) -> tuple[list[Trade], list[dict]]:
    trades: list[Trade] = []
    trade_rows: list[dict] = []
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
        action_mask = np.concatenate([[True], decision.token_mask])
        masked = np.asarray(pred, dtype=float).copy()
        masked[~action_mask] = -np.inf
        if not np.isfinite(masked).any():
            continue
        action = int(np.nanargmax(masked))
        if action == 0:
            continue
        edge = float(masked[action] - masked[0])
        if not np.isfinite(edge) or edge < trial.min_edge_vs_no_trade:
            continue
        token_idx = action - 1
        pnl = float(decision.labels[token_idx])
        if not np.isfinite(pnl):
            continue
        trade = Trade(
            session=decision.session,
            decision_time=decision.decision_time.isoformat(),
            pnl=pnl,
            score=edge,
            right=str(decision.rights[token_idx]),
            offset=float(decision.offsets[token_idx]),
            strategy=strategy,
        )
        trades.append(trade)
        token_features = decision.token_features[token_idx]
        selected_feature_names = {
            "pattern_count_norm",
            "abs_delta",
            "delta_atr_capture",
            "convexity_per_premium",
            "theta_burden_hold",
            "spread_tax",
            "breakeven_atr",
            "gamma_theta_ratio_scaled",
            "contract_value_score",
            "worth_spread_flag",
            "obvious_overpay_flag",
        }
        selected_features = {
            f"feature_{name}": float(token_features[idx])
            for idx, name in enumerate(feature_names)
            if name in selected_feature_names and idx < len(token_features)
        }
        trade_rows.append(
            {
                **trade.__dict__,
                "contract_id": str(decision.contract_ids[token_idx]),
                "token_idx": int(token_idx),
                "flat_score": float(masked[0]),
                "action_score": float(masked[action]),
                "pattern_present": float(decision.pattern_targets[token_idx]),
                "worth_spread_target": float(decision.value_targets[token_idx]),
                **selected_features,
            }
        )
        trades_by_session[decision.session] = trades_by_session.get(decision.session, 0) + 1
        pnl_by_session[decision.session] = pnl_by_session.get(decision.session, 0.0) + pnl
        next_time_by_session[decision.session] = decision.decision_time + timedelta(minutes=cooldown_minutes)
        if trial.daily_loss_stop is not None and pnl_by_session[decision.session] <= trial.daily_loss_stop:
            halted_sessions.add(decision.session)
    return trades, trade_rows


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _summarize_split(seed_rows: list[dict], split: str) -> dict:
    metrics = [row["metrics_by_split"][split] for row in seed_rows]
    randoms = [row["random_baseline_by_split"][split] for row in seed_rows]
    pnl = [float(row["total_pnl"]) for row in metrics]
    pf = [float(row["profit_factor"]) for row in metrics]
    trades = [float(row["trades"]) for row in metrics]
    random_pnl = [float(row["total_pnl_median"]) for row in randoms]
    return {
        "pnl_by_seed": pnl,
        "pnl_median": _median(pnl),
        "profit_factor_by_seed": pf,
        "profit_factor_median": _median(pf),
        "trades_by_seed": trades,
        "trades_median": _median(trades),
        "positive_seed_fraction": float(np.mean([x > 0 for x in pnl])) if pnl else 0.0,
        "random_pnl_by_seed": random_pnl,
        "random_pnl_median": _median(random_pnl),
        "edge_vs_random_median": _median(pnl) - _median(random_pnl),
    }


def _summarize_stress(seed_rows: list[dict], split: str) -> dict:
    out = {}
    for cost in ("25", "50", "100"):
        metrics = [row["slippage_stress_by_split"][split][cost] for row in seed_rows]
        pnl = [float(row["total_pnl"]) for row in metrics]
        pf = [float(row["profit_factor"]) for row in metrics]
        out[cost] = {
            "pnl_by_seed": pnl,
            "pnl_median": _median(pnl),
            "profit_factor_by_seed": pf,
            "profit_factor_median": _median(pf),
            "survives": bool(_median(pnl) > 0 and _median(pf) >= 1.05),
        }
    return out


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# A+ Strict Entry Stress 003B",
        "",
        payload["framing"],
        "",
        f"Fixed variant: `{payload['variant']['name']}`",
        f"Fixed policy: `policy{payload['policy_index']} / {payload['policy_name']}`",
        f"Fixed trial: `{payload['trial']['name']}`",
        f"Passes strict stress: `{payload['passes_strict_stress']}`",
        "",
        "## Split Summary",
        "",
        "| Split | Median PnL | Median PF | Trades | Positive Seeds | Random PnL | Edge vs Random |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("selection", "march", "q4"):
        row = payload["summary_by_split"][split]
        lines.append(
            f"| {split} | {row['pnl_median']:.0f} | {row['profit_factor_median']:.3f} | "
            f"{row['trades_median']:.0f} | {row['positive_seed_fraction']:.2f} | "
            f"{row['random_pnl_median']:.0f} | {row['edge_vs_random_median']:.0f} |"
        )
    lines += [
        "",
        "## Extra Cost Stress",
        "",
        "| Split | Extra Cost | Median PnL | Median PF | Survives |",
        "|---|---:|---:|---:|---|",
    ]
    for split in ("march", "q4"):
        for cost, row in payload["stress_by_split"][split].items():
            lines.append(
                f"| {split} | {cost} | {row['pnl_median']:.0f} | {row['profit_factor_median']:.3f} | {row['survives']} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    paths = _paths_by_split(args.data_dir)
    q4_paths = sorted(args.q4_data_dir.glob("*.pkl"))
    if not q4_paths:
        raise SystemExit(f"no Q4 pkl files found under {args.q4_data_dir}")
    window = _protocol_window(paths, q4_paths)
    variant = _variant_from_name(args.variant_name)
    policy_name, cooldown = POLICY_META[args.policy_index]
    trial = _trial_from_args(args)
    market_cache = MarketStructureCache()
    cache_dir = None if args.no_decision_cache else args.decision_cache_dir
    feature_names = token_feature_names(variant.token_mode)

    train_decisions = _load_surface_decisions_cached(
        paths["train"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="train",
        cache_dir=cache_dir,
    )
    validation_decisions = _load_surface_decisions_cached(
        paths["validation"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="validation",
        cache_dir=cache_dir,
    )
    calibration_decisions, selection_decisions = split_validation_by_session(validation_decisions)
    march_decisions = _load_surface_decisions_cached(
        paths["test"],
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="march",
        cache_dir=cache_dir,
    )
    q4_decisions = _load_surface_decisions_cached(
        q4_paths,
        policy_index=args.policy_index,
        variant=variant,
        market_cache=market_cache,
        split="q4",
        cache_dir=cache_dir,
    )
    decision_sets = {
        "selection": selection_decisions,
        "march": march_decisions,
        "q4": q4_decisions,
    }

    seed_rows = []
    selected_trades: dict[str, list[dict]] = {"selection": [], "march": [], "q4": []}
    for seed in args.seeds:
        effective_seed = window_seed(seed, window.window_id)
        config = PilotConfig(
            policy_index=args.policy_index,
            policy_name=policy_name,
            cooldown_minutes=cooldown,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=128,
            seed=effective_seed,
        )
        print(f"{LOOP_ID} policy={args.policy_index} variant={variant.name} seed={seed}", flush=True)
        model, standardizer, history = train_surface_model(
            train_decisions,
            calibration_decisions,
            config=config,
            variant=variant,
        )
        predictions = {
            split: predict_surface_actions(model, standardizer, decisions, target_scale=config.target_scale)
            for split, decisions in decision_sets.items()
        }
        metrics_by_split = {}
        random_baseline_by_split = {}
        slippage_stress_by_split = {}
        bootstrap_by_split = {}
        trade_counts = {}
        for split, decisions in decision_sets.items():
            trades, trade_rows = _simulate_with_contracts(
                decisions,
                predictions[split],
                trial=trial,
                cooldown_minutes=cooldown,
                strategy=f"{LOOP_ID}:{variant.name}:{trial.name}",
                feature_names=feature_names,
            )
            for row in trade_rows:
                selected_trades[split].append({"seed": int(seed), **row})
            metrics_by_split[split] = metrics_with_concentration(trades)
            trade_counts[split] = len(trades)
            random_baseline_by_split[split] = summarize_random_baseline(
                decisions,
                trial=trial,
                cooldown_minutes=cooldown,
                seed=effective_seed,
                target_trade_count=len(trades),
            )
            if split in {"march", "q4"}:
                bootstrap_by_split[split] = bootstrap_trade_pnl(trades, seed=effective_seed)
                slippage_stress_by_split[split] = {
                    str(extra_cost): metrics_with_concentration(
                        stress_trades(trades, extra_cost_per_trade=float(extra_cost))
                    )
                    for extra_cost in (25, 50, 100)
                }
        seed_rows.append(
            {
                "seed": int(seed),
                "effective_seed": int(effective_seed),
                "best_epoch": next((x["epoch"] for x in history if x["is_best"]), None),
                "history": history,
                "metrics_by_split": metrics_by_split,
                "random_baseline_by_split": random_baseline_by_split,
                "slippage_stress_by_split": slippage_stress_by_split,
                "bootstrap_by_split": bootstrap_by_split,
                "trade_counts": trade_counts,
            }
        )

    summary_by_split = {
        split: _summarize_split(seed_rows, split)
        for split in ("selection", "march", "q4")
    }
    stress_by_split = {
        split: _summarize_stress(seed_rows, split)
        for split in ("march", "q4")
    }
    passes_strict = bool(
        summary_by_split["march"]["pnl_median"] > 0
        and summary_by_split["march"]["profit_factor_median"] >= 1.05
        and summary_by_split["march"]["positive_seed_fraction"] >= 2 / 3
        and summary_by_split["q4"]["pnl_median"] > 0
        and summary_by_split["q4"]["profit_factor_median"] >= 1.05
        and summary_by_split["q4"]["positive_seed_fraction"] >= 2 / 3
        and stress_by_split["march"]["25"]["survives"]
        and stress_by_split["q4"]["25"]["survives"]
    )
    payload = {
        "loop_id": LOOP_ID,
        "framing": (
            "Fixed stricter stress test for the Protocol 003 A+ neural champion. "
            "This run does not choose among new trials; it asks whether the same "
            "model family survives a one-trade-per-day execution cap."
        ),
        "window": asdict(window) | {"window_id": window.window_id},
        "variant": asdict(variant) | {"variant_id": variant.variant_id},
        "policy_index": int(args.policy_index),
        "policy_name": policy_name,
        "trial": asdict(trial) | {"config_id": trial.config_id},
        "args": {
            "data_dir": str(args.data_dir),
            "q4_data_dir": str(args.q4_data_dir),
            "out_dir": str(args.out_dir),
            "decision_cache_dir": None if cache_dir is None else str(cache_dir),
            "seeds": args.seeds,
            "max_trades_per_day": args.max_trades_per_day,
            "min_edge_vs_no_trade": args.min_edge_vs_no_trade,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "summary_by_split": summary_by_split,
        "stress_by_split": stress_by_split,
        "passes_strict_stress": passes_strict,
        "interpretation": (
            "Passing this stress means the A+ neural lead can become more selective without disappearing. "
            "Failing it means the current champion still depends on taking too many marginal entries. "
            "Either way, this remains research evidence only."
        ),
        "seed_rows": seed_rows,
        "selected_trades": selected_trades,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    _write_markdown(md_path, payload)
    for split, rows in selected_trades.items():
        (args.out_dir / f"selected_trades_{split}.json").write_text(
            json.dumps(rows, indent=2, allow_nan=True) + "\n"
        )
    print(json_path)
    print(md_path)
    print(json.dumps({"passes_strict_stress": passes_strict, "summary_by_split": summary_by_split}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
