"""EXP_2026_05_22_BLENDED_DOLLAR_PREMIUM_POLICY_V1.

Historically Protocol232. This is a single-change follow-up to the
return-on-premium policy: train the same full-action/history neural policy on a
blended utility that keeps raw dollar opportunity primary while penalizing
capital-inefficient premium use.

Historical replay still scores executable dollars with ask entry, bid exit, one
account, one contract, no overlap, and affordability enforced. No paid data is
downloaded and no broker endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.scripts.run_protocol165_full_action_space_policy import (
    DEFAULT_PROTOCOL101_SUMMARY,
    DEFAULT_RECENT_BASELINE,
    FOLDS,
    MAX_ACTION_CANDIDATES,
    STARTING_CASH,
    aggregate,
    fmt,
    load_dataset,
    load_protocol101_baselines,
    reported_slices,
    simulate_first_affordable,
    simulate_matched_random,
    simulate_oracle,
    smoke_folds,
)
from v4.scripts.run_protocol172_full_action_value_policy import build_events, summarize_events
import v4.scripts.run_protocol183_two_stage_full_action_policy as p183
import v4.scripts.run_protocol221_return_on_premium_policy as p221


ROLE_LABEL = "EXP_2026_05_22_BLENDED_DOLLAR_PREMIUM_POLICY_V1"
HISTORICAL_ID = "Protocol232"
DEFAULT_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_211_full_action_history_feature_repair/full_action_surface_edge_with_history.parquet"
)
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_232_blended_dollar_premium_policy")
FEATURE_COLUMNS = p221.FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--seeds", nargs="*", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--dollar-weight", type=float, default=0.65)
    parser.add_argument("--premium-weight", type=float, default=0.35)
    parser.add_argument("--dollar-scale", type=float, default=500.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    p183.set_active_feature_columns(FEATURE_COLUMNS, dataset)
    events = build_events(dataset, starting_cash=float(args.starting_cash))
    if args.smoke:
        events = p221.smoke_events(events, max_sessions=3)
    oracle_summary = add_blended_advantages(
        events,
        dollar_weight=float(args.dollar_weight),
        premium_weight=float(args.premium_weight),
        dollar_scale=float(args.dollar_scale),
    )
    config = p221.ReturnOnPremiumConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        target_scale=0.25,
        target_clip=3.0,
    )
    folds = smoke_folds(events) if args.smoke else FOLDS
    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    baseline_trades: list[dict[str, Any]] = []
    for fold in folds:
        train_events = [event for event in events if event["split"] in set(fold["train_splits"])]
        validation_events = [event for event in events if event["split"] == fold["validation_split"]]
        if not train_events or not validation_events:
            fold_results.append({"fold": fold["name"], "skipped": True, "reason": "missing_train_or_validation_events", "splits": {}})
            continue
        for seed in args.seeds:
            model, scaler, history = p221.train_policy(train_events, validation_events, seed=int(seed), config=config)
            threshold = select_threshold(validation_events, model, scaler, config=config, starting_cash=float(args.starting_cash))
            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            (model_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
            (model_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "role_label": ROLE_LABEL,
                        "historical_protocol": HISTORICAL_ID,
                        "protocol": "232_blended_dollar_premium_policy",
                        "fold": fold["name"],
                        "seed": int(seed),
                        "feature_columns": p183.ACTIVE_FEATURE_COLUMNS,
                        "max_action_candidates": MAX_ACTION_CANDIDATES,
                        "config": asdict(config),
                        "utility": {
                            "dollar_weight": float(args.dollar_weight),
                            "premium_weight": float(args.premium_weight),
                            "dollar_scale": float(args.dollar_scale),
                            "formula": "dollar_weight * candidate_pnl / dollar_scale + premium_weight * candidate_pnl / entry_premium",
                        },
                        "threshold_selection": threshold,
                        "history": history,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
            result = {"fold": fold["name"], "seed": int(seed), "threshold": float(threshold["threshold"]), "splits": {}}
            for split_name, split_events in reported_slices(events, fold).items():
                model_base = p183.simulate_model(
                    split_events,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                    strategy="blended_dollar_premium_policy",
                )
                model_10 = p183.simulate_model(
                    split_events,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.10,
                    starting_cash=float(args.starting_cash),
                    strategy="blended_dollar_premium_policy_stress10",
                )
                model_25 = p183.simulate_model(
                    split_events,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.25,
                    starting_cash=float(args.starting_cash),
                    strategy="blended_dollar_premium_policy_stress25",
                )
                first = simulate_first_affordable(split_events, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                random_base = simulate_matched_random(split_events, seed=int(seed), slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                oracle = simulate_oracle(split_events, slippage_per_side=0.0, starting_cash=float(args.starting_cash))
                result["splits"][split_name] = {
                    "model": p221.add_premium_efficiency(model_base.summary, model_base.trades),
                    "model_stress_0_10": p221.add_premium_efficiency(model_10.summary, model_10.trades),
                    "model_stress_0_25": p221.add_premium_efficiency(model_25.summary, model_25.trades),
                    "first_affordable_baseline": first.summary,
                    "matched_random_baseline": random_base.summary,
                    "full_action_oracle": oracle.summary,
                    "edge_only_baseline": {"status": "not_available_no_surface_edge_feature"},
                }
                model_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in model_base.trades)
                baseline_trades.extend({**trade, "fold": fold["name"], "seed": int(seed), "reported_split": split_name} for trade in first.trades)
            fold_results.append(result)

    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)
    model_trade_frame = pd.DataFrame(model_trades)
    if not model_trade_frame.empty:
        model_trade_frame.to_csv(args.out_dir / "model_trades.csv", index=False)
    pd.DataFrame(baseline_trades).to_csv(args.out_dir / "first_affordable_baseline_trades.csv", index=False)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "experiment / model change",
        "changes_paper_default": False,
        "candidate_label": "CHALLENGER_BLENDED_DOLLAR_PREMIUM_FULL_ACTION_V1",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "baseline_challenger_label": "CHALLENGER_RETURN_ON_PREMIUM_FULL_ACTION_V1",
        "data_used": str(args.dataset),
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": True,
        "utility": {
            "dollar_weight": float(args.dollar_weight),
            "premium_weight": float(args.premium_weight),
            "dollar_scale": float(args.dollar_scale),
        },
        "feature_columns": p183.ACTIVE_FEATURE_COLUMNS,
        "max_action_candidates": MAX_ACTION_CANDIDATES,
        "event_summary": summarize_events(events),
        "blended_oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate": aggregate(fold_results, frozen_protocol101),
        "frozen_protocol101_baselines": frozen_protocol101,
        "moneyness_profile": p221.summarize_trade_profile(model_trade_frame),
        "decision": "",
        "next_experiment": (
            "If this improves dollar edge while keeping premium efficiency, test account-aware sizing on its trade stream. "
            "If it fails, return to lifecycle/action-state modeling rather than adding another moneyness knob."
        ),
    }
    payload["decision"] = decide(payload)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def add_blended_advantages(
    events: list[dict[str, Any]],
    *,
    dollar_weight: float,
    premium_weight: float,
    dollar_scale: float,
) -> dict[str, Any]:
    total_weight = max(abs(dollar_weight) + abs(premium_weight), 1e-9)
    dollar_weight = float(dollar_weight) / total_weight
    premium_weight = float(premium_weight) / total_weight
    dollar_scale = max(float(dollar_scale), 1e-9)
    by_split: dict[str, dict[str, int]] = {}
    advantage_rows: list[dict[str, float | str]] = []
    for _, session_events in p221.group_events(events).items():
        session_events.sort(key=lambda event: event["decision_dt"])
        times = np.asarray([event["decision_dt"].value for event in session_events], dtype=np.int64)
        values = np.zeros(len(session_events) + 1, dtype=float)
        actions = np.zeros(len(session_events), dtype=np.int64)
        advantages_by_event: list[np.ndarray] = [np.zeros(0, dtype=np.float32) for _ in session_events]
        for idx in range(len(session_events) - 1, -1, -1):
            wait_value = values[idx + 1]
            candidates = session_events[idx]["candidates"]
            advantages = np.full(len(candidates), -1e9, dtype=np.float32)
            best_value = wait_value
            best_action = 0
            for local_idx, row in candidates.iterrows():
                if float(row.get("entry_affordable_10k", 0.0)) < 1.0:
                    continue
                exit_ns = pd.Timestamp(row["candidate_exit_dt"]).value
                if exit_ns <= session_events[idx]["decision_dt"].value:
                    continue
                premium = float(row.get("entry_premium", 0.0) or 0.0)
                if premium <= 0.0 or not math.isfinite(premium):
                    continue
                candidate_pnl = float(row.get("candidate_pnl", 0.0) or 0.0)
                utility = dollar_weight * (candidate_pnl / dollar_scale) + premium_weight * (candidate_pnl / premium)
                next_idx = int(np.searchsorted(times, exit_ns, side="left"))
                take_value = utility + values[next_idx]
                advantages[int(local_idx)] = float(take_value - wait_value)
                if take_value > best_value:
                    best_value = take_value
                    best_action = int(local_idx) + 1
            values[idx] = best_value
            actions[idx] = best_action
            advantages_by_event[idx] = advantages
        for idx, event in enumerate(session_events):
            event["rop_oracle_action"] = int(actions[idx])
            event["rop_oracle_advantages"] = advantages_by_event[idx]
            event["rop_oracle_best_advantage"] = float(np.max(advantages_by_event[idx])) if len(advantages_by_event[idx]) else -1e9
            split = str(event["split"])
            item = by_split.setdefault(split, {"take": 0, "wait": 0})
            item["take" if int(actions[idx]) else "wait"] += 1
            finite_adv = advantages_by_event[idx][np.isfinite(advantages_by_event[idx]) & (advantages_by_event[idx] > -1e8)]
            if len(finite_adv):
                advantage_rows.append(
                    {
                        "split": split,
                        "max_blended_advantage": float(np.max(finite_adv)),
                        "positive_candidate_fraction": float((finite_adv > 0).mean()),
                    }
                )
    if not advantage_rows:
        return {"by_split": by_split, "advantage_summary": {}}
    advantage_frame = pd.DataFrame(advantage_rows)
    return {
        "by_split": by_split,
        "advantage_summary": {
            split: {
                "median_max_blended_advantage": float(group["max_blended_advantage"].median()),
                "mean_positive_candidate_fraction": float(group["positive_candidate_fraction"].mean()),
            }
            for split, group in advantage_frame.groupby("split")
        },
    }


def select_threshold(
    events: list[dict[str, Any]],
    model: p183.TwoStageFullActionPolicy,
    scaler: Any,
    *,
    config: p221.ReturnOnPremiumConfig,
    starting_cash: float,
) -> dict[str, Any]:
    scores = p183.event_scores(events, model, scaler)
    finite_scores = scores[np.isfinite(scores)]
    thresholds = [float("inf")] if len(finite_scores) == 0 else sorted(
        set(np.quantile(finite_scores, [0, .1, .2, .35, .5, .65, .8, .9, .95]).round(4).tolist() + [0.0, float(finite_scores.min()) - 1e-3])
    )
    first = simulate_first_affordable(events, slippage_per_side=0.0, starting_cash=starting_cash)
    sweep = []
    for threshold in thresholds:
        base = p183.simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.0, starting_cash=starting_cash, strategy="blended_validation")
        stress = p183.simulate_model(events, model, scaler, threshold=float(threshold), slippage_per_side=0.10, starting_cash=starting_cash, strategy="blended_validation_stress10")
        base_summary = p221.add_premium_efficiency(base.summary, base.trades)
        stress_summary = p221.add_premium_efficiency(stress.summary, stress.trades)
        sweep.append(
            {
                "threshold": float(threshold),
                "model": base_summary,
                "model_stress_0_10": stress_summary,
                "first_affordable_baseline": first.summary,
                "delta_vs_first": float(base.summary["total_pnl"] - first.summary["total_pnl"]),
            }
        )
    eligible = [
        row
        for row in sweep
        if row["model"]["trades"] >= config.min_validation_trades
        and row["model"]["total_pnl"] > config.min_validation_dollar_pnl
        and row["model_stress_0_10"]["total_pnl"] > 0.0
    ]
    pool = eligible if eligible else sweep
    best = max(
        pool,
        key=lambda row: (
            row["model_stress_0_10"]["total_pnl"],
            row["model_stress_0_10"].get("pnl_per_premium", -1e9),
            row["delta_vs_first"],
            row["model"]["profit_factor"],
        ),
    )
    return {
        "threshold": float(best["threshold"]),
        "objective": "validation stress_0_10 dollar PnL first, then premium efficiency",
        "selected": best,
        "sweep": sweep,
    }


def decide(payload: dict[str, Any]) -> str:
    aggregate_payload = payload.get("aggregate", {})
    if aggregate_payload.get("promotion_ready"):
        return "research_candidate_survives_frozen_protocol101_gate_with_blended_dollar_premium_training"
    return "research_only_blended_dollar_premium_objective_did_not_clear_frozen_protocol101_gate"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: {payload['candidate_label']}",
        f"Paper default baseline: {payload['paper_default_label']}",
        f"Baseline challenger: {payload['baseline_challenger_label']}",
        f"Data used: {payload['data_used']}",
        f"Paid data downloaded: {payload['paid_data_downloaded_by_runner']}",
        f"Broker endpoint called: {payload['broker_endpoint_called']}",
        f"Next experiment: {payload['next_experiment']}",
        "",
        f"Historical ID: `{payload['historical_protocol']}`",
        f"Decision: `{payload['decision']}`",
        "",
        "## Utility",
        "",
        f"- Dollar weight: {payload['utility']['dollar_weight']:.2f}",
        f"- Premium weight: {payload['utility']['premium_weight']:.2f}",
        f"- Dollar scale: ${payload['utility']['dollar_scale']:.0f}",
        "",
        "## Aggregate Dollar Replay",
        "",
        "| split | seeds | median PnL | frozen Protocol101 | delta | PF | stress 0.10 | pnl/premium | median premium | trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in payload["aggregate"].items():
        if not isinstance(item, dict) or item.get("seeds", 0) == 0:
            continue
        eff = split_efficiency(payload["fold_results"], split)
        lines.append(
            f"| {split} | {item['seeds']} | {fmt(item['median_total_pnl'])} | "
            f"{fmt(item.get('frozen_protocol101_total_pnl'))} | {fmt(item.get('median_delta_vs_frozen_protocol101'))} | "
            f"{fmt(item['median_profit_factor'])} | {fmt(item['median_stress_0_10_total_pnl'])} | "
            f"{eff['pnl_per_premium']:.4f} | {eff['median_entry_premium']:.0f} | {fmt(item['median_trades'])} |"
        )
    lines.extend(["", "## Moneyness Profile", ""])
    for row in payload.get("moneyness_profile", {}).get("ladder_moneyness", []):
        lines.append(
            f"- {row['ladder_moneyness']}: {row['trades']} trades, "
            f"PnL ${float(row['pnl']):,.0f}, median premium ${float(row['median_premium']):,.0f}"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Model trades: `{path.parent / 'model_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def split_efficiency(fold_results: list[dict[str, Any]], split: str) -> dict[str, float]:
    rows = []
    for result in fold_results:
        if result.get("skipped") or split not in result.get("splits", {}):
            continue
        rows.append(result["splits"][split]["model"])
    if not rows:
        return {"pnl_per_premium": 0.0, "median_entry_premium": 0.0}
    return {
        "pnl_per_premium": float(np.median([row.get("pnl_per_premium", 0.0) for row in rows])),
        "median_entry_premium": float(np.median([row.get("median_entry_premium", 0.0) for row in rows])),
    }


if __name__ == "__main__":
    raise SystemExit(main())
