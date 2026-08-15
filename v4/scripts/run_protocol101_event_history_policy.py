"""Protocol 101: sequential event policy with causal short-history features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from v4.model.serial_opportunity import ENTRY_FEATURE_COLUMNS, strict_serial_baseline
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS
from v4.scripts.run_protocol097_sequential_event_policy import (
    DEFAULT_PROTOCOL092_DIR,
    EventPolicyConfig,
    _aggregate_gate,
    _candidate_frame_from_events,
    _compare_to_protocol092,
    _event_summary,
    _json_dumps,
    _reported_event_slices,
    add_oracle_actions,
    build_events,
    select_margin_threshold,
    simulate_event_policy,
    train_event_policy,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
MODEL_SEEDS = [1, 2, 3, 4, 5]
HISTORY_FEATURE_COLUMNS = [
    "hist_events_seen",
    "hist_minutes_since_prev_event",
    "hist_prev_candidate_count",
    "hist_prev_max_edge",
    "hist_prev_mean_edge",
    "hist_prev_max_gamma",
    "hist_prev_mean_theta_burden",
    "hist_prev_min_spread_over_mid",
    "hist_prev_call_count",
    "hist_prev_put_count",
    "hist_prev_call_minus_put_edge",
    "hist_roll3_candidate_count_mean",
    "hist_roll3_max_edge",
    "hist_roll3_mean_edge",
    "hist_roll3_max_gamma",
    "hist_roll3_mean_theta_burden",
    "hist_roll3_min_spread_over_mid",
    "hist_roll3_call_minus_put_edge",
]
FEATURE_COLUMNS = ENTRY_FEATURE_COLUMNS + HISTORY_FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.protocol092_dir / "serial_opportunity_dataset.parquet")
    dataset["decision_dt"] = pd.to_datetime(dataset["decision_time"], utc=True)
    dataset["candidate_exit_dt"] = pd.to_datetime(dataset["candidate_exit_time"], utc=True)
    events = build_events(dataset)
    add_causal_history_features(events)
    oracle_summary = add_oracle_actions(events)
    config = EventPolicyConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        min_validation_trades=int(args.min_validation_trades),
    )

    fold_results: list[dict[str, Any]] = []
    trade_ledgers: list[dict[str, Any]] = []
    for fold in FOLDS:
        train_events = [event for event in events if event["split"] in set(fold["train_splits"])]
        validation_all = [event for event in events if event["split"] == fold["validation_split"]]
        for seed in args.seeds:
            model, scaler, history = train_event_policy(
                train_events,
                validation_all,
                seed=int(seed),
                config=config,
                feature_columns=FEATURE_COLUMNS,
            )
            model_dir = args.out_dir / "model_artifacts" / fold["name"] / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "model.pt")
            (model_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")
            (model_dir / "manifest.json").write_text(
                _json_dumps(
                    {
                        "protocol": "101_event_history_policy",
                        "fold": fold["name"],
                        "seed": int(seed),
                        "feature_columns": FEATURE_COLUMNS,
                        "history_feature_columns": HISTORY_FEATURE_COLUMNS,
                        "config": config.__dict__,
                        "history": history,
                    }
                )
            )

            validation_seed = [event for event in validation_all if int(event["seed"]) == int(seed)]
            threshold = select_margin_threshold(
                validation_seed,
                model,
                scaler,
                seed=int(seed),
                config=config,
                feature_columns=FEATURE_COLUMNS,
            )
            seed_result = {
                "fold": fold["name"],
                "seed": int(seed),
                "train_splits": fold["train_splits"],
                "validation_split": fold["validation_split"],
                "test_split": fold["test_split"],
                "threshold": float(threshold["threshold"]),
                "threshold_selection": threshold,
                "history_last": history[-1] if history else {},
                "splits": {},
            }
            for split_name, event_slice in _reported_event_slices(events, fold, seed).items():
                candidate_source = _candidate_frame_from_events(event_slice)
                base = simulate_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.0,
                    strategy=f"protocol101_{fold['name']}",
                    feature_columns=FEATURE_COLUMNS,
                )
                stress10 = simulate_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.10,
                    strategy=f"protocol101_{fold['name']}_stress10",
                    feature_columns=FEATURE_COLUMNS,
                )
                stress25 = simulate_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.25,
                    strategy=f"protocol101_{fold['name']}_stress25",
                    feature_columns=FEATURE_COLUMNS,
                )
                baseline = strict_serial_baseline(candidate_source, seed=int(seed), slippage_per_side=0.0)
                baseline10 = strict_serial_baseline(candidate_source, seed=int(seed), slippage_per_side=0.10)
                seed_result["splits"][split_name] = {
                    "model": base.summary,
                    "model_stress_0_10": stress10.summary,
                    "model_stress_0_25": stress25.summary,
                    "strict_serial_baseline": baseline.summary,
                    "strict_serial_baseline_stress_0_10": baseline10.summary,
                    "validation_threshold_source": fold["validation_split"],
                }
                for trade in base.trades:
                    row = dict(trade)
                    row["fold"] = fold["name"]
                    row["reported_split"] = split_name
                    trade_ledgers.append(row)
            fold_results.append(seed_result)

    aggregate = _aggregate_gate(fold_results)
    protocol092 = json.loads((args.protocol092_dir / "summary.json").read_text())
    protocol097 = json.loads(Path("v4/audit/autoresearch/v4_aplus_hypothesis_097_sequential_event_policy/summary.json").read_text())
    payload = {
        "protocol": "101_event_history_policy",
        "paid_data_downloaded": False,
        "live_orders": False,
        "source_protocol092_dir": str(args.protocol092_dir),
        "pre_registration": _pre_registration(),
        "feature_columns": FEATURE_COLUMNS,
        "history_feature_columns": HISTORY_FEATURE_COLUMNS,
        "config": config.__dict__,
        "event_summary": _event_summary(events),
        "oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate_gate": aggregate,
        "protocol092_comparison": _compare_to_protocol092(aggregate, protocol092["aggregate_gate"]),
        "protocol097_comparison": _compare_to_protocol092(aggregate, protocol097["aggregate_gate"]),
        "decision": _decision(aggregate, protocol092["aggregate_gate"], protocol097["aggregate_gate"]),
    }
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    (args.out_dir / "serial_policy_trades.json").write_text(_json_dumps(trade_ledgers))
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "vs_097": payload["protocol097_comparison"], "vs_092": payload["protocol092_comparison"]}, indent=2, sort_keys=True))
    print(args.out_dir / "report.md")
    return 0


def add_causal_history_features(events: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault((str(event["split"]), int(event["seed"]), str(event["session"])), []).append(event)
    for _, session_events in grouped.items():
        session_events.sort(key=lambda event: event["decision_dt"])
        summaries: list[dict[str, float]] = []
        previous_time = None
        for idx, event in enumerate(session_events):
            previous = summaries[-1] if summaries else _empty_summary()
            rolling = summaries[-3:]
            history = _history_features(
                events_seen=idx,
                minutes_since_prev=_minutes_between(previous_time, event["decision_dt"]),
                previous=previous,
                rolling=rolling,
            )
            candidates = event["candidates"].copy()
            for column, value in history.items():
                candidates[column] = float(value)
            event["candidates"] = candidates
            summaries.append(_event_candidate_summary(candidates))
            previous_time = event["decision_dt"]


def _event_candidate_summary(candidates: pd.DataFrame) -> dict[str, float]:
    calls = candidates[candidates["right"] == "C"]
    puts = candidates[candidates["right"] == "P"]
    max_call_edge = float(calls["edge"].max()) if not calls.empty else 0.0
    max_put_edge = float(puts["edge"].max()) if not puts.empty else 0.0
    return {
        "candidate_count": float(len(candidates)),
        "max_edge": float(candidates["edge"].max()),
        "mean_edge": float(candidates["edge"].mean()),
        "max_gamma": float(candidates["entry_gamma"].max()),
        "mean_theta_burden": float(candidates["entry_theta_burden"].mean()),
        "min_spread_over_mid": float(candidates["entry_spread_over_mid"].min()),
        "call_count": float(len(calls)),
        "put_count": float(len(puts)),
        "call_minus_put_edge": float(max_call_edge - max_put_edge),
    }


def _history_features(
    *,
    events_seen: int,
    minutes_since_prev: float,
    previous: dict[str, float],
    rolling: list[dict[str, float]],
) -> dict[str, float]:
    if not rolling:
        rolling = [_empty_summary()]
    return {
        "hist_events_seen": float(min(events_seen, 50)),
        "hist_minutes_since_prev_event": float(minutes_since_prev),
        "hist_prev_candidate_count": previous["candidate_count"],
        "hist_prev_max_edge": previous["max_edge"],
        "hist_prev_mean_edge": previous["mean_edge"],
        "hist_prev_max_gamma": previous["max_gamma"],
        "hist_prev_mean_theta_burden": previous["mean_theta_burden"],
        "hist_prev_min_spread_over_mid": previous["min_spread_over_mid"],
        "hist_prev_call_count": previous["call_count"],
        "hist_prev_put_count": previous["put_count"],
        "hist_prev_call_minus_put_edge": previous["call_minus_put_edge"],
        "hist_roll3_candidate_count_mean": _mean(rolling, "candidate_count"),
        "hist_roll3_max_edge": max(item["max_edge"] for item in rolling),
        "hist_roll3_mean_edge": _mean(rolling, "mean_edge"),
        "hist_roll3_max_gamma": max(item["max_gamma"] for item in rolling),
        "hist_roll3_mean_theta_burden": _mean(rolling, "mean_theta_burden"),
        "hist_roll3_min_spread_over_mid": min(item["min_spread_over_mid"] for item in rolling),
        "hist_roll3_call_minus_put_edge": _mean(rolling, "call_minus_put_edge"),
    }


def _empty_summary() -> dict[str, float]:
    return {
        "candidate_count": 0.0,
        "max_edge": 0.0,
        "mean_edge": 0.0,
        "max_gamma": 0.0,
        "mean_theta_burden": 0.0,
        "min_spread_over_mid": 0.0,
        "call_count": 0.0,
        "put_count": 0.0,
        "call_minus_put_edge": 0.0,
    }


def _minutes_between(previous, current) -> float:
    if previous is None:
        return 999.0
    return float((pd.Timestamp(current) - pd.Timestamp(previous)).total_seconds() / 60.0)


def _mean(rows: list[dict[str, float]], column: str) -> float:
    return float(sum(row[column] for row in rows) / max(len(rows), 1))


def _decision(current: dict[str, Any], protocol092: dict[str, Any], protocol097: dict[str, Any]) -> str:
    if current["promotion_ready"]:
        return "keep_promote_candidate: Protocol 101 clears the strict serial gate"
    vs_097 = _compare_to_protocol092(current, protocol097)
    q3 = vs_097["q3_2025"]["median_total_pnl_delta"] > 0
    q4 = vs_097["q4_2025"]["median_total_pnl_delta"] > 0
    if q3 and q4:
        return "keep_for_research_only: Protocol 101 improves Q3/Q4 versus Protocol 097 but still does not clear promotion"
    return "reject_and_pause_for_broader_data: short-history memory did not improve Q3/Q4 versus Protocol 097"


def _pre_registration() -> dict[str, Any]:
    return {
        "hypothesis": "Protocol 097's Q4 seed failure is caused by memoryless event decisions. Causal short-history summaries should help distinguish opportunity clusters and prevent under-trading post-open winners.",
        "single_change": "append causal previous-event and rolling-3 event summary features to each candidate",
        "features": "Protocol 092 entry-only features plus causal short-history state",
        "exits": "frozen Protocol 081 candidate exits",
        "paid_data": "forbidden",
        "live_orders": "forbidden",
        "folds": FOLDS,
        "seeds": MODEL_SEEDS,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 101: Event Policy With Short History",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Single change: `{payload['pre_registration']['single_change']}`",
        "",
        "## Gate",
        "",
        _table(
            [
                {
                    "split": split,
                    "median_pnl": payload["aggregate_gate"][split]["median_total_pnl"],
                    "pf": payload["aggregate_gate"][split]["median_profit_factor"],
                    "trades": payload["aggregate_gate"][split]["median_trades"],
                    "stress10": payload["aggregate_gate"][split]["median_stress_0_10_total_pnl"],
                    "stress25": payload["aggregate_gate"][split]["median_stress_0_25_total_pnl"],
                    "baseline": payload["aggregate_gate"][split]["strict_serial_baseline_median_total_pnl"],
                    "beats_baseline": payload["aggregate_gate"][split]["beats_strict_serial_baseline"],
                    "vs_protocol092": payload["protocol092_comparison"][split]["median_total_pnl_delta"],
                    "vs_protocol097": payload["protocol097_comparison"][split]["median_total_pnl_delta"],
                }
                for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]
            ],
            ["split", "median_pnl", "pf", "trades", "stress10", "stress25", "baseline", "beats_baseline", "vs_protocol092", "vs_protocol097"],
        ),
        "",
        "## Promotion Checks",
        "",
        _table(payload["aggregate_gate"]["promotion_checks"], ["split", "name", "value", "pass"]),
    ]
    path.write_text("\n".join(lines) + "\n")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
