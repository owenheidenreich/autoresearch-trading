"""Protocol 109: frozen Protocol 101 seed ensemble.

This is a no-training stability check. It averages the logits from the five
frozen Protocol 101 seed artifacts, selects thresholds only on the registered
validation split, and evaluates whether seed ensembling reduces the abstention
fragility observed in Q4 2024 external attribution.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v4.model.serial_opportunity import CONTRACT_MULTIPLIER, serial_metrics, strict_serial_baseline
from v4.model.supervised_pilot import FeatureScaler
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS
from v4.scripts.run_protocol097_sequential_event_policy import (
    EventPolicyConfig,
    EventSetPolicy,
    MAX_CANDIDATES,
    _aggregate_gate,
    _candidate_frame_from_events,
    _group_events,
    _reported_event_slices,
    _summarize_split,
    _table,
    event_tensors,
)
from v4.scripts.run_protocol101_event_history_policy import FEATURE_COLUMNS, add_causal_history_features


DEFAULT_PROTOCOL092_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy")
DEFAULT_PROTOCOL101_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy")
DEFAULT_PROTOCOL107_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_107_protocol101_q4_2024_external_stress")
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_109_frozen_protocol101_seed_ensemble")
MODEL_SEEDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol092-dir", type=Path, default=DEFAULT_PROTOCOL092_DIR)
    parser.add_argument("--protocol101-dir", type=Path, default=DEFAULT_PROTOCOL101_DIR)
    parser.add_argument("--protocol107-dir", type=Path, default=DEFAULT_PROTOCOL107_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--artifact-seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args.protocol092_dir / "serial_opportunity_dataset.parquet")
    events = _build_history_events(dataset)
    config = EventPolicyConfig(min_validation_trades=int(args.min_validation_trades))

    fold_results: list[dict[str, Any]] = []
    trade_ledgers: list[dict[str, Any]] = []
    thresholds: dict[tuple[str, int], dict[str, Any]] = {}
    for fold in FOLDS:
        ensemble = _load_ensemble(args.protocol101_dir, fold["name"], args.artifact_seeds)
        fold_splits = {fold["validation_split"], fold["test_split"]}
        fold_events = [event for event in events if event["split"] in fold_splits]
        attach_ensemble_predictions(fold_events, ensemble)
        validation_all = [event for event in events if event["split"] == fold["validation_split"]]
        for seed in args.seeds:
            validation_seed = [event for event in validation_all if int(event["seed"]) == int(seed)]
            threshold = select_ensemble_margin_threshold(validation_seed, ensemble, seed=int(seed), config=config)
            thresholds[(fold["name"], int(seed))] = threshold
            seed_result = {
                "fold": fold["name"],
                "seed": int(seed),
                "train_splits": fold["train_splits"],
                "validation_split": fold["validation_split"],
                "test_split": fold["test_split"],
                "threshold": float(threshold["threshold"]),
                "threshold_selection": threshold,
                "splits": {},
            }
            for split_name, event_slice in _reported_event_slices(events, fold, int(seed)).items():
                candidate_source = _candidate_frame_from_events(event_slice)
                base = simulate_ensemble_event_policy(
                    event_slice,
                    ensemble,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.0,
                    strategy=f"protocol109_{fold['name']}",
                )
                stress10 = simulate_ensemble_event_policy(
                    event_slice,
                    ensemble,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.10,
                    strategy=f"protocol109_{fold['name']}_stress10",
                )
                stress25 = simulate_ensemble_event_policy(
                    event_slice,
                    ensemble,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.25,
                    strategy=f"protocol109_{fold['name']}_stress25",
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

    external = _score_q4_2024_external(args.protocol107_dir, args.protocol101_dir, thresholds, args.artifact_seeds, args.seeds)
    aggregate = _aggregate_gate(fold_results)
    protocol101 = json.loads((args.protocol101_dir / "summary.json").read_text())
    payload = {
        "protocol": "109_frozen_protocol101_seed_ensemble",
        "paid_data_downloaded": False,
        "live_orders": False,
        "model_training": "none; reused frozen Protocol 101 artifacts",
        "source_protocol092_dir": str(args.protocol092_dir),
        "source_protocol101_dir": str(args.protocol101_dir),
        "source_protocol107_dir": str(args.protocol107_dir),
        "pre_registration": _pre_registration(args.artifact_seeds),
        "feature_columns": FEATURE_COLUMNS,
        "fold_results": fold_results,
        "aggregate_gate": aggregate,
        "protocol101_comparison": _compare_to_protocol101(aggregate, protocol101["aggregate_gate"]),
        "q4_2024_external": external,
        "decision": _decision(aggregate, protocol101["aggregate_gate"], external),
    }
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    (args.out_dir / "serial_policy_trades.json").write_text(_json_dumps(trade_ledgers))
    (args.out_dir / "q4_2024_external_trades.json").write_text(_json_dumps(external["trades"]))
    _write_report(args.out_dir / "report.md", payload)
    print(json.dumps({"decision": payload["decision"], "gate": _gate_brief(payload), "q4_2024_external": external["aggregate"]}, indent=2, sort_keys=True))
    print(args.out_dir / "report.md")
    return 0


def _load_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_parquet(path)
    dataset["decision_dt"] = pd.to_datetime(dataset["decision_time"], utc=True)
    dataset["candidate_exit_dt"] = pd.to_datetime(dataset["candidate_exit_time"], utc=True)
    return dataset


def _build_history_events(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    from v4.scripts.run_protocol097_sequential_event_policy import build_events

    events = build_events(dataset)
    add_causal_history_features(events)
    return events


def _load_ensemble(protocol101_dir: Path, fold_name: str, artifact_seeds: list[int]) -> list[dict[str, Any]]:
    members = []
    for seed in artifact_seeds:
        model_dir = protocol101_dir / "model_artifacts" / fold_name / f"seed_{seed}"
        manifest = json.loads((model_dir / "manifest.json").read_text())
        scaler_blob = json.loads((model_dir / "scaler.json").read_text())
        feature_columns = list(manifest["feature_columns"])
        if feature_columns != FEATURE_COLUMNS:
            raise ValueError(f"Feature mismatch for {model_dir}: {feature_columns}")
        config = manifest.get("config", {})
        model = EventSetPolicy(input_dim=len(feature_columns), hidden_dim=int(config.get("hidden_dim", 96)))
        state = torch.load(model_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        scaler = FeatureScaler(
            fill=np.asarray(scaler_blob["fill"], dtype=np.float32),
            mean=np.asarray(scaler_blob["mean"], dtype=np.float32),
            std=np.asarray(scaler_blob["std"], dtype=np.float32),
        )
        members.append({"seed": int(seed), "model": model, "scaler": scaler})
    return members


def ensemble_logits(event: dict[str, Any], ensemble: list[dict[str, Any]]) -> np.ndarray:
    logits = []
    for member in ensemble:
        x, mask, _, _, _ = event_tensors([event], member["scaler"], feature_columns=FEATURE_COLUMNS)
        with torch.no_grad():
            logits.append(member["model"](torch.from_numpy(x), torch.from_numpy(mask)).cpu().numpy()[0])
    return np.mean(np.vstack(logits), axis=0)


def attach_ensemble_predictions(events: list[dict[str, Any]], ensemble: list[dict[str, Any]]) -> None:
    if not events:
        return
    accumulator: np.ndarray | None = None
    mask_reference: np.ndarray | None = None
    for member in ensemble:
        x, mask, _, _, _ = event_tensors(events, member["scaler"], feature_columns=FEATURE_COLUMNS)
        mask_reference = mask
        chunks = []
        with torch.no_grad():
            for start in range(0, len(events), 4096):
                logits = member["model"](torch.from_numpy(x[start : start + 4096]), torch.from_numpy(mask[start : start + 4096]))
                chunks.append(logits.cpu().numpy())
        member_logits = np.vstack(chunks)
        accumulator = member_logits if accumulator is None else accumulator + member_logits
    assert accumulator is not None
    assert mask_reference is not None
    mean_logits = accumulator / float(len(ensemble))
    for idx, event in enumerate(events):
        wait = float(mean_logits[idx, 0])
        candidate_logits = mean_logits[idx, 1 : 1 + MAX_CANDIDATES].copy()
        candidate_logits[~mask_reference[idx]] = -1e9
        action = int(np.argmax(candidate_logits)) + 1
        margin = float(candidate_logits[action - 1] - wait)
        event["_protocol109_action"] = action
        event["_protocol109_margin"] = margin


def ensemble_action(event: dict[str, Any], ensemble: list[dict[str, Any]]) -> tuple[int, float]:
    if "_protocol109_action" in event and "_protocol109_margin" in event:
        return int(event["_protocol109_action"]), float(event["_protocol109_margin"])
    logits = ensemble_logits(event, ensemble)
    wait = float(logits[0])
    candidate_logits = logits[1:]
    action = int(np.argmax(candidate_logits)) + 1
    margin = float(candidate_logits[action - 1] - wait)
    return action, margin


def ensemble_margins(events: list[dict[str, Any]], ensemble: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([ensemble_action(event, ensemble)[1] for event in events], dtype=float)


def select_ensemble_margin_threshold(
    events: list[dict[str, Any]],
    ensemble: list[dict[str, Any]],
    *,
    seed: int,
    config: EventPolicyConfig,
) -> dict[str, Any]:
    margins = ensemble_margins(events, ensemble)
    finite = margins[np.isfinite(margins)]
    if len(finite) == 0:
        thresholds = [float("inf")]
    else:
        thresholds = sorted(set(np.quantile(finite, [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]).round(4).tolist() + [0.0, float(finite.min()) - 1e-3]))
    baseline = strict_serial_baseline(_candidate_frame_from_events(events), seed=int(seed), slippage_per_side=0.0)
    baseline10 = strict_serial_baseline(_candidate_frame_from_events(events), seed=int(seed), slippage_per_side=0.10)
    sweep = []
    for threshold in thresholds:
        base = simulate_ensemble_event_policy(events, ensemble, threshold=float(threshold), slippage_per_side=0.0, strategy="protocol109_validation")
        stress = simulate_ensemble_event_policy(events, ensemble, threshold=float(threshold), slippage_per_side=0.10, strategy="protocol109_validation_stress10")
        sweep.append(
            {
                "threshold": float(threshold),
                "model": base.summary,
                "model_stress_0_10": stress.summary,
                "strict_serial_baseline": baseline.summary,
                "strict_serial_baseline_stress_0_10": baseline10.summary,
                "delta_vs_baseline": float(base.summary["total_pnl"] - baseline.summary["total_pnl"]),
                "delta_vs_baseline_stress_0_10": float(stress.summary["total_pnl"] - baseline10.summary["total_pnl"]),
            }
        )
    eligible = [row for row in sweep if row["model"]["trades"] >= config.min_validation_trades and row["model_stress_0_10"]["total_pnl"] > 0.0]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda row: (row["delta_vs_baseline_stress_0_10"], row["delta_vs_baseline"], row["model"]["total_pnl"], row["model"]["trades"]))
    return {
        "threshold": float(best["threshold"]),
        "source_seed": int(seed),
        "source_rows": int(len(events)),
        "objective": "validation stress_0_10 delta vs strict serial baseline",
        "selected": best,
        "sweep": sweep,
    }


def simulate_ensemble_event_policy(
    events: list[dict[str, Any]],
    ensemble: list[dict[str, Any]],
    *,
    threshold: float,
    slippage_per_side: float,
    strategy: str,
) -> Any:
    trades: list[dict[str, Any]] = []
    round_trip_slippage = float(slippage_per_side) * 2.0 * CONTRACT_MULTIPLIER
    for _, session_events in _group_events(events).items():
        session_events = sorted(session_events, key=lambda event: event["decision_dt"])
        open_until: pd.Timestamp | None = None
        for event in session_events:
            if open_until is not None and event["decision_dt"] < open_until:
                continue
            action, margin = ensemble_action(event, ensemble)
            if action <= 0 or margin < threshold:
                continue
            row = event["candidates"].iloc[action - 1]
            pnl = float(row["candidate_pnl"]) - round_trip_slippage
            trades.append(
                {
                    "candidate_uid": str(row["candidate_uid"]),
                    "trade_uid": str(row["trade_uid"]),
                    "split": str(row["split"]),
                    "seed": int(row["seed"]),
                    "entry_seed": int(row["entry_seed"]),
                    "session": str(row["session"]),
                    "decision_time": pd.Timestamp(row["decision_dt"]).isoformat(),
                    "exit_time": pd.Timestamp(row["candidate_exit_dt"]).isoformat(),
                    "contract_id": str(row["contract_id"]),
                    "right": str(row["right"]),
                    "offset": float(row["offset"]),
                    "score": float(margin),
                    "threshold": float(threshold),
                    "pnl": pnl,
                    "raw_candidate_pnl": float(row["candidate_pnl"]),
                    "slippage_per_side": float(slippage_per_side),
                    "strategy": strategy,
                    "exit_reason": str(row["candidate_exit_reason"]),
                    "label_source": str(row["label_source"]),
                }
            )
            open_until = pd.Timestamp(row["candidate_exit_dt"])
    summary = serial_metrics(trades)
    summary.update({"input_events": int(len(events)), "max_concurrent_positions": 1 if trades else 0, "serial_status": "pass"})
    return type("EventSimulationResult", (), {"trades": trades, "summary": summary})()


def _score_q4_2024_external(
    protocol107_dir: Path,
    protocol101_dir: Path,
    thresholds: dict[tuple[str, int], dict[str, Any]],
    artifact_seeds: list[int],
    seeds: list[int],
) -> dict[str, Any]:
    dataset = _load_dataset(protocol107_dir / "q4_2024_external_serial_opportunity_dataset.parquet")
    events = _build_history_events(dataset)
    fold_name = "fold3_train_q1_q2_q3_validate_q4_test_q1_2026"
    ensemble = _load_ensemble(protocol101_dir, fold_name, artifact_seeds)
    attach_ensemble_predictions(events, ensemble)
    rows = []
    trades: list[dict[str, Any]] = []
    for seed in seeds:
        event_slice = [event for event in events if int(event["seed"]) == int(seed)]
        threshold = float(thresholds[(fold_name, int(seed))]["threshold"])
        candidate_source = _candidate_frame_from_events(event_slice)
        base = simulate_ensemble_event_policy(event_slice, ensemble, threshold=threshold, slippage_per_side=0.0, strategy="protocol109_q4_2024_external")
        stress10 = simulate_ensemble_event_policy(event_slice, ensemble, threshold=threshold, slippage_per_side=0.10, strategy="protocol109_q4_2024_external_stress10")
        stress25 = simulate_ensemble_event_policy(event_slice, ensemble, threshold=threshold, slippage_per_side=0.25, strategy="protocol109_q4_2024_external_stress25")
        baseline = strict_serial_baseline(candidate_source, seed=int(seed), slippage_per_side=0.0)
        rows.append(
            {
                "seed": int(seed),
                "fold": fold_name,
                "model": base.summary,
                "model_stress_0_10": stress10.summary,
                "model_stress_0_25": stress25.summary,
                "strict_serial_baseline": baseline.summary,
                "threshold": threshold,
                "validation_threshold_source": "q4_2025",
            }
        )
        for trade in base.trades:
            item = dict(trade)
            item["fold"] = fold_name
            item["reported_split"] = "q4_2024_external"
            trades.append(item)
    summary = _summarize_split(rows)
    summary["positive_seed_margin_fraction"] = float(np.mean([row["model"]["total_pnl"] > row["strict_serial_baseline"]["total_pnl"] for row in rows])) if rows else 0.0
    return {"rows": rows, "aggregate": summary, "trades": trades}


def _compare_to_protocol101(current: dict[str, Any], protocol101: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]:
        now = current[split]
        old = protocol101[split]
        out[split] = {
            "median_total_pnl_delta": float(now["median_total_pnl"] - old["median_total_pnl"]),
            "median_profit_factor_delta": float(now["median_profit_factor"] - old["median_profit_factor"]),
            "median_trades_delta": float(now["median_trades"] - old["median_trades"]),
            "stress_0_25_delta": float(now["median_stress_0_25_total_pnl"] - old["median_stress_0_25_total_pnl"]),
            "baseline_beat_before": bool(old["beats_strict_serial_baseline"]),
            "baseline_beat_after": bool(now["beats_strict_serial_baseline"]),
        }
    return out


def _decision(aggregate: dict[str, Any], protocol101: dict[str, Any], external: dict[str, Any]) -> str:
    comparison = _compare_to_protocol101(aggregate, protocol101)
    registered_better = all(comparison[split]["median_total_pnl_delta"] >= 0 for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"])
    registered_gate = bool(aggregate["promotion_ready"])
    external_margin = float(external["aggregate"]["median_total_pnl"] - external["aggregate"]["strict_serial_baseline_median_total_pnl"])
    external_seed_margin = float(external["aggregate"]["positive_seed_margin_fraction"])
    if registered_gate and registered_better and external_margin > 0.0 and external_seed_margin >= 0.80:
        return "keep_as_research_challenger: frozen seed ensemble improves stability without retraining"
    if registered_gate and external_margin > 0.0:
        return "keep_for_research_only: ensemble passes registered gate but does not fully solve external seed fragility"
    return "reject: frozen seed ensemble does not preserve Protocol 101 promotion gate"


def _pre_registration(artifact_seeds: list[int]) -> dict[str, Any]:
    return {
        "hypothesis": "Protocol 101's external fragility is partly seed instability. Averaging the frozen seed logits may reduce abstention noise without adding side/time rules or retraining.",
        "model_class": "frozen Protocol 101 logit ensemble",
        "artifact_seeds": [int(seed) for seed in artifact_seeds],
        "training": "none",
        "thresholds": "selected only on the registered validation split for each fold/seed",
        "external_thresholds": "Q4 2024 external uses fold3 thresholds selected on Q4 2025 validation",
        "paid_data": "forbidden",
        "live_orders": "forbidden",
    }


def _gate_brief(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        split: {
            "median_total_pnl": payload["aggregate_gate"][split]["median_total_pnl"],
            "baseline": payload["aggregate_gate"][split]["strict_serial_baseline_median_total_pnl"],
            "vs_protocol101": payload["protocol101_comparison"][split]["median_total_pnl_delta"],
        }
        for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol 109: Frozen Protocol 101 Seed Ensemble",
        "",
        "No paid market data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Model training: `{payload['model_training']}`",
        "",
        "## Registered Gate",
        "",
        _table(
            [
                {
                    "split": split,
                    "median_pnl": payload["aggregate_gate"][split]["median_total_pnl"],
                    "pf": payload["aggregate_gate"][split]["median_profit_factor"],
                    "trades": payload["aggregate_gate"][split]["median_trades"],
                    "stress25": payload["aggregate_gate"][split]["median_stress_0_25_total_pnl"],
                    "baseline": payload["aggregate_gate"][split]["strict_serial_baseline_median_total_pnl"],
                    "beats_baseline": payload["aggregate_gate"][split]["beats_strict_serial_baseline"],
                    "vs_protocol101": payload["protocol101_comparison"][split]["median_total_pnl_delta"],
                }
                for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026"]
            ],
            ["split", "median_pnl", "pf", "trades", "stress25", "baseline", "beats_baseline", "vs_protocol101"],
        ),
        "",
        "## Q4 2024 External",
        "",
        _table(
            [
                {
                    "seed": row["seed"],
                    "model_pnl": row["model"]["total_pnl"],
                    "baseline": row["strict_serial_baseline"]["total_pnl"],
                    "margin": row["model"]["total_pnl"] - row["strict_serial_baseline"]["total_pnl"],
                    "pf": row["model"]["profit_factor"],
                    "trades": row["model"]["trades"],
                    "stress25": row["model_stress_0_25"]["total_pnl"],
                }
                for row in payload["q4_2024_external"]["rows"]
            ],
            ["seed", "model_pnl", "baseline", "margin", "pf", "trades", "stress25"],
        ),
        "",
        "External aggregate:",
        "",
        "```json",
        json.dumps(payload["q4_2024_external"]["aggregate"], indent=2, sort_keys=True),
        "```",
        "",
        "## Promotion Checks",
        "",
        _table(payload["aggregate_gate"]["promotion_checks"], ["split", "name", "value", "pass"]),
    ]
    path.write_text("\n".join(lines) + "\n")


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
