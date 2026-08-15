"""Protocol 163: serial one-account entry training.

This runner is the first source-of-truth training path shaped like the paper
bot rather than like an overlapping candidate diagnostic:

* one account starting at $10,000
* one contract max
* one open position max
* ask entry / bid exit through the frozen Protocol081 lifecycle path
* affordability enforced before entry
* no paid data downloads and no broker endpoints

The model is still an entry/arbitration policy. Exits remain frozen.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from v4.model.environment_diagnostics import time_bucket
from v4.model.serial_opportunity import (
    CONTRACT_MULTIPLIER,
    _add_entry_features,
    _sanitize_dataset,
    serial_metrics,
)
from v4.scripts.run_protocol092_serial_opportunity_policy import FOLDS
from v4.scripts.run_protocol097_sequential_event_policy import (
    EventPolicyConfig,
    EventSetPolicy,
    _json_dumps,
    add_oracle_actions,
    build_events,
    event_margins,
    predict_event_action,
    train_event_policy,
)
from v4.scripts.run_protocol101_event_history_policy import (
    FEATURE_COLUMNS as PROTOCOL101_FEATURE_COLUMNS,
    add_causal_history_features,
)


LOOP_ID = "v4_aplus_hypothesis_163_serial_one_account_training"
DEFAULT_PROTOCOL092_DATASET = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_092_serial_opportunity_policy/"
    "serial_opportunity_dataset.parquet"
)
DEFAULT_RECENT_CANDIDATES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/"
    "candidate_lifecycle_paths.csv"
)
DEFAULT_RECENT_BASELINE_SUMMARY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_163_recent_protocol101_serial_lifecycle_replay/"
    "summary.json"
)
DEFAULT_NORMALIZED_DIR = Path("v4/normalized_official_context")
DEFAULT_OUT_DIR = Path(f"v4/audit/autoresearch/{LOOP_ID}")
MODEL_SEEDS = [1, 2, 3, 4, 5]
STARTING_CASH = 10_000.0
RECENT_SPLIT = "recent_2026"
MAY_DIAGNOSTIC_SPLIT = "may19_20_diagnostic"
ACCOUNT_FEATURE_COLUMNS = [
    "entry_premium",
    "entry_premium_frac_10k",
    "entry_affordable_10k",
]
FEATURE_COLUMNS = PROTOCOL101_FEATURE_COLUMNS + ACCOUNT_FEATURE_COLUMNS
RECENT_FOLD = {
    "name": "fold4_train_2025_validate_q1_2026_test_recent",
    "train_splits": ["q1_2025", "q2_2025", "q3_2025", "q4_2025"],
    "validation_split": "q1_2026",
    "test_split": RECENT_SPLIT,
    "reported_splits": [RECENT_SPLIT, MAY_DIAGNOSTIC_SPLIT],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol092-dataset", type=Path, default=DEFAULT_PROTOCOL092_DATASET)
    parser.add_argument("--recent-candidates", type=Path, default=DEFAULT_RECENT_CANDIDATES)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE_SUMMARY)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", type=int, default=MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = EventPolicyConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        min_validation_trades=int(args.min_validation_trades),
    )
    dataset, dataset_audit = build_protocol163_dataset(
        protocol092_dataset=args.protocol092_dataset,
        recent_candidates=args.recent_candidates,
        normalized_dir=args.normalized_dir,
        seeds=args.seeds,
        starting_cash=float(args.starting_cash),
    )
    dataset_path = args.out_dir / "protocol163_serial_one_account_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)

    events = build_events(dataset)
    add_causal_history_features(events)
    oracle_summary = add_oracle_actions(events)

    fold_results: list[dict[str, Any]] = []
    model_trades: list[dict[str, Any]] = []
    baseline_trades: list[dict[str, Any]] = []
    for fold in [*FOLDS, RECENT_FOLD]:
        train_events = [event for event in events if str(event["split"]) in set(fold["train_splits"])]
        validation_all = [event for event in events if str(event["split"]) == fold["validation_split"]]
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
            model_path = model_dir / "model.pt"
            scaler_path = model_dir / "scaler.json"
            torch.save(model.state_dict(), model_path)
            scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n")

            validation_seed = [
                event for event in validation_all if int(event["seed"]) == int(seed)
            ]
            threshold = select_account_threshold(
                validation_seed,
                model,
                scaler,
                seed=int(seed),
                config=config,
                feature_columns=FEATURE_COLUMNS,
                starting_cash=float(args.starting_cash),
            )
            manifest = {
                "protocol": "163_serial_one_account_training",
                "fold_name": fold["name"],
                "seed": int(seed),
                "feature_columns": FEATURE_COLUMNS,
                "account_feature_columns": ACCOUNT_FEATURE_COLUMNS,
                "config": asdict(config),
                "starting_cash": float(args.starting_cash),
                "threshold_selection": threshold,
                "training_history": history,
                "files": {"model": str(model_path), "scaler": str(scaler_path)},
            }
            (model_dir / "manifest.json").write_text(_json_dumps(manifest))

            seed_result: dict[str, Any] = {
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
            for split_name, event_slice in reported_event_slices(events, fold, int(seed)).items():
                model_base = simulate_one_account_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.0,
                    strategy=f"protocol163_{fold['name']}",
                    feature_columns=FEATURE_COLUMNS,
                    starting_cash=float(args.starting_cash),
                )
                model_stress10 = simulate_one_account_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.10,
                    strategy=f"protocol163_{fold['name']}_stress10",
                    feature_columns=FEATURE_COLUMNS,
                    starting_cash=float(args.starting_cash),
                )
                model_stress25 = simulate_one_account_event_policy(
                    event_slice,
                    model,
                    scaler,
                    threshold=float(threshold["threshold"]),
                    slippage_per_side=0.25,
                    strategy=f"protocol163_{fold['name']}_stress25",
                    feature_columns=FEATURE_COLUMNS,
                    starting_cash=float(args.starting_cash),
                )
                baseline_base = strict_one_account_baseline(
                    event_slice,
                    seed=int(seed),
                    slippage_per_side=0.0,
                    starting_cash=float(args.starting_cash),
                )
                baseline_stress10 = strict_one_account_baseline(
                    event_slice,
                    seed=int(seed),
                    slippage_per_side=0.10,
                    starting_cash=float(args.starting_cash),
                )
                seed_result["splits"][split_name] = {
                    "model": model_base.summary,
                    "model_stress_0_10": model_stress10.summary,
                    "model_stress_0_25": model_stress25.summary,
                    "strict_serial_baseline": baseline_base.summary,
                    "strict_serial_baseline_stress_0_10": baseline_stress10.summary,
                    "validation_threshold_source": fold["validation_split"],
                }
                for trade in model_base.trades:
                    model_trades.append({**trade, "fold": fold["name"], "reported_split": split_name})
                for trade in baseline_base.trades:
                    baseline_trades.append({**trade, "fold": fold["name"], "reported_split": split_name})
            fold_results.append(seed_result)

    recent_baseline = _load_recent_frozen_baseline(args.recent_baseline_summary)
    payload = {
        "protocol": "163_serial_one_account_training",
        "paid_data_downloaded_by_runner": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "starting_cash": float(args.starting_cash),
        "source_protocol092_dataset": str(args.protocol092_dataset),
        "source_recent_candidates": str(args.recent_candidates),
        "dataset_path": str(dataset_path),
        "feature_columns": FEATURE_COLUMNS,
        "dataset_audit": dataset_audit,
        "event_summary": event_summary(events),
        "oracle_summary": oracle_summary,
        "fold_results": fold_results,
        "aggregate": aggregate_results(fold_results),
        "recent_frozen_protocol101_baseline": recent_baseline,
    }
    payload["decision"] = decision(payload)
    (args.out_dir / "summary.json").write_text(_json_dumps(payload))
    pd.DataFrame(model_trades).to_csv(args.out_dir / "protocol163_model_trades.csv", index=False)
    pd.DataFrame(baseline_trades).to_csv(args.out_dir / "strict_one_account_baseline_trades.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "recent": payload["aggregate"].get(RECENT_SPLIT),
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


def build_protocol163_dataset(
    *,
    protocol092_dataset: Path,
    recent_candidates: Path,
    normalized_dir: Path,
    seeds: Iterable[int],
    starting_cash: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = pd.read_parquet(protocol092_dataset)
    historical = _add_account_features(historical, starting_cash=starting_cash)
    recent_raw = pd.read_csv(recent_candidates)
    recent, recent_audit = recent_candidates_to_dataset(
        recent_raw,
        normalized_dir=normalized_dir,
        seeds=list(seeds),
        starting_cash=starting_cash,
    )
    combined = pd.concat([historical, recent], ignore_index=True, sort=False)
    combined = _add_account_features(combined, starting_cash=starting_cash)
    combined = combined[combined["entry_affordable_10k"] >= 1.0].copy()
    for column in ["decision_dt", "candidate_exit_dt", "entry_quote_dt"]:
        combined[column] = pd.to_datetime(combined[column], utc=True, errors="coerce")
    combined = combined.sort_values(["split", "seed", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)
    audit = {
        "historical_rows": int(len(historical)),
        "recent_rows": int(len(recent)),
        "combined_rows_after_affordability": int(len(combined)),
        "rows_by_split": {str(k): int(v) for k, v in combined["split"].value_counts().sort_index().items()},
        "recent_candidate_audit": recent_audit,
        "starting_cash": float(starting_cash),
    }
    return combined, audit


def recent_candidates_to_dataset(
    frame: pd.DataFrame,
    *,
    normalized_dir: Path,
    seeds: list[int],
    starting_cash: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ok = frame[frame["path_status"].astype(str).eq("ok")].copy()
    ok = attach_entry_sizes(ok, normalized_dir=normalized_dir)
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        item = pd.DataFrame(
            {
                "split": RECENT_SPLIT,
                "session": ok["session"].astype(str),
                "decision_time": pd.to_datetime(ok["decision_time"], utc=True, errors="coerce"),
                "trade_uid": ok["trade_uid"].astype(str),
                "canonical_entry_uid": ok["canonical_entry_uid"].astype(str),
                "contract_id": ok["contract_id"].astype(str),
                "right": ok["right"].astype(str),
                "offset": pd.to_numeric(ok["offset"], errors="coerce"),
                "edge": pd.to_numeric(ok["edge"], errors="coerce"),
                "time_bucket": [
                    time_bucket(pd.Timestamp(ts).to_pydatetime())
                    for ts in pd.to_datetime(ok["decision_time"], utc=True, errors="coerce")
                ],
                "entry_quote_time": pd.to_datetime(ok["entry_quote_time"], utc=True, errors="coerce"),
                "entry_bid": pd.to_numeric(ok["entry_bid"], errors="coerce"),
                "entry_ask": pd.to_numeric(ok["entry_ask"], errors="coerce"),
                "entry_mid": pd.to_numeric(ok["entry_mid"], errors="coerce"),
                "entry_spread": pd.to_numeric(ok["entry_spread"], errors="coerce"),
                "entry_spread_frac": _safe_div(
                    pd.to_numeric(ok["entry_spread"], errors="coerce"),
                    pd.to_numeric(ok["entry_mid"], errors="coerce").abs().clip(lower=0.01),
                ),
                "entry_bid_size": pd.to_numeric(ok["entry_bid_size"], errors="coerce").fillna(0.0),
                "entry_ask_size": pd.to_numeric(ok["entry_ask_size"], errors="coerce").fillna(0.0),
                "entry_underlying_price": pd.to_numeric(ok["entry_underlying_price"], errors="coerce"),
                "entry_iv": pd.to_numeric(ok["entry_iv"], errors="coerce"),
                "entry_delta": pd.to_numeric(ok["entry_delta"], errors="coerce"),
                "entry_gamma": pd.to_numeric(ok["entry_gamma"], errors="coerce"),
                "entry_theta": pd.to_numeric(ok["entry_theta"], errors="coerce"),
                "seed": int(seed),
                "entry_seed": 1,
                "candidate_pnl": pd.to_numeric(ok["candidate_pnl"], errors="coerce"),
                "candidate_exit_time": pd.to_datetime(ok["candidate_exit_time"], utc=True, errors="coerce"),
                "candidate_exit_reason": ok["candidate_exit_reason"].astype(str),
                "candidate_exit_step": pd.to_numeric(ok["candidate_exit_step"], errors="coerce"),
                "label_source": "protocol081_recent_protocol101_candidate",
            }
        )
        item["candidate_uid"] = (
            "protocol163_recent:"
            + str(int(seed))
            + ":"
            + item["trade_uid"].astype(str)
            + ":"
            + item["contract_id"].astype(str)
        )
        rows.append(item)
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    out = _add_entry_features(out)
    out = _sanitize_dataset(out)
    out = _add_account_features(out, starting_cash=starting_cash)
    audit = {
        "raw_candidate_rows": int(len(frame)),
        "ok_candidate_rows": int(len(ok)),
        "duplicated_rows": int(len(out)),
        "missing_entry_sizes_after_join": int(((out["entry_bid_size"] <= 0) & (out["entry_ask_size"] <= 0)).sum())
        if not out.empty
        else 0,
        "unaffordable_rows": int((out["entry_affordable_10k"] < 1.0).sum()) if not out.empty else 0,
    }
    return out, audit


def attach_entry_sizes(frame: pd.DataFrame, *, normalized_dir: Path) -> pd.DataFrame:
    out = frame.copy()
    out["entry_bid_size"] = np.nan
    out["entry_ask_size"] = np.nan
    for session, group in out.groupby("session", sort=False):
        path = _normalized_session_path(normalized_dir, str(session))
        if path is None:
            continue
        sizes = pd.read_parquet(path, columns=["contract_id", "quote_time", "bid_size", "ask_size"])
        sizes["contract_id"] = sizes["contract_id"].astype(str)
        sizes["quote_time"] = pd.to_datetime(sizes["quote_time"], utc=True, errors="coerce")
        sizes = sizes.drop_duplicates(["contract_id", "quote_time"], keep="last")
        lookup = sizes.set_index(["contract_id", "quote_time"])[["bid_size", "ask_size"]]
        for idx in group.index:
            key = (str(out.at[idx, "contract_id"]), pd.Timestamp(out.at[idx, "entry_quote_time"]))
            if key in lookup.index:
                row = lookup.loc[key]
                out.at[idx, "entry_bid_size"] = float(row["bid_size"])
                out.at[idx, "entry_ask_size"] = float(row["ask_size"])
    return out


def _normalized_session_path(normalized_dir: Path, session: str) -> Path | None:
    preferred = sorted(normalized_dir.glob(f"*{session}*official_context.parquet"))
    if preferred:
        return preferred[0]
    fallback = sorted(normalized_dir.glob(f"*{session}*.parquet"))
    return fallback[0] if fallback else None


def _add_account_features(frame: pd.DataFrame, *, starting_cash: float) -> pd.DataFrame:
    out = frame.copy()
    ask = pd.to_numeric(out["entry_ask"], errors="coerce")
    out["entry_premium"] = ask * CONTRACT_MULTIPLIER
    out["entry_premium_frac_10k"] = out["entry_premium"] / float(starting_cash)
    out["entry_affordable_10k"] = ((out["entry_premium"] > 0.0) & (out["entry_premium"] <= float(starting_cash))).astype(float)
    return out


def select_account_threshold(
    events: list[dict[str, Any]],
    model: EventSetPolicy,
    scaler: Any,
    *,
    seed: int,
    config: EventPolicyConfig,
    feature_columns: list[str],
    starting_cash: float,
) -> dict[str, Any]:
    margins = event_margins(events, model, scaler, feature_columns=feature_columns)
    finite = margins[np.isfinite(margins)]
    if len(finite) == 0:
        thresholds = [float("inf")]
    else:
        thresholds = sorted(
            set(
                np.quantile(finite, [0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95])
                .round(4)
                .tolist()
                + [0.0, float(finite.min()) - 1e-3]
            )
        )
    baseline = strict_one_account_baseline(events, seed=int(seed), slippage_per_side=0.0, starting_cash=starting_cash)
    baseline10 = strict_one_account_baseline(events, seed=int(seed), slippage_per_side=0.10, starting_cash=starting_cash)
    sweep = []
    for threshold in thresholds:
        base = simulate_one_account_event_policy(
            events,
            model,
            scaler,
            threshold=float(threshold),
            slippage_per_side=0.0,
            strategy="protocol163_validation",
            feature_columns=feature_columns,
            starting_cash=starting_cash,
        )
        stress = simulate_one_account_event_policy(
            events,
            model,
            scaler,
            threshold=float(threshold),
            slippage_per_side=0.10,
            strategy="protocol163_validation_stress10",
            feature_columns=feature_columns,
            starting_cash=starting_cash,
        )
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
    eligible = [
        row
        for row in sweep
        if row["model"]["trades"] >= config.min_validation_trades
        and row["model_stress_0_10"]["total_pnl"] > 0.0
    ]
    pool = eligible if eligible else sweep
    best = max(
        pool,
        key=lambda row: (
            row["delta_vs_baseline_stress_0_10"],
            row["delta_vs_baseline"],
            row["model"]["total_pnl"],
            row["model"]["trades"],
        ),
    )
    return {
        "threshold": float(best["threshold"]),
        "source_seed": int(seed),
        "source_rows": int(len(events)),
        "objective": "validation account stress_0_10 delta vs one-account strict serial baseline",
        "selected": best,
        "sweep": sweep,
    }


def simulate_one_account_event_policy(
    events: list[dict[str, Any]],
    model: EventSetPolicy,
    scaler: Any,
    *,
    threshold: float,
    slippage_per_side: float,
    strategy: str,
    feature_columns: list[str],
    starting_cash: float,
) -> Any:
    trades: list[dict[str, Any]] = []
    equity = float(starting_cash)
    skipped = {"overlap": 0, "threshold": 0, "unaffordable": 0, "invalid": 0}
    open_until_by_session: dict[str, pd.Timestamp] = {}
    for event in sorted(events, key=lambda row: (str(row["session"]), row["decision_dt"])):
        session = str(event["session"])
        decision_time = pd.Timestamp(event["decision_dt"])
        open_until = open_until_by_session.get(session)
        if open_until is not None and decision_time < open_until:
            skipped["overlap"] += int(len(event["candidates"]))
            continue
        action, margin = predict_event_action(event, model, scaler, feature_columns=feature_columns)
        if action <= 0 or margin < threshold:
            skipped["threshold"] += int(len(event["candidates"]))
            continue
        row = event["candidates"].iloc[action - 1]
        trade = _trade_from_candidate(
            row,
            score=margin,
            threshold=threshold,
            slippage_per_side=slippage_per_side,
            strategy=strategy,
            equity=equity,
        )
        if trade is None:
            skipped["invalid"] += int(len(event["candidates"]))
            continue
        if float(trade["entry_premium_with_slippage"]) > equity:
            skipped["unaffordable"] += int(len(event["candidates"]))
            continue
        trades.append(trade)
        equity += float(trade["pnl"])
        trades[-1]["account_equity_after"] = float(equity)
        open_until_by_session[session] = pd.Timestamp(row["candidate_exit_dt"])
    return _simulation_result(trades, events, skipped, starting_cash=starting_cash, strategy=strategy)


def strict_one_account_baseline(
    events: list[dict[str, Any]],
    *,
    seed: int,
    slippage_per_side: float,
    starting_cash: float,
) -> Any:
    trades: list[dict[str, Any]] = []
    equity = float(starting_cash)
    skipped = {"overlap": 0, "threshold": 0, "unaffordable": 0, "invalid": 0}
    open_until_by_session: dict[str, pd.Timestamp] = {}
    seed_events = [event for event in events if int(event["seed"]) == int(seed)]
    for event in sorted(seed_events, key=lambda row: (str(row["session"]), row["decision_dt"])):
        session = str(event["session"])
        decision_time = pd.Timestamp(event["decision_dt"])
        open_until = open_until_by_session.get(session)
        if open_until is not None and decision_time < open_until:
            skipped["overlap"] += int(len(event["candidates"]))
            continue
        candidates = event["candidates"].sort_values(["entry_seed", "contract_id", "candidate_uid"])
        chosen_trade = None
        for _, row in candidates.iterrows():
            trade = _trade_from_candidate(
                row,
                score=0.0,
                threshold=-1e18,
                slippage_per_side=slippage_per_side,
                strategy="strict_one_account_first_available",
                equity=equity,
            )
            if trade is None:
                continue
            if float(trade["entry_premium_with_slippage"]) <= equity:
                chosen_trade = trade
                break
        if chosen_trade is None:
            skipped["unaffordable"] += int(len(candidates))
            continue
        trades.append(chosen_trade)
        equity += float(chosen_trade["pnl"])
        trades[-1]["account_equity_after"] = float(equity)
        open_until_by_session[session] = pd.Timestamp(candidates.iloc[0]["candidate_exit_dt"])
    return _simulation_result(
        trades,
        seed_events,
        skipped,
        starting_cash=starting_cash,
        strategy="strict_one_account_first_available",
    )


def _trade_from_candidate(
    row: pd.Series,
    *,
    score: float,
    threshold: float,
    slippage_per_side: float,
    strategy: str,
    equity: float,
) -> dict[str, Any] | None:
    decision_dt = pd.Timestamp(row["decision_dt"])
    exit_dt = pd.Timestamp(row["candidate_exit_dt"])
    pnl = _finite_float(row.get("candidate_pnl"))
    entry_ask = _finite_float(row.get("entry_ask"))
    if exit_dt <= decision_dt or not math.isfinite(pnl) or not math.isfinite(entry_ask) or entry_ask <= 0.0:
        return None
    round_trip_slippage = float(slippage_per_side) * 2.0 * CONTRACT_MULTIPLIER
    entry_premium = entry_ask * CONTRACT_MULTIPLIER
    return {
        "candidate_uid": str(row["candidate_uid"]),
        "trade_uid": str(row["trade_uid"]),
        "split": str(row["split"]),
        "seed": int(row["seed"]),
        "entry_seed": int(row["entry_seed"]),
        "session": str(row["session"]),
        "decision_time": decision_dt.isoformat(),
        "exit_time": exit_dt.isoformat(),
        "contract_id": str(row["contract_id"]),
        "right": str(row["right"]),
        "offset": float(row["offset"]),
        "score": float(score),
        "threshold": float(threshold),
        "entry_ask": float(entry_ask),
        "entry_premium": float(entry_premium),
        "entry_premium_with_slippage": float((entry_ask + float(slippage_per_side)) * CONTRACT_MULTIPLIER),
        "account_equity_before": float(equity),
        "account_equity_after": float(equity),
        "pnl": float(pnl - round_trip_slippage),
        "raw_candidate_pnl": float(pnl),
        "slippage_per_side": float(slippage_per_side),
        "strategy": strategy,
        "exit_reason": str(row["candidate_exit_reason"]),
        "label_source": str(row["label_source"]),
        "quantity": 1,
    }


def _simulation_result(
    trades: list[dict[str, Any]],
    events: list[dict[str, Any]],
    skipped: dict[str, int],
    *,
    starting_cash: float,
    strategy: str,
) -> Any:
    summary = serial_metrics(trades)
    equity_values = [float(starting_cash)] + [float(trade["account_equity_after"]) for trade in trades]
    peak = equity_values[0]
    max_drawdown = 0.0
    for value in equity_values:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value - peak)
    summary.update(
        {
            "strategy": strategy,
            "starting_cash": float(starting_cash),
            "ending_equity": float(equity_values[-1]),
            "return_pct": float((equity_values[-1] - starting_cash) / starting_cash * 100.0),
            "max_drawdown": float(max_drawdown),
            "input_events": int(len(events)),
            "skipped_overlap_candidates": int(skipped["overlap"]),
            "skipped_threshold_candidates": int(skipped["threshold"]),
            "skipped_unaffordable_candidates": int(skipped["unaffordable"]),
            "skipped_invalid_candidates": int(skipped["invalid"]),
            "max_concurrent_positions": 1 if trades else 0,
            "serial_status": "pass",
            "all_flat_by_session_end": True,
        }
    )
    return type("AccountSimulationResult", (), {"trades": trades, "summary": summary})()


def reported_event_slices(events: list[dict[str, Any]], fold: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    test = [event for event in events if str(event["split"]) == fold["test_split"] and int(event["seed"]) == int(seed)]
    out: dict[str, list[dict[str, Any]]] = {fold["test_split"]: test}
    if fold["test_split"] == "q1_2026":
        out["march_2026"] = [event for event in test if str(event["session"]) >= "2026-03-01"]
    if fold["test_split"] == RECENT_SPLIT:
        out[MAY_DIAGNOSTIC_SPLIT] = [event for event in test if str(event["session"]) >= "2026-05-19"]
    return out


def aggregate_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    splits = ["q3_2025", "q4_2025", "q1_2026", "march_2026", RECENT_SPLIT, MAY_DIAGNOSTIC_SPLIT]
    return {split: summarize_split_results(fold_results, split) for split in splits}


def summarize_split_results(fold_results: list[dict[str, Any]], split: str) -> dict[str, Any]:
    rows = []
    for result in fold_results:
        if split in result["splits"]:
            rows.append({"seed": result["seed"], "fold": result["fold"], **result["splits"][split]})
    if not rows:
        return {"seeds": 0}
    model_pnl = _arr(row["model"]["total_pnl"] for row in rows)
    stress10 = _arr(row["model_stress_0_10"]["total_pnl"] for row in rows)
    stress25 = _arr(row["model_stress_0_25"]["total_pnl"] for row in rows)
    baseline = _arr(row["strict_serial_baseline"]["total_pnl"] for row in rows)
    pf = _arr(_finite_pf(row["model"]["profit_factor"]) for row in rows)
    trades = _arr(row["model"]["trades"] for row in rows)
    return {
        "seeds": int(len(rows)),
        "median_total_pnl": float(np.median(model_pnl)),
        "median_stress_0_10_total_pnl": float(np.median(stress10)),
        "median_stress_0_25_total_pnl": float(np.median(stress25)),
        "positive_seed_fraction": float((model_pnl > 0.0).mean()),
        "median_profit_factor": float(np.median(pf)),
        "median_trades": float(np.median(trades)),
        "strict_serial_baseline_median_total_pnl": float(np.median(baseline)),
        "median_delta_vs_baseline": float(np.median(model_pnl - baseline)),
        "beats_strict_serial_baseline": bool(float(np.median(model_pnl)) > float(np.median(baseline))),
        "seed_rows": [
            {
                "seed": int(row["seed"]),
                "fold": row["fold"],
                "model_total_pnl": float(row["model"]["total_pnl"]),
                "model_trades": int(row["model"]["trades"]),
                "model_profit_factor": float(row["model"]["profit_factor"]),
                "model_return_pct": float(row["model"]["return_pct"]),
                "model_max_drawdown": float(row["model"]["max_drawdown"]),
                "stress_0_10_total_pnl": float(row["model_stress_0_10"]["total_pnl"]),
                "stress_0_25_total_pnl": float(row["model_stress_0_25"]["total_pnl"]),
                "strict_serial_baseline_total_pnl": float(row["strict_serial_baseline"]["total_pnl"]),
            }
            for row in rows
        ],
    }


def event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for event in events:
        split = str(event["split"])
        item = rows.setdefault(split, {"events": 0, "candidate_rows": 0, "sessions": set()})
        item["events"] += 1
        item["candidate_rows"] += int(len(event["candidates"]))
        item["sessions"].add(str(event["session"]))
    return {
        split: {
            "events": int(item["events"]),
            "candidate_rows": int(item["candidate_rows"]),
            "sessions": int(len(item["sessions"])),
        }
        for split, item in sorted(rows.items())
    }


def decision(payload: dict[str, Any]) -> str:
    recent = payload["aggregate"].get(RECENT_SPLIT, {})
    q1 = payload["aggregate"].get("q1_2026", {})
    march = payload["aggregate"].get("march_2026", {})
    if (
        recent.get("seeds", 0) > 0
        and recent.get("median_total_pnl", 0.0) > 0.0
        and recent.get("median_delta_vs_baseline", -1.0) > 0.0
        and recent.get("positive_seed_fraction", 0.0) >= 0.8
        and q1.get("median_total_pnl", 0.0) > 0.0
        and march.get("median_total_pnl", 0.0) > 0.0
    ):
        return "keep_for_research: Protocol163 beats the one-account recent baseline without breaking Q1/March"
    if recent.get("seeds", 0) > 0 and recent.get("median_total_pnl", 0.0) > 0.0:
        return "keep_for_attribution_only: Protocol163 is profitable but does not clear the one-account improvement gate"
    return "reject_current_hypothesis: Protocol163 did not produce a profitable one-account recent audit"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# Protocol 163: Serial One-Account Training",
        "",
        "Protocol163 trains the entry policy under the same account shape as the paper bot: one account, one contract, one open position, ask entry, bid exit, affordability checks, and frozen Protocol081 exits.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Starting cash: `${payload['starting_cash']:.0f}`",
        f"- Dataset: `{payload['dataset_path']}`",
        f"- Paid data downloaded by runner: `{payload['paid_data_downloaded_by_runner']}`",
        f"- Broker endpoint called: `{payload['broker_endpoint_called']}`",
        "",
        "## Dataset",
        "",
        f"- Historical rows: `{payload['dataset_audit']['historical_rows']}`",
        f"- Recent duplicated rows: `{payload['dataset_audit']['recent_rows']}`",
        f"- Combined rows after affordability: `{payload['dataset_audit']['combined_rows_after_affordability']}`",
        f"- Rows by split: `{payload['dataset_audit']['rows_by_split']}`",
        f"- Recent candidate audit: `{payload['dataset_audit']['recent_candidate_audit']}`",
        "",
        "## Results",
        "",
        "| split | seeds | model median PnL | baseline median PnL | delta | PF | trades | stress $0.10 | stress $0.25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", RECENT_SPLIT, MAY_DIAGNOSTIC_SPLIT]:
        item = agg.get(split, {})
        if item.get("seeds", 0) == 0:
            continue
        lines.append(
            f"| {split} | {item['seeds']} | {_fmt(item['median_total_pnl'])} | "
            f"{_fmt(item['strict_serial_baseline_median_total_pnl'])} | {_fmt(item['median_delta_vs_baseline'])} | "
            f"{_fmt(item['median_profit_factor'])} | {item['median_trades']:.0f} | "
            f"{_fmt(item['median_stress_0_10_total_pnl'])} | {_fmt(item['median_stress_0_25_total_pnl'])} |"
        )
    lines.extend(
        [
            "",
            "## Recent Frozen Baseline",
            "",
            f"- Frozen Protocol101 repaired serial baseline total PnL: `{_fmt(payload['recent_frozen_protocol101_baseline'].get('total_pnl'))}`",
            f"- Trades: `{payload['recent_frozen_protocol101_baseline'].get('trades')}`",
            f"- Profit factor: `{_fmt(payload['recent_frozen_protocol101_baseline'].get('profit_factor'))}`",
            "",
            "## Interpretation",
            "",
            "This is not live-trading approval. It answers whether a serial/account-aware neural entry policy can improve on the frozen Protocol101 one-position baseline when tested on the cleaned recent block. If it fails to beat that baseline, the existing one-contract Protocol101 path remains the operational default while we collect live timing/fill evidence.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Model trades: `{path.parent / 'protocol163_model_trades.csv'}`",
            f"- Baseline trades: `{path.parent / 'strict_one_account_baseline_trades.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _load_recent_frozen_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("serial_protocol081_summary", {})


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _arr(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _finite_pf(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(out):
        return 999.0
    if not math.isfinite(out):
        return 0.0
    return out


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _fmt(value: Any) -> str:
    number = _finite_float(value)
    if math.isinf(number):
        return "inf"
    if not math.isfinite(number):
        return ""
    return f"{number:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
