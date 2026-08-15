"""CHALLENGER_INTEGRATED_ENTRY_LIFECYCLE_SERIAL_REPLAY_V1.

Integrate the full-action flat entry policy from Protocol271 with the
position-state lifecycle policy from Protocol275 inside the strict unified
one-account simulator. This is the first actual replay-PnL check for the
Protocol275 lifecycle model.

No paid data is downloaded. No broker endpoint is called. This runner does not
change the paper default.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from v4.model.supervised_pilot import FeatureScaler, Trade, metrics_for_trades
from v4.scripts.run_protocol165_full_action_space_policy import FOLDS, STARTING_CASH, load_protocol101_baselines, reported_slices
from v4.scripts.run_protocol271_unified_action_advantage_policy import (
    DEFAULT_DATASET,
    DEFAULT_PROTOCOL101_SUMMARY,
    DEFAULT_RECENT_BASELINE,
    MAX_ACTION_CANDIDATES,
    UnifiedActionAdvantagePolicy,
    UnifiedActionPolicyConfig,
    build_events,
    load_dataset,
    tensors as entry_tensors,
)
from v4.scripts.run_protocol274_position_state_action_advantage_dataset import (
    DEFAULT_NORMALIZED_DIR,
    build_flat_value_maps,
    forced_flat_timestamp,
    labels_for_trade,
    load_session_quotes,
)
from v4.scripts.run_protocol275_position_state_lifecycle_policy import FEATURE_COLUMNS as LIFECYCLE_FEATURE_COLUMNS
from v4.scripts.run_protocol275_position_state_lifecycle_policy import PositionLifecycleMLP


ROLE_LABEL = "CHALLENGER_INTEGRATED_ENTRY_LIFECYCLE_SERIAL_REPLAY_V1"
HISTORICAL_ID = "Protocol276"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_276_integrated_entry_lifecycle_serial_replay")
DEFAULT_ENTRY_ARTIFACTS = Path("v4/audit/autoresearch/v4_aplus_hypothesis_271_unified_action_advantage_policy/model_artifacts")
DEFAULT_LIFECYCLE_ARTIFACTS = Path("v4/audit/autoresearch/v4_aplus_hypothesis_275_position_state_lifecycle_policy/model_artifacts")
CONTRACT_MULTIPLIER = 100.0


@dataclass
class EntryBundle:
    fold: str
    seed: int
    threshold: float
    feature_columns: list[str]
    model: UnifiedActionAdvantagePolicy
    scaler: FeatureScaler
    manifest: dict[str, Any]


@dataclass
class LifecycleBundle:
    fold: str
    seed: int
    threshold: float
    feature_columns: list[str]
    model: PositionLifecycleMLP
    scaler: FeatureScaler
    manifest: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument("--entry-artifacts", type=Path, default=DEFAULT_ENTRY_ARTIFACTS)
    parser.add_argument("--lifecycle-artifacts", type=Path, default=DEFAULT_LIFECYCLE_ARTIFACTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--recent-baseline-summary", type=Path, default=DEFAULT_RECENT_BASELINE)
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--slippage-per-side", type=float, default=0.0)
    parser.add_argument("--forced-flat-time", default="15:55")
    parser.add_argument("--max-events-per-split", type=int, default=0)
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, max_events=0)
    if int(args.max_events_per_split) > 0:
        dataset = limit_dataset_events(dataset, int(args.max_events_per_split))
    feature_columns = load_feature_columns(args.entry_artifacts, seed=int(args.seed))
    events = build_events(dataset, feature_columns=feature_columns, starting_cash=float(args.starting_cash))
    flat_values = build_flat_value_maps(dataset)
    frozen_protocol101 = load_protocol101_baselines(args.protocol101_summary, args.recent_baseline_summary)

    fold_results: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_skips: list[dict[str, Any]] = []
    for fold in FOLDS:
        entry = load_entry_bundle(args.entry_artifacts, fold["name"], int(args.seed))
        lifecycle = load_lifecycle_bundle(args.lifecycle_artifacts, fold["name"], int(args.seed))
        result = {"fold": fold["name"], "seed": int(args.seed), "splits": {}}
        for split_name, split_events in reported_slices(events, fold).items():
            sim = simulate_integrated(
                split_events,
                entry,
                lifecycle,
                flat_values,
                normalized_dir=args.normalized_dir,
                starting_cash=float(args.starting_cash),
                slippage_per_side=float(args.slippage_per_side),
                forced_flat_time=str(args.forced_flat_time),
            )
            result["splits"][split_name] = sim["summary"]
            print(
                json.dumps(
                    {
                        "fold": fold["name"],
                        "split": split_name,
                        "trades": sim["summary"].get("trades", 0),
                        "pnl": sim["summary"].get("total_pnl", 0.0),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            all_trades.extend({**trade, "fold": fold["name"], "reported_split": split_name} for trade in sim["trades"])
            all_skips.extend({**skip, "fold": fold["name"], "reported_split": split_name} for skip in sim["skips"])
        fold_results.append(result)

    aggregate_payload = aggregate(fold_results, frozen_protocol101)
    payload = {
        "role_label": ROLE_LABEL,
        "historical_protocol": HISTORICAL_ID,
        "what_is_this": "model integration / strict one-account serial replay of full-action entry plus lifecycle hold-exit policies",
        "changes_paper_default": False,
        "candidate_label": ROLE_LABEL,
        "entry_component": "CHALLENGER_UNIFIED_ACTION_ADVANTAGE_POLICY_V1 / Protocol271",
        "lifecycle_component": "CHALLENGER_POSITION_STATE_LIFECYCLE_POLICY_V1 / Protocol275",
        "paper_default_label": "PAPER_DEFAULT_PROTOCOL101",
        "baseline": "PAPER_DEFAULT_PROTOCOL101 strict serial replay summary",
        "data_used": {
            "action_dataset": str(args.dataset),
            "normalized_quotes": str(args.normalized_dir),
            "entry_artifacts": str(args.entry_artifacts),
            "lifecycle_artifacts": str(args.lifecycle_artifacts),
        },
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "starting_cash": float(args.starting_cash),
        "contracts_per_trade": 1,
        "max_concurrent_positions": 1,
        "execution": "ask entry, learned bid exit, mandatory flat by close",
        "slippage_per_side": float(args.slippage_per_side),
        "run_scope": "capped_integration_check" if int(args.max_events_per_split) > 0 else "full_available_split_replay",
        "max_events_per_split": int(args.max_events_per_split),
        "artifact_scope_note": artifact_scope_note(),
        "event_summary": summarize_events(events),
        "fold_results": fold_results,
        "aggregate": aggregate_payload,
        "frozen_protocol101_baselines": frozen_protocol101,
        "decision": decide(aggregate_payload),
        "next_experiment": next_experiment(aggregate_payload),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame(all_trades).to_csv(args.out_dir / "integrated_entry_lifecycle_trades.csv", index=False)
    pd.DataFrame(all_skips).to_csv(args.out_dir / "integrated_entry_lifecycle_skips.csv", index=False)
    write_report(args.out_dir / "report.md", payload)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def limit_dataset_events(frame: pd.DataFrame, per_split: int) -> pd.DataFrame:
    events = frame[["split", "session", "decision_dt"]].drop_duplicates().sort_values(["split", "session", "decision_dt"])
    keep = events.groupby("split", sort=True).head(per_split)
    return frame.merge(keep.assign(_keep=1), on=["split", "session", "decision_dt"], how="inner").drop(columns=["_keep"])


def load_feature_columns(root: Path, *, seed: int) -> list[str]:
    manifests = sorted(root.glob(f"*/seed_{seed}/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no entry manifests found under {root} for seed {seed}")
    manifest = json.loads(manifests[0].read_text())
    return list(manifest["feature_columns"])


def load_entry_bundle(root: Path, fold: str, seed: int) -> EntryBundle:
    artifact = root / fold / f"seed_{seed}"
    manifest = json.loads((artifact / "manifest.json").read_text())
    feature_columns = list(manifest["feature_columns"])
    hidden_dim = int(manifest.get("config", {}).get("hidden_dim", 160))
    model = UnifiedActionAdvantagePolicy(input_dim=len(feature_columns), hidden_dim=hidden_dim)
    model.load_state_dict(load_torch_state(artifact / "model.pt"))
    model.eval()
    return EntryBundle(
        fold=fold,
        seed=seed,
        threshold=float(manifest["threshold_selection"]["threshold"]),
        feature_columns=feature_columns,
        model=model,
        scaler=scaler_from_json(artifact / "scaler.json"),
        manifest=manifest,
    )


def load_lifecycle_bundle(root: Path, fold: str, seed: int) -> LifecycleBundle:
    artifact = root / fold / f"seed_{seed}"
    manifest = json.loads((artifact / "manifest.json").read_text())
    feature_columns = list(manifest.get("feature_columns", LIFECYCLE_FEATURE_COLUMNS))
    hidden_dim = int(manifest.get("config", {}).get("hidden_dim", 128))
    model = PositionLifecycleMLP(input_dim=len(feature_columns), hidden_dim=hidden_dim)
    model.load_state_dict(load_torch_state(artifact / "model.pt"))
    model.eval()
    return LifecycleBundle(
        fold=fold,
        seed=seed,
        threshold=float(manifest["threshold"]["threshold"]),
        feature_columns=feature_columns,
        model=model,
        scaler=scaler_from_json(artifact / "scaler.json"),
        manifest=manifest,
    )


def load_torch_state(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def scaler_from_json(path: Path) -> FeatureScaler:
    payload = json.loads(path.read_text())
    return FeatureScaler(
        fill=np.asarray(payload["fill"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def simulate_integrated(
    events: list[dict[str, Any]],
    entry: EntryBundle,
    lifecycle: LifecycleBundle,
    flat_values: dict[tuple[str, str], pd.DataFrame],
    *,
    normalized_dir: Path,
    starting_cash: float,
    slippage_per_side: float,
    forced_flat_time: str,
) -> dict[str, Any]:
    predictions = entry_predictions(events, entry)
    ordered = sorted(zip(events, predictions), key=lambda item: (item[0]["session"], item[0]["decision_dt"]))
    trades: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    equity = float(starting_cash)
    open_until: dict[str, pd.Timestamp] = {}
    selected_contracts = selected_contracts_by_session(ordered, entry.threshold)
    quote_cache: dict[str, pd.DataFrame] = {}
    skip_counts = {"overlap": 0, "threshold": 0, "unaffordable": 0, "path": 0, "invalid": 0}
    for event, pred in ordered:
        session = str(event["session"])
        decision_dt = pd.Timestamp(event["decision_dt"])
        if open_until.get(session) is not None and decision_dt < open_until[session]:
            skip_counts["overlap"] += len(event["candidates"])
            continue
        action = int(pred["action"])
        margin = float(pred["margin"])
        if action <= 0 or action > len(event["candidates"]) or margin < entry.threshold:
            skip_counts["threshold"] += len(event["candidates"])
            continue
        row = event["candidates"].iloc[action - 1].copy()
        premium = finite(row.get("entry_premium"), finite(row.get("entry_ask")) * CONTRACT_MULTIPLIER)
        if not math.isfinite(premium) or premium <= 0.0:
            skip_counts["invalid"] += 1
            skips.append(skip_row(event, row, "invalid_entry_premium"))
            continue
        if premium + slippage_per_side * CONTRACT_MULTIPLIER > equity:
            skip_counts["unaffordable"] += 1
            skips.append(skip_row(event, row, "unaffordable_current_equity", equity=equity))
            continue
        path = lifecycle_path_for_trade(
            row,
            flat_values,
            normalized_dir=normalized_dir,
            quote_cache=quote_cache,
            selected_contracts=selected_contracts,
            forced_flat_time=forced_flat_time,
        )
        if path["skip"] is not None:
            skip_counts["path"] += 1
            skips.append(path["skip"])
            continue
        exit_decision = choose_lifecycle_exit(path["rows"], lifecycle)
        if exit_decision is None:
            skip_counts["path"] += 1
            skips.append(skip_row(event, row, "empty_lifecycle_path"))
            continue
        trade = trade_from_decision(
            row,
            exit_decision,
            entry_margin=margin,
            entry_threshold=entry.threshold,
            lifecycle_threshold=lifecycle.threshold,
            equity_before=equity,
            slippage_per_side=slippage_per_side,
        )
        trades.append(trade)
        equity += float(trade["pnl"])
        trades[-1]["account_equity_after"] = float(equity)
        open_until[session] = pd.Timestamp(trade["exit_time"])
    summary = simulation_summary(trades, len(events), skip_counts, starting_cash=starting_cash)
    return {"trades": trades, "skips": skips, "summary": summary}


def selected_contracts_by_session(ordered: list[tuple[dict[str, Any], dict[str, Any]]], threshold: float) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for event, pred in ordered:
        action = int(pred["action"])
        margin = float(pred["margin"])
        if action <= 0 or action > len(event["candidates"]) or margin < threshold:
            continue
        row = event["candidates"].iloc[action - 1]
        contract_id = str(row.get("contract_id", ""))
        if contract_id:
            out.setdefault(str(event["session"]), set()).add(contract_id)
    return out


def entry_predictions(events: list[dict[str, Any]], entry: EntryBundle) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    config = UnifiedActionPolicyConfig()
    for start in range(0, len(events), 2048):
        chunk = events[start : start + 2048]
        if not chunk:
            continue
        x, mask, state, *_ = entry_tensors(chunk, entry.scaler, entry.feature_columns, config)
        with torch.no_grad():
            logits, _ = entry.model(torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(state))
        logits_np = logits.cpu().numpy()
        candidates = logits_np[:, 1:]
        actions = np.argmax(candidates, axis=1) + 1
        margins = candidates[np.arange(len(candidates)), actions - 1] - logits_np[:, 0]
        valid = np.isfinite(margins) & (margins > -1e8)
        for action, margin, ok in zip(actions, margins, valid):
            out.append({"action": int(action) if ok else 0, "margin": float(margin) if ok else -math.inf})
    return out


def lifecycle_path_for_trade(
    row: pd.Series,
    flat_values: dict[tuple[str, str], pd.DataFrame],
    *,
    normalized_dir: Path,
    quote_cache: dict[str, pd.DataFrame],
    selected_contracts: dict[str, set[str]],
    forced_flat_time: str,
) -> dict[str, Any]:
    session = str(row.get("session", ""))
    contract_id = str(row.get("contract_id", ""))
    if session not in quote_cache:
        contracts = selected_contracts.get(session, {contract_id})
        quote_cache[session] = load_session_quotes(normalized_dir, session, contracts)
    quotes = quote_cache[session]
    if not quotes.empty:
        quotes = quotes[quotes["contract_id"].astype(str).eq(contract_id)].copy()
    forced_flat = forced_flat_timestamp(session, forced_flat_time)
    state_rows, skip = labels_for_trade(row, quotes, flat_values.get((str(row.get("split", "")), session)), forced_flat)
    if skip:
        return {"rows": pd.DataFrame(), "skip": skip}
    frame = pd.DataFrame(state_rows)
    if not frame.empty:
        frame["state_dt"] = pd.to_datetime(frame["state_time"], utc=True, errors="coerce")
        frame = frame.sort_values("state_dt").reset_index(drop=True)
    return {"rows": frame, "skip": None}


def choose_lifecycle_exit(path: pd.DataFrame, lifecycle: LifecycleBundle) -> dict[str, Any] | None:
    if path.empty:
        return None
    for column in lifecycle.feature_columns:
        if column not in path.columns:
            path[column] = 0.0
    x = lifecycle.scaler.transform(path[lifecycle.feature_columns].to_numpy(dtype=np.float32))
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            logits, pred_adv = lifecycle.model(torch.from_numpy(x[start : start + 4096]))
            score = (logits[:, 1] - logits[:, 0]).cpu().numpy() + pred_adv.cpu().numpy() * 0.25
            scores.extend(float(value) for value in score)
    path = path.copy()
    path["lifecycle_score"] = scores
    path["lifecycle_action"] = np.where(path["lifecycle_score"].to_numpy(dtype=float) > lifecycle.threshold, "hold", "exit")
    exit_candidates = path[path["lifecycle_action"].eq("exit")]
    if exit_candidates.empty:
        row = path.iloc[-1]
        reason = "forced_flat_no_lifecycle_exit_signal"
    else:
        row = exit_candidates.iloc[0]
        reason = "lifecycle_model_exit"
    return {
        "exit_time": pd.Timestamp(row["state_dt"]),
        "exit_bid": finite(row.get("bid")),
        "exit_ask": finite(row.get("ask")),
        "exit_mid": finite(row.get("mid")),
        "exit_underlying_price": finite(row.get("underlying_price")),
        "exit_score": finite(row.get("lifecycle_score")),
        "exit_reason": reason,
        "state_index": int(row.get("state_index", 0)),
        "current_pnl_before_slippage": finite(row.get("current_pnl")),
        "mfe_to_exit": finite(row.get("mfe_to_now")),
        "mae_to_exit": finite(row.get("mae_to_now")),
        "giveback_from_mfe": finite(row.get("giveback_from_mfe")),
        "path_rows": int(len(path)),
    }


def trade_from_decision(
    row: pd.Series,
    exit_decision: dict[str, Any],
    *,
    entry_margin: float,
    entry_threshold: float,
    lifecycle_threshold: float,
    equity_before: float,
    slippage_per_side: float,
) -> dict[str, Any]:
    entry_ask = finite(row.get("entry_ask"))
    exit_bid = finite(exit_decision["exit_bid"])
    entry_dt = pd.Timestamp(row["decision_dt"])
    exit_dt = pd.Timestamp(exit_decision["exit_time"])
    pnl = (exit_bid - entry_ask - (2.0 * float(slippage_per_side))) * CONTRACT_MULTIPLIER
    return {
        "candidate_uid": str(row.get("candidate_uid", "")),
        "trade_uid": str(row.get("trade_uid", "")),
        "split": str(row.get("split", "")),
        "session": str(row.get("session", "")),
        "decision_time": entry_dt.isoformat(),
        "exit_time": exit_dt.isoformat(),
        "duration_minutes": float((exit_dt - entry_dt).total_seconds() / 60.0),
        "contract_id": str(row.get("contract_id", "")),
        "right": str(row.get("right", "")),
        "offset": finite(row.get("offset")),
        "entry_ask": entry_ask,
        "entry_bid": finite(row.get("entry_bid")),
        "entry_premium": float(entry_ask * CONTRACT_MULTIPLIER),
        "exit_bid": exit_bid,
        "exit_ask": finite(exit_decision.get("exit_ask")),
        "exit_mid": finite(exit_decision.get("exit_mid")),
        "pnl": float(pnl),
        "account_equity_before": float(equity_before),
        "account_equity_after": float(equity_before),
        "entry_model_margin": float(entry_margin),
        "entry_model_threshold": float(entry_threshold),
        "lifecycle_score_at_exit": finite(exit_decision.get("exit_score")),
        "lifecycle_threshold": float(lifecycle_threshold),
        "exit_reason": str(exit_decision.get("exit_reason", "")),
        "exit_state_index": int(exit_decision.get("state_index", 0)),
        "mfe_to_exit": finite(exit_decision.get("mfe_to_exit")),
        "mae_to_exit": finite(exit_decision.get("mae_to_exit")),
        "giveback_from_mfe": finite(exit_decision.get("giveback_from_mfe")),
        "entry_underlying_price": finite(row.get("entry_underlying_price")),
        "exit_underlying_price": finite(exit_decision.get("exit_underlying_price")),
        "a_enter": finite(row.get("a_enter")),
        "q_wait": finite(row.get("q_wait")),
        "q_enter": finite(row.get("q_enter")),
        "path_rows_seen": int(exit_decision.get("path_rows", 0)),
        "strategy": ROLE_LABEL,
    }


def simulation_summary(trades: list[dict[str, Any]], event_count: int, skipped: dict[str, int], *, starting_cash: float) -> dict[str, Any]:
    converted = [
        Trade(
            session=str(trade["session"]),
            decision_time=str(trade["decision_time"]),
            pnl=float(trade["pnl"]),
            score=float(trade.get("entry_model_margin", 0.0)),
            right=str(trade.get("right", "")),
            offset=float(trade.get("offset", 0.0)),
            strategy=ROLE_LABEL,
        )
        for trade in trades
    ]
    metrics = metrics_for_trades(converted)
    equity = [float(starting_cash)] + [float(trade["account_equity_after"]) for trade in trades]
    peak = float(starting_cash)
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = min(drawdown, value - peak)
    metrics.update(
        {
            "starting_cash": float(starting_cash),
            "ending_equity": float(equity[-1]),
            "return_pct": float((equity[-1] - starting_cash) / starting_cash * 100.0),
            "max_account_drawdown": float(drawdown),
            "input_events": int(event_count),
            "max_concurrent_positions": 1 if trades else 0,
            "serial_status": "pass",
            "all_flat_by_session_end": True,
            **{f"skipped_{key}": int(value) for key, value in skipped.items()},
        }
    )
    if trades:
        metrics["median_duration_minutes"] = float(np.median([float(t["duration_minutes"]) for t in trades]))
        metrics["side_counts"] = {"C": int(sum(1 for t in trades if t["right"] == "C")), "P": int(sum(1 for t in trades if t["right"] == "P"))}
        metrics["exit_reason_counts"] = count_by(pd.DataFrame(trades), "exit_reason")
    else:
        metrics["median_duration_minutes"] = 0.0
        metrics["side_counts"] = {"C": 0, "P": 0}
        metrics["exit_reason_counts"] = {}
    return metrics


def aggregate(fold_results: list[dict[str, Any]], frozen_protocol101: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ["q3_2025", "q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        rows = [result["splits"][split] for result in fold_results if split in result.get("splits", {})]
        out[split] = summarize_split(rows, frozen_protocol101.get(split))
    out["promotion_checks"] = promotion_checks(out)
    out["promotion_ready"] = bool(out["promotion_checks"] and all(item["pass"] for item in out["promotion_checks"]))
    return out


def summarize_split(rows: list[dict[str, Any]], frozen: float | None) -> dict[str, Any]:
    if not rows:
        return {"seeds": 0, "frozen_protocol101_total_pnl": frozen}
    pnl = arr(row["total_pnl"] for row in rows)
    pf = arr(finite_pf(row["profit_factor"]) for row in rows)
    trades = arr(row["trades"] for row in rows)
    win = arr(row["win_rate"] for row in rows)
    dd = arr(row.get("max_account_drawdown", row.get("max_drawdown", 0.0)) for row in rows)
    return {
        "seeds": len(rows),
        "median_total_pnl": float(np.median(pnl)),
        "positive_seed_fraction": float((pnl > 0).mean()),
        "median_profit_factor": float(np.median(pf)),
        "median_trades": float(np.median(trades)),
        "median_win_rate": float(np.median(win)),
        "median_account_drawdown": float(np.median(dd)),
        "frozen_protocol101_total_pnl": frozen,
        "median_delta_vs_frozen_protocol101": None if frozen is None else float(np.median(pnl) - frozen),
        "beats_frozen_protocol101": False if frozen is None else bool(np.median(pnl) > frozen),
    }


def promotion_checks(aggregate_payload: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for split in ["q4_2025", "q1_2026", "march_2026", "recent_2026"]:
        item = aggregate_payload.get(split, {})
        if item.get("seeds", 0) == 0:
            checks.append({"split": split, "name": "split_available", "pass": False, "value": 0})
            continue
        checks.extend(
            [
                {"split": split, "name": "positive_median_pnl", "pass": item["median_total_pnl"] > 0, "value": item["median_total_pnl"]},
                {"split": split, "name": "median_pf_ge_1_15", "pass": item["median_profit_factor"] >= 1.15, "value": item["median_profit_factor"]},
                {"split": split, "name": "beats_frozen_protocol101", "pass": bool(item["beats_frozen_protocol101"]), "value": item["median_delta_vs_frozen_protocol101"]},
            ]
        )
    return checks


def decide(aggregate_payload: dict[str, Any]) -> str:
    if aggregate_payload.get("promotion_ready"):
        return "integrated_entry_lifecycle_candidate_surpasses_protocol101_needs_multiseed_and_runtime_parity"
    return "research_only_integrated_entry_lifecycle_does_not_surpass_protocol101"


def next_experiment(aggregate_payload: dict[str, Any]) -> str:
    if aggregate_payload.get("promotion_ready"):
        return "Rerun uncapped with seeds 1-5, then build no-order runtime parity for the exact integrated policy before paper replacement."
    return "Do attribution on whether failure came from full-action entry misses, lifecycle early exits, or lifecycle overholding before adding architecture knobs."


def artifact_scope_note() -> str:
    summary_path = Path("v4/audit/autoresearch/v4_aplus_hypothesis_271_unified_action_advantage_policy/summary.json")
    if not summary_path.exists():
        return "entry/lifecycle artifact training scope unknown"
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return "entry/lifecycle artifact training scope unknown"
    return str(summary.get("run_scope", "entry/lifecycle artifact training scope unknown"))


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for event in events:
        item = out.setdefault(event["split"], {"events": 0, "candidate_rows": 0, "sessions": set()})
        item["events"] += 1
        item["candidate_rows"] += len(event["candidates"])
        item["sessions"].add(event["session"])
    return {key: {"events": value["events"], "candidate_rows": value["candidate_rows"], "sessions": len(value["sessions"])} for key, value in sorted(out.items())}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Candidate being tested: `{payload['candidate_label']}`",
        f"Baseline: `{payload['paper_default_label']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        f"Run scope: `{payload['run_scope']}`",
        f"Artifact scope note: `{payload['artifact_scope_note']}`",
        f"Decision: `{payload['decision']}`",
        f"Next experiment: {payload['next_experiment']}",
        "",
        "## Aggregate",
        "",
        "| split | seeds | PnL | Protocol101 | delta | trades | win rate | PF | acct DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in payload["aggregate"].items():
        if not isinstance(item, dict) or item.get("seeds", 0) == 0:
            continue
        lines.append(
            f"| {split} | {item['seeds']} | {fmt(item['median_total_pnl'])} | {fmt(item.get('frozen_protocol101_total_pnl'))} | "
            f"{fmt(item.get('median_delta_vs_frozen_protocol101'))} | {fmt(item['median_trades'])} | {item['median_win_rate']:.3f} | "
            f"{fmt(item['median_profit_factor'])} | {fmt(item['median_account_drawdown'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "This is an integration replay of saved Protocol271 and Protocol275 artifacts. It is actual one-account replay PnL, but it is not a paper-default replacement unless it beats Protocol101 and later passes multi-seed/runtime parity.",
            "",
            "## Outputs",
            "",
            f"- Summary: `{path.parent / 'summary.json'}`",
            f"- Trades: `{path.parent / 'integrated_entry_lifecycle_trades.csv'}`",
            f"- Skips: `{path.parent / 'integrated_entry_lifecycle_skips.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {HISTORICAL_ID} - {ROLE_LABEL}"
    if marker in ledger.read_text():
        return
    with ledger.open("a") as handle:
        handle.write(
            "\n".join(
                [
                    "",
                    marker,
                    "",
                    f"- What is this: {payload['what_is_this']}",
                    "- Changes paper default: no",
                    f"- Candidate: `{payload['candidate_label']}`",
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                ]
            )
            + "\n"
        )


def skip_row(event: dict[str, Any], row: pd.Series, reason: str, *, equity: float | None = None) -> dict[str, Any]:
    return {
        "split": str(event.get("split", "")),
        "session": str(event.get("session", "")),
        "decision_time": pd.Timestamp(event["decision_dt"]).isoformat(),
        "candidate_uid": str(row.get("candidate_uid", "")),
        "contract_id": str(row.get("contract_id", "")),
        "right": str(row.get("right", "")),
        "offset": finite(row.get("offset")),
        "entry_premium": finite(row.get("entry_premium")),
        "account_equity": equity,
        "skip_reason": reason,
    }


def count_by(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}


def arr(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_pf(value: Any) -> float:
    out = finite(value, 0.0)
    return 999.0 if math.isinf(out) else (out if math.isfinite(out) else 0.0)


def fmt(value: Any) -> str:
    out = finite(value)
    return "" if not math.isfinite(out) else f"{out:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
