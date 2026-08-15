"""Strict serial replay for the trained unified conservative neural policy.

This closes the replay-evidence loop for the trained artifacts only. It does
not change the paper default and does not make a Protocol101 promotion claim.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.supervised_pilot import FeatureScaler
from v4.model.unified_conservative_neural_policy import (
    ROLE_LABEL as TRAINED_POLICY_LABEL,
    ConservativeAdvantageMLP,
    ConservativeNeuralPolicyConfig,
)
from v4.model.unified_conservative_neural_replay import (
    ROLE_LABEL,
    ConservativeReplayConfig,
    choose_conservative_holding_exit,
    select_conservative_entry_candidate,
    summarize_replay_trades,
)
from v4.model.unified_conservative_policy import validate_no_future_feature_columns


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_conservative_neural_policy_strict_replay_v1")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_STRICT_REPLAY_V1.md")
DEFAULT_MODEL_ARTIFACTS = Path("v4/audit/autoresearch/unified_conservative_neural_policy_v1/model_artifacts")
DEFAULT_TRAINING_SUMMARY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_v1/summary.json")
DEFAULT_FLAT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_HOLDING_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/position_state_action_advantage.parquet")
DEFAULT_BASELINE_ACTIONS = Path("v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet")
DEFAULT_SESSION_MANIFEST = Path("v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_session_manifest.csv")
CORE_FLAT_COLUMNS = (
    "split",
    "session",
    "decision_time",
    "decision_dt",
    "candidate_uid",
    "contract_id",
    "right",
    "offset",
    "entry_bid",
    "entry_ask",
    "entry_premium",
    "candidate_pnl",
    "a_enter",
    "candidate_exit_time",
    "candidate_exit_dt",
)
CORE_HOLDING_COLUMNS = (
    "split",
    "session",
    "candidate_uid",
    "trade_uid",
    "contract_id",
    "right",
    "offset",
    "entry_time",
    "state_time",
    "state_index",
    "entry_ask",
    "entry_premium",
    "bid",
    "ask",
    "mid",
    "current_pnl",
    "mfe_to_now",
    "mae_to_now",
    "giveback_from_mfe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifacts", type=Path, default=DEFAULT_MODEL_ARTIFACTS)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
    parser.add_argument("--holding-dataset", type=Path, default=DEFAULT_HOLDING_DATASET)
    parser.add_argument("--baseline-actions", type=Path, default=DEFAULT_BASELINE_ACTIONS)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--slippage-grid", default="0,0.10,0.25")
    parser.add_argument("--max-events-per-split", type=int, default=0)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_policy_bundle(args.model_artifacts)
    validate_no_future_feature_columns(bundle["flat_feature_columns"])
    validate_no_future_feature_columns(bundle["holding_feature_columns"])
    training_summary = load_json(args.training_summary)
    session_manifest = pd.read_csv(args.session_manifest)
    included_sessions = included_session_keys(session_manifest)

    flat = load_flat_dataset(args.flat_dataset, bundle["flat_feature_columns"], included_sessions)
    holding = load_holding_dataset(args.holding_dataset, bundle["holding_feature_columns"], included_sessions)
    baseline = load_baseline_actions(args.baseline_actions, included_sessions, seed=int(args.seed))
    if int(args.max_events_per_split) > 0:
        flat = limit_events_per_split(flat, int(args.max_events_per_split))
        event_keys = flat[["split", "session", "decision_dt"]].drop_duplicates()
        baseline = baseline.merge(event_keys.assign(_keep=1), on=["split", "session", "decision_dt"], how="inner").drop(columns=["_keep"])

    flat = attach_predictions(flat, bundle, head="flat")
    holding_indices = holding.groupby("candidate_uid", sort=False).indices
    slippages = parse_slippage_grid(args.slippage_grid)

    stress_results: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    for slippage in slippages:
        replay = simulate_stress_grid(
            flat,
            baseline,
            holding,
            holding_indices,
            bundle,
            slippage_per_side=float(slippage),
            seed=int(args.seed),
        )
        stress_results.append(replay["summary"])
        all_trades.extend(replay["trades"])
        all_decisions.extend(replay["decisions"])
        suffix = slippage_suffix(slippage)
        pd.DataFrame(replay["trades"]).to_csv(args.out_dir / f"trades_slippage_{suffix}.csv", index=False)
        pd.DataFrame(replay["decisions"]).to_csv(args.out_dir / f"decisions_slippage_{suffix}.csv", index=False)

    challenge_blockers = [
        "calibrated stochastic fill model unavailable",
        "untouched holdout data pending",
        "live no-order full-action parity pending",
        "formal validation controls pending",
    ]
    total_challenger_entries = int(sum(result["totals"]["challenger_entries"] for result in stress_results))
    decision = (
        "strict_replay_complete_model_deferred_to_protocol101_flat_gate_too_conservative"
        if total_challenger_entries == 0
        else "strict_replay_complete_protocol101_challenge_still_blocked"
    )
    payload = {
        "role_label": ROLE_LABEL,
        "trained_policy_label": TRAINED_POLICY_LABEL,
        "what_is_this": "strict one-account serial replay of the trained conservative neural policy with Protocol101 defer fallback",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "strict_replay_run": True,
        "challenge_allowed": False,
        "decision": decision,
        "challenge_blockers": challenge_blockers,
        "interpretation": interpretation(total_challenger_entries),
        "replay_config": ConservativeReplayConfig().to_dict(),
        "policy_config": bundle["config"].to_dict(),
        "data_scope": {
            "seed": int(args.seed),
            "included_sessions": int(len(included_sessions)),
            "flat_rows_loaded": int(len(flat)),
            "holding_rows_loaded": int(len(holding)),
            "baseline_event_rows": int(len(baseline)),
            "training_decision": training_summary.get("decision", "missing"),
            "max_events_per_split": int(args.max_events_per_split),
        },
        "stress_results": stress_results,
        "artifacts": {
            "model_artifacts": str(args.model_artifacts),
            "training_summary": str(args.training_summary),
        },
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
        "next_required_evidence": next_required_evidence(total_challenger_entries),
    }
    write_json(args.out_dir / "summary.json", payload)
    report = render_report(payload)
    (args.out_dir / "report.md").write_text(report)
    if not args.skip_doc:
        args.doc_path.parent.mkdir(parents=True, exist_ok=True)
        args.doc_path.write_text(report)
    if not args.skip_ledger:
        append_ledger(payload, args.out_dir)
    print(json.dumps({"decision": payload["decision"], "report": str(args.out_dir / "report.md")}, indent=2, sort_keys=True))
    return 0


def load_policy_bundle(path: Path) -> dict[str, Any]:
    manifest = load_json(path / "manifest.json")
    config = ConservativeNeuralPolicyConfig(**manifest.get("config", {}))
    flat_features = list(manifest["flat_feature_columns"])
    holding_features = list(manifest["holding_feature_columns"])
    flat_model = ConservativeAdvantageMLP(input_dim=len(flat_features), hidden_dim=config.hidden_dim)
    holding_model = ConservativeAdvantageMLP(input_dim=len(holding_features), hidden_dim=config.hidden_dim)
    flat_model.load_state_dict(load_torch_state(path / "flat_entry_model.pt"))
    holding_model.load_state_dict(load_torch_state(path / "holding_lifecycle_model.pt"))
    flat_model.eval()
    holding_model.eval()
    return {
        "manifest": manifest,
        "config": config,
        "flat_feature_columns": flat_features,
        "holding_feature_columns": holding_features,
        "flat_model": flat_model,
        "holding_model": holding_model,
        "flat_scaler": scaler_from_json(path / "flat_scaler.json"),
        "holding_scaler": scaler_from_json(path / "holding_scaler.json"),
    }


def load_flat_dataset(path: Path, feature_columns: list[str], sessions: set[tuple[str, str]]) -> pd.DataFrame:
    columns = list(dict.fromkeys([*CORE_FLAT_COLUMNS, *feature_columns]))
    frame = pd.read_parquet(path, columns=columns)
    frame = filter_sessions(frame, sessions)
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["entry_ask"] = pd.to_numeric(frame["entry_ask"], errors="coerce")
    frame["entry_premium"] = pd.to_numeric(frame["entry_premium"], errors="coerce")
    frame["entry_premium"] = frame["entry_premium"].fillna(frame["entry_ask"] * ConservativeReplayConfig().contract_multiplier)
    return frame[frame["decision_dt"].notna()].sort_values(["split", "session", "decision_dt", "candidate_uid"]).reset_index(drop=True)


def load_holding_dataset(path: Path, feature_columns: list[str], sessions: set[tuple[str, str]]) -> pd.DataFrame:
    columns = list(dict.fromkeys([*CORE_HOLDING_COLUMNS, *feature_columns]))
    frame = pd.read_parquet(path, columns=columns)
    frame = filter_sessions(frame, sessions)
    frame["state_dt"] = pd.to_datetime(frame["state_time"], utc=True, errors="coerce")
    return frame[frame["state_dt"].notna()].sort_values(["candidate_uid", "state_dt"]).reset_index(drop=True)


def load_baseline_actions(path: Path, sessions: set[tuple[str, str]], *, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[pd.to_numeric(frame["seed"], errors="coerce").fillna(0).astype(int).eq(int(seed))].copy()
    frame = filter_sessions(frame, sessions)
    frame["decision_dt"] = pd.to_datetime(frame["decision_dt"], utc=True, errors="coerce")
    frame["baseline_exit_dt"] = pd.to_datetime(frame["baseline_exit_time"], utc=True, errors="coerce")
    return frame[frame["decision_dt"].notna()].sort_values(["split", "session", "decision_dt"]).reset_index(drop=True)


def attach_predictions(frame: pd.DataFrame, bundle: dict[str, Any], *, head: str) -> pd.DataFrame:
    features = bundle[f"{head}_feature_columns"]
    scaler = bundle[f"{head}_scaler"]
    model = bundle[f"{head}_model"]
    predictions = predict(model, scaler, frame, features, bundle["config"])
    out = frame.copy()
    out["predicted_advantage"] = predictions["predicted_advantage"]
    out["positive_probability"] = predictions["positive_probability"]
    out["tail_probability"] = predictions["tail_probability"]
    out["conservative_gate_without_affordability"] = (
        out["predicted_advantage"].ge(bundle["config"].min_advantage_margin)
        & out["positive_probability"].ge(bundle["config"].positive_probability_min)
        & out["tail_probability"].le(bundle["config"].tail_probability_max)
    )
    return out


def predict(
    model: ConservativeAdvantageMLP,
    scaler: FeatureScaler,
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: ConservativeNeuralPolicyConfig,
) -> dict[str, np.ndarray]:
    if frame.empty:
        return {
            "predicted_advantage": np.asarray([], dtype=float),
            "positive_probability": np.asarray([], dtype=float),
            "tail_probability": np.asarray([], dtype=float),
        }
    x = scaler.transform(frame[feature_columns].to_numpy(dtype=np.float32))
    adv: list[np.ndarray] = []
    pos: list[np.ndarray] = []
    tail: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 16_384):
            out = model(torch.from_numpy(x[start : start + 16_384]))
            adv.append(out["advantage"].cpu().numpy() * config.target_scale)
            pos.append(torch.sigmoid(out["positive_logit"]).cpu().numpy())
            tail.append(torch.sigmoid(out["tail_logit"]).cpu().numpy())
    return {
        "predicted_advantage": np.concatenate(adv),
        "positive_probability": np.concatenate(pos),
        "tail_probability": np.concatenate(tail),
    }


def simulate_stress_grid(
    flat: pd.DataFrame,
    baseline: pd.DataFrame,
    holding: pd.DataFrame,
    holding_indices: dict[str, np.ndarray],
    bundle: dict[str, Any],
    *,
    slippage_per_side: float,
    seed: int,
) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    replay_config = ConservativeReplayConfig()
    for split, split_flat in flat.groupby("split", sort=True):
        split_baseline = baseline[baseline["split"].astype(str).eq(str(split))].copy()
        sim = simulate_one_split(
            split_flat,
            split_baseline,
            holding,
            holding_indices,
            bundle,
            slippage_per_side=float(slippage_per_side),
            seed=int(seed),
            replay_config=replay_config,
        )
        split_summaries[str(split)] = sim["summary"]
        trades.extend(sim["trades"])
        decisions.extend(sim["decisions"])
    totals = aggregate_totals(split_summaries)
    return {
        "summary": {
            "slippage_per_side": float(slippage_per_side),
            "seed": int(seed),
            "splits": split_summaries,
            "totals": totals,
        },
        "trades": trades,
        "decisions": decisions,
    }


def simulate_one_split(
    flat: pd.DataFrame,
    baseline: pd.DataFrame,
    holding: pd.DataFrame,
    holding_indices: dict[str, np.ndarray],
    bundle: dict[str, Any],
    *,
    slippage_per_side: float,
    seed: int,
    replay_config: ConservativeReplayConfig,
) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    skipped = {
        "open_position_events": 0,
        "candidate_gate": 0,
        "baseline_wait": 0,
        "baseline_missing": 0,
        "baseline_unaffordable": 0,
        "challenger_missing_holding_path": 0,
        "challenger_invalid_exit": 0,
    }
    events = flat[["split", "session", "decision_dt", "decision_time"]].drop_duplicates().sort_values(["session", "decision_dt"])
    event_indices = flat.groupby(["split", "session", "decision_dt"], sort=False).indices
    baseline_by_key = {
        event_key(row["split"], row["session"], row["decision_dt"]): row
        for row in baseline.to_dict("records")
    }
    equity = float(replay_config.starting_cash)
    open_until: dict[str, pd.Timestamp] = {}
    for event in events.itertuples(index=False):
        split = str(event.split)
        session = str(event.session)
        decision_dt = pd.Timestamp(event.decision_dt)
        key = event_key(split, session, decision_dt)
        indices = event_indices.get((split, session, decision_dt), [])
        candidates = flat.iloc[indices].copy()
        equity_before = float(equity)
        if open_until.get(session) is not None and decision_dt < pd.Timestamp(open_until[session]):
            skipped["open_position_events"] += 1
            decisions.append(decision_row(event, slippage_per_side, equity_before, equity, "skip_open_position", "open_position", ""))
            continue

        stressed_candidates = candidates.copy()
        stressed_candidates["entry_premium"] = pd.to_numeric(stressed_candidates["entry_premium"], errors="coerce") + (
            float(slippage_per_side) * replay_config.contract_multiplier
        )
        selected_idx, selection_reason = select_conservative_entry_candidate(
            stressed_candidates,
            equity=equity,
            policy_config=bundle["config"],
            replay_config=replay_config,
        )
        if selected_idx is not None:
            challenger = challenger_trade(
                candidates.loc[selected_idx],
                holding,
                holding_indices,
                bundle,
                equity_before=equity_before,
                slippage_per_side=float(slippage_per_side),
                replay_config=replay_config,
            )
            if challenger["trade"] is not None:
                trade = challenger["trade"]
                equity += float(trade["pnl"])
                trade["account_equity_after"] = float(equity)
                trades.append(trade)
                open_until[session] = pd.Timestamp(trade["exit_time"])
                decisions.append(decision_row(event, slippage_per_side, equity_before, equity, "challenger_entry", selection_reason, trade["candidate_uid"]))
                continue
            skipped[challenger["skip_reason"]] += 1

        skipped["candidate_gate"] += 1
        baseline_trade = protocol101_defer_trade(
            baseline_by_key.get(key),
            candidates,
            equity_before=equity_before,
            slippage_per_side=float(slippage_per_side),
            replay_config=replay_config,
        )
        if baseline_trade["trade"] is not None:
            trade = baseline_trade["trade"]
            equity += float(trade["pnl"])
            trade["account_equity_after"] = float(equity)
            trades.append(trade)
            open_until[session] = pd.Timestamp(trade["exit_time"])
            decisions.append(decision_row(event, slippage_per_side, equity_before, equity, "protocol101_defer_enter", baseline_trade["reason"], trade["candidate_uid"]))
        else:
            skipped[baseline_trade["skip_reason"]] += 1
            decisions.append(decision_row(event, slippage_per_side, equity_before, equity, "protocol101_defer_wait", baseline_trade["reason"], ""))
    summary = summarize_replay_trades(trades, event_count=len(events), starting_cash=replay_config.starting_cash, skipped=skipped)
    summary["protocol101_same_scope_pnl"] = baseline_same_scope_pnl(baseline, slippage_per_side=slippage_per_side, multiplier=replay_config.contract_multiplier)
    summary["delta_vs_protocol101_same_scope"] = float(summary["total_pnl"] - summary["protocol101_same_scope_pnl"])
    summary["challenger_entries"] = int(sum(1 for trade in trades if trade["source"] == "challenger"))
    summary["protocol101_defer_entries"] = int(sum(1 for trade in trades if trade["source"] == "protocol101_defer"))
    return {"trades": trades, "decisions": decisions, "summary": summary}


def challenger_trade(
    row: pd.Series,
    holding: pd.DataFrame,
    holding_indices: dict[str, np.ndarray],
    bundle: dict[str, Any],
    *,
    equity_before: float,
    slippage_per_side: float,
    replay_config: ConservativeReplayConfig,
) -> dict[str, Any]:
    candidate_uid = str(row.get("candidate_uid", ""))
    indices = holding_indices.get(candidate_uid)
    if indices is None or len(indices) == 0:
        return {"trade": None, "skip_reason": "challenger_missing_holding_path"}
    path = holding.iloc[indices].copy().sort_values("state_dt").reset_index(drop=True)
    if path.empty:
        return {"trade": None, "skip_reason": "challenger_missing_holding_path"}
    path = attach_predictions(path, bundle, head="holding")
    exit_idx, exit_reason = choose_conservative_holding_exit(
        path,
        path[["predicted_advantage", "positive_probability", "tail_probability"]],
        policy_config=bundle["config"],
    )
    if exit_idx is None:
        return {"trade": None, "skip_reason": "challenger_invalid_exit"}
    exit_row = path.iloc[int(exit_idx)]
    entry_ask = finite(row.get("entry_ask"))
    exit_bid = finite(exit_row.get("bid"))
    if not math.isfinite(entry_ask) or not math.isfinite(exit_bid):
        return {"trade": None, "skip_reason": "challenger_invalid_exit"}
    entry_dt = pd.Timestamp(row["decision_dt"])
    exit_dt = pd.Timestamp(exit_row["state_dt"])
    pnl = (exit_bid - entry_ask - 2.0 * float(slippage_per_side)) * replay_config.contract_multiplier
    return {
        "trade": {
            "slippage_per_side": float(slippage_per_side),
            "split": str(row.get("split", "")),
            "session": str(row.get("session", "")),
            "decision_time": entry_dt.isoformat(),
            "exit_time": exit_dt.isoformat(),
            "duration_minutes": float((exit_dt - entry_dt).total_seconds() / 60.0),
            "candidate_uid": candidate_uid,
            "contract_id": str(row.get("contract_id", "")),
            "right": str(row.get("right", "")),
            "offset": finite(row.get("offset"), 0.0),
            "entry_ask": entry_ask,
            "exit_bid": exit_bid,
            "pnl": float(pnl),
            "account_equity_before": float(equity_before),
            "account_equity_after": float(equity_before),
            "predicted_advantage": finite(row.get("predicted_advantage"), 0.0),
            "positive_probability": finite(row.get("positive_probability"), 0.0),
            "tail_probability": finite(row.get("tail_probability"), 1.0),
            "exit_reason": exit_reason,
            "source": "challenger",
            "strategy": ROLE_LABEL,
        },
        "skip_reason": "",
    }


def protocol101_defer_trade(
    baseline_row: dict[str, Any] | None,
    candidates: pd.DataFrame,
    *,
    equity_before: float,
    slippage_per_side: float,
    replay_config: ConservativeReplayConfig,
) -> dict[str, Any]:
    if baseline_row is None:
        return {"trade": None, "reason": "missing_protocol101_event_action", "skip_reason": "baseline_missing"}
    action = str(baseline_row.get("protocol101_action", ""))
    if action != "enter":
        return {"trade": None, "reason": f"protocol101_{action or 'wait'}", "skip_reason": "baseline_wait"}
    match = match_baseline_candidate(candidates, baseline_row)
    premium = finite(match.get("entry_premium") if match is not None else math.nan, 0.0) + (
        float(slippage_per_side) * replay_config.contract_multiplier
    )
    if premium > float(equity_before):
        return {"trade": None, "reason": "protocol101_unaffordable_after_challenger_equity", "skip_reason": "baseline_unaffordable"}
    entry_dt = pd.Timestamp(baseline_row["decision_dt"])
    exit_dt = pd.Timestamp(baseline_row["baseline_exit_dt"])
    pnl = finite(baseline_row.get("baseline_trade_pnl"), 0.0) - 2.0 * float(slippage_per_side) * replay_config.contract_multiplier
    return {
        "trade": {
            "slippage_per_side": float(slippage_per_side),
            "split": str(baseline_row.get("split", "")),
            "session": str(baseline_row.get("session", "")),
            "decision_time": entry_dt.isoformat(),
            "exit_time": exit_dt.isoformat(),
            "duration_minutes": float((exit_dt - entry_dt).total_seconds() / 60.0),
            "candidate_uid": str(baseline_row.get("surface_candidate_uid", "")),
            "contract_id": str(baseline_row.get("contract_id", "")),
            "right": str(match.get("right", "")) if match is not None else "",
            "offset": finite(match.get("offset") if match is not None else 0.0, 0.0),
            "entry_ask": finite(match.get("entry_ask") if match is not None else math.nan),
            "exit_bid": math.nan,
            "pnl": float(pnl),
            "account_equity_before": float(equity_before),
            "account_equity_after": float(equity_before),
            "predicted_advantage": 0.0,
            "positive_probability": 0.0,
            "tail_probability": 0.0,
            "exit_reason": "protocol101_baseline_exit",
            "source": "protocol101_defer",
            "strategy": ROLE_LABEL,
        },
        "reason": "protocol101_enter",
        "skip_reason": "",
    }


def match_baseline_candidate(candidates: pd.DataFrame, baseline_row: dict[str, Any]) -> pd.Series | None:
    uid = str(baseline_row.get("surface_candidate_uid", ""))
    if uid:
        match = candidates[candidates["candidate_uid"].astype(str).eq(uid)]
        if not match.empty:
            return match.iloc[0]
    contract_id = str(baseline_row.get("contract_id", ""))
    if contract_id:
        match = candidates[candidates["contract_id"].astype(str).eq(contract_id)]
        if not match.empty:
            return match.iloc[0]
    return None


def aggregate_totals(split_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_pnl = float(sum(item.get("total_pnl", 0.0) for item in split_summaries.values()))
    baseline_pnl = float(sum(item.get("protocol101_same_scope_pnl", 0.0) for item in split_summaries.values()))
    trades = int(sum(item.get("trades", 0) for item in split_summaries.values()))
    challenger_entries = int(sum(item.get("challenger_entries", 0) for item in split_summaries.values()))
    protocol101_entries = int(sum(item.get("protocol101_defer_entries", 0) for item in split_summaries.values()))
    return {
        "total_pnl": total_pnl,
        "protocol101_same_scope_pnl": baseline_pnl,
        "delta_vs_protocol101_same_scope": float(total_pnl - baseline_pnl),
        "trades": trades,
        "challenger_entries": challenger_entries,
        "protocol101_defer_entries": protocol101_entries,
    }


def baseline_same_scope_pnl(baseline: pd.DataFrame, *, slippage_per_side: float, multiplier: float) -> float:
    entries = baseline[baseline["protocol101_action"].astype(str).eq("enter")]
    if entries.empty:
        return 0.0
    pnl = pd.to_numeric(entries["baseline_trade_pnl"], errors="coerce").fillna(0.0)
    stress = 2.0 * float(slippage_per_side) * float(multiplier)
    return float((pnl - stress).sum())


def limit_events_per_split(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    events = frame[["split", "session", "decision_dt"]].drop_duplicates().sort_values(["split", "session", "decision_dt"])
    keep = events.groupby("split", sort=True).head(int(limit))
    return frame.merge(keep.assign(_keep=1), on=["split", "session", "decision_dt"], how="inner").drop(columns=["_keep"])


def filter_sessions(frame: pd.DataFrame, sessions: set[tuple[str, str]]) -> pd.DataFrame:
    if frame.empty or not sessions:
        return frame.iloc[0:0].copy()
    allowed = pd.MultiIndex.from_tuples(sessions, names=["split", "session"])
    current = pd.MultiIndex.from_frame(frame[["split", "session"]].astype(str))
    return frame.loc[current.isin(allowed)].copy()


def included_session_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    included = frame[frame["included"].astype(bool)].copy()
    return set(zip(included["split"].astype(str), included["session"].astype(str)))


def scaler_from_json(path: Path) -> FeatureScaler:
    payload = load_json(path)
    return FeatureScaler(
        fill=np.asarray(payload["fill"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
    )


def load_torch_state(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def event_key(split: str, session: str, decision_dt: Any) -> tuple[str, str, pd.Timestamp]:
    return (str(split), str(session), pd.Timestamp(decision_dt))


def decision_row(event: Any, slippage: float, equity_before: float, equity_after: float, decision: str, reason: str, candidate_uid: str) -> dict[str, Any]:
    return {
        "slippage_per_side": float(slippage),
        "split": str(event.split),
        "session": str(event.session),
        "decision_time": pd.Timestamp(event.decision_dt).isoformat(),
        "decision": decision,
        "reason": reason,
        "candidate_uid": str(candidate_uid),
        "account_equity_before": float(equity_before),
        "account_equity_after": float(equity_after),
    }


def parse_slippage_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def slippage_suffix(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def interpretation(total_challenger_entries: int) -> str:
    if total_challenger_entries == 0:
        return (
            "The trained flat-entry head never cleared the conservative override gate in strict replay. "
            "The artifact is therefore an abstention/pass-through policy, not evidence of Protocol101 improvement."
        )
    return (
        "The trained model produced challenger overrides in strict replay, but promotion remains blocked until "
        "fill calibration, untouched holdout data, live no-order parity, and formal validation controls are complete."
    )


def next_required_evidence(total_challenger_entries: int) -> list[str]:
    items = []
    if total_challenger_entries == 0:
        items.append("Diagnose why the flat-entry head is over-conservative before retraining: target imbalance, margin thresholds, and positive-class weighting.")
    else:
        items.append("Attribute every challenger override versus Protocol101 by side, premium, time bucket, moneyness, and lifecycle exit reason.")
    items.extend(
        [
            "Keep Protocol101 as the paper default.",
            "Collect paper/no-order fill evidence before stochastic fill calibration.",
            "Reserve and acquire untouched holdout data before any promotion claim.",
            "Build live no-order full-action parity for this exact policy contract.",
            "Add formal validation controls before a better-than-Protocol101 claim.",
        ]
    )
    return items


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Trained policy: `{payload['trained_policy_label']}`",
        f"Paper default baseline: `{payload['paper_default_baseline']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training in this runner: no",
        f"Decision: `{payload['decision']}`",
        f"Challenge allowed: `{payload['challenge_allowed']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Scope",
        "",
        f"- Seed: `{payload['data_scope']['seed']}`",
        f"- Included sessions: `{payload['data_scope']['included_sessions']}`",
        f"- Flat rows loaded: `{payload['data_scope']['flat_rows_loaded']}`",
        f"- Holding rows loaded: `{payload['data_scope']['holding_rows_loaded']}`",
        f"- Baseline event rows: `{payload['data_scope']['baseline_event_rows']}`",
        "",
        "## Stress Results",
        "",
        "| slippage | split | PnL | same-scope Protocol101 | delta | trades | challenger | defer | PF |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stress in payload["stress_results"]:
        slip = stress["slippage_per_side"]
        for split, item in stress["splits"].items():
            lines.append(
                f"| {slip:.2f} | {split} | {item.get('total_pnl', 0.0):.2f} | "
                f"{item.get('protocol101_same_scope_pnl', 0.0):.2f} | {item.get('delta_vs_protocol101_same_scope', 0.0):.2f} | "
                f"{item.get('trades', 0)} | {item.get('challenger_entries', 0)} | {item.get('protocol101_defer_entries', 0)} | "
                f"{fmt(item.get('profit_factor', 0.0))} |"
            )
        totals = stress["totals"]
        lines.append(
            f"| {slip:.2f} | total | {totals['total_pnl']:.2f} | {totals['protocol101_same_scope_pnl']:.2f} | "
            f"{totals['delta_vs_protocol101_same_scope']:.2f} | {totals['trades']} | {totals['challenger_entries']} | "
            f"{totals['protocol101_defer_entries']} |  |"
        )
    lines.extend(["", "## Remaining Challenge Blockers", ""])
    lines.extend(f"- {item}" for item in payload["challenge_blockers"])
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(["", "## Outputs", ""])
    for name, path in payload["outputs"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def append_ledger(payload: dict[str, Any], out_dir: Path) -> None:
    ledger = Path("v4/ledger/RESEARCH_LEDGER.md")
    if not ledger.exists():
        return
    marker = f"## {ROLE_LABEL}"
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
                    "- Paid data downloaded: no",
                    "- Broker endpoint called: no",
                    "- Model training: no",
                    f"- Decision: `{payload['decision']}`",
                    f"- Report: `{out_dir / 'report.md'}`",
                    "- Result: Strict replay complete; Protocol101 challenge remains blocked.",
                ]
            )
            + "\n"
        )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any) -> str:
    out = finite(value)
    return "" if not math.isfinite(out) else f"{out:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
