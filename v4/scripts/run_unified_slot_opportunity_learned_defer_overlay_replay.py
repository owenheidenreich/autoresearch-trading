"""Strict replay with the learned slot opportunity-cost defer overlay."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4.model.unified_conservative_neural_policy import ConservativeNeuralPolicyConfig
from v4.model.unified_conservative_neural_replay import ConservativeReplayConfig, summarize_replay_trades
from v4.model.unified_slot_opportunity_cost_estimator import estimator_predictions, validate_slot_cost_feature_columns
from v4.model.unified_slot_opportunity_defer import SlotOpportunityDeferConfig, slot_opportunity_defer_decision
from v4.scripts.run_unified_conservative_neural_policy_strict_replay import (
    aggregate_totals,
    attach_predictions,
    baseline_same_scope_pnl,
    challenger_trade,
    decision_row,
    event_key,
    included_session_keys,
    limit_events_per_split,
    load_baseline_actions,
    load_holding_dataset,
    load_json,
    load_flat_dataset,
    load_policy_bundle,
    parse_slippage_grid,
    protocol101_defer_trade,
    slippage_suffix,
    write_json,
)


ROLE_LABEL = "REPLAY_UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_V1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/unified_slot_opportunity_learned_defer_overlay_replay")
DEFAULT_DOC_PATH = Path("v4/docs/UNIFIED_SLOT_OPPORTUNITY_LEARNED_DEFER_OVERLAY_REPLAY.md")
DEFAULT_MODEL_ARTIFACTS = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_v1/model_artifacts")
DEFAULT_TRAINING_SUMMARY = Path("v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_v1/summary.json")
DEFAULT_ESTIMATOR = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/model_artifacts/slot_opportunity_cost_estimator.joblib")
DEFAULT_ESTIMATOR_SUMMARY = Path("v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/summary.json")
DEFAULT_FLAT_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_270_full_surface_action_advantage_dataset/full_surface_action_advantage.parquet")
DEFAULT_HOLDING_DATASET = Path("v4/audit/autoresearch/v4_aplus_hypothesis_274_position_state_action_advantage_dataset/position_state_action_advantage.parquet")
DEFAULT_BASELINE_ACTIONS = Path("v4/audit/autoresearch/unified_protocol101_baseline_attachment/protocol101_baseline_event_actions_training_scope.parquet")
DEFAULT_SESSION_MANIFEST = Path("v4/audit/autoresearch/unified_serial_dp_oracle/serial_dp_session_manifest.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifacts", type=Path, default=DEFAULT_MODEL_ARTIFACTS)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--slot-estimator", type=Path, default=DEFAULT_ESTIMATOR)
    parser.add_argument("--slot-estimator-summary", type=Path, default=DEFAULT_ESTIMATOR_SUMMARY)
    parser.add_argument("--flat-dataset", type=Path, default=DEFAULT_FLAT_DATASET)
    parser.add_argument("--holding-dataset", type=Path, default=DEFAULT_HOLDING_DATASET)
    parser.add_argument("--baseline-actions", type=Path, default=DEFAULT_BASELINE_ACTIONS)
    parser.add_argument("--session-manifest", type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--slippage-grid", default="0,0.10,0.25")
    parser.add_argument("--max-events-per-split", type=int, default=0)
    parser.add_argument("--min-net-advantage-margin", type=float, default=250.0)
    parser.add_argument("--blocked-cost-uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--max-blocked-protocol101-entries", type=int, default=1)
    parser.add_argument("--skip-doc", action="store_true")
    parser.add_argument("--skip-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_bundle = load_policy_bundle(args.model_artifacts)
    estimator_bundle = joblib.load(args.slot_estimator)
    validate_slot_cost_feature_columns(list(estimator_bundle["feature_columns"]))
    training_summary = load_json(args.training_summary)
    estimator_summary = load_json(args.slot_estimator_summary)
    session_manifest = pd.read_csv(args.session_manifest)
    included_sessions = included_session_keys(session_manifest)

    flat_feature_columns = list(
        dict.fromkeys([*policy_bundle["flat_feature_columns"], *list(estimator_bundle["feature_columns"])])
    )
    flat = load_flat_dataset(args.flat_dataset, flat_feature_columns, included_sessions)
    holding = load_holding_dataset(args.holding_dataset, policy_bundle["holding_feature_columns"], included_sessions)
    baseline = load_baseline_actions(args.baseline_actions, included_sessions, seed=int(args.seed))
    if int(args.max_events_per_split) > 0:
        flat = limit_events_per_split(flat, int(args.max_events_per_split))
        event_keys = flat[["split", "session", "decision_dt"]].drop_duplicates()
        baseline = baseline.merge(event_keys.assign(_keep=1), on=["split", "session", "decision_dt"], how="inner").drop(columns=["_keep"])

    flat = attach_predictions(flat, policy_bundle, head="flat")
    flat = attach_slot_estimates(flat, estimator_bundle)
    holding_indices = holding.groupby("candidate_uid", sort=False).indices
    slippages = parse_slippage_grid(args.slippage_grid)
    defer_config = SlotOpportunityDeferConfig(
        min_net_advantage_margin=float(args.min_net_advantage_margin),
        blocked_cost_uncertainty_weight=float(args.blocked_cost_uncertainty_weight),
        max_blocked_protocol101_entries=int(args.max_blocked_protocol101_entries),
    )

    stress_results: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    for slippage in slippages:
        replay = simulate_stress_grid(
            flat,
            baseline,
            holding,
            holding_indices,
            policy_bundle,
            defer_config=defer_config,
            slippage_per_side=float(slippage),
            seed=int(args.seed),
        )
        stress_results.append(replay["summary"])
        all_trades.extend(replay["trades"])
        all_decisions.extend(replay["decisions"])
        suffix = slippage_suffix(slippage)
        pd.DataFrame(replay["trades"]).to_csv(args.out_dir / f"trades_slippage_{suffix}.csv", index=False)
        pd.DataFrame(replay["decisions"]).to_csv(args.out_dir / f"decisions_slippage_{suffix}.csv", index=False)

    decision = decide(stress_results)
    payload = {
        "role_label": ROLE_LABEL,
        "what_is_this": "strict replay / learned slot opportunity-cost defer overlay on the flat-calibrated conservative policy",
        "changes_paper_default": False,
        "paper_default_baseline": "PAPER_DEFAULT_PROTOCOL101",
        "paid_data_downloaded_by_runner": False,
        "broker_endpoint_called": False,
        "live_orders": False,
        "model_training": False,
        "strict_replay_run": True,
        "challenge_allowed": False,
        "decision": decision,
        "interpretation": interpretation(decision, stress_results),
        "defer_config": defer_config.to_dict(),
        "policy_config": policy_bundle["config"].to_dict(),
        "estimator_decision": estimator_summary.get("decision", "missing"),
        "estimator_q1_auc": estimator_summary.get("metrics", {}).get("q1_2026", {}).get("positive_auc"),
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
            "policy_model_artifacts": str(args.model_artifacts),
            "policy_training_summary": str(args.training_summary),
            "slot_estimator": str(args.slot_estimator),
            "slot_estimator_summary": str(args.slot_estimator_summary),
        },
        "challenge_blockers": [
            "calibrated stochastic fill model unavailable",
            "untouched holdout data pending",
            "live no-order full-action parity pending",
            "formal validation controls pending",
        ],
        "next_required_evidence": next_required_evidence(decision),
        "outputs": {
            "summary": str(args.out_dir / "summary.json"),
            "report": str(args.out_dir / "report.md"),
            "doc": None if args.skip_doc else str(args.doc_path),
        },
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


def attach_slot_estimates(frame: pd.DataFrame, estimator_bundle: dict[str, Any]) -> pd.DataFrame:
    predictions = estimator_predictions(estimator_bundle, frame)
    out = frame.copy()
    for column in predictions.columns:
        out[column] = predictions[column].to_numpy()
    return out


def select_learned_defer_entry_candidate(
    candidates: pd.DataFrame,
    *,
    equity: float,
    policy_config: ConservativeNeuralPolicyConfig,
    defer_config: SlotOpportunityDeferConfig,
) -> tuple[int | None, str]:
    if candidates.empty:
        return None, "empty_candidate_set"
    required = {
        "predicted_advantage",
        "positive_probability",
        "tail_probability",
        "entry_premium",
        "estimated_blocked_protocol101_cost",
        "blocked_cost_uncertainty",
        "estimated_blocked_entries",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"missing learned defer candidate columns: {missing}")
    numeric = candidates.copy()
    for column in required:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    base_allowed = (
        numeric["predicted_advantage"].ge(float(policy_config.min_advantage_margin))
        & numeric["positive_probability"].ge(float(policy_config.positive_probability_min))
        & numeric["tail_probability"].le(float(policy_config.tail_probability_max))
        & numeric["entry_premium"].gt(0.0)
        & numeric["entry_premium"].le(float(equity))
    )
    if not bool(base_allowed.any()):
        return None, "no_candidate_clears_policy_gate"
    adjusted: dict[int, float] = {}
    for idx, row in numeric.loc[base_allowed].iterrows():
        decision = slot_opportunity_defer_decision(
            predicted_challenger_advantage=float(row["predicted_advantage"]),
            estimated_blocked_protocol101_cost=float(row["estimated_blocked_protocol101_cost"]),
            blocked_cost_uncertainty=float(row["blocked_cost_uncertainty"]),
            estimated_blocked_entries=int(round(max(0.0, float(row["estimated_blocked_entries"])))),
            config=defer_config,
        )
        if decision["allowed"]:
            adjusted[int(idx)] = float(decision["adjusted_advantage"])
    if not adjusted:
        return None, "no_candidate_clears_learned_slot_defer_gate"
    return max(adjusted, key=adjusted.get), "challenger_entry_selected_after_learned_slot_defer"


def simulate_stress_grid(
    flat: pd.DataFrame,
    baseline: pd.DataFrame,
    holding: pd.DataFrame,
    holding_indices: dict[str, np.ndarray],
    bundle: dict[str, Any],
    *,
    defer_config: SlotOpportunityDeferConfig,
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
            defer_config=defer_config,
            slippage_per_side=float(slippage_per_side),
            seed=int(seed),
            replay_config=replay_config,
        )
        split_summaries[str(split)] = sim["summary"]
        trades.extend(sim["trades"])
        decisions.extend(sim["decisions"])
    totals = aggregate_totals(split_summaries)
    return {"summary": {"slippage_per_side": float(slippage_per_side), "seed": int(seed), "splits": split_summaries, "totals": totals}, "trades": trades, "decisions": decisions}


def simulate_one_split(
    flat: pd.DataFrame,
    baseline: pd.DataFrame,
    holding: pd.DataFrame,
    holding_indices: dict[str, np.ndarray],
    bundle: dict[str, Any],
    *,
    defer_config: SlotOpportunityDeferConfig,
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
        "learned_slot_defer_gate": 0,
    }
    events = flat[["split", "session", "decision_dt", "decision_time"]].drop_duplicates().sort_values(["session", "decision_dt"])
    event_indices = flat.groupby(["split", "session", "decision_dt"], sort=False).indices
    baseline_by_key = {event_key(row["split"], row["session"], row["decision_dt"]): row for row in baseline.to_dict("records")}
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
        selected_idx, selection_reason = select_learned_defer_entry_candidate(
            stressed_candidates,
            equity=equity,
            policy_config=bundle["config"],
            defer_config=defer_config,
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
                add_slot_fields(trade, candidates.loc[selected_idx])
                equity += float(trade["pnl"])
                trade["account_equity_after"] = float(equity)
                trades.append(trade)
                open_until[session] = pd.Timestamp(trade["exit_time"])
                decisions.append(decision_row(event, slippage_per_side, equity_before, equity, "challenger_entry", selection_reason, trade["candidate_uid"]))
                continue
            skipped[challenger["skip_reason"]] += 1

        skipped["candidate_gate"] += 1
        if selection_reason == "no_candidate_clears_learned_slot_defer_gate":
            skipped["learned_slot_defer_gate"] += 1
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


def add_slot_fields(trade: dict[str, Any], row: pd.Series) -> None:
    trade["estimated_blocked_protocol101_cost"] = finite(row.get("estimated_blocked_protocol101_cost"), 0.0)
    trade["estimated_blocked_entries"] = finite(row.get("estimated_blocked_entries"), 0.0)
    trade["blocked_cost_positive_probability"] = finite(row.get("blocked_cost_positive_probability"), 0.0)
    trade["blocked_cost_uncertainty"] = finite(row.get("blocked_cost_uncertainty"), 0.0)


def decide(stress_results: list[dict[str, Any]]) -> str:
    if not q1_q3_stress_pass(stress_results):
        return "learned_slot_opportunity_defer_overlay_replay_blocks_training_q1_q3_stress_failed"
    total_challenger = int(sum(item["totals"].get("challenger_entries", 0) for item in stress_results))
    if total_challenger <= 0:
        return "learned_slot_opportunity_defer_overlay_replay_safe_but_deferred_all_overconservative"
    return "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training"


def q1_q3_stress_pass(stress_results: list[dict[str, Any]]) -> bool:
    for stress in stress_results:
        for split in ("q1_2026", "q3_2025"):
            delta = finite(stress.get("splits", {}).get(split, {}).get("delta_vs_protocol101_same_scope"), -math.inf)
            if delta < -1e-9:
                return False
    return True


def interpretation(decision: str, stress_results: list[dict[str, Any]]) -> str:
    total_challenger = int(sum(item["totals"].get("challenger_entries", 0) for item in stress_results))
    if decision.endswith("q1_q3_stress_failed"):
        return "The learned slot-cost overlay did not repair the protected Q1/Q3 stress deltas; more neural training remains blocked."
    if total_challenger <= 0:
        return "The learned slot-cost overlay is safe but over-conservative: it defers all challenger overrides back to Protocol101."
    return "The learned slot-cost overlay preserves nonnegative Q1/Q3 stress deltas while allowing challenger overrides; it is ready as a fixed guardrail for the next preregistered training run."


def next_required_evidence(decision: str) -> list[str]:
    if decision == "learned_slot_opportunity_defer_overlay_replay_ready_for_next_preregistered_training":
        return [
            "Freeze this learned defer overlay configuration before the next preregistered neural policy run.",
            "Keep Protocol101 as paper default until fill, untouched holdout data, live no-order parity, and formal validation controls pass.",
        ]
    if decision == "learned_slot_opportunity_defer_overlay_replay_safe_but_deferred_all_overconservative":
        return [
            "Decide whether an all-defer overlay is acceptable as a safety guardrail or whether estimator calibration must be sharpened before retraining.",
            "Do not treat all-defer behavior as evidence of Protocol101 improvement.",
        ]
    return [
        "Diagnose learned-estimator false negatives/underestimated blocked costs in Q1/Q3.",
        "Do not run another neural policy until the learned overlay passes Q1/Q3 stress replay.",
    ]


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {ROLE_LABEL}",
        "",
        f"What is this: {payload['what_is_this']}",
        "Does it change the paper-trading default: no",
        f"Paper default baseline: `{payload['paper_default_baseline']}`",
        "Paid data downloaded: no",
        "Broker endpoint called: no",
        "Model training in this runner: no",
        f"Decision: `{payload['decision']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Learned Defer Config",
        "",
        f"- Min net advantage margin: `{payload['defer_config']['min_net_advantage_margin']}`",
        f"- Blocked-cost uncertainty weight: `{payload['defer_config']['blocked_cost_uncertainty_weight']}`",
        f"- Max blocked Protocol101 entries: `{payload['defer_config']['max_blocked_protocol101_entries']}`",
        f"- Estimator decision: `{payload['estimator_decision']}`",
        f"- Estimator Q1 AUC: `{payload['estimator_q1_auc']}`",
        "",
        "## Stress Results",
        "",
        "| slippage | split | PnL | same-scope Protocol101 | delta | trades | challenger | defer | learned gate skips | PF |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stress in payload["stress_results"]:
        slip = stress["slippage_per_side"]
        for split, item in stress["splits"].items():
            lines.append(
                f"| {slip:.2f} | {split} | {item.get('total_pnl', 0.0):.2f} | "
                f"{item.get('protocol101_same_scope_pnl', 0.0):.2f} | {item.get('delta_vs_protocol101_same_scope', 0.0):.2f} | "
                f"{item.get('trades', 0)} | {item.get('challenger_entries', 0)} | {item.get('protocol101_defer_entries', 0)} | "
                f"{item.get('skipped_learned_slot_defer_gate', 0)} | {fmt(item.get('profit_factor', 0.0))} |"
            )
        totals = stress["totals"]
        lines.append(
            f"| {slip:.2f} | total | {totals['total_pnl']:.2f} | {totals['protocol101_same_scope_pnl']:.2f} | "
            f"{totals['delta_vs_protocol101_same_scope']:.2f} | {totals['trades']} | {totals['challenger_entries']} | "
            f"{totals['protocol101_defer_entries']} |  |  |"
        )
    lines.extend(["", "## Next Required Evidence", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(payload["next_required_evidence"], start=1))
    lines.extend(["", "## Remaining Challenge Blockers", ""])
    lines.extend(f"- {item}" for item in payload["challenge_blockers"])
    lines.extend(["", "## Artifacts", ""])
    for name, path in payload["artifacts"].items():
        lines.append(f"- {name}: `{path}`")
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
                ]
            )
            + "\n"
        )


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
