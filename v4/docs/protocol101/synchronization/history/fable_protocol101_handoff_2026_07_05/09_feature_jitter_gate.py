"""Stress a fair-contract candidate against vendor-sensitive option-feature jitter.

This is an offline-only robustness gate. It does not train, tune thresholds,
contact brokers/vendors, download data, change defaults, promote models, or use
June/July recorder captures for model selection. It replays an already-trained
candidate over historical validation/diagnostic rows, then perturbs only the
option microstructure inputs that caused paired IBKR-vs-historical drift.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from v4.model.supervised_pilot import VENDOR_MICROSTRUCTURE_JITTER_SCENARIOS
from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    STRICT_REPLAY_CONTRACT_MULTIPLIER,
    candidate_allowed_by_entry_filter,
    candidate_records_from_row,
    load_json_optional,
    load_model,
    load_rows,
    score_candidate_records,
    top_candidate_with_margin,
)
from v4.scripts.run_protocol101_fair_contract_selected_candidate_replay_gate import (
    metrics_for_trades,
    replay_selected_candidates,
)


DEFAULT_RUNNER_PLAN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_vwap_pocket_plateau_threshold/"
    "attempts/attempt_107_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil50_dailyloss500_plateau_s42/"
    "training_runner/runner_plan.json"
)
DEFAULT_TRAINING_RESULT = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_model_search_expanded_jul_dec2025_128_q1_vwap_pocket_plateau_threshold/"
    "attempts/attempt_107_policy0_hgb_blend35_relative_put_near_after0940_vwap_m2_10_cap3_scoreceil50_dailyloss500_plateau_s42/"
    "training_runner/training_result.json"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_attempt107_feature_jitter_gate"
)
FEATURE_JITTER_GATE_VERSION = "Protocol101FeatureJitterGateV1"
IMPLEMENTATION_VERSION = "feature_jitter_gate_v1"


@dataclass(frozen=True)
class JitterScenario:
    name: str
    iv_delta: float = 0.0
    spread_delta: float = 0.0
    spread_frac_delta: float = 0.0
    bid_size_scale: float = 1.0
    ask_size_scale: float = 1.0


SCENARIOS = tuple(
    JitterScenario(name=name, **params)
    for name, params in VENDOR_MICROSTRUCTURE_JITTER_SCENARIOS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-plan", type=Path, default=DEFAULT_RUNNER_PLAN)
    parser.add_argument("--training-result", type=Path, default=DEFAULT_TRAINING_RESULT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "diagnostic_test"],
        choices=("train", "validation", "diagnostic_test"),
    )
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--stress-per-side", type=float, default=0.10)
    parser.add_argument("--min-action-presence-match-rate", type=float, default=0.98)
    parser.add_argument("--min-selected-contract-match-rate", type=float, default=0.90)
    parser.add_argument("--max-pnl-drift-fraction", type=float, default=0.35)
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clone_candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for row in records:
        copy = dict(row)
        if "_features" in copy:
            copy["_features"] = np.asarray(copy["_features"], dtype=np.float32).copy()
        cloned.append(copy)
    return cloned


def apply_option_feature_jitter(
    records: list[dict[str, Any]],
    scenario: JitterScenario,
) -> list[dict[str, Any]]:
    """Return copied records with deterministic option-feature perturbations.

    Feature positions are from ``candidate_feature_vector``:
    3 spread, 4 spread fraction, 5 bid size, 6 ask size, 9 repaired IV.
    """
    output = clone_candidate_records(records)
    if scenario.name == "baseline":
        return output
    for row in output:
        features = np.asarray(row.get("_features"), dtype=np.float32)
        if features.shape[0] <= 9:
            continue
        if math.isfinite(float(features[3])):
            features[3] = max(float(features[3]) + float(scenario.spread_delta), 0.0)
        if math.isfinite(float(features[4])):
            features[4] = max(float(features[4]) + float(scenario.spread_frac_delta), 0.0)
        if math.isfinite(float(features[5])):
            features[5] = max(float(features[5]) * float(scenario.bid_size_scale), 0.0)
        if math.isfinite(float(features[6])):
            features[6] = max(float(features[6]) * float(scenario.ask_size_scale), 0.0)
        if math.isfinite(float(features[9])):
            features[9] = max(float(features[9]) + float(scenario.iv_delta), 0.0)
        row["_features"] = features.astype(np.float32)
    return output


def public_selected_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def select_records_for_split_with_scenario(
    *,
    split: str,
    paths: list[Path],
    loaded: Any,
    scenario: JitterScenario,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    decision_groups: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    decision_rows = 0
    candidate_rows = 0
    threshold_waits = 0
    cooldown_blocks = 0
    session_trade_cap_blocks = 0
    daily_loss_stop_blocks = 0
    entry_filter_blocks = 0
    affordability_blocks = 0
    score_margin_waits = 0
    score_ceiling_waits = 0
    cash = 10_000.0
    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = {}
    realized_pnl_by_session: dict[str, float] = {}
    entry_filter = str(getattr(loaded, "entry_filter", "none") or "none")
    selection_mode = str(getattr(loaded, "selection_mode", "top_score") or "top_score")
    min_score_margin = float(getattr(loaded, "min_score_margin", 0.0) or 0.0)
    max_score_ceiling = max(float(getattr(loaded, "max_score_ceiling", 0.0) or 0.0), 0.0)
    max_trades_per_session = max(int(getattr(loaded, "max_trades_per_session", 0) or 0), 0)
    max_daily_loss = max(float(getattr(loaded, "max_daily_loss", 0.0) or 0.0), 0.0)

    for path in paths:
        session = path.name.removesuffix(".pkl")
        for row in load_rows(path):
            decision_rows += 1
            candidates = candidate_records_from_row(
                row,
                session=session,
                split=split,
                policy_index=loaded.policy_index,
            )
            candidate_rows += len(candidates)
            if candidates:
                decision_groups.append((session, row, candidates))

    all_candidates = [
        candidate
        for _session, _row, candidates in decision_groups
        for candidate in candidates
    ]
    jittered_all_candidates = apply_option_feature_jitter(all_candidates, scenario)
    cursor = 0
    jittered_groups: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for session, row, candidates in decision_groups:
        count = len(candidates)
        jittered_groups.append((session, row, jittered_all_candidates[cursor : cursor + count]))
        cursor += count
    score_candidate_records(jittered_all_candidates, loaded)

    for session, row, candidates in jittered_groups:
        decision_time = row.get("decision_time")
        if not isinstance(decision_time, datetime):
            decision_time = datetime.fromisoformat(str(decision_time))
        next_time = next_time_by_session.get(session)
        if next_time is not None and decision_time < next_time:
            cooldown_blocks += 1
            continue
        if max_trades_per_session > 0 and trades_by_session.get(session, 0) >= max_trades_per_session:
            session_trade_cap_blocks += 1
            continue
        if max_daily_loss > 0.0 and realized_pnl_by_session.get(session, 0.0) <= -max_daily_loss:
            daily_loss_stop_blocks += 1
            continue
        eligible_candidates = [
            candidate
            for candidate in candidates
            if candidate_allowed_by_entry_filter(candidate, row, entry_filter)
        ]
        if not eligible_candidates:
            entry_filter_blocks += 1
            continue
        affordable_candidates: list[dict[str, Any]] = []
        for candidate in eligible_candidates:
            ask = _safe_float(candidate.get("entry_ask"))
            if ask is not None and ask > 0.0 and ask * STRICT_REPLAY_CONTRACT_MULTIPLIER <= cash + 1e-9:
                affordable_candidates.append(candidate)
        if not affordable_candidates:
            affordability_blocks += 1
            continue
        top = top_candidate_with_margin(
            affordable_candidates,
            selection_mode=selection_mode,
        )
        if top is None:
            threshold_waits += 1
            continue
        best, score_margin = top
        if min_score_margin > 0.0 and score_margin < min_score_margin:
            score_margin_waits += 1
            continue
        if max_score_ceiling > 0.0 and float(best["score"]) >= max_score_ceiling:
            score_ceiling_waits += 1
            continue
        if float(best["score"]) < float(loaded.threshold):
            threshold_waits += 1
            continue
        best = public_selected_row(best)
        best["selected_rank"] = 1
        best["threshold"] = loaded.threshold
        best["policy_index"] = loaded.policy_index
        best["policy_name"] = loaded.policy_name
        best["cooldown_minutes"] = loaded.cooldown_minutes
        best["target_mode"] = loaded.target_mode
        best["entry_filter"] = entry_filter
        best["selection_mode"] = selection_mode
        best["min_score_margin"] = min_score_margin
        best["max_score_ceiling"] = max_score_ceiling
        best["max_trades_per_session"] = max_trades_per_session
        best["max_daily_loss"] = max_daily_loss
        best["score_margin"] = score_margin
        best["model_path"] = str(loaded.model_path)
        best["model_family"] = str(getattr(loaded, "model_family", "mlp") or "mlp")
        best["jitter_scenario"] = scenario.name
        selected.append(best)
        trades_by_session[session] = trades_by_session.get(session, 0) + 1
        realized_pnl_by_session[session] = realized_pnl_by_session.get(session, 0.0) + float(
            best.get("label_net_pnl") or 0.0
        )
        cash += float(best.get("label_net_pnl") or 0.0) - 20.0
        next_time_by_session[session] = decision_time + timedelta(minutes=loaded.cooldown_minutes)

    return selected, {
        "split": split,
        "jitter_scenario": scenario.name,
        "decision_rows": decision_rows,
        "candidate_rows": candidate_rows,
        "selected_entries": len(selected),
        "threshold_waits": threshold_waits,
        "cooldown_blocks": cooldown_blocks,
        "session_trade_cap_blocks": session_trade_cap_blocks,
        "daily_loss_stop_blocks": daily_loss_stop_blocks,
        "entry_filter": entry_filter,
        "selection_mode": selection_mode,
        "entry_filter_blocks": entry_filter_blocks,
        "affordability_blocks": affordability_blocks,
        "score_margin_waits": score_margin_waits,
        "score_ceiling_waits": score_ceiling_waits,
    }


def action_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("split")), str(row.get("session")), str(row.get("decision_time")))


def compare_selected_actions(
    baseline_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    *,
    total_decisions: int,
) -> dict[str, Any]:
    baseline = {action_key(row): str(row.get("contract_id")) for row in baseline_rows}
    scenario = {action_key(row): str(row.get("contract_id")) for row in scenario_rows}
    all_keys = set(baseline) | set(scenario)
    presence_mismatches = [key for key in all_keys if (key in baseline) != (key in scenario)]
    contract_mismatches = [
        key for key in all_keys
        if key in baseline and key in scenario and baseline[key] != scenario[key]
    ]
    action_presence_matches = int(total_decisions) - len(presence_mismatches)
    selected_contract_matches = [
        key for key in all_keys
        if key in baseline and key in scenario and baseline[key] == scenario[key]
    ]
    return {
        "baseline_entries": len(baseline),
        "scenario_entries": len(scenario),
        "total_decisions": int(total_decisions),
        "action_presence_mismatches": len(presence_mismatches),
        "selected_contract_mismatches": len(contract_mismatches),
        "action_presence_match_rate": (
            action_presence_matches / float(total_decisions)
            if total_decisions > 0
            else 0.0
        ),
        "selected_contract_match_rate": (
            len(selected_contract_matches) / float(len(baseline))
            if baseline
            else 1.0 if not scenario else 0.0
        ),
        "mismatch_examples": [
            {
                "split": key[0],
                "session": key[1],
                "decision_time": key[2],
                "baseline_contract": baseline.get(key),
                "scenario_contract": scenario.get(key),
            }
            for key in sorted(presence_mismatches + contract_mismatches)[:20]
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol101 Fair-Contract Feature Jitter Gate",
        "",
        "## Decision",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Model training executed here: `false`",
        f"- Threshold tuning executed here: `false`",
        f"- Broker endpoint called: `false`",
        f"- Paper-submit allowed: `false`",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | Split | Entries | PnL | PF | Presence match | Contract match | Result |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("scenario_rows") or []:
        metrics = row.get("metrics") or {}
        comparison = row.get("comparison") or {}
        lines.append(
            f"| `{row['scenario']}` | `{row['split']}` | "
            f"{int(metrics.get('trades') or 0)} | {float(metrics.get('total_pnl') or 0.0):.2f} | "
            f"{float(metrics.get('profit_factor') or 0.0):.3f} | "
            f"{float(comparison.get('action_presence_match_rate') or 0.0):.3f} | "
            f"{float(comparison.get('selected_contract_match_rate') or 0.0):.3f} | "
            f"`{row['status']}` |"
        )
    if payload.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{item}`" for item in payload["blockers"])
    lines.extend(["", "## Interpretation", ""])
    lines.append(str(payload.get("interpretation") or ""))
    lines.extend(["", "## Outputs", ""])
    for key, value in (payload.get("outputs") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner_plan = load_json_optional(args.runner_plan)
    training_result = load_json_optional(args.training_result)
    if not runner_plan or not training_result:
        raise SystemExit("runner plan and training result are required")
    loaded = load_model(training_result)
    split_files = runner_plan.get("split_files") or {}
    selected_by_scenario_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
    split_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario in SCENARIOS:
        for split in args.splits:
            paths = [Path(path) for path in split_files.get(split, [])]
            selected, summary = select_records_for_split_with_scenario(
                split=split,
                paths=paths,
                loaded=loaded,
                scenario=scenario,
            )
            selected_by_scenario_split[(scenario.name, split)] = selected
            split_summaries[(scenario.name, split)] = summary

    scenario_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    all_selected_rows: list[dict[str, Any]] = []
    for (scenario_name, split), selected in sorted(selected_by_scenario_split.items()):
        all_selected_rows.extend(selected)
        trades, _state = replay_selected_candidates(
            selected,
            starting_cash=float(args.starting_cash),
            contract_multiplier=STRICT_REPLAY_CONTRACT_MULTIPLIER,
            stress_per_side=float(args.stress_per_side),
        )
        metrics = metrics_for_trades(trades, starting_cash=float(args.starting_cash))
        baseline_selected = selected_by_scenario_split.get(("baseline", split), [])
        total_decisions = int(split_summaries[(scenario_name, split)].get("decision_rows") or 0)
        comparison = compare_selected_actions(
            baseline_selected,
            selected,
            total_decisions=total_decisions,
        )
        baseline_metrics = None
        if scenario_name != "baseline":
            baseline_trades, _ = replay_selected_candidates(
                baseline_selected,
                starting_cash=float(args.starting_cash),
                contract_multiplier=STRICT_REPLAY_CONTRACT_MULTIPLIER,
                stress_per_side=float(args.stress_per_side),
            )
            baseline_metrics = metrics_for_trades(
                baseline_trades,
                starting_cash=float(args.starting_cash),
            )
        pnl_drift_fraction = 0.0
        if baseline_metrics is not None:
            baseline_pnl = abs(float(baseline_metrics.get("total_pnl") or 0.0))
            pnl_drift_fraction = (
                abs(float(metrics.get("total_pnl") or 0.0) - float(baseline_metrics.get("total_pnl") or 0.0))
                / baseline_pnl
                if baseline_pnl > 1e-9
                else 0.0
            )
        row_status = "pass"
        if scenario_name != "baseline":
            if comparison["action_presence_match_rate"] < float(args.min_action_presence_match_rate):
                row_status = "fail"
                blockers.append(f"{scenario_name}:{split}:action_presence_match_rate")
            if comparison["selected_contract_match_rate"] < float(args.min_selected_contract_match_rate):
                row_status = "fail"
                blockers.append(f"{scenario_name}:{split}:selected_contract_match_rate")
            if pnl_drift_fraction > float(args.max_pnl_drift_fraction):
                row_status = "fail"
                blockers.append(f"{scenario_name}:{split}:pnl_drift_fraction")
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "split": split,
                "status": row_status,
                "metrics": metrics,
                "comparison": comparison,
                "pnl_drift_fraction": pnl_drift_fraction,
                "split_summary": split_summaries[(scenario_name, split)],
            }
        )

    selected_csv = args.out_dir / "jitter_selected_candidates.csv"
    rows_csv = args.out_dir / "scenario_rows.csv"
    write_csv(selected_csv, all_selected_rows)
    flat_rows = []
    for row in scenario_rows:
        metrics = row["metrics"]
        comparison = row["comparison"]
        flat_rows.append(
            {
                "scenario": row["scenario"],
                "split": row["split"],
                "status": row["status"],
                "trades": metrics.get("trades"),
                "total_pnl": metrics.get("total_pnl"),
                "profit_factor": metrics.get("profit_factor"),
                "action_presence_match_rate": comparison.get("action_presence_match_rate"),
                "selected_contract_match_rate": comparison.get("selected_contract_match_rate"),
                "action_presence_mismatches": comparison.get("action_presence_mismatches"),
                "selected_contract_mismatches": comparison.get("selected_contract_mismatches"),
                "pnl_drift_fraction": row.get("pnl_drift_fraction"),
            }
        )
    write_csv(rows_csv, flat_rows)
    status = "pass" if not blockers else "fail"
    payload = {
        "schema_version": FEATURE_JITTER_GATE_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": status,
        "decision": (
            "candidate_passes_option_feature_jitter_gate"
            if status == "pass"
            else "candidate_fails_option_feature_jitter_gate"
        ),
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
        "broker_endpoint_called": False,
        "paper_submit_allowed": False,
        "scenarios": [scenario.__dict__ for scenario in SCENARIOS],
        "requirements": {
            "min_action_presence_match_rate": float(args.min_action_presence_match_rate),
            "min_selected_contract_match_rate": float(args.min_selected_contract_match_rate),
            "max_pnl_drift_fraction": float(args.max_pnl_drift_fraction),
        },
        "scenario_rows": scenario_rows,
        "blockers": sorted(set(blockers)),
        "interpretation": (
            "A fail means the candidate's selected trades or PnL are too dependent on small, live-plausible "
            "option IV/spread/size perturbations. That is negative model-selection evidence, not a feed adapter bug."
        ),
        "outputs": {
            "jitter_selected_candidates_csv": str(selected_csv),
            "scenario_rows_csv": str(rows_csv),
            "report": str(args.out_dir / "report.md"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "report.md").write_text(render_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
