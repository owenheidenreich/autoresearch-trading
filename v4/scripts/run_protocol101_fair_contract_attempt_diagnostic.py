"""Diagnose a fair-contract candidate against labels and Protocol101 teacher rows.

This is an offline audit tool. It does not train, tune thresholds, contact
brokers, download data, change defaults, or promote a model. The purpose is to
answer whether a near-miss fair-contract candidate is losing because the model
chooses poor candidates, because the fair inputs cannot represent the older
Protocol101 behavior, or because the old teacher behavior is not a causal/live
target worth imitating.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol101_fair_contract_selected_candidate_export import (
    candidate_allowed_by_entry_filter,
    candidate_records_from_row,
    load_model,
    score_candidate_records,
    top_candidate_with_margin,
)


DEFAULT_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_labels_design/summary.json"
)
DEFAULT_MANIFEST = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_labels_preflight/"
    "canonical_processed_session_manifest.json"
)
DEFAULT_BASELINE_EVENTS = Path(
    "v4/audit/autoresearch/unified_protocol101_baseline_attachment/"
    "protocol101_baseline_event_actions_training_scope.parquet"
)
DEFAULT_ATTEMPT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_model_search/attempts/"
    "attempt_056_policy0_hgb_profit_classifier_near_offset_cap2_s42"
)
DEFAULT_OUT_DIR = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_attempt056_diagnostic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-events", type=Path, default=DEFAULT_BASELINE_EVENTS)
    parser.add_argument("--attempt-dir", type=Path, default=DEFAULT_ATTEMPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "diagnostic_test"),
        default=["validation", "diagnostic_test"],
    )
    parser.add_argument("--missed-limit", type=int, default=250)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path} expected list rows, got {type(rows).__name__}")
    return rows


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy_contract(value: Any) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() != "nan" else ""


def manifest_session_paths(manifest: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in manifest.get("included_sessions") or []:
        session = str(item.get("session") or "")
        processed = item.get("processed_file")
        if session and processed:
            paths[session] = Path(str(processed))
    return paths


def split_paths(design: dict[str, Any], manifest: dict[str, Any]) -> dict[str, list[Path]]:
    by_session = manifest_session_paths(manifest)
    policy = design.get("split_policy") or {}
    keys = {
        "train": "train_sessions",
        "validation": "validation_sessions",
        "diagnostic_test": "diagnostic_test_sessions",
    }
    out: dict[str, list[Path]] = {}
    missing: list[str] = []
    for split, key in keys.items():
        paths: list[Path] = []
        for session in policy.get(key) or []:
            path = by_session.get(str(session))
            if path is None:
                missing.append(f"{split}:{session}")
            else:
                paths.append(path)
        out[split] = paths
    if missing:
        raise FileNotFoundError(f"manifest missing split sessions: {missing}")
    return out


def load_baseline_teacher(path: Path, sessions: set[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[
        (df["split"].astype(str) == "q1_2026")
        & (df["session"].astype(str).isin(sessions))
        & (df["protocol101_action"].astype(str) == "enter")
    ].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "session",
                "decision_time",
                "contract_id",
                "teacher_seed_count",
                "teacher_mean_pnl",
                "teacher_median_pnl",
                "teacher_positive_seed_count",
                "teacher_seeds",
            ]
        )
    df["decision_time"] = df["decision_time"].astype(str)
    df["contract_id"] = df["contract_id"].astype(str)
    grouped = (
        df.groupby(["session", "decision_time", "contract_id"], dropna=False)
        .agg(
            teacher_seed_count=("seed", "nunique"),
            teacher_mean_pnl=("baseline_trade_pnl", "mean"),
            teacher_median_pnl=("baseline_trade_pnl", "median"),
            teacher_positive_seed_count=(
                "baseline_trade_pnl",
                lambda values: int((pd.Series(values) > 0).sum()),
            ),
            teacher_min_pnl=("baseline_trade_pnl", "min"),
            teacher_max_pnl=("baseline_trade_pnl", "max"),
            teacher_seeds=("seed", lambda values: ",".join(map(str, sorted(set(values))))),
        )
        .reset_index()
    )
    return grouped


def _candidate_lookup(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_truthy_contract(row.get("contract_id")): row for row in candidates}


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_features"}


def _metrics(pnls: list[float]) -> dict[str, Any]:
    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    equity = 10_000.0
    peak = equity
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return {
        "trades": len(pnls),
        "total_pnl": float(sum(pnls)),
        "avg_pnl": float(sum(pnls) / len(pnls)) if pnls else 0.0,
        "win_rate": float(sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else (999.0 if gains > 0 else 0.0),
        "max_drawdown": float(max_drawdown),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_split(
    *,
    split: str,
    paths: list[Path],
    loaded_model: Any,
    teacher: pd.DataFrame,
    missed_limit: int,
) -> dict[str, Any]:
    entry_filter = str(getattr(loaded_model, "entry_filter", "none") or "none")
    threshold = float(loaded_model.threshold)
    min_score_margin = float(getattr(loaded_model, "min_score_margin", 0.0) or 0.0)
    cooldown_minutes = int(getattr(loaded_model, "cooldown_minutes", 0) or 0)
    max_trades_per_session = int(getattr(loaded_model, "max_trades_per_session", 0) or 0)
    max_daily_loss = float(getattr(loaded_model, "max_daily_loss", 0.0) or 0.0)

    teacher_by_time: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    teacher_by_contract: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in teacher.to_dict("records"):
        key_time = (str(row["session"]), str(row["decision_time"]))
        contract = _truthy_contract(row.get("contract_id"))
        teacher_by_time[key_time].append(row)
        teacher_by_contract[(key_time[0], key_time[1], contract)] = row

    decision_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []

    next_time_by_session: dict[str, datetime] = {}
    trades_by_session: dict[str, int] = defaultdict(int)
    realized_pnl_by_session: dict[str, float] = defaultdict(float)

    oracle_next_time_by_session: dict[str, datetime] = {}
    oracle_trades_by_session: dict[str, int] = defaultdict(int)
    oracle_pnls: list[float] = []

    counters: dict[str, int] = defaultdict(int)
    selected_pnls: list[float] = []

    for path in paths:
        session = path.name.removesuffix(".pkl")
        for row in load_rows(path):
            counters["decision_rows"] += 1
            decision_time_value = row.get("decision_time")
            decision_time = (
                decision_time_value
                if isinstance(decision_time_value, datetime)
                else datetime.fromisoformat(str(decision_time_value))
            )
            decision_iso = _iso(decision_time)
            candidates = candidate_records_from_row(
                row,
                session=session,
                split=split,
                policy_index=int(loaded_model.policy_index),
            )
            counters["candidate_rows"] += len(candidates)
            if not candidates:
                continue
            score_candidate_records(candidates, loaded_model)
            eligible = [
                candidate
                for candidate in candidates
                if candidate_allowed_by_entry_filter(candidate, row, entry_filter)
            ]
            counters["eligible_candidate_rows"] += len(eligible)
            lookup = _candidate_lookup(candidates)
            eligible_lookup = _candidate_lookup(eligible)

            top = top_candidate_with_margin(eligible) if eligible else None
            top_candidate = top[0] if top else None
            score_margin = float(top[1]) if top else None
            best_label_candidate = max(
                eligible,
                key=lambda item: float(item.get("label_net_pnl") or -1e18),
                default=None,
            )

            teacher_entries = teacher_by_time.get((session, decision_iso), [])
            teacher_exact_contracts = [
                _truthy_contract(item.get("contract_id"))
                for item in teacher_entries
                if _truthy_contract(item.get("contract_id")) in lookup
            ]
            teacher_eligible_contracts = [
                contract for contract in teacher_exact_contracts if contract in eligible_lookup
            ]

            state_block = ""
            selected = False
            if next_time_by_session.get(session) and decision_time < next_time_by_session[session]:
                state_block = "cooldown"
            elif max_trades_per_session > 0 and trades_by_session[session] >= max_trades_per_session:
                state_block = "session_trade_cap"
            elif max_daily_loss > 0.0 and realized_pnl_by_session[session] <= -max_daily_loss:
                state_block = "daily_loss_stop"
            elif not eligible:
                state_block = "entry_filter"
            elif top_candidate is None:
                state_block = "no_score"
            elif min_score_margin > 0.0 and (score_margin is None or score_margin < min_score_margin):
                state_block = "score_margin"
            elif float(top_candidate.get("score") or -1e18) < threshold:
                state_block = "threshold"
            else:
                selected = True

            if selected and top_candidate is not None:
                selected_row = _flatten_for_csv(top_candidate)
                selected_row.update(
                    {
                        "threshold": threshold,
                        "score_margin": score_margin,
                        "selection_state": "selected",
                        "teacher_same_time_count": len(teacher_entries),
                        "teacher_exact_same_contract": int(
                            _truthy_contract(top_candidate.get("contract_id"))
                            in {
                                _truthy_contract(item.get("contract_id"))
                                for item in teacher_entries
                            }
                        ),
                        "teacher_exact_same_contract_seed_count": int(
                            teacher_by_contract.get(
                                (
                                    session,
                                    decision_iso,
                                    _truthy_contract(top_candidate.get("contract_id")),
                                ),
                                {},
                            ).get("teacher_seed_count", 0)
                            or 0
                        ),
                    }
                )
                selected_rows.append(selected_row)
                pnl = float(top_candidate.get("label_net_pnl") or 0.0)
                selected_pnls.append(pnl)
                trades_by_session[session] += 1
                realized_pnl_by_session[session] += pnl
                next_time_by_session[session] = decision_time + timedelta(minutes=cooldown_minutes)

            oracle_state_block = ""
            oracle_selected = False
            if (
                oracle_next_time_by_session.get(session)
                and decision_time < oracle_next_time_by_session[session]
            ):
                oracle_state_block = "cooldown"
            elif max_trades_per_session > 0 and oracle_trades_by_session[session] >= max_trades_per_session:
                oracle_state_block = "session_trade_cap"
            elif best_label_candidate is None:
                oracle_state_block = "no_eligible_candidate"
            elif float(best_label_candidate.get("label_net_pnl") or 0.0) <= 0.0:
                oracle_state_block = "nonpositive_best_label"
            else:
                oracle_selected = True
                oracle_pnl = float(best_label_candidate.get("label_net_pnl") or 0.0)
                oracle_pnls.append(oracle_pnl)
                oracle_trades_by_session[session] += 1
                oracle_next_time_by_session[session] = decision_time + timedelta(
                    minutes=cooldown_minutes
                )

            if best_label_candidate is not None:
                best_label = float(best_label_candidate.get("label_net_pnl") or 0.0)
                selected_label = (
                    float(top_candidate.get("label_net_pnl") or 0.0)
                    if selected and top_candidate is not None
                    else None
                )
                if best_label > 0 and (selected_label is None or best_label > selected_label):
                    missed = _flatten_for_csv(best_label_candidate)
                    missed.update(
                        {
                            "model_selected": int(selected),
                            "selected_contract_id": _truthy_contract(
                                top_candidate.get("contract_id") if top_candidate else ""
                            ),
                            "selected_label_net_pnl": selected_label,
                            "selected_score": (
                                _safe_float(top_candidate.get("score")) if top_candidate else None
                            ),
                            "missed_label_advantage": (
                                best_label - selected_label if selected_label is not None else best_label
                            ),
                            "selection_state_block": state_block,
                            "teacher_same_time_count": len(teacher_entries),
                        }
                    )
                    missed_rows.append(missed)

            for item in teacher_entries:
                contract = _truthy_contract(item.get("contract_id"))
                candidate = lookup.get(contract)
                eligible_candidate = eligible_lookup.get(contract)
                out = {
                    "split": split,
                    "session": session,
                    "decision_time": decision_iso,
                    "teacher_contract_id": contract,
                    "teacher_seed_count": int(item.get("teacher_seed_count") or 0),
                    "teacher_positive_seed_count": int(
                        item.get("teacher_positive_seed_count") or 0
                    ),
                    "teacher_mean_pnl": _safe_float(item.get("teacher_mean_pnl")),
                    "teacher_median_pnl": _safe_float(item.get("teacher_median_pnl")),
                    "teacher_min_pnl": _safe_float(item.get("teacher_min_pnl")),
                    "teacher_max_pnl": _safe_float(item.get("teacher_max_pnl")),
                    "teacher_seeds": item.get("teacher_seeds"),
                    "present_in_fair_universe": int(candidate is not None),
                    "eligible_under_attempt_filter": int(eligible_candidate is not None),
                    "selected_by_attempt": int(
                        selected
                        and top_candidate is not None
                        and _truthy_contract(top_candidate.get("contract_id")) == contract
                    ),
                    "selected_same_timestamp_other_contract": int(
                        selected
                        and top_candidate is not None
                        and _truthy_contract(top_candidate.get("contract_id")) != contract
                    ),
                }
                if candidate is not None:
                    out.update(
                        {
                            "fair_label_net_pnl": _safe_float(candidate.get("label_net_pnl")),
                            "fair_score": _safe_float(candidate.get("score")),
                            "fair_right": candidate.get("right"),
                            "fair_offset": _safe_float(candidate.get("offset")),
                            "fair_entry_ask": _safe_float(candidate.get("entry_ask")),
                        }
                    )
                teacher_rows.append(out)

            decision_rows.append(
                {
                    "split": split,
                    "session": session,
                    "decision_time": decision_iso,
                    "candidate_count": len(candidates),
                    "eligible_candidate_count": len(eligible),
                    "state_block": state_block,
                    "model_selected": int(selected),
                    "model_top_contract_id": _truthy_contract(
                        top_candidate.get("contract_id") if top_candidate else ""
                    ),
                    "model_top_right": top_candidate.get("right") if top_candidate else "",
                    "model_top_offset": _safe_float(top_candidate.get("offset"))
                    if top_candidate
                    else None,
                    "model_top_score": _safe_float(top_candidate.get("score"))
                    if top_candidate
                    else None,
                    "model_top_label": _safe_float(top_candidate.get("label_net_pnl"))
                    if top_candidate
                    else None,
                    "model_score_margin": score_margin,
                    "best_label_contract_id": _truthy_contract(
                        best_label_candidate.get("contract_id") if best_label_candidate else ""
                    ),
                    "best_label_right": best_label_candidate.get("right")
                    if best_label_candidate
                    else "",
                    "best_label_offset": _safe_float(best_label_candidate.get("offset"))
                    if best_label_candidate
                    else None,
                    "best_label_net_pnl": _safe_float(best_label_candidate.get("label_net_pnl"))
                    if best_label_candidate
                    else None,
                    "best_label_score": _safe_float(best_label_candidate.get("score"))
                    if best_label_candidate
                    else None,
                    "oracle_selected": int(oracle_selected),
                    "oracle_state_block": oracle_state_block,
                    "teacher_same_time_count": len(teacher_entries),
                    "teacher_exact_contracts_present": len(teacher_exact_contracts),
                    "teacher_eligible_contracts": len(teacher_eligible_contracts),
                }
            )

    missed_rows.sort(
        key=lambda row: (
            float(row.get("missed_label_advantage") or 0.0),
            float(row.get("label_net_pnl") or 0.0),
        ),
        reverse=True,
    )
    missed_rows = missed_rows[: max(int(missed_limit), 0)]
    teacher_present = [row for row in teacher_rows if row["present_in_fair_universe"]]
    teacher_eligible = [row for row in teacher_rows if row["eligible_under_attempt_filter"]]
    teacher_selected = [row for row in teacher_rows if row["selected_by_attempt"]]
    teacher_positive_fair = [
        row
        for row in teacher_present
        if (value := _safe_float(row.get("fair_label_net_pnl"))) is not None and value > 0
    ]
    summary = {
        "split": split,
        "decision_rows": int(counters["decision_rows"]),
        "candidate_rows": int(counters["candidate_rows"]),
        "eligible_candidate_rows": int(counters["eligible_candidate_rows"]),
        "attempt_selected_metrics_raw_labels": _metrics(selected_pnls),
        "oracle_positive_best_eligible_same_state_metrics_raw_labels": _metrics(oracle_pnls),
        "model_selected_rows": len(selected_rows),
        "teacher_enter_rows": len(teacher_rows),
        "teacher_present_in_fair_universe": len(teacher_present),
        "teacher_eligible_under_attempt_filter": len(teacher_eligible),
        "teacher_selected_by_attempt": len(teacher_selected),
        "teacher_positive_fair_label_rows": len(teacher_positive_fair),
        "teacher_present_rate": (
            len(teacher_present) / len(teacher_rows) if teacher_rows else 0.0
        ),
        "teacher_eligible_rate_when_present": (
            len(teacher_eligible) / len(teacher_present) if teacher_present else 0.0
        ),
        "teacher_selected_rate_when_present": (
            len(teacher_selected) / len(teacher_present) if teacher_present else 0.0
        ),
        "teacher_positive_fair_label_rate_when_present": (
            len(teacher_positive_fair) / len(teacher_present) if teacher_present else 0.0
        ),
        "state_blocks": dict(
            pd.Series([row["state_block"] for row in decision_rows]).value_counts().to_dict()
        )
        if decision_rows
        else {},
        "missed_profitable_examples_written": len(missed_rows),
    }
    return {
        "summary": summary,
        "decision_rows": decision_rows,
        "selected_rows": selected_rows,
        "missed_rows": missed_rows,
        "teacher_rows": teacher_rows,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protocol101 Fair-Contract Attempt Diagnostic",
        "",
        "This packet is an offline diagnostic only. It does not train, tune thresholds, "
        "touch broker endpoints, download paid data, change defaults, or promote a model.",
        "",
        f"- Attempt: `{payload['attempt_id']}`",
        f"- Feature contract: `{payload['feature_contract']}`",
        f"- Model family: `{payload['model_family']}`",
        f"- Threshold: `{payload['threshold']}`",
        f"- Entry filter: `{payload['entry_filter']}`",
        f"- Diagnostic decision: `{payload['decision']}`",
        "",
        "## Split Summary",
        "",
        "| split | selected trades | selected PnL | selected PF | oracle trades | oracle PnL | teacher present | teacher eligible | teacher selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, summary in payload["splits"].items():
        selected = summary["attempt_selected_metrics_raw_labels"]
        oracle = summary["oracle_positive_best_eligible_same_state_metrics_raw_labels"]
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(selected["trades"]),
                    f"{selected['total_pnl']:.2f}",
                    f"{selected['profit_factor']:.3f}",
                    str(oracle["trades"]),
                    f"{oracle['total_pnl']:.2f}",
                    str(summary["teacher_present_in_fair_universe"]),
                    str(summary["teacher_eligible_under_attempt_filter"]),
                    str(summary["teacher_selected_by_attempt"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `oracle_positive_best_eligible_same_state` is a diagnostic ceiling using future labels. It is not a deployable model.",
            "- Teacher rows are older Protocol101 event actions. They are labels/diagnostics only, never runtime inputs.",
            "- Low teacher eligibility under the attempt filter means the current hand filter may exclude behavior the old system relied on.",
            "- Low teacher positive fair-label rate means imitating that teacher directly may not recover fair-contract performance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    design = load_json(args.design)
    manifest = load_json(args.manifest)
    paths_by_split = split_paths(design, manifest)
    training_result_path = args.attempt_dir / "training_runner" / "training_result.json"
    training_result = load_json(training_result_path)
    loaded = load_model(training_result)

    sessions = {
        path.name.removesuffix(".pkl")
        for split in args.splits
        for path in paths_by_split.get(split, [])
    }
    teacher = load_baseline_teacher(args.baseline_events, sessions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_payloads: dict[str, Any] = {}
    for split in args.splits:
        result = analyze_split(
            split=split,
            paths=paths_by_split[split],
            loaded_model=loaded,
            teacher=teacher[teacher["session"].astype(str).isin(
                {path.name.removesuffix(".pkl") for path in paths_by_split[split]}
            )],
            missed_limit=int(args.missed_limit),
        )
        split_payloads[split] = result["summary"]
        write_csv(args.out_dir / f"{split}_decision_diagnostic.csv", result["decision_rows"])
        write_csv(args.out_dir / f"{split}_model_selected_actions.csv", result["selected_rows"])
        write_csv(args.out_dir / f"{split}_missed_profitable_candidates.csv", result["missed_rows"])
        write_csv(args.out_dir / f"{split}_teacher_alignment.csv", result["teacher_rows"])

    blockers: list[str] = []
    for split, summary in split_payloads.items():
        selected = summary["attempt_selected_metrics_raw_labels"]
        if split in {"validation", "diagnostic_test"}:
            if selected["trades"] < 20:
                blockers.append(f"{split}_selected_trade_count")
            if selected["profit_factor"] < 1.25:
                blockers.append(f"{split}_selected_profit_factor")
            if selected["total_pnl"] <= 0:
                blockers.append(f"{split}_selected_positive_pnl")

    decision = (
        "candidate_needs_more_model_or_filter_work"
        if blockers
        else "candidate_diagnostic_passes_basic_raw_label_quality"
    )
    payload = {
        "schema_version": "Protocol101FairContractAttemptDiagnosticV1",
        "status": "fail" if blockers else "pass",
        "decision": decision,
        "attempt_id": args.attempt_dir.name,
        "attempt_dir": str(args.attempt_dir),
        "feature_contract": design.get("selected_feature_contract"),
        "model_family": loaded.model_family,
        "target_mode": loaded.target_mode,
        "threshold": loaded.threshold,
        "entry_filter": loaded.entry_filter,
        "min_score_margin": loaded.min_score_margin,
        "max_trades_per_session": loaded.max_trades_per_session,
        "max_daily_loss": loaded.max_daily_loss,
        "paper_submit_allowed": False,
        "broker_endpoint_called": False,
        "model_training_executed_here": False,
        "threshold_tuning_executed_here": False,
        "forbidden_recorder_days_used_for_training_or_tuning": False,
        "blockers": blockers,
        "splits": split_payloads,
        "outputs": {
            "summary_json": str(args.out_dir / "summary.json"),
            "report_md": str(args.out_dir / "report.md"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_report(args.out_dir / "report.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
