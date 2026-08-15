"""Parity audit for Protocol101 feature recovery group 2.

Group 2 covers candidate geometry and moneyness features. The audit is
offline-only: it reads existing paired certified v2 recorder traces, derives
the frozen geometry features, and applies the preregistered parity gates before
any uplift training is allowed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from v4.scripts import run_protocol101_feature_recovery_group1_parity_audit as g1


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_PLAN_DIR = BASE_AUDIT / "protocol101_live_v2_feature_recovery_group2_candidate_geometry_plan"
DEFAULT_PREREGISTRATION = DEFAULT_PLAN_DIR / "preregistration.json"
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_live_v2_feature_recovery_group2_candidate_geometry_parity"
SCHEMA_VERSION = "Protocol101FeatureRecoveryGroup2ParityAuditV1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-prefix", default="protocol101_live_v2_microstructure_masked_greekgate")
    return parser.parse_args()


def candidate_offset(item: dict[str, Any], vector: list[Any] | None = None) -> tuple[float, bool]:
    value = g1.finite(item.get("offset"))
    if value is not None:
        return value, False
    if vector is not None:
        value = g1.value(vector, "option.distance_points")
        if value is not None:
            return value, False
    return 0.0, True


def geometry_context(
    row: dict[str, Any],
    vectors: dict[str, list[Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, float]]:
    candidates = g1.candidate_map(row)
    by_right: dict[str, list[str]] = {}
    offsets: dict[str, float] = {}
    for contract_id, item in candidates.items():
        right = str(item.get("right") or ("C" if contract_id.endswith("-C") else "P" if contract_id.endswith("-P") else ""))
        by_right.setdefault(right, []).append(contract_id)
        offset, _missing = candidate_offset(item, vectors.get(contract_id))
        offsets[contract_id] = offset
    for right, ids in by_right.items():
        ids.sort(key=lambda cid: (abs(offsets.get(cid, 0.0)), cid))
    return candidates, by_right, offsets


def derive_geometry_features(
    *,
    row: dict[str, Any],
    contract_id: str,
    vector: list[Any],
    definition: dict[str, Any],
) -> tuple[dict[str, float], dict[str, bool]]:
    candidates, by_right, offsets = geometry_context(row, g1.token_vectors(row))
    item = candidates.get(contract_id) or {}
    guards = definition["deterministic_guards"]
    min_denominator = float(guards["minimum_denominator_abs"])
    right = str(item.get("right") or ("C" if contract_id.endswith("-C") else "P" if contract_id.endswith("-P") else ""))
    offset, offset_missing = candidate_offset(item, vector)
    spx, spx_missing = g1.impute(g1.value(vector, "market_last.spx_close"))
    same_right_ids = by_right.get(right, [])
    rank = same_right_ids.index(contract_id) if contract_id in same_right_ids else 0
    same_right_count = len(same_right_ids)
    candidate_count = len(candidates)
    neighbor_count = sum(1 for cid in same_right_ids if abs(offsets.get(cid, 0.0) - offset) <= 10.0)
    abs_offset = abs(offset)
    recovered = {
        "right_is_call": float(right == "C"),
        "right_is_put": float(right == "P"),
        "offset_points": offset,
        "abs_offset_points": abs_offset,
        "offset_bps_underlying": g1.safe_div(offset, spx, min_denominator) * 10_000.0,
        "abs_offset_bps_underlying": g1.safe_div(abs_offset, spx, min_denominator) * 10_000.0,
        "offset_sign": -1.0 if offset < 0 else 1.0 if offset > 0 else 0.0,
        "out_of_the_money_flag": float((right == "C" and offset > 0.0) or (right == "P" and offset < 0.0)),
        "in_the_money_flag": float((right == "C" and offset < 0.0) or (right == "P" and offset > 0.0)),
        "same_right_rank_by_abs_offset": float(rank),
        "same_right_candidate_count": float(same_right_count),
        "candidate_count_total": float(candidate_count),
        "same_right_abs_offset_percentile": float(rank) / max(float(same_right_count - 1), 1.0),
        "local_same_right_neighbor_count_10_points": float(neighbor_count),
        "abs_offset_bucket_0_10": float(abs_offset <= 10.0),
        "abs_offset_bucket_10_25": float(10.0 < abs_offset <= 25.0),
        "abs_offset_bucket_25_50": float(25.0 < abs_offset <= 50.0),
        "abs_offset_bucket_gt_50": float(abs_offset > 50.0),
    }
    missing = {
        name: offset_missing or (spx_missing if "bps_underlying" in name else False)
        for name in definition["candidate_recovered_model_facing_features"]
    }
    return recovered, missing


def run_audit(preregistration_path: Path, out_dir: Path, trace_prefix: str) -> dict[str, Any]:
    prereg = g1.load_json(preregistration_path)
    definition_path = Path(prereg["feature_group_definition_path"])
    definition = g1.load_json(definition_path)
    blockers = g1.validate_machine_checkable(prereg, definition)
    parity = prereg["parity_evidence_required_before_uplift_claim"]
    drift_gates = parity["feature_drift_gates"]
    first_et = str(parity["required_first_decision_et"])
    last_et = str(parity["required_last_decision_et"])
    expected_rows = int(parity["paired_rows_per_full_day"])
    continuous_features = list(definition["continuous_recovered_features"])
    flag_features = list(definition["sign_or_bucket_recovered_features"])

    deltas: dict[str, list[float]] = {name: [] for name in continuous_features}
    historical_reference: dict[str, list[float]] = {name: [] for name in continuous_features}
    missing_counts: dict[str, int] = {name: 0 for name in definition["candidate_recovered_model_facing_features"]}
    daily: dict[str, Any] = {}
    minute_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    total_candidate_pairs = 0
    membership_mismatches = 0
    sign_matches = 0
    sign_total = 0
    sign_non_threshold_mismatches = 0
    action_flips = 0
    action_non_threshold_flips = 0
    selected_contract_changes = 0

    for session in parity["required_recorder_days"]:
        live_path, historical_path = g1.trace_paths(trace_prefix, session)
        if not live_path.exists():
            blockers.append(f"missing_live_trace:{session}:{live_path}")
            continue
        if not historical_path.exists():
            blockers.append(f"missing_historical_trace:{session}:{historical_path}")
            continue
        live_rows = g1.load_trace_rows(live_path)
        historical_rows = g1.load_trace_rows(historical_path)
        keys = [ts for ts in sorted(set(live_rows) & set(historical_rows)) if g1.in_window(ts, first_et, last_et)]
        day_blockers: list[str] = []
        if len(keys) != expected_rows:
            day_blockers.append(f"paired_row_count:{len(keys)}")
        day_membership_mismatches = 0
        day_action_flips = 0
        day_candidate_pairs = 0
        for ts in keys:
            live_row = live_rows[ts]
            historical_row = historical_rows[ts]
            expected_context = ts - timedelta(minutes=1)
            if g1.parse_ts(live_row.get("source_context_ts")) != expected_context:
                day_blockers.append(f"live_context_lag_not_one_minute:{session}:{ts.isoformat()}")
            if g1.parse_ts(historical_row.get("source_context_ts")) != expected_context:
                day_blockers.append(f"historical_context_lag_not_one_minute:{session}:{ts.isoformat()}")
            if live_row.get("feature_contract_version") != prereg["base_contract"]:
                day_blockers.append(f"live_contract_mismatch:{session}:{ts.isoformat()}")
            if historical_row.get("feature_contract_version") != prereg["base_contract"]:
                day_blockers.append(f"historical_contract_mismatch:{session}:{ts.isoformat()}")
            membership_match = g1.membership_fingerprint(live_row) == g1.membership_fingerprint(historical_row)
            if not membership_match:
                membership_mismatches += 1
                day_membership_mismatches += 1
            live_action = str(live_row.get("selected_action"))
            historical_action = str(historical_row.get("selected_action"))
            if live_action != historical_action:
                action_flips += 1
                day_action_flips += 1
                if not g1.action_threshold_adjacent(live_row, historical_row):
                    action_non_threshold_flips += 1
            live_selected = live_row.get("selected_contract_id")
            historical_selected = historical_row.get("selected_contract_id")
            if live_selected != historical_selected and (live_selected is not None or historical_selected is not None):
                selected_contract_changes += 1
            live_vectors = g1.token_vectors(live_row)
            historical_vectors = g1.token_vectors(historical_row)
            common_contracts = sorted(set(live_vectors) & set(historical_vectors))
            total_candidate_pairs += len(common_contracts)
            day_candidate_pairs += len(common_contracts)
            minute_rows.append(
                {
                    "session": session,
                    "decision_time_utc": ts.isoformat(),
                    "decision_time_et": g1.local_minute(ts),
                    "live_candidate_count": len(live_vectors),
                    "historical_candidate_count": len(historical_vectors),
                    "common_candidate_count": len(common_contracts),
                    "candidate_membership_match": membership_match,
                    "action_match": live_action == historical_action,
                }
            )
            for contract_id in common_contracts:
                live_recovered, live_missing = derive_geometry_features(
                    row=live_row,
                    contract_id=contract_id,
                    vector=live_vectors[contract_id],
                    definition=definition,
                )
                historical_recovered, historical_missing = derive_geometry_features(
                    row=historical_row,
                    contract_id=contract_id,
                    vector=historical_vectors[contract_id],
                    definition=definition,
                )
                for name in continuous_features:
                    deltas[name].append(live_recovered[name] - historical_recovered[name])
                    historical_reference[name].append(historical_recovered[name])
                for name in definition["candidate_recovered_model_facing_features"]:
                    if live_missing[name] or historical_missing[name]:
                        missing_counts[name] += 1
                for name in flag_features:
                    sign_total += 1
                    if live_recovered[name] == historical_recovered[name]:
                        sign_matches += 1
                    else:
                        sign_non_threshold_mismatches += 1
                        mismatch_rows.append(
                            {
                                "session": session,
                                "decision_time_utc": ts.isoformat(),
                                "decision_time_et": g1.local_minute(ts),
                                "contract_id": contract_id,
                                "feature": name,
                                "live_value": live_recovered[name],
                                "historical_value": historical_recovered[name],
                            }
                        )
        blockers.extend(day_blockers)
        daily[session] = {
            "live_trace": str(live_path),
            "historical_trace": str(historical_path),
            "paired_rows_in_required_window": len(keys),
            "first_decision_et": g1.local_minute(keys[0]) if keys else None,
            "last_decision_et": g1.local_minute(keys[-1]) if keys else None,
            "action_flips": day_action_flips,
            "candidate_membership_mismatches": day_membership_mismatches,
            "candidate_pairs": day_candidate_pairs,
            "blockers": sorted(set(day_blockers)),
        }

    scale_floor = float(definition["feature_drift_standardization_policy"]["scale_floor"])
    feature_rows: list[dict[str, Any]] = []
    for name in continuous_features:
        ref = historical_reference[name]
        scale = max(stdev(ref) if len(ref) > 1 else 0.0, scale_floor)
        standardized = [delta / scale for delta in deltas[name]]
        raw = g1.summary(deltas[name])
        scaled = g1.summary(standardized)
        finite_coverage = 1.0 - (missing_counts[name] / max(total_candidate_pairs, 1))
        passed = (
            finite_coverage >= float(drift_gates["finite_coverage_min"])
            and (scaled.get("median_abs") or 0.0)
            <= float(drift_gates["continuous_feature_standardized_abs_drift_median_max"])
            and (scaled.get("p95_abs") or 0.0)
            <= float(drift_gates["continuous_feature_standardized_abs_drift_p95_max"])
            and (scaled.get("p99_abs") or 0.0)
            <= float(drift_gates["continuous_feature_standardized_abs_drift_p99_max"])
        )
        feature_rows.append(
            {
                "feature": name,
                "family": "continuous",
                "standardization_scale": scale,
                "raw_median_abs": raw.get("median_abs"),
                "raw_p95_abs": raw.get("p95_abs"),
                "raw_p99_abs": raw.get("p99_abs"),
                "raw_max_abs": raw.get("max_abs"),
                "standardized_median_abs": scaled.get("median_abs"),
                "standardized_p95_abs": scaled.get("p95_abs"),
                "standardized_p99_abs": scaled.get("p99_abs"),
                "standardized_max_abs": scaled.get("max_abs"),
                "raw_missing_count": missing_counts[name],
                "final_finite_coverage_after_imputation": finite_coverage,
                "pass": passed,
            }
        )

    sign_match_rate = sign_matches / sign_total if sign_total else 0.0
    continuous_gate_pass = all(bool(row["pass"]) for row in feature_rows)
    sign_gate_pass = (
        sign_match_rate >= float(drift_gates["sign_or_bucket_match_rate_min"])
        and sign_non_threshold_mismatches <= int(drift_gates["non_threshold_sign_or_bucket_mismatch_count_max"])
    )
    timestamp_gate_pass = not blockers
    membership_gate_pass = membership_mismatches == 0
    action_gate_pass = action_non_threshold_flips <= int(
        parity["candidate_replay_action_gates"]["unclassified_non_threshold_action_flips_max"]
    )
    status = "pass" if timestamp_gate_pass and continuous_gate_pass and sign_gate_pass and membership_gate_pass and action_gate_pass else "fail"
    decision = "parity_passed_uplift_training_allowed_with_owner_approval" if status == "pass" else "reject_feature_group_before_cv_uplift_claim"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "decision": decision,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": prereg["experiment_id"],
        "feature_group": prereg["feature_group"],
        "base_contract": prereg["base_contract"],
        "base_model_facing_transform": prereg["base_model_facing_transform"],
        "feature_group_definition_path": str(definition_path),
        "feature_group_definition_sha256": g1.sha256_path(definition_path),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": g1.sha256_path(preregistration_path),
        "trace_prefix": trace_prefix,
        "daily": daily,
        "gate_results": {
            "machine_checkable_preregistration": not bool(blockers and any("hash" in item for item in blockers)),
            "timestamp_and_rows": timestamp_gate_pass,
            "candidate_membership_unchanged": membership_gate_pass,
            "continuous_feature_drift": continuous_gate_pass,
            "sign_or_bucket_match": sign_gate_pass,
            "action_flips": action_gate_pass,
        },
        "blockers": sorted(set(blockers)),
        "continuous_feature_gate": {
            "failed_features": [row["feature"] for row in feature_rows if not row["pass"]],
            "finite_coverage_min": drift_gates["finite_coverage_min"],
        },
        "sign_or_bucket_gate": {
            "match_rate": sign_match_rate,
            "match_rate_min": drift_gates["sign_or_bucket_match_rate_min"],
            "sign_total": sign_total,
            "sign_matches": sign_matches,
            "non_threshold_mismatches": sign_non_threshold_mismatches,
            "non_threshold_mismatch_max": drift_gates["non_threshold_sign_or_bucket_mismatch_count_max"],
        },
        "candidate_membership": {
            "checked_rows": sum(row["paired_rows_in_required_window"] for row in daily.values()),
            "mismatch_count": membership_mismatches,
            "membership_hash_type": "contract_id_right_offset_strike_idx_right_idx",
        },
        "action_replay": {
            "action_flips": action_flips,
            "unclassified_non_threshold_action_flips": action_non_threshold_flips,
            "selected_contract_changes": selected_contract_changes,
        },
        "side_effect_policy": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_download": False,
            "promotion_or_default_changed": False,
            "runtime_flags_edited": False,
            "launchd_changed": False,
            "real_money_path_changed": False,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    g1.write_json(out_dir / "summary.json", payload)
    g1.write_csv(out_dir / "feature_drift_summary.csv", feature_rows)
    g1.write_csv(out_dir / "sign_bucket_mismatches.csv", mismatch_rows)
    g1.write_csv(out_dir / "minute_parity.csv", minute_rows)
    write_report(out_dir / "report.md", payload, feature_rows)
    return payload


def write_report(path: Path, payload: dict[str, Any], feature_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Protocol101 Feature Recovery Group 2 Parity Audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Experiment: `{payload['experiment_id']}`",
        f"- Feature group: `{payload['feature_group']}`",
        f"- Contract: `{payload['base_contract']}`",
        f"- Transform: `{payload['base_model_facing_transform']}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gate_results"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Daily Evidence", ""])
    for session, row in payload["daily"].items():
        lines.append(
            f"- `{session}`: rows=`{row['paired_rows_in_required_window']}` first/last=`{row['first_decision_et']}`/`{row['last_decision_et']}` "
            f"candidate_pairs=`{row['candidate_pairs']}` action_flips=`{row['action_flips']}` membership_mismatches=`{row['candidate_membership_mismatches']}`"
        )
    lines.extend(["", "## Feature Drift", ""])
    for row in sorted(feature_rows, key=lambda item: float(item.get("standardized_p99_abs") or 0.0), reverse=True):
        lines.append(
            f"- `{row['feature']}`: pass=`{row['pass']}` std median/p95/p99=`{row['standardized_median_abs']}`/`{row['standardized_p95_abs']}`/`{row['standardized_p99_abs']}`"
        )
    lines.extend(
        [
            "",
            "## Side Effects",
            "",
        ]
    )
    for key, value in payload["side_effect_policy"].items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    payload = run_audit(args.preregistration, args.out_dir, str(args.trace_prefix))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "blockers": payload["blockers"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
