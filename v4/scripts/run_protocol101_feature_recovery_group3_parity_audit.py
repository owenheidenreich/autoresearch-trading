"""Parity audit for Protocol101 feature recovery group 3.

Group 3 covers internally computed IV and Greeks. The audit is offline-only:
it reads existing paired certified v2 recorder traces, derives the frozen
repaired Black-Scholes features, and applies the preregistered parity gates
before any uplift training is allowed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from statistics import stdev
from typing import Any

from v4.greeks.repair import compute_repaired_greeks
from v4.scripts import run_protocol101_feature_recovery_group1_parity_audit as g1


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_PLAN_DIR = BASE_AUDIT / "protocol101_live_v2_feature_recovery_group3_internal_greeks_plan"
DEFAULT_PREREGISTRATION = DEFAULT_PLAN_DIR / "preregistration.json"
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_live_v2_feature_recovery_group3_internal_greeks_parity"
SCHEMA_VERSION = "Protocol101FeatureRecoveryGroup3ParityAuditV1"
CONTRACT_RE = re.compile(r"^SPXW-(?P<expiry>\d{8})-(?P<strike>\d+(?:\.\d+)?)-(?P<right>[CP])$")
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-prefix", default="protocol101_live_v2_microstructure_masked_greekgate")
    return parser.parse_args()


def contract_terms(contract_id: str) -> tuple[float | None, str | None]:
    match = CONTRACT_RE.match(str(contract_id))
    if not match:
        return None, None
    return float(match.group("strike")), match.group("right")


def settlement_time_utc(row: dict[str, Any], decision_ts: datetime) -> datetime:
    for item in g1.candidate_map(row).values():
        value = item.get("settlement_time_utc")
        parsed = g1.parse_ts(value)
        if parsed is not None:
            return parsed
    local = decision_ts.astimezone(g1.NY)
    settlement_local = datetime.combine(local.date(), time(16, 0), tzinfo=g1.NY)
    return settlement_local.astimezone(UTC)


def finite_item(item: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = g1.finite(item.get(name))
        if value is not None:
            return value
    return None


def derive_internal_greek_features(
    *,
    row: dict[str, Any],
    contract_id: str,
    vector: list[Any],
    definition: dict[str, Any],
) -> tuple[dict[str, float], dict[str, bool]]:
    item = g1.candidate_map(row).get(contract_id) or {}
    decision_ts = g1.parse_ts(row.get("decision_ts"))
    strike, right = contract_terms(contract_id)
    if right is None:
        right = str(item.get("right") or "")
    spx = g1.value(vector, "market_last.spx_close")
    bid = finite_item(item, "entry_bid", "bid")
    ask = finite_item(item, "entry_ask", "ask")
    mid = finite_item(item, "entry_mid", "mid")
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0

    policy = definition["greek_repair_policy"]
    missing_input = (
        decision_ts is None
        or strike is None
        or right not in {"C", "P"}
        or spx is None
        or not math.isfinite(float(spx))
        or float(spx) <= float(definition["deterministic_guards"]["minimum_underlying_abs"])
    )
    estimate = None
    if not missing_input:
        expiry = settlement_time_utc(row, decision_ts)
        t_years = (expiry - decision_ts).total_seconds() / SECONDS_PER_YEAR
        if t_years > 0.0:
            estimate = compute_repaired_greeks(
                S=float(spx),
                K=float(strike),
                T=t_years,
                is_call=right == "C",
                mid=mid,
                ask=ask,
                bid=bid,
                r=float(policy["risk_free_rate"]),
                q=float(policy["dividend_yield"]),
            )

    if estimate is None:
        recovered = {
            "repaired_iv": 0.0,
            "repaired_delta": 0.0,
            "repaired_abs_delta": 0.0,
            "repaired_gamma": 0.0,
            "repaired_theta_per_day": 0.0,
            "repaired_abs_theta_per_day": 0.0,
            "repaired_vega": 0.0,
            "repaired_price_source_mid": 0.0,
            "repaired_price_source_ask": 0.0,
            "repaired_price_source_bid": 0.0,
            "repaired_success_flag": 0.0,
        }
        missing = {
            name: name in definition["continuous_recovered_features"]
            for name in definition["candidate_recovered_model_facing_features"]
        }
        missing["repaired_success_flag"] = False
        return recovered, missing

    recovered = {
        "repaired_iv": float(estimate.iv),
        "repaired_delta": float(estimate.delta),
        "repaired_abs_delta": abs(float(estimate.delta)),
        "repaired_gamma": float(estimate.gamma),
        "repaired_theta_per_day": float(estimate.theta_per_day),
        "repaired_abs_theta_per_day": abs(float(estimate.theta_per_day)),
        "repaired_vega": float(estimate.vega),
        "repaired_price_source_mid": float(estimate.source == "black_scholes_mid"),
        "repaired_price_source_ask": float(estimate.source == "black_scholes_ask_repair"),
        "repaired_price_source_bid": float(estimate.source == "black_scholes_bid_repair"),
        "repaired_success_flag": 1.0,
    }
    missing = {name: False for name in definition["candidate_recovered_model_facing_features"]}
    return recovered, missing


def run_audit(preregistration_path: Path, out_dir: Path, trace_prefix: str) -> dict[str, Any]:
    prereg = g1.load_json(preregistration_path)
    definition_path = Path(prereg["feature_group_definition_path"])
    definition = g1.load_json(definition_path)
    blockers = g1.validate_machine_checkable(prereg, definition)
    if definition.get("greek_repair_policy", {}).get("vendor_greek_fallback_allowed") is not False:
        blockers.append("vendor_greek_fallback_not_explicitly_forbidden")
    if definition.get("greek_repair_policy", {}).get("raw_quote_direct_alpha_allowed") is not False:
        blockers.append("raw_quote_direct_alpha_not_explicitly_forbidden")

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
        if keys and g1.local_minute(keys[0]) != first_et:
            day_blockers.append(f"first_decision_et:{g1.local_minute(keys[0])}")
        if keys and g1.local_minute(keys[-1]) != last_et:
            day_blockers.append(f"last_decision_et:{g1.local_minute(keys[-1])}")

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
                live_recovered, live_missing = derive_internal_greek_features(
                    row=live_row,
                    contract_id=contract_id,
                    vector=live_vectors[contract_id],
                    definition=definition,
                )
                historical_recovered, historical_missing = derive_internal_greek_features(
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
        "greek_repair_policy": definition["greek_repair_policy"],
        "daily": daily,
        "gate_results": {
            "machine_checkable_preregistration": not bool(blockers and any("hash" in item for item in blockers)),
            "timestamp_and_rows": timestamp_gate_pass,
            "candidate_membership_unchanged": membership_gate_pass,
            "continuous_feature_drift": continuous_gate_pass,
            "source_flag_match": sign_gate_pass,
            "action_flips": action_gate_pass,
            "vendor_greek_fallback_disallowed": definition["greek_repair_policy"]["vendor_greek_fallback_allowed"] is False,
            "raw_quote_direct_alpha_disallowed": definition["greek_repair_policy"]["raw_quote_direct_alpha_allowed"] is False,
        },
        "blockers": sorted(set(blockers)),
        "continuous_feature_gate": {
            "failed_features": [row["feature"] for row in feature_rows if not row["pass"]],
            "finite_coverage_min": drift_gates["finite_coverage_min"],
        },
        "source_flag_gate": {
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
    g1.write_csv(out_dir / "source_flag_mismatches.csv", mismatch_rows)
    g1.write_csv(out_dir / "minute_parity.csv", minute_rows)
    write_report(out_dir / "report.md", payload, feature_rows)
    return payload


def write_report(path: Path, payload: dict[str, Any], feature_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Protocol101 Feature Recovery Group 3 Parity Audit",
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
            f"- `{row['feature']}`: pass=`{row['pass']}` std median/p95/p99=`{row['standardized_median_abs']}`/`{row['standardized_p95_abs']}`/`{row['standardized_p99_abs']}` coverage=`{row['final_finite_coverage_after_imputation']}`"
        )
    lines.extend(["", "## Source Flags", ""])
    gate = payload["source_flag_gate"]
    lines.append(f"- match_rate: `{gate['match_rate']}`")
    lines.append(f"- non_threshold_mismatches: `{gate['non_threshold_mismatches']}`")
    lines.extend(["", "## Side Effects", ""])
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
