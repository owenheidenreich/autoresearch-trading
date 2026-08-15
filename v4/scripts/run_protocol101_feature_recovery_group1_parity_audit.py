"""Parity audit for Protocol101 feature recovery group 1.

This script is offline-only. It reads existing paired IBKR-capture and
historical replay traces for the certified v2 microstructure-masked contract,
derives the preregistered stable index/context feature group, and applies the
frozen parity gates before any CV uplift training is allowed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any
from zoneinfo import ZoneInfo

from v4.dataset.spxw_0dte_neural import MARKET_FEATURE_NAMES, OPTION_FEATURE_NAMES


NY = ZoneInfo("America/New_York")
BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_PLAN_DIR = BASE_AUDIT / "protocol101_live_v2_feature_recovery_group1_stable_index_context_plan"
DEFAULT_PREREGISTRATION = DEFAULT_PLAN_DIR / "preregistration.json"
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_live_v2_feature_recovery_group1_stable_index_context_parity"
SCHEMA_VERSION = "Protocol101FeatureRecoveryGroup1ParityAuditV1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--trace-prefix",
        default="protocol101_live_v2_microstructure_masked_greekgate",
        help="Audit artifact prefix containing historical_replay and ibkr_capture_replay trace directories.",
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def payload(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("payload")
    return nested if row.get("schema_version") == "Protocol101DecisionTraceV1" and isinstance(nested, dict) else row


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=UTC)
    return out.astimezone(UTC)


def local_minute(ts: datetime) -> str:
    return ts.astimezone(NY).strftime("%H:%M")


def in_window(ts: datetime, first_et: str, last_et: str) -> bool:
    minute = local_minute(ts)
    return str(first_et) <= minute <= str(last_et)


def load_trace_rows(path: Path) -> dict[datetime, dict[str, Any]]:
    rows: dict[datetime, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = payload(json.loads(line))
            ts = parse_ts(row.get("decision_ts"))
            if ts is not None:
                rows[ts] = row
    return rows


def candidate_feature_names() -> list[str]:
    names: list[str] = list(OPTION_FEATURE_NAMES)
    for prefix in ("market_last", "market_mean", "market_std", "market_delta"):
        names.extend(f"{prefix}.{name}" for name in MARKET_FEATURE_NAMES)
    names.extend(("side.is_call", "side.is_put", "shape.offset_norm", "shape.abs_offset_norm"))
    names.extend(
        (
            "environment.vwap_gap_over_range",
            "environment.range_pct",
            "environment.above_vwap",
            "environment.below_vwap",
            "environment.omar_pos",
            "environment.omar_neg",
            "environment.mom5_pos",
            "environment.mom5_neg",
            "environment.mom15_pos",
            "environment.mom15_neg",
            "environment.vwap_trend_aligned",
            "environment.vwap_mean_reversion_side",
            "environment.omar_aligned",
            "environment.omar_counter",
            "environment.momentum15_aligned",
            "environment.momentum15_counter",
        )
    )
    names.extend(
        (
            "time.session_progress",
            "time.session_progress_remaining",
            "time.session_progress_sin",
            "time.session_progress_cos",
            "time.first_30",
            "time.post_open_morning",
            "time.midday",
            "time.late_afternoon",
        )
    )
    return names


FEATURE_INDEX = {name: idx for idx, name in enumerate(candidate_feature_names())}


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_div(numerator: float, denominator: float, floor: float) -> float:
    return float(numerator) / max(abs(float(denominator)), float(floor))


def token_vectors(row: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for item in (row.get("features") or {}).get("token_features") or []:
        if not isinstance(item, dict) or not item.get("contract_id"):
            continue
        vector = item.get("features", item.get("token_features"))
        if isinstance(vector, list):
            out[str(item["contract_id"])] = vector
    return out


def candidate_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["contract_id"]): dict(item)
        for item in row.get("candidate_universe") or []
        if isinstance(item, dict) and item.get("contract_id")
    }


def membership_fingerprint(row: dict[str, Any]) -> str:
    members: list[dict[str, Any]] = []
    for contract_id, item in sorted(candidate_map(row).items()):
        members.append(
            {
                "contract_id": contract_id,
                "right": item.get("right"),
                "right_idx": item.get("right_idx"),
                "strike_idx": item.get("strike_idx"),
                "offset": item.get("offset"),
            }
        )
    return hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def value(vector: list[Any], name: str) -> float | None:
    index = FEATURE_INDEX[name]
    return finite(vector[index]) if index < len(vector) else None


def impute(value_or_none: float | None) -> tuple[float, bool]:
    if value_or_none is None or not math.isfinite(float(value_or_none)):
        return 0.0, True
    return float(value_or_none), False


def right_for(row: dict[str, Any], contract_id: str) -> str:
    item = candidate_map(row).get(contract_id) or {}
    right = str(item.get("right") or "")
    if right:
        return right
    return "C" if contract_id.endswith("-C") else "P" if contract_id.endswith("-P") else ""


def vix_by_decision_ts(rows: dict[datetime, dict[str, Any]]) -> dict[datetime, float | None]:
    out: dict[datetime, float | None] = {}
    for ts, row in rows.items():
        vectors = token_vectors(row)
        first = next(iter(vectors.values()), None)
        out[ts] = value(first, "market_last.vix_close") if first is not None else None
    return out


def derive_recovered_features(
    *,
    row: dict[str, Any],
    contract_id: str,
    vector: list[Any],
    vix_lookup: dict[datetime, float | None],
    definition: dict[str, Any],
) -> tuple[dict[str, float], dict[str, bool], dict[str, float]]:
    guards = definition["deterministic_guards"]
    min_denominator = float(guards["minimum_denominator_abs"])
    range_floor = float(guards["session_range_floor_points"])
    vix_floor = float(guards["vix_denominator_floor"])
    omar_min = float(guards["omar_clip_min"])
    omar_max = float(guards["omar_clip_max"])

    spx, spx_missing = impute(value(vector, "market_last.spx_close"))
    vix, vix_missing = impute(value(vector, "market_last.vix_close"))
    vwap, vwap_missing = impute(value(vector, "market_last.spx_vwap"))
    omar, omar_missing = impute(value(vector, "market_last.omar"))
    session_range, range_missing = impute(value(vector, "market_last.session_range"))
    momentum_5m, momentum_5m_missing = impute(value(vector, "market_last.momentum_5m"))
    momentum_15m, momentum_15m_missing = impute(value(vector, "market_last.momentum_15m"))

    decision_ts = parse_ts(row.get("decision_ts"))
    assert decision_ts is not None
    vix_5m_raw = None
    vix_15m_raw = None
    if not vix_missing:
        prior_5m = vix_lookup.get(decision_ts - timedelta(minutes=5))
        prior_15m = vix_lookup.get(decision_ts - timedelta(minutes=15))
        if prior_5m is not None:
            vix_5m_raw = vix - float(prior_5m)
        if prior_15m is not None:
            vix_15m_raw = vix - float(prior_15m)
    vix_5m, vix_5m_missing = impute(vix_5m_raw)
    vix_15m, vix_15m_missing = impute(vix_15m_raw)

    gap = spx - vwap
    right = right_for(row, contract_id)
    recovered = {
        "spx_vwap_gap_points": gap,
        "spx_vwap_gap_bps": safe_div(gap, spx, min_denominator) * 10_000.0,
        "spx_vwap_gap_over_session_range": safe_div(gap, session_range, range_floor),
        "session_range_bps": safe_div(session_range, spx, min_denominator) * 10_000.0,
        "momentum_5m_bps": safe_div(momentum_5m, spx, min_denominator) * 10_000.0,
        "momentum_15m_bps": safe_div(momentum_15m, spx, min_denominator) * 10_000.0,
        "momentum_5m_over_session_range": safe_div(momentum_5m, session_range, range_floor),
        "momentum_15m_over_session_range": safe_div(momentum_15m, session_range, range_floor),
        "vix_change_5m": vix_5m,
        "vix_change_15m": vix_15m,
        "vix_change_5m_bps": safe_div(vix_5m, vix, vix_floor) * 10_000.0,
        "vix_change_15m_bps": safe_div(vix_15m, vix, vix_floor) * 10_000.0,
        "omar_clipped_neg3_pos3": min(max(omar, omar_min), omar_max),
        "vwap_side_alignment_flag": float((right == "C" and gap > 0.0) or (right == "P" and gap < 0.0)),
        "omar_side_alignment_flag": float((right == "C" and omar > 0.0) or (right == "P" and omar < 0.0)),
        "momentum15_side_alignment_flag": float(
            (right == "C" and momentum_15m > 0.0) or (right == "P" and momentum_15m < 0.0)
        ),
    }
    missing = {
        "spx_vwap_gap_points": spx_missing or vwap_missing,
        "spx_vwap_gap_bps": spx_missing or vwap_missing,
        "spx_vwap_gap_over_session_range": spx_missing or vwap_missing or range_missing,
        "session_range_bps": range_missing or spx_missing,
        "momentum_5m_bps": momentum_5m_missing or spx_missing,
        "momentum_15m_bps": momentum_15m_missing or spx_missing,
        "momentum_5m_over_session_range": momentum_5m_missing or range_missing,
        "momentum_15m_over_session_range": momentum_15m_missing or range_missing,
        "vix_change_5m": vix_5m_missing,
        "vix_change_15m": vix_15m_missing,
        "vix_change_5m_bps": vix_5m_missing or vix_missing,
        "vix_change_15m_bps": vix_15m_missing or vix_missing,
        "omar_clipped_neg3_pos3": omar_missing,
        "vwap_side_alignment_flag": spx_missing or vwap_missing,
        "omar_side_alignment_flag": omar_missing,
        "momentum15_side_alignment_flag": momentum_15m_missing,
    }
    adjacency_sources = {
        "spx_vwap_gap_points": abs(gap),
        "omar": abs(omar),
        "momentum_15m": abs(momentum_15m),
        "vix_change_5m": abs(vix_5m),
        "vix_change_15m": abs(vix_15m),
    }
    return recovered, missing, adjacency_sources


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = int(round((len(ordered) - 1) * float(quantile)))
    return ordered[rank]


def summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered_abs = sorted(abs(float(value)) for value in values)
    return {
        "n": len(values),
        "median_abs": median(ordered_abs),
        "mean_abs": mean(ordered_abs),
        "p95_abs": percentile(ordered_abs, 0.95),
        "p99_abs": percentile(ordered_abs, 0.99),
        "max_abs": ordered_abs[-1],
    }


def sign_threshold_adjacent(
    feature: str,
    live_sources: dict[str, float],
    historical_sources: dict[str, float],
    rules: dict[str, Any],
) -> bool:
    if feature == "vwap_side_alignment_flag":
        limit = float(rules["spx_vwap_gap_points_abs_max_for_threshold_adjacent"])
        return min(live_sources["spx_vwap_gap_points"], historical_sources["spx_vwap_gap_points"]) <= limit
    if feature == "omar_side_alignment_flag":
        limit = float(rules["omar_abs_max_for_threshold_adjacent"])
        return min(live_sources["omar"], historical_sources["omar"]) <= limit
    if feature == "momentum15_side_alignment_flag":
        limit = float(rules["momentum_points_abs_max_for_threshold_adjacent"])
        return min(live_sources["momentum_15m"], historical_sources["momentum_15m"]) <= limit
    return False


def action_threshold_adjacent(live_row: dict[str, Any], historical_row: dict[str, Any]) -> bool:
    distances = [
        finite(live_row.get("threshold_distance")),
        finite(historical_row.get("threshold_distance")),
    ]
    distances.extend(
        [
            finite(live_row.get("selected_score", 0.0)) - finite(live_row.get("decision_threshold", 0.0))
            if finite(live_row.get("selected_score")) is not None and finite(live_row.get("decision_threshold")) is not None
            else None,
            finite(historical_row.get("selected_score", 0.0)) - finite(historical_row.get("decision_threshold", 0.0))
            if finite(historical_row.get("selected_score")) is not None
            and finite(historical_row.get("decision_threshold")) is not None
            else None,
        ]
    )
    finite_distances = [abs(float(value)) for value in distances if value is not None]
    return bool(finite_distances and min(finite_distances) <= 0.02)


def validate_machine_checkable(prereg: dict[str, Any], definition: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    required = set(definition.get("candidate_recovered_model_facing_features") or [])
    expressions = set((definition.get("feature_expressions") or {}).keys())
    continuous = set(definition.get("continuous_recovered_features") or [])
    flags = set(definition.get("sign_or_bucket_recovered_features") or [])
    forbidden = set(definition.get("forbidden_model_facing_sources") or [])
    if not required:
        blockers.append("missing_candidate_recovered_model_facing_features")
    if expressions != required:
        blockers.append("feature_expression_set_mismatch")
    if (continuous | flags) != required or (continuous & flags):
        blockers.append("continuous_sign_bucket_partition_mismatch")
    if forbidden & required:
        blockers.append("forbidden_feature_declared_as_recovered")
    if definition.get("feature_drift_standardization_policy", {}).get("name") != "historical_reference_stddev_with_floor_v1":
        blockers.append("missing_or_unexpected_feature_drift_standardization_policy")
    expected_hash = str(prereg.get("feature_group_definition_sha256") or "")
    actual_hash = sha256_path(Path(prereg["feature_group_definition_path"]))
    if expected_hash != actual_hash:
        blockers.append("feature_group_definition_hash_mismatch")
    if prereg.get("base_contract") != definition.get("base_contract"):
        blockers.append("base_contract_mismatch")
    if prereg.get("base_model_facing_transform") != definition.get("base_transform"):
        blockers.append("base_transform_mismatch")
    return blockers


def trace_paths(prefix: str, session: str) -> tuple[Path, Path]:
    day = session.replace("-", "_")
    historical = BASE_AUDIT / f"{prefix}_historical_replay_{day}" / "decision_traces.jsonl"
    live = BASE_AUDIT / f"{prefix}_ibkr_capture_replay_{day}" / "decision_traces.jsonl"
    return live, historical


def run_audit(preregistration_path: Path, out_dir: Path, trace_prefix: str) -> dict[str, Any]:
    prereg = load_json(preregistration_path)
    definition_path = Path(prereg["feature_group_definition_path"])
    definition = load_json(definition_path)
    blockers = validate_machine_checkable(prereg, definition)
    parity = prereg["parity_evidence_required_before_uplift_claim"]
    drift_gates = parity["feature_drift_gates"]
    timestamp_gates = parity["timestamp_gates"]
    first_et = str(parity["required_first_decision_et"])
    last_et = str(parity["required_last_decision_et"])
    expected_rows = int(parity["paired_rows_per_full_day"])
    continuous_features = list(definition["continuous_recovered_features"])
    flag_features = list(definition["sign_or_bucket_recovered_features"])

    all_continuous_deltas: dict[str, list[float]] = {name: [] for name in continuous_features}
    historical_reference: dict[str, list[float]] = {name: [] for name in continuous_features}
    standardized_deltas: dict[str, list[float]] = {name: [] for name in continuous_features}
    missing_counts: dict[str, int] = {name: 0 for name in definition["candidate_recovered_model_facing_features"]}
    minute_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    daily: dict[str, Any] = {}
    sign_matches = 0
    sign_total = 0
    sign_threshold_adjacent_mismatches = 0
    sign_non_threshold_mismatches = 0
    action_flips = 0
    action_threshold_adjacent_flips = 0
    action_non_threshold_flips = 0
    selected_contract_changes = 0
    membership_mismatches = 0
    total_candidate_pairs = 0

    paired_rows: list[tuple[str, datetime, dict[str, Any], dict[str, Any], dict[str, list[Any]], dict[str, list[Any]]]] = []
    required_sessions = list(parity["required_recorder_days"])
    for session in required_sessions:
        live_path, historical_path = trace_paths(trace_prefix, session)
        if not live_path.exists():
            blockers.append(f"missing_live_trace:{session}:{live_path}")
            continue
        if not historical_path.exists():
            blockers.append(f"missing_historical_trace:{session}:{historical_path}")
            continue
        live_rows_all = load_trace_rows(live_path)
        historical_rows_all = load_trace_rows(historical_path)
        live_vix = vix_by_decision_ts(live_rows_all)
        historical_vix = vix_by_decision_ts(historical_rows_all)
        common_keys = sorted(set(live_rows_all) & set(historical_rows_all))
        window_keys = [ts for ts in common_keys if in_window(ts, first_et, last_et)]
        day_blockers: list[str] = []
        if len(window_keys) != expected_rows:
            day_blockers.append(f"paired_row_count:{len(window_keys)}")
        if window_keys and local_minute(window_keys[0]) != first_et:
            day_blockers.append(f"first_decision_et:{local_minute(window_keys[0])}")
        if window_keys and local_minute(window_keys[-1]) != last_et:
            day_blockers.append(f"last_decision_et:{local_minute(window_keys[-1])}")

        day_action_flips = 0
        day_membership_mismatches = 0
        day_candidate_pairs = 0
        for ts in window_keys:
            live_row = live_rows_all[ts]
            historical_row = historical_rows_all[ts]
            live_context_ts = parse_ts(live_row.get("source_context_ts"))
            historical_context_ts = parse_ts(historical_row.get("source_context_ts"))
            expected_context = ts - timedelta(minutes=1)
            if timestamp_gates.get("source_context_time_equals_decision_minus_one_minute"):
                if live_context_ts != expected_context:
                    day_blockers.append(f"live_context_lag_not_one_minute:{session}:{ts.isoformat()}")
                if historical_context_ts != expected_context:
                    day_blockers.append(f"historical_context_lag_not_one_minute:{session}:{ts.isoformat()}")
            if timestamp_gates.get("no_future_context"):
                if live_context_ts is None or live_context_ts >= ts:
                    day_blockers.append(f"live_future_or_missing_context:{session}:{ts.isoformat()}")
                if historical_context_ts is None or historical_context_ts >= ts:
                    day_blockers.append(f"historical_future_or_missing_context:{session}:{ts.isoformat()}")
            if live_row.get("feature_contract_version") != prereg["base_contract"]:
                day_blockers.append(f"live_contract_mismatch:{session}:{ts.isoformat()}")
            if historical_row.get("feature_contract_version") != prereg["base_contract"]:
                day_blockers.append(f"historical_contract_mismatch:{session}:{ts.isoformat()}")
            if bool(live_row.get("broker_order_endpoint_called")) or bool(historical_row.get("broker_order_endpoint_called")):
                day_blockers.append(f"broker_endpoint_called_in_trace:{session}:{ts.isoformat()}")

            live_membership = membership_fingerprint(live_row)
            historical_membership = membership_fingerprint(historical_row)
            membership_match = live_membership == historical_membership
            if not membership_match:
                membership_mismatches += 1
                day_membership_mismatches += 1

            live_action = str(live_row.get("selected_action"))
            historical_action = str(historical_row.get("selected_action"))
            action_match = live_action == historical_action
            if not action_match:
                action_flips += 1
                day_action_flips += 1
                if action_threshold_adjacent(live_row, historical_row):
                    action_threshold_adjacent_flips += 1
                else:
                    action_non_threshold_flips += 1
            live_selected = live_row.get("selected_contract_id")
            historical_selected = historical_row.get("selected_contract_id")
            if live_selected != historical_selected and (live_selected is not None or historical_selected is not None):
                selected_contract_changes += 1

            live_vectors = token_vectors(live_row)
            historical_vectors = token_vectors(historical_row)
            common_contracts = sorted(set(live_vectors) & set(historical_vectors))
            day_candidate_pairs += len(common_contracts)
            total_candidate_pairs += len(common_contracts)
            paired_rows.append((session, ts, live_row, historical_row, live_vectors, historical_vectors))

            minute_rows.append(
                {
                    "session": session,
                    "decision_time_utc": ts.isoformat(),
                    "decision_time_et": local_minute(ts),
                    "live_candidate_count": len(live_vectors),
                    "historical_candidate_count": len(historical_vectors),
                    "common_candidate_count": len(common_contracts),
                    "candidate_membership_match": membership_match,
                    "action_match": action_match,
                    "live_action": live_action,
                    "historical_action": historical_action,
                    "selected_contract_match": live_selected == historical_selected,
                }
            )

            for contract_id in common_contracts:
                live_recovered, live_missing, live_sources = derive_recovered_features(
                    row=live_row,
                    contract_id=contract_id,
                    vector=live_vectors[contract_id],
                    vix_lookup=live_vix,
                    definition=definition,
                )
                historical_recovered, historical_missing, historical_sources = derive_recovered_features(
                    row=historical_row,
                    contract_id=contract_id,
                    vector=historical_vectors[contract_id],
                    vix_lookup=historical_vix,
                    definition=definition,
                )
                for name in continuous_features:
                    live_value = live_recovered[name]
                    historical_value = historical_recovered[name]
                    all_continuous_deltas[name].append(live_value - historical_value)
                    historical_reference[name].append(historical_value)
                for name in definition["candidate_recovered_model_facing_features"]:
                    if live_missing.get(name) or historical_missing.get(name):
                        missing_counts[name] += 1
                for name in flag_features:
                    sign_total += 1
                    if live_recovered[name] == historical_recovered[name]:
                        sign_matches += 1
                        continue
                    adjacent = sign_threshold_adjacent(
                        name,
                        live_sources,
                        historical_sources,
                        parity["threshold_adjacency_rules"],
                    )
                    if adjacent:
                        sign_threshold_adjacent_mismatches += 1
                    else:
                        sign_non_threshold_mismatches += 1
                    mismatch_rows.append(
                        {
                            "session": session,
                            "decision_time_utc": ts.isoformat(),
                            "decision_time_et": local_minute(ts),
                            "contract_id": contract_id,
                            "feature": name,
                            "live_value": live_recovered[name],
                            "historical_value": historical_recovered[name],
                            "threshold_adjacent": adjacent,
                        }
                    )
        if day_blockers:
            blockers.extend(sorted(set(day_blockers)))
        daily[session] = {
            "live_trace": str(live_path),
            "historical_trace": str(historical_path),
            "paired_rows_in_required_window": len(window_keys),
            "first_decision_et": local_minute(window_keys[0]) if window_keys else None,
            "last_decision_et": local_minute(window_keys[-1]) if window_keys else None,
            "action_flips": day_action_flips,
            "candidate_membership_mismatches": day_membership_mismatches,
            "candidate_pairs": day_candidate_pairs,
            "blockers": sorted(set(day_blockers)),
        }

    standardization_scales: dict[str, float] = {}
    scale_floor = float(definition["feature_drift_standardization_policy"]["scale_floor"])
    for name in continuous_features:
        ref = historical_reference[name]
        scale = stdev(ref) if len(ref) > 1 else 0.0
        scale = max(float(scale), scale_floor)
        standardization_scales[name] = scale
        standardized_deltas[name] = [delta / scale for delta in all_continuous_deltas[name]]

    feature_rows: list[dict[str, Any]] = []
    for name in continuous_features:
        raw = summary(all_continuous_deltas[name])
        scaled = summary(standardized_deltas[name])
        total = max(total_candidate_pairs, 1)
        finite_coverage = 1.0 - (missing_counts[name] / total)
        feature_rows.append(
            {
                "feature": name,
                "family": "continuous",
                "standardization_scale": standardization_scales[name],
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
                "pass": bool(
                    finite_coverage >= float(drift_gates["finite_coverage_min"])
                    and (scaled.get("median_abs") or 0.0)
                    <= float(drift_gates["continuous_feature_standardized_abs_drift_median_max"])
                    and (scaled.get("p95_abs") or 0.0)
                    <= float(drift_gates["continuous_feature_standardized_abs_drift_p95_max"])
                    and (scaled.get("p99_abs") or 0.0)
                    <= float(drift_gates["continuous_feature_standardized_abs_drift_p99_max"])
                ),
            }
        )
    sign_match_rate = (sign_matches / sign_total) if sign_total else 0.0
    sign_gate_pass = (
        sign_match_rate >= float(drift_gates["sign_or_bucket_match_rate_min"])
        and sign_non_threshold_mismatches
        <= int(drift_gates["non_threshold_sign_or_bucket_mismatch_count_max"])
    )
    continuous_gate_pass = all(bool(row["pass"]) for row in feature_rows)
    timestamp_gate_pass = not any(
        "context_lag" in item
        or "future_or_missing_context" in item
        or "paired_row_count" in item
        or "first_decision_et" in item
        or "last_decision_et" in item
        or "contract_mismatch" in item
        for item in blockers
    )
    membership_gate_pass = membership_mismatches == 0
    action_gate_pass = (
        action_non_threshold_flips
        <= int(parity["candidate_replay_action_gates"]["unclassified_non_threshold_action_flips_max"])
        and action_threshold_adjacent_flips
        <= int(parity["candidate_replay_action_gates"]["threshold_adjacent_action_flips_max"])
    )
    status = "pass" if not blockers and continuous_gate_pass and sign_gate_pass and membership_gate_pass and action_gate_pass else "fail"
    decision = "parity_passed_uplift_training_allowed_with_owner_approval" if status == "pass" else "reject_feature_group_before_cv_uplift_claim"

    gate_results = {
        "machine_checkable_preregistration": not any(
            item
            in {
                "missing_candidate_recovered_model_facing_features",
                "feature_expression_set_mismatch",
                "continuous_sign_bucket_partition_mismatch",
                "forbidden_feature_declared_as_recovered",
                "missing_or_unexpected_feature_drift_standardization_policy",
                "feature_group_definition_hash_mismatch",
                "base_contract_mismatch",
                "base_transform_mismatch",
            }
            for item in blockers
        ),
        "timestamp_and_rows": timestamp_gate_pass,
        "candidate_membership_unchanged": membership_gate_pass,
        "continuous_feature_drift": continuous_gate_pass,
        "sign_or_bucket_match": sign_gate_pass,
        "action_flips": action_gate_pass,
    }
    payload_out = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "decision": decision,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": prereg["experiment_id"],
        "feature_group": prereg["feature_group"],
        "base_contract": prereg["base_contract"],
        "base_model_facing_transform": prereg["base_model_facing_transform"],
        "feature_group_definition_path": str(definition_path),
        "feature_group_definition_sha256": sha256_path(definition_path),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": sha256_path(preregistration_path),
        "trace_prefix": trace_prefix,
        "required_sessions": required_sessions,
        "required_first_decision_et": first_et,
        "required_last_decision_et": last_et,
        "required_rows_per_day": expected_rows,
        "daily": daily,
        "gate_results": gate_results,
        "blockers": sorted(set(blockers)),
        "continuous_feature_gate": {
            "median_max": drift_gates["continuous_feature_standardized_abs_drift_median_max"],
            "p95_max": drift_gates["continuous_feature_standardized_abs_drift_p95_max"],
            "p99_max": drift_gates["continuous_feature_standardized_abs_drift_p99_max"],
            "finite_coverage_min": drift_gates["finite_coverage_min"],
            "failed_features": [row["feature"] for row in feature_rows if not row["pass"]],
        },
        "sign_or_bucket_gate": {
            "match_rate": sign_match_rate,
            "match_rate_min": drift_gates["sign_or_bucket_match_rate_min"],
            "sign_total": sign_total,
            "sign_matches": sign_matches,
            "threshold_adjacent_mismatches": sign_threshold_adjacent_mismatches,
            "non_threshold_mismatches": sign_non_threshold_mismatches,
            "non_threshold_mismatch_max": drift_gates["non_threshold_sign_or_bucket_mismatch_count_max"],
        },
        "candidate_membership": {
            "mismatch_count": membership_mismatches,
            "checked_rows": sum(item["paired_rows_in_required_window"] for item in daily.values()),
            "membership_hash_type": "contract_id_right_offset_strike_idx_right_idx",
        },
        "action_replay": {
            "action_flips": action_flips,
            "threshold_adjacent_action_flips": action_threshold_adjacent_flips,
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
    write_json(out_dir / "summary.json", payload_out)
    write_csv(out_dir / "feature_drift_summary.csv", feature_rows)
    write_csv(out_dir / "sign_bucket_mismatches.csv", mismatch_rows)
    write_csv(out_dir / "minute_parity.csv", minute_rows)
    write_report(out_dir / "report.md", payload_out, feature_rows)
    return payload_out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "no_rows"}])


def write_report(path: Path, payload: dict[str, Any], feature_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Protocol101 Feature Recovery Group 1 Parity Audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Experiment: `{payload['experiment_id']}`",
        f"- Feature group: `{payload['feature_group']}`",
        f"- Contract: `{payload['base_contract']}`",
        f"- Transform: `{payload['base_model_facing_transform']}`",
        f"- Feature definition hash: `{payload['feature_group_definition_sha256']}`",
        "",
        "## Gates",
        "",
    ]
    for name, passed in payload["gate_results"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.extend(["", "## Daily Evidence", ""])
    for session, row in payload["daily"].items():
        lines.append(
            f"- `{session}`: rows=`{row['paired_rows_in_required_window']}` "
            f"first/last=`{row['first_decision_et']}`/`{row['last_decision_et']}` "
            f"candidate_pairs=`{row['candidate_pairs']}` action_flips=`{row['action_flips']}` "
            f"membership_mismatches=`{row['candidate_membership_mismatches']}`"
        )
    lines.extend(["", "## Feature Drift", ""])
    for row in sorted(feature_rows, key=lambda item: float(item.get("standardized_p99_abs") or 0.0), reverse=True):
        lines.append(
            f"- `{row['feature']}`: pass=`{row['pass']}` "
            f"std median/p95/p99=`{row['standardized_median_abs']}`/`{row['standardized_p95_abs']}`/`{row['standardized_p99_abs']}` "
            f"raw_missing=`{row['raw_missing_count']}`"
        )
    sign = payload["sign_or_bucket_gate"]
    lines.extend(
        [
            "",
            "## Sign And Action",
            "",
            f"- Sign/bucket match rate: `{sign['match_rate']}`",
            f"- Sign/bucket threshold-adjacent mismatches: `{sign['threshold_adjacent_mismatches']}`",
            f"- Sign/bucket non-threshold mismatches: `{sign['non_threshold_mismatches']}`",
            f"- Action flips: `{payload['action_replay']['action_flips']}`",
            f"- Unclassified non-threshold action flips: `{payload['action_replay']['unclassified_non_threshold_action_flips']}`",
            "",
            "## Side Effects",
            "",
        ]
    )
    for key, value in payload["side_effect_policy"].items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    if payload["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in payload["blockers"]:
            lines.append(f"- `{blocker}`")
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
