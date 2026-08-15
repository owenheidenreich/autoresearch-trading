"""Audit Protocol101 static-ladder boundary-stable feature recovery policy.

This script is offline-only. It reads already-local paired source-aligned
historical and IBKR replay traces, treats the 42-slot candidate filter trace as
the model-facing static ladder for non-quote features, and reruns Group 1/2
parity checks under that policy. It does not train, tune, submit orders, or
change runtime defaults.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import stdev
from typing import Any

from v4.scripts import run_protocol101_feature_recovery_group1_parity_audit as g1


BASE_AUDIT = Path("v4/audit/autoresearch")
DEFAULT_OUT_DIR = BASE_AUDIT / "protocol101_live_v2_static_ladder_boundary_stable_policy_audit"
DEFAULT_POLICY_OPTIONS = BASE_AUDIT / "protocol101_live_v2_option_quote_source_policy_resolution" / "policy_options.json"
DEFAULT_TRACE_PREFIX = "protocol101_live_v2_candidate_universe_parity_source_aligned"
DEFAULT_GROUP1_PREREGISTRATION = (
    BASE_AUDIT / "protocol101_live_v2_feature_recovery_group1_stable_index_context_plan" / "preregistration.json"
)
DEFAULT_GROUP2_PREREGISTRATION = (
    BASE_AUDIT / "protocol101_live_v2_feature_recovery_group2_candidate_geometry_plan" / "preregistration.json"
)
CONTRACT = "protocol101-live-v2-microstructure-masked"
TRANSFORM = "mask_vendor_sensitive_option_quote_greek_microstructure"
POLICY_ID = "boundary_stable_tradability_plus_static_ladder_for_non_quote_features_v1"
SCHEMA_VERSION = "Protocol101StaticLadderBoundaryStablePolicyAuditV1"
CONTRACT_MULTIPLIER = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy-options", type=Path, default=DEFAULT_POLICY_OPTIONS)
    parser.add_argument("--trace-prefix", default=DEFAULT_TRACE_PREFIX)
    parser.add_argument("--group1-preregistration", type=Path, default=DEFAULT_GROUP1_PREREGISTRATION)
    parser.add_argument("--group2-preregistration", type=Path, default=DEFAULT_GROUP2_PREREGISTRATION)
    return parser.parse_args()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def canonical_float(value: Any, digits: int = 6) -> float | None:
    number = finite(value)
    return None if number is None else round(number, digits)


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_policy_margins(path: Path) -> dict[str, float]:
    payload = g1.load_json(path)
    margins = (
        payload.get("options", {})
        .get("B_boundary_stable_tradability_policy", {})
        .get("frozen_margins_for_next_audit", {})
    )
    required = {
        "min_mid",
        "max_mid",
        "max_spread_abs",
        "max_spread_frac",
        "max_quote_age_ms",
        "min_bid_size",
        "min_ask_size",
        "stable_min_mid",
        "stable_max_mid",
        "stable_max_spread_abs",
        "stable_max_spread_frac",
        "stable_max_quote_age_ms",
        "stable_min_bid_size",
        "stable_min_ask_size",
        "max_affordability_utilization",
    }
    missing = sorted(required - set(margins))
    if missing:
        raise ValueError(f"policy_options_missing_frozen_margins:{missing}")
    return {name: float(margins[name]) for name in required}


def static_slots(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in row.get("candidate_filter_trace") or [] if isinstance(item, dict)]


def static_slot_identity(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": str(item.get("contract_id") or ""),
        "right": str(item.get("right") or ""),
        "strike": canonical_float(item.get("strike")),
        "offset": canonical_float(item.get("offset")),
        "strike_idx": int_or_none(item.get("strike_idx")),
        "right_idx": int_or_none(item.get("right_idx")),
        "atm_strike": canonical_float(item.get("atm_strike")),
        "strike_step": int_or_none(item.get("strike_step")),
        "spx_for_ladder": canonical_float(item.get("spx_for_ladder")),
        "source_context_ts": str(item.get("source_context_ts") or item.get("source_context_time") or ""),
    }


def static_slot_key(item: dict[str, Any]) -> str:
    return stable_hash(static_slot_identity(item))


def static_universe_fingerprint(row: dict[str, Any]) -> str:
    return stable_hash([static_slot_identity(item) for item in static_slots(row)])


def observed_field(item: dict[str, Any], name: str) -> Any:
    observed = (item.get("candidate_filter") or {}).get("observed") or {}
    return observed.get(name, item.get(name))


def derived_observed_quote(item: dict[str, Any]) -> dict[str, float | None]:
    bid = finite(observed_field(item, "bid"))
    ask = finite(observed_field(item, "ask"))
    mid = finite(observed_field(item, "mid"))
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    spread = finite(observed_field(item, "spread"))
    if spread is None and bid is not None and ask is not None:
        spread = ask - bid
    spread_frac = finite(observed_field(item, "spread_frac"))
    if spread_frac is None and spread is not None and mid is not None and abs(mid) > 0.0:
        spread_frac = spread / abs(mid)
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "spread_frac": spread_frac,
        "bid_size": finite(observed_field(item, "bid_size")),
        "ask_size": finite(observed_field(item, "ask_size")),
        "quote_age_ms": finite(observed_field(item, "quote_age_ms")),
    }


def guard_status(
    item: dict[str, Any],
    margins: dict[str, float],
    *,
    cash: float,
    contract_multiplier: float = CONTRACT_MULTIPLIER,
) -> dict[str, Any]:
    quote = derived_observed_quote(item)
    ask = quote["ask"]
    affordability_utilization = None
    if ask is not None and cash > 0.0:
        affordability_utilization = (ask * contract_multiplier) / cash

    missing_quote = quote["bid"] is None or ask is None or quote["mid"] is None
    reasons = list(item.get("filter_reasons") or (item.get("candidate_filter") or {}).get("reasons") or [])
    if missing_quote:
        status = "quote_missing"
        base_tradable = False
        boundary_stable = False
    elif quote["quote_age_ms"] is None:
        status = "quote_age_missing"
        base_tradable = False
        boundary_stable = False
    elif quote["quote_age_ms"] > margins["max_quote_age_ms"]:
        status = "stale"
        base_tradable = False
        boundary_stable = False
    elif affordability_utilization is not None and affordability_utilization > 1.0:
        status = "unaffordable"
        base_tradable = False
        boundary_stable = False
    else:
        base_checks = {
            "mid_min": quote["mid"] is not None and quote["mid"] >= margins["min_mid"],
            "mid_max": quote["mid"] is not None and quote["mid"] <= margins["max_mid"],
            "spread_abs": quote["spread"] is not None and quote["spread"] <= margins["max_spread_abs"],
            "spread_frac": quote["spread_frac"] is not None and quote["spread_frac"] <= margins["max_spread_frac"],
            "bid_size": quote["bid_size"] is not None and quote["bid_size"] >= margins["min_bid_size"],
            "ask_size": quote["ask_size"] is not None and quote["ask_size"] >= margins["min_ask_size"],
        }
        base_tradable = all(base_checks.values())
        if not base_tradable:
            status = "untradable"
            boundary_stable = False
        else:
            stable_checks = {
                "stable_mid_min": quote["mid"] is not None and quote["mid"] >= margins["stable_min_mid"],
                "stable_mid_max": quote["mid"] is not None and quote["mid"] <= margins["stable_max_mid"],
                "stable_spread_abs": quote["spread"] is not None and quote["spread"] <= margins["stable_max_spread_abs"],
                "stable_spread_frac": quote["spread_frac"] is not None
                and quote["spread_frac"] <= margins["stable_max_spread_frac"],
                "stable_quote_age": quote["quote_age_ms"] is not None
                and quote["quote_age_ms"] <= margins["stable_max_quote_age_ms"],
                "stable_bid_size": quote["bid_size"] is not None
                and quote["bid_size"] >= margins["stable_min_bid_size"],
                "stable_ask_size": quote["ask_size"] is not None
                and quote["ask_size"] >= margins["stable_min_ask_size"],
                "stable_affordability": affordability_utilization is None
                or affordability_utilization <= margins["max_affordability_utilization"],
            }
            boundary_stable = all(stable_checks.values())
            status = "tradable_boundary_stable" if boundary_stable else "tradable_unstable"

    candidate_filter = item.get("candidate_filter") or {}
    return {
        "status": status,
        "base_tradable": base_tradable,
        "boundary_stable": boundary_stable,
        "candidate_filter_passed": bool(candidate_filter.get("passed", item.get("post_filter_candidate", False))),
        "post_filter_candidate": bool(item.get("post_filter_candidate", False)),
        "freshness_pass": bool(item.get("freshness_pass", candidate_filter.get("freshness_pass", False))),
        "tradability_pass": bool(item.get("tradability_pass", candidate_filter.get("tradability_pass", False))),
        "filter_reasons": reasons,
        "quote": quote,
        "affordability_utilization": affordability_utilization,
        "quote_age_source": item.get("quote_age_source"),
        "source_quote_ts": item.get("source_quote_ts") or item.get("source_quote_time"),
    }


def cash_for_row(row: dict[str, Any]) -> float:
    return finite((row.get("account_state") or {}).get("cash")) or 10_000.0


def load_required_inputs(
    trace_prefix: str,
    session: str,
) -> tuple[Path, Path, dict[datetime, dict[str, Any]], dict[datetime, dict[str, Any]]]:
    live_path, historical_path = g1.trace_paths(trace_prefix, session)
    return live_path, historical_path, g1.load_trace_rows(live_path), g1.load_trace_rows(historical_path)


def collect_policy_metrics(
    *,
    trace_prefix: str,
    required_sessions: list[str],
    first_et: str,
    last_et: str,
    expected_rows: int,
    margins: dict[str, float],
    out_dir: Path,
) -> dict[str, Any]:
    daily: dict[str, Any] = {}
    blockers: list[str] = []
    universe_mismatch_rows: list[dict[str, Any]] = []
    guard_status_counts: dict[str, dict[str, int]] = {"historical": {}, "ibkr": {}}
    guard_divergence_counts: dict[str, int] = {}
    stable_counts = {"historical": 0, "ibkr": 0}
    base_tradable_counts = {"historical": 0, "ibkr": 0}
    status_divergence_count = 0
    stable_divergence_count = 0
    post_filter_divergence_count = 0
    slot_pairs_checked = 0
    rows_checked = 0
    action_flips = 0
    broker_endpoint_called = False
    replay_artifacts: list[str] = []

    for session in required_sessions:
        live_path, historical_path, live_rows_all, historical_rows_all = load_required_inputs(trace_prefix, session)
        common_keys = sorted(set(live_rows_all) & set(historical_rows_all))
        window_keys = [ts for ts in common_keys if g1.in_window(ts, first_et, last_et)]
        day_blockers: list[str] = []
        if len(window_keys) != expected_rows:
            day_blockers.append(f"paired_row_count:{len(window_keys)}")
        if window_keys and g1.local_minute(window_keys[0]) != first_et:
            day_blockers.append(f"first_decision_et:{g1.local_minute(window_keys[0])}")
        if window_keys and g1.local_minute(window_keys[-1]) != last_et:
            day_blockers.append(f"last_decision_et:{g1.local_minute(window_keys[-1])}")

        replay_path = out_dir / f"policy_replay_{session.replace('-', '_')}.jsonl"
        replay_artifacts.append(str(replay_path))
        with replay_path.open("w") as replay_handle:
            day_guard_divergences = 0
            day_stable_hist = 0
            day_stable_live = 0
            day_slot_pairs = 0
            day_universe_mismatches = 0
            for ts in window_keys:
                live_row = live_rows_all[ts]
                historical_row = historical_rows_all[ts]
                rows_checked += 1
                if bool(live_row.get("broker_order_endpoint_called")) or bool(
                    historical_row.get("broker_order_endpoint_called")
                ):
                    broker_endpoint_called = True
                    day_blockers.append(f"broker_endpoint_called_in_trace:{session}:{ts.isoformat()}")
                if live_row.get("feature_contract_version") != CONTRACT:
                    day_blockers.append(f"live_contract_mismatch:{session}:{ts.isoformat()}")
                if historical_row.get("feature_contract_version") != CONTRACT:
                    day_blockers.append(f"historical_contract_mismatch:{session}:{ts.isoformat()}")
                expected_context = ts - timedelta(minutes=1)
                if g1.parse_ts(live_row.get("source_context_ts")) != expected_context:
                    day_blockers.append(f"live_context_lag_not_one_minute:{session}:{ts.isoformat()}")
                if g1.parse_ts(historical_row.get("source_context_ts")) != expected_context:
                    day_blockers.append(f"historical_context_lag_not_one_minute:{session}:{ts.isoformat()}")

                live_slots = static_slots(live_row)
                historical_slots = static_slots(historical_row)
                live_fingerprint = static_universe_fingerprint(live_row)
                historical_fingerprint = static_universe_fingerprint(historical_row)
                universe_match = live_fingerprint == historical_fingerprint and len(live_slots) == len(historical_slots)
                if not universe_match:
                    day_universe_mismatches += 1
                    universe_mismatch_rows.append(
                        {
                            "session": session,
                            "decision_time_utc": ts.isoformat(),
                            "decision_time_et": g1.local_minute(ts),
                            "historical_static_slot_count": len(historical_slots),
                            "ibkr_static_slot_count": len(live_slots),
                            "historical_static_ladder_hash": historical_fingerprint,
                            "ibkr_static_ladder_hash": live_fingerprint,
                        }
                    )

                if str(live_row.get("selected_action")) != str(historical_row.get("selected_action")):
                    action_flips += 1

                historical_by_key = {static_slot_key(item): item for item in historical_slots}
                live_by_key = {static_slot_key(item): item for item in live_slots}
                common_slot_keys = sorted(set(historical_by_key) & set(live_by_key))
                row_guard_divergences = 0
                row_stable_hist = 0
                row_stable_live = 0
                row_status_counts = {"historical": {}, "ibkr": {}}
                for slot_key in common_slot_keys:
                    historical_item = historical_by_key[slot_key]
                    live_item = live_by_key[slot_key]
                    historical_status = guard_status(
                        historical_item,
                        margins,
                        cash=cash_for_row(historical_row),
                    )
                    live_status = guard_status(live_item, margins, cash=cash_for_row(live_row))
                    slot_pairs_checked += 1
                    day_slot_pairs += 1
                    for side, status_payload in (("historical", historical_status), ("ibkr", live_status)):
                        status_name = str(status_payload["status"])
                        guard_status_counts[side][status_name] = guard_status_counts[side].get(status_name, 0) + 1
                        row_status_counts[side][status_name] = row_status_counts[side].get(status_name, 0) + 1
                        if status_payload["base_tradable"]:
                            base_tradable_counts[side] += 1
                        if status_payload["boundary_stable"]:
                            stable_counts[side] += 1
                    if historical_status["boundary_stable"]:
                        row_stable_hist += 1
                        day_stable_hist += 1
                    if live_status["boundary_stable"]:
                        row_stable_live += 1
                        day_stable_live += 1
                    if historical_status["status"] != live_status["status"]:
                        status_divergence_count += 1
                        row_guard_divergences += 1
                        day_guard_divergences += 1
                        key = f"{historical_status['status']}->{live_status['status']}"
                        guard_divergence_counts[key] = guard_divergence_counts.get(key, 0) + 1
                    if historical_status["boundary_stable"] != live_status["boundary_stable"]:
                        stable_divergence_count += 1
                    if historical_status["post_filter_candidate"] != live_status["post_filter_candidate"]:
                        post_filter_divergence_count += 1

                replay_handle.write(
                    json.dumps(
                        {
                            "schema_version": "Protocol101StaticLadderBoundaryStablePolicyReplayRowV1",
                            "policy_id": POLICY_ID,
                            "session": session,
                            "decision_time_utc": ts.isoformat(),
                            "decision_time_et": g1.local_minute(ts),
                            "static_ladder_universe_match": universe_match,
                            "historical_static_ladder_hash": historical_fingerprint,
                            "ibkr_static_ladder_hash": live_fingerprint,
                            "historical_static_slot_count": len(historical_slots),
                            "ibkr_static_slot_count": len(live_slots),
                            "paired_static_slot_count": len(common_slot_keys),
                            "historical_guard_status_counts": row_status_counts["historical"],
                            "ibkr_guard_status_counts": row_status_counts["ibkr"],
                            "guard_status_divergence_count": row_guard_divergences,
                            "historical_boundary_stable_count": row_stable_hist,
                            "ibkr_boundary_stable_count": row_stable_live,
                            "historical_action": historical_row.get("selected_action"),
                            "ibkr_action": live_row.get("selected_action"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if day_blockers:
                blockers.extend(sorted(set(day_blockers)))
            daily[session] = {
                "historical_trace": str(historical_path),
                "ibkr_trace": str(live_path),
                "policy_replay_artifact": str(replay_path),
                "paired_rows_in_required_window": len(window_keys),
                "first_decision_et": g1.local_minute(window_keys[0]) if window_keys else None,
                "last_decision_et": g1.local_minute(window_keys[-1]) if window_keys else None,
                "static_ladder_universe_mismatch_rows": day_universe_mismatches,
                "paired_static_slot_count": day_slot_pairs,
                "guard_status_divergences": day_guard_divergences,
                "historical_boundary_stable_count": day_stable_hist,
                "ibkr_boundary_stable_count": day_stable_live,
                "blockers": sorted(set(day_blockers)),
            }

    expected_total_rows = expected_rows * len(required_sessions)
    expected_slots = expected_total_rows * 42
    universe_match_rate = 1.0 - (len(universe_mismatch_rows) / max(rows_checked, 1))
    return {
        "policy_id": POLICY_ID,
        "trace_prefix": trace_prefix,
        "required_sessions": required_sessions,
        "required_first_decision_et": first_et,
        "required_last_decision_et": last_et,
        "required_rows_per_day": expected_rows,
        "decision_rows_checked": rows_checked,
        "expected_decision_rows": expected_total_rows,
        "static_ladder_slots_per_row": 42,
        "expected_static_slot_pairs": expected_slots,
        "paired_static_slot_pairs_checked": slot_pairs_checked,
        "static_ladder_universe_match_rate": universe_match_rate,
        "static_ladder_universe_mismatch_rows": universe_mismatch_rows,
        "model_facing_candidate_count_parity": slot_pairs_checked == expected_slots
        and rows_checked == expected_total_rows
        and not universe_mismatch_rows,
        "static_ladder_universe_parity_pass": universe_match_rate == 1.0
        and rows_checked == expected_total_rows
        and slot_pairs_checked == expected_slots,
        "daily": daily,
        "guard_status_counts": guard_status_counts,
        "guard_status_divergence_counts": dict(sorted(guard_divergence_counts.items())),
        "guard_status_divergence_count": status_divergence_count,
        "boundary_stable_divergence_count": stable_divergence_count,
        "post_filter_candidate_divergence_count": post_filter_divergence_count,
        "boundary_stable_candidate_counts": stable_counts,
        "base_tradable_candidate_counts": base_tradable_counts,
        "action_flips_in_existing_replay": action_flips,
        "blockers": sorted(set(blockers)),
        "replay_artifacts": replay_artifacts,
        "side_effect_policy": side_effect_policy(broker_endpoint_called=broker_endpoint_called),
    }


def context_vector(row: dict[str, Any]) -> list[Any] | None:
    vectors = g1.token_vectors(row)
    return next(iter(vectors.values()), None)


def summarize_continuous_features(
    *,
    continuous_features: list[str],
    deltas: dict[str, list[float]],
    historical_reference: dict[str, list[float]],
    missing_counts: dict[str, int],
    total_pairs: int,
    drift_gates: dict[str, Any],
    definition: dict[str, Any],
) -> list[dict[str, Any]]:
    feature_rows: list[dict[str, Any]] = []
    scale_floor = float(definition["feature_drift_standardization_policy"]["scale_floor"])
    for name in continuous_features:
        ref = historical_reference[name]
        scale = max(stdev(ref) if len(ref) > 1 else 0.0, scale_floor)
        standardized = [delta / scale for delta in deltas[name]]
        raw = g1.summary(deltas[name])
        scaled = g1.summary(standardized)
        finite_coverage = 1.0 - (missing_counts[name] / max(total_pairs, 1))
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
    return feature_rows


def evaluate_group1(
    *,
    preregistration_path: Path,
    trace_prefix: str,
    policy_metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    feature_names = list(definition["candidate_recovered_model_facing_features"])
    deltas: dict[str, list[float]] = {name: [] for name in continuous_features}
    historical_reference: dict[str, list[float]] = {name: [] for name in continuous_features}
    missing_counts: dict[str, int] = {name: 0 for name in feature_names}
    mismatch_rows: list[dict[str, Any]] = []
    daily: dict[str, Any] = {}
    total_pairs = 0
    sign_total = 0
    sign_matches = 0
    sign_threshold_adjacent_mismatches = 0
    sign_non_threshold_mismatches = 0
    action_flips = 0
    action_non_threshold_flips = 0

    for session in parity["required_recorder_days"]:
        _live_path, _historical_path, live_rows_all, historical_rows_all = load_required_inputs(trace_prefix, session)
        live_vix = g1.vix_by_decision_ts(live_rows_all)
        historical_vix = g1.vix_by_decision_ts(historical_rows_all)
        window_keys = [
            ts for ts in sorted(set(live_rows_all) & set(historical_rows_all)) if g1.in_window(ts, first_et, last_et)
        ]
        day_blockers: list[str] = []
        if len(window_keys) != expected_rows:
            day_blockers.append(f"paired_row_count:{len(window_keys)}")
        day_pairs = 0
        day_action_flips = 0
        for ts in window_keys:
            live_row = live_rows_all[ts]
            historical_row = historical_rows_all[ts]
            live_vector = context_vector(live_row)
            historical_vector = context_vector(historical_row)
            if live_vector is None or historical_vector is None:
                day_blockers.append(f"missing_context_token_vector:{session}:{ts.isoformat()}")
                continue
            if str(live_row.get("selected_action")) != str(historical_row.get("selected_action")):
                action_flips += 1
                day_action_flips += 1
                if not g1.action_threshold_adjacent(live_row, historical_row):
                    action_non_threshold_flips += 1
            live_slots = {static_slot_key(item): item for item in static_slots(live_row)}
            historical_slots = {static_slot_key(item): item for item in static_slots(historical_row)}
            for slot_key in sorted(set(live_slots) & set(historical_slots)):
                live_item = live_slots[slot_key]
                historical_item = historical_slots[slot_key]
                contract_id = str(live_item.get("contract_id") or historical_item.get("contract_id") or "")
                live_recovered, live_missing, live_sources = g1.derive_recovered_features(
                    row=live_row,
                    contract_id=contract_id,
                    vector=live_vector,
                    vix_lookup=live_vix,
                    definition=definition,
                )
                historical_recovered, historical_missing, historical_sources = g1.derive_recovered_features(
                    row=historical_row,
                    contract_id=contract_id,
                    vector=historical_vector,
                    vix_lookup=historical_vix,
                    definition=definition,
                )
                total_pairs += 1
                day_pairs += 1
                for name in continuous_features:
                    deltas[name].append(live_recovered[name] - historical_recovered[name])
                    historical_reference[name].append(historical_recovered[name])
                for name in feature_names:
                    if live_missing.get(name) or historical_missing.get(name):
                        missing_counts[name] += 1
                for name in flag_features:
                    sign_total += 1
                    if live_recovered[name] == historical_recovered[name]:
                        sign_matches += 1
                        continue
                    adjacent = g1.sign_threshold_adjacent(
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
                            "decision_time_et": g1.local_minute(ts),
                            "contract_id": contract_id,
                            "feature": name,
                            "live_value": live_recovered[name],
                            "historical_value": historical_recovered[name],
                            "threshold_adjacent": adjacent,
                        }
                    )
        blockers.extend(day_blockers)
        daily[session] = {
            "paired_rows_in_required_window": len(window_keys),
            "static_ladder_candidate_pairs": day_pairs,
            "action_flips": day_action_flips,
            "blockers": sorted(set(day_blockers)),
        }

    feature_rows = summarize_continuous_features(
        continuous_features=continuous_features,
        deltas=deltas,
        historical_reference=historical_reference,
        missing_counts=missing_counts,
        total_pairs=total_pairs,
        drift_gates=drift_gates,
        definition=definition,
    )
    continuous_gate_pass = all(bool(row["pass"]) for row in feature_rows)
    sign_match_rate = sign_matches / sign_total if sign_total else 0.0
    sign_gate_pass = (
        sign_match_rate >= float(drift_gates["sign_or_bucket_match_rate_min"])
        and sign_non_threshold_mismatches <= int(drift_gates["non_threshold_sign_or_bucket_mismatch_count_max"])
    )
    static_universe_pass = bool(policy_metrics["static_ladder_universe_parity_pass"])
    action_gate_pass = action_non_threshold_flips <= int(
        parity["candidate_replay_action_gates"]["unclassified_non_threshold_action_flips_max"]
    )
    timestamp_gate_pass = not blockers and total_pairs == policy_metrics["expected_static_slot_pairs"]
    eligible = static_universe_pass and timestamp_gate_pass and continuous_gate_pass and sign_gate_pass and action_gate_pass
    result = {
        "feature_group": prereg["feature_group"],
        "experiment_id": prereg["experiment_id"],
        "preregistration_path": str(preregistration_path),
        "feature_group_definition_path": str(definition_path),
        "static_ladder_policy_id": POLICY_ID,
        "status": "eligible_for_later_uplift_testing" if eligible else "not_eligible_for_later_uplift_testing",
        "eligibility": bool(eligible),
        "decision": (
            "policy_parity_passed_uplift_training_allowed_only_with_separate_owner_approval"
            if eligible
            else "blocked_before_uplift_under_static_ladder_policy"
        ),
        "gate_results": {
            "static_ladder_universe": static_universe_pass,
            "timestamp_and_rows": timestamp_gate_pass,
            "continuous_feature_drift": continuous_gate_pass,
            "sign_or_bucket_match": sign_gate_pass,
            "action_flips": action_gate_pass,
        },
        "continuous_feature_gate": {
            "failed_features": [row["feature"] for row in feature_rows if not row["pass"]],
            "finite_coverage_min": drift_gates["finite_coverage_min"],
        },
        "sign_or_bucket_gate": {
            "match_rate": sign_match_rate,
            "match_rate_min": drift_gates["sign_or_bucket_match_rate_min"],
            "sign_total": sign_total,
            "sign_matches": sign_matches,
            "threshold_adjacent_mismatches": sign_threshold_adjacent_mismatches,
            "non_threshold_mismatches": sign_non_threshold_mismatches,
        },
        "action_replay": {
            "action_flips": action_flips,
            "unclassified_non_threshold_action_flips": action_non_threshold_flips,
        },
        "daily": daily,
        "candidate_pairs": total_pairs,
        "blockers": sorted(set(blockers)),
    }
    return result, feature_rows, mismatch_rows


def geometry_context(slots: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    by_right: dict[str, list[dict[str, Any]]] = {}
    offsets: dict[str, float] = {}
    for item in slots:
        right = str(item.get("right") or "")
        by_right.setdefault(right, []).append(item)
        offsets[static_slot_key(item)] = finite(item.get("offset")) or 0.0
    for right, items in by_right.items():
        items.sort(key=lambda item: (abs(offsets[static_slot_key(item)]), str(item.get("contract_id") or "")))
    return by_right, offsets


def derive_static_geometry_features(
    *,
    item: dict[str, Any],
    slots: list[dict[str, Any]],
    definition: dict[str, Any],
) -> tuple[dict[str, float], dict[str, bool]]:
    by_right, offsets = geometry_context(slots)
    item_key = static_slot_key(item)
    right = str(item.get("right") or "")
    offset = offsets.get(item_key, 0.0)
    spx = finite(item.get("spx_for_ladder")) or 0.0
    same_right_items = by_right.get(right, [])
    same_right_keys = [static_slot_key(candidate) for candidate in same_right_items]
    rank = same_right_keys.index(item_key) if item_key in same_right_keys else 0
    same_right_count = len(same_right_items)
    abs_offset = abs(offset)
    neighbor_count = sum(
        1 for candidate in same_right_items if abs((offsets.get(static_slot_key(candidate)) or 0.0) - offset) <= 10.0
    )
    min_denominator = float(definition["deterministic_guards"]["minimum_denominator_abs"])
    recovered = {
        "right_is_call": float(right == "C"),
        "right_is_put": float(right == "P"),
        "offset_points": offset,
        "abs_offset_points": abs_offset,
        "offset_bps_underlying": g1.safe_div(offset, spx, min_denominator) * 10_000.0,
        "abs_offset_bps_underlying": g1.safe_div(abs_offset, spx, min_denominator) * 10_000.0,
        "offset_sign": -1.0 if offset < 0.0 else 1.0 if offset > 0.0 else 0.0,
        "out_of_the_money_flag": float((right == "C" and offset > 0.0) or (right == "P" and offset < 0.0)),
        "in_the_money_flag": float((right == "C" and offset < 0.0) or (right == "P" and offset > 0.0)),
        "same_right_rank_by_abs_offset": float(rank),
        "same_right_candidate_count": float(same_right_count),
        "candidate_count_total": float(len(slots)),
        "same_right_abs_offset_percentile": float(rank) / max(float(same_right_count - 1), 1.0),
        "local_same_right_neighbor_count_10_points": float(neighbor_count),
        "abs_offset_bucket_0_10": float(abs_offset <= 10.0),
        "abs_offset_bucket_10_25": float(10.0 < abs_offset <= 25.0),
        "abs_offset_bucket_25_50": float(25.0 < abs_offset <= 50.0),
        "abs_offset_bucket_gt_50": float(abs_offset > 50.0),
    }
    missing = {
        name: (spx == 0.0 if "bps_underlying" in name else False)
        for name in definition["candidate_recovered_model_facing_features"]
    }
    return recovered, missing


def evaluate_group2(
    *,
    preregistration_path: Path,
    trace_prefix: str,
    policy_metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    feature_names = list(definition["candidate_recovered_model_facing_features"])
    deltas: dict[str, list[float]] = {name: [] for name in continuous_features}
    historical_reference: dict[str, list[float]] = {name: [] for name in continuous_features}
    missing_counts: dict[str, int] = {name: 0 for name in feature_names}
    mismatch_rows: list[dict[str, Any]] = []
    daily: dict[str, Any] = {}
    total_pairs = 0
    sign_total = 0
    sign_matches = 0
    sign_non_threshold_mismatches = 0
    action_flips = 0
    action_non_threshold_flips = 0

    for session in parity["required_recorder_days"]:
        _live_path, _historical_path, live_rows_all, historical_rows_all = load_required_inputs(trace_prefix, session)
        window_keys = [
            ts for ts in sorted(set(live_rows_all) & set(historical_rows_all)) if g1.in_window(ts, first_et, last_et)
        ]
        day_blockers: list[str] = []
        if len(window_keys) != expected_rows:
            day_blockers.append(f"paired_row_count:{len(window_keys)}")
        day_pairs = 0
        day_action_flips = 0
        for ts in window_keys:
            live_row = live_rows_all[ts]
            historical_row = historical_rows_all[ts]
            if str(live_row.get("selected_action")) != str(historical_row.get("selected_action")):
                action_flips += 1
                day_action_flips += 1
                if not g1.action_threshold_adjacent(live_row, historical_row):
                    action_non_threshold_flips += 1
            live_slots = {static_slot_key(item): item for item in static_slots(live_row)}
            historical_slots = {static_slot_key(item): item for item in static_slots(historical_row)}
            live_slot_list = static_slots(live_row)
            historical_slot_list = static_slots(historical_row)
            for slot_key in sorted(set(live_slots) & set(historical_slots)):
                live_recovered, live_missing = derive_static_geometry_features(
                    item=live_slots[slot_key],
                    slots=live_slot_list,
                    definition=definition,
                )
                historical_recovered, historical_missing = derive_static_geometry_features(
                    item=historical_slots[slot_key],
                    slots=historical_slot_list,
                    definition=definition,
                )
                total_pairs += 1
                day_pairs += 1
                for name in continuous_features:
                    deltas[name].append(live_recovered[name] - historical_recovered[name])
                    historical_reference[name].append(historical_recovered[name])
                for name in feature_names:
                    if live_missing.get(name) or historical_missing.get(name):
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
                                "contract_id": live_slots[slot_key].get("contract_id"),
                                "feature": name,
                                "live_value": live_recovered[name],
                                "historical_value": historical_recovered[name],
                            }
                        )
        blockers.extend(day_blockers)
        daily[session] = {
            "paired_rows_in_required_window": len(window_keys),
            "static_ladder_candidate_pairs": day_pairs,
            "action_flips": day_action_flips,
            "blockers": sorted(set(day_blockers)),
        }

    feature_rows = summarize_continuous_features(
        continuous_features=continuous_features,
        deltas=deltas,
        historical_reference=historical_reference,
        missing_counts=missing_counts,
        total_pairs=total_pairs,
        drift_gates=drift_gates,
        definition=definition,
    )
    continuous_gate_pass = all(bool(row["pass"]) for row in feature_rows)
    sign_match_rate = sign_matches / sign_total if sign_total else 0.0
    sign_gate_pass = (
        sign_match_rate >= float(drift_gates["sign_or_bucket_match_rate_min"])
        and sign_non_threshold_mismatches <= int(drift_gates["non_threshold_sign_or_bucket_mismatch_count_max"])
    )
    static_universe_pass = bool(policy_metrics["static_ladder_universe_parity_pass"])
    action_gate_pass = action_non_threshold_flips <= int(
        parity["candidate_replay_action_gates"]["unclassified_non_threshold_action_flips_max"]
    )
    timestamp_gate_pass = not blockers and total_pairs == policy_metrics["expected_static_slot_pairs"]
    eligible = static_universe_pass and timestamp_gate_pass and continuous_gate_pass and sign_gate_pass and action_gate_pass
    result = {
        "feature_group": prereg["feature_group"],
        "experiment_id": prereg["experiment_id"],
        "preregistration_path": str(preregistration_path),
        "feature_group_definition_path": str(definition_path),
        "static_ladder_policy_id": POLICY_ID,
        "status": "eligible_for_later_uplift_testing" if eligible else "not_eligible_for_later_uplift_testing",
        "eligibility": bool(eligible),
        "decision": (
            "policy_parity_passed_uplift_training_allowed_only_with_separate_owner_approval"
            if eligible
            else "blocked_before_uplift_under_static_ladder_policy"
        ),
        "gate_results": {
            "static_ladder_universe": static_universe_pass,
            "timestamp_and_rows": timestamp_gate_pass,
            "continuous_feature_drift": continuous_gate_pass,
            "sign_or_bucket_match": sign_gate_pass,
            "action_flips": action_gate_pass,
        },
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
        },
        "action_replay": {
            "action_flips": action_flips,
            "unclassified_non_threshold_action_flips": action_non_threshold_flips,
        },
        "daily": daily,
        "candidate_pairs": total_pairs,
        "blockers": sorted(set(blockers)),
    }
    return result, feature_rows, mismatch_rows


def side_effect_policy(*, broker_endpoint_called: bool = False) -> dict[str, bool]:
    return {
        "model_training_executed": False,
        "threshold_selection_executed": False,
        "feature_uplift_cv_executed": False,
        "broker_endpoint_called": bool(broker_endpoint_called),
        "paper_submit_allowed": False,
        "paid_data_download": False,
        "promotion_or_default_changed": False,
        "runtime_flags_edited": False,
        "launchd_changed": False,
        "real_money_path_changed": False,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "no_rows"}])


def write_group_report(
    path: Path,
    *,
    title: str,
    result: dict[str, Any],
    feature_rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Status: `{result['status']}`",
        f"- Decision: `{result['decision']}`",
        f"- Experiment: `{result['experiment_id']}`",
        f"- Feature group: `{result['feature_group']}`",
        f"- Static-ladder policy: `{POLICY_ID}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in result["gate_results"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Daily Evidence", ""])
    for session, row in result["daily"].items():
        lines.append(
            f"- `{session}`: rows=`{row['paired_rows_in_required_window']}` "
            f"static_pairs=`{row['static_ladder_candidate_pairs']}` action_flips=`{row['action_flips']}`"
        )
    lines.extend(["", "## Feature Drift", ""])
    for row in sorted(feature_rows, key=lambda item: float(item.get("standardized_p99_abs") or 0.0), reverse=True):
        lines.append(
            f"- `{row['feature']}`: pass=`{row['pass']}` "
            f"std median/p95/p99=`{row['standardized_median_abs']}`/`{row['standardized_p95_abs']}`/`{row['standardized_p99_abs']}` "
            f"raw_missing=`{row['raw_missing_count']}` coverage=`{row['final_finite_coverage_after_imputation']}`"
        )
    if result["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in result["blockers"]:
            lines.append(f"- `{blocker}`")
    path.write_text("\n".join(lines) + "\n")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["policy_replay_metrics"]
    lines = [
        "# Protocol101 Static-Ladder Boundary-Stable Policy Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Highest allowed claim: `{summary['highest_allowed_claim']}`",
        f"- Policy: `{summary['policy_id']}`",
        f"- Contract: `{summary['contract']}`",
        f"- Transform: `{summary['required_transform']}`",
        f"- Decision rows checked: `{metrics['decision_rows_checked']}`",
        f"- Static-ladder universe match rate: `{metrics['static_ladder_universe_match_rate']}`",
        f"- Model-facing candidate count parity: `{metrics['model_facing_candidate_count_parity']}`",
        f"- Paired static slot pairs checked: `{metrics['paired_static_slot_pairs_checked']}`",
        "",
        "## Replay Evidence",
        "",
    ]
    for session, row in metrics["daily"].items():
        lines.append(
            f"- `{session}`: rows=`{row['paired_rows_in_required_window']}` "
            f"static_mismatches=`{row['static_ladder_universe_mismatch_rows']}` "
            f"guard_divergences=`{row['guard_status_divergences']}` "
            f"historical_stable=`{row['historical_boundary_stable_count']}` "
            f"ibkr_stable=`{row['ibkr_boundary_stable_count']}`"
        )
    lines.extend(
        [
            "",
            "## Guard Divergence",
            "",
            f"- Guard status divergence count: `{metrics['guard_status_divergence_count']}`",
            f"- Boundary-stable divergence count: `{metrics['boundary_stable_divergence_count']}`",
            f"- Post-filter candidate divergence count: `{metrics['post_filter_candidate_divergence_count']}`",
            f"- Historical boundary-stable candidates: `{metrics['boundary_stable_candidate_counts']['historical']}`",
            f"- IBKR boundary-stable candidates: `{metrics['boundary_stable_candidate_counts']['ibkr']}`",
            "",
            "The policy preserves quote, freshness, tradability, and affordability differences as guard/audit fields. "
            "Those fields are not promoted to model-facing alpha by this audit.",
            "",
            "## Group Results",
            "",
            f"- Group 1 stable index/context: `{summary['group_results']['group1']['status']}`",
            f"- Group 1 failed continuous features: `{summary['group_results']['group1']['continuous_feature_gate']['failed_features']}`",
            f"- Group 2 candidate geometry/moneyness: `{summary['group_results']['group2']['status']}`",
            f"- Group 2 failed continuous features: `{summary['group_results']['group2']['continuous_feature_gate']['failed_features']}`",
            "",
            "Quote/liquidity/Greek alpha remains blocked. Group 3 and Group 4 were not rerun as active pass/fail claims.",
            "",
            "## Resume Decision",
            "",
            f"- Feature recovery can resume for Group 1: `{summary['feature_recovery_resume']['group1']}`",
            f"- Feature recovery can resume for Group 2: `{summary['feature_recovery_resume']['group2']}`",
            f"- Recommended next goal: `{summary['recommended_next_goal']}`",
            "",
            "## Side Effects",
            "",
        ]
    )
    for key, value in summary["side_effect_policy"].items():
        lines.append(f"- {key}: `{str(value).lower()}`")
    if summary["blockers"]:
        lines.extend(["", "## Blockers", ""])
        for blocker in summary["blockers"]:
            lines.append(f"- `{blocker}`")
    path.write_text("\n".join(lines) + "\n")


def run_audit(
    *,
    out_dir: Path,
    policy_options: Path,
    trace_prefix: str,
    group1_preregistration: Path,
    group2_preregistration: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    margins = load_policy_margins(policy_options)
    group1_prereg = g1.load_json(group1_preregistration)
    parity = group1_prereg["parity_evidence_required_before_uplift_claim"]
    required_sessions = list(parity["required_recorder_days"])
    first_et = str(parity["required_first_decision_et"])
    last_et = str(parity["required_last_decision_et"])
    expected_rows = int(parity["paired_rows_per_full_day"])

    policy_metrics = collect_policy_metrics(
        trace_prefix=trace_prefix,
        required_sessions=required_sessions,
        first_et=first_et,
        last_et=last_et,
        expected_rows=expected_rows,
        margins=margins,
        out_dir=out_dir,
    )
    group1_result, group1_features, group1_mismatches = evaluate_group1(
        preregistration_path=group1_preregistration,
        trace_prefix=trace_prefix,
        policy_metrics=policy_metrics,
    )
    group2_result, group2_features, group2_mismatches = evaluate_group2(
        preregistration_path=group2_preregistration,
        trace_prefix=trace_prefix,
        policy_metrics=policy_metrics,
    )
    blockers = list(policy_metrics["blockers"])
    static_policy_success = (
        policy_metrics["static_ladder_universe_parity_pass"]
        and policy_metrics["model_facing_candidate_count_parity"]
        and not blockers
    )
    group1_resume = "yes" if group1_result["eligibility"] else "no"
    group2_resume = "yes" if group2_result["eligibility"] else "no"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if static_policy_success else "blocked",
        "highest_allowed_claim": "static-ladder boundary-stable policy audit complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy_id": POLICY_ID,
        "contract": CONTRACT,
        "required_transform": TRANSFORM,
        "trace_prefix": trace_prefix,
        "policy_options_path": str(policy_options),
        "policy_options_sha256": g1.sha256_path(policy_options),
        "frozen_margins": margins,
        "policy_replay_metrics": policy_metrics,
        "group_results": {
            "group1": group1_result,
            "group2": group2_result,
        },
        "feature_recovery_resume": {
            "group1": group1_resume,
            "group2": group2_resume,
            "quote_liquidity_greek_alpha": "blocked",
        },
        "quote_liquidity_greek_alpha_remains_blocked": True,
        "recommended_next_goal": (
            "Run preregistered uplift testing for the eligible non-quote feature group(s), "
            "without quote/liquidity/Greek alpha."
        )
        if group1_result["eligibility"] or group2_result["eligibility"]
        else "Write a blocked packet before any uplift testing.",
        "blockers": sorted(set(blockers)),
        "side_effect_policy": side_effect_policy(
            broker_endpoint_called=bool(policy_metrics["side_effect_policy"]["broker_endpoint_called"])
        ),
    }

    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "policy_replay_metrics.json", policy_metrics)
    write_csv(out_dir / "group1_feature_drift_summary.csv", group1_features)
    write_csv(out_dir / "group1_sign_bucket_mismatches.csv", group1_mismatches)
    write_csv(out_dir / "group2_feature_drift_summary.csv", group2_features)
    write_csv(out_dir / "group2_sign_bucket_mismatches.csv", group2_mismatches)
    write_group_report(
        out_dir / "group1_parity_report.md",
        title="Protocol101 Static-Ladder Policy Group 1 Parity Report",
        result=group1_result,
        feature_rows=group1_features,
    )
    write_group_report(
        out_dir / "group2_parity_report.md",
        title="Protocol101 Static-Ladder Policy Group 2 Parity Report",
        result=group2_result,
        feature_rows=group2_features,
    )
    write_report(out_dir / "report.md", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = run_audit(
        out_dir=args.out_dir,
        policy_options=args.policy_options,
        trace_prefix=str(args.trace_prefix),
        group1_preregistration=args.group1_preregistration,
        group2_preregistration=args.group2_preregistration,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "static_ladder_universe_parity_pass": summary["policy_replay_metrics"][
                    "static_ladder_universe_parity_pass"
                ],
                "group1": summary["group_results"]["group1"]["status"],
                "group2": summary["group_results"]["group2"]["status"],
                "report": str(args.out_dir / "report.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
