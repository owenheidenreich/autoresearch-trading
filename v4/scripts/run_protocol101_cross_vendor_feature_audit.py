"""Field-level audit for Protocol101 IBKR-capture versus historical traces."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any
from zoneinfo import ZoneInfo


MARKET_FEATURE_NAMES = (
    "spx_close",
    "vix_close",
    "spx_vwap",
    "omar",
    "session_range",
    "momentum_5m",
    "momentum_15m",
)
STRUCTURE_FEATURE_NAMES = (
    "true_spx_close",
    "true_vix_close",
    "sigma_pos",
    "abs_sigma_pos",
    "spx_vwap_est",
    "vwap_dist_pct",
    "vwap_slope_5m_pct",
    "omar_high",
    "omar_low",
    "omar_mid",
    "omar_range",
    "omar_range_pct",
    "omar_mid_pos_units",
    "omar_retest_dist_norm",
    "first15_available",
    "first15_range_pct",
    "first15_close_position",
    "first15_acceptance",
    "inside_first15",
    "opening_gap_pct",
    "last10_range_over_omar",
    "last10_break_state",
    "atr15_pct",
    "minute_fraction",
    "bucket_first_30",
    "bucket_post_open_morning",
    "bucket_midday",
    "bucket_late_afternoon",
)
OPTION_FEATURE_NAMES = (
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_frac",
    "bid_size",
    "ask_size",
    "option_ohlcv_volume",
    "stat_open_interest",
    "iv",
    "delta",
    "gamma",
    "theta",
    "distance_points",
    "breakeven_distance",
)
SIDE_FEATURE_NAMES = ("is_call", "is_put")
SHAPE_FEATURE_NAMES = ("offset_norm", "abs_offset_norm")
ENVIRONMENT_PRIOR_FEATURE_NAMES = (
    "vwap_gap_over_range",
    "range_pct",
    "above_vwap",
    "below_vwap",
    "omar_pos",
    "omar_neg",
    "mom5_pos",
    "mom5_neg",
    "mom15_pos",
    "mom15_neg",
    "vwap_trend_aligned",
    "vwap_mean_reversion_side",
    "omar_aligned",
    "omar_counter",
    "momentum15_aligned",
    "momentum15_counter",
)
TIME_FEATURE_NAMES = (
    "session_progress",
    "session_progress_remaining",
    "session_progress_sin",
    "session_progress_cos",
    "bucket_first_30",
    "bucket_post_open_morning",
    "bucket_midday",
    "bucket_late_afternoon",
)
NY = ZoneInfo("America/New_York")
ALLOWED_ENTRY_BUCKETS = frozenset({"post_open_morning", "late_afternoon"})
DIAGNOSTIC_EDGE_THRESHOLDS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-traces", type=Path, required=True)
    parser.add_argument("--historical-traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--session",
        default=None,
        help="Optional single-session filter. Omit to audit every session present in both trace files.",
    )
    parser.add_argument("--minimum-candidate-overlap", type=float, default=0.80)
    parser.add_argument(
        "--surface-standardizer",
        type=Path,
        help="Optional frozen surface standardizer used to report drift in model-standard-deviation units.",
    )
    return parser.parse_args()


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("payload")
    return nested if row.get("schema_version") == "Protocol101DecisionTraceV1" and isinstance(nested, dict) else row


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        out = datetime.fromisoformat(text)
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _key(row: dict[str, Any]) -> str | None:
    ts = _timestamp(row.get("decision_ts") or row.get("decision_time") or row.get("timestamp"))
    if ts is None:
        return None
    session = str(row.get("session") or ts.date().isoformat())
    return f"{session}|{ts.replace(second=0, microsecond=0).isoformat()}"


def _load(path: Path, session: str | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = _payload(json.loads(line))
            if session is not None and str(row.get("session")) != session:
                continue
            key = _key(row)
            if key:
                out[key] = row
    return out


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _universe(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in row.get("candidate_universe") or []:
        if isinstance(item, dict) and item.get("contract_id"):
            contract_id = str(item["contract_id"])
            out[contract_id] = dict(item)
            score = _finite(item.get("edge", item.get("score")))
            if score is not None:
                out[contract_id]["edge"] = score

    features = row.get("features") or {}
    for item in features.get("token_features") or []:
        if not isinstance(item, dict) or not item.get("contract_id"):
            continue
        contract_id = str(item["contract_id"])
        candidate = out.setdefault(contract_id, {"contract_id": contract_id})
        token_features = item.get("token_features", item.get("features"))
        if token_features is not None:
            candidate["token_features"] = token_features
        if item.get("feature_hash") is not None:
            candidate["feature_hash"] = item.get("feature_hash")

    model_scores = row.get("model_scores") or {}
    for item in model_scores.get("candidate_scores") or []:
        if not isinstance(item, dict) or not item.get("contract_id"):
            continue
        contract_id = str(item["contract_id"])
        candidate = out.setdefault(contract_id, {"contract_id": contract_id})
        score = _finite(item.get("edge", item.get("score")))
        if score is not None:
            candidate["edge"] = score
    return out


def _scalar_names(width: int) -> list[str]:
    names = []
    for prefix in ("market_last", "market_mean", "market_std", "market_delta"):
        names.extend(f"{prefix}.{name}" for name in MARKET_FEATURE_NAMES)
    names.extend(f"structure.{name}" for name in STRUCTURE_FEATURE_NAMES)
    names.extend(f"scalar_{index}" for index in range(len(names), width))
    return names[:width]


def _token_names(width: int) -> list[str]:
    names = list(OPTION_FEATURE_NAMES)
    for prefix in ("market_last", "market_mean", "market_std", "market_delta"):
        names.extend(f"{prefix}.{name}" for name in MARKET_FEATURE_NAMES)
    names.extend(f"side.{name}" for name in SIDE_FEATURE_NAMES)
    names.extend(f"shape.{name}" for name in SHAPE_FEATURE_NAMES)
    names.extend(f"environment.{name}" for name in ENVIRONMENT_PRIOR_FEATURE_NAMES)
    names.extend(f"time.{name}" for name in TIME_FEATURE_NAMES)
    names.extend(f"derived_token_{index}" for index in range(len(names), width))
    return names[:width]


def _summary(values: list[float]) -> dict[str, Any]:
    ordered_abs = sorted(abs(value) for value in values)
    if not values:
        return {"n": 0}
    rank = int(round((len(ordered_abs) - 1) * 0.95))
    return {
        "n": len(values),
        "signed_mean": mean(values),
        "signed_median": median(values),
        "mean_abs": mean(ordered_abs),
        "median_abs": median(ordered_abs),
        "p95_abs": ordered_abs[rank],
        "max_abs": ordered_abs[-1],
    }


def _time_bucket(value: str) -> str:
    ts = _timestamp(value)
    if ts is None:
        return "unknown"
    local = ts.astimezone(NY)
    minutes = local.hour * 60 + local.minute
    if minutes < 10 * 60:
        return "first_30"
    if minutes < 11 * 60 + 30:
        return "post_open_morning"
    if minutes < 13 * 60 + 30:
        return "midday"
    return "late_afternoon"


def _local_minute(value: str) -> str | None:
    ts = _timestamp(value)
    return None if ts is None else ts.astimezone(NY).strftime("%H:%M")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = int(round((len(ordered) - 1) * float(quantile)))
    return ordered[rank]


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    denominator = left_scale * right_scale
    return None if denominator <= 0 else numerator / denominator


def _rank_correlation(
    live_universe: dict[str, dict[str, Any]],
    historical_universe: dict[str, dict[str, Any]],
) -> float | None:
    common = set(live_universe) & set(historical_universe)
    if len(common) < 2:
        return None
    live_order = sorted(common, key=lambda contract_id: float(live_universe[contract_id].get("edge") or -math.inf), reverse=True)
    historical_order = sorted(
        common,
        key=lambda contract_id: float(historical_universe[contract_id].get("edge") or -math.inf),
        reverse=True,
    )
    live_rank = {contract_id: rank for rank, contract_id in enumerate(live_order)}
    historical_rank = {contract_id: rank for rank, contract_id in enumerate(historical_order)}
    ordered_ids = sorted(common)
    return _pearson(
        [float(live_rank[contract_id]) for contract_id in ordered_ids],
        [float(historical_rank[contract_id]) for contract_id in ordered_ids],
    )


def _top_contract(universe: dict[str, dict[str, Any]]) -> tuple[str | None, float | None]:
    eligible = [
        (contract_id, _finite(item.get("edge")))
        for contract_id, item in universe.items()
    ]
    eligible = [(contract_id, edge) for contract_id, edge in eligible if edge is not None]
    if not eligible:
        return None, None
    contract_id, edge = max(eligible, key=lambda item: float(item[1]))
    return contract_id, edge


def _top_k_ids(universe: dict[str, dict[str, Any]], count: int) -> set[str]:
    ordered = sorted(
        universe,
        key=lambda contract_id: float(_finite(universe[contract_id].get("edge")) or -math.inf),
        reverse=True,
    )
    return set(ordered[: int(count)])


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _load_standardizer_stds(path: Path | None) -> tuple[list[float], list[float]]:
    if path is None:
        return [], []
    payload = json.loads(path.read_text())
    return (
        [max(float(value), 1e-12) for value in payload["scalar"]["std"]],
        [max(float(value), 1e-12) for value in payload["token"]["std"]],
    )


def _standardized_drift_details(
    left: list[Any],
    right: list[Any],
    stds: list[float],
    names: list[str],
) -> tuple[float | None, list[dict[str, Any]]]:
    values: list[tuple[float, str, float]] = []
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        left_value = _finite(left_item)
        right_value = _finite(right_item)
        if left_value is None or right_value is None or index >= len(stds):
            continue
        raw_delta = left_value - right_value
        values.append((abs(raw_delta) / stds[index], names[index], raw_delta))
    if not values:
        return None, []
    rms = math.sqrt(mean([item[0] ** 2 for item in values]))
    details = [
        {"feature": name, "standardized_abs": magnitude, "raw_delta": raw_delta}
        for magnitude, name, raw_delta in sorted(values, reverse=True)[:5]
    ]
    return rms, details


def _bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"minutes": 0}
    edge_deltas = [float(row["max_edge_delta"]) for row in rows if row.get("max_edge_delta") is not None]
    rank_correlations = [
        float(row["candidate_rank_correlation"])
        for row in rows
        if row.get("candidate_rank_correlation") is not None
    ]
    return {
        "minutes": len(rows),
        "max_edge_delta": _summary(edge_deltas),
        "top_contract_match_rate": mean([bool(row["top_contract_match"]) for row in rows]),
        "top3_overlap_median": median([float(row["top3_identity_overlap"]) for row in rows]),
        "candidate_rank_correlation_median": median(rank_correlations) if rank_correlations else None,
        "candidate_exact_rate": mean([float(row["candidate_identity_overlap"]) == 1.0 for row in rows]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "no_rows"}])


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    live = _load(args.live_traces, args.session)
    historical = _load(args.historical_traces, args.session)
    scalar_stds, token_stds = _load_standardizer_stds(args.surface_standardizer)
    keys = sorted(set(live) & set(historical))
    minute_rows: list[dict[str, Any]] = []
    scalar_deltas: dict[int, list[float]] = {}
    token_deltas: dict[int, list[float]] = {}
    scalar_standardized_deltas: dict[int, list[float]] = {}
    token_standardized_deltas: dict[int, list[float]] = {}
    common_candidate_edge_deltas: list[float] = []
    quote_lags: dict[str, list[float]] = {"live": [], "historical": []}

    for key in keys:
        live_row = live[key]
        historical_row = historical[key]
        decision_key = key.split("|", 1)[-1]
        live_universe = _universe(live_row)
        historical_universe = _universe(historical_row)
        live_ids = set(live_universe)
        historical_ids = set(historical_universe)
        union = live_ids | historical_ids
        overlap = 1.0 if not union else len(live_ids & historical_ids) / len(union)
        live_features = live_row.get("features") or {}
        historical_features = historical_row.get("features") or {}
        live_scalar = live_features.get("scalar_features") or []
        historical_scalar = historical_features.get("scalar_features") or []
        for index, (left, right) in enumerate(zip(live_scalar, historical_scalar)):
            left_value = _finite(left)
            right_value = _finite(right)
            if left_value is not None and right_value is not None:
                delta = left_value - right_value
                scalar_deltas.setdefault(index, []).append(delta)
                if index < len(scalar_stds):
                    scalar_standardized_deltas.setdefault(index, []).append(delta / scalar_stds[index])
        for contract_id in live_ids & historical_ids:
            live_token = live_universe[contract_id].get("token_features") or []
            historical_token = historical_universe[contract_id].get("token_features") or []
            for index, (left, right) in enumerate(zip(live_token, historical_token)):
                left_value = _finite(left)
                right_value = _finite(right)
                if left_value is not None and right_value is not None:
                    delta = left_value - right_value
                    token_deltas.setdefault(index, []).append(delta)
                    if index < len(token_stds):
                        token_standardized_deltas.setdefault(index, []).append(delta / token_stds[index])

            live_edge = _finite(live_universe[contract_id].get("edge"))
            historical_edge = _finite(historical_universe[contract_id].get("edge"))
            if live_edge is not None and historical_edge is not None:
                common_candidate_edge_deltas.append(live_edge - historical_edge)

        live_scores = (live_row.get("model_scores") or {}).get("surface_scores") or []
        historical_scores = (historical_row.get("model_scores") or {}).get("surface_scores") or []
        live_edges = [_finite(item.get("edge")) for item in live_universe.values()]
        historical_edges = [_finite(item.get("edge")) for item in historical_universe.values()]
        live_edges = [value for value in live_edges if value is not None]
        historical_edges = [value for value in historical_edges if value is not None]
        live_top_contract, live_top_edge = _top_contract(live_universe)
        historical_top_contract, historical_top_edge = _top_contract(historical_universe)
        top3_overlap = _jaccard(_top_k_ids(live_universe, 3), _top_k_ids(historical_universe, 3))
        rank_correlation = _rank_correlation(live_universe, historical_universe)
        common_edge_deltas = []
        for contract_id in live_ids & historical_ids:
            live_edge = _finite(live_universe[contract_id].get("edge"))
            historical_edge = _finite(historical_universe[contract_id].get("edge"))
            if live_edge is not None and historical_edge is not None:
                common_edge_deltas.append(live_edge - historical_edge)
        scalar_rms, scalar_top_drift = _standardized_drift_details(
            list(live_scalar),
            list(historical_scalar),
            scalar_stds,
            _scalar_names(min(len(live_scalar), len(historical_scalar))),
        )
        token_reference_contract = live_top_contract if live_top_contract in historical_ids else None
        if token_reference_contract is None and live_ids & historical_ids:
            token_reference_contract = max(
                live_ids & historical_ids,
                key=lambda contract_id: float(_finite(live_universe[contract_id].get("edge")) or -math.inf),
            )
        token_rms = None
        token_top_drift: list[dict[str, Any]] = []
        if token_reference_contract is not None:
            live_token = live_universe[token_reference_contract].get("token_features") or []
            historical_token = historical_universe[token_reference_contract].get("token_features") or []
            token_rms, token_top_drift = _standardized_drift_details(
                list(live_token),
                list(historical_token),
                token_stds,
                _token_names(min(len(live_token), len(historical_token))),
            )
        bucket = _time_bucket(decision_key)
        minute_rows.append(
            {
                "session": str(live_row.get("session") or historical_row.get("session") or ""),
                "decision_minute_utc": decision_key,
                "decision_minute_et": _local_minute(decision_key),
                "time_bucket": bucket,
                "entry_bucket_allowed": bucket in ALLOWED_ENTRY_BUCKETS,
                "live_action": live_row.get("selected_action"),
                "historical_action": historical_row.get("selected_action"),
                "action_match": live_row.get("selected_action") == historical_row.get("selected_action"),
                "live_candidate_count": len(live_ids),
                "historical_candidate_count": len(historical_ids),
                "candidate_identity_overlap": overlap,
                "live_only_candidates": len(live_ids - historical_ids),
                "historical_only_candidates": len(historical_ids - live_ids),
                "live_flat_score": _finite(live_scores[0]) if live_scores else None,
                "historical_flat_score": _finite(historical_scores[0]) if historical_scores else None,
                "flat_score_delta": (
                    _finite(live_scores[0]) - _finite(historical_scores[0])
                    if live_scores
                    and historical_scores
                    and _finite(live_scores[0]) is not None
                    and _finite(historical_scores[0]) is not None
                    else None
                ),
                "live_max_edge": live_top_edge,
                "historical_max_edge": historical_top_edge,
                "max_edge_delta": (
                    live_top_edge - historical_top_edge
                    if live_top_edge is not None and historical_top_edge is not None
                    else None
                ),
                "live_top_contract_id": live_top_contract,
                "historical_top_contract_id": historical_top_contract,
                "top_contract_match": live_top_contract == historical_top_contract,
                "top3_identity_overlap": top3_overlap,
                "candidate_rank_correlation": rank_correlation,
                "common_candidate_edge_delta_mean_abs": (
                    mean([abs(value) for value in common_edge_deltas]) if common_edge_deltas else None
                ),
                "common_candidate_edge_delta_p95_abs": (
                    _percentile([abs(value) for value in common_edge_deltas], 0.95)
                    if common_edge_deltas
                    else None
                ),
                "common_candidate_edge_delta_max_abs": (
                    max(abs(value) for value in common_edge_deltas) if common_edge_deltas else None
                ),
                "scalar_standardized_rms_drift": scalar_rms,
                "token_standardized_rms_drift": token_rms,
                "token_drift_reference_contract_id": token_reference_contract,
                "largest_scalar_standardized_drifts": json.dumps(scalar_top_drift, sort_keys=True),
                "largest_token_standardized_drifts": json.dumps(token_top_drift, sort_keys=True),
            }
        )
        for name, row in (("live", live_row), ("historical", historical_row)):
            decision_ts = _timestamp(row.get("decision_ts"))
            quote_ts = _timestamp(row.get("source_quote_ts"))
            if decision_ts and quote_ts:
                quote_lags[name].append((decision_ts - quote_ts).total_seconds() * 1000.0)

    scalar_width = max(scalar_deltas, default=-1) + 1
    token_width = max(token_deltas, default=-1) + 1
    scalar_names = _scalar_names(scalar_width)
    token_names = _token_names(token_width)
    feature_rows = []
    for family, deltas, standardized_deltas, names in (
        ("scalar", scalar_deltas, scalar_standardized_deltas, scalar_names),
        ("token", token_deltas, token_standardized_deltas, token_names),
    ):
        for index, values in sorted(deltas.items()):
            scaled_summary = _summary(standardized_deltas.get(index, []))
            feature_rows.append(
                {
                    "family": family,
                    "index": index,
                    "feature": names[index],
                    **_summary(values),
                    "standardized_mean_abs": scaled_summary.get("mean_abs"),
                    "standardized_median_abs": scaled_summary.get("median_abs"),
                    "standardized_p95_abs": scaled_summary.get("p95_abs"),
                    "standardized_max_abs": scaled_summary.get("max_abs"),
                }
            )

    overlaps = [float(row["candidate_identity_overlap"]) for row in minute_rows]
    flat_score_deltas = [
        float(row["live_flat_score"]) - float(row["historical_flat_score"])
        for row in minute_rows
        if row.get("live_flat_score") is not None and row.get("historical_flat_score") is not None
    ]
    max_edge_deltas = [
        float(row["live_max_edge"]) - float(row["historical_max_edge"])
        for row in minute_rows
        if row.get("live_max_edge") is not None and row.get("historical_max_edge") is not None
    ]
    action_mismatches = sum(not bool(row["action_match"]) for row in minute_rows)
    low_overlap = sum(value < args.minimum_candidate_overlap for value in overlaps)
    allowed_rows = [row for row in minute_rows if bool(row.get("entry_bucket_allowed"))]
    threshold_disagreements = {}
    for threshold in DIAGNOSTIC_EDGE_THRESHOLDS:
        threshold_disagreements[str(threshold)] = sum(
            (float(row["live_max_edge"]) >= threshold) != (float(row["historical_max_edge"]) >= threshold)
            for row in allowed_rows
            if row.get("live_max_edge") is not None and row.get("historical_max_edge") is not None
        )
    by_time_bucket = {
        bucket: _bucket_summary([row for row in minute_rows if row.get("time_bucket") == bucket])
        for bucket in ("first_30", "post_open_morning", "midday", "late_afternoon", "unknown")
        if any(row.get("time_bucket") == bucket for row in minute_rows)
    }
    top_contract_matches = [bool(row["top_contract_match"]) for row in minute_rows]
    top3_overlaps = [float(row["top3_identity_overlap"]) for row in minute_rows]
    rank_correlations = [
        float(row["candidate_rank_correlation"])
        for row in minute_rows
        if row.get("candidate_rank_correlation") is not None
    ]
    largest_edge_drift_minutes = [
        {
            key: row.get(key)
            for key in (
                "decision_minute_et",
                "time_bucket",
                "entry_bucket_allowed",
                "live_max_edge",
                "historical_max_edge",
                "max_edge_delta",
                "live_top_contract_id",
                "historical_top_contract_id",
                "top_contract_match",
                "candidate_identity_overlap",
                "candidate_rank_correlation",
                "scalar_standardized_rms_drift",
                "token_standardized_rms_drift",
                "largest_scalar_standardized_drifts",
                "largest_token_standardized_drifts",
            )
        }
        for row in sorted(
            minute_rows,
            key=lambda item: abs(float(item.get("max_edge_delta") or 0.0)),
            reverse=True,
        )[:20]
    ]
    summary = {
        "schema_version": "Protocol101CrossVendorFeatureAuditV1",
        "session": args.session or "ALL",
        "status": "fail" if action_mismatches or low_overlap else "review",
        "paired_minutes": len(keys),
        "missing_live_minutes": len(set(historical) - set(live)),
        "missing_historical_minutes": len(set(live) - set(historical)),
        "action_mismatches": action_mismatches,
        "candidate_exact_minutes": sum(value == 1.0 for value in overlaps),
        "candidate_overlap_below_minimum_minutes": low_overlap,
        "candidate_overlap_minimum": min(overlaps) if overlaps else None,
        "candidate_overlap_mean": mean(overlaps) if overlaps else None,
        "candidate_overlap_median": median(overlaps) if overlaps else None,
        "minimum_candidate_overlap_gate": args.minimum_candidate_overlap,
        "top_contract_match_minutes": sum(top_contract_matches),
        "top_contract_match_rate": mean(top_contract_matches) if top_contract_matches else None,
        "top3_identity_overlap_median": median(top3_overlaps) if top3_overlaps else None,
        "candidate_rank_correlation": _summary(rank_correlations),
        "common_candidate_edge_delta": _summary(common_candidate_edge_deltas),
        "flat_score_delta": _summary(flat_score_deltas),
        "max_edge_delta": _summary(max_edge_deltas),
        "live_minutes_at_or_above_edge_25": sum(
            row.get("live_max_edge") is not None and float(row["live_max_edge"]) >= 25.0 for row in minute_rows
        ),
        "historical_minutes_at_or_above_edge_25": sum(
            row.get("historical_max_edge") is not None and float(row["historical_max_edge"]) >= 25.0 for row in minute_rows
        ),
        "allowed_entry_minutes": len(allowed_rows),
        "allowed_minutes_live_within_5_points_of_edge_gate": sum(
            row.get("live_max_edge") is not None and float(row["live_max_edge"]) >= 20.0 for row in allowed_rows
        ),
        "allowed_minutes_historical_within_5_points_of_edge_gate": sum(
            row.get("historical_max_edge") is not None and float(row["historical_max_edge"]) >= 20.0 for row in allowed_rows
        ),
        "diagnostic_edge_gate_disagreement_minutes": threshold_disagreements,
        "by_time_bucket": by_time_bucket,
        "largest_edge_drift_minutes": largest_edge_drift_minutes,
        "live_quote_lag_ms": _summary(quote_lags["live"]),
        "historical_quote_lag_ms": _summary(quote_lags["historical"]),
        "largest_scalar_drifts": sorted(
            (row for row in feature_rows if row["family"] == "scalar"),
            key=lambda row: float(row.get("p95_abs") or 0.0),
            reverse=True,
        )[:15],
        "largest_base_token_drifts": sorted(
            (row for row in feature_rows if row["family"] == "token" and int(row["index"]) < len(OPTION_FEATURE_NAMES)),
            key=lambda row: float(row.get("p95_abs") or 0.0),
            reverse=True,
        ),
        "largest_token_drifts": sorted(
            (row for row in feature_rows if row["family"] == "token"),
            key=lambda row: float(row.get("p95_abs") or 0.0),
            reverse=True,
        )[:25],
        "largest_standardized_scalar_drifts": sorted(
            (row for row in feature_rows if row["family"] == "scalar"),
            key=lambda row: float(row.get("standardized_p95_abs") or 0.0),
            reverse=True,
        )[:15],
        "largest_standardized_token_drifts": sorted(
            (row for row in feature_rows if row["family"] == "token"),
            key=lambda row: float(row.get("standardized_p95_abs") or 0.0),
            reverse=True,
        )[:15],
        "inputs": {
            "live_traces": str(args.live_traces),
            "historical_traces": str(args.historical_traces),
            "surface_standardizer": str(args.surface_standardizer) if args.surface_standardizer else None,
        },
    }
    _write_csv(args.out_dir / "minute_parity.csv", minute_rows)
    _write_csv(
        args.out_dir / "largest_edge_drift_minutes.csv",
        sorted(minute_rows, key=lambda item: abs(float(item.get("max_edge_delta") or 0.0)), reverse=True)[:50],
    )
    _write_csv(args.out_dir / "feature_drift_summary.csv", feature_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = [
        "# Protocol101 Cross-Vendor Feature Audit",
        "",
        f"- Session: `{args.session or 'ALL'}`",
        f"- Status: `{summary['status']}`",
        f"- Paired minutes: `{summary['paired_minutes']}`",
        f"- Action mismatches: `{action_mismatches}`",
        f"- Exact candidate sets: `{summary['candidate_exact_minutes']}`",
        f"- Candidate overlap min/median/mean: `{summary['candidate_overlap_minimum']}` / `{summary['candidate_overlap_median']}` / `{summary['candidate_overlap_mean']}`",
        f"- Candidate overlap below `{args.minimum_candidate_overlap}`: `{low_overlap}`",
        f"- Top-contract match rate: `{summary['top_contract_match_rate']}`",
        f"- Top-3 identity overlap median: `{summary['top3_identity_overlap_median']}`",
        f"- Candidate-rank correlation median: `{summary['candidate_rank_correlation'].get('median_abs')}`",
        f"- Flat-score delta median/p95 abs: `{summary['flat_score_delta'].get('median_abs')}` / `{summary['flat_score_delta'].get('p95_abs')}`",
        f"- Max-edge delta median/p95 abs: `{summary['max_edge_delta'].get('median_abs')}` / `{summary['max_edge_delta'].get('p95_abs')}`",
        f"- Frozen edge-25 disagreement minutes in allowed entry buckets: `{summary['diagnostic_edge_gate_disagreement_minutes']['25.0']}`",
        f"- Allowed minutes within five edge points of gate, live/historical: `{summary['allowed_minutes_live_within_5_points_of_edge_gate']}` / `{summary['allowed_minutes_historical_within_5_points_of_edge_gate']}`",
        "",
        "## Interpretation",
        "",
        "Matching wait actions are coarse evidence only. Tail score drift, top-contract changes, and candidate-rank drift remain parity risks even when neither stream reaches the frozen entry gate.",
        "The diagnostic threshold sweep measures sensitivity only; it does not change or tune the frozen edge threshold.",
        "",
        "## Time Buckets",
        "",
    ]
    for bucket, payload in summary["by_time_bucket"].items():
        edge = payload.get("max_edge_delta") or {}
        report.append(
            f"- `{bucket}`: minutes=`{payload.get('minutes')}` top-contract-match=`{payload.get('top_contract_match_rate')}` "
            f"edge median/p95/max abs=`{edge.get('median_abs')}` / `{edge.get('p95_abs')}` / `{edge.get('max_abs')}`"
        )
    report.extend(["", "## Largest Edge Drifts", ""])
    for row in summary["largest_edge_drift_minutes"][:10]:
        report.append(
            f"- `{row.get('decision_minute_et')} ET` bucket=`{row.get('time_bucket')}` "
            f"live/historical=`{row.get('live_max_edge')}` / `{row.get('historical_max_edge')}` "
            f"delta=`{row.get('max_edge_delta')}` top-match=`{row.get('top_contract_match')}`"
        )
    report.extend(["", "## Standardized Feature Drifts", ""])
    if args.surface_standardizer:
        report.extend(
            [
                "Model-standardized drift ranks differences by the scale presented to the frozen surface model.",
                "",
            ]
        )
        for row in summary["largest_standardized_scalar_drifts"][:10]:
            report.append(
                f"- scalar `{row['feature']}`: p95=`{row.get('standardized_p95_abs')}` "
                f"max=`{row.get('standardized_max_abs')}` model standard deviations"
            )
        for row in summary["largest_standardized_token_drifts"][:10]:
            report.append(
                f"- token `{row['feature']}`: p95=`{row.get('standardized_p95_abs')}` "
                f"max=`{row.get('standardized_max_abs')}` model standard deviations"
            )
    else:
        report.append("Not computed; pass `--surface-standardizer` to measure drift in frozen-model scale units.")
    report.extend([
        "",
        "## Largest Scalar Drifts",
        "",
    ])
    for row in summary["largest_scalar_drifts"][:10]:
        report.append(f"- `{row['feature']}`: median_abs=`{row['median_abs']}` p95_abs=`{row['p95_abs']}`")
    report.extend(["", "## Token Drifts", ""])
    for row in summary["largest_token_drifts"][:15]:
        report.append(f"- `{row['feature']}`: median_abs=`{row['median_abs']}` p95_abs=`{row['p95_abs']}`")
    report.extend(["", "## Option Token Drifts", ""])
    for row in summary["largest_base_token_drifts"]:
        report.append(f"- `{row['feature']}`: median_abs=`{row['median_abs']}` p95_abs=`{row['p95_abs']}`")
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
