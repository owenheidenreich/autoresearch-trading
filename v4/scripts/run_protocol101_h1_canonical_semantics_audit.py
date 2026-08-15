"""Audit H1: canonical semantics as the remaining Protocol101 parity repair path.

This is an offline evidence reducer. It reads existing Q1 comparison artifacts
and classifies the historical degradation into repair routes. It does not train,
tune thresholds, contact vendors, touch launchd, or call broker paths.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.protocol101_synchronization import Protocol101CanonicalSemanticsAuditV1


DEFAULT_TRADE_ATTRIBUTION = Path(
    "v4/audit/autoresearch/protocol101_q1_2026_trade_pnl_attribution"
)
DEFAULT_Q1_ROOT = Path("v4/audit/autoresearch/protocol101_q1_2026_contract_comparison")
DEFAULT_SYNC_ROOT = Path("v4/audit/autoresearch/protocol101_synchronization_resolution")
DEFAULT_OUT = Path("v4/audit/autoresearch/protocol101_h1_canonical_semantics_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-attribution-dir", type=Path, default=DEFAULT_TRADE_ATTRIBUTION)
    parser.add_argument("--q1-root", type=Path, default=DEFAULT_Q1_ROOT)
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if pd.notna(number) else default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_h1_feature_route(feature: str) -> dict[str, str]:
    name = str(feature or "").lower()
    if name.startswith("pattern_") or "pattern" in name:
        return {
            "repair_route": "canonical_pattern_semantics",
            "causal_status": "causal_if_built_from_completed_minutes_only",
            "offline_action": "audit_pattern_token_generation_against_canonical_minutes",
            "july6_dependency": "none_for_formula_audit",
            "decision": (
                "Do not tune thresholds. Verify that every pattern token is derived "
                "from completed 09:30+ session minutes under the shared contract."
            ),
        }
    if name.startswith("structure."):
        return {
            "repair_route": "canonical_structure_semantics",
            "causal_status": "causal_if_built_from_completed_minutes_only",
            "offline_action": "keep_same_input_structure_fixture_and_audit_window_boundaries",
            "july6_dependency": "future_capture_confirms_distribution_not_formula",
            "decision": (
                "Implementation matches for identical canonical minutes; remaining "
                "drift should be treated as raw input/window policy drift."
            ),
        }
    if "open_interest" in name or name == "stat_open_interest":
        return {
            "repair_route": "runtime_zeroed_pressure_diagnostics",
            "causal_status": "causal_only_if_publication_timestamp_and_live_semantics_match",
            "offline_action": "do_not_restore_as_databento_field_without_live_equivalent",
            "july6_dependency": "capture_ibkr_generic_tick_101_for_diagnostics",
            "decision": (
                "High raw drift is expected because live-v1 intentionally zeroes OI. "
                "IBKR tick 101 can be tested, but it is not automatically equivalent "
                "to Databento statistics."
            ),
        }
    if "volume" in name:
        return {
            "repair_route": "runtime_zeroed_pressure_diagnostics",
            "causal_status": "vendor_sensitive_daily_or_interval_semantics",
            "offline_action": "do_not_restore_minute_ohlcv_without_live_equivalent",
            "july6_dependency": "capture_ibkr_generic_tick_100_for_diagnostics",
            "decision": (
                "High raw drift is expected because live-v1 intentionally zeroes "
                "option OHLCV volume. Test as diagnostic evidence, not as a direct "
                "replacement."
            ),
        }
    if "distance_points" in name or "offset" in name:
        return {
            "repair_route": "canonical_candidate_geometry",
            "causal_status": "causal_if_underlying_source_and_atm_rounding_policy_match",
            "offline_action": "audit_atm_strike_rounding_and_underlying_source_policy",
            "july6_dependency": "future_capture_confirms_spx_source_distribution",
            "decision": (
                "A same contract can be scored with different ATM-relative geometry. "
                "Audit SPX source, context lag, and ATM rounding before treating this "
                "as model degradation."
            ),
        }
    if name.startswith("market_") or "spx" in name or "vix" in name or "omar" in name:
        return {
            "repair_route": "canonical_index_context_semantics",
            "causal_status": "causal_if_timestamp_policy_matches",
            "offline_action": "audit_completed_minute_index_window_and_source_policy",
            "july6_dependency": "future_capture_confirms_live_input_distribution",
            "decision": (
                "Same-input market-window parity passes, so remaining drift is a "
                "source/timestamp/data-policy question rather than a formula mismatch."
            ),
        }
    if "spread" in name or "overpay" in name or "worth" in name or name in {
        "bid",
        "ask",
        "mid",
        "bid_size",
        "ask_size",
    }:
        return {
            "repair_route": "quote_tradability_guardrail",
            "causal_status": "causal_but_vendor_sensitive",
            "offline_action": "compare_quote_freshness_spread_and_mask_rules",
            "july6_dependency": "future_capture_confirms_stability_across_sessions",
            "decision": (
                "Keep as a guardrail. Do not loosen spread/freshness just to recreate "
                "historical trades."
            ),
        }
    if any(part in name for part in ("gamma", "theta", "delta", "iv", "vega", "breakeven", "premium")):
        return {
            "repair_route": "greeks_decay_quote_path",
            "causal_status": "causal_through_shared_repair_when_inputs_exist",
            "offline_action": "validate_shared_greek_repair_and_input_price_timestamps",
            "july6_dependency": "future_capture_confirms_quote_path_distribution",
            "decision": (
                "Use repaired internal Greeks for inference; keep vendor Greeks as "
                "diagnostics only."
            ),
        }
    return {
        "repair_route": "manual_review",
        "causal_status": "unknown_until_field_review",
        "offline_action": "classify_before_restoring_to_runtime_contract",
        "july6_dependency": "unknown",
        "decision": "Do not restore or remove this field without a field-level causal review.",
    }


def dominant_group_route(group: str) -> str:
    name = str(group or "").lower()
    if name in {"tokens", "pattern_context", "discrete_break_pattern", "omar_patterns"}:
        return "canonical_pattern_semantics"
    if name in {"scalar_market_window", "scalar"}:
        return "canonical_index_context_semantics"
    if name in {"scalar_structure", "scalar_omar_structure"}:
        return "canonical_structure_semantics"
    if name in {"option_volume", "open_interest"}:
        return "runtime_zeroed_pressure_diagnostics"
    if name == "quote_spread":
        return "quote_tradability_guardrail"
    if name == "greeks_decay":
        return "greeks_decay_quote_path"
    return "manual_review"


def scenario_closures(score_summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    scenarios = score_summary.get("scenarios") or {}
    route_map = {
        "canonical_pattern_semantics": [
            "live_with_historical_tokens",
            "live_with_historical_pattern_context",
            "live_with_historical_discrete_break_pattern",
            "live_with_historical_omar_patterns",
        ],
        "canonical_index_context_semantics": [
            "live_with_historical_scalar",
            "live_with_historical_scalar_market_window",
        ],
        "canonical_structure_semantics": [
            "live_with_historical_scalar_structure",
            "live_with_historical_scalar_omar_structure",
        ],
        "canonical_candidate_geometry": [
            "live_with_historical_scalar_market_window",
            "live_with_historical_scalar",
        ],
        "runtime_zeroed_pressure_diagnostics": [
            "live_with_historical_open_interest",
            "live_with_historical_option_volume",
        ],
        "quote_tradability_guardrail": ["live_with_historical_quote_spread"],
        "greeks_decay_quote_path": ["live_with_historical_greeks_decay"],
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for route, names in route_map.items():
        rows = []
        for name in names:
            closure = (scenarios.get(name) or {}).get("historical_gap_closure") or {}
            rows.append(
                {
                    "scenario": name,
                    "signed_mean": closure.get("signed_mean"),
                    "mean_abs": closure.get("mean_abs"),
                    "p95_abs": closure.get("p95_abs"),
                }
            )
        out[route] = rows
    return out


def route_priority(route: str, signed_mean: float, exposure: float) -> int:
    if route == "canonical_pattern_semantics":
        return 1
    if route == "canonical_index_context_semantics":
        return 2
    if route == "canonical_structure_semantics":
        return 3
    if route == "canonical_candidate_geometry":
        return 3
    if route == "runtime_zeroed_pressure_diagnostics":
        return 4
    if exposure > 0 or signed_mean > 0:
        return 5
    return 9


def build_field_rows(field_frequency: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, raw in field_frequency.iterrows():
        feature = str(raw.get("feature") or "")
        rows.append(
            {
                **classify_h1_feature_route(feature),
                "feature": feature,
                "drift_family": str(raw.get("drift_family") or ""),
                "top_lost_occurrences": int(finite(raw.get("occurrences"))),
                "field_row_pnl_exposure": finite(raw.get("pnl")),
                "mean_standardized_abs": finite(raw.get("mean_standardized_abs")),
            }
        )
    return rows


def aggregate_routes(
    *,
    field_rows: list[dict[str, Any]],
    field_report: pd.DataFrame,
    lost_trades: pd.DataFrame,
    closures_by_route: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    route_rows: dict[str, dict[str, Any]] = {}
    for row in field_rows:
        route = str(row["repair_route"])
        bucket = route_rows.setdefault(
            route,
            {
                "repair_route": route,
                "features": set(),
                "top_lost_occurrences": 0,
                "field_frequency_pnl_exposure": 0.0,
                "field_report_rows": 0,
                "field_report_pnl_exposure": 0.0,
                "lost_trade_rows": 0,
                "lost_trade_pnl_exposure": 0.0,
                "max_mean_standardized_abs": 0.0,
                "primary_gap_closure_scenario": None,
                "primary_gap_closure_signed_mean": None,
                "primary_gap_closure_mean_abs": None,
                "recommended_status": None,
                "recommended_action": None,
            },
        )
        bucket["features"].add(str(row["feature"]))
        bucket["top_lost_occurrences"] += int(row["top_lost_occurrences"])
        bucket["field_frequency_pnl_exposure"] += float(row["field_row_pnl_exposure"])
        bucket["max_mean_standardized_abs"] = max(
            float(bucket["max_mean_standardized_abs"]),
            float(row["mean_standardized_abs"]),
        )

    for _, raw in field_report.iterrows():
        route = classify_h1_feature_route(str(raw.get("feature") or ""))["repair_route"]
        bucket = route_rows.setdefault(
            route,
            {
                "repair_route": route,
                "features": set(),
                "top_lost_occurrences": 0,
                "field_frequency_pnl_exposure": 0.0,
                "field_report_rows": 0,
                "field_report_pnl_exposure": 0.0,
                "lost_trade_rows": 0,
                "lost_trade_pnl_exposure": 0.0,
                "max_mean_standardized_abs": 0.0,
                "primary_gap_closure_scenario": None,
                "primary_gap_closure_signed_mean": None,
                "primary_gap_closure_mean_abs": None,
                "recommended_status": None,
                "recommended_action": None,
            },
        )
        bucket["field_report_rows"] += 1
        bucket["field_report_pnl_exposure"] += finite(raw.get("pnl"))

    for _, raw in lost_trades.iterrows():
        route = dominant_group_route(str(raw.get("dominant_gap_closure_group") or ""))
        bucket = route_rows.setdefault(
            route,
            {
                "repair_route": route,
                "features": set(),
                "top_lost_occurrences": 0,
                "field_frequency_pnl_exposure": 0.0,
                "field_report_rows": 0,
                "field_report_pnl_exposure": 0.0,
                "lost_trade_rows": 0,
                "lost_trade_pnl_exposure": 0.0,
                "max_mean_standardized_abs": 0.0,
                "primary_gap_closure_scenario": None,
                "primary_gap_closure_signed_mean": None,
                "primary_gap_closure_mean_abs": None,
                "recommended_status": None,
                "recommended_action": None,
            },
        )
        bucket["lost_trade_rows"] += 1
        bucket["lost_trade_pnl_exposure"] += finite(raw.get("pnl"))

    recommendations = {
        "canonical_pattern_semantics": (
            "test_now_offline",
            "Audit pattern token generation and completed-minute semantics, then rerun Q1 without tuning.",
        ),
        "canonical_index_context_semantics": (
            "test_now_offline",
            "Keep the same-input market-window fixture and audit source/timestamp policy on high-loss examples.",
        ),
        "canonical_structure_semantics": (
            "regression_guard_present",
            "Structure formulas match on identical inputs; investigate real input/window boundaries, not formula drift.",
        ),
        "canonical_candidate_geometry": (
            "test_now_offline",
            "Audit ATM strike rounding and SPX context source where the same contract has different distance_points.",
        ),
        "runtime_zeroed_pressure_diagnostics": (
            "collect_july6_diagnostics",
            "Capture IBKR ticks 100/101, but do not restore Databento volume/OI fields without causal equivalence.",
        ),
        "quote_tradability_guardrail": (
            "guardrail_compare",
            "Compare masks and freshness; do not loosen tradability rules to force historical trades.",
        ),
        "greeks_decay_quote_path": (
            "regression_guard_present",
            "Retain shared repaired Greeks; validate input quotes and timestamps.",
        ),
        "manual_review": (
            "manual_review_required",
            "Classify individual fields before changing the runtime contract.",
        ),
    }

    routes = []
    for route, bucket in route_rows.items():
        closures = closures_by_route.get(route, [])
        known_closures = [
            item for item in closures if item.get("signed_mean") is not None
        ]
        primary = max(
            known_closures,
            key=lambda item: float(item.get("signed_mean") or 0.0),
            default={},
        )
        status, action = recommendations.get(route, recommendations["manual_review"])
        signed_mean = finite(primary.get("signed_mean"))
        exposure = float(bucket["lost_trade_pnl_exposure"]) or float(
            bucket["field_frequency_pnl_exposure"]
        )
        bucket["features"] = ",".join(sorted(str(item) for item in bucket["features"]))
        bucket["primary_gap_closure_scenario"] = primary.get("scenario")
        bucket["primary_gap_closure_signed_mean"] = primary.get("signed_mean")
        bucket["primary_gap_closure_mean_abs"] = primary.get("mean_abs")
        bucket["recommended_status"] = status
        bucket["recommended_action"] = action
        bucket["priority"] = route_priority(route, signed_mean, exposure)
        routes.append(bucket)
    return sorted(
        routes,
        key=lambda row: (
            int(row["priority"]),
            -float(row["lost_trade_pnl_exposure"]),
            -float(row["field_frequency_pnl_exposure"]),
        ),
    )


def top_examples(field_report: pd.DataFrame, limit_per_route: int = 5) -> list[dict[str, Any]]:
    rows = []
    working = field_report.copy()
    working["_abs_pnl"] = working["pnl"].map(lambda value: abs(finite(value)))
    for route, group in working.groupby(
        working["feature"].map(lambda value: classify_h1_feature_route(str(value))["repair_route"])
    ):
        for _, raw in group.sort_values("_abs_pnl", ascending=False).head(limit_per_route).iterrows():
            rows.append(
                {
                    "repair_route": route,
                    "session": raw.get("session"),
                    "decision_time": raw.get("decision_time"),
                    "contract_id": raw.get("contract_id"),
                    "right": raw.get("right"),
                    "pnl": finite(raw.get("pnl")),
                    "feature": raw.get("feature"),
                    "dominant_gap_closure_group": raw.get("dominant_gap_closure_group"),
                    "dominant_gap_closure_points": finite(raw.get("dominant_gap_closure_points")),
                    "legacy_max_edge": finite(raw.get("legacy_max_edge")),
                    "live_max_edge": finite(raw.get("live_max_edge")),
                    "standardized_abs": finite(raw.get("standardized_abs")),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trade_summary = load_json(args.trade_attribution_dir / "summary.json")
    score_summary = load_json(args.q1_root / "score_attribution" / "summary.json")
    feature_summary = load_json(args.q1_root / "feature_audit" / "summary.json")
    decision_map = load_json(args.sync_root / "data_plane_decision_map_summary.json")
    field_frequency = pd.read_csv(args.trade_attribution_dir / "top_lost_field_frequency.csv")
    field_report = pd.read_csv(args.trade_attribution_dir / "top_lost_trade_field_report.csv")
    lost_trades = pd.read_csv(args.trade_attribution_dir / "lost_legacy_trades_attribution.csv")

    field_rows = build_field_rows(field_frequency)
    closures = scenario_closures(score_summary)
    route_rows = aggregate_routes(
        field_rows=field_rows,
        field_report=field_report,
        lost_trades=lost_trades,
        closures_by_route=closures,
    )
    examples = top_examples(field_report)

    headline = {
        "status": "offline_h1_audit_complete",
        "same_runner_legacy_trades": (trade_summary.get("legacy") or {}).get("serial_trades"),
        "same_runner_legacy_pnl": (trade_summary.get("legacy") or {}).get("serial_pnl"),
        "live_contract_trades": (trade_summary.get("protocol101_live_v1") or {}).get("serial_trades"),
        "live_contract_pnl": (trade_summary.get("protocol101_live_v1") or {}).get("serial_pnl"),
        "net_degradation": trade_summary.get("net_pnl_delta_live_minus_legacy"),
        "lost_legacy_trades": trade_summary.get("lost_legacy_trades"),
        "candidate_overlap_median_on_lost": (
            trade_summary.get("candidate_identity_overlap_on_lost") or {}
        ).get("median"),
        "q1_action_mismatches": feature_summary.get("action_mismatches"),
        "q1_top_contract_match_rate": feature_summary.get("top_contract_match_rate"),
        "frozen_model_reconstruction": (
            score_summary.get("frozen_model_baseline_reconstruction") or {}
        ).get("status"),
        "data_plane_decision_map_present": bool(decision_map),
        "broker_endpoint_called": False,
        "model_training": False,
        "threshold_tuning": False,
    }
    conclusions = [
        "July 2 entry/lifecycle synchronization can be true while Q1 historical non-inferiority still fails.",
        "The lost-trade candidate universe is mostly present, so the main Q1 issue is scoring/ranking semantics, not missing strikes.",
        "Same-input structure and market-window fixtures now pass; remaining Q1 drift is a feature-contract/data-policy problem.",
        "Pattern/token semantics are the first offline repair hypothesis; OI/volume are diagnostics unless a causal live equivalent is proven.",
        "Do not use the old 359-trade event-policy headline as a live benchmark unless it can be reproduced by the selected causal contract.",
    ]
    next_actions = [
        "Inspect the highest-PnL canonical-pattern and index-context examples for completed-minute/source-policy drift.",
        "Rerun Q1 only after a causal contract repair, with frozen weights and thresholds.",
        "Use July 6+ recorder days to confirm distribution and capture IBKR tick 100/101 diagnostics, not to tune thresholds.",
        "If H1/H2/H3 cannot recover non-inferiority causally, freeze the live-reproducible contract and retrain under purged/embargoed validation.",
    ]
    packet = Protocol101CanonicalSemanticsAuditV1(
        headline=headline,
        repair_routes=route_rows,
        field_routes=field_rows,
        conclusions=conclusions,
        next_actions=next_actions,
    )

    write_csv(args.out_dir / "field_route_summary.csv", field_rows)
    write_csv(args.out_dir / "repair_route_summary.csv", route_rows)
    write_csv(args.out_dir / "top_lost_examples_by_route.csv", examples)
    (args.out_dir / "summary.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Protocol101 H1 Canonical Semantics Audit",
        "",
        "## Scope",
        "",
        "- Offline only: no training, no threshold tuning, no paid download, no broker path.",
        "- Purpose: decide what can be tested before July 6 and what must wait for more recorder evidence.",
        "",
        "## Headline",
        "",
        (
            f"- Same-runner legacy: `{headline['same_runner_legacy_trades']}` trades / "
            f"`${float(headline['same_runner_legacy_pnl'] or 0.0):,.0f}`."
        ),
        (
            f"- Live-reproducible contract: `{headline['live_contract_trades']}` trades / "
            f"`${float(headline['live_contract_pnl'] or 0.0):,.0f}`."
        ),
        f"- Net degradation: `${float(headline['net_degradation'] or 0.0):,.0f}`.",
        (
            "- Lost-trade candidate identity overlap median: "
            f"`{headline['candidate_overlap_median_on_lost']}`."
        ),
        (
            "- Frozen model score reconstruction: "
            f"`{headline['frozen_model_reconstruction']}`."
        ),
        "",
        "## Repair Routes",
        "",
    ]
    for row in route_rows:
        lines.append(
            f"- `{row['repair_route']}`: status=`{row['recommended_status']}`, "
            f"lost-trade-pnl-exposure=`${float(row['lost_trade_pnl_exposure']):,.0f}`, "
            f"field-frequency-exposure=`${float(row['field_frequency_pnl_exposure']):,.0f}`, "
            f"primary-closure=`{row['primary_gap_closure_scenario']}`/"
            f"`{row['primary_gap_closure_signed_mean']}`, "
            f"action={row['recommended_action']}"
        )
    lines.extend(["", "## Conclusions", ""])
    lines.extend(f"- {item}" for item in conclusions)
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in next_actions)
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- H1 does not authorize restoring unavailable historical fields to live inference.",
            "- H1 does not authorize model training, threshold tuning, or paper-submit.",
            "- Any recovered performance must come from one causal contract used by both history and live replay.",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "status": headline["status"],
                "top_route": route_rows[0]["repair_route"] if route_rows else "UNKNOWN",
                "net_degradation": headline["net_degradation"],
                "model_training": False,
                "threshold_tuning": False,
                "broker_endpoint_called": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
