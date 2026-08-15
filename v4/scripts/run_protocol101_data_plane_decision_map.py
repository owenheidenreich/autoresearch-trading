"""Build the Protocol101 data-plane decision map from existing Q1 evidence.

This is an offline governance/diagnostic artifact. It reads completed Q1
contract-comparison and trade/PnL attribution outputs, then classifies the
remaining degradation into repair hypotheses. It does not train, tune, contact
vendors, touch broker paths, or mutate runtime defaults.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v4.live.protocol101_synchronization import Protocol101DataPlaneDecisionMapV1


DEFAULT_SYNC_ROOT = Path("v4/audit/autoresearch/protocol101_synchronization_resolution")
DEFAULT_TRADE_ATTRIBUTION = Path(
    "v4/audit/autoresearch/protocol101_q1_2026_trade_pnl_attribution"
)
DEFAULT_Q1_ROOT = Path("v4/audit/autoresearch/protocol101_q1_2026_contract_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SYNC_ROOT)
    parser.add_argument("--trade-attribution-dir", type=Path, default=DEFAULT_TRADE_ATTRIBUTION)
    parser.add_argument("--q1-root", type=Path, default=DEFAULT_Q1_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if pd.notna(number) else default


def classify_feature(feature: str) -> dict[str, str]:
    name = feature.lower()
    if "open_interest" in name:
        return {
            "family": "open_interest",
            "live_causal_status": "causal_if_publication_timestamp_is_respected",
            "primary_path": "test_enriched_ibkr_and_same_vendor_statistics",
            "decision": (
                "Capture IBKR generic tick 101 as diagnostic evidence, but do not "
                "treat it as Databento minute statistics. Prefer same-vendor OPRA "
                "statistics only if publication timing is causal."
            ),
        }
    if "volume" in name:
        return {
            "family": "option_volume",
            "live_causal_status": "partly_causal_but_vendor_semantics_differ",
            "primary_path": "test_enriched_ibkr_and_databento_live_ohlcv",
            "decision": (
                "Capture IBKR generic tick 100 as diagnostic evidence. It may help "
                "feature distribution, but it is not one-minute OHLCV and cannot be "
                "assumed equivalent."
            ),
        }
    if name.startswith("pattern_") or "pattern" in name:
        return {
            "family": "pattern_tokens",
            "live_causal_status": "causal_if_built_from_completed_minutes_only",
            "primary_path": "repair_or_freeze_canonical_pattern_semantics",
            "decision": (
                "Highest-priority semantic audit. Same raw canonical minutes now pass "
                "structure-feature parity, so remaining drift should be traced to input "
                "data policy, completed-minute policy, or intentionally changed live-v1 "
                "token semantics."
            ),
        }
    if name.startswith("market_") or "spx" in name or "vix" in name or "omar" in name:
        return {
            "family": "index_context",
            "live_causal_status": "causal_if_timestamp_policy_matches",
            "primary_path": "canonicalize_spx_vix_source_and_minute_lag",
            "decision": (
                "Use one completed-minute index contract. This is not solved by IBKR "
                "settings alone; the same source/timestamp semantics must feed both "
                "history and live replay."
            ),
        }
    if name.startswith("structure."):
        return {
            "family": "structure_context",
            "live_causal_status": "causal_if_built_from_completed_minutes_only",
            "primary_path": "keep_raw_input_parity_fixture_and_audit_real_inputs",
            "decision": (
                "The implementation matches for identical canonical minutes. Treat "
                "remaining Q1 drift as raw input/window policy drift until proven "
                "otherwise."
            ),
        }
    if "spread" in name or "overpay" in name or "worth" in name:
        return {
            "family": "quote_tradability",
            "live_causal_status": "causal_but_vendor_sensitive",
            "primary_path": "compare_quote_freshness_and_tradability_masks",
            "decision": (
                "Keep as parity guardrail. It can flip candidate selection, but Q1 "
                "evidence does not make it the first rescue path."
            ),
        }
    if any(part in name for part in ("gamma", "theta", "delta", "iv", "vega")):
        return {
            "family": "greeks_decay",
            "live_causal_status": "causal_through_shared_repair_when_inputs_exist",
            "primary_path": "preserve_shared_greek_repair",
            "decision": (
                "Use internally repaired Greeks for inference and keep vendor Greeks "
                "diagnostic. Validate input price/timestamp drift rather than trusting "
                "IBKR Greeks as replacements."
            ),
        }
    return {
        "family": "other_model_token",
        "live_causal_status": "requires_field_level_review",
        "primary_path": "manual_feature_contract_review",
        "decision": "Classify before any feature is restored to the runtime contract.",
    }


def family_rows(field_frequency: pd.DataFrame, score_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for _, raw in field_frequency.iterrows():
        feature = str(raw.get("feature") or "")
        classification = classify_feature(feature)
        row = {
            **classification,
            "feature": feature,
            "drift_family": str(raw.get("drift_family") or ""),
            "top_lost_occurrences": int(finite(raw.get("occurrences"))),
            "top_lost_occurrence_pnl_exposure": finite(raw.get("pnl")),
            "mean_standardized_abs": finite(raw.get("mean_standardized_abs")),
        }
        rows.append(row)

    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = row["family"]
        bucket = by_family.setdefault(
            family,
            {
                "family": family,
                "features": [],
                "top_lost_occurrences": 0,
                "top_lost_occurrence_pnl_exposure": 0.0,
                "max_mean_standardized_abs": 0.0,
                "live_causal_status": row["live_causal_status"],
                "primary_path": row["primary_path"],
                "decision": row["decision"],
            },
        )
        bucket["features"].append(row["feature"])
        bucket["top_lost_occurrences"] += int(row["top_lost_occurrences"])
        bucket["top_lost_occurrence_pnl_exposure"] += float(row["top_lost_occurrence_pnl_exposure"])
        bucket["max_mean_standardized_abs"] = max(
            float(bucket["max_mean_standardized_abs"]),
            float(row["mean_standardized_abs"]),
        )

    scenario_closures = {
        "pattern_tokens": "live_with_historical_tokens",
        "pattern_context": "live_with_historical_pattern_context",
        "option_volume": "live_with_historical_option_volume",
        "open_interest": "live_with_historical_open_interest",
        "quote_tradability": "live_with_historical_quote_spread",
        "greeks_decay": "live_with_historical_greeks_decay",
        "index_context": "live_with_historical_scalar_market_window",
        "structure_context": "live_with_historical_scalar_structure",
    }
    scenarios = score_summary.get("scenarios") or {}
    family_list = []
    for payload in by_family.values():
        family = str(payload["family"])
        scenario_name = scenario_closures.get(family)
        closure = None
        if scenario_name:
            closure = ((scenarios.get(scenario_name) or {}).get("historical_gap_closure") or {}).get(
                "signed_mean"
            )
        payload["counterfactual_mean_gap_closure_points"] = closure
        payload["features"] = ",".join(sorted(set(str(item) for item in payload["features"])))
        family_list.append(payload)
    return sorted(
        family_list,
        key=lambda item: (
            float(item.get("counterfactual_mean_gap_closure_points") or 0.0),
            float(item["top_lost_occurrence_pnl_exposure"]),
        ),
        reverse=True,
    )


def hypotheses(summary: dict[str, Any], families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    top_family = families[0]["family"] if families else "UNKNOWN"
    return [
        {
            "id": "H1_canonical_pattern_context_repair",
            "status": "test_now_offline",
            "claim": (
                "Most frozen-model degradation is caused by changed pattern/token/index "
                "semantics rather than missing contracts."
            ),
            "evidence": (
                f"Top family by current decision map is {top_family}; lost-trade "
                f"candidate identity overlap median is "
                f"{(summary.get('candidate_identity_overlap_on_lost') or {}).get('median')}."
            ),
            "test": (
                "Audit and, only if causal, restore canonical pattern/context semantics "
                "on Q1. Rerun Q1 non-inferiority with frozen weights and thresholds."
            ),
            "success": "Q1 live-causal contract reaches non-inferiority without lookahead.",
            "failure": "Proceed to H4 retraining unless H2/H3 materially improves the same gate.",
        },
        {
            "id": "H2_enriched_ibkr_diagnostics",
            "status": "collect_july6_plus_evidence",
            "claim": (
                "IBKR generic ticks 100/101 may recover some distribution shift but are "
                "not equivalent to Databento minute volume/statistics."
            ),
            "evidence": "OI/volume appear in high-drift lost fields, but average score closure is small or mixed.",
            "test": (
                "Use July 6+ recorder fields to measure whether generic tick values are "
                "stable, causal, and correlated with historical feature bins."
            ),
            "success": "Feature distribution improves without increasing cross-vendor action drift.",
            "failure": "Do not use IBKR ticks as direct replacements; keep them diagnostic or retrain.",
        },
        {
            "id": "H3_same_vendor_model_feed",
            "status": "quote_then_trial_if_needed",
            "claim": (
                "Databento live OPRA plus matching historical OPRA may reduce vendor "
                "quote/statistics drift; IBKR stays execution-only."
            ),
            "evidence": "Raw quote prices match more closely than derived semantics, so this is not guaranteed.",
            "test": "Get portal quotes, then run one live/history same-schema shadow day if cost is approved.",
            "success": "Same-vendor feed improves parity and Q1 non-inferiority enough to preserve Protocol101.",
            "failure": "Freeze canonical live-reproducible contract and retrain later.",
        },
        {
            "id": "H4_retrain_on_canonical_contract",
            "status": "blocked_until_H1_H2_H3_fail_or_are_rejected",
            "claim": "If frozen Protocol101 depends on unavailable semantics, a new model is required.",
            "evidence": (
                f"Current live-v1 Q1 degradation is ${finite(summary.get('net_pnl_delta_live_minus_legacy')):,.0f}."
            ),
            "test": "Train only after canonical contract is frozen, with purged/embargoed validation.",
            "success": "New candidate beats Protocol101 under the same live-causal game.",
            "failure": "Do not paper-submit a model whose historical edge cannot be reproduced live.",
        },
    ]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trade_summary = load_json(args.trade_attribution_dir / "summary.json")
    score_summary = load_json(args.q1_root / "score_attribution" / "summary.json")
    feature_summary = load_json(args.q1_root / "feature_audit" / "summary.json")
    field_frequency = pd.read_csv(args.trade_attribution_dir / "top_lost_field_frequency.csv")

    field_rows = []
    for _, raw in field_frequency.iterrows():
        feature = str(raw.get("feature") or "")
        field_rows.append(
            {
                **classify_feature(feature),
                "feature": feature,
                "drift_family": str(raw.get("drift_family") or ""),
                "top_lost_occurrences": int(finite(raw.get("occurrences"))),
                "top_lost_occurrence_pnl_exposure": finite(raw.get("pnl")),
                "mean_standardized_abs": finite(raw.get("mean_standardized_abs")),
            }
        )
    families = family_rows(field_frequency, score_summary)
    headline = {
        "same_runner_legacy_trades": (trade_summary.get("legacy") or {}).get("serial_trades"),
        "same_runner_legacy_pnl": (trade_summary.get("legacy") or {}).get("serial_pnl"),
        "live_contract_trades": (trade_summary.get("protocol101_live_v1") or {}).get("serial_trades"),
        "live_contract_pnl": (trade_summary.get("protocol101_live_v1") or {}).get("serial_pnl"),
        "net_degradation": trade_summary.get("net_pnl_delta_live_minus_legacy"),
        "lost_legacy_trades": trade_summary.get("lost_legacy_trades"),
        "live_only_trades": trade_summary.get("live_only_trades"),
        "candidate_overlap_median_on_lost": (
            trade_summary.get("candidate_identity_overlap_on_lost") or {}
        ).get("median"),
        "q1_action_mismatches": feature_summary.get("action_mismatches"),
        "top_contract_match_rate": feature_summary.get("top_contract_match_rate"),
        "model_training": False,
        "threshold_tuning": False,
        "broker_endpoint_called": False,
    }
    decisions = [
        "Do not treat the 359-trade older event-policy headline as the live benchmark.",
        "Do not paper-submit or hill-climb from the current live-v1 contract; historical non-inferiority still fails.",
        "Use July 6+ evidence to broaden confidence, but use Q1 attribution now to repair/evaluate the data plane.",
        "Prioritize canonical pattern/index/structure semantics before assuming IBKR subscription settings can restore the old result.",
        "Keep OI/volume enrichment as evidence, not an automatic replacement for Databento historical fields.",
    ]
    blockers = [
        "Q1 live-reproducible performance remains materially below same-runner legacy.",
        "Frozen Protocol101 may depend on feature semantics that are unavailable or differently defined live.",
        "July 2 is strong but narrow single-day entry/lifecycle evidence; more days are needed for confirmation, not for initial diagnosis.",
    ]
    packet = Protocol101DataPlaneDecisionMapV1(
        headline=headline,
        feature_families=families,
        hypotheses=hypotheses(trade_summary, families),
        decisions=decisions,
        blockers=blockers,
    )

    write_csv(args.out_dir / "data_plane_decision_map_fields.csv", field_rows)
    write_csv(args.out_dir / "data_plane_decision_map_families.csv", families)
    (args.out_dir / "data_plane_decision_map.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "data_plane_decision_map_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "Protocol101DataPlaneDecisionMapSummaryV1",
                "headline": headline,
                "top_feature_families": families[:8],
                "hypotheses": packet.hypotheses,
                "decisions": decisions,
                "blockers": blockers,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    lines = [
        "# Protocol101 Data-Plane Decision Map",
        "",
        "## Scope",
        "",
        "- Offline only: no training, no threshold tuning, no vendor call, no broker path.",
        "- Purpose: decide what can be fixed now versus what needs more July 6+ evidence or later retraining.",
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
            "- Q1 action mismatches / top-contract match rate: "
            f"`{headline['q1_action_mismatches']}` / `{headline['top_contract_match_rate']}`."
        ),
        "",
        "## Read",
        "",
        (
            "- The main Q1 issue is not that the live contract cannot see the same strike ladder. "
            "Candidate overlap is high. The problem is that the frozen model ranks/scores the "
            "same broad candidate universe differently under changed feature semantics."
        ),
        (
            "- July 2 remains strong synchronization evidence, but it is one narrow day. "
            "Use it as proof that the repaired replay path can match; do not use it as proof "
            "that the whole Q1 feature-distribution problem is solved."
        ),
        (
            "- Pattern tokens are the strongest counterfactual score-closure family, while "
            "index/context fields carry the largest top-lost field exposure. Treat both as "
            "first-class evidence, not as competing slogans."
        ),
        (
            "- Same-input fixtures now cover both structure features and market-window "
            "features. Both match when live and historical are fed identical canonical "
            "minutes, so the remaining Q1 degradation should be treated as data-policy/"
            "feature-contract shift unless a later fixture proves otherwise."
        ),
        "",
        "## Feature Families",
        "",
    ]
    for row in families[:10]:
        lines.append(
            f"- `{row['family']}`: closure=`{row.get('counterfactual_mean_gap_closure_points')}`, "
            f"occurrences=`{row['top_lost_occurrences']}`, "
            f"pnl-exposure=`${float(row['top_lost_occurrence_pnl_exposure']):,.0f}`, "
            f"path=`{row['primary_path']}`."
        )
    lines.extend(["", "## Hypotheses", ""])
    for item in packet.hypotheses:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- Status: `{item['status']}`",
                f"- Claim: {item['claim']}",
                f"- Test: {item['test']}",
                f"- Success: {item['success']}",
                f"- Failure: {item['failure']}",
                "",
            ]
        )
    lines.extend(["## Decisions", ""])
    lines.extend(f"- {item}" for item in decisions)
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in blockers)
    (args.out_dir / "data_plane_decision_map_report.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(packet.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
