"""Build the Protocol101 synchronization lineage and current gate packet."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from v4.live.protocol101_synchronization import (
    Protocol101BaselineLineageV1,
    Protocol101DataPlaneSelectionV1,
    Protocol101FairGamePolicyV1,
    Protocol101FeatureAvailabilityMatrixV1,
    Protocol101PaperReadinessGateV2,
    Protocol101SingleDaySynchronizationGateV1,
    evaluate_historical_non_inferiority,
)


DEFAULT_OLD_SUMMARY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json"
)
DEFAULT_OLD_TRADES = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/serial_policy_trades.json"
)
DEFAULT_Q1_COMPARISON = Path(
    "v4/audit/autoresearch/protocol101_q1_2026_contract_comparison/comparison/summary.json"
)
DEFAULT_Q1_FEATURE_AUDIT = Path(
    "v4/audit/autoresearch/protocol101_q1_2026_contract_comparison/feature_audit/summary.json"
)
DEFAULT_Q1_SCORE_ATTRIBUTION = Path(
    "v4/audit/autoresearch/protocol101_q1_2026_contract_comparison/score_attribution/summary.json"
)
DEFAULT_Q1_TRADE_PNL_ATTRIBUTION = Path(
    "v4/audit/autoresearch/protocol101_q1_2026_trade_pnl_attribution/summary.json"
)
DEFAULT_Q1_DIAGNOSTIC_SOURCE_POLICY_COMPARISON = Path(
    "v4/audit/autoresearch/protocol101_q1_diagnostic_source_policy_context_lag0/comparison/summary.json"
)
DEFAULT_Q1_FAIR_TIMING_SOURCE_POLICY_COMPARISON = Path(
    "v4/audit/autoresearch/protocol101_q1_fair_source_policy_lag0_decision_plus1/comparison/summary.json"
)
DEFAULT_DATA_PLANE_DECISION_MAP = Path(
    "v4/audit/autoresearch/protocol101_synchronization_resolution/data_plane_decision_map_summary.json"
)
DEFAULT_H1_CANONICAL_SEMANTICS_AUDIT = Path(
    "v4/audit/autoresearch/protocol101_h1_canonical_semantics_audit/summary.json"
)
DEFAULT_H1_TOP_EXAMPLE_INSPECTION = Path(
    "v4/audit/autoresearch/protocol101_h1_top_example_inspection/summary.json"
)
DEFAULT_H1_REPAIRABILITY_DECISION = Path(
    "v4/audit/autoresearch/protocol101_h1_repairability_decision/repairability_decision.json"
)
DEFAULT_EXISTING_CAPTURE_REPAIRABILITY_DIAGNOSTIC = Path(
    "v4/audit/autoresearch/protocol101_existing_capture_repairability_diagnostic/summary.json"
)
DEFAULT_FAIR_CONTRACT_TRAINING_PREFLIGHT = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_preflight/summary.json"
)
DEFAULT_FAIR_CONTRACT_TRAINING_DESIGN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_design/summary.json"
)
DEFAULT_FAIR_CONTRACT_TRAINING_DRY_RUN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_dry_run/summary.json"
)
DEFAULT_FAIR_CONTRACT_TRAINING_RUNNER_PLAN = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_training_runner/runner_plan.json"
)
DEFAULT_FAIR_CONTRACT_CANDIDATE_VALIDATION_GATE = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_candidate_validation_gate/summary.json"
)
DEFAULT_FAIR_CONTRACT_SELECTED_CANDIDATE_EXPORT = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_selected_candidate_export/summary.json"
)
DEFAULT_FAIR_CONTRACT_SELECTED_CANDIDATE_REPLAY_GATE = Path(
    "v4/audit/autoresearch/protocol101_fair_contract_selected_candidate_replay_gate/summary.json"
)
DEFAULT_JULY2 = (
    Path.home()
    / ".autoresearch-trading/live_runtime/ibkr_capture/2026-07-02/"
    "protocol101-recorder-2026-07-02/same_input_replay_summary.json"
)
DEFAULT_JULY2_PAIRED_DIFF = Path(
    "v4/audit/autoresearch/protocol101_2026_07_02_ibkr_vs_historical_paired_diff/summary.json"
)
DEFAULT_JULY2_ENTRY_SUMMARY = Path(
    "v4/audit/autoresearch/protocol101_2026_07_02_parity_resolution/entry_intent_summary.json"
)
DEFAULT_JULY2_LIFECYCLE_SUMMARY = Path(
    "v4/audit/autoresearch/protocol101_2026_07_02_parity_resolution/lifecycle_exit_summary.json"
)
DEFAULT_OUT = Path("v4/audit/autoresearch/protocol101_synchronization_resolution")
DEFAULT_CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-summary", type=Path, default=DEFAULT_OLD_SUMMARY)
    parser.add_argument("--old-trades", type=Path, default=DEFAULT_OLD_TRADES)
    parser.add_argument("--q1-comparison", type=Path, default=DEFAULT_Q1_COMPARISON)
    parser.add_argument("--q1-feature-audit", type=Path, default=DEFAULT_Q1_FEATURE_AUDIT)
    parser.add_argument("--q1-score-attribution", type=Path, default=DEFAULT_Q1_SCORE_ATTRIBUTION)
    parser.add_argument("--q1-trade-pnl-attribution", type=Path, default=DEFAULT_Q1_TRADE_PNL_ATTRIBUTION)
    parser.add_argument(
        "--q1-diagnostic-source-policy-comparison",
        type=Path,
        default=DEFAULT_Q1_DIAGNOSTIC_SOURCE_POLICY_COMPARISON,
    )
    parser.add_argument(
        "--q1-fair-timing-source-policy-comparison",
        type=Path,
        default=DEFAULT_Q1_FAIR_TIMING_SOURCE_POLICY_COMPARISON,
    )
    parser.add_argument("--data-plane-decision-map", type=Path, default=DEFAULT_DATA_PLANE_DECISION_MAP)
    parser.add_argument("--h1-canonical-semantics-audit", type=Path, default=DEFAULT_H1_CANONICAL_SEMANTICS_AUDIT)
    parser.add_argument("--h1-top-example-inspection", type=Path, default=DEFAULT_H1_TOP_EXAMPLE_INSPECTION)
    parser.add_argument("--h1-repairability-decision", type=Path, default=DEFAULT_H1_REPAIRABILITY_DECISION)
    parser.add_argument(
        "--existing-capture-repairability-diagnostic",
        type=Path,
        default=DEFAULT_EXISTING_CAPTURE_REPAIRABILITY_DIAGNOSTIC,
    )
    parser.add_argument(
        "--fair-contract-training-preflight",
        type=Path,
        default=DEFAULT_FAIR_CONTRACT_TRAINING_PREFLIGHT,
    )
    parser.add_argument(
        "--fair-contract-training-design",
        type=Path,
        default=DEFAULT_FAIR_CONTRACT_TRAINING_DESIGN,
    )
    parser.add_argument(
        "--fair-contract-training-dry-run",
        type=Path,
        default=DEFAULT_FAIR_CONTRACT_TRAINING_DRY_RUN,
    )
    parser.add_argument(
        "--fair-contract-training-runner-plan",
        type=Path,
        default=DEFAULT_FAIR_CONTRACT_TRAINING_RUNNER_PLAN,
    )
    parser.add_argument(
        "--fair-contract-candidate-validation-gate",
        type=Path,
        default=DEFAULT_FAIR_CONTRACT_CANDIDATE_VALIDATION_GATE,
    )
    parser.add_argument(
        "--fair-contract-selected-candidate-export",
        type=Path,
        default=DEFAULT_FAIR_CONTRACT_SELECTED_CANDIDATE_EXPORT,
    )
    parser.add_argument(
        "--fair-contract-selected-candidate-replay-gate",
        type=Path,
        default=DEFAULT_FAIR_CONTRACT_SELECTED_CANDIDATE_REPLAY_GATE,
    )
    parser.add_argument("--july2-replay", type=Path, default=DEFAULT_JULY2)
    parser.add_argument("--july2-paired-diff", type=Path, default=DEFAULT_JULY2_PAIRED_DIFF)
    parser.add_argument("--july2-entry-summary", type=Path, default=DEFAULT_JULY2_ENTRY_SUMMARY)
    parser.add_argument("--july2-lifecycle-summary", type=Path, default=DEFAULT_JULY2_LIFECYCLE_SUMMARY)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text())


def load_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        return load_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def feature_rows(score_attribution: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = score_attribution.get("scenarios") or {}

    def closure(name: str) -> str:
        payload = scenarios.get(name) or {}
        value = (payload.get("historical_gap_closure") or {}).get("signed_mean")
        return "UNKNOWN" if value is None else f"mean edge-gap closure {float(value):.4f} points"

    return [
        {
            "feature_group": "option_quotes",
            "features": "bid,ask,mid,spread,bid_size,ask_size",
            "historical_source": "Databento OPRA CBBO-1m at decision interval end",
            "ibkr_availability": "live event updates and minute checkpoints",
            "databento_live_availability": "OPRA live schemas",
            "thetadata_availability": "not used for options",
            "causal": True,
            "vendor_sensitive": True,
            "current_degradation_contribution": closure("live_with_historical_quote_spread"),
        },
        {
            "feature_group": "option_volume",
            "features": "option_ohlcv_volume,log_option_volume",
            "historical_source": "Databento OPRA OHLCV-1m",
            "ibkr_availability": "generic tick 100 daily aggregate diagnostic",
            "databento_live_availability": "available with live OPRA schema",
            "thetadata_availability": "not used for options",
            "causal": True,
            "vendor_sensitive": True,
            "current_degradation_contribution": closure("live_with_historical_option_volume"),
        },
        {
            "feature_group": "open_interest",
            "features": "stat_open_interest,log_open_interest",
            "historical_source": "Databento OPRA statistics",
            "ibkr_availability": "generic tick 101 daily aggregate diagnostic",
            "databento_live_availability": "schema/entitlement verification required",
            "thetadata_availability": "not used for options",
            "causal": "depends on publication timestamp",
            "vendor_sensitive": True,
            "current_degradation_contribution": closure("live_with_historical_open_interest"),
        },
        {
            "feature_group": "greeks_iv",
            "features": "iv,delta,gamma,theta,vega",
            "historical_source": "shared repaired Greeks from causal quote/index inputs",
            "ibkr_availability": "shared repaired Greeks; IBKR Greeks diagnostic only",
            "databento_live_availability": "shared repaired Greeks",
            "thetadata_availability": "index inputs only",
            "causal": True,
            "vendor_sensitive": "input prices and timestamps only",
            "current_degradation_contribution": closure("live_with_historical_greeks_decay"),
        },
        {
            "feature_group": "index_context",
            "features": "SPX,VIX,VWAP,momentum,range",
            "historical_source": "ThetaData one-minute SPX/VIX",
            "ibkr_availability": "live SPX/VIX event updates",
            "databento_live_availability": "not selected for current index contract",
            "thetadata_availability": "live and historical account capability must be verified",
            "causal": True,
            "vendor_sensitive": True,
            "current_degradation_contribution": closure("live_with_historical_scalar_market_window"),
        },
        {
            "feature_group": "opening_structure",
            "features": "OMAR,first15,breakout-pattern features",
            "historical_source": "completed 09:30 and subsequent causal index minutes",
            "ibkr_availability": "reconstructed from captured index events",
            "databento_live_availability": "derived from selected index source",
            "thetadata_availability": "derived from live/historical bars",
            "causal": True,
            "vendor_sensitive": True,
            "current_degradation_contribution": (
                f"pattern context {closure('live_with_historical_pattern_context')}; "
                f"discrete break {closure('live_with_historical_discrete_break_pattern')}; "
                f"OMAR-only {closure('live_with_historical_omar_patterns')}"
            ),
        },
        {
            "feature_group": "candidate_mask",
            "features": "strike ladder,freshness,spread,affordability,tradability",
            "historical_source": "canonical contract adapter",
            "ibkr_availability": "canonical contract adapter",
            "databento_live_availability": "canonical contract adapter",
            "thetadata_availability": "not applicable",
            "causal": True,
            "vendor_sensitive": True,
            "current_degradation_contribution": "candidate overlap and ranking reports",
        },
        {
            "feature_group": "lifecycle",
            "features": "full causal sequence,current PnL,MFE,MAE,forced-flat clock",
            "historical_source": "normalized quote path through 15:55 ET",
            "ibkr_availability": "captured minute path through 15:55 ET",
            "databento_live_availability": "canonical live quote path",
            "thetadata_availability": "index context only",
            "causal": True,
            "vendor_sensitive": True,
            "current_degradation_contribution": (
                "full-sequence/forced-flat replay defect repaired; July 2 lifecycle "
                "paths now match historical serial exits by timestamp and reason"
            ),
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_findings() -> list[dict[str, Any]]:
    paths = [
        Path("v4/scripts/run_protocol101_event_history_policy.py"),
        Path("v4/scripts/run_protocol097_sequential_event_policy.py"),
    ]
    needles = ("candidate_pnl", "candidate_exit_time", "add_oracle_actions", "simulate_event_policy")
    findings: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            for needle in needles:
                if needle in line:
                    findings.append(
                        {
                            "file": str(path),
                            "line": line_number,
                            "field_or_operation": needle,
                            "source": line.strip(),
                        }
                    )
    return findings


def reproduce_old_split(path: Path, split: str) -> dict[str, Any]:
    rows = json.loads(path.read_text())
    selected = [
        row
        for row in rows
        if int(row.get("seed") or 0) == 1
        and str(row.get("fold")) == "fold3_train_q1_q2_q3_validate_q4_test_q1_2026"
        and str(row.get("reported_split")) == split
        and float(row.get("slippage_per_side") or 0.0) == 0.0
    ]
    return {
        "trades": len(selected),
        "total_pnl": sum(float(row.get("pnl") or 0.0) for row in selected),
        "unique_trade_uids": len({str(row.get("trade_uid")) for row in selected}),
        "source": str(path),
    }


def shadow_evidence_inventory(capture_root: Path) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    for quality_path in sorted(capture_root.expanduser().glob("*/protocol101-recorder-*/ibkr_capture_quality.json")):
        replay_path = quality_path.parent / "same_input_replay_summary.json"
        try:
            quality = load_json(quality_path)
            replay = load_json(replay_path)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        lifecycle = replay.get("lifecycle_actions") or {}
        terminal = sum(int(lifecycle.get(name, 0)) for name in ("exit", "stop", "forced_flat"))
        stale_terminal = int(replay.get("lifecycle_terminal_stale_quote_rows") or 0)
        sessions.append(
            {
                "session": str(replay.get("session") or quality.get("session") or quality_path.parts[-3]),
                "capture_dir": str(quality_path.parent),
                "capture_quality": quality.get("status"),
                "same_input_exact": bool(replay.get("same_input_exact")),
                "entry_intents": int(replay.get("entry_actions") or 0),
                "terminal_lifecycle_paths": terminal,
                "stale_terminal_paths": stale_terminal,
                "complete_lifecycle_paths": max(terminal - stale_terminal, 0),
                "quote_missing_rows": int(replay.get("lifecycle_quote_missing_rows") or 0),
            }
        )
    complete = [
        row for row in sessions
        if row["capture_quality"] == "pass" and row["same_input_exact"]
    ]
    return {
        "schema_version": "Protocol101ShadowEvidenceInventoryV1",
        "sessions": sessions,
        "complete_sessions": len(complete),
        "entry_intents": sum(row["entry_intents"] for row in complete),
        "complete_lifecycle_paths": sum(row["complete_lifecycle_paths"] for row in complete),
        "unflattened_or_stale_positions": sum(
            1
            for row in complete
            if row["entry_intents"] > 0
            and (
                row["terminal_lifecycle_paths"] == 0
                or row["stale_terminal_paths"] > 0
            )
        ),
        "all_complete_same_input_exact": bool(complete)
        and all(row["same_input_exact"] for row in complete),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    old = load_json(args.old_summary)
    comparison = load_json(args.q1_comparison)
    feature_audit = load_json(args.q1_feature_audit)
    score_attribution = load_json(args.q1_score_attribution)
    trade_attribution = load_json_optional(args.q1_trade_pnl_attribution)
    diagnostic_source_policy_comparison = load_json_optional(args.q1_diagnostic_source_policy_comparison)
    fair_timing_source_policy_comparison = load_json_optional(
        args.q1_fair_timing_source_policy_comparison
    )
    data_plane_decision_map = load_json_optional(args.data_plane_decision_map)
    h1_canonical_semantics_audit = load_json_optional(args.h1_canonical_semantics_audit)
    h1_top_example_inspection = load_json_optional(args.h1_top_example_inspection)
    h1_repairability_decision = load_json_optional(args.h1_repairability_decision)
    existing_capture_repairability_diagnostic = load_json_optional(
        args.existing_capture_repairability_diagnostic
    )
    fair_contract_training_preflight = load_json_optional(args.fair_contract_training_preflight)
    fair_contract_training_design = load_json_optional(args.fair_contract_training_design)
    fair_contract_training_dry_run = load_json_optional(args.fair_contract_training_dry_run)
    fair_contract_training_runner_plan = load_json_optional(args.fair_contract_training_runner_plan)
    fair_contract_candidate_validation_gate = load_json_optional(
        args.fair_contract_candidate_validation_gate
    )
    fair_contract_selected_candidate_export = load_json_optional(
        args.fair_contract_selected_candidate_export
    )
    fair_contract_selected_candidate_replay_gate = load_json_optional(
        args.fair_contract_selected_candidate_replay_gate
    )
    july2 = load_json(args.july2_replay)
    july2_paired = load_json_optional(args.july2_paired_diff)
    july2_entry = load_json_optional(args.july2_entry_summary)
    july2_lifecycle = load_json_optional(args.july2_lifecycle_summary)
    old_q1 = old["aggregate_gate"]["q1_2026"]["seed_rows"][0]
    old_march = old["aggregate_gate"]["march_2026"]["seed_rows"][0]
    old_q1_ledger = reproduce_old_split(args.old_trades, "q1_2026")
    old_march_ledger = reproduce_old_split(args.old_trades, "march_2026")
    legacy = comparison["legacy"]
    live = comparison["protocol101_live_v1"]
    diagnostic_source_policy = (
        diagnostic_source_policy_comparison.get("protocol101_live_v1")
        if diagnostic_source_policy_comparison
        else None
    )
    fair_timing_source_policy = (
        fair_timing_source_policy_comparison.get("protocol101_live_v1")
        if fair_timing_source_policy_comparison
        else None
    )

    stages = [
        {
            "stage": "older_official_q1_2026_event_policy",
            "scope": "q1_2026",
            "trades": old_q1["model_trades"],
            "total_pnl": old_q1["model_total_pnl"],
            "causal_raw_market_replay": False,
            "frozen_ledger_reproduction": old_q1_ledger,
            "ledger_matches_summary": (
                old_q1_ledger["trades"] == old_q1["model_trades"]
                and old_q1_ledger["total_pnl"] == old_q1["model_total_pnl"]
            ),
            "note": "Uses precomputed candidate exits/PnL in evaluation.",
        },
        {
            "stage": "older_official_march_2026_reported_split",
            "scope": "march_2026",
            "trades": old_march["model_trades"],
            "total_pnl": old_march["model_total_pnl"],
            "causal_raw_market_replay": False,
            "frozen_ledger_reproduction": old_march_ledger,
            "ledger_matches_summary": (
                old_march_ledger["trades"] == old_march["model_trades"]
                and old_march_ledger["total_pnl"] == old_march["model_total_pnl"]
            ),
            "note": "Separate reported split; do not add to Q1 for an apples-to-apples benchmark.",
        },
        {
            "stage": "older_reported_combined_headline",
            "scope": "q1_2026_plus_march_2026_reported_splits",
            "trades": old_q1["model_trades"] + old_march["model_trades"],
            "total_pnl": old_q1["model_total_pnl"] + old_march["model_total_pnl"],
            "causal_raw_market_replay": False,
            "note": "359/$136,800 is a combined headline, not a 60-session Q1-only result.",
        },
        {
            "stage": "same_runner_legacy_contract",
            "scope": "q1_2026_current_serial_runner",
            **legacy,
            "causal_raw_market_replay": True,
        },
        {
            "stage": "protocol101_live_v1_contract",
            "scope": "q1_2026_current_serial_runner",
            **live,
            "causal_raw_market_replay": True,
        },
        *(
            [
                {
                    "stage": "diagnostic_context_lag0_same_minute_source_policy",
                    "scope": "q1_2026_current_serial_runner_diagnostic_only",
                    **diagnostic_source_policy,
                    "causal_raw_market_replay": True,
                    "production_contract_mutation": False,
                    "note": (
                        "Diagnostic-only rebuild: protocol101-live-v1 rows with "
                        "index_context_lag_minutes=0 to test repairability of source/timing policy."
                    ),
                }
            ]
            if diagnostic_source_policy
            else []
        ),
        *(
            [
                {
                    "stage": "fair_timing_context_lag0_decision_plus1",
                    "scope": "q1_2026_current_serial_runner_diagnostic_only",
                    **fair_timing_source_policy,
                    "causal_raw_market_replay": True,
                    "production_contract_mutation": False,
                    "note": (
                        "Fair-timing check: uses completed minute T source features but shifts "
                        "the effective decision/entry to minute T+1. This is the live-plausible "
                        "test for whether context-lag-0 can be a production repair."
                    ),
                }
            ]
            if fair_timing_source_policy
            else []
        ),
        {
            "stage": "ibkr_captured_feed_july2",
            "scope": "2026-07-02_development_session",
            "entry_intents": july2.get("entry_actions"),
            "same_input_exact": july2.get("same_input_exact"),
            "lifecycle_actions": july2.get("lifecycle_actions"),
            "causal_raw_market_replay": True,
        },
    ]
    lineage = Protocol101BaselineLineageV1(
        stages=stages,
        disclosures=[
            "The accepted equity benchmark must be generated by the same causal contract used live.",
            "The 359-trade headline combines separately reported Q1 and March scopes.",
            "IB Gateway is transport/execution infrastructure and cannot recreate unavailable model inputs.",
            "Model weights and thresholds remain frozen during synchronization.",
        ],
    )
    (args.out_dir / "baseline_lineage.json").write_text(
        json.dumps(lineage.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.out_dir / "performance_waterfall.csv", stages)

    findings = source_findings()
    write_csv(args.out_dir / "precomputed_path_dependency_audit.csv", findings)

    features = feature_rows(score_attribution)
    matrix = Protocol101FeatureAvailabilityMatrixV1(features=features)
    (args.out_dir / "feature_availability_matrix.json").write_text(
        json.dumps(matrix.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.out_dir / "feature_availability_matrix.csv", features)

    non_inferiority = evaluate_historical_non_inferiority(
        legacy=legacy,
        candidate=live,
        adverse_fill_total_pnl=None,
    )
    (args.out_dir / "historical_non_inferiority_gate.json").write_text(
        json.dumps(non_inferiority, indent=2, sort_keys=True) + "\n"
    )
    diagnostic_non_inferiority = None
    if diagnostic_source_policy:
        diagnostic_non_inferiority = evaluate_historical_non_inferiority(
            legacy=legacy,
            candidate=diagnostic_source_policy,
            adverse_fill_total_pnl=None,
        )
        (args.out_dir / "historical_non_inferiority_gate_diagnostic_context_lag0.json").write_text(
            json.dumps(diagnostic_non_inferiority, indent=2, sort_keys=True) + "\n"
        )
    fair_timing_non_inferiority = None
    if fair_timing_source_policy:
        fair_timing_non_inferiority = evaluate_historical_non_inferiority(
            legacy=legacy,
            candidate=fair_timing_source_policy,
            adverse_fill_total_pnl=None,
        )
        (args.out_dir / "historical_non_inferiority_gate_fair_timing_context_lag0_decision_plus1.json").write_text(
            json.dumps(fair_timing_non_inferiority, indent=2, sort_keys=True) + "\n"
        )
    fair_game_policy = Protocol101FairGamePolicyV1.evaluate(
        old_event_policy={
            "trades": old_q1["model_trades"],
            "total_pnl": old_q1["model_total_pnl"],
        },
        legacy_serial=legacy,
        live_contract=live,
        diagnostic_source_policy=diagnostic_source_policy,
        fair_timing_source_policy=fair_timing_source_policy,
        live_non_inferiority=non_inferiority,
        diagnostic_non_inferiority=diagnostic_non_inferiority,
        fair_timing_non_inferiority=fair_timing_non_inferiority,
    )
    (args.out_dir / "fair_game_policy.json").write_text(
        json.dumps(fair_game_policy.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    selection = Protocol101DataPlaneSelectionV1.evaluate(
        enriched_ibkr={
            "evaluation_complete": False,
            "parity_pass": False,
            "historical_non_inferiority_pass": False,
            "next_evidence": (
                "capture generic ticks 100/101 and test them, but do not expect a feed-only "
                "repair because Q1 same-source attribution identifies derived pattern semantics "
                "as the dominant edge-gap contributor"
            ),
        },
        same_vendor_model_feed={
            "evaluation_complete": False,
            "parity_pass": False,
            "historical_non_inferiority_pass": False,
            "next_evidence": "obtain portal quote and run a prospective same-schema live/history trial",
        },
    )
    (args.out_dir / "data_plane_selection.json").write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    quote_request = {
        "schema_version": "Protocol101DataPlaneQuoteRequestV1",
        "status": "owner_portal_action_required",
        "purchase_authorized": False,
        "requests": [
            {
                "vendor": "Databento",
                "product": "OPRA live plus historical access",
                "required_quote_fields": [
                    "professional_or_nonprofessional_classification",
                    "monthly_subscription_usd",
                    "exchange_license_usd",
                    "API_or_usage_fees",
                    "simultaneous_live_and_historical_access",
                ],
                "exact_portal_quote": "UNKNOWN",
            },
            {
                "vendor": "ThetaData",
                "product": "live and historical SPX/VIX one-minute or streaming index data",
                "required_quote_fields": [
                    "current_plan",
                    "live_index_entitlement",
                    "historical_index_entitlement",
                    "streaming_semantics",
                    "monthly_cost_usd",
                ],
                "exact_portal_quote": "UNKNOWN",
            },
            {
                "vendor": "IBKR",
                "product": "SPX/SPXW/VIX API market data",
                "required_quote_fields": [
                    "active_subscriptions",
                    "API_snapshot_or_streaming_eligibility",
                    "market_data_line_limit",
                ],
                "exact_portal_quote": "UNKNOWN",
            },
        ],
    }
    (args.out_dir / "data_plane_quote_request.json").write_text(
        json.dumps(quote_request, indent=2, sort_keys=True) + "\n"
    )

    shadow_inventory = shadow_evidence_inventory(args.capture_root)
    (args.out_dir / "shadow_evidence_inventory.json").write_text(
        json.dumps(shadow_inventory, indent=2, sort_keys=True) + "\n"
    )
    readiness = Protocol101PaperReadinessGateV2.evaluate(
        complete_shadow_sessions=int(shadow_inventory["complete_sessions"]),
        entry_intents=int(shadow_inventory["entry_intents"]),
        complete_lifecycle_paths=int(shadow_inventory["complete_lifecycle_paths"]),
        same_input_exact=bool(shadow_inventory["all_complete_same_input_exact"]),
        unclassified_nonthreshold_mismatches=None,
        unflattened_positions=int(shadow_inventory["unflattened_or_stale_positions"]),
    )
    (args.out_dir / "paper_readiness_gate.json").write_text(
        json.dumps(readiness.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    historical_serial_trades = int((july2_lifecycle or {}).get("historical_serial_trades") or 0)
    ibkr_terminal_paths = int((july2_lifecycle or {}).get("ibkr_terminal_paths") or 0)
    july2_unflattened = 0 if historical_serial_trades > 0 and ibkr_terminal_paths >= historical_serial_trades else 1
    single_day_sync = Protocol101SingleDaySynchronizationGateV1.evaluate(
        session=str(july2.get("session") or "2026-07-02"),
        same_input_exact=bool(july2.get("same_input_exact")),
        entry_intents=int((july2_entry or {}).get("live_entry_count") or july2.get("entry_actions") or 0),
        action_mismatches=(
            int(july2_paired["action_mismatches"])
            if july2_paired and july2_paired.get("action_mismatches") is not None
            else None
        ),
        selected_contract_mismatches=(
            int(july2_paired["selected_contract_mismatches"])
            if july2_paired and july2_paired.get("selected_contract_mismatches") is not None
            else None
        ),
        historical_serial_trades=historical_serial_trades,
        ibkr_terminal_paths=ibkr_terminal_paths,
        lifecycle_exit_time_matches=int((july2_lifecycle or {}).get("exit_time_matches") or 0),
        lifecycle_reason_matches=int((july2_lifecycle or {}).get("reason_matches") or 0),
        lifecycle_quote_missing_rows=int(july2.get("lifecycle_quote_missing_rows") or 0),
        lifecycle_terminal_stale_quote_rows=int(july2.get("lifecycle_terminal_stale_quote_rows") or 0),
        unflattened_positions=july2_unflattened,
    )
    (args.out_dir / "single_day_synchronization_gate.json").write_text(
        json.dumps(single_day_sync.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    report = [
        "# Protocol101 Synchronization Resolution",
        "",
        "## Current Decision",
        "",
        f"- July 2 single-day synchronization: `{single_day_sync.status}`",
        "- Data-plane selection: `evidence_pending`",
        f"- Historical non-inferiority: `{non_inferiority['status']}`",
        (
            "- Diagnostic source-policy non-inferiority: "
            f"`{diagnostic_non_inferiority['status']}`"
        )
        if diagnostic_non_inferiority
        else "- Diagnostic source-policy non-inferiority: `not_run`",
        (
            "- Fair-timing source-policy non-inferiority: "
            f"`{fair_timing_non_inferiority['status']}`"
        )
        if fair_timing_non_inferiority
        else "- Fair-timing source-policy non-inferiority: `not_run`",
        f"- Paper readiness: `{readiness.status}`",
        "- Frozen model weights changed: `false`",
        "- Frozen thresholds changed: `false`",
        "- Paper default changed: `false`",
        "- Paper-submit runtime authorization: `disabled until Protocol101PaperReadinessGateV2 passes`",
        f"- Fair-game policy: `{fair_game_policy.status}`",
        (
            "- Fair-contract training preflight: "
            f"`{fair_contract_training_preflight.get('status', 'missing')}`"
        )
        if fair_contract_training_preflight
        else "- Fair-contract training preflight: `missing`",
        (
            "- Fair-contract training design: "
            f"`{fair_contract_training_design.get('status', 'missing')}`"
        )
        if fair_contract_training_design
        else "- Fair-contract training design: `missing`",
        (
            "- Fair-contract training dry-run: "
            f"`{fair_contract_training_dry_run.get('status', 'missing')}`"
        )
        if fair_contract_training_dry_run
        else "- Fair-contract training dry-run: `missing`",
        (
            "- Fair-contract training runner: "
            f"`{fair_contract_training_runner_plan.get('status', 'missing')}`"
        )
        if fair_contract_training_runner_plan
        else "- Fair-contract training runner: `missing`",
        (
            "- Fair-contract candidate validation gate: "
            f"`{fair_contract_candidate_validation_gate.get('status', 'missing')}`"
        )
        if fair_contract_candidate_validation_gate
        else "- Fair-contract candidate validation gate: `missing`",
        (
            "- Fair-contract selected-candidate export: "
            f"`{fair_contract_selected_candidate_export.get('status', 'missing')}`"
        )
        if fair_contract_selected_candidate_export
        else "- Fair-contract selected-candidate export: `missing`",
        (
            "- Fair-contract selected-candidate replay gate: "
            f"`{fair_contract_selected_candidate_replay_gate.get('status', 'missing')}`"
        )
        if fair_contract_selected_candidate_replay_gate
        else "- Fair-contract selected-candidate replay gate: `missing`",
        "",
        "## Fair-Game Policy",
        "",
        f"- Objective: {fair_game_policy.objective}",
        "- Accepted degradation: old event-policy/precomputed-path edge is not the target; strict serial replay is the trustworthy benchmark.",
        "- Repairable degradation: live-contract losses are investigated only where the missing or changed feature semantics can be proven causal and live-available.",
        "- Forbidden: do not restore precomputed exits/PnL, overlapping-entry totals, unavailable pressure fields, threshold tuning, or paper-submit from diagnostic evidence alone.",
        f"- Retraining trigger: {fair_game_policy.retraining_trigger['condition']}",
        "",
        "## Baseline Truth",
        "",
        f"- Older Q1 event-policy result: `{old_q1['model_trades']}` trades / `${old_q1['model_total_pnl']:.0f}`.",
        f"- Separate March reported split: `{old_march['model_trades']}` trades / `${old_march['model_total_pnl']:.0f}`.",
        f"- Same-runner legacy contract: `{legacy['trades']}` trades / `${legacy['total_pnl']:.0f}`.",
        f"- Live-reproducible contract: `{live['trades']}` trades / `${live['total_pnl']:.0f}`.",
        *(
            [
                (
                    "- Diagnostic context-lag-0 source policy: "
                    f"`{diagnostic_source_policy['trades']}` trades / "
                    f"`${diagnostic_source_policy['total_pnl']:.0f}` "
                    "(diagnostic only; production contract unchanged)."
                ),
                (
                    "- Diagnostic context-lag-0 versus same-runner legacy: "
                    f"trade-count ratio `{diagnostic_non_inferiority['ratios']['trade_count']:.3f}`, "
                    f"PnL ratio `{diagnostic_non_inferiority['ratios']['total_pnl']:.3f}`, "
                    f"drawdown ratio `{diagnostic_non_inferiority['ratios']['drawdown']:.3f}`."
                ),
            ]
            if diagnostic_source_policy and diagnostic_non_inferiority
            else []
        ),
        *(
            [
                (
                    "- Fair-timing context-lag-0 decision-plus-one policy: "
                    f"`{fair_timing_source_policy['trades']}` trades / "
                    f"`${fair_timing_source_policy['total_pnl']:.0f}` "
                    "(diagnostic only; production contract unchanged)."
                ),
                (
                    "- Fair-timing versus same-runner legacy: "
                    f"trade-count ratio `{fair_timing_non_inferiority['ratios']['trade_count']:.3f}`, "
                    f"PnL ratio `{fair_timing_non_inferiority['ratios']['total_pnl']:.3f}`, "
                    f"drawdown ratio `{fair_timing_non_inferiority['ratios']['drawdown']:.3f}`."
                ),
                (
                    "- Interpretation: context-lag-0 recovered same-minute score behavior, "
                    "but the live-plausible one-minute execution delay destroyed profitability; "
                    "do not mutate production toward same-minute completed-bar semantics."
                ),
            ]
            if fair_timing_source_policy and fair_timing_non_inferiority
            else []
        ),
        "- The combined 359/$136,800 headline is not an apples-to-apples 60-session Q1 benchmark.",
        f"- Q1 paired feature minutes: `{feature_audit.get('paired_minutes')}`.",
        f"- Q1 action mismatches: `{feature_audit.get('action_mismatches')}`.",
        f"- Q1 top-contract match rate: `{feature_audit.get('top_contract_match_rate')}`.",
        (
            "- Q1 pattern-context mean edge-gap closure: "
            f"`{score_attribution['scenarios']['live_with_historical_pattern_context']['historical_gap_closure']['signed_mean']}`."
        ),
        (
            "- Q1 option-volume/open-interest mean edge-gap closure: "
            f"`{score_attribution['scenarios']['live_with_historical_option_volume']['historical_gap_closure']['signed_mean']}` / "
            f"`{score_attribution['scenarios']['live_with_historical_open_interest']['historical_gap_closure']['signed_mean']}`."
        ),
        (
            "- Complete exact-replay shadow sessions / entry intents / complete lifecycle paths: "
            f"`{shadow_inventory['complete_sessions']}` / `{shadow_inventory['entry_intents']}` / "
            f"`{shadow_inventory['complete_lifecycle_paths']}`."
        ),
        (
            "- July 2 repaired lifecycle actions: "
            f"`{july2.get('lifecycle_actions')}`; quote-missing rows: "
            f"`{july2.get('lifecycle_quote_missing_rows')}`; stale terminal rows: "
            f"`{july2.get('lifecycle_terminal_stale_quote_rows')}`."
        ),
    ]
    if july2_paired:
        report.append(
            "- July 2 cross-vendor paired diff: "
            f"`{july2_paired.get('status')}` with "
            f"`{july2_paired.get('action_matches')}` action matches, "
            f"`{july2_paired.get('action_mismatches')}` action mismatches, "
            f"`{july2_paired.get('selected_contract_matches')}` selected-contract matches, "
            f"and `{july2_paired.get('selected_contract_mismatches')}` selected-contract mismatches."
        )
    if july2_entry:
        report.append(
            "- July 2 threshold-crossing entry intents: "
            f"`{july2_entry.get('live_entry_count')}` IBKR / "
            f"`{july2_entry.get('historical_entry_count')}` historical, "
            f"contract match rate `{july2_entry.get('contract_match_rate')}`."
        )
    if july2_lifecycle:
        report.append(
            "- July 2 serial lifecycle parity: "
            f"`{july2_lifecycle.get('exit_time_matches')}` exit-time matches and "
            f"`{july2_lifecycle.get('reason_matches')}` reason matches across "
            f"`{july2_lifecycle.get('historical_serial_trades')}` historical serial trades; "
            f"max PnL delta `${float(july2_lifecycle.get('pnl_delta_max_abs') or 0.0):.2f}`."
        )
    if trade_attribution:
        lost_groups = trade_attribution.get("lost_dominant_gap_closure_group") or []
        top_lost_group = lost_groups[0] if lost_groups else {}
        report.extend(
            [
                "",
                "## Q1 Trade/PnL Attribution",
                "",
                (
                    "- Legacy-only trades / PnL: "
                    f"`{trade_attribution.get('lost_legacy_trades')}` / "
                    f"`${float(trade_attribution.get('lost_legacy_pnl') or 0.0):.0f}`."
                ),
                (
                    "- Live-contract-only trades / PnL: "
                    f"`{trade_attribution.get('live_only_trades')}` / "
                    f"`${float(trade_attribution.get('live_only_pnl') or 0.0):.0f}`."
                ),
                (
                    "- Net live-contract degradation versus same-runner legacy: "
                    f"`${float(trade_attribution.get('net_pnl_delta_live_minus_legacy') or 0.0):.0f}`."
                ),
                (
                    "- Lost-trade candidate identity overlap median: "
                    f"`{(trade_attribution.get('candidate_identity_overlap_on_lost') or {}).get('median')}`."
                ),
                (
                    "- Top lost-trade feature group by positive legacy PnL: "
                    f"`{top_lost_group.get('dominant_gap_closure_group', 'UNKNOWN')}` / "
                    f"`${float(top_lost_group.get('pnl') or 0.0):.0f}`."
                ),
                (
                    "- Interpretation: Q1 degradation is mainly score/ranking/feature-semantics drift, "
                    "not missing candidate-universe coverage."
                ),
            ]
        )
    if data_plane_decision_map:
        top_families = data_plane_decision_map.get("top_feature_families") or []
        top_family = top_families[0] if top_families else {}
        second_family = top_families[1] if len(top_families) > 1 else {}
        exposure_family = max(
            top_families,
            key=lambda row: float(row.get("top_lost_occurrence_pnl_exposure") or 0.0),
            default={},
        )
        first_hypothesis = (data_plane_decision_map.get("hypotheses") or [{}])[0]
        report.extend(
            [
                "",
                "## Data-Plane Decision Map",
                "",
                f"- Current first offline hypothesis: `{first_hypothesis.get('id', 'UNKNOWN')}`.",
                (
                    "- Top counterfactual feature family: "
                    f"`{top_family.get('family', 'UNKNOWN')}` with "
                    f"`{top_family.get('counterfactual_mean_gap_closure_points')}` "
                    "mean edge-gap closure points."
                ),
                (
                    "- Next material family by this ranking: "
                    f"`{second_family.get('family', 'UNKNOWN')}` with "
                    f"`${float(second_family.get('top_lost_occurrence_pnl_exposure') or 0.0):.0f}` "
                    "top-lost field exposure."
                ),
                (
                    "- Largest top-lost field exposure family: "
                    f"`{exposure_family.get('family', 'UNKNOWN')}` with "
                    f"`${float(exposure_family.get('top_lost_occurrence_pnl_exposure') or 0.0):.0f}` "
                    "field-exposure dollars."
                ),
                (
                    "- Decision: use July 6+ evidence to broaden confidence, but use Q1 "
                    "attribution now to repair/evaluate the data plane."
                ),
                (
                    "- Same-input regression fixtures now pass for both structure and "
                    "market-window features, so remaining Q1 degradation is classified as "
                    "feature-contract/data-policy shift rather than a simple live-vs-"
                    "historical formula mismatch."
                ),
            ]
        )
    if h1_canonical_semantics_audit:
        h1_headline = h1_canonical_semantics_audit.get("headline") or {}
        h1_routes = h1_canonical_semantics_audit.get("repair_routes") or []
        h1_top_route = h1_routes[0] if h1_routes else {}
        h1_second_route = h1_routes[1] if len(h1_routes) > 1 else {}
        report.extend(
            [
                "",
                "## H1 Canonical Semantics Audit",
                "",
                f"- H1 audit status: `{h1_headline.get('status', 'UNKNOWN')}`.",
                (
                    "- First offline repair route: "
                    f"`{h1_top_route.get('repair_route', 'UNKNOWN')}` with "
                    f"`${float(h1_top_route.get('lost_trade_pnl_exposure') or 0.0):.0f}` "
                    "lost-trade PnL exposure and "
                    f"`{h1_top_route.get('primary_gap_closure_signed_mean')}` "
                    "mean edge-gap closure points."
                ),
                (
                    "- Second repair route: "
                    f"`{h1_second_route.get('repair_route', 'UNKNOWN')}` with "
                    f"`${float(h1_second_route.get('lost_trade_pnl_exposure') or 0.0):.0f}` "
                    "lost-trade PnL exposure."
                ),
                (
                    "- H1 conclusion: July 2 can prove entry/lifecycle synchronization "
                    "for one useful day while Q1 non-inferiority still fails from "
                    "feature-contract/data-policy drift."
                ),
                (
                    "- H1 guardrail: OI/volume remain diagnostics unless a causal live "
                    "equivalent is proven; do not restore those historical fields just "
                    "to recover the old equity curve."
                ),
            ]
        )
    if h1_top_example_inspection:
        signatures = h1_top_example_inspection.get("signatures") or []
        top_signature = signatures[0] if signatures else {}
        diff_routes = h1_top_example_inspection.get("diff_routes") or []
        route_counts = {
            str(row.get("repair_route")): row.get("diff_rows")
            for row in diff_routes
        }
        report.extend(
            [
                "",
                "## H1 Top Example Inspection",
                "",
                (
                    "- Top lost legacy examples signature: "
                    f"`{top_signature.get('signature', 'UNKNOWN')}` across "
                    f"`{top_signature.get('examples', 'UNKNOWN')}` examples with "
                    f"`${float(top_signature.get('pnl') or 0.0):.0f}` PnL exposure "
                    f"and `{top_signature.get('mean_edge_delta')}` mean legacy-minus-live edge delta."
                ),
                (
                    "- Inspected token-diff route counts: "
                    f"pattern=`{route_counts.get('canonical_pattern_semantics', 'UNKNOWN')}`, "
                    f"pressure=`{route_counts.get('runtime_zeroed_pressure_diagnostics', 'UNKNOWN')}`, "
                    f"candidate-geometry=`{route_counts.get('canonical_candidate_geometry', 'UNKNOWN')}`, "
                    f"greeks/quote-path=`{route_counts.get('greeks_decay_quote_path', 'UNKNOWN')}`."
                ),
                (
                    "- Interpretation: the biggest Q1 lost trades were present in "
                    "both traces at the same quote, so the failure is not primarily "
                    "missing contracts or obvious quote absence."
                ),
            ]
        )
    if h1_repairability_decision:
        report.extend(
            [
                "",
                "## H1 Repairability Decision",
                "",
                f"- Status: `{h1_repairability_decision.get('status', 'UNKNOWN')}`.",
                f"- Decision: `{h1_repairability_decision.get('decision', 'UNKNOWN')}`.",
                (
                    "- Immediate contract change allowed: "
                    f"`{str(h1_repairability_decision.get('immediate_contract_change_allowed')).lower()}`."
                ),
                (
                    "- Q1 rerun required now: "
                    f"`{str(h1_repairability_decision.get('q1_rerun_required_now')).lower()}`."
                ),
                (
                    "- Meaning: pattern/context and candidate-geometry drift are real, "
                    "but the same-minute completed-bar mutation path is rejected after "
                    "the fair timing replay. Future repairs must use a genuinely causal "
                    "intra-minute contract, a different data plane, or retraining."
                ),
            ]
        )
    if existing_capture_repairability_diagnostic:
        existing_headline = existing_capture_repairability_diagnostic.get("headline") or {}
        q1_policy = existing_capture_repairability_diagnostic.get("q1_lost_trade_policy") or {}
        capture_timing = existing_capture_repairability_diagnostic.get("capture_timing") or []
        routes = existing_capture_repairability_diagnostic.get("repairability_routes") or []
        source_policy_route = next(
            (row for row in routes if row.get("route") == "source_policy_timing_and_atm_geometry"),
            {},
        )
        current_contract_route = next(
            (row for row in routes if row.get("route") == "current_recorder_contract_causality"),
            {},
        )
        report.extend(
            [
                "",
                "## Existing Capture Repairability Diagnostic",
                "",
                f"- Status: `{existing_headline.get('status', 'UNKNOWN')}`.",
                (
                    "- Existing exact same-input capture sessions / entry actions: "
                    f"`{existing_headline.get('same_input_exact_sessions', 'UNKNOWN')}` / "
                    f"`{existing_headline.get('entry_actions_in_existing_captures', 'UNKNOWN')}`."
                ),
                (
                    "- Lost Q1 legacy trades with a live-contract signal within one minute "
                    "and five strike points: "
                    f"`{q1_policy.get('one_minute_near_trades', 'UNKNOWN')}` / "
                    f"`${float(q1_policy.get('one_minute_near_pnl') or 0.0):.0f}` "
                    f"({float(q1_policy.get('one_minute_near_pnl_share') or 0.0):.1%} of lost PnL)."
                ),
                (
                    "- Current recorder contract route: "
                    f"`{current_contract_route.get('status', 'UNKNOWN')}` / "
                    f"`{current_contract_route.get('repairability', 'UNKNOWN')}`."
                ),
                (
                    "- Source-policy route: "
                    f"`{source_policy_route.get('status', 'UNKNOWN')}` / "
                    f"`{source_policy_route.get('repairability', 'UNKNOWN')}`."
                ),
                (
                    "- Capture timing: "
                    + "; ".join(
                        f"{row.get('session')} ctx-lag={row.get('context_lag_min_median')} "
                        f"future-context={row.get('future_context_rows')} "
                        f"future-quote={row.get('future_quote_rows')}"
                        for row in capture_timing
                    )
                ),
                (
                    "- Interpretation: use existing captures now for repeatable offline "
                    "diagnostics; July 6+ is confirmation/out-of-sample evidence, not a "
                    "reason to sit idle."
                ),
                (
                    "- Guardrail: the diagnostic authorizes an explicitly labeled "
                    "alternate-source-policy rebuild only; it does not authorize mutating "
                    "`protocol101-live-v1`, rerunning Q1 as a fixed claim, training, tuning, "
                    "or paper-submit."
                ),
            ]
        )
    if fair_contract_training_preflight:
        dataset_checks = fair_contract_training_preflight.get("dataset_checks") or {}
        processed = dataset_checks.get("processed") or {}
        row_checks = dataset_checks.get("row_checks") or {}
        sync_checks = fair_contract_training_preflight.get("synchronization_checks") or {}
        report.extend(
            [
                "",
                "## Fair-Contract Training Preflight",
                "",
                f"- Status: `{fair_contract_training_preflight.get('status', 'UNKNOWN')}`.",
                f"- Decision: `{fair_contract_training_preflight.get('decision', 'UNKNOWN')}`.",
                (
                    "- Model training authorized: "
                    f"`{str(fair_contract_training_preflight.get('model_training_authorized')).lower()}`."
                ),
                (
                    "- Paper-submit allowed: "
                    f"`{str(fair_contract_training_preflight.get('paper_submit_allowed')).lower()}`."
                ),
                (
                    "- Processed exact sessions / total processed files: "
                    f"`{dataset_checks.get('processed_exact_session_count', 'UNKNOWN')}` / "
                    f"`{processed.get('file_count', 'UNKNOWN')}`."
                ),
                (
                    "- Processed rows inspected / feature contract versions: "
                    f"`{row_checks.get('row_count_total', 'UNKNOWN')}` / "
                    f"`{row_checks.get('version_counts', 'UNKNOWN')}`."
                ),
                (
                    "- Future quote/context rows: "
                    f"`{row_checks.get('future_quote_rows', 'UNKNOWN')}` / "
                    f"`{row_checks.get('future_context_rows', 'UNKNOWN')}`."
                ),
                (
                    "- Training-preflight blockers: "
                    f"`{fair_contract_training_preflight.get('blockers', [])}`."
                ),
                (
                    "- Synchronization status used by preflight: historical="
                    f"`{sync_checks.get('historical_non_inferiority_status', 'UNKNOWN')}`, "
                    f"fair_timing=`{sync_checks.get('fair_timing_non_inferiority_status', 'UNKNOWN')}`, "
                    f"single_day=`{sync_checks.get('single_day_synchronization_status', 'UNKNOWN')}`."
                ),
            ]
        )
    if fair_contract_training_design:
        allowed = fair_contract_training_design.get("allowed_data") or {}
        split = fair_contract_training_design.get("split_policy") or {}
        report.extend(
            [
                "",
                "## Fair-Contract Training Design",
                "",
                f"- Status: `{fair_contract_training_design.get('status', 'UNKNOWN')}`.",
                f"- Decision: `{fair_contract_training_design.get('decision', 'UNKNOWN')}`.",
                (
                    "- Model training authorized: "
                    f"`{str(fair_contract_training_design.get('model_training_authorized')).lower()}`."
                ),
                (
                    "- Threshold tuning authorized: "
                    f"`{str(fair_contract_training_design.get('threshold_tuning_authorized')).lower()}`."
                ),
                (
                    "- Paper-submit allowed: "
                    f"`{str(fair_contract_training_design.get('paper_submit_allowed')).lower()}`."
                ),
                (
                    "- Allowed manifest sessions / manifest loading required: "
                    f"`{allowed.get('included_session_count', 'UNKNOWN')}` / "
                    f"`{str(allowed.get('require_manifest_loading')).lower()}`."
                ),
                (
                    "- Split policy: train=`"
                    f"{len(split.get('train_sessions') or [])}`, validation=`"
                    f"{len(split.get('validation_sessions') or [])}`, diagnostic_test=`"
                    f"{len(split.get('diagnostic_test_sessions') or [])}`."
                ),
                (
                    "- Meaning: this is a ready design packet, not permission to train. "
                    "Owner approval is still required before any model-training command."
                ),
            ]
        )
    if fair_contract_training_dry_run:
        feature_summary = fair_contract_training_dry_run.get("feature_summary") or {}
        label_summary = fair_contract_training_dry_run.get("label_summary") or {}
        split_summary = fair_contract_training_dry_run.get("split_summary") or {}
        report.extend(
            [
                "",
                "## Fair-Contract Training Dry Run",
                "",
                f"- Status: `{fair_contract_training_dry_run.get('status', 'UNKNOWN')}`.",
                f"- Decision: `{fair_contract_training_dry_run.get('decision', 'UNKNOWN')}`.",
                (
                    "- Feature dimensions / sampled vectors: "
                    f"`{feature_summary.get('feature_dimensions', 'UNKNOWN')}` / "
                    f"`{feature_summary.get('feature_samples', 'UNKNOWN')}`."
                ),
                (
                    "- Feature imputation required: "
                    f"`{str(feature_summary.get('feature_imputation_required')).lower()}` "
                    f"from `{feature_summary.get('nonfinite_feature_values', 'UNKNOWN')}` "
                    "sampled nonfinite values."
                ),
                (
                    "- Split rows: train=`"
                    f"{(split_summary.get('train') or {}).get('decision_rows', 'UNKNOWN')}`, "
                    "validation=`"
                    f"{(split_summary.get('validation') or {}).get('decision_rows', 'UNKNOWN')}`, "
                    "diagnostic_test=`"
                    f"{(split_summary.get('diagnostic_test') or {}).get('decision_rows', 'UNKNOWN')}`."
                ),
                (
                    "- Label positive rates: "
                    + ", ".join(
                        f"{name}={payload.get('positive_rate')}"
                        for name, payload in sorted(label_summary.items())
                    )
                ),
                (
                    "- Dry-run blockers: "
                    f"`{fair_contract_training_dry_run.get('blockers', [])}`."
                ),
                (
                    "- Meaning: manifest ingestion and label availability are ready for a "
                    "future owner-approved training runner; this still is not a training run."
                ),
            ]
        )
    if fair_contract_training_runner_plan:
        split_sessions = fair_contract_training_runner_plan.get("split_sessions") or {}
        report.extend(
            [
                "",
                "## Fair-Contract Training Runner",
                "",
                f"- Status: `{fair_contract_training_runner_plan.get('status', 'UNKNOWN')}`.",
                f"- Decision: `{fair_contract_training_runner_plan.get('decision', 'UNKNOWN')}`.",
                f"- Mode: `{fair_contract_training_runner_plan.get('mode', 'UNKNOWN')}`.",
                (
                    "- Model training / threshold selection executed: "
                    f"`{str(fair_contract_training_runner_plan.get('model_training_executed')).lower()}` / "
                    f"`{str(fair_contract_training_runner_plan.get('threshold_selection_executed')).lower()}`."
                ),
                (
                    "- Broker endpoint called / paper-submit allowed: "
                    f"`{str(fair_contract_training_runner_plan.get('broker_endpoint_called')).lower()}` / "
                    f"`{str(fair_contract_training_runner_plan.get('paper_submit_allowed')).lower()}`."
                ),
                (
                    "- Runner split sessions: train=`"
                    f"{len(split_sessions.get('train') or [])}`, validation=`"
                    f"{len(split_sessions.get('validation') or [])}`, diagnostic_test=`"
                    f"{len(split_sessions.get('diagnostic_test') or [])}`."
                ),
                (
                    "- Runner blockers: "
                    f"`{fair_contract_training_runner_plan.get('blockers', [])}`."
                ),
                (
                    "- Meaning: the future training command path is now wired and locked; "
                    "actual execution still needs explicit owner approval flags."
                ),
            ]
        )
    if fair_contract_candidate_validation_gate:
        checks = fair_contract_candidate_validation_gate.get("checks") or {}
        report.extend(
            [
                "",
                "## Fair-Contract Candidate Validation Gate",
                "",
                f"- Status: `{fair_contract_candidate_validation_gate.get('status', 'UNKNOWN')}`.",
                f"- Decision: `{fair_contract_candidate_validation_gate.get('decision', 'UNKNOWN')}`.",
                (
                    "- Model training / threshold selection executed: "
                    f"`{str(fair_contract_candidate_validation_gate.get('model_training_executed')).lower()}` / "
                    f"`{str(fair_contract_candidate_validation_gate.get('threshold_selection_executed')).lower()}`."
                ),
                (
                    "- Broker endpoint called / paper-submit allowed: "
                    f"`{str(fair_contract_candidate_validation_gate.get('broker_endpoint_called')).lower()}` / "
                    f"`{str(fair_contract_candidate_validation_gate.get('paper_submit_allowed')).lower()}`."
                ),
                (
                    "- Training result present: "
                    f"`{str((checks.get('training_result_present') or {}).get('value')).lower()}`."
                ),
                (
                    "- Meaning: no candidate can advance to strict serial/lifecycle replay "
                    "until a future owner-approved training result passes this gate."
                ),
            ]
        )
    if fair_contract_selected_candidate_export:
        report.extend(
            [
                "",
                "## Fair-Contract Selected-Candidate Export",
                "",
                f"- Status: `{fair_contract_selected_candidate_export.get('status', 'UNKNOWN')}`.",
                f"- Decision: `{fair_contract_selected_candidate_export.get('decision', 'UNKNOWN')}`.",
                (
                    "- Model loaded / broker endpoint called / paper-submit allowed: "
                    f"`{str(fair_contract_selected_candidate_export.get('model_loaded')).lower()}` / "
                    f"`{str(fair_contract_selected_candidate_export.get('broker_endpoint_called')).lower()}` / "
                    f"`{str(fair_contract_selected_candidate_export.get('paper_submit_allowed')).lower()}`."
                ),
                (
                    "- Export blockers: "
                    f"`{fair_contract_selected_candidate_export.get('blockers', [])}`."
                ),
                (
                    "- Meaning: once a future candidate passes validation, this bridge will "
                    "export selected entries with contract IDs, source timestamps, scores, "
                    "labels, entry quotes, and feature hashes for strict replay."
                ),
            ]
        )
    if fair_contract_selected_candidate_replay_gate:
        report.extend(
            [
                "",
                "## Fair-Contract Selected-Candidate Replay Gate",
                "",
                f"- Status: `{fair_contract_selected_candidate_replay_gate.get('status', 'UNKNOWN')}`.",
                f"- Decision: `{fair_contract_selected_candidate_replay_gate.get('decision', 'UNKNOWN')}`.",
                (
                    "- Strict replay / broker endpoint / paper-submit: "
                    f"`{str(fair_contract_selected_candidate_replay_gate.get('strict_replay_executed')).lower()}` / "
                    f"`{str(fair_contract_selected_candidate_replay_gate.get('broker_endpoint_called')).lower()}` / "
                    f"`{str(fair_contract_selected_candidate_replay_gate.get('paper_submit_allowed')).lower()}`."
                ),
                (
                    "- Replay blockers: "
                    f"`{fair_contract_selected_candidate_replay_gate.get('blockers', [])}`."
                ),
                (
                    "- Meaning: after selected entries exist, this gate enforces one-account "
                    "affordability, serial behavior, ask-entry premium, and adverse-fill stress "
                    "before strict lifecycle or recorder-shadow validation."
                ),
            ]
        )
    if diagnostic_source_policy and diagnostic_non_inferiority:
        checks = diagnostic_non_inferiority["checks"]
        report.extend(
            [
                "",
                "## Diagnostic Source-Policy Q1 Rebuild",
                "",
                "- Mode: `diagnostic_context_lag0_same_minute`.",
                "- Production `protocol101-live-v1` mutation: `false`.",
                (
                    "- Result: "
                    f"`{diagnostic_source_policy['trades']}` serial trades / "
                    f"`${diagnostic_source_policy['total_pnl']:.0f}` PnL / "
                    f"profit factor `{diagnostic_source_policy.get('profit_factor')}` / "
                    f"max drawdown `${float(diagnostic_source_policy.get('max_drawdown_dollars') or 0.0):.0f}`."
                ),
                (
                    "- Compared with production live-v1: "
                    f"`+{int(diagnostic_source_policy['trades']) - int(live['trades'])}` trades and "
                    f"`+${float(diagnostic_source_policy['total_pnl']) - float(live['total_pnl']):.0f}` PnL."
                ),
                (
                    "- Compared with same-runner legacy: "
                    f"`{diagnostic_non_inferiority['ratios']['trade_count']:.1%}` of trade count and "
                    f"`{diagnostic_non_inferiority['ratios']['total_pnl']:.1%}` of PnL."
                ),
                (
                    "- Non-inferiority checks: "
                    + ", ".join(
                        f"{name}={str(value).lower()}"
                        for name, value in checks.items()
                    )
                    + "."
                ),
                (
                    "- Interpretation: source/timing policy is a meaningful forensic clue, "
                    "but by itself it is not a production repair because this same-minute "
                    "diagnostic does not prove the model could have known the completed "
                    "minute before entering."
                ),
                (
                    "- Fair-game decision: keep this as a repairability clue, not as permission "
                    "to recover old edge by weakening causal/live-reproducible rules."
                ),
            ]
        )
    if fair_timing_source_policy and fair_timing_non_inferiority:
        checks = fair_timing_non_inferiority["checks"]
        report.extend(
            [
                "",
                "## Fair-Timing Q1 Rebuild",
                "",
                "- Mode: `diagnostic_context_lag0_decision_plus1`.",
                "- Production `protocol101-live-v1` mutation: `false`.",
                (
                    "- Result: "
                    f"`{fair_timing_source_policy['trades']}` serial trades / "
                    f"`${fair_timing_source_policy['total_pnl']:.0f}` PnL / "
                    f"profit factor `{fair_timing_source_policy.get('profit_factor')}` / "
                    f"max drawdown `${float(fair_timing_source_policy.get('max_drawdown_dollars') or 0.0):.0f}`."
                ),
                (
                    "- Compared with same-runner legacy: "
                    f"`{fair_timing_non_inferiority['ratios']['trade_count']:.1%}` of trade count and "
                    f"`{fair_timing_non_inferiority['ratios']['total_pnl']:.1%}` of PnL."
                ),
                (
                    "- Non-inferiority checks: "
                    + ", ".join(
                        f"{name}={str(value).lower()}"
                        for name, value in checks.items()
                    )
                    + "."
                ),
                (
                    "- Interpretation: the apparent context-lag-0 recovery does not survive "
                    "the live-plausible rule of observing completed minute T and entering on "
                    "minute T+1. Treat the recovered same-minute edge as timing privilege, "
                    "not as a contract repair."
                ),
            ]
        )
    report.extend(
        [
            "",
            "## Required Next Evidence",
            "",
            "1. Preserve July 2 as passed single-day synchronization evidence, while keeping strict feature/score drift as review evidence.",
            "2. Run the next recorder-first packet on July 6-10, with July 6 as the ungated development day and later days gated on July 6 collection integrity.",
            "3. Use future days to broaden evidence beyond one morning put sequence: calls, other time buckets, and non-time-flat exits remain under-sampled.",
            "4. Keep the raw-input structure and market-window parity fixtures as regression guards; the focused lost-trade token-field report is now present.",
            "5. Reject same-minute completed-bar recovery as a production repair; only a truly intra-minute causal model/feed contract could revisit that path.",
            "6. Capture enriched IBKR generic ticks as diagnostics on a future session.",
            "7. Price, but do not purchase, the same-vendor live OPRA and SPX/VIX alternative.",
            "8. Select a canonical causal data plane or declare retraining required.",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "historical_non_inferiority": non_inferiority["status"],
                "diagnostic_source_policy_non_inferiority": (
                    diagnostic_non_inferiority["status"] if diagnostic_non_inferiority else "not_run"
                ),
                "fair_timing_source_policy_non_inferiority": (
                    fair_timing_non_inferiority["status"] if fair_timing_non_inferiority else "not_run"
                ),
                "single_day_synchronization": single_day_sync.status,
                "data_plane_selection": selection.status,
                "paper_readiness": readiness.status,
                "trade_pnl_attribution": "present" if trade_attribution else "missing",
                "data_plane_decision_map": "present" if data_plane_decision_map else "missing",
                "h1_canonical_semantics_audit": "present" if h1_canonical_semantics_audit else "missing",
                "h1_top_example_inspection": "present" if h1_top_example_inspection else "missing",
                "h1_repairability_decision": "present" if h1_repairability_decision else "missing",
                "existing_capture_repairability_diagnostic": (
                    "present" if existing_capture_repairability_diagnostic else "missing"
                ),
                "fair_contract_training_preflight": (
                    fair_contract_training_preflight.get("status")
                    if fair_contract_training_preflight
                    else "missing"
                ),
                "fair_contract_training_design": (
                    fair_contract_training_design.get("status")
                    if fair_contract_training_design
                    else "missing"
                ),
                "fair_contract_training_dry_run": (
                    fair_contract_training_dry_run.get("status")
                    if fair_contract_training_dry_run
                    else "missing"
                ),
                "fair_contract_training_runner": (
                    fair_contract_training_runner_plan.get("status")
                    if fair_contract_training_runner_plan
                    else "missing"
                ),
                "fair_contract_candidate_validation_gate": (
                    fair_contract_candidate_validation_gate.get("status")
                    if fair_contract_candidate_validation_gate
                    else "missing"
                ),
                "fair_contract_selected_candidate_export": (
                    fair_contract_selected_candidate_export.get("status")
                    if fair_contract_selected_candidate_export
                    else "missing"
                ),
                "fair_contract_selected_candidate_replay_gate": (
                    fair_contract_selected_candidate_replay_gate.get("status")
                    if fair_contract_selected_candidate_replay_gate
                    else "missing"
                ),
                "q1_diagnostic_source_policy": (
                    "present" if diagnostic_source_policy_comparison else "missing"
                ),
                "q1_fair_timing_source_policy": (
                    "present" if fair_timing_source_policy_comparison else "missing"
                ),
                "fair_game_policy": fair_game_policy.status,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
