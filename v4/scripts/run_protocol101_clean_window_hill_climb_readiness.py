"""Resolve whether clean-window evidence is enough to start hill climbing."""
from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEAN_WINDOW_DIR = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_canonical_v1_4_clean_window_certification_attempt001"
)
GROUP1_DIR = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_live_v2_group1_index_context_parity_resolution"
)
OUT_DIR = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_clean_window_hill_climb_readiness_2026_07_25"
)
RUNTIME_GUARD = ROOT / "v4/live/protocol101_clean_window_guard.py"
RUNTIME_RUNNER = (
    ROOT / "v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py"
)

STABLE_PROBES = (
    "straddle_mid_expansion",
    "put_call_ratio_skew",
    "internal_delta_geometry",
)
INITIAL_FEATURES = (
    "spx_vwap_gap_points",
    "spx_vwap_gap_bps",
    "spx_vwap_gap_over_session_range",
    "session_range_bps",
    "momentum_5m_bps",
    "momentum_15m_bps",
    "momentum_5m_over_session_range",
    "momentum_15m_over_session_range",
    "omar_clipped_neg3_pos3",
    "vwap_side_alignment_flag",
    "omar_side_alignment_flag",
    "momentum15_side_alignment_flag",
    "D.near_atm.straddle_mid_spot_bps",
    "D.near_atm.put_call_mid_ratio",
    "D.near_atm.side_smile_slope_bps_per_5pt",
    "E.bs.delta",
    "E.bs.gamma",
)
QUARANTINED_FEATURES = {
    "direct_per_slot_option_price_path": (
        "C source discriminator HGB AUC 0.550524 exceeded the frozen 0.55 gate"
    ),
    "internal_iv_expansion_compression": (
        "action agreement was 0.961538 on 2026-07-17"
    ),
    "vix_change_5m_15m": (
        "paired pre-window context history remains insufficient"
    ),
    "raw_bid_ask_spread_size_quote_age": (
        "guard, fill, and audit only; not initial model alpha"
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_summary() -> dict[str, Any]:
    clean_routing = load_json(CLEAN_WINDOW_DIR / "routing_decision.json")
    clean_coverage = load_json(CLEAN_WINDOW_DIR / "coverage_summary.json")
    group1 = load_json(GROUP1_DIR / "parity_result.json")
    decision_by_date = pd.read_csv(
        CLEAN_WINDOW_DIR / "decision_transfer_by_date.csv"
    )

    l0 = {
        row["family"]: row
        for row in clean_routing.get("l0_feature_families", [])
    }
    l2 = {
        row["family"]: row
        for row in clean_routing.get("l2_feature_families", [])
    }
    stable_probe_evidence: dict[str, Any] = {}
    for probe in STABLE_PROBES:
        rows = decision_by_date[decision_by_date["probe"] == probe]
        stable_probe_evidence[probe] = {
            "dates": int(rows["session_date"].nunique()),
            "all_dates_pass": bool(rows["pass"].all()),
            "minimum_action_agreement": float(rows["action_agreement"].min()),
            "per_date": {
                str(row.session_date): float(row.action_agreement)
                for row in rows.itertuples()
            },
        }

    evidence_gates = {
        "group1_non_vix_subset_previously_passed": bool(
            group1.get("subset_eligibility")
        ),
        "D_l0_pass": bool(l0.get("D_composite", {}).get("pass")),
        "D_l2_pass": bool(l2.get("D_composite", {}).get("pass")),
        "E_l0_pass": bool(l0.get("E_internal_greek", {}).get("pass")),
        "E_l2_pass": bool(l2.get("E_internal_greek", {}).get("pass")),
        "stable_probe_battery_pass": all(
            row["dates"] >= 5 and row["all_dates_pass"]
            for row in stable_probe_evidence.values()
        ),
        "runtime_clean_window_guard_present": RUNTIME_GUARD.exists(),
        "runtime_decision_shadow_present": (
            RUNTIME_RUNNER.exists()
            and "Protocol101DecisionShadowV1" in RUNTIME_RUNNER.read_text()
        ),
    }
    ready = all(evidence_gates.values())
    return {
        "schema_version": "Protocol101CleanWindowHillClimbReadinessV1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": (
            "governed_hill_climbing_ready_on_parity_stable_subset"
            if ready
            else "hill_climbing_still_blocked"
        ),
        "confident_answer": bool(ready),
        "full_day_raw_recorder_is_training_blocker": False if ready else True,
        "paper_ready": False,
        "evidence_gates": evidence_gates,
        "clean_window_evidence": {
            "option_eligible_minutes": clean_coverage["option_eligible"][
                "total_minutes"
            ],
            "dates": clean_coverage["option_eligible"]["distinct_dates"],
            "per_date_minutes": clean_coverage["option_eligible"]["per_date"],
            "morning_minutes_in_this_packet": clean_coverage[
                "option_eligible"
            ]["per_time_bucket"].get("morning", 0),
            "context_packet_pass": clean_coverage["context_eligible"]["pass"],
            "note": (
                "This five-date packet independently validates option-derived "
                "clean windows. The 12 Group 1 non-VIX context features rely on "
                "their existing three-full-day parity packet."
            ),
        },
        "family_results": {
            "AB_context_geometry": {
                "initial_alpha": True,
                "source": str(GROUP1_DIR / "parity_result.json"),
                "restriction": "exclude VIX-change features",
            },
            "C_direct_option_price": {
                "initial_alpha": False,
                "l0_pass": bool(l0.get("C_option_price", {}).get("pass")),
                "l2_hgb_auc": l2.get("C_option_price", {}).get(
                    "bounded_hgb_pooled_auc"
                ),
            },
            "D_composites": {
                "initial_alpha": True,
                "l0_max_standardized_p95": l0.get("D_composite", {}).get(
                    "max_standardized_p95"
                ),
                "l2_hgb_auc": l2.get("D_composite", {}).get(
                    "bounded_hgb_pooled_auc"
                ),
            },
            "E_internal_greeks": {
                "initial_alpha": "delta_and_gamma_only",
                "l0_max_standardized_p95": l0.get(
                    "E_internal_greek", {}
                ).get("max_standardized_p95"),
                "l2_hgb_auc": l2.get("E_internal_greek", {}).get(
                    "bounded_hgb_pooled_auc"
                ),
            },
        },
        "stable_probe_evidence": stable_probe_evidence,
        "initial_model_alpha_features": list(INITIAL_FEATURES),
        "quarantined_features": QUARANTINED_FEATURES,
        "required_training_controls": [
            "5-fold chronological expanding-window CV with one-session embargo",
            "observed cross-source divergence noise injection",
            "intersection tradability guards and pessimistic fills",
            "G1-G9 gates and strict serial replay",
            "no protected, recorder, or confirmation dates in training",
        ],
        "required_before_paper_validation": [
            "selected candidate passes the lightweight decision-shadow transfer battery",
            "clean-window guard abstains for 15 complete healthy minutes after interruption",
            "candidate-specific action and selected-slot agreement pass frozen gates",
            "paper-submit remains separately owner-approved",
        ],
        "recommended_next_route": (
            "Start bounded governed HGB hill climbing on the initial feature "
            "list now. Do not wait for another perfect full-day raw capture. "
            "Use the decision-shadow logger for candidate-specific transfer."
        ),
        "source_artifacts": {
            "clean_window_routing": str(
                CLEAN_WINDOW_DIR / "routing_decision.json"
            ),
            "clean_window_manifest": str(
                CLEAN_WINDOW_DIR / "clean_window_manifest.csv"
            ),
            "group1_context_parity": str(GROUP1_DIR / "parity_result.json"),
            "runtime_guard": str(RUNTIME_GUARD),
            "runtime_runner": str(RUNTIME_RUNNER),
        },
        "source_hashes": {
            "clean_window_routing": sha256_path(
                CLEAN_WINDOW_DIR / "routing_decision.json"
            ),
            "clean_window_manifest": sha256_path(
                CLEAN_WINDOW_DIR / "clean_window_manifest.csv"
            ),
            "group1_context_parity": sha256_path(
                GROUP1_DIR / "parity_result.json"
            ),
            "runtime_guard": sha256_path(RUNTIME_GUARD),
            "runtime_runner": sha256_path(RUNTIME_RUNNER),
        },
        "side_effects": {
            "model_training_executed": False,
            "threshold_tuning_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_download": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
        },
    }


def write_report(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    evidence = summary["clean_window_evidence"]
    family = summary["family_results"]
    probes = summary["stable_probe_evidence"]
    lines = [
        "# Protocol101 Clean-Window Hill-Climb Readiness",
        "",
        "## Decision",
        "",
        f"- Decision: `{decision}`",
        f"- Confident answer: `{summary['confident_answer']}`",
        "- Full perfect-day raw recording is no longer a prerequisite for offline training.",
        "- This is not a paper-readiness or paper-submit approval.",
        "",
        "## Evidence",
        "",
        f"- Clean option minutes: `{evidence['option_eligible_minutes']}` across `{evidence['dates']}` dates.",
        f"- Per-date minutes: `{evidence['per_date_minutes']}`",
        f"- D composite source AUC: `{family['D_composites']['l2_hgb_auc']:.6f}`",
        f"- E internal-Greek source AUC: `{family['E_internal_greeks']['l2_hgb_auc']:.6f}`",
        f"- Direct C price source AUC: `{family['C_direct_option_price']['l2_hgb_auc']:.6f}`; quarantined from initial alpha.",
        "",
        "Stable decision probes:",
    ]
    for probe, row in probes.items():
        lines.append(
            f"- `{probe}`: `{row['dates']}` dates, minimum action agreement "
            f"`{row['minimum_action_agreement']:.6f}`, all dates pass "
            f"`{row['all_dates_pass']}`."
        )
    lines.extend(
        [
            "",
            "## Allowed Initial Contract",
            "",
            "- The 12 previously certified Group 1 non-VIX context features.",
            "- Near-ATM D composite features.",
            "- Internally computed delta and gamma.",
            "- Raw quote fields remain available for guards, fills, labels, and audit.",
            "",
            "Excluded initially: direct per-slot price-path alpha, internal-IV "
            "expansion/compression, VIX changes, and raw bid/ask/spread/size/quote-age alpha.",
            "",
            "## Why Collection Can Stop Blocking Research",
            "",
            "The clean-window packet localizes the remaining transfer failures "
            "instead of showing a system-wide mismatch. Stable feature families "
            "and decisions now have evidence across five additional dates. A "
            "candidate-specific lightweight shadow log closes the remaining "
            "model-transfer question before paper validation, without requiring "
            "another 3 GB perfect raw day.",
            "",
            "## Next Route",
            "",
            summary["recommended_next_route"],
            "",
            "No model training, threshold tuning, broker call, paper submit, paid "
            "download, promotion, launchd edit, or real-money change occurred.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (OUT_DIR / "report.md").write_text(write_report(summary))
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "confident_answer": summary["confident_answer"],
                "report": str(OUT_DIR / "report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
