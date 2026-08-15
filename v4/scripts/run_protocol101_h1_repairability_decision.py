"""Decide whether H1 evidence justifies changing Protocol101's feature contract.

This script turns the H1 audits into a concrete synchronization decision. It is
offline-only and does not rebuild Q1, train, tune, contact vendors, or touch
broker/runtime defaults.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from v4.live.protocol101_synchronization import Protocol101RepairabilityDecisionV1


DEFAULT_H1_AUDIT = Path("v4/audit/autoresearch/protocol101_h1_canonical_semantics_audit/summary.json")
DEFAULT_TOP_EXAMPLES = Path("v4/audit/autoresearch/protocol101_h1_top_example_inspection/summary.json")
DEFAULT_SYNC_ROOT = Path("v4/audit/autoresearch/protocol101_synchronization_resolution")
DEFAULT_FAIR_TIMING_COMPARISON = Path(
    "v4/audit/autoresearch/protocol101_q1_fair_source_policy_lag0_decision_plus1/comparison/summary.json"
)
DEFAULT_OUT = Path("v4/audit/autoresearch/protocol101_h1_repairability_decision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h1-audit", type=Path, default=DEFAULT_H1_AUDIT)
    parser.add_argument("--top-examples", type=Path, default=DEFAULT_TOP_EXAMPLES)
    parser.add_argument("--sync-root", type=Path, default=DEFAULT_SYNC_ROOT)
    parser.add_argument("--fair-timing-comparison", type=Path, default=DEFAULT_FAIR_TIMING_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        return load_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def route_decisions(
    h1: dict[str, Any],
    top_examples: dict[str, Any],
    fair_timing: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    routes = {str(row.get("repair_route")): row for row in h1.get("repair_routes") or []}
    diff_counts = {
        str(row.get("repair_route")): int(row.get("diff_rows") or 0)
        for row in top_examples.get("diff_routes") or []
    }
    signatures = top_examples.get("signatures") or []
    top_signature = signatures[0] if signatures else {}
    fair_candidate = (fair_timing or {}).get("protocol101_live_v1") or {}
    fair_pnl = fair_candidate.get("total_pnl")
    fair_timing_rejected = fair_pnl is not None and float(fair_pnl) <= 0.0
    pattern_repairability = (
        "rejected_as_same_minute_completed_bar_repair"
        if fair_timing_rejected
        else "not_safe_to_mutate_contract_yet"
    )
    pattern_reason = (
        "The context-lag-0 diagnostic recovered score behavior, but the live-plausible "
        f"decision-plus-one replay produced ${float(fair_pnl):.0f}. That treats the "
        "same-minute completed-bar recovery as timing privilege, not a production repair."
        if fair_timing_rejected
        else (
            "Pattern tokens are causal only if built from the same completed-minute policy. "
            "The old legacy replay appears to use the decision minute, while live-v1 uses the "
            "prior completed minute. Simply switching live-v1 to decision-minute context could "
            "introduce lookahead unless the live recorder proves that minute is complete before inference."
        )
    )
    return [
        {
            "route": "canonical_pattern_semantics",
            "evidence": (
                f"{diff_counts.get('canonical_pattern_semantics', 0)} inspected top-example token diffs; "
                f"route lost-trade PnL exposure ${float((routes.get('canonical_pattern_semantics') or {}).get('lost_trade_pnl_exposure') or 0.0):.0f}; "
                f"top signature {top_signature.get('signature', 'UNKNOWN')}"
            ),
            "repairability": pattern_repairability,
            "reason": pattern_reason,
            "allowed_now": "inspection_and_diagnostic_rebuild_only",
            "blocked_change": (
                "do_not_change_protocol101_live_v1_timestamp_policy_to_same_minute_completed_bar"
                if fair_timing_rejected
                else "do_not_change_protocol101_live_v1_timestamp_policy_today"
            ),
        },
        {
            "route": "canonical_candidate_geometry",
            "evidence": (
                f"{diff_counts.get('canonical_candidate_geometry', 0)} inspected top-example diffs, "
                "all from distance_points/ATM-relative geometry."
            ),
            "repairability": "potentially_repairable_after_source_policy_proof",
            "reason": (
                "The same contract can have different distance_points when SPX context/ATM rounding uses "
                "a different minute. This is a real clue, but it should be repaired only by one shared "
                "SPX/ATM timestamp policy, not by per-example patching."
            ),
            "allowed_now": "audit_atm_rounding_and_spx_source_policy",
            "blocked_change": "do_not_recenter_ladder_to_match_legacy_without_live_causal_proof",
        },
        {
            "route": "runtime_zeroed_pressure_diagnostics",
            "evidence": (
                f"{diff_counts.get('runtime_zeroed_pressure_diagnostics', 0)} inspected top-example diffs; "
                "live-v1 intentionally zeroes option volume/open-interest fields."
            ),
            "repairability": "diagnostic_only_until_live_equivalent_proven",
            "reason": (
                "IBKR generic ticks 100/101 may be captured on July 6+, but they are not Databento "
                "one-minute OHLCV/statistics. Restoring historical pressure fields would recreate a "
                "non-live game unless publication timing and semantics are proven equivalent."
            ),
            "allowed_now": "capture_and_compare_diagnostics",
            "blocked_change": "do_not_restore_databento_volume_or_open_interest_to_live_inference",
        },
        {
            "route": "greeks_decay_quote_path",
            "evidence": f"{diff_counts.get('greeks_decay_quote_path', 0)} inspected top-example diffs.",
            "repairability": "not_a_contract_rescue_path",
            "reason": (
                "Live-v1 intentionally uses shared repaired Greeks. The old replay's vendor/row Greek "
                "semantics may be favorable, but that is not a safe live input unless reproduced causally."
            ),
            "allowed_now": "keep_repaired_greek_regression_tests",
            "blocked_change": "do_not_switch_live_inference_back_to_vendor_greeks",
        },
        {
            "route": "quote_tradability_guardrail",
            "evidence": f"{diff_counts.get('quote_tradability_guardrail', 0)} inspected top-example diffs.",
            "repairability": "guardrail_not_profit_repair",
            "reason": (
                "Spread/overpay/worth features are allowed to block trades. They should be compared for "
                "parity, but loosening them to recover old PnL would make execution less realistic."
            ),
            "allowed_now": "compare_mask_parity",
            "blocked_change": "do_not_loosen_tradability_masks_for_backtest_recovery",
        },
    ]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    h1 = load_json(args.h1_audit)
    top_examples = load_json(args.top_examples)
    fair_timing = load_json_optional(args.fair_timing_comparison)
    fair_candidate = (fair_timing or {}).get("protocol101_live_v1") or {}
    fair_pnl = fair_candidate.get("total_pnl")
    fair_timing_rejected = fair_pnl is not None and float(fair_pnl) <= 0.0
    routes = route_decisions(h1, top_examples, fair_timing)
    next_actions = [
        "Do not mutate protocol101-live-v1 toward same-minute completed-bar semantics.",
        "Keep diagnostic lag/source-policy rebuilds as audit artifacts, not as the new default contract.",
        "Use July 6+ recorder evidence for confirmation breadth, not as permission to revive rejected same-minute timing.",
        "Keep capturing IBKR generic ticks 100/101 as diagnostics, but do not use them as direct volume/OI replacements.",
        "If diagnostic repair cannot restore Q1 non-inferiority causally, freeze the canonical live-reproducible contract and plan retraining under that contract.",
    ]
    packet = Protocol101RepairabilityDecisionV1(
        status=(
            "same_minute_completed_bar_repair_rejected"
            if fair_timing_rejected
            else "contract_change_blocked_pending_causal_source_policy_proof"
        ),
        decision=(
            "do_not_update_shared_feature_contract_to_same_minute_completed_bar_semantics"
            if fair_timing_rejected
            else "do_not_update_shared_feature_contract_or_rerun_q1_as_promotion_claim_yet"
        ),
        immediate_contract_change_allowed=False,
        q1_rerun_required_now=False,
        route_decisions=routes,
        next_actions=next_actions,
    )
    (args.out_dir / "repairability_decision.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.out_dir / "route_decisions.csv", routes)
    headline = h1.get("headline") or {}
    signatures = top_examples.get("signatures") or []
    top_signature = signatures[0] if signatures else {}
    report = [
        "# Protocol101 H1 Repairability Decision",
        "",
        "## Decision",
        "",
        f"- Status: `{packet.status}`",
        f"- Decision: `{packet.decision}`",
        f"- Immediate contract change allowed: `{str(packet.immediate_contract_change_allowed).lower()}`",
        f"- Q1 rerun required now: `{str(packet.q1_rerun_required_now).lower()}`",
        "- Model training: `false`",
        "- Threshold tuning: `false`",
        "- Broker endpoint called: `false`",
        "",
        "## Why",
        "",
        (
            f"- Q1 live-v1 degradation remains `${float(headline.get('net_degradation') or 0.0):,.0f}` "
            "versus same-runner legacy."
        ),
        (
            "- Top lost examples signature: "
            f"`{top_signature.get('signature', 'UNKNOWN')}` across "
            f"`{top_signature.get('examples', 'UNKNOWN')}` examples with "
            f"`${float(top_signature.get('pnl') or 0.0):,.0f}` PnL exposure."
        ),
        (
            "- The biggest lost trades had the same contract and effectively the same quote, "
            "so the immediate problem is feature/token semantics, not missing contracts."
        ),
        (
            "- The strongest apparent repair, using the decision minute's pattern/index/ATM state, "
            + (
                f"failed the fair timing test with ${float(fair_pnl):,.0f} PnL."
                if fair_timing_rejected
                else "is only safe if the live recorder proves that state is complete before inference."
            )
        ),
        "",
        "## Route Decisions",
        "",
    ]
    for row in routes:
        report.extend(
            [
                f"### {row['route']}",
                "",
                f"- Repairability: `{row['repairability']}`",
                f"- Evidence: {row['evidence']}",
                f"- Reason: {row['reason']}",
                f"- Allowed now: `{row['allowed_now']}`",
                f"- Blocked change: `{row['blocked_change']}`",
                "",
            ]
        )
    report.extend(["## Next Actions", ""])
    report.extend(f"- {item}" for item in next_actions)
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(packet.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
