"""Model-free FT1C selection across the frozen 28-row Stage-1 family."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from v4.model.protocol101_stage1_controller_journal import (
    ControllerJournalError,
    JOURNAL_NODES,
    file_sha256,
    validate_journal,
)
from v4.model.protocol101_stage1_gate_contract import (
    AUDIT_SCHEMA,
    HYPOTHESES,
    INDEPENDENT_AUDIT_FREEZE_SCHEMA,
    POLICIES,
    ROWS,
    SELECTION_SCHEMA,
    stable_hash,
)


SELECTED_ROUTE = "selected_candidate_awaiting_owner_authorized_G9"
ATTRIBUTION_ROUTE = (
    "multiplicity_adjusted_real_signal_requires_fixed_exit_"
    "binding_attribution"
)
REGIME_ROUTE = "regime_bound_requires_owner_review"
STOP_ROUTE = "stage1_no_accepted_edge_stop_and_redesign"
INVALID_ROUTE = "campaign_invalid_no_selection"

SELECTION_RULE = {
    "primary": (
        "descending_median_seed_fee_adjusted_continuous_"
        "strict_serial_net_pnl"
    ),
    "tie_break": "H0_H1_H2_H3_then_P0_through_P6",
    "maximum_selected_candidates": 1,
    "rank_excludes": [
        "ECE",
        "win_rate",
        "drawdown",
        "feature_count",
        "model_complexity",
        "risk_adjusted_utility",
        "post_result_preference",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--controller-journal", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--campaign-namespace", required=True)
    parser.add_argument("--campaign-execution-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _invalid(blockers: list[str]) -> dict[str, Any]:
    result = {
        "schema_version": SELECTION_SCHEMA,
        "valid": False,
        "routing_decision": INVALID_ROUTE,
        "selected_candidate": None,
        "eligible_row_count": 0,
        "adjusted_signal_row_count": 0,
        "ranked_eligible_rows": [],
        "selection_rule": SELECTION_RULE,
        "blockers": sorted(set(blockers)),
        "G8_used_for_eligibility_or_ranking": False,
        "G9_executed": False,
        "selection_sha256": None,
    }
    result["selection_sha256"] = stable_hash(
        {**result, "selection_sha256": None}
    )
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _freeze_blockers(audit_result: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    freeze = audit_result.get("freeze")
    freeze_sha256 = audit_result.get("freeze_sha256")
    if not isinstance(freeze, Mapping):
        return ["audit_freeze_missing"]
    if not _is_sha256(freeze_sha256):
        blockers.append("audit_freeze_hash_malformed")
    elif stable_hash(freeze) != freeze_sha256:
        blockers.append("audit_freeze_hash_mismatch")
    required_fields = {
        "schema_version",
        "audit_route",
        "campaign_packet_sha256",
        "execution_provenance_authority_sha256",
        "control_authority_sha256",
        "producer_aggregation_sha256",
        "independent_aggregation_sha256",
        "row_count",
        "row_result_sha256",
        "independent_result_sha256",
        "source_hashes",
        "agreement",
    }
    if set(freeze) != required_fields:
        blockers.append("audit_freeze_schema_fields_mismatch")
    if freeze.get("schema_version") != INDEPENDENT_AUDIT_FREEZE_SCHEMA:
        blockers.append("audit_freeze_schema_mismatch")
    if freeze.get("audit_route") != "fresh_28_row_independent_audit_accepted":
        blockers.append("audit_freeze_route_mismatch")
    if freeze.get("agreement") is not True:
        blockers.append("audit_freeze_agreement_missing")
    independent = audit_result.get("independent_result")
    producer = audit_result.get("producer_result")
    if not isinstance(independent, Mapping):
        blockers.append("independent_result_missing")
        return blockers
    if not isinstance(producer, Mapping):
        blockers.append("producer_result_missing")
        producer = {}
    hashes = (
        "campaign_packet_sha256",
        "execution_provenance_authority_sha256",
        "control_authority_sha256",
        "producer_aggregation_sha256",
        "independent_aggregation_sha256",
        "row_result_sha256",
        "independent_result_sha256",
    )
    if any(not _is_sha256(freeze.get(field)) for field in hashes):
        blockers.append("audit_freeze_binding_hash_malformed")
    if freeze.get("campaign_packet_sha256") != independent.get(
        "campaign_packet_sha256"
    ):
        blockers.append("audit_freeze_campaign_hash_mismatch")
    if freeze.get("execution_provenance_authority_sha256") != audit_result.get(
        "execution_provenance_authority_sha256"
    ) or freeze.get("execution_provenance_authority_sha256") != independent.get(
        "execution_provenance_authority_sha256"
    ):
        blockers.append("audit_freeze_execution_authority_mismatch")
    if freeze.get("control_authority_sha256") != audit_result.get(
        "control_authority_sha256"
    ) or freeze.get("control_authority_sha256") != independent.get(
        "control_authority_sha256"
    ):
        blockers.append("audit_freeze_control_authority_mismatch")
    if freeze.get("producer_aggregation_sha256") != producer.get(
        "aggregation_sha256"
    ):
        blockers.append("audit_freeze_producer_aggregation_mismatch")
    if freeze.get("independent_aggregation_sha256") != independent.get(
        "aggregation_sha256"
    ):
        blockers.append("audit_freeze_independent_aggregation_mismatch")
    rows = independent.get("rows")
    if freeze.get("row_count") != 28 or not isinstance(rows, list):
        blockers.append("audit_freeze_row_count_mismatch")
    elif freeze.get("row_result_sha256") != stable_hash(rows):
        blockers.append("audit_freeze_row_payload_mismatch")
    if freeze.get("independent_result_sha256") != stable_hash(independent):
        blockers.append("audit_freeze_independent_payload_mismatch")
    source_hashes = freeze.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or not source_hashes
        or any(
            not isinstance(path, str) or not _is_sha256(value)
            for path, value in source_hashes.items()
        )
    ):
        blockers.append("audit_freeze_source_hashes_invalid")
    return blockers


def _journal_blockers(
    *,
    audit_result: Mapping[str, Any],
    controller_journal_path: Path | None,
    audit_result_path: Path | None,
    workspace_root: Path | None,
    campaign_namespace: str | None,
    campaign_execution_id: str | None,
) -> tuple[list[str], dict[str, Any] | None]:
    blockers: list[str] = []
    if controller_journal_path is None:
        blockers.append("controller_journal_required")
    if audit_result_path is None:
        blockers.append("journal_bound_audit_result_path_required")
    if workspace_root is None:
        blockers.append("journal_workspace_root_required")
    if not isinstance(campaign_namespace, str) or not campaign_namespace:
        blockers.append("journal_campaign_namespace_required")
    if not isinstance(campaign_execution_id, str) or not campaign_execution_id:
        blockers.append("journal_campaign_execution_id_required")
    if blockers:
        return blockers, None
    assert controller_journal_path is not None
    assert audit_result_path is not None
    assert workspace_root is not None
    try:
        state = validate_journal(
            controller_journal_path,
            workspace_root=workspace_root,
            expected_campaign_namespace=campaign_namespace,
            expected_campaign_execution_id=campaign_execution_id,
        )
    except ControllerJournalError as exc:
        return [f"controller_journal_invalid:{exc}"], None
    expected_prefix = list(JOURNAL_NODES[:7])
    if state["completed_prefix"] != expected_prefix:
        blockers.append("journal_not_exactly_at_independent_audit_checkpoint")
        return blockers, state
    binding = state["artifact_bindings"].get("INDEPENDENT_AUDIT")
    if not isinstance(binding, Mapping):
        blockers.append("journal_independent_audit_binding_missing")
        return blockers, state
    try:
        root = workspace_root.resolve()
        resolved = audit_result_path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (FileNotFoundError, ValueError):
        blockers.append("audit_result_path_outside_workspace_or_missing")
        return blockers, state
    if relative != binding.get("artifact_path"):
        blockers.append("audit_result_path_not_journaled")
    if file_sha256(resolved) != binding.get("artifact_sha256"):
        blockers.append("audit_result_file_hash_not_journaled")
    try:
        disk_payload = load_json(resolved)
    except (OSError, json.JSONDecodeError):
        blockers.append("audit_result_file_malformed")
    else:
        if dict(audit_result) != disk_payload:
            blockers.append("audit_result_payload_differs_from_journaled_file")
    return blockers, state


def select_candidate(
    audit_result: Mapping[str, Any],
    *,
    controller_journal_path: Path | None = None,
    audit_result_path: Path | None = None,
    workspace_root: Path | None = None,
    campaign_namespace: str | None = None,
    campaign_execution_id: str | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    journal_blockers, journal_state = _journal_blockers(
        audit_result=audit_result,
        controller_journal_path=controller_journal_path,
        audit_result_path=audit_result_path,
        workspace_root=workspace_root,
        campaign_namespace=campaign_namespace,
        campaign_execution_id=campaign_execution_id,
    )
    blockers.extend(journal_blockers)
    if audit_result.get("schema_version") != AUDIT_SCHEMA:
        blockers.append("audit_schema_mismatch")
    if audit_result.get("accepted") is not True:
        blockers.append("independent_audit_not_accepted")
    blockers.extend(_freeze_blockers(audit_result))
    independent = audit_result.get("independent_result")
    if not isinstance(independent, Mapping):
        blockers.append("independent_result_missing")
        return _invalid(blockers)
    rows = independent.get("rows")
    if not isinstance(rows, list) or [
        row.get("row_id") for row in rows if isinstance(row, Mapping)
    ] != list(ROWS):
        blockers.append("independent_28_row_grid_mismatch")
    controls = independent.get("global_controls")
    if not isinstance(controls, Mapping) or controls.get(
        "all_global_controls_pass"
    ) is not True:
        blockers.append("global_controls_invalid")
    if audit_result.get("campaign_valid") is not True:
        blockers.append("campaign_invalid")
    if blockers:
        return _invalid(blockers)

    eligible: list[Mapping[str, Any]] = []
    signal_rows: list[Mapping[str, Any]] = []
    g6_only: list[Mapping[str, Any]] = []
    for row in rows:
        pnl = row.get(
            "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
        )
        if not isinstance(pnl, (int, float)) or not math.isfinite(float(pnl)):
            return _invalid([f"nonfinite_selection_pnl:{row.get('row_id')}"])
        if row.get("hard_gate_eligible") is True:
            eligible.append(row)
        if row.get("multiplicity_adjusted_real_entry_signal") is True:
            signal_rows.append(row)
        if row.get("G6_only_blocked") is True:
            g6_only.append(row)
    hypothesis_rank = {value: index for index, value in enumerate(HYPOTHESES)}
    policy_rank = {value: index for index, value in enumerate(POLICIES)}
    eligible.sort(
        key=lambda row: (
            -float(
                row[
                    "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
                ]
            ),
            hypothesis_rank[str(row["hypothesis"])],
            policy_rank[str(row["policy"])],
        )
    )
    ranked = [
        {
            "rank": index,
            "row_id": row["row_id"],
            "hypothesis": row["hypothesis"],
            "policy": row["policy"],
            "median_seed_fee_adjusted_continuous_strict_serial_net_pnl": row[
                "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
            ],
        }
        for index, row in enumerate(eligible, start=1)
    ]
    selected = ranked[0] if ranked else None
    if selected is not None:
        route = SELECTED_ROUTE
    elif g6_only:
        route = REGIME_ROUTE
    elif signal_rows:
        route = ATTRIBUTION_ROUTE
    else:
        route = STOP_ROUTE
    result = {
        "schema_version": SELECTION_SCHEMA,
        "valid": True,
        "routing_decision": route,
        "selected_candidate": selected,
        "eligible_row_count": len(eligible),
        "adjusted_signal_row_count": len(signal_rows),
        "G6_only_blocked_row_count": len(g6_only),
        "ranked_eligible_rows": ranked,
        "selection_rule": SELECTION_RULE,
        "blockers": [],
        "audit_freeze_sha256": audit_result["freeze_sha256"],
        "controller_journal_head_sha256": journal_state[
            "journal_head_sha256"
        ],
        "journaled_independent_audit_sha256": journal_state[
            "artifact_bindings"
        ]["INDEPENDENT_AUDIT"]["artifact_sha256"],
        "G8_used_for_eligibility_or_ranking": False,
        "G9_executed": False,
        "selection_sha256": None,
    }
    result["selection_sha256"] = stable_hash(
        {**result, "selection_sha256": None}
    )
    return result


def compare(
    *,
    audit_result: Mapping[str, Any],
    controller_journal_path: Path | None = None,
    audit_result_path: Path | None = None,
    workspace_root: Path | None = None,
    campaign_namespace: str | None = None,
    campaign_execution_id: str | None = None,
) -> dict[str, Any]:
    return select_candidate(
        audit_result,
        controller_journal_path=controller_journal_path,
        audit_result_path=audit_result_path,
        workspace_root=workspace_root,
        campaign_namespace=campaign_namespace,
        campaign_execution_id=campaign_execution_id,
    )


def selection_contract() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101FT1CSelectionContractV1",
        "selection_rule": SELECTION_RULE,
        "routes": [
            SELECTED_ROUTE,
            ATTRIBUTION_ROUTE,
            REGIME_ROUTE,
            STOP_ROUTE,
            INVALID_ROUTE,
        ],
        "requires_independent_28_row_freeze": True,
        "requires_exact_controller_journal_independent_audit_binding": True,
        "G8_role": "report_only",
        "G9_run": False,
    }


def main() -> int:
    args = parse_args()
    result = select_candidate(
        load_json(args.audit_result),
        controller_journal_path=args.controller_journal,
        audit_result_path=args.audit_result,
        workspace_root=args.workspace_root,
        campaign_namespace=args.campaign_namespace,
        campaign_execution_id=args.campaign_execution_id,
    )
    write_json(args.out_dir / "selection.json", result)
    write_json(
        args.out_dir / "summary.json",
        {
            "routing_decision": result["routing_decision"],
            "selected_candidate": result["selected_candidate"],
            "eligible_row_count": result["eligible_row_count"],
            "G9_executed": False,
        },
    )
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
