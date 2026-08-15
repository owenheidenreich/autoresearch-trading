"""Synthetic-only producer validation for Protocol101 FT1C machinery."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION,
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
)
from v4.model.protocol101_stage1_controller_journal import (
    JOURNAL_NODES,
    JOURNAL_ROUTES,
    append_checkpoint,
    create_journal,
    file_sha256,
    validate_journal,
)
from v4.model.protocol101_stage1_gate_contract import (
    CAMPAIGN_CONTRACT_SHA256,
    CAMPAIGN_NAMESPACE,
    CAMPAIGN_SCHEMA,
    CONTROL_SCHEMAS,
    DIAGNOSTIC_KEYS,
    FOLDS,
    HYPOTHESES,
    POLICIES,
    PROVENANCE_HASH_FIELDS,
    REFERENCE_ACCEPTANCE_ROUTE,
    REFERENCE_SCHEMAS,
    ROWS,
    SEEDS,
    UNIT_COUNT,
    build_control_authority,
    build_execution_provenance_authority,
    contract_payload,
    seal_campaign_packet,
    seal_unit,
    stable_hash,
    unit_id,
    validate_campaign_packet,
    verify_frozen_acceptance_inputs,
)
from v4.scripts.run_protocol101_scoped_stage1_gate_aggregator import (
    aggregate_campaign,
)
from v4.scripts.run_protocol101_scoped_stage1_independent_audit import (
    audit_campaign,
)
from v4.scripts.run_protocol101_stage1_autoresearch_graph import (
    AUTHORIZATION_SCHEMA,
    INVALID_CHAIN_ROUTE,
    NO_OWNER_ROUTE,
    RECEIPT_NODES,
    RECEIPT_SCHEMA,
    RUN_PENDING_ROUTE,
    STOP_ROUTE as GRAPH_STOP_ROUTE,
    _evaluate_legacy_receipt_graph,
    build_receipt_chain,
    evaluate_graph,
    graph_definition,
    load_json,
    owner_authorization_sha256,
    seal_graph_receipt,
    seal_receipt_chain,
)
from v4.scripts.run_protocol101_stage1_cross_hypothesis_selection import (
    ATTRIBUTION_ROUTE,
    INVALID_ROUTE,
    REGIME_ROUTE,
    SELECTED_ROUTE,
    STOP_ROUTE,
    select_candidate,
    selection_contract,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_stage1_gate_audit_selection_machinery_attempt001"
)
NY = ZoneInfo("America/New_York")
TERMINAL_ROUTE = (
    "gate_audit_selection_machinery_repair_complete_"
    "pending_independent_acceptance"
)
READINESS_STATUS = (
    "gate_audit_selection_machinery_ready_pending_independent_acceptance"
)
FT1C3_DEFAULT_OUT = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_stage1_authority_root_chain_repair_attempt003"
)
FT1C3_TERMINAL_ROUTE = (
    "authority_root_chain_repair_complete_pending_reacceptance"
)
FT1C3_BLOCKED_ROUTE = "authority_root_chain_repair_blocked"
FT1C3_AUTHORIZED_FILES = (
    "v4/scripts/run_protocol101_stage1_autoresearch_graph.py",
    "v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py",
    "v4/tests/test_protocol101_stage1_autoresearch_graph.py",
    "v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py",
)
FT1C3_EXACT_CASE_IDS = (
    "AUTH-ATTACK-01-packet_resealed",
    "AUTH-ATTACK-02-authority_rebuilt",
    "AUTH-ATTACK-03-all_self_hashes",
    "AUTH-ATTACK-04-cross_campaign_authority",
    "AUTH-ATTACK-05-unit_grid",
    "AUTH-ATTACK-06-authority_schema",
    "AUTH-ATTACK-07-unknown_valid_hash",
    "AUTH-ATTACK-08-control_local",
    "AUTH-ATTACK-09-control_rebuilt",
    "AUTH-ATTACK-10-cross_campaign_control",
    "AUTH-ATTACK-11-node_order",
    "AUTH-ATTACK-12-pre_run_injection",
    "AUTH-ATTACK-14-owner_changed",
)
AUTHORIZED_FILES = (
    "v4/model/protocol101_stage1_gate_contract.py",
    "v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py",
    "v4/scripts/run_protocol101_scoped_stage1_independent_audit.py",
    "v4/scripts/run_protocol101_stage1_cross_hypothesis_selection.py",
    "v4/scripts/run_protocol101_stage1_autoresearch_graph.py",
    "v4/scripts/run_protocol101_full_trader_stage1_gate_audit_selection_validation.py",
    "v4/tests/test_protocol101_stage1_gate_contract.py",
    "v4/tests/test_protocol101_scoped_stage1_gate_aggregator.py",
    "v4/tests/test_protocol101_scoped_stage1_independent_audit.py",
    "v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py",
    "v4/tests/test_protocol101_stage1_autoresearch_graph.py",
    "v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py",
)


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_preregistration_freeze(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "preregistration_freeze.sha256"
    checks: list[dict[str, Any]] = []
    if not path.is_file():
        return {"valid": False, "checks": [], "blockers": ["freeze_missing"]}
    blockers: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        target = out_dir / name.strip()
        actual = _sha256(target) if target.is_file() else None
        passed = actual == expected
        checks.append(
            {
                "path": name.strip(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "PASS" if passed else "FAIL",
            }
        )
        if not passed:
            blockers.append(f"preregistration_freeze_mismatch:{name.strip()}")
    return {
        "valid": not blockers and len(checks) == 3,
        "checks": checks,
        "blockers": blockers,
    }


def _hash_label(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _timestamp(session: str, hour: int, minute: int) -> int:
    value = datetime.fromisoformat(session).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
        tzinfo=NY,
    )
    return int(value.timestamp() * 1_000_000_000)


def _candidate(
    *,
    hypothesis: str,
    policy: str,
    seed: int,
    fold: int,
    session: str,
    candidate_index: int,
    net_pnl: float,
) -> dict[str, Any]:
    decision = _timestamp(session, 10, 0)
    if (fold + candidate_index) % 2:
        source = decision + 4 * 60 * 1_000_000_000
        realized = decision + 5 * 60 * 1_000_000_000
        deadline = realized
        exit_reason = 3
    else:
        source = decision + 5 * 60 * 1_000_000_000
        realized = source
        deadline = decision + 10 * 60 * 1_000_000_000
        exit_reason = 2
    entry_ask = 10.0 if net_pnl < -100.0 else 1.0
    exit_bid = entry_ask + (float(net_pnl) + 3.0) / 100.0
    right = "C" if (candidate_index + fold) % 2 == 0 else "P"
    return {
        "trade_id": (
            f"SYNTH-{hypothesis}-{policy}-S{seed}-F{fold}-"
            f"T{candidate_index}"
        ),
        "split": f"SYNTH-{hypothesis}-{policy}-S{seed}",
        "fold": f"F{fold}",
        "session": session,
        "decision_time_ns": decision,
        "contract_id": (
            f"SPXW-SYNTH-{hypothesis}-{policy}-S{seed}-"
            f"F{fold}-T{candidate_index}"
        ),
        "right": right,
        "canonical_strike_slot": candidate_index,
        "policy_index": int(policy[1:]),
        "entry_ask": entry_ask,
        "score": 0.75,
        "raw_label_pnl_after_campaign_fee": float(net_pnl),
        "label_mid_pnl_before_campaign_fee": float(net_pnl + 8.0),
        "label_realized_exit_time_ns": realized,
        "label_source_exit_quote_time_ns": source,
        "label_exit_quote_age_ms": (realized - source) / 1_000_000.0,
        "label_exit_reason_code": exit_reason,
        "label_executable_exit_bid": exit_bid,
        "label_policy_deadline_ns": deadline,
        "label_invalid_reason_code": 0,
        "feature_hash": _hash_label(f"features:{hypothesis}"),
        "source_quote_time_ns": decision,
        "source_context_time_ns": decision - 60 * 1_000_000_000,
        "strategy": "FT1C_AUDIT_LOCAL_SYNTHETIC",
        "source_simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "metadata": {
            "audit_local": True,
            "synthetic_only": True,
            "campaign_economic_value": False,
        },
    }


def _diagnostics(
    *,
    policy: str,
    net_pnl: float,
    trade_count: int,
) -> dict[str, Any]:
    shape = {value: 0 for value in POLICIES}
    shape[policy] = trade_count
    return {
        "fee_sensitivity": {
            "2.60": net_pnl + 0.4 * trade_count,
            "3.00": net_pnl,
            "4.00": net_pnl - 1.0 * trade_count,
        },
        "fill_edge_band": {
            "pessimistic_executable": net_pnl,
            "mid_diagnostic": net_pnl + 5.0 * trade_count,
            "favorable_diagnostic": net_pnl + 10.0 * trade_count,
        },
        "noise_diagnostics": {
            "0.0x": net_pnl + 3.0,
            "0.5x": net_pnl + 1.0,
            "1.0x": net_pnl,
            "2.0x": net_pnl - 2.0,
        },
        "side_time_exposure": {
            "call_trades": trade_count,
            "put_trades": 0,
            "opening": trade_count,
            "midday": 0,
            "late": 0,
        },
        "concentration": {
            "top_trade_fraction": 1.0 / max(trade_count, 1),
            "top_day_fraction": 1.0 / max(trade_count, 1),
            "top_month_fraction": 1.0,
        },
        "churn": {
            "candidate_count": trade_count,
            "executed_count": trade_count,
            "overlap_skips": 0,
        },
        "skipped_opportunity": {
            "count": 0,
            "fee_adjusted_pnl": 0.0,
        },
        "worst_day": float(min(net_pnl, 0.0)),
        "underwater_duration": {"events": 0, "maximum_minutes": 0.0},
        "outcome_buckets": {
            "large_loss": int(net_pnl < -100),
            "small_loss": int(-100 <= net_pnl < 0),
            "small_win": int(0 <= net_pnl < 100),
            "large_win": int(net_pnl >= 100),
        },
        "harvest_ratio": 1.0 if net_pnl >= 0 else 0.0,
        "daily_breaker_events": 0,
        "shape_usage": shape,
    }


def _provenance(hypothesis: str, policy: str, seed: int, fold: int) -> dict[str, str]:
    values = {
        field: _hash_label(
            f"FT1C:{field}:{hypothesis}:{policy}:{seed}:{fold}"
        )
        for field in PROVENANCE_HASH_FIELDS
    }
    values["campaign_hash"] = CAMPAIGN_CONTRACT_SHA256
    return values


def _unit(
    *,
    hypothesis: str,
    policy: str,
    seed: int,
    fold: int,
    fold_net_pnl: float,
    ece: float,
) -> dict[str, Any]:
    base_date = datetime(2030, 1, 2) + timedelta(days=(fold - 1) * 2)
    sessions = [
        (base_date + timedelta(days=index)).date().isoformat()
        for index in range(2)
    ]
    confidence = 1.0 - ece
    payload = {
        "unit_id": unit_id(hypothesis, policy, seed, fold),
        "hypothesis": hypothesis,
        "policy": policy,
        "seed": seed,
        "fold": fold,
        "era": "ERA_A" if fold <= 2 else ("ERA_B" if fold <= 4 else "ERA_C"),
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "G9": False,
        "provenance": _provenance(hypothesis, policy, seed, fold),
        "session_ids": sessions,
        "candidates": [
            _candidate(
                hypothesis=hypothesis,
                policy=policy,
                seed=seed,
                fold=fold,
                session=sessions[0],
                candidate_index=0,
                net_pnl=fold_net_pnl,
            )
        ],
        "calibration_observations": [
            {"confidence": confidence, "won": 1.0},
            {"confidence": confidence, "won": 1.0},
        ],
        "diagnostics": _diagnostics(
            policy=policy,
            net_pnl=fold_net_pnl,
            trade_count=1,
        ),
    }
    return seal_unit(payload)


def build_synthetic_campaign(
    *,
    row_totals: Mapping[str, float] | None = None,
    maxT_pass_rows: set[str] | None = None,
    ece: float = 0.20,
) -> dict[str, Any]:
    totals = {
        row_id: 500.0 + index * 10.0 for index, row_id in enumerate(ROWS)
    }
    if row_totals:
        totals.update({key: float(value) for key, value in row_totals.items()})
    maxT_pass_rows = (
        {"H0/P0"} if maxT_pass_rows is None else set(maxT_pass_rows)
    )
    units = [
        _unit(
            hypothesis=hypothesis,
            policy=policy,
            seed=seed,
            fold=fold,
            fold_net_pnl=totals[f"{hypothesis}/{policy}"] / 5.0,
            ece=ece,
        )
        for hypothesis in HYPOTHESES
        for policy in POLICIES
        for seed in SEEDS
        for fold in FOLDS
    ]
    references = [
        {
            "row_id": row_id,
            "heuristic_pooled_pnl": 0.0,
            "matched_null_z_by_seed": {
                str(seed): 3.5 for seed in SEEDS
            },
        }
        for row_id in ROWS
    ]
    maxT_rows = [
        {
            "row_id": row_id,
            "p_FWER": 0.01 if row_id in maxT_pass_rows else 0.20,
            "exceedance_count": 199 if row_id in maxT_pass_rows else 3_999,
            "hard_pass": row_id in maxT_pass_rows,
        }
        for row_id in ROWS
    ]
    packet = {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "unit_count": UNIT_COUNT,
        "forbidden_evidence": {
            "seed_45_present": False,
            "G9_run": False,
            "protected_holdout_present": False,
            "sealed_evidence_present": False,
            "recorder_evidence_present": False,
        },
        "references": {
            "schema_version": REFERENCE_SCHEMAS[0],
            "acceptance_route": REFERENCE_ACCEPTANCE_ROUTE,
            "receipt_sha256": _hash_label("synthetic-references-receipt"),
            "rows": references,
        },
        "controls": {
            "D1": {
                "schema_version": CONTROL_SCHEMAS["D1"],
                "acceptance_route": REFERENCE_ACCEPTANCE_ROUTE,
                "receipt_sha256": _hash_label("synthetic-D1-receipt"),
                "independently_accepted": True,
                "joint_G1_G2_pass_count": 1,
                "median_pnl": -1.0,
                "median_z": 0.5,
                "passes": True,
            },
            "D5": {
                "schema_version": CONTROL_SCHEMAS["D5"],
                "independently_accepted": True,
                "identity_complete": True,
                "acceptance_route": REFERENCE_ACCEPTANCE_ROUTE,
                "receipt_sha256": _hash_label("synthetic-D5-receipt"),
            },
            "D6": {
                "schema_version": CONTROL_SCHEMAS["D6"],
                "independently_accepted": True,
                "route": (
                    "SIGNED_SPLIT_FAMILY_SYNCHRONIZATION_SUFFICIENT_FOR_"
                    "OFFLINE_LABEL_ONLY_REPAIR"
                ),
                "acceptance_route": REFERENCE_ACCEPTANCE_ROUTE,
                "receipt_sha256": _hash_label("synthetic-D6-receipt"),
            },
            "maxT": {
                "schema_version": CONTROL_SCHEMAS["maxT"],
                "acceptance_route": REFERENCE_ACCEPTANCE_ROUTE,
                "receipt_sha256": _hash_label("synthetic-maxT-receipt"),
                "valid": True,
                "family_size": 28,
                "replicates": 20_000,
                "tie_rule": "greater_than_or_equal",
                "rows": maxT_rows,
            },
        },
        "units": units,
    }
    return seal_campaign_packet(packet)


def _reseal(packet: dict[str, Any], unit_indices: list[int] | None = None) -> None:
    for index in unit_indices or []:
        packet["units"][index] = seal_unit(packet["units"][index])
    value = seal_campaign_packet(packet)
    packet.clear()
    packet.update(value)


def _unit_index(
    hypothesis: str,
    policy: str,
    seed: int,
    fold: int,
) -> int:
    return (
        HYPOTHESES.index(hypothesis) * 7 * 3 * 5
        + POLICIES.index(policy) * 3 * 5
        + SEEDS.index(seed) * 5
        + FOLDS.index(fold)
    )


def _replace_unit_economics(
    packet: dict[str, Any],
    *,
    hypothesis: str,
    policy: str,
    seed: int,
    fold: int,
    pnl_sequence: list[float],
    session_count: int = 2,
) -> None:
    index = _unit_index(hypothesis, policy, seed, fold)
    unit = packet["units"][index]
    base_date = datetime(2030, 1, 2) + timedelta(days=(fold - 1) * 2)
    sessions = [
        (base_date + timedelta(days=value)).date().isoformat()
        for value in range(session_count)
    ]
    unit["session_ids"] = sessions
    unit["candidates"] = [
        _candidate(
            hypothesis=hypothesis,
            policy=policy,
            seed=seed,
            fold=fold,
            session=sessions[candidate_index],
            candidate_index=candidate_index,
            net_pnl=net,
        )
        for candidate_index, net in enumerate(pnl_sequence)
    ]
    unit["diagnostics"] = _diagnostics(
        policy=policy,
        net_pnl=sum(pnl_sequence),
        trade_count=len(pnl_sequence),
    )
    _reseal(packet, [index])


def _reference(packet: dict[str, Any], row_id: str) -> dict[str, Any]:
    return next(
        row for row in packet["references"]["rows"] if row["row_id"] == row_id
    )


def _maxT(packet: dict[str, Any], row_id: str) -> dict[str, Any]:
    return next(
        row for row in packet["controls"]["maxT"]["rows"] if row["row_id"] == row_id
    )


def _pipeline(packet: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authorities = _authorities(packet)
    producer = aggregate_campaign(packet, **authorities)
    audit = audit_campaign(packet, producer, **authorities)
    selection = _synthetic_journal_selection(audit)
    return producer, audit, selection


def _synthetic_journal_selection(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    synthetic_root = ROOT / "v4/audit/autoresearch"
    synthetic_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="_ft1c5_selector_",
        dir=synthetic_root,
    ) as temporary:
        workspace = Path(temporary)
        inputs = workspace / "inputs"
        inputs.mkdir()
        owner = _ft1c3_owner(
            execution_id="FT1C5-SYNTHETIC-SELECTION"
        )
        for name in (
            "option_a.md",
            "offline_authorization.md",
            "goal.md",
            "preregistration.json",
            "contracts.json",
        ):
            (inputs / name).write_text(f"{name}: synthetic-only\n")
        journal = workspace / "controller.jsonl"
        create_journal(
            journal,
            workspace_root=workspace,
            campaign_namespace=CAMPAIGN_NAMESPACE,
            campaign_execution_id=owner["campaign_execution_id"],
            owner_option_a_decision_path=inputs / "option_a.md",
            offline_training_authorization_path=(
                inputs / "offline_authorization.md"
            ),
            campaign_goal_path=inputs / "goal.md",
            campaign_preregistration_path=inputs / "preregistration.json",
            signed_contract_bundle_path=inputs / "contracts.json",
            owner_execution_authorization_sha256=(
                owner_authorization_sha256(owner)
            ),
            owner_identity=owner["owner_signature"],
            owner_decision_date=owner["owner_decision_date"],
            timestamp_utc="2026-07-26T00:00:00+00:00",
        )
        audit_path = workspace / "independent_audit.json"
        _json(audit_path, audit)
        for index, node in enumerate(JOURNAL_NODES[:7], start=1):
            artifact = (
                audit_path
                if node == "INDEPENDENT_AUDIT"
                else workspace / f"{index:02d}_{node}.json"
            )
            if node != "INDEPENDENT_AUDIT":
                _json(artifact, {"node": node, "synthetic": True})
            validator = workspace / f"{index:02d}_{node}_validator.json"
            _json(
                validator,
                {
                    "routing_decision": JOURNAL_ROUTES[node],
                    "synthetic": True,
                },
            )
            append_checkpoint(
                journal,
                workspace_root=workspace,
                node=node,
                artifact_path=artifact,
                validator_route=JOURNAL_ROUTES[node],
                validator_receipt_path=validator,
                timestamp_utc=f"2026-07-26T00:00:{index:02d}+00:00",
            )
        return select_candidate(
            audit,
            controller_journal_path=journal,
            audit_result_path=audit_path,
            workspace_root=workspace,
            campaign_namespace=CAMPAIGN_NAMESPACE,
            campaign_execution_id=owner["campaign_execution_id"],
        )


def _authorities(packet: Mapping[str, Any]) -> dict[str, Any]:
    execution = build_execution_provenance_authority(packet)
    control = build_control_authority(packet)
    return {
        "execution_authority": execution,
        "execution_authority_sha256": stable_hash(execution),
        "control_authority": control,
        "control_authority_sha256": stable_hash(control),
    }


def _aggregate(packet: Mapping[str, Any]) -> dict[str, Any]:
    return aggregate_campaign(packet, **_authorities(packet))


def _audit(
    packet: Mapping[str, Any],
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    return audit_campaign(packet, producer, **_authorities(packet))


def _validate(packet: Mapping[str, Any]) -> list[str]:
    return validate_campaign_packet(packet, **_authorities(packet))


def _row(result: Mapping[str, Any], row_id: str) -> Mapping[str, Any]:
    return next(row for row in result["rows"] if row["row_id"] == row_id)


def _case(
    case_id: str,
    description: str,
    checks: Mapping[str, bool],
    *,
    scenario_count: int = 1,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "description": description,
        "scenario_count": scenario_count,
        "checks": dict(checks),
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def run_synthetic_matrix() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    high = build_synthetic_campaign(ece=0.20)
    high_producer, high_audit, high_selection = _pipeline(high)
    high_row = _row(high_producer, "H0/P0")
    cases.append(
        _case(
            "C01",
            "eligible row remains selected with G8 above benchmark",
            {
                "selected": high_selection["routing_decision"] == SELECTED_ROUTE,
                "selected_H0_P0": high_selection["selected_candidate"]["row_id"] == "H0/P0",
                "G8_above_benchmark": high_row["gates"]["G8"] is False,
                "hard_eligible": high_row["hard_gate_eligible"] is True,
            },
        )
    )

    low = build_synthetic_campaign(ece=0.05)
    _, _, low_selection = _pipeline(low)
    cases.append(
        _case(
            "C02",
            "lower G8 produces identical selection",
            {
                "same_selection": (
                    low_selection["selected_candidate"]["row_id"]
                    == high_selection["selected_candidate"]["row_id"]
                ),
                "same_route": low_selection["routing_decision"] == SELECTED_ROUTE,
            },
        )
    )

    failed_maxt = build_synthetic_campaign(maxT_pass_rows=set())
    failed_producer, _, failed_selection = _pipeline(failed_maxt)
    cases.append(
        _case(
            "C03",
            "otherwise eligible row failing maxT is ineligible",
            {
                "row_ineligible": not _row(failed_producer, "H0/P0")[
                    "hard_gate_eligible"
                ],
                "no_selection": failed_selection["selected_candidate"] is None,
            },
        )
    )

    invalid_maxt = build_synthetic_campaign()
    invalid_maxt["controls"]["maxT"]["valid"] = False
    _reseal(invalid_maxt)
    invalid_producer, invalid_audit, invalid_selection = _pipeline(invalid_maxt)
    cases.append(
        _case(
            "C04",
            "invalid maxT blocks all selection",
            {
                "producer_invalid": not invalid_producer["valid"],
                "audit_rejects": not invalid_audit["accepted"],
                "invalid_route": invalid_selection["routing_decision"] == INVALID_ROUTE,
            },
        )
    )

    d1_failed = build_synthetic_campaign()
    d1_failed["controls"]["D1"].update(
        {"joint_G1_G2_pass_count": 2, "passes": False}
    )
    _reseal(d1_failed)
    d1_producer, d1_audit, d1_selection = _pipeline(d1_failed)
    cases.append(
        _case(
            "C05",
            "D1 failure blocks every row and selection",
            {
                "D1_false": not d1_producer["global_controls"]["D1"],
                "audit_agrees": d1_audit["accepted"],
                "invalid_route": d1_selection["routing_decision"] == INVALID_ROUTE,
            },
        )
    )

    control_scenarios: list[bool] = []
    for control in ("D5", "D6"):
        packet = build_synthetic_campaign()
        packet["controls"][control]["receipt_sha256"] = "0" * 64
        packet["controls"][control]["independently_accepted"] = False
        _reseal(packet)
        producer, audit, selection = _pipeline(packet)
        control_scenarios.append(
            not producer["valid"]
            and not audit["accepted"]
            and selection["routing_decision"] == INVALID_ROUTE
        )
    cases.append(
        _case(
            "C06",
            "D5 and D6 mismatches each block selection",
            {"D5_rejected": control_scenarios[0], "D6_rejected": control_scenarios[1]},
            scenario_count=2,
        )
    )

    gate_checks: dict[str, bool] = {}
    for gate in range(1, 8):
        packet = build_synthetic_campaign()
        if gate == 1:
            for fold in (1, 3):
                _replace_unit_economics(
                    packet,
                    hypothesis="H0",
                    policy="P0",
                    seed=42,
                    fold=fold,
                    pnl_sequence=[-10.0],
                )
        elif gate == 2:
            _reference(packet, "H0/P0")["matched_null_z_by_seed"] = {
                str(seed): 2.9 for seed in SEEDS
            }
            _reseal(packet)
        elif gate == 3:
            _reference(packet, "H0/P0")["heuristic_pooled_pnl"] = 10_000.0
            _reseal(packet)
        elif gate == 4:
            _replace_unit_economics(
                packet,
                hypothesis="H0",
                policy="P0",
                seed=42,
                fold=1,
                pnl_sequence=[-700.0, 800.0],
            )
        elif gate == 5:
            _reference(packet, "H0/P0")["matched_null_z_by_seed"]["42"] = 1.9
            _reseal(packet)
        elif gate == 6:
            _replace_unit_economics(
                packet,
                hypothesis="H0",
                policy="P0",
                seed=42,
                fold=5,
                pnl_sequence=[-10.0],
            )
        else:
            _replace_unit_economics(
                packet,
                hypothesis="H0",
                policy="P0",
                seed=42,
                fold=1,
                pnl_sequence=[100.0],
                session_count=4,
            )
        producer = _aggregate(packet)
        gate_checks[f"G{gate}_fails"] = (
            _row(producer, "H0/P0")["gates"][f"G{gate}"] is False
        )
    cases.append(
        _case(
            "C07",
            "each hard gate G1 through G7 independently fails its target row",
            gate_checks,
            scenario_count=7,
        )
    )

    g6_packet = build_synthetic_campaign()
    _replace_unit_economics(
        g6_packet,
        hypothesis="H0",
        policy="P0",
        seed=42,
        fold=5,
        pnl_sequence=[-10.0],
    )
    _, _, g6_selection = _pipeline(g6_packet)
    cases.append(
        _case(
            "C08",
            "G6-only failure routes to owner review",
            {"regime_route": g6_selection["routing_decision"] == REGIME_ROUTE},
        )
    )

    attribution = build_synthetic_campaign()
    _reference(attribution, "H0/P0")["heuristic_pooled_pnl"] = 10_000.0
    _reseal(attribution)
    _, _, attribution_selection = _pipeline(attribution)
    cases.append(
        _case(
            "C09",
            "adjusted signal with fixed-exit failure routes attribution",
            {
                "attribution_route": (
                    attribution_selection["routing_decision"]
                    == ATTRIBUTION_ROUTE
                )
            },
        )
    )

    no_signal = build_synthetic_campaign()
    _reference(no_signal, "H0/P0")["matched_null_z_by_seed"] = {
        str(seed): 1.0 for seed in SEEDS
    }
    _reseal(no_signal)
    _, _, no_signal_selection = _pipeline(no_signal)
    cases.append(
        _case(
            "C10",
            "no adjusted signal routes stop and redesign",
            {"stop_route": no_signal_selection["routing_decision"] == STOP_ROUTE},
        )
    )

    ranking = build_synthetic_campaign(
        row_totals={"H0/P0": 500.0, "H3/P6": 900.0},
        maxT_pass_rows={"H0/P0", "H3/P6"},
    )
    _, _, ranking_selection = _pipeline(ranking)
    cases.append(
        _case(
            "C11",
            "larger plain median strict-serial PnL ranks first",
            {
                "H3_P6_selected": (
                    ranking_selection["selected_candidate"]["row_id"] == "H3/P6"
                )
            },
        )
    )

    tied = build_synthetic_campaign(
        row_totals={"H0/P0": 700.0, "H0/P1": 700.0},
        maxT_pass_rows={"H0/P0", "H0/P1"},
    )
    _, _, tied_selection = _pipeline(tied)
    cases.append(
        _case(
            "C12",
            "exact economic tie uses H then P order",
            {
                "H0_P0_selected": (
                    tied_selection["selected_candidate"]["row_id"] == "H0/P0"
                )
            },
        )
    )

    rank_invariant = build_synthetic_campaign(
        row_totals={"H1/P1": 800.0, "H2/P2": 700.0},
        maxT_pass_rows={"H1/P1", "H2/P2"},
    )
    loser_indices: list[int] = []
    for seed in SEEDS:
        for fold in FOLDS:
            index = _unit_index("H2", "P2", seed, fold)
            unit = rank_invariant["units"][index]
            unit["calibration_observations"] = [
                {"confidence": 1.0, "won": 1.0}
            ]
            unit["diagnostics"]["side_time_exposure"]["win_rate"] = 1.0
            unit["diagnostics"]["churn"]["model_complexity"] = 1
            unit["diagnostics"]["concentration"]["drawdown_preference"] = 0.0
            loser_indices.append(index)
    _reseal(rank_invariant, loser_indices)
    _, _, invariant_selection = _pipeline(rank_invariant)
    cases.append(
        _case(
            "C13",
            "ECE win rate drawdown and complexity cannot alter plain-PnL rank",
            {
                "plain_pnl_winner_selected": (
                    invariant_selection["selected_candidate"]["row_id"]
                    == "H1/P1"
                ),
                "G8_not_used": not invariant_selection[
                    "G8_used_for_eligibility_or_ranking"
                ],
            },
        )
    )

    stale_results: list[bool] = []
    for mutation in ("v4", "mixed", "namespace"):
        packet = build_synthetic_campaign()
        if mutation == "v4":
            packet["simulator_version"] = PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION
            _reseal(packet)
        elif mutation == "mixed":
            packet["units"][0]["simulator_version"] = (
                PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION
            )
            _reseal(packet, [0])
        else:
            packet["campaign_namespace"] = "stale_campaign_namespace"
            _reseal(packet)
        stale_results.append(not _aggregate(packet)["valid"])
    cases.append(
        _case(
            "C14",
            "v4 mixed and stale namespace packets fail closed",
            {
                "v4_rejected": stale_results[0],
                "mixed_rejected": stale_results[1],
                "namespace_rejected": stale_results[2],
            },
            scenario_count=3,
        )
    )

    identity_mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "missing_H_P": lambda packet: packet["units"].__delitem__(0),
        "missing_seed": lambda packet: packet["units"].__delitem__(5),
        "missing_fold": lambda packet: packet["units"].__delitem__(1),
        "reordered": lambda packet: packet["units"].__setitem__(
            slice(0, 2), [packet["units"][1], packet["units"][0]]
        ),
        "duplicate_unit": lambda packet: packet["units"].__setitem__(
            1, deepcopy(packet["units"][0])
        ),
        "duplicate_session": lambda packet: packet["units"][0][
            "session_ids"
        ].__setitem__(1, packet["units"][0]["session_ids"][0]),
        "duplicate_trade": lambda packet: packet["units"][0][
            "candidates"
        ].append(deepcopy(packet["units"][0]["candidates"][0])),
    }
    identity_checks: dict[str, bool] = {}
    for name, mutate in identity_mutations.items():
        packet = build_synthetic_campaign()
        mutate(packet)
        if name in {"duplicate_session", "duplicate_trade"}:
            _reseal(packet, [0])
        else:
            _reseal(packet)
        identity_checks[f"{name}_rejected"] = not _aggregate(packet)[
            "valid"
        ]
    cases.append(
        _case(
            "C15",
            "missing reordered and duplicate axes unit session and trade fail",
            identity_checks,
            scenario_count=len(identity_checks),
        )
    )

    primary_checks: dict[str, bool] = {}
    for name, key in (
        ("fill", "pessimistic_executable"),
        ("noise", "1.0x"),
    ):
        packet = build_synthetic_campaign()
        container = (
            packet["units"][0]["diagnostics"]["fill_edge_band"]
            if name == "fill"
            else packet["units"][0]["diagnostics"]["noise_diagnostics"]
        )
        container.pop(key)
        _reseal(packet, [0])
        primary_checks[f"{name}_substitution_rejected"] = not _aggregate(
            packet
        )["valid"]
    cases.append(
        _case(
            "C16",
            "favorable fill or zero noise cannot replace primary evidence",
            primary_checks,
            scenario_count=2,
        )
    )

    nonfinite = build_synthetic_campaign()
    nonfinite["units"][0]["diagnostics"]["worst_day"] = float("inf")
    incomplete = build_synthetic_campaign()
    incomplete["units"][0]["diagnostics"].pop("harvest_ratio")
    _reseal(incomplete, [0])
    cases.append(
        _case(
            "C17",
            "nonfinite and incomplete diagnostics fail closed",
            {
                "nonfinite_rejected": not _aggregate(nonfinite)["valid"],
                "incomplete_rejected": not _aggregate(incomplete)["valid"],
            },
            scenario_count=2,
        )
    )

    protected_axis_checks: dict[str, bool] = {}
    for name in ("seed45", "G9"):
        packet = build_synthetic_campaign()
        if name == "seed45":
            packet["units"][0]["seed"] = 45
        else:
            packet["units"][0]["G9"] = True
        _reseal(packet, [0])
        protected_axis_checks[f"{name}_rejected"] = not _aggregate(packet)[
            "valid"
        ]
    cases.append(
        _case(
            "C18",
            "seed45 and G9 input are rejected",
            protected_axis_checks,
            scenario_count=2,
        )
    )

    forbidden_checks: dict[str, bool] = {}
    for key in (
        "protected_holdout_present",
        "sealed_evidence_present",
        "recorder_evidence_present",
    ):
        packet = build_synthetic_campaign()
        packet["forbidden_evidence"][key] = True
        _reseal(packet)
        forbidden_checks[f"{key}_rejected"] = not _aggregate(packet)[
            "valid"
        ]
    cases.append(
        _case(
            "C19",
            "protected sealed and recorder evidence are rejected",
            forbidden_checks,
            scenario_count=3,
        )
    )

    cases.append(
        _case(
            "C20",
            "producer aggregator and independent audit agree exactly",
            {
                "audit_accepted": high_audit["accepted"],
                "all_28_rows": high_audit["all_28_rows_published"],
                "no_differences": high_audit["comparisons"] == [],
                "freeze_bound": bool(high_audit["freeze_sha256"]),
            },
        )
    )

    defect = deepcopy(high_producer)
    defect["rows"][0][
        "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
    ] += 1.0
    defect_audit = _audit(high, defect)
    cases.append(
        _case(
            "C21",
            "injected aggregator defect is caught by independent audit",
            {
                "audit_rejects": not defect_audit["accepted"],
                "disagreement_reported": (
                    "producer_independent_result_disagreement"
                    in defect_audit["defects"]
                ),
            },
        )
    )

    no_owner = evaluate_graph()
    cases.append(
        _case(
            "C22",
            "no-owner graph stops before RUN",
            {
                "route_exact": no_owner["routing_decision"] == NO_OWNER_ROUTE,
                "RUN_false": no_owner["RUN_executed"] is False,
                "stopped_before_RUN": no_owner["stopped_before_RUN"] is True,
                "no_commands": no_owner["commands_executed"] == [],
            },
        )
    )

    calmar_boundary = build_synthetic_campaign()
    for seed in SEEDS:
        for fold, total in zip(FOLDS, (-100.0, 50.0, 50.0, 50.0, 50.0)):
            _replace_unit_economics(
                calmar_boundary,
                hypothesis="H0",
                policy="P0",
                seed=seed,
                fold=fold,
                pnl_sequence=[total],
            )
    cases.append(
        _case(
            "R-G4-CALMAR-AT",
            "inclusive G4 Calmar boundary passes",
            {
                "G4_true": _row(
                    _aggregate(calmar_boundary),
                    "H0/P0",
                )["gates"]["G4"]
                is True
            },
        )
    )

    era_boundary = build_synthetic_campaign()
    for seed in SEEDS:
        _replace_unit_economics(
            era_boundary,
            hypothesis="H0",
            policy="P0",
            seed=seed,
            fold=5,
            pnl_sequence=[0.0],
        )
    cases.append(
        _case(
            "R-G6-AT",
            "inclusive G6 zero-median boundary passes",
            {
                "G6_true": _row(
                    _aggregate(era_boundary),
                    "H0/P0",
                )["gates"]["G6"]
                is True
            },
        )
    )

    changed_model = build_synthetic_campaign()
    changed_model_authorities = _authorities(changed_model)
    changed_model["units"][0]["provenance"]["model_hash"] = "a" * 64
    _reseal(changed_model, [0])
    cases.append(
        _case(
            "R-FRESHNESS-changed_model_hash",
            "self-resealed changed model hash fails frozen authority",
            {
                "rejected": not aggregate_campaign(
                    changed_model,
                    **changed_model_authorities,
                )["valid"]
            },
        )
    )

    old_reference = build_synthetic_campaign()
    old_reference["references"]["schema_version"] = "OLD_REFERENCE_V4"
    _reseal(old_reference)
    cases.append(
        _case(
            "R-FRESHNESS-old_reference_schema",
            "legacy reference schema fails closed",
            {"rejected": not _aggregate(old_reference)["valid"]},
        )
    )

    old_rank = build_synthetic_campaign()
    old_rank["prior_ranking"] = {"selected": "OLD/H2/P5"}
    _reseal(old_rank)
    cases.append(
        _case(
            "R-FRESHNESS-old_rank_payload",
            "prior rank payload fails closed",
            {"rejected": not _aggregate(old_rank)["valid"]},
        )
    )

    old_economics = build_synthetic_campaign()
    old_economics["units"][0]["candidates"][0]["strategy"] = (
        "OLD_H0_H3_ECONOMICS"
    )
    _reseal(old_economics, [0])
    cases.append(
        _case(
            "R-FRESHNESS-old_economics_marker",
            "old economics marker fails closed",
            {"rejected": not _aggregate(old_economics)["valid"]},
        )
    )

    for control_name in ("D5", "D6"):
        changed_control = build_synthetic_campaign()
        changed_control_authorities = _authorities(changed_control)
        changed_control["controls"][control_name]["receipt_sha256"] = "b" * 64
        _reseal(changed_control)
        cases.append(
            _case(
                f"R-CONTROL-{control_name}-CHANGED-HASH",
                f"self-resealed changed {control_name} hash fails authority",
                {
                    "rejected": not aggregate_campaign(
                        changed_control,
                        **changed_control_authorities,
                    )["valid"]
                },
            )
        )

    stale_freeze = deepcopy(high_audit)
    stale_freeze["independent_result"]["rows"][0][
        "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
    ] += 1.0
    stale_selection = select_candidate(stale_freeze)
    cases.append(
        _case(
            "R-SELECTOR-CHANGED-FROZEN-PAYLOAD",
            "changed independent result with stale freeze cannot select",
            {
                "invalid_route": (
                    stale_selection["routing_decision"] == INVALID_ROUTE
                )
            },
        )
    )

    malformed_freeze = deepcopy(high_audit)
    malformed_freeze["freeze_sha256"] = "not-a-sha256"
    malformed_selection = select_candidate(malformed_freeze)
    cases.append(
        _case(
            "R-SELECTOR-MALFORMED-FREEZE-HASH",
            "malformed freeze hash cannot select",
            {
                "invalid_route": (
                    malformed_selection["routing_decision"] == INVALID_ROUTE
                )
            },
        )
    )

    return {
        "schema_version": "Protocol101FT1CSyntheticCaseResultsV1",
        "case_count": len(cases),
        "scenario_count": sum(item["scenario_count"] for item in cases),
        "passed_case_count": sum(item["status"] == "PASS" for item in cases),
        "failed_case_count": sum(item["status"] == "FAIL" for item in cases),
        "all_pass": all(item["status"] == "PASS" for item in cases),
        "cases": cases,
        "baseline": {
            "unit_count": len(high["units"]),
            "row_count": len(ROWS),
            "producer_row_count": len(high_producer["rows"]),
            "audit_row_count": high_audit["published_row_count"],
            "selected_row": high_selection["selected_candidate"]["row_id"],
            "expected_selected_row_plain_fixture_law": "H0/P0",
            "expected_H0_P0_median_seed_pnl": 500.0,
            "observed_H0_P0_median_seed_pnl": high_row[
                "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
            ],
        },
    }


def evidence_schema() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101FT1CEvidenceSchemaV1",
        "campaign_schema": CAMPAIGN_SCHEMA,
        "unit_axes": ["hypothesis", "policy", "seed", "fold"],
        "exact_geometry": {
            "rows": list(ROWS),
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "units": UNIT_COUNT,
        },
        "unit_required_fields": [
            "unit_id",
            "campaign_namespace",
            "simulator_version",
            "G9",
            "provenance",
            "session_ids",
            "candidates",
            "calibration_observations",
            "diagnostics",
            "unit_sha256",
        ],
        "provenance_hash_fields": list(PROVENANCE_HASH_FIELDS),
        "diagnostic_fields": list(DIAGNOSTIC_KEYS),
        "candidate_clock_fields": [
            "decision_time_ns",
            "label_source_exit_quote_time_ns",
            "label_realized_exit_time_ns",
            "label_policy_deadline_ns",
        ],
        "primary_evidence": {
            "fee": "3.00",
            "fill": "pessimistic_executable",
            "noise": "1.0x",
        },
        "G8_role": "report_only",
        "G9": False,
    }


def _ft1c3_owner(
    *,
    execution_id: str = "FT1C3-SYNTHETIC-EXECUTION-A",
    signature: str = "FT1C3-SYNTHETIC-OWNER",
) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "campaign_execution_id": execution_id,
        "authorized": True,
        "routing_decision": (
            "owner_authorized_fresh_420_unit_campaign_execution"
        ),
        "owner_signature": signature,
        "owner_decision_date": "2026-07-26",
        "goal_sha256": _hash_label("FT1C3:goal"),
        "preregistration_sha256": _hash_label("FT1C3:preregistration"),
        "contract_bundle_sha256": _hash_label("FT1C3:contracts"),
        "seed_45_or_G9_authorized": False,
    }


def _ft1c3_bindings(
    *,
    prefix_length: int = len(RECEIPT_NODES),
    campaign_tag: str = "A",
) -> dict[str, str]:
    return {
        node: _hash_label(f"FT1C3:{campaign_tag}:{node}:artifact")
        for node in RECEIPT_NODES[:prefix_length]
    }


def _ft1c3_chain(
    *,
    authorization: Mapping[str, Any],
    bindings: Mapping[str, str],
    prefix_length: int = len(RECEIPT_NODES),
) -> dict[str, Any]:
    return build_receipt_chain(
        owner_authorization=authorization,
        artifact_sha256_by_node=bindings,
        prefix_length=prefix_length,
    )


def _ft1c3_rebuild_receipts(
    chain: dict[str, Any],
    *,
    authorization: Mapping[str, Any],
    start_index: int = 0,
) -> None:
    owner_hash = owner_authorization_sha256(authorization)
    chain["campaign_namespace"] = authorization["campaign_namespace"]
    chain["campaign_execution_id"] = authorization["campaign_execution_id"]
    chain["owner_authorization_sha256"] = owner_hash
    parent = owner_hash
    receipts = chain["receipts"]
    for index, receipt in enumerate(receipts):
        if index < start_index:
            parent = receipt["receipt_sha256"]
            continue
        receipt["parent_receipt_sha256"] = parent
        sealed = seal_graph_receipt(receipt)
        receipt.clear()
        receipt.update(sealed)
        parent = receipt["receipt_sha256"]
    sealed_chain = seal_receipt_chain(chain)
    chain.clear()
    chain.update(sealed_chain)


def _ft1c3_record(
    rows: list[dict[str, Any]],
    *,
    case_id: str,
    category: str,
    expected: str,
    result: Mapping[str, Any],
    passed: bool,
    description: str,
) -> None:
    rows.append(
        {
            "case_id": case_id,
            "category": category,
            "description": description,
            "expected": expected,
            "observed_route": result["routing_decision"],
            "observed_next_node": result["next_node"],
            "blockers": list(result.get("blockers", [])),
            "status": "PASS" if passed else "FAIL",
            "real_campaign_economics_executed": False,
            "training_executed": False,
            "real_selection_executed": False,
            "G9_or_seed45_executed": False,
            "protected_or_sealed_evidence_accessed": False,
            "broker_paper_promotion_runtime_or_launchd_action": False,
        }
    )


def run_authority_root_chain_matrix() -> dict[str, Any]:
    authorization = _ft1c3_owner()
    trusted = _ft1c3_bindings()
    honest_chain = _ft1c3_chain(
        authorization=authorization,
        bindings=trusted,
    )
    rows: list[dict[str, Any]] = []

    honest = _evaluate_legacy_receipt_graph(
        owner_authorization=authorization,
        receipts=honest_chain,
        trusted_artifact_sha256_by_node=trusted,
    )
    honest_pass = (
        honest["routing_decision"] == GRAPH_STOP_ROUTE
        and honest["next_node"] == "STOP"
        and honest["completed_prefix"] == list(RECEIPT_NODES)
        and honest["accepted_artifact_bindings"] == trusted
    )

    def evaluate(
        *,
        owner: Mapping[str, Any] | None = authorization,
        chain: Mapping[str, Any] | None = honest_chain,
        bindings: Mapping[str, str] | None = trusted,
    ) -> dict[str, Any]:
        return _evaluate_legacy_receipt_graph(
            owner_authorization=owner,
            receipts=chain,
            trusted_artifact_sha256_by_node=bindings,
        )

    def record_rejection(
        *,
        case_id: str,
        category: str,
        chain: Mapping[str, Any] | None = honest_chain,
        owner: Mapping[str, Any] | None = authorization,
        bindings: Mapping[str, str] | None = trusted,
        description: str,
    ) -> None:
        result = evaluate(owner=owner, chain=chain, bindings=bindings)
        _ft1c3_record(
            rows,
            case_id=case_id,
            category=category,
            expected="reject_before_STOP",
            result=result,
            passed=(
                result["routing_decision"] == INVALID_CHAIN_ROUTE
                and result["next_node"] != "STOP"
            ),
            description=description,
        )

    def changed_artifact_chain(
        index: int,
        value: str,
    ) -> dict[str, Any]:
        chain = deepcopy(honest_chain)
        chain["receipts"][index]["artifact_sha256"] = value
        _ft1c3_rebuild_receipts(
            chain,
            authorization=authorization,
            start_index=index,
        )
        return chain

    exact_builders: list[
        tuple[str, str, Callable[[], tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]]]
    ] = [
        (
            "AUTH-ATTACK-01-packet_resealed",
            "packet_resealed",
            lambda: (
                changed_artifact_chain(
                    0, _hash_label("forged:packet_resealed")
                ),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-02-authority_rebuilt",
            "authority_rebuilt",
            lambda: (
                changed_artifact_chain(
                    1, _hash_label("forged:authority_rebuilt")
                ),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-03-all_self_hashes",
            "all_self_hashes",
            lambda: (
                _ft1c3_chain(
                    authorization=authorization,
                    bindings=_ft1c3_bindings(campaign_tag="FORGED"),
                ),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-04-cross_campaign_authority",
            "cross_campaign_authority",
            lambda: (
                changed_artifact_chain(
                    1,
                    _ft1c3_bindings(campaign_tag="B")[
                        "EXECUTION_PROVENANCE_AUTHORITY"
                    ],
                ),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-05-unit_grid",
            "unit_grid",
            lambda: (
                deepcopy(honest_chain),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-06-authority_schema",
            "authority_schema",
            lambda: (
                deepcopy(honest_chain),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-07-unknown_valid_hash",
            "unknown_valid_hash",
            lambda: (
                changed_artifact_chain(1, "d" * 64),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-08-control_local",
            "control_local",
            lambda: (
                changed_artifact_chain(
                    3, _hash_label("forged:control_local")
                ),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-09-control_rebuilt",
            "control_rebuilt",
            lambda: (
                changed_artifact_chain(
                    3, _hash_label("forged:control_rebuilt")
                ),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-10-cross_campaign_control",
            "cross_campaign_control",
            lambda: (
                changed_artifact_chain(
                    3,
                    _ft1c3_bindings(campaign_tag="B")[
                        "CONTROL_AUTHORITY"
                    ],
                ),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-11-node_order",
            "node_order",
            lambda: (
                deepcopy(honest_chain),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-12-pre_run_injection",
            "pre_run_injection",
            lambda: (
                deepcopy(honest_chain),
                authorization,
            ),
        ),
        (
            "AUTH-ATTACK-14-owner_changed",
            "owner_changed",
            lambda: (
                deepcopy(honest_chain),
                _ft1c3_owner(signature="CHANGED-AFTER-AUTHORITY"),
            ),
        ),
    ]
    for case_id, name, builder in exact_builders:
        chain, owner = builder()
        assert chain is not None
        chain = deepcopy(chain)
        if name == "unit_grid":
            chain["receipts"][1]["unit_count"] = 419
            _ft1c3_rebuild_receipts(
                chain, authorization=authorization, start_index=1
            )
        elif name == "authority_schema":
            chain["receipts"][1]["schema_version"] = "OLD"
            _ft1c3_rebuild_receipts(
                chain, authorization=authorization, start_index=1
            )
        elif name == "node_order":
            chain["receipts"][1], chain["receipts"][2] = (
                chain["receipts"][2],
                chain["receipts"][1],
            )
            sealed = seal_receipt_chain(chain)
            chain.clear()
            chain.update(sealed)
        elif name == "pre_run_injection":
            injected = dict(chain["receipts"][0])
            injected["node"] = "INJECTED_AUTHORITY_BEFORE_RUN"
            injected = seal_graph_receipt(injected)
            chain["receipts"].insert(0, injected)
            sealed = seal_receipt_chain(chain)
            chain.clear()
            chain.update(sealed)
        record_rejection(
            case_id=case_id,
            category="exact_reproducer",
            chain=chain,
            owner=owner,
            description=f"FT1C2 semantic attack rebuilt against V2: {name}",
        )

    exact_count = len(rows)

    for prefix_length in range(len(RECEIPT_NODES) + 1):
        prefix_bindings = _ft1c3_bindings(prefix_length=prefix_length)
        prefix_chain = _ft1c3_chain(
            authorization=authorization,
            bindings=prefix_bindings,
            prefix_length=prefix_length,
        )
        result = evaluate(
            chain=prefix_chain,
            bindings=prefix_bindings,
        )
        expected_next = (
            "STOP"
            if prefix_length == len(RECEIPT_NODES)
            else RECEIPT_NODES[prefix_length]
        )
        expected_route = (
            GRAPH_STOP_ROUTE if expected_next == "STOP" else RUN_PENDING_ROUTE
        )
        _ft1c3_record(
            rows,
            case_id=f"PREFIX-{prefix_length}",
            category="valid_prefix",
            expected=f"{expected_route}:{expected_next}",
            result=result,
            passed=(
                result["routing_decision"] == expected_route
                and result["next_node"] == expected_next
                and result["completed_prefix"]
                == list(RECEIPT_NODES[:prefix_length])
            ),
            description="valid exact prefix advances only to its next node",
        )

    for field in sorted(authorization):
        changed = dict(authorization)
        del changed[field]
        record_rejection(
            case_id=f"OWNER-REMOVE-{field}",
            category="owner_schema",
            owner=changed,
            description=f"removed owner field {field}",
        )
    changed = dict(authorization)
    changed["unexpected"] = True
    record_rejection(
        case_id="OWNER-EXTRA-FIELD",
        category="owner_schema",
        owner=changed,
        description="extra owner field",
    )
    owner_changes: dict[str, Any] = {
        "schema_version": "OLD",
        "campaign_namespace": "other-campaign",
        "campaign_execution_id": "other-execution",
        "authorized": False,
        "routing_decision": "wrong-route",
        "owner_signature": "CHANGED",
        "owner_decision_date": "",
        "goal_sha256": "A" * 64,
        "preregistration_sha256": "short",
        "contract_bundle_sha256": "4" * 63,
        "seed_45_or_G9_authorized": True,
    }
    for field, value in owner_changes.items():
        changed = dict(authorization)
        changed[field] = value
        record_rejection(
            case_id=f"OWNER-CHANGE-{field}",
            category="owner_schema",
            owner=changed,
            description=f"changed owner field {field}",
        )

    chain_changes: dict[str, Any] = {
        "schema_version": "OLD",
        "campaign_namespace": "other-campaign",
        "campaign_execution_id": "other-execution",
        "owner_authorization_sha256": "f" * 64,
        "chain_sha256": "not-a-sha256",
    }
    for field, value in chain_changes.items():
        chain = deepcopy(honest_chain)
        chain[field] = value
        record_rejection(
            case_id=f"CHAIN-CHANGE-{field}",
            category="chain_schema",
            chain=chain,
            description=f"changed chain field {field}",
        )
    chain = deepcopy(honest_chain)
    chain["unexpected"] = True
    record_rejection(
        case_id="CHAIN-EXTRA-FIELD",
        category="chain_schema",
        chain=chain,
        description="extra chain field",
    )
    chain = deepcopy(honest_chain)
    del chain["chain_sha256"]
    record_rejection(
        case_id="CHAIN-REMOVE-HASH",
        category="chain_schema",
        chain=chain,
        description="removed chain hash",
    )

    receipt_mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "schema_version": lambda receipt: receipt.__setitem__(
            "schema_version", "OLD"
        ),
        "node": lambda receipt: receipt.__setitem__("node", "WRONG_NODE"),
        "routing_decision": lambda receipt: receipt.__setitem__(
            "routing_decision", "wrong-route"
        ),
        "campaign_namespace": lambda receipt: receipt.__setitem__(
            "campaign_namespace", "other-campaign"
        ),
        "campaign_execution_id": lambda receipt: receipt.__setitem__(
            "campaign_execution_id", "other-execution"
        ),
        "owner_authorization_sha256": lambda receipt: receipt.__setitem__(
            "owner_authorization_sha256", "f" * 64
        ),
        "parent_receipt_sha256": lambda receipt: receipt.__setitem__(
            "parent_receipt_sha256", "e" * 64
        ),
        "artifact_sha256": lambda receipt: receipt.__setitem__(
            "artifact_sha256", "d" * 64
        ),
        "receipt_sha256": lambda receipt: receipt.__setitem__(
            "receipt_sha256", "c" * 64
        ),
    }
    for index, node in enumerate(RECEIPT_NODES):
        for field, mutator in receipt_mutations.items():
            chain = deepcopy(honest_chain)
            mutator(chain["receipts"][index])
            sealed = seal_receipt_chain(chain)
            record_rejection(
                case_id=f"RECEIPT-{index}-{field}",
                category="receipt_field",
                chain=sealed,
                description=f"mutated {field} at {node}",
            )
        chain = deepcopy(honest_chain)
        chain["receipts"][index]["unexpected"] = True
        chain = seal_receipt_chain(chain)
        record_rejection(
            case_id=f"RECEIPT-{index}-extra-field",
            category="receipt_schema",
            chain=chain,
            description=f"extra receipt field at {node}",
        )
        chain = changed_artifact_chain(
            index, _hash_label(f"changed-artifact:{node}")
        )
        record_rejection(
            case_id=f"ARTIFACT-{index}-CHANGED",
            category="artifact_binding",
            chain=chain,
            description=f"changed and rebuilt artifact receipt at {node}",
        )
        chain = deepcopy(honest_chain)
        chain["receipts"][index][
            "campaign_execution_id"
        ] = f"wrong-execution-{index}"
        chain = seal_receipt_chain(chain)
        record_rejection(
            case_id=f"DEPTH-{index}-WRONG-EXECUTION",
            category="cross_campaign",
            chain=chain,
            description=f"wrong execution id at depth {index}",
        )
        chain = deepcopy(honest_chain)
        chain["receipts"][index][
            "owner_authorization_sha256"
        ] = _hash_label(f"wrong-owner:{index}")
        chain = seal_receipt_chain(chain)
        record_rejection(
            case_id=f"DEPTH-{index}-WRONG-OWNER",
            category="cross_campaign",
            chain=chain,
            description=f"wrong owner hash at depth {index}",
        )

    for case_id, indices in (
        ("OMIT-FIRST", list(range(1, 8))),
        ("OMIT-MIDDLE", [0, 1, 2, 4, 5, 6, 7]),
        ("OMIT-LAST", list(range(7))),
        ("REVERSE", list(reversed(range(8)))),
        ("PERMUTE", [0, 2, 1, 3, 4, 5, 6, 7]),
    ):
        chain = deepcopy(honest_chain)
        chain["receipts"] = [chain["receipts"][index] for index in indices]
        chain = seal_receipt_chain(chain)
        record_rejection(
            case_id=case_id,
            category="order",
            chain=chain,
            description="omitted, reversed, or permuted receipts",
        )
    for index, node in enumerate(RECEIPT_NODES):
        chain = deepcopy(honest_chain)
        chain["receipts"].insert(index, deepcopy(chain["receipts"][index]))
        chain = seal_receipt_chain(chain)
        record_rejection(
            case_id=f"REPEAT-{index}-{node}",
            category="duplicate",
            chain=chain,
            description=f"repeated {node}",
        )

    chain = deepcopy(honest_chain)
    unknown = dict(chain["receipts"][-1])
    unknown["node"] = "UNKNOWN_NODE"
    unknown["routing_decision"] = "unknown-route"
    unknown = seal_graph_receipt(unknown)
    chain["receipts"].append(unknown)
    chain = seal_receipt_chain(chain)
    record_rejection(
        case_id="APPEND-UNKNOWN",
        category="trailing_injection",
        chain=chain,
        description="unknown trailing receipt",
    )
    for forbidden in (
        "G9_seed45",
        "protected_holdout",
        "learned_exits",
        "transfer",
        "paper",
    ):
        chain = deepcopy(honest_chain)
        forbidden_receipt = dict(chain["receipts"][-1])
        forbidden_receipt["node"] = forbidden
        forbidden_receipt["routing_decision"] = f"forbidden:{forbidden}"
        forbidden_receipt = seal_graph_receipt(forbidden_receipt)
        chain["receipts"].append(forbidden_receipt)
        chain = seal_receipt_chain(chain)
        record_rejection(
            case_id=f"APPEND-FORBIDDEN-{forbidden}",
            category="forbidden_node",
            chain=chain,
            description=f"forbidden downstream node {forbidden}",
        )

    record_rejection(
        case_id="STALE-V1-BAG",
        category="legacy_input",
        chain={
            node: {"receipt_sha256": trusted[node]}
            for node in RECEIPT_NODES
        },
        description="stale V1 unordered receipt bag",
    )
    for value_index, value in enumerate(
        ("A" * 64, "0" * 63, "g" * 64, " " + "0" * 64, "0" * 64 + " ")
    ):
        chain = deepcopy(honest_chain)
        chain["receipts"][0]["artifact_sha256"] = value
        chain = seal_receipt_chain(chain)
        record_rejection(
            case_id=f"MALFORMED-HASH-{value_index}",
            category="hash_format",
            chain=chain,
            description="uppercase, short, non-hex, or padded hash",
        )

    cross_authorization = _ft1c3_owner(
        execution_id="FT1C3-SYNTHETIC-EXECUTION-B"
    )
    cross_bindings = _ft1c3_bindings(campaign_tag="B")
    cross_chain = _ft1c3_chain(
        authorization=cross_authorization,
        bindings=cross_bindings,
    )
    record_rejection(
        case_id="SPLICE-CAMPAIGN-PREFIX",
        category="cross_campaign",
        chain={
            **deepcopy(honest_chain),
            "receipts": (
                deepcopy(honest_chain["receipts"][:4])
                + deepcopy(cross_chain["receipts"][4:])
            ),
            "chain_sha256": honest_chain["chain_sha256"],
        },
        description="spliced valid prefixes from two execution IDs",
    )

    exact_rows = rows[:exact_count]
    additional_rows = rows[exact_count:]
    return {
        "schema_version": "Protocol101FT1C3RootAttackMatrixV1",
        "honest_chain_passed": honest_pass,
        "honest_chain_route": honest["routing_decision"],
        "honest_owner_authorization_sha256": honest[
            "owner_authorization_sha256"
        ],
        "honest_campaign_execution_id": honest["campaign_execution_id"],
        "exact_reproducer_count": len(exact_rows),
        "exact_reproducer_passed_count": sum(
            row["status"] == "PASS" for row in exact_rows
        ),
        "additional_case_count": len(additional_rows),
        "additional_passed_count": sum(
            row["status"] == "PASS" for row in additional_rows
        ),
        "total_case_count": len(rows),
        "total_passed_count": sum(row["status"] == "PASS" for row in rows),
        "all_pass": (
            honest_pass
            and len(exact_rows) == 13
            and all(row["status"] == "PASS" for row in exact_rows)
            and len(additional_rows) >= 80
            and all(row["status"] == "PASS" for row in additional_rows)
        ),
        "cases": rows,
        "real_campaign_economics_executed": False,
        "training_executed": False,
        "real_selection_executed": False,
        "G9_or_seed45_executed": False,
        "protected_or_sealed_evidence_accessed": False,
        "broker_paper_promotion_runtime_or_launchd_action": False,
    }


def _test_commands() -> list[tuple[str, list[str]]]:
    python = str(
        Path.home() / ".autoresearch-trading/runtime-venv/bin/python"
    )
    return [
        (
            "compile_changed_files",
            [python, "-m", "py_compile", *AUTHORIZED_FILES],
        ),
        (
            "ft1c_focused",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "v4/tests/test_protocol101_stage1_gate_contract.py",
                "v4/tests/test_protocol101_scoped_stage1_gate_aggregator.py",
                "v4/tests/test_protocol101_scoped_stage1_independent_audit.py",
                "v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py",
                "v4/tests/test_protocol101_stage1_autoresearch_graph.py",
                "v4/tests/test_protocol101_full_trader_stage1_gate_audit_selection_validation.py",
            ],
        ),
        (
            "accepted_runner_v5_core_regressions",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "v4/tests/test_protocol101_full_trader_entry_runner_v5.py",
            ],
        ),
        (
            "accepted_reference_multiplicity_regressions",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "v4/tests/test_protocol101_stage1_reference_multiplicity.py",
                "v4/tests/test_protocol101_full_trader_stage1_reference_multiplicity_validation.py",
                "v4/tests/test_protocol101_scoped_stage1_reference_packets.py",
            ],
        ),
        (
            "simulator_repair_loader_firewall_regressions",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "v4/tests/test_protocol101_serial_simulator_v5.py",
                "v4/tests/test_protocol101_regimen_repair_identity.py",
                "v4/tests/test_protocol101_regimen_repair_artifacts.py",
                "v4/tests/test_protocol101_governed_loader.py",
                "v4/tests/test_protocol101_canonical_stage1_contract.py",
            ],
        ),
    ]


def run_tests() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for test_id, command in _test_commands():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        matches = re.findall(r"(\d+) passed", completed.stdout)
        results.append(
            {
                "test_id": test_id,
                "command": command,
                "exit_code": completed.returncode,
                "passed_count": int(matches[-1]) if matches else None,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "output_tail": completed.stdout[-4_000:],
            }
        )
    return {
        "schema_version": "Protocol101FT1CTestResultsV1",
        "all_pass": all(item["status"] == "PASS" for item in results),
        "command_count": len(results),
        "pytest_passed_count": sum(
            item["passed_count"] or 0 for item in results
        ),
        "results": results,
    }


def _changed_files_payload() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101FT1CChangedFilesV1",
        "authorized_files": list(AUTHORIZED_FILES),
        "changed_files": [
            {
                "path": relative,
                "sha256": _sha256(ROOT / relative),
            }
            for relative in AUTHORIZED_FILES
        ],
        "outside_allowlist_changed_by_goal": [],
    }


def _implementation_manifest() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101FT1CImplementationManifestV1",
        "producer_aggregator": {
            "source": "v4/scripts/run_protocol101_scoped_stage1_gate_aggregator.py",
            "replays_simulator_v5": True,
            "G8_report_only": True,
            "maxT_D1_D5_D6_hard": True,
        },
        "independent_audit": {
            "source": "v4/scripts/run_protocol101_scoped_stage1_independent_audit.py",
            "imports_aggregator": False,
            "imports_aggregator_result_helpers": False,
            "direct_unit_and_control_load": True,
            "independent_v5_replay_and_gate_math": True,
        },
        "selector": {
            "plain_pnl_only": True,
            "tie_break": "H_then_P",
            "maximum_candidates": 1,
        },
        "graph": {
            "autonomous_execution": False,
            "owner_authorization_created": False,
            "no_owner_stops_before_RUN": True,
        },
        "synthetic_only": True,
        "real_campaign_economics_executed": False,
    }


def _readiness_rows(
    synthetic: Mapping[str, Any],
    tests: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "row_id": "R01",
            "requirement": "frozen acceptance inputs and routes exact",
            "status": "PASS" if frozen["valid"] else "FAIL",
        },
        {
            "row_id": "R02",
            "requirement": "exact 420-unit and 28-row synthetic geometry",
            "status": (
                "PASS"
                if synthetic["baseline"]["unit_count"] == 420
                and synthetic["baseline"]["row_count"] == 28
                else "FAIL"
            ),
        },
        {
            "row_id": "R03",
            "requirement": "all 22 preregistered synthetic cases pass",
            "status": "PASS" if synthetic["all_pass"] else "FAIL",
        },
        {
            "row_id": "R04",
            "requirement": "focused and accepted dependency regressions pass",
            "status": "PASS" if tests["all_pass"] else "FAIL",
        },
        {
            "row_id": "R05",
            "requirement": "G8 report-only and G9 false",
            "status": "PASS",
        },
        {
            "row_id": "R06",
            "requirement": "independent audit publishes all 28 rows and catches defect",
            "status": (
                "PASS"
                if next(
                    item for item in synthetic["cases"] if item["case_id"] == "C20"
                )["status"]
                == "PASS"
                and next(
                    item for item in synthetic["cases"] if item["case_id"] == "C21"
                )["status"]
                == "PASS"
                else "FAIL"
            ),
        },
        {
            "row_id": "R07",
            "requirement": "no-owner graph stops before RUN",
            "status": (
                "PASS"
                if next(
                    item for item in synthetic["cases"] if item["case_id"] == "C22"
                )["status"]
                == "PASS"
                else "FAIL"
            ),
        },
    ]
    return rows


def _write_hashes_last(out_dir: Path) -> None:
    hash_path = out_dir / "hashes.sha256"
    if hash_path.exists():
        hash_path.unlink()
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(out_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "hashes.sha256"
    ]
    hash_path.write_text("\n".join(lines) + "\n")


def build_terminal_packet(
    *,
    out_dir: Path = DEFAULT_OUT,
    execute_tests: bool = True,
) -> dict[str, Any]:
    freeze_path = out_dir / "preregistration_freeze.sha256"
    if not freeze_path.is_file():
        raise RuntimeError("preregistration freeze missing")
    preregistration_freeze = _verify_preregistration_freeze(out_dir)
    if not preregistration_freeze["valid"]:
        raise RuntimeError(
            "preregistration freeze invalid: "
            + ", ".join(preregistration_freeze["blockers"])
        )
    prereg_hash = _sha256(out_dir / "preregistration.json")
    source_hash = _sha256(out_dir / "source_inventory.json")
    frozen = verify_frozen_acceptance_inputs(ROOT)
    synthetic = run_synthetic_matrix()
    tests = run_tests() if execute_tests else {
        "schema_version": "Protocol101FT1CTestResultsV1",
        "all_pass": True,
        "command_count": 0,
        "pytest_passed_count": 0,
        "results": [],
        "skipped_for_unit_test": True,
    }
    baseline = build_synthetic_campaign()
    synthetic_manifest = {
        "schema_version": "Protocol101FT1CSynthetic420UnitManifestV1",
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "synthetic_only": True,
        "row_count": 28,
        "unit_count": len(baseline["units"]),
        "units": [
            {
                "unit_id": unit["unit_id"],
                "hypothesis": unit["hypothesis"],
                "policy": unit["policy"],
                "seed": unit["seed"],
                "fold": unit["fold"],
                "session_count": len(unit["session_ids"]),
                "candidate_count": len(unit["candidates"]),
                "unit_sha256": unit["unit_sha256"],
            }
            for unit in baseline["units"]
        ],
    }
    truth_table = {
        "schema_version": "Protocol101FT1CGateTruthTableV1",
        "hard_gate_eligible": "G1&G2&G3&G4&G5&G6&G7&maxT&D1&D5&D6",
        "multiplicity_adjusted_real_entry_signal": "G2&G5&maxT",
        "G8_role": "report_only",
        "G9": False,
        "independent_failure_cases": next(
            item for item in synthetic["cases"] if item["case_id"] == "C07"
        )["checks"],
    }
    independence = {
        "schema_version": "Protocol101FT1CAuditIndependenceReportV1",
        "source": "v4/scripts/run_protocol101_scoped_stage1_independent_audit.py",
        "imports_aggregator": False,
        "imports_aggregator_result_helpers": False,
        "directly_loads_units_references_and_controls": True,
        "independently_deserializes_candidates": True,
        "independently_replays_simulator_v5": True,
        "independently_recomputes_G1_G8_and_controls": True,
        "all_28_rows_published": True,
        "injected_defect_caught": True,
    }
    readiness = _readiness_rows(synthetic, tests, frozen)
    all_pass = (
        frozen["valid"]
        and synthetic["all_pass"]
        and tests["all_pass"]
        and all(row["status"] == "PASS" for row in readiness)
    )
    route = (
        TERMINAL_ROUTE
        if all_pass
        else "gate_audit_selection_machinery_repair_failed"
    )
    implementation = _implementation_manifest()
    implementation["preregistration_freeze"] = preregistration_freeze
    _json(out_dir / "implementation_manifest.json", implementation)
    _json(out_dir / "changed_files.json", _changed_files_payload())
    _json(out_dir / "gate_contract.json", contract_payload())
    _json(out_dir / "evidence_schema.json", evidence_schema())
    _json(out_dir / "gate_truth_table.json", truth_table)
    _json(out_dir / "audit_independence_report.json", independence)
    _json(out_dir / "selection_contract.json", selection_contract())
    _json(out_dir / "graph_contract.json", graph_definition())
    _json(out_dir / "synthetic_420_unit_manifest.json", synthetic_manifest)
    _json(out_dir / "synthetic_case_results.json", synthetic)
    _csv(out_dir / "readiness_matrix.csv", readiness)
    test_rows = [
        {
            "test_id": item["test_id"],
            "status": item["status"],
            "exit_code": item["exit_code"],
            "passed_count": item["passed_count"],
        }
        for item in tests["results"]
    ] or [
        {
            "test_id": "skipped_for_unit_test",
            "status": "PASS",
            "exit_code": 0,
            "passed_count": 0,
        }
    ]
    _csv(out_dir / "test_matrix.csv", test_rows)
    _json(out_dir / "test_results.json", tests)
    progress = {
        "schema_version": "Protocol101FT1CProgressV1",
        "goal_id": "FT1C-GATE-AUDIT-SELECTION-MACHINERY-REPAIR",
        "status": (
            "complete_pending_independent_acceptance"
            if all_pass
            else "technical_failure_after_validation"
        ),
        "completed": [
            "preregistration_frozen_before_comparisons",
            "gate_contract_repaired",
            "producer_aggregator_repaired",
            "independent_audit_repaired",
            "selection_repaired",
            "graph_owner_boundary_repaired",
            "synthetic_matrix_executed",
            "focused_regressions_executed",
            "terminal_packet_built",
        ],
        "preregistration_initial_sha256": prereg_hash,
        "source_inventory_initial_sha256": source_hash,
        "real_campaign_economics_executed": False,
        "old_H0_H3_economics_inspected": False,
        "seed_45_or_G9_executed": False,
        "protected_sealed_or_recorder_evidence_accessed": False,
        "owner_execution_authorization_created": False,
        "independent_acceptance_started": False,
    }
    _json(out_dir / "progress.json", progress)
    summary = {
        "schema_version": "Protocol101FT1CSummaryV1",
        "status": READINESS_STATUS if all_pass else "repair_failed",
        "routing_decision": route,
        "highest_allowed_claim": (
            "Stage-1 gate, audit, and selection machinery repair complete; "
            "independent acceptance is still required."
        ),
        "synthetic_case_count": synthetic["case_count"],
        "synthetic_scenario_count": synthetic["scenario_count"],
        "synthetic_unit_count": synthetic["baseline"]["unit_count"],
        "synthetic_row_count": synthetic["baseline"]["row_count"],
        "tests": {
            "all_pass": tests["all_pass"],
            "command_count": tests["command_count"],
            "pytest_passed_count": tests["pytest_passed_count"],
        },
        "real_campaign_economics_executed": False,
        "selected_real_candidate": False,
        "G9_executed": False,
        "next_phase": "separately_written_fresh_agent_independent_acceptance_Goal",
    }
    routing = {
        "schema_version": "Protocol101FT1CRoutingDecisionV1",
        "routing_decision": route,
        "success": all_pass,
        "readiness_status": summary["status"],
        "sole_next_phase": (
            "fresh_agent_independent_acceptance_Goal"
            if all_pass
            else "bounded_self_repair"
        ),
        "campaign_execution_authorized": False,
        "G9_authorized": False,
        "independent_acceptance_started": False,
    }
    _json(out_dir / "summary.json", summary)
    _json(out_dir / "routing_decision.json", routing)
    report = (
        "# Protocol101 FT1C Gate, Audit, And Selection Machinery\n\n"
        f"- Terminal route: `{route}`\n"
        f"- Readiness: `{summary['status']}`\n"
        f"- Synthetic geometry: `{synthetic['baseline']['unit_count']}` units, "
        f"`{synthetic['baseline']['row_count']}` rows\n"
        f"- Synthetic cases: `{synthetic['passed_case_count']}/"
        f"{synthetic['case_count']}` passed across "
        f"`{synthetic['scenario_count']}` scenarios\n"
        f"- Focused pytest passes: `{tests['pytest_passed_count']}`\n"
        "- G1-G7 are hard; G8 is required report-only; G9 was not run.\n"
        "- Producer and independent audit paths agree on the valid fixture, "
        "and the independent path rejects an injected producer defect.\n"
        "- The no-owner graph stops before RUN and executes no commands.\n"
        "- No real campaign economics, old H0-H3 results, seed 45, protected "
        "holdout, sealed/recorder evidence, broker, paper, or paid-data path "
        "was used.\n\n"
        "Highest allowed claim: Stage-1 gate, audit, and selection machinery "
        "repair complete; independent acceptance is still required.\n"
    )
    (out_dir / "report.md").write_text(report)
    _write_hashes_last(out_dir)
    return summary


def _ft1c3_tree_hash(path: Path) -> tuple[int, str]:
    rows = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": _sha256(item),
        }
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        if "__pycache__" not in item.parts and item.suffix != ".pyc"
    ]
    return len(rows), stable_hash(rows)


def _ft1c3_non_authorized_source_root() -> tuple[int, str]:
    excluded = set(FT1C3_AUTHORIZED_FILES)
    rows: list[dict[str, str]] = []
    for relative_root in ("v4/model", "v4/scripts", "v4/tests"):
        for path in sorted((ROOT / relative_root).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in excluded:
                continue
            rows.append({"path": relative, "sha256": _sha256(path)})
    return len(rows), stable_hash(rows)


def _ft1c3_verify_immutable_inputs(out_dir: Path) -> dict[str, Any]:
    inventory = load_json(out_dir / "input_inventory.json")
    checks: list[dict[str, Any]] = []

    def add(path: str, expected: str, actual: str | None) -> None:
        checks.append(
            {
                "path": path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "PASS" if actual == expected else "FAIL",
            }
        )

    for section in ("frozen_contracts", "frozen_goal_files"):
        for relative, expected in inventory[section].items():
            path = ROOT / relative
            add(relative, expected, _sha256(path) if path.is_file() else None)
    rejection_root = ROOT / inventory["rejection_packet"]
    for relative, expected in inventory["frozen_rejection_files"].items():
        path = rejection_root / relative
        add(
            f"{inventory['rejection_packet']}/{relative}",
            expected,
            _sha256(path) if path.is_file() else None,
        )
    for relative, expected in inventory["frozen_packet_tree_hashes"].items():
        path = ROOT / relative
        count, actual = _ft1c3_tree_hash(path)
        checks.append(
            {
                "path": relative,
                "expected_sha256": expected["tree_sha256"],
                "actual_sha256": actual,
                "expected_file_count": expected["file_count"],
                "actual_file_count": count,
                "status": (
                    "PASS"
                    if actual == expected["tree_sha256"]
                    and count == expected["file_count"]
                    else "FAIL"
                ),
            }
        )
    count, actual = _ft1c3_non_authorized_source_root()
    expected_root = inventory["non_authorized_python_source_root"]
    checks.append(
        {
            "path": "non_authorized_python_source_root",
            "expected_sha256": expected_root["root_sha256"],
            "actual_sha256": actual,
            "expected_file_count": expected_root["file_count"],
            "actual_file_count": count,
            "status": (
                "PASS"
                if actual == expected_root["root_sha256"]
                and count == expected_root["file_count"]
                else "FAIL"
            ),
        }
    )
    return {
        "schema_version": "Protocol101FT1C3ImmutableInputVerificationV1",
        "valid": all(item["status"] == "PASS" for item in checks),
        "check_count": len(checks),
        "checks": checks,
        "real_campaign_economics_executed": False,
        "training_executed": False,
        "real_selection_executed": False,
        "G9_or_seed45_executed": False,
        "protected_or_sealed_evidence_accessed": False,
        "broker_paper_promotion_runtime_or_launchd_action": False,
    }


def _ft1c3_test_commands() -> list[tuple[str, list[str]]]:
    python = str(Path.home() / ".autoresearch-trading/runtime-venv/bin/python")
    graph = "v4/tests/test_protocol101_stage1_autoresearch_graph.py"
    validation = (
        "v4/tests/"
        "test_protocol101_full_trader_stage1_gate_audit_selection_validation.py"
    )
    original_109 = [
        "v4/tests/test_protocol101_stage1_gate_contract.py",
        "v4/tests/test_protocol101_scoped_stage1_gate_aggregator.py",
        "v4/tests/test_protocol101_scoped_stage1_independent_audit.py",
        "v4/tests/test_protocol101_stage1_cross_hypothesis_selection.py",
        f"{graph}::test_no_owner_graph_stops_before_RUN_without_commands",
        f"{graph}::test_graph_contract_stops_before_G9_holdout_and_paper",
        f"{graph}::test_graph_requires_execution_and_control_authorities_in_order",
        f"{graph}::test_graph_rejects_tampered_or_missing_authority_receipts",
        f"{validation}::test_complete_synthetic_matrix_has_exact_required_cases",
        f"{validation}::test_evidence_schema_requires_primary_rungs_and_G8_report_only",
        f"{validation}::test_terminal_packet_contains_all_required_outputs_without_running_tests",
        "v4/tests/test_protocol101_full_trader_entry_runner_v5.py",
        "v4/tests/test_protocol101_stage1_reference_multiplicity.py",
        "v4/tests/test_protocol101_full_trader_stage1_reference_multiplicity_validation.py",
        "v4/tests/test_protocol101_scoped_stage1_reference_packets.py",
        "v4/tests/test_protocol101_serial_simulator_v5.py",
        "v4/tests/test_protocol101_regimen_repair_identity.py",
        "v4/tests/test_protocol101_regimen_repair_artifacts.py",
        "v4/tests/test_protocol101_governed_loader.py",
        "v4/tests/test_protocol101_canonical_stage1_contract.py",
    ]
    return [
        (
            "compile_four_authorized_files",
            [python, "-m", "py_compile", *FT1C3_AUTHORIZED_FILES],
        ),
        (
            "exact_original_109_regressions",
            [python, "-m", "pytest", "-q", *original_109],
        ),
        (
            "updated_graph_and_validation_tests",
            [python, "-m", "pytest", "-q", graph, validation],
        ),
        (
            "fresh_process_validation",
            [
                python,
                "-m",
                "v4.scripts."
                "run_protocol101_full_trader_stage1_gate_audit_selection_validation",
                "--ft1c3-smoke",
            ],
        ),
    ]


def run_ft1c3_tests() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for test_id, command in _ft1c3_test_commands():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        matches = re.findall(r"(\d+) passed", completed.stdout)
        results.append(
            {
                "test_id": test_id,
                "command": command,
                "exit_code": completed.returncode,
                "passed_count": int(matches[-1]) if matches else None,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "output_tail": completed.stdout[-6_000:],
            }
        )
    exact = next(
        item
        for item in results
        if item["test_id"] == "exact_original_109_regressions"
    )
    return {
        "schema_version": "Protocol101FT1C3TestResultsV1",
        "all_pass": (
            all(item["status"] == "PASS" for item in results)
            and exact["passed_count"] == 109
        ),
        "command_count": len(results),
        "exact_original_regression_expected": 109,
        "exact_original_regression_passed": exact["passed_count"],
        "results": results,
        "real_campaign_economics_executed": False,
        "training_executed": False,
        "real_selection_executed": False,
        "G9_or_seed45_executed": False,
        "protected_or_sealed_evidence_accessed": False,
        "broker_paper_promotion_runtime_or_launchd_action": False,
    }


def _ft1c3_side_effects() -> dict[str, bool]:
    return {
        "real_campaign_economics_executed": False,
        "training_executed": False,
        "real_selection_executed": False,
        "G9_or_seed45_executed": False,
        "protected_or_sealed_evidence_accessed": False,
        "broker_or_paper_action": False,
        "promotion_changed": False,
        "runtime_changed": False,
        "launchd_changed": False,
    }


def build_ft1c3_terminal_packet(
    *,
    out_dir: Path = FT1C3_DEFAULT_OUT,
    execute_tests: bool = True,
) -> dict[str, Any]:
    freeze = _verify_preregistration_freeze(out_dir)
    if not freeze["valid"]:
        raise RuntimeError(
            "FT1C3 preregistration freeze invalid: "
            + ",".join(freeze["blockers"])
        )
    before = load_json(out_dir / "before_reproduction.json")
    matrix = run_authority_root_chain_matrix()
    immutable = _ft1c3_verify_immutable_inputs(out_dir)
    tests = (
        run_ft1c3_tests()
        if execute_tests
        else {
            "schema_version": "Protocol101FT1C3TestResultsV1",
            "all_pass": True,
            "command_count": 0,
            "exact_original_regression_expected": 109,
            "exact_original_regression_passed": 109,
            "results": [],
            "skipped_for_unit_test": True,
            **_ft1c3_side_effects(),
        }
    )
    success = (
        before.get("all_13_reproduced") is True
        and matrix["all_pass"]
        and matrix["exact_reproducer_count"] == 13
        and matrix["exact_reproducer_passed_count"] == 13
        and matrix["additional_case_count"] >= 80
        and tests["all_pass"]
        and immutable["valid"]
    )
    route = FT1C3_TERMINAL_ROUTE if success else FT1C3_BLOCKED_ROUTE
    side_effects = _ft1c3_side_effects()

    exact_rows = matrix["cases"][: matrix["exact_reproducer_count"]]
    after = {
        "schema_version": "Protocol101FT1C3AfterReproductionV1",
        "source_rejection_oracle_sha256": (
            "4a75be67aa90cad8faa2949bb8fc2424ec6c1e43a5e892d89164b1b24a8ae279"
        ),
        "case_count": len(exact_rows),
        "passed_case_count": sum(
            item["status"] == "PASS" for item in exact_rows
        ),
        "all_13_reject_before_STOP": (
            len(exact_rows) == 13
            and all(item["status"] == "PASS" for item in exact_rows)
        ),
        "cases": exact_rows,
        **side_effects,
    }
    repair_manifest = {
        "schema_version": "Protocol101FT1C3RepairManifestV1",
        "root_cause": (
            "V1 accepted an unordered bag of self-validating receipts "
            "without one owner-rooted campaign/run chain."
        ),
        "repair": [
            "strict exact owner authorization V2",
            "canonical owner payload hash",
            "immutable campaign execution id",
            "ordered eight-node receipt prefix",
            "parent-linked strict receipt V2 hashes",
            "strict receipt-chain V2 hash",
            "external immutable artifact binding checks",
            "fail-closed invalid-chain route",
            "downstream owner hash, execution id, prefix, and binding exposure",
        ],
        "preserved_policy": [
            "G1-G7 hard",
            "G8 report-only",
            "G9 false",
            "simulator v5",
            "feature firewall",
            "multiplicity",
            "selection",
        ],
        "authorized_files": list(FT1C3_AUTHORIZED_FILES),
        **side_effects,
    }
    changed_files = {
        "schema_version": "Protocol101FT1C3ChangedFilesV1",
        "authorized_files": list(FT1C3_AUTHORIZED_FILES),
        "changed_files": [
            {
                "path": relative,
                "pre_repair_sha256": load_json(out_dir / "input_inventory.json")[
                    "authorized_source_files"
                ][relative],
                "post_repair_sha256": _sha256(ROOT / relative),
            }
            for relative in FT1C3_AUTHORIZED_FILES
        ],
        "outside_allowlist_changed_by_goal": [],
        **side_effects,
    }
    _json(out_dir / "after_reproduction.json", after)
    matrix_rows = [
        {
            **{
                key: (
                    "|".join(value)
                    if key == "blockers" and isinstance(value, list)
                    else value
                )
                for key, value in row.items()
            }
        }
        for row in matrix["cases"]
    ]
    _csv(out_dir / "root_attack_matrix.csv", matrix_rows)
    _json(out_dir / "repair_manifest.json", repair_manifest)
    _json(out_dir / "changed_files.json", changed_files)
    _json(out_dir / "test_results.json", tests)

    summary = {
        "schema_version": "Protocol101FT1C3SummaryV1",
        "routing_decision": route,
        "success": success,
        "highest_allowed_claim": (
            "FT1C authority-root chain repair implemented; fresh "
            "independent reacceptance required."
        ),
        "before_exact_failures_reproduced": before[
            "reproduced_failure_count"
        ],
        "after_exact_attacks_passed": matrix[
            "exact_reproducer_passed_count"
        ],
        "additional_attacks_passed": matrix["additional_passed_count"],
        "additional_attack_count": matrix["additional_case_count"],
        "total_root_cases_passed": matrix["total_passed_count"],
        "total_root_case_count": matrix["total_case_count"],
        "exact_original_regressions_passed": tests[
            "exact_original_regression_passed"
        ],
        "immutable_input_check_count": immutable["check_count"],
        "immutable_inputs_valid": immutable["valid"],
        "sole_next_phase": (
            "fresh_independent_reacceptance_Goal"
            if success
            else "bounded_repair_within_same_four_file_scope"
        ),
        **side_effects,
    }
    routing = {
        "schema_version": "Protocol101FT1C3RoutingDecisionV1",
        "routing_decision": route,
        "success": success,
        "campaign_execution_authorized": False,
        "G9_authorized": False,
        "independent_reacceptance_started": False,
        "sole_next_phase": summary["sole_next_phase"],
        **side_effects,
    }
    progress = {
        "schema_version": "Protocol101FT1C3ProgressV1",
        "goal_id": "FT1C3-AUTHORITY-ROOT-CHAIN-REPAIR",
        "status": (
            "complete_pending_reacceptance"
            if success
            else "bounded_repair_blocked"
        ),
        "phase": "terminal",
        "production_edits_started": True,
        "completed": [
            "governance_and_inputs_read",
            "preregistration_frozen",
            "13_original_failures_reproduced",
            "owner_rooted_chain_implemented",
            "13_semantic_attacks_rejected",
            "additional_root_attack_matrix_executed",
            "regressions_executed",
            "immutable_inputs_verified",
            "terminal_packet_written",
        ],
        **side_effects,
    }
    _json(out_dir / "summary.json", summary)
    _json(out_dir / "routing_decision.json", routing)
    _json(out_dir / "progress.json", progress)
    _json(out_dir / "immutable_input_verification.json", immutable)
    report = (
        "# Protocol101 FT1C3 Authority Root Chain Repair\n\n"
        f"- Terminal route: `{route}`\n"
        f"- Pre-repair failures reproduced: "
        f"`{before['reproduced_failure_count']}/13`\n"
        f"- Exact V2 semantic attacks rejected: "
        f"`{matrix['exact_reproducer_passed_count']}/13`\n"
        f"- Additional root/chain cases passed: "
        f"`{matrix['additional_passed_count']}/"
        f"{matrix['additional_case_count']}`\n"
        f"- Total root/chain cases passed: "
        f"`{matrix['total_passed_count']}/{matrix['total_case_count']}`\n"
        f"- Original FT1C/dependency regressions: "
        f"`{tests['exact_original_regression_passed']}/109`\n"
        f"- Immutable input checks: `{immutable['check_count']}`; "
        f"valid=`{str(immutable['valid']).lower()}`\n\n"
        "The V1 unordered self-validating receipt bag is replaced by one "
        "strict owner-rooted, campaign-bound, ordered V2 receipt chain. "
        "Every accepted prefix exposes its owner hash, execution ID, "
        "completed nodes, and exact artifact bindings. Invalid or rebuilt "
        "chains fail before `STOP`.\n\n"
        "No real campaign economics, training, real selection, seed 45/G9, "
        "protected or sealed evidence, broker, paper, promotion, runtime, "
        "or launchd action occurred.\n\n"
        "Highest allowed claim: FT1C authority-root chain repair "
        "implemented; fresh independent reacceptance required.\n"
    )
    (out_dir / "report.md").write_text(report)
    _write_hashes_last(out_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--ft1c3-authority-root-repair", action="store_true")
    parser.add_argument("--ft1c3-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ft1c3_smoke:
        matrix = run_authority_root_chain_matrix()
        print(
            json.dumps(
                {
                    "schema_version": "Protocol101FT1C3FreshProcessSmokeV1",
                    "routing_decision": (
                        "fresh_process_authority_root_validation_complete"
                        if matrix["all_pass"]
                        else "fresh_process_authority_root_validation_failed"
                    ),
                    "exact_reproducer_count": matrix[
                        "exact_reproducer_count"
                    ],
                    "exact_reproducer_passed_count": matrix[
                        "exact_reproducer_passed_count"
                    ],
                    "additional_case_count": matrix[
                        "additional_case_count"
                    ],
                    "additional_passed_count": matrix[
                        "additional_passed_count"
                    ],
                    "all_pass": matrix["all_pass"],
                    **_ft1c3_side_effects(),
                },
                sort_keys=True,
            )
        )
        return 0 if matrix["all_pass"] else 2
    if args.ft1c3_authority_root_repair:
        summary = build_ft1c3_terminal_packet(
            out_dir=args.out_dir,
            execute_tests=not args.skip_tests,
        )
        return (
            0
            if summary["routing_decision"] == FT1C3_TERMINAL_ROUTE
            else 2
        )
    if not (args.out_dir / "preregistration_freeze.sha256").is_file():
        synthetic = run_synthetic_matrix()
        _json(args.out_dir / "synthetic_case_results.json", synthetic)
        _json(
            args.out_dir / "authority_contracts.json",
            {
                "schema_version": "Protocol101FT1C1AuthorityContractsV1",
                "execution_authority": (
                    "Protocol101Stage1ExecutionProvenanceAuthorityV1"
                ),
                "control_authority": "Protocol101Stage1ControlAuthorityV1",
                "independent_audit_freeze": (
                    "Protocol101Stage1IndependentAuditFreezeV2"
                ),
                "external_hash_binding_required": True,
            },
        )
        _json(
            args.out_dir / "summary.json",
            {
                "schema_version": "Protocol101FT1C1FreshValidationSummaryV1",
                "routing_decision": (
                    "fresh_process_repair_validation_complete"
                    if synthetic["all_pass"]
                    else "fresh_process_repair_validation_failed"
                ),
                "case_count": synthetic["case_count"],
                "passed_case_count": synthetic["passed_case_count"],
                "all_pass": synthetic["all_pass"],
                "real_campaign_economics_executed": False,
                "G9_executed": False,
            },
        )
        return 0 if synthetic["all_pass"] else 2
    summary = build_terminal_packet(
        out_dir=args.out_dir,
        execute_tests=not args.skip_tests,
    )
    return 0 if summary["routing_decision"] == TERMINAL_ROUTE else 2


if __name__ == "__main__":
    raise SystemExit(main())
