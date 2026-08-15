"""Frozen Protocol101 FT1C gate and campaign-evidence contract.

This module contains validation and gate-law primitives only. It does not fit,
score, replay real campaign economics, rank candidates, or authorize G9.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
)


CAMPAIGN_NAMESPACE = "protocol101_full_trader_stage1_entry_fresh_attempt001"
CAMPAIGN_SCHEMA = "Protocol101FT1CSyntheticCampaignPacketV1"
AGGREGATION_SCHEMA = "Protocol101FT1CGateAggregationV1"
AUDIT_SCHEMA = "Protocol101FT1CIndependentAuditV1"
SELECTION_SCHEMA = "Protocol101FT1CSelectionV1"
EXECUTION_PROVENANCE_AUTHORITY_SCHEMA = (
    "Protocol101Stage1ExecutionProvenanceAuthorityV1"
)
CONTROL_AUTHORITY_SCHEMA = "Protocol101Stage1ControlAuthorityV1"
INDEPENDENT_AUDIT_FREEZE_SCHEMA = "Protocol101Stage1IndependentAuditFreezeV2"
REFERENCE_SCHEMAS = (
    "Protocol101FT1CSyntheticReferencesV1",
    "Protocol101FT1CFreshIndependentReferencesV1",
)
CONTROL_SCHEMAS = {
    "D1": "Protocol101FT1CD1ControlV1",
    "D5": "Protocol101FT1CD5ControlV1",
    "D6": "Protocol101FT1CD6ControlV1",
    "maxT": "Protocol101FT1CMaxTControlV1",
}
# V1 remains readable only so the frozen campaign can be reconstructed. It is
# not current selection authority. D1 V2 and the candidate incremental-edge
# gate live in protocol101_d1_v2_contract.py.
LEGACY_D1_SCHEMA = CONTROL_SCHEMAS["D1"]
CURRENT_D1_SCHEMA = "Protocol101D1MatchedRandomControlV2"
HYPOTHESES = ("H0", "H1", "H2", "H3")
POLICIES = tuple(f"P{index}" for index in range(7))
SEEDS = (42, 43, 44)
FOLDS = (1, 2, 3, 4, 5)
ROWS = tuple(f"{hypothesis}/{policy}" for hypothesis in HYPOTHESES for policy in POLICIES)
UNIT_COUNT = len(ROWS) * len(SEEDS) * len(FOLDS)
PRIMARY_FEE = 3.0
PRIMARY_NOISE = "1.0x"
PRIMARY_FILL = "pessimistic_executable"
MAXT_REPLICATES = 20_000
MAXT_EXCEEDANCE_LIMIT = 999
MAXT_ALPHA = 0.05
FLOAT_ABS_TOL = 1e-9

CAMPAIGN_CONTRACT_SHA256 = (
    "7a6f747718419041ca3ce9590fb192c64e915800f3dd0dafac5d9ffdc5ec03f0"
)
RUNNER_ACCEPTANCE_ROUTE = "entry_runner_v5_core_independently_accepted"
REFERENCE_ACCEPTANCE_ROUTE = (
    "reference_multiplicity_machinery_independently_accepted"
)
D6_ROUTE = (
    "SIGNED_SPLIT_FAMILY_SYNCHRONIZATION_SUFFICIENT_FOR_"
    "OFFLINE_LABEL_ONLY_REPAIR"
)

FROZEN_ACCEPTANCE_INPUTS = {
    "v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/acceptance_decision.json": "d6211d8260fe43aadd30037da5d9df373bb53eec2d78ab49866f73dc52b4022f",
    "v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/summary.json": "5a6582895ee23396386fc443f6e98dfba72376863d2456a305b8c5bfcf5eca11",
    "v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/maxT_independent_acceptance.json": "58e5fd26ec885b4563b8bd3511e4075e8d41a498d4755f94ff240a45731e0af2",
    "v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/hashes.sha256": "b9247dd06a808adee8bdbbcdb0adf1cdca1b5f0e5536716ff5861473a0af93e1",
    "v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/acceptance_decision.json": "72cbe1443cbddeb5af6dbb09bf384d9ea479658a4cefacbcda5b9c8463600bd2",
    "v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/hashes.sha256": "b5a70e0dcc6df7c0bd95408a65d1561beaccb615c75ea978060eab15c60b41e0",
    "v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/campaign_contract.json": CAMPAIGN_CONTRACT_SHA256,
    "v4/audit/autoresearch/protocol101_full_trader_entry_campaign_preregistration_attempt001/preregistration.json": "40c3fa07c6fc94aaafdb1abf2b454ede5567c92728f814c38870c8f0eed969c5",
}

PROVENANCE_HASH_FIELDS = (
    "campaign_hash",
    "fold_hash",
    "registry_hash",
    "feature_hash",
    "simulator_hash",
    "schema_hash",
    "model_hash",
    "threshold_hash",
    "epsilon_hash",
    "source_hash",
    "code_hash",
)

DIAGNOSTIC_KEYS = (
    "fee_sensitivity",
    "fill_edge_band",
    "noise_diagnostics",
    "side_time_exposure",
    "concentration",
    "churn",
    "skipped_opportunity",
    "worst_day",
    "underwater_duration",
    "outcome_buckets",
    "harvest_ratio",
    "daily_breaker_events",
    "shape_usage",
)

FORBIDDEN_EVIDENCE_FLAGS = (
    "seed_45_present",
    "G9_run",
    "protected_holdout_present",
    "sealed_evidence_present",
    "recorder_evidence_present",
)

CAMPAIGN_PACKET_FIELDS = (
    "schema_version",
    "campaign_namespace",
    "simulator_version",
    "unit_count",
    "forbidden_evidence",
    "references",
    "controls",
    "units",
    "packet_sha256",
)

FORBIDDEN_FRESHNESS_FIELDS = frozenset(
    {
        "prior_ranking",
        "selected_candidate",
        "promotion",
        "promotion_candidate",
        "g9_result",
        "g9_results",
        "holdout_payload",
        "holdout_result",
        "holdout_results",
        "old_h0_h3_economics",
        "benchmark_economics",
        "stale_campaign",
    }
)
FORBIDDEN_FRESHNESS_MARKERS = frozenset(
    {
        "old_h0_h3_economics",
        "benchmark_economics",
        "stale_campaign_economics",
    }
)


class Protocol101Stage1GateContractError(RuntimeError):
    """Raised when campaign evidence cannot safely enter aggregation."""

    def __init__(self, blockers: str | Iterable[str]):
        values = [blockers] if isinstance(blockers, str) else list(blockers)
        self.blockers = tuple(sorted(set(str(value) for value in values)))
        super().__init__("; ".join(self.blockers))


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop(field, None)
    return result


def seal_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    result = _without_hash(unit, "unit_sha256")
    result["unit_sha256"] = stable_hash(result)
    return result


def seal_campaign_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    result = _without_hash(packet, "packet_sha256")
    result["packet_sha256"] = stable_hash(result)
    return result


def expected_unit_axes() -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (hypothesis, policy, seed, fold)
        for hypothesis in HYPOTHESES
        for policy in POLICIES
        for seed in SEEDS
        for fold in FOLDS
    )


def unit_id(hypothesis: str, policy: str, seed: int, fold: int) -> str:
    return f"{hypothesis}/{policy}/S{seed}/F{fold}"


def verify_frozen_acceptance_inputs(workspace: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    for relative, expected in FROZEN_ACCEPTANCE_INPUTS.items():
        path = workspace / relative
        actual = sha256_path(path) if path.is_file() else None
        passed = actual == expected
        checks.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "PASS" if passed else "FAIL",
            }
        )
        if not passed:
            blockers.append(f"frozen_input_hash_mismatch:{relative}")
    route_paths = {
        "runner": workspace
        / "v4/audit/autoresearch/protocol101_full_trader_entry_runner_v5_core_independent_acceptance_attempt001/acceptance_decision.json",
        "reference": workspace
        / "v4/audit/autoresearch/protocol101_full_trader_stage1_reference_multiplicity_independent_acceptance_attempt001/acceptance_decision.json",
    }
    observed_routes: dict[str, str | None] = {}
    if not blockers:
        for name, path in route_paths.items():
            payload = json.loads(path.read_text())
            observed_routes[name] = payload.get("routing_decision")
        if observed_routes["runner"] != RUNNER_ACCEPTANCE_ROUTE:
            blockers.append("runner_acceptance_route_mismatch")
        if observed_routes["reference"] != REFERENCE_ACCEPTANCE_ROUTE:
            blockers.append("reference_acceptance_route_mismatch")
    return {
        "valid": not blockers,
        "checks": checks,
        "routes": observed_routes,
        "blockers": sorted(blockers),
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def inclusive_lower_bound(value: float, boundary: float) -> bool:
    """Compare an inclusive float boundary without softening strict gates."""
    numeric = float(value)
    threshold = float(boundary)
    if math.isnan(numeric) or math.isnan(threshold):
        return False
    return numeric > threshold or math.isclose(
        numeric,
        threshold,
        rel_tol=0.0,
        abs_tol=FLOAT_ABS_TOL,
    )


def inclusive_upper_bound(value: float, boundary: float) -> bool:
    """Upper-bound twin used only by mathematically inclusive gates."""
    numeric = float(value)
    threshold = float(boundary)
    if math.isnan(numeric) or math.isnan(threshold):
        return False
    return numeric < threshold or math.isclose(
        numeric,
        threshold,
        rel_tol=0.0,
        abs_tol=FLOAT_ABS_TOL,
    )


def _freshness_blockers(value: Any, *, path: str = "$") -> list[str]:
    blockers: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_FRESHNESS_FIELDS:
                blockers.append(f"forbidden_freshness_field:{path}.{key}")
            blockers.extend(
                _freshness_blockers(item, path=f"{path}.{key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            blockers.extend(
                _freshness_blockers(item, path=f"{path}[{index}]")
            )
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in FORBIDDEN_FRESHNESS_MARKERS:
            blockers.append(f"forbidden_freshness_marker:{path}")
    return blockers


def build_execution_provenance_authority(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the post-RUN authority that a graph receipt must freeze."""
    units = packet.get("units")
    if not isinstance(units, list):
        units = []
    bindings: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        bindings.append(
            {
                "unit_id": unit.get("unit_id"),
                "hypothesis": unit.get("hypothesis"),
                "policy": unit.get("policy"),
                "seed": unit.get("seed"),
                "fold": unit.get("fold"),
                "provenance": deepcopy(unit.get("provenance")),
                "unit_sha256": unit.get("unit_sha256"),
            }
        )
    return {
        "schema_version": EXECUTION_PROVENANCE_AUTHORITY_SCHEMA,
        "campaign_namespace": packet.get("campaign_namespace"),
        "unit_count": len(bindings),
        "units": bindings,
    }


def build_control_authority(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Build the post-control authority that a graph receipt must freeze."""
    references = packet.get("references")
    controls = packet.get("controls")
    references = references if isinstance(references, Mapping) else {}
    controls = controls if isinstance(controls, Mapping) else {}
    bindings: dict[str, dict[str, Any]] = {}
    for name, payload in (("references", references), *controls.items()):
        if name not in {"references", "D1", "D5", "D6", "maxT"}:
            continue
        item = payload if isinstance(payload, Mapping) else {}
        bindings[name] = {
            "schema_version": item.get("schema_version"),
            "payload_sha256": stable_hash(item),
            "acceptance_route": item.get("acceptance_route"),
            "receipt_sha256": item.get("receipt_sha256"),
        }
    return {
        "schema_version": CONTROL_AUTHORITY_SCHEMA,
        "campaign_namespace": packet.get("campaign_namespace"),
        "bindings": bindings,
    }


def validate_execution_provenance_authority(
    packet: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
    expected_sha256: str | None,
) -> list[str]:
    blockers: list[str] = []
    if not _is_sha256(expected_sha256):
        blockers.append("execution_authority_expected_hash_invalid")
    if not isinstance(authority, Mapping):
        return blockers + ["execution_authority_missing"]
    if stable_hash(authority) != expected_sha256:
        blockers.append("execution_authority_hash_mismatch")
    expected = build_execution_provenance_authority(packet)
    if set(authority) != set(expected):
        blockers.append("execution_authority_schema_fields_mismatch")
    if authority.get("schema_version") != EXECUTION_PROVENANCE_AUTHORITY_SCHEMA:
        blockers.append("execution_authority_schema_mismatch")
    if authority.get("campaign_namespace") != CAMPAIGN_NAMESPACE:
        blockers.append("execution_authority_namespace_mismatch")
    if authority.get("unit_count") != UNIT_COUNT:
        blockers.append("execution_authority_unit_count_mismatch")
    if authority != expected:
        blockers.append("execution_authority_binding_mismatch")
    units = authority.get("units")
    if isinstance(units, list):
        observed_ids = [
            item.get("unit_id") for item in units if isinstance(item, Mapping)
        ]
        expected_ids = [unit_id(*axis) for axis in expected_unit_axes()]
        if observed_ids != expected_ids:
            blockers.append(
                "execution_authority_unit_grid_missing_reordered_duplicate_or_extra"
            )
    return blockers


def validate_control_authority(
    packet: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
    expected_sha256: str | None,
) -> list[str]:
    blockers: list[str] = []
    if not _is_sha256(expected_sha256):
        blockers.append("control_authority_expected_hash_invalid")
    if not isinstance(authority, Mapping):
        return blockers + ["control_authority_missing"]
    if stable_hash(authority) != expected_sha256:
        blockers.append("control_authority_hash_mismatch")
    expected = build_control_authority(packet)
    if set(authority) != set(expected):
        blockers.append("control_authority_schema_fields_mismatch")
    if authority.get("schema_version") != CONTROL_AUTHORITY_SCHEMA:
        blockers.append("control_authority_schema_mismatch")
    if authority.get("campaign_namespace") != CAMPAIGN_NAMESPACE:
        blockers.append("control_authority_namespace_mismatch")
    bindings = authority.get("bindings")
    if not isinstance(bindings, Mapping) or tuple(bindings) != (
        "references",
        "D1",
        "D5",
        "D6",
        "maxT",
    ):
        blockers.append(
            "control_authority_binding_grid_missing_reordered_duplicate_or_extra"
        )
    if authority != expected:
        blockers.append("control_authority_binding_mismatch")
    return blockers


def _validate_unit(
    unit: Mapping[str, Any],
    expected_axis: tuple[str, str, int, int],
) -> list[str]:
    blockers: list[str] = []
    hypothesis, policy, seed, fold = expected_axis
    identity = unit_id(hypothesis, policy, seed, fold)
    if unit.get("unit_id") != identity:
        blockers.append(f"unit_identity_mismatch:{identity}")
    if (
        unit.get("hypothesis"),
        unit.get("policy"),
        unit.get("seed"),
        unit.get("fold"),
    ) != expected_axis:
        blockers.append(f"unit_axis_mismatch:{identity}")
    if unit.get("campaign_namespace") != CAMPAIGN_NAMESPACE:
        blockers.append(f"unit_namespace_mismatch:{identity}")
    if unit.get("simulator_version") != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION:
        blockers.append(f"unit_simulator_mismatch:{identity}")
    if seed == 45 or unit.get("G9") is not False:
        blockers.append(f"seed45_or_G9_present:{identity}")
    try:
        expected_hash = stable_hash(_without_hash(unit, "unit_sha256"))
    except (TypeError, ValueError):
        expected_hash = None
    if unit.get("unit_sha256") != expected_hash:
        blockers.append(f"unit_hash_mismatch:{identity}")

    provenance = unit.get("provenance")
    if not isinstance(provenance, Mapping):
        blockers.append(f"unit_provenance_missing:{identity}")
    else:
        if tuple(provenance.keys()) != PROVENANCE_HASH_FIELDS:
            blockers.append(f"unit_provenance_schema_mismatch:{identity}")
        if any(not _is_sha256(provenance.get(key)) for key in PROVENANCE_HASH_FIELDS):
            blockers.append(f"unit_provenance_hash_invalid:{identity}")
        if provenance.get("campaign_hash") != CAMPAIGN_CONTRACT_SHA256:
            blockers.append(f"unit_campaign_hash_mismatch:{identity}")

    sessions = unit.get("session_ids")
    candidates = unit.get("candidates")
    if not isinstance(sessions, list) or not sessions:
        blockers.append(f"unit_sessions_missing:{identity}")
        sessions = []
    if len(sessions) != len(set(sessions)):
        blockers.append(f"unit_session_duplicate:{identity}")
    if not isinstance(candidates, list) or not candidates:
        blockers.append(f"unit_candidates_missing:{identity}")
        candidates = []
    candidate_ids: list[str] = []
    candidate_sessions: list[str] = []
    previous_key: tuple[str, int, str] | None = None
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            blockers.append(f"candidate_not_mapping:{identity}:{index}")
            continue
        candidate_id = candidate.get("trade_id")
        candidate_ids.append(str(candidate_id))
        session = str(candidate.get("session"))
        candidate_sessions.append(session)
        ordering_key = (
            session,
            int(candidate.get("decision_time_ns", -1)),
            str(candidate.get("contract_id")),
        )
        if previous_key is not None and ordering_key <= previous_key:
            blockers.append(f"candidate_order_invalid:{identity}")
        previous_key = ordering_key
        if candidate.get("split") != f"SYNTH-{hypothesis}-{policy}-S{seed}":
            blockers.append(f"candidate_split_mismatch:{identity}:{index}")
        if candidate.get("fold") != f"F{fold}":
            blockers.append(f"candidate_fold_mismatch:{identity}:{index}")
        if candidate.get("source_simulator_version") != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION:
            blockers.append(f"candidate_simulator_mismatch:{identity}:{index}")
        if candidate.get("policy_index") != int(policy[1:]):
            blockers.append(f"candidate_policy_mismatch:{identity}:{index}")
        clocks = (
            candidate.get("decision_time_ns"),
            candidate.get("label_source_exit_quote_time_ns"),
            candidate.get("label_realized_exit_time_ns"),
            candidate.get("label_policy_deadline_ns"),
        )
        if any(not isinstance(value, int) for value in clocks):
            blockers.append(f"candidate_clock_missing:{identity}:{index}")
        elif not clocks[0] < clocks[1] <= clocks[2] <= clocks[3]:
            blockers.append(f"candidate_clock_order_invalid:{identity}:{index}")
    if len(candidate_ids) != len(set(candidate_ids)):
        blockers.append(f"unit_trade_duplicate:{identity}")
    if any(session not in sessions for session in candidate_sessions):
        blockers.append(f"candidate_session_not_declared:{identity}")

    diagnostics = unit.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        blockers.append(f"unit_diagnostics_missing:{identity}")
    else:
        if tuple(diagnostics.keys()) != DIAGNOSTIC_KEYS:
            blockers.append(f"unit_diagnostics_schema_mismatch:{identity}")
        if not _all_numbers_finite(diagnostics):
            blockers.append(f"unit_diagnostics_nonfinite:{identity}")
        fees = diagnostics.get("fee_sensitivity", {})
        fills = diagnostics.get("fill_edge_band", {})
        noise = diagnostics.get("noise_diagnostics", {})
        shape = diagnostics.get("shape_usage", {})
        if set(fees) != {"2.60", "3.00", "4.00"}:
            blockers.append(f"fee_diagnostics_incomplete:{identity}")
        if set(fills) != {
            PRIMARY_FILL,
            "mid_diagnostic",
            "favorable_diagnostic",
        }:
            blockers.append(f"fill_diagnostics_incomplete:{identity}")
        if set(noise) != {"0.0x", "0.5x", PRIMARY_NOISE, "2.0x"}:
            blockers.append(f"noise_diagnostics_incomplete:{identity}")
        if set(shape) != set(POLICIES):
            blockers.append(f"shape_usage_incomplete:{identity}")

    observations = unit.get("calibration_observations")
    if not isinstance(observations, list) or not observations:
        blockers.append(f"ece_observations_missing:{identity}")
    elif any(
        not isinstance(item, Mapping)
        or not _finite_number(item.get("confidence"))
        or not _finite_number(item.get("won"))
        or not 0.0 <= float(item["confidence"]) <= 1.0
        or float(item["won"]) not in {0.0, 1.0}
        for item in observations
    ):
        blockers.append(f"ece_observations_invalid:{identity}")
    if not _all_numbers_finite(unit):
        blockers.append(f"unit_nonfinite:{identity}")
    return blockers


def _validate_controls(packet: Mapping[str, Any]) -> list[str]:
    """Validate the immutable FT1C packet under its historical V1 schema.

    A packet passing this function is structurally reproducible, not eligible
    for current selection. Current selection requires a model-free D1 V2
    reaggregation and candidate incremental-edge evidence.
    """
    blockers: list[str] = []
    controls = packet.get("controls")
    if not isinstance(controls, Mapping):
        return ["campaign_controls_missing"]
    if tuple(controls) != ("D1", "D5", "D6", "maxT"):
        blockers.append("campaign_controls_schema_mismatch")
    d1 = controls.get("D1")
    if not isinstance(d1, Mapping):
        blockers.append("D1_missing")
    else:
        if d1.get("schema_version") != CONTROL_SCHEMAS["D1"]:
            blockers.append("D1_schema_mismatch")
        if d1.get("acceptance_route") != REFERENCE_ACCEPTANCE_ROUTE:
            blockers.append("D1_acceptance_route_mismatch")
        if not _is_sha256(d1.get("receipt_sha256")):
            blockers.append("D1_receipt_hash_invalid")
        values_valid = (
            isinstance(d1.get("joint_G1_G2_pass_count"), int)
            and _finite_number(d1.get("median_pnl"))
            and _finite_number(d1.get("median_z"))
        )
        derived = bool(
            values_valid
            and d1["joint_G1_G2_pass_count"] <= 1
            and float(d1["median_pnl"]) <= 0.0
            and float(d1["median_z"]) < 1.0
        )
        if d1.get("independently_accepted") is not True:
            blockers.append("D1_not_independently_accepted")
        if d1.get("passes") is not derived:
            blockers.append("D1_law_mismatch")

    d5 = controls.get("D5")
    if not isinstance(d5, Mapping):
        blockers.append("D5_missing")
    elif not (
        d5.get("schema_version") == CONTROL_SCHEMAS["D5"]
        and d5.get("independently_accepted") is True
        and d5.get("identity_complete") is True
        and d5.get("acceptance_route") == REFERENCE_ACCEPTANCE_ROUTE
        and _is_sha256(d5.get("receipt_sha256"))
    ):
        blockers.append("D5_mismatch")

    d6 = controls.get("D6")
    if not isinstance(d6, Mapping):
        blockers.append("D6_missing")
    elif not (
        d6.get("schema_version") == CONTROL_SCHEMAS["D6"]
        and d6.get("independently_accepted") is True
        and d6.get("route") == D6_ROUTE
        and d6.get("acceptance_route") == REFERENCE_ACCEPTANCE_ROUTE
        and _is_sha256(d6.get("receipt_sha256"))
    ):
        blockers.append("D6_mismatch")

    maxt = controls.get("maxT")
    if not isinstance(maxt, Mapping):
        blockers.append("maxT_missing")
        return blockers
    if not (
        maxt.get("schema_version") == CONTROL_SCHEMAS["maxT"]
        and maxt.get("acceptance_route") == REFERENCE_ACCEPTANCE_ROUTE
        and _is_sha256(maxt.get("receipt_sha256"))
        and maxt.get("valid") is True
        and maxt.get("family_size") == len(ROWS)
        and maxt.get("replicates") == MAXT_REPLICATES
        and maxt.get("tie_rule") == "greater_than_or_equal"
    ):
        blockers.append("maxT_invalid")
    rows = maxt.get("rows")
    if not isinstance(rows, list) or [item.get("row_id") for item in rows if isinstance(item, Mapping)] != list(ROWS):
        blockers.append("maxT_row_grid_mismatch")
        return blockers
    for expected_row, item in zip(ROWS, rows):
        if not isinstance(item, Mapping):
            blockers.append(f"maxT_row_invalid:{expected_row}")
            continue
        p_fwer = item.get("p_FWER")
        exceedance = item.get("exceedance_count")
        if (
            not _finite_number(p_fwer)
            or not 0.0 <= float(p_fwer) <= 1.0
            or not isinstance(exceedance, int)
            or not 0 <= exceedance <= MAXT_REPLICATES
        ):
            blockers.append(f"maxT_row_invalid:{expected_row}")
            continue
        derived = bool(
            float(p_fwer) <= MAXT_ALPHA
            and exceedance <= MAXT_EXCEEDANCE_LIMIT
        )
        if item.get("hard_pass") is not derived:
            blockers.append(f"maxT_row_law_mismatch:{expected_row}")
    return blockers


def validate_campaign_packet(
    packet: Mapping[str, Any],
    *,
    execution_authority: Mapping[str, Any] | None = None,
    execution_authority_sha256: str | None = None,
    control_authority: Mapping[str, Any] | None = None,
    control_authority_sha256: str | None = None,
) -> list[str]:
    blockers: list[str] = []
    if tuple(packet) != CAMPAIGN_PACKET_FIELDS:
        blockers.append("campaign_packet_schema_fields_mismatch")
    blockers.extend(_freshness_blockers(packet))
    if packet.get("schema_version") != CAMPAIGN_SCHEMA:
        blockers.append("campaign_schema_mismatch")
    if packet.get("campaign_namespace") != CAMPAIGN_NAMESPACE:
        blockers.append("campaign_namespace_mismatch")
    if packet.get("simulator_version") != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION:
        blockers.append("campaign_simulator_mismatch")
    if packet.get("unit_count") != UNIT_COUNT:
        blockers.append("campaign_unit_count_mismatch")
    flags = packet.get("forbidden_evidence")
    if not isinstance(flags, Mapping):
        blockers.append("forbidden_evidence_flags_missing")
    else:
        if tuple(flags.keys()) != FORBIDDEN_EVIDENCE_FLAGS:
            blockers.append("forbidden_evidence_flags_schema_mismatch")
        if any(flags.get(key) is not False for key in FORBIDDEN_EVIDENCE_FLAGS):
            blockers.append("forbidden_evidence_present")

    units = packet.get("units")
    if not isinstance(units, list):
        blockers.append("campaign_units_missing")
        units = []
    expected_axes = expected_unit_axes()
    observed_axes = [
        (
            unit.get("hypothesis"),
            unit.get("policy"),
            unit.get("seed"),
            unit.get("fold"),
        )
        for unit in units
        if isinstance(unit, Mapping)
    ]
    if tuple(observed_axes) != expected_axes:
        blockers.append("unit_axis_grid_missing_reordered_duplicate_or_extra")
    if len(units) == UNIT_COUNT:
        for unit, axis in zip(units, expected_axes):
            if isinstance(unit, Mapping):
                blockers.extend(_validate_unit(unit, axis))
            else:
                blockers.append(f"unit_not_mapping:{unit_id(*axis)}")

    references = packet.get("references")
    if not isinstance(references, Mapping):
        blockers.append("references_missing")
    else:
        if references.get("schema_version") not in REFERENCE_SCHEMAS:
            blockers.append("reference_schema_mismatch")
        if references.get("acceptance_route") != REFERENCE_ACCEPTANCE_ROUTE:
            blockers.append("reference_acceptance_route_mismatch")
        if not _is_sha256(references.get("receipt_sha256")):
            blockers.append("reference_receipt_hash_invalid")
        rows = references.get("rows")
        if not isinstance(rows, list) or [
            item.get("row_id") for item in rows if isinstance(item, Mapping)
        ] != list(ROWS):
            blockers.append("reference_row_grid_mismatch")
        else:
            for row_id, row in zip(ROWS, rows):
                if not _finite_number(row.get("heuristic_pooled_pnl")):
                    blockers.append(f"heuristic_nonfinite:{row_id}")
                seed_z = row.get("matched_null_z_by_seed")
                if not isinstance(seed_z, Mapping) or tuple(seed_z.keys()) != tuple(str(seed) for seed in SEEDS):
                    blockers.append(f"matched_null_seed_grid_mismatch:{row_id}")
                elif any(not _finite_number(value) for value in seed_z.values()):
                    blockers.append(f"matched_null_nonfinite:{row_id}")

    blockers.extend(_validate_controls(packet))
    blockers.extend(
        validate_execution_provenance_authority(
            packet,
            execution_authority,
            execution_authority_sha256,
        )
    )
    blockers.extend(
        validate_control_authority(
            packet,
            control_authority,
            control_authority_sha256,
        )
    )
    try:
        expected_packet_hash = stable_hash(
            _without_hash(packet, "packet_sha256")
        )
    except (TypeError, ValueError):
        expected_packet_hash = None
    if packet.get("packet_sha256") != expected_packet_hash:
        blockers.append("campaign_packet_hash_mismatch")
    if not _all_numbers_finite(packet):
        blockers.append("campaign_packet_nonfinite")
    return sorted(set(blockers))


def assert_campaign_packet(
    packet: Mapping[str, Any],
    *,
    execution_authority: Mapping[str, Any] | None = None,
    execution_authority_sha256: str | None = None,
    control_authority: Mapping[str, Any] | None = None,
    control_authority_sha256: str | None = None,
) -> None:
    blockers = validate_campaign_packet(
        packet,
        execution_authority=execution_authority,
        execution_authority_sha256=execution_authority_sha256,
        control_authority=control_authority,
        control_authority_sha256=control_authority_sha256,
    )
    if blockers:
        raise Protocol101Stage1GateContractError(blockers)


def ten_bin_ece(observations: Sequence[Mapping[str, Any]]) -> float:
    total = len(observations)
    if total <= 0:
        raise Protocol101Stage1GateContractError("ece_observations_missing")
    weighted_error = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        members = [
            item
            for item in observations
            if lower <= float(item["confidence"]) < upper
            or (index == 9 and float(item["confidence"]) == 1.0)
        ]
        if not members:
            continue
        confidence = statistics.fmean(float(item["confidence"]) for item in members)
        outcome = statistics.fmean(float(item["won"]) for item in members)
        weighted_error += len(members) / total * abs(confidence - outcome)
    return float(weighted_error)


def calmar_payload(pnl: float, max_drawdown: float) -> dict[str, Any]:
    if pnl > 0.0 and math.isclose(max_drawdown, 0.0, abs_tol=FLOAT_ABS_TOL):
        return {"value": None, "classification": "positive_zero_drawdown_infinite"}
    value = pnl / max_drawdown if max_drawdown > 0.0 else None
    return {"value": value, "classification": "finite" if value is not None else "nonpositive_zero_drawdown"}


def evaluate_row_gates(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    heuristic_pooled_pnl: float,
    maxT_pass: bool,
    global_controls_pass: bool,
) -> dict[str, Any]:
    if [int(item["seed"]) for item in seed_results] != list(SEEDS):
        raise Protocol101Stage1GateContractError("seed_result_grid_mismatch")
    seed_g1: list[bool] = []
    seed_g4: list[bool] = []
    seed_g6: list[bool] = []
    seed_g7: list[bool] = []
    pooled_pnls: list[float] = []
    null_zs: list[float] = []
    eces: list[float] = []
    for seed in seed_results:
        folds = seed["fold_results"]
        fold_pnls = [float(item["fee_adjusted_net_pnl"]) for item in folds]
        pooled_pnl = float(seed["continuous_oof_replay"]["fee_adjusted_net_pnl"])
        pooled_pnls.append(pooled_pnl)
        null_zs.append(float(seed["matched_null_z"]))
        eces.append(float(seed["ece"]))
        seed_g1.append(
            sum(value > 0.0 for value in fold_pnls) >= 4 and pooled_pnl > 0.0
        )
        max_drawdown = float(seed["continuous_oof_replay"]["max_drawdown"])
        calmar_pass = (
            pooled_pnl > 0.0
            and (
                math.isclose(max_drawdown, 0.0, abs_tol=FLOAT_ABS_TOL)
                or inclusive_lower_bound(
                    pooled_pnl / max_drawdown,
                    1.0,
                )
            )
        )
        equity_pass = all(
            inclusive_lower_bound(
                float(item["minimum_equity"]),
                5_000.0,
            )
            for item in folds
        )
        seed_g4.append(calmar_pass and equity_pass)
        era_values: dict[str, list[float]] = {}
        for fold in folds:
            era_values.setdefault(str(fold["era"]), []).append(
                float(fold["fee_adjusted_net_pnl"])
            )
        seed_g6.append(
            all(
                inclusive_lower_bound(statistics.median(values), 0.0)
                for values in era_values.values()
            )
        )
        seed_g7.append(
            all(
                inclusive_lower_bound(
                    float(item["trades_per_session"]),
                    0.3,
                )
                and inclusive_upper_bound(
                    float(item["trades_per_session"]),
                    6.0,
                )
                for item in folds
            )
        )
    gates = {
        "G1": all(seed_g1),
        "G2": statistics.median(null_zs) >= 3.0,
        "G3": statistics.median(pooled_pnls) > float(heuristic_pooled_pnl),
        "G4": all(seed_g4),
        "G5": all(seed_g1) and min(null_zs) >= 2.0,
        "G6": all(seed_g6),
        "G7": all(seed_g7),
        "G8": all(value <= 0.10 for value in eces),
        "G9": False,
    }
    hard_gate_eligible = bool(
        global_controls_pass
        and maxT_pass
        and all(gates[f"G{index}"] for index in range(1, 8))
    )
    adjusted_signal = bool(gates["G2"] and gates["G5"] and maxT_pass)
    g6_only_blocked = bool(
        global_controls_pass
        and maxT_pass
        and not gates["G6"]
        and all(gates[f"G{index}"] for index in (1, 2, 3, 4, 5, 7))
    )
    return {
        "gates": gates,
        "G8_role": "report_only",
        "G8_benchmark": 0.10,
        "hard_gate_eligible": hard_gate_eligible,
        "multiplicity_adjusted_real_entry_signal": adjusted_signal,
        "G6_only_blocked": g6_only_blocked,
        "median_seed_fee_adjusted_continuous_strict_serial_net_pnl": float(
            statistics.median(pooled_pnls)
        ),
        "median_seed_matched_null_z": float(statistics.median(null_zs)),
        "worst_seed_matched_null_z": float(min(null_zs)),
        "median_seed_ece": float(statistics.median(eces)),
    }


def contract_payload() -> dict[str, Any]:
    return {
        "schema_version": "Protocol101FT1CGateContractV1",
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "rows": list(ROWS),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "unit_count": UNIT_COUNT,
        "execution_provenance_authority_schema": (
            EXECUTION_PROVENANCE_AUTHORITY_SCHEMA
        ),
        "control_authority_schema": CONTROL_AUTHORITY_SCHEMA,
        "independent_audit_freeze_schema": INDEPENDENT_AUDIT_FREEZE_SCHEMA,
        "hard_gates": [f"G{index}" for index in range(1, 8)],
        "G8": {"required": True, "role": "report_only", "benchmark": 0.10},
        "G9": {"run": False, "authorized": False},
        "maxT": {
            "family_size": len(ROWS),
            "replicates": MAXT_REPLICATES,
            "p_FWER_max": MAXT_ALPHA,
            "exceedance_count_max": MAXT_EXCEEDANCE_LIMIT,
        },
        "hard_gate_eligible": "G1&G2&G3&G4&G5&G6&G7&maxT&D1&D5&D6",
        "selection_authority": {
            "historical_packet_D1_schema": LEGACY_D1_SCHEMA,
            "current_D1_schema": CURRENT_D1_SCHEMA,
            "current_D1_absolute_pnl_gate": False,
            "candidate_incremental_edge_required": True,
            "candidate_incremental_ci_lower": "strictly_greater_than_zero",
            "candidate_incremental_multiplicity_adjusted_p_max": 0.05,
        },
        "multiplicity_adjusted_real_entry_signal": "G2&G5&maxT",
        "primary_evidence": {
            "noise": PRIMARY_NOISE,
            "fill": PRIMARY_FILL,
            "fee_usd": PRIMARY_FEE,
        },
        "required_diagnostics": list(DIAGNOSTIC_KEYS),
        "inclusive_float_boundary_policy": {
            "absolute_tolerance": FLOAT_ABS_TOL,
            "relative_tolerance": 0.0,
            "only": ["G4_Calmar", "G4_equity", "G6_era", "G7_frequency"],
        },
        "fail_closed": True,
    }
