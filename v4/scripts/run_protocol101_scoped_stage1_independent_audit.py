"""Read-only independent FT1C audit of fresh Stage-1 campaign evidence.

The implementation deliberately does not import the producer aggregator or
its result helpers. It loads units and controls directly, independently
deserializes candidates, replays simulator v5, and reconstructs all 28 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)


PACKET_SCHEMA = "Protocol101FT1CSyntheticCampaignPacketV1"
RESULT_SCHEMA = "Protocol101FT1CGateAggregationV1"
AUDIT_SCHEMA = "Protocol101FT1CIndependentAuditV1"
EXECUTION_AUTHORITY_SCHEMA = "Protocol101Stage1ExecutionProvenanceAuthorityV1"
CONTROL_AUTHORITY_SCHEMA = "Protocol101Stage1ControlAuthorityV1"
FREEZE_SCHEMA = "Protocol101Stage1IndependentAuditFreezeV2"
REFERENCE_SCHEMAS = {
    "Protocol101FT1CSyntheticReferencesV1",
    "Protocol101FT1CFreshIndependentReferencesV1",
}
CONTROL_SCHEMAS = {
    "D1": "Protocol101FT1CD1ControlV1",
    "D5": "Protocol101FT1CD5ControlV1",
    "D6": "Protocol101FT1CD6ControlV1",
    "maxT": "Protocol101FT1CMaxTControlV1",
}
NAMESPACE = "protocol101_full_trader_stage1_entry_fresh_attempt001"
HYPOTHESES = ("H0", "H1", "H2", "H3")
POLICIES = tuple(f"P{value}" for value in range(7))
SEEDS = (42, 43, 44)
FOLDS = (1, 2, 3, 4, 5)
ROWS = tuple(f"{h}/{p}" for h in HYPOTHESES for p in POLICIES)
EXPECTED_UNITS = 420
PRIMARY_FEE = 3.0
D6_ROUTE = (
    "SIGNED_SPLIT_FAMILY_SYNCHRONIZATION_SUFFICIENT_FOR_"
    "OFFLINE_LABEL_ONLY_REPAIR"
)
REFERENCE_ROUTE = "reference_multiplicity_machinery_independently_accepted"
CAMPAIGN_CONTRACT_SHA256 = (
    "7a6f747718419041ca3ce9590fb192c64e915800f3dd0dafac5d9ffdc5ec03f0"
)
PROVENANCE_FIELDS = (
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
DIAGNOSTICS = (
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
FLOAT_ABS_TOL = 1e-9
FORBIDDEN_FRESHNESS_FIELDS = {
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
FORBIDDEN_FRESHNESS_MARKERS = {
    "old_h0_h3_economics",
    "benchmark_economics",
    "stale_campaign_economics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-packet", type=Path, required=True)
    parser.add_argument("--producer-result", type=Path, required=True)
    parser.add_argument(
        "--execution-provenance-authority",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--execution-provenance-authority-sha256",
        required=True,
    )
    parser.add_argument("--control-authority", type=Path, required=True)
    parser.add_argument("--control-authority-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _without(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = deepcopy(dict(payload))
    value.pop(key, None)
    return value


def payload_differences(
    expected: Any,
    observed: Any,
    *,
    path: str = "$",
    exact_keys: bool = True,
    limit: int = 200,
) -> list[str]:
    """Return bounded structural/numeric differences for audit reporting."""
    differences: list[str] = []

    def visit(left: Any, right: Any, location: str) -> None:
        if len(differences) >= limit:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_keys = set(left)
            right_keys = set(right)
            if exact_keys and left_keys != right_keys:
                differences.append(
                    f"{location}:keys:{sorted(left_keys)}!={sorted(right_keys)}"
                )
            for key in sorted(left_keys & right_keys):
                visit(left[key], right[key], f"{location}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(
                    f"{location}:length:{len(left)}!={len(right)}"
                )
            for index, (left_item, right_item) in enumerate(
                zip(left, right)
            ):
                visit(left_item, right_item, f"{location}[{index}]")
            return
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            if not math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=1e-9
            ):
                differences.append(f"{location}:{left}!={right}")
            return
        if left != right:
            differences.append(f"{location}:{left!r}!={right!r}")

    visit(expected, observed, path)
    return differences


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _inclusive_lower(value: float, boundary: float) -> bool:
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


def _inclusive_upper(value: float, boundary: float) -> bool:
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


def _freshness_defects(value: Any, *, path: str = "$") -> list[str]:
    defects: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_FRESHNESS_FIELDS:
                defects.append(f"forbidden_freshness_field:{path}.{key}")
            defects.extend(
                _freshness_defects(item, path=f"{path}.{key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            defects.extend(
                _freshness_defects(item, path=f"{path}[{index}]")
            )
    elif isinstance(value, str):
        if value.strip().lower() in FORBIDDEN_FRESHNESS_MARKERS:
            defects.append(f"forbidden_freshness_marker:{path}")
    return defects


def _expected_execution_authority(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    units = packet.get("units")
    units = units if isinstance(units, list) else []
    bindings = []
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
        "schema_version": EXECUTION_AUTHORITY_SCHEMA,
        "campaign_namespace": packet.get("campaign_namespace"),
        "unit_count": len(bindings),
        "units": bindings,
    }


def _expected_control_authority(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
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


def _validate_authorities(
    packet: Mapping[str, Any],
    *,
    execution_authority: Mapping[str, Any] | None,
    execution_authority_sha256: str | None,
    control_authority: Mapping[str, Any] | None,
    control_authority_sha256: str | None,
) -> list[str]:
    defects: list[str] = []
    if not _is_sha256(execution_authority_sha256):
        defects.append("execution_authority_expected_hash_invalid")
    if not isinstance(execution_authority, Mapping):
        defects.append("execution_authority_missing")
    else:
        if stable_hash(execution_authority) != execution_authority_sha256:
            defects.append("execution_authority_hash_mismatch")
        if execution_authority != _expected_execution_authority(packet):
            defects.append("execution_authority_binding_mismatch")
        if execution_authority.get("schema_version") != EXECUTION_AUTHORITY_SCHEMA:
            defects.append("execution_authority_schema_mismatch")
        if execution_authority.get("unit_count") != EXPECTED_UNITS:
            defects.append("execution_authority_unit_count_mismatch")
    if not _is_sha256(control_authority_sha256):
        defects.append("control_authority_expected_hash_invalid")
    if not isinstance(control_authority, Mapping):
        defects.append("control_authority_missing")
    else:
        if stable_hash(control_authority) != control_authority_sha256:
            defects.append("control_authority_hash_mismatch")
        if control_authority != _expected_control_authority(packet):
            defects.append("control_authority_binding_mismatch")
        if control_authority.get("schema_version") != CONTROL_AUTHORITY_SCHEMA:
            defects.append("control_authority_schema_mismatch")
        bindings = control_authority.get("bindings")
        if not isinstance(bindings, Mapping) or tuple(bindings) != (
            "references",
            "D1",
            "D5",
            "D6",
            "maxT",
        ):
            defects.append("control_authority_binding_grid_mismatch")
    return defects


def _expected_axes() -> list[tuple[str, str, int, int]]:
    return [
        (hypothesis, policy, seed, fold)
        for hypothesis in HYPOTHESES
        for policy in POLICIES
        for seed in SEEDS
        for fold in FOLDS
    ]


def validate_packet_independently(
    packet: Mapping[str, Any],
    *,
    execution_authority: Mapping[str, Any] | None = None,
    execution_authority_sha256: str | None = None,
    control_authority: Mapping[str, Any] | None = None,
    control_authority_sha256: str | None = None,
) -> list[str]:
    defects: list[str] = []
    defects.extend(_freshness_defects(packet))
    if tuple(packet) != (
        "schema_version",
        "campaign_namespace",
        "simulator_version",
        "unit_count",
        "forbidden_evidence",
        "references",
        "controls",
        "units",
        "packet_sha256",
    ):
        defects.append("packet_schema_fields_mismatch")
    if packet.get("schema_version") != PACKET_SCHEMA:
        defects.append("packet_schema_mismatch")
    if packet.get("campaign_namespace") != NAMESPACE:
        defects.append("packet_namespace_mismatch")
    if packet.get("simulator_version") != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION:
        defects.append("packet_simulator_mismatch")
    if packet.get("unit_count") != EXPECTED_UNITS:
        defects.append("packet_unit_count_mismatch")
    flags = packet.get("forbidden_evidence")
    if not isinstance(flags, Mapping) or any(value is not False for value in flags.values()):
        defects.append("forbidden_evidence_present_or_unreported")
    units = packet.get("units")
    if not isinstance(units, list):
        return defects + ["units_missing"]
    axes = [
        (
            item.get("hypothesis"),
            item.get("policy"),
            item.get("seed"),
            item.get("fold"),
        )
        for item in units
        if isinstance(item, Mapping)
    ]
    if axes != _expected_axes():
        defects.append("unit_grid_missing_reordered_duplicate_or_extra")
    for expected, unit in zip(_expected_axes(), units):
        if not isinstance(unit, Mapping):
            defects.append(f"unit_not_mapping:{expected}")
            continue
        hypothesis, policy, seed, fold = expected
        expected_id = f"{hypothesis}/{policy}/S{seed}/F{fold}"
        if unit.get("unit_id") != expected_id:
            defects.append(f"unit_id_mismatch:{expected_id}")
        try:
            expected_unit_hash = stable_hash(_without(unit, "unit_sha256"))
        except (TypeError, ValueError):
            expected_unit_hash = None
        if unit.get("unit_sha256") != expected_unit_hash:
            defects.append(f"unit_hash_mismatch:{expected_id}")
        if unit.get("campaign_namespace") != NAMESPACE:
            defects.append(f"unit_namespace_mismatch:{expected_id}")
        if unit.get("simulator_version") != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION:
            defects.append(f"unit_simulator_mismatch:{expected_id}")
        if unit.get("G9") is not False or seed == 45:
            defects.append(f"G9_or_seed45_present:{expected_id}")
        provenance = unit.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or tuple(provenance) != PROVENANCE_FIELDS
            or any(
                not _is_sha256(provenance.get(field))
                for field in PROVENANCE_FIELDS
            )
            or provenance.get("campaign_hash") != CAMPAIGN_CONTRACT_SHA256
        ):
            defects.append(f"provenance_invalid:{expected_id}")
        sessions = unit.get("session_ids")
        candidates = unit.get("candidates")
        if (
            not isinstance(sessions, list)
            or not sessions
            or len(sessions) != len(set(sessions))
        ):
            defects.append(f"session_identity_invalid:{expected_id}")
            sessions = []
        if not isinstance(candidates, list) or not candidates:
            defects.append(f"candidate_stream_missing:{expected_id}")
            candidates = []
        trade_ids = [item.get("trade_id") for item in candidates if isinstance(item, Mapping)]
        if len(trade_ids) != len(set(trade_ids)):
            defects.append(f"trade_identity_duplicate:{expected_id}")
        observed_order = [
            (
                item.get("session"),
                item.get("decision_time_ns"),
                item.get("contract_id"),
            )
            for item in candidates
            if isinstance(item, Mapping)
        ]
        if observed_order != sorted(observed_order):
            defects.append(f"candidate_order_invalid:{expected_id}")
        if any(item.get("session") not in sessions for item in candidates if isinstance(item, Mapping)):
            defects.append(f"candidate_session_undeclared:{expected_id}")
        diagnostics = unit.get("diagnostics")
        if not isinstance(diagnostics, Mapping) or tuple(diagnostics) != DIAGNOSTICS:
            defects.append(f"diagnostics_incomplete:{expected_id}")
        elif (
            set(diagnostics.get("fee_sensitivity", {}))
            != {"2.60", "3.00", "4.00"}
            or set(diagnostics.get("fill_edge_band", {}))
            != {
                "pessimistic_executable",
                "mid_diagnostic",
                "favorable_diagnostic",
            }
            or set(diagnostics.get("noise_diagnostics", {}))
            != {"0.0x", "0.5x", "1.0x", "2.0x"}
            or set(diagnostics.get("shape_usage", {})) != set(POLICIES)
        ):
            defects.append(f"primary_or_diagnostic_rung_missing:{expected_id}")
        observations = unit.get("calibration_observations")
        if not isinstance(observations, list) or not observations:
            defects.append(f"ece_observations_missing:{expected_id}")
    reference_payload = packet.get("references")
    if not isinstance(reference_payload, Mapping) or (
        reference_payload.get("schema_version") not in REFERENCE_SCHEMAS
        or reference_payload.get("acceptance_route") != REFERENCE_ROUTE
        or not _is_sha256(reference_payload.get("receipt_sha256"))
    ):
        defects.append("reference_schema_or_receipt_invalid")
        reference_payload = {}
    references = reference_payload.get("rows")
    if not isinstance(references, list) or [
        item.get("row_id") for item in references if isinstance(item, Mapping)
    ] != list(ROWS):
        defects.append("reference_grid_invalid")
    else:
        for row_id, reference in zip(ROWS, references):
            heuristic = reference.get("heuristic_pooled_pnl")
            seed_z = reference.get("matched_null_z_by_seed")
            if (
                not isinstance(heuristic, (int, float))
                or isinstance(heuristic, bool)
                or not math.isfinite(float(heuristic))
            ):
                defects.append(f"heuristic_invalid:{row_id}")
            if (
                not isinstance(seed_z, Mapping)
                or tuple(seed_z) != tuple(str(seed) for seed in SEEDS)
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    for value in seed_z.values()
                )
            ):
                defects.append(f"matched_null_invalid:{row_id}")
    controls = packet.get("controls")
    if not isinstance(controls, Mapping):
        defects.append("controls_missing")
    else:
        if tuple(controls) != ("D1", "D5", "D6", "maxT"):
            defects.append("controls_schema_fields_mismatch")
        d1 = controls.get("D1")
        if not isinstance(d1, Mapping):
            defects.append("D1_missing")
        else:
            if (
                d1.get("schema_version") != CONTROL_SCHEMAS["D1"]
                or d1.get("acceptance_route") != REFERENCE_ROUTE
                or not _is_sha256(d1.get("receipt_sha256"))
            ):
                defects.append("D1_schema_or_receipt_invalid")
            d1_law = bool(
                isinstance(d1.get("joint_G1_G2_pass_count"), int)
                and isinstance(d1.get("median_pnl"), (int, float))
                and not isinstance(d1.get("median_pnl"), bool)
                and math.isfinite(float(d1["median_pnl"]))
                and isinstance(d1.get("median_z"), (int, float))
                and not isinstance(d1.get("median_z"), bool)
                and math.isfinite(float(d1["median_z"]))
                and d1["joint_G1_G2_pass_count"] <= 1
                and float(d1["median_pnl"]) <= 0.0
                and float(d1["median_z"]) < 1.0
            )
            if d1.get("passes") is not d1_law:
                defects.append("D1_law_mismatch")
            if d1.get("independently_accepted") is not True:
                defects.append("D1_not_independently_accepted")
        d5 = controls.get("D5")
        if not isinstance(d5, Mapping) or not (
            d5.get("schema_version") == CONTROL_SCHEMAS["D5"]
            and d5.get("independently_accepted") is True
            and d5.get("identity_complete") is True
            and d5.get("acceptance_route") == REFERENCE_ROUTE
            and _is_sha256(d5.get("receipt_sha256"))
        ):
            defects.append("D5_binding_invalid")
        d6 = controls.get("D6")
        if not isinstance(d6, Mapping) or not (
            d6.get("schema_version") == CONTROL_SCHEMAS["D6"]
            and d6.get("independently_accepted") is True
            and d6.get("route") == D6_ROUTE
            and d6.get("acceptance_route") == REFERENCE_ROUTE
            and _is_sha256(d6.get("receipt_sha256"))
        ):
            defects.append("D6_binding_invalid")
        maxT = controls.get("maxT")
        if not isinstance(maxT, Mapping) or not (
            maxT.get("schema_version") == CONTROL_SCHEMAS["maxT"]
            and maxT.get("acceptance_route") == REFERENCE_ROUTE
            and _is_sha256(maxT.get("receipt_sha256"))
            and maxT.get("valid") is True
            and maxT.get("family_size") == 28
            and maxT.get("replicates") == 20_000
            and maxT.get("tie_rule") == "greater_than_or_equal"
        ):
            defects.append("maxT_global_contract_invalid")
            maxT = {}
        max_rows = maxT.get("rows")
        if not isinstance(max_rows, list) or [
            item.get("row_id") for item in max_rows if isinstance(item, Mapping)
        ] != list(ROWS):
            defects.append("maxT_grid_invalid")
        else:
            for row_id, row in zip(ROWS, max_rows):
                p_fwer = row.get("p_FWER")
                exceedance = row.get("exceedance_count")
                values_valid = bool(
                    isinstance(p_fwer, (int, float))
                    and not isinstance(p_fwer, bool)
                    and math.isfinite(float(p_fwer))
                    and 0.0 <= float(p_fwer) <= 1.0
                    and isinstance(exceedance, int)
                    and not isinstance(exceedance, bool)
                    and 0 <= exceedance <= 20_000
                )
                derived = bool(
                    values_valid
                    and float(p_fwer) <= 0.05
                    and exceedance <= 999
                )
                if not values_valid or row.get("hard_pass") is not derived:
                    defects.append(f"maxT_row_law_invalid:{row_id}")
    try:
        expected_packet_hash = stable_hash(
            _without(packet, "packet_sha256")
        )
    except (TypeError, ValueError):
        expected_packet_hash = None
    if packet.get("packet_sha256") != expected_packet_hash:
        defects.append("packet_hash_mismatch")
    if not _finite_tree(packet):
        defects.append("packet_nonfinite")
    defects.extend(
        _validate_authorities(
            packet,
            execution_authority=execution_authority,
            execution_authority_sha256=execution_authority_sha256,
            control_authority=control_authority,
            control_authority_sha256=control_authority_sha256,
        )
    )
    return sorted(set(defects))


def _candidate(payload: Mapping[str, Any]) -> SerialCandidateV5:
    return SerialCandidateV5(
        **{
            key: payload[key]
            for key in SerialCandidateV5.__dataclass_fields__
            if key in payload
        }
    )


def _fee_candidate(item: SerialCandidateV5, fee: float) -> SerialCandidateV5:
    expected = (
        (item.label_executable_exit_bid - item.entry_ask) * 100.0 - fee
    )
    return replace(item, raw_label_pnl_after_campaign_fee=float(expected))


def _independent_replay(
    candidates: Sequence[SerialCandidateV5],
    fee: float = PRIMARY_FEE,
) -> dict[str, Any]:
    adjusted = tuple(_fee_candidate(item, fee) for item in candidates)
    trades, state = simulate_serial_candidates_v5(
        adjusted,
        config=SerialSimulatorV5Config(
            campaign_round_trip_fee_dollars=fee,
            affordability_reserve_per_trade=fee,
        ),
    )
    events = list(next(iter(state.equity_events_by_account.values()), ()))
    equities = [10_000.0] + [float(item["equity"]) for item in events]
    peak = equities[0]
    drawdown = 0.0
    for value in equities:
        peak = value if value > peak else peak
        drawdown = max(drawdown, peak - value)
    return {
        "fee_adjusted_net_pnl": float(
            sum(item.raw_label_pnl_after_campaign_fee for item in trades)
        ),
        "max_drawdown": float(drawdown),
        "minimum_equity": float(min(equities)),
        "trade_count": len(trades),
        "trade_identity_hash": state.trade_identity_hash,
        "candidate_stream_hash": state.candidate_stream_hash,
        "candidate_payload_hash": state.candidate_payload_hash,
        "simulator_version": state.semantics["simulator_version"],
        "skipped": dict(state.skipped),
        "skipped_events": [
            {
                "candidate_identity": list(event.candidate_identity),
                "reason_code": event.reason_code,
                "decision_time_ns": event.decision_time_ns,
                "pending_contract_id_or_null": (
                    event.pending_contract_id_or_null
                ),
                "pending_source_quote_time_ns_or_null": (
                    event.pending_source_quote_time_ns_or_null
                ),
                "pending_realized_exit_time_ns_or_null": (
                    event.pending_realized_exit_time_ns_or_null
                ),
            }
            for event in state.skipped_events
        ],
        "trades": [
            {
                "session": item.session,
                "decision_time_ns": item.decision_time_ns,
                "contract_id": item.contract_id,
                "label_source_exit_quote_time_ns": item.label_source_exit_quote_time_ns,
                "label_realized_exit_time_ns": item.label_realized_exit_time_ns,
                "raw_label_pnl_after_campaign_fee": item.raw_label_pnl_after_campaign_fee,
            }
            for item in trades
        ],
    }


def _ece(observations: Sequence[Mapping[str, Any]]) -> float:
    total = len(observations)
    value = 0.0
    for bin_index in range(10):
        low = bin_index / 10.0
        high = (bin_index + 1) / 10.0
        members = [
            item
            for item in observations
            if low <= float(item["confidence"]) < high
            or (bin_index == 9 and float(item["confidence"]) == 1.0)
        ]
        if members:
            confidence = statistics.fmean(float(item["confidence"]) for item in members)
            won = statistics.fmean(float(item["won"]) for item in members)
            value += len(members) / total * abs(confidence - won)
    return float(value)


def _controls(packet: Mapping[str, Any]) -> dict[str, bool]:
    values = packet["controls"]
    d1 = values["D1"]
    d1_pass = bool(
        d1["independently_accepted"]
        and d1["passes"]
        and d1["joint_G1_G2_pass_count"] <= 1
        and d1["median_pnl"] <= 0
        and d1["median_z"] < 1
    )
    d5 = values["D5"]
    d5_pass = bool(
        d5["independently_accepted"]
        and d5["identity_complete"]
        and d5["acceptance_route"] == REFERENCE_ROUTE
        and _is_sha256(d5["receipt_sha256"])
    )
    d6 = values["D6"]
    d6_pass = bool(
        d6["independently_accepted"]
        and d6["route"] == D6_ROUTE
        and d6["acceptance_route"] == REFERENCE_ROUTE
        and _is_sha256(d6["receipt_sha256"])
    )
    maxT = values["maxT"]
    row_laws = all(
        row["hard_pass"]
        is (
            float(row["p_FWER"]) <= 0.05
            and int(row["exceedance_count"]) <= 999
        )
        for row in maxT["rows"]
    )
    maxt_valid = bool(
        maxT["valid"]
        and maxT["family_size"] == 28
        and maxT["replicates"] == 20_000
        and maxT["tie_rule"] == "greater_than_or_equal"
        and row_laws
    )
    return {
        "D1": d1_pass,
        "D5": d5_pass,
        "D6": d6_pass,
        "maxT_valid": maxt_valid,
        "all_global_controls_pass": (
            d1_pass and d5_pass and d6_pass and maxt_valid
        ),
    }


def _seed(
    units: Sequence[Mapping[str, Any]],
    *,
    hypothesis: str,
    policy: str,
    seed: int,
    matched_null_z: float,
) -> tuple[dict[str, Any], list[str]]:
    folds: list[dict[str, Any]] = []
    candidates: list[SerialCandidateV5] = []
    observations: list[Mapping[str, Any]] = []
    defects: list[str] = []
    fees = {"2.60": 0.0, "3.00": 0.0, "4.00": 0.0}
    for unit in units:
        fold_candidates = [_candidate(value) for value in unit["candidates"]]
        replay = _independent_replay(fold_candidates)
        primary_values = (
            unit["diagnostics"]["fee_sensitivity"]["3.00"],
            unit["diagnostics"]["fill_edge_band"]["pessimistic_executable"],
            unit["diagnostics"]["noise_diagnostics"]["1.0x"],
        )
        if any(
            not math.isclose(
                float(value),
                replay["fee_adjusted_net_pnl"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for value in primary_values
        ):
            defects.append(f"primary_diagnostic_mismatch:{unit['unit_id']}")
        candidates.extend(fold_candidates)
        observations.extend(unit["calibration_observations"])
        session_count = len(unit["session_ids"])
        folds.append(
            {
                "fold": int(unit["fold"]),
                "era": str(unit["era"]),
                "fee_adjusted_net_pnl": replay["fee_adjusted_net_pnl"],
                "max_drawdown": replay["max_drawdown"],
                "minimum_equity": replay["minimum_equity"],
                "trade_count": replay["trade_count"],
                "session_count": session_count,
                "trades_per_session": replay["trade_count"] / session_count,
                "three_per_day_rail_report_only": (
                    replay["trade_count"] / session_count <= 3.0
                ),
                "trade_identity_hash": replay["trade_identity_hash"],
                "candidate_stream_hash": replay["candidate_stream_hash"],
                "candidate_payload_hash": replay["candidate_payload_hash"],
                "skipped": replay["skipped"],
                "skipped_events": replay["skipped_events"],
                "source_and_realized_clocks": [
                    {
                        "trade_id": item["trade_id"],
                        "source_exit_quote_time_ns": item[
                            "label_source_exit_quote_time_ns"
                        ],
                        "realized_exit_time_ns": item[
                            "label_realized_exit_time_ns"
                        ],
                    }
                    for item in unit["candidates"]
                ],
                "diagnostics": unit["diagnostics"],
            }
        )
        for fee in (2.6, 3.0, 4.0):
            fees[f"{fee:.2f}"] += _independent_replay(
                fold_candidates, fee
            )["fee_adjusted_net_pnl"]
    return (
        {
            "seed": seed,
            "hypothesis": hypothesis,
            "policy": policy,
            "fold_results": folds,
            "continuous_oof_replay": _independent_replay(candidates),
            "matched_null_z": float(matched_null_z),
            "ece": _ece(observations),
            "ece_observations": len(observations),
            "calibration_observations": list(observations),
            "fee_sensitivity_pooled_net_pnl": fees,
        },
        defects,
    )


def _row_law(
    seeds: Sequence[Mapping[str, Any]],
    *,
    heuristic: float,
    maxT_pass: bool,
    controls_pass: bool,
) -> dict[str, Any]:
    pooled = [
        float(item["continuous_oof_replay"]["fee_adjusted_net_pnl"])
        for item in seeds
    ]
    z_values = [float(item["matched_null_z"]) for item in seeds]
    ece_values = [float(item["ece"]) for item in seeds]
    seed_g1: list[bool] = []
    seed_g4: list[bool] = []
    seed_g6: list[bool] = []
    seed_g7: list[bool] = []
    for item, pnl in zip(seeds, pooled):
        folds = item["fold_results"]
        seed_g1.append(
            sum(float(fold["fee_adjusted_net_pnl"]) > 0 for fold in folds) >= 4
            and pnl > 0
        )
        drawdown = float(item["continuous_oof_replay"]["max_drawdown"])
        seed_g4.append(
            pnl > 0
            and (
                math.isclose(
                    drawdown,
                    0.0,
                    rel_tol=0.0,
                    abs_tol=FLOAT_ABS_TOL,
                )
                or _inclusive_lower(pnl / drawdown, 1.0)
            )
            and all(
                _inclusive_lower(float(fold["minimum_equity"]), 5_000.0)
                for fold in folds
            )
        )
        eras: dict[str, list[float]] = {}
        for fold in folds:
            eras.setdefault(str(fold["era"]), []).append(
                float(fold["fee_adjusted_net_pnl"])
            )
        seed_g6.append(
            all(
                _inclusive_lower(statistics.median(values), 0.0)
                for values in eras.values()
            )
        )
        seed_g7.append(
            all(
                _inclusive_lower(
                    float(fold["trades_per_session"]),
                    0.3,
                )
                and _inclusive_upper(
                    float(fold["trades_per_session"]),
                    6.0,
                )
                for fold in folds
            )
        )
    gates = {
        "G1": all(seed_g1),
        "G2": statistics.median(z_values) >= 3.0,
        "G3": statistics.median(pooled) > float(heuristic),
        "G4": all(seed_g4),
        "G5": all(seed_g1) and min(z_values) >= 2.0,
        "G6": all(seed_g6),
        "G7": all(seed_g7),
        "G8": all(value <= 0.10 for value in ece_values),
        "G9": False,
    }
    hard = bool(
        controls_pass
        and maxT_pass
        and all(gates[f"G{value}"] for value in range(1, 8))
    )
    signal = bool(gates["G2"] and gates["G5"] and maxT_pass)
    g6_only = bool(
        controls_pass
        and maxT_pass
        and not gates["G6"]
        and all(gates[f"G{value}"] for value in (1, 2, 3, 4, 5, 7))
    )
    return {
        "gates": gates,
        "G8_role": "report_only",
        "G8_benchmark": 0.10,
        "hard_gate_eligible": hard,
        "multiplicity_adjusted_real_entry_signal": signal,
        "G6_only_blocked": g6_only,
        "median_seed_fee_adjusted_continuous_strict_serial_net_pnl": float(
            statistics.median(pooled)
        ),
        "median_seed_matched_null_z": float(statistics.median(z_values)),
        "worst_seed_matched_null_z": float(min(z_values)),
        "median_seed_ece": float(statistics.median(ece_values)),
    }


def reconstruct_independently(
    packet: Mapping[str, Any],
    *,
    execution_authority_sha256: str | None = None,
    control_authority_sha256: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    controls = _controls(packet)
    unit_map = {
        (
            item["hypothesis"],
            item["policy"],
            int(item["seed"]),
            int(item["fold"]),
        ): item
        for item in packet["units"]
    }
    references = {
        item["row_id"]: item for item in packet["references"]["rows"]
    }
    maxT = {
        item["row_id"]: item for item in packet["controls"]["maxT"]["rows"]
    }
    rows: list[dict[str, Any]] = []
    defects: list[str] = []
    for hypothesis in HYPOTHESES:
        for policy in POLICIES:
            row_id = f"{hypothesis}/{policy}"
            seed_rows: list[dict[str, Any]] = []
            for seed in SEEDS:
                value, seed_defects = _seed(
                    [
                        unit_map[(hypothesis, policy, seed, fold)]
                        for fold in FOLDS
                    ],
                    hypothesis=hypothesis,
                    policy=policy,
                    seed=seed,
                    matched_null_z=references[row_id][
                        "matched_null_z_by_seed"
                    ][str(seed)],
                )
                seed_rows.append(value)
                defects.extend(seed_defects)
            law = _row_law(
                seed_rows,
                heuristic=references[row_id]["heuristic_pooled_pnl"],
                maxT_pass=bool(maxT[row_id]["hard_pass"]),
                controls_pass=controls["all_global_controls_pass"],
            )
            rows.append(
                {
                    "row_id": row_id,
                    "hypothesis": hypothesis,
                    "policy": policy,
                    "policy_index": int(policy[1:]),
                    "seed_results": seed_rows,
                    "heuristic_pooled_pnl": references[row_id][
                        "heuristic_pooled_pnl"
                    ],
                    "maxT": dict(maxT[row_id]),
                    **law,
                }
            )
    result = {
        "schema_version": RESULT_SCHEMA,
        "valid": bool(controls["all_global_controls_pass"]),
        "campaign_packet_sha256": packet["packet_sha256"],
        "execution_provenance_authority_sha256": (
            execution_authority_sha256
        ),
        "control_authority_sha256": control_authority_sha256,
        "blockers": (
            []
            if controls["all_global_controls_pass"]
            else [
                name
                for name in ("D1", "D5", "D6", "maxT_valid")
                if not controls[name]
            ]
        ),
        "global_controls": controls,
        "rows": rows,
        "row_count": len(rows),
        "unit_count": len(packet["units"]),
        "G8_role": "report_only",
        "G9_executed": False,
        "real_campaign_economics_executed": False,
        "aggregation_sha256": None,
    }
    result["aggregation_sha256"] = stable_hash(
        {**result, "aggregation_sha256": None}
    )
    return result, sorted(set(defects))


def audit_campaign(
    packet: Mapping[str, Any],
    producer_result: Mapping[str, Any],
    *,
    execution_authority: Mapping[str, Any] | None = None,
    execution_authority_sha256: str | None = None,
    control_authority: Mapping[str, Any] | None = None,
    control_authority_sha256: str | None = None,
) -> dict[str, Any]:
    defects = validate_packet_independently(
        packet,
        execution_authority=execution_authority,
        execution_authority_sha256=execution_authority_sha256,
        control_authority=control_authority,
        control_authority_sha256=control_authority_sha256,
    )
    independent: dict[str, Any] | None = None
    comparisons: list[str] = []
    if not defects:
        try:
            independent, replay_defects = reconstruct_independently(
                packet,
                execution_authority_sha256=execution_authority_sha256,
                control_authority_sha256=control_authority_sha256,
            )
            defects.extend(replay_defects)
        except Exception as exc:
            defects.append(
                f"independent_reconstruction_failed:{type(exc).__name__}:{exc}"
            )
    if independent is not None and not defects:
        comparisons = payload_differences(independent, producer_result)
        if comparisons:
            defects.append("producer_independent_result_disagreement")
    agreement = independent is not None and not comparisons and not defects
    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1]
        / "model"
        / "protocol101_serial_simulator_v5.py",
    )
    source_hashes = {
        str(path.relative_to(Path(__file__).resolve().parents[2])): (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in source_paths
    }
    freeze_payload = {
        "schema_version": FREEZE_SCHEMA,
        "audit_route": "fresh_28_row_independent_audit_accepted",
        "campaign_packet_sha256": packet.get("packet_sha256"),
        "execution_provenance_authority_sha256": (
            execution_authority_sha256
        ),
        "control_authority_sha256": control_authority_sha256,
        "producer_aggregation_sha256": producer_result.get(
            "aggregation_sha256"
        ),
        "independent_aggregation_sha256": (
            independent.get("aggregation_sha256")
            if independent is not None
            else None
        ),
        "row_count": (
            len(independent["rows"]) if independent is not None else 0
        ),
        "row_result_sha256": (
            stable_hash(independent["rows"])
            if independent is not None
            else None
        ),
        "independent_result_sha256": (
            stable_hash(independent) if independent is not None else None
        ),
        "source_hashes": source_hashes,
        "agreement": agreement,
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "accepted": agreement,
        "campaign_valid": bool(
            agreement and independent is not None and independent["valid"]
        ),
        "defects": sorted(set(defects)),
        "comparisons": comparisons,
        "producer_result": dict(producer_result),
        "independent_result": independent,
        "published_row_count": (
            len(independent["rows"]) if independent is not None else 0
        ),
        "all_28_rows_published": bool(
            independent is not None and len(independent["rows"]) == 28
        ),
        "freeze": freeze_payload if agreement else None,
        "freeze_sha256": stable_hash(freeze_payload) if agreement else None,
        "execution_provenance_authority_sha256": (
            execution_authority_sha256
        ),
        "control_authority_sha256": control_authority_sha256,
        "imports_producer_aggregator": False,
        "imports_producer_result_helpers": False,
        "G8_role": "report_only",
        "G9_executed": False,
    }


def independent_audit(
    *,
    campaign_packet: Mapping[str, Any],
    producer_result: Mapping[str, Any],
    execution_authority: Mapping[str, Any] | None = None,
    execution_authority_sha256: str | None = None,
    control_authority: Mapping[str, Any] | None = None,
    control_authority_sha256: str | None = None,
) -> dict[str, Any]:
    return audit_campaign(
        campaign_packet,
        producer_result,
        execution_authority=execution_authority,
        execution_authority_sha256=execution_authority_sha256,
        control_authority=control_authority,
        control_authority_sha256=control_authority_sha256,
    )


def main() -> int:
    args = parse_args()
    result = audit_campaign(
        load_json(args.campaign_packet),
        load_json(args.producer_result),
        execution_authority=load_json(
            args.execution_provenance_authority
        ),
        execution_authority_sha256=(
            args.execution_provenance_authority_sha256
        ),
        control_authority=load_json(args.control_authority),
        control_authority_sha256=args.control_authority_sha256,
    )
    write_json(args.out_dir / "independent_audit.json", result)
    write_json(
        args.out_dir / "audit_freeze.json",
        result["freeze"]
        or {
            "accepted": False,
            "defects": result["defects"],
        },
    )
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
