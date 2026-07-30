#!/usr/bin/env python3
"""Executable consistency gate for Protocol101 FT2 final repair attempt 002."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PACKET_ROOT = ROOT / "v4/audit/autoresearch"
FT208 = PACKET_ROOT / "protocol101_ft2_08_data_tensor_label_contract"
FT210 = PACKET_ROOT / "protocol101_ft2_10_entry_science_contract"
FT211 = PACKET_ROOT / "protocol101_ft2_11_evidence_statistics_contract"
FT205 = PACKET_ROOT / "protocol101_ft2_05_opportunity_census"
HERE = Path(__file__).resolve().parent

AUTHORITY_HASH = (
    "2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a"
)
INTENT_LAW_HASH = (
    "5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0"
)
GENERATOR_HASH = (
    "8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754"
)
SIMULATOR_HASH = (
    "7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548"
)
CENSUS_RECEIPT_HASH = (
    "a252feb2007b2aece77f377461bc6fa5a882e929bd2f7d3f23a47c07401cfdce"
)
PARENT_RECEIPTS = {
    "FT2-08": "584612af00f8e04b9b903f1fb4b56396a3dc950bd21f18718fa72fd4ff8dfb9c",
    "FT2-10": "5108b909740fad187c4c0349196243ed47aa1cd2785a46e9f5c7c6f2f336bda8",
    "FT2-11": "61746ee2c71f279c4ae5c864e8bd8ed242d0281cb4084abeb7e0fd0a274d7871",
}
PACKETS = {
    "FT2-08": FT208,
    "FT2-10": FT210,
    "FT2-11": FT211,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"expected object: {path}")
    return payload


def resolve_json_pointer(payload: Any, pointer: str) -> Any:
    value = payload
    for raw in pointer.strip("/").split("/"):
        if not raw:
            continue
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(key)]
        else:
            value = value[key]
    return value


def delete_block_values(series: list[float], block_length: int) -> tuple[list[float], float, float]:
    n = len(series)
    k = n - block_length + 1
    if n <= block_length or k < 2:
        raise ValueError("uncomputable delete-block jackknife")
    means: list[float] = []
    for start in range(k):
        retained = series[:start] + series[start + block_length :]
        means.append(sum(retained) / len(retained))
    center = sum(means) / len(means)
    variance = (
        (n - block_length)
        / (block_length * k)
        * sum((value - center) ** 2 for value in means)
    )
    return means, center, variance


def terminal_resolve(example: dict[str, Any]) -> tuple[str, list[str]]:
    packet = example["packet"]
    invalid: list[str] = []
    insufficient: list[str] = []
    no_signal: list[str] = []
    if packet.get("integrity") != "valid":
        invalid.append("integrity is not valid")
    if packet.get("synthetic_AR_coverage") != "pass":
        invalid.append("synthetic AR coverage is not pass")
    controls = packet.get("D1_controls", {})
    if not controls.get("eight_random_schedules_complete_and_matched", False):
        invalid.append("canonical eight-attempt matched-random control is incomplete")
    if controls.get("shuffled_significant_positive_increment", False):
        invalid.append("strong shuffle significantly beats matched random")

    action = packet.get("action_calibration", {})
    regret = action.get("q90_normalized_regret_upper_bound")
    if regret is None or not math.isfinite(float(regret)):
        invalid.append("q90 normalized-regret upper bound is missing or nonfinite")
    elif float(regret) > 0.10:
        no_signal.append("q90 normalized-regret upper bound is above 0.10")

    quality = packet.get("quality", {})
    interval = quality.get("DeltaQ_vs_P5_adjusted_interval")
    if interval is not None:
        if float(interval[0]) <= 0.0 <= float(interval[1]) and float(
            quality.get("DeltaQ_MDE", math.inf)
        ) > 0.05:
            insufficient.append(
                "quality interval contains zero and DeltaQ MDE exceeds 0.05"
            )
    if quality.get("negative_fifth_attribution") == "ambiguous":
        insufficient.append("bounded fifth-fold attribution is ambiguous")

    if "distinct_sessions" in action and "required_minimum_sessions" in action:
        if int(action["distinct_sessions"]) < int(
            action["required_minimum_sessions"]
        ):
            insufficient.append(
                "action-calibration distinct-session minimum is unmet"
            )
    if "ECE_adjusted_interval" in action:
        lower, upper = map(float, action["ECE_adjusted_interval"])
        threshold = float(action["ECE_threshold"])
        if lower <= threshold <= upper:
            insufficient.append(
                "action-calibration ECE bound overlaps 0.10"
            )

    serial = packet.get("serial_dollar_no_harm", {})
    ambiguous_serial = False
    for fee_path in ("$3", "$4"):
        key = f"{fee_path}_adjusted_interval"
        if key not in serial:
            continue
        lower, upper = map(float, serial[key])
        if upper < 0.0:
            no_signal.append(
                f"serial-dollar no-harm adjusted upper bound is below zero on {fee_path}"
            )
        elif lower < 0.0 <= upper:
            ambiguous_serial = True
    if ambiguous_serial:
        insufficient.append(
            "serial-dollar no-harm intervals contain zero with negative lower bounds"
        )

    if invalid:
        return "invalid_evidence", invalid
    if insufficient:
        return "insufficient_evidence", insufficient
    if no_signal:
        return "no_genuine_entry_signal", no_signal
    return "entry_component_freeze_pass", []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "consistency_checker_output.json",
    )
    args = parser.parse_args()
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {"pass": bool(passed), "evidence": evidence}

    authority = ROOT / (
        "v4/docs/protocol101/training/contracts/"
        "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
    )
    simulator = ROOT / "v4/model/protocol101_serial_simulator_v5.py"
    law_path = FT208 / "intent_fill_recheck_law.json"
    generator_path = FT210 / "matched_random_generator_spec.json"
    check("authority_hash", sha256(authority) == AUTHORITY_HASH, sha256(authority))
    check("simulator_v5_unchanged", sha256(simulator) == SIMULATOR_HASH, sha256(simulator))
    check("intent_law_hash", sha256(law_path) == INTENT_LAW_HASH, sha256(law_path))
    check("canonical_generator_hash", sha256(generator_path) == GENERATOR_HASH, sha256(generator_path))
    check("census_v3_receipt_hash", sha256(FT205 / "receipt.json") == CENSUS_RECEIPT_HASH, sha256(FT205 / "receipt.json"))
    census_receipt = load(FT205 / "receipt.json")
    census_deliverable_results = {
        relative: {
            "exists": (FT205 / relative).is_file(),
            "hash_matches": (
                (FT205 / relative).is_file()
                and sha256(FT205 / relative) == expected
            ),
        }
        for relative, expected in census_receipt["deliverable_hashes"].items()
    }
    check(
        "census_v3_receipt_deliverables",
        all(
            value["exists"] and value["hash_matches"]
            for value in census_deliverable_results.values()
        ),
        {
            "deliverable_count": len(census_deliverable_results),
            "failures": {
                relative: value
                for relative, value in census_deliverable_results.items()
                if not (value["exists"] and value["hash_matches"])
            },
        },
    )

    law = load(law_path)
    objective = load(FT210 / "objective_spec.json")
    evidence = load(FT211 / "evidence_standard.json")
    primitives = law["numeric_primitives"]
    fees = law["fee_paths"]
    expected_constants = {
        "numeric_unit": primitives["money_unit"],
        "contract_multiplier": primitives["contract_multiplier"],
        "premium_floor_quote_cents": primitives["premium_floor_quote_cents"],
        "fee_3_round_trip_cents": fees["primary"]["round_trip_fee_cents"],
        "fee_4_round_trip_cents": fees["stress"]["round_trip_fee_cents"],
        "fee_3_floor_cost_cents": fees["primary"]["premium_floor_plus_fee_cents"],
        "fee_4_floor_cost_cents": fees["stress"]["premium_floor_plus_fee_cents"],
        "daily_budget": primitives["daily_budget_cents"],
    }
    constants_equal = (
        objective["cross_contract_constants"]
        == evidence["cross_contract_constants"]
        == expected_constants
    )
    check(
        "cross_contract_constants_identical",
        constants_equal,
        {
            "FT2-08": expected_constants,
            "FT2-10": objective["cross_contract_constants"],
            "FT2-11": evidence["cross_contract_constants"],
        },
    )
    check(
        "fee_path_threshold_formula",
        (
            100 * 100 + 300 == 10300
            and 100 * 100 + 400 == 10400
            and fees["threshold_formula"]
            == "premium_floor_plus_fee_cents=premium_floor_quote_cents*contract_multiplier+active_round_trip_fee_cents"
            and fees["hard_coded_shared_threshold_forbidden"] is True
        ),
        {"fee_3": 10300, "fee_4": 10400},
    )

    intent_consumers = [
        FT210 / "objective_spec.json",
        FT210 / "composer_spec.json",
        FT210 / "calibration_spec.json",
        FT210 / "forecast_heads.json",
        FT210 / "controls_spec.json",
        FT210 / "matched_random_generator_spec.json",
        FT211 / "bootstrap_spec.json",
        FT211 / "evidence_standard.json",
        FT211 / "mde_spec.json",
        FT211 / "multiplicity_spec.json",
        FT211 / "shadow_sufficiency_spec.json",
        FT211 / "terminal_decision_spec.json",
        FT211 / "tripwire_spec.json",
    ]
    intent_refs: dict[str, str] = {}
    intent_ref_ok = True
    for path in intent_consumers:
        ref = load(path).get("intent_fill_recheck_law", {})
        value = str(ref.get("sha256"))
        intent_refs[str(path.relative_to(ROOT))] = value
        intent_ref_ok &= value == INTENT_LAW_HASH
    check("all_FT2_10_FT2_11_intent_refs_exact", intent_ref_ok, intent_refs)

    controls = load(FT210 / "controls_spec.json")
    multiplicity = load(FT211 / "multiplicity_spec.json")
    control_ref = controls["frozen_nonselectable_random_controls"]
    multiplicity_ref = multiplicity["frozen_nonselectable_matched_random_algorithm"]
    comparator_ok = (
        control_ref["authority_sha256"]
        == multiplicity_ref["source_sha256"]
        == GENERATOR_HASH
        and control_ref["authority_path"] == multiplicity_ref["source_contract"]
        and multiplicity_ref["candidate_specific_seed_namespace"] is False
        and multiplicity_ref["consumer_local_override"] is False
    )
    check(
        "FT2_10_FT2_11_comparator_identity",
        comparator_ok,
        {"FT2-10": control_ref, "FT2-11": multiplicity_ref},
    )

    stale_pattern = re.compile(r"census(?:[ _-]*)v[12]\b", re.IGNORECASE)
    stale_hits: list[str] = []
    for packet in PACKETS.values():
        for path in sorted(packet.iterdir()):
            if (
                not path.is_file()
                or path.name in {"receipt.json", "findings_crosswalk.json"}
                or path.suffix.lower() not in {".json", ".md", ".py"}
            ):
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if stale_pattern.search(line):
                    stale_hits.append(
                        f"{path.relative_to(ROOT)}:{line_no}:{line.strip()}"
                    )
    check("no_current_v1_v2_census_spec_reference", not stale_hits, stale_hits)

    census_receipt = load(FT205 / "receipt.json")
    census_hashes_ok = (
        census_receipt["schema_version"] == "Protocol101FT205NodeReceiptV3"
        and census_receipt["repair_of"]["receipt_sha256"]
        == "0cec80e1ce029bcbec200c991f9183a9b05f1e21b28016fe944fee0ae029b33a"
        and census_receipt["input_hashes"]["intent_fill_recheck_law.json"]
        == INTENT_LAW_HASH
        and census_receipt["session_count"] == 45
        and census_receipt["outcome"] == "feasible"
    )
    check("census_v3_chain_and_scope", census_hashes_ok, census_receipt)

    account = load(FT208 / "account_state_ledger_spec.json")
    tensor = load(FT208 / "tensor_schema.json")
    session_start = account["session_start"]
    equity_ok = (
        session_start["canonical_frozen_timestamp"]
        == "session date 09:30:00 America/New_York"
        and session_start["live_paper_value"][
            "maximum_source_age_at_freeze_seconds"
        ]
        == 60
        and "dedicated" in session_start["account_scope"]["dedication_requirement"]
        and "block all new entries"
        in session_start["live_paper_value"][
            "missing_stale_ambiguous_or_late"
        ]
    )
    check("deterministic_session_start_equity", equity_ok, session_start)
    terminal_boundary_ok = (
        tensor["open_state"]["last_learned_exit_decision_et"] == "15:54"
        and tensor["open_state"]["last_learned_exit_fill_et"] == "15:55"
        and tensor["open_state"]["actions_at_1555"] == []
        and tensor["open_state"]["terminal_minute_1555"]["model_invoked"] is False
    )
    check("terminal_boundary_no_learned_exit_at_1555", terminal_boundary_ok, tensor["open_state"])

    composer = load(FT210 / "composer_spec.json")
    calibration = load(FT210 / "calibration_spec.json")
    cluster = composer["uncertainty_wait"]["directional_substitute_cluster"]
    regret = composer["mandatory_action_conditioned_gate"][
        "selected_contract_regret_action_constraint"
    ]
    check(
        "directional_substitute_cluster_widened",
        (
            cluster["breadth_points_each_side"] == 10
            and cluster["breadth_strike_milli_points_each_side"] == 10000
        ),
        cluster,
    )
    check(
        "regret_bound_constrains_action",
        (
            regret["maximum"] == 0.1
            and regret["missing_nonfinite_or_above"] == "WAIT"
            and regret["not_report_only"] is True
            and calibration["action_conditioned_gate"]["selected_contract_regret"][
                "report_only"
            ]
            is False
        ),
        regret,
    )
    selection_order = calibration["joint_conservatism"]["selection_order"]
    census_role = calibration["joint_conservatism"][
        "census_v3_trade_rate_provenance"
    ]["role"]
    check(
        "census_activity_report_only",
        (
            all("trade-rate band" not in value for value in selection_order)
            and "cannot discard" in census_role
            and calibration["joint_conservatism"][
                "design_trade_rate_band_mean_trades_per_session"
            ]["selection_gate"]
            is False
        ),
        {"selection_order": selection_order, "role": census_role},
    )
    full_window = objective["nested_empirical_cdf"]["full_window_filter"]
    check(
        "finite_censored_CDF_rows_excluded",
        "excluded" in full_window["finite_but_censored_near_close"],
        full_window,
    )

    bootstrap = load(FT211 / "bootstrap_spec.json")
    example = bootstrap["worked_studentization_example"]
    observed_series = [float(value) for value in example["observed_series"]]
    means, center, variance = delete_block_values(
        observed_series, int(example["block_length_L"])
    )
    observed_se = math.sqrt(variance)
    observed_t = sum(observed_series) / len(observed_series) / observed_se
    centered = [float(value) for value in example["centered_outer_resample"]]
    star_means, star_center, star_variance = delete_block_values(
        centered, int(example["block_length_L"])
    )
    star_se = math.sqrt(star_variance)
    star_t = sum(centered) / len(centered) / star_se
    tol = 1e-12
    se_ok = (
        all(
            math.isclose(a, b, rel_tol=1e-10, abs_tol=tol)
            for a, b in zip(
                means,
                map(float, example["observed_delete_block_means_in_order"]),
                strict=True,
            )
        )
        and math.isclose(center, float(example["observed_delete_mean_average"]), rel_tol=1e-10, abs_tol=tol)
        and math.isclose(variance, float(example["observed_variance"]), rel_tol=1e-10, abs_tol=tol)
        and math.isclose(observed_se, float(example["SE_h"]), rel_tol=1e-10, abs_tol=tol)
        and math.isclose(observed_t, float(example["observed_t_h"]), rel_tol=1e-10, abs_tol=tol)
        and all(
            math.isclose(a, b, rel_tol=1e-10, abs_tol=tol)
            for a, b in zip(
                star_means,
                map(float, example["replicate_delete_block_means_in_order"]),
                strict=True,
            )
        )
        and math.isclose(star_center, float(example["replicate_delete_mean_average"]), rel_tol=1e-10, abs_tol=tol)
        and math.isclose(star_se, float(example["SE_h_star_b"]), rel_tol=1e-10, abs_tol=tol)
        and math.isclose(star_t, float(example["t_h_star_b"]), rel_tol=1e-10, abs_tol=tol)
    )
    check(
        "SE_h_and_SE_h_star_numeric_fixture",
        se_ok,
        {
            "observed_delete_means": means,
            "observed_variance": variance,
            "SE_h": observed_se,
            "t_h": observed_t,
            "replicate_delete_means": star_means,
            "replicate_variance": star_variance,
            "SE_h_star_b": star_se,
            "t_h_star_b": star_t,
        },
    )
    shadow = load(FT211 / "shadow_sufficiency_spec.json")
    phase_ok = (
        bootstrap["phase_F_shadow_intervals"][
            "minimum_complete_sessions_for_hard_bounds"
        ]
        == 45
        and shadow["minimum_no_order_live_shadow_evidence"][
            "complete_sessions_for_hard_gate"
        ]
        == 45
        and "report-only"
        in shadow["minimum_no_order_live_shadow_evidence"][
            "sessions_1_through_44"
        ]
    )
    check("phase_F_hard_floor_matches_AR_coverage", phase_ok, {"bootstrap": bootstrap["phase_F_shadow_intervals"], "shadow": shadow["minimum_no_order_live_shadow_evidence"]})

    topology = evidence["source_transfer_graph_topology"]
    topology_ok = (
        topology["entry_component_acceptance_transfer_state"]
        == "transfer_not_yet_run is allowed and is not a component-acceptance failure"
        and topology["FT2_93_prerequisite"] == "FT2-92 transfer pass"
        and shadow["graph_topology"]["FT2_92_produces_transfer_outcome"] is True
        and shadow["graph_topology"]["circular_precondition_forbidden"] is True
    )
    check("source_transfer_graph_topology", topology_ok, {"evidence": topology, "shadow": shadow["graph_topology"]})
    mnar_refit = evidence["mnar_no_bid_sensitivity"]["alternate_refit_protocol"]
    check(
        "MNAR_alternate_full_refit",
        (
            mnar_refit["refit_required"] is True
            and len(mnar_refit["refit_all_label_dependent_machinery"]) >= 6
            and mnar_refit["primary_outcome_tuning"] is False
            and "forbidden" in mnar_refit["cross_view_artifact_reuse"]
        ),
        mnar_refit,
    )

    terminal_examples = load(FT211 / "worked_terminal_examples.json")
    terminal_results: list[dict[str, Any]] = []
    terminals_ok = True
    for item in terminal_examples["examples"]:
        terminal, triggered = terminal_resolve(item)
        passed = (
            terminal == item["expected_terminal"]
            and triggered == item.get("expected_triggered_predicates", [])
        )
        terminals_ok &= passed
        terminal_results.append(
            {
                "id": item["id"],
                "terminal": terminal,
                "triggered_predicates": triggered,
                "expected_terminal": item["expected_terminal"],
                "expected_triggered_predicates": item.get(
                    "expected_triggered_predicates", []
                ),
                "pass": passed,
            }
        )
    check("three_worked_terminal_examples_executed", terminals_ok and len(terminal_results) == 3, terminal_results)

    crosswalk = load(HERE / "findings_crosswalk.json")
    findings = crosswalk["findings"]
    original_expected = {
        *(f"S1-{index:02d}" for index in range(1, 8)),
        *(f"S2-{index:02d}" for index in range(1, 12)),
        *(f"S3-{index:02d}" for index in range(1, 11)),
    }
    fresh_expected = {
        "FRESH-S1-B1",
        "FRESH-S1-B2",
        "FRESH-S1-B3",
        "FRESH-S2-B1",
        "FRESH-S2-B2",
        "FRESH-S2-B3",
        "FRESH-S2-B4",
        "FRESH-S2-B5",
        "FRESH-S2-B6",
        "FRESH-S2-MINOR-RECEIPT",
        "FRESH-S3-B1",
        "FRESH-S3-B2",
        "FRESH-S3-B3",
        "FRESH-S3-B4",
        "FRESH-S3-B5",
        "FRESH-S3-B6",
        "FRESH-S3-B7",
        "FRESH-S3-B8",
        "FRESH-S3-MINOR-RECEIPT",
    }
    original_actual = {
        item["finding_id"]
        for item in findings
        if item["finding_class"] == "original"
    }
    fresh_actual = {
        item["finding_id"]
        for item in findings
        if item["finding_class"] == "fresh"
    }
    crosswalk_ok = (
        len(findings) == 47
        and len({item["finding_id"] for item in findings}) == 47
        and original_actual == original_expected
        and fresh_actual == fresh_expected
        and all(item["verdict"] == "ANSWERED" for item in findings)
        and all(item["repair_citations"] for item in findings)
        and sum(item["severity"] == "BLOCKING" for item in findings) == 26
        and sum(item["severity"] == "MATERIAL" for item in findings) == 19
        and sum(item["severity"] == "MINOR" for item in findings) == 2
    )
    check(
        "all_47_rerun_findings_answered",
        crosswalk_ok,
        {
            "total": len(findings),
            "original": len(original_actual),
            "fresh": len(fresh_actual),
            "verdicts": sorted({item["verdict"] for item in findings}),
        },
    )
    citation_failures: list[str] = []
    for item in findings:
        for citation in item["repair_citations"]:
            if ":/" in citation:
                relative, pointer = citation.split(":", 1)
            else:
                relative, pointer = citation, ""
            path = ROOT / relative
            if not path.is_file():
                citation_failures.append(
                    f"{item['finding_id']}: missing {relative}"
                )
                continue
            if pointer and path.suffix == ".json":
                try:
                    resolve_json_pointer(load(path), pointer)
                except (KeyError, IndexError, ValueError, TypeError) as exc:
                    citation_failures.append(
                        f"{item['finding_id']}: invalid {citation}: {exc}"
                    )
    check("crosswalk_citations_resolve", not citation_failures, citation_failures)

    receipt_results: dict[str, Any] = {}
    receipts_ok = True
    for name, packet in PACKETS.items():
        receipt = load(packet / "receipt.json")
        result = {
            "schema_version": receipt.get("schema_version"),
            "outcome": receipt.get("outcome"),
            "product_contract_hash": receipt.get("product_contract_hash"),
            "repair_of": receipt.get("repair_of"),
            "has_authority_sha256_field": "authority_sha256" in receipt,
            "deliverable_hashes_exact": all(
                (packet / relative).is_file()
                and sha256(packet / relative) == expected
                for relative, expected in receipt.get(
                    "deliverable_hashes", {}
                ).items()
            ),
        }
        result["pass"] = (
            str(receipt.get("schema_version", "")).endswith("V3")
            and receipt.get("outcome") == "producer_repaired"
            and receipt.get("product_contract_hash") == AUTHORITY_HASH
            and "authority_sha256" not in receipt
            and receipt["repair_of"]["receipt_sha256"]
            == PARENT_RECEIPTS[name]
            and sha256(
                packet / receipt["repair_of"]["preserved_path"]
            )
            == PARENT_RECEIPTS[name]
            and result["deliverable_hashes_exact"]
        )
        receipts_ok &= result["pass"]
        receipt_results[name] = result
    check("standardized_chained_packet_receipts", receipts_ok, receipt_results)

    failed = [name for name, value in checks.items() if not value["pass"]]
    output = {
        "schema_version": "Protocol101FT2FinalRepairConsistencyOutputV1",
        "goal": "FT2-FINAL-REPAIR",
        "attempt": 2,
        "product_contract_hash": AUTHORITY_HASH,
        "outcome": "pass" if not failed else "fail",
        "check_count": len(checks),
        "failed_checks": failed,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outcome": output["outcome"], "failed_checks": failed}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
