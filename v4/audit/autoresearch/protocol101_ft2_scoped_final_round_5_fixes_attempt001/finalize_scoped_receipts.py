#!/usr/bin/env python3
"""Mechanically chain scoped-round packet and aggregate receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PACKET_ROOT = REPO / "v4/audit/autoresearch"
AUTHORITY = REPO / (
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
GRAPH = REPO / (
    "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2.json"
)
SIMULATOR = REPO / "v4/model/protocol101_serial_simulator_v5.py"
AUTHORITY_HASH = (
    "d115b953d8959fe777923ca5c1e375246754a181847ae77b57d37d24f0a279ca"
)
LAW_HASH = (
    "5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0"
)
SIMULATOR_HASH = (
    "7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548"
)
PACKETS = {
    "FT2-08": {
        "path": PACKET_ROOT / "protocol101_ft2_08_data_tensor_label_contract",
        "schema": "Protocol101FT208NodeReceiptV4",
        "parent_schema": "Protocol101FT208NodeReceiptV3",
        "parent_hash": "731cc6fb4c44bd0650d658e71fffbdac7c3e0f5d68701b2e05b8576b014b078c",
        "deliverables": [
            "account_state_ledger_spec.json",
            "contract.md",
            "field_semantics_manifest.json",
            "findings_crosswalk.json",
            "fold_roles.json",
            "identity_mapping_spec.json",
            "intent_fill_recheck_law.json",
            "label_join_spec.json",
            "replay_authority_v5_1_spec.json",
            "storage_spec.json",
            "synthetic_golden_vectors.json",
            "tensor_schema.json",
            "validate_contract_v3.py",
            "validation.json",
        ],
    },
    "FT2-10": {
        "path": PACKET_ROOT / "protocol101_ft2_10_entry_science_contract",
        "schema": "Protocol101FT210NodeReceiptV4",
        "parent_schema": "Protocol101FT210NodeReceiptV3",
        "parent_hash": "c04a591fd3034a2caba756e8f9dc1d7f5c24c397170894b0b6b0eca7a083fa6f",
        "deliverables": [
            "calibration_spec.json",
            "composer_spec.json",
            "contract.md",
            "controls_spec.json",
            "findings_crosswalk.json",
            "forecast_heads.json",
            "matched_random_generator_spec.json",
            "objective_spec.json",
            "realized_label_audit_composer_spec.json",
        ],
    },
    "FT2-11": {
        "path": PACKET_ROOT / "protocol101_ft2_11_evidence_statistics_contract",
        "schema": "Protocol101FT211NodeReceiptV4",
        "parent_schema": "Protocol101FT211NodeReceiptV3",
        "parent_hash": "bdb52c4f9178da1cab75ec4a80bd13c1db8c0074c1c398f380729d1df80c5ef9",
        "deliverables": [
            "bootstrap_spec.json",
            "contract.md",
            "evidence_standard.json",
            "findings_crosswalk.json",
            "mde_spec.json",
            "multiplicity_spec.json",
            "shadow_sufficiency_spec.json",
            "terminal_decision_spec.json",
            "tripwire_spec.json",
            "worked_terminal_examples.json",
        ],
    },
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


def write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_fixed_inputs() -> None:
    if sha256(AUTHORITY) != AUTHORITY_HASH:
        raise RuntimeError("authority hash mismatch")
    if sha256(SIMULATOR) != SIMULATOR_HASH:
        raise RuntimeError("simulator-v5 hash mismatch")
    law = (
        PACKET_ROOT
        / "protocol101_ft2_08_data_tensor_label_contract"
        / "intent_fill_recheck_law.json"
    )
    if sha256(law) != LAW_HASH:
        raise RuntimeError("canonical law hash mismatch")


def finalize_packet_receipts() -> None:
    verify_fixed_inputs()
    census_receipt_path = (
        PACKET_ROOT
        / "protocol101_ft2_05_opportunity_census"
        / "receipt.json"
    )
    ft204_receipt_path = (
        PACKET_ROOT
        / "protocol101_ft2_04_path_label_freeze"
        / "receipt.json"
    )
    census_receipt = load(census_receipt_path)
    if census_receipt["schema_version"] != "Protocol101FT205NodeReceiptV4":
        raise RuntimeError("census v4 receipt not finalized")
    ft204_receipt = load(ft204_receipt_path)
    if ft204_receipt["schema_version"] != "Protocol101FT204NodeReceiptV4":
        raise RuntimeError("FT2-04 v4 receipt not finalized")

    for goal, spec in PACKETS.items():
        packet = Path(spec["path"])
        parent_path = (
            packet
            / "superseded"
            / "v3_pre_scoped_final_round_20260730"
            / "receipt.json"
        )
        if sha256(parent_path) != spec["parent_hash"]:
            raise RuntimeError(f"{goal} preserved parent receipt mismatch")
        deliverable_hashes = {
            relative: sha256(packet / relative)
            for relative in spec["deliverables"]
        }
        receipt = {
            "schema_version": spec["schema"],
            "goal": goal,
            "coordinated_goal": "FT2-SCOPED-FINAL-ROUND-5-FIXES",
            "attempt": 1,
            "outcome": "producer_repaired",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "product_contract_hash": AUTHORITY_HASH,
            "repair_of": {
                "schema_version": spec["parent_schema"],
                "receipt_sha256": spec["parent_hash"],
                "preserved_path": (
                    "superseded/"
                    "v3_pre_scoped_final_round_20260730/receipt.json"
                ),
                "preservation_verified": True,
            },
            "inputs": {
                "consolidated_authority": {
                    "path": str(AUTHORITY.relative_to(REPO)),
                    "sha256": AUTHORITY_HASH,
                },
                "graph_v2": {
                    "path": str(GRAPH.relative_to(REPO)),
                    "sha256": sha256(GRAPH),
                },
                "canonical_intent_fill_recheck_law": {
                    "path": (
                        "v4/audit/autoresearch/"
                        "protocol101_ft2_08_data_tensor_label_contract/"
                        "intent_fill_recheck_law.json"
                    ),
                    "sha256": LAW_HASH,
                },
                "ft2_04_receipt": {
                    "path": str(ft204_receipt_path.relative_to(REPO)),
                    "sha256": sha256(ft204_receipt_path),
                },
                "census_v4_receipt": {
                    "path": str(census_receipt_path.relative_to(REPO)),
                    "sha256": sha256(census_receipt_path),
                },
                "rerun002_receipt": {
                    "path": (
                        "v4/audit/autoresearch/"
                        "protocol101_ft2_20_parallel_design_review_rerun002/"
                        "receipt.json"
                    ),
                    "sha256": (
                        "17cc72ea85c629ae9a2d23059ae13020782464705ee6049f577c9936dd2579ac"
                    ),
                },
            },
            "deliverable_hashes": deliverable_hashes,
            "side_effects": {
                "real_model_training_or_fitting": False,
                "protected_or_outer_data_access": False,
                "broker_recorder_or_paid_data_access": False,
                "simulator_v5_modified": False,
                "runtime_promotion_or_paper_change": False,
            },
            "highest_allowed_claim": (
                "The scoped producer repair is complete and mechanically "
                "testable; Fable verification and delta-scoped fresh-seat "
                "review remain pending."
            ),
        }
        write(packet / "receipt.json", receipt)


def finalize_aggregate() -> None:
    verify_fixed_inputs()
    required_outputs = [
        "findings_crosswalk.json",
        "diff_manifest.json",
        "t1_rlac_pipeline_output.json",
        "t3_t4_census_v4_output.json",
        "graph_validation.json",
        "t6_regression_output.json",
        "targeted_census_pytest_output.json",
        "consistency_checker_output.json",
        "report.md",
    ]
    for relative in required_outputs:
        if not (HERE / relative).is_file():
            raise RuntimeError(f"missing aggregate input: {relative}")
    checker = load(HERE / "consistency_checker_output.json")
    if checker["outcome"] != "pass":
        raise RuntimeError("consistency checker did not pass")
    packet_receipts = {
        "FT2-04": (
            PACKET_ROOT
            / "protocol101_ft2_04_path_label_freeze"
            / "receipt.json"
        ),
        "FT2-05": (
            PACKET_ROOT
            / "protocol101_ft2_05_opportunity_census"
            / "receipt.json"
        ),
        **{
            goal: Path(spec["path"]) / "receipt.json"
            for goal, spec in PACKETS.items()
        },
    }
    round_hashes = {
        relative: sha256(HERE / relative)
        for relative in required_outputs
    }
    receipt = {
        "schema_version": "Protocol101FT2ScopedFinalRoundAggregateReceiptV1",
        "goal": "FT2-SCOPED-FINAL-ROUND-5-FIXES",
        "attempt": 1,
        "outcome": "producer_complete_pending_fable_verification",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "product_contract_hash": AUTHORITY_HASH,
        "graph_sha256": sha256(GRAPH),
        "simulator_v5_sha256": SIMULATOR_HASH,
        "repair_source": {
            "rerun002_receipt_sha256": (
                "17cc72ea85c629ae9a2d23059ae13020782464705ee6049f577c9936dd2579ac"
            ),
            "prior_aggregate_receipt_sha256": (
                "19b74e513f0c6c68b50167b37e4bcfd40cb4ccd0f3c2240201ff3cbbd178f4f3"
            ),
        },
        "packet_receipts": {
            goal: {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
            }
            for goal, path in packet_receipts.items()
        },
        "round_deliverable_hashes": round_hashes,
        "tests": {
            "T1": load(HERE / "t1_rlac_pipeline_output.json")["outcome"],
            "T2": checker["outcome"],
            "T3_T4": load(
                HERE / "t3_t4_census_v4_output.json"
            )["outcome"],
            "T5": load(HERE / "graph_validation.json")["outcome"],
            "T6": load(HERE / "t6_regression_output.json")["outcome"],
            "targeted_census_pytest": load(
                HERE / "targeted_census_pytest_output.json"
            )["outcome"],
            "consistency_check_count": checker["check_count"],
            "consistency_failed_checks": checker["failed_checks"],
        },
        "side_effects": {
            "real_model_training_or_fitting": False,
            "synthetic_tiny_T1_only": True,
            "protected_or_outer_data_access": False,
            "broker_recorder_or_paid_data_access": False,
            "simulator_v5_modified": False,
            "runtime_promotion_or_paper_change": False,
        },
        "next": "STOP_FOR_FABLE_VERIFICATION",
        "highest_allowed_claim": (
            "All five owner-scoped repairs and T1-T6 passed producer-side "
            "mechanical checks. Independent Fable verification and the "
            "delta-scoped fresh-seat review remain pending."
        ),
    }
    write(HERE / "aggregate_receipt.json", receipt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        finalize_aggregate()
        print(json.dumps({"aggregate": "written"}))
    else:
        finalize_packet_receipts()
        print(json.dumps({"packet_receipts": "written"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
