#!/usr/bin/env python3
"""Deterministically finalize FT2 final-repair child and aggregate receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "v4/audit/autoresearch"
HERE = Path(__file__).resolve().parent

AUTHORITY_PATH = (
    ROOT
    / "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
REVIEW = AUDIT / "protocol101_ft2_20_parallel_design_review_rerun001"
FT205 = AUDIT / "protocol101_ft2_05_opportunity_census"
FT208 = AUDIT / "protocol101_ft2_08_data_tensor_label_contract"
FT210 = AUDIT / "protocol101_ft2_10_entry_science_contract"
FT211 = AUDIT / "protocol101_ft2_11_evidence_statistics_contract"
SIMULATOR = ROOT / "v4/model/protocol101_serial_simulator_v5.py"

AUTHORITY_HASH = (
    "2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a"
)
REVIEW_HASH = (
    "068fd6e7f49362d69ffaa935abd1fc043a62251f2c1ef6922a62d9416265dc4a"
)
REVIEW_RECEIPT_HASH = (
    "8338b51ecd1ba33d33ed9fa6635456f71092745cdd62559857a9bbca280c0abc"
)
LAW_HASH = (
    "5c117d716cea3c986605faf7b58d510eedce3264a0c04f9368f6dc509dea6bd0"
)
GENERATOR_HASH = (
    "8bdbe8beb4734852526cd2981be76098f792d5c4b9fee30f89f8a2d952abf754"
)
CENSUS_RECEIPT_HASH = (
    "a252feb2007b2aece77f377461bc6fa5a882e929bd2f7d3f23a47c07401cfdce"
)
SIMULATOR_HASH = (
    "7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548"
)
CLAIM = (
    "All rerun001 findings are answered under one coordinated repair with "
    "machine-verified cross-contract consistency; the final re-review is "
    "unblocked."
)

PACKETS = {
    "FT2-08": {
        "path": FT208,
        "schema": "Protocol101FT208NodeReceiptV3",
        "parent_hash": (
            "584612af00f8e04b9b903f1fb4b56396a3dc950bd21f18718fa72fd4ff8dfb9c"
        ),
        "parent_schema": "Protocol101FT208NodeReceiptV2",
    },
    "FT2-10": {
        "path": FT210,
        "schema": "Protocol101FT210NodeReceiptV3",
        "parent_hash": (
            "5108b909740fad187c4c0349196243ed47aa1cd2785a46e9f5c7c6f2f336bda8"
        ),
        "parent_schema": "Protocol101FT210NodeReceiptV2",
    },
    "FT2-11": {
        "path": FT211,
        "schema": "Protocol101FT211NodeReceiptV3",
        "parent_hash": (
            "61746ee2c71f279c4ae5c864e8bd8ed242d0281cb4084abeb7e0fd0a274d7871"
        ),
        "parent_schema": "Protocol101FT211NodeReceiptV2",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def top_level_deliverables(packet: Path) -> dict[str, str]:
    return {
        path.name: sha256(path)
        for path in sorted(packet.iterdir())
        if path.is_file() and path.name != "receipt.json"
    }


def common_inputs() -> dict[str, dict[str, str]]:
    return {
        "consolidated_authority": {
            "path": relative(AUTHORITY_PATH),
            "sha256": sha256(AUTHORITY_PATH),
        },
        "rerun001_joined_review": {
            "path": relative(REVIEW / "joined_review.md"),
            "sha256": sha256(REVIEW / "joined_review.md"),
        },
        "rerun001_receipt": {
            "path": relative(REVIEW / "receipt.json"),
            "sha256": sha256(REVIEW / "receipt.json"),
        },
        "intent_fill_recheck_law": {
            "path": relative(FT208 / "intent_fill_recheck_law.json"),
            "sha256": sha256(FT208 / "intent_fill_recheck_law.json"),
        },
        "census_v3_receipt": {
            "path": relative(FT205 / "receipt.json"),
            "sha256": sha256(FT205 / "receipt.json"),
        },
        "simulator_v5": {
            "path": relative(SIMULATOR),
            "sha256": sha256(SIMULATOR),
        },
    }


def assert_fixed_inputs() -> None:
    observed = {
        "authority": sha256(AUTHORITY_PATH),
        "rerun001_review": sha256(REVIEW / "joined_review.md"),
        "rerun001_receipt": sha256(REVIEW / "receipt.json"),
        "intent_law": sha256(FT208 / "intent_fill_recheck_law.json"),
        "generator": sha256(FT210 / "matched_random_generator_spec.json"),
        "census_receipt": sha256(FT205 / "receipt.json"),
        "simulator": sha256(SIMULATOR),
    }
    expected = {
        "authority": AUTHORITY_HASH,
        "rerun001_review": REVIEW_HASH,
        "rerun001_receipt": REVIEW_RECEIPT_HASH,
        "intent_law": LAW_HASH,
        "generator": GENERATOR_HASH,
        "census_receipt": CENSUS_RECEIPT_HASH,
        "simulator": SIMULATOR_HASH,
    }
    if observed != expected:
        raise SystemExit(
            "fixed-input hash mismatch:\n"
            + json.dumps({"expected": expected, "observed": observed}, indent=2)
        )


def finalize_children() -> None:
    assert_fixed_inputs()
    base_inputs = common_inputs()
    upstream_receipts: dict[str, dict[str, str]] = {}
    for name in ("FT2-08", "FT2-10", "FT2-11"):
        config = PACKETS[name]
        packet = config["path"]
        parent_relative = Path("superseded/v2_pre_final_repair_20260729/receipt.json")
        parent_path = packet / parent_relative
        if sha256(parent_path) != config["parent_hash"]:
            raise SystemExit(f"{name} preserved v2 receipt hash mismatch")

        inputs = dict(base_inputs)
        if name == "FT2-10":
            inputs["upstream_FT2_08_receipt"] = upstream_receipts["FT2-08"]
            inputs["canonical_matched_random_generator"] = {
                "path": relative(FT210 / "matched_random_generator_spec.json"),
                "sha256": GENERATOR_HASH,
            }
        elif name == "FT2-11":
            inputs["upstream_FT2_08_receipt"] = upstream_receipts["FT2-08"]
            inputs["upstream_FT2_10_receipt"] = upstream_receipts["FT2-10"]
            inputs["canonical_matched_random_generator"] = {
                "path": relative(FT210 / "matched_random_generator_spec.json"),
                "sha256": GENERATOR_HASH,
            }

        payload = {
            "schema_version": config["schema"],
            "goal": name,
            "coordinated_goal": "FT2-FINAL-REPAIR",
            "attempt": 2,
            "design_repair_budget": {
                "attempt": 2,
                "maximum_attempts": 2,
                "remaining_after_completion": 0,
            },
            "outcome": "producer_repaired",
            "product_contract_hash": AUTHORITY_HASH,
            "repair_of": {
                "schema_version": config["parent_schema"],
                "receipt_sha256": config["parent_hash"],
                "preserved_path": parent_relative.as_posix(),
                "preservation_verified": True,
            },
            "inputs": inputs,
            "deliverable_hashes": top_level_deliverables(packet),
            "finding_disposition": {
                "verdict": "ANSWERED",
                "unanswered_count": 0,
                "partially_answered_count": 0,
                "rebutted_count": 0,
                "unified_crosswalk": relative(HERE / "findings_crosswalk.json"),
                "unified_crosswalk_sha256": sha256(HERE / "findings_crosswalk.json"),
            },
            "side_effects": {
                "training_or_fitting": False,
                "new_session_statistics_beyond_census_v3": False,
                "protected_data_access": False,
                "recorder_access": False,
                "broker_access": False,
                "paid_compute": False,
                "simulator_v5_modified": False,
                "final_re_review_started": False,
            },
            "next_goal": {
                "goal": "FT2-20",
                "phase": "final re-review with fresh seats",
                "status": "not_started",
            },
            "highest_allowed_claim": CLAIM,
        }
        write_json(packet / "receipt.json", payload)
        upstream_receipts[name] = {
            "path": relative(packet / "receipt.json"),
            "sha256": sha256(packet / "receipt.json"),
        }
    print(
        json.dumps(
            {
                name: sha256(config["path"] / "receipt.json")
                for name, config in PACKETS.items()
            },
            sort_keys=True,
        )
    )


def finalize_aggregate() -> None:
    assert_fixed_inputs()
    checker = HERE / "check_cross_contract_consistency.py"
    checker_output = HERE / "consistency_checker_output.json"
    if not checker_output.exists():
        raise SystemExit("consistency checker output is missing")
    checker_result = json.loads(checker_output.read_text(encoding="utf-8"))
    if checker_result.get("outcome") != "pass":
        raise SystemExit("consistency checker has not passed")

    child_receipts = {
        name: {
            "path": relative(config["path"] / "receipt.json"),
            "sha256": sha256(config["path"] / "receipt.json"),
            "schema_version": json.loads(
                (config["path"] / "receipt.json").read_text(encoding="utf-8")
            )["schema_version"],
        }
        for name, config in PACKETS.items()
    }
    census_impact = FT205 / "v2_v3_impact.json"
    payload = {
        "schema_version": "Protocol101FT2FinalRepairAggregateReceiptV1",
        "goal": "FT2-FINAL-REPAIR",
        "attempt": 2,
        "outcome": "producer_repaired",
        "product_contract_hash": AUTHORITY_HASH,
        "child_receipts": child_receipts,
        "unified_findings_crosswalk": {
            "path": relative(HERE / "findings_crosswalk.json"),
            "sha256": sha256(HERE / "findings_crosswalk.json"),
            "finding_count": 47,
            "answered_count": 47,
            "partially_answered_count": 0,
            "rebutted_count": 0,
            "unanswered_count": 0,
        },
        "consistency_verification": {
            "checker_path": relative(checker),
            "checker_sha256": sha256(checker),
            "output_path": relative(checker_output),
            "output_sha256": sha256(checker_output),
            "outcome": "pass",
            "check_count": checker_result["check_count"],
            "failed_checks": checker_result["failed_checks"],
        },
        "census_v3": {
            "receipt_path": relative(FT205 / "receipt.json"),
            "receipt_sha256": sha256(FT205 / "receipt.json"),
            "impact_note_path": relative(census_impact),
            "impact_note_sha256": sha256(census_impact),
        },
        "fixed_inputs": {
            "consolidated_authority_sha256": sha256(AUTHORITY_PATH),
            "rerun001_joined_review_sha256": sha256(
                REVIEW / "joined_review.md"
            ),
            "rerun001_receipt_sha256": sha256(REVIEW / "receipt.json"),
            "intent_fill_recheck_law_sha256": sha256(
                FT208 / "intent_fill_recheck_law.json"
            ),
            "canonical_matched_random_generator_sha256": sha256(
                FT210 / "matched_random_generator_spec.json"
            ),
            "simulator_v5_sha256": sha256(SIMULATOR),
        },
        "side_effects": {
            "training_or_fitting": False,
            "new_session_statistics_beyond_census_v3": False,
            "protected_data_access": False,
            "recorder_access": False,
            "broker_access": False,
            "paid_compute": False,
            "simulator_v5_modified": False,
            "final_re_review_started": False,
        },
        "next_goal": {
            "goal": "FT2-20",
            "phase": "final re-review with fresh seats",
            "status": "not_started",
        },
        "highest_allowed_claim": CLAIM,
    }
    write_json(HERE / "aggregate_receipt.json", payload)
    print(
        json.dumps(
            {
                "aggregate_receipt_sha256": sha256(
                    HERE / "aggregate_receipt.json"
                )
            }
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("children", "aggregate", "all"),
        help="receipt set to finalize",
    )
    args = parser.parse_args()
    if args.mode in {"children", "all"}:
        finalize_children()
    if args.mode in {"aggregate", "all"}:
        finalize_aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
