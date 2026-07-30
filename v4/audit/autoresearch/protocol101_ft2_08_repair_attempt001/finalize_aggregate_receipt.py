#!/usr/bin/env python3
"""Issue the aggregate receipt after independent completion review."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ATTEMPT_DIR = Path(__file__).resolve().parent
FT204_DIR = ROOT / "v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze"
FT205_DIR = ROOT / "v4/audit/autoresearch/protocol101_ft2_05_opportunity_census"
FT208_DIR = (
    ROOT / "v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract"
)
AUTHORITY = (
    ROOT
    / "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
NEW_AUTHORITY_SHA256 = (
    "2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a"
)
HIGHEST_ALLOWED_CLAIM = (
    "FT2-08 (with its FT2-04/05 dependencies) is repaired against the "
    "FT2-20 findings under the owner's t+1 fill convention; FT2-10 repair "
    "is unblocked."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    audit_json_path = ATTEMPT_DIR / "independent_completion_audit.json"
    audit_md_path = ATTEMPT_DIR / "independent_completion_audit.md"
    audit = read_json(audit_json_path)
    if audit.get("outcome") != "accepted":
        raise RuntimeError(
            "aggregate receipt forbidden: independent audit did not accept"
        )
    if sha256(AUTHORITY) != NEW_AUTHORITY_SHA256:
        raise RuntimeError("aggregate receipt forbidden: authority hash changed")
    node_receipts = {
        "ft2_04": FT204_DIR / "receipt.json",
        "ft2_05": FT205_DIR / "receipt.json",
        "ft2_08": FT208_DIR / "receipt.json",
    }
    expected_outcomes = {
        "ft2_04": "labels_frozen_v2",
        "ft2_05": "feasible",
        "ft2_08": "producer_repaired",
    }
    for name, path in node_receipts.items():
        outcome = read_json(path).get("outcome")
        if outcome != expected_outcomes[name]:
            raise RuntimeError(
                f"aggregate receipt forbidden: {name} outcome {outcome!r}"
            )
    deliverables = [
        ATTEMPT_DIR / "report.md",
        ATTEMPT_DIR / "findings_crosswalk.json",
        ATTEMPT_DIR / "step0_data_feasibility.json",
        ATTEMPT_DIR / "step0_quote_per_minute.csv",
        ATTEMPT_DIR / "finalize_producer_repair.py",
        ATTEMPT_DIR / "finalize_aggregate_receipt.py",
        audit_json_path,
        audit_md_path,
        *node_receipts.values(),
    ]
    receipt = {
        "schema_version": "Protocol101FT208RepairAttemptReceiptV2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "goal": "FT2-08-REPAIR",
        "node": "design_repair_data_tensor_label",
        "repair_attempt": 1,
        "repair_budget": 2,
        "outcome": "producer_repaired",
        "highest_allowed_claim": HIGHEST_ALLOWED_CLAIM,
        "authority_sha256": NEW_AUTHORITY_SHA256,
        "repair_of": {
            "step0_receipt_sha256": (
                "5704e2fab4eaead28a49c240f5ac61fe97ae4e59df0c52fd7150210a4a0a8e2f"
            ),
            "ft2_04_receipt_sha256": (
                "0b4a4b14b250e377211f29e3fcd4ab2051bacd9f65079bb73f0a3cc20928ce91"
            ),
            "ft2_05_receipt_sha256": (
                "9085a09cbc1583973fdb372c78d78568e3fb1baa0fa1d9b42c23b05543ee24de"
            ),
            "ft2_08_receipt_sha256": (
                "56885fad09ea7034ce112c356edcce92bad0d9c52e93fb8e32cf66934f5ce518"
            ),
            "authority_sha256": (
                "893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832"
            ),
        },
        "current_node_receipts": {
            name: {
                "path": relative(path),
                "sha256": sha256(path),
                "outcome": expected_outcomes[name],
            }
            for name, path in node_receipts.items()
        },
        "independent_completion_audit": {
            "outcome": audit["outcome"],
            "json_sha256": sha256(audit_json_path),
            "markdown_sha256": sha256(audit_md_path),
        },
        "deliverable_hashes": {
            relative(path): sha256(path) for path in deliverables
        },
        "assertions": {
            "steps_1_through_4_complete": True,
            "all_28_findings_crosswalked": True,
            "independent_completion_audit_accepted": True,
            "repair_of_chain_complete": True,
            "superseded_packets_preserved": True,
            "ft2_10_repair_unblocked_but_not_started": True,
            "ft2_20_fresh_seat_rerun_deferred_until_all_three_repairs": True,
            "highest_allowed_claim_not_exceeded": True,
        },
        "side_effects": {
            "model_training_or_tuning": False,
            "protected_holdout_accessed": False,
            "outer_test_accessed": False,
            "recorder_data_accessed": False,
            "broker_contacted": False,
            "paid_data_or_compute_used": False,
            "simulator_v5_source_modified": False,
            "runtime_or_promotion_modified": False,
            "ft2_10_repaired_or_started": False,
            "ft2_11_repaired_or_started": False,
        },
        "next": "FT2-10-ENTRY-SCIENCE-CONTRACT-REPAIR",
        "required_stop_observed": True,
        "self_hash": {
            "included_in_deliverable_hashes": False,
            "reason": "the aggregate receipt cannot hash itself",
        },
    }
    (ATTEMPT_DIR / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "outcome": receipt["outcome"],
                "receipt_sha256": sha256(ATTEMPT_DIR / "receipt.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
