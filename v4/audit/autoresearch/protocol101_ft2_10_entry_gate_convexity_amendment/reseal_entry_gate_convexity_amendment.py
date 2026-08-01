#!/usr/bin/env python3
"""Mechanically re-pin the owner-authorized FT2-10 A6 amendment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
AUTHORITY = ROOT / (
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
GRAPH = ROOT / (
    "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2.json"
)
FT204 = ROOT / "v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze"
FT205 = ROOT / "v4/audit/autoresearch/protocol101_ft2_05_opportunity_census"
FT208 = ROOT / "v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract"
FT210 = ROOT / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract"
FT211 = ROOT / "v4/audit/autoresearch/protocol101_ft2_11_evidence_statistics_contract"
CHECKER = ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_ft2_scoped_final_round_5_fixes_attempt001/"
    "check_cross_contract_consistency_v3.py"
)

OLD_AUTHORITY = "edcbee06ebfc5ac3a26fa13da043754589ba55fbbd11b207906e19459d4103f3"
NEW_AUTHORITY = "82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2"
GRAPH_SHA256 = "9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08"
OLD_CALIBRATION = "fce2756ee109f6e6ceb2df2fdb1666c92deb1545b79f648cd86972b7f6ee0371"
OLD_CENSUS_RECEIPT = "4ba3c1cc89505066e80a7abb4877c121356234cde6ce7be112eb150a38747c43"

REPINS = (
    FT204 / "oracle_rules.json",
    FT204 / "receipt.json",
    FT205 / "census_results.json",
    FT205 / "d48_reference_transition_audit.json",
    FT205 / "receipt.json",
    FT205 / "report.md",
    FT205 / "v3_v4_impact.json",
    FT208 / "receipt.json",
    FT208 / "validate_contract_v3.py",
    FT210 / "calibration_spec.json",
    FT210 / "composer_spec.json",
    FT210 / "contract.md",
    FT210 / "forecast_heads.json",
    FT210 / "realized_label_audit_composer_spec.json",
    FT210 / "receipt.json",
    FT211 / "contract.md",
    FT211 / "evidence_standard.json",
    FT211 / "mde_spec.json",
    FT211 / "receipt.json",
    FT211 / "shadow_sufficiency_spec.json",
    CHECKER,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def replace_exact(path: Path, old: str, new: str, *, required: bool = True) -> int:
    text = path.read_text()
    count = text.count(old)
    if required and count == 0:
        raise AssertionError(f"expected pin absent from {path}: {old}")
    if count:
        path.write_text(text.replace(old, new))
    return count


def refresh_deliverables(packet: Path, receipt: dict[str, Any]) -> None:
    receipt["deliverable_hashes"] = {
        relative: sha256(packet / relative)
        for relative in receipt["deliverable_hashes"]
    }


def refresh_common_inputs(
    receipt: dict[str, Any], ft204_receipt_sha: str, ft205_receipt_sha: str
) -> None:
    receipt["inputs"]["consolidated_authority"]["sha256"] = NEW_AUTHORITY
    receipt["inputs"]["ft2_04_receipt"]["sha256"] = ft204_receipt_sha
    receipt["inputs"]["census_v5_receipt"]["sha256"] = ft205_receipt_sha


def main() -> int:
    if sha256(AUTHORITY) != NEW_AUTHORITY:
        raise AssertionError("amended authority hash is not the reviewed A6 hash")
    if sha256(GRAPH) != GRAPH_SHA256:
        raise AssertionError("Graph V2 changed during a graph-unchanged amendment")
    missing_old_pins = [
        str(path.relative_to(ROOT))
        for path in REPINS
        if OLD_AUTHORITY not in path.read_text()
    ]
    if missing_old_pins:
        raise AssertionError(f"pre-reseal authority pins absent: {missing_old_pins}")
    before = {str(path.relative_to(ROOT)): sha256(path) for path in REPINS}
    replacement_counts = {
        str(path.relative_to(ROOT)): replace_exact(
            path, OLD_AUTHORITY, NEW_AUTHORITY
        )
        for path in REPINS
    }

    # FT2-04 is the first receipt in the active provenance chain.
    ft204_receipt = load(FT204 / "receipt.json")
    ft204_receipt["product_contract_hash"] = NEW_AUTHORITY
    ft204_receipt["source_hashes"]["consolidated_authority_sha256"] = NEW_AUTHORITY
    refresh_deliverables(FT204, ft204_receipt)
    write_json(FT204 / "receipt.json", ft204_receipt)
    ft204_receipt_sha = sha256(FT204 / "receipt.json")

    # The census data and economics are unchanged. Only provenance pins and
    # hashes induced by the authority reseal are refreshed.
    ft205_receipt = load(FT205 / "receipt.json")
    ft205_receipt["product_contract_hash"] = NEW_AUTHORITY
    ft205_receipt["input_hashes"]["oracle_rules.json"] = sha256(
        FT204 / "oracle_rules.json"
    )
    refresh_deliverables(FT205, ft205_receipt)
    write_json(FT205 / "receipt.json", ft205_receipt)
    ft205_receipt_sha = sha256(FT205 / "receipt.json")

    # The calibration spec changed, so direct consumers must pin its new hash.
    new_calibration_sha = sha256(FT210 / "calibration_spec.json")
    if new_calibration_sha == OLD_CALIBRATION:
        raise AssertionError("calibration spec did not incorporate A6")
    for path in (FT211 / "evidence_standard.json", FT211 / "shadow_sufficiency_spec.json"):
        replace_exact(path, OLD_CALIBRATION, new_calibration_sha)

    for packet in (FT208, FT210, FT211):
        receipt = load(packet / "receipt.json")
        receipt["product_contract_hash"] = NEW_AUTHORITY
        refresh_common_inputs(receipt, ft204_receipt_sha, ft205_receipt_sha)
        refresh_deliverables(packet, receipt)
        write_json(packet / "receipt.json", receipt)

    # Checker pins are updated only after the new census receipt is final.
    replace_exact(CHECKER, OLD_CENSUS_RECEIPT, ft205_receipt_sha)

    after = {str(path.relative_to(ROOT)): sha256(path) for path in REPINS}
    output = {
        "schema_version": "Protocol101FT210EntryGateConvexityResealV1",
        "outcome": "RESEALED_PENDING_CONSISTENCY_CHECK",
        "authority_pre_amendment_sha256": OLD_AUTHORITY,
        "authority_post_amendment_sha256": NEW_AUTHORITY,
        "graph_sha256_before_and_after": GRAPH_SHA256,
        "graph_changed": False,
        "calibration_spec_sha256": new_calibration_sha,
        "ft2_04_receipt_sha256": ft204_receipt_sha,
        "ft2_05_receipt_sha256": ft205_receipt_sha,
        "replacement_counts": replacement_counts,
        "before_sha256": before,
        "after_sha256": after,
        "scope": "entry-gate criterion and mechanically induced authority/direct-dependency/receipt/checker pins only",
    }
    write_json(HERE / "reseal_receipt.json", output)
    print(json.dumps({"outcome": output["outcome"], "authority": NEW_AUTHORITY}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
