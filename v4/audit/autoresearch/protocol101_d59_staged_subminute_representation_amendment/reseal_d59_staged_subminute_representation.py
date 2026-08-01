#!/usr/bin/env python3
"""Mechanically re-pin the owner-authorized D59 A7 authority amendment."""
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
POLICY = HERE / "sub_minute_tier_policy.json"
REGISTRY = HERE / "sub_minute_corpus_registry.json"

OLD_AUTHORITY = "82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2"
NEW_AUTHORITY = "1d215845cf7b853550c5cf27af5bafca66db2355e0f12493e2c5a8922278d4bc"
GRAPH_SHA256 = "9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08"
OLD_CENSUS_RECEIPT = "3493afd12a0dfc85bde24736346d89995b4fce6c3480579062d462253d3add06"

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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def replace_exact(path: Path, old: str, new: str) -> int:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count:
        path.write_text(text.replace(old, new), encoding="utf-8")
    elif new not in text:
        raise AssertionError(f"neither old nor new pin exists in {path}")
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
        raise AssertionError("amended authority hash is not the reviewed A7 hash")
    if sha256(GRAPH) != GRAPH_SHA256:
        raise AssertionError("Graph V2 changed during graph-unchanged A7")
    if load(POLICY).get("authority_sha256") != NEW_AUTHORITY:
        raise AssertionError("A7 tier policy is not pinned to the new authority")
    if load(REGISTRY).get("authority_sha256") != NEW_AUTHORITY:
        raise AssertionError("A7 corpus registry is not pinned to the new authority")

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

    # A7 changes authority/provenance only for the existing census artifacts;
    # no census value, label, or economic result is recomputed.
    ft205_receipt = load(FT205 / "receipt.json")
    ft205_receipt["product_contract_hash"] = NEW_AUTHORITY
    ft205_receipt["input_hashes"]["oracle_rules.json"] = sha256(
        FT204 / "oracle_rules.json"
    )
    refresh_deliverables(FT205, ft205_receipt)
    write_json(FT205 / "receipt.json", ft205_receipt)
    ft205_receipt_sha = sha256(FT205 / "receipt.json")

    for packet in (FT208, FT210, FT211):
        receipt = load(packet / "receipt.json")
        receipt["product_contract_hash"] = NEW_AUTHORITY
        refresh_common_inputs(receipt, ft204_receipt_sha, ft205_receipt_sha)
        refresh_deliverables(packet, receipt)
        write_json(packet / "receipt.json", receipt)

    # Checker pins update only after the active census receipt is final.
    replace_exact(CHECKER, OLD_CENSUS_RECEIPT, ft205_receipt_sha)

    after = {str(path.relative_to(ROOT)): sha256(path) for path in REPINS}
    output = {
        "schema_version": "Protocol101D59StagedSubMinuteA7ResealV1",
        "outcome": "RESEALED_PENDING_CONSISTENCY_CHECK",
        "authority_pre_amendment_sha256": OLD_AUTHORITY,
        "authority_post_amendment_sha256": NEW_AUTHORITY,
        "graph_sha256_before_and_after": GRAPH_SHA256,
        "graph_changed": False,
        "ft2_04_receipt_sha256": ft204_receipt_sha,
        "ft2_05_receipt_sha256": ft205_receipt_sha,
        "tier_policy_sha256": sha256(POLICY),
        "corpus_registry_sha256": sha256(REGISTRY),
        "replacement_counts": replacement_counts,
        "repinned_files": sorted(
            path for path, count in replacement_counts.items() if count > 0
        ),
        "already_new_pinned_files": sorted(
            path for path, count in replacement_counts.items() if count == 0
        ),
        "before_sha256": before,
        "after_sha256": after,
        "scope": "A7 D59 tier policy and mechanically induced active authority/spec/receipt/checker pins only",
        "side_effects": {
            "broker_or_recorder_contacted": False,
            "data_download_or_purchase": False,
            "graph_edited": False,
            "model_training_or_fitting": False,
            "protected_resource_access": False,
            "runtime_or_default_change": False
        }
    }
    write_json(HERE / "reseal_receipt.json", output)
    print(
        json.dumps(
            {
                "outcome": output["outcome"],
                "authority": NEW_AUTHORITY,
                "repinned_file_count": len(output["repinned_files"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

