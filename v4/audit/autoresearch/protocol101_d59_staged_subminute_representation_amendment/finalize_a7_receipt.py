#!/usr/bin/env python3
"""Emit the final A7 implementation receipt after the governed checker passes."""
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
CHECKER = ROOT / (
    "v4/audit/autoresearch/"
    "protocol101_ft2_scoped_final_round_5_fixes_attempt001/"
    "consistency_checker_output.json"
)
PROPOSAL = ROOT / (
    "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_D59_STAGED_SUBMINUTE_REPRESENTATION_AMENDMENT_PROPOSAL_2026_07_31.md"
)
LEDGER = ROOT / (
    "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_WALKING_SKELETON_LEARNINGS_LEDGER_2026_07_30.md"
)
REVIEW = HERE / "codex_review.md"
POLICY = HERE / "sub_minute_tier_policy.json"
REGISTRY = HERE / "sub_minute_corpus_registry.json"
RESEAL = HERE / "reseal_receipt.json"

OLD_AUTHORITY = "82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2"
NEW_AUTHORITY = "1d215845cf7b853550c5cf27af5bafca66db2355e0f12493e2c5a8922278d4bc"
GRAPH_SHA256 = "9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    reseal = load(RESEAL)
    checker = load(CHECKER)
    if sha256(AUTHORITY) != NEW_AUTHORITY:
        raise AssertionError("A7 authority hash drift")
    if sha256(GRAPH) != GRAPH_SHA256:
        raise AssertionError("Graph V2 changed")
    if reseal.get("authority_pre_amendment_sha256") != OLD_AUTHORITY:
        raise AssertionError("reseal old-authority identity mismatch")
    if reseal.get("authority_post_amendment_sha256") != NEW_AUTHORITY:
        raise AssertionError("reseal new-authority identity mismatch")
    if reseal.get("graph_changed") is not False:
        raise AssertionError("reseal claims graph change")
    if checker.get("outcome") != "pass" or checker.get("failed_checks"):
        raise AssertionError("cross-contract checker is not green")
    if checker.get("check_count") != 33:
        raise AssertionError("unexpected checker count")
    tier_check = checker.get("checks", {}).get(
        "sub_minute_corpus_tier_tag_required", {}
    )
    if tier_check.get("pass") is not True:
        raise AssertionError("A7 anti-regression checker rule is not passing")
    if not all(tier_check["evidence"]["negative_fixtures"].values()):
        raise AssertionError("A7 checker negative fixtures did not all reject")

    artifacts = {
        "authority": AUTHORITY,
        "graph": GRAPH,
        "proposal": PROPOSAL,
        "codex_review": REVIEW,
        "sub_minute_tier_policy": POLICY,
        "sub_minute_corpus_registry": REGISTRY,
        "consistency_checker_output": CHECKER,
        "reseal_receipt": RESEAL,
        "learnings_ledger": LEDGER,
    }
    receipt: dict[str, Any] = {
        "schema_version": "Protocol101D59StagedSubMinuteA7ImplementationReceiptV1",
        "amendment_id": "A7",
        "status": "IMPLEMENTED_PENDING_CLAUDE_VERIFICATION_AND_OWNER_SIGNATURE",
        "review": {
            "verdict": "ENDORSED_WITH_REQUIRED_IMPLEMENTATION_CLARIFICATIONS",
            "required_clarifications": [
                "cbbo-1s is last consolidated BBO in time space and not an event-complete path",
                "cmbp-1 is the consolidated top-of-book event stream relevant to crossings",
                "trusted floor/stop labels inspect raw events before downsampling",
                "A7 does not silently activate one-second training/runtime cadence",
                "tier enforcement uses a machine-readable policy and corpus registry"
            ],
            "path": str(REVIEW.relative_to(ROOT)),
            "sha256": sha256(REVIEW),
        },
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "old_sha256": OLD_AUTHORITY,
            "new_sha256": NEW_AUTHORITY,
        },
        "graph": {
            "path": str(GRAPH.relative_to(ROOT)),
            "sha256_before": GRAPH_SHA256,
            "sha256_after": GRAPH_SHA256,
            "changed": False,
        },
        "checker": {
            "path": str(CHECKER.relative_to(ROOT)),
            "sha256": sha256(CHECKER),
            "outcome": checker["outcome"],
            "checks_passed": checker["check_count"],
            "checks_total": checker["check_count"],
            "failed_checks": checker["failed_checks"],
            "new_rule": "sub_minute_corpus_tier_tag_required",
            "new_rule_pass": tier_check["pass"],
            "negative_fixtures": tier_check["evidence"]["negative_fixtures"],
            "discovered_sub_minute_manifest_count": tier_check["evidence"][
                "discovered_sub_minute_manifest_count"
            ],
            "registered_corpus_count": tier_check["evidence"][
                "registered_corpus_count"
            ],
        },
        "tier_contract": {
            "tier_s": "cbbo-1s; skeleton|prototype|probe; floor/stop labels 1-second-approximate; never trusted-consumer eligible",
            "tier_t": "cmbp-1-derived 1-second model input; raw events retained and used for trusted floor/stop labels; downsampler certified",
            "tier_s_combined_evaluation_earliest_session": "2025-02-20",
            "cbbo_1s_acquisition_requires": [
                "free per-acquisition cost estimate",
                "hard cost cap",
                "explicit owner green-light",
                "quarantined output"
            ],
            "cmbp_1_purchase_authorized": False,
            "d57_remains_deferred": True,
        },
        "repin": {
            "repinned_file_count": len(reseal["repinned_files"]),
            "repinned_files": reseal["repinned_files"],
            "already_new_pinned_files": reseal["already_new_pinned_files"],
            "ft2_04_receipt_sha256": reseal["ft2_04_receipt_sha256"],
            "ft2_05_receipt_sha256": reseal["ft2_05_receipt_sha256"],
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in artifacts.items()
        },
        "side_effects": {
            "broker_contacted": False,
            "cmbp_1_purchase_or_download": False,
            "data_download_or_purchase": False,
            "graph_edited": False,
            "model_training_or_fitting": False,
            "protected_resource_access": False,
            "recorder_contacted_or_activated": False,
            "runtime_or_default_change": False,
        },
        "highest_allowed_claim": "D59 staged into Tier-S/Tier-T via A7; trusted floor/stop guarantee preserved; graph unchanged; awaiting Claude verification + owner signature.",
        "stop": "STOP_FOR_CLAUDE_VERIFICATION",
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    (HERE / "implementation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "authority": NEW_AUTHORITY,
                "graph": GRAPH_SHA256,
                "checker": f"{checker['check_count']}/{checker['check_count']}",
                "receipt_hash": receipt["receipt_hash"],
                "stop": receipt["stop"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

