"""Integration guards for Job 49's local-only foundation receipt."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from v5.ops import record_human_policy_foundation as cli
from v5.research import human_decision_log as human
from v5.research import human_policy_foundation_receipt as foundation


REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "v5/work/human-policy-foundation"


def _required_names() -> list[str]:
    base_contract = json.loads(
        (WORK / "PROGRAM_CONTRACT_V1.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (WORK / "PROGRAM_CONTRACT_V2.json").read_text(encoding="utf-8")
    )
    gates = base_contract["evidence_gates"]
    return (
        gates["H_human_instrumentation"]["required_tests"]
        + gates["C_offline_cmbp_catalogue"]["required_tests"]
        + contract["evidence_gate_H_v2_delta"][
            "required_tests_added_to_all_v1_gate_H_tests"
        ]
    )


def _junit(path: Path, *, failure: bool = False, omit_last: bool = False) -> Path:
    names = _required_names()
    if omit_last:
        names = names[:-1]
    cases = []
    for index, name in enumerate(names):
        body = "<failure message='synthetic failure'/>" if failure and index == 0 else ""
        cases.append(f"<testcase classname='job49' name='{name}'>{body}</testcase>")
    path.write_text(
        f"<testsuite tests='{len(cases)}'>" + "".join(cases) + "</testsuite>",
        encoding="utf-8",
    )
    return path


def test_program_contract_semantic_hash_normalizes_storage_status() -> None:
    base_path = WORK / "PROGRAM_CONTRACT_V1.json"
    contract_path = WORK / "PROGRAM_CONTRACT_V2.json"
    base = foundation.verify_program_contract(base_path)
    contract = foundation.verify_program_contract(contract_path)

    for verified in (base, contract):
        claimed = verified["self_hash"]["value"]
        assert foundation.contract_semantic_sha256(verified) == claimed
        changed = copy.deepcopy(verified)
        changed["self_hash"]["status"] = "UNSEALED"
        assert foundation.contract_semantic_sha256(changed) == claimed

    supersession = contract["supersession"]
    assert supersession["base_contract_semantic_sha256"] == base["self_hash"][
        "value"
    ]
    assert supersession["base_contract_file_sha256"] == foundation.file_sha256(
        base_path
    )
    prior_receipt_path = WORK / "LOCAL_FOUNDATION_RECEIPT_V1.json"
    prior_receipt = foundation.strict_json(prior_receipt_path)
    assert supersession["base_receipt_semantic_sha256"] == prior_receipt[
        "receipt_sha256"
    ]
    assert supersession["base_receipt_file_sha256"] == foundation.file_sha256(
        prior_receipt_path
    )


def test_test_report_requires_zero_failures_and_every_frozen_case(tmp_path: Path) -> None:
    with pytest.raises(foundation.FoundationReceiptError, match="failures=1"):
        foundation.verify_junit_report(
            _junit(tmp_path / "failed.xml", failure=True),
            required_test_names=_required_names(),
        )
    with pytest.raises(foundation.FoundationReceiptError, match="missing="):
        foundation.verify_junit_report(
            _junit(tmp_path / "missing.xml", omit_last=True),
            required_test_names=_required_names(),
        )


def test_local_receipt_binds_scope_journal_code_and_zero_external_actions(
    tmp_path: Path,
) -> None:
    report = _junit(tmp_path / "pass.xml")
    journal_path = WORK / "SYNTHETIC_HUMAN_JOURNAL_V2.jsonl"
    receipt = foundation.build_local_foundation_receipt(
        repo_root=REPO,
        test_report_path=report,
        scope_manifest_path=WORK / "CMBP_SCOPE_MANIFEST_V1.json",
        catalogue_declaration_path=WORK / "CMBP_CATALOGUE_DECLARATION_V1.json",
        synthetic_journal_path=journal_path,
    )
    assert receipt["status"] == "JOB49_LOCAL_FOUNDATION_PASS_ONLY_V2"
    assert receipt["next_state"] == "BLOCKED_AWAITING_JOB50_OWNER_GATE"
    assert receipt["integrity"] == foundation.INTEGRITY_ZERO

    journal = human.verify_log(journal_path)
    watermark = human.verify_watermark(journal_path)
    watermark_path = human.watermark_path(journal_path)
    human_gate = receipt["human_gate"]
    assert human_gate["journal_file_sha256"] == foundation.file_sha256(journal_path)
    assert human_gate["watermark_file_sha256"] == foundation.file_sha256(
        watermark_path
    )
    assert human_gate["watermark_sha256"] == watermark.watermark_sha256
    assert human_gate["watermark_file_mode"] == "0600"
    assert human_gate["log_id"] == journal.log_id == watermark.log_id
    assert (
        human_gate["terminal_sequence"]
        == journal.terminal_sequence
        == watermark.terminal_sequence
    )
    assert human_gate["terminal_head"] == journal.head == watermark.terminal_head
    assert human_gate["schema_version"] == watermark.journal_schema_version
    assert (
        human_gate["program_contract_sha256"]
        == watermark.program_contract_sha256
        == human.PROGRAM_CONTRACT_SHA256
    )
    foundation.validate_local_foundation_receipt(
        receipt,
        repo_root=REPO,
        test_report_path=report,
        scope_manifest_path=WORK / "CMBP_SCOPE_MANIFEST_V1.json",
        catalogue_declaration_path=WORK / "CMBP_CATALOGUE_DECLARATION_V1.json",
        synthetic_journal_path=journal_path,
    )
    tampered = copy.deepcopy(receipt)
    tampered["integrity"]["vendor_calls"] = 1
    with pytest.raises(foundation.FoundationReceiptError):
        foundation.validate_local_foundation_receipt(
            tampered,
            repo_root=REPO,
            test_report_path=report,
            scope_manifest_path=WORK / "CMBP_SCOPE_MANIFEST_V1.json",
            catalogue_declaration_path=WORK / "CMBP_CATALOGUE_DECLARATION_V1.json",
            synthetic_journal_path=journal_path,
        )


def test_receipt_writer_has_no_network_vendor_broker_market_or_order_import() -> None:
    forbidden = {"databento", "requests", "httpx", "socket", "ib_insync", "ibapi"}
    for module_path in (Path(foundation.__file__), Path(cli.__file__)):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden
