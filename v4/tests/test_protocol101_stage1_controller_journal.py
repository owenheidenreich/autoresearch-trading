from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from v4.model.protocol101_stage1_controller_journal import (
    ControllerJournalError,
    JOURNAL_NODES,
    JOURNAL_ROUTES,
    append_checkpoint,
    create_journal,
    stable_hash,
    validate_journal,
)
from v4.scripts.run_protocol101_full_trader_stage1_gate_audit_selection_validation import (
    _authorities,
    build_synthetic_campaign,
)
from v4.scripts.run_protocol101_scoped_stage1_gate_aggregator import (
    aggregate_campaign,
)
from v4.scripts.run_protocol101_scoped_stage1_independent_audit import (
    audit_campaign,
)
from v4.scripts.run_protocol101_stage1_autoresearch_graph import (
    AUTHORIZATION_SCHEMA,
    CAMPAIGN_NAMESPACE,
    INVALID_CHAIN_ROUTE,
    STOP_ROUTE,
    evaluate_graph,
    owner_authorization_sha256,
)
from v4.scripts.run_protocol101_stage1_cross_hypothesis_selection import (
    INVALID_ROUTE,
    SELECTED_ROUTE,
    select_candidate,
)


def _owner(execution_id: str = "FT1C5-JOURNAL-TEST-001") -> dict:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "campaign_execution_id": execution_id,
        "authorized": True,
        "routing_decision": "owner_authorized_fresh_420_unit_campaign_execution",
        "owner_signature": "Owen Heidenreich",
        "owner_decision_date": "2026-07-26",
        "goal_sha256": "1" * 64,
        "preregistration_sha256": "2" * 64,
        "contract_bundle_sha256": "3" * 64,
        "seed_45_or_G9_authorized": False,
    }


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _create(
    tmp_path: Path,
    *,
    prefix_length: int = 0,
    execution_id: str = "FT1C5-JOURNAL-TEST-001",
    audit=None,
) -> tuple[Path, dict, dict[str, Path]]:
    owner = _owner(execution_id)
    inputs: dict[str, Path] = {}
    for name in ("option", "offline", "goal", "prereg", "contracts"):
        path = tmp_path / f"{name}.json"
        _json(path, {"name": name})
        inputs[name] = path
    journal = tmp_path / "controller.jsonl"
    create_journal(
        journal,
        workspace_root=tmp_path,
        campaign_namespace=CAMPAIGN_NAMESPACE,
        campaign_execution_id=execution_id,
        owner_option_a_decision_path=inputs["option"],
        offline_training_authorization_path=inputs["offline"],
        campaign_goal_path=inputs["goal"],
        campaign_preregistration_path=inputs["prereg"],
        signed_contract_bundle_path=inputs["contracts"],
        owner_execution_authorization_sha256=owner_authorization_sha256(owner),
        owner_identity=owner["owner_signature"],
        owner_decision_date=owner["owner_decision_date"],
        timestamp_utc="2026-07-26T00:00:00+00:00",
    )
    artifacts: dict[str, Path] = {}
    for index, node in enumerate(JOURNAL_NODES[:prefix_length], start=1):
        artifact = tmp_path / f"{index:02d}_{node}.json"
        validator = tmp_path / f"{index:02d}_{node}_validator.json"
        _json(
            artifact,
            audit if node == "INDEPENDENT_AUDIT" and audit is not None else {
                "node": node,
                "accepted": True,
            },
        )
        _json(validator, {"routing_decision": JOURNAL_ROUTES[node]})
        artifacts[node] = artifact
        append_checkpoint(
            journal,
            workspace_root=tmp_path,
            node=node,
            artifact_path=artifact,
            validator_route=JOURNAL_ROUTES[node],
            validator_receipt_path=validator,
            timestamp_utc=f"2026-07-26T00:00:{index:02d}+00:00",
        )
    return journal, owner, artifacts


@pytest.mark.parametrize("prefix_length", range(9))
def test_all_valid_prefixes_and_crash_resume(tmp_path, prefix_length) -> None:
    journal, owner, _ = _create(tmp_path, prefix_length=prefix_length)
    first = validate_journal(
        journal,
        workspace_root=tmp_path,
        expected_campaign_namespace=CAMPAIGN_NAMESPACE,
        expected_campaign_execution_id=owner["campaign_execution_id"],
        expected_owner_authorization_sha256=owner_authorization_sha256(owner),
    )
    second = validate_journal(
        journal,
        workspace_root=tmp_path,
        expected_campaign_namespace=CAMPAIGN_NAMESPACE,
        expected_campaign_execution_id=owner["campaign_execution_id"],
        expected_owner_authorization_sha256=owner_authorization_sha256(owner),
    )
    assert first["journal_head_sha256"] == second["journal_head_sha256"]
    assert first["completed_prefix"] == list(JOURNAL_NODES[:prefix_length])


@pytest.mark.parametrize(
    ("line_index", "field", "value"),
    [
        (0, "campaign_namespace", "other"),
        (0, "campaign_execution_id", "other"),
        (0, "owner_identity", "changed"),
        (0, "owner_decision_date", "changed"),
        (0, "previous_record_sha256", "f" * 64),
        (0, "record_sha256", "f" * 64),
        (1, "ordinal", 2),
        (1, "node", "CONTROL_AUTHORITY"),
        (1, "routing_decision", "changed"),
        (1, "campaign_namespace", "other"),
        (1, "campaign_execution_id", "other"),
        (1, "owner_genesis_sha256", "f" * 64),
        (1, "previous_record_sha256", "f" * 64),
        (1, "artifact_sha256", "f" * 64),
        (1, "validator_route", "changed"),
        (1, "validator_receipt_sha256", "f" * 64),
        (1, "record_sha256", "f" * 64),
    ],
)
def test_changed_journal_line_fails_closed(
    tmp_path,
    line_index,
    field,
    value,
) -> None:
    journal, _, _ = _create(tmp_path, prefix_length=1)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    records[line_index][field] = value
    journal.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in records
        )
    )
    with pytest.raises(ControllerJournalError):
        validate_journal(journal, workspace_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "owner_option_a_decision_path",
        "offline_training_authorization_path",
        "campaign_goal_path",
        "campaign_preregistration_path",
        "signed_contract_bundle_path",
        "owner_execution_authorization_sha256",
        "journal_schema",
        "record_kind",
    ],
)
def test_missing_genesis_binding_fails_closed(tmp_path, field) -> None:
    journal, _, _ = _create(tmp_path)
    genesis = json.loads(journal.read_text())
    del genesis[field]
    journal.write_text(
        json.dumps(genesis, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(ControllerJournalError):
        validate_journal(journal, workspace_root=tmp_path)


@pytest.mark.parametrize(
    "node",
    [
        "G9_seed45",
        "protected_holdout",
        "learned_exits",
        "paper",
        "broker",
        "runtime_promotion",
        "UNKNOWN",
    ],
)
def test_forbidden_or_unknown_nodes_reject(tmp_path, node) -> None:
    journal, _, _ = _create(tmp_path)
    artifact = tmp_path / "artifact.json"
    validator = tmp_path / "validator.json"
    _json(artifact, {})
    _json(validator, {})
    with pytest.raises(ControllerJournalError):
        append_checkpoint(
            journal,
            workspace_root=tmp_path,
            node=node,
            artifact_path=artifact,
            validator_route="invalid",
            validator_receipt_path=validator,
        )


@pytest.mark.parametrize("wrong_node", JOURNAL_NODES[1:])
def test_skipped_node_rejects(tmp_path, wrong_node) -> None:
    journal, _, _ = _create(tmp_path)
    artifact = tmp_path / "artifact.json"
    validator = tmp_path / "validator.json"
    _json(artifact, {})
    _json(validator, {})
    with pytest.raises(ControllerJournalError):
        append_checkpoint(
            journal,
            workspace_root=tmp_path,
            node=wrong_node,
            artifact_path=artifact,
            validator_route=JOURNAL_ROUTES[wrong_node],
            validator_receipt_path=validator,
        )


def test_duplicate_append_rejects(tmp_path) -> None:
    journal, _, _ = _create(tmp_path, prefix_length=1)
    artifact = tmp_path / "duplicate.json"
    validator = tmp_path / "duplicate_validator.json"
    _json(artifact, {})
    _json(validator, {})
    with pytest.raises(ControllerJournalError):
        append_checkpoint(
            journal,
            workspace_root=tmp_path,
            node=JOURNAL_NODES[0],
            artifact_path=artifact,
            validator_route=JOURNAL_ROUTES[JOURNAL_NODES[0]],
            validator_receipt_path=validator,
        )


@pytest.mark.parametrize("target", ["artifact", "validator"])
def test_changed_accepted_file_rejects(tmp_path, target) -> None:
    journal, _, artifacts = _create(tmp_path, prefix_length=1)
    state = validate_journal(journal, workspace_root=tmp_path)
    record = state["records"][1]
    path = (
        artifacts[JOURNAL_NODES[0]]
        if target == "artifact"
        else tmp_path / record["validator_receipt_path"]
    )
    path.write_text("changed\n")
    with pytest.raises(ControllerJournalError):
        validate_journal(journal, workspace_root=tmp_path)


def test_partial_final_line_rejects(tmp_path) -> None:
    journal, _, _ = _create(tmp_path, prefix_length=1)
    with journal.open("ab") as handle:
        handle.write(b'{"partial":')
    with pytest.raises(
        ControllerJournalError,
        match="journal_partial_final_line",
    ):
        validate_journal(journal, workspace_root=tmp_path)


@pytest.mark.parametrize("mode", [0o644, 0o666, 0o604])
def test_non_owner_permissions_reject(tmp_path, mode) -> None:
    journal, _, _ = _create(tmp_path)
    os.chmod(journal, mode)
    with pytest.raises(
        ControllerJournalError,
        match="journal_permissions_not_owner_only",
    ):
        validate_journal(journal, workspace_root=tmp_path)


def test_exclusive_genesis_refuses_overwrite(tmp_path) -> None:
    journal, owner, _ = _create(tmp_path)
    with pytest.raises(FileExistsError):
        create_journal(
            journal,
            workspace_root=tmp_path,
            campaign_namespace=CAMPAIGN_NAMESPACE,
            campaign_execution_id=owner["campaign_execution_id"],
            owner_option_a_decision_path=tmp_path / "option.json",
            offline_training_authorization_path=tmp_path / "offline.json",
            campaign_goal_path=tmp_path / "goal.json",
            campaign_preregistration_path=tmp_path / "prereg.json",
            signed_contract_bundle_path=tmp_path / "contracts.json",
            owner_execution_authorization_sha256=owner_authorization_sha256(
                owner
            ),
            owner_identity=owner["owner_signature"],
            owner_decision_date=owner["owner_decision_date"],
        )


def test_stale_or_cross_campaign_journal_rejects(tmp_path) -> None:
    journal, owner, _ = _create(tmp_path, execution_id="EXECUTION-A")
    with pytest.raises(ControllerJournalError):
        validate_journal(
            journal,
            workspace_root=tmp_path,
            expected_campaign_execution_id="EXECUTION-B",
        )
    result = evaluate_graph(
        owner_authorization=_owner("EXECUTION-B"),
        controller_journal_path=journal,
        workspace_root=tmp_path,
    )
    assert result["routing_decision"] == INVALID_CHAIN_ROUTE
    assert result["next_node"] != "STOP"
    assert owner["campaign_execution_id"] == "EXECUTION-A"


def test_concurrent_append_allows_exactly_one_writer(tmp_path) -> None:
    journal, _, _ = _create(tmp_path)
    artifact = tmp_path / "artifact.json"
    validator = tmp_path / "validator.json"
    _json(artifact, {})
    _json(validator, {})

    def attempt():
        try:
            append_checkpoint(
                journal,
                workspace_root=tmp_path,
                node=JOURNAL_NODES[0],
                artifact_path=artifact,
                validator_route=JOURNAL_ROUTES[JOURNAL_NODES[0]],
                validator_receipt_path=validator,
            )
            return "accepted"
        except ControllerJournalError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(results) == ["accepted", "rejected"]


@pytest.mark.parametrize(
    "case_id",
    [
        "AUTH-ATTACK-02-authority_rebuilt",
        "AUTH-ATTACK-03-all_self_hashes",
        "AUTH-ATTACK-09-control_rebuilt",
    ],
)
def test_old_self_resealed_receipt_attacks_cannot_reach_stop(
    tmp_path,
    case_id,
) -> None:
    root = Path(__file__).resolve().parents[2]
    reproducer = json.loads(
        (
            root
            / "v4/audit/autoresearch/"
            "protocol101_full_trader_stage1_gate_audit_selection_"
            "independent_acceptance_attempt003/reproducers/"
            f"{case_id}.json"
        ).read_text()
    )
    result = evaluate_graph(
        owner_authorization=reproducer["owner"],
        receipts=reproducer["chain"],
        trusted_artifact_sha256_by_node=reproducer["bindings"],
        workspace_root=tmp_path,
    )
    assert result["routing_decision"] == INVALID_CHAIN_ROUTE
    assert result["next_node"] != "STOP"


def _audit_fixture():
    packet = build_synthetic_campaign()
    authorities = _authorities(packet)
    producer = aggregate_campaign(packet, **authorities)
    return audit_campaign(packet, producer, **authorities)


def test_selector_accepts_only_exact_journaled_audit(tmp_path) -> None:
    audit = _audit_fixture()
    journal, owner, artifacts = _create(
        tmp_path,
        prefix_length=7,
        audit=audit,
    )
    result = select_candidate(
        audit,
        controller_journal_path=journal,
        audit_result_path=artifacts["INDEPENDENT_AUDIT"],
        workspace_root=tmp_path,
        campaign_namespace=CAMPAIGN_NAMESPACE,
        campaign_execution_id=owner["campaign_execution_id"],
    )
    assert result["routing_decision"] == SELECTED_ROUTE


@pytest.mark.parametrize(
    "case_id",
    [
        "SEL-fully-resealed-result",
        "SEL-fully-resealed-producer-independent",
    ],
)
def test_fully_resealed_unjournaled_audit_rejects(tmp_path, case_id) -> None:
    baseline = _audit_fixture()
    journal, owner, artifacts = _create(
        tmp_path,
        prefix_length=7,
        audit=baseline,
    )
    root = Path(__file__).resolve().parents[2]
    forged = json.loads(
        (
            root
            / "v4/audit/autoresearch/"
            "protocol101_full_trader_stage1_gate_audit_selection_"
            "independent_acceptance_attempt003/reproducers/"
            f"{case_id}.json"
        ).read_text()
    )
    result = select_candidate(
        forged,
        controller_journal_path=journal,
        audit_result_path=artifacts["INDEPENDENT_AUDIT"],
        workspace_root=tmp_path,
        campaign_namespace=CAMPAIGN_NAMESPACE,
        campaign_execution_id=owner["campaign_execution_id"],
    )
    assert result["routing_decision"] == INVALID_ROUTE
    assert result["selected_candidate"] is None


def test_in_memory_audit_rewrite_rejects(tmp_path) -> None:
    audit = _audit_fixture()
    journal, owner, artifacts = _create(
        tmp_path,
        prefix_length=7,
        audit=audit,
    )
    changed = deepcopy(audit)
    changed["independent_result"]["rows"][0][
        "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
    ] += 10
    result = select_candidate(
        changed,
        controller_journal_path=journal,
        audit_result_path=artifacts["INDEPENDENT_AUDIT"],
        workspace_root=tmp_path,
        campaign_namespace=CAMPAIGN_NAMESPACE,
        campaign_execution_id=owner["campaign_execution_id"],
    )
    assert result["routing_decision"] == INVALID_ROUTE
