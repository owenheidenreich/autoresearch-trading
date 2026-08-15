from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from v4.model.protocol101_stage1_controller_journal import (
    JOURNAL_NODES,
    JOURNAL_ROUTES,
    append_checkpoint,
    create_journal,
)
from v4.scripts.run_protocol101_stage1_autoresearch_graph import (
    AUTHORIZATION_SCHEMA,
    CAMPAIGN_NAMESPACE,
    INVALID_CHAIN_ROUTE,
    NODE_DEFINITIONS,
    NO_OWNER_ROUTE,
    RECEIPT_NODES,
    RUN_PENDING_ROUTE,
    STOP_ROUTE,
    build_receipt_chain,
    evaluate_graph,
    graph_definition,
    owner_authorization_sha256,
    seal_receipt_chain,
)


def _authorization(
    *,
    execution_id: str = "FT1C3-SYNTHETIC-RUN-001",
    signature: str = "SYNTHETIC-TEST-OWNER",
) -> dict:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "campaign_namespace": CAMPAIGN_NAMESPACE,
        "campaign_execution_id": execution_id,
        "authorized": True,
        "routing_decision": "owner_authorized_fresh_420_unit_campaign_execution",
        "owner_signature": signature,
        "owner_decision_date": "2026-07-26",
        "goal_sha256": "1" * 64,
        "preregistration_sha256": "2" * 64,
        "contract_bundle_sha256": "3" * 64,
        "seed_45_or_G9_authorized": False,
    }


def _bindings(prefix_length: int = 8, *, salt: str = "a") -> dict[str, str]:
    return {
        node: f"{index:x}" * 64
        for index, node in enumerate(
            RECEIPT_NODES[:prefix_length],
            start=1,
        )
    }


def _chain(
    prefix_length: int = 8,
    *,
    authorization: dict | None = None,
    bindings: dict[str, str] | None = None,
) -> tuple[dict, dict[str, str]]:
    authorization = authorization or _authorization()
    bindings = bindings or _bindings(prefix_length)
    return (
        build_receipt_chain(
            owner_authorization=authorization,
            artifact_sha256_by_node=bindings,
            prefix_length=prefix_length,
        ),
        bindings,
    )


def _journal(
    tmp_path: Path,
    *,
    prefix_length: int,
    authorization: dict | None = None,
) -> Path:
    authorization = authorization or _authorization()
    for name in ("option.md", "offline.md", "goal.md", "prereg.json", "contracts.json"):
        (tmp_path / name).write_text(name)
    journal = tmp_path / "controller.jsonl"
    create_journal(
        journal,
        workspace_root=tmp_path,
        campaign_namespace=CAMPAIGN_NAMESPACE,
        campaign_execution_id=authorization["campaign_execution_id"],
        owner_option_a_decision_path=tmp_path / "option.md",
        offline_training_authorization_path=tmp_path / "offline.md",
        campaign_goal_path=tmp_path / "goal.md",
        campaign_preregistration_path=tmp_path / "prereg.json",
        signed_contract_bundle_path=tmp_path / "contracts.json",
        owner_execution_authorization_sha256=(
            owner_authorization_sha256(authorization)
        ),
        owner_identity=authorization["owner_signature"],
        owner_decision_date=authorization["owner_decision_date"],
        timestamp_utc="2026-07-26T00:00:00+00:00",
    )
    for index, node in enumerate(JOURNAL_NODES[:prefix_length], start=1):
        artifact = tmp_path / f"{index:02d}_{node}.json"
        validator = tmp_path / f"{index:02d}_{node}_validator.json"
        artifact.write_text('{"accepted":true}\n')
        validator.write_text('{"valid":true}\n')
        append_checkpoint(
            journal,
            workspace_root=tmp_path,
            node=node,
            artifact_path=artifact,
            validator_route=JOURNAL_ROUTES[node],
            validator_receipt_path=validator,
            timestamp_utc=f"2026-07-26T00:00:{index:02d}+00:00",
        )
    return journal


def test_no_owner_graph_stops_before_RUN_without_commands() -> None:
    result = evaluate_graph()
    assert result["routing_decision"] == NO_OWNER_ROUTE
    assert result["next_node"] == "OWNER_EXECUTION_AUTHORIZATION"
    assert result["RUN_executed"] is False
    assert result["commands_executed"] == []
    assert result["owner_authorization_created"] is False
    assert result["G9_executed"] is False
    assert result["stopped_before_RUN"] is True


def test_graph_contract_stops_before_G9_holdout_and_paper() -> None:
    contract = graph_definition()
    assert contract["autonomous_execution"] is False
    assert contract["permission_escalation"] is False
    assert contract["owner_authorization_created_by_graph"] is False
    assert contract["trusted_artifact_bindings_are_authorization"] is False
    assert set(contract["forbidden_downstream_nodes"]) == {
        "G9_seed45",
        "protected_holdout",
        "learned_exits",
        "transfer",
        "paper",
    }


def test_valid_owner_with_genesis_journal_waits_for_RUN(tmp_path) -> None:
    authorization = _authorization()
    journal = _journal(tmp_path, prefix_length=0, authorization=authorization)
    result = evaluate_graph(
        owner_authorization=authorization,
        controller_journal_path=journal,
        workspace_root=tmp_path,
    )
    assert result["routing_decision"] == RUN_PENDING_ROUTE
    assert result["next_node"] == "FRESH_420_UNIT_RUN"
    assert result["owner_authorization_sha256"] == owner_authorization_sha256(
        authorization
    )
    assert result["completed_prefix"] == []


@pytest.mark.parametrize("prefix_length", range(9))
def test_every_valid_chain_prefix_advances_to_exact_next_node(
    prefix_length: int,
    tmp_path,
) -> None:
    authorization = _authorization()
    journal = _journal(
        tmp_path,
        prefix_length=prefix_length,
        authorization=authorization,
    )
    result = evaluate_graph(
        owner_authorization=authorization,
        controller_journal_path=journal,
        workspace_root=tmp_path,
    )
    expected_next = (
        "STOP"
        if prefix_length == len(RECEIPT_NODES)
        else RECEIPT_NODES[prefix_length]
    )
    expected_route = STOP_ROUTE if expected_next == "STOP" else RUN_PENDING_ROUTE
    assert result["routing_decision"] == expected_route
    assert result["next_node"] == expected_next
    assert result["completed_prefix"] == list(RECEIPT_NODES[:prefix_length])
    assert set(result["accepted_artifact_bindings"]) == set(
        JOURNAL_NODES[:prefix_length]
    )
    assert result["RUN_executed"] is False
    assert result["G9_executed"] is False


def test_stale_v1_receipt_bag_rejects() -> None:
    result = evaluate_graph(
        owner_authorization=_authorization(),
        receipts={"FRESH_420_UNIT_RUN": {"receipt_sha256": "a" * 64}},
    )
    assert result["routing_decision"] == INVALID_CHAIN_ROUTE
    assert result["next_node"] == "INVALID_CONTROLLER_JOURNAL"


def test_changed_owner_with_old_chain_rejects(tmp_path) -> None:
    authorization = _authorization()
    journal = _journal(tmp_path, prefix_length=8, authorization=authorization)
    changed = dict(authorization)
    changed["owner_signature"] = "CHANGED-AFTER-CHAIN"
    result = evaluate_graph(
        owner_authorization=changed,
        controller_journal_path=journal,
        workspace_root=tmp_path,
    )
    assert result["routing_decision"] == INVALID_CHAIN_ROUTE
    assert any(
        "journal_owner_authorization_mismatch" in blocker
        for blocker in result["blockers"]
    )


def test_rebuilt_descendants_with_frozen_artifact_bindings_reject() -> None:
    authorization = _authorization()
    chain, trusted = _chain(8, authorization=authorization)
    forged_bindings = dict(trusted)
    forged_bindings["EXECUTION_PROVENANCE_AUTHORITY"] = "f" * 64
    forged_chain = build_receipt_chain(
        owner_authorization=authorization,
        artifact_sha256_by_node=forged_bindings,
        prefix_length=8,
    )
    result = evaluate_graph(
        owner_authorization=authorization,
        receipts=forged_chain,
        trusted_artifact_sha256_by_node=trusted,
    )
    assert result["routing_decision"] == INVALID_CHAIN_ROUTE
    assert "caller_supplied_receipt_chain_not_authoritative" in result["blockers"]
    assert result["next_node"] != "STOP"


def test_tampered_parent_or_chain_hash_rejects() -> None:
    authorization = _authorization()
    chain, bindings = _chain(8, authorization=authorization)
    tampered = deepcopy(chain)
    tampered["receipts"][3]["parent_receipt_sha256"] = "f" * 64
    tampered = seal_receipt_chain(tampered)
    result = evaluate_graph(
        owner_authorization=authorization,
        receipts=tampered,
        trusted_artifact_sha256_by_node=bindings,
    )
    assert result["routing_decision"] == INVALID_CHAIN_ROUTE
    assert "caller_supplied_receipt_chain_not_authoritative" in result["blockers"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda authorization: authorization.__setitem__("extra", True),
        lambda authorization: authorization.__delitem__("goal_sha256"),
        lambda authorization: authorization.__setitem__(
            "goal_sha256", "A" * 64
        ),
        lambda authorization: authorization.__setitem__(
            "campaign_execution_id", " padded "
        ),
        lambda authorization: authorization.__setitem__(
            "seed_45_or_G9_authorized", True
        ),
    ],
)
def test_owner_schema_fails_closed(mutator) -> None:
    authorization = _authorization()
    mutator(authorization)
    result = evaluate_graph(owner_authorization=authorization)
    assert result["routing_decision"] == INVALID_CHAIN_ROUTE
    assert result["next_node"] != "STOP"


def test_graph_definitions_keep_canonical_order() -> None:
    assert tuple(item["node"] for item in NODE_DEFINITIONS[1:-1]) == RECEIPT_NODES


def test_graph_requires_execution_and_control_authorities_in_order(
    tmp_path,
) -> None:
    authorization = _authorization()
    journal = _journal(tmp_path, prefix_length=8, authorization=authorization)
    result = evaluate_graph(
        owner_authorization=authorization,
        controller_journal_path=journal,
        workspace_root=tmp_path,
    )
    assert result["routing_decision"] == STOP_ROUTE
    assert result["next_node"] == "STOP"
    assert result["completed_prefix"] == list(RECEIPT_NODES)


def test_graph_rejects_tampered_or_missing_authority_receipts() -> None:
    authorization = _authorization()
    chain, bindings = _chain(2, authorization=authorization)
    missing = evaluate_graph(
        owner_authorization=authorization,
        receipts=chain,
        trusted_artifact_sha256_by_node={
            "FRESH_420_UNIT_RUN": bindings["FRESH_420_UNIT_RUN"],
        },
    )
    assert missing["routing_decision"] == INVALID_CHAIN_ROUTE
    tampered = deepcopy(chain)
    tampered["receipts"][1]["artifact_sha256"] = "f" * 64
    tampered = seal_receipt_chain(tampered)
    result = evaluate_graph(
        owner_authorization=authorization,
        receipts=tampered,
        trusted_artifact_sha256_by_node=bindings,
    )
    assert result["routing_decision"] == INVALID_CHAIN_ROUTE
