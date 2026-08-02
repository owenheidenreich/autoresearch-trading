"""Focused pre-fit regressions for the corrected Path-D foundation generation."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES
from v4.research import pathd_entry_exit as foundation
from v4.research.pathd_feature_live_twin import (
    ENTRY17_FEATURE_NAMES,
    EXIT47_CORRECTED_FEATURE_NAMES,
    EXIT49_FEATURE_NAMES,
    DROP_UNTIL_EXACT_ADAPTER_RECEIPT,
    NO_INTRADAY_LIVE_TWIN,
    PRIOR_DAY_EOD_STATIC,
    live_twin_record,
)


def test_p1_forensic_contract_is_benign_zero_access_and_restores_five() -> None:
    burn = foundation._assert_superseded_foundation_history()
    correction = foundation.foundation_correction_spec()["p1_foundation_restoration"]

    assert burn["access_count"] == 0
    assert burn["access_receipt_sha256"] is None
    assert burn["dataset_receipt_sha256"] is None
    assert burn["result_sha256"] is None
    assert burn["transaction_id"] == "frozen-gate-transaction"
    assert correction["classification"] == "BENIGN_TEST_CONTAMINATION"
    assert correction["scientific_data_changed"] is False
    assert correction["restored_outer_folds"] == [1, 2, 3, 4, 5]
    assert correction["acceptance_rule_unchanged"] == (
        "pooled AND at least 4 of 5 chronological outer folds"
    )


def test_corrected_generation_binds_restoration_and_stability_receipts() -> None:
    contracts = foundation.receipt_contract_spec()

    assert contracts["foundation_restoration_receipt_path"] == (
        foundation.repo_path_label(foundation.FOUNDATION_RESTORATION_RECEIPT_PATH)
    )
    assert contracts["foundation_stability_receipt_path"] == (
        foundation.repo_path_label(foundation.FOUNDATION_STABILITY_RECEIPT_PATH)
    )
    assert foundation.AUDIT_ROOT != foundation.SUPERSEDED_AUDIT_ROOT
    assert foundation.AUDIT_ROOT != foundation.INTERMEDIATE_CORRECTED_AUDIT_ROOT


def test_v3_preserves_corrected_v2_bytes_and_remains_build_only() -> None:
    correction = foundation.foundation_correction_spec()
    assert foundation.AUDIT_ROOT != foundation.PREVIOUS_CORRECTED_V2_AUDIT_ROOT
    assert correction['previous_corrected_v2_audit_root'] == foundation.repo_path_label(
        foundation.PREVIOUS_CORRECTED_V2_AUDIT_ROOT
    )
    assert correction['previous_corrected_v2_foundation_files'] == [
        dict(row) for row in foundation.PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS
    ]
    assert correction['previous_corrected_v2_foundation_generation_sha256'] == (
        foundation.PREVIOUS_CORRECTED_V2_FOUNDATION_GENERATION_SHA256
    )
    for expected in foundation.PREVIOUS_CORRECTED_V2_FOUNDATION_FILE_BINDINGS:
        assert foundation._raw_regular_file_binding(expected['path']) == expected
    pause = correction['prefit_pause']
    assert pause['unresolved_owner_decision'] is None
    assert pause['resolved_owner_decisions'] == [
        'AREF_ACTION_CALIBRATION_ROLE',
        'COMPOSITE_CALIBRATION_TERMINAL_RULE',
    ]
    assert pause['claude_verification_pending'] is True
    assert pause['machinery_seal_authorized'] is False
    assert pause['foundation_stability_seal_authorized'] is False
    assert pause['model_fit_authorized'] is False
    assert pause['corpus_decode_authorized'] is False
    assert pause['nested_or_outer_evidence_open_authorized'] is False
    assert pause['protected_holdout_open_authorized'] is False
    payload, _assignments, _lineage = foundation.preregistration_payload()
    with pytest.raises(RuntimeError, match='STOP_FOR_CLAUDE_VERIFICATION'):
        foundation.assert_correction_prefit_release(payload)


def test_refined_br_requires_exact_five_valid_folds_and_rejects_rescue() -> None:
    spec = foundation.composite_calibration_terminal_rule_spec()
    assert spec['policy'] == 'B_R_STATUS_PRESERVING_WHOLE_RUN_HARD_STOP'
    assert spec['status_precedence'] == [
        'invalid_result', 'insufficient_evidence', 'owner_decision_required',
        'no_genuine_signal', 'PASS',
    ]
    assert spec['all_five_manifest']['outer_folds_in_exact_order'] == [1, 2, 3, 4, 5]
    assert spec['all_five_manifest']['valid_fold_count'] == 5
    assert spec['all_five_manifest']['four_as_five_or_survivor_pooling'] is False
    expected_skips = [
        {'outer_fold': 1, 'inner_fold': 1}, {'outer_fold': 1, 'inner_fold': 2},
        {'outer_fold': 1, 'inner_fold': 3}, {'outer_fold': 1, 'inner_fold': 4},
        {'outer_fold': 2, 'inner_fold': 1}, {'outer_fold': 2, 'inner_fold': 2},
        {'outer_fold': 2, 'inner_fold': 3}, {'outer_fold': 3, 'inner_fold': 1},
        {'outer_fold': 3, 'inner_fold': 2}, {'outer_fold': 4, 'inner_fold': 1},
        {'outer_fold': 5, 'inner_fold': 1},
    ]
    assert spec['nested_structural_skip']['exact_blocks'] == expected_skips
    assert spec['nested_structural_skip']['computed_failure_in_calibration_valid_block_can_be_relabelled_skip'] is False
    manifest = []
    for fold in range(1, 6):
        scope = f'OUTER_{fold}'
        nodes = foundation.entry_required_calibration_node_ids(scope)
        manifest.append({
            'outer_fold': fold,
            'scope': scope,
            'status': 'VALID',
            'required_node_count': len(nodes),
            'valid_node_count': len(nodes),
            'required_node_ids_sha256': foundation.stable_hash(list(nodes)),
            'scope_gate_receipt_sha256': f'{fold:x}' * 64,
            'outer_result_receipt_sha256': f'{fold + 5:x}' * 64,
            'deleted_node_count': 0,
            'imputed_node_count': 0,
        })
    foundation.validate_br_five_fold_manifest(manifest)
    invalid_manifests = []
    invalid_manifests.append(copy.deepcopy(manifest[:-1]))
    changed = copy.deepcopy(manifest); changed[2]['status'] = 'INSUFFICIENT_EVIDENCE'; invalid_manifests.append(changed)
    changed = copy.deepcopy(manifest); changed[1]['valid_node_count'] -= 1; invalid_manifests.append(changed)
    changed = copy.deepcopy(manifest); changed[1]['deleted_node_count'] = 1; invalid_manifests.append(changed)
    changed = copy.deepcopy(manifest); changed[1]['imputed_node_count'] = 1; invalid_manifests.append(changed)
    changed = copy.deepcopy(manifest); changed[1]['required_node_ids_sha256'] = '0' * 64; invalid_manifests.append(changed)
    for changed in invalid_manifests:
        with pytest.raises(ValueError, match='B-R'):
            foundation.validate_br_five_fold_manifest(changed)
    payload, assignments, lineage = foundation.preregistration_payload()
    changed_payload = copy.deepcopy(payload)
    changed_payload['composite_calibration_terminal_rule']['forbidden_rescue'].remove('zero imputation')
    with pytest.raises(ValueError):
        foundation.validate_preregistration_payload(changed_payload, assignments, lineage)


def test_refined_br_preserves_entry_verdict_on_exit_power_stop() -> None:
    spec = foundation.composite_calibration_terminal_rule_spec()
    stop = spec['current_run_exit_power_stop']
    assert stop == {
        'known_exit_weight_session_counts_by_fold': [0, 0, 19, 48, 56],
        'minimum_model_fit_sessions': 60,
        'terminal_status_after_entry_pass': 'insufficient_evidence',
        'stop_reason': 'insufficient_exit_evidence',
        'entry_only_verdict_preserved_by_hash': True,
        'exit_fit_executed': False,
        'four_box_instantiated': False,
        'full_exit_fit_executed': False,
        'preholdout_packet_frozen': False,
        'holdout_open_count': 0,
    }
    assert max(stop['known_exit_weight_session_counts_by_fold']) < stop['minimum_model_fit_sessions']
    assert spec['full_preholdout']['entry_instantiation'] == 'only after immutable outer entry PASS'
    assert spec['full_preholdout']['entry_only_pass_authorizes_combined_full_exit_or_holdout'] is False
    assert spec['failure_mapping']['INVALID_TARGET_COVERAGE'] == 'invalid_result'
    assert spec['failure_mapping']['INSUFFICIENT_EVIDENCE'] == 'insufficient_evidence'
    assert spec['same_generation_retry_after_invalid_result'] is False
    assert {'fold deletion', 'target or quantile deletion', 'row deletion after prediction',
            'null imputation', 'zero imputation', 'status relabeling',
            'survivor-only pooling', 'treating four valid folds as five'} <= set(spec['forbidden_rescue'])


def test_foundation_byte_root_changes_on_any_file_or_generation_change() -> None:
    files = [{"path": "a", "bytes": 1, "sha256": "a" * 64}]
    baseline = foundation._foundation_root_sha256(
        generation_sha256="b" * 64,
        files=files,
        fit_environment_sha256="c" * 64,
    )

    changed_file = copy.deepcopy(files)
    changed_file[0]["sha256"] = "d" * 64
    assert baseline != foundation._foundation_root_sha256(
        generation_sha256="b" * 64,
        files=changed_file,
        fit_environment_sha256="c" * 64,
    )
    assert baseline != foundation._foundation_root_sha256(
        generation_sha256="e" * 64,
        files=files,
        fit_environment_sha256="c" * 64,
    )


def test_stability_seal_requires_all_corrected_fold_namespaces_pristine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "entry_outer_folds"
    monkeypatch.setattr(foundation, "ENTRY_FOLD_ARTIFACT_ROOT", root)

    rows = foundation._pristine_corrected_fold_namespaces()
    assert [row["outer_fold"] for row in rows] == [1, 2, 3, 4, 5]
    assert {row["state"] for row in rows} == {"PRISTINE_ABSENT"}

    occupied = root / "fold_1"
    occupied.mkdir(parents=True)
    (occupied / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not pristine"):
        foundation._pristine_corrected_fold_namespaces()


def test_p2_live_twin_inventory_keeps_signed17_and_corrects_exit47() -> None:
    exit_contract = foundation.exit_feature_spec()
    lineage = foundation.feature_lineage()["live_twin_inventory"]
    open_interest = live_twin_record("exit", "last_causal_open_interest")
    minute_volume = live_twin_record("exit", "last_causal_minute_volume")

    assert tuple(FEATURE_NAMES) == ENTRY17_FEATURE_NAMES
    assert exit_contract["feature_names_in_exact_order"] == list(
        EXIT47_CORRECTED_FEATURE_NAMES
    )
    assert exit_contract["feature_count"] == 47
    assert exit_contract["original_exit49_audit_binding"][
        "feature_names_in_exact_order"
    ] == list(EXIT49_FEATURE_NAMES)
    assert open_interest.live_twin_class == NO_INTRADAY_LIVE_TWIN
    assert open_interest.prior_day_static_class == PRIOR_DAY_EOD_STATIC
    assert "last_causal_open_interest" not in EXIT47_CORRECTED_FEATURE_NAMES
    assert "last_causal_minute_volume" not in EXIT47_CORRECTED_FEATURE_NAMES
    assert minute_volume.recommended_action == DROP_UNTIL_EXACT_ADAPTER_RECEIPT
    assert lineage["minute_volume_current_run_disposition"] == (
        "DROPPED_UNTIL_EXACT_SHARED_ADAPTER_RECEIPT"
    )
    assert lineage["unresolved_live_twin_allowed_in_current_alpha"] is False
    assert foundation.require_all_negative_fixtures_rejected()[
        "feature_without_live_twin"
    ] is True


def test_fold_stage_preflight_stops_before_dispatch_on_foundation_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from argparse import Namespace
    from v4.scripts import run_pathd_entry_exit_research as runner

    fold_root = tmp_path / "entry_outer_folds"
    called: list[int] = []
    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: Namespace(stage="run-entry-fold", outer_fold=1, inner_fold=None),
    )
    monkeypatch.setattr(runner.prereg, "ENTRY_FOLD_ARTIFACT_ROOT", fold_root)

    def mismatch() -> None:
        raise RuntimeError("synthetic global foundation mismatch")

    monkeypatch.setattr(
        runner.prereg, "assert_research_foundation_stable", mismatch
    )
    monkeypatch.setattr(
        runner,
        "run_canonical_entry_fold",
        lambda *, outer_fold: called.append(outer_fold),
    )

    with pytest.raises(RuntimeError, match="synthetic global foundation mismatch"):
        runner.main()

    assert called == []
    assert not fold_root.exists()


def test_p3_widen_entry_lead_is_forward_only() -> None:
    correction = foundation.foundation_correction_spec()
    note = correction["p3_forward_only_entry_enrichment"]

    assert note["current_entry_baseline"] == "EXACTLY_SIGNED_17_UNCHANGED"
    assert note["candidate"] == "size_imbalance"
    assert note["optional_companions"] == ["bid_size", "ask_size"]
    assert note["current_run_alpha_allowed"] is False
    assert correction["prefit_pause"] == {
        "terminal_marker": "STOP_FOR_CLAUDE_VERIFICATION",
        "foundation_stability_gate_implemented": True,
        "foundation_stability_receipt_sealed": False,
        "machinery_seal_authorized": False,
        "foundation_stability_seal_authorized": False,
        "model_fit_authorized": False,
        "corpus_decode_authorized": False,
        "nested_or_outer_evidence_open_authorized": False,
        "protected_holdout_open_authorized": False,
        "unresolved_owner_decision": None,
        "resolved_owner_decisions": [
            "AREF_ACTION_CALIBRATION_ROLE",
            "COMPOSITE_CALIBRATION_TERMINAL_RULE",
        ],
        "claude_verification_pending": True,
        "separate_post_verification_release_required": True,
    }
    payload, _assignments, _lineage = foundation.preregistration_payload()
    with pytest.raises(RuntimeError, match="STOP_FOR_CLAUDE_VERIFICATION"):
        foundation.assert_correction_prefit_release(payload)


def _transition(
    *,
    schema: str,
    prior_authorization: str | None,
    next_authorization: str,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "event_schema_version": schema,
        "prior_authorization_sha256": prior_authorization,
        "next_authorization_sha256": next_authorization,
        "event_payload": payload,
    }


def test_nested_dataset_membership_uses_only_current_authorization_segment() -> None:
    prior = "a" * 64
    current = "b" * 64
    old_example = "c" * 64
    current_example = "d" * 64
    control = "pathd.research_ledger_control_event.v1"
    coverage = "pathd.entry_frame_coverage.v1"
    decision = "pathd.entry_action_decision.v1"
    observation = "pathd.entry_action_observation.v1"
    transitions = [
        _transition(
            schema=control,
            prior_authorization=None,
            next_authorization=prior,
            payload={"event_kind": "GENESIS"},
        ),
        _transition(
            schema=coverage,
            prior_authorization=prior,
            next_authorization=prior,
            payload={
                "example_sha256": old_example,
                "disposition": "STATE_INELIGIBLE",
            },
        ),
        _transition(
            schema=control,
            prior_authorization=prior,
            next_authorization=current,
            payload={"event_kind": "AUTHORIZATION_HANDOFF"},
        ),
        _transition(
            schema=coverage,
            prior_authorization=current,
            next_authorization=current,
            payload={
                "example_sha256": current_example,
                "disposition": "ACTION_DECISION",
            },
        ),
        _transition(
            schema=decision,
            prior_authorization=current,
            next_authorization=current,
            payload={"example_sha256": current_example},
        ),
        _transition(
            schema=observation,
            prior_authorization=current,
            next_authorization=current,
            payload={"example_sha256": current_example},
        ),
    ]
    value = {
        "authorization_sha256": current,
        "terminal_journal": {"transitions": transitions},
    }
    dataset = {
        "ordered_example_hashes": [current_example],
        "session_example_hashes": [
            {
                "session": "2026-01-02",
                "ordered_example_hashes": [current_example],
            }
        ],
        "sessions": ["2026-01-02"],
    }

    foundation._validate_policy_evaluation_dataset_membership(value, dataset)

    forged = copy.deepcopy(value)
    forged["terminal_journal"]["transitions"][3]["event_payload"][
        "example_sha256"
    ] = old_example
    with pytest.raises(RuntimeError, match="unsealed example"):
        foundation._validate_policy_evaluation_dataset_membership(forged, dataset)
