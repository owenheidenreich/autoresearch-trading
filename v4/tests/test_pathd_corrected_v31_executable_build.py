from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from v4.research import pathd_entry_exit as foundation
from v4.research import pathd_entry_models as models
from v4.scripts import run_pathd_entry_exit_research as runner


HEX_A = "a" * 64
HEX_B = "b" * 64


def _br_rows() -> list[dict[str, object]]:
    rows = []
    for fold in range(1, 6):
        scope = f"OUTER_{fold}"
        nodes = foundation.entry_required_calibration_node_ids(scope)
        rows.append(
            {
                "outer_fold": fold,
                "scope": scope,
                "status": "VALID",
                "required_node_count": len(nodes),
                "valid_node_count": len(nodes),
                "required_node_ids_sha256": foundation.stable_hash(list(nodes)),
                "scope_gate_receipt_sha256": HEX_A,
                "outer_result_receipt_sha256": HEX_B,
                "deleted_node_count": 0,
                "imputed_node_count": 0,
            }
        )
    return rows


def _candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "outer_fold": 1,
            "session": "2026-01-02",
            "call_or_put": right,
            "ATM_NEAR_WING": money,
            "decision_time_premium_band": band,
            "action": "ENTER",
            "decision_sha256": f"{index + 1:064x}",
        }
        for index, (right, money, band) in enumerate(
            (
                ("C", "ATM", "small_1_3"),
                ("C", "ATM", "small_1_3"),
                ("P", "WING", "large_8_20"),
            )
        )
    ]


def _calibration_payload(*, wait_status: str = "VALID") -> dict[str, object]:
    heads = []
    for head in foundation.ENTRY_HEAD_TARGETS:
        semantic = {
            "head": head,
            "row_count": 100,
            "session_count": 12,
            "population_sha256": HEX_A,
            "mean_lcb_seed_receipt": {},
            "mean_lcb_correction": 0.0,
            "q10_correction": 0.0,
        }
        heads.append(
            {**semantic, "receipt_sha256": foundation.stable_hash(semantic)}
        )
    composites = {}
    for action, status, rows in (
        ("enter", "VALID", 20),
        ("wait", wait_status, 40 if wait_status == "VALID" else 0),
    ):
        semantic = {
            "status": status,
            "correction": 0.0 if status == "VALID" else None,
            "row_count": rows,
            "session_count": 12,
            "missing_outcome_count": 0,
        }
        receipt = (
            {**semantic, "receipt_sha256": foundation.stable_hash(semantic)}
            if status == "VALID"
            else semantic
        )
        composites[f"{action}_composite_q10_status"] = status
        composites[f"{action}_composite_calibration_receipt"] = receipt
    return {"head_calibration_receipts": heads, **composites}


def _proposals(quotas: dict[tuple[object, ...], int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for cell, count in quotas.items():
        for _ in range(count + 2):
            index += 1
            rows.append(
                {
                    "id": f"proposal-{index}",
                    "cell": cell,
                    "session": str(cell[1]),
                    "decision_time_ns": 1_000_000_000 * index,
                    "source_neutral_contract_id": f"contract-{index}",
                    "expiry_yyyymmdd": 20260102,
                    "strike_milli_points": 6_000_000 + index,
                    "right_code": str(cell[2]),
                }
            )
    return rows


def test_static_generation_binding_and_campaign_order_are_exact() -> None:
    release = runner.assert_corrected_v31_authorization_release()
    assert release["preregistration_sha256"] == runner.V31_PREREGISTRATION_SHA256
    assert release["holdout_open_count"] == 0
    assert runner.canonical_entry_campaign_stage_order() == runner.ENTRY_CAMPAIGN_STAGE_ORDER
    assert runner.ENTRY_CAMPAIGN_STAGE_ORDER[4:9] == tuple(
        f"OUTER_FOLD_{fold}" for fold in range(1, 6)
    )
    generation = runner._corrected_v31_executable_generation_semantic()
    assert generation["science_authorization_file_sha256s"] == (
        runner.V31_AUTHORIZATION_FILE_SHA256S
    )
    assert generation["hard_stops_at_generation_freeze"]["model_or_weight_fit"] is False


def test_br_is_first_and_terminal_precedence_is_fail_closed() -> None:
    scope = "NESTED_OUTER_1_INNER_1"
    passed = runner.evaluate_entry_calibration_scope_gate(
        scope=scope,
        calibration_bundles={
            "HGB": {"payload": _calibration_payload()},
            "NEURAL": {"payload": _calibration_payload()},
        },
    )
    assert passed["status"] == "VALID"
    assert passed["required_node_count"] == 84
    assert passed["failure_node_ids"] == []

    failed = runner.evaluate_entry_calibration_scope_gate(
        scope=scope,
        calibration_bundles={
            "HGB": {"payload": _calibration_payload(wait_status="INSUFFICIENT_EVIDENCE")},
            "NEURAL": {"payload": _calibration_payload()},
        },
    )
    assert failed["status"] == "FAILED_CLOSED"
    assert failed["failure_node_ids"] == [
        "ENTRY::NESTED_OUTER_1_INNER_1::HGB::ACTION_COMPOSITE::WAIT"
    ]

    rows = _br_rows()
    foundation.validate_br_five_fold_manifest(rows)
    for mutation in ("drop", "delete", "impute", "survivor"):
        broken = [dict(row) for row in rows]
        if mutation in {"drop", "survivor"}:
            broken.pop()
        elif mutation == "delete":
            broken[2]["deleted_node_count"] = 1
        else:
            broken[2]["imputed_node_count"] = 1
        with pytest.raises(ValueError, match="B-R"):
            foundation.validate_br_five_fold_manifest(broken)


def test_matched_random_budget_and_all_eight_schedules_are_outcome_blind() -> None:
    budget = models.derive_matched_random_candidate_budget(
        outer_fold=1,
        owner_policy_id="HGB",
        channel="SHARED_TRANSPARENT_FEE3",
        fee_path=3,
        candidate_buy_intents=_candidate_rows(),
    )
    quotas = models.matched_random_budget_quotas(budget)
    assert sum(quotas.values()) == 3
    proposals = _proposals(quotas)
    schedules = tuple(
        models.build_matched_random_schedule(
            proposals=proposals,
            quotas=quotas,
            seed=seed,
            policy_id=f"MATCHED_RANDOM_{index:02d}",
            outer_fold=1,
        )
        for index, seed in enumerate(foundation.MATCHED_RANDOM_SEEDS, 1)
    )
    assert len(schedules) == 8
    assert len({row["schedule_sha256"] for row in schedules}) == 8
    for row in schedules:
        realized = models.realize_frozen_matched_random_schedule(
            row,
            proposal_population=proposals,
            outcomes={proposal["id"]: "NO_FILL" for proposal in proposals},
        )
        assert realized["filled_ids"] == []
        assert realized["substituted_or_backfilled_ids"] == []
        assert realized["attempted_ids"] == row["ordered_ids"]

    missing_classification = _candidate_rows()
    missing_classification[0].pop("ATM_NEAR_WING")
    with pytest.raises(ValueError, match="schema|matching field"):
        models.derive_matched_random_candidate_budget(
            outer_fold=1,
            owner_policy_id="HGB",
            channel="SHARED_TRANSPARENT_FEE3",
            fee_path=3,
            candidate_buy_intents=missing_classification,
        )


def test_shared_control_exit_requires_complete_panel_and_selects_real_economics() -> None:
    sessions = tuple(foundation.session_assignments()["folds"][0]["model_fit"])
    rows = []
    for index, policy_id in enumerate(models.entry_control_exit_policy_ids()):
        pnl = tuple([index] * len(sessions))
        semantic = {
            "schema_version": models.EntryControlExitCandidateEvaluationV1.SCHEMA_VERSION,
            "outer_fold": 1,
            "policy_id": policy_id,
            "model_fit_sessions_sha256_newline": foundation.canonical_session_hash(sessions),
            "session_pnl_micros": pnl,
            "session_terminal_journal_sha256s": tuple(f"{index + 1:064x}" for _ in sessions),
            "valid_session_count": len(sessions),
            "total_net_pnl_micros": sum(pnl),
        }
        rows.append(
            models.EntryControlExitCandidateEvaluationV1(
                **semantic,
                evaluation_sha256=foundation.stable_hash(semantic),
            )
        )
    selected = models.select_entry_control_exit(outer_fold=1, evaluations=rows)
    assert selected.selected_policy_id == models.entry_control_exit_policy_ids()[-1]
    with pytest.raises(ValueError, match="incomplete"):
        models.select_entry_control_exit(outer_fold=1, evaluations=rows[:-1])


def test_negative_controls_fail_closed_and_exit_geometry_abstains() -> None:
    identities = tuple(
        {
            "outer_fold": 1,
            "session": "2026-01-02",
            "decision_time_ns": index,
            "source_neutral_contract_id": f"C{index}",
            "expiry_yyyymmdd": 20260102,
            "strike_milli_points": 6_000_000 + index,
            "right_code": "C" if index % 2 == 0 else "P",
        }
        for index in range(4)
    )
    targets = tuple({"mean": [float(index)], "q10": [-float(index)]} for index in range(4))
    validity = tuple({"mean": [True], "q10": [True]} for _ in range(4))
    histories = tuple({"current": [index], "lag30": [index - 30]} for index in range(4))
    for control in foundation.ENTRY_NEGATIVE_CONTROL_IDS:
        seed_index = 0 if not control.startswith("SHUFFLED_TARGET_") else int(control[-2:]) - 1
        value = models.transform_entry_negative_control_bundle(
            control_id=control,
            target_bundles=targets,
            target_validity=validity,
            feature_histories=histories,
            row_identities=identities,
            seed=foundation.MATCHED_RANDOM_SEEDS[seed_index],
        )
        assert value["control_id"] == control
        assert value["transform_sha256"] == foundation.stable_hash(
            {key: item for key, item in value.items() if key != "transform_sha256"}
        )
    with pytest.raises(ValueError):
        models.transform_entry_negative_control_bundle(
            control_id="SHUFFLED_TARGET_09",
            target_bundles=targets,
            target_validity=validity,
            feature_histories=histories,
            row_identities=identities,
            seed=foundation.MATCHED_RANDOM_SEEDS[0],
        )

    exit_spec = foundation.exit_action_composer_spec()
    assert exit_spec["action_rule"] == "HOLD iff U_hold>0; otherwise request EXIT"
    assert exit_spec["action_inputs"] == ["A_ref_mean", "A_ref_q10"]
    assert exit_spec["exit_reliability_only_support"] == ["A_ref_q90"]
    hold = models.compose_exit_action_from_calibrated_aref(
        mean_lcb_aref=2.0, q10_aref=-1.0, q50_aref=0.0, q90_aref=2.0
    )
    assert hold["action"] == "HOLD"
    exit_action = models.compose_exit_action_from_calibrated_aref(
        mean_lcb_aref=0.1, q10_aref=-1.0, q50_aref=0.0, q90_aref=2.0
    )
    assert exit_action["action"] == "EXIT"
    abstain = models.compose_exit_action_from_calibrated_aref(
        mean_lcb_aref=1.0, q10_aref=2.0, q50_aref=1.0, q90_aref=3.0
    )
    assert abstain["status"] == "invalid_result"
    assert abstain["action"] is None


def test_build_does_not_create_any_real_execution_namespace() -> None:
    assert foundation.PROTECTED_HOLDOUT_ROOT.parent == foundation.AUDIT_ROOT
    for fold in range(1, 6):
        assert not Path(foundation.ENTRY_FOLD_ARTIFACT_ROOT / f"fold_{fold}").exists()
