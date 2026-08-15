from __future__ import annotations

from v4.scripts import run_protocol101_scoped_stage1_readiness as readiness


def test_preflight_reconciles_signed_scope_without_blockers() -> None:
    summary = readiness.build_summary()
    assert summary["contract_id"] == "protocol101-scoped-canonical-stage1-v1"
    assert len(summary["feature_names"]) == 17
    assert summary["training_scope"]["registry_sessions"] == 301
    assert summary["training_scope"]["fold_eligible_unique_sessions"] == 271
    assert summary["training_scope"]["protected_holdout_sessions_present"] == []
    assert summary["global_all_feature_synchronization_claimed"] is False
    assert summary["side_effects"]["model_training_executed"] is False
    assert summary["status"] == "ready_for_plumbing_smoke"
    assert summary["blockers"] == []
    assert all(summary["signatures"].values())


def test_reference_packet_validator_accepts_exact_v4_contract() -> None:
    scope = type(
        "Scope",
        (),
        {
            "fold_governance_hash": "fold-hash",
            "acceptance_registry_hash": "registry-hash",
            "sessions": tuple((str(index), None) for index in range(271)),
        },
    )()
    common = {
        "status": "pass",
        "contract_id": readiness.CONTRACT_ID,
        "fold_governance_hash": "fold-hash",
        "acceptance_registry_hash": "registry-hash",
        "fold_eligible_sessions": 271,
        "preregistration_hash": "prereg-hash",
        "side_effects": {
            "model_training_executed": False,
            "broker_endpoint_called": False,
        },
    }
    heuristic = {
        **common,
        "g3_fixed_baseline": {
            "folds": [
                {
                    "min_equity": 5_000.0,
                    "simulator_semantics": {
                        "simulator_version": readiness.PROTOCOL101_SERIAL_SIMULATOR_VERSION,
                        "affordability_reserve_per_trade": 3.0,
                    },
                }
                for _ in range(5)
            ]
        },
    }

    assert readiness.reference_packet_blockers(
        scope=scope,
        null_summary=dict(common),
        heuristic_summary=heuristic,
    ) == []


def test_reference_packet_validator_rejects_stale_simulator() -> None:
    scope = type(
        "Scope",
        (),
        {
            "fold_governance_hash": "fold-hash",
            "acceptance_registry_hash": "registry-hash",
            "sessions": (("2026-01-02", None),),
        },
    )()
    common = {
        "status": "pass",
        "contract_id": readiness.CONTRACT_ID,
        "fold_governance_hash": "fold-hash",
        "acceptance_registry_hash": "registry-hash",
        "fold_eligible_sessions": 1,
        "preregistration_hash": "same",
        "side_effects": {},
    }
    heuristic = {
        **common,
        "g3_fixed_baseline": {
            "folds": [
                {
                    "min_equity": -2.0,
                    "simulator_semantics": {
                        "simulator_version": "protocol101_serial_simulator_v2",
                        "affordability_reserve_per_trade": 0.0,
                    },
                }
                for _ in range(5)
            ]
        },
    }

    blockers = readiness.reference_packet_blockers(
        scope=scope,
        null_summary=dict(common),
        heuristic_summary=heuristic,
    )
    assert "exact_contract_heuristic_simulator_version_mismatch" in blockers
    assert "exact_contract_heuristic_fee_reserve_mismatch" in blockers
    assert "exact_contract_heuristic_negative_equity" in blockers
