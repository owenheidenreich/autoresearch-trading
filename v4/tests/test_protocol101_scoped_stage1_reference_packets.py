from __future__ import annotations

from v4.scripts.run_protocol101_scoped_stage1_reference_packets import (
    Candidate,
    FRESH_REFERENCE_NAMESPACE_PREFIX,
    LEGACY_COMPATIBILITY_MODE,
    Minute,
    preregistration,
    serial_candidates,
)


class Scope:
    fold_governance_hash = "fold"
    acceptance_registry_hash = "registry"
    folds = [
        {
            "fold_id": "fold_01",
            "validation_sessions": ["2025-01-02"],
        }
    ]


def test_reference_preregistration_freezes_contract_and_daily_breaker() -> None:
    payload = preregistration(Scope(), 200)
    assert payload["contract_id"] == "protocol101-scoped-canonical-stage1-v1"
    assert payload["random_null"]["draws"] == 200
    assert payload["daily_loss_fraction_of_session_start_equity"] == 0.05
    assert payload["results_inspected_before_preregistration"] is False
    assert LEGACY_COMPATIBILITY_MODE == "legacy-v4-compatibility"
    assert FRESH_REFERENCE_NAMESPACE_PREFIX.endswith("reference_v5_")


def test_vwap_heuristic_chooses_nearest_atm_on_indicated_side() -> None:
    import pandas as pd

    minute = Minute(
        session="2025-01-02",
        decision_time=pd.Timestamp("2025-01-02T14:32:00Z"),
        vwap_side="P",
        candidates=(
            Candidate("far-put", "P", -20.0, 1.0, (10.0,) * 7),
            Candidate("near-put", "P", -5.0, 1.0, (20.0,) * 7),
            Candidate("call", "C", 0.0, 1.0, (30.0,) * 7),
        ),
    )
    selected = serial_candidates([minute], policy_idx=0, fee=3.0, heuristic=True)
    assert selected[0].contract_id == "near-put"
    assert selected[0].raw_label_pnl == 17.0
