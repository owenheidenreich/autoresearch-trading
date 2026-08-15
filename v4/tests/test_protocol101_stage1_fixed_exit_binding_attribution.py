from __future__ import annotations

from pathlib import Path

import pytest

from v4.scripts import (
    run_protocol101_stage1_fixed_exit_binding_attribution as attribution,
)


def test_preregistration_freezes_strict_stage2_route() -> None:
    payload = attribution.preregistration()
    assert payload["input_route_required"] == "real_signal_attribution_required"
    stage2_rule = payload["routing_law"]["stage2_design_may_be_drafted"]
    assert "no single fixed policy repairs" in stage2_rule
    assert "G1 and repairs G4 and G7" in stage2_rule
    assert payload["path_evidence"]["minimum_coverage"] == 0.95
    assert payload["counterfactuals"][
        "minimum_all_seven_label_lookup_coverage"
    ] == 0.99
    assert payload["path_evidence"]["minimum_source_policy_losers"] == 30
    assert payload["preregistration_hash"] == attribution.stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "preregistration_hash"
        }
    )


def test_write_preregistration_is_immutable(tmp_path: Path) -> None:
    first = attribution.write_preregistration(tmp_path / "design")
    before = (tmp_path / "design" / "preregistration.json").read_bytes()
    second = attribution.write_preregistration(tmp_path / "design")
    assert first == second
    assert (tmp_path / "design" / "preregistration.json").read_bytes() == before


def test_rank_auc_is_orientation_free() -> None:
    assert attribution.rank_auc([0.0, 1.0, 2.0, 3.0], [True, True, False, False]) == 1.0
    assert attribution.rank_auc([0.0, 1.0, 2.0, 3.0], [False, False, True, True]) == 1.0
    assert attribution.rank_auc([1.0, 1.0], [False, True]) == 0.5


def test_deserialize_executed_trade_uses_frozen_policy_cooldown() -> None:
    payload = {
        "split": "validation",
        "session": "2025-03-12",
        "decision_time": "2025-03-12T13:33:00+00:00",
        "contract_id": "SPXW-20250312-05645.000-P",
        "right": "P",
        "offset": 10.0,
        "entry_ask": 34.6,
        "score": -0.18,
        "raw_label_pnl": 687.0,
    }
    candidate = attribution.deserialize_intent(
        payload,
        default_cooldown_minutes=384.0,
    )
    assert candidate.cooldown_minutes == 384.0
    assert candidate.max_hold_minutes == 384.0

    with pytest.raises(KeyError, match="cooldown_minutes"):
        attribution.deserialize_intent(payload)
