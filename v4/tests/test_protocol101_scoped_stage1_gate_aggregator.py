from __future__ import annotations

from v4.scripts.run_protocol101_full_trader_stage1_gate_audit_selection_validation import (
    _aggregate,
    _reference,
    _replace_unit_economics,
    _reseal,
    build_synthetic_campaign,
)
def _row(result: dict, row_id: str) -> dict:
    return next(row for row in result["rows"] if row["row_id"] == row_id)


def test_aggregates_all_420_units_and_reports_high_G8_without_gating() -> None:
    result = _aggregate(build_synthetic_campaign(ece=0.20))
    row = _row(result, "H0/P0")
    assert result["valid"] is True
    assert result["unit_count"] == 420
    assert result["row_count"] == 28
    assert row["gates"]["G8"] is False
    assert row["G8_role"] == "report_only"
    assert row["hard_gate_eligible"] is True
    assert result["G9_executed"] is False


def test_G6_only_failure_is_distinct_and_signal_is_preserved() -> None:
    packet = build_synthetic_campaign()
    _replace_unit_economics(
        packet,
        hypothesis="H0",
        policy="P0",
        seed=42,
        fold=5,
        pnl_sequence=[-10.0],
    )
    row = _row(_aggregate(packet), "H0/P0")
    assert row["gates"]["G6"] is False
    assert row["G6_only_blocked"] is True
    assert row["multiplicity_adjusted_real_entry_signal"] is True
    assert row["hard_gate_eligible"] is False


def test_fixed_exit_failure_routes_as_adjusted_signal_not_eligibility() -> None:
    packet = build_synthetic_campaign()
    _reference(packet, "H0/P0")["heuristic_pooled_pnl"] = 10_000.0
    _reseal(packet)
    row = _row(_aggregate(packet), "H0/P0")
    assert row["gates"]["G3"] is False
    assert row["multiplicity_adjusted_real_entry_signal"] is True
    assert row["hard_gate_eligible"] is False


def test_D1_and_invalid_maxT_block_all_rows() -> None:
    d1 = build_synthetic_campaign()
    d1["controls"]["D1"].update(
        {"joint_G1_G2_pass_count": 2, "passes": False}
    )
    _reseal(d1)
    d1_result = _aggregate(d1)
    assert d1_result["global_controls"]["D1"] is False
    assert not any(row["hard_gate_eligible"] for row in d1_result["rows"])
    invalid = build_synthetic_campaign()
    invalid["controls"]["maxT"]["valid"] = False
    _reseal(invalid)
    assert _aggregate(invalid)["valid"] is False
