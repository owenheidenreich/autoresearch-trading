from __future__ import annotations

from copy import deepcopy

from v4.scripts.run_protocol101_full_trader_stage1_gate_audit_selection_validation import (
    _pipeline,
    _reference,
    _replace_unit_economics,
    _reseal,
    build_synthetic_campaign,
)
from v4.scripts.run_protocol101_stage1_cross_hypothesis_selection import (
    ATTRIBUTION_ROUTE,
    INVALID_ROUTE,
    REGIME_ROUTE,
    SELECTED_ROUTE,
    STOP_ROUTE,
    select_candidate,
)


def test_plain_pnl_ranking_and_H_P_tie_break() -> None:
    larger = build_synthetic_campaign(
        row_totals={"H0/P0": 500.0, "H3/P6": 900.0},
        maxT_pass_rows={"H0/P0", "H3/P6"},
    )
    assert _pipeline(larger)[2]["selected_candidate"]["row_id"] == "H3/P6"
    tied = build_synthetic_campaign(
        row_totals={"H0/P0": 700.0, "H0/P1": 700.0},
        maxT_pass_rows={"H0/P0", "H0/P1"},
    )
    assert _pipeline(tied)[2]["selected_candidate"]["row_id"] == "H0/P0"


def test_selection_routes_are_exact() -> None:
    selected = _pipeline(build_synthetic_campaign())[2]
    assert selected["routing_decision"] == SELECTED_ROUTE
    attribution = build_synthetic_campaign()
    _reference(attribution, "H0/P0")["heuristic_pooled_pnl"] = 10_000.0
    _reseal(attribution)
    assert _pipeline(attribution)[2]["routing_decision"] == ATTRIBUTION_ROUTE
    regime = build_synthetic_campaign()
    _replace_unit_economics(
        regime,
        hypothesis="H0",
        policy="P0",
        seed=42,
        fold=5,
        pnl_sequence=[-10.0],
    )
    assert _pipeline(regime)[2]["routing_decision"] == REGIME_ROUTE
    stop = build_synthetic_campaign()
    _reference(stop, "H0/P0")["matched_null_z_by_seed"] = {
        "42": 1.0,
        "43": 1.0,
        "44": 1.0,
    }
    _reseal(stop)
    assert _pipeline(stop)[2]["routing_decision"] == STOP_ROUTE


def test_invalid_campaign_cannot_select() -> None:
    packet = build_synthetic_campaign()
    packet["controls"]["maxT"]["valid"] = False
    _reseal(packet)
    selection = _pipeline(packet)[2]
    assert selection["routing_decision"] == INVALID_ROUTE
    assert selection["selected_candidate"] is None


def test_G8_is_never_a_rank_input() -> None:
    high = _pipeline(build_synthetic_campaign(ece=0.20))[2]
    low = _pipeline(build_synthetic_campaign(ece=0.05))[2]
    assert high["selected_candidate"]["row_id"] == low["selected_candidate"]["row_id"]
    assert high["G8_used_for_eligibility_or_ranking"] is False
    assert high["selection_rule"]["primary"].startswith("descending_median_seed")


def test_selector_rejects_changed_payload_and_malformed_freeze_hash() -> None:
    audit = _pipeline(build_synthetic_campaign())[1]
    changed = deepcopy(audit)
    changed["independent_result"]["rows"][0][
        "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
    ] += 1.0
    assert select_candidate(changed)["routing_decision"] == INVALID_ROUTE
    malformed = deepcopy(audit)
    malformed["freeze_sha256"] = "not-a-sha256"
    assert select_candidate(malformed)["routing_decision"] == INVALID_ROUTE


def test_selector_rejects_partial_or_extra_freeze_bindings() -> None:
    audit = _pipeline(build_synthetic_campaign())[1]
    partial = deepcopy(audit)
    partial["freeze"].pop("row_result_sha256")
    partial["freeze_sha256"] = "a" * 64
    assert select_candidate(partial)["routing_decision"] == INVALID_ROUTE
    extra = deepcopy(audit)
    extra["freeze"]["producer_only_summary"] = True
    extra["freeze_sha256"] = "b" * 64
    assert select_candidate(extra)["routing_decision"] == INVALID_ROUTE
