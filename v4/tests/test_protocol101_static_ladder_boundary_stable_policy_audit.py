from __future__ import annotations

from v4.scripts import run_protocol101_static_ladder_boundary_stable_policy_audit as audit


MARGINS = {
    "min_mid": 0.5,
    "max_mid": 35.0,
    "max_spread_abs": 0.5,
    "max_spread_frac": 0.25,
    "max_quote_age_ms": 90_000.0,
    "min_bid_size": 1,
    "min_ask_size": 1,
    "stable_min_mid": 0.55,
    "stable_max_mid": 34.75,
    "stable_max_spread_abs": 0.45,
    "stable_max_spread_frac": 0.225,
    "stable_max_quote_age_ms": 85_000.0,
    "stable_min_bid_size": 2,
    "stable_min_ask_size": 2,
    "max_affordability_utilization": 0.975,
}


def _slot(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "contract_id": "SPXW-20260701-05000.000-C",
        "right": "C",
        "strike": 5000.0,
        "offset": 0.0,
        "strike_idx": 10,
        "right_idx": 0,
        "atm_strike": 5000,
        "strike_step": 5,
        "spx_for_ladder": 5001.25,
        "source_context_ts": "2026-07-01T13:31:00+00:00",
        "bid": 9.95,
        "ask": 10.05,
        "mid": 10.0,
        "spread": 0.10,
        "spread_frac": 0.01,
        "bid_size": 3,
        "ask_size": 4,
        "quote_age_ms": 1_000.0,
        "post_filter_candidate": True,
        "candidate_filter": {
            "passed": True,
            "tradability_pass": True,
            "freshness_pass": True,
            "reasons": [],
            "observed": {
                "bid": 9.95,
                "ask": 10.05,
                "mid": 10.0,
                "spread": 0.10,
                "spread_frac": 0.01,
                "bid_size": 3,
                "ask_size": 4,
                "quote_age_ms": 1_000.0,
            },
        },
    }
    item.update(overrides)
    return item


def test_static_slot_identity_ignores_quote_and_guard_fields() -> None:
    first = _slot(bid=9.95, ask=10.05, mid=10.0, post_filter_candidate=True)
    second = _slot(bid=8.0, ask=12.0, mid=10.0, post_filter_candidate=False)

    assert audit.static_slot_identity(first) == audit.static_slot_identity(second)
    assert audit.static_slot_key(first) == audit.static_slot_key(second)


def test_guard_status_preserves_boundary_states() -> None:
    assert audit.guard_status(_slot(), MARGINS, cash=10_000.0)["status"] == "tradable_boundary_stable"

    unstable = _slot(
        bid=34.75,
        ask=35.15,
        mid=34.95,
        spread=0.40,
        spread_frac=0.0115,
        candidate_filter={"passed": True, "observed": {}},
    )
    assert audit.guard_status(unstable, MARGINS, cash=10_000.0)["status"] == "tradable_unstable"

    wide = _slot(
        bid=9.5,
        ask=10.2,
        mid=9.85,
        spread=0.70,
        spread_frac=0.071,
        candidate_filter={"passed": False, "observed": {}},
    )
    assert audit.guard_status(wide, MARGINS, cash=10_000.0)["status"] == "untradable"

    stale = _slot(quote_age_ms=95_000.0, candidate_filter={"passed": False, "observed": {}})
    assert audit.guard_status(stale, MARGINS, cash=10_000.0)["status"] == "stale"

    missing = _slot(ask=None, candidate_filter={"passed": False, "observed": {"bid": 9.95}})
    assert audit.guard_status(missing, MARGINS, cash=10_000.0)["status"] == "quote_missing"


def test_static_geometry_uses_full_ladder_slots() -> None:
    definition = audit.g1.load_json(audit.DEFAULT_GROUP2_PREREGISTRATION.parent / "feature_group_definition.json")
    slots = [
        _slot(contract_id="SPXW-20260701-04995.000-C", right="C", offset=-5.0, strike=4995.0, strike_idx=9),
        _slot(contract_id="SPXW-20260701-05000.000-C", right="C", offset=0.0, strike=5000.0, strike_idx=10),
        _slot(contract_id="SPXW-20260701-05005.000-C", right="C", offset=5.0, strike=5005.0, strike_idx=11),
        _slot(
            contract_id="SPXW-20260701-04995.000-P",
            right="P",
            right_idx=1,
            offset=-5.0,
            strike=4995.0,
            strike_idx=9,
        ),
    ]

    recovered, missing = audit.derive_static_geometry_features(item=slots[1], slots=slots, definition=definition)
    put_recovered, _put_missing = audit.derive_static_geometry_features(item=slots[3], slots=slots, definition=definition)

    assert recovered["candidate_count_total"] == 4.0
    assert recovered["same_right_candidate_count"] == 3.0
    assert recovered["same_right_rank_by_abs_offset"] == 0.0
    assert recovered["local_same_right_neighbor_count_10_points"] == 3.0
    assert recovered["abs_offset_bucket_0_10"] == 1.0
    assert put_recovered["out_of_the_money_flag"] == 1.0
    assert not any(missing.values())
