from __future__ import annotations

import pandas as pd

from v4.model.protocol101_walking_skeleton_lifecycle import (
    FEATURE_NAMES,
    FloorSpec,
    apply_entry_safety_prefilter,
    apply_lifecycle_policy,
    contract_id_to_raw_symbol,
    next_floor,
)


def _trajectory(*, bid: float = 4.0, probability: float = 0.0) -> pd.DataFrame:
    start = pd.Timestamp("2025-03-06 15:50:00", tz="America/New_York").tz_convert("UTC")
    rows = []
    for second in range(301):
        timestamp = start + pd.Timedelta(seconds=second)
        rows.append(
            {
                "session": "2025-03-06",
                "split": "plumbing_replay",
                "contract_id": "SPXW-20250306-05695.000-P",
                "decision_time_ns": int((start - pd.Timedelta(minutes=1)).value),
                "entry_fill_time_ns": int(start.value),
                "sample_time_ns": int(timestamp.value),
                "source_event_time_ns": int(timestamp.value),
                "current_bid": bid,
                "current_ask": bid + 0.1,
                "entry_ask": 4.0,
                "elapsed_seconds": second,
                "minutes_to_forced_flat": (300 - second) / 60.0,
                "exit_probability": probability,
                "market_quote_age_ms": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_normalized_contract_converts_to_owned_opra_symbol() -> None:
    assert (
        contract_id_to_raw_symbol("SPXW-20250306-05695.000-P")
        == "SPXW  250306P05695000"
    )


def test_floor_is_upward_only() -> None:
    spec = FloorSpec()
    first = next_floor(
        committed_floor=2.4,
        entry_ask=4.0,
        maximum_bid=5.0,
        current_spread=0.10,
        spec=spec,
    )
    second = next_floor(
        committed_floor=first,
        entry_ask=4.0,
        maximum_bid=4.5,
        current_spread=0.10,
        spec=spec,
    )
    assert second >= first


def test_lifecycle_routes_fire_hold_learned_floor_and_forced_flat() -> None:
    learned, learned_actions = apply_lifecycle_policy(
        _trajectory(probability=0.9), threshold=0.5, route="full_policy"
    )
    assert learned["trigger"] == "learned_exit"
    assert any(row["action"] == "HOLD" for row in learned_actions)

    floor, _ = apply_lifecycle_policy(
        _trajectory(bid=3.0), threshold=1.0, route="floor_path_canary"
    )
    assert floor["trigger"] == "floor_trigger"

    forced, _ = apply_lifecycle_policy(
        _trajectory(probability=1.0), threshold=0.0, route="forced_flat_path_canary"
    )
    assert forced["trigger"] == "forced_flat"
    assert forced["exit_time_ns"] == int(
        pd.Timestamp("2025-03-06 15:55:00", tz="America/New_York").tz_convert("UTC").value
    )


def test_d48_d49_prefilter_masks_overlap_and_budget_excess() -> None:
    base = int(pd.Timestamp("2025-03-06 10:00:00", tz="America/New_York").tz_convert("UTC").value)
    candidates = [
        {
            "session": "2025-03-06",
            "decision_time_ns": base,
            "exit_time_ns": base + 120 * 1_000_000_000,
            "contract_id": "first",
            "entry_ask": 4.0,
            "pnl_after_fee": -200.0,
        },
        {
            "session": "2025-03-06",
            "decision_time_ns": base + 60 * 1_000_000_000,
            "exit_time_ns": base + 180 * 1_000_000_000,
            "contract_id": "overlap",
            "entry_ask": 4.0,
            "pnl_after_fee": 0.0,
        },
        {
            "session": "2025-03-06",
            "decision_time_ns": base + 180 * 1_000_000_000,
            "exit_time_ns": base + 240 * 1_000_000_000,
            "contract_id": "budget",
            "entry_ask": 4.0,
            "pnl_after_fee": 0.0,
        },
    ]
    accepted, rejected, summary = apply_entry_safety_prefilter(candidates)
    assert [row["contract_id"] for row in accepted] == ["first"]
    assert {row["entry_safety_reason"] for row in rejected} == {
        "one_open_position_overlap",
        "d49_remaining_budget_mask",
    }
    assert summary["all_accepted_d48_pass"]
    assert summary["all_accepted_d49_pass"]


def test_entry_safety_prefilter_rejects_15_30_decision_boundary() -> None:
    decision = int(
        pd.Timestamp("2025-03-06 15:30:00", tz="America/New_York")
        .tz_convert("UTC")
        .value
    )
    candidates = [
        {
            "session": "2025-03-06",
            "decision_time_ns": decision,
            "exit_time_ns": decision + 60 * 1_000_000_000,
            "contract_id": "boundary",
            "entry_ask": 4.0,
            "pnl_after_fee": 0.0,
        }
    ]
    accepted, rejected, _ = apply_entry_safety_prefilter(candidates)
    assert not accepted
    assert [row["entry_safety_reason"] for row in rejected] == [
        "at_or_after_15_30_entry_cutoff"
    ]


def test_runtime_features_exclude_future_labels() -> None:
    assert not set(FEATURE_NAMES) & {
        "exit_target",
        "future_max_bid_300s_label",
        "future_min_bid_300s_label",
    }
