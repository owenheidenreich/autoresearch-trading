from __future__ import annotations

import pandas as pd

from v4.model.protocol101_baseline_attachment import (
    build_protocol101_baseline_event_actions,
    summarize_protocol101_baseline_attachment,
)


def test_baseline_attachment_marks_enter_holding_exit_and_wait() -> None:
    flat = pd.DataFrame(
        [
            _flat_row("2026-01-02T15:00:00+00:00", "SPXW-A", "surface-a"),
            _flat_row("2026-01-02T15:01:00+00:00", "SPXW-A", "surface-a"),
            _flat_row("2026-01-02T15:02:00+00:00", "SPXW-A", "surface-a"),
            _flat_row("2026-01-02T15:03:00+00:00", "SPXW-A", "surface-a"),
        ]
    )
    baseline = pd.DataFrame([_trade()])

    actions = build_protocol101_baseline_event_actions(flat, baseline)

    assert actions["protocol101_action"].tolist() == ["enter", "holding", "exit_then_wait", "wait"]
    assert actions.loc[0, "surface_candidate_uid"] == "surface-a"


def test_baseline_attachment_flags_splits_without_baseline() -> None:
    flat = pd.DataFrame([_flat_row("2026-01-02T15:00:00+00:00", "SPXW-A", "surface-a", split="future")])
    baseline = pd.DataFrame([_trade()])

    actions = build_protocol101_baseline_event_actions(flat, baseline)
    summary = summarize_protocol101_baseline_attachment(actions, baseline)

    assert actions.loc[0, "protocol101_action"] == "baseline_not_available_for_split"
    assert summary.splits_without_baseline == ("future",)


def test_baseline_attachment_prefers_same_minute_reentry_for_flat_event() -> None:
    flat = pd.DataFrame(
        [
            _flat_row("2026-01-02T15:00:00+00:00", "SPXW-A", "surface-a"),
            _flat_row("2026-01-02T15:02:00+00:00", "SPXW-B", "surface-b"),
        ]
    )
    baseline = pd.DataFrame(
        [
            _trade(),
            {
                **_trade(),
                "decision_time": "2026-01-02T15:02:00+00:00",
                "exit_time": "2026-01-02T15:03:00+00:00",
                "contract_id": "SPXW-B",
                "candidate_uid": "protocol101-b",
            },
        ]
    )

    actions = build_protocol101_baseline_event_actions(flat, baseline)

    assert actions["protocol101_action"].tolist() == ["enter", "enter"]
    assert actions.loc[1, "surface_candidate_uid"] == "surface-b"


def test_baseline_attachment_emits_wait_rows_for_seed_without_session_trade() -> None:
    flat = pd.DataFrame([_flat_row("2026-01-03T15:00:00+00:00", "SPXW-A", "surface-a", session="2026-01-03")])
    baseline = pd.DataFrame(
        [
            _trade(seed=1),
            _trade(seed=2),
        ]
    )

    actions = build_protocol101_baseline_event_actions(flat, baseline)

    assert actions["seed"].tolist() == [1, 2]
    assert actions["protocol101_action"].tolist() == ["wait", "wait"]


def _flat_row(
    decision: str,
    contract_id: str,
    candidate_uid: str,
    *,
    split: str = "unit",
    session: str = "2026-01-02",
) -> dict:
    return {
        "split": split,
        "session": session,
        "decision_time": decision,
        "decision_dt": decision,
        "candidate_uid": candidate_uid,
        "contract_id": contract_id,
    }


def _trade(*, seed: int = 1) -> dict:
    return {
        "reported_split": "unit",
        "seed": seed,
        "session": "2026-01-02",
        "decision_time": "2026-01-02T15:00:00+00:00",
        "exit_time": "2026-01-02T15:02:00+00:00",
        "contract_id": "SPXW-A",
        "candidate_uid": "protocol101-a",
        "pnl": 100.0,
    }
