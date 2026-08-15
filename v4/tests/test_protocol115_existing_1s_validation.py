from __future__ import annotations

import pandas as pd

from v4.scripts.run_protocol115_protocol101_existing_1s_validation import (
    decide,
    prepare_rows,
    summarize_splits,
)


def _row(split: str, session: str, status: str, diff: float | None = None) -> dict:
    row = {
        "split": split,
        "session": session,
        "decision_time": f"{session}T15:00:00Z",
        "audit_status": status,
        "right": "C",
        "sign_flip": False,
        "planned_sign_flip": False,
        "mandatory_event_before_lifecycle_exit": False,
    }
    if diff is not None:
        row |= {
            "pnl_1m": 100.0,
            "pnl_1s": 100.0 + diff,
            "pnl_diff_1s_minus_1m": diff,
            "pnl_1s_planned_exit": 100.0,
            "pnl_diff_1s_planned_minus_1m": 0.0,
        }
    return row


def test_prepare_rows_adds_march_overlay() -> None:
    rows = pd.DataFrame(
        [
            _row("q1_2026", "2026-02-27", "audited", 0.0),
            _row("q1_2026", "2026-03-06", "audited", 0.0),
        ]
    )

    prepared = prepare_rows(rows)

    assert len(prepared[prepared["split"].eq("march_2026")]) == 1
    assert set(prepared["time_bucket"]) == {"post_open_morning"}


def test_decision_requires_missing_critical_splits_to_remain_partial() -> None:
    rows = pd.DataFrame(
        [
            _row("q4_2024_external", "2024-10-01", "missing_1s_session"),
            _row("q3_2025", "2025-07-01", "missing_1s_session"),
            _row("q4_2025", "2025-10-17", "audited", 0.0),
            _row("q1_2026", "2026-03-06", "audited", 0.0),
        ]
    )

    summary = summarize_splits(prepare_rows(rows))

    assert decide(summary) == "partial_support_needs_targeted_1s_or_live_shadow"


def test_decision_rejects_large_covered_repricing() -> None:
    rows = pd.DataFrame(
        [
            _row("q4_2024_external", "2024-10-01", "audited", 0.0),
            _row("q3_2025", "2025-07-01", "audited", 0.0),
            _row("q4_2025", "2025-10-17", "audited", -20.0),
            _row("q1_2026", "2026-03-06", "audited", 0.0),
        ]
    )

    summary = summarize_splits(prepare_rows(rows))

    assert decide(summary) == "reject_or_reprice_protocol101_timing_assumption"
