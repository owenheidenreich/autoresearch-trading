import pandas as pd

from v4.scripts.run_protocol101_canonical_v1_4_rehearsal_battery import (
    count_material_true_reorderings,
    full_session_trace_blockers,
)


def _readiness(*, ibkr_rows: int = 360, ibkr_slot_rows: int = 15_120):
    return {
        "trace_readiness": {
            "sessions": {
                "2026-07-20": {
                    "historical_rows": 360,
                    "historical_slot_rows": 15_120,
                    "ibkr_rows": ibkr_rows,
                    "ibkr_slot_rows": ibkr_slot_rows,
                }
            }
        }
    }


def test_full_session_trace_blockers_accepts_complete_pair():
    assert full_session_trace_blockers(_readiness(), ("2026-07-20",)) == []


def test_full_session_trace_blockers_rejects_partial_ibkr_capture():
    blockers = full_session_trace_blockers(
        _readiness(ibkr_rows=346, ibkr_slot_rows=14_532),
        ("2026-07-20",),
    )

    assert blockers == [
        "incomplete_ibkr_trace_rows:2026-07-20:346!=360",
        "incomplete_ibkr_slot_rows:2026-07-20:14532!=15120",
    ]


def test_count_material_true_reorderings_uses_current_schema():
    merged = pd.DataFrame(
        {
            "true_score_reordering": [True, True, False],
            "economically_material": [True, False, True],
        }
    )

    assert count_material_true_reorderings(merged) == 1


def test_count_material_true_reorderings_accepts_legacy_schema():
    merged = pd.DataFrame(
        {
            "true_score_reordering": [True, True],
            "selected_slot_swap_economically_material": [False, True],
        }
    )

    assert count_material_true_reorderings(merged) == 1
