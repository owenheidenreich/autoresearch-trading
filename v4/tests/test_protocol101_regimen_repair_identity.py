from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from v4.model.protocol101_regimen_repair import (
    TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    Protocol101DuplicateCanonicalSlotError,
    Protocol101DuplicateContractIdentityError,
    Protocol101DuplicateDecisionIdentityError,
    Protocol101DuplicateSessionMembershipError,
    Protocol101PolicyAxisAlignmentError,
    Protocol101SessionRoleOverlapError,
    Protocol101ValidationFoldOverlapError,
    assert_decision_row_identities,
    assert_fold_session_identities,
    assert_manifest_session_identities,
    assert_processed_row_identities,
    render_utc_ns,
)


def _row(decision: str = "2025-01-02 15:00:00+00:00") -> dict:
    shape = (2, 2, 7)
    return {
        "decision_time": pd.Timestamp(decision),
        "contract_ids": np.asarray(
            [["c0", "p0"], ["c1", "p1"]], dtype=object
        ),
        "strike_offsets": np.asarray([-5.0, 0.0]),
        "rights": ("C", "P"),
        "processed_row_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        "label_policy_index": np.arange(7, dtype=np.uint8),
        "labels_net_pnl": np.zeros(shape),
        "labels_mid_pnl": np.zeros(shape),
        "label_realized_exit_time_ns": np.ones(shape, dtype=np.int64),
        "label_source_exit_quote_time_ns": np.ones(shape, dtype=np.int64),
        "label_exit_quote_age_ms": np.zeros(shape),
        "label_exit_reason_code": np.ones(shape, dtype=np.uint8),
        "label_executable_exit_bid": np.ones(shape),
        "label_policy_deadline_ns": np.ones(shape, dtype=np.int64),
        "label_invalid_reason_code": np.zeros(shape, dtype=np.uint8),
    }


def test_duplicate_session_membership_precedes_downstream_ID_SESSION_001() -> None:
    downstream_calls = []
    with pytest.raises(Protocol101DuplicateSessionMembershipError) as caught:
        assert_manifest_session_identities(
            [{"session": "2025-01-02"}, {"session": "2025-01-02"}]
        )
        downstream_calls.append("load")
    assert caught.value.payload["blocker_code"] == (
        "P101_ID_DUPLICATE_SESSION_MEMBERSHIP"
    )
    assert downstream_calls == []


def test_duplicate_decision_precedes_downstream_ID_DECISION_001() -> None:
    downstream_calls = []
    row = _row()
    with pytest.raises(Protocol101DuplicateDecisionIdentityError):
        assert_processed_row_identities(
            [row, copy.deepcopy(row)],
            split="train",
            session="2025-01-02",
        )
        downstream_calls.append("target_or_score")
    assert downstream_calls == []


def test_duplicate_contract_precedes_downstream_ID_CONTRACT_001() -> None:
    downstream_calls = []
    row = _row()
    row["contract_ids"][1, 1] = "c0"
    with pytest.raises(Protocol101DuplicateContractIdentityError):
        assert_decision_row_identities(
            row, split="train", session="2025-01-02"
        )
        downstream_calls.append("hash_or_replay")
    assert downstream_calls == []


def test_duplicate_slot_precedes_downstream_ID_SLOT_001() -> None:
    downstream_calls = []
    row = _row()
    row["strike_offsets"] = np.asarray([-5.0, -5.0])
    with pytest.raises(Protocol101DuplicateCanonicalSlotError):
        assert_decision_row_identities(
            row, split="train", session="2025-01-02"
        )
        downstream_calls.append("fit")
    assert downstream_calls == []


@pytest.mark.parametrize(
    ("folds", "error"),
    [
        (
            [
                {
                    "fold_id": "f1",
                    "train_sessions": ["2025-01-02"],
                    "validation_sessions": ["2025-01-02"],
                }
            ],
            Protocol101SessionRoleOverlapError,
        ),
        (
            [
                {
                    "fold_id": "f1",
                    "train_sessions": ["2025-01-01"],
                    "validation_sessions": ["2025-01-02"],
                },
                {
                    "fold_id": "f2",
                    "train_sessions": ["2025-01-03"],
                    "validation_sessions": ["2025-01-02"],
                },
            ],
            Protocol101ValidationFoldOverlapError,
        ),
    ],
    ids=["role-overlap", "validation-fold-overlap"],
)
def test_role_and_fold_overlap_precede_load_ID_ROLE_001(
    folds: list[dict], error: type[Exception]
) -> None:
    downstream_calls = []
    with pytest.raises(error):
        assert_fold_session_identities(folds, attempt_id="attempt001")
        downstream_calls.append("load")
    assert downstream_calls == []


def test_unordered_policy_axis_fails_closed() -> None:
    row = _row()
    row["label_policy_index"] = np.asarray(
        [0, 1, 3, 2, 4, 5, 6], dtype=np.uint8
    )
    with pytest.raises(Protocol101PolicyAxisAlignmentError):
        assert_decision_row_identities(
            row, split="train", session="2025-01-02"
        )


def test_canonical_nanosecond_rendering() -> None:
    value = int(pd.Timestamp("2025-01-02T15:00:00.123456789Z").value)
    assert render_utc_ns(value) == "2025-01-02T15:00:00.123456789Z"
