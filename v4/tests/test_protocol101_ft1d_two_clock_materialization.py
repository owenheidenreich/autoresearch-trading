from __future__ import annotations

import copy

import numpy as np
import pytest

from v4.model.protocol101_regimen_repair import TWO_CLOCK_PROCESSED_ROW_SCHEMA
from v4.scripts import materialize_protocol101_ft1d_two_clock_rows as materializer


def _legacy_row() -> dict:
    offsets = np.array([-5.0])
    rights = np.array(["C"])
    return {
        "decision_time": "2025-01-02T14:32:00+00:00",
        "strike_offsets": offsets,
        "rights": rights,
        "contract_ids": np.array([["SPXW-TEST-C"]]),
        "candidate_mask": np.array([[True]]),
        "contract_quote_metadata": {
            "SPXW-TEST-C": {
                "source_quote_time": "2025-01-02T14:32:00+00:00",
                "source_context_time": "2025-01-02T14:31:00+00:00",
            }
        },
        "labels_net_pnl": np.zeros((1, 1, 7)),
        "labels_mid_pnl": np.zeros((1, 1, 7)),
    }


def _materialized_row() -> dict:
    row = copy.deepcopy(_legacy_row())
    row["contract_quote_metadata"]["SPXW-TEST-C"]["diagnostic_only"] = True
    row.update(
        {
            "processed_row_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
            "label_realized_exit_time_ns": np.ones((1, 1, 7), dtype=np.int64),
            "label_source_exit_quote_time_ns": np.ones(
                (1, 1, 7), dtype=np.int64
            ),
            "label_exit_quote_age_ms": np.zeros((1, 1, 7)),
            "label_exit_reason_code": np.ones((1, 1, 7), dtype=np.uint8),
            "label_executable_exit_bid": np.ones((1, 1, 7)),
            "label_policy_deadline_ns": np.ones(
                (1, 1, 7), dtype=np.int64
            ),
            "label_policy_index": np.arange(7, dtype=np.uint8),
            "label_invalid_reason_code": np.zeros(
                (1, 1, 7), dtype=np.uint8
            ),
            "candidate_filter_trace": {},
            "ladder_context": {},
        }
    )
    return row


def test_values_equal_allows_only_metadata_superset() -> None:
    legacy = _legacy_row()
    materialized = _materialized_row()
    assert materializer.values_equal(
        legacy["contract_quote_metadata"],
        materialized["contract_quote_metadata"],
    )


def test_additive_parity_rejects_changed_legacy_field() -> None:
    legacy = [_legacy_row() for _ in range(359)]
    materialized = [_materialized_row() for _ in range(359)]
    materialized[100]["candidate_mask"][0, 0] = False
    with pytest.raises(RuntimeError, match="legacy_field_mismatch"):
        materializer.assert_additive_parity(
            legacy,
            materialized,
            session="2025-01-02",
        )


def test_additive_parity_rejects_unknown_field() -> None:
    legacy = [_legacy_row() for _ in range(359)]
    materialized = [_materialized_row() for _ in range(359)]
    materialized[0]["not_signed"] = 1
    with pytest.raises(RuntimeError, match="unexpected_additive_fields"):
        materializer.assert_additive_parity(
            legacy,
            materialized,
            session="2025-01-02",
        )
