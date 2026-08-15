from __future__ import annotations

import math
import inspect

import numpy as np
import pandas as pd

from v4.model.protocol101_canonical_stage1_contract import HYPOTHESES
from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_scoped_stage1_hgb import (
    CanonicalDecision,
    HGBUnitConfig,
    RepairedCanonicalDecision,
    run_hgb_unit,
    selection_rows_v5,
    split_fit_calibration_sessions,
)
from v4.model.protocol101_serial_simulator import (
    PROTOCOL101_SERIAL_SIMULATOR_VERSION,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
)


def _decision(index: int, *, session: str) -> CanonicalDecision:
    base = float(index) / 10.0
    features = np.asarray(
        [
            [base + candidate / 20.0] * len(HYPOTHESES["H0"])
            for candidate in range(3)
        ],
        dtype=float,
    )
    labels = np.asarray([-30.0, 20.0 + index, 80.0 + index], dtype=float)
    return CanonicalDecision(
        session=session,
        decision_time=pd.Timestamp(f"{session} 15:{30 + index:02d}:00", tz="UTC"),
        features=features,
        labels=labels,
        mid_labels=labels + 10.0,
        entry_asks=np.asarray([2.0, 2.5, 3.0], dtype=float),
        offsets=np.asarray([-5.0, 0.0, 5.0], dtype=float),
        rights=np.asarray(["P", "C", "C"], dtype=object),
        contract_ids=np.asarray(
            [f"{session}-P-{index}", f"{session}-ATM-{index}", f"{session}-C-{index}"],
            dtype=object,
        ),
        strike_indices=np.asarray([9, 10, 11], dtype=int),
        right_indices=np.asarray([1, 0, 0], dtype=int),
    )


def test_split_fit_calibration_sessions_is_chronological() -> None:
    fit, calibration = split_fit_calibration_sessions(
        ["2025-01-05", "2025-01-02", "2025-01-03", "2025-01-04"]
    )
    assert fit == ["2025-01-02", "2025-01-03", "2025-01-04"]
    assert calibration == ["2025-01-05"]
    assert max(fit) < min(calibration)


def test_historical_v4_unit_remains_dormant_for_compatibility() -> None:
    source = inspect.getsource(run_hgb_unit)
    assert "replay_candidates(" in source
    assert PROTOCOL101_SERIAL_SIMULATOR_VERSION in source or (
        "simulator_version" in source
    )


def test_repaired_selection_adapter_carries_two_clocks_without_alpha() -> None:
    base = _decision(0, session="2025-01-02")
    source = int(base.decision_time.value + 60 * 1_000_000_000)
    realized = int(base.decision_time.value + 10 * 60 * 1_000_000_000)
    repaired = RepairedCanonicalDecision(
        base=base,
        realized_exit_time_ns=np.asarray([realized] * 3, dtype=np.int64),
        source_exit_quote_time_ns=np.asarray([source] * 3, dtype=np.int64),
        exit_quote_age_ms=np.asarray([540_000.0] * 3),
        exit_reason_codes=np.asarray([3] * 3, dtype=np.uint8),
        executable_exit_bids=np.asarray([1.73, 2.73, 3.80]),
        policy_deadline_ns=np.asarray([realized] * 3, dtype=np.int64),
        invalid_reason_codes=np.zeros(3, dtype=np.uint8),
        canonical_strike_slots=np.asarray([9, 10, 11], dtype=np.int64),
        source_quote_time_ns=np.asarray(
            [base.decision_time.value] * 3, dtype=np.int64
        ),
        source_context_time_ns=np.asarray(
            [base.decision_time.value] * 3, dtype=np.int64
        ),
    )
    config = HGBUnitConfig(hypothesis="H0", policy_index=0, seed=42)
    candidates, _ = selection_rows_v5(
        [repaired],
        [np.asarray([0.0, 0.5, 1.0])],
        threshold=-1.0,
        epsilon=0.0,
        config=config,
        split="validation",
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source_simulator_version == (
        PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
    )
    assert candidate.label_source_exit_quote_time_ns == source
    assert candidate.label_realized_exit_time_ns == realized
    assert candidate.metadata["pessimistic_label_before_fee"] == 80.0
