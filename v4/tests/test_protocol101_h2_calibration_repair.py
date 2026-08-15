from __future__ import annotations

import numpy as np

from v4.model.protocol101_h2_calibration_repair import (
    CalibrationRow,
    chronological_inner_splits,
    confidence_from_state,
    fit_map,
    select_and_fit_map,
)


def rows() -> list[CalibrationRow]:
    result: list[CalibrationRow] = []
    for session in range(12):
        for index in range(10):
            score = -1.0 + 0.2 * index
            result.append(
                CalibrationRow(
                    session=f"2026-01-{session + 1:02d}",
                    decision_time=f"2026-01-{session + 1:02d}T10:{index:02d}:00",
                    score=score,
                    outcome=float(score + (session % 3) * 0.05 > 0.0),
                    selected_contract_id=f"C{index}",
                )
            )
    return result


def test_chronological_inner_splits_are_forward_only() -> None:
    splits = chronological_inner_splits(rows())
    assert len(splits) == 3
    for split in splits:
        assert max(split["train_sessions"]) < min(split["validation_sessions"])


def test_bounded_methods_emit_probabilities() -> None:
    data = rows()
    for method in ("training_tail_isotonic_v1", "training_tail_platt_v1"):
        state = fit_map(data, method=method)
        confidence = confidence_from_state(
            np.asarray([-2.0, 0.0, 2.0]),
            state,
        )
        assert np.isfinite(confidence).all()
        assert ((confidence >= 0.0) & (confidence <= 1.0)).all()


def test_method_selection_is_deterministic() -> None:
    first = select_and_fit_map(rows())
    second = select_and_fit_map(rows())
    assert first == second
    assert first["selected_method"] in {
        "training_tail_isotonic_v1",
        "training_tail_platt_v1",
    }
