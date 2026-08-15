from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops.evaluate_causal_day_action_value import (
    attach_causal_fields,
    attach_outcome,
    fold_evidence,
    session_values,
)


def _selected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["2025-08-01", "2025-08-04"],
            "entry_minute": ["10:00", "13:30"],
            "contract_id": ["a", "b"],
            "fold": [1, 2],
        }
    )


def _causal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["2025-08-01", "2025-08-04"],
            "entry_minute": ["10:00", "13:30"],
            "contract_id": ["a", "b"],
            "entry_regime": ["morning", "afternoon"],
            "right": ["C", "P"],
            "self_delta": [0.4, -0.3],
            "entry_ask_usd": [400.0, 300.0],
            "entry_mid_usd": [395.0, 295.0],
        }
    )


def test_causal_and_outcome_attachment_are_one_to_one() -> None:
    causal = attach_causal_fields(_selected(), _causal())
    outcomes = _causal()[["session", "entry_minute", "contract_id"]].copy()
    outcomes["q_enter_mid_120m_usd"] = [10.0, -20.0]
    got = attach_outcome(causal, outcomes, column="q_enter_mid_120m_usd")
    assert got["q_enter_mid_120m_usd"].tolist() == [10.0, -20.0]


def test_missing_selected_outcome_is_refused() -> None:
    causal = attach_causal_fields(_selected(), _causal())
    outcomes = _causal().iloc[:1][["session", "entry_minute", "contract_id"]].copy()
    outcomes["q_enter_mid_120m_usd"] = 10.0
    with pytest.raises(RuntimeError, match="finite"):
        attach_outcome(causal, outcomes, column="q_enter_mid_120m_usd")


def test_session_vector_keeps_abstentions_as_zero() -> None:
    trades = pd.DataFrame({"session": ["a", "c"], "pnl": [10.0, -3.0]})
    got = session_values(trades, ["a", "b", "c"], column="pnl")
    assert got.tolist() == [10.0, 0.0, -3.0]


def test_fold_evidence_requires_and_reports_all_five_folds() -> None:
    sessions = [f"s{i}" for i in range(10)]
    fold_map = {session: index // 2 + 1 for index, session in enumerate(sessions)}
    values = np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, -1.0, -1.0])
    count, means = fold_evidence(values, sessions, fold_map)
    assert count == 4
    assert means == {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": -1.0}
