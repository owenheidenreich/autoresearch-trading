from __future__ import annotations

import pandas as pd
import pytest

from v5.ops.audit_causal_day_coverage import ENTRY_MINUTES
from v5.research.causal_day_action_value_selection import (
    ActionValueSelectionError,
    select_first_positive_advantage,
    select_outcome_blind_matched_control,
)


def _minutes(*, wait: float = 5.0, null: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": "2025-08-01",
            "entry_minute": ENTRY_MINUTES,
            "fold": 1,
            "shuffled_label_null": null,
            "predicted_q_wait_bid_120m_usd": wait,
        }
    )


def _candidates(*, null: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["2025-08-01"] * 4,
            "entry_minute": ["09:35", "09:35", "09:36", "09:36"],
            "contract_id": ["b", "a", "c", "d"],
            "fold": 1,
            "shuffled_label_null": null,
            "predicted_q_enter_bid_120m_usd": [5.0, 5.0, 9.0, 8.0],
        }
    )


def test_selector_walks_causally_and_uses_strict_advantage() -> None:
    got = select_first_positive_advantage(_candidates(), _minutes())
    assert len(got) == 1
    assert got.iloc[0]["entry_minute"] == "09:36"
    assert got.iloc[0]["contract_id"] == "c"
    assert got.iloc[0]["predicted_action_advantage_usd"] == pytest.approx(4.0)


def test_structural_zero_floor_prevents_negative_forced_trade() -> None:
    candidates = _candidates()
    candidates["predicted_q_enter_bid_120m_usd"] = -1.0
    got = select_first_positive_advantage(candidates, _minutes(wait=-10.0))
    assert got.empty


def test_tie_break_does_not_read_a_future_outcome() -> None:
    candidates = _candidates()
    candidates.loc[candidates["entry_minute"].eq("09:35"), "predicted_q_enter_bid_120m_usd"] = 6.0
    candidates["future_pnl"] = [10_000.0, -10_000.0, 0.0, 0.0]
    got = select_first_positive_advantage(candidates, _minutes())
    assert got.iloc[0]["contract_id"] == "a"


def test_incomplete_wait_clock_is_refused() -> None:
    with pytest.raises(ActionValueSelectionError, match="incomplete"):
        select_first_positive_advantage(_candidates(), _minutes().iloc[:-1])


def _population() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["2025-08-01"] * 4,
            "entry_minute": ["09:35", "09:35", "09:35", "09:36"],
            "contract_id": ["chosen", "near", "wrong-side", "later"],
            "entry_regime": "morning",
            "right": ["C", "C", "P", "C"],
            "self_delta": [0.40, 0.41, -0.40, 0.4001],
            "entry_ask_usd": [500.0, 510.0, 500.0, 501.0],
            "future_pnl": [0.0, -999.0, 999.0, 999.0],
        }
    )


def test_control_prefers_same_minute_and_side_without_outcome() -> None:
    population = _population()
    target = population[population["contract_id"].eq("chosen")]
    got = select_outcome_blind_matched_control(target, population)
    assert got.iloc[0]["contract_id"] == "near"
    changed = population.copy()
    changed["future_pnl"] *= -1_000.0
    again = select_outcome_blind_matched_control(target, changed)
    assert again.iloc[0]["contract_id"] == "near"


def test_control_falls_back_to_nearest_minute_with_same_regime_and_side() -> None:
    population = _population()
    population = population[~population["contract_id"].eq("near")]
    target = population[population["contract_id"].eq("chosen")]
    got = select_outcome_blind_matched_control(target, population)
    assert got.iloc[0]["contract_id"] == "later"
    assert got.iloc[0]["match_minute_distance"] == 1
