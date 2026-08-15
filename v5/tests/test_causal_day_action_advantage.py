from __future__ import annotations

import pandas as pd
import pytest

from v5.ops.audit_causal_day_coverage import ENTRY_MINUTES
from v5.research.causal_day_action_advantage import (
    ActionAdvantageError,
    label_one_session,
)


def _frame(minute_values: dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for minute in ENTRY_MINUTES:
        bid, mid = minute_values.get(minute, (-10.0, -5.0))
        for suffix, offset in (("C", 0.0), ("P", -1.0)):
            rows.append(
                {
                    "session": "2025-08-01",
                    "entry_minute": minute,
                    "entry_regime": "morning" if minute < "12:46" else "afternoon",
                    "trade_id": f"{minute}|{suffix}",
                    "contract_id": suffix,
                    "right": suffix,
                    "self_delta": 0.4 if suffix == "C" else -0.4,
                    "entry_ask_usd": 500.0,
                    "entry_mid_usd": 490.0,
                    "net_bid_120m_usd": bid + offset,
                    "net_mid_120m_usd": mid + offset,
                    "clock_exit_minute_120m": "15:59",
                    "clock_exit_type_120m": "executable_bid",
                    "realised_hold_120m": 120.0,
                }
            )
    return pd.DataFrame(rows)


def test_wait_is_best_strictly_later_value_and_excludes_current() -> None:
    got = label_one_session(
        _frame({"09:35": (50.0, 40.0), "09:36": (20.0, 10.0), "09:37": (30.0, 25.0)})
    )
    minute = got.minutes.set_index("entry_minute")
    assert minute.loc["09:35", "q_wait_bid_120m_usd"] == 30.0
    assert minute.loc["09:36", "q_wait_bid_120m_usd"] == 30.0
    assert minute.loc["09:37", "q_wait_bid_120m_usd"] == 0.0
    assert minute["q_wait_bid_120m_usd"].is_monotonic_decreasing


def test_primary_oracle_uses_earliest_global_maximum_and_contract_tie_break() -> None:
    frame = _frame({"10:00": (100.0, 90.0), "13:30": (100.0, 95.0)})
    # Give the put the same bid reward as the call at 10:00; contract_id C wins.
    frame.loc[
        frame["entry_minute"].eq("10:00") & frame["contract_id"].eq("P"),
        "net_bid_120m_usd",
    ] = 100.0
    got = label_one_session(frame)
    assert got.summary["oracle_entry_minute"] == "10:00"
    assert got.summary["oracle_contract_id"] == "C"
    selected = got.candidates[got.candidates["is_primary_oracle_action"]]
    assert len(selected) == 1
    assert selected.iloc[0]["trade_id"] == "10:00|C"


def test_never_trade_is_the_oracle_when_every_reward_is_nonpositive() -> None:
    got = label_one_session(_frame({}))
    assert got.summary["global_oracle_bid_120m_usd"] == 0.0
    assert got.summary["oracle_entry_minute"] is None
    assert not got.candidates["is_primary_oracle_action"].any()
    assert (got.minutes["q_wait_bid_120m_usd"] == 0.0).all()


def test_bid_and_mid_wait_values_are_independent_declared_surfaces() -> None:
    got = label_one_session(
        _frame({"10:00": (10.0, 100.0), "10:01": (50.0, 20.0)})
    )
    minute = got.minutes.set_index("entry_minute")
    assert minute.loc["09:59", "q_wait_bid_120m_usd"] == 50.0
    assert minute.loc["09:59", "q_wait_mid_120m_usd"] == 100.0


def test_minute_without_eligible_contract_is_an_explicit_wait_only_state() -> None:
    frame = _frame({})
    frame = frame[~frame["entry_minute"].eq("10:00")]
    got = label_one_session(frame)
    minute = got.minutes.set_index("entry_minute")
    assert pd.isna(minute.loc["10:00", "best_q_enter_bid_120m_usd"])
    assert minute.loc["10:00", "q_wait_bid_120m_usd"] == 0.0
    assert got.summary["minutes_without_eligible_action"] == 1


def test_candidate_outside_declared_entry_clock_is_refused() -> None:
    frame = _frame({})
    bad = frame.iloc[[0]].copy()
    bad["entry_minute"] = "15:01"
    bad["trade_id"] = "15:01|C"
    with pytest.raises(ActionAdvantageError, match="outside"):
        label_one_session(pd.concat([frame, bad], ignore_index=True))


def test_blocked_terminal_exit_is_refused() -> None:
    frame = _frame({})
    frame.loc[frame.index[0], "clock_exit_type_120m"] = "blocked"
    with pytest.raises(ActionAdvantageError, match="blocked or unknown"):
        label_one_session(frame)
