from __future__ import annotations

import pandas as pd

from v4.scripts.run_protocol101_early_sealed_failure_attribution import (
    SPX_TOLERANCES,
    add_causal_history_health,
    health_strata,
)


def test_health_strata_separate_raw_and_recovered_minutes() -> None:
    minute = pd.DataFrame(
        [
            {
                "session_date": "2026-07-22",
                "decision_minute_et": "09:31",
                "raw_checkpoint": True,
                "recovered_checkpoint": False,
                "atm_aligned": True,
                "opening_context_ready_for_session": True,
                "spx_abs_drift_points": 0.2,
            },
            {
                "session_date": "2026-07-22",
                "decision_minute_et": "09:32",
                "raw_checkpoint": False,
                "recovered_checkpoint": True,
                "atm_aligned": False,
                "opening_context_ready_for_session": True,
                "spx_abs_drift_points": 5.0,
            },
        ]
    )
    for tolerance in SPX_TOLERANCES:
        label = str(tolerance).replace(".", "_")
        minute[f"spx_within_{label}"] = (
            minute["spx_abs_drift_points"] <= tolerance
        )
    minute["raw_atm_aligned_spx_within_0_75"] = (
        minute["raw_checkpoint"]
        & minute["atm_aligned"]
        & minute["spx_within_0_75"]
    )
    minute["fully_healthy"] = (
        minute["raw_atm_aligned_spx_within_0_75"]
        & minute["opening_context_ready_for_session"]
    )
    minute["degraded"] = ~minute["fully_healthy"]
    minute = add_causal_history_health(minute)

    strata = health_strata(minute)

    assert strata["fully_healthy"] == {("2026-07-22", "09:31")}
    assert strata["recovered_checkpoint"] == {("2026-07-22", "09:32")}
    assert strata["degraded"] == {("2026-07-22", "09:32")}


def test_causal_history_health_requires_complete_consecutive_lookback() -> None:
    minute = pd.DataFrame(
        {
            "session_date": ["2026-07-22"] * 18,
            "decision_minute_et": [
                pd.Timestamp("2026-07-22 09:32") + pd.Timedelta(minutes=i)
                for i in range(18)
            ],
            "raw_atm_aligned_spx_within_0_75": [True] * 7 + [False] + [True] * 10,
            "fully_healthy": [True] * 7 + [False] + [True] * 10,
        }
    )
    minute["decision_minute_et"] = pd.to_datetime(
        minute["decision_minute_et"]
    ).dt.strftime("%H:%M")

    result = add_causal_history_health(minute)

    assert result.loc[5, "fully_healthy_5m_history"]
    assert not result.loc[8, "fully_healthy_5m_history"]
    assert result.loc[13, "fully_healthy_5m_history"]
    assert not result["fully_healthy_15m_history"].any()
