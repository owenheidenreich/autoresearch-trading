from __future__ import annotations

import pandas as pd

from v4.scripts.run_protocol101_clean_window_certification import (
    WARMUP_MINUTES,
    apply_window_policy,
)


def _minutes(count: int, *, interruption: int | None = None) -> pd.DataFrame:
    healthy = [True] * count
    if interruption is not None:
        healthy[interruption] = False
    start = pd.Timestamp("2026-07-15 09:32")
    return pd.DataFrame(
        {
            "session_date": ["2026-07-15"] * count,
            "decision_minute_et": [
                (start + pd.Timedelta(minutes=index)).strftime("%H:%M")
                for index in range(count)
            ],
            "base_healthy": healthy,
            "opening_context_ready": [True] * count,
        }
    )


def test_option_window_requires_causal_warmup_and_minimum_window() -> None:
    result = apply_window_policy(_minutes(46))

    assert not result.loc[WARMUP_MINUTES - 1, "option_preliminary_eligible"]
    assert result.loc[WARMUP_MINUTES, "option_preliminary_eligible"]
    assert result["option_eligible"].sum() == 31


def test_interruption_resets_warmup_and_permanently_breaks_context_prefix() -> None:
    result = apply_window_policy(_minutes(80, interruption=35))

    assert not result.loc[35, "option_preliminary_eligible"]
    assert not result.loc[50, "option_preliminary_eligible"]
    assert result.loc[51, "option_preliminary_eligible"]
    assert not result.loc[35:, "context_prefix_clean"].any()


def test_short_post_warmup_fragment_is_quarantined() -> None:
    result = apply_window_policy(_minutes(40))

    assert result["option_preliminary_eligible"].sum() == 25
    assert not result["option_eligible"].any()


def test_prewindow_history_row_does_not_poison_decision_context() -> None:
    decision_rows = _minutes(46)
    prewindow = pd.DataFrame(
        {
            "session_date": ["2026-07-15"],
            "decision_minute_et": ["09:31"],
            "base_healthy": [False],
            "opening_context_ready": [False],
        }
    )

    result = apply_window_policy(pd.concat([prewindow, decision_rows], ignore_index=True))
    scored = result[result["decision_minute_et"] != "09:31"]

    assert scored["context_eligible"].all()
    assert scored["option_eligible"].sum() == 31
