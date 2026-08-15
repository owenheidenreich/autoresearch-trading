from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from v5.research.causal_day_architectures import ArchitectureDimensions
from v5.research.causal_day_tensorizer import (
    AccountObservation,
    CANDLE_FEATURES,
    LADDER_FEATURES,
    tensorize_observation,
)


SESSION = "2025-09-03"


def _candles() -> pd.DataFrame:
    rows = []
    for i, minute in enumerate(("09:30", "09:31", "09:32", "09:33", "09:34", "09:35")):
        rows.append(
            {
                "session": SESSION,
                "bar_minute": minute,
                "knowable_at": f"09:{31 + i:02d}",
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 10.0 + i,
            }
        )
    return pd.DataFrame(rows)


def _ladder() -> pd.DataFrame:
    rows = []
    for minute, strike in (("09:35", 105.0), ("09:36", 110.0)):
        row = {
            "session": SESSION,
            "minute": minute,
            "contract_id": f"c-{minute}",
            "right": "C",
            "strike": strike,
            "underlying_price": 100.0,
            "entry_eligible": minute == "09:35",
        }
        row.update({column: float(index + 1) for index, column in enumerate(LADDER_FEATURES)})
        rows.append(row)
    return pd.DataFrame(rows)


def _account() -> AccountObservation:
    return AccountObservation(10_000.0, 0.0, 0, 1, False)


def test_tensorizer_uses_only_completed_prefix_and_current_ladder() -> None:
    got = tensorize_observation(
        _candles(),
        _ladder(),
        session=SESSION,
        minute="09:35",
        role="morning_entry",
        account=_account(),
    )
    assert got.candle_minutes == ("09:30", "09:31", "09:32", "09:33", "09:34")
    assert got.contract_ids == ("c-09:35",)
    assert got.batch.candles.shape == (1, 5, len(CANDLE_FEATURES))
    assert got.batch.ladder.shape == (1, 1, len(LADDER_FEATURES))
    assert got.batch.entry_action_mask.tolist() == [[True]]
    got.batch.validate(
        ArchitectureDimensions(
            len(CANDLE_FEATURES), len(LADDER_FEATURES), 5, 10, 5, hidden_size=8
        )
    )


def test_mutating_future_rows_cannot_change_decision_tensor() -> None:
    candles = _candles()
    ladder = _ladder()
    first = tensorize_observation(
        candles,
        ladder,
        session=SESSION,
        minute="09:35",
        role="morning_entry",
        account=_account(),
    )
    dirty_candles = candles.copy()
    dirty_candles.loc[
        dirty_candles["knowable_at"].gt("09:35"),
        ["open", "high", "low", "close", "volume"],
    ] *= 1000.0
    dirty_ladder = ladder.copy()
    dirty_ladder.loc[dirty_ladder["minute"].gt("09:35"), list(LADDER_FEATURES)] *= -1000.0
    second = tensorize_observation(
        dirty_candles,
        dirty_ladder,
        session=SESSION,
        minute="09:35",
        role="morning_entry",
        account=_account(),
    )
    assert torch.equal(first.batch.candles, second.batch.candles)
    assert torch.equal(first.batch.ladder, second.batch.ladder)


def test_afternoon_tensor_retains_the_opening_candle() -> None:
    minutes = pd.date_range("2000-01-01 09:30", "2000-01-01 15:59", freq="min")
    candles = pd.DataFrame(
        {
            "session": SESSION,
            "bar_minute": minutes.strftime("%H:%M"),
            "knowable_at": (minutes + pd.Timedelta(minutes=1)).strftime("%H:%M"),
            "open": 100.0 + np.arange(390),
            "high": 101.0 + np.arange(390),
            "low": 99.0 + np.arange(390),
            "close": 100.5 + np.arange(390),
            "volume": 10.0 + np.arange(390),
        }
    )
    ladder = _ladder().copy()
    ladder.loc[:, "minute"] = "15:00"
    got = tensorize_observation(
        candles,
        ladder,
        session=SESSION,
        minute="15:00",
        role="afternoon_entry",
        account=_account(),
    )
    assert got.candle_minutes[0] == "09:30"
    assert got.candle_minutes[-1] == "14:59"
    assert len(got.candle_minutes) == 330


def test_missing_volume_and_greek_are_encoded_with_observation_flags() -> None:
    ladder = _ladder()
    ladder.loc[ladder["minute"].eq("09:35"), ["volume", "self_iv"]] = np.nan
    got = tensorize_observation(
        _candles(),
        ladder,
        session=SESSION,
        minute="09:35",
        role="morning_entry",
        account=_account(),
    )
    values = got.batch.ladder[0, 0]
    assert values[LADDER_FEATURES.index("volume")] == 0.0
    assert values[LADDER_FEATURES.index("volume_observed")] == 0.0
    assert values[LADDER_FEATURES.index("self_iv")] == 0.0
    assert values[LADDER_FEATURES.index("self_iv_observed")] == 0.0
