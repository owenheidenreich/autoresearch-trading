from __future__ import annotations

import numpy as np
import pandas as pd

from v5.ops import build_causal_day_dataset as day
from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES


def _candles() -> pd.DataFrame:
    minutes = [day.minute_label(value) for value in range(570, 960)]
    close = 5000.0 + np.arange(390, dtype=float)
    return pd.DataFrame(
        {
            "session": "2025-09-03",
            "bar_minute": minutes,
            "knowable_at": [day.minute_label(day.minute_number(value) + 1) for value in minutes],
            "open": close - 0.25,
            "high": close + 0.50,
            "low": close - 0.50,
            "close": close,
            "volume": np.arange(1, 391, dtype=float),
        }
    )


def _quote_path(*, no_quotes_after_entry: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, minute in enumerate(QUOTE_MINUTES):
        spot = 100.0 if i < 64 else 100.0 + (i - 63) * 5.0
        bid, ask, mid, bid_size = 0.9, 1.1, 1.0, 1.0
        if no_quotes_after_entry and i > 4:
            bid = ask = mid = np.nan
            bid_size = 0.0
        rows.append(
            {
                "minute": minute,
                "contract_id": "c1",
                "right": "C",
                "strike": 105.0,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "bid_size": bid_size,
                "ask_size": 1.0,
                "quote_age_ms": 0.0,
                "underlying_price": spot,
            }
        )
    quotes = pd.DataFrame(rows)
    candidate = pd.DataFrame(
        {
            "session": ["2025-09-03"],
            "entry_minute": ["09:35"],
            "contract_id": ["c1"],
            "strike": [105.0],
            "is_call": [True],
            "underlying_price": [100.0],
            "entry_ask_usd": [110.0],
            "entry_mid_usd": [100.0],
            "trade_id": ["2025-09-03|09:35|c1"],
        }
    )
    return quotes, candidate


def test_regime_router_has_no_boundary_gap() -> None:
    assert day.regime_for_minute("12:45") == "morning"
    assert day.regime_for_minute("12:46") == "afternoon"
    assert day.regime_for_minute("16:00") == "afternoon"


def test_decision_uses_the_completed_previous_es_bar() -> None:
    state = day.candle_states(_candles())
    ten = state[state["minute"].eq("10:00")].iloc[0]
    assert ten["latest_es_bar_minute"] == "09:59"
    assert ten["history_minutes"] == 30


def test_mutating_a_future_candle_cannot_move_an_earlier_state() -> None:
    clean = _candles()
    dirty = clean.copy()
    dirty.loc[dirty["bar_minute"].gt("10:00"), ["open", "high", "low", "close"]] *= 4.0
    a = day.candle_states(clean).set_index("minute")
    b = day.candle_states(dirty).set_index("minute")
    columns = [column for column in a.columns if column not in {"session", "regime"}]
    pd.testing.assert_frame_equal(a.loc[:"10:00", columns], b.loc[:"10:00", columns])


def test_clock_exit_takes_first_later_bid_not_later_best() -> None:
    quotes, candidate = _quote_path()
    target = quotes.index[quotes["minute"].eq("10:35")][0]
    quotes.loc[target, ["bid", "ask", "mid"]] = np.nan
    quotes.loc[target, "bid_size"] = 0.0
    quotes.loc[target + 1, ["bid", "ask", "mid"]] = [2.0, 2.2, 2.1]
    quotes.loc[target + 2, ["bid", "ask", "mid"]] = [20.0, 20.2, 20.1]

    got = day.attach_candidate_outcomes(candidate, quotes).iloc[0]
    assert got["clock_exit_delay_60m"] == 1
    assert got["clock_exit_bid_60m"] == 2.0
    assert got["clock_exit_status_60m"] == "delayed_first_later_bid"


def test_candidate_survives_when_every_later_quote_is_missing() -> None:
    quotes, candidate = _quote_path(no_quotes_after_entry=True)
    got = day.attach_candidate_outcomes(candidate, quotes)
    assert len(got) == 1
    assert got.iloc[0]["clock_exit_status_60m"] == "blocked_no_executable_bid"
    assert np.isnan(got.iloc[0]["net_bid_60m_usd"])


def test_validated_terminal_settlement_is_not_mislabeled_as_a_bid() -> None:
    quotes, candidate = _quote_path(no_quotes_after_entry=True)
    got = day.attach_candidate_outcomes(
        candidate, quotes, settlement_spx=130.0
    ).iloc[0]
    assert got["clock_exit_status_60m"] == "validated_cash_settlement"
    assert got["clock_exit_type_60m"] == "validated_cash_settlement"
    assert np.isnan(got["clock_exit_bid_60m"])
    assert got["clock_terminal_intrinsic_60m"] == 25.0
    assert got["net_bid_60m_usd"] == 25.0 * 100.0 - 110.0 - 3.08
    assert got["net_bid_zero_recovery_60m_usd"] == -110.0 - 3.08


def test_otm_contract_crosses_and_reaches_declared_itm_depth() -> None:
    quotes, candidate = _quote_path()
    got = day.attach_candidate_outcomes(candidate, quotes).iloc[0]
    assert got["reached_cross_60m"]
    assert got["reached_10_itm_90m"]
    assert got["reached_30_itm_90m"]
    assert got["maximum_itm_depth_90m"] >= 30.0


def test_future_quote_mutation_cannot_change_earlier_ladder_or_candidates() -> None:
    quotes, _ = _quote_path()
    quotes["raw_symbol"] = quotes["contract_id"]
    quotes["volume"] = 1.0
    quotes["open_interest"] = 2.0
    clean_ladder, clean_candidates = day.ladder_state(quotes, "2025-09-03")
    dirty = quotes.copy()
    future = dirty["minute"].gt("10:00")
    dirty.loc[future, ["bid", "ask", "mid", "underlying_price"]] *= 10.0
    dirty_ladder, dirty_candidates = day.ladder_state(dirty, "2025-09-03")
    pd.testing.assert_frame_equal(
        clean_ladder[clean_ladder["minute"].le("10:00")].reset_index(drop=True),
        dirty_ladder[dirty_ladder["minute"].le("10:00")].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        clean_candidates[clean_candidates["entry_minute"].le("10:00")].reset_index(drop=True),
        dirty_candidates[dirty_candidates["entry_minute"].le("10:00")].reset_index(drop=True),
    )
