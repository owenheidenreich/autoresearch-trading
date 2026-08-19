from __future__ import annotations

import pandas as pd
import pytest

from v5.ops.normalize_lifecycle_quote_backfill import (
    QuoteNormalizationError,
    normalize_session,
)


def _osi(session: str, right: str, strike: float) -> str:
    stamp = session.replace("-", "")[2:]
    return f"SPXW  {stamp}{right}{int(strike * 1000):08d}"


def _raw_pair(session: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions, quotes = [], []
    for index, strike in enumerate((4975.0, 4980.0, 4985.0, 4990.0, 4995.0), 1):
        for right, value in (("C", 10.0 + index), ("P", 9.0 + index)):
            symbol = _osi(session, right, strike)
            instrument_id = index * 10 + (right == "P")
            definitions.append({"symbol": symbol, "instrument_id": instrument_id})
            quotes.append(
                {
                    "ts_recv": f"{session}T14:35:00Z",
                    "symbol": symbol,
                    "instrument_id": instrument_id,
                    "bid_px_00": value - 0.1,
                    "ask_px_00": value + 0.1,
                    "bid_sz_00": 2,
                    "ask_sz_00": 3,
                }
            )
    return pd.DataFrame(definitions), pd.DataFrame(quotes)


def test_normalize_session_maps_quotes_to_same_day_definitions_and_parity_spot() -> None:
    definitions, quotes = _raw_pair("2024-01-02")

    got = normalize_session(definitions, quotes, session="2024-01-02")

    assert len(got) == 10
    assert got["expiry"].eq("2024-01-02").all()
    assert got["underlying_price"].notna().all()
    assert got["quote_age_ms"].eq(0.0).all()
    assert got["contract_id"].str.startswith("SPXW-20240102-").all()


def test_normalize_session_refuses_a_future_expiry_even_if_the_row_is_present() -> None:
    definitions, quotes = _raw_pair("2024-01-02")
    definitions.loc[0, "symbol"] = _osi("2024-01-03", "C", 4975.0)
    quotes.loc[0, "symbol"] = definitions.loc[0, "symbol"]

    with pytest.raises(QuoteNormalizationError, match="non-same-day expiry"):
        normalize_session(definitions, quotes, session="2024-01-02")
