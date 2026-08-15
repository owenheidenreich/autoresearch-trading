from __future__ import annotations

import pandas as pd
import pytest

from v5.ops.audit_causal_day_settlement import SettlementAuditError, validate_session


SESSION = "2025-08-01"
SETTLEMENT = pd.Timestamp("2025-08-01 20:00:00", tz="UTC")


def _official(close: float = 6238.01) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": [SETTLEMENT - pd.Timedelta(minutes=1), SETTLEMENT],
            "symbol": ["SPX", "SPX"],
            "close": [6237.30, close],
            "context_source": ["owned/thetadata/spx", "owned/thetadata/spx"],
            "is_derived": [False, False],
            "is_proxy": [False, False],
            "is_official_index_data": [True, True],
        }
    )


def _quotes(underlying: float = 6238.01) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": [SETTLEMENT, SETTLEMENT],
            "expiry": [SESSION, SESSION],
            "root": ["SPXW", "SPXW"],
            "settlement_style": ["PM", "PM"],
            "settlement_time_utc": [SETTLEMENT, SETTLEMENT],
            "underlying_price": [underlying, underlying],
        }
    )


def test_validates_unique_official_close_against_declared_pm_settlement() -> None:
    record = validate_session(SESSION, _official(), _quotes())
    assert record["settlement_spx"] == pytest.approx(6238.01)
    assert record["exact_aligned_identity"]
    assert record["settlement_time_et"].startswith("2025-08-01T16:00:00")


def test_rejects_aligned_underlying_that_differs_from_official_close() -> None:
    with pytest.raises(SettlementAuditError, match="aligned terminal SPX"):
        validate_session(SESSION, _official(), _quotes(6237.99))


def test_rejects_proxy_or_non_pm_source() -> None:
    official = _official()
    official["is_proxy"] = True
    with pytest.raises(SettlementAuditError, match="proxy"):
        validate_session(SESSION, official, _quotes())

    quotes = _quotes()
    quotes["settlement_style"] = "AM"
    with pytest.raises(SettlementAuditError, match="non-PM"):
        validate_session(SESSION, _official(), quotes)
