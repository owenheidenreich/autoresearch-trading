from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace

from v4.scripts import run_protocol163_recent_data_catchup_request as p163


def test_protocol163_expected_sessions_excludes_weekends_and_good_friday() -> None:
    sessions = p163.expected_trading_sessions(date(2026, 4, 1), date(2026, 4, 6))

    assert [session.isoformat() for session in sessions] == [
        "2026-04-01",
        "2026-04-02",
        "2026-04-06",
    ]


def test_protocol163_metadata_estimate_does_not_need_timeseries_get_range(monkeypatch) -> None:
    calls = []

    class FakeMetadata:
        def get_cost(self, **kwargs):
            calls.append(kwargs)
            return 0.123

    class FakeHistorical:
        def __init__(self):
            self.metadata = FakeMetadata()

    monkeypatch.setitem(sys.modules, "databento", SimpleNamespace(Historical=FakeHistorical))

    status, estimates = p163.estimate_definition_costs(
        [date(2026, 4, 1)],
        skip_metadata=False,
    )

    assert status == "ok"
    assert estimates == {"2026-04-01": 0.123}
    assert calls[0]["schema"] == "definition"


def test_protocol163_hard_cap_rounds_conservatively() -> None:
    assert p163.proposed_hard_cap(18.01) == 25.0
    assert p163.proposed_hard_cap(0.0) == 10.0
