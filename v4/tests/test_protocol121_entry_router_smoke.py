from __future__ import annotations

from v4.scripts.run_protocol121_protocol101_entry_router_smoke import _numeric_summary


def test_protocol121_numeric_summary() -> None:
    summary = _numeric_summary([3.0, 1.0, 2.0])

    assert summary == {"count": 3, "min": 1.0, "median": 2.0, "max": 3.0}


def test_protocol121_numeric_summary_handles_empty() -> None:
    summary = _numeric_summary([])

    assert summary["count"] == 0
    assert summary["median"] is None
