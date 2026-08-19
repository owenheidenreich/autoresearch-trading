"""Tests for the backfill request-scope diagnostic (no vendor contact)."""
from __future__ import annotations

import json

import pytest

from v5.ops.probe_backfill_request_scope import (
    RECORDED_LADDER_SYMBOLS_MAX,
    probe_session,
    summarize,
    write_probe,
)


class FakeMetadata:
    """Records every cost query so scope arguments can be asserted."""

    def __init__(self, parent_usd: float, per_symbol_usd: float) -> None:
        self.parent_usd = parent_usd
        self.per_symbol_usd = per_symbol_usd
        self.calls: list[dict] = []

    def get_cost(self, **kwargs) -> float:
        self.calls.append(kwargs)
        if kwargs["stype_in"] == "parent":
            return self.parent_usd
        return self.per_symbol_usd * len(kwargs["symbols"])


class FakeClient:
    def __init__(self, parent_usd: float = 0.84, per_symbol_usd: float = 0.00005) -> None:
        self.metadata = FakeMetadata(parent_usd, per_symbol_usd)


def test_probe_session_queries_both_scopes_on_one_window() -> None:
    client = FakeClient()
    symbols = [f"SPXW  240315C0{i:06d}" for i in range(300)]
    row = probe_session(client, "2024-03-15", symbols)

    assert len(client.metadata.calls) == 2
    parent_call, symbol_call = client.metadata.calls
    assert parent_call["stype_in"] == "parent"
    assert symbol_call["stype_in"] == "raw_symbol"
    assert symbol_call["symbols"] == symbols
    # Both scopes must be priced over the identical window, or the comparison
    # would measure the window rather than the scope.
    assert parent_call["start"] == symbol_call["start"]
    assert parent_call["end"] == symbol_call["end"]
    assert row["traded_symbols"] == 300
    assert row["scope_ratio"] == pytest.approx(0.84 / (0.00005 * 300))


def test_probe_never_requests_data() -> None:
    """The probe may only touch cost metadata; any data call is a defect."""

    class ExplodingTimeseries:
        def get_range(self, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("the probe requested data instead of cost metadata")

    client = FakeClient()
    client.timeseries = ExplodingTimeseries()
    probe_session(client, "2024-03-15", ["SPXW  240315C03000000"])
    assert len(client.metadata.calls) == 2


def test_summarize_projects_full_ladder_above_traded_subset() -> None:
    rows = [
        {
            "parent_scope_usd": 0.80,
            "symbol_scope_usd": 0.015,
            "traded_symbols": 300,
            "usd_per_symbol": 0.00005,
        },
        {
            "parent_scope_usd": 0.90,
            "symbol_scope_usd": 0.020,
            "traded_symbols": 400,
            "usd_per_symbol": 0.00005,
        },
    ]
    summary = summarize(rows, sessions_total=794)
    assert summary["scope_ratio_mean"] == pytest.approx(0.85 / 0.0175)
    # The full-ladder bound must exceed the traded-only projection, because the
    # traded subset understates the quoted ladder.
    assert (
        summary["projected_full_ladder_total_usd"]
        > summary["projected_traded_symbol_total_usd"]
    )
    assert summary["projected_full_ladder_total_usd"] == pytest.approx(
        0.00005 * RECORDED_LADDER_SYMBOLS_MAX * 794
    )


def test_write_probe_refuses_to_overwrite(tmp_path) -> None:
    out = tmp_path / "probe.json"
    out.write_text("{}")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_probe(output_path=out, client=FakeClient())


def test_receipt_records_no_spend_and_no_download(tmp_path, monkeypatch) -> None:
    import v5.ops.probe_backfill_request_scope as module

    monkeypatch.setattr(
        module, "source_sessions", lambda *a, **k: (["2024-03-15", "2024-03-18"], "hash")
    )
    monkeypatch.setattr(
        module, "zero_dte_symbols", lambda root, session: ["SPXW  240315C03000000"]
    )
    out = tmp_path / "probe.json"
    receipt = write_probe(output_path=out, client=FakeClient(), stride=1)

    assert receipt["integrity"] == {
        "money_spent": False,
        "data_downloaded": False,
        "declaration_modified": False,
        "cost_metadata_only": True,
    }
    assert receipt["classification"].startswith("DIAGNOSTIC")
    stored = json.loads(out.read_text())
    assert stored["receipt_sha256"] == receipt["receipt_sha256"]
