"""Tests for the corrected symbol-scoped backfill acquirer (no vendor contact)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from v5.ops.acquire_lifecycle_backfill import (
    AcquisitionError,
    acquire,
    load_declaration,
    resolve_zero_dte_ladder,
    session_costs,
    write_preflight,
)
from v5.ops import acquire_lifecycle_backfill as module
from v5.ops.download_spxw_history import canonical_json, file_sha256

MODULE = Path("v5/ops/acquire_lifecycle_backfill.py")


class FakeSymbology:
    def __init__(self, universe: dict[str, list]) -> None:
        self.universe = universe
        self.calls: list[dict] = []

    def resolve(self, **kwargs):
        self.calls.append(kwargs)
        return {"result": self.universe}


class FakeMetadata:
    def __init__(self, definition_usd: float, per_symbol_usd: float) -> None:
        self.definition_usd = definition_usd
        self.per_symbol_usd = per_symbol_usd
        self.calls: list[dict] = []

    def get_cost(self, **kwargs) -> float:
        self.calls.append(kwargs)
        if kwargs["schema"] == "definition":
            return self.definition_usd
        return self.per_symbol_usd * len(kwargs["symbols"])


class FakeStore:
    def to_df(self):
        import pandas as pd

        return pd.DataFrame({"symbol": pd.Series(dtype="object")})


class FakeTimeseries:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_range(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStore()


class FakeClient:
    def __init__(
        self,
        *,
        universe: dict[str, list] | None = None,
        definition_usd: float = 0.0,
        per_symbol_usd: float = 0.00005,
    ) -> None:
        if universe is None:
            universe = {
                # two same-day contracts, one far-dated contract that must be excluded
                "SPXW  240315C05000000": [],
                "SPXW  240315P05000000": [],
                "SPXW  240517P04750000": [],
            }
        self.symbology = FakeSymbology(universe)
        self.metadata = FakeMetadata(definition_usd, per_symbol_usd)
        self.timeseries = FakeTimeseries()


def test_ladder_keeps_only_same_day_expiry() -> None:
    ladder = resolve_zero_dte_ladder(FakeClient(), "2024-03-15")
    assert ladder == ["SPXW  240315C05000000", "SPXW  240315P05000000"]


def test_ladder_refuses_when_nothing_resolves() -> None:
    client = FakeClient(universe={"SPXW  240517P04750000": []})
    with pytest.raises(AcquisitionError, match="no same-day SPXW ladder"):
        resolve_zero_dte_ladder(client, "2024-03-15")


def test_cbbo_is_priced_at_symbol_scope_not_parent() -> None:
    client = FakeClient()
    row = session_costs(client, "2024-03-15", cbbo_close="16:01")

    cbbo_call = [c for c in client.metadata.calls if c["schema"] == "cbbo-1m"][0]
    assert cbbo_call["stype_in"] == "raw_symbol"
    assert cbbo_call["symbols"] == ["SPXW  240315C05000000", "SPXW  240315P05000000"]
    # The defect being corrected: parent scope must never price cbbo again.
    assert "SPXW.OPT" not in cbbo_call["symbols"]
    assert row["ladder_symbols"] == 2
    assert row["session_usd"] == pytest.approx(0.0001)


def _declaration(
    tmp_path: Path,
    *,
    cap: float = 75.0,
    stype: str = "raw_symbol",
    cbbo_close: str | None = "16:01",
) -> Path:
    request: dict[str, object] = {"cbbo_stype_in": stype}
    if cbbo_close is not None:
        request["cbbo_close_minute"] = cbbo_close
    body = {
        "schema_version": "v5.lifecycle-quote-backfill-declaration.v2",
        "hard_cap_usd": cap,
        "implementation_sha256": file_sha256(MODULE),
        "request": request,
        "destination": {"root": str(tmp_path / "dest")},
        "source_ohlcv_inventory": {
            "root": str(tmp_path / "src"),
            "start": "2024-03-15",
            "end": "2024-03-15",
            "sessions_sha256": "filled-in-by-test",
        },
    }
    body["declaration_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True))
    return path


def test_declaration_must_pin_this_runner(tmp_path) -> None:
    path = _declaration(tmp_path)
    body = json.loads(path.read_text())
    body["implementation_sha256"] = "0" * 64
    body.pop("declaration_sha256")
    body["declaration_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    path.write_text(json.dumps(body))
    with pytest.raises(AcquisitionError, match="pins a different runner"):
        load_declaration(path)


def test_declaration_must_request_symbol_scope(tmp_path) -> None:
    path = _declaration(tmp_path, stype="parent")
    with pytest.raises(AcquisitionError, match="resolved-symbol scope"):
        load_declaration(path)


def test_declaration_rejects_ceiling_above_authorization(tmp_path) -> None:
    path = _declaration(tmp_path, cap=100.0)
    with pytest.raises(AcquisitionError, match="outside the owner-authorized range"):
        load_declaration(path)


def test_acquire_refuses_a_failed_preflight(tmp_path) -> None:
    declaration_path = _declaration(tmp_path)
    declaration = json.loads(declaration_path.read_text())
    receipt = {
        "gate": "STOP_OVER_HARD_CAP",
        "declaration_sha256": declaration["declaration_sha256"],
        "implementation_sha256": file_sha256(MODULE),
        "estimated_total_usd": 671.9,
        "session_costs": [],
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(receipt))

    with pytest.raises(AcquisitionError, match="preflight did not pass"):
        acquire(
            declaration_path=declaration_path,
            preflight_path=preflight_path,
            receipt_path=tmp_path / "acq.json",
            client=FakeClient(),
        )


def test_acquire_refuses_when_running_total_would_exceed_ceiling(tmp_path) -> None:
    """A passing total must still not permit per-session overrun."""

    declaration_path = _declaration(tmp_path, cap=1.0)
    declaration = json.loads(declaration_path.read_text())
    receipt = {
        "gate": "PASS",
        "declaration_sha256": declaration["declaration_sha256"],
        "implementation_sha256": file_sha256(MODULE),
        "estimated_total_usd": 0.5,  # understates the rows below
        "session_costs": [
            {"session": "2024-03-15", "ladder_symbols": 2, "session_usd": 0.6},
            {"session": "2024-03-18", "ladder_symbols": 2, "session_usd": 0.6},
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(receipt))

    with pytest.raises(AcquisitionError, match="would exceed the ceiling"):
        acquire(
            declaration_path=declaration_path,
            preflight_path=preflight_path,
            receipt_path=tmp_path / "acq.json",
            client=FakeClient(),
        )


def test_complete_sessions_are_skipped_without_any_vendor_call(tmp_path) -> None:
    """The 2026-08-16 fix: a restart must not re-resolve what it already owns.

    Before this, `acquire` resolved every declared session's ladder before
    checking whether its outputs existed, so a restart made hundreds of vendor
    calls to reach new work and one transient 504 discarded the attempt.
    """

    import pandas as pd

    declaration_path = _declaration(tmp_path)
    declaration = json.loads(declaration_path.read_text())
    root = Path(declaration["destination"]["root"])
    # Pre-create both outputs for the only quoted session.
    for schema in ("definition", "cbbo-1m"):
        path = root / "raw" / "databento" / f"opra_spxw_{schema.replace('-', '_')}"
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"symbol": ["SPXW  240315C05000000"]}).to_parquet(
            path / f"2024-03-15.{schema}.parquet"
        )

    receipt = {
        "gate": "PASS",
        "declaration_sha256": declaration["declaration_sha256"],
        "implementation_sha256": file_sha256(MODULE),
        "estimated_total_usd": 0.01,
        "session_costs": [
            {"session": "2024-03-15", "ladder_symbols": 2, "session_usd": 0.01}
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(receipt))

    client = FakeClient()
    result = acquire(
        declaration_path=declaration_path,
        preflight_path=preflight_path,
        receipt_path=tmp_path / "acq.json",
        client=client,
    )

    assert client.symbology.calls == [], "a complete session must not be resolved again"
    assert client.timeseries.calls == [], "a complete session must not be downloaded again"
    assert result["completion"]["downloaded"] == 0
    assert all(f["status"] == "EXISTING_DECLARED_OUTPUT" for f in result["files"])


def test_transient_vendor_failures_are_retried_then_surface(monkeypatch) -> None:
    import v5.ops.acquire_lifecycle_backfill as module

    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("504 gateway timed out")
        return "ok"

    assert module._with_retry("probe", flaky) == "ok"
    assert calls["n"] == 3

    def always_fails():
        raise RuntimeError("504 gateway timed out")

    with pytest.raises(AcquisitionError, match="failed after"):
        module._with_retry("probe", always_fails)


def test_a_refusal_is_never_retried(monkeypatch) -> None:
    """Ceiling and drift guards are decisions, not blips."""

    import v5.ops.acquire_lifecycle_backfill as module

    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def refuses():
        calls["n"] += 1
        raise AcquisitionError("running spend would exceed the ceiling")

    with pytest.raises(AcquisitionError, match="exceed the ceiling"):
        module._with_retry("probe", refuses)
    assert calls["n"] == 1, "a refusal must surface on the first attempt"


def test_preflight_refuses_to_overwrite(tmp_path) -> None:
    out = tmp_path / "preflight.json"
    out.write_text("{}")
    with pytest.raises(AcquisitionError, match="refusing to overwrite"):
        write_preflight(
            declaration_path=_declaration(tmp_path), output_path=out, client=FakeClient()
        )


def test_cbbo_window_end_is_declared_and_exclusive() -> None:
    """The window that lost the closing bar must be a declared value, not a constant.

    Vendor ranges are half-open and CBBO-1m bars are stamped at their end, so a
    16:00 close drops the 16:00-stamped bar — measured on the V4 acquisition as
    389 bars per contract instead of 390. A 16:01 close includes it.
    """

    _, end_1600 = module._window("2024-03-15", "cbbo-1m", cbbo_close="16:00")
    _, end_1601 = module._window("2024-03-15", "cbbo-1m", cbbo_close="16:01")
    assert end_1600 == "2024-03-15T20:00:00Z"
    assert end_1601 == "2024-03-15T20:01:00Z", "the close must reach past the 16:00 stamp"


def test_cbbo_window_is_dst_aware() -> None:
    """The same wall-clock close is a different UTC instant across DST."""

    _, summer = module._window("2024-07-11", "cbbo-1m", cbbo_close="16:01")
    _, winter = module._window("2024-12-11", "cbbo-1m", cbbo_close="16:01")
    assert summer == "2024-07-11T20:01:00Z"
    assert winter == "2024-12-11T21:01:00Z"


def test_definition_window_ignores_the_cbbo_close() -> None:
    """Only cbbo-1m is affected; definitions keep their full-UTC-day window."""

    assert module._window("2024-03-15", "definition", cbbo_close="16:01") == module._bounds(
        "2024-03-15", "definition"
    )


def test_declaration_without_a_close_minute_is_refused(tmp_path) -> None:
    with pytest.raises(AcquisitionError, match="cbbo_close_minute"):
        module.load_declaration(_declaration(tmp_path, cbbo_close=None))


def test_declaration_with_a_malformed_close_minute_is_refused(tmp_path) -> None:
    with pytest.raises(AcquisitionError, match="cbbo_close_minute"):
        module.load_declaration(_declaration(tmp_path, cbbo_close="4pm"))


def test_a_drifted_window_dependency_is_refused(tmp_path) -> None:
    """The pinned runner hash never covered the imported window logic.

    `_bounds` lives in another module, so the window could change without
    invalidating `implementation_sha256`. A declaration that pins the dependency
    must refuse when those bytes move.
    """

    path = _declaration(tmp_path)
    body = json.loads(path.read_text())
    body["implementation_dependencies"] = [
        {"path": "v5/ops/download_spxw_history.py", "sha256": "0" * 64}
    ]
    body.pop("declaration_sha256", None)
    body["declaration_sha256"] = module._payload_sha256(body, "declaration_sha256")
    path.write_text(json.dumps(body))
    with pytest.raises(AcquisitionError, match="dependency drifted"):
        module.load_declaration(path)
