from __future__ import annotations

import json
from pathlib import Path

import databento as db
import pandas as pd
import pytest
import zstandard as zstd

from v4.scripts import run_protocol101_ft2_d55_acquisition as d55


def test_evenly_spaced_indices_match_owner_rule() -> None:
    assert d55.evenly_spaced_indices(218, 30) == [
        0,
        7,
        15,
        22,
        30,
        37,
        45,
        52,
        60,
        67,
        75,
        82,
        90,
        97,
        105,
        112,
        120,
        127,
        135,
        142,
        150,
        157,
        165,
        172,
        180,
        187,
        195,
        202,
        210,
        217,
    ]


def test_evenly_spaced_indices_repairs_duplicates() -> None:
    indices = d55.evenly_spaced_indices(5, 5)
    assert indices == [0, 1, 2, 3, 4]
    assert len(indices) == len(set(indices))


def test_session_symbols_requires_exact_spxw_pm_0dte(tmp_path: Path) -> None:
    path = tmp_path / "day.parquet"
    pd.DataFrame(
        {
            "raw_symbol": ["SPXW  250220C06000000"],
            "root": ["SPXW"],
            "expiry": ["2025-02-20"],
            "settlement_style": ["PM"],
        }
    ).to_parquet(path)
    symbols, rows = d55.session_symbols(path, "2025-02-20")
    assert symbols == ["SPXW  250220C06000000"]
    assert rows == 1

    bad = tmp_path / "bad.parquet"
    pd.DataFrame(
        {
            "raw_symbol": ["SPX   250220C06000000"],
            "root": ["SPX"],
            "expiry": ["2025-02-20"],
            "settlement_style": ["AM"],
        }
    ).to_parquet(bad)
    with pytest.raises(RuntimeError, match="outside the exact"):
        d55.session_symbols(bad, "2025-02-20")


class _Metadata:
    def __init__(self, costs: list[float]) -> None:
        self.costs = list(costs)
        self.calls: list[dict] = []

    def get_cost(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise RuntimeError(
                "Request exceeds maximum limit of 2,000 symbols (was 2001)"
            )
        return self.costs.pop(0)


class _Client:
    def __init__(self, costs: list[float]) -> None:
        self.metadata = _Metadata(costs)


def _small_plan() -> dict:
    first = ["A"] * 1000
    second = ["B"] * 1001
    return {
        "sessions": [
            {
                "sequence": 1,
                "selection_index": 0,
                "date": "2025-02-20",
                "symbol_count": len(first),
                "symbols_sha256": d55.canonical_sha256(first),
                "symbols": first,
            },
            {
                "sequence": 2,
                "selection_index": 1,
                "date": "2025-02-21",
                "symbol_count": len(second),
                "symbols_sha256": d55.canonical_sha256(second),
                "symbols": second,
            },
        ]
    }


def test_cost_gate_prices_every_exact_session_before_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _small_plan()
    d55.write_json(tmp_path / "acquisition_plan.json", plan)
    monkeypatch.setattr(d55, "load_and_verify_plan", lambda *_: plan)
    client = _Client([10.0, 11.0])
    receipt = d55.cost_gate(
        client=client,
        normalized_dir=tmp_path,
        out_dir=tmp_path,
    )
    assert receipt["gate"] == "PASS"
    assert receipt["estimated_total_usd"] == 21.0
    assert len(client.metadata.calls) == 3
    assert client.metadata.calls[1]["dataset"] == "OPRA.PILLAR"
    assert client.metadata.calls[1]["schema"] == "cbbo-1s"
    assert client.metadata.calls[1]["stype_in"] == "raw_symbol"
    assert json.loads((tmp_path / "cost_receipt.json").read_text())["gate"] == "PASS"


def test_cost_gate_stops_over_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _small_plan()
    d55.write_json(tmp_path / "acquisition_plan.json", plan)
    monkeypatch.setattr(d55, "load_and_verify_plan", lambda *_: plan)
    with pytest.raises(SystemExit, match="STOP"):
        d55.cost_gate(
            client=_Client([20.0, 11.0]),
            normalized_dir=tmp_path,
            out_dir=tmp_path,
        )
    assert json.loads((tmp_path / "cost_receipt.json").read_text())["gate"] == "STOP_OVER_CAP"


def test_partial_stream_billable_bound_accepts_only_interrupted_first_frame(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "day.partial.cbbo-1s.dbn.zst"
    complete = zstd.ZstdCompressor(
        write_content_size=True,
        write_checksum=True,
    ).compress(b"x" * 50_000)
    partial.write_bytes(complete[:-8])

    bound = d55.partial_stream_billable_bound(partial)

    assert bound["eligible_for_byte_bound"] is True
    assert bound["uncompressed_billable_bytes_upper_bound"] == 50_000
    assert bound["first_frame_complete"] is False


def test_reconcile_failures_refreshes_hash_and_uses_vendor_byte_bound(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    out_dir = tmp_path / "audit"
    archive = raw_root / "2025-02-20" / "failed_attempts"
    archive.mkdir(parents=True)
    partial = archive / "attempt01_day.partial.cbbo-1s.dbn.zst"
    compressed = zstd.ZstdCompressor(
        write_content_size=True,
        write_checksum=True,
    ).compress(b"x" * 100_000)
    partial.write_bytes(compressed[:-8])
    d55.write_json(
        out_dir / "failed_download_attempts.json",
        {
            "schema_version": "Protocol101FT2D55FailedDownloadAttemptsV1",
            "attempts": [
                {
                    "date": "2025-02-20",
                    "reserved_cost_usd": 1.25,
                    "archived_partial_files": [
                        {
                            "path": str(partial),
                            "bytes": 0,
                            "sha256": "stale",
                        }
                    ],
                }
            ],
        },
    )

    rows = d55.reconcile_failures(
        raw_root=raw_root,
        out_dir=out_dir,
        unit_price_usd_per_gb=2.0,
    )

    assert rows[0]["archived_partial_files"][0]["bytes"] == partial.stat().st_size
    assert rows[0]["archived_partial_files"][0]["sha256"] == d55.sha256_file(partial)
    assert rows[0]["billable_uncompressed_bytes_upper_bound"] == 100_000
    assert rows[0]["billable_cost_upper_bound_usd"] == pytest.approx(0.0002)
    assert rows[0]["preflight_session_cost_usd"] == 1.25
    assert "reserved_cost_usd" not in rows[0]

    repeated = d55.reconcile_failures(
        raw_root=raw_root,
        out_dir=out_dir,
        unit_price_usd_per_gb=2.0,
    )
    assert repeated[0]["billable_cost_upper_bound_usd"] == pytest.approx(0.0002)


def test_batch_fallback_submits_only_exact_approved_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeStore:
        def to_df(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "symbol": ["SPXW  251209C06000000"],
                    "bid_px_00": [1.0],
                    "ask_px_00": [1.1],
                    "ts_event": [pd.Timestamp("2025-12-09T14:31:00Z")],
                },
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2025-12-09T14:31:00Z")],
                    name="ts_recv",
                ),
            )

    class FakeBatch:
        def __init__(self) -> None:
            self.submission: dict | None = None

        def submit_job(self, **kwargs):
            self.submission = kwargs
            return {"id": "job-exact", "state": "queued"}

        def list_jobs(self, **kwargs):
            return [{"id": "job-exact", "state": "done"}]

        def list_files(self, job_id: str):
            assert job_id == "job-exact"
            return [{"filename": "exact.dbn.zst", "size": 3}]

        def download(self, job_id: str, output_dir: Path, filename_to_download: str):
            path = Path(output_dir) / job_id / filename_to_download
            path.parent.mkdir(parents=True)
            path.write_bytes(b"dbn")
            return [path]

    class FakeClient:
        def __init__(self) -> None:
            self.batch = FakeBatch()

    monkeypatch.setattr(
        db.DBNStore,
        "from_file",
        staticmethod(lambda _: FakeStore()),
    )
    client = FakeClient()
    symbol = "SPXW  251209C06000000"
    manifest = d55._download_one_batch(
        client=client,
        raw_root=tmp_path / "raw",
        out_dir=tmp_path / "audit",
        plan_row={
            "date": "2025-12-09",
            "symbol_count": 1,
            "symbols_sha256": d55.canonical_sha256([symbol]),
            "symbols": [symbol],
            "normalized_path": "owned.parquet",
            "normalized_sha256": "owned-hash",
            "normalized_rows": 1,
        },
        cost_row={
            "start": "2025-12-09T00:00:00+00:00",
            "end": "2025-12-10T00:00:00+00:00",
            "cost_estimate_usd": 0.5,
        },
    )

    request = client.batch.submission
    assert request is not None
    assert request["dataset"] == "OPRA.PILLAR"
    assert request["schema"] == "cbbo-1s"
    assert request["symbols"] == [symbol]
    assert request["stype_in"] == "raw_symbol"
    assert request["delivery"] == "download"
    assert request["map_symbols"] is False
    assert manifest["request"]["batch_job_id"] == "job-exact"


def test_pinned_unit_price_avoids_network_dependency(tmp_path: Path) -> None:
    class NoMetadataCalls:
        def list_unit_prices(self, **kwargs):
            raise AssertionError("pinned resume must not contact metadata")

    class Client:
        metadata = NoMetadataCalls()

    d55.write_json(
        tmp_path / "failed_download_attempts.json",
        {
            "vendor_billing_evidence": {
                "dataset": "OPRA.PILLAR",
                "schema": "cbbo-1s",
                "mode": "historical-streaming",
                "unit_price_usd_per_gb": 2.0,
                "decimal_bytes_per_gb": 1_000_000_000,
                "source": "Databento metadata.list_unit_prices",
            }
        },
    )

    assert d55.pinned_or_current_streaming_unit_price(
        client=Client(),
        out_dir=tmp_path,
    ) == 2.0
