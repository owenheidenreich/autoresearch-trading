"""The resumable normalization driver must reuse frozen semantics, provably."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from v5.ops import normalize_lifecycle_corpus as driver
from v5.ops.normalize_lifecycle_quote_backfill import QuoteNormalizationError

SESSION = "2024-03-15"


def _receipt(tmp_path: Path, sessions: list[str]) -> Path:
    path = tmp_path / "acquisition_receipt.json"
    path.write_text(
        json.dumps(
            {
                # Deliberately without `completion.expected_files`: this is the
                # V5 receipt shape the pinned run() refuses.
                "completion": {"sessions": len(sessions), "recorded_files": 2 * len(sessions)},
                "files": [
                    {"session": s, "path": f"/x/{s}.{kind}.parquet"}
                    for s in sessions
                    for kind in ("definition", "cbbo-1m")
                ],
            }
        )
    )
    return path


def _valid_raw_root(tmp_path: Path, sessions: tuple[str, ...] = ("2024-03-01", "2024-03-04")) -> Path:
    """A source root that passes the 389-bar guard, for orchestration tests."""

    from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES

    root = tmp_path / "raw"
    for session in sessions:
        _cbbo(root, session, list(QUOTE_MINUTES))
    return root


def test_sessions_are_read_from_the_v5_receipt_shape(tmp_path: Path) -> None:
    receipt = json.loads(_receipt(tmp_path, ["2024-03-15", "2024-03-18"]).read_text())
    assert driver.acquired_sessions(receipt) == ["2024-03-15", "2024-03-18"]


def test_a_receipt_naming_no_sessions_is_refused() -> None:
    with pytest.raises(QuoteNormalizationError, match="names no sessions"):
        driver.acquired_sessions({"files": []})


def test_frozen_semantics_are_verified_before_any_work(tmp_path: Path) -> None:
    """A drifted normalizer must stop the run, not silently reshape the corpus."""

    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"source_files": {driver.PINNED_SOURCE: "0" * 64}})
    )
    with pytest.raises(QuoteNormalizationError, match="frozen normalizer drifted"):
        driver.assert_frozen_semantics(freeze)


def test_the_real_normalizer_matches_its_frozen_hash() -> None:
    """Guards the actual repository state, not a fixture."""

    assert len(driver.assert_frozen_semantics()) == 64


def test_a_freeze_that_does_not_pin_the_normalizer_is_refused(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"source_files": {}}))
    with pytest.raises(QuoteNormalizationError, match="does not pin"):
        driver.assert_frozen_semantics(freeze)


def test_existing_outputs_are_skipped_so_a_resumed_pass_is_cheap(tmp_path: Path) -> None:
    """The defect this driver exists to fix: a rerun must not degrade prior work."""

    out = tmp_path / "out"
    out.mkdir()
    (out / f"databento_spxw_0dte_{SESSION}.parquet").write_bytes(b"already-normalized")

    payload = driver.run(
        raw_root=_valid_raw_root(tmp_path),
        output_root=out,
        acquisition_receipt=_receipt(tmp_path, [SESSION]),
        receipt_path=tmp_path / "receipt.json",
    )

    assert payload["summary"]["already_present"] == 1
    assert payload["summary"]["degraded"] == 0
    assert payload["gate"] == "PASS"
    assert payload["sessions"][0]["classification"] == "ALREADY_PRESENT"


def test_a_missing_source_is_recorded_as_degraded_not_raised(tmp_path: Path) -> None:
    payload = driver.run(
        raw_root=_valid_raw_root(tmp_path),
        output_root=tmp_path / "out",
        acquisition_receipt=_receipt(tmp_path, [SESSION]),
        receipt_path=tmp_path / "receipt.json",
    )
    assert payload["gate"] == "DEGRADED_SESSIONS_PRESENT"
    assert payload["sessions"][0]["classification"] == "DEGRADED"


def test_the_receipt_records_the_frozen_hash_and_self_hashes(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    payload = driver.run(
        raw_root=_valid_raw_root(tmp_path),
        output_root=tmp_path / "out",
        acquisition_receipt=_receipt(tmp_path, [SESSION]),
        receipt_path=receipt_path,
    )
    written = json.loads(receipt_path.read_text())
    assert written["frozen_normalizer"]["path"] == driver.PINNED_SOURCE
    assert len(written["frozen_normalizer"]["sha256"]) == 64
    assert written["receipt_sha256"] == payload["receipt_sha256"]


def test_an_interrupted_write_leaves_no_output_a_resume_would_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging then rename: a crash mid-write must not look like completed work."""

    raw = _valid_raw_root(tmp_path)
    (raw / "raw/databento/opra_spxw_definition").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"a": [1]}).to_parquet(
        raw / f"raw/databento/opra_spxw_definition/{SESSION}.definition.parquet"
    )
    from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES

    _cbbo(raw, SESSION, list(QUOTE_MINUTES))

    def explode(*_args, **_kwargs):
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(driver, "normalize_session", explode)
    payload = driver.run(
        raw_root=raw,
        output_root=tmp_path / "out",
        acquisition_receipt=_receipt(tmp_path, [SESSION]),
        receipt_path=tmp_path / "receipt.json",
    )

    assert payload["sessions"][0]["classification"] == "DEGRADED"
    assert not (tmp_path / "out" / f"databento_spxw_0dte_{SESSION}.parquet").exists()


def _cbbo(root: Path, session: str, minutes: list[str]) -> None:
    directory = root / "raw/databento/opra_spxw_cbbo_1m"
    directory.mkdir(parents=True, exist_ok=True)
    stamps = (
        pd.to_datetime([f"2024-03-15 {m}" for m in minutes])
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    pd.DataFrame({"ts_recv": stamps, "symbol": ["SPXW  240315C05000000"] * len(minutes)}).to_parquet(
        directory / f"{session}.cbbo-1m.parquet", index=False
    )


def test_the_superseded_389_bar_root_is_refused(tmp_path: Path) -> None:
    """Two 1.7GB corpora differ by one bar; the wrong one must be unusable.

    Naming the old root helps a reader. This is what stops a later step from
    building from it silently.
    """

    from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES

    root = tmp_path / "superseded"
    short = [m for m in QUOTE_MINUTES if m != "16:00"]
    for day in range(1, 5):
        _cbbo(root, f"2024-03-0{day}", short)

    with pytest.raises(QuoteNormalizationError, match="superseded 389-bar acquisition"):
        driver.assert_source_clock(root)


def test_the_fixed_root_is_accepted(tmp_path: Path) -> None:
    from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES

    root = tmp_path / "fixed"
    for day in range(1, 5):
        _cbbo(root, f"2024-03-0{day}", list(QUOTE_MINUTES))

    report = driver.assert_source_clock(root)
    assert report["terminal_minute_share"] == 1.0


def test_a_few_genuine_early_closes_do_not_refuse_a_good_root(tmp_path: Path) -> None:
    """The guard is a majority test: real early closes lack 16:00 legitimately."""

    from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES

    root = tmp_path / "mostly_good"
    for day in range(1, 10):
        _cbbo(root, f"2024-03-{day:02d}", list(QUOTE_MINUTES))
    _cbbo(root, "2024-03-10", [m for m in QUOTE_MINUTES if m <= "13:00"])

    assert driver.assert_source_clock(root)["terminal_minute_share"] > 0.6
