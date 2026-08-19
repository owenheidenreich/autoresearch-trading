"""Parity repair must fill only what it can solve, and label what it filled."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v5.ops.repair_parity_spot import (
    SOURCE_RELAXED,
    SOURCE_STRICT,
    SOURCE_UNAVAILABLE,
    ParityRepairError,
    relaxed_parity_spot,
    repair_session,
    run,
)

SESSION = "2022-06-01"
SPOT = 4100.0


def _rows(minute: str, strikes: list[float], *, paired: bool, spot: float = SPOT):
    out = []
    for strike in strikes:
        for right in ("C", "P"):
            intrinsic = max(0.0, spot - strike) if right == "C" else max(0.0, strike - spot)
            live = paired or right == "C"
            out.append(
                {
                    "event_time": pd.Timestamp(f"2022-06-01 {minute}", tz="America/New_York").tz_convert("UTC"),
                    "strike": strike,
                    "right": right,
                    "bid": intrinsic + 9.0 if live else np.nan,
                    "ask": intrinsic + 11.0 if live else np.nan,
                    "mid": intrinsic + 10.0,
                    "underlying_price": np.nan,
                }
            )
    return out


def _session(minutes_paired: dict[str, bool]) -> pd.DataFrame:
    rows: list[dict] = []
    for minute, paired in minutes_paired.items():
        rows += _rows(minute, [4080.0, 4090.0, 4100.0, 4110.0], paired=paired)
    return pd.DataFrame(rows)


def test_exact_parity_recovers_spot_from_a_single_paired_strike() -> None:
    """European parity is exact per strike; one pair is mathematically enough."""

    frame = pd.DataFrame(_rows("16:00", [4100.0], paired=True))
    spot, used = relaxed_parity_spot(frame)
    assert used == 1
    assert spot == pytest.approx(SPOT, abs=1e-6)


def test_no_paired_strike_yields_nothing_rather_than_a_guess() -> None:
    frame = pd.DataFrame(_rows("16:00", [4100.0], paired=False))
    spot, used = relaxed_parity_spot(frame)
    assert used == 0
    assert np.isnan(spot)


def test_strict_values_are_never_overwritten() -> None:
    frame = _session({"09:31": True})
    frame["underlying_price"] = 1234.5
    out, report = repair_session(frame)
    assert (out["underlying_price"] == 1234.5).all()
    assert (out["underlying_price_source"] == SOURCE_STRICT).all()
    assert report["repaired_minutes"] == 0


def test_an_empty_minute_is_repaired_and_labelled_with_its_strike_count() -> None:
    out, report = repair_session(_session({"09:31": True}))
    assert report["repaired_minutes"] == 1
    assert report["repaired_detail"]["09:31"] == 4
    assert (out["underlying_price_source"] == SOURCE_RELAXED).all()
    assert (out["underlying_parity_strikes"] == 4).all()
    assert out["underlying_price"].iloc[0] == pytest.approx(SPOT, abs=1e-6)


def test_an_unsolvable_minute_stays_missing_and_is_reported() -> None:
    """Nothing is forward-filled: a gap is surfaced, never patched over."""

    out, report = repair_session(_session({"09:31": False}))
    assert report["repaired_minutes"] == 0
    assert report["unrepaired_minutes"] == ["09:31"]
    assert report["clock_complete"] is False
    assert out["underlying_price"].isna().all()
    assert (out["underlying_price_source"] == SOURCE_UNAVAILABLE).all()


def test_repair_does_not_carry_a_value_between_minutes() -> None:
    """A solvable minute must not lend its spot to an unsolvable neighbour."""

    out, report = repair_session(_session({"09:31": True, "09:32": False}))
    stamped = pd.to_datetime(out["event_time"], utc=True).dt.tz_convert("America/New_York")
    later = out[stamped.dt.strftime("%H:%M").eq("09:32")]
    assert later["underlying_price"].isna().all()
    assert report["unrepaired_minutes"] == ["09:32"]


def test_an_empty_source_root_is_refused(tmp_path) -> None:
    with pytest.raises(ParityRepairError, match="no normalized sessions"):
        run(
            source_root=tmp_path,
            output_root=tmp_path / "out",
            receipt_path=tmp_path / "receipt.json",
        )


def test_existing_outputs_are_skipped_so_the_pass_resumes(tmp_path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _session({"09:31": True}).to_parquet(source / f"databento_spxw_0dte_{SESSION}.parquet")
    out = tmp_path / "out"
    out.mkdir()
    (out / f"databento_spxw_0dte_{SESSION}.parquet").write_bytes(b"done")

    payload = run(
        source_root=source, output_root=out, receipt_path=tmp_path / "receipt.json"
    )
    assert payload["sessions"][0]["classification"] == "ALREADY_PRESENT"
