"""The SPX tape must be causal, complete, and honest about what it is not."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v5.ops.build_causal_day_dataset import prepare_es
from v5.ops.build_parity_spot_candles import (
    BAR_MINUTES,
    TAPE_SOURCE,
    ParitySpotCandleError,
    parity_spot_candles,
    run,
)
from v5.ops.audit_causal_day_coverage import QUOTE_MINUTES

SESSION = "2024-03-15"


def _spot(*, missing: tuple[str, ...] = (), start: float = 5000.0) -> pd.Series:
    values = start + np.arange(len(QUOTE_MINUTES), dtype=float) * 0.25
    series = pd.Series(values, index=list(QUOTE_MINUTES), dtype=float)
    for minute in missing:
        series[minute] = np.nan
    return series


def test_the_grid_is_the_full_es_session_clock() -> None:
    assert BAR_MINUTES[0] == "09:30"
    assert BAR_MINUTES[-1] == "15:59"
    assert len(BAR_MINUTES) == 390


def test_a_bar_is_filled_from_the_snapshot_one_minute_later() -> None:
    """The causal offset, asserted rather than commented.

    An ES bar stamped `t` is knowable at `t+1`. A quote snapshot stamped `t` is
    the market at `t`. Bar `t` therefore takes the `t+1` snapshot, so it stays
    first-readable one minute after the state it describes.
    """

    spot = _spot()
    candles = parity_spot_candles(spot, SESSION)
    assert list(candles["bar_observation_minute"][:3]) == ["09:31", "09:32", "09:33"]
    assert candles["bar_observation_minute"].iloc[-1] == "16:00"
    assert float(candles["close"].iloc[0]) == pytest.approx(float(spot["09:31"]))
    assert float(candles["close"].iloc[-1]) == pytest.approx(float(spot["16:00"]))


def test_open_high_and_low_equal_the_close_rather_than_an_invented_range() -> None:
    candles = parity_spot_candles(_spot(), SESSION)
    for column in ("open", "high", "low"):
        np.testing.assert_allclose(candles[column].to_numpy(), candles["close"].to_numpy())
    assert (candles["volume"] == 0.0).all()
    assert (candles["tape_source"] == TAPE_SOURCE).all()


def test_a_missing_minute_is_refused_rather_than_carried_forward() -> None:
    with pytest.raises(ParitySpotCandleError, match="parity spot absent"):
        parity_spot_candles(_spot(missing=("11:20",)), SESSION)


def test_a_non_positive_spot_is_refused() -> None:
    spot = _spot()
    spot["10:00"] = -1.0
    with pytest.raises(ParitySpotCandleError, match="non-positive"):
        parity_spot_candles(spot, SESSION)


def test_the_pinned_es_reader_accepts_the_emitted_shape(tmp_path: Path) -> None:
    """The whole point of the file shape: `build_session` must need no edit.

    `prepare_es` is inside the semantic freeze. If it accepts this frame, the
    tape swaps out without touching a pinned line.
    """

    path = tmp_path / f"{SESSION}.parquet"
    parity_spot_candles(_spot(), SESSION).to_parquet(path)
    prepared = prepare_es(pd.read_parquet(path), SESSION)
    assert len(prepared) == 390
    assert prepared["bar_minute"].iloc[0] == "09:30"
    assert prepared["bar_minute"].iloc[-1] == "15:59"
    assert prepared["knowable_at"].iloc[0] == "09:31"
    assert prepared["knowable_at"].iloc[-1] == "16:00"


def test_the_run_receipt_reports_a_failure_instead_of_swallowing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quotes = tmp_path / "quotes"
    quotes.mkdir()
    (quotes / f"databento_spxw_0dte_{SESSION}.parquet").touch()

    import v5.ops.build_parity_spot_candles as module

    monkeypatch.setattr(module, "parity_spot_by_minute", lambda path, session: _spot())
    receipt = tmp_path / "receipt.json"
    payload = run(
        quote_roots=[quotes],
        sessions=[SESSION, "2024-03-16"],
        out_dir=tmp_path / "tape",
        receipt_path=receipt,
    )
    assert payload["gate"] == "SESSIONS_FAILED"
    assert payload["summary"] == {
        **payload["summary"],
        "built": 1,
        "failed": 1,
        "already_present": 0,
    }
    written = json.loads(receipt.read_text())
    failed = [row for row in written["sessions"] if row["classification"] == "FAILED"]
    assert failed and "no quote file" in failed[0]["reason"]


def test_a_second_pass_skips_what_it_already_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quotes = tmp_path / "quotes"
    quotes.mkdir()
    (quotes / f"databento_spxw_0dte_{SESSION}.parquet").touch()

    import v5.ops.build_parity_spot_candles as module

    monkeypatch.setattr(module, "parity_spot_by_minute", lambda path, session: _spot())
    kwargs = dict(
        quote_roots=[quotes],
        sessions=[SESSION],
        out_dir=tmp_path / "tape",
        receipt_path=tmp_path / "receipt.json",
    )
    assert run(**kwargs)["summary"]["built"] == 1
    again = run(**kwargs)
    assert again["summary"]["built"] == 0
    assert again["summary"]["already_present"] == 1
    assert again["gate"] == "PASS"
