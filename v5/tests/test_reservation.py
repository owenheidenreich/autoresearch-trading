"""The signed forward reservation must be enforced by code, not by memory."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from v5.research import reservation as rv


ROOT = Path(__file__).resolve().parents[2]


def test_the_declaration_it_cites_exists_and_is_signed() -> None:
    text = (ROOT / rv.DECLARATION).read_text(errors="ignore")
    assert "SIGNED AND IN FORCE" in text
    assert rv.RESERVATION_START in text


def test_the_boundary_is_the_declared_date() -> None:
    assert rv.is_reserved("2026-08-06")
    assert rv.is_reserved("2027-01-04")
    assert not rv.is_reserved("2026-08-05")
    assert not rv.is_reserved("2026-07-31")


def test_a_development_index_passes_untouched() -> None:
    frame = pd.DataFrame(
        {"session": ["2026-07-30", "2026-07-31", "2026-08-05"], "net_pnl": [1.0, 2.0, 3.0]}
    )
    result = rv.development_frame(frame, purpose="G1 screen")
    pd.testing.assert_frame_equal(result, frame)


def test_a_reserved_session_stops_the_run_and_says_what_was_refused() -> None:
    frame = pd.DataFrame(
        {"session": ["2026-08-05", "2026-08-06"], "net_pnl": [1.0, 2.0]}
    )
    with pytest.raises(rv.ReservationError) as excinfo:
        rv.development_frame(frame, purpose="G1 economic replay")
    message = str(excinfo.value)
    assert "G1 economic replay" in message
    assert "2026-08-06" in message
    assert rv.DECLARATION in message


def test_reserved_rows_are_refused_rather_than_silently_dropped() -> None:
    """Filtering would let a run continue on a quietly different index."""

    sessions = ["2026-08-04", "2026-08-06", "2026-08-07"]
    with pytest.raises(rv.ReservationError):
        rv.assert_development_only(sessions, purpose="anything")
    assert rv.reserved_sessions(sessions) == ("2026-08-06", "2026-08-07")


def test_the_owned_corpus_is_entirely_development_data() -> None:
    """The G1 index must sit wholly on the development side of the firewall."""

    corpus = Path(
        "/Users/gduby/.autoresearch-trading/pathd_2025-08-01_2026-07-31"
        "/raw/databento/glbx_es_ohlcv_1m"
    )
    if not corpus.is_dir():
        pytest.skip("owned corpus not mounted")
    sessions = [path.name.split(".")[0] for path in corpus.glob("*.parquet")]
    assert sessions, "corpus is present but empty"
    rv.assert_development_only(sessions, purpose="the owned ES corpus")
