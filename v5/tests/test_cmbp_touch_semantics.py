"""The signing law, pinned. Structural tests run everywhere; corpus tests skip if absent."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from v5.ops.verify_cmbp_touch_semantics import (
    MIN_SIGNED_SHARE,
    CmbpSemanticError,
    build_receipt,
    classify_session,
    verify_parquet,
)

SLICE = Path("data/raw/audit/protocol101_highres_opra/cmbp-1")
needs_slice = pytest.mark.skipif(not SLICE.is_dir(), reason="owned cmbp-1 slice not present")


def _rows(rows):
    return pd.DataFrame(rows, columns=["instrument_id", "ts_event", "action", "side",
                                       "price", "size", "bid_px_00", "ask_px_00",
                                       "symbol", "publisher_id"])


def _quote(i, t, bid, ask):
    return (1, t, "A", "B", float("nan"), 0, bid, ask, "X", 1)


def test_a_print_at_the_prior_ask_is_buyer_initiated() -> None:
    f = _rows([_quote(1, 1, 1.00, 1.20), (1, 2, "T", "N", 1.20, 5, 1.00, 1.20, "X", 1)])
    r = classify_session(f, "s")
    assert r.at_ask_buyer_initiated == 1 and r.at_bid_seller_initiated == 0
    assert r.signed == 1 and r.signed_share == 1.0


def test_a_print_at_the_prior_bid_is_seller_initiated() -> None:
    f = _rows([_quote(1, 1, 1.00, 1.20), (1, 2, "T", "N", 1.00, 5, 1.00, 1.20, "X", 1)])
    r = classify_session(f, "s")
    assert r.at_bid_seller_initiated == 1 and r.at_ask_buyer_initiated == 0


def test_a_print_inside_the_touch_is_ambiguous_and_is_not_guessed() -> None:
    """44% of the real tape lands here. Guessing it would be the whole defect."""

    f = _rows([_quote(1, 1, 1.00, 1.20), (1, 2, "T", "N", 1.10, 5, 1.00, 1.20, "X", 1)])
    r = classify_session(f, "s")
    assert r.inside_ambiguous == 1
    assert r.signed == 0, "an inside print must never be signed"


def test_the_prior_touch_is_used_not_the_trade_rows_own_book() -> None:
    """The signing law is 'strictly before'. This fails if the row's own book is used."""

    f = _rows([
        _quote(1, 1, 1.00, 1.20),
        # Trade prints at 1.20 == the PRIOR ask, while its own row shows a moved book.
        (1, 2, "T", "N", 1.20, 5, 1.30, 1.50, "X", 1),
    ])
    r = classify_session(f, "s")
    assert r.at_ask_buyer_initiated == 1, "must sign against the prior touch, not the row's own"


def test_a_locked_prior_book_is_excluded_from_signing() -> None:
    f = _rows([_quote(1, 1, 1.10, 1.10), (1, 2, "T", "N", 1.10, 5, 1.10, 1.10, "X", 1)])
    r = classify_session(f, "s")
    assert r.locked_prior == 1 and r.signed == 0


def test_a_crossed_prior_book_is_excluded_from_signing() -> None:
    f = _rows([_quote(1, 1, 1.30, 1.20), (1, 2, "T", "N", 1.20, 5, 1.30, 1.20, "X", 1)])
    r = classify_session(f, "s")
    assert r.crossed_prior == 1 and r.signed == 0


def test_a_trade_with_no_prior_quote_is_counted_not_dropped() -> None:
    f = _rows([(1, 1, "T", "N", 1.20, 5, float("nan"), float("nan"), "X", 1)])
    r = classify_session(f, "s")
    assert r.no_prior_quote == 1 and r.trades == 1 and r.signed == 0


def test_instruments_do_not_leak_into_each_others_touch() -> None:
    """Shifting without grouping would sign instrument 2 off instrument 1's book."""

    f = _rows([_quote(1, 1, 1.00, 1.20), (2, 2, "T", "N", 1.20, 5, 9.00, 9.20, "X", 1)])
    r = classify_session(f, "s")
    assert r.no_prior_quote == 1 and r.signed == 0


def test_an_empty_or_malformed_slice_fails_closed() -> None:
    with pytest.raises(CmbpSemanticError, match="no rows"):
        classify_session(_rows([]), "s")
    with pytest.raises(CmbpSemanticError, match="missing required fields"):
        classify_session(_rows([_quote(1, 1, 1.0, 1.2)]).drop(columns=["bid_px_00"]), "s")
    with pytest.raises(CmbpSemanticError, match="no parquet"):
        verify_parquet(Path("/nonexistent/x.parquet"), "s")


def test_the_receipt_states_it_read_no_outcome_and_charged_no_alpha() -> None:
    f = _rows([_quote(1, 1, 1.00, 1.20), (1, 2, "T", "N", 1.20, 5, 1.00, 1.20, "X", 1)])
    r = build_receipt([classify_session(f, "s")], manifest_sha256="abc")
    assert r["reads_no_outcome"] is True and r["alpha_charged"] is False
    assert r["verdict"] == "SEMANTICS_PASS_ONLY"
    assert "parser possibility only" in r["caveat"]


def test_an_unsignable_slice_stops_rather_than_passing() -> None:
    rows = [_quote(1, 1, 1.00, 1.20)]
    rows += [(1, i + 2, "T", "N", 1.10, 5, 1.00, 1.20, "X", 1) for i in range(20)]
    r = build_receipt([classify_session(_rows(rows), "s")], manifest_sha256="abc")
    assert r["signed_share"] < MIN_SIGNED_SHARE
    assert r["verdict"] == "SEMANTIC_STOP_UNSIGNABLE"


@needs_slice
def test_the_owned_slice_signs_a_real_share_of_its_tape() -> None:
    r = verify_parquet(SLICE / "2024-10-01.cmbp-1.parquet", "2024-10-01")
    assert r.trades > 10_000
    assert r.locked_prior == 0 and r.crossed_prior == 0
    assert 0.40 < r.signed_share < 0.70, f"signed share drifted: {r.signed_share:.3f}"
    assert r.inside_ambiguous > 0, "a real tape always has inside prints"
