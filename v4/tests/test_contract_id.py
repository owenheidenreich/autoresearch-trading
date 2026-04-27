"""Tests for canonical contract ID parser."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from v4.parser import ContractId, from_columns, parse
from v4.schema.types import ContractRoot, OptionRight


def test_canonical_format_round_trip() -> None:
    cid = ContractId(
        root=ContractRoot.SPXW,
        expiry=date(2023, 1, 2),
        strike=Decimal("4000.000"),
        right=OptionRight.CALL,
    )
    s = cid.to_canonical()
    assert s == "SPXW-20230102-04000.000-C"
    assert parse(s) == cid


def test_occ21_round_trip() -> None:
    cid = ContractId(
        root=ContractRoot.SPXW,
        expiry=date(2023, 1, 2),
        strike=Decimal("4000.000"),
        right=OptionRight.CALL,
    )
    occ = cid.to_occ21()
    # SPXW + 230102 + C + 04000000 → "SPXW  230102C04000000"
    assert occ == "SPXW  230102C04000000"
    assert parse(occ) == cid


def test_parse_occ21_no_spaces() -> None:
    cid = parse("SPX230120P03950000")
    assert cid.root == ContractRoot.SPX
    assert cid.expiry == date(2023, 1, 20)
    assert cid.strike == Decimal("3950.000")
    assert cid.right == OptionRight.PUT


def test_from_columns_optionsdx_style() -> None:
    cid = from_columns(
        root="SPXW",
        expiry=date(2023, 1, 2),
        strike=4000.0,
        right="C",
    )
    assert cid.to_canonical() == "SPXW-20230102-04000.000-C"


def test_from_columns_strike_string() -> None:
    cid = from_columns(
        root=ContractRoot.SPX,
        expiry=date(2023, 6, 15),
        strike="4250.5",
        right=OptionRight.PUT,
    )
    assert cid.strike == Decimal("4250.5")
    assert cid.to_canonical() == "SPX-20230615-04250.500-P"


def test_canonical_lexicographic_order_by_date_then_strike() -> None:
    a = ContractId(ContractRoot.SPXW, date(2023, 1, 2), Decimal("3950.000"), OptionRight.CALL)
    b = ContractId(ContractRoot.SPXW, date(2023, 1, 2), Decimal("4000.000"), OptionRight.CALL)
    c = ContractId(ContractRoot.SPXW, date(2023, 1, 3), Decimal("3900.000"), OptionRight.CALL)
    assert a.to_canonical() < b.to_canonical() < c.to_canonical()


def test_negative_strike_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ContractId(ContractRoot.SPX, date(2023, 1, 2), Decimal("-1"), OptionRight.CALL)


def test_excessive_strike_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ContractId(ContractRoot.SPX, date(2023, 1, 2), Decimal("200000"), OptionRight.CALL)


def test_parse_unknown_format() -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        parse("not_an_option_id")


def test_parse_empty_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse("")


def test_contractid_is_hashable() -> None:
    cid = from_columns("SPXW", date(2023, 1, 2), 4000.0, "C")
    s = {cid}
    assert cid in s


def test_pre_2070_year_disambiguation() -> None:
    """OCC convention: yy < 70 → 2000s, yy >= 70 → 1900s.

    SPX 0DTE didn't exist before 2005 so this only matters for the parser
    being correct, not for any real legacy data.
    """
    cid_2025 = parse("SPX250101C04000000")
    assert cid_2025.expiry == date(2025, 1, 1)
    cid_1995 = parse("SPX950101C04000000")
    assert cid_1995.expiry == date(1995, 1, 1)
