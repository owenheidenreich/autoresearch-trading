"""Canonical SPX / SPXW contract ID parser.

Supports parsing multiple vendor formats (OCC-21, OptionsDX columnar, Databento
definition records) into a single canonical form, and serializing back.

Canonical form (used everywhere in v4):

    {root}-{YYYYMMDD}-{strike:0>9.3f}-{right}

Examples:

    SPXW-20230102-04000.000-C
    SPX-20230120-03950.000-P
    SPY-20230106-00400.000-C

This form is human-readable, lexicographically orderable by (date, strike),
and round-trip parseable. The strike padding (9 chars total, 3 decimals)
covers SPX strikes up to $99,999.999 — comfortable for the foreseeable
SPX range.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from v4.schema.types import ContractRoot, OptionRight


# OCC-21 format: 6-char root padded with spaces, yymmdd, C/P, strike*1000 padded to 8 digits.
# In stored data, the space-padded version is sometimes written without spaces.
OCC21_RE = re.compile(
    r"^(?P<root>[A-Z ]{1,6}?)\s*"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<right>[CP])"
    r"(?P<strike_thousands>\d{8})$"
)

CANONICAL_RE = re.compile(
    r"^(?P<root>[A-Z]+)-"
    r"(?P<date>\d{8})-"
    r"(?P<strike>\d{1,5}\.\d{3})-"
    r"(?P<right>[CP])$"
)


@dataclass(frozen=True)
class ContractId:
    """Canonical option-contract identity.

    Frozen + hashable so it can be used as a dict key or set member.
    """

    root: ContractRoot
    expiry: date
    strike: Decimal
    right: OptionRight

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"strike must be positive; got {self.strike}")
        # Strike must be representable in 5+3 decimal places
        if self.strike >= Decimal("100000.000"):
            raise ValueError(f"strike out of range for SPX-family: {self.strike}")

    def __str__(self) -> str:
        return self.to_canonical()

    def to_canonical(self) -> str:
        """Render as canonical string: ROOT-YYYYMMDD-STRIKE.padded-RIGHT."""
        return (
            f"{self.root.value}-"
            f"{self.expiry.strftime('%Y%m%d')}-"
            f"{self.strike:09.3f}-"
            f"{self.right.value}"
        )

    def to_occ21(self) -> str:
        """Render as OCC-21 (no spaces, root left-aligned to 6 chars).

        Used by Databento definition records. Strike is multiplied by 1000
        and padded to 8 digits.
        """
        root_padded = self.root.value.ljust(6)
        strike_thousands = int(self.strike * 1000)
        return (
            f"{root_padded}"
            f"{self.expiry.strftime('%y%m%d')}"
            f"{self.right.value}"
            f"{strike_thousands:08d}"
        )


def parse(s: str) -> ContractId:
    """Parse any supported format into a ContractId.

    Tries canonical form first, then OCC-21. Raises ValueError on failure.
    """
    s = s.strip()
    if not s:
        raise ValueError("empty contract id")

    # Try canonical form first
    if m := CANONICAL_RE.match(s):
        root = ContractRoot(m.group("root"))
        expiry = date(
            int(m.group("date")[:4]),
            int(m.group("date")[4:6]),
            int(m.group("date")[6:8]),
        )
        strike = Decimal(m.group("strike"))
        right = OptionRight(m.group("right"))
        return ContractId(root=root, expiry=expiry, strike=strike, right=right)

    # Try OCC-21 form
    if m := OCC21_RE.match(s):
        root_raw = m.group("root").strip()
        try:
            root = ContractRoot(root_raw)
        except ValueError as e:
            raise ValueError(f"unknown root in OCC-21 id {s!r}: {root_raw!r}") from e
        yy = int(m.group("yy"))
        # Two-digit year disambiguation: OCC convention treats 70-99 as 1970s-1990s,
        # 00-69 as 2000s-2060s. SPX 0DTE didn't exist before 2005, so this is safe.
        full_year = 2000 + yy if yy < 70 else 1900 + yy
        expiry = date(full_year, int(m.group("mm")), int(m.group("dd")))
        strike = Decimal(m.group("strike_thousands")) / Decimal(1000)
        right = OptionRight(m.group("right"))
        return ContractId(root=root, expiry=expiry, strike=strike, right=right)

    raise ValueError(f"unrecognized contract id format: {s!r}")


def from_columns(
    root: str | ContractRoot,
    expiry: date,
    strike: float | Decimal | str,
    right: str | OptionRight,
) -> ContractId:
    """Build a ContractId from separate columnar fields (OptionsDX style)."""
    root_e = ContractRoot(root) if isinstance(root, str) else root
    right_e = OptionRight(right) if isinstance(right, str) else right
    if isinstance(strike, float):
        # Convert via str to avoid binary float artifacts
        strike_d = Decimal(str(strike))
    elif isinstance(strike, str):
        strike_d = Decimal(strike)
    else:
        strike_d = strike
    return ContractId(root=root_e, expiry=expiry, strike=strike_d, right=right_e)
