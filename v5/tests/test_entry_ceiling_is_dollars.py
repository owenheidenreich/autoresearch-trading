"""The entry ceiling is a fixed dollar amount and the moneyness band is separate.

Both guards were ruled load-bearing by the owner on 2026-08-16 after the scale
sensitivity study. These tests exist so a later refactor cannot quietly
reintroduce either defect:

- an equity-derived ceiling, which is inert only while equity is frozen and
  drifts into deep ITM the moment live equity is threaded through; and
- the belief that a price cap implies an out-of-the-money contract, which is
  false — 37.3% of contracts cheap enough to clear $2,000 are in the money.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest

from v5.ops import audit_causal_day_coverage as coverage
from v5.ops.audit_causal_day_coverage import (
    ENTRY_FEES_USD,
    MAX_ENTRY_TICKET_USD,
    eligible_entry,
)

EQUITY_TOKENS = (
    "SESSION_START_EQUITY",
    "TICKET_CEILING_SHARE",
    "equity",
    "account_usd",
    "starting_equity",
)


def _row(*, ask: float, money: float, spot: float = 5000.0, right: str = "C") -> pd.DataFrame:
    """One ladder row at a chosen ask and signed moneyness."""

    strike = spot - money if right == "C" else spot + money
    return pd.DataFrame(
        {
            "underlying_price": [spot],
            "strike": [strike],
            "right": [right],
            "ask": [ask],
            "bid": [ask - 0.05],
            "ask_size": [10.0],
            "quote_age_ms": [0.0],
        }
    )


# --------------------------------------------------------- dollar ceiling


def test_ceiling_is_the_signed_dollar_amount() -> None:
    assert MAX_ENTRY_TICKET_USD == 2_000.0


def test_equity_constants_are_deleted_not_merely_unused() -> None:
    """The owner's instruction: delete the expression, don't leave it computed."""

    for name in ("SESSION_START_EQUITY_USD", "TICKET_CEILING_SHARE"):
        assert not hasattr(coverage, name), f"{name} must not exist"


def _referenced_names(source: str) -> set[str]:
    """Identifiers the code actually reads, ignoring docstrings and comments.

    A textual scan would flag prose that *forbids* equity, so this parses the
    code and inspects real name references instead.
    """

    tree = ast.parse(textwrap.dedent(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_no_equity_reference_anywhere_in_the_eligibility_path() -> None:
    """The ceiling must be structurally incapable of drifting with the account."""

    referenced = _referenced_names(inspect.getsource(eligible_entry))
    for token in EQUITY_TOKENS:
        assert token not in referenced, f"eligibility reads {token!r}"

    # The ceiling constants must be literals, not expressions over anything.
    module = ast.parse(inspect.getsource(coverage))
    ceilings = {
        target.id: node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in ("MAX_ENTRY_TICKET_USD", "MAX_ENTRY_ASK_USD")
    }
    assert "MAX_ENTRY_TICKET_USD" in ceilings, "the ticket ceiling must be module-level"
    assert isinstance(
        ceilings["MAX_ENTRY_TICKET_USD"], ast.Constant
    ), "the ticket ceiling must be a literal, not a computed expression"
    for name, value in ceilings.items():
        for token in EQUITY_TOKENS:
            assert token not in _referenced_names(
                ast.unparse(value)
            ), f"{name} derives from {token!r}"


def test_ceiling_charges_fees_as_the_amendment_states() -> None:
    """The signed law is 'entry premium plus fees at most $2,000'."""

    just_under = (MAX_ENTRY_TICKET_USD - ENTRY_FEES_USD) / 100.0
    assert bool(eligible_entry(_row(ask=just_under - 0.01, money=-5.0)).iloc[0])
    # A ticket whose premium alone is $2,000 breaches once fees are added.
    assert not bool(eligible_entry(_row(ask=20.00, money=-5.0)).iloc[0])


def test_expensive_otm_contract_is_refused() -> None:
    assert not bool(eligible_entry(_row(ask=25.00, money=-5.0)).iloc[0])


# ------------------------------------------------- moneyness is separate


def test_a_cheap_in_the_money_contract_is_still_refused() -> None:
    """Price and moneyness are different axes; the cap does not imply OTM.

    Measured 2026-08-16: 37.3% of contracts under a $2,000 ticket are in the
    money, 99th percentile +18.4 points. If this test ever fails, the two
    guards have been collapsed into one.
    """

    cheap_itm = _row(ask=2.00, money=+5.0)  # $200 ticket, 5 points in the money
    assert not bool(eligible_entry(cheap_itm).iloc[0])


def test_a_cheap_deep_in_the_money_contract_is_refused() -> None:
    assert not bool(eligible_entry(_row(ask=1.00, money=+20.0)).iloc[0])


def test_an_otm_contract_beyond_the_band_is_refused() -> None:
    assert not bool(eligible_entry(_row(ask=1.00, money=-40.0)).iloc[0])


def test_the_two_guards_are_independently_necessary() -> None:
    """Neither guard alone admits only what the charter permits."""

    cheap_and_otm = _row(ask=4.00, money=-5.0)
    cheap_but_itm = _row(ask=4.00, money=+5.0)
    dear_but_otm = _row(ask=30.00, money=-5.0)

    assert bool(eligible_entry(cheap_and_otm).iloc[0])
    assert not bool(eligible_entry(cheap_but_itm).iloc[0]), "moneyness guard is load-bearing"
    assert not bool(eligible_entry(dear_but_otm).iloc[0]), "dollar guard is load-bearing"


def test_eligibility_is_unchanged_by_any_notion_of_account_size() -> None:
    """Same row, same verdict — there is no account argument to vary."""

    signature = inspect.signature(eligible_entry)
    assert list(signature.parameters) == ["frame"]
    row = _row(ask=15.00, money=-5.0)
    assert bool(eligible_entry(row).iloc[0]) is bool(eligible_entry(row).iloc[0])


# ------------------------------------------------- risk simulator ceiling


def test_risk_simulator_ceiling_is_dollars_not_a_share() -> None:
    from v5.ops.check_occupancy_risk import CHARTER_PREMIUM_CEILING_USD, simulate

    assert CHARTER_PREMIUM_CEILING_USD == 2_000.0
    parameters = inspect.signature(simulate).parameters
    assert "premium_ceiling_usd" in parameters
    assert "premium_ceiling" not in parameters, "the share parameter must be gone"

    # A ticket above the dollar ceiling stays unaffordable however large the
    # account grows — the drift the study warned about.
    rng = np.random.default_rng(0)
    result = simulate(
        trades_per_session=1,
        win_quantiles_usd=np.array([100.0] * 60),
        loss_quantiles_usd=np.array([-100.0] * 60),
        mean_premium_usd=5_000.0,
        accuracy=0.5,
        account_usd=1_000_000.0,
        sessions=5,
        paths=50,
        rng=rng,
    )
    assert result["affordable_under_the_charter_ceiling"] is False
