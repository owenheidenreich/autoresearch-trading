"""Single provisional fill-law implementation for Path-D offline research."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from v4.path_d.contracts import ContractIdentityV1, ExecutionIntentV1
from v4.research import pathd_entry_exit as prereg
from v4.research.pathd_entry_dataset import VerifiedResearchQuoteRowV1


def _contract_dict(contract: ContractIdentityV1) -> dict[str, Any]:
    if type(contract) is not ContractIdentityV1:
        raise TypeError("research quote contract must be ContractIdentityV1")
    return {field.name: getattr(contract, field.name) for field in fields(contract)}


@dataclass(frozen=True)
class ResearchFillLawV1:
    SCHEMA_VERSION = "pathd.research_fill_law.v1"

    schema_version: str
    fill_law_hash: str
    headline_delay_ms: int
    delay_sensitivity_ms: tuple[int, ...]
    paired_quote_latency_bounds_ms: tuple[int, ...]
    entry_fee_micros: int
    exit_fee_micros: int
    fill_price_mode: str


@dataclass(frozen=True)
class ResearchQuoteV1:
    SCHEMA_VERSION = "pathd.research_quote.v1"

    schema_version: str
    session: str
    contract: ContractIdentityV1
    source_vendor: str
    represented_interval_end_ns: int
    ts_recv_ns: int
    available_at_ns: int
    bid_micros: int
    ask_micros: int
    actionable: bool
    invalid_reason: str | None
    source_relative_path: str
    source_file_sha256: str
    row_group: int
    row_index: int
    canonical_row_sha256: str
    source_receipt_sha256: str
    source_record_sha256: str
    quote_sha256: str


@dataclass(frozen=True)
class ResearchFillDecisionV1:
    SCHEMA_VERSION = "pathd.research_fill_decision.v1"

    schema_version: str
    filled: bool
    fill_price_micros: int | None
    hard_limit_micros: int
    reason: str
    position_unchanged: bool


def research_fill_law_from_preregistration(payload: Any, /) -> ResearchFillLawV1:
    if type(payload) is not dict or type(payload.get("fill_law")) is not dict:
        raise TypeError("research fill law requires the exact preregistration payload")
    observed = payload["fill_law"]
    expected = prereg.fill_law()
    if prereg.stable_hash(observed) != prereg.stable_hash(expected):
        raise ValueError("research fill-law preregistration drift")
    return ResearchFillLawV1(
        schema_version=ResearchFillLawV1.SCHEMA_VERSION,
        fill_law_hash=observed["fill_law_hash"],
        headline_delay_ms=observed["headline_delay_ms"],
        delay_sensitivity_ms=tuple(observed["delay_sensitivity_ms"]),
        paired_quote_latency_bounds_ms=tuple(observed["paired_quote_latency_bounds_ms"]),
        entry_fee_micros=observed["fee_per_filled_side_cash_micros"],
        exit_fee_micros=observed["fee_per_filled_side_cash_micros"],
        fill_price_mode=observed["fill_price_mode"],
    )


def option_tick_micros(reference_price_micros: int, /) -> int:
    if type(reference_price_micros) is not int or reference_price_micros < 0:
        raise ValueError("reference option price must be nonnegative integer micros")
    return 50_000 if reference_price_micros < 3_000_000 else 100_000


def _quote_invalid_reason(*, bid_micros: int, ask_micros: int) -> str | None:
    if bid_micros == 0:
        return "ZERO_BID"
    if ask_micros == 0:
        return "ZERO_ASK"
    if bid_micros < 0 or ask_micros < 0:
        return "NEGATIVE_BBO"
    if bid_micros == ask_micros:
        return "LOCKED_BBO"
    if bid_micros > ask_micros:
        return "CROSSED_BBO"
    return None


def seal_research_quote(source_row: Any, /) -> ResearchQuoteV1:
    if type(source_row) is not VerifiedResearchQuoteRowV1:
        raise TypeError("research quote requires VerifiedResearchQuoteRowV1")
    if source_row.schema_version != source_row.SCHEMA_VERSION:
        raise ValueError("verified quote schema drift")
    semantic = {
        field.name: getattr(source_row, field.name)
        for field in fields(source_row)
        if field.name != "record_sha256"
    }
    semantic["contract"] = _contract_dict(source_row.contract)
    if source_row.record_sha256 != prereg.stable_hash(semantic):
        raise ValueError("verified quote record hash drift")
    if source_row.source_vendor != "DATABENTO_OPRA":
        raise ValueError("research option quote vendor drift")
    if source_row.available_at_ns > source_row.query_at_or_before_ns:
        raise ValueError("research quote is post-query")
    if (
        source_row.query_at_or_before_ns - source_row.available_at_ns
        > source_row.query_maximum_age_ms * 1_000_000
    ):
        raise ValueError("research quote is stale")
    selection_key = tuple(source_row.selection_key)
    if len(selection_key) != 6 or tuple(selection_key) != (
        source_row.available_at_ns,
        source_row.ts_recv_ns,
        source_row.represented_interval_end_ns,
        source_row.source_relative_path,
        source_row.row_group,
        source_row.row_index,
    ):
        raise ValueError("research quote selection key drift")
    # The selector, which has the complete candidate set, owns proof
    # construction.  This downstream seal can authenticate that proof through
    # record_sha256 but must not pretend the selected row was the only
    # candidate.
    if (
        type(source_row.eligible_row_count) is not int
        or source_row.eligible_row_count < 1
        or type(source_row.selection_proof_sha256) is not str
        or len(source_row.selection_proof_sha256) != 64
        or source_row.selection_proof_sha256.lower()
        != source_row.selection_proof_sha256
    ):
        raise ValueError("research quote selection proof drift")
    try:
        int(source_row.selection_proof_sha256, 16)
    except ValueError as exc:
        raise ValueError("research quote selection proof drift") from exc
    reason = _quote_invalid_reason(
        bid_micros=source_row.bid_micros, ask_micros=source_row.ask_micros
    )
    actionable = reason is None
    values = {
        "schema_version": ResearchQuoteV1.SCHEMA_VERSION,
        "session": source_row.session,
        "contract": source_row.contract,
        "source_vendor": source_row.source_vendor,
        "represented_interval_end_ns": source_row.represented_interval_end_ns,
        "ts_recv_ns": source_row.ts_recv_ns,
        "available_at_ns": source_row.available_at_ns,
        "bid_micros": source_row.bid_micros,
        "ask_micros": source_row.ask_micros,
        "actionable": actionable,
        "invalid_reason": reason,
        "source_relative_path": source_row.source_relative_path,
        "source_file_sha256": source_row.source_file_sha256,
        "row_group": source_row.row_group,
        "row_index": source_row.row_index,
        "canonical_row_sha256": source_row.canonical_row_sha256,
        "source_receipt_sha256": source_row.source_receipt_sha256,
        "source_record_sha256": source_row.record_sha256,
    }
    hash_values = dict(values)
    hash_values["contract"] = _contract_dict(source_row.contract)
    return ResearchQuoteV1(**values, quote_sha256=prereg.stable_hash(hash_values))


def _validate_law(law: ResearchFillLawV1) -> None:
    if type(law) is not ResearchFillLawV1 or law.schema_version != law.SCHEMA_VERSION:
        raise TypeError("research arrival requires ResearchFillLawV1")
    expected = research_fill_law_from_preregistration(prereg.preregistration_payload()[0])
    if law != expected:
        raise ValueError("research fill law drift")


def _validate_quote(quote: ResearchQuoteV1) -> None:
    if type(quote) is not ResearchQuoteV1 or quote.schema_version != quote.SCHEMA_VERSION:
        raise TypeError("research arrival requires ResearchQuoteV1")
    semantic = {
        field.name: getattr(quote, field.name)
        for field in fields(quote)
        if field.name != "quote_sha256"
    }
    semantic["contract"] = _contract_dict(quote.contract)
    if quote.quote_sha256 != prereg.stable_hash(semantic):
        raise ValueError("research quote seal drift")
    reason = _quote_invalid_reason(
        bid_micros=quote.bid_micros, ask_micros=quote.ask_micros
    )
    if quote.actionable is not (reason is None) or quote.invalid_reason != reason:
        raise ValueError("research quote actionability drift")


def evaluate_research_arrival(
    intent: Any, /, *, arrival: Any, law: Any
) -> ResearchFillDecisionV1:
    if type(intent) is not ExecutionIntentV1:
        raise TypeError("research arrival requires ExecutionIntentV1")
    _validate_law(law)
    _validate_quote(arrival)
    if arrival.contract != intent.contract:
        raise ValueError("arrival quote contract differs from intent")
    hard_limit = intent.price_budget.hard_limit_micros
    side = intent.decision.side
    if side not in ("BUY", "SELL"):
        raise ValueError("research fill law supports BUY or SELL only")
    if not arrival.actionable:
        return ResearchFillDecisionV1(
            schema_version=ResearchFillDecisionV1.SCHEMA_VERSION,
            filled=False,
            fill_price_micros=None,
            hard_limit_micros=hard_limit,
            reason=arrival.invalid_reason or "NON_ACTIONABLE_BBO",
            position_unchanged=True,
        )
    marketable = (
        arrival.ask_micros <= hard_limit
        if side == "BUY"
        else arrival.bid_micros >= hard_limit
    )
    if not marketable:
        return ResearchFillDecisionV1(
            schema_version=ResearchFillDecisionV1.SCHEMA_VERSION,
            filled=False,
            fill_price_micros=None,
            hard_limit_micros=hard_limit,
            reason="ARRIVAL_NOT_MARKETABLE",
            position_unchanged=True,
        )
    return ResearchFillDecisionV1(
        schema_version=ResearchFillDecisionV1.SCHEMA_VERSION,
        filled=True,
        fill_price_micros=hard_limit,
        hard_limit_micros=hard_limit,
        reason="FILLED_AT_SUBMITTED_HARD_LIMIT",
        position_unchanged=False,
    )


__all__ = [
    "ResearchFillLawV1",
    "ResearchQuoteV1",
    "ResearchFillDecisionV1",
    "research_fill_law_from_preregistration",
    "option_tick_micros",
    "evaluate_research_arrival",
    "seal_research_quote",
]
