"""Protocol166 live/training candidate-generation parity contract.

This module defines the candidate surface that a future Protocol165 runtime
must share with Protocol164 historical training. It is intentionally small and
deterministic so tests can catch drift before paper trading uses a new model.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


CONTRACT_MULTIPLIER = 100.0


@dataclass(frozen=True)
class Protocol166Contract:
    protocol: str = "166_live_training_parity_contract"
    symbol_root: str = "SPXW"
    settlement_style: str = "PM"
    strike_spacing: float = 5.0
    strike_window: float = 50.0
    starting_cash: float = 10_000.0
    max_contracts: int = 1
    max_concurrent_positions: int = 1
    decision_window_start_et: str = "09:31"
    no_new_entries_after_et: str = "15:30"
    entry_price: str = "ask"
    exit_price: str = "bid"
    fees_included: bool = False
    required_candidate_fields: tuple[str, ...] = (
        "decision_time",
        "contract_id",
        "root",
        "settlement_style",
        "right",
        "offset",
        "entry_bid",
        "entry_ask",
        "entry_mid",
        "entry_spread",
        "entry_bid_size",
        "entry_ask_size",
        "entry_premium",
        "entry_delta",
        "entry_gamma",
        "entry_theta",
        "entry_iv",
    )
    required_account_fields: tuple[str, ...] = (
        "account_equity",
        "cash_available",
        "open_position_count",
        "max_concurrent_positions",
        "max_contracts",
    )
    disallowed_pre_entry_gates: tuple[str, ...] = (
        "protocol101_min_edge_gate",
        "protocol101_allowed_time_bucket_gate",
        "protocol101_selected_candidate_dependency",
    )
    flat_actions: tuple[str, ...] = ("wait", "enter_call", "enter_put")
    holding_actions: tuple[str, ...] = ("hold", "exit")
    position_states: tuple[str, ...] = ("flat", "holding")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_protocol166_contract() -> Protocol166Contract:
    return Protocol166Contract()


def validate_candidate(candidate: dict[str, Any], account_state: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = default_protocol166_contract()
    errors: list[str] = []
    for field in contract.required_candidate_fields:
        if field not in candidate:
            errors.append(f"missing_candidate_field:{field}")
    if account_state is not None:
        for field in contract.required_account_fields:
            if field not in account_state:
                errors.append(f"missing_account_field:{field}")
    if errors:
        return {"status": "fail", "errors": errors}

    contract_id = str(candidate.get("contract_id"))
    if str(candidate.get("root")) != contract.symbol_root or not contract_id.startswith(f"{contract.symbol_root}-"):
        errors.append("wrong_root")
    if str(candidate.get("settlement_style")) != contract.settlement_style:
        errors.append("wrong_settlement_style")
    if str(candidate.get("right")) not in {"C", "P"}:
        errors.append("invalid_right")
    offset = _finite(candidate.get("offset"))
    if not math.isfinite(offset) or abs(offset) > contract.strike_window or abs(offset % contract.strike_spacing) > 1e-6:
        errors.append("invalid_strike_offset")
    bid = _finite(candidate.get("entry_bid"))
    ask = _finite(candidate.get("entry_ask"))
    mid = _finite(candidate.get("entry_mid"))
    spread = _finite(candidate.get("entry_spread"))
    bid_size = _finite(candidate.get("entry_bid_size"))
    ask_size = _finite(candidate.get("entry_ask_size"))
    if bid <= 0.0 or ask <= 0.0 or mid <= 0.0 or spread < 0.0 or ask < bid:
        errors.append("invalid_nbbo")
    if not math.isfinite(bid_size) or not math.isfinite(ask_size):
        errors.append("missing_quote_size")
    for greek in ("entry_delta", "entry_gamma", "entry_theta", "entry_iv"):
        if not math.isfinite(_finite(candidate.get(greek))):
            errors.append(f"invalid_greek:{greek}")
    premium = _finite(candidate.get("entry_premium"))
    expected_premium = ask * CONTRACT_MULTIPLIER
    if premium <= 0.0 or abs(premium - expected_premium) > 1e-6:
        errors.append("invalid_entry_premium")
    if account_state is not None:
        cash = _finite(account_state.get("cash_available"))
        open_positions = int(_finite(account_state.get("open_position_count"), 99.0))
        if premium > cash:
            errors.append("unaffordable")
        if open_positions >= int(account_state.get("max_concurrent_positions", contract.max_concurrent_positions)):
            errors.append("max_concurrency_reached")
        if int(account_state.get("max_contracts", contract.max_contracts)) != contract.max_contracts:
            errors.append("max_contracts_mismatch")
    return {"status": "pass" if not errors else "fail", "errors": errors}


def validate_position_action(
    *,
    position_state: str,
    action: str,
    selected_candidate: dict[str, Any] | None = None,
    account_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the unified position-state action space."""

    contract = default_protocol166_contract()
    errors: list[str] = []
    if position_state not in contract.position_states:
        errors.append("invalid_position_state")
    if position_state == "flat":
        if action not in contract.flat_actions:
            errors.append("invalid_flat_action")
        if action in {"enter_call", "enter_put"}:
            if selected_candidate is None:
                errors.append("missing_selected_candidate")
            else:
                expected_right = "C" if action == "enter_call" else "P"
                if str(selected_candidate.get("right")) != expected_right:
                    errors.append("action_candidate_side_mismatch")
                candidate_result = validate_candidate(selected_candidate, account_state)
                errors.extend(candidate_result["errors"])
    if position_state == "holding":
        if action not in contract.holding_actions:
            errors.append("invalid_holding_action")
        if action == "hold" and selected_candidate is not None:
            errors.append("hold_must_not_select_new_candidate")
    return {"status": "pass" if not errors else "fail", "errors": sorted(set(errors))}


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default
