from __future__ import annotations

from v4.live.paper_trade_log import make_trade_log_event, validate_observability_contract


SESSION = "2026-05-25"
RUN_ID = "fake_observability_run"
TS = "2026-05-25T14:35:00+00:00"
QUOTE_TS = "2026-05-25T14:34:59.900000+00:00"
RECEIVED_TS = "2026-05-25T14:35:00+00:00"
DECISION_TS = "2026-05-25T14:35:00+00:00"


def _market_snapshot(selected: dict | None = None) -> dict:
    option = selected or {
        "quote_count": 2,
        "freshness": {"count": 2, "known_count": 2, "missing_count": 0, "min_ms": 100, "median_ms": 100, "max_ms": 100},
        "option_quotes_digest": "quotes_hash",
    }
    return {
        "underlying": {
            "spx": 6700.0,
            "vix": 16.0,
            "spx_raw_quote_timestamp_utc": QUOTE_TS,
            "vix_raw_quote_timestamp_utc": QUOTE_TS,
            "spx_quote_age_ms": 100,
            "vix_quote_age_ms": 100,
            "spx_market_data_type": "live",
            "vix_market_data_type": "live",
        },
        "option_nbbo": option,
        "context": {
            "source": "fake_fixture",
            "context_age_ms": 0,
            "context_ready": True,
            "context_rows": 60,
            "context_minute_rows": 31,
        },
    }


def _selected_contract(**overrides) -> dict:
    row = {
        "contract_id": "SPXW-20260525-06700.000-C",
        "symbol": "SPX",
        "root": "SPXW",
        "trading_class": "SPXW",
        "settlement": "PM",
        "expiry": "20260525",
        "strike": 6700.0,
        "right": "C",
        "exchange": "SMART",
        "currency": "USD",
        "bid": 9.9,
        "ask": 10.0,
        "bid_size": 12,
        "ask_size": 11,
        "quote_age_ms": 100,
        "raw_quote_timestamp_utc": QUOTE_TS,
        "received_timestamp_utc": RECEIVED_TS,
        "decision_timestamp_utc": DECISION_TS,
    }
    row.update(overrides)
    return row


def _order(**overrides) -> dict:
    row = {
        "action": "BUY",
        "quantity": 1,
        "limit_price": 10.0,
        "contract_payload": {"symbol": "SPX", "trading_class": "SPXW", "settlement": "PM"},
        "order_payload": {"action": "BUY", "totalQuantity": 1, "lmtPrice": 10.0},
        "dry_run": False,
        "would_submit": False,
        "submit_allowed": True,
        "broker_order_id": "fake-order-1",
        "status": "Submitted",
        "final_status": "Submitted",
        "filled": 0,
        "remaining": 1,
    }
    row.update(overrides)
    return row


def _risk_gate(passed: bool = True, reason: str = "pass") -> dict:
    return {
        "passed": passed,
        "guard_passed": passed,
        "guard_block_reasons": [] if passed else [reason],
        "reason": reason,
        "permission_enable_flag_present": True,
        "permission_ack_flag_present": True,
        "permission_env_present": True,
        "account_prefix_ok": True,
        "paper_account_confirmed": True,
        "real_money_false_confirmed": True,
        "quantity_ok": True,
        "one_open_position_ok": True,
        "quote_freshness_ok": True,
        "context_freshness_ok": True,
        "affordability_ok": True,
        "account_cash": 10_000.0,
        "account_equity": 10_000.0,
        "open_positions": 0,
        "premium_required": 1000.0,
    }


def _base_event(event_type: str, **kwargs) -> dict:
    return make_trade_log_event(
        event_type=event_type,
        timestamp=TS,
        session=SESSION,
        run_id=RUN_ID,
        mode=kwargs.pop("mode", "intent-shadow"),
        selected_contract=kwargs.pop("selected_contract", {}),
        order=kwargs.pop("order", {}),
        account=kwargs.pop("account", {"account_id_redacted": "DU***45", "cash": 10_000.0, "equity": 10_000.0, "open_positions": 0}),
        market_snapshot=kwargs.pop("market_snapshot", _market_snapshot()),
        model_decision=kwargs.pop("model_decision", {"action": "wait", "selected_action": "wait", "reason": "fixture", "no_entry_reason": "fixture", "action_mask": {"wait": True}, "raw_logits": [], "candidate_logits": [], "threshold": -1.0}),
        risk_gate=kwargs.pop("risk_gate", _risk_gate()),
        broker_order_endpoint_called=kwargs.pop("broker_order_endpoint_called", False),
        intent_id=kwargs.pop("intent_id", "intent_fixture"),
        artifact_ids=kwargs.pop("artifact_ids", {"protocol101": "fake_protocol101", "surface": "fake_surface", "lifecycle": "fake_lifecycle"}),
        runtime_flag_digest=kwargs.pop("runtime_flag_digest", "fake_runtime_digest"),
        extra=kwargs.pop("extra", {}),
        **kwargs,
    )


def test_observability_contract_passes_complete_no_order_fixture() -> None:
    candidate_hash = "candidate_hash"
    feature_hash = "feature_hash"
    rows = [
        _base_event("market_snapshot"),
        _base_event(
            "candidate_set",
            extra={
                "candidate_set_hash": candidate_hash,
                "feature_vector_hash": feature_hash,
                "candidate_count": 0,
                "candidate_gate_diagnostics": {"filter_reason": "below_min_edge"},
            },
        ),
        _base_event(
            "model_decision",
            extra={"candidate_set_hash": candidate_hash, "feature_vector_hash": feature_hash},
            model_decision={
                "action": "wait",
                "selected_action": "wait",
                "score": None,
                "selected_margin": None,
                "threshold": -1.365,
                "threshold_source": "protocol101_artifact",
                "reason": "no_candidates",
                "no_entry_reason": "no_candidates",
                "action_mask": {"wait": True, "candidate_count": 0, "enter": False},
                "raw_logits": [],
                "wait_logit": None,
                "candidate_logits": [],
                "candidate_set_hash": candidate_hash,
                "feature_vector_hash": feature_hash,
            },
        ),
        _base_event("risk_gate", risk_gate=_risk_gate(True, "no_entry_intent")),
    ]

    result = validate_observability_contract(rows)

    assert result["status"] == "pass"
    assert result["readiness_status"] == "ready"


def test_observability_contract_fails_closed_on_real_money_or_unknown() -> None:
    row = _base_event("market_snapshot", real_money_trading=True)

    result = validate_observability_contract([row])

    assert result["status"] == "fail"
    assert any("real_money must be false" in error for error in result["errors"])


def test_observability_contract_fails_closed_on_no_order_broker_call() -> None:
    row = _base_event(
        "paper_order_dry_run",
        mode="paper-dry-run",
        selected_contract=_selected_contract(),
        order=_order(dry_run=True, would_submit=True),
        market_snapshot=_market_snapshot(_selected_contract()),
        broker_order_endpoint_called=True,
    )

    result = validate_observability_contract([row])

    assert result["status"] == "fail"
    assert any("no-order/dry-run row called broker endpoint" in error for error in result["errors"])


def test_observability_contract_fails_closed_when_submit_lacks_prior_guard_pass() -> None:
    submit = _base_event(
        "paper_order_submitted",
        mode="paper-submit",
        selected_contract=_selected_contract(),
        order=_order(),
        market_snapshot=_market_snapshot(_selected_contract()),
        broker_order_endpoint_called=True,
    )

    result = validate_observability_contract([submit])

    assert result["status"] == "fail"
    assert any("submitted order lacks prior guard pass" in error for error in result["errors"])


def test_observability_contract_accepts_fake_submit_chain_with_prior_guard() -> None:
    guard = _base_event(
        "risk_gate",
        mode="paper-submit",
        selected_contract=_selected_contract(),
        order=_order(status="guard_passed", broker_order_id=None),
        market_snapshot=_market_snapshot(_selected_contract()),
        risk_gate=_risk_gate(True, "pass"),
        intent_id="intent_submit",
    )
    submit = _base_event(
        "paper_order_submitted",
        mode="paper-submit",
        selected_contract=_selected_contract(),
        order=_order(),
        market_snapshot=_market_snapshot(_selected_contract()),
        broker_order_endpoint_called=True,
        intent_id="intent_submit",
    )

    result = validate_observability_contract([guard, submit])

    assert result["status"] == "pass"


def test_observability_contract_fails_closed_on_quantity_exceeds_one() -> None:
    row = _base_event(
        "paper_order_dry_run",
        mode="paper-dry-run",
        selected_contract=_selected_contract(),
        order=_order(quantity=2, dry_run=True, would_submit=True),
        market_snapshot=_market_snapshot(_selected_contract()),
    )

    result = validate_observability_contract([row])

    assert result["status"] == "fail"
    assert any("quantity exceeds one contract" in error for error in result["errors"])


def test_observability_contract_fails_closed_on_selected_quote_without_timestamp() -> None:
    selected = _selected_contract(raw_quote_timestamp_utc=None)
    selected.pop("quote_timestamp", None)
    row = _base_event(
        "paper_order_dry_run",
        mode="paper-dry-run",
        selected_contract=selected,
        order=_order(dry_run=True, would_submit=True),
        market_snapshot=_market_snapshot(selected),
    )

    result = validate_observability_contract([row])

    assert result["status"] == "fail"
    assert any("selected quote lacks raw timestamp" in error for error in result["errors"])


def test_observability_contract_fails_closed_on_no_entry_without_reason() -> None:
    row = _base_event(
        "model_decision",
        extra={"candidate_set_hash": "candidate_hash", "feature_vector_hash": "feature_hash"},
        model_decision={
            "action": "wait",
            "selected_action": "wait",
            "threshold": -1.0,
            "action_mask": {"wait": True},
            "raw_logits": [],
            "candidate_logits": [],
            "no_entry_reason": "",
        },
    )

    result = validate_observability_contract([row])

    assert result["status"] == "fail"
    assert any("no-entry decision lacks structured no-entry reason" in error for error in result["errors"])


def test_observability_contract_fails_closed_on_raw_account_id() -> None:
    row = _base_event("market_snapshot", account={"account_id": "DU12345", "cash": 10_000.0})

    result = validate_observability_contract([row])

    assert result["status"] == "fail"
    assert any("raw account id" in error for error in result["errors"])


def test_observability_contract_covers_fake_broker_status_and_lifecycle() -> None:
    guard = _base_event(
        "risk_gate",
        mode="paper-submit",
        selected_contract=_selected_contract(),
        order=_order(status="guard_passed", broker_order_id=None),
        market_snapshot=_market_snapshot(_selected_contract()),
        intent_id="intent_lifecycle",
    )
    submit = _base_event(
        "paper_order_submitted",
        mode="paper-submit",
        selected_contract=_selected_contract(),
        order=_order(),
        market_snapshot=_market_snapshot(_selected_contract()),
        broker_order_endpoint_called=True,
        intent_id="intent_lifecycle",
    )
    status = _base_event(
        "paper_order_status",
        mode="paper-submit",
        selected_contract=_selected_contract(),
        order=_order(status="Filled", final_status="Filled", filled=1, remaining=0),
        market_snapshot=_market_snapshot(_selected_contract()),
        broker_order_endpoint_called=True,
        intent_id="intent_lifecycle",
    )
    lifecycle = _base_event(
        "lifecycle_state",
        mode="paper-submit",
        selected_contract=_selected_contract(),
        order=_order(status="Filled", final_status="Filled", filled=1, remaining=0),
        market_snapshot=_market_snapshot(_selected_contract()),
        extra={
            "lifecycle": {
                "position_detected": True,
                "runtime_state_hash": "runtime_hash",
                "lifecycle_action": "hold",
                "final_position_state": "holding",
            }
        },
        intent_id="intent_lifecycle",
    )

    result = validate_observability_contract([guard, submit, status, lifecycle])

    assert result["status"] == "pass"
