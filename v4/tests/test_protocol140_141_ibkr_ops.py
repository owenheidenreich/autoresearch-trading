from __future__ import annotations

from pathlib import Path

from v4.live.ibkr_paper_guard import (
    PaperOrderGuardConfig,
    PaperOrderIntent,
    paper_order_permission,
    validate_order_intent,
)
from v4.scripts.run_protocol140_ibkr_autostart_prep import (
    candidate_api_ports,
    launchd_payload,
    parse_jts_ini,
    readiness_checks,
)
from v4.scripts.run_protocol141_ibkr_paper_order_guard import decide


def test_parse_jts_ini_reads_paper_api_config(tmp_path: Path) -> None:
    jts = tmp_path / "jts.ini"
    jts.write_text(
        """
[IBGateway]
TrustedIPs=127.0.0.1
LocalServerPort=4000
ApiOnly=true
[Logon]
tradingMode=p
"""
    )

    parsed = parse_jts_ini(jts)
    checks = readiness_checks(app_path=tmp_path, jts=parsed, jts_path=jts)

    assert parsed["IBGateway"]["LocalServerPort"] == "4000"
    assert parsed["Logon"]["tradingMode"] == "p"
    assert {check["name"]: check["passed"] for check in checks}["paper_mode_configured"] is True
    assert {check["name"]: check["passed"] for check in checks}["trusted_localhost"] is True


def test_launchd_payload_runs_on_calendar_with_logs(tmp_path: Path) -> None:
    payload = launchd_payload(
        label="com.example.test",
        program_arguments=["/bin/echo", "hello"],
        hour=6,
        minute=20,
        stdout=tmp_path / "out.log",
        stderr=tmp_path / "err.log",
        environment={"A": "B"},
    )

    assert payload["Label"] == "com.example.test"
    assert payload["StartCalendarInterval"] == {"Hour": 6, "Minute": 20}
    assert payload["EnvironmentVariables"] == {"A": "B"}
    assert payload["ProgramArguments"] == ["/bin/echo", "hello"]


def test_candidate_api_ports_include_configured_and_paper_default() -> None:
    assert candidate_api_ports(4002, configured_port=4000) == [4002, 4000, 7497, 7496, 4001]
    assert candidate_api_ports(4000) == [4000, 4002, 7497, 7496, 4001]


def test_paper_order_permission_blocks_by_default_and_passes_with_du_account() -> None:
    blocked = paper_order_permission(
        enable_paper_orders=False,
        acknowledge_paper_loss=False,
        account_id="DU12345",
        environ={},
    )
    allowed = paper_order_permission(
        enable_paper_orders=True,
        acknowledge_paper_loss=True,
        account_id="DU12345",
        environ={"V4_ALLOW_IBKR_PAPER_ORDERS": "YES"},
    )

    assert blocked["passed"] is False
    assert "enable_paper_orders_flag_missing" in blocked["reasons"]
    assert allowed["passed"] is True


def test_paper_order_permission_rejects_nonpaper_account() -> None:
    result = paper_order_permission(
        enable_paper_orders=True,
        acknowledge_paper_loss=True,
        account_id="U12345",
        environ={"V4_ALLOW_IBKR_PAPER_ORDERS": "YES"},
    )

    assert result["passed"] is False
    assert "account_not_recognized_as_paper" in result["reasons"]


def test_validate_order_intent_rejects_stale_or_unaffordable_entry() -> None:
    intent = PaperOrderIntent(
        action="BUY",
        symbol="SPX",
        expiry="20260320",
        strike=6700.0,
        right="C",
        quantity=1,
        limit_price=100.0,
    )

    result = validate_order_intent(
        intent,
        account_cash=5_000.0,
        open_positions=0,
        quote={"bid": 99.5, "ask": 100.0, "reference_ask": 100.0, "quote_age_ms": 3_000},
        context={"context_age_ms": 100},
    )

    assert result["passed"] is False
    assert "stale_option_quote" in result["reasons"]
    assert "insufficient_paper_cash" in result["reasons"]


def test_validate_order_intent_accepts_clean_spxw_paper_intent() -> None:
    intent = PaperOrderIntent(
        action="BUY",
        symbol="SPX",
        expiry="20260320",
        strike=6700.0,
        right="P",
        quantity=1,
        limit_price=10.0,
    )

    result = validate_order_intent(
        intent,
        account_cash=10_000.0,
        open_positions=0,
        quote={"bid": 9.9, "ask": 10.0, "reference_ask": 10.0, "quote_age_ms": 50},
        context={"context_age_ms": 50},
        config=PaperOrderGuardConfig(),
    )

    assert result["passed"] is True
    assert result["premium_required"] == 1000.0


def test_protocol141_decision_default_guard_smoke_passes_because_it_blocks_orders() -> None:
    permission = {"passed": False, "reason": "enable_paper_orders_flag_missing"}
    intent_validation = {"passed": True, "reason": "pass"}

    assert (
        decide("guard-smoke", permission=permission, account_probe={}, intent_validation=intent_validation)
        == "pass_default_blocks_paper_orders_until_explicitly_enabled"
    )


def test_protocol141_decision_account_probe_passes_without_order_flags() -> None:
    permission = {"passed": False, "reason": "enable_paper_orders_flag_missing"}
    account_probe = {"connected": True}
    intent_validation = {"passed": True, "reason": "pass"}

    assert (
        decide("account-probe", permission=permission, account_probe=account_probe, intent_validation=intent_validation)
        == "pass_account_probe_connected_orders_still_disabled"
    )
