from __future__ import annotations

from v4.ops.ibkr.write_ibc_runtime_config import build_config, escape_ini
from v4.ops.ibkr.probe_ibkr_api import candidate_ports, hold_connection, redact_account_id, sanitize_attempts
from v4.scripts.run_protocol146_ibc_credential_readiness import decide


def test_build_config_uses_paper_mode_and_api_port_without_newlines() -> None:
    config = build_config(
        username="user\nbad",
        password="pass\rbad",
        trading_mode="paper",
        api_port=4002,
        auto_restart_time="",
        cold_restart_time="",
    )

    assert "IbLoginId=userbad" in config
    assert "IbPassword=passbad" in config
    assert "TradingMode=paper" in config
    assert "AcceptNonBrokerageAccountWarning=yes" in config
    assert "OverrideTwsApiPort=4002" in config
    assert "ReadOnlyApi=no" in config


def test_escape_ini_strips_newlines() -> None:
    assert escape_ini("a\nb\rc") == "abc"


def test_protocol146_decision_blocks_missing_keychain_credentials() -> None:
    decision = decide(
        {"installed": True},
        {"username_present": True, "password_present": False},
        {"status": "skipped_missing_credentials"},
    )

    assert decision == "expected_blocker_missing_keychain_credentials"


def test_protocol146_decision_passes_when_config_written() -> None:
    decision = decide(
        {"installed": True},
        {"username_present": True, "password_present": True},
        {"status": "written"},
    )

    assert decision == "pass_ibc_credentials_ready_for_cold_start_rehearsal"


def test_ibkr_api_probe_helpers_redact_accounts() -> None:
    attempts = sanitize_attempts([{"primary_account_id": "DU123456", "port": 4002}])

    assert candidate_ports("4002,4000,bad,4002") == [4002, 4000]
    assert redact_account_id("DU123456") == "DU***56"
    assert attempts == [{"primary_account_id_redacted": "DU***56", "port": 4002}]


def test_hold_connection_blocks_when_dependency_missing(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ib_insync":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert hold_connection(host="127.0.0.1", port=4002, client_id=1, hold_seconds=1, poll_seconds=1) == 1
