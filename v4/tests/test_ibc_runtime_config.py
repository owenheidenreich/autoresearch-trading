from __future__ import annotations

from v4.ops.ibkr.write_ibc_runtime_config import build_config, escape_ini
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
