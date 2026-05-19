from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from v4.scripts.run_protocol140_ibkr_autostart_prep import GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL
from v4.scripts.run_protocol156_ibkr_autostart_observability import (
    PortProbe,
    aggregate_signals,
    collect_entitlement_summary,
    collect_logs,
    decide,
    entitlement_live_ready,
    file_digest,
    parse_launchctl_print,
    ports_from_args,
    write_report,
)


def loaded_status(label: str) -> dict[str, object]:
    return {
        "label": label,
        "loaded": True,
        "state": "not running",
        "runs": 4,
        "last_exit_code": 1,
    }


def test_parse_launchctl_print_extracts_state_and_log_paths() -> None:
    text = """
    state = not running
    runs = 4
    last exit code = 1
    path = /Users/me/Library/LaunchAgents/com.example.plist
    stdout path = /Users/me/Library/Logs/out.log
    stderr path = /Users/me/Library/Logs/err.log
    """

    status = parse_launchctl_print(
        label="com.example",
        target="gui/501/com.example",
        stdout=text,
        stderr="",
        returncode=0,
    )

    assert status["loaded"] is True
    assert status["state"] == "not running"
    assert status["runs"] == 4
    assert status["last_exit_code"] == 1
    assert status["stdout_path"].endswith("out.log")
    assert status["stderr_path"].endswith("err.log")


def test_file_digest_detects_known_failure_signals(tmp_path: Path) -> None:
    log = tmp_path / "session.err.log"
    log.write_text(
        "\n".join(
            [
                "/usr/bin/python3: Error while finding module specification for 'v4.scripts.x'",
                "ModuleNotFoundError: No module named 'v4'",
                json.dumps({"status": "port_open", "port": 4002}),
                json.dumps({"connected": True, "status": "pass"}),
                json.dumps({"blocked_reason": "missing_live_market_data_entitlements"}),
            ]
        )
        + "\n"
    )

    digest = file_digest(log, tail_lines=20)

    assert "pythonpath_missing_for_session_runner" in digest["signals"]
    assert "api_port_open_detected" in digest["signals"]
    assert "ibkr_api_connected_detected" in digest["signals"]
    assert "pass_status_detected" in digest["signals"]
    assert "missing_live_market_data_entitlements" in digest["signals"]
    assert digest["json_events"][-1]["blocked_reason"] == "missing_live_market_data_entitlements"


def test_file_digest_detects_launchd_python_runtime_failure(tmp_path: Path) -> None:
    log = tmp_path / "session.err.log"
    log.write_text(
        "\n".join(
            [
                "Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encoding",
                "PermissionError: [Errno 1] Operation not permitted",
            ]
        )
        + "\n"
    )

    digest = file_digest(log, tail_lines=20)

    assert "launchd_python_runtime_failed" in digest["signals"]
    assert "launchd_permission_denied" in digest["signals"]


def test_collect_logs_and_decide_prioritize_session_import_failure(tmp_path: Path) -> None:
    (tmp_path / "protocol101-paper-session.err.log").write_text("ModuleNotFoundError: No module named 'v4'\n")
    (tmp_path / "protocol101-paper-preflight.out.log").write_text('{"status": "port_open"}\n')
    (tmp_path / "protocol101-paper-preflight.err.log").write_text("Requested market data is not subscribed\n")
    (tmp_path / "ibgateway-paper.err.log").write_text('{"connected": true, "status": "pass"}\n')

    logs = collect_logs(tmp_path, tail_lines=20)
    signals = aggregate_signals(logs)
    launchd = {label: loaded_status(label) for label in (GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL)}
    decision = decide(launchd=launchd, logs=logs, ports=[PortProbe(port=4002, open=True)], signals=signals)

    assert signals["pythonpath_missing_for_session_runner"] == 1
    assert decision == "blocked_session_runner_pythonpath_missing"


def test_decide_moves_past_historical_import_failure_when_runtime_wrapper_is_patched() -> None:
    launchd = {label: loaded_status(label) for label in (GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL)}
    runtime_wrappers = {
        "run_protocol101_paper_session.sh": {"exists": True, "exports_pythonpath": True},
    }

    decision = decide(
        launchd=launchd,
        logs={},
        ports=[PortProbe(port=4002, open=False)],
        signals=Counter(
            {
                "pythonpath_missing_for_session_runner": 1,
                "ibkr_market_data_not_subscribed": 1,
            }
        ),
        runtime_wrappers=runtime_wrappers,
    )

    assert decision == "blocked_live_market_data_entitlements"


def test_decide_blocks_unpatched_launchd_python_runtime_failure() -> None:
    launchd = {label: loaded_status(label) for label in (GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL)}
    runtime_wrappers = {
        "run_protocol101_paper_session.sh": {"exists": True, "exports_pythonpath": True, "prefers_project_venv": False},
        "run_protocol101_paper_preflight.sh": {"exists": True, "exports_pythonpath": True, "prefers_project_venv": False},
    }

    decision = decide(
        launchd=launchd,
        logs={},
        ports=[PortProbe(port=4002, open=True)],
        signals=Counter({"launchd_python_runtime_failed": 1}),
        runtime_wrappers=runtime_wrappers,
    )

    assert decision == "blocked_launchd_python_runtime_failed"


def test_decide_moves_past_historical_launchd_python_failure_when_wrappers_are_patched() -> None:
    launchd = {label: loaded_status(label) for label in (GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL)}
    runtime_wrappers = {
        "run_protocol101_paper_session.sh": {"exists": True, "exports_pythonpath": True, "prefers_project_venv": True},
        "run_protocol101_paper_preflight.sh": {"exists": True, "exports_pythonpath": True, "prefers_project_venv": True},
    }

    decision = decide(
        launchd=launchd,
        logs={},
        ports=[PortProbe(port=4002, open=True)],
        signals=Counter({"launchd_python_runtime_failed": 1, "missing_live_market_data_entitlements": 1}),
        runtime_wrappers=runtime_wrappers,
    )

    assert decision == "blocked_live_market_data_entitlements"


def test_decide_entitlement_block_after_loaded_api_port() -> None:
    launchd = {label: loaded_status(label) for label in (GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL)}

    decision = decide(
        launchd=launchd,
        logs={},
        ports=[PortProbe(port=4002, open=True)],
        signals=Counter({"api_port_open_detected": 1, "missing_live_market_data_entitlements": 1}),
    )

    assert decision == "blocked_live_market_data_entitlements"


def test_entitlement_live_ready_requires_live_index_and_option_nbbo() -> None:
    payload = {
        "decision": "pass",
        "ibkr_connected": True,
        "broker_order_endpoint_called": False,
        "feed_status": {
            "spx": {"live_price_available": True},
            "vix": {"live_price_available": True},
            "spxw_options": {"live_nbbo_rows": 6},
        },
    }

    assert entitlement_live_ready(payload) is True
    payload["feed_status"]["spxw_options"]["live_nbbo_rows"] = 0
    assert entitlement_live_ready(payload) is False


def test_collect_entitlement_summary_reads_no_order_probe(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "checked_at": "2026-05-19T11:09:54-04:00",
                "decision": "pass",
                "blocked_reason": None,
                "ibkr_connected": True,
                "ibkr_port": 4002,
                "feed_status": {"spx": {"live_price_available": True}},
                "subscription_errors": [],
                "no_order_guarantee": {"broker_order_endpoint_called": False},
            }
        )
    )

    result = collect_entitlement_summary(summary)

    assert result["decision"] == "pass"
    assert result["broker_order_endpoint_called"] is False
    assert result["subscription_error_count"] == 0


def test_decide_passes_when_latest_entitlement_probe_passes_despite_stale_log_errors() -> None:
    launchd = {label: loaded_status(label) for label in (GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL)}
    entitlement = {
        "decision": "pass",
        "ibkr_connected": True,
        "broker_order_endpoint_called": False,
        "feed_status": {
            "spx": {"live_price_available": True},
            "vix": {"live_price_available": True},
            "spxw_options": {"live_nbbo_rows": 6},
        },
    }

    decision = decide(
        launchd=launchd,
        logs={},
        ports=[PortProbe(port=4002, open=True)],
        signals=Counter({"missing_live_market_data_entitlements": 1, "ibkr_market_data_not_subscribed": 1}),
        entitlement=entitlement,
    )

    assert decision == "pass_live_market_data_entitlements"


def test_decide_blocks_when_launchagents_missing() -> None:
    launchd = {
        GATEWAY_LABEL: loaded_status(GATEWAY_LABEL),
        PREFLIGHT_LABEL: loaded_status(PREFLIGHT_LABEL),
        SESSION_LABEL: {"label": SESSION_LABEL, "loaded": False},
    }

    decision = decide(launchd=launchd, logs={}, ports=[PortProbe(port=4002, open=True)], signals=Counter())

    assert decision == "blocked_launchagents_not_loaded"


def test_ports_from_args_keeps_paper_port_candidates() -> None:
    assert ports_from_args(4002, "4002,4000") == [4002, 4000, 7497, 7496, 4001]


def test_write_report_includes_log_paths_and_next_action(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    payload = {
        "generated_at_pacific": "2026-05-18T12:00:00-07:00",
        "session_date": "2026-05-19",
        "decision": "blocked_session_runner_pythonpath_missing",
        "log_dir": str(tmp_path),
        "next_action": "copy patched wrapper",
        "interpretation": ["import failed"],
        "launchd": {label: loaded_status(label) for label in (GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL)},
        "ports": [{"port": 4002, "open": False, "error": "ConnectionRefusedError"}],
        "signals": {"pythonpath_missing_for_session_runner": 1},
        "logs": {
            "session_stderr": {
                "path": str(tmp_path / "err.log"),
                "exists": True,
                "size_bytes": 10,
                "modified_at_pacific": "2026-05-18T12:00:00-07:00",
                "signals": ["pythonpath_missing_for_session_runner"],
                "json_events": [],
                "tail": ["ModuleNotFoundError: No module named 'v4'"],
            },
            "ibc_recent": [],
        },
        "live_trade_logs": {"exists": False, "root": "v4/logs/paper_trading", "jsonl_files": []},
    }

    write_report(report, payload)

    text = report.read_text()
    assert "Protocol 156" in text
    assert "blocked_session_runner_pythonpath_missing" in text
    assert "copy patched wrapper" in text
    assert "ModuleNotFoundError" in text
