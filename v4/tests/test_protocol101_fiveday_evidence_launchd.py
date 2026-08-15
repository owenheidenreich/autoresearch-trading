from __future__ import annotations

import plistlib
from pathlib import Path

from v4.scripts.run_protocol101_fiveday_evidence_packet import build_packet


ROOT = Path(__file__).resolve().parents[2]
LAUNCHD = ROOT / "v4" / "ops" / "launchd"
OPS_IBKR = ROOT / "v4" / "ops" / "ibkr"

TARGET_DATES = [
    (6, 8),
    (6, 9),
    (6, 10),
    (6, 11),
    (6, 12),
]


def load_plist(name: str) -> dict:
    with (LAUNCHD / name).open("rb") as fh:
        return plistlib.load(fh)


def interval_pairs(plist: dict) -> list[tuple[int, int, int, int]]:
    return [
        (int(item["Month"]), int(item["Day"]), int(item["Hour"]), int(item["Minute"]))
        for item in plist["StartCalendarInterval"]
    ]


def test_fiveday_launchd_jobs_are_date_scoped() -> None:
    expected = {
        "com.autoresearch.protocol101.fiveday.ibgateway.paper.plist": 28,
        "com.autoresearch.protocol101.fiveday.paper-preflight.plist": 29,
        "com.autoresearch.protocol101.fiveday.paper-session.plist": 30,
        "com.autoresearch.protocol101.fiveday.daily-monitor.plist": 31,
    }
    for plist_name, minute in expected.items():
        plist = load_plist(plist_name)
        assert interval_pairs(plist) == [(month, day, 6, minute) for month, day in TARGET_DATES]

    shutdown = load_plist("com.autoresearch.protocol101.fiveday.paper-shutdown.plist")
    post = load_plist("com.autoresearch.protocol101.fiveday.evidence-postsession.plist")
    assert interval_pairs(shutdown) == [(month, day, 13, 5) for month, day in TARGET_DATES]
    assert interval_pairs(post) == [(month, day, 13, 15) for month, day in TARGET_DATES]

    watchdog = load_plist("com.autoresearch.protocol101.fiveday.recovery-watchdog.plist")
    assert interval_pairs(watchdog) == [
        (month, day, hour, minute)
        for month, day in TARGET_DATES
        for hour, minute in ((6, 35), (6, 45), (7, 0), (7, 15), (8, 0), (10, 0), (12, 0))
    ]


def test_fiveday_paper_session_is_guarded_paper_submit() -> None:
    plist = load_plist("com.autoresearch.protocol101.fiveday.paper-session.plist")
    env = plist["EnvironmentVariables"]

    assert plist["Label"] == "com.autoresearch.protocol101.fiveday.paper-session"
    assert env["DAILY_PAPER_AUTOPILOT_MODE"] == "paper-submit"
    assert env["PROTOCOL101_ENTRY_BRIDGE_MODE"] == "paper-submit"
    assert env["PROTOCOL101_ENABLE_PAPER_ORDERS"] == "YES"
    assert env["PROTOCOL101_ACKNOWLEDGE_PAPER_LOSS"] == "YES"
    assert env["V4_ALLOW_IBKR_PAPER_ORDERS"] == "YES"
    assert env["DAILY_PAPER_AUTOPILOT_MODEL_REGISTRY"].endswith("v4/promotion/PAPER_TRADING_DEFAULT.json")
    assert env["DAILY_PAPER_AUTOPILOT_STARTUP_ATTEMPTS"] == "40"
    assert env["DAILY_PAPER_AUTOPILOT_RETRY_UNTIL_PACIFIC"] == "13:00"
    assert "v4.scripts.run_protocol160_protocol101_persistent_paper_trader" in env["DAILY_PAPER_AUTOPILOT_WARM_MODULES"]
    assert env["LAUNCHD_PYTHON_IMPORT_SHELL_TIMEOUT_SECONDS"] == "180"
    assert env["LAUNCHD_PYTHON_IMPORT_TIMEOUT_SECONDS"] == "120"
    assert env["PYTHON_BIN"] == "/Users/gduby/.autoresearch-trading/runtime-venv/bin/python"
    assert env["DAILY_PAPER_AUTOPILOT_TRADE_LOG_ROOT"] == "/Users/gduby/.autoresearch-trading/live_runtime/paper_trading"
    assert env["DAILY_PAPER_AUTOPILOT_LIVE_INDEX_CONTEXT_LOG"] == "/Users/gduby/.autoresearch-trading/live_runtime/runtime/protocol101_live_index_context.jsonl"


def test_fiveday_installer_disables_daily_duplicates_and_loads_fiveday_labels() -> None:
    installer = (LAUNCHD / "install_protocol101_fiveday_evidence.sh").read_text()

    assert "com.autoresearch.protocol101.paper-session" in installer
    assert "launchctl disable \"gui/$UID/$label\"" in installer
    assert "com.autoresearch.protocol101.fiveday.paper-session" in installer
    assert "com.autoresearch.protocol101.fiveday.recovery-watchdog" in installer
    assert "run_protocol101_fiveday_recovery_watchdog.sh" in installer
    assert "launchctl bootstrap \"gui/$UID\" \"$LAUNCH_AGENT_DIR/$plist\"" in installer


def test_fiveday_recovery_watchdog_kickstarts_missing_stack() -> None:
    wrapper = (OPS_IBKR / "run_protocol101_fiveday_recovery_watchdog.sh").read_text()

    assert "com.autoresearch.protocol101.fiveday.ibgateway.paper" in wrapper
    assert "com.autoresearch.protocol101.fiveday.paper-session" in wrapper
    assert "com.autoresearch.protocol101.fiveday.daily-monitor" in wrapper
    assert "launchctl kickstart -k" in wrapper
    assert "broker_order_endpoint_called" in wrapper


def test_paper_runtime_wrappers_are_selection_safe_and_launchd_hardened() -> None:
    autopilot = (OPS_IBKR / "run_daily_paper_autopilot.sh").read_text()
    session = (OPS_IBKR / "run_protocol101_paper_session.sh").read_text()
    gateway = (OPS_IBKR / "start_ib_gateway_paper_ibc.sh").read_text()
    preflight = (OPS_IBKR / "run_protocol101_paper_preflight.sh").read_text()

    assert 'cmd+=("$@")' in autopilot
    assert 'exec /bin/bash "$SCRIPT_DIR/run_daily_paper_autopilot.sh" "$@"' in session
    assert "AUTORESEARCH_TMP_DIR" in gateway
    assert 'export TMPDIR="$AUTORESEARCH_TMP_DIR/"' in gateway
    assert 'cd "$AUTORESEARCH_TMP_DIR"' in gateway
    assert "RUNTIME_PYTHON_BIN" in autopilot
    assert "DAILY_PAPER_AUTOPILOT_TRADE_LOG_ROOT" in autopilot
    assert "--trade-log-root" in autopilot
    assert "--live-index-context-log" in autopilot
    assert "continuing to IBKR API preflight" in preflight
    assert "warm_python_deps" in autopilot
    assert "DAILY_PAPER_AUTOPILOT_RETRY_UNTIL_PACIFIC" in autopilot


def test_shutdown_wrapper_handles_fiveday_labels_without_unloading_schedules() -> None:
    shutdown = (OPS_IBKR / "shutdown_ibkr_paper_stack.sh").read_text()

    assert 'kill_launchd_job "com.autoresearch.protocol101.fiveday.ibgateway.paper"' in shutdown
    assert 'kill_launchd_job "com.autoresearch.protocol101.fiveday.paper-session"' in shutdown
    assert 'kill_launchd_job "com.autoresearch.protocol101.fiveday.daily-monitor"' in shutdown
    assert "launchctl bootout" not in shutdown


def test_fiveday_postsession_wrapper_runs_analyzer_monitor_and_packet() -> None:
    wrapper = (OPS_IBKR / "run_protocol101_fiveday_post_session_evidence.sh").read_text()

    assert "run_protocol148_protocol101_post_session_analyzer" in wrapper
    assert "run_protocol157_protocol101_daily_ops_monitor" in wrapper
    assert "run_protocol101_fiveday_evidence_packet" in wrapper
    assert "--skip-ibkr-account-snapshot" in wrapper
    assert "--trade-log-root" in wrapper


def test_fiveday_evidence_packet_marks_complete_session(tmp_path: Path) -> None:
    session = "2026-06-08"
    run_id = f"daily_paper_autopilot_{session}"
    log_dir = tmp_path / "v4/logs/paper_trading" / session
    runtime_dir = tmp_path / "v4/runtime"
    entitlement_dir = tmp_path / "v4/audit/ibkr_live_data_entitlements"
    log_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    entitlement_dir.mkdir(parents=True)
    (runtime_dir / "protocol101_paper_order_enablement.json").write_text(
        '{"paper_orders_enabled": true, "real_money_trading": false, "required_env": "V4_ALLOW_IBKR_PAPER_ORDERS=YES"}'
    )
    (entitlement_dir / "summary.json").write_text('{"decision": "pass"}')
    (log_dir / f"{run_id}.jsonl").write_text(
        "\n".join(
            [
                '{"timestamp":"2026-06-08T13:30:00+00:00","event_type":"market_snapshot","session":"2026-06-08","run_id":"daily_paper_autopilot_2026-06-08","broker_order_endpoint_called":false}',
                '{"timestamp":"2026-06-08T13:31:00+00:00","event_type":"candidate_set","session":"2026-06-08","run_id":"daily_paper_autopilot_2026-06-08","broker_order_endpoint_called":false}',
                '{"timestamp":"2026-06-08T13:32:00+00:00","event_type":"model_decision","session":"2026-06-08","run_id":"daily_paper_autopilot_2026-06-08","broker_order_endpoint_called":false,"model_decision":{"action":"wait"}}',
                '{"timestamp":"2026-06-08T13:33:00+00:00","event_type":"risk_gate","session":"2026-06-08","run_id":"daily_paper_autopilot_2026-06-08","broker_order_endpoint_called":false,"risk_gate":{"reason":"no_entry_intent"}}',
                '{"timestamp":"2026-06-08T20:00:00+00:00","event_type":"paper_account_state","session":"2026-06-08","run_id":"daily_paper_autopilot_2026-06-08","broker_order_endpoint_called":false}',
            ]
        )
        + "\n"
    )

    packet = build_packet(
        repo_root=tmp_path,
        session=session,
        run_id=run_id,
        trade_log_root=Path("v4/logs/paper_trading"),
        runtime_flag=Path("v4/runtime/protocol101_paper_order_enablement.json"),
        entitlement_summary=Path("v4/audit/ibkr_live_data_entitlements/summary.json"),
    )

    assert packet["decision"] == "complete_fiveday_paper_submit_session_ready_for_paired_replay_review"
    assert packet["completion_checks"]["has_model_decisions"] is True
