from __future__ import annotations

import json
import plistlib
import subprocess
from pathlib import Path

from v4.live.paper_model_registry import DEFAULT_PAPER_TRADING_DEFAULT, load_paper_model_selection
from v4.scripts.run_daily_paper_autopilot import build_child_args


ROOT = Path(__file__).resolve().parents[2]


def test_paper_default_registry_points_to_protocol101_approved_default() -> None:
    selection = load_paper_model_selection(ROOT / DEFAULT_PAPER_TRADING_DEFAULT)

    assert selection.model_id == "protocol101"
    assert selection.display_name == "Protocol101"
    assert selection.runner_kind == "protocol101_persistent"
    assert selection.default_run_id("2026-05-27") == "daily_paper_autopilot_2026-05-27"
    assert "protocol101_manifest" in selection.arguments


def test_daily_paper_autopilot_print_selection_does_not_run_broker() -> None:
    result = subprocess.run(
        [
            "python3",
            "-m",
            "v4.scripts.run_daily_paper_autopilot",
            "--paper-default-registry",
            str(ROOT / DEFAULT_PAPER_TRADING_DEFAULT),
            "--session-date",
            "2026-05-27",
            "--print-selection",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["feature_name"] == "daily paper autopilot"
    assert payload["model_id"] == "protocol101"
    assert payload["run_id"] == "daily_paper_autopilot_2026-05-27"


def test_paper_session_plist_uses_daily_paper_autopilot_registry() -> None:
    with (ROOT / "v4/ops/launchd/com.autoresearch.protocol101.paper-session.plist").open("rb") as fh:
        plist = plistlib.load(fh)

    env = plist["EnvironmentVariables"]
    assert env["DAILY_PAPER_AUTOPILOT_MODEL_REGISTRY"].endswith("v4/promotion/PAPER_TRADING_DEFAULT.json")
    assert env["DAILY_PAPER_AUTOPILOT_MODE"] == "paper-submit"


def test_protocol101_child_args_are_built_from_registry() -> None:
    selection = load_paper_model_selection(ROOT / DEFAULT_PAPER_TRADING_DEFAULT)

    class Args:
        mode = "paper-submit"
        out_root = None
        trade_log_root = None
        ibkr_host = "127.0.0.1"
        ibkr_port = 4002
        ibkr_auto_ports = "4002"
        ibkr_client_id = 160
        account_id = None
        paper_cash = 10_000.0
        open_positions = 0
        quantity = 1
        order_timeout_seconds = 15.0
        forced_flat_time = "15:55"
        market_close_time = "16:00"
        min_edge = 25.0
        live_strikes_around_atm = 10
        quote_loop_seconds = 1.0
        entry_decision_interval_seconds = 60.0
        entry_decision_mode = "minute"
        decision_interval_seconds = None
        heartbeat_seconds = 30.0
        contract_refresh_seconds = 60.0
        refresh_contracts_drift_points = 15.0
        min_live_context_minutes = 30.0
        reconnect_sleep_seconds = 5.0
        max_reconnects = 200
        max_decisions = 100_000
        allow_delayed_market_data = False
        enable_paper_orders = True
        acknowledge_paper_loss = True
        skip_market_clock = False

    args = build_child_args(selection, args=Args(), session="2026-05-27", run_id="daily_paper_autopilot_2026-05-27")

    assert "--protocol101-manifest" in args
    assert selection.arguments["protocol101_manifest"] in args
    assert "--enable-paper-orders" in args
    assert "--acknowledge-paper-loss" in args
