from __future__ import annotations

import plistlib
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from v4.scripts.run_tuesday_no_order_evidence_packet import build_packet
from v4.scripts.run_tuesday_protocol101_paper_fill_observation import choose_observation_candidate, local_observation_approval


ROOT = Path(__file__).resolve().parents[2]
OPS_IBKR = ROOT / "v4" / "ops" / "ibkr"
LAUNCHD = ROOT / "v4" / "ops" / "launchd"


def load_plist(name: str) -> dict:
    with (LAUNCHD / name).open("rb") as fh:
        return plistlib.load(fh)


def test_tuesday_paper_fill_observation_script_requires_explicit_paper_approval() -> None:
    script = (OPS_IBKR / "run_tuesday_paper_fill_observation_collection.sh").read_text()

    assert "export V4_ALLOW_IBKR_PAPER_ORDERS=YES" in script
    assert "export TUESDAY_PAPER_FILL_OBSERVATIONS_APPROVED=YES" in script
    assert "export PROTOCOL101_ENTRY_BRIDGE_MODE=paper-submit" in script
    assert "export PROTOCOL101_ENABLE_PAPER_ORDERS=YES" in script
    assert "export PROTOCOL101_ACKNOWLEDGE_PAPER_LOSS=YES" in script
    assert "--enable-paper-orders" in script
    assert "--acknowledge-paper-loss" in script
    assert "--max-probes" in script
    assert "--max-filled-round-trips" in script
    assert "warm_launchd_python_deps.py" in script
    assert "blocked_ibkr_api_unavailable_before_paper_fill_observation" in script
    assert "tuesday-paper-fill-observation-api-probe.json" in script


def test_tuesday_paper_fill_observation_launchd_is_date_guarded_and_capped() -> None:
    plist = load_plist("com.autoresearch.tuesday.paper-fill-observation.plist")

    assert plist["Label"] == "com.autoresearch.tuesday.paper-fill-observation"
    assert plist["StartCalendarInterval"] == {"Month": 5, "Day": 26, "Hour": 6, "Minute": 31}
    env = plist["EnvironmentVariables"]
    assert env["TUESDAY_EVIDENCE_TARGET_DATE"] == "2026-05-26"
    assert env["V4_ALLOW_IBKR_PAPER_ORDERS"] == "YES"
    assert env["TUESDAY_PAPER_FILL_OBSERVATIONS_APPROVED"] == "YES"
    assert env["PROTOCOL101_ENTRY_BRIDGE_MODE"] == "paper-submit"
    assert env["PROTOCOL101_ENABLE_PAPER_ORDERS"] == "YES"
    assert env["PROTOCOL101_ACKNOWLEDGE_PAPER_LOSS"] == "YES"
    assert env["TUESDAY_EVIDENCE_MAX_PROBES"] == "8"
    assert env["TUESDAY_EVIDENCE_MAX_FILLED_ROUND_TRIPS"] == "3"


def test_tuesday_gateway_preflight_postsession_schedule() -> None:
    gateway = load_plist("com.autoresearch.tuesday.ibgateway.paper.plist")
    preflight = load_plist("com.autoresearch.tuesday.protocol101.paper-preflight.plist")
    post = load_plist("com.autoresearch.tuesday.evidence-postsession.plist")
    shutdown = load_plist("com.autoresearch.tuesday.ibgateway.paper-shutdown.plist")

    assert gateway["StartCalendarInterval"] == {"Month": 5, "Day": 26, "Hour": 6, "Minute": 5}
    assert preflight["StartCalendarInterval"] == {"Month": 5, "Day": 26, "Hour": 6, "Minute": 20}
    assert post["StartCalendarInterval"] == {"Month": 5, "Day": 26, "Hour": 13, "Minute": 15}
    assert shutdown["StartCalendarInterval"] == {"Month": 5, "Day": 26, "Hour": 13, "Minute": 35}


def test_installer_does_not_install_order_enabled_paper_session() -> None:
    installer = (LAUNCHD / "install_tuesday_no_order_evidence.sh").read_text()

    assert "com.autoresearch.protocol101.paper-session" in installer
    assert "com.autoresearch.premiumblend.no-order-surface-check" in installer
    assert "com.autoresearch.tuesday.no-order-evidence" in installer
    assert "launchctl disable \"gui/$UID/$existing_label\"" in installer
    assert "com.autoresearch.tuesday.paper-fill-observation" in installer
    assert "com.autoresearch.protocol101.paper-session.plist" not in installer


def test_tuesday_evidence_packet_detects_no_order_success(tmp_path: Path) -> None:
    session = "2026-05-26"
    p245 = tmp_path / "v4/audit/autoresearch/v4_aplus_hypothesis_245_premium_blend_live_surface_autotest" / session / "surface"
    p160 = tmp_path / "v4/audit/autoresearch/v4_aplus_hypothesis_160_protocol101_persistent_paper_trader" / session / "intent"
    logs = tmp_path / "v4/logs/paper_trading" / session
    p245.mkdir(parents=True)
    p160.mkdir(parents=True)
    logs.mkdir(parents=True)
    (p245 / "summary.json").write_text('{"decision":"pass_live_surface_autotest_ready_for_challenger_shadow_protocol101_default_unchanged","paper_orders_submitted":0,"broker_order_endpoint_called":false}')
    (p160 / "summary.json").write_text('{"decision":"pass_persistent_live_intent_shadow_logged","paper_orders_submitted":0,"broker_order_endpoint_called":false,"decision_count":3}')
    (logs / "evidence.jsonl").write_text(
        '{"event_type":"model_decision","broker_order_endpoint_called":false,"session":"2026-05-26"}\n'
    )

    packet = build_packet(repo_root=tmp_path, session_date=session)

    assert packet["decision"] == "pass_tuesday_no_order_evidence_ready_for_review"
    assert packet["trade_log"]["paper_order_submitted_rows"] == 0


def test_tuesday_evidence_packet_detects_approved_paper_fill_observations(tmp_path: Path) -> None:
    session = "2026-05-26"
    paper = tmp_path / "v4/audit/autoresearch/tuesday_protocol101_paper_fill_observation" / session / "fills"
    logs = tmp_path / "v4/logs/paper_trading" / session
    paper.mkdir(parents=True)
    logs.mkdir(parents=True)
    (paper / "summary.json").write_text(
        '{"decision":"paper_fill_observations_collected_for_execution_truth_review","paper_orders_submitted":2,"broker_order_endpoint_called":true}'
    )
    (paper / "execution_observations.jsonl").write_text(
        '{"fill_status":"filled","exit_fill_status":"filled","broker_order_endpoint_called":true,"open_position_risk":false}\n'
    )
    (logs / "paper.jsonl").write_text(
        '{"event_type":"paper_order_submitted","broker_order_endpoint_called":true,"session":"2026-05-26"}\n'
    )

    packet = build_packet(repo_root=tmp_path, session_date=session)

    assert packet["decision"] == "pass_tuesday_paper_fill_observations_ready_for_review"
    assert packet["execution_observations"]["filled_round_trips"] == 1


def test_tuesday_evidence_packet_flags_unattributed_broker_rows(tmp_path: Path) -> None:
    session = "2026-05-26"
    logs = tmp_path / "v4/logs/paper_trading" / session
    logs.mkdir(parents=True)
    (logs / "unsafe.jsonl").write_text(
        '{"event_type":"paper_order_submitted","broker_order_endpoint_called":true,"session":"2026-05-26"}\n'
    )

    packet = build_packet(repo_root=tmp_path, session_date=session)

    assert packet["decision"] == "fail_unattributed_broker_order_endpoint_rows"


def test_paper_fill_observation_permission_requires_dedicated_approval_env() -> None:
    args = SimpleNamespace(enable_paper_orders=True, acknowledge_paper_loss=True)

    blocked = local_observation_approval(args, {"V4_ALLOW_IBKR_PAPER_ORDERS": "YES"})
    passed = local_observation_approval(
        args,
        {"V4_ALLOW_IBKR_PAPER_ORDERS": "YES", "TUESDAY_PAPER_FILL_OBSERVATIONS_APPROVED": "YES"},
    )

    assert blocked["passed"] is False
    assert "paper_fill_observation_approval_env_not_set" in blocked["reasons"]
    assert passed["passed"] is True


def test_choose_observation_candidate_prefers_protocol101_selection_but_checks_fillability() -> None:
    candidates = pd.DataFrame(
        [
            {"contract_id": "bad", "score": 100.0},
            {"contract_id": "selected", "score": 80.0},
        ]
    )
    lookup = {
        "bad": {"bid": 1.0, "ask": 200.0, "quote_age_ms": 100, "right": "C"},
        "selected": {"bid": 9.9, "ask": 10.0, "quote_age_ms": 100, "right": "P"},
    }

    row, quote, reason = choose_observation_candidate(
        candidates=candidates,
        prediction={"action": "enter", "selected": {"contract_id": "selected"}},
        lookup=lookup,
        observed_contract_counts={},
        paper_cash=10_000.0,
        max_observations_per_contract=2,
    )

    assert row["contract_id"] == "selected"
    assert quote["ask"] == 10.0
    assert reason == "protocol101_selected"
