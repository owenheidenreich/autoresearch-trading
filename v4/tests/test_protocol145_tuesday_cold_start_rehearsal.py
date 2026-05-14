from __future__ import annotations

from v4.scripts.run_protocol145_tuesday_cold_start_rehearsal import candidate_ports, decide


def _launchd(loaded: bool = True) -> dict:
    return {
        "gateway": {"loaded": loaded},
        "preflight": {"loaded": loaded},
    }


def test_candidate_ports_deduplicates_and_keeps_order() -> None:
    assert candidate_ports("4000,4002,4000,bad,7497") == [4000, 4002, 7497]


def test_decide_classifies_expected_login_blocker() -> None:
    decision = decide(
        before={"any_open": False},
        after={"any_open": False},
        launchd=_launchd(True),
        start_result={"returncode": 1},
        skip_start=False,
    )

    assert decision == "expected_blocker_gateway_login_required_or_api_port_closed"


def test_decide_passes_when_api_opens_after_start() -> None:
    decision = decide(
        before={"any_open": False},
        after={"any_open": True},
        launchd=_launchd(True),
        start_result={"returncode": 0},
        skip_start=False,
    )

    assert decision == "pass_cold_start_api_ready"


def test_decide_blocks_when_launchagents_missing() -> None:
    decision = decide(
        before={"any_open": False},
        after={"any_open": False},
        launchd=_launchd(False),
        start_result={"returncode": 1},
        skip_start=False,
    )

    assert decision == "blocked_launchagents_not_loaded"
