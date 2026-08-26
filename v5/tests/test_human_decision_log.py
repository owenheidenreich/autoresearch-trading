"""Outcome-blind tests for the local human-decision journal foundation."""
from __future__ import annotations

import ast
import hashlib
import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from v5.ops import record_human_decision as cli
from v5.research import human_decision_log as hdl


T0 = datetime(2026, 8, 24, 17, 0, tzinfo=timezone.utc)
STATE_HASH = "a" * 64
UNIVERSE_HASH = "b" * 64
CALL = "SPXW  260824C07650000"
PUT = "SPXW  260824P07650000"


def _at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _journal(tmp_path: Path) -> hdl.HumanDecisionLog:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return hdl.HumanDecisionLog.initialize(
        tmp_path / "human.jsonl", log_id="synthetic-human-log", created_at=T0
    )


def _on(log: hdl.HumanDecisionLog, *, second: int = 1, event_id: str = "on") -> None:
    log._append_for_test(
        kind=hdl.MONITORING_ON,
        event_id=event_id,
        session="2026-08-24",
        occurred_at=_at(second),
        appended_at=_at(second),
    )


def _prompt(
    log: hdl.HumanDecisionLog, *, second: int = 2, event_id: str = "prompt"
) -> None:
    log._append_for_test(
        kind=hdl.PROMPT,
        event_id=event_id,
        session="2026-08-24",
        occurred_at=_at(second),
        appended_at=_at(second),
        market_state_sha256=STATE_HASH,
        universe_sha256=UNIVERSE_HASH,
    )


def _decision(
    log: hdl.HumanDecisionLog,
    *,
    second: int,
    event_id: str,
    decision_id: str,
    action: str,
    position_state: str,
    contract_osi: str | None = None,
    appended_second: int | None = None,
    prompt_event_id: str | None = None,
    include_owner_intent: bool = True,
    owner_intent_override: str | None = None,
    information_sources: list[str] | None = None,
    reason_codes: list[str] | None = None,
    note: str | None = None,
) -> dict:
    is_open = action in {hdl.OPEN_CALL, hdl.OPEN_PUT}
    default_owner_intent = {
        hdl.OPEN_CALL: "ENTER_LONG_CALL",
        hdl.OPEN_PUT: "ENTER_LONG_PUT",
        hdl.EXIT: "PROFIT_TARGET",
    }.get(action)
    return log._append_for_test(
        kind=hdl.DECISION,
        event_id=event_id,
        decision_id=decision_id,
        session="2026-08-24",
        occurred_at=_at(second),
        appended_at=_at(second if appended_second is None else appended_second),
        monitoring_state="ON",
        position_state=position_state,
        action=action,
        contract_osi=contract_osi,
        market_state_sha256=STATE_HASH,
        universe_sha256=UNIVERSE_HASH,
        prompt_event_id=prompt_event_id,
        spontaneous=prompt_event_id is None,
        information_sources=information_sources or ["SPX_CHART"],
        confidence=0.75,
        reason_codes=reason_codes or ["SYNTHETIC_FIXTURE"],
        program_contract_sha256=hdl.PROGRAM_CONTRACT_SHA256,
        risk_contract_sha256=hdl.RISK_CONTRACT_SHA256,
        quantity=1 if is_open else None,
        intended_order_type="LIMIT" if is_open else None,
        intended_limit_price=10.0 if is_open else None,
        estimated_entry_debit_usd=1001.0 if is_open else None,
        declared_stop_fraction=-0.4 if is_open else None,
        owner_intent=(
            owner_intent_override or default_owner_intent
            if include_owner_intent and (is_open or action == hdl.EXIT)
            else None
        ),
        note=note,
    )


def test_initialize_is_exclusive_private_and_deterministic_utc(tmp_path: Path) -> None:
    path = tmp_path / "human.jsonl"
    header = hdl.initialize_log(
        path,
        log_id="synthetic",
        created_at="2026-08-24T10:00:00-07:00",
    )
    assert header["created_at"] == "2026-08-24T17:00:00.000000Z"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    line = path.read_text(encoding="utf-8").rstrip("\n")
    assert line == json.dumps(
        json.loads(line), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    with pytest.raises(hdl.HumanDecisionLogError, match="will not be overwritten"):
        hdl.initialize_log(path, log_id="replacement", created_at=T0)


def test_event_kind_vocabulary_is_exact() -> None:
    assert hdl.EVENT_KINDS == frozenset(
        {"MONITORING_ON", "MONITORING_OFF", "PROMPT", "DECISION", "CORRECTION"}
    )


def test_action_vocabulary_is_exact() -> None:
    assert hdl.DECISIONS == frozenset(
        {"WAIT", "OPEN_CALL", "OPEN_PUT", "HOLD", "EXIT"}
    )


def test_monitoring_off_unanswered_prompt_and_missing_are_not_wait(tmp_path: Path) -> None:
    unanswered = _journal(tmp_path / "unanswered")
    _on(unanswered)
    _prompt(unanswered)
    pending = unanswered.status(now=_at(20))
    assert pending["pending_prompts"] == 1
    assert pending["no_response_prompts"] == 0
    assert pending["explicit_waits"] == 0
    expired = unanswered.status(now=_at(33))
    assert expired["pending_prompts"] == 0
    assert expired["no_response_prompts"] == 1
    assert expired["explicit_waits"] == 0
    assert expired["observation_status"] == "OBSERVED"

    waited = _journal(tmp_path / "waited")
    _on(waited)
    _prompt(waited)
    _decision(
        waited,
        second=3,
        event_id="wait",
        decision_id="decision-wait",
        action=hdl.WAIT,
        position_state="FLAT",
        prompt_event_id="prompt",
    )
    waited._append_for_test(
        kind=hdl.MONITORING_OFF,
        event_id="off",
        session="2026-08-24",
        occurred_at=_at(4),
        appended_at=_at(4),
    )
    status = waited.status(now=_at(40))
    assert status["explicit_waits"] == 1
    assert status["answered_prompts"] == 1
    assert status["no_response_prompts"] == 0
    assert status["observation_status"] == "UNOBSERVED"


def test_decision_requires_explicit_monitoring_and_snapshot_hashes(tmp_path: Path) -> None:
    log = _journal(tmp_path)
    with pytest.raises(hdl.HumanDecisionLogError, match="monitoring_state ON"):
        log._append_for_test(
            kind=hdl.DECISION,
            event_id="outside",
            decision_id="d-outside",
            session="2026-08-24",
            occurred_at=_at(1),
            appended_at=_at(1),
            monitoring_state="ON",
            position_state="FLAT",
            action=hdl.WAIT,
            market_state_sha256=STATE_HASH,
            universe_sha256=UNIVERSE_HASH,
        )
    _on(log)
    with pytest.raises(hdl.HumanDecisionLogError, match="explicit monitoring_state"):
        log._append_for_test(
            kind=hdl.DECISION,
            event_id="implicit",
            decision_id="d-implicit",
            session="2026-08-24",
            occurred_at=_at(2),
            appended_at=_at(2),
            position_state="FLAT",
            action=hdl.WAIT,
            market_state_sha256=STATE_HASH,
            universe_sha256=UNIVERSE_HASH,
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="market_state_sha256"):
        log._append_for_test(
            kind=hdl.DECISION,
            event_id="no-state",
            decision_id="d-no-state",
            session="2026-08-24",
            occurred_at=_at(2),
            appended_at=_at(2),
            monitoring_state="ON",
            position_state="FLAT",
            action=hdl.WAIT,
            universe_sha256=UNIVERSE_HASH,
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="universe_sha256"):
        log._append_for_test(
            kind=hdl.DECISION,
            event_id="no-universe",
            decision_id="d-no-universe",
            session="2026-08-24",
            occurred_at=_at(2),
            appended_at=_at(2),
            monitoring_state="ON",
            position_state="FLAT",
            action=hdl.WAIT,
            market_state_sha256=STATE_HASH,
        )
    assert log.verify(now=_at(2)).events == 1


def test_decision_origin_and_prompt_link_are_explicit(tmp_path: Path) -> None:
    log = _journal(tmp_path)
    _on(log)
    _prompt(log)
    prompted = _decision(
        log,
        second=3,
        event_id="prompted-wait",
        decision_id="d-prompted-wait",
        action=hdl.WAIT,
        position_state="FLAT",
        prompt_event_id="prompt",
    )
    assert prompted["spontaneous"] is False
    assert prompted["prompt_event_id"] == "prompt"
    spontaneous = _decision(
        log,
        second=4,
        event_id="spontaneous-wait",
        decision_id="d-spontaneous-wait",
        action=hdl.WAIT,
        position_state="FLAT",
    )
    assert spontaneous["spontaneous"] is True
    assert spontaneous["prompt_event_id"] is None


def test_position_state_machine_open_hold_exit_and_wait(tmp_path: Path) -> None:
    log = _journal(tmp_path)
    _on(log)
    opened = _decision(
        log,
        second=2,
        event_id="open",
        decision_id="d-open",
        action=hdl.OPEN_CALL,
        position_state="FLAT",
        contract_osi=CALL,
    )
    assert opened["position_before"] == "FLAT"
    assert opened["position_after"] == "OPEN"
    with pytest.raises(hdl.HumanDecisionLogError, match="WAIT is a flat-state"):
        _decision(
            log,
            second=3,
            event_id="bad-wait",
            decision_id="d-bad-wait",
            action=hdl.WAIT,
            position_state="OPEN",
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="OPEN requires position_state FLAT"):
        _decision(
            log,
            second=3,
            event_id="second-open",
            decision_id="d-second-open",
            action=hdl.OPEN_PUT,
            position_state="OPEN",
            contract_osi=PUT,
        )
    held = _decision(
        log,
        second=3,
        event_id="hold",
        decision_id="d-hold",
        action=hdl.HOLD,
        position_state="OPEN",
        contract_osi=CALL,
    )
    assert held["position_after"] == "OPEN"
    with pytest.raises(hdl.HumanDecisionLogError, match="while a position is open"):
        log._append_for_test(
            kind=hdl.MONITORING_OFF,
            event_id="premature-off",
            session="2026-08-24",
            occurred_at=_at(4),
            appended_at=_at(4),
        )
    exited = _decision(
        log,
        second=4,
        event_id="exit",
        decision_id="d-exit",
        action=hdl.EXIT,
        position_state="OPEN",
        contract_osi=CALL,
    )
    assert exited["position_after"] == "FLAT"
    with pytest.raises(hdl.HumanDecisionLogError, match="requires explicit position_state OPEN"):
        _decision(
            log,
            second=5,
            event_id="flat-exit",
            decision_id="d-flat-exit",
            action=hdl.EXIT,
            position_state="FLAT",
            contract_osi=CALL,
        )


@pytest.mark.parametrize(
    ("symbol", "action", "message"),
    [
        ("SPXW 260824C07650000", hdl.OPEN_CALL, "21-character"),
        ("SPXW  260825C07650000", hdl.OPEN_CALL, "does not equal session"),
        ("SPXW  260824P07650000", hdl.OPEN_CALL, "does not match C"),
        ("SPXW  260832C07650000", hdl.OPEN_CALL, "invalid expiry"),
        ("SPXW  260824C00000000", hdl.OPEN_CALL, "strike must be positive"),
    ],
)
def test_open_requires_exact_same_day_spxw_osi_and_owner_intent(
    tmp_path: Path, symbol: str, action: str, message: str
) -> None:
    log = _journal(tmp_path)
    _on(log)
    with pytest.raises(hdl.HumanDecisionLogError, match=message):
        _decision(
            log,
            second=2,
            event_id="open",
            decision_id="d-open",
            action=action,
            position_state="FLAT",
            contract_osi=symbol,
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="owner_intent"):
        _decision(
            log,
            second=2,
            event_id="open-without-intent",
            decision_id="d-open-without-intent",
            action=hdl.OPEN_CALL,
            position_state="FLAT",
            contract_osi=CALL,
            include_owner_intent=False,
        )
    assert log.verify(now=_at(2)).position_state == "FLAT"


def test_risk_metadata_matches_frozen_contract_without_simulation() -> None:
    v2_contract_path = (
        Path(__file__).resolve().parents[1]
        / "work/human-policy-foundation/PROGRAM_CONTRACT_V2.json"
    )
    v1_contract_path = v2_contract_path.with_name("PROGRAM_CONTRACT_V1.json")
    v2_contract = json.loads(v2_contract_path.read_text(encoding="utf-8"))
    v1_contract = json.loads(v1_contract_path.read_text(encoding="utf-8"))
    assert v2_contract["self_hash"]["value"] == hdl.PROGRAM_CONTRACT_SHA256
    assert (
        v2_contract["supersession"]["base_contract_semantic_sha256"]
        == v1_contract["self_hash"]["value"]
    )
    encoded = json.dumps(
        v1_contract["risk_contract_v1"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == hdl.RISK_CONTRACT_SHA256
    assert hdl.ENTRY_QUANTITY_CONTRACTS == 1
    assert hdl.MAX_TOTAL_DEBIT_USD == 2500.0
    assert hdl.MAXIMUM_DECLARED_STOP_FRACTION == -0.4


def test_hold_and_exit_require_the_exact_held_contract(tmp_path: Path) -> None:
    log = _journal(tmp_path)
    _on(log)
    _decision(
        log,
        second=2,
        event_id="open",
        decision_id="d-open",
        action=hdl.OPEN_CALL,
        position_state="FLAT",
        contract_osi=CALL,
    )
    with pytest.raises(hdl.HumanDecisionLogError, match="differs from the held contract"):
        _decision(
            log,
            second=3,
            event_id="wrong-hold",
            decision_id="d-wrong-hold",
            action=hdl.HOLD,
            position_state="OPEN",
            contract_osi=PUT,
        )


def test_decision_append_lag_boundary_future_and_over_limit(tmp_path: Path) -> None:
    at_boundary = _journal(tmp_path / "boundary")
    _on(at_boundary)
    _decision(
        at_boundary,
        second=2,
        appended_second=2 + hdl.MAX_DECISION_APPEND_LAG_SECONDS,
        event_id="boundary-wait",
        decision_id="d-boundary-wait",
        action=hdl.WAIT,
        position_state="FLAT",
    )
    assert at_boundary.verify(now=_at(40)).decisions == 1

    too_late = _journal(tmp_path / "late")
    _on(too_late)
    with pytest.raises(hdl.HumanDecisionLogError, match="appended retroactively"):
        _decision(
            too_late,
            second=2,
            appended_second=33,
            event_id="late-wait",
            decision_id="d-late-wait",
            action=hdl.WAIT,
            position_state="FLAT",
        )

    future = _journal(tmp_path / "future")
    _on(future)
    with pytest.raises(hdl.HumanDecisionLogError, match="in the future"):
        _decision(
            future,
            second=3,
            appended_second=2,
            event_id="future-wait",
            decision_id="d-future-wait",
            action=hdl.WAIT,
            position_state="FLAT",
        )


def test_production_append_api_owns_both_seal_clocks(tmp_path: Path) -> None:
    log = _journal(tmp_path)
    with pytest.raises(TypeError, match="appended_at"):
        log.append(
            kind=hdl.MONITORING_ON,
            event_id="forged-clock",
            session="2026-08-24",
            occurred_at=_at(1),
            appended_at=_at(1),
            local_monotonic_ns=1,
        )
    assert log.verify(now=_at(1)).events == 0
    with pytest.raises(TypeError, match="now"):
        cli.main(["verify", "--log", str(log.path)], now=_at(1))


def test_closed_vocabularies_and_training_projection_refuse_free_text(
    tmp_path: Path,
) -> None:
    log = _journal(tmp_path)
    _on(log)
    with pytest.raises(hdl.HumanDecisionLogError, match="note is audit-only"):
        _decision(
            log,
            second=2,
            event_id="decision-note",
            decision_id="d-decision-note",
            action=hdl.WAIT,
            position_state="FLAT",
            note="future P&L was positive",
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="frozen vocabulary"):
        _decision(
            log,
            second=2,
            event_id="free-source",
            decision_id="d-free-source",
            action=hdl.WAIT,
            position_state="FLAT",
            information_sources=["FUTURE_PNL"],
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="frozen vocabulary"):
        _decision(
            log,
            second=2,
            event_id="free-reason",
            decision_id="d-free-reason",
            action=hdl.WAIT,
            position_state="FLAT",
            reason_codes=["RETURN_WAS_37_PERCENT"],
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="owner_intent"):
        _decision(
            log,
            second=2,
            event_id="free-intent",
            decision_id="d-free-intent",
            action=hdl.OPEN_CALL,
            position_state="FLAT",
            contract_osi=CALL,
            owner_intent_override="I know this trade wins",
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="note is audit-only"):
        log._append_for_test(
            kind=hdl.PROMPT,
            event_id="prompt-note",
            session="2026-08-24",
            occurred_at=_at(2),
            appended_at=_at(2),
            market_state_sha256=STATE_HASH,
            universe_sha256=UNIVERSE_HASH,
            note="future outcome text",
        )

    decision = _decision(
        log,
        second=2,
        event_id="structured-wait",
        decision_id="d-structured-wait",
        action=hdl.WAIT,
        position_state="FLAT",
    )
    with pytest.raises(hdl.HumanDecisionLogError, match="frozen audit annotation"):
        log._append_for_test(
            kind=hdl.CORRECTION,
            event_id="free-correction",
            session="2026-08-24",
            occurred_at=_at(3),
            appended_at=_at(3),
            corrects_event_id=decision["event_id"],
            note="winner after the fact",
        )
    log._append_for_test(
        kind=hdl.CORRECTION,
        event_id="typed-correction",
        session="2026-08-24",
        occurred_at=_at(3),
        appended_at=_at(3),
        corrects_event_id=decision["event_id"],
        note="CONFIDENCE_TYPO",
    )
    projected = log.training_decisions()
    assert len(projected) == 1
    assert tuple(projected[0]) == hdl.TRAINING_DECISION_FIELDS
    assert projected[0]["action"] == hdl.WAIT
    assert {
        "appended_at",
        "decision_id",
        "event_id",
        "local_monotonic_ns",
        "note",
        "prompt_event_id",
        "corrects_event_id",
        "previous_hash",
        "record_hash",
    }.isdisjoint(projected[0])


def test_backdated_event_and_backward_append_clock_are_refused(tmp_path: Path) -> None:
    log = _journal(tmp_path)
    log._append_for_test(
        kind=hdl.MONITORING_ON,
        event_id="on",
        session="2026-08-24",
        occurred_at=_at(1),
        appended_at=_at(10),
    )
    with pytest.raises(hdl.HumanDecisionLogError, match="appended_at moved backward"):
        log._append_for_test(
            kind=hdl.PROMPT,
            event_id="backward-append",
            session="2026-08-24",
            occurred_at=_at(2),
            appended_at=_at(9),
            market_state_sha256=STATE_HASH,
            universe_sha256=UNIVERSE_HASH,
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="backdated"):
        log._append_for_test(
            kind=hdl.PROMPT,
            event_id="backdated",
            session="2026-08-24",
            occurred_at=_at(0),
            appended_at=_at(11),
            market_state_sha256=STATE_HASH,
            universe_sha256=UNIVERSE_HASH,
        )


def test_prompt_link_is_backward_single_use_and_causal(tmp_path: Path) -> None:
    log = _journal(tmp_path)
    _on(log)
    _prompt(log)
    _decision(
        log,
        second=3,
        event_id="wait",
        decision_id="d-wait",
        action=hdl.WAIT,
        position_state="FLAT",
        prompt_event_id="prompt",
    )
    with pytest.raises(hdl.HumanDecisionLogError, match="already has"):
        _decision(
            log,
            second=4,
            event_id="second-response",
            decision_id="d-second-response",
            action=hdl.WAIT,
            position_state="FLAT",
            prompt_event_id="prompt",
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="point backward"):
        _decision(
            log,
            second=4,
            event_id="missing-prompt",
            decision_id="d-missing-prompt",
            action=hdl.WAIT,
            position_state="FLAT",
            prompt_event_id="not-yet-written",
        )


def test_duplicate_event_and_decision_ids_are_refused_without_writing(tmp_path: Path) -> None:
    log = _journal(tmp_path)
    _on(log)
    _decision(
        log,
        second=2,
        event_id="wait",
        decision_id="same-decision",
        action=hdl.WAIT,
        position_state="FLAT",
    )
    with pytest.raises(hdl.HumanDecisionLogError, match="duplicate event_id"):
        log._append_for_test(
            kind=hdl.CORRECTION,
            event_id="wait",
            session="2026-08-24",
            occurred_at=_at(3),
            appended_at=_at(3),
            corrects_event_id="wait",
            note="AUDIT_TEST_FIXTURE",
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="duplicate decision_id"):
        _decision(
            log,
            second=3,
            event_id="another-wait",
            decision_id="same-decision",
            action=hdl.WAIT,
            position_state="FLAT",
        )
    assert log.verify(now=_at(3)).events == 2


def test_correction_is_a_new_linked_event_not_an_overwrite(tmp_path: Path) -> None:
    log = _journal(tmp_path)
    _on(log)
    _decision(
        log,
        second=2,
        event_id="wait",
        decision_id="d-wait",
        action=hdl.WAIT,
        position_state="FLAT",
    )
    correction = log._append_for_test(
        kind=hdl.CORRECTION,
        event_id="correction",
        session="2026-08-24",
        occurred_at=_at(100),
        appended_at=_at(100),
        corrects_event_id="wait",
        note="CONFIDENCE_TYPO",
    )
    assert correction["corrects_event_id"] == "wait"
    status = log.status(now=_at(100))
    assert status["decisions"] == 1
    assert status["explicit_waits"] == 1
    assert status["position_state"] == "FLAT"
    with pytest.raises(hdl.HumanDecisionLogError, match="point backward"):
        log._append_for_test(
            kind=hdl.CORRECTION,
            event_id="bad-correction",
            session="2026-08-24",
            occurred_at=_at(101),
            appended_at=_at(101),
            corrects_event_id="future-event",
            note="AUDIT_TEST_FIXTURE",
        )
    with pytest.raises(hdl.HumanDecisionLogError, match="cannot carry decision"):
        log._append_for_test(
            kind=hdl.CORRECTION,
            event_id="mutation-attempt",
            session="2026-08-24",
            occurred_at=_at(101),
            appended_at=_at(101),
            action=hdl.OPEN_CALL,
            contract_osi=CALL,
            corrects_event_id="wait",
            note="AUDIT_TEST_FIXTURE",
        )


def test_append_chain_rejects_mutation_deletion_duplicate_and_backdated_sequence(tmp_path: Path) -> None:
    path = tmp_path / "human.jsonl"
    log = hdl.HumanDecisionLog.initialize(
        path, log_id="synthetic", created_at=T0
    )
    _on(log)
    _prompt(log)
    _decision(
        log,
        second=3,
        event_id="wait",
        decision_id="d-wait",
        action=hdl.WAIT,
        position_state="FLAT",
        prompt_event_id="prompt",
    )
    original = path.read_bytes()
    lines = original.decode().splitlines()

    changed = [json.loads(line) for line in lines]
    changed[-1]["action"] = hdl.OPEN_CALL
    path.write_text("\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in changed) + "\n")
    with pytest.raises(hdl.HumanDecisionLogError, match="record hash mismatch"):
        hdl.verify_log(path, now=_at(4))

    path.write_bytes(original)
    path.write_text("\n".join([lines[0], lines[1], lines[3]]) + "\n")
    with pytest.raises(hdl.HumanDecisionLogError, match="sequence|chain broken"):
        hdl.verify_log(path, now=_at(4))

    path.write_bytes(original)
    path.write_text("\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n")
    with pytest.raises(hdl.HumanDecisionLogError, match="sequence|chain broken"):
        hdl.verify_log(path, now=_at(4))

    path.write_bytes(original[:-1])
    with pytest.raises(hdl.HumanDecisionLogError, match="incomplete final"):
        hdl.verify_log(path, now=_at(4))


def test_tail_truncation_is_refused_against_a_recorded_watermark(
    tmp_path: Path,
) -> None:
    """Every proper JSONL prefix is valid as a chain but invalid as this log."""

    path = tmp_path / "human.jsonl"
    log = hdl.HumanDecisionLog.initialize(
        path, log_id="synthetic-tail-attack", created_at=T0
    )
    _on(log, second=1, event_id="on")
    _prompt(log, second=2, event_id="prompt-1")
    _decision(
        log,
        second=3,
        event_id="wait",
        decision_id="d-wait",
        action=hdl.WAIT,
        position_state="FLAT",
        prompt_event_id="prompt-1",
    )
    _prompt(log, second=4, event_id="prompt-2")
    _decision(
        log,
        second=5,
        event_id="open",
        decision_id="d-open",
        action=hdl.OPEN_CALL,
        position_state="FLAT",
        contract_osi=CALL,
        prompt_event_id="prompt-2",
    )
    _decision(
        log,
        second=6,
        event_id="hold",
        decision_id="d-hold",
        action=hdl.HOLD,
        position_state="OPEN",
        contract_osi=CALL,
    )
    _decision(
        log,
        second=7,
        event_id="exit",
        decision_id="d-exit",
        action=hdl.EXIT,
        position_state="OPEN",
        contract_osi=CALL,
    )
    log._append_for_test(
        kind=hdl.MONITORING_OFF,
        event_id="off",
        session="2026-08-24",
        occurred_at=_at(8),
        appended_at=_at(8),
    )

    original = path.read_bytes()
    lines = original.decode("utf-8").splitlines()
    assert len(lines) == 9  # header plus eight events
    assert log.verify(now=_at(9)).events == 8

    # Each candidate ends on a complete, canonical record.  The bare chain for
    # several of these prefixes is internally valid; the external high-water
    # mark is what must make all of them fail, including a header-only journal.
    for retained_lines in range(1, len(lines)):
        path.write_text(
            "\n".join(lines[:retained_lines]) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(
            hdl.HumanDecisionLogError,
            match="watermark|terminal|sequence|head|truncat",
        ):
            hdl.verify_log(path, now=_at(9))
        with pytest.raises(
            hdl.HumanDecisionLogError,
            match="watermark|terminal|sequence|head|truncat",
        ):
            hdl.read_training_decisions(path)

    path.write_bytes(original)
    assert log.verify(now=_at(9)).events == 8


def test_watermark_is_required_and_advances_on_every_successful_append(
    tmp_path: Path,
) -> None:
    path = tmp_path / "human.jsonl"
    log = hdl.HumanDecisionLog.initialize(
        path, log_id="synthetic-watermark", created_at=T0
    )
    watermark = hdl.watermark_path(path)
    assert watermark.is_file()
    assert stat.S_IMODE(watermark.stat().st_mode) == 0o600

    initial_watermark = watermark.read_bytes()
    initial = log.verify(now=T0)
    _on(log)
    after_on_watermark = watermark.read_bytes()
    after_on = log.verify(now=_at(1))
    assert after_on_watermark != initial_watermark
    assert after_on.head != initial.head
    assert after_on.events == initial.events + 1

    _prompt(log)
    after_prompt_watermark = watermark.read_bytes()
    after_prompt = log.verify(now=_at(2))
    assert after_prompt_watermark != after_on_watermark
    assert after_prompt.head != after_on.head
    assert after_prompt.events == after_on.events + 1

    watermark.unlink()
    with pytest.raises(
        hdl.HumanDecisionLogError, match="watermark.*missing|missing.*watermark"
    ):
        hdl.verify_log(path, now=_at(2))


def test_watermark_tamper_and_regression_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "human.jsonl"
    log = hdl.HumanDecisionLog.initialize(
        path, log_id="synthetic-watermark-tamper", created_at=T0
    )
    watermark = hdl.watermark_path(path)
    initial_watermark = watermark.read_bytes()

    _on(log)
    current_watermark = watermark.read_bytes()
    assert current_watermark != initial_watermark

    watermark.write_bytes(b'{}\n')
    with pytest.raises(hdl.HumanDecisionLogError, match="watermark"):
        hdl.verify_log(path, now=_at(1))

    # A previously authentic watermark is still a regression once the journal
    # has advanced; it must not silently degrade verification to a bare chain.
    watermark.write_bytes(initial_watermark)
    with pytest.raises(
        hdl.HumanDecisionLogError,
        match="watermark|terminal|sequence|head|regress",
    ):
        hdl.verify_log(path, now=_at(1))

    watermark.write_bytes(current_watermark)
    assert hdl.verify_log(path, now=_at(1)).events == 1


def test_writer_refuses_to_extend_or_launder_a_truncated_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "human.jsonl"
    log = hdl.HumanDecisionLog.initialize(
        path, log_id="synthetic-truncated-writer", created_at=T0
    )
    _on(log)
    _prompt(log)
    watermark = hdl.watermark_path(path)
    recorded_watermark = watermark.read_bytes()

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    truncated_bytes = path.read_bytes()
    with pytest.raises(
        hdl.HumanDecisionLogError,
        match="watermark|terminal|sequence|head|truncat",
    ):
        log._append_for_test(
            kind=hdl.PROMPT,
            event_id="laundered-prompt",
            session="2026-08-24",
            occurred_at=_at(3),
            appended_at=_at(3),
            market_state_sha256=STATE_HASH,
            universe_sha256=UNIVERSE_HASH,
        )
    assert path.read_bytes() == truncated_bytes
    assert watermark.read_bytes() == recorded_watermark


def test_expected_terminal_assertions_are_additional_external_anchors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "human.jsonl"
    log = hdl.HumanDecisionLog.initialize(
        path, log_id="synthetic-expected-terminal", created_at=T0
    )
    _on(log)
    verified = log.verify(now=_at(1))

    assert hdl.verify_log(
        path,
        now=_at(1),
        expected_head=verified.head,
        expected_min_sequence=verified.events,
    ).head == verified.head

    with pytest.raises(hdl.HumanDecisionLogError, match="expected.*head|head.*expected"):
        hdl.verify_log(
            path,
            now=_at(1),
            expected_head=hdl.GENESIS_HASH,
        )
    with pytest.raises(
        hdl.HumanDecisionLogError,
        match="expected.*sequence|sequence.*expected|minimum",
    ):
        hdl.verify_log(
            path,
            now=_at(1),
            expected_min_sequence=verified.events + 1,
        )


def test_expected_terminal_anchor_rejects_coordinated_prefix_rollback(
    tmp_path: Path,
) -> None:
    """An out-of-band expected head catches rollback of both local files."""

    path = tmp_path / "human.jsonl"
    log = hdl.HumanDecisionLog.initialize(
        path, log_id="synthetic-coordinated-rollback", created_at=T0
    )
    _on(log)
    watermark = hdl.watermark_path(path)
    earlier_log = path.read_bytes()
    earlier_watermark = watermark.read_bytes()

    _prompt(log)
    terminal = log.verify(now=_at(2))
    assert terminal.events == 2

    # Simulate rollback of both files to an earlier mutually-consistent pair.
    # No self-authenticating local sidecar can distinguish that pair by itself;
    # the caller-supplied terminal anchor must refuse it.
    path.write_bytes(earlier_log)
    watermark.write_bytes(earlier_watermark)
    with pytest.raises(
        hdl.HumanDecisionLogError,
        match="expected.*head|head.*expected|expected.*sequence|sequence.*expected|minimum",
    ):
        hdl.verify_log(
            path,
            now=_at(2),
            expected_head=terminal.head,
            expected_min_sequence=terminal.events,
        )


def test_outcome_fill_return_and_pnl_fields_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "human.jsonl"
    log = hdl.HumanDecisionLog.initialize(path, log_id="synthetic", created_at=T0)
    _on(log)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["pnl"] = 315.0
    records[-1]["record_hash"] = hdl._record_hash(records[-1])
    path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) for row in records
        )
        + "\n"
    )
    with pytest.raises(hdl.HumanDecisionLogError, match="schema mismatch"):
        hdl.verify_log(path, now=_at(2))


def test_jsonl_and_symlink_boundaries_are_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(hdl.HumanDecisionLogError, match="local .jsonl"):
        hdl.initialize_log(tmp_path / "human.json", log_id="x", created_at=T0)
    target = tmp_path / "target.jsonl"
    hdl.initialize_log(target, log_id="target", created_at=T0)
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(hdl.HumanDecisionLogError, match="symbolic link"):
        hdl.verify_log(link, now=T0)


def test_synthetic_fixture_round_trips_and_verifies_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cli.jsonl"
    assert cli._main_for_test(
        ["initialize", "--log", str(path), "--log-id", "synthetic-cli"], now=T0
    ) == 0
    capsys.readouterr()
    assert cli._main_for_test(
        [
            "append",
            "--log",
            str(path),
            "--kind",
            hdl.MONITORING_ON,
            "--event-id",
            "on",
            "--session",
            "2026-08-24",
            "--occurred-at",
            hdl.canonical_utc(_at(1)),
        ],
        now=_at(1),
    ) == 0
    capsys.readouterr()
    assert cli._main_for_test(["status", "--log", str(path)], now=_at(2)) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["observation_status"] == "OBSERVED"
    assert cli._main_for_test(["verify", "--log", str(path)], now=_at(2)) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"
    assert cli._main_for_test(
        ["initialize", "--log", str(path), "--log-id", "overwrite"], now=_at(2)
    ) == 2
    assert "will not be overwritten" in capsys.readouterr().err


def test_logger_has_no_network_broker_or_market_data_dependency() -> None:
    roots = {
        "fcntl",
        "hashlib",
        "json",
        "os",
        "re",
        "time",
        "uuid",
        "dataclasses",
        "datetime",
        "pathlib",
        "typing",
        "zoneinfo",
    }
    module_path = Path(hdl.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= roots | {"__future__"}
    assert set(hdl.event_source_imports()) == roots
