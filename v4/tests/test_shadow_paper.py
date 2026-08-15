from __future__ import annotations

import copy

from v4.sim.shadow_paper import ShadowPaperConfig, replay_shadow_paper


def _row(*, trade_uid: str = "t1", action: str = "hold", bid: float = 1.2, ask: float = 1.3, ts: int = 1770000000000) -> dict:
    return {
        "protocol_id": "protocol081",
        "trade_uid": trade_uid,
        "timestamp_ms": ts,
        "decision_time": "2026-03-02T15:01:00+00:00",
        "entry_decision_time": "2026-03-02T15:00:00+00:00",
        "position_state": "holding",
        "intended_size": 1,
        "contract_id": "SPXW-20260302-06700.000-C",
        "nbbo": {"bid": bid, "ask": ask, "timestamp_ms": ts},
        "features": {
            "bid": bid,
            "ask": ask,
            "bid_over_entry_ask": bid / 1.0,
            "current_pnl": (bid - 1.0) * 100.0,
            "entry_is_call": 1.0,
            "entry_is_put": 0.0,
        },
        "decision": {"action": action, "reason": "test"},
        "order_intent": None,
    }


def test_shadow_paper_reconstructs_entry_and_exit_bid_pnl() -> None:
    first = _row(action="hold", bid=1.2, ask=1.3)
    second = _row(action="exit", bid=1.8, ask=1.9, ts=1770000060000)
    second["decision_time"] = "2026-03-02T15:02:00+00:00"

    summary = replay_shadow_paper([first, second])

    assert summary["status"] == "pass"
    assert summary["closed_trades"] == 1
    ledger = summary["trade_ledgers"][0]
    assert ledger["entry_fill_price"] == 1.0
    assert ledger["exit_fill_price"] == 1.8
    assert round(ledger["realized_pnl"], 6) == 80.0


def test_shadow_paper_warns_when_stream_ends_open() -> None:
    summary = replay_shadow_paper([_row(action="hold")])

    assert summary["status"] == "warn"
    assert summary["open_trades"] == 1
    assert summary["trade_ledgers"][0]["status"] == "open_at_stream_end"


def test_shadow_paper_blocks_order_intent() -> None:
    row = _row(action="exit")
    row["order_intent"] = {"side": "BUY", "quantity": 1}

    summary = replay_shadow_paper([row])

    assert summary["status"] == "fail"
    assert summary["checks"][0]["name"] == "no_order_fields"
    assert summary["checks"][0]["status"] == "fail"


def test_shadow_paper_flags_rows_after_terminal_action() -> None:
    first = _row(action="exit", bid=1.8, ask=1.9)
    later = _row(action="hold", bid=1.7, ask=1.8, ts=1770000060000)
    later["decision_time"] = "2026-03-02T15:02:00+00:00"

    summary = replay_shadow_paper([first, later])

    assert summary["status"] == "warn"
    ledger = summary["trade_ledgers"][0]
    assert ledger["post_terminal_row_count"] == 1
    checks = {check["name"]: check for check in summary["checks"]}
    assert checks["terminal_action_final"]["status"] == "warn"


def test_shadow_paper_can_require_terminal_action_to_be_final() -> None:
    first = _row(action="exit", bid=1.8, ask=1.9)
    later = _row(action="hold", bid=1.7, ask=1.8, ts=1770000060000)
    later["decision_time"] = "2026-03-02T15:02:00+00:00"

    summary = replay_shadow_paper(
        [first, later],
        config=ShadowPaperConfig(require_terminal_final=True),
    )

    assert summary["status"] == "fail"
    checks = {check["name"]: check for check in summary["checks"]}
    assert checks["terminal_action_final"]["status"] == "fail"


def test_shadow_paper_can_enforce_global_one_position() -> None:
    a1 = _row(trade_uid="a", action="hold", bid=1.2, ts=1770000000000)
    a2 = _row(trade_uid="a", action="exit", bid=1.4, ts=1770000120000)
    a2["decision_time"] = "2026-03-02T15:03:00+00:00"
    b1 = copy.deepcopy(_row(trade_uid="b", action="hold", bid=2.2, ts=1770000060000))
    b1["contract_id"] = "SPXW-20260302-06705.000-P"
    b1["decision_time"] = "2026-03-02T15:02:00+00:00"
    b1["entry_decision_time"] = "2026-03-02T15:02:00+00:00"
    b2 = copy.deepcopy(b1)
    b2["decision"] = {"action": "exit", "reason": "test"}
    b2["timestamp_ms"] = 1770000180000
    b2["decision_time"] = "2026-03-02T15:04:00+00:00"

    summary = replay_shadow_paper(
        [a1, b1, a2, b2],
        config=ShadowPaperConfig(enforce_global_one_position=True),
    )

    assert summary["status"] == "fail"
    checks = {check["name"]: check for check in summary["checks"]}
    assert checks["global_one_position"]["status"] == "fail"
    assert checks["global_one_position"]["value"] == 2
