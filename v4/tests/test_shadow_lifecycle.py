from __future__ import annotations

from v4.live.shadow_lifecycle import strict_serial_shadow_rows


def _row(
    trade_uid: str,
    *,
    entry: str,
    decision: str,
    action: str,
    ts: int,
    step: int,
) -> dict:
    return {
        "protocol_id": "protocol081",
        "trade_uid": trade_uid,
        "timestamp_ms": ts,
        "sequence_step_index": step,
        "entry_decision_time": entry,
        "decision_time": decision,
        "decision": {"action": action},
        "position_state": "holding",
        "intended_size": 1,
        "contract_id": "SPXW-20260302-06700.000-C",
        "nbbo": {"bid": 1.0, "ask": 1.1, "timestamp_ms": ts},
        "context": {"spx": 6700.0, "vix": 20.0, "timestamp_ms": ts},
        "features": {},
        "model": {"artifact": "test"},
        "order_intent": None,
    }


def test_strict_serial_shadow_rows_truncates_after_terminal_action() -> None:
    rows = [
        _row("a", entry="2026-03-02T15:00:00+00:00", decision="2026-03-02T15:01:00+00:00", action="hold", ts=1, step=0),
        _row("a", entry="2026-03-02T15:00:00+00:00", decision="2026-03-02T15:02:00+00:00", action="exit", ts=2, step=1),
        _row("a", entry="2026-03-02T15:00:00+00:00", decision="2026-03-02T15:03:00+00:00", action="hold", ts=3, step=2),
    ]

    result = strict_serial_shadow_rows(rows)

    assert result.summary["selected_trades"] == 1
    assert result.summary["post_terminal_rows_removed"] == 1
    assert [row["decision"]["action"] for row in result.rows] == ["hold", "exit"]


def test_strict_serial_shadow_rows_skips_overlapping_candidates() -> None:
    rows = [
        _row("a", entry="2026-03-02T15:00:00+00:00", decision="2026-03-02T15:01:00+00:00", action="hold", ts=1, step=0),
        _row("a", entry="2026-03-02T15:00:00+00:00", decision="2026-03-02T15:10:00+00:00", action="forced_flat", ts=10, step=1),
        _row("b", entry="2026-03-02T15:05:00+00:00", decision="2026-03-02T15:06:00+00:00", action="hold", ts=6, step=0),
        _row("b", entry="2026-03-02T15:05:00+00:00", decision="2026-03-02T15:08:00+00:00", action="exit", ts=8, step=1),
        _row("c", entry="2026-03-02T15:11:00+00:00", decision="2026-03-02T15:12:00+00:00", action="exit", ts=12, step=0),
    ]

    result = strict_serial_shadow_rows(rows)

    assert result.summary["selected_trade_uids"] == ["a", "c"]
    assert result.summary["skipped_overlap_trade_uids"] == ["b"]
    assert result.summary["skipped_overlap_trades"] == 1
