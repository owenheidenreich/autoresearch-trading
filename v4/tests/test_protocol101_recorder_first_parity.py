from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from v4.scripts.run_protocol101_fair_contract_ibkr_capture_replay import (
    _annotate_affordability,
    _json_feature_vector,
    _trace_candidate_filter_payload,
    _trace_candidate_payload,
)
from v4.live.ibkr_market_capture import (
    CapturePaths,
    CaptureWriter,
    apply_option_capture_event,
    iter_capture_rows,
    option_snapshot_delta,
)
from v4.live.protocol101_capture_replay import (
    ReplayArtifacts,
    _index_value,
    build_replay_inputs,
    deterministic_trace_hash,
    replay_lifecycle,
    replay_capture,
)
from v4.live.protocol101_live_entry import LiveIndexState
from v4.ops.ibkr.protocol101_recorder_control import (
    checkpoint_overlay_path,
    checkpoint_progress_status,
    expected_minutes,
    scan,
)
from v4.ops.ibkr.protocol101_gateway_ready import handshake
from v4.ops.ibkr.run_protocol101_ibkr_recorder import completed_minute_label, connect


REPO_ROOT = Path(__file__).resolve().parents[2]
SURFACE = REPO_ROOT / "v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/model_artifacts/train_through_q4_2025_test_q1_2026/seed_11/manifest.json"
ENTRY = REPO_ROOT / "v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/model_artifacts/fold3_train_q1_q2_q3_validate_q4_test_q1_2026/seed_1/manifest.json"
SUMMARY = REPO_ROOT / "v4/audit/autoresearch/v4_aplus_hypothesis_101_event_history_policy/summary.json"
LIFECYCLE = REPO_ROOT / "v4/audit/autoresearch/v4_aplus_hypothesis_081_q4start_residual_sequence_deterministic_artifacts/model_artifacts/train_q1_2025_q2_2025_q3_2025_q4_2025_test_q1_2026/seed_1/manifest.json"


def test_capture_writer_restart_appends_monotonic_sequences_and_locks(tmp_path: Path) -> None:
    paths = CapturePaths.for_capture(tmp_path, "2026-06-29", "unit")
    first = CaptureWriter(paths, session="2026-06-29", capture_id="unit")
    first.append("connection", {"status": "connected"}, flush_to_disk=True)
    with pytest.raises(RuntimeError, match="already active"):
        CaptureWriter(paths, session="2026-06-29", capture_id="unit")
    first.close()

    with CaptureWriter(paths, session="2026-06-29", capture_id="unit") as second:
        second.new_connection_epoch()
        second.append("heartbeat", {"connected": True}, flush_to_disk=True)

    rows = [row for _, row in iter_capture_rows(paths.events) if row]
    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[0]["producer_instance_id"] != rows[1]["producer_instance_id"]
    assert rows[1]["connection_epoch"] == 1


def test_fair_contract_trace_vector_preserves_missing_values() -> None:
    vector = _json_feature_vector([1.5, float("nan"), float("inf"), float("-inf"), 0.0])

    assert vector == [1.5, None, None, None, 0.0]


def test_fair_contract_trace_payload_carries_filter_and_affordability_diagnostics() -> None:
    candidates = [
        {
            "contract_id": "SPXW-20260102-06000.000-C",
            "right": "C",
            "offset": 0.0,
            "strike_idx": 10,
            "right_idx": 0,
            "entry_bid": 1.0,
            "entry_ask": 1.2,
            "entry_mid": 1.1,
            "entry_spread": 0.2,
            "entry_spread_frac": 0.1818,
            "quote_age_ms": 500.0,
            "candidate_filter": {"passed": True, "reasons": []},
            "filter_reasons": [],
            "spx_for_ladder": 6000.1,
            "atm_strike": 6000,
            "feature_hash": "abc",
        }
    ]
    _annotate_affordability(candidates, cash=100.0)

    payload = _trace_candidate_payload(candidates)
    filter_payload = _trace_candidate_filter_payload(
        {
            "candidate_filter_trace": [
                {
                    "contract_id": "SPXW-20260102-06005.000-C",
                    "filter_reasons": ["stale_quote"],
                    "raw_vendor_fields": {"large": "not-exported"},
                }
            ]
        }
    )

    assert payload[0]["affordable_at_decision"] is False
    assert payload[0]["affordability_result"]["required_cash"] == 120.0
    assert payload[0]["candidate_filter"]["passed"] is True
    assert payload[0]["spx_for_ladder"] == 6000.1
    assert filter_payload == [
        {
            "contract_id": "SPXW-20260102-06005.000-C",
            "filter_reasons": ["stale_quote"],
        }
    ]


def test_option_delta_round_trip_preserves_changed_and_cleared_fields() -> None:
    first = {"contract_id": "SPXW-1", "bid": 1.0, "ask": 1.2, "ask_size": 10.0}
    second = {"contract_id": "SPXW-1", "bid": 1.1, "ask": 1.2, "ask_size": None}
    delta = option_snapshot_delta(first, second)
    assert delta == {"contract_id": "SPXW-1", "changes": {"bid": 1.1, "ask_size": None}}
    state: dict[str, dict[str, object]] = {}
    apply_option_capture_event(state, {"event_type": "option_update", "payload": first})
    apply_option_capture_event(state, {"event_type": "option_delta", "payload": delta})
    assert state["SPXW-1"] == second


def test_completed_opening_minute_maps_to_0931_decision() -> None:
    assert completed_minute_label(datetime(2026, 6, 29, 13, 31, 0, tzinfo=timezone.utc)) == "2026-06-29T09:30:00-04:00"


def test_completed_closing_minute_is_emitted_during_1600_et() -> None:
    assert completed_minute_label(datetime(2026, 6, 29, 20, 0, 30, tzinfo=timezone.utc)) == "2026-06-29T15:59:00-04:00"
    assert completed_minute_label(datetime(2026, 6, 29, 20, 1, 0, tzinfo=timezone.utc)) is None


def test_expected_regular_session_has_390_minutes() -> None:
    minutes = expected_minutes("2026-06-29")
    assert len(minutes) == 390
    assert minutes[0].endswith("09:30:00-04:00")
    assert minutes[-1].endswith("15:59:00-04:00")


def test_recorder_connect_is_market_data_only() -> None:
    calls: list[tuple[str, int, int, int]] = []

    class Client:
        async def connectAsync(self, host: str, port: int, client_id: int, timeout: int):
            calls.append((host, port, client_id, timeout))

    class Wrapper:
        clientId = None

    class FakeIB:
        def __init__(self) -> None:
            self.client = Client()
            self.wrapper = Wrapper()

        def run(self, awaitable):
            import asyncio
            return asyncio.run(awaitable)

        def isConnected(self) -> bool:
            return True

    args = type("Args", (), {"ports": "4002", "host": "127.0.0.1", "client_id": 159})()
    ib, port, attempts = connect(FakeIB, args)

    assert ib is not None
    assert port == 4002
    assert calls == [("127.0.0.1", 4002, 159, 8)]
    assert ib.wrapper.clientId == 159
    assert attempts == [{"port": 4002, "status": "connected", "connection_mode": "market_data_only"}]


def test_gateway_readiness_handshake_is_market_data_only() -> None:
    calls: list[tuple[str, int, int, int]] = []

    class Client:
        async def connectAsync(self, host: str, port: int, client_id: int, timeout: int):
            calls.append((host, port, client_id, timeout))

        def getAccounts(self):
            return ["DU123"]

    class Wrapper:
        clientId = None

    class FakeIB:
        def __init__(self) -> None:
            self.client = Client()
            self.wrapper = Wrapper()

        def run(self, awaitable):
            import asyncio
            return asyncio.run(awaitable)

        def isConnected(self) -> bool:
            return True

        def disconnect(self) -> None:
            pass

    result = handshake(FakeIB, host="127.0.0.1", port=4002, client_id=157)

    assert calls == [("127.0.0.1", 4002, 157, 8)]
    assert result == {"connected": True, "account_count": 1, "connection_mode": "market_data_only"}


def test_checkpoint_progress_status_fails_stale_opening_checkpoint() -> None:
    ok, details = checkpoint_progress_status(
        "2026-07-15",
        None,
        max_lag_seconds=150,
        now=datetime(2026, 7, 15, 13, 35, tzinfo=timezone.utc),
    )

    assert ok is False
    assert details["required"] is True
    assert details["last_checkpoint_decision_utc"] is None


def test_checkpoint_progress_status_allows_recent_checkpoint() -> None:
    ok, details = checkpoint_progress_status(
        "2026-07-15",
        "2026-07-15T09:33:00-04:00",
        max_lag_seconds=150,
        now=datetime(2026, 7, 15, 13, 34, 20, tzinfo=timezone.utc),
    )

    assert ok is True
    assert details["required"] is True
    assert details["checkpoint_lag_seconds"] == pytest.approx(20.0)


def test_live_index_state_retains_opening_context_for_full_tick_session() -> None:
    state = LiveIndexState()
    start = pd.Timestamp("2026-06-29 09:30", tz="America/New_York").tz_convert("UTC")
    for index in range(50_100):
        state.add(
            timestamp=start + pd.Timedelta(milliseconds=index * 450),
            spx=6000.0 + (index % 10) * 0.1,
            vix=16.0,
        )

    summary = state.session_context_summary(start + pd.Timedelta(hours=6))
    assert summary["opening_context_ready"] is True
    assert summary["missing_opening_minutes"] == 0


def test_captured_replay_uses_last_option_event_at_or_before_boundary(tmp_path: Path) -> None:
    session = "2026-06-29"
    boundary = pd.Timestamp(f"{session} 09:31", tz="America/New_York").tz_convert("UTC")
    before = (boundary - pd.Timedelta(milliseconds=100)).isoformat()
    after = (boundary + pd.Timedelta(milliseconds=100)).isoformat()
    checkpoint_time = (boundary + pd.Timedelta(milliseconds=200)).isoformat()
    contract = {
        "contract_id": "SPXW-20260629-06000.000-C",
        "expiry": "20260629",
        "strike": 6000.0,
        "right": "C",
        "bid": 1.0,
        "ask": 1.2,
        "source_timestamp_utc": before,
        "last_received_timestamp_utc": before,
    }
    rows = [
        _event(1, session, "index_update", before, {"symbol": "VIX", "price": 16.0}),
        _event(2, session, "index_update", before, {"symbol": "SPX", "price": 6000.0}),
        _event(3, session, "option_update", before, contract),
        _event(4, session, "option_delta", after, {"contract_id": contract["contract_id"], "changes": {"bid": 2.0}}),
        _event(
            5,
            session,
            "ladder_checkpoint",
            checkpoint_time,
            {
                "completed_minute_et": (boundary.tz_convert("America/New_York") - pd.Timedelta(minutes=1)).isoformat(),
                "decision_time_et": boundary.tz_convert("America/New_York").isoformat(),
                "contracts": [{**contract, "bid": 2.0}],
                "spx": {"symbol": "SPX", "price": 6000.0},
                "vix": {"symbol": "VIX", "price": 16.0},
            },
        ),
    ]
    events = tmp_path / "market_events.jsonl"
    events.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    checkpoints, index_state = build_replay_inputs(events, session=session)

    assert checkpoints[0]["replay_checkpoint_source"] == "event_boundary_at_or_before_decision"
    assert checkpoints[0]["contracts"][0]["bid"] == 1.0
    assert len(index_state.rows) == 4


def test_captured_replay_does_not_treat_prior_close_fallback_as_opening_print() -> None:
    assert _index_value({"price": 7499.36, "last": None, "close": 7499.36}) is None
    assert _index_value({"price": 7479.24, "last": 7479.24, "close": 7499.36}) == 7479.24
    assert _index_value({"price": 7479.24}) == 7479.24


def test_live_index_state_uses_captured_prior_close_for_opening_gap() -> None:
    opening = pd.Timestamp("2026-07-01 09:30", tz="America/New_York").tz_convert("UTC")
    state = LiveIndexState(prior_session_close=7499.36)
    state.add(timestamp=opening, spx=7471.52, vix=17.0)

    structure = state.structure_features(opening + pd.Timedelta(minutes=1))

    assert structure[19] == pytest.approx((7471.52 - 7499.36) / 7499.36)


def test_scan_detects_parse_error_without_modifying_raw_capture(tmp_path: Path) -> None:
    paths = CapturePaths.for_capture(tmp_path, "2026-06-29", "unit")
    paths.root.mkdir(parents=True)
    paths.events.write_text('{"schema_version":"IBKRMarketCaptureV1","sequence":1,"session":"2026-06-29","event_type":"heartbeat","producer_instance_id":"a","connection_epoch":1,"payload":{}}\n{bad\n')
    before = paths.events.read_bytes()
    result = scan(paths, "2026-06-29")
    assert result["parse_errors"] == 1
    assert paths.events.read_bytes() == before


def test_scan_uses_provenanced_overlay_to_replace_bad_restart_checkpoint(tmp_path: Path) -> None:
    session = "2026-06-29"
    paths = CapturePaths.for_capture(tmp_path, session, "unit")
    paths.root.mkdir(parents=True)
    minute = "2026-06-29T09:30:00-04:00"
    bad = _event(
        1,
        session,
        "ladder_checkpoint",
        "2026-06-29T13:31:01+00:00",
        {"completed_minute_et": minute, "contracts": []},
    )
    paths.events.write_text(json.dumps(bad, sort_keys=True) + "\n")
    contracts = [
        {
            "contract_id": f"SPXW-{index}",
            "strike": 6000.0 + index,
            "right": "C" if index % 2 == 0 else "P",
            "market_data_type_name": "live",
            "bid": 1.0,
            "ask": 1.1,
            "last_received_timestamp_utc": "2026-06-29T13:30:59+00:00",
        }
        for index in range(42)
    ]
    replacement = {
        "schema_version": "Protocol101DerivedCheckpointV1",
        "session": session,
        "event_type": "ladder_checkpoint",
        "payload": {
            "completed_minute_et": minute,
            "replacement_for_raw_checkpoint": True,
            "contracts": contracts,
        },
    }
    checkpoint_overlay_path(paths).write_text(json.dumps(replacement, sort_keys=True) + "\n")

    result = scan(paths, session)

    assert result["checkpoint_count"] == 1
    assert result["incomplete_checkpoint_rows"] == 0
    assert result["low_tradable_checkpoint_rows"] == 0
    assert result["recovered_checkpoint_minutes"] == [minute]


def test_retry_wrapper_survives_two_failures_then_succeeds(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    command = tmp_path / "flaky.sh"
    command.write_text(
        "#!/usr/bin/env bash\n"
        "n=0\n"
        "[[ -f \"$1\" ]] && n=$(cat \"$1\")\n"
        "n=$((n+1))\n"
        "printf '%s' \"$n\" >\"$1\"\n"
        "[[ $n -ge 3 ]]\n"
    )
    command.chmod(0o700)
    env = {"RETRY_ATTEMPTS": "3", "RETRY_INITIAL_SLEEP_SECONDS": "0", "RETRY_MAX_SLEEP_SECONDS": "0", "RETRY_LOG_PATH": str(tmp_path / "retry.log")}
    result = subprocess.run(
        [str(REPO_ROOT / "v4/ops/ibkr/retry_command.sh"), str(command), str(counter)],
        cwd=REPO_ROOT,
        env={**__import__("os").environ, **env},
        check=False,
    )
    assert result.returncode == 0
    assert counter.read_text() == "3"


def test_packet_assigns_preflight_and_recorder_distinct_client_ids() -> None:
    script = (REPO_ROOT / "v4/ops/ibkr/run_protocol101_recorder_packet.sh").read_text()
    assert '--session "$SESSION" --client-id 156' in script
    assert '--capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT"' in script
    assert '--client-id 159 --strikes-around-atm 10' in script
    assert 'PROTOCOL101_PACKET_DEVELOPMENT_SESSION' in script
    assert 'PROTOCOL101_PACKET_ALLOWED_SESSIONS' in script
    assert 'session_allowed' in script
    assert 'session == development_session' in script
    assert 'recorder evidence unhealthy; restarting' in script
    assert 'gateway API port closed; restarting' in script
    assert 'protocol101_gateway_ready' in script
    assert 'pkill -TERM -f "v4.ops.ibkr.run_protocol101_ibkr_recorder.*--session $SESSION"' in script
    assert 'launchctl kill TERM "gui/$UID/$LABEL_PREFIX.recorder"' in script
    assert 'WATCHDOG_RESTART_COOLDOWN_SECONDS' in script
    assert script.index('gateway API port closed; restarting') < script.index('launchctl kickstart -k "gui/$UID/$LABEL_PREFIX.recorder"')
    assert '10#$raw' in script
    assert '--require-checkpoint-progress --max-checkpoint-lag-seconds 150' in script
    assert 'recover-checkpoints' in script
    assert script.index("recover-checkpoints") < script.index("audit \\")


def test_deploy_recorder_packet_accepts_date_scoped_arguments() -> None:
    source = (REPO_ROOT / "v4/scripts/deploy_protocol101_recorder_parity.py").read_text()
    assert "--packet-dates" in source
    assert "--development-session" in source
    assert "PROTOCOL101_PACKET_ALLOWED_SESSIONS" in source
    assert "DEFAULT_PACKET_DATES" in source
    assert '"gateway": ("gateway", 4, 45' in source
    assert '"preflight": ("preflight", 5, 25' in source
    assert '"recorder": ("recorder", 5, 40' in source
    assert '"watchdog": ("watchdog", 6, 5' in source
    assert '((5, 50), (6, 0), (6, 10), (6, 20), (6, 28))' in source


def test_capture_to_protocol101_trace_is_exactly_repeatable(tmp_path: Path) -> None:
    session = "2026-06-29"
    events = tmp_path / "market_events.jsonl"
    _write_synthetic_capture(events, session=session, minute_count=32)
    artifacts = ReplayArtifacts(SURFACE, ENTRY, SUMMARY, LIFECYCLE)

    canonical_a, traces_a = replay_capture(events, artifacts, session=session, run_id="unit")
    canonical_b, traces_b = replay_capture(events, artifacts, session=session, run_id="unit")

    assert canonical_a
    assert traces_a
    assert canonical_a[0]["schema_version"] == "CanonicalMarketMinuteV2"
    assert canonical_a[0]["decision_time"].endswith("13:31:00+00:00")
    assert canonical_a[0]["source_quote_time_utc"]
    assert canonical_a[0]["source_context_time_utc"]
    assert canonical_a[0]["tradable_token_count"] > 0
    assert traces_a[0]["candidate_count"] > 0
    assert traces_a[0]["feature_contract_version"] == "protocol101-live-v1"
    assert deterministic_trace_hash(traces_a) == deterministic_trace_hash(traces_b)
    assert [row["feature_hash"] for row in traces_a] == [row["feature_hash"] for row in traces_b]


def test_lifecycle_replay_uses_full_sequence_and_forces_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    class Artifact:
        feature_columns = ["current_pnl", "minutes_to_forced_flat"]
        selected_override_threshold = float("inf")

    sequence_lengths: list[int] = []

    def fake_predict(artifact: object, frame: pd.DataFrame):
        del artifact
        sequence_lengths.append(len(frame))
        zeros = __import__("numpy").zeros(len(frame), dtype=float)
        return zeros, zeros, zeros

    def fake_feature_row(**kwargs: object) -> pd.Series:
        return pd.Series(
            {
                "current_pnl": 0.0,
                "minutes_to_forced_flat": 999.0,
                "is_baseline_exit_step": False,
                "baseline_exit_reason": "",
            }
        )

    monkeypatch.setattr(
        "v4.live.protocol101_capture_replay.load_protocol066_artifact",
        lambda path: Artifact(),
    )
    monkeypatch.setattr(
        "v4.live.protocol101_capture_replay.predict_protocol066_sequence",
        fake_predict,
    )
    monkeypatch.setattr(
        "v4.live.protocol101_capture_replay._live_feature_row",
        fake_feature_row,
    )
    session = "2026-07-02"
    contract_id = "SPXW-20260702-06000.000-C"

    def canonical(minute: str) -> dict:
        decision = pd.Timestamp(f"{session} {minute}", tz="America/New_York")
        checkpoint = {
            "contracts": [
                {
                    "contract_id": contract_id,
                    "expiry": "20260702",
                    "strike": 6000.0,
                    "right": "C",
                    "bid": 1.0,
                    "ask": 1.1,
                    "mid": 1.05,
                    "bid_size": 10.0,
                    "ask_size": 10.0,
                }
            ]
        }
        return {
            "decision_time": decision.tz_convert("UTC").isoformat(),
            "spx": 6000.0,
            "atm_strike": 6000,
            "raw_vendor_checkpoint_json": json.dumps(checkpoint),
        }

    rows = [canonical("15:53"), canonical("15:54"), canonical("15:55")]
    traces = [
        {
            "decision_ts": pd.Timestamp(
                f"{session} 15:53",
                tz="America/New_York",
            ).tz_convert("UTC").isoformat(),
            "selected_action": "enter",
            "selected_contract_id": contract_id,
            "selected_score": 30.0,
        }
    ]

    lifecycle = replay_lifecycle(rows, traces, Path("unused"))

    assert sequence_lengths == [1, 2]
    assert [row["sequence_length"] for row in lifecycle] == [1, 2]
    assert lifecycle[-1]["action"] == "forced_flat"
    assert lifecycle[-1]["reason"] == "mandatory_time_flat"


def test_lifecycle_replay_uses_trade_deadline_not_only_session_forced_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    class Artifact:
        feature_columns = ["current_pnl", "minutes_to_deadline", "minutes_to_forced_flat"]
        selected_override_threshold = float("inf")

    observed_deadlines: list[float] = []

    def fake_predict(artifact: object, frame: pd.DataFrame):
        del artifact
        observed_deadlines.append(float(frame.iloc[-1]["minutes_to_deadline"]))
        zeros = __import__("numpy").zeros(len(frame), dtype=float)
        return zeros, zeros, zeros

    def fake_feature_row(**kwargs: object) -> pd.Series:
        return pd.Series(
            {
                "current_pnl": 0.0,
                "minutes_to_deadline": 999.0,
                "minutes_to_forced_flat": 999.0,
                "is_baseline_exit_step": False,
                "baseline_exit_reason": "",
            }
        )

    monkeypatch.setattr(
        "v4.live.protocol101_capture_replay.load_protocol066_artifact",
        lambda path: Artifact(),
    )
    monkeypatch.setattr(
        "v4.live.protocol101_capture_replay.predict_protocol066_sequence",
        fake_predict,
    )
    monkeypatch.setattr(
        "v4.live.protocol101_capture_replay._live_feature_row",
        fake_feature_row,
    )
    session = "2026-07-02"
    contract_id = "SPXW-20260702-06000.000-C"

    def canonical(minute: str) -> dict:
        decision = pd.Timestamp(f"{session} {minute}", tz="America/New_York")
        checkpoint = {
            "contracts": [
                {
                    "contract_id": contract_id,
                    "expiry": "20260702",
                    "strike": 6000.0,
                    "right": "C",
                    "bid": 1.0,
                    "ask": 1.1,
                    "mid": 1.05,
                    "bid_size": 10.0,
                    "ask_size": 10.0,
                }
            ]
        }
        return {
            "decision_time": decision.tz_convert("UTC").isoformat(),
            "spx": 6000.0,
            "atm_strike": 6000,
            "raw_vendor_checkpoint_json": json.dumps(checkpoint),
        }

    rows = [canonical(f"09:{minute:02d}") for minute in range(31, 57)]
    traces = [
        {
            "decision_ts": pd.Timestamp(
                f"{session} 09:31",
                tz="America/New_York",
            ).tz_convert("UTC").isoformat(),
            "selected_action": "enter",
            "selected_contract_id": contract_id,
            "selected_score": 30.0,
        }
    ]

    lifecycle = replay_lifecycle(rows, traces, Path("unused"))

    assert observed_deadlines[0] == pytest.approx(24.0)
    assert observed_deadlines[-1] == pytest.approx(0.0)
    assert lifecycle[-1]["action"] == "forced_flat"
    assert lifecycle[-1]["reason"] == "mandatory_time_flat"


def test_lifecycle_replay_can_reenter_on_same_minute_as_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    class Artifact:
        feature_columns = ["current_pnl", "minutes_to_deadline", "minutes_to_forced_flat"]
        selected_override_threshold = float("inf")

    def fake_predict(artifact: object, frame: pd.DataFrame):
        del artifact
        zeros = __import__("numpy").zeros(len(frame), dtype=float)
        return zeros, zeros, zeros

    def fake_feature_row(**kwargs: object) -> pd.Series:
        return pd.Series(
            {
                "current_pnl": 0.0,
                "minutes_to_deadline": 999.0,
                "minutes_to_forced_flat": 999.0,
                "is_baseline_exit_step": False,
                "baseline_exit_reason": "",
            }
        )

    monkeypatch.setattr(
        "v4.live.protocol101_capture_replay.load_protocol066_artifact",
        lambda path: Artifact(),
    )
    monkeypatch.setattr(
        "v4.live.protocol101_capture_replay.predict_protocol066_sequence",
        fake_predict,
    )
    monkeypatch.setattr(
        "v4.live.protocol101_capture_replay._live_feature_row",
        fake_feature_row,
    )
    session = "2026-07-02"
    first_contract = "SPXW-20260702-06000.000-C"
    second_contract = "SPXW-20260702-06005.000-C"

    def canonical(timestamp: pd.Timestamp) -> dict:
        checkpoint = {
            "contracts": [
                {
                    "contract_id": first_contract,
                    "expiry": "20260702",
                    "strike": 6000.0,
                    "right": "C",
                    "bid": 1.0,
                    "ask": 1.1,
                    "mid": 1.05,
                    "bid_size": 10.0,
                    "ask_size": 10.0,
                },
                {
                    "contract_id": second_contract,
                    "expiry": "20260702",
                    "strike": 6005.0,
                    "right": "C",
                    "bid": 1.2,
                    "ask": 1.3,
                    "mid": 1.25,
                    "bid_size": 10.0,
                    "ask_size": 10.0,
                },
            ]
        }
        return {
            "decision_time": timestamp.tz_convert("UTC").isoformat(),
            "spx": 6000.0,
            "atm_strike": 6000,
            "raw_vendor_checkpoint_json": json.dumps(checkpoint),
        }

    start = pd.Timestamp(f"{session} 09:31", tz="America/New_York")
    rows = [canonical(start + pd.Timedelta(minutes=minute)) for minute in range(51)]
    traces = [
        {
            "decision_ts": start.tz_convert("UTC").isoformat(),
            "selected_action": "enter",
            "selected_contract_id": first_contract,
            "selected_score": 30.0,
        },
        {
            "decision_ts": (start + pd.Timedelta(minutes=25)).tz_convert("UTC").isoformat(),
            "selected_action": "enter",
            "selected_contract_id": second_contract,
            "selected_score": 30.0,
        },
    ]

    lifecycle = replay_lifecycle(rows, traces, Path("unused"))
    terminals = [row for row in lifecycle if row["action"] == "forced_flat"]

    assert [row["contract_id"] for row in terminals] == [first_contract, second_contract]
    assert [row["sequence_length"] for row in terminals] == [25, 25]


def _write_synthetic_capture(path: Path, *, session: str, minute_count: int) -> None:
    start = pd.Timestamp(f"{session} 09:30", tz="America/New_York")
    rows = []
    sequence = 0
    for minute_index in range(minute_count):
        completed = start + pd.Timedelta(minutes=minute_index)
        decision = completed + pd.Timedelta(minutes=1)
        received = (decision.tz_convert("UTC") - pd.Timedelta(milliseconds=100)).isoformat()
        spx = 6000.0 + minute_index * 0.25
        vix = 16.0 + minute_index * 0.01
        for symbol, value in (("SPX", spx), ("VIX", vix)):
            sequence += 1
            rows.append(_event(sequence, session, "index_update", received, {"symbol": symbol, "price": value, "market_data_type": 1, "market_data_type_name": "live"}))
        contracts = []
        atm = int(round(spx / 5.0) * 5)
        for offset in range(-10, 11):
            strike = float(atm + offset * 5)
            for right in ("C", "P"):
                intrinsic = max(spx - strike, 0.0) if right == "C" else max(strike - spx, 0.0)
                mid = min(34.0, max(1.0, 8.0 + intrinsic * 0.15))
                contracts.append({
                    "contract_id": f"SPXW-{session.replace('-', '')}-{strike:09.3f}-{right}",
                    "symbol": "SPX",
                    "expiry": session.replace("-", ""),
                    "strike": strike,
                    "right": right,
                    "trading_class": "SPXW",
                    "settlement": "PM",
                    "exchange": "SMART",
                    "currency": "USD",
                    "bid": mid - 0.1,
                    "ask": mid + 0.1,
                    "mid": mid,
                    "bid_size": 10.0,
                    "ask_size": 12.0,
                    "source_timestamp_utc": received,
                    "last_received_timestamp_utc": received,
                    "market_data_type": 1,
                    "market_data_type_name": "live",
                    "model_greeks": {
                        "implied_vol": 0.20,
                        "delta": 0.50 if right == "C" else -0.50,
                        "gamma": 0.01,
                        "theta": -0.25,
                        "vega": 0.08,
                        "underlying_price": spx,
                    },
                })
        sequence += 1
        rows.append(_event(sequence, session, "ladder_checkpoint", received, {
            "completed_minute_et": completed.isoformat(),
            "decision_time_et": decision.isoformat(),
            "spx": {"symbol": "SPX", "price": spx, "market_data_type": 1, "market_data_type_name": "live"},
            "vix": {"symbol": "VIX", "price": vix, "market_data_type": 1, "market_data_type_name": "live"},
            "atm_strike": atm,
            "contracts": contracts,
            "contract_count": len(contracts),
        }))
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _event(sequence: int, session: str, event_type: str, received: str, payload: dict) -> dict:
    return {
        "schema_version": "IBKRMarketCaptureV1",
        "sequence": sequence,
        "producer_instance_id": "synthetic",
        "connection_epoch": 1,
        "session": session,
        "capture_id": "synthetic",
        "event_type": event_type,
        "event_timestamp_utc": received,
        "received_timestamp_utc": received,
        "payload": payload,
    }
