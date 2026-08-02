"""Generate the bounded offline Path-D E2E transcript and latency evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from v4.path_d.contracts import CanonicalMarketEventV1, PositionV1
from v4.path_d.decision import ExitFixtureState, OfflineDecisionService
from v4.path_d.execution.latency import owned_input_manifest, run_latency_bounds, write_latency_report
from v4.path_d.execution.simulated import ExecutionScenario, SimulatedExecutor, SimulatedQuote, VirtualMonotonicClock
from v4.path_d.risk import DeterministicGovernor, FeedHealthV1, fake_broker_state


DEFAULT_OUT = Path("v4/audit/autoresearch/path_d_offline_foundation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    events = _canonical_fixture_events()
    held = events[-1].osi_symbol
    assert held is not None
    decision = OfflineDecisionService().replay(
        events,
        exit_state=ExitFixtureState(
            held_osi_symbol=held,
            position_snapshot_version="fake-state-1",
            entry_bid_micros=2_500_000,
            running_max_bid_micros=3_000_000,
            opened_at_utc="2026-06-30T13:30:00Z",
        ),
    )
    if decision.intent is None:
        raise RuntimeError("deterministic fixture failed to produce an exit intent")
    broker_state = fake_broker_state(
        captured_at_utc="2026-06-30T13:30:02Z",
        positions=(PositionV1(held, 1, 2_500_000, "2026-06-30T13:30:00Z"),),
    )
    feed = FeedHealthV1(
        option_received_timestamp_utc="2026-06-30T13:30:02Z",
        spx_received_timestamp_utc="2026-06-30T13:30:01Z",
    )
    authorization = DeterministicGovernor().evaluate(
        decision.intent,
        broker_state=broker_state,
        feed_health=feed,
        now_utc="2026-06-30T13:30:02Z",
    )
    executor = SimulatedExecutor(
        clock=VirtualMonotonicClock(datetime(2026, 6, 30, 13, 30, 2, tzinfo=timezone.utc)),
        quote_tape=(SimulatedQuote(0, 2_000_000, 2_100_000), SimulatedQuote(120, 1_950_000, 2_050_000)),
        scenario=ExecutionScenario(outcome="FULL", submit_latency_ms=100),
    )
    execution_events = executor.submit(decision.intent, authorization)
    executor.write_transcript(out_dir / "simulated_execution_transcript.jsonl")

    transcript = []
    transcript.extend({"record_type": "CANONICAL_MARKET_EVENT", "payload": event.to_dict()} for event in events)
    transcript.append({"record_type": "FEATURE_SNAPSHOT", "payload": decision.snapshot.to_dict()})
    transcript.append({"record_type": "EXECUTION_INTENT", "payload": decision.intent.to_dict()})
    transcript.append({"record_type": "GOVERNOR_DECISION", "payload": authorization.to_dict()})
    transcript.extend({"record_type": "EXECUTION_EVENT", "payload": event.to_dict()} for event in execution_events)
    (out_dir / "offline_e2e_transcript.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in transcript),
        encoding="utf-8",
    )

    latency = run_latency_bounds(repo_root)
    write_latency_report(latency, out_dir)
    input_manifest = owned_input_manifest(repo_root)
    (out_dir / "latency_input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "PathDOfflineFoundationRunV1",
        "mode": "OFFLINE_ONLY",
        "canonical_event_count": len(events),
        "execution_event_count": len(execution_events),
        "latency_bound_rows": len(latency),
        "latency_sessions": sorted({row.session_date for row in latency}),
        "latency_rungs_ms": sorted({row.latency_ms for row in latency}),
        "latency_input_manifest": "latency_input_manifest.json",
        "final_execution_state": execution_events[-1].state_to,
        "actual_fill_claim": False,
        "paper_runtime_touched": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def _canonical_fixture_events() -> list[CanonicalMarketEventV1]:
    common = {"session_date": "2026-06-30"}
    spx0 = CanonicalMarketEventV1.create(
        source="THETADATA_SPX", event_type="SPX_INDEX",
        source_timestamp_utc="2026-06-30T13:30:00Z", received_timestamp_utc="2026-06-30T13:30:00Z",
        index_symbol="SPX", index_price_micros=7_490_000_000, volume=10, **common,
    )
    spx1 = CanonicalMarketEventV1.create(
        source="THETADATA_SPX", event_type="SPX_INDEX",
        source_timestamp_utc="2026-06-30T13:30:01Z", received_timestamp_utc="2026-06-30T13:30:01Z",
        index_symbol="SPX", index_price_micros=7_491_000_000, volume=20, **common,
    )
    option = CanonicalMarketEventV1.create(
        source="DATABENTO_OPRA", event_type="OPTION_QUOTE",
        source_timestamp_utc="2026-06-30T13:30:01Z", received_timestamp_utc="2026-06-30T13:30:02Z",
        osi_symbol="SPXW  260630C07490000", bid_price_micros=2_000_000, ask_price_micros=2_100_000,
        bid_size=10, ask_size=8, **common,
    )
    return [spx0, spx1, option]


if __name__ == "__main__":
    main()
