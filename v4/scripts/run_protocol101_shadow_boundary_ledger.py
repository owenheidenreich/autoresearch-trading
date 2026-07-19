"""Shadow boundary ledger: record, at every minute boundary during the live
session, exactly how much capture data had arrived (bytes + last complete
event). This is the real-time measurement for equivalence layer 1
(live-vs-recorded): since the decision pipeline is deterministic over bytes,
the model's true live per-minute decision stream is reconstructed later by
replaying the capture truncated to each boundary's offset — with zero loss
of fidelity and zero mid-week computation.

Runs once per minute via launchd; self-gated to weekdays 06:29-13:06 PT.
Read-only on the capture; failures log and exit 0.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture"


def main() -> int:
    now = datetime.now(PT)
    manual = "--manual" in sys.argv
    if not manual:
        if now.weekday() >= 5:
            return 0
        if not ((6, 29) <= (now.hour, now.minute) <= (13, 6)):
            return 0
    session = now.strftime("%Y-%m-%d")
    capture_dir = CAPTURE_ROOT / session / f"protocol101-recorder-{session}"
    events = capture_dir / "market_events.jsonl"
    shadow_dir = capture_dir / "shadow_asof"
    try:
        if not events.exists():
            return 0
        size = events.stat().st_size
        # last complete line's tail fields (cheap: read final 4KB)
        last_seq = None
        last_event_type = None
        with events.open("rb") as handle:
            back = min(size, 4096)
            handle.seek(size - back)
            tail = handle.read(back)
        complete = tail[: tail.rfind(b"\n")] if b"\n" in tail else b""
        if complete:
            try:
                last = json.loads(complete.rsplit(b"\n", 1)[-1])
                last_seq = last.get("sequence")
                last_event_type = last.get("event_type")
            except Exception:
                pass
        offset = size - (len(tail) - tail.rfind(b"\n") - 1) if b"\n" in tail else 0
        shadow_dir.mkdir(parents=True, exist_ok=True)
        with (shadow_dir / "boundary_ledger.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "schema_version": "Protocol101ShadowBoundaryLedgerV1",
                "wall_clock_pt": now.isoformat(),
                "boundary_minute_pt": now.strftime("%H:%M"),
                "events_bytes_total": size,
                "events_bytes_last_complete_line": offset,
                "last_sequence": last_seq,
                "last_event_type": last_event_type,
                "evidence_grade": "shadow_asof_diagnostic",
            }) + "\n")
        return 0
    except Exception as exc:
        try:
            shadow_dir.mkdir(parents=True, exist_ok=True)
            with (shadow_dir / "snapshot_errors.log").open("a") as handle:
                handle.write(f"{now.isoformat()} ledger exception: {exc!r}\n")
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
