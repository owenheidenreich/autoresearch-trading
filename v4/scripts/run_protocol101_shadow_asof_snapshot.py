"""Shadow-as-of snapshot: run the deterministic capture replay over the
current session's capture-so-far and log the decisions visible at this
wall-clock moment.

Design (see v4/docs/PROTOCOL101_SHADOW_ASOF_WEEK_PLAN_2026_07_18.md):
- Read-only on the live capture: the events file is copied, truncated to the
  last complete line, into a temp capture root; the unmodified replay script
  runs against the copy.
- Outputs land INSIDE the session capture dir under shadow_asof/, so sealed
  days sweep their shadow evidence into the vault automatically.
- Self-gated: weekday, 06:35-13:10 PT window, capture present and >= 10 MB.
  Any failure logs and exits 0 — this job must never disturb the recorder.
- Evidence grade: shadow_asof_diagnostic. Never part of the confirmation
  battery.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
REPO = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture"
PYTHON = str(Path.home() / ".autoresearch-trading/runtime-venv/bin/python")
MIN_BYTES = 10 * 1024 * 1024
WINDOW = ((6, 35), (13, 10))
COMPACT_KEYS = (
    "decision_ts",
    "selected_action",
    "selected_contract_id",
    "selected_score",
    "decision_threshold",
    "candidate_count",
)


def log_error(shadow_dir: Path, message: str) -> None:
    try:
        shadow_dir.mkdir(parents=True, exist_ok=True)
        with (shadow_dir / "snapshot_errors.log").open("a") as handle:
            handle.write(f"{datetime.now(PT).isoformat()} {message}\n")
    except Exception:
        pass


def last_newline_offset(path: Path) -> int:
    size = path.stat().st_size
    with path.open("rb") as handle:
        back = min(size, 1 << 20)
        handle.seek(size - back)
        tail = handle.read(back)
    cut = tail.rfind(b"\n")
    if cut < 0:
        return 0
    return size - back + cut + 1


def copy_truncated(src: Path, dst: Path, limit: int) -> None:
    with src.open("rb") as fin, dst.open("wb") as fout:
        remaining = limit
        while remaining > 0:
            chunk = fin.read(min(1 << 22, remaining))
            if not chunk:
                break
            fout.write(chunk)
            remaining -= len(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=None, help="default: today in PT")
    parser.add_argument("--manual", action="store_true",
                        help="bypass weekday/window gates (dry-runs, catch-up)")
    parser.add_argument("--truncate-fraction", type=float, default=None,
                        help="manual testing: use only this fraction of the events file")
    args = parser.parse_args()

    now = datetime.now(PT)
    session = args.session or now.strftime("%Y-%m-%d")
    capture_dir = CAPTURE_ROOT / session / f"protocol101-recorder-{session}"
    shadow_dir = capture_dir / "shadow_asof"
    if not args.manual:
        if now.weekday() >= 5:
            return 0
        hm = (now.hour, now.minute)
        if not (WINDOW[0] <= hm <= WINDOW[1]):
            return 0
    events = capture_dir / "market_events.jsonl"
    if not events.exists() or events.stat().st_size < MIN_BYTES:
        return 0
    lock = shadow_dir / ".snapshot.lock"
    try:
        shadow_dir.mkdir(parents=True, exist_ok=True)
        try:
            lock.touch(exist_ok=False)
        except FileExistsError:
            age = datetime.now().timestamp() - lock.stat().st_mtime
            if age < 20 * 60:
                return 0  # previous snapshot still running
            lock.touch()

        limit = last_newline_offset(events)
        if args.truncate_fraction:
            limit = int(limit * args.truncate_fraction)
            with events.open("rb") as handle:
                handle.seek(max(limit - (1 << 20), 0))
                tail = handle.read(min(limit, 1 << 20))
            limit = max(limit - (len(tail) - tail.rfind(b"\n") - 1), 0)
        stamp = now.strftime("%H%M")

        with tempfile.TemporaryDirectory(prefix="shadow_asof_") as tmp:
            tmp_root = Path(tmp) / "capture_root"
            tmp_capture = tmp_root / session / f"protocol101-recorder-{session}"
            tmp_capture.mkdir(parents=True)
            copy_truncated(events, tmp_capture / "market_events.jsonl", limit)
            tmp_out = Path(tmp) / "replay_out"
            proc = subprocess.run(
                [PYTHON, "v4/scripts/run_protocol101_fair_contract_ibkr_capture_replay.py",
                 "--session", session, "--capture-root", str(tmp_root),
                 "--out-dir", str(tmp_out)],
                cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                capture_output=True, text=True, timeout=13 * 60,
            )
            if proc.returncode != 0:
                log_error(shadow_dir, f"replay failed rc={proc.returncode}: {proc.stderr[-400:]}")
                return 0
            traces = tmp_out / "decision_traces.jsonl"
            rows = []
            with traces.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    rows.append({k: row.get(k) for k in COMPACT_KEYS})
        out_path = shadow_dir / f"asof_{stamp}.decision_log.jsonl"
        with out_path.open("w") as handle:
            handle.write(json.dumps({
                "meta": True,
                "schema_version": "Protocol101ShadowAsofSnapshotV1",
                "session": session,
                "wall_clock_pt": now.isoformat(),
                "events_bytes_read": limit,
                "rows_decided": len(rows),
                "evidence_grade": "shadow_asof_diagnostic",
            }) + "\n")
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return 0
    except Exception as exc:  # never disturb the recorder
        log_error(shadow_dir, f"snapshot exception: {exc!r}")
        return 0
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
