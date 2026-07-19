"""Shadow-as-of diff: compare intra-session as-of decision snapshots against
the final end-of-day replay of the same capture.

Sealing rules (v4/docs/PROTOCOL101_SHADOW_ASOF_WEEK_PLAN_2026_07_18.md):
- SEALED sessions: no decision-level analysis. Writes a count-only marker to
  the governance dir and exits; snapshots seal with the capture at 13:30.
- Development/validation sessions: full diff, written into
  <capture_dir>/shadow_asof/ plus a summary copy in the governance dir.

Disagreements are classified `trailing_edge` when the minute lies within the
final 2 minutes a snapshot had seen (data still arriving — expected), else
`settled` (the real live-vs-replay signal). Preregistered bar: settled
agreement >= 0.995 on action + selected contract.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
REPO = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture"
SEALED_ROOT = Path.home() / ".autoresearch-trading/live_runtime/ibkr_capture_sealed"
GOVERNANCE = REPO / "v4/audit/autoresearch/protocol101_shadow_asof"
PYTHON = str(Path.home() / ".autoresearch-trading/runtime-venv/bin/python")
TRAILING_MINUTES = 2

sys.path.insert(0, str(REPO))
from v4.scripts.run_protocol101_sealed_day_assignment import classify  # noqa: E402


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_snapshots(shadow_dir: Path) -> list[tuple[str, dict, list[dict]]]:
    out = []
    for path in sorted(shadow_dir.glob("asof_*.decision_log.jsonl")):
        with path.open() as handle:
            lines = [json.loads(l) for l in handle if l.strip()]
        if not lines or not lines[0].get("meta"):
            continue
        out.append((path.name, lines[0], lines[1:]))
    return out


def final_replay(session: str, capture_root: Path) -> dict[str, dict]:
    with tempfile.TemporaryDirectory(prefix="shadow_final_") as tmp:
        out_dir = Path(tmp) / "final"
        proc = subprocess.run(
            [PYTHON, "v4/scripts/run_protocol101_fair_contract_ibkr_capture_replay.py",
             "--session", session, "--capture-root", str(capture_root),
             "--out-dir", str(out_dir)],
            cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
            capture_output=True, text=True, timeout=20 * 60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"final replay failed: {proc.stderr[-400:]}")
        finals: dict[str, dict] = {}
        with (out_dir / "decision_traces.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                finals[str(row.get("decision_ts"))] = row
        return finals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=None, help="default: today in PT")
    parser.add_argument("--manual", action="store_true", help="bypass weekday gate")
    args = parser.parse_args()
    now = datetime.now(PT)
    if not args.manual and now.weekday() >= 5:
        return 0
    session = args.session or now.strftime("%Y-%m-%d")
    cls = classify(session)

    capture_dir = CAPTURE_ROOT / session / f"protocol101-recorder-{session}"
    sealed_dir = SEALED_ROOT / session / f"protocol101-recorder-{session}"
    if cls == "sealed":
        # Count-only marker; never read decision content. The capture may
        # already have been moved by the 13:30 sealing job.
        base = capture_dir if capture_dir.exists() else sealed_dir
        count = len(list((base / "shadow_asof").glob("asof_*.decision_log.jsonl"))) if (base / "shadow_asof").exists() else 0
        write_json(GOVERNANCE / f"marker_{session.replace('-', '_')}.json", {
            "session": session, "class": cls, "shadow_snapshots": count,
            "status": "sealed_diff_deferred_until_after_exam",
            "recorded_at": now.isoformat(),
        })
        return 0
    if cls not in ("development", "validation"):
        return 0
    shadow_dir = capture_dir / "shadow_asof"
    snapshots = load_snapshots(shadow_dir)
    if not snapshots:
        write_json(GOVERNANCE / f"marker_{session.replace('-', '_')}.json", {
            "session": session, "class": cls, "shadow_snapshots": 0,
            "status": "no_snapshots", "recorded_at": now.isoformat()})
        return 0

    finals = final_replay(session, CAPTURE_ROOT)
    per_snapshot, settled_total, settled_agree, trailing_total, trailing_agree = [], 0, 0, 0, 0
    disagreements = []
    for name, meta, rows in snapshots:
        if not rows:
            continue
        max_ts = max(str(r["decision_ts"]) for r in rows)
        s_tot = s_ok = t_tot = t_ok = 0
        for row in rows:
            ts = str(row["decision_ts"])
            final = finals.get(ts)
            if final is None:
                continue
            same = (row.get("selected_action") == final.get("selected_action")
                    and row.get("selected_contract_id") == final.get("selected_contract_id"))
            trailing = (max_ts_minutes_between(ts, max_ts) < TRAILING_MINUTES)
            if trailing:
                t_tot += 1
                t_ok += int(same)
            else:
                s_tot += 1
                s_ok += int(same)
            if not same:
                disagreements.append({
                    "snapshot": name, "decision_ts": ts,
                    "zone": "trailing_edge" if trailing else "settled",
                    "asof": {k: row.get(k) for k in ("selected_action", "selected_contract_id")},
                    "final": {k: final.get(k) for k in ("selected_action", "selected_contract_id")},
                })
        per_snapshot.append({
            "snapshot": name, "wall_clock_pt": meta.get("wall_clock_pt"),
            "rows": len(rows), "settled": s_tot, "settled_agree": s_ok,
            "trailing": t_tot, "trailing_agree": t_ok,
        })
        settled_total += s_tot
        settled_agree += s_ok
        trailing_total += t_tot
        trailing_agree += t_ok

    settled_rate = settled_agree / settled_total if settled_total else None
    payload = {
        "schema_version": "Protocol101ShadowAsofDiffV1",
        "session": session, "class": cls,
        "evidence_grade": "shadow_asof_diagnostic",
        "snapshots": per_snapshot,
        "settled_agreement": settled_rate,
        "trailing_agreement": trailing_agree / trailing_total if trailing_total else None,
        "preregistered_bar_settled": 0.995,
        "pass_settled_bar": (settled_rate is None or settled_rate >= 0.995),
        "disagreements": disagreements[:200],
        "generated_at": now.isoformat(),
    }
    write_json(shadow_dir / "shadow_vs_final_diff.json", payload)
    write_json(GOVERNANCE / f"diff_summary_{session.replace('-', '_')}.json",
               {k: payload[k] for k in payload if k != "disagreements"})
    lines = [f"# Shadow-as-of diff — {session} ({cls})", "",
             f"- settled agreement: {settled_rate}",
             f"- trailing-edge agreement: {payload['trailing_agreement']}",
             f"- passes 0.995 settled bar: {payload['pass_settled_bar']}",
             f"- snapshots: {len(per_snapshot)}; disagreements listed: {len(disagreements)}"]
    (shadow_dir / "shadow_vs_final_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: payload[k] for k in ("session", "settled_agreement", "pass_settled_bar")}))
    return 0


def max_ts_minutes_between(ts: str, max_ts: str) -> float:
    try:
        a = datetime.fromisoformat(ts)
        b = datetime.fromisoformat(max_ts)
        return abs((b - a).total_seconds()) / 60.0
    except Exception:
        return 1e9


if __name__ == "__main__":
    sys.exit(main())
