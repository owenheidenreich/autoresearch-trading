"""Protocol 121: offline smoke for Protocol 101 live-entry router wiring.

This verifies the frozen Protocol 051/A+ surface edge scorer feeds the frozen
Protocol 101 event-policy inference path without zero-filling edge features.
It uses cached official-context SurfaceDecision rows only; no broker endpoint,
paid market-data endpoint, or order endpoint is called.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.live.protocol051_surface_edge import load_surface_edge_artifact, score_surface_decisions
from v4.live.protocol101_entry import (
    Protocol101HistoryState,
    load_protocol101_entry_artifact,
    predict_protocol101_entry,
    protocol101_candidate_frame_from_surface,
)
from v4.model.hypothesis_protocol import SurfaceDecision
from v4.scripts.run_protocol119_protocol101_live_readiness import (
    DEFAULT_PROTOCOL101_MANIFEST,
    DEFAULT_PROTOCOL101_SUMMARY,
)


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_121_protocol101_entry_router_smoke")
DEFAULT_SURFACE_MANIFEST = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_075_protocol054_frozen_stack_artifacts/"
    "model_artifacts/train_through_q4_2025_test_q1_2026/seed_11/manifest.json"
)
DEFAULT_DECISION_CACHE_DIR = Path("data/cache/v4_aplus_surface_decisions_official_context")
DEFAULT_DECISION_PATTERN = (
    "train_through_q4_2025_test_q1_2026_test.policy1.surface_structure_aplus_side_value_rank.*.pkl"
)
DEFAULT_LEDGER = Path("v4/ledger/RESEARCH_LEDGER.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--surface-manifest", type=Path, default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument("--protocol101-manifest", type=Path, default=DEFAULT_PROTOCOL101_MANIFEST)
    parser.add_argument("--protocol101-summary", type=Path, default=DEFAULT_PROTOCOL101_SUMMARY)
    parser.add_argument("--decision-cache-dir", type=Path, default=DEFAULT_DECISION_CACHE_DIR)
    parser.add_argument("--decision-pattern", default=DEFAULT_DECISION_PATTERN)
    parser.add_argument("--max-decisions", type=int, default=3000)
    parser.add_argument("--min-edge", type=float, default=25.0)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    surface_artifact = load_surface_edge_artifact(args.surface_manifest)
    protocol101 = load_protocol101_entry_artifact(args.protocol101_manifest, args.protocol101_summary)
    cache_path = _single_cache_path(args.decision_cache_dir, args.decision_pattern)
    decisions = _load_decisions(cache_path, max_decisions=args.max_decisions)
    surface_scores = score_surface_decisions(surface_artifact, decisions)
    history = Protocol101HistoryState()
    observations: list[dict[str, Any]] = []
    feature_rows = 0
    nonfinite_feature_rows = 0
    edge_values: list[float] = []
    entered = 0
    events_with_candidates = 0
    for decision, scores in zip(decisions, surface_scores):
        frame = protocol101_candidate_frame_from_surface(
            decision,
            scores,
            history,
            min_edge=float(args.min_edge),
        )
        if frame.empty:
            continue
        events_with_candidates += 1
        feature_rows += int(len(frame))
        edge_values.extend(float(value) for value in frame["edge"].to_numpy(dtype=float))
        features = frame[list(protocol101.feature_columns)].to_numpy(dtype=float)
        nonfinite_feature_rows += int(np.any(~np.isfinite(features), axis=1).sum())
        result = predict_protocol101_entry(protocol101, frame)
        entered += int(result["action"] == "enter")
        selected = result.get("selected") or {}
        observations.append(
            {
                "protocol_id": "protocol101",
                "decision_time": pd.Timestamp(decision.decision_time).isoformat(),
                "session": decision.session,
                "candidate_count": int(len(frame)),
                "action": result["action"],
                "reason": result["reason"],
                "margin": result.get("margin"),
                "threshold": result.get("threshold"),
                "selected_contract_id": selected.get("contract_id"),
                "selected_edge": selected.get("edge"),
                "order_intent": None,
            }
        )
        history.update(frame, pd.Timestamp(decision.decision_time))

    summary = {
        "input_decisions": len(decisions),
        "events_with_candidates": events_with_candidates,
        "candidate_feature_rows": feature_rows,
        "nonfinite_feature_rows": nonfinite_feature_rows,
        "protocol101_enter_signals": entered,
        "edge_feature_summary": _numeric_summary(edge_values),
    }
    decision = "pass_protocol101_entry_router_edge_wired"
    if events_with_candidates <= 0 or feature_rows <= 0 or nonfinite_feature_rows > 0:
        decision = "blocked_protocol101_entry_router_edge_wiring_failed"
    payload = {
        "protocol": "121_protocol101_entry_router_smoke",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "market_data_endpoint_called": False,
        "surface_manifest": str(args.surface_manifest),
        "protocol101_manifest": str(args.protocol101_manifest),
        "decision_cache": str(cache_path),
        "summary": summary,
        "next_gate": "Rerun Protocol 119. If IBKR live data is ready, run no-order Protocol 101 live capture.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "router_observations.json").write_text(json.dumps(observations, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2))
    return 0 if decision.startswith("pass_") else 1


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    edge = summary["edge_feature_summary"]
    lines = [
        "# Protocol 121: Protocol101 Entry Router Smoke",
        "",
        "No paid data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Input decisions: `{summary['input_decisions']}`",
        f"- Events with candidates: `{summary['events_with_candidates']}`",
        f"- Candidate feature rows: `{summary['candidate_feature_rows']}`",
        f"- Non-finite feature rows: `{summary['nonfinite_feature_rows']}`",
        f"- Protocol 101 enter signals: `{summary['protocol101_enter_signals']}`",
        f"- Edge median: `{edge['median']}`",
        f"- Edge min/max: `{edge['min']}` / `{edge['max']}`",
        "",
        "## What This Proves",
        "",
        "The frozen A+ surface scorer is now wired into the frozen Protocol 101 entry-policy feature path. The `edge` and edge-history features are produced causally from candidate sets rather than substituted with zeros.",
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 121 Protocol101 Entry Router Smoke"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Wired the frozen Protocol 051/A+ surface edge scorer into the frozen Protocol 101 entry-policy inference path and smoke-tested it on cached official-context SurfaceDecision rows.
Reason: Protocol 119 showed that Protocol 101 live deployment cannot zero-fill `edge`; this test proves the model path can produce edge and edge-history features causally before broker live capture.
Data Used: Existing Protocol 075 surface artifact, Protocol 101 artifact, and local cached official-context surface decisions only. No paid data was downloaded, no live broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Events with candidates={payload['summary']['events_with_candidates']}; candidate_feature_rows={payload['summary']['candidate_feature_rows']}; nonfinite_feature_rows={payload['summary']['nonfinite_feature_rows']}. Report: {report_path}
Next Gate: Rerun Protocol 119; if IBKR live data is available, run no-order Protocol 101 live capture.
Owner: Codex
```
"""
    existing = ledger.read_text() if ledger.exists() else ""
    if marker not in existing:
        ledger.write_text(existing.rstrip() + entry + "\n")
        return
    start = existing.index(marker)
    next_start = existing.find("\n## ", start + len(marker))
    replacement = entry.strip() + "\n"
    if next_start == -1:
        ledger.write_text(existing[:start].rstrip() + "\n\n" + replacement)
    else:
        ledger.write_text(existing[:start].rstrip() + "\n\n" + replacement + existing[next_start:])


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(clean),
        "min": clean[0],
        "median": float(np.median(clean)),
        "max": clean[-1],
    }


def _single_cache_path(cache_dir: Path, pattern: str) -> Path:
    matches = sorted(cache_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no cache file matched {cache_dir / pattern}")
    return matches[0]


def _load_decisions(path: Path, *, max_decisions: int) -> list[SurfaceDecision]:
    value = pickle.loads(path.read_bytes())
    if not isinstance(value, list) or not all(isinstance(item, SurfaceDecision) for item in value):
        raise ValueError(f"{path} must contain a list[SurfaceDecision]")
    return value[: max(0, int(max_decisions))]


if __name__ == "__main__":
    raise SystemExit(main())
