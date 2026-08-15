"""Protocol 120: offline portability check for the Protocol 051/A+ edge scorer."""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from v4.live.protocol051_surface_edge import load_surface_edge_artifact, surface_edge_rows
from v4.model.hypothesis_protocol import SurfaceDecision


DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/v4_aplus_hypothesis_120_surface_edge_portability")
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
    parser.add_argument("--decision-cache-dir", type=Path, default=DEFAULT_DECISION_CACHE_DIR)
    parser.add_argument("--decision-pattern", default=DEFAULT_DECISION_PATTERN)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = load_surface_edge_artifact(args.surface_manifest)
    cache_path = _single_cache_path(args.decision_cache_dir, args.decision_pattern)
    decisions = _load_decisions(cache_path, max_decisions=args.max_decisions)
    rows = surface_edge_rows(artifact, decisions)
    summary = summarize_edge_rows(rows)
    decision = "pass_surface_edge_generator_loads_offline_live_router_next"
    if summary["finite_edge_fraction"] < 0.99 or summary["scored_decisions"] == 0:
        decision = "blocked_surface_edge_generator_not_reliable"
    payload = {
        "protocol": "120_surface_edge_portability",
        "decision": decision,
        "paid_data_downloaded": False,
        "live_orders": False,
        "broker_endpoint_called": False,
        "surface_manifest": str(args.surface_manifest),
        "decision_cache": str(cache_path),
        "artifact": {
            "variant_name": artifact.variant_name,
            "trial_name": artifact.trial_name,
            "policy_index": artifact.policy_index,
            "target_scale": artifact.target_scale,
            "model_path": str(artifact.model_path),
            "standardizer_path": str(artifact.standardizer_path),
        },
        "summary": summary,
        "next_gate": (
            "Wire this loaded surface edge scorer into the Protocol 101 no-order live entry router, then rerun Protocol 119."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "edge_rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir / "report.md", payload)
    if not args.no_ledger:
        append_ledger(args.ledger, payload, args.out_dir / "report.md")
    print(json.dumps({"decision": decision, "report": str(args.out_dir / "report.md")}, indent=2))
    return 0 if decision.startswith("pass_") else 1


def summarize_edge_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite_edges = [float(row["edge"]) for row in rows if row.get("edge") is not None and math.isfinite(float(row["edge"]))]
    side_counts: dict[str, int] = {}
    for row in rows:
        right = row.get("right")
        if right:
            side_counts[str(right)] = side_counts.get(str(right), 0) + 1
    return {
        "scored_decisions": len(rows),
        "finite_edge_count": len(finite_edges),
        "finite_edge_fraction": float(len(finite_edges) / max(len(rows), 1)),
        "positive_edge_fraction": float(sum(edge > 0.0 for edge in finite_edges) / max(len(finite_edges), 1)),
        "edge_min": float(min(finite_edges)) if finite_edges else None,
        "edge_median": float(np.median(finite_edges)) if finite_edges else None,
        "edge_max": float(max(finite_edges)) if finite_edges else None,
        "selected_side_counts": side_counts,
        "valid_token_count_min": int(min((row.get("valid_token_count", 0) for row in rows), default=0)),
        "valid_token_count_median": float(np.median([row.get("valid_token_count", 0) for row in rows])) if rows else 0.0,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Protocol 120: Surface Edge Portability",
        "",
        "No paid data was downloaded. No live broker data or order endpoint was used.",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Surface artifact: `{payload['artifact']['variant_name']}`",
        f"- Trial: `{payload['artifact']['trial_name']}`",
        f"- Decisions scored: `{summary['scored_decisions']}`",
        f"- Finite edge fraction: `{summary['finite_edge_fraction']:.3f}`",
        f"- Positive edge fraction: `{summary['positive_edge_fraction']:.3f}`",
        f"- Edge median: `{summary['edge_median']:.3f}`",
        f"- Selected side counts: `{summary['selected_side_counts']}`",
        "",
        "## Next Gate",
        "",
        payload["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n")


def append_ledger(ledger: Path, payload: dict[str, Any], report_path: Path) -> None:
    marker = "## 2026-05-14 Protocol 120 Surface Edge Portability"
    entry = f"""

{marker}

```text
Date: 2026-05-14
Decision / Experiment: Loaded the frozen Protocol 051/A+ surface scorer as a reusable edge generator and scored cached official-context SurfaceDecision rows.
Reason: Protocol 119 identified the upstream `edge` feature family as a blocker for Protocol 101 live deployment. The project needed to prove the frozen scorer can be loaded independently before wiring it into the live entry router.
Data Used: Existing Protocol 075 surface artifact and local cached official-context surface decisions only. No paid data was downloaded, no live broker endpoint was called, and no orders were placed.
Cost: $0 incremental paid data.
Result: Decision {payload['decision']}. Scored {payload['summary']['scored_decisions']} decisions with finite_edge_fraction={payload['summary']['finite_edge_fraction']:.3f}. Report: {report_path}
Next Gate: Wire the surface edge scorer into the Protocol 101 no-order live entry router, then rerun Protocol 119.
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
