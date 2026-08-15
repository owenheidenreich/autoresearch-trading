"""CLI for the frozen Path-D Wave-2 fixed-score discovery gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from v4.research.pathd_wave2_signal_discovery import (
    DataContractError,
    PREREGISTRATION_PATH,
    Wave2Paths,
    run_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Volumes/AR_TRADING_DATA"))
    parser.add_argument("--counterfactuals", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/AR_TRADING_DATA/artifacts/pathd_wave2_signal_discovery_2026_08_03"),
    )
    parser.add_argument("--preregistration", type=Path, default=PREREGISTRATION_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = Wave2Paths.from_root(args.root, counterfactuals=args.counterfactuals)
    try:
        result = run_gate(
            paths,
            args.output_root,
            preregistration=args.preregistration,
            repo_root=args.repo_root,
        )
    except DataContractError as exc:
        args.output_root.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": "pathd.wave2-causal-60m-signal-discovery.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": "BLOCKED_CLOCK_OR_DATA_CONTRACT",
            "error": str(exc),
            "protected_holdout_opened": False,
            "model_fit_executed": False,
            "broker_accessed": False,
            "paper_order_submitted": False,
        }
        (args.output_root / "results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        (args.output_root / "report.md").write_text(
            "# Path-D Wave 2 — Blocked\n\n"
            f"**Verdict:** `BLOCKED_CLOCK_OR_DATA_CONTRACT`\n\n{exc}\n\n"
            "No model, holdout, broker, paper, runtime, promotion, or default path was used.\n\n"
            "STOP_FOR_CLAUDE_VERIFICATION\n"
        )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "output_root": str(args.output_root),
                "result_semantic_sha256": result.get("result_semantic_sha256"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
