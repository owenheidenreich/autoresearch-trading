"""Safe Section 3 preflight before any model hill-climb work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v4.foundation.model_experiment_preflight import (
    DEFAULT_OUT_DIR,
    build_model_experiment_preflight,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hypothesis-packet", type=Path, default=None)
    parser.add_argument("--require-hypothesis", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_model_experiment_preflight(
        args.repo_root,
        hypothesis_packet=args.hypothesis_packet,
        require_hypothesis=bool(args.require_hypothesis),
    )
    if not args.no_write:
        summary_path, report_path = write_outputs(payload, args.out_dir)
        payload["outputs"] = {"summary": str(summary_path), "report": str(report_path)}
    print(
        json.dumps(
            {
                "section3_model_experiment_decision": payload["section3_model_experiment_decision"],
                "training_blockers": payload["training_blockers"],
                "protocol101_challenge_blockers": payload["protocol101_challenge_blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
