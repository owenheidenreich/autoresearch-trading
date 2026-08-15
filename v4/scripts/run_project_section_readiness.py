"""Read-only readiness report for project sections 1-5.

This runner does not train, tune, download paid data, call broker endpoints, or
mutate operational runtime configs. It writes an audit summary/report only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v4.foundation.project_sections import build_readiness_payload, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("v4/audit/autoresearch/project_section_readiness"))
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_readiness_payload(args.repo_root)
    if not args.no_write:
        summary_path, report_path = write_outputs(payload, args.out_dir)
        payload["outputs"] = {"summary": str(summary_path), "report": str(report_path)}
    print(
        json.dumps(
            {
                "section_1_2_decision": payload["section_1_2_decision"],
                "model_hill_climb_decision": payload["model_hill_climb_decision"],
                "model_hill_climb_blockers": payload["model_hill_climb_blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

