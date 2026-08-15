"""Safe untouched-holdout availability preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v4.foundation.holdout_availability import DEFAULT_OUT_DIR, build_holdout_availability, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_holdout_availability(args.repo_root)
    if not args.no_write:
        summary_path, report_path = write_outputs(payload, args.out_dir)
        payload["outputs"] = {"summary": str(summary_path), "report": str(report_path)}
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "data_available": payload["data_available"],
                "data_status": payload["data_status"],
                "protected_holdout_scored": payload["protected_holdout_scored"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
