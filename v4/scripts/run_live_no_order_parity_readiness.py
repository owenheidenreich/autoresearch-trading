"""Safe readiness check for a future live no-order parity session."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from v4.foundation.live_no_order_parity_readiness import (
    DEFAULT_OUT_DIR,
    build_live_no_order_parity_readiness,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--now", default=None, help="Optional ISO timestamp for tests/audits.")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.fromisoformat(args.now) if args.now else None
    payload = build_live_no_order_parity_readiness(args.repo_root, now=now)
    if not args.no_write:
        summary_path, report_path = write_outputs(payload, args.out_dir)
        payload["outputs"] = {"summary": str(summary_path), "report": str(report_path)}
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "market_open": payload["market_status"]["is_regular_market_session"],
                "missing_scaffolds": payload["missing_scaffolds"],
                "broker_endpoint_called": payload["broker_endpoint_called"],
                "live_orders": payload["live_orders"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
