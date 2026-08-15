"""Build formal validation controls from existing replay summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v4.foundation.formal_validation_governance import (
    DEFAULT_AUDIT_ROOT,
    DEFAULT_OUT_DIR,
    build_formal_validation_governance,
    strategy_matrix,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_formal_validation_governance(args.audit_root)
    matrix = strategy_matrix(args.audit_root)
    if not args.no_write:
        summary_path, report_path = write_outputs(payload, matrix, args.out_dir)
        payload["outputs"] = {"summary": str(summary_path), "report": str(report_path)}
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "control_status": payload["control_status"],
                "pbo_cscv_status": payload["pbo_cscv_status"],
                "comparable_strategy_count": payload["comparable_strategy_count"],
                "comparable_split_count": payload["comparable_split_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
