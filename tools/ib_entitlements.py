#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.live.entitlements import probe_entitlements


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe IBKR live paper-data entitlements")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=91)
    args = parser.parse_args()

    report = probe_entitlements(host=args.host, port=args.port, client_id=args.client_id)
    output = {
        "passed": report.passed,
        "created_at": report.created_at,
        "host": report.host,
        "port": report.port,
        "account": report.account,
        "warnings": report.warnings,
        "symbols": {k: vars(v) for k, v in report.symbols.items()},
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
