#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Auto-load .env so API keys are always available
_env_path = os.path.join(ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from training.live.service import PaperLiveConfig, PaperTradingService


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time IBKR paper trader with Polygon context bootstrap")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002, help="IBKR paper port default=4002")
    parser.add_argument("--client-id", type=int, default=70)
    parser.add_argument("--model", default="training/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--train-py", default=None, help="Optional train.py/best_train.py for dynamic architecture loading")
    parser.add_argument("--context-days", type=int, default=30)
    parser.add_argument("--context-only", action="store_true", help="Run context_refresh job only and exit")
    parser.add_argument("--no-context-refresh", action="store_true", help="Skip refresh and use latest saved context bundle")
    parser.add_argument("--dry-run", action="store_true", help="Do not place orders; still runs full decision loop")
    parser.add_argument("--paper-auto", action="store_true", help="Enable automatic paper order placement")
    parser.add_argument("--kill-switch", default=None, help="Path to kill-switch file (content: 1/on/true/stop/kill)")
    parser.add_argument("--audit-path", default="results/live/audit.jsonl")
    parser.add_argument("--max-position-size", type=int, default=1)
    parser.add_argument("--min-trade-prob", type=float, default=0.55)
    parser.add_argument("--start-time-et", default="09:30")
    parser.add_argument("--end-time-et", default="16:00")
    parser.add_argument("--max-minutes", type=int, default=None, help="Optional cap for test runs")
    args = parser.parse_args()

    cfg = PaperLiveConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        model_path=args.model,
        train_py_path=args.train_py,
        context_days=args.context_days,
        refresh_context=not args.no_context_refresh,
        paper_auto=args.paper_auto,
        dry_run=args.dry_run,
        kill_switch_path=args.kill_switch,
        audit_path=args.audit_path,
        max_position_size=args.max_position_size,
        min_trade_prob=args.min_trade_prob,
        start_time_et=args.start_time_et,
        end_time_et=args.end_time_et,
        max_minutes=args.max_minutes,
    )
    svc = PaperTradingService(cfg)
    if args.context_only:
        path = svc.run_context_refresh()
        print(f"context_refresh completed: {path}")
        return
    svc.run_session()


if __name__ == "__main__":
    main()
