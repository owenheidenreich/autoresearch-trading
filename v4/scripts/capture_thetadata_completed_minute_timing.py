"""Measure no-order ThetaData SPX completed-minute availability.

The command polls only ``index_history_ohlc`` and records the first local time
each newly completed bar appears.  It has no broker, order, model, registry, or
holdout imports.  The resulting receipt is development evidence; callers must
accumulate enough samples across sessions before freezing an emission lag.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time as clock
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.checks.paid_data_guard import add_paid_data_approval_args, require_paid_data_approval
from v4.scripts.download_thetadata_index_bars import _client, _load_env_file, _normalize_theta_ohlc


NY = ZoneInfo("America/New_York")
SCHEMA = "autoresearch.thetadata-completed-minute-timing.v1"
DEFAULT_APPROVAL = Path(
    "v4/audit/autoresearch/thetadata_completed_minute_timing_2026_08_03/authorization.json"
)


def _stable_hash(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("receipt_sha256", None)
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _quantiles(values: list[int]) -> dict[str, int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()),
        "p50": int(np.quantile(array, 0.50, method="higher")),
        "p90": int(np.quantile(array, 0.90, method="higher")),
        "p99": int(np.quantile(array, 0.99, method="higher")),
        "max": int(array.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--duration-seconds", type=int, default=360)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-samples-to-freeze", type=int, default=5)
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    add_paid_data_approval_args(parser, default_manifest=DEFAULT_APPROVAL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 60 <= args.duration_seconds <= 1_800:
        raise SystemExit("--duration-seconds must be in 60..1800")
    if not 0.5 <= args.poll_seconds <= 10.0:
        raise SystemExit("--poll-seconds must be in 0.5..10")
    if not 5 <= args.minimum_samples_to_freeze <= 100:
        raise SystemExit("--minimum-samples-to-freeze must be in 5..100")
    plan = {
        "source": "ThetaData",
        "endpoint": "index_history_ohlc",
        "symbol": "SPX",
        "session_date": args.session_date.isoformat(),
        "duration_seconds": args.duration_seconds,
        "poll_seconds": args.poll_seconds,
        "output": str(args.output.resolve()),
        "broker_or_order_path": False,
        "protected_holdout_path": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite timing receipt: {args.output}")
    require_paid_data_approval(
        manifest_path=args.approval_manifest,
        approval_text=args.approval_text,
        approval_env_var=args.approval_env_var,
        operation="no-order ThetaData SPX completed-minute timing measurement",
    )
    _load_env_file(args.env_file)
    client = _client()
    session_open = time(9, 30)
    session_close = time(16, 0)
    latest_open_ns: int | None = None
    first_seen: dict[int, int] = {}
    polls = 0
    errors: list[str] = []
    started_ns = clock.time_ns()
    deadline = clock.monotonic() + args.duration_seconds
    while clock.monotonic() < deadline:
        poll_started_ns = clock.time_ns()
        try:
            raw = client.index_history_ohlc(
                symbol="SPX",
                start_date=args.session_date,
                end_date=args.session_date,
                interval="1m",
                start_time=session_open,
                end_time=session_close,
            )
            normalized = _normalize_theta_ohlc(raw, symbol="SPX")
            event_ns = set(
                pd.to_datetime(normalized["event_time"], utc=True)
                .astype("datetime64[ns, UTC]")
                .astype("int64")
            )
            if event_ns:
                observed_latest = max(int(value) for value in event_ns)
                if latest_open_ns is not None and observed_latest > latest_open_ns:
                    # ThetaData exposes the current, still-forming bar.  A bar
                    # stamped t is complete only when the following bar stamped
                    # t+60s becomes observable.  Therefore the new bar's open is
                    # the represented interval end for the preceding bar.
                    newly_opened = sorted(
                        int(value) for value in event_ns if int(value) > latest_open_ns
                    )
                    for next_open_ns in newly_opened:
                        completed_event_ns = next_open_ns - 60_000_000_000
                        first_seen.setdefault(completed_event_ns, poll_started_ns)
                latest_open_ns = observed_latest
            polls += 1
        except Exception as exc:  # vendor/network evidence, never include credentials
            errors.append(f"{type(exc).__name__}:{str(exc)[:240]}")
        remaining = deadline - clock.monotonic()
        if remaining > 0:
            clock.sleep(min(args.poll_seconds, remaining))
    finished_ns = clock.time_ns()
    samples = []
    for event_ns, receipt_ns in sorted(first_seen.items()):
        represented_end_ns = event_ns + 60_000_000_000
        lag_ns = receipt_ns - represented_end_ns
        if lag_ns < 0:
            errors.append(f"bar_seen_before_represented_end:{event_ns}")
            continue
        samples.append(
            {
                "event_time_utc": datetime.fromtimestamp(event_ns / 1e9, tz=timezone.utc).isoformat(),
                "represented_interval_end_utc": datetime.fromtimestamp(
                    represented_end_ns / 1e9, tz=timezone.utc
                ).isoformat(),
                "first_local_receipt_utc": datetime.fromtimestamp(
                    receipt_ns / 1e9, tz=timezone.utc
                ).isoformat(),
                "receipt_lag_ns": lag_ns,
            }
        )
    lags = [int(row["receipt_lag_ns"]) for row in samples]
    quantiles = _quantiles(lags)
    within_supported_lag = (
        quantiles is not None and quantiles["p99"] <= 10_000_000_000
    )
    enough = (
        len(samples) >= args.minimum_samples_to_freeze
        and not errors
        and within_supported_lag
    )
    frozen_lag_ms = None
    if enough and quantiles is not None:
        frozen_lag_ms = int(math.ceil(quantiles["p99"] / 1_000_000))
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS_FREEZE_READY" if enough else "COLLECTED_NOT_FREEZE_READY",
        "plan": plan,
        "capture_started_unix_ns": started_ns,
        "capture_finished_unix_ns": finished_ns,
        "poll_count": polls,
        "minimum_samples_to_freeze": args.minimum_samples_to_freeze,
        "sample_count": len(samples),
        "all_samples_within_supported_10s_lag": within_supported_lag,
        "samples": samples,
        "receipt_lag_ns": quantiles,
        "frozen_emission_lag_ms": frozen_lag_ms,
        "errors": errors,
        "secrets_recorded": False,
    }
    result["receipt_sha256"] = _stable_hash(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if enough else 3


if __name__ == "__main__":
    raise SystemExit(main())
