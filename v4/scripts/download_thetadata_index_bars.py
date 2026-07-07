"""Download official ThetaData SPX/VIX 1-minute index bars.

This script is intentionally narrow: it only downloads index OHLC bars for the
official-context promotion gate. It reads ThetaData account credentials from a
local env file and never prints secrets.

Paid-data guardrail: use ``--auth-check`` for a no-market-data credential check.
Do not run download mode until the exact date range and symbols are approved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import date, time
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)


_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class ThetaIndexDownloadRecord:
    symbol: str
    start_date: str
    end_date: str
    interval: str
    output_dir: str
    files_written: int
    rows_written: int
    first_event_time: str | None
    last_event_time: str | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    p.add_argument("--start-date", default="2025-01-02")
    p.add_argument("--end-date", default="2026-03-31")
    p.add_argument("--symbols", nargs="+", default=["SPX", "VIX"])
    p.add_argument("--interval", default="1m")
    p.add_argument("--start-time", default="09:30:00")
    p.add_argument("--end-time", default="16:00:00")
    p.add_argument("--output-root", type=Path, default=Path("data/vendor/thetadata/index"))
    p.add_argument(
        "--max-request-days",
        type=int,
        default=365,
        help="ThetaData rejects overly long date ranges; split requests into chunks no longer than this many days.",
    )
    p.add_argument(
        "--audit-out",
        type=Path,
        default=Path("v4/audit/thetadata_index_downloads.jsonl"),
    )
    p.add_argument("--auth-check", action="store_true", help="instantiate ThetaClient only; no market-data request")
    p.add_argument("--dry-run", action="store_true", help="print planned request; no ThetaData request")
    p.add_argument("--overwrite", action="store_true")
    add_paid_data_approval_args(p)
    return p.parse_args()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _date(value: str) -> date:
    return pd.Timestamp(value).date()


def _time(value: str) -> time:
    parts = value.split(":")
    if len(parts) < 2:
        raise ValueError(f"invalid time: {value}")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(float(parts[2])) if len(parts) > 2 else 0
    return time(hour, minute, second)


def _date_chunks(start_date: date, end_date: date, max_days: int) -> Iterable[tuple[date, date]]:
    if max_days <= 0:
        raise ValueError("--max-request-days must be positive")
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, (pd.Timestamp(cursor) + pd.Timedelta(days=max_days - 1)).date())
        yield cursor, chunk_end
        cursor = (pd.Timestamp(chunk_end) + pd.Timedelta(days=1)).date()


def _client() -> Any:
    email = os.environ.get("THETADATA_EMAIL")
    password = os.environ.get("THETADATA_PASSWORD")
    if not email or not password:
        raise SystemExit("missing THETADATA_EMAIL/THETADATA_PASSWORD in environment or env file")
    try:
        from thetadata import ThetaClient
    except ImportError:
        print("thetadata is not installed. Run with uv after installing project dependencies.", file=sys.stderr)
        raise SystemExit(2)
    return ThetaClient(email=email, password=password, dataframe_type="pandas")


def _event_time_utc(series: pd.Series) -> pd.Series:
    """Interpret ThetaData naive timestamps as exchange-local New York time."""
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize(_NY, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert(_UTC)
    return parsed.dt.tz_convert(_UTC)


def _normalize_theta_ohlc(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "event_time",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "count",
                "vwap",
                "context_source",
                "is_derived",
                "is_proxy",
                "is_official_index_data",
            ]
        )

    working = frame.reset_index() if frame.index.name else frame.copy()
    lower_to_col = {str(col).lower(): col for col in working.columns}
    time_col = None
    for candidate in ("timestamp", "event_time", "datetime", "time", "ts"):
        if candidate in lower_to_col:
            time_col = lower_to_col[candidate]
            break
    if time_col is None:
        raise ValueError(f"could not identify ThetaData timestamp column in {list(working.columns)}")

    out = pd.DataFrame()
    out["event_time"] = _event_time_utc(working[time_col])
    out["symbol"] = symbol.upper()
    for col in ("open", "high", "low", "close"):
        source = lower_to_col.get(col)
        if source is None:
            raise ValueError(f"missing ThetaData {col} column in {list(working.columns)}")
        out[col] = pd.to_numeric(working[source], errors="coerce")
    for col in ("volume", "count"):
        source = lower_to_col.get(col)
        out[col] = pd.to_numeric(working[source], errors="coerce").fillna(0).astype("int64") if source else 0
    source = lower_to_col.get("vwap")
    out["vwap"] = pd.to_numeric(working[source], errors="coerce") if source else pd.NA
    out["context_source"] = "thetadata_index_history_ohlc"
    out["is_derived"] = False
    out["is_proxy"] = False
    out["is_official_index_data"] = True
    out = out.dropna(subset=["event_time", "open", "high", "low", "close"]).sort_values("event_time")
    return out.reset_index(drop=True)


def _output_dir(root: Path, symbol: str, interval: str) -> Path:
    return root / f"{symbol.lower()}_{interval}"


def _write_by_session(frame: pd.DataFrame, *, output_dir: Path, overwrite: bool) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        return 0, 0
    sessions = frame["event_time"].dt.tz_convert(_NY).dt.date
    files = 0
    rows = 0
    for session, day_frame in frame.groupby(sessions):
        out_path = output_dir / f"{session.isoformat()}.parquet"
        if out_path.exists() and not overwrite:
            continue
        day_frame = day_frame.sort_values("event_time").reset_index(drop=True)
        day_frame.to_parquet(out_path, index=False)
        files += 1
        rows += len(day_frame)
    return files, rows


def _record(path: Path, record: ThetaIndexDownloadRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _download_symbol(
    client: Any,
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    interval: str,
    start_time: time,
    end_time: time,
    output_root: Path,
    audit_out: Path,
    overwrite: bool,
) -> ThetaIndexDownloadRecord:
    try:
        frame = client.index_history_ohlc(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as exc:  # pragma: no cover - vendor/network failure path
        message = str(exc)
        if "PERMISSION_DENIED" in message or "standard subscription" in message:
            raise SystemExit(
                "ThetaData permission denied for index_history_ohlc. "
                "The credentials authenticated, but the account does not have "
                "an active Index Standard/Pro subscription for SPX/VIX endpoints."
            ) from exc
        if os.environ.get("V4_DEBUG_THETADATA"):
            traceback.print_exc()
        raise SystemExit(f"ThetaData index_history_ohlc request failed for {symbol}: {exc}") from exc
    normalized = _normalize_theta_ohlc(frame, symbol=symbol)
    out_dir = _output_dir(output_root, symbol, interval)
    files, rows = _write_by_session(normalized, output_dir=out_dir, overwrite=overwrite)
    record = ThetaIndexDownloadRecord(
        symbol=symbol.upper(),
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        interval=interval,
        output_dir=str(out_dir),
        files_written=files,
        rows_written=rows,
        first_event_time=None if normalized.empty else normalized["event_time"].iloc[0].isoformat(),
        last_event_time=None if normalized.empty else normalized["event_time"].iloc[-1].isoformat(),
    )
    _record(audit_out, record)
    return record


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file)
    symbols = [symbol.upper() for symbol in args.symbols]
    start_date = _date(args.start_date)
    end_date = _date(args.end_date)
    start_time = _time(args.start_time)
    end_time = _time(args.end_time)

    if args.dry_run:
        chunks = [
            {"start_date": chunk_start.isoformat(), "end_date": chunk_end.isoformat()}
            for chunk_start, chunk_end in _date_chunks(start_date, end_date, args.max_request_days)
        ]
        print(
            json.dumps(
                {
                    "source": "ThetaData",
                    "product": "index_history_ohlc",
                    "symbols": symbols,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "chunks": chunks,
                    "interval": args.interval,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "output_root": str(args.output_root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.auth_check:
        client = _client()
        print("ThetaData client initialized. No market-data request was made.")
        return 0

    require_paid_data_approval(
        manifest_path=args.approval_manifest,
        approval_text=args.approval_text,
        approval_env_var=args.approval_env_var,
        operation=f"ThetaData index_history_ohlc download for {','.join(symbols)} {start_date} through {end_date}",
    )
    client = _client()

    records = []
    for symbol in symbols:
        for chunk_start, chunk_end in _date_chunks(start_date, end_date, args.max_request_days):
            records.append(
                _download_symbol(
                    client,
                    symbol=symbol,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    interval=args.interval,
                    start_time=start_time,
                    end_time=end_time,
                    output_root=args.output_root,
                    audit_out=args.audit_out,
                    overwrite=args.overwrite,
                )
            )
    print(json.dumps([asdict(record) for record in records], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
