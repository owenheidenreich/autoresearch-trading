"""Download low-cost Databento context proxies for SPX/VIX index bars.

This is a stopgap for local prototype wiring when licensed SPX/VIX index bars
are not yet available. ES futures are written in the SPX bar shape and VX
futures are written in the VIX bar shape, with explicit proxy provenance
columns. Do not treat these files as true index data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)


@dataclass(frozen=True)
class ProxySpec:
    target_symbol: str
    source_dataset: str
    parent_symbol: str
    continuous_symbol: str
    available_start: str | None
    raw_dir: str
    output_dir: str


@dataclass(frozen=True)
class ProxyDownloadRecord:
    date: str
    target_symbol: str
    proxy_symbol: str
    source_dataset: str
    parent_symbol: str
    request_symbol: str
    stype_in: str
    request_start: str
    request_end: str
    cost_estimate_usd: float
    raw_dbn_path: str
    raw_parquet_path: str
    output_parquet_path: str
    raw_rows: int
    output_rows: int


PROXIES = (
    ProxySpec(
        target_symbol="SPX",
        source_dataset="GLBX.MDP3",
        parent_symbol="ES.FUT",
        continuous_symbol="ES.c.0",
        available_start=None,
        raw_dir="databento/glbx_es_ohlcv_1m",
        output_dir="index/spx_1m",
    ),
    ProxySpec(
        target_symbol="VIX",
        source_dataset="XCBF.PITCH",
        parent_symbol="VX.FUT",
        continuous_symbol="VX.c.0",
        available_start="2026-04-01",
        raw_dir="databento/xcbf_vx_ohlcv_1m",
        output_dir="index/vix_1m",
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-01-02")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--max-cost", type=float, default=2.00)
    p.add_argument(
        "--rth-continuous-front",
        action="store_true",
        help=(
            "request only the calendar front continuous contract during the "
            "09:30-16:00 America/New_York option session"
        ),
    )
    p.add_argument(
        "--parquet-only",
        action="store_true",
        help="keep only the Parquet copy and do not materialize DBN.zst",
    )
    p.add_argument(
        "--cost-only",
        action="store_true",
        help="estimate the full requested scope and make no download calls",
    )
    p.add_argument("--cost-out", type=Path, default=None)
    p.add_argument(
        "--cost-ledger",
        type=Path,
        default=None,
        help="reuse a completed cost-only JSON ledger for the matching download scope",
    )
    p.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--audit-out",
        type=Path,
        default=Path("v4/audit/databento_context_proxy_downloads.jsonl"),
    )
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


def _client():
    try:
        import databento as db
    except ImportError:
        print("databento is not installed. Run: .venv/bin/python -m pip install databento", file=sys.stderr)
        raise SystemExit(2)
    return db.Historical()


def _retry(label: str, call: Any, attempts: int = 5) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"{label}: transient failure on attempt {attempt}/{attempts}: "
                f"{exc}; retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def _session_dates(start: str, days: int) -> list[date]:
    first = pd.Timestamp(start).date()
    out: list[date] = []
    cursor = first
    while len(out) < days:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _bounds(session: date, *, rth_only: bool = False) -> tuple[str, str]:
    if rth_only:
        ny = ZoneInfo("America/New_York")
        start = datetime.combine(session, datetime.min.time(), tzinfo=ny).replace(
            hour=9,
            minute=30,
        ).astimezone(timezone.utc)
        end = datetime.combine(session, datetime.min.time(), tzinfo=ny).replace(
            hour=16,
        ).astimezone(timezone.utc)
        return (
            start.isoformat().replace("+00:00", "Z"),
            end.isoformat().replace("+00:00", "Z"),
        )
    start = datetime.combine(session, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _download_parent_ohlcv(
    client: Any,
    *,
    spec: ProxySpec,
    session: date,
    start: str,
    end: str,
    raw_root: Path,
    request_symbol: str,
    stype_in: str,
    parquet_only: bool,
) -> tuple[pd.DataFrame, Path, Path]:
    out_dir = raw_root / spec.raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{session.isoformat()}.{request_symbol.lower().replace('.', '_')}.ohlcv-1m"
    dbn_path = out_dir / f"{stem}.dbn.zst"
    parquet_path = out_dir / f"{stem}.parquet"
    if parquet_path.exists() and (parquet_only or dbn_path.exists()):
        return pd.read_parquet(parquet_path), dbn_path, parquet_path

    kwargs = {
        "dataset": spec.source_dataset,
        "schema": "ohlcv-1m",
        "symbols": request_symbol,
        "stype_in": stype_in,
        "start": start,
        "end": end,
    }
    if not parquet_only:
        kwargs["path"] = dbn_path
    store = _retry(
        f"{session.isoformat()} {request_symbol} download",
        lambda: client.timeseries.get_range(**kwargs),
    )
    frame = store.to_df()
    frame.to_parquet(parquet_path, index=True)
    return frame, dbn_path, parquet_path


def _time_col(frame: pd.DataFrame) -> str:
    for col in ("event_time", "ts_event", "timestamp", "ts_recv"):
        if col in frame.columns:
            return col
    if frame.index.name in {"ts_event", "ts_recv"}:
        frame.reset_index(inplace=True)
        return frame.columns[0]
    raise ValueError(f"could not find time column in {list(frame.columns)}")


def _select_most_liquid(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    working = frame.copy()
    key_col = "symbol" if "symbol" in working.columns else "instrument_id"
    if key_col not in working.columns:
        raise ValueError(f"could not identify futures contract in {list(working.columns)}")
    volumes = working.groupby(key_col)["volume"].sum().sort_values(ascending=False)
    if volumes.empty:
        raise ValueError("no futures volume rows found")
    key = volumes.index[0]
    return working[working[key_col] == key].copy(), str(key)


def _write_proxy_bars(
    frame: pd.DataFrame,
    *,
    spec: ProxySpec,
    session: date,
    proxy_symbol: str,
    raw_root: Path,
) -> Path:
    time_col = _time_col(frame)
    out = pd.DataFrame()
    out["event_time"] = pd.to_datetime(frame[time_col], utc=True)
    out["symbol"] = spec.target_symbol
    for col in ("open", "high", "low", "close", "volume"):
        if col not in frame.columns:
            raise ValueError(f"missing {col} in proxy frame")
        out[col] = pd.to_numeric(frame[col], errors="coerce")
    out["proxy_source"] = f"{spec.source_dataset}:{spec.continuous_symbol}"
    out["proxy_symbol"] = proxy_symbol
    out["is_proxy"] = True
    out = out.dropna(subset=["event_time", "close"]).sort_values("event_time")

    out_dir = raw_root / spec.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session.isoformat()}.proxy_{spec.parent_symbol.lower().replace('.', '_')}.parquet"
    out.to_parquet(out_path, index=False)
    return out_path


def _record(path: Path, record: ProxyDownloadRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file)
    client = _client()
    remaining = args.max_cost
    records: list[ProxyDownloadRecord] = []
    cost_rows: list[dict[str, Any]] = []
    ledger_costs: dict[tuple[str, str, str, str, str], float] = {}
    if args.cost_ledger is not None:
        ledger = json.loads(args.cost_ledger.read_text())
        if ledger.get("mode") != "cost_only" or ledger.get("download_calls") != 0:
            raise SystemExit("abort: --cost-ledger must be a completed cost-only ledger")
        ledger_total = float(ledger.get("estimated_total_usd", float("inf")))
        if ledger_total > args.max_cost:
            raise SystemExit(
                f"abort: ledger total {ledger_total:.4f} exceeds cap {args.max_cost:.4f}"
            )
        for row in ledger.get("requests", []):
            key = (
                str(row["date"]),
                str(row["dataset"]),
                str(row["request_symbol"]),
                str(row["start"]),
                str(row["end"]),
            )
            ledger_costs[key] = float(row["cost_estimate_usd"])

    for session in _session_dates(args.start_date, args.days):
        start, end = _bounds(session, rth_only=args.rth_continuous_front)
        for spec in PROXIES:
            if spec.available_start is not None and session < pd.Timestamp(spec.available_start).date():
                continue
            request_symbol = (
                spec.continuous_symbol
                if args.rth_continuous_front
                else spec.parent_symbol
            )
            stype_in = "continuous" if args.rth_continuous_front else "parent"
            ledger_key = (
                session.isoformat(),
                spec.source_dataset,
                request_symbol,
                start,
                end,
            )
            if args.cost_ledger is not None and ledger_key not in ledger_costs:
                raise SystemExit(
                    f"abort: request missing from cost ledger: {ledger_key}"
                )
            cost = ledger_costs.get(ledger_key)
            if cost is None:
                cost = float(
                    _retry(
                        f"{session.isoformat()} {request_symbol} cost",
                        lambda: client.metadata.get_cost(
                            dataset=spec.source_dataset,
                            schema="ohlcv-1m",
                            symbols=request_symbol,
                            stype_in=stype_in,
                            start=start,
                            end=end,
                        ),
                    )
                )
            cost_rows.append(
                {
                    "date": session.isoformat(),
                    "dataset": spec.source_dataset,
                    "request_symbol": request_symbol,
                    "stype_in": stype_in,
                    "start": start,
                    "end": end,
                    "cost_estimate_usd": cost,
                }
            )
            if args.cost_only:
                continue
            if cost > remaining:
                raise SystemExit(
                    f"abort: {spec.parent_symbol} cost {cost:.4f} exceeds remaining cap {remaining:.4f}"
                )
            require_paid_data_approval(
                manifest_path=args.approval_manifest,
                approval_text=args.approval_text,
                approval_env_var=args.approval_env_var,
                operation=f"Databento context proxy download for {session.isoformat()} {spec.parent_symbol}",
            )
            raw, dbn_path, parquet_path = _download_parent_ohlcv(
                client,
                spec=spec,
                session=session,
                start=start,
                end=end,
                raw_root=args.raw_root,
                request_symbol=request_symbol,
                stype_in=stype_in,
                parquet_only=args.parquet_only,
            )
            if raw.empty:
                print(f"{session}: no {request_symbol} rows returned; skipping closed/unavailable session")
                continue
            selected, proxy_symbol = _select_most_liquid(raw.reset_index())
            out_path = _write_proxy_bars(
                selected,
                spec=spec,
                session=session,
                proxy_symbol=proxy_symbol,
                raw_root=args.raw_root,
            )
            remaining -= cost
            record = ProxyDownloadRecord(
                date=session.isoformat(),
                target_symbol=spec.target_symbol,
                proxy_symbol=proxy_symbol,
                source_dataset=spec.source_dataset,
                parent_symbol=spec.parent_symbol,
                request_symbol=request_symbol,
                stype_in=stype_in,
                request_start=start,
                request_end=end,
                cost_estimate_usd=cost,
                raw_dbn_path=str(dbn_path),
                raw_parquet_path=str(parquet_path),
                output_parquet_path=str(out_path),
                raw_rows=len(raw),
                output_rows=len(selected),
            )
            _record(args.audit_out, record)
            records.append(record)

    if args.cost_only:
        payload = {
            "mode": "cost_only",
            "requests": cost_rows,
            "estimated_total_usd": sum(
                row["cost_estimate_usd"] for row in cost_rows
            ),
            "max_cost_usd": args.max_cost,
            "download_calls": 0,
        }
        if args.cost_out is not None:
            args.cost_out.parent.mkdir(parents=True, exist_ok=True)
            args.cost_out.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["estimated_total_usd"] > args.max_cost:
            raise SystemExit(
                f"abort: estimated total {payload['estimated_total_usd']:.4f} "
                f"exceeds cap {args.max_cost:.4f}"
            )
        return 0

    print(json.dumps([asdict(r) for r in records], indent=2))
    print(f"estimated spend: {args.max_cost - remaining:.4f} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
