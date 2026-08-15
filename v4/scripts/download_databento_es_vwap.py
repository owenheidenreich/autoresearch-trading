"""Download Databento ES 1-minute bars for the v4 VWAP signal test.

This script is intentionally ES-only. It does not download VX/VIX data and it
enforces a hard estimated-cost cap before any paid get_range call.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)


_SESSION_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.pkl$")


@dataclass(frozen=True)
class EsDownloadRecord:
    date: str
    dataset: str
    parent_symbol: str
    schema: str
    cost_estimate_usd: float
    raw_dbn_path: str
    raw_parquet_path: str
    continuous_parquet_path: str
    selected_symbol: str
    raw_rows: int
    continuous_rows: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2025-01-02")
    p.add_argument("--end-date", default="2026-03-31")
    p.add_argument("--max-cost", type=float, default=5.00)
    p.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--processed-dirs",
        nargs="*",
        type=Path,
        default=[
            Path("data/processed/spxw_0dte_neural_q1_2025_nofee"),
            Path("data/processed/spxw_0dte_neural_q2_2025_nofee"),
            Path("data/processed/spxw_0dte_neural_q3_2025_nofee"),
            Path("data/processed/spxw_0dte_neural_q4_2025_nofee"),
            Path("data/processed/spxw_0dte_neural_derived_nofee"),
        ],
        help="Use existing processed SPXW sessions as the approved ES download calendar.",
    )
    p.add_argument(
        "--audit-out",
        type=Path,
        default=Path("v4/audit/databento_es_vwap_downloads.jsonl"),
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


def _client() -> Any:
    try:
        import databento as db
    except ImportError:
        print("databento is not installed. Run with uv or install databento.", file=sys.stderr)
        raise SystemExit(2)
    return db.Historical()


def _sessions_from_processed_dirs(directories: list[Path], start: date, end: date) -> list[date]:
    sessions: set[date] = set()
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.glob("*.pkl"):
            match = _SESSION_RE.match(path.name)
            if not match:
                continue
            session = pd.Timestamp(match.group(1)).date()
            if start <= session <= end:
                sessions.add(session)
    if not sessions:
        raise SystemExit("no processed sessions found for requested ES download window")
    return sorted(sessions)


def _bounds(session: date) -> tuple[str, str]:
    start = datetime.combine(session, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _range_bounds(start_date: date, end_date: date) -> tuple[str, str]:
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _stem(session: date) -> str:
    return f"{session.isoformat()}.es_fut.ohlcv-1m"


def _range_stem(start_date: date, end_date: date) -> str:
    return f"{start_date.isoformat()}_{end_date.isoformat()}.es_fut.ohlcv-1m"


def _time_col(frame: pd.DataFrame) -> str:
    for col in ("event_time", "ts_event", "timestamp", "ts_recv"):
        if col in frame.columns:
            return col
    if frame.index.name in {"ts_event", "ts_recv"}:
        frame.reset_index(inplace=True)
        return frame.columns[0]
    raise ValueError(f"could not find time column in {list(frame.columns)}")


def _select_daily_contract(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    working = frame.copy()
    key_col = "symbol" if "symbol" in working.columns else "instrument_id"
    if key_col not in working.columns:
        raise ValueError(f"could not identify ES contract column in {list(working.columns)}")
    if key_col == "symbol":
        working = working[~working[key_col].astype(str).str.contains("-", regex=False)].copy()
        if working.empty:
            raise ValueError("no outright ES contracts found after excluding spread symbols")
    volume = pd.to_numeric(working["volume"], errors="coerce").fillna(0.0)
    volumes = working.assign(_volume=volume).groupby(key_col)["_volume"].sum().sort_values(ascending=False)
    if volumes.empty:
        raise ValueError("no ES volume rows found")
    key = volumes.index[0]
    selected = working[working[key_col] == key].copy()
    return selected, str(key)


def _write_continuous_session(frame: pd.DataFrame, *, session: date, selected_symbol: str, raw_root: Path) -> Path:
    time_col = _time_col(frame)
    out = pd.DataFrame()
    out["event_time"] = pd.to_datetime(frame[time_col], utc=True)
    out["symbol"] = "ES"
    out["selected_symbol"] = selected_symbol
    for col in ("open", "high", "low", "close", "volume"):
        if col not in frame.columns:
            raise ValueError(f"missing {col} in ES ohlcv frame")
        out[col] = pd.to_numeric(frame[col], errors="coerce")
    out["source_dataset"] = "GLBX.MDP3"
    out["source_parent_symbol"] = "ES.FUT"
    out["vwap_price_basis"] = "typical_price_from_ohlcv"
    out = out.dropna(subset=["event_time", "close"]).sort_values("event_time")

    out_dir = raw_root / "futures/es_ohlcv_1m_continuous"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session.isoformat()}.es_continuous_ohlcv_1m.parquet"
    out.to_parquet(out_path, index=False)
    return out_path


def _download_day(client: Any, *, session: date, raw_root: Path) -> tuple[pd.DataFrame, Path, Path]:
    out_dir = raw_root / "databento/glbx_es_ohlcv_1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(session)
    dbn_path = out_dir / f"{stem}.dbn.zst"
    parquet_path = out_dir / f"{stem}.parquet"
    if dbn_path.exists() and parquet_path.exists():
        return pd.read_parquet(parquet_path), dbn_path, parquet_path

    start, end = _bounds(session)
    store = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="ohlcv-1m",
        symbols="ES.FUT",
        stype_in="parent",
        start=start,
        end=end,
        path=dbn_path,
    )
    frame = store.to_df()
    frame.to_parquet(parquet_path, index=True)
    return frame, dbn_path, parquet_path


def _download_range(
    client: Any,
    *,
    start_date: date,
    end_date: date,
    raw_root: Path,
) -> tuple[pd.DataFrame, Path, Path]:
    out_dir = raw_root / "databento/glbx_es_ohlcv_1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _range_stem(start_date, end_date)
    dbn_path = out_dir / f"{stem}.dbn.zst"
    parquet_path = out_dir / f"{stem}.parquet"
    if dbn_path.exists() and parquet_path.exists():
        return pd.read_parquet(parquet_path), dbn_path, parquet_path

    start, end = _range_bounds(start_date, end_date)
    store = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="ohlcv-1m",
        symbols="ES.FUT",
        stype_in="parent",
        start=start,
        end=end,
        path=dbn_path,
    )
    frame = store.to_df()
    frame.to_parquet(parquet_path, index=True)
    return frame, dbn_path, parquet_path


def _write_continuous_sessions_from_range(
    frame: pd.DataFrame,
    *,
    sessions: list[date],
    raw_root: Path,
) -> list[tuple[date, Path, str, int]]:
    working = frame.reset_index().copy()
    time_col = _time_col(working)
    key_col = "symbol" if "symbol" in working.columns else "instrument_id"
    if key_col not in working.columns:
        raise ValueError(f"could not identify ES contract column in {list(working.columns)}")
    working["event_time"] = pd.to_datetime(working[time_col], utc=True)
    local = working["event_time"].dt.tz_convert("America/New_York")
    working["_session"] = local.dt.date
    minutes = local.dt.hour * 60 + local.dt.minute
    working["_is_rth"] = (minutes >= 9 * 60 + 30) & (minutes <= 16 * 60)
    session_set = set(sessions)
    working = working[working["_session"].isin(session_set)].copy()

    written: list[tuple[date, Path, str, int]] = []
    for session in sessions:
        day = working[working["_session"] == session].copy()
        if day.empty:
            continue
        rth = day[day["_is_rth"]].copy()
        selector_frame = rth if not rth.empty else day
        selected, selected_symbol = _select_daily_contract(selector_frame)
        selected_key = selected[key_col].iloc[0]
        selected_day = day[day[key_col] == selected_key].copy()
        out_path = _write_continuous_session(
            selected_day,
            session=session,
            selected_symbol=str(selected_symbol),
            raw_root=raw_root,
        )
        written.append((session, out_path, str(selected_symbol), int(len(selected_day))))
    return written


def _record(path: Path, record: EsDownloadRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file)
    client = _client()

    start_date = pd.Timestamp(args.start_date).date()
    end_date = pd.Timestamp(args.end_date).date()
    sessions = _sessions_from_processed_dirs(args.processed_dirs, start_date, end_date)

    continuous_dir = args.raw_root / "futures/es_ohlcv_1m_continuous"
    missing_sessions = [
        session
        for session in sessions
        if not (continuous_dir / f"{session.isoformat()}.es_continuous_ohlcv_1m.parquet").exists()
    ]

    range_start, range_end = _range_bounds(start_date, end_date)
    estimated_total = float(
        client.metadata.get_cost(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            symbols="ES.FUT",
            stype_in="parent",
            start=range_start,
            end=range_end,
        )
    )
    if estimated_total > args.max_cost:
        raise SystemExit(
            f"abort: estimated ES cost ${estimated_total:.4f} exceeds hard cap ${args.max_cost:.4f}"
        )

    records: list[EsDownloadRecord] = []
    if missing_sessions:
        require_paid_data_approval(
            manifest_path=args.approval_manifest,
            approval_text=args.approval_text,
            approval_env_var=args.approval_env_var,
            operation=f"Databento ES VWAP range download from {start_date.isoformat()} to {end_date.isoformat()}",
        )
        frame, dbn_path, parquet_path = _download_range(
            client,
            start_date=start_date,
            end_date=end_date,
            raw_root=args.raw_root,
        )
        written = _write_continuous_sessions_from_range(
            frame,
            sessions=sessions,
            raw_root=args.raw_root,
        )
        cost_per_session = estimated_total / max(len(written), 1)
        for session, continuous_path, selected_symbol, continuous_rows in written:
            record = EsDownloadRecord(
                date=session.isoformat(),
                dataset="GLBX.MDP3",
                parent_symbol="ES.FUT",
                schema="ohlcv-1m",
                cost_estimate_usd=float(cost_per_session),
                raw_dbn_path=str(dbn_path),
                raw_parquet_path=str(parquet_path),
                continuous_parquet_path=str(continuous_path),
                selected_symbol=selected_symbol,
                raw_rows=int(len(frame)),
                continuous_rows=int(continuous_rows),
            )
            _record(args.audit_out, record)
            records.append(record)

    summary = {
        "sessions_requested": len(sessions),
        "sessions_missing_before_run": len(missing_sessions),
        "sessions_written": len(records),
        "estimated_range_cost_usd": estimated_total,
        "hard_cap_usd": args.max_cost,
        "continuous_dir": str(continuous_dir),
        "audit_out": str(args.audit_out),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
