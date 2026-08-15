"""Download a tiny Databento SPXW 0DTE pilot sample.

The script estimates cost before each paid request and aborts if the projected
spend exceeds ``--max-cost``. It writes vendor DBN files plus lightweight
Parquet copies for local inspection.
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
from typing import Any, Sequence

import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)
from v4.ingest.databento_opra import filter_0dte_definitions


DATASET = "OPRA.PILLAR"
PARENT_SYMBOL = "SPXW.OPT"


@dataclass(frozen=True)
class DownloadRecord:
    date: str
    schema: str
    symbols: int | str
    cost_estimate_usd: float
    dbn_path: str
    parquet_path: str
    rows: int
    status: str = "success"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2026-01-02")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--max-cost", type=float, default=5.00)
    p.add_argument("--include-ohlcv", action="store_true")
    p.add_argument("--include-statistics", action="store_true")
    p.add_argument(
        "--include-cbbo-1s",
        action="store_true",
        help="also download full-session CBBO-1s for the resolved SPXW PM 0DTE symbols",
    )
    p.add_argument(
        "--parquet-only",
        action="store_true",
        help="keep only the Parquet copy and do not materialize a DBN.zst file",
    )
    p.add_argument(
        "--end-utc",
        default=None,
        help=(
            "Optional UTC end timestamp override for every requested session, "
            "for same-day post-close downloads before the full UTC day is available."
        ),
    )
    p.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--audit-out", type=Path, default=Path("v4/audit/databento_pilot_downloads.jsonl"))
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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _client():
    try:
        import databento as db
    except ImportError:
        print("databento is not installed. Run: .venv/bin/python -m pip install databento", file=sys.stderr)
        raise SystemExit(2)
    return db.Historical()


def _session_dates(start: str, days: int) -> list[date]:
    first = pd.Timestamp(start).date()
    out: list[date] = []
    cursor = first
    while len(out) < days:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _bounds(session: date, *, end_utc: str | None = None) -> tuple[str, str]:
    start = datetime.combine(session, datetime.min.time(), tzinfo=timezone.utc)
    end = pd.Timestamp(end_utc).to_pydatetime() if end_utc else start + timedelta(days=1)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)
    if end <= start:
        raise ValueError(f"--end-utc must be after {start.isoformat()}, got {end.isoformat()}")
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _estimate(
    client: Any,
    *,
    schema: str,
    symbols: Sequence[str] | str,
    stype_in: str,
    start: str,
    end: str,
) -> float:
    return float(
        client.metadata.get_cost(
            dataset=DATASET,
            schema=schema,
            symbols=symbols,
            stype_in=stype_in,
            start=start,
            end=end,
        )
    )


def _is_unresolved_symbol_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "None of the symbols could be resolved" in text
        or "symbology_invalid_request" in text
    )


def _is_transient_download_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "504",
            "gateway timed out",
            "remote gateway timed out",
            "temporarily unavailable",
            "connection reset",
            "read timed out",
            "response ended prematurely",
            "500",
            "502",
            "503",
        )
    )


def _existing_audit_cost(audit_out: Path) -> float:
    if not audit_out.exists():
        return 0.0
    total = 0.0
    for line in audit_out.read_text().splitlines():
        if not line.strip():
            continue
        try:
            total += float(json.loads(line).get("cost_estimate_usd", 0.0) or 0.0)
        except json.JSONDecodeError:
            continue
    return total


def _download(
    client: Any,
    *,
    schema: str,
    symbols: Sequence[str] | str,
    stype_in: str,
    stype_out: str | None,
    start: str,
    end: str,
    out_dir: Path,
    stem: str,
    parquet_only: bool = False,
) -> tuple[pd.DataFrame, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dbn_path = out_dir / f"{stem}.{schema}.dbn.zst"
    parquet_path = out_dir / f"{stem}.{schema}.parquet"
    kwargs = {
        "dataset": DATASET,
        "schema": schema,
        "symbols": symbols,
        "stype_in": stype_in,
        "start": start,
        "end": end,
    }
    if not parquet_only:
        kwargs["path"] = dbn_path
    if stype_out is not None:
        kwargs["stype_out"] = stype_out
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            store = client.timeseries.get_range(**kwargs)
            frame = store.to_df()
            frame.to_parquet(parquet_path, index=True)
            return frame, dbn_path, parquet_path
        except Exception as exc:
            if attempt >= max_attempts or not _is_transient_download_error(exc):
                raise
            dbn_path.unlink(missing_ok=True)
            parquet_path.unlink(missing_ok=True)
            print(
                f"{stem}: transient {schema} download error, retrying attempt "
                f"{attempt + 1}/{max_attempts}: {exc}"
            )
            time.sleep(2 * attempt)
    raise RuntimeError(f"unreachable retry state for {stem} {schema}")



def _maybe_existing(
    out_dir: Path,
    stem: str,
    schema: str,
    *,
    parquet_only: bool = False,
) -> tuple[pd.DataFrame, Path, Path] | None:
    dbn_path = out_dir / f"{stem}.{schema}.dbn.zst"
    parquet_path = out_dir / f"{stem}.{schema}.parquet"
    if parquet_path.exists() and (parquet_only or dbn_path.exists()):
        return pd.read_parquet(parquet_path), dbn_path, parquet_path
    return None


def _record(
    audit_out: Path,
    *,
    session: date,
    schema: str,
    symbols: Sequence[str] | str,
    cost: float,
    dbn_path: Path,
    parquet_path: Path,
    rows: int,
) -> DownloadRecord:
    rec = DownloadRecord(
        date=session.isoformat(),
        schema=schema,
        symbols=len(symbols) if not isinstance(symbols, str) else symbols,
        cost_estimate_usd=cost,
        dbn_path=str(dbn_path),
        parquet_path=str(parquet_path),
        rows=rows,
    )
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    with audit_out.open("a") as f:
        f.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
    return rec


def _record_failure(
    audit_out: Path,
    *,
    session: date,
    schema: str,
    symbols: Sequence[str] | str,
    exc: Exception,
) -> None:
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "date": session.isoformat(),
        "schema": schema,
        "symbols": len(symbols) if not isinstance(symbols, str) else symbols,
        "cost_estimate_usd": 0.0,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "live_license_blocked": "live data license is required" in str(exc).lower(),
    }
    with audit_out.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file)
    client = _client()
    prior_estimated_spend = _existing_audit_cost(args.audit_out)
    remaining = args.max_cost - prior_estimated_spend
    if remaining <= 0:
        raise SystemExit(
            f"abort: prior audit spend {prior_estimated_spend:.4f} meets/exceeds cap {args.max_cost:.4f}"
        )
    if prior_estimated_spend:
        print(
            f"resuming with prior estimated spend {prior_estimated_spend:.4f} USD; "
            f"remaining approved cap {remaining:.4f} USD"
        )
    records: list[DownloadRecord] = []

    for session in _session_dates(args.start_date, args.days):
        start, end = _bounds(session, end_utc=args.end_utc)
        stem = session.isoformat()

        definition_dir = args.raw_root / "databento" / "opra_spxw_definition"
        existing = _maybe_existing(
            definition_dir,
            stem,
            "definition",
            parquet_only=args.parquet_only,
        )
        if existing is not None:
            defs, dbn_path, parquet_path = existing
            print(f"{session}: reusing existing definition file")
        else:
            try:
                definition_cost = _estimate(
                    client,
                    schema="definition",
                    symbols=PARENT_SYMBOL,
                    stype_in="parent",
                    start=start,
                    end=end,
                )
            except Exception as e:
                if _is_unresolved_symbol_error(e):
                    print(f"{session}: skipping unresolved/closed session")
                    continue
                raise
            if definition_cost > remaining:
                raise SystemExit(
                    f"abort: definition cost {definition_cost:.4f} exceeds remaining cap {remaining:.4f}"
                )
            require_paid_data_approval(
                manifest_path=args.approval_manifest,
                approval_text=args.approval_text,
                approval_env_var=args.approval_env_var,
                operation=f"Databento {DATASET} definition download for {session}",
            )
            try:
                defs, dbn_path, parquet_path = _download(
                    client,
                    schema="definition",
                    symbols=PARENT_SYMBOL,
                    stype_in="parent",
                    stype_out=None,
                    start=start,
                    end=end,
                    out_dir=definition_dir,
                    stem=stem,
                    parquet_only=args.parquet_only,
                )
            except Exception as exc:
                _record_failure(
                    args.audit_out,
                    session=session,
                    schema="definition",
                    symbols=PARENT_SYMBOL,
                    exc=exc,
                )
                raise
            remaining -= definition_cost
            records.append(
                _record(
                    args.audit_out,
                    session=session,
                    schema="definition",
                    symbols=PARENT_SYMBOL,
                    cost=definition_cost,
                    dbn_path=dbn_path,
                    parquet_path=parquet_path,
                    rows=len(defs),
                )
            )

        filtered_defs = filter_0dte_definitions(defs.reset_index(), session)
        raw_symbols = filtered_defs["raw_symbol"].drop_duplicates().tolist()
        if not raw_symbols:
            print(f"{session}: no SPXW 0DTE symbols found after filtering")
            continue

        for schema, out_name, enabled in (
            ("cbbo-1m", "opra_spxw_cbbo_1m", True),
            ("ohlcv-1m", "opra_spxw_ohlcv_1m", args.include_ohlcv),
            ("statistics", "opra_spxw_statistics", args.include_statistics),
            ("cbbo-1s", "opra_spxw_cbbo_1s", args.include_cbbo_1s),
        ):
            if not enabled:
                continue
            out_dir = args.raw_root / "databento" / out_name
            existing = _maybe_existing(
                out_dir,
                stem,
                schema,
                parquet_only=args.parquet_only,
            )
            if existing is not None:
                frame, _, _ = existing
                print(f"{session}: reusing existing {schema} file ({len(frame)} rows)")
                continue
            cost = _estimate(
                client,
                schema=schema,
                symbols=raw_symbols,
                stype_in="raw_symbol",
                start=start,
                end=end,
            )
            if cost > remaining:
                raise SystemExit(
                    f"abort: {schema} cost {cost:.4f} exceeds remaining cap {remaining:.4f}"
                )
            require_paid_data_approval(
                manifest_path=args.approval_manifest,
                approval_text=args.approval_text,
                approval_env_var=args.approval_env_var,
                operation=f"Databento {DATASET} {schema} download for {session}",
            )
            try:
                frame, dbn_path, parquet_path = _download(
                    client,
                    schema=schema,
                    symbols=raw_symbols,
                    stype_in="raw_symbol",
                    stype_out=None,
                    start=start,
                    end=end,
                    out_dir=out_dir,
                    stem=stem,
                    parquet_only=args.parquet_only,
                )
            except Exception as exc:
                _record_failure(
                    args.audit_out,
                    session=session,
                    schema=schema,
                    symbols=raw_symbols,
                    exc=exc,
                )
                raise
            remaining -= cost
            records.append(
                _record(
                    args.audit_out,
                    session=session,
                    schema=schema,
                    symbols=raw_symbols,
                    cost=cost,
                    dbn_path=dbn_path,
                    parquet_path=parquet_path,
                    rows=len(frame),
                )
            )

    print(json.dumps([asdict(r) for r in records], indent=2))
    print(f"estimated spend this run: {sum(r.cost_estimate_usd for r in records):.4f} USD")
    print(f"estimated spend cumulative: {args.max_cost - remaining:.4f} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
