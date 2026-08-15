"""Download a small Databento CBBO-1s audit slice for SPXW 0DTE quotes.

The audit universe is selected from the normalized pilot data: for each session,
keep SPXW contracts that were within ATM +/- $50 at any minute according to the
derived underlying context. This gives a cheap high-resolution slice for checking
whether CBBO-1m is acceptable for prototype labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)


DATASET = "OPRA.PILLAR"
DEFAULT_SESSIONS = (
    "2026-01-02",
    "2026-01-16",
    "2026-01-30",
    "2026-02-13",
    "2026-02-27",
    "2026-03-06",
    "2026-03-13",
    "2026-03-20",
    "2026-03-27",
    "2026-03-31",
)


@dataclass(frozen=True)
class AuditDownloadRecord:
    date: str
    schema: str
    symbols: int
    cost_estimate_usd: float
    dbn_path: str
    parquet_path: str
    rows: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="*", default=list(DEFAULT_SESSIONS))
    p.add_argument("--max-cost", type=float, default=5.0)
    p.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    p.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--audit-out",
        type=Path,
        default=Path("v4/audit/databento_cbbo_1s_audit_downloads.jsonl"),
    )
    p.add_argument("--ladder-dollars", type=float, default=50.0)
    p.add_argument("--strike-step", type=float, default=5.0)
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
        print("databento is not installed. Run: .venv/bin/python -m pip install databento", file=sys.stderr)
        raise SystemExit(2)
    return db.Historical()


def _bounds(session: str) -> tuple[str, str]:
    session_date = pd.Timestamp(session).date()
    start = datetime.combine(session_date, time(0, 0), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _atm(value: pd.Series, step: float) -> pd.Series:
    return (value.astype(float) / step).round() * step


def _audit_symbols(
    normalized_path: Path,
    *,
    ladder_dollars: float,
    strike_step: float,
) -> list[str]:
    frame = pd.read_parquet(
        normalized_path,
        columns=["raw_symbol", "strike", "underlying_price", "root", "settlement_style"],
    )
    frame = frame[
        (frame["root"] == "SPXW")
        & (frame["settlement_style"] == "PM")
        & frame["underlying_price"].notna()
    ].copy()
    frame["atm_strike"] = _atm(frame["underlying_price"], strike_step)
    frame["strike_float"] = frame["strike"].astype(float)
    frame = frame[(frame["strike_float"] - frame["atm_strike"]).abs() <= ladder_dollars]
    return sorted(frame["raw_symbol"].dropna().astype(str).unique().tolist())


def _estimate(
    client: Any,
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
) -> float:
    return float(
        client.metadata.get_cost(
            dataset=DATASET,
            schema="cbbo-1s",
            symbols=list(symbols),
            stype_in="raw_symbol",
            start=start,
            end=end,
        )
    )


def _download(
    client: Any,
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    out_dir: Path,
    stem: str,
) -> tuple[pd.DataFrame, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dbn_path = out_dir / f"{stem}.cbbo-1s.dbn.zst"
    parquet_path = out_dir / f"{stem}.cbbo-1s.parquet"
    store = client.timeseries.get_range(
        dataset=DATASET,
        schema="cbbo-1s",
        symbols=list(symbols),
        stype_in="raw_symbol",
        start=start,
        end=end,
        path=dbn_path,
    )
    frame = store.to_df()
    frame.to_parquet(parquet_path, index=True)
    return frame, dbn_path, parquet_path


def _existing(out_dir: Path, stem: str) -> tuple[pd.DataFrame, Path, Path] | None:
    dbn_path = out_dir / f"{stem}.cbbo-1s.dbn.zst"
    parquet_path = out_dir / f"{stem}.cbbo-1s.parquet"
    if dbn_path.exists() and parquet_path.exists():
        return pd.read_parquet(parquet_path), dbn_path, parquet_path
    return None


def _record(audit_out: Path, record: AuditDownloadRecord) -> None:
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    with audit_out.open("a") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file)
    client = _client()
    out_dir = args.raw_root / "audit" / "opra_spxw_cbbo_1s"
    remaining = args.max_cost
    records: list[AuditDownloadRecord] = []

    for session in args.sessions:
        stem = str(session)
        normalized_path = args.normalized_dir / f"databento_spxw_0dte_{stem}_derived_context.parquet"
        if not normalized_path.exists():
            raise SystemExit(f"missing normalized derived file: {normalized_path}")
        symbols = _audit_symbols(
            normalized_path,
            ladder_dollars=args.ladder_dollars,
            strike_step=args.strike_step,
        )
        if not symbols:
            raise SystemExit(f"{stem}: no audit symbols selected")

        existing = _existing(out_dir, stem)
        if existing is not None:
            frame, dbn_path, parquet_path = existing
            print(f"{stem}: reusing existing cbbo-1s file ({len(frame)} rows)")
            continue

        start, end = _bounds(stem)
        cost = _estimate(client, symbols=symbols, start=start, end=end)
        if cost > remaining:
            raise SystemExit(f"abort: {stem} cost {cost:.4f} exceeds remaining cap {remaining:.4f}")
        require_paid_data_approval(
            manifest_path=args.approval_manifest,
            approval_text=args.approval_text,
            approval_env_var=args.approval_env_var,
            operation=f"Databento CBBO-1s audit download for {stem}",
        )
        frame, dbn_path, parquet_path = _download(
            client,
            symbols=symbols,
            start=start,
            end=end,
            out_dir=out_dir,
            stem=stem,
        )
        remaining -= cost
        record = AuditDownloadRecord(
            date=stem,
            schema="cbbo-1s",
            symbols=len(symbols),
            cost_estimate_usd=cost,
            dbn_path=str(dbn_path),
            parquet_path=str(parquet_path),
            rows=int(len(frame)),
        )
        _record(args.audit_out, record)
        records.append(record)
        print(json.dumps(asdict(record), sort_keys=True), flush=True)

    print(json.dumps([asdict(r) for r in records], indent=2))
    print(f"estimated spend: {args.max_cost - remaining:.4f} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
