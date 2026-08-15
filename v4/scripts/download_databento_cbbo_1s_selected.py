"""Download targeted Databento CBBO-1s for Protocol 039 selected trades.

This is the narrow Protocol 041 audit downloader. It downloads only exact raw
symbols selected by Protocol 039 on pre-registered target sessions, not an
ATM ladder or full chain.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)


DATASET = "OPRA.PILLAR"
SCHEMA = "cbbo-1s"
TARGET_SESSIONS = (
    "2025-04-14",
    "2025-04-21",
    "2025-04-23",
    "2025-04-28",
    "2025-05-01",
    "2025-05-02",
    "2025-05-05",
    "2025-05-06",
    "2025-05-09",
    "2025-05-13",
    "2025-05-16",
    "2025-05-22",
    "2025-05-27",
    "2025-05-30",
    "2025-10-17",
    "2025-11-06",
    "2025-11-13",
    "2025-11-18",
    "2025-11-21",
    "2025-12-12",
    "2025-12-16",
    "2025-12-17",
    "2026-03-02",
    "2026-03-05",
    "2026-03-11",
    "2026-03-23",
    "2026-03-25",
)


@dataclass(frozen=True)
class SelectedDownloadPlan:
    session: str
    symbols: int
    selected_trades: int
    estimate_usd: float
    existing: bool
    parquet_path: str


@dataclass(frozen=True)
class SelectedDownloadRecord:
    session: str
    dataset: str
    schema: str
    symbols: int
    selected_trades: int
    cost_estimate_usd: float
    dbn_path: str
    parquet_path: str
    rows: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    p.add_argument(
        "--selected-trades-dir",
        type=Path,
        default=Path("v4/audit/autoresearch/v4_aplus_hypothesis_039_broader_baseline_validation/selected_trades"),
    )
    p.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--sessions", nargs="*", default=list(TARGET_SESSIONS))
    p.add_argument("--max-cost", type=float, default=2.00)
    p.add_argument(
        "--audit-out",
        type=Path,
        default=Path("v4/audit/databento_cbbo_1s_selected_downloads.jsonl"),
    )
    p.add_argument("--dry-run", action="store_true")
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


def _client() -> Any:
    try:
        import databento as db
    except ImportError:
        print("databento is not installed. Run with uv or install databento.", file=sys.stderr)
        raise SystemExit(2)
    return db.Historical()


def _bounds(session: str) -> tuple[str, str]:
    session_date = pd.Timestamp(session).date()
    start = datetime.combine(session_date, time(0, 0), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _selected_trade_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise SystemExit(f"missing selected trades directory: {directory}")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise SystemExit(f"no selected trade JSON files found in {directory}")
    return paths


def _load_selected(paths: Iterable[Path], sessions: set[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        seed_text = path.stem.rsplit("_seed", 1)[-1]
        try:
            seed = int(seed_text)
        except ValueError:
            seed = -1
        data = json.loads(path.read_text())
        for row in data:
            if str(row.get("session")) in sessions:
                rows.append({"source_file": str(path), "seed_from_file": seed, **row})
    return rows


def _raw_symbol_map(normalized_dir: Path, session: str) -> dict[str, str]:
    path = normalized_dir / f"databento_spxw_0dte_{session}_derived_context.parquet"
    if not path.exists():
        raise SystemExit(f"missing normalized symbol map for {session}: {path}")
    frame = pd.read_parquet(path, columns=["contract_id", "raw_symbol", "root", "settlement_style"])
    frame = frame[
        (frame["root"].astype(str) == "SPXW")
        & (frame["settlement_style"].astype(str) == "PM")
    ].dropna(subset=["contract_id", "raw_symbol"])
    frame = frame.drop_duplicates("contract_id")
    return dict(zip(frame["contract_id"].astype(str), frame["raw_symbol"].astype(str)))


def _session_symbols(rows: list[dict], normalized_dir: Path) -> dict[str, list[str]]:
    by_session: dict[str, list[dict]] = {}
    for row in rows:
        by_session.setdefault(str(row["session"]), []).append(row)
    out: dict[str, list[str]] = {}
    missing = []
    for session, session_rows in sorted(by_session.items()):
        mapping = _raw_symbol_map(normalized_dir, session)
        symbols = set()
        for row in session_rows:
            contract_id = str(row.get("contract_id"))
            raw_symbol = mapping.get(contract_id)
            if raw_symbol is None:
                missing.append((session, contract_id))
            else:
                symbols.add(raw_symbol)
        if symbols:
            out[session] = sorted(symbols)
    if missing:
        preview = ", ".join(f"{session}:{contract}" for session, contract in missing[:5])
        raise SystemExit(f"missing raw symbol mappings for {len(missing)} selected trades: {preview}")
    return out


def _out_paths(raw_root: Path, session: str) -> tuple[Path, Path]:
    out_dir = raw_root / "audit" / "opra_spxw_cbbo_1s"
    return out_dir / f"{session}.cbbo-1s.dbn.zst", out_dir / f"{session}.cbbo-1s.parquet"


def _estimate(client: Any, *, symbols: Sequence[str], start: str, end: str) -> float:
    return float(
        client.metadata.get_cost(
            dataset=DATASET,
            schema=SCHEMA,
            symbols=list(symbols),
            stype_in="raw_symbol",
            start=start,
            end=end,
        )
    )


def _plan(
    client: Any,
    *,
    rows: list[dict],
    symbols_by_session: dict[str, list[str]],
    raw_root: Path,
) -> list[SelectedDownloadPlan]:
    trade_counts = pd.Series([row["session"] for row in rows], dtype="object").value_counts().to_dict()
    plans: list[SelectedDownloadPlan] = []
    for session, symbols in sorted(symbols_by_session.items()):
        dbn_path, parquet_path = _out_paths(raw_root, session)
        existing = dbn_path.exists() and parquet_path.exists()
        estimate = 0.0
        if not existing:
            start, end = _bounds(session)
            estimate = _estimate(client, symbols=symbols, start=start, end=end)
        plans.append(
            SelectedDownloadPlan(
                session=session,
                symbols=len(symbols),
                selected_trades=int(trade_counts.get(session, 0)),
                estimate_usd=estimate,
                existing=existing,
                parquet_path=str(parquet_path),
            )
        )
    return plans


def _download(
    client: Any,
    *,
    session: str,
    symbols: Sequence[str],
    raw_root: Path,
    overwrite: bool,
) -> tuple[pd.DataFrame, Path, Path]:
    dbn_path, parquet_path = _out_paths(raw_root, session)
    dbn_path.parent.mkdir(parents=True, exist_ok=True)
    if dbn_path.exists() and parquet_path.exists() and not overwrite:
        return pd.read_parquet(parquet_path), dbn_path, parquet_path
    start, end = _bounds(session)
    store = client.timeseries.get_range(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=list(symbols),
        stype_in="raw_symbol",
        start=start,
        end=end,
        path=dbn_path,
    )
    frame = store.to_df()
    frame.to_parquet(parquet_path, index=True)
    return frame, dbn_path, parquet_path


def _record(path: Path, record: SelectedDownloadRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file)
    sessions = set(args.sessions)
    rows = _load_selected(_selected_trade_files(args.selected_trades_dir), sessions)
    if not rows:
        raise SystemExit("no selected trades found for requested sessions")
    symbols_by_session = _session_symbols(rows, args.normalized_dir)
    client = _client()
    plans = _plan(client, rows=rows, symbols_by_session=symbols_by_session, raw_root=args.raw_root)
    estimated_total = sum(plan.estimate_usd for plan in plans)
    payload = {
        "dataset": DATASET,
        "schema": SCHEMA,
        "stype_in": "raw_symbol",
        "sessions": len(plans),
        "selected_trades": len(rows),
        "unique_symbols_total": sum(plan.symbols for plan in plans),
        "estimated_total_usd": estimated_total,
        "max_cost_usd": args.max_cost,
        "plans": [asdict(plan) for plan in plans],
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if estimated_total > args.max_cost:
        raise SystemExit(f"abort: estimated cost {estimated_total:.4f} exceeds cap {args.max_cost:.4f}")
    if args.dry_run:
        return 0

    records: list[SelectedDownloadRecord] = []
    for plan in plans:
        if plan.existing and not args.overwrite:
            continue
        symbols = symbols_by_session[plan.session]
        require_paid_data_approval(
            manifest_path=args.approval_manifest,
            approval_text=args.approval_text,
            approval_env_var=args.approval_env_var,
            operation=f"Databento {DATASET} {SCHEMA} selected-symbol download for {plan.session}",
        )
        frame, dbn_path, parquet_path = _download(
            client,
            session=plan.session,
            symbols=symbols,
            raw_root=args.raw_root,
            overwrite=args.overwrite,
        )
        record = SelectedDownloadRecord(
            session=plan.session,
            dataset=DATASET,
            schema=SCHEMA,
            symbols=len(symbols),
            selected_trades=plan.selected_trades,
            cost_estimate_usd=plan.estimate_usd,
            dbn_path=str(dbn_path),
            parquet_path=str(parquet_path),
            rows=int(len(frame)),
        )
        _record(args.audit_out, record)
        records.append(record)
        print(json.dumps(asdict(record), sort_keys=True), flush=True)
    print(json.dumps([asdict(record) for record in records], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
