"""Rebuild v4 neural decision rows from existing normalized/context files.

This is useful when label policy code changes but the OPRA normalization does
not. It does not download data and does not re-normalize raw OPRA files.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from v4.dataset.spxw_0dte_neural import build_neural_dataset
from v4.ingest.index_bars import load_spx_1m, load_vix_1m


@dataclass(frozen=True)
class RebuildRecord:
    session: str
    normalized_rows: int
    spx_rows: int
    vix_rows: int
    neural_rows: int
    normalized_path: str
    spx_path: str
    vix_path: str
    neural_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized"))
    parser.add_argument("--spx-dir", type=Path, default=Path("data/raw/index/spx_1m"))
    parser.add_argument("--vix-dir", type=Path, default=Path("data/raw/index/vix_1m"))
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--context-suffix", default="derived_context", choices=("derived_context", "official_context"))
    parser.add_argument("--summary-out", type=Path, required=True)
    return parser.parse_args()


def _sessions(start: str, end: str) -> list[str]:
    cursor = pd.Timestamp(start).date()
    final = pd.Timestamp(end).date()
    out: list[str] = []
    while cursor <= final:
        if cursor.weekday() < 5:
            out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def _first_existing(directory: Path, session: str, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        path = directory / pattern.format(session=session)
        if path.exists():
            return path
    for path in sorted(directory.glob(f"{session}.*")):
        if path.exists() and path.is_file():
            return path
    return None


def _build_session(args: argparse.Namespace, session: str) -> RebuildRecord | None:
    normalized_path = (
        args.normalized_dir / f"databento_spxw_0dte_{session}_{args.context_suffix}.parquet"
    )
    if not normalized_path.exists():
        return None
    spx_path = _first_existing(
        args.spx_dir,
        session,
        [
            "{session}.official_spx.parquet",
            "{session}.derived_spxw_parity.parquet",
            "{session}.parquet",
            "{session}.csv",
            "{session}.jsonl",
        ],
    )
    vix_path = _first_existing(
        args.vix_dir,
        session,
        [
            "{session}.official_vix.parquet",
            "{session}.derived_spxw_atm_iv.parquet",
            "{session}.parquet",
            "{session}.csv",
            "{session}.jsonl",
        ],
    )
    if spx_path is None or vix_path is None:
        return None
    normalized = pq.read_table(normalized_path)
    spx = load_spx_1m(spx_path)
    vix = load_vix_1m(vix_path)
    rows = build_neural_dataset(normalized, spx, vix)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    neural_path = args.processed_dir / f"{session}.pkl"
    with neural_path.open("wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return RebuildRecord(
        session=session,
        normalized_rows=int(normalized.num_rows),
        spx_rows=int(len(spx)),
        vix_rows=int(len(vix)),
        neural_rows=int(len(rows)),
        normalized_path=str(normalized_path),
        spx_path=str(spx_path),
        vix_path=str(vix_path),
        neural_path=str(neural_path),
    )


def main() -> int:
    args = parse_args()
    records: list[RebuildRecord] = []
    skipped: list[str] = []
    for session in _sessions(args.start_date, args.end_date):
        record = _build_session(args, session)
        if record is None:
            skipped.append(session)
            print(f"{session}: skipped missing normalized/context files", flush=True)
            continue
        records.append(record)
        print(json.dumps({"session": session, "neural_rows": record.neural_rows}), flush=True)
    payload = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "processed_dir": str(args.processed_dir),
        "context_suffix": args.context_suffix,
        "sessions_built": len(records),
        "sessions_skipped": skipped,
        "totals": {
            "normalized_rows": sum(record.normalized_rows for record in records),
            "spx_rows": sum(record.spx_rows for record in records),
            "vix_rows": sum(record.vix_rows for record in records),
            "neural_rows": sum(record.neural_rows for record in records),
        },
        "records": [asdict(record) for record in records],
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.summary_out)
    print(json.dumps(payload["totals"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
