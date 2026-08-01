"""Six-session paired-quote latency bound harness; offline local files only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from .simulated import VirtualMonotonicClock


OWNED_PAIRED_SESSIONS = (
    "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-10", "2026-07-13", "2026-07-14",
)
LATENCY_RUNGS_MS = (100, 250, 500, 1000)


@dataclass(frozen=True)
class LatencyBoundRow:
    session_date: str
    osi_symbol: str
    decision_second_utc: int
    latency_ms: int
    virtual_monotonic_ns: int
    virtual_arrival_timestamp_utc: str
    databento_decision_bid_micros: int
    databento_decision_ask_micros: int
    ibkr_arrival_bid_micros: int
    ibkr_arrival_ask_micros: int
    sell_limit_micros: int
    marketable_at_observed_arrival: bool
    fill_quantity_lower_bound: int
    fill_quantity_upper_bound: int
    fill_bound_reason: str
    source_scope: str


def run_latency_bounds(repo_root: Path) -> list[LatencyBoundRow]:
    rows: list[LatencyBoundRow] = []
    for session in OWNED_PAIRED_SESSIONS:
        sample = _paired_consecutive_sample(repo_root, session)
        base = sample.iloc[0]
        next_row = sample.iloc[1]
        for latency_ms in LATENCY_RUNGS_MS:
            arrival = next_row if latency_ms >= 1000 else base
            decision_wall = datetime.fromisoformat(session).replace(tzinfo=timezone.utc) + timedelta(seconds=int(base.second))
            clock = VirtualMonotonicClock(decision_wall)
            clock.advance_ms(latency_ms)
            db_bid = _micros(base.bid_px_00)
            db_ask = _micros(base.ask_px_00)
            ib_bid = _micros(arrival.bid)
            ib_ask = _micros(arrival.ask)
            limit = max(0, db_bid - 100_000)
            marketable = ib_bid >= limit
            rows.append(
                LatencyBoundRow(
                    session_date=session,
                    osi_symbol=str(base.symbol),
                    decision_second_utc=int(base.second),
                    latency_ms=latency_ms,
                    virtual_monotonic_ns=clock.monotonic_ns,
                    virtual_arrival_timestamp_utc=clock.iso(),
                    databento_decision_bid_micros=db_bid,
                    databento_decision_ask_micros=db_ask,
                    ibkr_arrival_bid_micros=ib_bid,
                    ibkr_arrival_ask_micros=ib_ask,
                    sell_limit_micros=limit,
                    marketable_at_observed_arrival=marketable,
                    fill_quantity_lower_bound=0,
                    fill_quantity_upper_bound=1 if marketable else 0,
                    fill_bound_reason=(
                        "BBO_SHOWS_MARKETABLE_BUT_CANNOT_PROVE_TRADE_OR_QUEUE_FILL"
                        if marketable else "OBSERVED_BBO_IS_NOT_MARKETABLE_AT_LIMIT"
                    ),
                    source_scope="OWNED_DATABENTO_CBBO_1S_PLUS_OWNED_IBKR_LAST_BBO_1S",
                )
            )
    return rows


def write_latency_report(rows: Iterable[LatencyBoundRow], out_dir: Path) -> None:
    materialized = list(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in materialized])
    frame.to_csv(out_dir / "latency_bounds.csv", index=False)
    (out_dir / "latency_bounds.json").write_text(
        json.dumps([asdict(row) for row in materialized], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = frame.groupby("latency_ms", as_index=False).agg(
        sessions=("session_date", "nunique"),
        marketable_upper_bound_count=("marketable_at_observed_arrival", "sum"),
        fill_lower_bound_total=("fill_quantity_lower_bound", "sum"),
        fill_upper_bound_total=("fill_quantity_upper_bound", "sum"),
    )
    lines = [
        "# Path-D offline latency bounds", "",
        "These are quote-supported bounds, not actual-fill observations. A marketable BBO sets an upper bound of one contract and a lower bound of zero because one-second quotes contain neither trades nor queue position.", "",
        _markdown_table(summary), "",
        "## Per-session table", "", _markdown_table(frame), "",
    ]
    (out_dir / "latency_bounds.md").write_text("\n".join(lines), encoding="utf-8")


def owned_input_manifest(repo_root: Path) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for session in OWNED_PAIRED_SESSIONS:
        inputs = {
            "databento_cbbo_1s": repo_root / "v4/raw/opra_1s_parity_probe" / f"parity_probe_{session}.cbbo-1s.parquet",
            "ibkr_last_bbo_1s": repo_root / "v4/audit/autoresearch/protocol101_ws2_parity_probe/derived" / f"ibkr_last_bbo_1s_{session}.parquet",
        }
        record: dict[str, object] = {"session_date": session}
        for name, path in inputs.items():
            record[name] = {
                "path": str(path.relative_to(repo_root)),
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "sha256": file_sha256(path),
            }
        manifest.append(record)
    return manifest


def _paired_consecutive_sample(repo_root: Path, session: str) -> pd.DataFrame:
    ibkr_path = repo_root / "v4/audit/autoresearch/protocol101_ws2_parity_probe/derived" / f"ibkr_last_bbo_1s_{session}.parquet"
    databento_path = repo_root / "v4/raw/opra_1s_parity_probe" / f"parity_probe_{session}.cbbo-1s.parquet"
    if not ibkr_path.is_file() or not databento_path.is_file():
        raise FileNotFoundError(f"missing owned paired inputs for {session}")
    ibkr = pq.read_table(ibkr_path, columns=["symbol", "second", "bid", "ask", "clean"]).to_pandas()
    ibkr = ibkr[ibkr["clean"] & (ibkr["bid"] >= 0) & (ibkr["ask"] >= ibkr["bid"])]
    counts = ibkr.groupby("symbol").size().sort_values(ascending=False)
    for symbol in counts.head(32).index:
        databento = pq.read_table(
            databento_path,
            columns=["symbol", "ts_recv", "bid_px_00", "ask_px_00"],
            filters=[("symbol", "=", str(symbol))],
        ).to_pandas().reset_index()
        if databento.empty:
            continue
        received = pd.to_datetime(databento["ts_recv"], utc=True) - pd.Timedelta(seconds=1)
        databento["second"] = received.dt.hour * 3600 + received.dt.minute * 60 + received.dt.second
        paired = ibkr[ibkr["symbol"] == symbol].merge(
            databento[["symbol", "second", "bid_px_00", "ask_px_00"]],
            on=["symbol", "second"],
            how="inner",
        )
        paired = paired[
            (paired["bid_px_00"] >= 0)
            & (paired["ask_px_00"] >= paired["bid_px_00"])
        ].sort_values("second").drop_duplicates("second")
        seconds = paired["second"].to_numpy()
        consecutive = (seconds[1:] - seconds[:-1]) == 1
        if consecutive.any():
            index = int(consecutive.argmax())
            return paired.iloc[index : index + 2].reset_index(drop=True)
    raise ValueError(f"no consecutive clean paired quote sample for {session}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _micros(value: float) -> int:
    return int(round(float(value) * 1_000_000))


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)
