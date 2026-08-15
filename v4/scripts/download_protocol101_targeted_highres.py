"""Download the approved Protocol 101 targeted high-resolution OPRA batch.

This downloader is intentionally manifest-driven. It only downloads the raw
symbols and sessions listed by Protocol 116, and it writes into a Protocol 101
overlay directory instead of overwriting older one-second audit slices.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from v4.checks.paid_data_guard import (
    add_paid_data_approval_args,
    require_paid_data_approval,
)


DATASET = "OPRA.PILLAR"
DEFAULT_MANIFEST = Path("v4/promotion/PROTOCOL_101_TARGETED_CBBO_1S_DOWNLOAD_MANIFEST.json")
DEFAULT_OUT_ROOT = Path("data/raw/audit/protocol101_highres_opra")
DEFAULT_AUDIT_OUT = Path("v4/audit/databento_protocol101_highres_downloads.jsonl")
DEFAULT_SUMMARY = Path(
    "v4/audit/autoresearch/v4_aplus_hypothesis_116_protocol101_targeted_1s_download/download_summary.json"
)


@dataclass(frozen=True)
class DownloadPlan:
    session: str
    schema: str
    symbols: list[str]
    selected_trades: int
    estimate_usd: float
    existing: bool
    dbn_path: str
    parquet_path: str


@dataclass(frozen=True)
class DownloadRecord:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-manifest-estimates", action="store_true")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    add_paid_data_approval_args(parser, default_manifest=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_env_file(args.env_file)
    manifest = _load_manifest(args.manifest)
    selected_sessions = set(args.sessions or manifest["requested_data"]["sessions"])
    rows = _missing_replay_rows(manifest, selected_sessions)
    if rows.empty:
        raise SystemExit("no missing Protocol 101 high-resolution rows matched the requested sessions")

    client = _client()
    plans = build_plan(
        client,
        manifest=manifest,
        rows=rows,
        out_root=args.out_root,
        selected_sessions=selected_sessions,
        overwrite=args.overwrite,
        use_manifest_estimates=args.use_manifest_estimates,
    )
    total_estimate = float(sum(plan.estimate_usd for plan in plans if not plan.existing or args.overwrite))
    max_cost = float(args.max_cost if args.max_cost is not None else manifest["cost_estimate"]["hard_cap_usd"])
    payload = {
        "protocol": "116_protocol101_targeted_highres_download",
        "dataset": DATASET,
        "sessions": len(plans),
        "schemas": sorted({plan.schema for plan in plans}),
        "symbol_session_count": int(sum(len(plan.symbols) for plan in plans)),
        "selected_trades": int(rows.shape[0]),
        "estimated_total_usd": total_estimate,
        "max_cost_usd": max_cost,
        "dry_run": bool(args.dry_run),
        "plans": [plan_for_json(plan) for plan in plans],
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if total_estimate > max_cost:
        raise SystemExit(f"abort: estimated cost {total_estimate:.4f} exceeds cap {max_cost:.4f}")
    if args.dry_run:
        return 0

    records: list[DownloadRecord] = []
    for plan in plans:
        if plan.existing and not args.overwrite:
            continue
        require_paid_data_approval(
            manifest_path=args.approval_manifest,
            approval_text=args.approval_text,
            approval_env_var=args.approval_env_var,
            operation=f"Databento {DATASET} {plan.schema} Protocol 101 targeted download for {plan.session}",
        )
        frame = _download(
            client,
            session=plan.session,
            schema=plan.schema,
            symbols=plan.symbols,
            dbn_path=Path(plan.dbn_path),
            parquet_path=Path(plan.parquet_path),
            overwrite=args.overwrite,
            retries=args.download_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
        )
        record = DownloadRecord(
            session=plan.session,
            dataset=DATASET,
            schema=plan.schema,
            symbols=len(plan.symbols),
            selected_trades=plan.selected_trades,
            cost_estimate_usd=plan.estimate_usd,
            dbn_path=plan.dbn_path,
            parquet_path=plan.parquet_path,
            rows=int(len(frame)),
        )
        records.append(record)
        _append_jsonl(args.audit_out, asdict(record))
        print(json.dumps(asdict(record), sort_keys=True), flush=True)

    summary = {**payload, "dry_run": False, "downloaded": [asdict(record) for record in records]}
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"downloaded_sessions": len(records), "summary": str(args.summary_out)}, indent=2, sort_keys=True))
    return 0


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("protocol") != "116_protocol101_targeted_1s_request":
        raise SystemExit(f"unexpected manifest protocol in {path}: {manifest.get('protocol')}")
    return manifest


def _client() -> Any:
    try:
        import databento as db
    except ImportError:
        print("databento is not installed. Run inside the project venv.", file=sys.stderr)
        raise SystemExit(2)
    return db.Historical()


def _bounds(session: str) -> tuple[str, str]:
    session_date = pd.Timestamp(session).date()
    start = datetime.combine(session_date, time(0, 0), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _missing_replay_rows(manifest: dict[str, Any], selected_sessions: set[str]) -> pd.DataFrame:
    replay_json = Path(manifest["source_replay_json"])
    if not replay_json.exists():
        raise SystemExit(f"source replay JSON not found: {replay_json}")
    rows = pd.DataFrame(json.loads(replay_json.read_text())["rows"])
    rows["session"] = rows["session"].astype(str)
    rows["raw_symbol"] = rows["raw_symbol"].astype(str)
    out = rows[
        rows["session"].isin(selected_sessions)
        & ~rows["audit_status"].astype(str).eq("audited")
        & rows["raw_symbol"].notna()
        & rows["raw_symbol"].ne("")
        & rows["raw_symbol"].ne("nan")
    ].copy()
    return out


def _request_by_session(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["session"]): row for row in manifest.get("requests", [])}


def _out_paths(out_root: Path, *, session: str, schema: str) -> tuple[Path, Path]:
    directory = out_root / schema
    return directory / f"{session}.{schema}.dbn.zst", directory / f"{session}.{schema}.parquet"


def _estimate(client: Any, *, schema: str, symbols: Sequence[str], start: str, end: str) -> float:
    return float(
        client.metadata.get_cost(
            dataset=DATASET,
            schema=schema,
            symbols=list(symbols),
            stype_in="raw_symbol",
            start=start,
            end=end,
        )
    )


def build_plan(
    client: Any,
    *,
    manifest: dict[str, Any],
    rows: pd.DataFrame,
    out_root: Path,
    selected_sessions: set[str],
    overwrite: bool,
    use_manifest_estimates: bool,
) -> list[DownloadPlan]:
    requests = _request_by_session(manifest)
    plans: list[DownloadPlan] = []
    for session, group in rows.groupby("session", sort=True):
        if session not in selected_sessions:
            continue
        request = requests.get(str(session))
        if request is None:
            raise SystemExit(f"session {session} is missing from Protocol 116 manifest requests")
        schema = str(request["schema"])
        symbols = sorted(set(group["raw_symbol"].astype(str)))
        dbn_path, parquet_path = _out_paths(out_root, session=str(session), schema=schema)
        existing = dbn_path.exists() and parquet_path.exists()
        estimate = 0.0
        if overwrite or not existing:
            if use_manifest_estimates:
                estimate = float(request.get("estimated_cost_usd", 0.0) or 0.0)
            else:
                start, end = _bounds(str(session))
                estimate = _estimate(client, schema=schema, symbols=symbols, start=start, end=end)
        plans.append(
            DownloadPlan(
                session=str(session),
                schema=schema,
                symbols=symbols,
                selected_trades=int(len(group)),
                estimate_usd=float(estimate),
                existing=bool(existing),
                dbn_path=str(dbn_path),
                parquet_path=str(parquet_path),
            )
        )
    return plans


def _download(
    client: Any,
    *,
    session: str,
    schema: str,
    symbols: Sequence[str],
    dbn_path: Path,
    parquet_path: Path,
    overwrite: bool,
    retries: int,
    retry_sleep_seconds: float,
) -> pd.DataFrame:
    if dbn_path.exists() and parquet_path.exists() and not overwrite:
        return pd.read_parquet(parquet_path)
    dbn_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    start, end = _bounds(session)
    attempts = max(int(retries), 1)
    store = None
    for attempt in range(1, attempts + 1):
        if dbn_path.exists() and not parquet_path.exists():
            dbn_path.unlink()
        try:
            store = client.timeseries.get_range(
                dataset=DATASET,
                schema=schema,
                symbols=list(symbols),
                stype_in="raw_symbol",
                start=start,
                end=end,
                path=dbn_path,
            )
            break
        except Exception:
            if attempt >= attempts:
                raise
            time_module.sleep(float(retry_sleep_seconds) * attempt)
    if store is None:
        raise RuntimeError(f"download failed without an exception for {session} {schema}")
    frame = store.to_df()
    frame.to_parquet(parquet_path, index=True)
    return frame


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def plan_for_json(plan: DownloadPlan) -> dict[str, Any]:
    row = asdict(plan)
    row["symbols"] = len(plan.symbols)
    return row


if __name__ == "__main__":
    raise SystemExit(main())
