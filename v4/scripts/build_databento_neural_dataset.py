"""Build normalized and neural SPXW 0DTE datasets from downloaded Databento files."""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from v4.dataset.spxw_0dte_neural import NeuralDatasetConfig, build_neural_dataset
from v4.live.protocol101_feature_contract import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED,
    feature_contract_version,
    is_live_feature_contract,
)
from v4.ingest.databento_opra import normalize_spxw_0dte_day
from v4.ingest.derived_context import write_derived_context
from v4.ingest.index_bars import load_spx_1m, load_vix_1m
from v4.model.protocol101_regimen_repair import (
    LEGACY_PROCESSED_ROW_SCHEMA,
    TWO_CLOCK_PROCESSED_ROW_SCHEMA,
)


@dataclass(frozen=True)
class BuildRecord:
    session: str
    definition_rows: int
    cbbo_rows: int
    ohlcv_rows: int
    statistics_rows: int
    normalized_rows: int
    derived_spx_rows: int
    derived_vix_rows: int
    neural_rows: int
    normalized_path: str
    derived_normalized_path: str
    spx_path: str
    vix_path: str
    neural_path: str
    context_mode: str
    context_source: str
    feature_contract: str
    diagnostic_index_context_lag_minutes: int | None = None
    diagnostic_source_policy: str | None = None
    production_contract_mutation: bool = False
    compute_policy_labels: bool = True
    label_mode: str = "historical_default"
    processed_row_schema_version: str = LEGACY_PROCESSED_ROW_SCHEMA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument(
        "--sessions-file",
        type=Path,
        default=None,
        help=(
            "optional newline-delimited YYYY-MM-DD session list. When provided, "
            "only these sessions inside the requested date range are considered."
        ),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "skip sessions whose processed file and normalized/context parquet files "
            "already exist, making long historical fair-contract builds resumable"
        ),
    )
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--normalized-dir", type=Path, default=Path("v4/normalized"))
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument(
        "--context-mode",
        choices=("derived", "official", "auto"),
        default="derived",
        help="derived uses SPXW parity/ATM IV; official requires SPX/VIX files; auto uses official when both files exist",
    )
    p.add_argument(
        "--official-spx-dir",
        type=Path,
        default=None,
        help="directory containing official/live-equivalent SPX bars named with the session date",
    )
    p.add_argument(
        "--official-vix-dir",
        type=Path,
        default=None,
        help="directory containing official/live-equivalent VIX bars named with the session date",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=Path("v4/audit/databento_neural_dataset_build_summary.json"),
    )
    p.add_argument(
        "--feature-contract",
        choices=("historical-default", FEATURE_CONTRACT_VERSION, FEATURE_CONTRACT_VERSION_MICROSTRUCTURE_MASKED),
        default="historical-default",
        help=(
            "historical-default preserves legacy dataset behavior; "
            "protocol101-live-v1 builds causal live-reproducible rows for parity/sanity gates; "
            "protocol101-live-v2-microstructure-masked additionally declares the model-scoring "
            "mask for vendor-sensitive option microstructure"
        ),
    )
    p.add_argument(
        "--diagnostic-index-context-lag-minutes",
        type=int,
        default=None,
        help=(
            "diagnostic-only override for protocol101-live-v1 index context lag; "
            "writes separate artifacts and does not mutate the production contract"
        ),
    )
    p.add_argument(
        "--diagnostic-source-policy",
        default=None,
        help="diagnostic-only label recorded in build summaries and row feature-contract metadata",
    )
    p.add_argument(
        "--compute-live-policy-labels",
        action="store_true",
        help=(
            "offline training-only mode for protocol101-live-v1: keep live-reproducible "
            "features but compute future ask-entry/bid-exit labels. Use a separate "
            "processed directory; this does not mutate the production feature contract."
        ),
    )
    p.add_argument(
        "--processed-row-schema",
        choices=(
            LEGACY_PROCESSED_ROW_SCHEMA,
            TWO_CLOCK_PROCESSED_ROW_SCHEMA,
        ),
        default=LEGACY_PROCESSED_ROW_SCHEMA,
        help=(
            "legacy preserves historical row shape; the owner-signed additive "
            "v2 schema persists source-pricing and occupancy-exit clocks"
        ),
    )
    return p.parse_args()


def _sessions(start: str, end: str) -> list[str]:
    cursor = pd.Timestamp(start).date()
    final = pd.Timestamp(end).date()
    out: list[str] = []
    while cursor <= final:
        if cursor.weekday() < 5:
            out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def _requested_sessions(args: argparse.Namespace) -> list[str]:
    sessions = _sessions(args.start_date, args.end_date)
    if args.sessions_file is None:
        return sessions
    requested = {
        line.strip()
        for line in args.sessions_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return [session for session in sessions if session in requested]


def _existing_outputs(
    *,
    session: str,
    normalized_dir: Path,
    processed_dir: Path,
    context_mode: str,
    official_spx_dir: Path | None,
    official_vix_dir: Path | None,
    processed_row_schema_version: str = LEGACY_PROCESSED_ROW_SCHEMA,
) -> bool:
    neural_path = processed_dir / f"{session}.pkl"
    normalized_path = normalized_dir / f"databento_spxw_0dte_{session}.parquet"
    official_spx_path = _find_index_file(official_spx_dir, session)
    official_vix_path = _find_index_file(official_vix_dir, session)
    use_official = context_mode == "official" or (
        context_mode == "auto" and official_spx_path is not None and official_vix_path is not None
    )
    context_suffix = "official_context" if use_official else "derived_context"
    context_path = normalized_dir / f"databento_spxw_0dte_{session}_{context_suffix}.parquet"
    complete = (
        neural_path.exists()
        and normalized_path.exists()
        and context_path.exists()
    )
    if (
        not complete
        or processed_row_schema_version != TWO_CLOCK_PROCESSED_ROW_SCHEMA
    ):
        return complete
    with neural_path.open("rb") as handle:
        rows = pickle.load(handle)
    return bool(
        isinstance(rows, list)
        and rows
        and all(
            isinstance(row, dict)
            and row.get("processed_row_schema_version")
            == TWO_CLOCK_PROCESSED_ROW_SCHEMA
            for row in rows
        )
    )


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _find_index_file(directory: Path | None, session: str) -> Path | None:
    if directory is None:
        return None
    candidates = [
        directory / f"{session}.parquet",
        directory / f"{session}.csv",
        directory / f"{session}.jsonl",
        directory / f"{session}.txt",
    ]
    candidates.extend(sorted(directory.glob(f"{session}.*")))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _stamp_context(frame: pd.DataFrame, *, source: str, official: bool) -> pd.DataFrame:
    out = frame.copy()
    out["context_source"] = source
    out["is_derived"] = False
    out["is_proxy"] = False
    out["is_official_index_data"] = bool(official)
    return out


def _paths(raw_root: Path, session: str) -> dict[str, Path]:
    return {
        "definition": raw_root
        / "databento"
        / "opra_spxw_definition"
        / f"{session}.definition.parquet",
        "cbbo": raw_root
        / "databento"
        / "opra_spxw_cbbo_1m"
        / f"{session}.cbbo-1m.parquet",
        "ohlcv": raw_root
        / "databento"
        / "opra_spxw_ohlcv_1m"
        / f"{session}.ohlcv-1m.parquet",
        "statistics": raw_root
        / "databento"
        / "opra_spxw_statistics"
        / f"{session}.statistics.parquet",
    }


def _build_session(
    *,
    session: str,
    raw_root: Path,
    normalized_dir: Path,
    processed_dir: Path,
    context_mode: str,
    official_spx_dir: Path | None,
    official_vix_dir: Path | None,
    feature_contract: str,
    diagnostic_index_context_lag_minutes: int | None = None,
    diagnostic_source_policy: str | None = None,
    compute_live_policy_labels: bool = False,
    processed_row_schema_version: str = LEGACY_PROCESSED_ROW_SCHEMA,
) -> BuildRecord | None:
    paths = _paths(raw_root, session)
    if not paths["definition"].exists() or not paths["cbbo"].exists():
        return None

    definitions = _read(paths["definition"]).reset_index()
    cbbo = _read(paths["cbbo"]).reset_index()
    ohlcv = _read(paths["ohlcv"]).reset_index()
    statistics = _read(paths["statistics"]).reset_index()
    if definitions.empty or cbbo.empty:
        return None

    normalized_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = normalized_dir / f"databento_spxw_0dte_{session}.parquet"
    context_normalized_path = normalized_dir / f"databento_spxw_0dte_{session}_derived_context.parquet"
    spx_path = raw_root / "index" / "spx_1m" / f"{session}.derived_spxw_parity.parquet"
    vix_path = raw_root / "index" / "vix_1m" / f"{session}.derived_spxw_atm_iv.parquet"
    neural_path = processed_dir / f"{session}.pkl"
    if (
        processed_row_schema_version == TWO_CLOCK_PROCESSED_ROW_SCHEMA
        and neural_path.exists()
    ):
        raise FileExistsError(
            "two-clock processed output already exists and is immutable; "
            "use --skip-existing only for a matching v2 packet or choose a "
            f"new processed namespace: {neural_path}"
        )

    official_spx_path = _find_index_file(official_spx_dir, session)
    official_vix_path = _find_index_file(official_vix_dir, session)
    use_official = context_mode == "official" or (
        context_mode == "auto" and official_spx_path is not None and official_vix_path is not None
    )
    if context_mode == "official" and (official_spx_path is None or official_vix_path is None):
        raise FileNotFoundError(
            f"official context requested for {session}, but missing SPX/VIX files: "
            f"spx={official_spx_path} vix={official_vix_path}"
        )

    live_contract = is_live_feature_contract(feature_contract)
    compute_policy_labels = (not live_contract) or bool(compute_live_policy_labels)
    label_mode = (
        "live_contract_offline_training_labels"
        if live_contract and compute_policy_labels
        else "live_contract_zero_label_placeholders"
        if live_contract
        else "historical_default_policy_labels"
    )
    # Live-contract replay consumes normalized quotes for lifecycle outcomes.
    # By default keep the live-compatible zero label placeholders required by
    # the frozen surface wrapper. Offline training builds may explicitly compute
    # future labels in a separate processed directory; these labels are targets,
    # not runtime features.
    neural_config = NeuralDatasetConfig(
        feature_contract=feature_contract,
        compute_policy_labels=compute_policy_labels,
        diagnostic_index_context_lag_minutes=diagnostic_index_context_lag_minutes,
        diagnostic_source_policy=diagnostic_source_policy,
        processed_row_schema_version=processed_row_schema_version,
    )

    if use_official:
        spx_bars = _stamp_context(load_spx_1m(official_spx_path), source=str(official_spx_path), official=True)
        vix_bars = _stamp_context(load_vix_1m(official_vix_path), source=str(official_vix_path), official=True)
        context_normalized_path = normalized_dir / f"databento_spxw_0dte_{session}_official_context.parquet"
        spx_path = raw_root / "index" / "spx_1m" / f"{session}.official_spx.parquet"
        vix_path = raw_root / "index" / "vix_1m" / f"{session}.official_vix.parquet"
        spx_path.parent.mkdir(parents=True, exist_ok=True)
        vix_path.parent.mkdir(parents=True, exist_ok=True)
        spx_bars.to_parquet(spx_path, index=False)
        vix_bars.to_parquet(vix_path, index=False)
        normalized = normalize_spxw_0dte_day(
            definitions,
            cbbo,
            ohlcv_1m=ohlcv,
            statistics=statistics,
            index_bars=spx_bars,
        )
        pq.write_table(normalized, normalized_path)
        pq.write_table(normalized, context_normalized_path)
        neural_rows = build_neural_dataset(
            normalized,
            spx_bars,
            vix_bars,
            config=neural_config,
        )
        context_source = "official_index_bars"
    else:
        normalized = normalize_spxw_0dte_day(
            definitions,
            cbbo,
            ohlcv_1m=ohlcv,
            statistics=statistics,
        )
        pq.write_table(normalized, normalized_path)
        derived = write_derived_context(
            normalized,
            session=session,
            spx_path=spx_path,
            vol_path=vix_path,
            normalized_out=context_normalized_path,
        )
        neural_rows = build_neural_dataset(
            derived.normalized,
            derived.spx_bars,
            derived.vol_bars,
            config=neural_config,
        )
        spx_bars = derived.spx_bars
        vix_bars = derived.vol_bars
        context_source = "spxw_derived_context"

    with neural_path.open("wb") as f:
        pickle.dump(neural_rows, f)

    return BuildRecord(
        session=session,
        definition_rows=len(definitions),
        cbbo_rows=len(cbbo),
        ohlcv_rows=len(ohlcv),
        statistics_rows=len(statistics),
        normalized_rows=int(normalized.num_rows),
        derived_spx_rows=int(len(spx_bars)),
        derived_vix_rows=int(len(vix_bars)),
        neural_rows=int(len(neural_rows)),
        normalized_path=str(normalized_path),
        derived_normalized_path=str(context_normalized_path),
        spx_path=str(spx_path),
        vix_path=str(vix_path),
        neural_path=str(neural_path),
        context_mode="official" if use_official else "derived",
        context_source=context_source,
        feature_contract=feature_contract_version(feature_contract),
        diagnostic_index_context_lag_minutes=diagnostic_index_context_lag_minutes,
        diagnostic_source_policy=diagnostic_source_policy,
        production_contract_mutation=False,
        compute_policy_labels=compute_policy_labels,
        label_mode=label_mode,
        processed_row_schema_version=processed_row_schema_version,
    )


def main() -> int:
    args = parse_args()
    records: list[BuildRecord] = []
    skipped: list[str] = []
    existing_skipped: list[str] = []
    for session in _requested_sessions(args):
        if args.skip_existing and _existing_outputs(
            session=session,
            normalized_dir=args.normalized_dir,
            processed_dir=args.processed_dir,
            context_mode=args.context_mode,
            official_spx_dir=args.official_spx_dir,
            official_vix_dir=args.official_vix_dir,
            processed_row_schema_version=args.processed_row_schema,
        ):
            existing_skipped.append(session)
            print(f"{session}: skipped existing outputs", flush=True)
            continue
        record = _build_session(
            session=session,
            raw_root=args.raw_root,
            normalized_dir=args.normalized_dir,
            processed_dir=args.processed_dir,
            context_mode=args.context_mode,
            official_spx_dir=args.official_spx_dir,
            official_vix_dir=args.official_vix_dir,
            feature_contract=args.feature_contract,
            diagnostic_index_context_lag_minutes=args.diagnostic_index_context_lag_minutes,
            diagnostic_source_policy=args.diagnostic_source_policy,
            compute_live_policy_labels=bool(args.compute_live_policy_labels),
            processed_row_schema_version=args.processed_row_schema,
        )
        if record is None:
            skipped.append(session)
            print(f"{session}: skipped missing raw files", flush=True)
            continue
        records.append(record)
        print(
            json.dumps(
                {
                    "session": session,
                    "normalized_rows": record.normalized_rows,
                    "neural_rows": record.neural_rows,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    payload = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "processed_dir": str(args.processed_dir),
        "feature_contract": feature_contract_version(args.feature_contract),
        "diagnostic_index_context_lag_minutes": args.diagnostic_index_context_lag_minutes,
        "diagnostic_source_policy": args.diagnostic_source_policy,
        "compute_live_policy_labels": bool(args.compute_live_policy_labels),
        "processed_row_schema_version": args.processed_row_schema,
        "production_contract_mutation": False,
        "sessions_considered": len(_requested_sessions(args)),
        "sessions_built": len(records),
        "sessions_skipped": skipped,
        "sessions_existing_skipped": existing_skipped,
        "totals": {
            "definition_rows": sum(r.definition_rows for r in records),
            "cbbo_rows": sum(r.cbbo_rows for r in records),
            "ohlcv_rows": sum(r.ohlcv_rows for r in records),
            "statistics_rows": sum(r.statistics_rows for r in records),
            "normalized_rows": sum(r.normalized_rows for r in records),
            "derived_spx_rows": sum(r.derived_spx_rows for r in records),
            "derived_vix_rows": sum(r.derived_vix_rows for r in records),
            "neural_rows": sum(r.neural_rows for r in records),
        },
        "records": [asdict(r) for r in records],
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.summary_out)
    print(json.dumps(payload["totals"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
