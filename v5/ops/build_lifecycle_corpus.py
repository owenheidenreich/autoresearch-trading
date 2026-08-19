"""Build the two-era causal corpus, with settlement source labelled per session.

**Why a driver rather than the pinned `run()`.** `build_causal_day_dataset.py` is
hash-pinned by `PREACQUISITION_SEMANTIC_FREEZE_V1`, and its `run()` can only take
settlement from `load_validated_settlements`, which requires a receipt whose
status is `VALIDATED_FOR_TERMINAL_ACCOUNTING`. That receipt exists for the owned
year and **for nothing else**: the official, non-derived SPX 16:00 source covers
251 sessions, so roughly **76% of this corpus has no official settlement** and the
pinned orchestration cannot express what the other three quarters need. The
*semantics* — `build_session`, its labels, its first-later-bid law — are reused
unchanged; only the orchestration around them is new.

**Settlement law, per the approved plan and the design's era section.**

* `official_1600` — the owned era, from the validated audit receipt.
* `parity_close` — the backfill era, from the session's own parity spot at the
  terminal minute, which `repair_parity_spot` supplies and labels. Parity is exact
  for European options, and the repair records how many strikes backed each value.

Every session carries its source in the receipt, and a cash settlement is never
called a fill. Because the pinned builder already emits `net_*_zero_recovery_*`
columns beside its settled ones, the paired zero-recovery sensitivity the plan
requires is available on the built corpus without a second pass.

The pass is resumable: a session whose outputs already exist is skipped, so an
interrupted build costs only the sessions it had not finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v5.ops.audit_causal_day_coverage import ET, LAST_QUOTE_MINUTE
from v5.ops.build_causal_day_dataset import (
    QUOTE_COLUMNS,
    build_session,
    canonical_json,
    load_validated_settlements,
    prepare_quotes,
)

SCHEMA_VERSION = "v5.lifecycle-corpus-build.v1"
TABLES = ("candles", "minutes", "ladder", "candidates", "atlas")
SOURCE_OFFICIAL = "official_1600"
SOURCE_PARITY = "parity_close"
PRIMARY_LABEL = "first_touch_50pct_before_loss_30pct_60m"


class CorpusBuildError(RuntimeError):
    """A session cannot be built under the declared settlement law."""


@dataclass(frozen=True)
class EraSource:
    """One data root and the settlement law that governs it."""

    era: str
    quote_root: Path
    settlement_source: str


def parity_close(quote_path: Path) -> tuple[float, bool, int]:
    """The session's own parity spot at the terminal minute.

    Read through `prepare_quotes` so the value is the one the builder will see,
    not a separately-derived number that could drift from it. Returns
    ``(spot, carried, carry_minutes)`` so a carried close travels into the
    receipt rather than being invisible once the corpus is built.
    """

    session = quote_path.stem.split("_")[-1]
    columns = list(QUOTE_COLUMNS)
    # Read the schema from parquet metadata, never via `read_parquet(columns=[])`:
    # that returns a frame with zero columns rather than the schema, so it reports
    # every optional column as absent and silently disables the provenance it is
    # meant to detect. That defect shipped once here and produced a carried-close
    # footnote of zero against 214 genuinely carried sessions.
    available = set(pq.ParquetFile(quote_path).schema_arrow.names)
    provenance = [
        name
        for name in ("underlying_price_source", "underlying_carry_minutes")
        if name in available
    ]
    raw = pd.read_parquet(quote_path, columns=columns + provenance)
    quotes = prepare_quotes(raw[columns], session)
    terminal_mask = quotes["minute"].eq(LAST_QUOTE_MINUTE)
    value = float(
        pd.to_numeric(quotes.loc[terminal_mask, "underlying_price"], errors="coerce").median()
    )
    if not np.isfinite(value) or value <= 0.0:
        raise CorpusBuildError(
            f"{quote_path.name}: no finite parity spot at {LAST_QUOTE_MINUTE}; "
            "run repair_parity_spot before building"
        )

    carried, carry_minutes = False, 0
    if "underlying_price_source" in provenance:
        stamped = pd.to_datetime(raw["event_time"], utc=True).dt.tz_convert(ET)
        at_close = raw[stamped.dt.strftime("%H:%M").eq(LAST_QUOTE_MINUTE)]
        if len(at_close):
            carried = bool((at_close["underlying_price_source"] == "carried_parity").any())
            if carried and "underlying_carry_minutes" in provenance:
                carry_minutes = int(
                    pd.to_numeric(at_close["underlying_carry_minutes"], errors="coerce").max()
                )
    return value, carried, carry_minutes


def tape_source(es_path: Path) -> str:
    """What the chart state is actually made of, read from the file itself.

    Owner ruling 2026-08-19: the tape is SPX parity spot, not ES futures. The
    pinned `build_session` writes the values into columns named `es_open`,
    `es_high` and so on, and it may not be edited to rename them -- so the
    corpus stamps what the tape really is beside them. A later reader seeing
    `es_close` next to `tape_source=spx_parity_spot` cannot stay misled for
    long, which is the whole point of the ruling.
    """

    available = set(pq.ParquetFile(es_path).schema_arrow.names)
    if "tape_source" not in available:
        return "es_futures"
    values = pd.read_parquet(es_path, columns=["tape_source"])["tape_source"].unique()
    if len(values) != 1:
        raise CorpusBuildError(f"{es_path.name}: candle file declares {len(values)} tape sources")
    return str(values[0])


def locate(sources: Sequence[EraSource], session: str) -> tuple[EraSource, Path]:
    for source in sources:
        candidate = source.quote_root / f"databento_spxw_0dte_{session}.parquet"
        if candidate.exists():
            return source, candidate
    raise CorpusBuildError(f"{session}: no quote file in any declared era root")


def build_one(
    session: str,
    *,
    sources: Sequence[EraSource],
    es_root: Path,
    out_dir: Path,
    official_settlements: dict[str, float],
) -> dict[str, Any]:
    source, quote_path = locate(sources, session)
    es_path = es_root / f"{session}.es_c_0.ohlcv-1m.parquet"
    if not es_path.exists():
        raise CorpusBuildError(f"{session}: missing ES candles at {es_path}")

    if source.settlement_source == SOURCE_OFFICIAL:
        if session not in official_settlements:
            raise CorpusBuildError(
                f"{session}: era declares an official settlement but the validated "
                "receipt does not carry one"
            )
        settlement = official_settlements[session]
        carried, carry_minutes = False, 0
    else:
        settlement, carried, carry_minutes = parity_close(quote_path)

    tape = tape_source(es_path)
    tables = build_session(quote_path, es_path, settlement_spx=settlement)
    for name in TABLES:
        destination = out_dir / name / f"{session}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_suffix(".partial")
        frame = tables[name].copy()
        frame["era"] = source.era
        frame["settlement_source"] = source.settlement_source
        frame["tape_source"] = tape
        frame.to_parquet(staging, index=False)
        staging.replace(destination)

    candidates = tables["candidates"]
    label = candidates.get(PRIMARY_LABEL)
    # Only a cash-settled exit can read the terminal underlying at all; a trade
    # closed on an executable bid never touches it. On a carried close that share
    # *is* the exposure, so it is measured per session rather than assumed small.
    cash_settled = {
        f"{horizon}m": float(
            candidates[f"clock_exit_type_{horizon}m"].eq("validated_cash_settlement").mean()
        )
        for horizon in (60, 90, 120)
        if f"clock_exit_type_{horizon}m" in candidates.columns
    }
    return {
        "session": session,
        "classification": "BUILT",
        "tape_source": tape,
        "era": source.era,
        "settlement_source": source.settlement_source,
        "settlement_spx": settlement,
        "settlement_close_carried": carried,
        "settlement_carry_minutes": carry_minutes,
        "cash_settled_share": cash_settled,
        "rows": {name: int(len(tables[name])) for name in TABLES},
        "label_coverage": float(label.notna().mean()) if label is not None else None,
        "label_base_rate": float(label.mean()) if label is not None else None,
    }


def _outputs_exist(out_dir: Path, session: str) -> bool:
    return all((out_dir / name / f"{session}.parquet").exists() for name in TABLES)


def run(
    *,
    sources: Sequence[EraSource],
    es_root: Path,
    sessions: Sequence[str],
    out_dir: Path,
    receipt_path: Path,
    settlement_receipt: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if not sessions:
        raise CorpusBuildError("no eligible sessions supplied")
    official = load_validated_settlements(settlement_receipt) if settlement_receipt else {}
    ordered = sorted(str(session) for session in sessions)
    if limit is not None:
        ordered = ordered[:limit]

    results: list[dict[str, Any]] = []
    for session in ordered:
        if _outputs_exist(out_dir, session):
            results.append({"session": session, "classification": "ALREADY_PRESENT"})
            continue
        try:
            results.append(
                build_one(
                    session,
                    sources=sources,
                    es_root=es_root,
                    out_dir=out_dir,
                    official_settlements=official,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "session": session,
                    "classification": "FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    built = [row for row in results if row["classification"] == "BUILT"]
    by_source: dict[str, int] = {}
    for row in built:
        by_source[row["settlement_source"]] = by_source.get(row["settlement_source"], 0) + 1

    def _era_rate(era: str) -> float | None:
        rates = [r["label_base_rate"] for r in built if r["era"] == era and r["label_base_rate"] is not None]
        return float(np.mean(rates)) if rates else None

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "build the two-era causal corpus reusing the pinned build_session; the "
            "pinned run() cannot express a parity-close settlement, which governs "
            "roughly three quarters of this corpus"
        ),
        "eras": [
            {
                "era": source.era,
                "quote_root": str(source.quote_root),
                "settlement_source": source.settlement_source,
            }
            for source in sources
        ],
        "out_dir": str(out_dir),
        "sessions": results,
        "summary": {
            "requested": len(ordered),
            "built": len(built),
            "already_present": int(
                sum(r["classification"] == "ALREADY_PRESENT" for r in results)
            ),
            "failed": int(sum(r["classification"] == "FAILED" for r in results)),
            "by_settlement_source": by_source,
            # Per-era base rates are reported, never pooled: the design requires
            # them measured rather than inherited from the owned year.
            "label_base_rate_by_era": {
                source.era: _era_rate(source.era) for source in sources
            },
        },
    }
    # Phase-5 footnote, owner-requested 2026-08-18: report how much of the
    # corpus rests on a carried close, and what share of trades there could
    # possibly depend on it. Reported, never gated.
    carried_rows = [r for r in built if r.get("settlement_close_carried")]
    carried_cash = [
        share
        for row in carried_rows
        for key, share in (row.get("cash_settled_share") or {}).items()
        if key == "60m"
    ]
    payload["summary"]["carried_close"] = {
        "sessions": len(carried_rows),
        "max_carry_minutes": max(
            (r.get("settlement_carry_minutes", 0) for r in built), default=0
        ),
        "mean_cash_settled_share_60m": float(np.mean(carried_cash)) if carried_cash else None,
        "note": (
            "entries stop at 15:00 by construction, and only a cash-settled exit "
            "reads the terminal underlying; this bounds what a carried close can "
            "affect. SPX is cash settled, so a position held through the close "
            "carries no assignment risk."
        ),
    }
    payload["gate"] = "PASS" if payload["summary"]["failed"] == 0 else "SESSIONS_FAILED"
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def eligible_from_coverage(paths: Sequence[Path]) -> list[str]:
    """Sessions marked build-eligible by one or more coverage CSVs."""

    sessions: set[str] = set()
    for path in paths:
        frame = pd.read_csv(path)
        if "included_for_episode_build" not in frame.columns:
            raise CorpusBuildError(f"coverage csv lacks eligibility column: {path}")
        included = frame.loc[frame["included_for_episode_build"], "session"]
        sessions.update(included.astype(str))
    if not sessions:
        raise CorpusBuildError("coverage certifies no eligible session")
    return sorted(sessions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill-root", type=Path, required=True)
    parser.add_argument("--owned-root", type=Path, required=True)
    parser.add_argument("--es-root", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--settlement-receipt", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sources = (
        EraSource("owned", args.owned_root, SOURCE_OFFICIAL),
        EraSource("backfill", args.backfill_root, SOURCE_PARITY),
    )
    payload = run(
        sources=sources,
        es_root=args.es_root,
        sessions=eligible_from_coverage(args.coverage),
        out_dir=args.out_dir,
        receipt_path=args.receipt,
        settlement_receipt=args.settlement_receipt,
        limit=args.limit,
    )
    summary = payload["summary"]
    print(
        f"corpus build {payload['gate']}: {summary['built']} built, "
        f"{summary['already_present']} present, {summary['failed']} failed "
        f"of {summary['requested']}"
    )
    print(f"  settlement sources: {summary['by_settlement_source']}")
    print(f"  base rate by era:   {summary['label_base_rate_by_era']}")
    for row in payload["sessions"]:
        if row["classification"] == "FAILED":
            print(f"  FAILED {row['session']}: {row['reason']}")
    return 0 if payload["gate"] == "PASS" else 7


if __name__ == "__main__":
    raise SystemExit(main())
