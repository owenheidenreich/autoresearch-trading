"""Diagnostic: price the SPXW 0DTE backfill at both request scopes.

**This is a defect diagnostic, not an acquisition step.** The 2026-08-15 exact
preflight returned $671.90 against a $75 cap and correctly STOPPED. That receipt
priced `stype_in="parent"` (`SPXW.OPT`), which covers **every SPXW expiration
listed that day** — the owned definition files carry 39-41 of them — while the
plan needs only the same-day (0DTE) subset. The already-owned quote corpus was
bought with resolved 0DTE symbol lists at a recorded $0.0271/session over 420
sessions, 31x below the parent-scope price.

This probe measures both scopes on the same sessions with the vendor's own cost
metadata, so the correction rests on measurement rather than on that inference.

It contacts the vendor's **cost metadata endpoint only**: no data is requested,
nothing is downloaded, and no money can be spent. It does not read, write, or
cancel the frozen acquisition declaration — changing the declared request shape
remains an owner decision under that declaration's own semantic-freeze rule.

Symbol-scope coverage note: the local 0DTE symbol lists come from owned OHLCV,
which records only contracts that **traded**. The true quoted ladder is larger
(504-1288 symbols in recorded acquisitions), so this probe reports cost per
symbol and a conservative projection at full ladder width rather than treating
its own symbol-scope total as the final answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from v5.ops.download_spxw_history import (
    DATASET,
    PARENT,
    _bounds,
    _load_env,
    canonical_json,
    source_sessions,
)

# Recorded symbol-scope evidence from the owned acquisition receipts, used only
# for projection and comparison. Source: databento*_downloads.jsonl, 420 sessions.
RECORDED_SYMBOL_SCOPE_MEAN_USD = 0.027149744686642857
RECORDED_SYMBOL_SCOPE_SESSIONS = 420
RECORDED_LADDER_SYMBOLS_MAX = 1288

SOURCE_ROOT = Path("/Volumes/AR_TRADING_DATA/spxw_0dte_2022-06-01_2026-07-31/raw")
PROBE_STRIDE = 100  # deterministic, outcome-free session selection


def zero_dte_symbols(root: Path, session: str) -> list[str]:
    """Owned 0DTE symbols for one session, from local files only."""

    path = root / f"{session}.spxw_0dte.ohlcv-1m.parquet"
    frame = pd.read_parquet(path, columns=["symbol"])
    symbols = sorted({str(value) for value in frame["symbol"].unique()})
    if not symbols:
        raise RuntimeError(f"no owned symbols for {session}")
    return symbols


def probe_session(client: Any, session: str, symbols: list[str]) -> dict[str, Any]:
    """Cost the same session and window at both request scopes."""

    start, end = _bounds(session, "cbbo-1m")
    parent_cost = float(
        client.metadata.get_cost(
            dataset=DATASET,
            schema="cbbo-1m",
            symbols=[PARENT],
            stype_in="parent",
            start=start,
            end=end,
        )
    )
    symbol_cost = float(
        client.metadata.get_cost(
            dataset=DATASET,
            schema="cbbo-1m",
            symbols=symbols,
            stype_in="raw_symbol",
            start=start,
            end=end,
        )
    )
    return {
        "session": session,
        "start": start,
        "end": end,
        "parent_scope_usd": parent_cost,
        "symbol_scope_usd": symbol_cost,
        "traded_symbols": len(symbols),
        "usd_per_symbol": symbol_cost / len(symbols) if symbols else None,
        "scope_ratio": parent_cost / symbol_cost if symbol_cost > 0 else None,
    }


def summarize(rows: list[dict[str, Any]], sessions_total: int) -> dict[str, Any]:
    """Project both scopes over the full inventory, conservatively."""

    parent_mean = sum(r["parent_scope_usd"] for r in rows) / len(rows)
    symbol_mean = sum(r["symbol_scope_usd"] for r in rows) / len(rows)
    per_symbol = [r["usd_per_symbol"] for r in rows if r["usd_per_symbol"]]
    per_symbol_mean = sum(per_symbol) / len(per_symbol) if per_symbol else 0.0
    full_ladder_projection = per_symbol_mean * RECORDED_LADDER_SYMBOLS_MAX * sessions_total
    return {
        "sessions_probed": len(rows),
        "sessions_total": sessions_total,
        "parent_scope_mean_usd": parent_mean,
        "symbol_scope_mean_usd": symbol_mean,
        "scope_ratio_mean": parent_mean / symbol_mean if symbol_mean > 0 else None,
        "usd_per_symbol_mean": per_symbol_mean,
        "projected_parent_scope_total_usd": parent_mean * sessions_total,
        "projected_traded_symbol_total_usd": symbol_mean * sessions_total,
        "projected_full_ladder_total_usd": full_ladder_projection,
        "recorded_symbol_scope_projection_usd": (
            RECORDED_SYMBOL_SCOPE_MEAN_USD * sessions_total
        ),
    }


def write_probe(*, output_path: Path, client: Any, stride: int = PROBE_STRIDE) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite probe receipt: {output_path}")
    sessions, sessions_hash = source_sessions(
        SOURCE_ROOT, start="2022-06-01", end="2025-07-31"
    )
    selected = list(sessions)[::stride]
    rows = [probe_session(client, s, zero_dte_symbols(SOURCE_ROOT, s)) for s in selected]
    receipt: dict[str, Any] = {
        "schema_version": "v5.backfill-request-scope-probe.v1",
        "purpose": (
            "measure whether the 2026-08-15 STOP priced a request 31x wider than "
            "the plan requires; parent scope covers every listed SPXW expiration"
        ),
        "classification": "DIAGNOSTIC — not an acquisition preflight",
        "dataset": DATASET,
        "schema": "cbbo-1m",
        "selection_law": f"every {stride}th session of the frozen inventory, outcome-free",
        "source_inventory": {
            "root": str(SOURCE_ROOT),
            "sessions": len(sessions),
            "sessions_sha256": sessions_hash,
        },
        "symbol_source": (
            "owned OHLCV 0DTE files; traded contracts only, a strict subset of the "
            "quoted ladder, so symbol-scope figures here are lower bounds"
        ),
        "recorded_symbol_scope_reference": {
            "mean_usd_per_session": RECORDED_SYMBOL_SCOPE_MEAN_USD,
            "sessions": RECORDED_SYMBOL_SCOPE_SESSIONS,
            "ladder_symbols_max": RECORDED_LADDER_SYMBOLS_MAX,
        },
        "integrity": {
            "money_spent": False,
            "data_downloaded": False,
            "declaration_modified": False,
            "cost_metadata_only": True,
        },
        "sessions": rows,
        "summary": summarize(rows, len(sessions)),
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path("v4/.env"))
    parser.add_argument("--stride", type=int, default=PROBE_STRIDE)
    args = parser.parse_args()

    _load_env(args.env_file)
    import databento as db

    client = db.Historical(os.environ.get("DATABENTO_API_KEY"))
    receipt = write_probe(output_path=args.out, client=client, stride=args.stride)
    summary = receipt["summary"]
    for row in receipt["sessions"]:
        print(
            f"{row['session']}  parent ${row['parent_scope_usd']:.4f}"
            f"  symbols({row['traded_symbols']:4d}) ${row['symbol_scope_usd']:.4f}"
            f"  ratio {row['scope_ratio']:.1f}x"
        )
    print(
        f"\nmean parent ${summary['parent_scope_mean_usd']:.4f}/session"
        f" vs traded-symbol ${summary['symbol_scope_mean_usd']:.4f}/session"
        f" ({summary['scope_ratio_mean']:.1f}x)"
    )
    print(
        f"projected over {summary['sessions_total']} sessions:"
        f" parent ${summary['projected_parent_scope_total_usd']:.2f}"
        f" | traded-symbol ${summary['projected_traded_symbol_total_usd']:.2f}"
        f" | full-ladder bound ${summary['projected_full_ladder_total_usd']:.2f}"
        f" | recorded-rate ${summary['recorded_symbol_scope_projection_usd']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
