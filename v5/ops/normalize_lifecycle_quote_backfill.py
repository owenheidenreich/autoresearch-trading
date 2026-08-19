"""Normalize the declared SPXW definitions + CBBO-1m backfill without v4 code.

This is deliberately a narrow v5 bridge: it joins each quote to that session's
definition, derives contract geometry from the raw OSI symbol, recomputes the
parity spot from contemporaneous quote mids, and emits only the columns the
causal-day dataset reads.  It never forward-fills a quote, substitutes an
underlying price, or treats a missing quote as a tradeable one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ET = "America/New_York"
MIN_PAIRED_STRIKES = 5
PARITY_WINDOW_POINTS = 30.0
OSI = re.compile(r"^SPXW\s+(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")
RTH_FIRST, RTH_LAST = "09:31", "16:00"


class QuoteNormalizationError(RuntimeError):
    """A raw session cannot be safely converted into a causal quote episode."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("receipt_sha256", None)
    return hashlib.sha256(canonical_json(semantic)).hexdigest()


@dataclass(frozen=True)
class Definition:
    raw_symbol: str
    contract_id: str
    instrument_id: int
    expiry: str
    strike: float
    right: str


def _flat(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reset_index() if frame.index.name is not None else frame.copy()


def _parse_osi(raw_symbol: object, session: str) -> tuple[float, str]:
    match = OSI.fullmatch(str(raw_symbol).strip())
    if match is None:
        raise QuoteNormalizationError(f"{session}: invalid SPXW OSI symbol {raw_symbol!r}")
    expiry = pd.to_datetime(match.group("expiry"), format="%y%m%d").strftime("%Y-%m-%d")
    if expiry != session:
        raise QuoteNormalizationError(
            f"{session}: non-same-day expiry {expiry} in saved backfill row"
        )
    return float(match.group("strike")) / 1000.0, match.group("right")


def definition_map(definitions: pd.DataFrame, session: str) -> dict[str, Definition]:
    """Build a one-to-one current-session definition map for CBBO rows."""

    frame = _flat(definitions)
    required = {"symbol", "instrument_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise QuoteNormalizationError(f"{session}: definition columns missing {missing}")
    mapping: dict[str, Definition] = {}
    for row in frame.loc[:, ["symbol", "instrument_id"]].drop_duplicates().to_dict("records"):
        raw_symbol = str(row["symbol"]).strip()
        strike, right = _parse_osi(raw_symbol, session)
        try:
            instrument_id = int(row["instrument_id"])
        except (TypeError, ValueError) as exc:
            raise QuoteNormalizationError(f"{session}: invalid definition instrument ID") from exc
        contract_id = f"SPXW-{session.replace('-', '')}-{strike:09.3f}-{right}"
        definition = Definition(
            raw_symbol=raw_symbol,
            contract_id=contract_id,
            instrument_id=instrument_id,
            expiry=session,
            strike=strike,
            right=right,
        )
        for key in (raw_symbol, str(instrument_id)):
            previous = mapping.get(key)
            if previous is not None and previous != definition:
                raise QuoteNormalizationError(f"{session}: conflicting definition identity for {key}")
            mapping[key] = definition
    if not mapping:
        raise QuoteNormalizationError(f"{session}: no valid same-day SPXW definitions")
    return mapping


def _quote_timestamp(frame: pd.DataFrame, session: str) -> pd.Series:
    for column in ("ts_recv", "event_time", "ts_event", "timestamp"):
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if parsed.notna().all():
                return parsed
    raise QuoteNormalizationError(f"{session}: CBBO rows have no complete receive timestamp")


def _parity_spot(frame: pd.DataFrame) -> pd.Series:
    """Average near-the-money put/call parity at each completed minute."""

    valid = frame[(frame["bid"] > 0.0) & (frame["ask"] > frame["bid"])].copy()
    if valid.empty:
        return pd.Series(dtype=float)
    pivot = valid.pivot_table(
        index="minute", columns=["strike", "right"], values="mid", aggfunc="last"
    )
    calls = pivot.loc[:, pivot.columns.get_level_values("right") == "C"].copy()
    puts = pivot.loc[:, pivot.columns.get_level_values("right") == "P"].copy()
    calls.columns = calls.columns.get_level_values("strike")
    puts.columns = puts.columns.get_level_values("strike")
    shared = calls.columns.intersection(puts.columns)
    if len(shared) < MIN_PAIRED_STRIKES:
        return pd.Series(index=pivot.index, dtype=float)
    implied = (calls[shared] - puts[shared]).add(pd.Series(shared, index=shared))
    coarse = implied.median(axis=1)
    distances = np.abs(implied.columns.to_numpy(float) - coarse.to_numpy()[:, None])
    near = implied.where(distances <= PARITY_WINDOW_POINTS)
    enough = near.notna().sum(axis=1) >= MIN_PAIRED_STRIKES
    return near.mean(axis=1).where(enough)


def normalize_session(
    definitions: pd.DataFrame,
    cbbo: pd.DataFrame,
    *,
    session: str,
) -> pd.DataFrame:
    """Normalize one acquired session; all retained quotes are provably 0DTE."""

    mapping = definition_map(definitions, session)
    frame = _flat(cbbo)
    required = {"symbol", "instrument_id", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise QuoteNormalizationError(f"{session}: CBBO columns missing {missing}")
    timestamps = _quote_timestamp(frame, session)
    records = []
    for position, row in enumerate(frame.to_dict("records")):
        raw_symbol = str(row["symbol"]).strip()
        definition = mapping.get(raw_symbol) or mapping.get(str(row["instrument_id"]))
        if definition is None:
            raise QuoteNormalizationError(f"{session}: CBBO contract lacks a same-session definition")
        # Re-parse at the join boundary: a malformed filter output cannot become
        # a trainable future expiry merely because it shares an instrument ID.
        strike, right = _parse_osi(raw_symbol, session)
        if (strike, right) != (definition.strike, definition.right):
            raise QuoteNormalizationError(f"{session}: quote/definition geometry mismatch")
        bid = pd.to_numeric(pd.Series([row["bid_px_00"]]), errors="coerce").iloc[0]
        ask = pd.to_numeric(pd.Series([row["ask_px_00"]]), errors="coerce").iloc[0]
        bid_size = pd.to_numeric(pd.Series([row["bid_sz_00"]]), errors="coerce").iloc[0]
        ask_size = pd.to_numeric(pd.Series([row["ask_sz_00"]]), errors="coerce").iloc[0]
        bid = float(bid) if pd.notna(bid) else np.nan
        ask = float(ask) if pd.notna(ask) else np.nan
        records.append(
            {
                "event_time": timestamps.iloc[position],
                "contract_id": definition.contract_id,
                "raw_symbol": definition.raw_symbol,
                "instrument_id": definition.instrument_id,
                "expiry": definition.expiry,
                "strike": definition.strike,
                "right": definition.right,
                "bid": bid,
                "ask": ask,
                "bid_size": float(bid_size) if pd.notna(bid_size) else 0.0,
                "ask_size": float(ask_size) if pd.notna(ask_size) else 0.0,
                "mid": (bid + ask) / 2.0 if np.isfinite(bid) and np.isfinite(ask) else np.nan,
                "quote_age_ms": 0.0,
                "volume": np.nan,
                "open_interest": np.nan,
            }
        )
    result = pd.DataFrame(records)
    if result.empty:
        raise QuoteNormalizationError(f"{session}: zero 0DTE CBBO rows")
    result["minute"] = (
        pd.to_datetime(result["event_time"], utc=True)
        .dt.tz_convert(ET)
        .dt.strftime("%H:%M")
    )
    result = result[result["minute"].between(RTH_FIRST, RTH_LAST)].copy()
    if result.empty:
        raise QuoteNormalizationError(f"{session}: no regular-session CBBO rows")
    result = result.sort_values(["minute", "contract_id", "event_time"]).drop_duplicates(
        ["minute", "contract_id"], keep="last"
    )
    result["underlying_price"] = result["minute"].map(_parity_spot(result))
    return result.drop(columns="minute").reset_index(drop=True)


def run(
    *,
    raw_root: Path,
    output_root: Path,
    acquisition_receipt: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Normalize every acquired pair and classify every structural failure."""

    if receipt_path.exists():
        raise QuoteNormalizationError(f"refusing to overwrite receipt: {receipt_path}")
    acquisition = json.loads(acquisition_receipt.read_text())
    expected = acquisition.get("completion", {}).get("expected_files")
    if acquisition.get("completion", {}).get("recorded_files") != expected:
        raise QuoteNormalizationError("acquisition receipt is incomplete")
    sessions = sorted({str(row["session"]) for row in acquisition.get("files", [])})
    results = []
    for session in sessions:
        definition_path = raw_root / "raw/databento/opra_spxw_definition" / f"{session}.definition.parquet"
        cbbo_path = raw_root / "raw/databento/opra_spxw_cbbo_1m" / f"{session}.cbbo-1m.parquet"
        output = output_root / f"databento_spxw_0dte_{session}.parquet"
        try:
            if output.exists():
                raise QuoteNormalizationError("normalized output already exists")
            normalized = normalize_session(
                pd.read_parquet(definition_path), pd.read_parquet(cbbo_path), session=session
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(output, index=False)
            results.append(
                {
                    "session": session,
                    "classification": "NORMALIZED",
                    "rows": int(len(normalized)),
                    "rth_minutes_with_parity_spot": int(
                        pd.to_datetime(normalized["event_time"], utc=True)
                        .dt.tz_convert(ET)
                        .dt.strftime("%H:%M")
                        .where(normalized["underlying_price"].notna())
                        .nunique()
                    ),
                    "path": str(output),
                    "sha256": file_sha256(output),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "session": session,
                    "classification": "DEGRADED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "definition_path": str(definition_path),
                    "cbbo_path": str(cbbo_path),
                }
            )
    receipt: dict[str, Any] = {
        "schema_version": "v5.lifecycle-quote-normalization.v1",
        "acquisition_receipt": str(acquisition_receipt),
        "acquisition_receipt_sha256": file_sha256(acquisition_receipt),
        "implementation_sha256": file_sha256(Path(__file__)),
        "source_raw_root": str(raw_root),
        "output_root": str(output_root),
        "sessions": results,
        "summary": {
            "acquired": len(sessions),
            "normalized": int(sum(row["classification"] == "NORMALIZED" for row in results)),
            "degraded": int(sum(row["classification"] == "DEGRADED" for row in results)),
            "all_retained_contracts_proven_same_day": True,
        },
    }
    receipt["receipt_sha256"] = _payload_sha256(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(
        raw_root=args.raw_root,
        output_root=args.output_root,
        acquisition_receipt=args.acquisition_receipt,
        receipt_path=args.receipt,
    )
    print(json.dumps(receipt["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
