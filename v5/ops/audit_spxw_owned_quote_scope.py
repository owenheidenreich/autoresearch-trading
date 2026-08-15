"""Audit owned SPXW quote expirations and debit-vertical entry feasibility.

This audit is deliberately outcome blind.  It reads contract definitions,
quoted instrument identifiers, frozen selected-entry keys, and entry-time
quotes.  It never reads a future option price or P&L column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD


SCHEMA = "v5.spxw-owned-quote-scope-audit.v1"
WIDTHS = (5, 10, 15, 20, 25, 30)
STARTING_EQUITY_USD = 10_000.0
MAX_LOSS_SHARE = 0.05


class QuoteScopeAuditError(RuntimeError):
    """The owned quote inventory cannot support an unambiguous audit."""


def _aggregate_manifest(paths: Iterable[Path]) -> tuple[int, str]:
    rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(paths)
    ]
    return len(rows), hashlib.sha256(canonical_json(rows)).hexdigest()


def quoted_expirations(quote_path: Path, definition_path: Path) -> dict[str, Any]:
    """Map every quoted instrument to exactly one definition expiry."""

    session = quote_path.name.split(".", 1)[0]
    quote_ids = pd.read_parquet(quote_path, columns=["instrument_id"])[
        "instrument_id"
    ].drop_duplicates()
    definitions = pd.read_parquet(
        definition_path, columns=["instrument_id", "expiration"]
    ).drop_duplicates()
    definitions["expiration"] = pd.to_datetime(
        definitions["expiration"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    if definitions["expiration"].isna().any():
        raise QuoteScopeAuditError(f"{session}: definition has invalid expiration")
    conflicts = definitions.groupby("instrument_id")["expiration"].nunique()
    if conflicts.gt(1).any():
        raise QuoteScopeAuditError(f"{session}: instrument maps to multiple expirations")
    mapping = definitions.drop_duplicates("instrument_id").set_index("instrument_id")[
        "expiration"
    ]
    expirations = mapping.reindex(quote_ids)
    if expirations.isna().any():
        raise QuoteScopeAuditError(f"{session}: quoted instrument lacks a definition")
    counts = expirations.value_counts().sort_index()
    return {
        "session": session,
        "quoted_instruments": int(len(quote_ids)),
        "quoted_expirations": [str(value) for value in counts.index],
        "same_day_only": bool(len(counts) == 1 and str(counts.index[0]) == session),
        "future_expiration_instruments": int(
            counts[[str(value) > session for value in counts.index]].sum()
        ),
    }


def vertical_entry_feasibility(
    selected: pd.DataFrame,
    ladder: pd.DataFrame,
    *,
    widths: Iterable[int] = WIDTHS,
) -> list[dict[str, Any]]:
    """Count live exact-width short legs without opening any future outcome."""

    selected_columns = {"session", "entry_minute", "contract_id", "fold"}
    ladder_columns = {
        "session",
        "minute",
        "contract_id",
        "strike",
        "right",
        "bid",
        "ask",
    }
    if not selected_columns.issubset(selected.columns):
        raise QuoteScopeAuditError("selected-entry key columns are incomplete")
    if not ladder_columns.issubset(ladder.columns):
        raise QuoteScopeAuditError("entry ladder columns are incomplete")
    keys = ["session", "entry_minute", "contract_id"]
    if selected.duplicated(keys).any():
        raise QuoteScopeAuditError("selected-entry key is duplicated")
    long_rows = selected[list(selected_columns)].merge(
        ladder[list(ladder_columns)],
        left_on=["session", "entry_minute", "contract_id"],
        right_on=["session", "minute", "contract_id"],
        how="left",
        validate="one_to_one",
    )
    if long_rows[["strike", "right", "bid", "ask"]].isna().any().any():
        raise QuoteScopeAuditError("a frozen selected entry is absent from the ladder")

    short = ladder[list(ladder_columns)].rename(
        columns={
            "minute": "entry_minute",
            "contract_id": "short_contract_id",
            "strike": "short_strike",
            "right": "short_right",
            "bid": "short_bid",
            "ask": "short_ask",
        }
    )
    if short.duplicated(
        ["session", "entry_minute", "short_strike", "short_right"]
    ).any():
        raise QuoteScopeAuditError("entry ladder has duplicate strike/right rows")

    results = []
    maximum_loss = STARTING_EQUITY_USD * MAX_LOSS_SHARE
    fees = 2 * FEES_PER_ROUND_TRIP_USD
    for width in widths:
        value = long_rows.copy()
        value["short_strike"] = value["strike"] + value["right"].map(
            {"C": float(width), "P": -float(width)}
        )
        if value["short_strike"].isna().any():
            raise QuoteScopeAuditError("selected entry has invalid option right")
        paired = value.merge(
            short[
                [
                    "session",
                    "entry_minute",
                    "short_contract_id",
                    "short_strike",
                    "short_right",
                    "short_bid",
                    "short_ask",
                ]
            ],
            left_on=["session", "entry_minute", "short_strike", "right"],
            right_on=["session", "entry_minute", "short_strike", "short_right"],
            how="left",
            validate="one_to_one",
        )
        paired = paired[paired["short_contract_id"].notna()].copy()
        paired["entry_max_loss_usd"] = (
            (paired["ask"] - paired["short_bid"]) * 100.0 + fees
        )
        risk_eligible = paired[
            paired["entry_max_loss_usd"].gt(0.0)
            & paired["entry_max_loss_usd"].le(maximum_loss)
        ]
        fold_counts = {
            str(int(fold)): int(count)
            for fold, count in risk_eligible.groupby("fold", sort=True).size().items()
        }
        results.append(
            {
                "width_points": int(width),
                "frozen_entries": int(len(selected)),
                "exact_live_pairs": int(len(paired)),
                "risk_eligible_pairs": int(len(risk_eligible)),
                "risk_eligible_folds": sorted(fold_counts),
                "risk_eligible_fold_count": int(len(fold_counts)),
                "risk_eligible_by_fold": fold_counts,
                "four_of_five_chronology_attainable": bool(len(fold_counts) >= 4),
                "median_entry_max_loss_usd": (
                    float(risk_eligible["entry_max_loss_usd"].median())
                    if len(risk_eligible)
                    else None
                ),
            }
        )
    return results


def run(
    *,
    quote_root: Path,
    definition_root: Path,
    selected_path: Path,
    ladder_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise QuoteScopeAuditError("refusing to overwrite scope-audit receipt")
    quote_paths = sorted(quote_root.glob("*.cbbo-1m.parquet"))
    if not quote_paths:
        raise QuoteScopeAuditError("no owned CBBO-1m quote files found")
    definition_paths = [
        definition_root / f"{path.name.split('.', 1)[0]}.definition.parquet"
        for path in quote_paths
    ]
    missing = [str(path) for path in definition_paths if not path.is_file()]
    if missing:
        raise QuoteScopeAuditError(f"missing definition files: {missing[:3]}")

    sessions = [
        quoted_expirations(quote_path, definition_path)
        for quote_path, definition_path in zip(quote_paths, definition_paths, strict=True)
    ]
    selected = pd.read_parquet(
        selected_path, columns=["session", "entry_minute", "contract_id", "fold"]
    )
    ladder = pd.read_parquet(
        ladder_path,
        columns=["session", "minute", "contract_id", "strike", "right", "bid", "ask"],
    )
    verticals = vertical_entry_feasibility(selected, ladder)
    quote_count, quote_manifest_sha256 = _aggregate_manifest(quote_paths)
    definition_count, definition_manifest_sha256 = _aggregate_manifest(definition_paths)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_on": "2026-08-14",
        "purpose": "outcome-blind inventory of owned SPXW quote expirations and frozen-selector debit-vertical attainability",
        "owned_quote_inventory": {
            "sessions": len(sessions),
            "first_session": sessions[0]["session"],
            "last_session": sessions[-1]["session"],
            "same_day_only_sessions": int(sum(row["same_day_only"] for row in sessions)),
            "sessions_with_future_expiration_quotes": int(
                sum(row["future_expiration_instruments"] > 0 for row in sessions)
            ),
            "future_expiration_quoted_instruments": int(
                sum(row["future_expiration_instruments"] for row in sessions)
            ),
            "all_quoted_contracts_are_0dte": bool(all(row["same_day_only"] for row in sessions)),
        },
        "debit_vertical_translation": {
            "frozen_selector_entries": int(len(selected)),
            "widths": verticals,
            "maximum_fold_coverage": max(row["risk_eligible_fold_count"] for row in verticals),
            "four_of_five_chronology_attainable_for_any_audited_width": bool(
                any(row["four_of_five_chronology_attainable"] for row in verticals)
            ),
            "decision": "DO_NOT_OPEN_ECONOMICS_UNSATISFIABLE_CHRONOLOGY",
        },
        "inputs": {
            "quote_root": str(quote_root),
            "quote_files": quote_count,
            "quote_manifest_sha256": quote_manifest_sha256,
            "definition_root": str(definition_root),
            "definition_files": definition_count,
            "definition_manifest_sha256": definition_manifest_sha256,
            "selected_path": str(selected_path),
            "selected_sha256": file_sha256(selected_path),
            "ladder_path": str(ladder_path),
            "ladder_sha256": file_sha256(ladder_path),
        },
        "integrity": {
            "future_option_prices_read": False,
            "pnl_columns_read": False,
            "entry_quotes_only_for_verticals": True,
            "model_fit": False,
            "economic_cell_opened": False,
            "reserved_sessions_used": False,
        },
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quote-root", type=Path, required=True)
    parser.add_argument("--definition-root", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        quote_root=args.quote_root,
        definition_root=args.definition_root,
        selected_path=args.selected,
        ladder_path=args.ladder,
        output_path=args.output,
    )
    print(json.dumps({
        "owned_quote_inventory": payload["owned_quote_inventory"],
        "debit_vertical_translation": payload["debit_vertical_translation"],
        "receipt_sha256": payload["receipt_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
