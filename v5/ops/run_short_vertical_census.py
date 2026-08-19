"""Run the preregistered defined-risk short-vertical census and write receipts.

Owner-authorized 2026-08-15. The runner refuses to execute unless the
declaration's implementation hashes match the code about to run and the frozen
inputs match their declared digests, mirroring the iron-fly runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import (
    QUOTE_COLUMNS,
    canonical_json,
    file_sha256,
    load_validated_settlements,
    prepare_quotes,
)
from v5.ops.measure_fill_quality import QUOTE_CORPUS
from v5.research.capacity_campaign import stable_seed
from v5.research.short_vertical_census import (
    FAMILY,
    FAMILY_SIZE,
    evaluate_session,
)

DECLARATION_SCHEMA = "v5.short-vertical-census-declaration.v1"
BOOTSTRAP_DRAWS = 20_000
ALPHA = 0.05
EXECUTION_LAWS = ("net_touch_usd", "net_fee_only_usd")


def _verified(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value.get("declaration_sha256")
    unsigned = dict(value)
    unsigned.pop("declaration_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"declaration self-hash mismatch: {path}")
    if value.get("schema_version") != DECLARATION_SCHEMA:
        raise RuntimeError(f"declaration schema mismatch: {path}")
    return value


def _corrected_lcb(values: np.ndarray, *, seed: int) -> float:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, ALPHA / FAMILY_SIZE))


def summarize_cells(result: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-cell means, corrected bounds and chronological fold signs.

    Folds split session *indices* — `np.array_split` on a DataFrame itself
    returns raw ndarrays whose column access fails, the defect that voided the
    first run before any outcome was read.
    """

    cells: list[dict[str, Any]] = []
    for cell in FAMILY:
        name = str(cell["cell"])
        part = result[result["cell"].eq(name)].sort_values("session")
        summary: dict[str, Any] = {
            **cell,
            "sessions": len(part),
            "trades": int(part["traded"].sum()),
        }
        for law in EXECUTION_LAWS:
            values = part[law].to_numpy(float)
            fold_means = [
                float(part.iloc[indices][law].mean())
                for indices in np.array_split(np.arange(len(part)), 5)
            ]
            law_key = law.replace("_usd", "")
            summary[f"{law_key}_mean_per_session_usd"] = float(values.mean())
            summary[f"{law_key}_corrected_lcb_usd"] = _corrected_lcb(
                values, seed=stable_seed("vertical-census", name, law)
            )
            summary[f"{law_key}_positive_folds"] = int(
                sum(mean > 0.0 for mean in fold_means)
            )
        summary["net_passive_diag_mean_per_session_usd"] = float(
            part["net_passive_diag_usd"].mean()
        )
        summary["clears_fee_only"] = bool(
            summary["net_fee_only_mean_per_session_usd"] > 0.0
            and summary["net_fee_only_corrected_lcb_usd"] > 0.0
            and summary["net_fee_only_positive_folds"] >= 4
        )
        summary["clears_touch"] = bool(
            summary["net_touch_mean_per_session_usd"] > 0.0
            and summary["net_touch_corrected_lcb_usd"] > 0.0
            and summary["net_touch_positive_folds"] >= 4
        )
        cells.append(summary)
    return cells


def run(
    *,
    declaration_path: Path,
    quote_root: Path,
    out_root: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    if out_root.exists() or evidence_dir.exists():
        raise RuntimeError("refusing to overwrite census output or evidence")
    declaration = _verified(declaration_path)
    for relative, digest in declaration["implementation_hashes"].items():
        path = Path(relative)
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"census implementation hash mismatch: {relative}")
    coverage_path = Path(declaration["inputs"]["coverage_path"])
    settlement_path = Path(declaration["inputs"]["settlement_receipt_path"])
    if file_sha256(coverage_path) != declaration["inputs"]["coverage_sha256"]:
        raise RuntimeError("census coverage input drift")
    if file_sha256(settlement_path) != declaration["inputs"]["settlement_receipt_sha256"]:
        raise RuntimeError("census settlement input drift")

    coverage = pd.read_csv(coverage_path)
    sessions = sorted(
        coverage.loc[coverage["included_for_episode_build"], "session"].astype(str)
    )
    if len(sessions) != 243:
        raise RuntimeError("census requires the same 243 coverage-included sessions")
    settlements = load_validated_settlements(settlement_path)

    rows: list[dict[str, Any]] = []
    for session in sessions:
        path = quote_root / f"databento_spxw_0dte_{session}.parquet"
        raw = pd.read_parquet(path, columns=list(QUOTE_COLUMNS))
        quotes = prepare_quotes(raw, session)
        rows.extend(
            evaluate_session(
                quotes, session=session, settlement_spx=settlements[session]
            ).rows
        )
    result = pd.DataFrame(rows).sort_values(["cell", "session"]).reset_index(drop=True)
    expected_rows = 243 * FAMILY_SIZE
    if len(result) != expected_rows:
        raise RuntimeError(
            f"census population drift: {len(result)} rows against {expected_rows}"
        )

    out_root.mkdir(parents=True, exist_ok=False)
    table_path = out_root / "session_cell_results.parquet"
    result.to_parquet(table_path, index=False)

    cells = summarize_cells(result)
    fee_only_pass = [c["cell"] for c in cells if c["clears_fee_only"]]
    touch_pass = [c["cell"] for c in cells if c["clears_touch"]]
    if not fee_only_pass:
        decision = "STOP_CONDITION_MET_NO_CELL_CLEARS_FEE_ONLY"
    elif touch_pass:
        decision = "SELLER_MECHANISM_CLEARS_AT_TOUCH"
    else:
        decision = "SELLER_MECHANISM_CLEARS_FEE_ONLY_ONLY"

    payload: dict[str, Any] = {
        "schema_version": "v5.short-vertical-census.v1",
        "created_on": "2026-08-15",
        "declaration": {
            "path": str(declaration_path),
            "sha256": declaration["declaration_sha256"],
        },
        "population": {
            "sessions": 243,
            "family_size": FAMILY_SIZE,
            "rows": len(result),
            "total_trades": int(result["traded"].sum()),
        },
        "cells": cells,
        "decision": decision,
        "fee_only_clearing_cells": fee_only_pass,
        "touch_clearing_cells": touch_pass,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "alpha": ALPHA,
            "correction_family": FAMILY_SIZE,
            "seed_law": "stable_seed('vertical-census', cell, law)",
        },
        "artifact": {
            "path": str(table_path),
            "rows": len(result),
            "sha256": file_sha256(table_path),
        },
        "integrity": {
            "model_fit": False,
            "entry_strikes_use_entry_snapshot_only": True,
            "hold_to_validated_cash_settlement": True,
            "abstentions_count_as_zero_sessions": True,
            "max_loss_capped_at_5pct_of_account": True,
            "reserved_sessions_used": False,
            "vendor_contacted": False,
            "money_spent": False,
            "orders_submitted": False,
        },
        "implementation_hashes": declaration["implementation_hashes"],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    receipt_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out_root / "receipt.json").write_text(receipt_text)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    (evidence_dir / "receipt.json").write_text(receipt_text)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--quotes", type=Path, default=QUOTE_CORPUS)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        declaration_path=args.declaration,
        quote_root=args.quotes,
        out_root=args.out_root,
        evidence_dir=args.evidence_dir,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "fee_only_clearing_cells": payload["fee_only_clearing_cells"],
                "touch_clearing_cells": payload["touch_clearing_cells"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
