"""Run the preregistered 15:00 five-point iron-fly strategic comparison."""
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
from v5.research.defined_risk_iron_fly import evaluate_session


DECLARATION_SCHEMA = "v5.defined-risk-iron-fly-declaration.v1"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202_608_14_1500
MULTIPLICITY_FAMILY = 33


def _verified(path: Path, schema: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"self-hash mismatch: {path}")
    if schema is not None and value.get("schema_version") != schema:
        raise RuntimeError(f"schema mismatch: {path}")
    return value


def _one_sided_lcb(values: np.ndarray) -> float:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(values)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = values[rng.integers(0, n, size=n)].mean()
    return float(np.quantile(draws, 0.05 / MULTIPLICITY_FAMILY))


def run(
    *,
    declaration_path: Path,
    quote_root: Path,
    coverage_path: Path,
    settlement_receipt_path: Path,
    out_root: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    if out_root.exists() or evidence_dir.exists():
        raise RuntimeError("refusing to overwrite iron-fly output or evidence")
    declaration = _verified(declaration_path, DECLARATION_SCHEMA)
    for relative, digest in declaration["implementation_hashes"].items():
        path = Path(relative)
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"iron-fly implementation hash mismatch: {relative}")
    if file_sha256(coverage_path) != declaration["inputs"]["coverage_sha256"]:
        raise RuntimeError("iron-fly coverage input drift")
    if file_sha256(settlement_receipt_path) != declaration["inputs"]["settlement_receipt_sha256"]:
        raise RuntimeError("iron-fly settlement input drift")
    coverage = pd.read_csv(coverage_path)
    sessions = sorted(
        coverage.loc[coverage["included_for_episode_build"], "session"].astype(str)
    )
    if len(sessions) != 243:
        raise RuntimeError("iron-fly comparison requires the same 243 sessions")
    settlements = load_validated_settlements(settlement_receipt_path)
    rows = []
    for session in sessions:
        path = quote_root / f"databento_spxw_0dte_{session}.parquet"
        raw = pd.read_parquet(path, columns=list(QUOTE_COLUMNS))
        quotes = prepare_quotes(raw, session)
        rows.append(evaluate_session(quotes, session=session, settlement_spx=settlements[session]).row)
    result = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
    if len(result) != 243 or result["session"].duplicated().any():
        raise RuntimeError("iron-fly session population drift")
    out_root.mkdir(parents=True, exist_ok=False)
    table_path = out_root / "session_results.parquet"
    result.to_parquet(table_path, index=False)

    net = result["net_touch_usd"].to_numpy(float)
    mid = result["gross_mid_usd"].to_numpy(float)
    folds = []
    for number, indices in enumerate(np.array_split(np.arange(len(result)), 5), 1):
        part = result.iloc[indices]
        folds.append(
            {
                "fold": number,
                "first_session": str(part["session"].min()),
                "last_session": str(part["session"].max()),
                "sessions": len(part),
                "trades": int(part["traded"].sum()),
                "mean_net_per_session_usd": float(part["net_touch_usd"].mean()),
                "mean_gross_mid_per_session_usd": float(part["gross_mid_usd"].mean()),
            }
        )
    traded = result[result["traded"]]
    positive_folds = sum(row["mean_net_per_session_usd"] > 0.0 for row in folds)
    lcb = _one_sided_lcb(net)
    payload: dict[str, Any] = {
        "schema_version": "v5.defined-risk-iron-fly.v1",
        "created_on": "2026-08-14",
        "declaration": {"path": str(declaration_path), "sha256": file_sha256(declaration_path)},
        "population": {
            "sessions": len(result),
            "trades": int(result["traded"].sum()),
            "abstentions": int((~result["traded"]).sum()),
        },
        "primary": {
            "mean_net_touch_per_session_usd": float(net.mean()),
            "mean_gross_mid_per_session_usd": float(mid.mean()),
            "mean_net_touch_per_trade_usd": float(traded["net_touch_usd"].mean()) if len(traded) else None,
            "median_net_touch_per_trade_usd": float(traded["net_touch_usd"].median()) if len(traded) else None,
            "win_rate": float(traded["net_touch_usd"].gt(0.0).mean()) if len(traded) else None,
            "worst_trade_usd": float(traded["net_touch_usd"].min()) if len(traded) else None,
            "largest_declared_max_loss_usd": float(traded["declared_max_loss_usd"].max()) if len(traded) else None,
            "multiplicity_family": MULTIPLICITY_FAMILY,
            "one_sided_corrected_lcb_per_session_usd": lcb,
            "positive_chronological_folds": positive_folds,
            "folds_required": 4,
        },
        "controls": {
            "naked_short_mean_net_per_session_usd": float(result["naked_short_net_touch_usd"].mean()),
            "long_straddle_mean_net_per_session_usd": float(result["long_straddle_net_touch_usd"].mean()),
        },
        "chronological_folds": folds,
        "decision": (
            "STRATEGIC_BREAKTHROUGH_DEFINED_RISK"
            if net.mean() > 0.0 and lcb > 0.0 and positive_folds >= 4
            else "DEFINED_RISK_PRIMARY_DOES_NOT_CLEAR"
        ),
        "artifact": {"path": str(table_path), "rows": len(result), "sha256": file_sha256(table_path)},
        "integrity": {
            "model_fit": False,
            "entry_strike_uses_entry_snapshot_only": True,
            "touch_prices_all_legs": True,
            "four_round_trip_fees": True,
            "abstentions_count_as_zero_sessions": True,
            "reserved_sessions_used": False,
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
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--settlement-receipt", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        declaration_path=args.declaration,
        quote_root=args.quotes,
        coverage_path=args.coverage,
        settlement_receipt_path=args.settlement_receipt,
        out_root=args.out_root,
        evidence_dir=args.evidence_dir,
    )
    print(json.dumps({"decision": payload["decision"], "primary": payload["primary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
