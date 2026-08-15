"""Materialize the full-chain, whole-prefix cache before the reopened fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import (
    QUOTE_COLUMNS,
    canonical_json,
    file_sha256,
    full_ladder_state,
    prepare_quotes,
)
from v5.ops.measure_fill_quality import QUOTE_CORPUS
from v5.research.causal_day_fit_cache import cache_path
from v5.research.causal_day_magnitude import HORIZONS
from v5.research.causal_day_tensorizer import (
    CANDLE_FEATURES,
    LADDER_FEATURES,
    _candle_feature_frame,
    _ladder_feature_frame,
)


DEFAULT_DATASET_RECEIPT = Path(
    "v4/audit/autoresearch/causal_day_dataset_settlement_validated_2026_08_14/receipt.json"
)
DEFAULT_COVERAGE = Path(
    "v4/audit/autoresearch/causal_day_trader_coverage_2026_08_14_attempt002/session_coverage.csv"
)


def _write_session(
    session: str,
    *,
    quote_root: Path,
    candles: pd.DataFrame,
    candidates: pd.DataFrame,
    out_root: Path,
) -> dict[str, int | str]:
    quote_path = quote_root / f"databento_spxw_0dte_{session}.parquet"
    raw = pd.read_parquet(quote_path, columns=list(QUOTE_COLUMNS))
    quotes = prepare_quotes(raw, session)
    chain, _ = full_ladder_state(quotes, session)
    chain = chain[chain["minute"].between("09:35", "15:00")].copy()
    chain = chain.sort_values(["minute", "right", "strike", "contract_id"], kind="mergesort")
    minutes = np.asarray(sorted(chain["minute"].unique()), dtype="U5")
    if len(minutes) != 326 or minutes[0] != "09:35" or minutes[-1] != "15:00":
        raise RuntimeError(f"{session}: full-chain entry clock is incomplete")
    ladder_values = _ladder_feature_frame(chain).loc[:, LADDER_FEATURES].to_numpy(np.float32)
    if not np.isfinite(ladder_values).all():
        raise RuntimeError(f"{session}: full-chain cache contains non-finite features")
    groups = chain.groupby("minute", sort=True).size().reindex(minutes).to_numpy(int)
    offsets = np.r_[0, np.cumsum(groups)].astype(np.int64)
    action = chain["entry_eligible"].fillna(False).to_numpy(bool)
    ids = chain["contract_id"].astype(str).to_numpy(dtype="U96")

    target = np.zeros((len(chain), len(HORIZONS)), np.float32)
    target[:] = np.nan
    candidate = candidates[candidates["session"].astype(str).eq(session)].copy()
    target_columns = [f"maximum_itm_depth_{horizon}m" for horizon in HORIZONS]
    keyed = candidate.set_index(["entry_minute", "contract_id"])[target_columns]
    keys = pd.MultiIndex.from_arrays(
        [chain["minute"].astype(str), chain["contract_id"].astype(str)]
    )
    aligned = keyed.reindex(keys).to_numpy(float)
    target[action] = aligned[action]
    if not np.isfinite(target[action]).all():
        raise RuntimeError(f"{session}: eligible action lacks a magnitude label")
    if int(action.sum()) != len(candidate):
        raise RuntimeError(f"{session}: candidate/cache action population differs")

    candle = candles[candles["session"].astype(str).eq(session)].sort_values("knowable_at")
    candle_values = _candle_feature_frame(candle).loc[:, CANDLE_FEATURES].to_numpy(np.float32)
    if candle_values.shape != (390, len(CANDLE_FEATURES)) or not np.isfinite(candle_values).all():
        raise RuntimeError(f"{session}: candle cache violates the complete prefix contract")
    candle_lengths = np.asarray(
        [int(value[:2]) * 60 + int(value[3:]) - 570 for value in minutes], dtype=np.int16
    )

    path = cache_path(out_root, session)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        session=np.asarray(session),
        minutes=minutes,
        candle_lengths=candle_lengths,
        candles=candle_values,
        ladder_offsets=offsets,
        ladder=ladder_values,
        action_mask=action,
        targets=target,
        contract_ids=ids,
    )
    return {
        "session": session,
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "minutes": len(minutes),
        "whole_chain_nodes": len(chain),
        "eligible_actions": int(action.sum()),
    }


def run(
    *,
    out_root: Path,
    quote_root: Path,
    dataset_receipt_path: Path,
    coverage_path: Path,
    declaration_path: Path,
    limit: int | None = None,
) -> dict:
    if out_root.exists():
        raise RuntimeError(f"refusing to overwrite fit cache: {out_root}")
    receipt = json.loads(dataset_receipt_path.read_text())
    candles = pd.read_parquet(receipt["outputs"]["candles"]["path"])
    candidates = pd.read_parquet(receipt["outputs"]["candidates"]["path"])
    coverage = pd.read_csv(coverage_path)
    sessions = sorted(
        coverage.loc[coverage["included_for_episode_build"], "session"].astype(str)
    )
    if limit is not None:
        sessions = sessions[:limit]
    out_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, session in enumerate(sessions, 1):
        rows.append(
            _write_session(
                session,
                quote_root=quote_root,
                candles=candles,
                candidates=candidates,
                out_root=out_root,
            )
        )
        if index % 10 == 0 or index == len(sessions):
            print(f"cached {index}/{len(sessions)} sessions", flush=True)
    manifest = pd.DataFrame(rows)
    manifest_path = out_root / "cache_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    payload = {
        "schema_version": "v5.causal-day-fit-cache.v1",
        "purpose": "whole completed prefix and whole live chain for the scoped magnitude fit",
        "declaration": {"path": str(declaration_path), "sha256": file_sha256(declaration_path)},
        "dataset_receipt": {
            "path": str(dataset_receipt_path),
            "sha256": file_sha256(dataset_receipt_path),
        },
        "sessions": len(sessions),
        "first_session": min(sessions),
        "last_session": max(sessions),
        "minutes": int(manifest["minutes"].sum()),
        "whole_chain_nodes": int(manifest["whole_chain_nodes"].sum()),
        "eligible_actions": int(manifest["eligible_actions"].sum()),
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    (out_root / "receipt.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out_root / "receipt.json")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--quotes", type=Path, default=QUOTE_CORPUS)
    parser.add_argument("--dataset-receipt", type=Path, default=DEFAULT_DATASET_RECEIPT)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(
        out_root=args.out_root,
        quote_root=args.quotes,
        dataset_receipt_path=args.dataset_receipt,
        coverage_path=args.coverage,
        declaration_path=args.declaration,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
