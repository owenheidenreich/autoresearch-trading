"""Exercise the compact shared lifecycle on real causal observations, unfitted.

Only feature arrays, masks and identities are opened.  The cached outcome
target member is deliberately never accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import cos, pi, sin
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256, minute_number
from v5.research.causal_day_architectures import CausalPolicyBatch
from v5.research.causal_day_compact_shared_lifecycle import (
    ARCHITECTURE_NAME,
    CompactSharedLifecyclePolicy,
    computed_parameter_count,
)


SCHEMA = "v5.compact-shared-lifecycle-real-interface.v1"
AUDIT_MINUTES = ("10:00", "13:30", "15:00")


def _scaled(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - mean) / scale, -10.0, 10.0).astype(np.float32)


def _batch_from_feature_cache(
    path: Path,
    *,
    minute: str,
    scaler: dict[str, np.ndarray],
    role: str,
) -> tuple[CausalPolicyBatch, dict[str, int]]:
    with np.load(path, allow_pickle=False) as value:
        # Do not access value["targets"]. This audit is feature/interface only.
        minutes = value["minutes"].astype(str)
        matches = np.flatnonzero(minutes == minute)
        if len(matches) != 1:
            raise RuntimeError(f"{path.stem}: {minute} is missing or duplicated")
        index = int(matches[0])
        candle_count = int(value["candle_lengths"][index])
        candles = _scaled(
            value["candles"][:candle_count], scaler["candle_mean"], scaler["candle_scale"]
        )
        offsets = value["ladder_offsets"]
        start, stop = int(offsets[index]), int(offsets[index + 1])
        ladder = _scaled(
            value["ladder"][start:stop], scaler["ladder_mean"], scaler["ladder_scale"]
        )
        actions = value["action_mask"][start:stop].astype(bool)
    current = minute_number(minute)
    elapsed = (current - 570) / 390.0
    angle = 2.0 * pi * elapsed
    clock = np.asarray(
        [elapsed, (960 - current) / 390.0, current < 766, sin(angle), cos(angle)],
        np.float32,
    )
    position = np.zeros((1, 10), np.float32)
    if role.endswith("exit"):
        position[0, [0, 1, 3, 7, 9]] = [1.0, 1.0, 0.04, 0.25, 1.0]
        actions[:] = False
    batch = CausalPolicyBatch(
        candles=torch.from_numpy(candles).unsqueeze(0),
        candle_mask=torch.ones((1, candle_count), dtype=torch.bool),
        ladder=torch.from_numpy(ladder).unsqueeze(0),
        ladder_mask=torch.ones((1, len(ladder)), dtype=torch.bool),
        entry_action_mask=torch.from_numpy(actions.copy()).unsqueeze(0),
        account=torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        position=torch.from_numpy(position),
        clock=torch.from_numpy(clock).unsqueeze(0),
        roles=(role,),
    )
    return batch, {
        "completed_candles": candle_count,
        "visible_contracts": len(ladder),
        "eligible_actions": int(actions.sum()),
    }


def _assert_mask_invariance(model: CompactSharedLifecyclePolicy, batch: CausalPolicyBatch) -> None:
    future_candle = torch.full((1, 1, batch.candles.shape[-1]), 1e6)
    invisible_contract = torch.full((1, 1, batch.ladder.shape[-1]), -1e6)
    mutated = CausalPolicyBatch(
        candles=torch.cat((batch.candles, future_candle), dim=1),
        candle_mask=torch.cat((batch.candle_mask, torch.zeros((1, 1), dtype=torch.bool)), dim=1),
        ladder=torch.cat((batch.ladder, invisible_contract), dim=1),
        ladder_mask=torch.cat((batch.ladder_mask, torch.zeros((1, 1), dtype=torch.bool)), dim=1),
        entry_action_mask=torch.cat(
            (batch.entry_action_mask, torch.zeros((1, 1), dtype=torch.bool)), dim=1
        ),
        account=batch.account,
        position=batch.position,
        clock=batch.clock,
        roles=batch.roles,
    )
    original = model(batch)
    changed = model(mutated)
    if not torch.allclose(original.abstain_logits, changed.abstain_logits):
        raise RuntimeError("masked future candle/contract changed WAIT")
    if not torch.allclose(original.exit_logits, changed.exit_logits):
        raise RuntimeError("masked future candle/contract changed exit scores")
    finite = torch.isfinite(original.contract_logits)
    if not torch.allclose(original.contract_logits[finite], changed.contract_logits[:, :-1][finite]):
        raise RuntimeError("masked future candle/contract changed entry scores")


def run(
    *,
    cache_root: Path,
    scaler_path: Path,
    design_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("refusing to overwrite real-interface receipt")
    design = json.loads(design_receipt_path.read_text())
    if int(design["architecture"]["built_parameter_count"]) != computed_parameter_count():
        raise RuntimeError("design receipt and built parameter count differ")
    manifest_path = cache_root / "cache_manifest.parquet"
    manifest = pd.read_parquet(manifest_path).sort_values("session")
    if len(manifest) < 3:
        raise RuntimeError("real-interface audit requires first/middle/last sessions")
    selected = manifest.iloc[[0, len(manifest) // 2, -1]].copy()
    raw_scaler = json.loads(scaler_path.read_text())
    scaler = {name: np.asarray(raw_scaler[name], np.float32) for name in raw_scaler}
    torch.manual_seed(20260814)
    model = CompactSharedLifecyclePolicy().eval()
    cells: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        path = Path(str(row.path))
        if file_sha256(path) != str(row.sha256):
            raise RuntimeError(f"feature cache hash mismatch: {row.session}")
        for minute in AUDIT_MINUTES:
            entry_role = "morning_entry" if minute < "12:46" else "afternoon_entry"
            entry, dimensions = _batch_from_feature_cache(
                path, minute=minute, scaler=scaler, role=entry_role
            )
            entry_scores = model(entry)
            if not torch.isfinite(entry_scores.abstain_logits).all():
                raise RuntimeError("real entry WAIT score is non-finite")
            eligible = entry.entry_action_mask
            if eligible.any() and not torch.isfinite(entry_scores.contract_logits[eligible]).all():
                raise RuntimeError("real eligible entry score is non-finite")
            if dimensions["visible_contracts"] <= dimensions["eligible_actions"]:
                raise RuntimeError("real whole-chain context collapsed to eligible actions")
            _assert_mask_invariance(model, entry)

            exit_role = "morning_exit" if minute < "12:46" else "afternoon_exit"
            exit_batch, _ = _batch_from_feature_cache(
                path, minute=minute, scaler=scaler, role=exit_role
            )
            exit_scores = model(exit_batch)
            if not torch.isfinite(exit_scores.exit_logits).all():
                raise RuntimeError("real HOLD/SELL scores are non-finite")
            _assert_mask_invariance(model, exit_batch)
            cells.append({"session": str(row.session), "minute": minute, **dimensions})

    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_on": "2026-08-14",
        "architecture": ARCHITECTURE_NAME,
        "built_parameter_count": computed_parameter_count(),
        "cells": cells,
        "checks": {
            "first_middle_last_sessions": True,
            "morning_and_afternoon_entry_routes": True,
            "origin_owned_exit_routes": True,
            "whole_chain_exceeds_entry_actions": True,
            "masked_future_invariance": True,
            "finite_entry_wait_hold_sell_scores": True,
        },
        "inputs": {
            "cache_manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            "scaler": {"path": str(scaler_path), "sha256": file_sha256(scaler_path)},
            "design_receipt": {
                "path": str(design_receipt_path),
                "sha256": file_sha256(design_receipt_path),
            },
        },
        "integrity": {
            "target_array_accessed": False,
            "fit_performed": False,
            "economics_read": False,
            "vendor_contacted": False,
            "data_downloaded": False,
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
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--design-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        cache_root=args.cache_root,
        scaler_path=args.scaler,
        design_receipt_path=args.design_receipt,
        output_path=args.output,
    )
    print(json.dumps({"cells": len(payload["cells"]), "checks": payload["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
