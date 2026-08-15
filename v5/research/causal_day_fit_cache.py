"""External full-chain cache used by the reopened causal-day comparison."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256, minute_number
from v5.research.causal_day_architectures import CausalPolicyBatch


@dataclass(frozen=True)
class FeatureScaler:
    candle_mean: np.ndarray
    candle_scale: np.ndarray
    ladder_mean: np.ndarray
    ladder_scale: np.ndarray

    def to_json(self) -> dict[str, list[float]]:
        return {
            "candle_mean": self.candle_mean.tolist(),
            "candle_scale": self.candle_scale.tolist(),
            "ladder_mean": self.ladder_mean.tolist(),
            "ladder_scale": self.ladder_scale.tolist(),
        }


@dataclass(frozen=True)
class CachedSession:
    session: str
    minutes: np.ndarray
    candle_lengths: np.ndarray
    candles: np.ndarray
    ladder_offsets: np.ndarray
    ladder: np.ndarray
    action_mask: np.ndarray
    targets: np.ndarray
    contract_ids: np.ndarray

    @property
    def minute_count(self) -> int:
        return int(len(self.minutes))


def cache_path(root: Path, session: str) -> Path:
    return root / "sessions" / f"{session}.npz"


def verify_cache_index(root: Path) -> tuple[dict, pd.DataFrame]:
    """Verify the immutable receipt, manifest and every cached session file."""

    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    expected = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError("fit-cache receipt self-hash mismatch")
    if receipt.get("schema_version") != "v5.causal-day-fit-cache.v1":
        raise RuntimeError("fit-cache receipt schema drift")
    manifest_info = receipt.get("manifest", {})
    manifest_path = Path(str(manifest_info.get("path", "")))
    if not manifest_path.is_file() or file_sha256(manifest_path) != manifest_info.get("sha256"):
        raise RuntimeError("fit-cache manifest is missing or hash-mismatched")
    manifest = pd.read_parquet(manifest_path)
    required = {
        "session",
        "path",
        "sha256",
        "minutes",
        "whole_chain_nodes",
        "eligible_actions",
    }
    if not required <= set(manifest) or manifest["session"].astype(str).duplicated().any():
        raise RuntimeError("fit-cache manifest schema or session identity drift")
    if len(manifest) != int(receipt.get("sessions", -1)):
        raise RuntimeError("fit-cache manifest session count drift")
    for row in manifest.itertuples(index=False):
        path = Path(str(row.path))
        if not path.is_file() or file_sha256(path) != str(row.sha256):
            raise RuntimeError(f"fit-cache session is missing or hash-mismatched: {row.session}")
    for column, field in (
        ("minutes", "minutes"),
        ("whole_chain_nodes", "whole_chain_nodes"),
        ("eligible_actions", "eligible_actions"),
    ):
        if int(pd.to_numeric(manifest[column], errors="raise").sum()) != int(
            receipt.get(field, -1)
        ):
            raise RuntimeError(f"fit-cache aggregate drift: {field}")
    return receipt, manifest


def load_cached_session(path: Path) -> CachedSession:
    with np.load(path, allow_pickle=False) as value:
        return CachedSession(
            session=str(value["session"].item()),
            minutes=value["minutes"],
            candle_lengths=value["candle_lengths"],
            candles=value["candles"],
            ladder_offsets=value["ladder_offsets"],
            ladder=value["ladder"],
            action_mask=value["action_mask"].astype(bool),
            targets=value["targets"],
            contract_ids=value["contract_ids"],
        )


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - mean) / scale, -10.0, 10.0).astype(np.float32)


def collate_minutes(
    cached: CachedSession,
    indices: np.ndarray,
    scaler: FeatureScaler,
    *,
    trade_cap: int = 2,
    targets_override: np.ndarray | None = None,
) -> tuple[CausalPolicyBatch, torch.Tensor, list[tuple[str, str, str]]]:
    """Pad a group of causal minute states without changing their content."""

    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        raise ValueError("cannot collate an empty minute group")
    candle_lengths = cached.candle_lengths[indices]
    ladder_lengths = np.diff(cached.ladder_offsets)[indices]
    max_candles = int(candle_lengths.max())
    max_ladder = int(ladder_lengths.max())
    batch_size = len(indices)
    candles = np.zeros((batch_size, max_candles, cached.candles.shape[1]), np.float32)
    candle_mask = np.zeros((batch_size, max_candles), bool)
    ladder = np.zeros((batch_size, max_ladder, cached.ladder.shape[1]), np.float32)
    ladder_mask = np.zeros((batch_size, max_ladder), bool)
    action_mask = np.zeros((batch_size, max_ladder), bool)
    targets = np.zeros((batch_size, max_ladder, cached.targets.shape[1]), np.float32)
    metadata: list[tuple[str, str, str]] = []
    source_targets = cached.targets if targets_override is None else targets_override

    for row, minute_index in enumerate(indices):
        candle_count = int(cached.candle_lengths[minute_index])
        candles[row, :candle_count] = _standardize(
            cached.candles[:candle_count], scaler.candle_mean, scaler.candle_scale
        )
        candle_mask[row, :candle_count] = True
        start = int(cached.ladder_offsets[minute_index])
        stop = int(cached.ladder_offsets[minute_index + 1])
        count = stop - start
        ladder[row, :count] = _standardize(
            cached.ladder[start:stop], scaler.ladder_mean, scaler.ladder_scale
        )
        ladder_mask[row, :count] = True
        action_mask[row, :count] = cached.action_mask[start:stop]
        targets[row, :count] = source_targets[start:stop]
        minute = str(cached.minutes[minute_index])
        for offset in np.flatnonzero(cached.action_mask[start:stop]):
            metadata.append((cached.session, minute, str(cached.contract_ids[start + offset])))

    roles = tuple(
        "morning_entry" if str(cached.minutes[index]) < "12:46" else "afternoon_entry"
        for index in indices
    )
    clock = np.zeros((batch_size, 5), np.float32)
    for row, index in enumerate(indices):
        minute = str(cached.minutes[index])
        value = minute_number(minute)
        elapsed = (value - 570) / 390.0
        angle = 2.0 * np.pi * elapsed
        clock[row] = [elapsed, (960 - value) / 390.0, value < 766, np.sin(angle), np.cos(angle)]
    account = np.tile(
        np.asarray([1.0, 0.0, 0.0, 1.0, 0.0], np.float32), (batch_size, 1)
    )
    batch = CausalPolicyBatch(
        candles=torch.from_numpy(candles),
        candle_mask=torch.from_numpy(candle_mask),
        ladder=torch.from_numpy(ladder),
        ladder_mask=torch.from_numpy(ladder_mask),
        entry_action_mask=torch.from_numpy(action_mask),
        account=torch.from_numpy(account),
        position=torch.zeros((batch_size, 10), dtype=torch.float32),
        clock=torch.from_numpy(clock),
        roles=roles,
    )
    return batch, torch.from_numpy(targets), metadata


def _moments(paths: list[Path], field: str) -> tuple[np.ndarray, np.ndarray]:
    total = None
    square = None
    count = 0
    for path in paths:
        cached = load_cached_session(path)
        values = np.asarray(getattr(cached, field), dtype=np.float64)
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            square = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
        count += len(values)
    assert total is not None and square is not None and count
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-12)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def fit_initial_scaler(paths: list[Path]) -> FeatureScaler:
    candle_mean, candle_scale = _moments(paths, "candles")
    ladder_mean, ladder_scale = _moments(paths, "ladder")
    return FeatureScaler(candle_mean, candle_scale, ladder_mean, ladder_scale)
