from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    build_export_rows_for_day,
    build_folds_for_dataset,
    build_labeled_day,
    export_metadata,
    finalize_export_dataframe,
    fold_test_day_map,
    minute_to_abs_index,
)
from v3.logger.schema import BarRecord, ContractRecord


DEFAULT_SURFACE_DATASET_PATH = os.path.join("v3", "artifacts", "layer2_surface_dataset.pkl")
DEFAULT_HISTORY_BARS = 20
DEFAULT_TOP_K_CONTRACTS = 8

SEQUENCE_FEATURE_CANDIDATES = (
    "underlying_close",
    "vwap_dist",
    "vwap_slope",
    "volume_ratio",
    "first15_range_pct",
    "bars_since_break_above_first15",
    "bars_since_break_below_first15",
    "sigma_pos",
    "omar_retest_dist_norm",
    "omar_range_pct",
    "last10_range_over_omar",
    "inside_first15",
    "late_window_40_120_flag",
    "omar_mid_pos_units",
    "last10_break_state",
)

CONTRACT_FEATURE_NAMES = (
    "strike_offset_pct",
    "premium",
    "delta",
    "abs_delta",
    "spread_fraction",
    "gamma_dollar",
    "theta_to_premium",
    "selection_score",
)


def resolve_sequence_feature_names(dataset: V2Dataset) -> tuple[str, ...]:
    feature_names = tuple(
        name for name in SEQUENCE_FEATURE_CANDIDATES
        if name == "underlying_close" or name in dataset.idx
    )
    if len(feature_names) < 8:
        raise RuntimeError(
            "Surface export resolved too few sequential features from the current data.pt. "
            f"Available={feature_names}"
        )
    return feature_names


def _sequence_value(dataset: V2Dataset, abs_idx: int, name: str) -> float:
    if name == "underlying_close":
        return float(dataset.spot_prices[abs_idx])
    return float(dataset.X_sim[abs_idx, dataset.idx[name]])


def build_sequence_window(
    dataset: V2Dataset,
    day: str,
    abs_idx: int,
    history_bars: int,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    day_start_abs, _ = dataset.day_bar_range(day)
    start_abs = max(day_start_abs, abs_idx - history_bars + 1)
    window_abs = list(range(start_abs, abs_idx + 1))

    seq = np.zeros((history_bars, len(feature_names)), dtype=np.float32)
    mask = np.zeros(history_bars, dtype=np.float32)
    offset = history_bars - len(window_abs)
    for pos, row_abs_idx in enumerate(window_abs, start=offset):
        seq[pos] = np.asarray(
            [_sequence_value(dataset, row_abs_idx, name) for name in feature_names],
            dtype=np.float32,
        )
        mask[pos] = 1.0
    return seq, mask


def _selection_score(contract: ContractRecord) -> float:
    if contract.gamma_dollar is None or contract.theta_to_premium is None:
        return 0.0
    if contract.theta_to_premium <= 0.0:
        return 0.0
    return float(contract.gamma_dollar / contract.theta_to_premium)


def _contract_feature_row(contract: ContractRecord, spot: float) -> np.ndarray:
    gamma = float(contract.gamma_dollar) if contract.gamma_dollar is not None else 0.0
    theta = float(contract.theta_to_premium) if contract.theta_to_premium is not None else 0.0
    spot_safe = max(abs(float(spot)), 1e-6)
    return np.asarray(
        [
            float((contract.strike - spot) / spot_safe),
            float(contract.premium),
            float(contract.delta),
            float(contract.abs_delta),
            float(contract.spread_fraction),
            gamma,
            theta,
            _selection_score(contract),
        ],
        dtype=np.float32,
    )


def _surface_contract_sort_key(contract: ContractRecord, spot: float) -> tuple[float, float, float]:
    return (
        abs(float(contract.strike) - float(spot)),
        float(contract.spread_fraction),
        abs(float(contract.abs_delta) - 0.25),
    )


def build_contract_surface(
    bar: BarRecord,
    top_k: int,
    feature_names: tuple[str, ...] = CONTRACT_FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del feature_names
    call_tokens = np.zeros((top_k, len(CONTRACT_FEATURE_NAMES)), dtype=np.float32)
    put_tokens = np.zeros((top_k, len(CONTRACT_FEATURE_NAMES)), dtype=np.float32)
    call_mask = np.zeros(top_k, dtype=np.float32)
    put_mask = np.zeros(top_k, dtype=np.float32)

    spot = float(bar.underlying_close)
    by_right = {
        "C": sorted(
            [c for c in bar.contracts if c.passed and c.right == "C"],
            key=lambda c: _surface_contract_sort_key(c, spot),
        ),
        "P": sorted(
            [c for c in bar.contracts if c.passed and c.right == "P"],
            key=lambda c: _surface_contract_sort_key(c, spot),
        ),
    }

    for i, contract in enumerate(by_right["C"][:top_k]):
        call_tokens[i] = _contract_feature_row(contract, spot)
        call_mask[i] = 1.0
    for i, contract in enumerate(by_right["P"][:top_k]):
        put_tokens[i] = _contract_feature_row(contract, spot)
        put_mask[i] = 1.0
    return call_tokens, call_mask, put_tokens, put_mask


def build_surface_export_bundle(
    dataset: V2Dataset,
    cfg: GuardrailConfig,
    equity: float,
    *,
    history_bars: int = DEFAULT_HISTORY_BARS,
    top_k_contracts: int = DEFAULT_TOP_K_CONTRACTS,
) -> dict[str, Any]:
    sequence_feature_names = resolve_sequence_feature_names(dataset)
    folds = build_folds_for_dataset(dataset)
    test_fold_map = fold_test_day_map(folds)
    all_days = sorted(set(dataset.dates))

    rows: list[dict[str, Any]] = []
    seq_rows: list[np.ndarray] = []
    seq_masks: list[np.ndarray] = []
    call_rows: list[np.ndarray] = []
    call_masks: list[np.ndarray] = []
    put_rows: list[np.ndarray] = []
    put_masks: list[np.ndarray] = []

    kept_days = 0
    for i, day in enumerate(all_days):
        log, sidecar = build_labeled_day(dataset, day, cfg, equity=equity)
        if log is None or sidecar is None:
            continue
        kept_days += 1

        day_rows = build_export_rows_for_day(dataset, day, log, test_fold_map.get(day, -1), sidecar=sidecar)
        minute_map = minute_to_abs_index(dataset, day)
        day_seq_rows: list[np.ndarray] = []
        day_seq_masks: list[np.ndarray] = []
        day_call_rows: list[np.ndarray] = []
        day_call_masks: list[np.ndarray] = []
        day_put_rows: list[np.ndarray] = []
        day_put_masks: list[np.ndarray] = []

        for bar in log.bars:
            abs_idx = minute_map.get(bar.bar_index)
            if abs_idx is None:
                continue
            seq, seq_mask = build_sequence_window(
                dataset,
                day,
                abs_idx,
                history_bars=history_bars,
                feature_names=sequence_feature_names,
            )
            call_tokens, call_mask, put_tokens, put_mask = build_contract_surface(
                bar,
                top_k=top_k_contracts,
                feature_names=CONTRACT_FEATURE_NAMES,
            )
            day_seq_rows.append(seq)
            day_seq_masks.append(seq_mask)
            day_call_rows.append(call_tokens)
            day_call_masks.append(call_mask)
            day_put_rows.append(put_tokens)
            day_put_masks.append(put_mask)

        if len(day_rows) != len(day_seq_rows):
            raise RuntimeError(
                f"Surface export row alignment failed on {day}: "
                f"{len(day_rows)} tabular rows vs {len(day_seq_rows)} surface rows"
            )

        rows.extend(day_rows)
        seq_rows.extend(day_seq_rows)
        seq_masks.extend(day_seq_masks)
        call_rows.extend(day_call_rows)
        call_masks.extend(day_call_masks)
        put_rows.extend(day_put_rows)
        put_masks.extend(day_put_masks)

        if (i + 1) % 100 == 0:
            print(
                f"  surface-exported {i+1}/{len(all_days)} days "
                f"({len(rows):,} rows)",
                flush=True,
            )

    df = finalize_export_dataframe(rows)
    meta = export_metadata(dataset, df, folds)
    meta.update(
        {
            "scalar_feature_names": list(meta["feature_names"]),
            "sequence_feature_names": list(sequence_feature_names),
            "contract_feature_names": list(CONTRACT_FEATURE_NAMES),
            "history_bars": int(history_bars),
            "top_k_contracts": int(top_k_contracts),
            "kept_days": int(kept_days),
        }
    )

    return {
        "rows": df,
        "meta": meta,
        "sequence_features": np.asarray(seq_rows, dtype=np.float32),
        "sequence_mask": np.asarray(seq_masks, dtype=np.float32),
        "call_contract_features": np.asarray(call_rows, dtype=np.float32),
        "call_contract_mask": np.asarray(call_masks, dtype=np.float32),
        "put_contract_features": np.asarray(put_rows, dtype=np.float32),
        "put_contract_mask": np.asarray(put_masks, dtype=np.float32),
    }
