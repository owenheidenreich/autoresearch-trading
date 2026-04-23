from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import (
    V2Dataset,
    bar_context_from_dataset,
    chain_from_sidecar,
    extra_context_from_dataset,
    has_chain_snapshot,
    load_day_sidecar,
)
from v3.layer2.common import (
    build_export_rows_for_day,
    build_folds_for_dataset,
    build_teachers,
    export_metadata,
    finalize_export_dataframe,
    fold_test_day_map,
    minute_to_abs_index,
)
from v3.layer2.surface_dataset import (
    build_sequence_window,
    resolve_sequence_feature_names,
)
from v3.logger.builder import build_bar_record
from v3.logger.schema import BarRecord, ContractRecord, DayLog
from v3.oracles.exit_headroom import _horizon_pnl, _time_stop_pnl, apply_exit_headroom_oracle
from v3.oracles.opportunity import (
    CIDX_VALID,
    _best_and_worst_forward_pnl,
    _build_contract_paths,
    _mfe_mae_at_horizon,
    apply_opportunity_oracle,
)
from v2.core.chain_data import contract_snapshot


DEFAULT_ACTION_SURFACE_DATASET_PATH = os.path.join(
    "v3",
    "artifacts",
    "layer2_action_surface_dataset.pkl",
)
DEFAULT_HISTORY_BARS = 20
DEFAULT_TOP_K_CONTRACTS = 12
DEFAULT_EXECUTION_START_BAR = 15   # 09:45 ET
DEFAULT_EXECUTION_END_BAR = 120    # 11:30 ET
DEFAULT_CLEAN_MAE_FLOOR_PCT = -20.0
DEFAULT_STOPOUT_TARGET_PCT = 10.0
DEFAULT_STOPOUT_HORIZON_BARS = 10
# Hold-aware utility target: champion L3 robust_90 median hold is 62 bars on
# seeds 42/43 and 81 bars on seed 44. Default horizon 60 sits near the lower
# end of that band so the signal is representable by L3 exit times that do
# fire rather than by session-end holds.
DEFAULT_UTILITY_HORIZON_BARS = 60

CONTRACT_FEATURE_NAMES = (
    "right_is_call",
    "strike_offset_pct",
    "premium",
    "delta",
    "abs_delta",
    "spread_fraction",
    "gamma_dollar",
    "theta_to_premium",
    "selection_score",
    "passed",
    "premium_cap_ok",
    "premium_floor_ok",
    "delta_floor_ok",
    "spread_cap_ok",
)

ACTION_LABEL_NAMES = (
    "utility_raw",
    "utility_arcsinh",
    "best_exit_pnl",
    "horizon_pnl",
    "mfe_5",
    "mae_5",
    "mfe_10",
    "mae_10",
    "mfe_20",
    "mae_20",
    "clean_entry",
    "stopout_risk",
    "available_mask",
    "tradeable_mask",
)


def _selection_score(contract: ContractRecord) -> float:
    if contract.gamma_dollar is None or contract.theta_to_premium is None:
        return 0.0
    if contract.theta_to_premium <= 0.0:
        return 0.0
    return float(contract.gamma_dollar / contract.theta_to_premium)


def _surface_contract_sort_key(contract: ContractRecord, spot: float) -> tuple[float, float, float, float]:
    return (
        abs(float(contract.strike) - float(spot)),
        float(contract.spread_fraction),
        abs(float(contract.abs_delta) - 0.25),
        float(contract.strike),
    )


def _contract_feature_row(contract: ContractRecord, spot: float) -> np.ndarray:
    gamma = float(contract.gamma_dollar) if contract.gamma_dollar is not None else 0.0
    theta = float(contract.theta_to_premium) if contract.theta_to_premium is not None else 0.0
    spot_safe = max(abs(float(spot)), 1e-6)
    return np.asarray(
        [
            1.0 if contract.right == "C" else 0.0,
            float((contract.strike - spot) / spot_safe),
            float(contract.premium),
            float(contract.delta),
            float(contract.abs_delta),
            float(contract.spread_fraction),
            gamma,
            theta,
            _selection_score(contract),
            1.0 if contract.passed else 0.0,
            1.0 if contract.premium_cap_ok else 0.0,
            1.0 if contract.premium_floor_ok else 0.0,
            1.0 if contract.delta_floor_ok else 0.0,
            1.0 if contract.spread_cap_ok else 0.0,
        ],
        dtype=np.float32,
    )


def _bar_contract_id_lookup(sidecar: dict[str, Any], local_bar: int) -> dict[tuple[float, str], int]:
    feats, _, cids = contract_snapshot(sidecar, local_bar)
    out: dict[tuple[float, str], int] = {}
    for i in range(feats.shape[0]):
        row = feats[i]
        if float(row[CIDX_VALID]) < 0.5:
            continue
        strike = float(row[1])
        right = "P" if float(row[2]) > 0.5 else "C"
        out[(strike, right)] = int(cids[i])
    return out


def _stopout_before_target(
    mids: np.ndarray,
    entry_bar: int,
    entry_mid: float,
    horizon_bars: int,
    stop_pct: float,
    target_pct: float,
) -> float:
    if entry_mid <= 0:
        return float("nan")
    end = min(entry_bar + horizon_bars, len(mids) - 1)
    window = mids[entry_bar + 1 : end + 1]
    if window.size == 0:
        return float("nan")
    for mid in window:
        if not np.isfinite(mid):
            continue
        pct = (float(mid) - float(entry_mid)) / float(entry_mid) * 100.0
        if pct <= stop_pct:
            return 1.0
        if pct >= target_pct:
            return 0.0
    return 0.0


def _time_bucket(time_stop_pnl: float, mae_10: float, mae_floor_pct: float) -> str:
    if not np.isfinite(time_stop_pnl) or not np.isfinite(mae_10):
        return "missing"
    if time_stop_pnl > 0 and mae_10 > mae_floor_pct:
        return "clean_winner"
    if time_stop_pnl > 0:
        return "shakeout_winner"
    if mae_10 <= mae_floor_pct:
        return "fast_loser"
    return "drift_loser"


def build_action_labeled_day(
    dataset: V2Dataset,
    day: str,
    cfg: GuardrailConfig,
    equity: float,
    *,
    execution_start_bar: int,
    execution_end_bar: int,
) -> tuple[DayLog, dict[str, Any]] | tuple[None, None]:
    teachers = build_teachers()
    sidecar = load_day_sidecar(dataset, day)
    if sidecar is None:
        return None, None

    day_start_abs, day_end_abs = dataset.day_bar_range(day)
    log = DayLog(day=day)
    for abs_idx in range(day_start_abs, day_end_abs):
        local_bar = abs_idx - day_start_abs
        minute = int(dataset.bar_of_day[abs_idx])
        if minute < execution_start_bar or minute > execution_end_bar:
            continue
        if not has_chain_snapshot(sidecar, local_bar):
            continue

        bar_ctx = bar_context_from_dataset(dataset, abs_idx)
        extra = extra_context_from_dataset(dataset, abs_idx)
        chain = chain_from_sidecar(sidecar, local_bar)
        record = build_bar_record(
            day=day,
            bar_index=minute,
            timestamp_ms=0,
            bar_ctx=bar_ctx,
            chain=chain,
            teachers=teachers,
            equity=equity,
            cfg=cfg,
            vix=extra["vix"],
            atm_iv=extra["atm_iv"],
            iv_percentile=extra["iv_percentile"],
        )
        log.bars.append(record)

    if not log.bars:
        return None, None

    apply_opportunity_oracle(log, sidecar, day_start_abs, bars_per_day=390)
    apply_exit_headroom_oracle(log, sidecar, bars_per_day=390)
    return log, sidecar


def _empty_contract_block(top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    token_count = top_k * 2
    labels = {name: np.full(token_count + 1, np.nan, dtype=np.float32) for name in ACTION_LABEL_NAMES}
    labels["available_mask"] = np.zeros(token_count + 1, dtype=np.float32)
    labels["tradeable_mask"] = np.zeros(token_count + 1, dtype=np.float32)
    labels["utility_raw"][0] = 0.0
    labels["utility_arcsinh"][0] = 0.0
    labels["best_exit_pnl"][0] = 0.0
    labels["horizon_pnl"][0] = 0.0
    labels["clean_entry"][0] = 0.0
    labels["stopout_risk"][0] = 0.0
    labels["available_mask"][0] = 1.0
    labels["tradeable_mask"][0] = 1.0
    return (
        np.zeros((token_count, len(CONTRACT_FEATURE_NAMES)), dtype=np.float32),
        np.zeros(token_count, dtype=np.float32),
        np.full(token_count, np.nan, dtype=np.float32),
        labels,
    )


def build_contract_action_surface(
    bar: BarRecord,
    *,
    sidecar: dict[str, Any],
    paths: dict[int, Any],
    top_k: int,
    mae_floor_pct: float,
    stopout_horizon_bars: int,
    stopout_target_pct: float,
    utility_horizon_bars: int = DEFAULT_UTILITY_HORIZON_BARS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    token_features, token_mask, token_strikes, action_labels = _empty_contract_block(top_k)
    spot = float(bar.underlying_close)
    cid_lookup = _bar_contract_id_lookup(sidecar, int(bar.bar_index))

    for side_idx, right in enumerate(("C", "P")):
        contracts = [c for c in bar.contracts if c.contract_valid and c.right == right]
        contracts = sorted(contracts, key=lambda c: _surface_contract_sort_key(c, spot))
        for slot, contract in enumerate(contracts[:top_k]):
            token_idx = side_idx * top_k + slot
            token_features[token_idx] = _contract_feature_row(contract, spot)
            token_mask[token_idx] = 1.0
            token_strikes[token_idx] = float(contract.strike)

            action_idx = token_idx + 1
            action_labels["available_mask"][action_idx] = 1.0
            action_labels["tradeable_mask"][action_idx] = 1.0 if contract.passed else 0.0

            cid = cid_lookup.get((float(contract.strike), contract.right))
            if cid is None or cid not in paths:
                continue
            path = paths[cid]

            time_stop_pnl = _time_stop_pnl(
                path.mids,
                entry_bar=int(bar.bar_index),
                entry_mid=float(contract.mid),
                entry_spread_frac=float(contract.spread_fraction),
                session_end_bar=375,
                commission=1.0,
            )
            horizon_pnl = _horizon_pnl(
                path.mids,
                entry_bar=int(bar.bar_index),
                entry_mid=float(contract.mid),
                entry_spread_frac=float(contract.spread_fraction),
                horizon_bars=int(utility_horizon_bars),
                session_end_bar=375,
                commission=1.0,
            )
            best_exit_pnl, _ = _best_and_worst_forward_pnl(
                path,
                entry_bar=int(bar.bar_index),
                entry_mid=float(contract.mid),
                entry_spread_frac=float(contract.spread_fraction),
                session_end_bar=375,
                commission=1.0,
            )
            mfe_5, mae_5 = _mfe_mae_at_horizon(path, int(bar.bar_index), float(contract.mid), 5, 375)
            mfe_10, mae_10 = _mfe_mae_at_horizon(path, int(bar.bar_index), float(contract.mid), 10, 375)
            mfe_20, mae_20 = _mfe_mae_at_horizon(path, int(bar.bar_index), float(contract.mid), 20, 375)
            stopout_risk = _stopout_before_target(
                path.mids,
                entry_bar=int(bar.bar_index),
                entry_mid=float(contract.mid),
                horizon_bars=stopout_horizon_bars,
                stop_pct=mae_floor_pct,
                target_pct=stopout_target_pct,
            )

            action_labels["utility_raw"][action_idx] = (
                float(time_stop_pnl) if time_stop_pnl is not None and np.isfinite(time_stop_pnl) else np.nan
            )
            action_labels["utility_arcsinh"][action_idx] = (
                float(np.arcsinh(time_stop_pnl / 100.0))
                if time_stop_pnl is not None and np.isfinite(time_stop_pnl)
                else np.nan
            )
            action_labels["best_exit_pnl"][action_idx] = (
                float(best_exit_pnl) if best_exit_pnl is not None and np.isfinite(best_exit_pnl) else np.nan
            )
            action_labels["horizon_pnl"][action_idx] = (
                float(horizon_pnl) if horizon_pnl is not None and np.isfinite(horizon_pnl) else np.nan
            )
            action_labels["mfe_5"][action_idx] = float(mfe_5) if mfe_5 is not None and np.isfinite(mfe_5) else np.nan
            action_labels["mae_5"][action_idx] = float(mae_5) if mae_5 is not None and np.isfinite(mae_5) else np.nan
            action_labels["mfe_10"][action_idx] = float(mfe_10) if mfe_10 is not None and np.isfinite(mfe_10) else np.nan
            action_labels["mae_10"][action_idx] = float(mae_10) if mae_10 is not None and np.isfinite(mae_10) else np.nan
            action_labels["mfe_20"][action_idx] = float(mfe_20) if mfe_20 is not None and np.isfinite(mfe_20) else np.nan
            action_labels["mae_20"][action_idx] = float(mae_20) if mae_20 is not None and np.isfinite(mae_20) else np.nan
            if time_stop_pnl is not None and np.isfinite(time_stop_pnl) and mae_10 is not None and np.isfinite(mae_10):
                action_labels["clean_entry"][action_idx] = 1.0 if time_stop_pnl > 0 and mae_10 > mae_floor_pct else 0.0
            if np.isfinite(stopout_risk):
                action_labels["stopout_risk"][action_idx] = float(stopout_risk)

    return token_features, token_mask, token_strikes, action_labels


def build_action_surface_bundle(
    dataset: V2Dataset,
    cfg: GuardrailConfig,
    equity: float,
    *,
    history_bars: int = DEFAULT_HISTORY_BARS,
    top_k_contracts: int = DEFAULT_TOP_K_CONTRACTS,
    execution_start_bar: int = DEFAULT_EXECUTION_START_BAR,
    execution_end_bar: int = DEFAULT_EXECUTION_END_BAR,
    mae_floor_pct: float = DEFAULT_CLEAN_MAE_FLOOR_PCT,
    stopout_target_pct: float = DEFAULT_STOPOUT_TARGET_PCT,
    stopout_horizon_bars: int = DEFAULT_STOPOUT_HORIZON_BARS,
    utility_horizon_bars: int = DEFAULT_UTILITY_HORIZON_BARS,
) -> dict[str, Any]:
    sequence_feature_names = resolve_sequence_feature_names(dataset)
    folds = build_folds_for_dataset(dataset)
    test_fold_map = fold_test_day_map(folds)
    all_days = sorted(set(dataset.dates))

    rows: list[dict[str, Any]] = []
    seq_rows: list[np.ndarray] = []
    seq_masks: list[np.ndarray] = []
    contract_rows: list[np.ndarray] = []
    contract_masks: list[np.ndarray] = []
    contract_strikes: list[np.ndarray] = []
    label_rows = {
        name: [] for name in ACTION_LABEL_NAMES
    }
    kept_days = 0

    for i, day in enumerate(all_days):
        log, sidecar = build_action_labeled_day(
            dataset,
            day,
            cfg,
            equity,
            execution_start_bar=execution_start_bar,
            execution_end_bar=execution_end_bar,
        )
        if log is None or sidecar is None:
            continue
        kept_days += 1

        day_rows = build_export_rows_for_day(dataset, day, log, test_fold_map.get(day, -1), sidecar=sidecar)
        minute_map = minute_to_abs_index(dataset, day)
        paths = _build_contract_paths(sidecar, 390)

        day_seq_rows: list[np.ndarray] = []
        day_seq_masks: list[np.ndarray] = []
        day_contract_rows: list[np.ndarray] = []
        day_contract_masks: list[np.ndarray] = []
        day_contract_strikes: list[np.ndarray] = []
        day_label_rows = {name: [] for name in ACTION_LABEL_NAMES}

        for row, bar in zip(day_rows, log.bars):
            abs_idx = minute_map.get(int(bar.bar_index))
            if abs_idx is None:
                continue
            seq, seq_mask = build_sequence_window(
                dataset,
                day,
                abs_idx,
                history_bars=history_bars,
                feature_names=sequence_feature_names,
            )
            token_features, token_mask, token_strike_row, labels = build_contract_action_surface(
                bar,
                sidecar=sidecar,
                paths=paths,
                top_k=top_k_contracts,
                mae_floor_pct=mae_floor_pct,
                stopout_horizon_bars=stopout_horizon_bars,
                stopout_target_pct=stopout_target_pct,
                utility_horizon_bars=utility_horizon_bars,
            )
            row["chosen_flat_time_bucket"] = _time_bucket(
                float(np.nanmax(labels["utility_raw"][1:])) if np.isfinite(labels["utility_raw"][1:]).any() else np.nan,
                float(np.nanmax(labels["mae_10"][1:])) if np.isfinite(labels["mae_10"][1:]).any() else np.nan,
                mae_floor_pct,
            )

            day_seq_rows.append(seq)
            day_seq_masks.append(seq_mask)
            day_contract_rows.append(token_features)
            day_contract_masks.append(token_mask)
            day_contract_strikes.append(token_strike_row)
            for name in ACTION_LABEL_NAMES:
                day_label_rows[name].append(labels[name])

        if len(day_rows) != len(day_seq_rows):
            raise RuntimeError(
                f"Action surface row alignment failed on {day}: "
                f"{len(day_rows)} rows vs {len(day_seq_rows)} tensors"
            )

        rows.extend(day_rows)
        seq_rows.extend(day_seq_rows)
        seq_masks.extend(day_seq_masks)
        contract_rows.extend(day_contract_rows)
        contract_masks.extend(day_contract_masks)
        contract_strikes.extend(day_contract_strikes)
        for name in ACTION_LABEL_NAMES:
            label_rows[name].extend(day_label_rows[name])

        if (i + 1) % 100 == 0:
            print(
                f"  action-surface-exported {i+1}/{len(all_days)} days "
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
            "action_label_names": list(ACTION_LABEL_NAMES),
            "history_bars": int(history_bars),
            "top_k_contracts_per_side": int(top_k_contracts),
            "n_action_contract_tokens": int(top_k_contracts * 2),
            "kept_days": int(kept_days),
            "execution_window": {
                "start_bar": int(execution_start_bar),
                "end_bar": int(execution_end_bar),
                "label": "09:45-11:30 ET",
            },
            "guardrail_semantics": {
                "layer0_execution_enforced": True,
                "blocked_tokens_present_in_context": True,
                "tradeable_actions_restricted_to_passed_contracts": True,
            },
            "token_schema": [
                {
                    "action_id": int(i + 1),
                    "slot": int(i % top_k_contracts),
                    "side": "call" if i < top_k_contracts else "put",
                }
                for i in range(top_k_contracts * 2)
            ],
            "clean_entry_rule": {
                "time_stop_positive_required": True,
                "mae_floor_pct": float(mae_floor_pct),
                "horizon_bars": 10,
            },
            "stopout_rule": {
                "horizon_bars": int(stopout_horizon_bars),
                "stop_pct": float(mae_floor_pct),
                "target_pct": float(stopout_target_pct),
            },
            "utility_horizon_bars": int(utility_horizon_bars),
        }
    )

    return {
        "rows": df,
        "meta": meta,
        "sequence_features": np.asarray(seq_rows, dtype=np.float32),
        "sequence_mask": np.asarray(seq_masks, dtype=np.float32),
        "contract_features": np.asarray(contract_rows, dtype=np.float32),
        "contract_mask": np.asarray(contract_masks, dtype=np.float32),
        "contract_strike": np.asarray(contract_strikes, dtype=np.float32),
        "action_labels": {
            name: np.asarray(values, dtype=np.float32)
            for name, values in label_rows.items()
        },
    }
