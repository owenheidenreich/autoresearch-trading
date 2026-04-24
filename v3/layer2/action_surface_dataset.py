from __future__ import annotations

import hashlib
import json
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
DEFAULT_CONTRACT_SELECTION_MODE = "risk_band"
# Hold-aware utility target: champion L3 robust_90 median hold is 62 bars on
# seeds 42/43 and 81 bars on seed 44. Default horizon 60 sits near the lower
# end of that band so the signal is representable by L3 exit times that do
# fire rather than by session-end holds.
DEFAULT_UTILITY_HORIZON_BARS = 60

RISK_BAND_SLOTS: tuple[tuple[str, float, float, int], ...] = (
    ("high_delta", 0.45, 0.70, 3),
    ("balanced", 0.25, 0.45, 4),
    ("convex", 0.12, 0.25, 4),
)
RISK_BAND_CODES = {
    "high_delta": 0.0,
    "balanced": 1.0,
    "convex": 2.0,
    "efficiency": 3.0,
    "fill": 4.0,
}
MONEYNESS_ATM_EPS = 0.00025
MONEYNESS_CODES = {
    "itm": -1.0,
    "atm": 0.0,
    "otm": 1.0,
}

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
    "slot_role",
    "moneyness_bucket",
    "risk_band",
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
    "return_on_premium",
    "win_label",
    "entry_fill_bar",
    "entry_fill_mid",
    "entry_spread_fraction",
    "spread_cost",
    "hybrid_live_utility",
    "hybrid_live_utility_arcsinh",
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


def required_action_labels_for_target(utility_target: str) -> set[str]:
    base = {
        "utility_raw",
        "utility_arcsinh",
        "best_exit_pnl",
        "clean_entry",
        "stopout_risk",
        "available_mask",
        "tradeable_mask",
    }
    if utility_target == "horizon":
        base.add("horizon_pnl")
    if utility_target == "hybrid_live":
        base.update(
            {
                "return_on_premium",
                "win_label",
                "spread_cost",
                "hybrid_live_utility",
                "hybrid_live_utility_arcsinh",
            }
        )
    return base


def validate_action_surface_bundle(
    bundle: dict[str, Any],
    *,
    required_labels: set[str] | None = None,
    required_contract_features: set[str] | None = None,
) -> None:
    """Fail fast when an action-surface artifact is stale or internally misaligned."""
    required_top = {
        "rows",
        "meta",
        "sequence_features",
        "sequence_mask",
        "contract_features",
        "contract_mask",
        "contract_strike",
        "action_labels",
    }
    missing_top = sorted(required_top - set(bundle))
    if missing_top:
        raise RuntimeError(f"Invalid action-surface bundle; missing top-level keys: {missing_top}")

    rows = bundle["rows"]
    meta = bundle["meta"]
    labels = bundle["action_labels"]
    n_rows = len(rows)
    expected_tokens = int(meta.get("n_action_contract_tokens", 0))
    expected_actions = expected_tokens + 1
    if bundle["contract_features"].shape[:2] != (n_rows, expected_tokens):
        raise RuntimeError(
            "Invalid action-surface bundle; contract_features shape "
            f"{bundle['contract_features'].shape} does not match rows/tokens {(n_rows, expected_tokens)}"
        )
    if bundle["contract_mask"].shape != (n_rows, expected_tokens):
        raise RuntimeError(
            "Invalid action-surface bundle; contract_mask shape "
            f"{bundle['contract_mask'].shape} does not match rows/tokens {(n_rows, expected_tokens)}"
        )
    if bundle["contract_strike"].shape != (n_rows, expected_tokens):
        raise RuntimeError(
            "Invalid action-surface bundle; contract_strike shape "
            f"{bundle['contract_strike'].shape} does not match rows/tokens {(n_rows, expected_tokens)}"
        )

    meta_labels = set(meta.get("action_label_names", ()))
    actual_labels = set(labels)
    missing_meta_labels = sorted(meta_labels - actual_labels)
    if missing_meta_labels:
        raise RuntimeError(
            "Invalid action-surface bundle; meta lists labels absent from action_labels: "
            f"{missing_meta_labels}"
        )
    wanted_labels = set(required_labels or ())
    missing_required = sorted(wanted_labels - actual_labels)
    if missing_required:
        raise RuntimeError(
            "Action-surface artifact is stale for this training target; missing labels: "
            f"{missing_required}. Re-run v3.layer2.export_action_surface_dataset."
        )
    for name, arr in labels.items():
        if arr.shape != (n_rows, expected_actions):
            raise RuntimeError(
                f"Invalid action-surface label {name!r}; shape {arr.shape} "
                f"does not match rows/actions {(n_rows, expected_actions)}"
            )

    contract_features = set(meta.get("contract_feature_names", ()))
    wanted_contract_features = set(required_contract_features or ())
    missing_contract_features = sorted(wanted_contract_features - contract_features)
    if missing_contract_features:
        raise RuntimeError(
            "Action-surface artifact is stale for this model; missing contract features: "
            f"{missing_contract_features}. Re-run v3.layer2.export_action_surface_dataset."
        )


def action_surface_dataset_fingerprint(rows: pd.DataFrame, meta: dict[str, Any]) -> str:
    identity_columns = list(meta.get("identity_columns", ("day", "bar_index", "fold_id")))
    missing = [col for col in identity_columns if col not in rows.columns]
    if missing:
        raise RuntimeError(f"Cannot fingerprint action-surface rows; missing identity columns: {missing}")

    identity_hashes = pd.util.hash_pandas_object(
        rows.loc[:, identity_columns].reset_index(drop=True),
        index=False,
    ).to_numpy(dtype=np.uint64, copy=False)

    fingerprint_meta = {
        "identity_columns": identity_columns,
        "n_rows": int(len(rows)),
        "history_bars": meta.get("history_bars"),
        "top_k_contracts_per_side": meta.get("top_k_contracts_per_side"),
        "n_action_contract_tokens": meta.get("n_action_contract_tokens"),
        "execution_window": meta.get("execution_window"),
        "feature_contract_version": meta.get("feature_contract_version"),
        "contract_feature_names": meta.get("contract_feature_names"),
        "action_label_names": meta.get("action_label_names"),
        "token_schema": meta.get("token_schema"),
        "contract_selection_mode": meta.get("contract_selection_mode"),
        "label_timing": meta.get("label_timing"),
        "hybrid_live_utility": meta.get("hybrid_live_utility"),
    }
    h = hashlib.sha256()
    h.update(identity_hashes.tobytes())
    h.update(json.dumps(fingerprint_meta, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def _selection_score(contract: ContractRecord) -> float:
    if contract.gamma_dollar is None or contract.theta_to_premium is None:
        return 0.0
    if contract.theta_to_premium <= 0.0:
        return 0.0
    return float(contract.gamma_dollar / contract.theta_to_premium)


def _moneyness_bucket(contract: ContractRecord, spot: float) -> float:
    spot_safe = max(abs(float(spot)), 1e-6)
    offset = float((contract.strike - spot) / spot_safe)
    if abs(offset) <= MONEYNESS_ATM_EPS:
        return MONEYNESS_CODES["atm"]
    if contract.right == "C":
        return MONEYNESS_CODES["otm"] if offset > 0 else MONEYNESS_CODES["itm"]
    return MONEYNESS_CODES["otm"] if offset < 0 else MONEYNESS_CODES["itm"]


def _spread_cost_dollars(entry_mid: float, spread_fraction: float) -> float:
    if not np.isfinite(entry_mid) or entry_mid <= 0 or not np.isfinite(spread_fraction):
        return float("nan")
    return float(entry_mid) * float(spread_fraction) * 100.0


def _return_on_premium(pnl: float | None, entry_mid: float) -> float:
    if pnl is None or not np.isfinite(pnl) or not np.isfinite(entry_mid) or entry_mid <= 0:
        return float("nan")
    return float(pnl) / max(float(entry_mid) * 100.0, 1e-6)


def hybrid_live_utility(
    pnl: float | None,
    *,
    entry_mid: float,
    spread_fraction: float,
    stopout_risk: float,
    entry_bar: int,
    exit_bar: int | None = None,
    session_end_bar: int = 375,
) -> float:
    """Live-oriented utility in dollars.

    Dollar PnL stays the base unit for account-level PF/DD, while premium
    return lets cheaper convex contracts compete. Spread, stopout, and late
    hold penalties keep the label closer to what an executable trader can
    reasonably prefer.
    """
    if pnl is None or not np.isfinite(pnl) or not np.isfinite(entry_mid) or entry_mid <= 0:
        return float("nan")
    premium = max(float(entry_mid) * 100.0, 1e-6)
    ret_multiple = float(pnl) / premium
    spread_cost = _spread_cost_dollars(entry_mid, spread_fraction)
    spread_penalty = 0.35 * spread_cost if np.isfinite(spread_cost) else 0.0
    stop_penalty = 60.0 * float(np.nan_to_num(stopout_risk, nan=0.0, posinf=1.0, neginf=0.0))
    if exit_bar is None or not np.isfinite(exit_bar):
        hold_frac = max(0.0, min(1.0, (session_end_bar - int(entry_bar)) / max(session_end_bar, 1)))
        late_penalty = 10.0 * (1.0 - hold_frac)
    else:
        hold_bars = max(0, int(exit_bar) - int(entry_bar))
        late_penalty = 0.08 * max(0, hold_bars - 90)
    return float(pnl) + 125.0 * float(np.clip(ret_multiple, -2.0, 4.0)) - spread_penalty - stop_penalty - late_penalty


def _surface_contract_sort_key(contract: ContractRecord, spot: float) -> tuple[float, float, float, float]:
    return (
        abs(float(contract.strike) - float(spot)),
        float(contract.spread_fraction),
        abs(float(contract.abs_delta) - 0.25),
        float(contract.strike),
    )


def _risk_band_sort_key(
    contract: ContractRecord,
    spot: float,
    *,
    target_delta: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    delta_term = 0.0 if target_delta is None else abs(float(contract.abs_delta) - float(target_delta))
    return (
        0.0 if contract.passed else 1.0,
        float(contract.spread_fraction),
        delta_term,
        -_selection_score(contract),
        abs(float(contract.strike) - float(spot)),
        float(contract.strike),
    )


def _risk_band_for_contract(contract: ContractRecord, default: str = "fill") -> str:
    abs_delta = float(contract.abs_delta)
    for name, lo, hi, _ in RISK_BAND_SLOTS:
        if lo <= abs_delta <= hi:
            return name
    return default


def _risk_band_contracts_for_side(
    contracts: list[ContractRecord],
    spot: float,
    *,
    top_k: int,
) -> list[tuple[ContractRecord, str]]:
    if top_k != DEFAULT_TOP_K_CONTRACTS:
        selected = sorted(contracts, key=lambda c: _surface_contract_sort_key(c, spot))[:top_k]
        return [(c, _risk_band_for_contract(c)) for c in selected]

    selected: list[tuple[ContractRecord, str]] = []
    selected_keys: set[tuple[float, str]] = set()

    for band_name, lo, hi, count in RISK_BAND_SLOTS:
        target = (lo + hi) / 2.0
        band_contracts = [
            c for c in contracts
            if lo <= float(c.abs_delta) <= hi
            and (float(c.strike), c.right) not in selected_keys
        ]
        band_contracts = sorted(
            band_contracts,
            key=lambda c, target=target: _risk_band_sort_key(c, spot, target_delta=target),
        )
        for contract in band_contracts[:count]:
            selected.append((contract, band_name))
            selected_keys.add((float(contract.strike), contract.right))

    remaining = [c for c in contracts if (float(c.strike), c.right) not in selected_keys]
    efficiency_sorted = sorted(
        remaining,
        key=lambda c: (
            0.0 if c.passed else 1.0,
            -_selection_score(c),
            float(c.spread_fraction),
            abs(float(c.strike) - float(spot)),
            float(c.strike),
        ),
    )
    if efficiency_sorted and len(selected) < top_k:
        contract = efficiency_sorted[0]
        selected.append((contract, "efficiency"))
        selected_keys.add((float(contract.strike), contract.right))

    remaining = [c for c in contracts if (float(c.strike), c.right) not in selected_keys]
    for contract in sorted(remaining, key=lambda c: _surface_contract_sort_key(c, spot)):
        if len(selected) >= top_k:
            break
        selected.append((contract, "fill"))
        selected_keys.add((float(contract.strike), contract.right))

    return selected[:top_k]


def _contract_feature_row(contract: ContractRecord, spot: float, slot_role: str) -> np.ndarray:
    gamma = float(contract.gamma_dollar) if contract.gamma_dollar is not None else 0.0
    theta = float(contract.theta_to_premium) if contract.theta_to_premium is not None else 0.0
    spot_safe = max(abs(float(spot)), 1e-6)
    role_code = RISK_BAND_CODES.get(slot_role, RISK_BAND_CODES["fill"])
    risk_band = RISK_BAND_CODES.get(_risk_band_for_contract(contract), RISK_BAND_CODES["fill"])
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
            role_code,
            _moneyness_bucket(contract, spot),
            risk_band,
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
    labels["return_on_premium"][0] = 0.0
    labels["win_label"][0] = 0.0
    labels["entry_fill_bar"][0] = -1.0
    labels["entry_fill_mid"][0] = 0.0
    labels["entry_spread_fraction"][0] = 0.0
    labels["spread_cost"][0] = 0.0
    labels["hybrid_live_utility"][0] = 0.0
    labels["hybrid_live_utility_arcsinh"][0] = 0.0
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
    contract_selection_mode: str = DEFAULT_CONTRACT_SELECTION_MODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    token_features, token_mask, token_strikes, action_labels = _empty_contract_block(top_k)
    spot = float(bar.underlying_close)
    cid_lookup = _bar_contract_id_lookup(sidecar, int(bar.bar_index))

    for side_idx, right in enumerate(("C", "P")):
        contracts = [c for c in bar.contracts if c.contract_valid and c.right == right]
        if contract_selection_mode == "risk_band":
            selected_contracts = _risk_band_contracts_for_side(contracts, spot, top_k=top_k)
        elif contract_selection_mode == "nearest":
            selected_contracts = [
                (c, _risk_band_for_contract(c))
                for c in sorted(contracts, key=lambda c: _surface_contract_sort_key(c, spot))[:top_k]
            ]
        else:
            raise ValueError(f"unknown contract_selection_mode: {contract_selection_mode}")
        for slot, (contract, slot_role) in enumerate(selected_contracts):
            token_idx = side_idx * top_k + slot
            token_features[token_idx] = _contract_feature_row(contract, spot, slot_role)
            token_mask[token_idx] = 1.0
            token_strikes[token_idx] = float(contract.strike)

            action_idx = token_idx + 1
            action_labels["available_mask"][action_idx] = 1.0

            cid = cid_lookup.get((float(contract.strike), contract.right))
            if cid is None or cid not in paths:
                continue
            path = paths[cid]
            entry_bar = int(bar.bar_index) + 1
            if entry_bar >= len(path.mids) or not np.isfinite(path.mids[entry_bar]):
                continue
            entry_mid = float(path.mids[entry_bar])
            entry_spread_frac = (
                float(path.spread_fracs[entry_bar])
                if np.isfinite(path.spread_fracs[entry_bar])
                else float(contract.spread_fraction)
            )
            action_labels["tradeable_mask"][action_idx] = 1.0 if contract.passed else 0.0

            time_stop_pnl = _time_stop_pnl(
                path.mids,
                entry_bar=entry_bar,
                entry_mid=entry_mid,
                entry_spread_frac=entry_spread_frac,
                session_end_bar=375,
                commission=1.0,
            )
            horizon_pnl = _horizon_pnl(
                path.mids,
                entry_bar=entry_bar,
                entry_mid=entry_mid,
                entry_spread_frac=entry_spread_frac,
                horizon_bars=int(utility_horizon_bars),
                session_end_bar=375,
                commission=1.0,
            )
            best_exit_pnl, _ = _best_and_worst_forward_pnl(
                path,
                entry_bar=entry_bar,
                entry_mid=entry_mid,
                entry_spread_frac=entry_spread_frac,
                session_end_bar=375,
                commission=1.0,
            )
            mfe_5, mae_5 = _mfe_mae_at_horizon(path, entry_bar, entry_mid, 5, 375)
            mfe_10, mae_10 = _mfe_mae_at_horizon(path, entry_bar, entry_mid, 10, 375)
            mfe_20, mae_20 = _mfe_mae_at_horizon(path, entry_bar, entry_mid, 20, 375)
            stopout_risk = _stopout_before_target(
                path.mids,
                entry_bar=entry_bar,
                entry_mid=entry_mid,
                horizon_bars=stopout_horizon_bars,
                stop_pct=mae_floor_pct,
                target_pct=stopout_target_pct,
            )
            return_on_premium = _return_on_premium(time_stop_pnl, entry_mid)
            spread_cost = _spread_cost_dollars(entry_mid, entry_spread_frac)
            hybrid = hybrid_live_utility(
                time_stop_pnl,
                entry_mid=entry_mid,
                spread_fraction=entry_spread_frac,
                stopout_risk=stopout_risk,
                entry_bar=entry_bar,
                exit_bar=None,
                session_end_bar=375,
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
            action_labels["return_on_premium"][action_idx] = float(return_on_premium)
            action_labels["win_label"][action_idx] = (
                1.0 if time_stop_pnl is not None and np.isfinite(time_stop_pnl) and time_stop_pnl > 0 else 0.0
            )
            action_labels["entry_fill_bar"][action_idx] = float(entry_bar)
            action_labels["entry_fill_mid"][action_idx] = float(entry_mid)
            action_labels["entry_spread_fraction"][action_idx] = float(entry_spread_frac)
            action_labels["spread_cost"][action_idx] = float(spread_cost)
            action_labels["hybrid_live_utility"][action_idx] = float(hybrid)
            action_labels["hybrid_live_utility_arcsinh"][action_idx] = (
                float(np.arcsinh(hybrid / 100.0)) if np.isfinite(hybrid) else np.nan
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
    contract_selection_mode: str = DEFAULT_CONTRACT_SELECTION_MODE,
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
                contract_selection_mode=contract_selection_mode,
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
                "label": f"{9 + (30 + execution_start_bar) // 60:02d}:{(30 + execution_start_bar) % 60:02d}-"
                f"{9 + (30 + execution_end_bar) // 60:02d}:{(30 + execution_end_bar) % 60:02d} ET",
            },
            "contract_selection_mode": str(contract_selection_mode),
            "label_timing": {
                "decision_bar_observed": "bar_t",
                "entry_fill_bar": "bar_t_plus_1",
                "reason": "live executable decisions observe a completed bar before sending an order",
            },
            "risk_band_slots": [
                {"role": name, "abs_delta_min": float(lo), "abs_delta_max": float(hi), "slots_per_side": int(count)}
                for name, lo, hi, count in RISK_BAND_SLOTS
            ],
            "risk_band_codes": {k: float(v) for k, v in RISK_BAND_CODES.items()},
            "moneyness_codes": {k: float(v) for k, v in MONEYNESS_CODES.items()},
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
                    "slot_role": (
                        "high_delta"
                        if (i % top_k_contracts) <= 2
                        else "balanced"
                        if (i % top_k_contracts) <= 6
                        else "convex"
                        if (i % top_k_contracts) <= 10
                        else "efficiency"
                    ),
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
            "hybrid_live_utility": {
                "base": "time_stop_pnl in exported artifact; train_unified_policy recomputes from simulated_l3 when requested",
                "dollar_weight": 1.0,
                "return_on_premium_weight_dollars": 125.0,
                "spread_cost_penalty_weight": 0.35,
                "stopout_penalty_dollars": 60.0,
                "late_hold_penalty_after_bars": 90,
            },
        }
    )
    meta["dataset_fingerprint"] = action_surface_dataset_fingerprint(df, meta)

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
