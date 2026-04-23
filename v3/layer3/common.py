from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import build_labeled_day, load_json, replay_metrics_from_pnls
from v3.layer25.common import DEFAULT_OUT_DIR as DEFAULT_LAYER25_DIR
from v3.layer25.common import policy_trades
from v3.logger.builder import select_contract
from v3.oracles.exit_headroom import (
    DEFAULT_COMMISSION_PER_CONTRACT,
    DEFAULT_SESSION_END_BAR,
)
from v3.oracles.opportunity import (
    _build_contract_paths,
    _contract_idx_for_record,
)


DEFAULT_SURFACE_DIR = DEFAULT_LAYER25_DIR
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer3_rolling_entry_patience")
DEFAULT_UNIFIED_CHOSEN_TRADES = os.path.join(
    "v3",
    "artifacts",
    "v3_unified_promo_001",
    "seed_42",
    "chosen_trades.pkl",
)
DEFAULT_EXIT_THRESHOLDS = (0.15, 0.19, 0.20, 0.25, 0.30)
DEFAULT_EQUITY = 25_000.0
DEFAULT_SEED = 42
DEFAULT_MIN_TRAIN_TRADES = 40
DEFAULT_FOLD0_FALLBACK = "time_stop"
DEFAULT_FOLD0_FALLBACK_BARS = 90

TRADE_STATE_NAMES = [
    "bars_since_entry",
    "bars_to_session_end",
    "current_pnl_norm",
    "mfe_norm",
    "mae_norm",
    "mfe_bar_age",
    "direction_is_call",
]


def load_layer25_policy(
    surface_dir: str,
    patience_threshold: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report_path = os.path.join(surface_dir, "entry_patience_surface.json")
    csv_path = os.path.join(surface_dir, "surface_predictions.csv")

    report = load_json(report_path)
    surface_df = pd.read_csv(csv_path)
    threshold = (
        float(report["recommended_threshold_by_pf"]["threshold"])
        if patience_threshold is None
        else float(patience_threshold)
    )
    thresholds_by_window = {
        int(k): v for k, v in report["policy_meta"]["thresholds_by_window"].items()
    }
    manifest = {
        "direction_mode": report["policy_meta"]["direction_mode"],
        "score_mode": report["policy_meta"]["score_mode"],
        "side_score_weight": float(report["policy_meta"]["side_score_weight"]),
    }
    trades = policy_trades(
        surface_df,
        manifest=manifest,
        thresholds_by_window=thresholds_by_window,
        patience_threshold=threshold,
    ).copy()
    if trades.empty:
        raise RuntimeError("Layer 2.5 policy produced zero trades")
    trades["direction"] = trades["effective_direction"].astype(str)
    trades["time_stop_pnl"] = trades["pnl"].astype(float)
    trades["entry_patience_threshold"] = float(threshold)
    return trades, {
        "source": "layer25",
        "threshold": float(threshold),
        "direction_mode": report["policy_meta"]["direction_mode"],
        "score_mode": report["policy_meta"]["score_mode"],
        "side_score_weight": float(report["policy_meta"]["side_score_weight"]),
    }


def load_unified_policy_trades(chosen_trades_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    trades = pd.read_pickle(chosen_trades_path).copy()
    if trades.empty:
        raise RuntimeError(f"Unified-policy chosen trades are empty: {chosen_trades_path}")

    trades = trades.loc[trades["chosen_action_id"].fillna(0).astype(int) > 0].copy()
    if trades.empty:
        raise RuntimeError(f"Unified-policy artifact has no executable trades: {chosen_trades_path}")

    direction = trades["chosen_side"].astype(str).str.lower()
    trades["direction"] = direction
    trades["time_stop_pnl"] = trades["chosen_time_stop_pnl"].astype(float)
    trades["clean_entry_prob"] = trades["pred_clean_entry_prob"].astype(float)
    trades["selected_strike"] = trades["chosen_strike"].astype(float)
    trades["selected_right"] = np.where(direction == "put", "P", "C")
    return trades, {
        "source": "unified_policy",
        "chosen_trades_path": chosen_trades_path,
        "n_trades": int(len(trades)),
        "call_share": float((direction == "call").mean()),
        "put_share": float((direction == "put").mean()),
    }


def _build_trade_data(
    trade: pd.Series,
    log: Any,
    sidecar: dict,
    paths_cache: dict[int, Any],
    minute_map_cache: dict[str, dict[int, int]],
    ds: V2Dataset,
    session_end_bar: int,
    commission: float,
) -> dict[str, Any] | None:
    bar_index = int(trade["bar_index"])
    direction = str(trade["direction"])
    day = str(trade["day"])
    bar = next((b for b in log.bars if b.bar_index == bar_index), None)
    if bar is None:
        return None

    preferred_strike = trade.get("selected_strike", np.nan)
    preferred_right = trade.get("selected_right", "")
    match = None
    if np.isfinite(preferred_strike) and preferred_right in {"C", "P"}:
        match = next(
            (
                c
                for c in bar.contracts
                if float(c.strike) == float(preferred_strike) and c.right == preferred_right
            ),
            None,
        )
    if match is None:
        sel = select_contract(bar.contracts, direction, "layer2")
        if sel is None:
            return None
        right = "P" if direction == "put" else "C"
        match = next(
            (c for c in bar.contracts if c.strike == sel.strike and c.right == right),
            None,
        )
    if match is None:
        return None

    cache_key = id(sidecar)
    paths = paths_cache.get(cache_key)
    if paths is None:
        paths = _build_contract_paths(sidecar, 390)
        paths_cache[cache_key] = paths

    cid = _contract_idx_for_record(sidecar, bar_index, match)
    if cid is None or cid not in paths:
        return None
    path = paths[cid]

    if day not in minute_map_cache:
        start, end = ds.day_bar_range(day)
        minute_map_cache[day] = {int(ds.bar_of_day[i]): i for i in range(start, end)}
    minute_map = minute_map_cache[day]

    entry_mid = float(match.mid)
    entry_sf = float(match.spread_fraction)
    entry_premium_dollars = entry_mid * 100.0
    entry_ask = entry_mid * (1.0 + entry_sf / 2.0)
    end_bar = min(session_end_bar, len(path.mids) - 1)

    per_bar: list[dict[str, Any]] = []
    for t in range(bar_index + 1, end_bar + 1):
        mid = path.mids[t]
        if not np.isfinite(mid):
            continue
        # Match the current Layer-2 / Layer-2.5 time-stop convention so
        # the exit layer is evaluated on the same economics as the entry stack.
        exit_bid = float(mid) * (1.0 - entry_sf / 2.0)
        current_pnl = 100.0 * (exit_bid - entry_ask) - commission
        abs_idx = minute_map.get(t)
        if abs_idx is None:
            continue
        per_bar.append(
            {
                "bar": t,
                "current_pnl": current_pnl,
                "state_features": ds.X_sim[abs_idx].astype(np.float32),
            }
        )

    if len(per_bar) < 2:
        return None

    suffix_max = np.full(len(per_bar), -np.inf, dtype=np.float64)
    for i in range(len(per_bar) - 2, -1, -1):
        suffix_max[i] = max(per_bar[i + 1]["current_pnl"], suffix_max[i + 1])

    mfe = -np.inf
    mae = np.inf
    mfe_bar = bar_index
    denom = max(abs(entry_premium_dollars), 1e-9)
    for i, payload in enumerate(per_bar):
        current_pnl = payload["current_pnl"]
        t = payload["bar"]
        if current_pnl > mfe:
            mfe = current_pnl
            mfe_bar = t
        if current_pnl < mae:
            mae = current_pnl
        trade_state = np.array(
            [
                t - bar_index,
                end_bar - t,
                current_pnl / denom,
                mfe / denom,
                mae / denom,
                t - mfe_bar,
                1.0 if direction == "call" else 0.0,
            ],
            dtype=np.float32,
        )
        payload["trade_state"] = trade_state
        payload["target"] = int(current_pnl >= suffix_max[i]) if np.isfinite(suffix_max[i]) else 1

    return {
        "window_idx": int(trade["window_idx"]),
        "day": day,
        "entry_bar": bar_index,
        "direction": direction,
        "selected_strike": float(match.strike),
        "selected_right": str(match.right),
        "clean_entry_prob": float(trade.get("clean_entry_prob", np.nan)),
        "time_stop_pnl": float(trade["time_stop_pnl"]),
        "per_bar": per_bar,
    }


def build_trade_dataset(
    trades: pd.DataFrame,
    equity: float,
    session_end_bar: int,
    commission: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    day_cache: dict[str, tuple[Any, Any]] = {}
    paths_cache: dict[int, Any] = {}
    minute_map_cache: dict[str, dict[int, int]] = {}

    trade_data: list[dict[str, Any]] = []
    skipped = 0
    for trade in trades.itertuples(index=False):
        trade_series = pd.Series(trade._asdict())
        day = str(trade_series["day"])
        if day not in day_cache:
            day_cache[day] = build_labeled_day(ds, day, cfg, equity=equity)
        log, sidecar = day_cache[day]
        if log is None or sidecar is None:
            skipped += 1
            continue
        payload = _build_trade_data(
            trade_series,
            log,
            sidecar,
            paths_cache,
            minute_map_cache,
            ds,
            session_end_bar,
            commission,
        )
        if payload is None:
            skipped += 1
            continue
        trade_data.append(payload)

    return trade_data, {
        "n_input_trades": int(len(trades)),
        "n_trade_datasets": int(len(trade_data)),
        "n_skipped": int(skipped),
    }


def _flatten_to_rows(trade_data: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for td in trade_data:
        for payload in td["per_bar"]:
            X_rows.append(np.concatenate([payload["state_features"], payload["trade_state"]]))
            y_rows.append(int(payload["target"]))
    return (
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int8),
    )


def train_models_by_window(
    trade_data: list[dict[str, Any]],
    seed: int,
    min_train_trades: int,
) -> tuple[dict[int, HistGradientBoostingClassifier | None], list[dict[str, Any]]]:
    models: dict[int, HistGradientBoostingClassifier | None] = {}
    reports: list[dict[str, Any]] = []

    for window_idx in sorted({int(td["window_idx"]) for td in trade_data}):
        train_data = [td for td in trade_data if int(td["window_idx"]) < window_idx]
        test_data = [td for td in trade_data if int(td["window_idx"]) == window_idx]
        report = {
            "window_idx": int(window_idx),
            "train_trades": int(len(train_data)),
            "test_trades": int(len(test_data)),
            "mode": "model",
            "train_rows": 0,
            "train_pos_rate": None,
        }

        if len(train_data) < min_train_trades:
            models[window_idx] = None
            report["mode"] = "fallback"
            reports.append(report)
            continue

        X_train, y_train = _flatten_to_rows(train_data)
        report["train_rows"] = int(len(X_train))
        report["train_pos_rate"] = float(y_train.mean()) if len(y_train) else None
        if len(X_train) == 0 or len(np.unique(y_train)) < 2:
            models[window_idx] = None
            report["mode"] = "fallback"
            reports.append(report)
            continue

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=200,
            max_depth=4,
            min_samples_leaf=50,
            random_state=seed + window_idx,
            early_stopping=False,
        )
        model.fit(X_train, y_train)
        models[window_idx] = model
        reports.append(report)

    return models, reports


def _time_stop_row(td: dict[str, Any], trigger: str) -> dict[str, Any]:
    last = td["per_bar"][-1]
    return {
        "window_idx": int(td["window_idx"]),
        "day": td["day"],
        "entry_bar": int(td["entry_bar"]),
        "exit_bar": int(last["bar"]),
        "direction": td["direction"],
        "exit_pnl": float(td["time_stop_pnl"]),
        "trigger": trigger,
        "bars_held": int(last["bar"] - td["entry_bar"]),
        "clean_entry_prob": float(td["clean_entry_prob"]),
    }


def replay_trade_set(
    trade_data: list[dict[str, Any]],
    model: HistGradientBoostingClassifier | None,
    threshold: float,
    *,
    fallback_policy: str,
    fallback_bars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for td in trade_data:
        if model is None:
            if fallback_policy == "time_of_day_90":
                chosen = None
                for payload in td["per_bar"]:
                    if (payload["bar"] - td["entry_bar"]) >= fallback_bars:
                        chosen = {
                            "window_idx": int(td["window_idx"]),
                            "day": td["day"],
                            "entry_bar": int(td["entry_bar"]),
                            "exit_bar": int(payload["bar"]),
                            "direction": td["direction"],
                            "exit_pnl": float(payload["current_pnl"]),
                            "trigger": "time_of_day_90",
                            "bars_held": int(payload["bar"] - td["entry_bar"]),
                            "clean_entry_prob": float(td["clean_entry_prob"]),
                        }
                        break
                rows.append(chosen if chosen is not None else _time_stop_row(td, "time_stop"))
            else:
                rows.append(_time_stop_row(td, "time_stop"))
            continue

        features = np.stack(
            [np.concatenate([payload["state_features"], payload["trade_state"]]) for payload in td["per_bar"]]
        )
        probs = model.predict_proba(features)[:, 1]
        chosen = None
        for i, payload in enumerate(td["per_bar"]):
            if probs[i] >= threshold:
                chosen = {
                    "window_idx": int(td["window_idx"]),
                    "day": td["day"],
                    "entry_bar": int(td["entry_bar"]),
                    "exit_bar": int(payload["bar"]),
                    "direction": td["direction"],
                    "exit_pnl": float(payload["current_pnl"]),
                    "trigger": "model",
                    "bars_held": int(payload["bar"] - td["entry_bar"]),
                    "clean_entry_prob": float(td["clean_entry_prob"]),
                    "max_exit_prob": float(np.max(probs)),
                }
                break
        rows.append(chosen if chosen is not None else _time_stop_row(td, "time_stop_fallback"))
    return rows


def agg_exit_metrics(trades_df: pd.DataFrame, equity: float) -> dict[str, float]:
    if trades_df.empty:
        return {
            "pf": 0.0,
            "max_dd_pct": 0.0,
            "mean_pnl": 0.0,
            "trades": 0.0,
            "mean_bars_held": 0.0,
        }
    ordered = trades_df.sort_values(["day", "entry_bar"])
    pnls = ordered["exit_pnl"].astype(float).tolist()
    metrics = replay_metrics_from_pnls(pnls, equity)
    metrics["trades"] = float(len(ordered))
    metrics["mean_pnl"] = float(ordered["exit_pnl"].mean())
    metrics["mean_bars_held"] = float(ordered["bars_held"].mean())
    return metrics


def baseline_time_stop_metrics(trade_data: list[dict[str, Any]], equity: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [_time_stop_row(td, "time_stop") for td in trade_data]
    return agg_exit_metrics(pd.DataFrame(rows), equity), rows
