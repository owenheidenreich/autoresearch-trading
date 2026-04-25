from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from v3.harness.rolling_windows import (
    generate_rolling_windows,
    print_window_summary,
    verify_windows,
)
from v3.layer2.action_surface_dataset import (
    DEFAULT_ACTION_SURFACE_DATASET_PATH,
    action_surface_dataset_fingerprint,
    hybrid_live_utility,
    required_action_labels_for_target,
    validate_action_surface_bundle,
)
from v3.layer2.common import (
    ensure_dir,
    load_export_bundle,
    replay_metrics_from_pnls,
    save_json,
    save_pickle,
)
from v3.layer2.unified_policy import train_unified_action_model


DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer2_unified_policy_rolling")
DEFAULT_BASELINE_PF = 1.132
DEFAULT_BASELINE_LAYER25_DD = 21.4
DEFAULT_SIZZLE_SLIPPAGE = (0.0, 10.0, 25.0)
OBJECTIVE_PNL_COL = "chosen_objective_pnl"
TIME_STOP_PNL_COL = "chosen_time_stop_pnl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the rolling Layer-2 unified action policy.")
    p.add_argument("--dataset", default=DEFAULT_ACTION_SURFACE_DATASET_PATH)
    p.add_argument("--run-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--tier", default="dev", choices=("smoke", "dev", "promotion"))
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default="", help="Optional comma-separated seed list. Promotion defaults to 3 seeds.")
    p.add_argument("--hidden-dim", type=int, default=160)
    p.add_argument("--seq-hidden-dim", type=int, default=96)
    p.add_argument("--contract-hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--max-epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--latest-only", action="store_true")
    # Regression keeps the scale signal; ranking is what actually decides
    # flat-vs-trade at the margin. After the smoke regression-heavy run
    # collapsed to call-only with a negative margin, ranking is now primary.
    p.add_argument("--w-regression", type=float, default=0.5)
    p.add_argument("--w-ranking", type=float, default=1.0)
    p.add_argument("--w-side-contrastive", type=float, default=0.0)
    p.add_argument("--w-clean", type=float, default=0.35)
    p.add_argument("--w-stopout", type=float, default=0.35)
    p.add_argument("--equity", type=float, default=25_000.0)
    # Utility-target blend for outer-loop retrain. alpha=0 reproduces the
    # time-stop target; alpha=1 uses the oracle best-exit upper bound.
    # The intermediate regime lets the entry policy learn from a composed
    # utility without collapsing flat vs trade.
    p.add_argument("--utility-blend", type=float, default=0.0,
                   help="Blend between time_stop_pnl (0.0) and best_exit_pnl (1.0)")
    # Hold-aware utility target. "time_stop" preserves the original target
    # (session-end PnL). "horizon" uses horizon_pnl, a fixed-horizon PnL
    # label tuned to the champion Layer-3 median hold. "simulated_l3" uses
    # the simulated-L3 oracle per candidate contract (requires --simulated-l3-oracle).
    p.add_argument("--utility-target", default="time_stop",
                   choices=("time_stop", "horizon", "simulated_l3", "hybrid_live"),
                   help="Base utility target before --utility-blend mixing.")
    p.add_argument("--simulated-l3-oracle", default="",
                   help="Path to .npz produced by v3.layer2.build_simulated_l3_oracle.")
    p.add_argument("--w-dollar", type=float, default=0.25)
    p.add_argument("--w-return", type=float, default=0.25)
    p.add_argument("--w-win", type=float, default=0.25)
    p.add_argument("--side-balance-weight", type=float, default=0.0,
                   help="Per-bar inverse-frequency sample weight for truth-best-side balance "
                        "(0.0=current, 1.0=full inverse-frequency). Multiplies into reg_weight "
                        "so regression/dollar/return heads see balanced gradient mass across "
                        "call-best vs put-best vs flat-best bars. Targets the 2.83:1 train "
                        "imbalance documented in spx_w_side_sweep_001 falsification.")
    p.add_argument("--golden-day", default="", help="Optional YYYY-MM-DD day to trace per epoch when it is in OOS.")
    p.add_argument(
        "--golden-day-out-dir",
        default="",
        help="Directory for per-epoch golden-day traces. Defaults under --run-dir.",
    )
    return p.parse_args()


def _resolve_device(arg: str) -> str:
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return "cuda"
    if arg == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds.strip():
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.tier == "promotion":
        return [args.seed, args.seed + 1, args.seed + 2]
    return [args.seed]


def _objective_label(utility_target: str, utility_blend: float) -> str:
    if utility_blend > 0.0:
        return f"{utility_target}+best_exit@{utility_blend:.2f}"
    return utility_target


def _apply_tier_defaults(args: argparse.Namespace, meta: dict[str, Any], resolved_device: str) -> None:
    if args.tier == "smoke":
        args.latest_only = True
        # 4 epochs was too short: the ranking head hadn't separated flat from
        # contracts before we stopped. 8 / patience 3 keeps smoke CPU-fast
        # (one window, ~30s/epoch) while giving enough budget to see whether
        # the new loss actually converges.
        args.max_epochs = min(args.max_epochs, 8)
        args.patience = min(args.patience, 3)
    if args.tier == "promotion" and resolved_device != "cuda":
        raise RuntimeError("Promotion tier requires GPU/CUDA per the new regimen.")
    if (
        meta["history_bars"] > 20
        or meta["top_k_contracts_per_side"] > 12
        or args.max_epochs > 20
    ) and resolved_device != "cuda":
        raise RuntimeError(
            "This experiment exceeds the CPU-only budget policy. "
            "Use CUDA for sequence>20, top_k>12, or max_epochs>20."
        )


def _slice_inputs(
    mask: np.ndarray,
    bundle: dict[str, Any],
    utility_blend: float = 0.0,
    utility_target: str = "time_stop",
    simulated_l3_pnl: np.ndarray | None = None,
    simulated_l3_exit_bar: np.ndarray | None = None,
) -> dict[str, Any]:
    rows: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    action_labels = bundle["action_labels"]
    time_stop_raw = action_labels["utility_raw"][mask]
    if utility_target == "horizon":
        if "horizon_pnl" not in action_labels:
            raise RuntimeError(
                "--utility-target=horizon requires a dataset exported with horizon_pnl "
                "(re-run v3.layer2.export_action_surface_dataset)."
            )
        base_raw = action_labels["horizon_pnl"][mask]
    elif utility_target in {"simulated_l3", "hybrid_live"}:
        if simulated_l3_pnl is None:
            raise RuntimeError(
                f"--utility-target={utility_target} requires --simulated-l3-oracle. "
                "Run v3.layer2.build_simulated_l3_oracle first."
            )
        base_raw = simulated_l3_pnl[mask]
    elif utility_target == "time_stop":
        base_raw = time_stop_raw
    else:
        raise ValueError(f"unknown utility_target: {utility_target}")

    if utility_target == "hybrid_live":
        entry_mid = action_labels["entry_fill_mid"][mask]
        entry_spread = action_labels["entry_spread_fraction"][mask]
        entry_bar = action_labels["entry_fill_bar"][mask]
        stopout = action_labels["stopout_risk"][mask]
        exit_bar = simulated_l3_exit_bar[mask] if simulated_l3_exit_bar is not None else np.full_like(base_raw, np.nan)
        utility_raw = np.full_like(base_raw, np.nan, dtype=np.float32)
        for row_i in range(base_raw.shape[0]):
            for action_i in range(base_raw.shape[1]):
                utility_raw[row_i, action_i] = np.float32(
                    hybrid_live_utility(
                        float(base_raw[row_i, action_i]) if np.isfinite(base_raw[row_i, action_i]) else None,
                        entry_mid=float(entry_mid[row_i, action_i]),
                        spread_fraction=float(entry_spread[row_i, action_i]),
                        stopout_risk=float(stopout[row_i, action_i]),
                        entry_bar=int(entry_bar[row_i, action_i]) if np.isfinite(entry_bar[row_i, action_i]) else 0,
                        exit_bar=int(exit_bar[row_i, action_i]) if np.isfinite(exit_bar[row_i, action_i]) and exit_bar[row_i, action_i] >= 0 else None,
                    )
                )
        utility_raw[:, 0] = 0.0
        utility_arcsinh = np.arcsinh(utility_raw / 100.0).astype(np.float32)
    elif utility_blend > 0.0:
        alpha = float(utility_blend)
        best_exit = action_labels["best_exit_pnl"][mask]
        nan_mask = ~np.isfinite(base_raw) | ~np.isfinite(best_exit)
        blended = (1.0 - alpha) * np.nan_to_num(base_raw, nan=0.0) + alpha * np.nan_to_num(best_exit, nan=0.0)
        blended[nan_mask] = np.nan
        utility_raw = blended.astype(np.float32)
        utility_arcsinh = np.arcsinh(utility_raw / 100.0).astype(np.float32)
    elif utility_target in ("horizon", "simulated_l3"):
        utility_raw = base_raw.astype(np.float32)
        utility_arcsinh = np.arcsinh(utility_raw / 100.0).astype(np.float32)
    else:
        utility_raw = time_stop_raw
        utility_arcsinh = action_labels["utility_arcsinh"][mask]
    entry_mid_all = action_labels.get("entry_fill_mid")
    if entry_mid_all is None:
        premium = np.ones_like(utility_raw, dtype=np.float32) * 100.0
    else:
        premium = np.maximum(entry_mid_all[mask].astype(np.float32) * 100.0, 1.0)
    return_multiple = (base_raw.astype(np.float32) / premium).astype(np.float32)
    return_multiple[:, 0] = 0.0
    dollar_target = np.arcsinh(base_raw.astype(np.float32) / 100.0).astype(np.float32)
    dollar_target[:, 0] = 0.0
    win_target = (base_raw > 0.0).astype(np.float32)
    win_target[:, 0] = 0.0

    contract_feature_names = list(meta.get("contract_feature_names", ()))
    if "risk_band" in contract_feature_names:
        risk_idx = contract_feature_names.index("risk_band")
        risk_tokens = bundle["contract_features"][mask, :, risk_idx].astype(np.float32)
    else:
        risk_tokens = np.zeros(bundle["contract_features"][mask].shape[:2], dtype=np.float32)
    risk_band = np.concatenate(
        [np.full((risk_tokens.shape[0], 1), -1.0, dtype=np.float32), risk_tokens],
        axis=1,
    )
    return_on_premium_label = (
        action_labels["return_on_premium"][mask]
        if "return_on_premium" in action_labels
        else np.full_like(utility_raw, np.nan, dtype=np.float32)
    )
    out = {
        "rows": rows.loc[mask].copy().reset_index(drop=True),
        "scalar": rows.loc[mask, meta["scalar_feature_names"]].to_numpy(dtype=np.float32),
        "seq": bundle["sequence_features"][mask],
        "seq_mask": bundle["sequence_mask"][mask],
        "contracts": bundle["contract_features"][mask],
        "contract_mask": bundle["contract_mask"][mask],
        "contract_strike": bundle["contract_strike"][mask],
        "utility": utility_arcsinh,
        "utility_raw": utility_raw,
        "dollar": dollar_target,
        "return_multiple": np.clip(return_multiple, -5.0, 5.0).astype(np.float32),
        "win": win_target,
        "risk_band": risk_band,
        "time_stop_raw": time_stop_raw,
        "return_on_premium": return_on_premium_label,
        "clean": action_labels["clean_entry"][mask],
        "stopout": action_labels["stopout_risk"][mask],
        "mae_10": action_labels["mae_10"][mask],
        "available_mask": np.nan_to_num(action_labels["available_mask"][mask], nan=0.0).astype(np.float32),
        "tradeable_mask": np.nan_to_num(action_labels["tradeable_mask"][mask], nan=0.0).astype(np.float32),
    }
    return out


def _timing_bucket(pnl: float, mae_10: float, mae_floor_pct: float = -20.0) -> str:
    if not np.isfinite(pnl) or not np.isfinite(mae_10):
        return "missing"
    if pnl > 0 and mae_10 > mae_floor_pct:
        return "clean_winner"
    if pnl > 0:
        return "shakeout_winner"
    if mae_10 <= mae_floor_pct:
        return "fast_loser"
    return "drift_loser"


def _prediction_frame(
    subset: dict[str, Any],
    pred: dict[str, np.ndarray],
    top_k_contracts: int,
    contract_feature_names: list[str] | None = None,
) -> pd.DataFrame:
    rows = subset["rows"].copy()
    utility = pred["utility"]
    clean_prob = pred["clean_prob"]
    stopout_prob = pred["stopout_prob"]
    win_prob = pred.get("win_prob", clean_prob)
    dollar_utility = pred.get("dollar_utility")
    return_multiple_pred = pred.get("return_multiple")
    tradeable_token_mask = np.nan_to_num(subset["tradeable_mask"][:, 1:], nan=0.0) > 0.5

    token_scores = utility[:, 1:].copy()
    token_scores[~tradeable_token_mask] = -np.inf
    best_token_idx = token_scores.argmax(axis=1)
    best_token_score = token_scores[np.arange(len(rows)), best_token_idx]
    has_trade = np.isfinite(best_token_score)
    chosen_action_id = np.where(has_trade, best_token_idx + 1, 0).astype(int)
    chosen_side = np.where(best_token_idx < top_k_contracts, "call", "put")
    chosen_side = np.where(has_trade, chosen_side, "")
    chosen_strike = np.where(
        has_trade,
        subset["contract_strike"][np.arange(len(rows)), best_token_idx],
        np.nan,
    )

    chosen_objective = np.zeros(len(rows), dtype=np.float32)
    chosen_time_stop = np.zeros(len(rows), dtype=np.float32)
    chosen_clean_prob = np.zeros(len(rows), dtype=np.float32)
    chosen_stopout_prob = np.zeros(len(rows), dtype=np.float32)
    chosen_win_prob = np.zeros(len(rows), dtype=np.float32)
    chosen_dollar_score = np.zeros(len(rows), dtype=np.float32)
    chosen_return_score = np.zeros(len(rows), dtype=np.float32)
    chosen_clean_label = np.zeros(len(rows), dtype=np.float32)
    chosen_stopout_label = np.zeros(len(rows), dtype=np.float32)
    chosen_mae_10 = np.full(len(rows), np.nan, dtype=np.float32)
    chosen_return_on_premium = np.full(len(rows), np.nan, dtype=np.float32)

    chosen_rows = np.arange(len(rows))[has_trade]
    chosen_cols = chosen_action_id[has_trade]
    chosen_token_cols = best_token_idx[has_trade]
    if len(chosen_rows) > 0:
        chosen_objective[chosen_rows] = subset["utility_raw"][chosen_rows, chosen_cols]
        chosen_time_stop[chosen_rows] = subset["time_stop_raw"][chosen_rows, chosen_cols]
        chosen_clean_prob[chosen_rows] = clean_prob[chosen_rows, chosen_cols]
        chosen_stopout_prob[chosen_rows] = stopout_prob[chosen_rows, chosen_cols]
        chosen_win_prob[chosen_rows] = win_prob[chosen_rows, chosen_cols]
        if dollar_utility is not None:
            chosen_dollar_score[chosen_rows] = dollar_utility[chosen_rows, chosen_cols]
        if return_multiple_pred is not None:
            chosen_return_score[chosen_rows] = return_multiple_pred[chosen_rows, chosen_cols]
        chosen_clean_label[chosen_rows] = subset["clean"][chosen_rows, chosen_cols]
        chosen_stopout_label[chosen_rows] = subset["stopout"][chosen_rows, chosen_cols]
        chosen_mae_10[chosen_rows] = subset["mae_10"][chosen_rows, chosen_cols]
        chosen_return_on_premium[chosen_rows] = subset["return_on_premium"][chosen_rows, chosen_cols]

    rows["flat_score"] = utility[:, 0]
    rows["best_nonflat_score"] = best_token_score
    rows["decision_margin"] = rows["best_nonflat_score"] - rows["flat_score"]
    rows["chosen_action_id"] = chosen_action_id
    rows["chosen_side"] = chosen_side
    rows["chosen_strike"] = chosen_strike
    rows[OBJECTIVE_PNL_COL] = chosen_objective
    rows["chosen_time_stop_pnl"] = chosen_time_stop
    rows["pred_clean_entry_prob"] = chosen_clean_prob
    rows["pred_stopout_risk"] = chosen_stopout_prob
    rows["pred_win_prob"] = chosen_win_prob
    rows["pred_dollar_score"] = chosen_dollar_score
    rows["pred_return_score"] = chosen_return_score
    rows["chosen_clean_entry_label"] = chosen_clean_label
    rows["chosen_stopout_label"] = chosen_stopout_label
    rows["chosen_mae_10"] = chosen_mae_10
    rows["chosen_return_on_premium"] = chosen_return_on_premium
    if contract_feature_names:
        feature_idx = {name: i for i, name in enumerate(contract_feature_names)}
        for feature_name, out_name in (
            ("premium", "chosen_premium"),
            ("abs_delta", "chosen_abs_delta"),
            ("slot_role", "chosen_slot_role"),
            ("moneyness_bucket", "chosen_moneyness_bucket"),
            ("risk_band", "chosen_risk_band"),
            ("spread_fraction", "chosen_spread_fraction"),
        ):
            values = np.full(len(rows), np.nan, dtype=np.float32)
            if feature_name in feature_idx and len(chosen_rows) > 0:
                values[chosen_rows] = subset["contracts"][chosen_rows, chosen_token_cols, feature_idx[feature_name]]
            rows[out_name] = values
    rows["chosen_time_bucket"] = [
        _timing_bucket(float(pnl), float(mae_10))
        for pnl, mae_10 in zip(chosen_time_stop, chosen_mae_10)
    ]
    rows["beats_v1_same_bar"] = (
        rows["chosen_action_id"] > 0
    ) & (
        rows["chosen_time_stop_pnl"] > rows["time_stop_pnl_put"].fillna(-np.inf)
    )
    return rows


def _golden_epoch_trace_rows(
    subset: dict[str, Any],
    pred: dict[str, np.ndarray],
    *,
    epoch: int,
    epoch_parts: dict[str, float],
    top_k_contracts: int,
) -> list[dict[str, Any]]:
    rows = subset["rows"].reset_index(drop=True)
    utility = pred["utility"]
    labels = subset["utility_raw"]
    tradeable = np.nan_to_num(subset["tradeable_mask"], nan=0.0) > 0.5
    token_tradeable = tradeable[:, 1:]
    token_scores = utility[:, 1:].copy()
    token_scores[~token_tradeable] = -np.inf
    token_labels = labels[:, 1:].copy()
    token_labels[~token_tradeable] = -np.inf

    out: list[dict[str, Any]] = []
    for i, row in rows.iterrows():
        call_slice = slice(0, top_k_contracts)
        put_slice = slice(top_k_contracts, top_k_contracts * 2)
        call_scores = token_scores[i, call_slice]
        put_scores = token_scores[i, put_slice]
        call_labels = token_labels[i, call_slice]
        put_labels = token_labels[i, put_slice]
        best_call_score = float(np.nanmax(call_scores)) if np.isfinite(call_scores).any() else float("nan")
        best_put_score = float(np.nanmax(put_scores)) if np.isfinite(put_scores).any() else float("nan")
        best_call_label = float(np.nanmax(call_labels)) if np.isfinite(call_labels).any() else float("nan")
        best_put_label = float(np.nanmax(put_labels)) if np.isfinite(put_labels).any() else float("nan")
        pred_action = int(np.nanargmax(np.where(tradeable[i], utility[i], -np.inf)))
        true_action = int(np.nanargmax(np.where(tradeable[i], labels[i], -np.inf)))
        out.append(
            {
                "epoch": int(epoch),
                "val_loss": float(epoch_parts.get("val_loss", np.nan)),
                "bar_index": int(row["bar_index"]),
                "flat_score": float(utility[i, 0]),
                "best_call_score": best_call_score,
                "best_put_score": best_put_score,
                "score_put_minus_call": float(best_put_score - best_call_score)
                if np.isfinite(best_put_score) and np.isfinite(best_call_score)
                else float("nan"),
                "best_call_label": best_call_label,
                "best_put_label": best_put_label,
                "label_put_minus_call": float(best_put_label - best_call_label)
                if np.isfinite(best_put_label) and np.isfinite(best_call_label)
                else float("nan"),
                "chosen_action_id": pred_action,
                "chosen_side": "flat" if pred_action == 0 else ("call" if pred_action <= top_k_contracts else "put"),
                "true_best_action_id": true_action,
                "true_best_side": "flat" if true_action == 0 else ("call" if true_action <= top_k_contracts else "put"),
            }
        )
    return out


def _select_daily_trades(
    pred_df: pd.DataFrame,
    decision_margin: float | None = None,
    *,
    abstention_policy: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if abstention_policy is not None:
        decision_margin = float(abstention_policy.get("decision_margin", 0.0))
        min_win_prob = float(abstention_policy.get("min_win_prob", 0.0))
        max_stopout_prob = float(abstention_policy.get("max_stopout_prob", 1.0))
        min_l3_train_trades = int(abstention_policy.get("min_l3_train_trades", 0))
    else:
        decision_margin = 0.0 if decision_margin is None else float(decision_margin)
        min_win_prob = 0.0
        max_stopout_prob = 1.0
        min_l3_train_trades = 0
    eligible_mask = (
        (pred_df["chosen_action_id"] > 0)
        & np.isfinite(pred_df["best_nonflat_score"])
        & (pred_df["decision_margin"] >= decision_margin)
        & (pred_df["pred_win_prob"].fillna(0.0) >= min_win_prob)
        & (pred_df["pred_stopout_risk"].fillna(1.0) <= max_stopout_prob)
    )
    if min_l3_train_trades > 0 and "l3_train_trades" in pred_df:
        eligible_mask &= pred_df["l3_train_trades"].fillna(0).astype(int) >= min_l3_train_trades
    eligible = pred_df[eligible_mask].copy()
    if eligible.empty:
        return eligible
    idx = eligible.groupby("day")["best_nonflat_score"].idxmax()
    return eligible.loc[idx].sort_values(["day", "bar_index"]).reset_index(drop=True)


def _stress_metrics(trades: pd.DataFrame, equity: float, slippage: float) -> dict[str, float]:
    return _stress_metrics_for_column(trades, equity, slippage, pnl_col=TIME_STOP_PNL_COL)


def _stress_metrics_for_column(
    trades: pd.DataFrame,
    equity: float,
    slippage: float,
    *,
    pnl_col: str,
) -> dict[str, float]:
    stressed = (trades[pnl_col] - float(slippage)).tolist() if not trades.empty else []
    metrics = replay_metrics_from_pnls(stressed, equity)
    return {
        "slippage_round_trip": float(slippage),
        "pf": float(metrics["pf"]) if stressed else 0.0,
        "max_dd_pct": float(metrics["max_dd_pct"]),
        "mean_pnl": float(metrics["mean_pnl"]),
        "trades": int(len(stressed)),
    }


def _trade_metrics(
    trades: pd.DataFrame,
    equity: float,
    total_days: int,
    *,
    pnl_col: str = TIME_STOP_PNL_COL,
) -> dict[str, Any]:
    metrics = replay_metrics_from_pnls(trades[pnl_col].tolist() if not trades.empty else [], equity)
    bucket_counts = trades["chosen_time_bucket"].value_counts().to_dict() if not trades.empty else {}
    return {
        "trades": int(len(trades)),
        "trade_share": float(len(trades) / max(total_days, 1)),
        "pf": float(metrics["pf"]) if not trades.empty else 0.0,
        "max_dd_pct": float(metrics["max_dd_pct"]),
        "mean_pnl": float(metrics["mean_pnl"]),
        "clean_entry_rate": float((trades["chosen_time_bucket"] == "clean_winner").mean()) if not trades.empty else 0.0,
        "fast_loser_rate": float((trades["chosen_time_bucket"] == "fast_loser").mean()) if not trades.empty else 0.0,
        "shakeout_winner_rate": float((trades["chosen_time_bucket"] == "shakeout_winner").mean()) if not trades.empty else 0.0,
        "beats_v1_same_bar_rate": float(trades["beats_v1_same_bar"].mean()) if not trades.empty else 0.0,
        "timing_bucket_counts": {str(k): int(v) for k, v in bucket_counts.items()},
    }


def _calibrate_decision_margin(
    val_pred: pd.DataFrame,
    equity: float,
    *,
    pnl_col: str = OBJECTIVE_PNL_COL,
) -> dict[str, Any]:
    total_days = int(val_pred["day"].nunique())
    candidate_base = val_pred.loc[
        np.isfinite(val_pred["decision_margin"]) & (val_pred["chosen_action_id"] > 0),
        "decision_margin",
    ].to_numpy(dtype=np.float64)
    if candidate_base.size == 0:
        return {
            "decision_margin": float("inf"),
            "objective_pf": 0.0,
            "objective_mean_pnl": 0.0,
            "trade_share": 0.0,
            "pf": 0.0,
            "trades": 0,
            "pf_qualified": False,
            "in_band": False,
        }

    # Never calibrate to a negative margin: trading when the model's own flat
    # prediction beats its best contract pick is incoherent. The smoke pass
    # produced a -0.083 margin on mean-PnL maximization, which turned the model
    # into a nearly-always-trade call-only gambler. Floor the grid at 0.
    positive_margins = candidate_base[candidate_base >= 0.0]
    if positive_margins.size >= 5:
        quantiles = np.quantile(positive_margins, np.linspace(0.0, 0.95, 20))
    else:
        quantiles = np.array([], dtype=np.float64)
    grid = np.unique(
        np.concatenate(
            ([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18], quantiles)
        )
    )
    grid = grid[grid >= 0.0]

    # Require a meaningful sample before trusting PF. For a 40-day validation
    # window this is ~6 trades; for a 60-day one it is ~9. Below that, we need
    # more evidence than PF can provide at the validation layer.
    min_trades_required = max(6, int(round(0.10 * total_days)))

    best_payload = None
    best_key = None
    for margin in grid:
        trades = _select_daily_trades(val_pred, float(margin))
        metrics = _trade_metrics(trades, equity, total_days, pnl_col=pnl_col)
        in_band = 0.25 <= metrics["trade_share"] <= 0.70
        enough_trades = metrics["trades"] >= min_trades_required
        pf_qualified = enough_trades and float(metrics["pf"]) >= 1.0
        # Lexicographic priority: PF-qualified + in-band > PF-qualified alone >
        # enough-trades (even if PF<1, gives us signal to tune with) > raw PF.
        # Within each tier, prefer higher PF, then higher mean_pnl, then more
        # trades so ties break toward the richer sample.
        key = (
            1 if (pf_qualified and in_band) else 0,
            1 if pf_qualified else 0,
            1 if enough_trades else 0,
            float(metrics["pf"]),
            float(metrics["mean_pnl"]),
            float(metrics["trades"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_payload = {
                "decision_margin": float(margin),
                "objective_pf": float(metrics["pf"]),
                "objective_mean_pnl": float(metrics["mean_pnl"]),
                "trade_share": float(metrics["trade_share"]),
                "pf": float(metrics["pf"]),
                "trades": int(metrics["trades"]),
                "pf_qualified": bool(pf_qualified),
                "in_band": bool(in_band),
            }
    assert best_payload is not None
    return best_payload


def _calibrate_abstention_policy(
    val_pred: pd.DataFrame,
    equity: float,
    *,
    pnl_col: str = OBJECTIVE_PNL_COL,
) -> dict[str, Any]:
    total_days = int(val_pred["day"].nunique())
    candidate_base = val_pred.loc[
        np.isfinite(val_pred["decision_margin"]) & (val_pred["chosen_action_id"] > 0),
        "decision_margin",
    ].to_numpy(dtype=np.float64)
    if candidate_base.size == 0:
        return {
            "decision_margin": float("inf"),
            "min_win_prob": 1.0,
            "max_stopout_prob": 0.0,
            "objective_pf": 0.0,
            "objective_mean_pnl": 0.0,
            "trade_share": 0.0,
            "pf": 0.0,
            "trades": 0,
            "pf_qualified": False,
            "in_band": False,
            "min_l3_train_trades": 40,
        }

    positive_margins = candidate_base[candidate_base >= 0.0]
    quantiles = (
        np.quantile(positive_margins, np.linspace(0.0, 0.95, 16))
        if positive_margins.size >= 5
        else np.array([], dtype=np.float64)
    )
    margin_grid = np.unique(np.concatenate(([0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18], quantiles)))
    margin_grid = margin_grid[margin_grid >= 0.0]
    win_grid = [0.0, 0.35, 0.45, 0.55]
    stopout_grid = [1.0, 0.70, 0.55, 0.45, 0.35]
    min_trades_required = max(6, int(round(0.10 * total_days)))

    best_payload = None
    best_key = None
    for margin in margin_grid:
        for min_win in win_grid:
            for max_stopout in stopout_grid:
                policy = {
                    "decision_margin": float(margin),
                    "min_win_prob": float(min_win),
                    "max_stopout_prob": float(max_stopout),
                    "min_l3_train_trades": 40,
                }
                trades = _select_daily_trades(val_pred, abstention_policy=policy)
                metrics = _trade_metrics(trades, equity, total_days, pnl_col=pnl_col)
                in_band = 0.20 <= metrics["trade_share"] <= 0.70
                enough_trades = metrics["trades"] >= min_trades_required
                pf_qualified = enough_trades and float(metrics["pf"]) >= 1.0
                key = (
                    1 if (pf_qualified and in_band) else 0,
                    1 if pf_qualified else 0,
                    1 if enough_trades else 0,
                    float(metrics["pf"]),
                    float(metrics["mean_pnl"]),
                    -abs(float(metrics["trade_share"]) - 0.45),
                    float(metrics["trades"]),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_payload = {
                        **policy,
                        "objective_pf": float(metrics["pf"]),
                        "objective_mean_pnl": float(metrics["mean_pnl"]),
                        "trade_share": float(metrics["trade_share"]),
                        "pf": float(metrics["pf"]),
                        "trades": int(metrics["trades"]),
                        "pf_qualified": bool(pf_qualified),
                        "in_band": bool(in_band),
                    }
    assert best_payload is not None
    return best_payload


def _window_report(
    window_idx: int,
    trades: pd.DataFrame,
    equity: float,
    total_days: int,
    decision_margin: float,
    *,
    objective_label: str,
    objective_matches_time_stop: bool,
) -> dict[str, Any]:
    metrics = _trade_metrics(trades, equity, total_days, pnl_col=OBJECTIVE_PNL_COL)
    report = {
        "window_idx": int(window_idx),
        "decision_margin": float(decision_margin),
        "calibration_objective": objective_label,
        **metrics,
    }
    if not objective_matches_time_stop:
        report["time_stop_reference"] = _trade_metrics(
            trades,
            equity,
            total_days,
            pnl_col=TIME_STOP_PNL_COL,
        )
    return report


def _aggregate_report(
    chosen_trades: pd.DataFrame,
    per_window: list[dict[str, Any]],
    total_days: int,
    equity: float,
    seed: int,
    tier: str,
    *,
    objective_label: str,
    objective_matches_time_stop: bool,
) -> dict[str, Any]:
    metrics = _trade_metrics(chosen_trades, equity, total_days, pnl_col=OBJECTIVE_PNL_COL)
    gate_metrics = metrics
    time_stop_reference = None
    slippage_time_stop_reference = None
    if not objective_matches_time_stop:
        time_stop_reference = _trade_metrics(
            chosen_trades,
            equity,
            total_days,
            pnl_col=TIME_STOP_PNL_COL,
        )
        gate_metrics = time_stop_reference
        slippage_time_stop_reference = {
            str(int(slip)): _stress_metrics_for_column(
                chosen_trades,
                equity,
                float(slip),
                pnl_col=TIME_STOP_PNL_COL,
            )
            for slip in DEFAULT_SIZZLE_SLIPPAGE
        }
    window_pfs = np.asarray([w["pf"] for w in per_window], dtype=np.float64)
    slippage = {
        str(int(slip)): _stress_metrics_for_column(
            chosen_trades,
            equity,
            float(slip),
            pnl_col=OBJECTIVE_PNL_COL,
        )
        for slip in DEFAULT_SIZZLE_SLIPPAGE
    }
    report = {
        "seed": int(seed),
        "tier": tier,
        "calibration_objective": objective_label,
        "aggregate": metrics,
        "per_window": per_window,
        "per_window_pf": {
            "mean": float(np.nanmean(window_pfs)) if len(window_pfs) else None,
            "std": float(np.nanstd(window_pfs)) if len(window_pfs) else None,
            "min": float(np.nanmin(window_pfs)) if len(window_pfs) else None,
            "max": float(np.nanmax(window_pfs)) if len(window_pfs) else None,
        },
        "slippage_stress": slippage,
        "promotion_gates": {
            "basis": "objective" if objective_matches_time_stop else "time_stop_reference",
            "w1_gate_pf_vs_baseline": float(gate_metrics["pf"]) > DEFAULT_BASELINE_PF,
            "w1_gate_dd_vs_layer25": (
                float(gate_metrics["max_dd_pct"]) <= DEFAULT_BASELINE_LAYER25_DD
                or float(gate_metrics["pf"]) >= DEFAULT_BASELINE_PF + 0.15
            ),
            "w1_gate_trade_share": 0.25 <= float(gate_metrics["trade_share"]) <= 0.70,
            "patience_gate_fast_loser_improvement_vs_old_baseline": float(gate_metrics["fast_loser_rate"]) <= (113.0 / 432.0) * 0.80,
            "exit_gate_placeholder": False,
        },
    }
    report["chosen_contract_diagnostics"] = _chosen_contract_diagnostics(chosen_trades)
    if time_stop_reference is not None:
        time_stop_window_pfs = np.asarray(
            [w["time_stop_reference"]["pf"] for w in per_window],
            dtype=np.float64,
        )
        report["aggregate_time_stop_reference"] = time_stop_reference
        report["per_window_pf_time_stop_reference"] = {
            "mean": float(np.nanmean(time_stop_window_pfs)) if len(time_stop_window_pfs) else None,
            "std": float(np.nanstd(time_stop_window_pfs)) if len(time_stop_window_pfs) else None,
            "min": float(np.nanmin(time_stop_window_pfs)) if len(time_stop_window_pfs) else None,
            "max": float(np.nanmax(time_stop_window_pfs)) if len(time_stop_window_pfs) else None,
        }
        report["slippage_stress_time_stop_reference"] = slippage_time_stop_reference
    return report


def _chosen_contract_diagnostics(chosen_trades: pd.DataFrame) -> dict[str, Any]:
    if chosen_trades.empty:
        return {"trades": 0}
    out: dict[str, Any] = {
        "trades": int(len(chosen_trades)),
        "side_counts": {str(k): int(v) for k, v in chosen_trades["chosen_side"].value_counts().items()},
    }
    for col in (
        "chosen_risk_band",
        "chosen_slot_role",
        "chosen_moneyness_bucket",
    ):
        if col in chosen_trades:
            out[f"{col}_counts"] = {
                str(k): int(v)
                for k, v in chosen_trades[col].round(3).astype(str).value_counts().items()
            }
    for col in (
        "chosen_premium",
        "chosen_abs_delta",
        "chosen_spread_fraction",
        "chosen_return_on_premium",
        "pred_win_prob",
        "pred_stopout_risk",
    ):
        if col in chosen_trades:
            values = chosen_trades[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if not values.empty:
                out[col] = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q10": float(values.quantile(0.10)),
                    "q90": float(values.quantile(0.90)),
                }
    if "bar_index" in chosen_trades:
        out["bar_index_counts"] = {
            str(k): int(v)
            for k, v in pd.cut(
                chosen_trades["bar_index"].astype(int),
                bins=[14, 30, 60, 90, 120, 180, 270, 389],
                labels=["0945-1000", "1001-1030", "1031-1100", "1101-1130", "1131-1230", "1231-1400", "1401-1559"],
            ).value_counts(sort=False).items()
        }
    return out


def _run_single_seed(args: argparse.Namespace, seed: int, device: str) -> dict[str, Any]:
    ensure_dir(args.run_dir)
    seed_dir = os.path.join(args.run_dir, f"seed_{seed}")
    ensure_dir(seed_dir)

    bundle = load_export_bundle(args.dataset)
    rows: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    validate_action_surface_bundle(
        bundle,
        required_labels=required_action_labels_for_target(args.utility_target),
        required_contract_features=(
            {"risk_band", "slot_role", "moneyness_bucket"}
            if args.utility_target == "hybrid_live"
            else set()
        ),
    )
    objective_label = _objective_label(args.utility_target, float(args.utility_blend))
    objective_matches_time_stop = args.utility_target == "time_stop" and float(args.utility_blend) == 0.0

    simulated_l3_pnl = None
    simulated_l3_exit_bar = None
    if args.utility_target in {"simulated_l3", "hybrid_live"}:
        if not args.simulated_l3_oracle:
            raise RuntimeError(f"--utility-target={args.utility_target} requires --simulated-l3-oracle <npz>")
        z = np.load(args.simulated_l3_oracle, allow_pickle=True)
        simulated_l3_pnl = z["l3_exit_pnl"]
        simulated_l3_exit_bar = z["l3_exit_bar"] if "l3_exit_bar" in z.files else None
        if simulated_l3_pnl.shape[0] != len(rows):
            raise RuntimeError(
                f"simulated-L3 oracle row count {simulated_l3_pnl.shape[0]} does not "
                f"match dataset row count {len(rows)}. Rebuild the oracle against the current dataset."
            )
        expected_actions = int(bundle["action_labels"]["tradeable_mask"].shape[1])
        if simulated_l3_pnl.shape[1] != expected_actions:
            raise RuntimeError(
                f"simulated-L3 oracle action count {simulated_l3_pnl.shape[1]} does not "
                f"match dataset action count {expected_actions}. Rebuild the oracle against the current dataset."
            )
        oracle_meta: dict[str, Any] = {}
        if "meta_json" in z.files:
            oracle_meta = json.loads(str(z["meta_json"].item()))
        dataset_fingerprint = meta.get("dataset_fingerprint") or action_surface_dataset_fingerprint(rows, meta)
        oracle_fingerprint = str(oracle_meta.get("dataset_fingerprint", "")).strip()
        if oracle_fingerprint:
            if oracle_fingerprint != dataset_fingerprint:
                raise RuntimeError(
                    "simulated-L3 oracle fingerprint does not match the current dataset. "
                    "Rebuild the oracle against the exact action-surface bundle you are training on."
                )
        else:
            oracle_dataset_path = str(oracle_meta.get("dataset_path", ""))
            if oracle_dataset_path and os.path.normpath(oracle_dataset_path) != os.path.normpath(args.dataset):
                raise RuntimeError(
                    "Legacy simulated-L3 oracle was built from a different dataset path and has no "
                    "fingerprint for row-identity verification. Rebuild the oracle against the current dataset."
                )
            print(
                "Warning: simulated-L3 oracle has no dataset fingerprint; falling back to "
                "legacy path/shape compatibility only. Rebuild it for strict row-identity checks.",
                flush=True,
            )
        if args.utility_target == "hybrid_live" and simulated_l3_exit_bar is None:
            raise RuntimeError("hybrid_live target requires l3_exit_bar in the simulated-L3 oracle sidecar.")
        print(f"Loaded simulated-L3 oracle: shape={simulated_l3_pnl.shape} from {args.simulated_l3_oracle}", flush=True)

    windows = generate_rolling_windows(sorted(rows["day"].unique().tolist()))
    verify_windows(windows)
    if args.latest_only:
        windows = [windows[-1]]

    print_window_summary(windows)

    all_oos_frames: list[pd.DataFrame] = []
    chosen_trade_frames: list[pd.DataFrame] = []
    window_reports: list[dict[str, Any]] = []

    train_days_by_window = {w.window_idx: list(w.train_days) for w in windows}
    total_t0 = time.time()
    golden_day = str(args.golden_day).strip()
    golden_day_mask = rows["day"].astype(str).eq(golden_day).to_numpy() if golden_day else None

    for window in windows:
        wi = int(window.window_idx)
        print(
            f"Window {wi}: fit_train={len(window.train_days) - len(window.val_days)} "
            f"val={len(window.val_days)} oos={len(window.oos_days)}",
            flush=True,
        )
        fit_days = [d for d in window.train_days if d not in set(window.val_days)]
        fit_mask = rows["day"].isin(fit_days).to_numpy()
        val_mask = rows["day"].isin(window.val_days).to_numpy()
        oos_mask = rows["day"].isin(window.oos_days).to_numpy()

        fit = _slice_inputs(
            fit_mask,
            bundle,
            utility_blend=args.utility_blend,
            utility_target=args.utility_target,
            simulated_l3_pnl=simulated_l3_pnl,
            simulated_l3_exit_bar=simulated_l3_exit_bar,
        )
        val = _slice_inputs(
            val_mask,
            bundle,
            utility_blend=args.utility_blend,
            utility_target=args.utility_target,
            simulated_l3_pnl=simulated_l3_pnl,
            simulated_l3_exit_bar=simulated_l3_exit_bar,
        )
        oos = _slice_inputs(
            oos_mask,
            bundle,
            utility_blend=args.utility_blend,
            utility_target=args.utility_target,
            simulated_l3_pnl=simulated_l3_pnl,
            simulated_l3_exit_bar=simulated_l3_exit_bar,
        )
        golden_trace_rows: list[dict[str, Any]] = []
        golden_trace_subset = None
        if golden_day and golden_day in set(window.oos_days) and golden_day_mask is not None and golden_day_mask.any():
            golden_trace_subset = _slice_inputs(
                golden_day_mask,
                bundle,
                utility_blend=args.utility_blend,
                utility_target=args.utility_target,
                simulated_l3_pnl=simulated_l3_pnl,
                simulated_l3_exit_bar=simulated_l3_exit_bar,
            )

        def _trace_callback(epoch: int, epoch_parts: dict[str, float], pred: dict[str, np.ndarray]) -> None:
            if golden_trace_subset is None:
                return
            golden_trace_rows.extend(
                _golden_epoch_trace_rows(
                    golden_trace_subset,
                    pred,
                    epoch=epoch,
                    epoch_parts=epoch_parts,
                    top_k_contracts=int(meta["top_k_contracts_per_side"]),
                )
            )

        predictor, train_info = train_unified_action_model(
            scalar_train=fit["scalar"],
            seq_train=fit["seq"],
            seq_mask_train=fit["seq_mask"],
            contracts_train=fit["contracts"],
            contract_mask_train=fit["contract_mask"],
            utility_train=fit["utility"],
            utility_raw_train=fit["utility_raw"],
            dollar_train=fit["dollar"],
            return_train=fit["return_multiple"],
            win_train=fit["win"],
            risk_band_train=fit["risk_band"],
            clean_train=fit["clean"],
            stopout_train=fit["stopout"],
            available_mask_train=fit["available_mask"],
            tradeable_mask_train=fit["tradeable_mask"],
            scalar_val=val["scalar"],
            seq_val=val["seq"],
            seq_mask_val=val["seq_mask"],
            contracts_val=val["contracts"],
            contract_mask_val=val["contract_mask"],
            utility_val=val["utility"],
            utility_raw_val=val["utility_raw"],
            dollar_val=val["dollar"],
            return_val=val["return_multiple"],
            win_val=val["win"],
            risk_band_val=val["risk_band"],
            clean_val=val["clean"],
            stopout_val=val["stopout"],
            available_mask_val=val["available_mask"],
            tradeable_mask_val=val["tradeable_mask"],
            device=device,
            seed=seed + wi,
            hidden_dim=args.hidden_dim,
            seq_hidden_dim=args.seq_hidden_dim,
            contract_hidden_dim=args.contract_hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            w_regression=args.w_regression,
            w_ranking=args.w_ranking,
            w_side_contrastive=args.w_side_contrastive,
            w_dollar=args.w_dollar,
            w_return=args.w_return,
            w_win=args.w_win,
            w_clean=args.w_clean,
            w_stopout=args.w_stopout,
            side_balance_weight=args.side_balance_weight,
            trace_eval=golden_trace_subset,
            trace_callback=_trace_callback if golden_trace_subset is not None else None,
        )

        val_pred = _prediction_frame(
            val,
            predictor.predict(
                val["scalar"],
                val["seq"],
                val["seq_mask"],
                val["contracts"],
                val["contract_mask"],
            ),
            top_k_contracts=int(meta["top_k_contracts_per_side"]),
            contract_feature_names=list(meta.get("contract_feature_names", ())),
        )
        calibration = _calibrate_abstention_policy(
            val_pred,
            args.equity,
            pnl_col=OBJECTIVE_PNL_COL,
        )
        calibration["calibration_objective"] = objective_label
        val_chosen = _select_daily_trades(val_pred, abstention_policy=calibration).copy()
        calibration["objective_metrics"] = _trade_metrics(
            val_chosen,
            args.equity,
            total_days=len(window.val_days),
            pnl_col=OBJECTIVE_PNL_COL,
        )
        if not objective_matches_time_stop:
            calibration["time_stop_reference"] = _trade_metrics(
                val_chosen,
                args.equity,
                total_days=len(window.val_days),
                pnl_col=TIME_STOP_PNL_COL,
            )

        oos_pred = _prediction_frame(
            oos,
            predictor.predict(
                oos["scalar"],
                oos["seq"],
                oos["seq_mask"],
                oos["contracts"],
                oos["contract_mask"],
            ),
            top_k_contracts=int(meta["top_k_contracts_per_side"]),
            contract_feature_names=list(meta.get("contract_feature_names", ())),
        )
        oos_pred["window_idx"] = wi
        oos_pred["seed"] = seed
        chosen_trades = _select_daily_trades(oos_pred, abstention_policy=calibration).copy()
        chosen_trades["window_idx"] = wi
        chosen_trades["seed"] = seed

        window_dir = os.path.join(seed_dir, f"window_{wi:02d}")
        ensure_dir(window_dir)
        if golden_trace_rows:
            golden_out_dir = args.golden_day_out_dir or os.path.join(args.run_dir, "golden_day_training_trace")
            ensure_dir(golden_out_dir)
            pd.DataFrame(golden_trace_rows).to_csv(
                os.path.join(golden_out_dir, f"seed_{seed}_window_{wi:02d}_{golden_day}.csv"),
                index=False,
            )
        save_pickle(os.path.join(window_dir, "model.pkl"), predictor)
        save_json(os.path.join(window_dir, "training_info.json"), train_info)
        save_json(os.path.join(window_dir, "calibration.json"), calibration)
        save_pickle(os.path.join(window_dir, "oos_predictions.pkl"), oos_pred)
        save_pickle(os.path.join(window_dir, "chosen_trades.pkl"), chosen_trades)

        all_oos_frames.append(oos_pred)
        chosen_trade_frames.append(chosen_trades)
        window_reports.append(
            _window_report(
                wi,
                chosen_trades,
                args.equity,
                total_days=len(window.oos_days),
                decision_margin=float(calibration["decision_margin"]),
                objective_label=objective_label,
                objective_matches_time_stop=objective_matches_time_stop,
            )
        )

    oos_all = pd.concat(all_oos_frames, ignore_index=True).sort_values(["window_idx", "day", "bar_index"]).reset_index(drop=True)
    if chosen_trade_frames and any(len(df) > 0 for df in chosen_trade_frames):
        chosen_all = pd.concat(chosen_trade_frames, ignore_index=True).sort_values(["window_idx", "day", "bar_index"]).reset_index(drop=True)
    else:
        chosen_all = pd.DataFrame(columns=[
            "window_idx",
            "day",
            "bar_index",
            "chosen_action_id",
            "chosen_side",
            "chosen_strike",
            "decision_margin",
            "pred_clean_entry_prob",
            "pred_stopout_risk",
            OBJECTIVE_PNL_COL,
            "chosen_time_stop_pnl",
            "chosen_time_bucket",
            "beats_v1_same_bar",
        ])
    save_pickle(os.path.join(seed_dir, "oof_predictions.pkl"), oos_all)
    save_pickle(os.path.join(seed_dir, "chosen_trades.pkl"), chosen_all)

    report = _aggregate_report(
        chosen_all,
        window_reports,
        total_days=int(sum(len(w.oos_days) for w in windows)),
        equity=args.equity,
        seed=seed,
        tier=args.tier,
        objective_label=objective_label,
        objective_matches_time_stop=objective_matches_time_stop,
    )
    report["runtime_seconds"] = float(time.time() - total_t0)
    report["windows"] = [
        {
            "window_idx": int(w.window_idx),
            "window_id": w.window_id,
            "train_days": train_days_by_window[w.window_idx],
            "val_days": list(w.val_days),
            "oos_days": list(w.oos_days),
        }
        for w in windows
    ]
    save_json(os.path.join(seed_dir, "report.json"), report)
    return report


def main() -> int:
    args = parse_args()
    bundle = load_export_bundle(args.dataset)
    device = _resolve_device(args.device)
    _apply_tier_defaults(args, bundle["meta"], device)
    seeds = _resolve_seeds(args)
    ensure_dir(args.run_dir)

    reports: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"\n=== Unified policy seed {seed} ({args.tier}) ===", flush=True)
        reports.append(_run_single_seed(args, seed, device))

    save_json(
        os.path.join(args.run_dir, "manifest.json"),
        {
            "dataset": args.dataset,
            "tier": args.tier,
            "device": device,
            "seeds": seeds,
            "history_bars": int(bundle["meta"]["history_bars"]),
            "top_k_contracts_per_side": int(bundle["meta"]["top_k_contracts_per_side"]),
            "hyperparams": {
                "hidden_dim": int(args.hidden_dim),
                "seq_hidden_dim": int(args.seq_hidden_dim),
                "contract_hidden_dim": int(args.contract_hidden_dim),
                "depth": int(args.depth),
                "dropout": float(args.dropout),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "batch_size": int(args.batch_size),
                "max_epochs": int(args.max_epochs),
                "patience": int(args.patience),
                "w_regression": float(args.w_regression),
                "w_ranking": float(args.w_ranking),
                "w_side_contrastive": float(args.w_side_contrastive),
                "w_dollar": float(args.w_dollar),
                "w_return": float(args.w_return),
                "w_win": float(args.w_win),
                "w_clean": float(args.w_clean),
                "w_stopout": float(args.w_stopout),
                "utility_blend": float(args.utility_blend),
                "utility_target": str(args.utility_target),
                "simulated_l3_oracle": str(args.simulated_l3_oracle) if args.simulated_l3_oracle else None,
            },
            "aggregate_by_seed": reports,
        },
    )
    print(f"Saved unified-policy manifest: {os.path.join(args.run_dir, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
