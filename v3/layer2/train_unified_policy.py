from __future__ import annotations

import argparse
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
from v3.layer2.action_surface_dataset import DEFAULT_ACTION_SURFACE_DATASET_PATH
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
) -> dict[str, Any]:
    rows: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
    action_labels = bundle["action_labels"]
    time_stop_raw = action_labels["utility_raw"][mask]
    if utility_blend > 0.0:
        alpha = float(utility_blend)
        best_exit = action_labels["best_exit_pnl"][mask]
        nan_mask = ~np.isfinite(time_stop_raw) | ~np.isfinite(best_exit)
        blended = (1.0 - alpha) * np.nan_to_num(time_stop_raw, nan=0.0) + alpha * np.nan_to_num(best_exit, nan=0.0)
        blended[nan_mask] = np.nan
        utility_raw = blended.astype(np.float32)
        utility_arcsinh = np.arcsinh(utility_raw / 100.0).astype(np.float32)
    else:
        utility_raw = time_stop_raw
        utility_arcsinh = action_labels["utility_arcsinh"][mask]
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
        "time_stop_raw": time_stop_raw,
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
) -> pd.DataFrame:
    rows = subset["rows"].copy()
    utility = pred["utility"]
    clean_prob = pred["clean_prob"]
    stopout_prob = pred["stopout_prob"]
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

    chosen_time_stop = np.zeros(len(rows), dtype=np.float32)
    chosen_clean_prob = np.zeros(len(rows), dtype=np.float32)
    chosen_stopout_prob = np.zeros(len(rows), dtype=np.float32)
    chosen_clean_label = np.zeros(len(rows), dtype=np.float32)
    chosen_stopout_label = np.zeros(len(rows), dtype=np.float32)
    chosen_mae_10 = np.full(len(rows), np.nan, dtype=np.float32)

    chosen_rows = np.arange(len(rows))[has_trade]
    chosen_cols = chosen_action_id[has_trade]
    if len(chosen_rows) > 0:
        chosen_time_stop[chosen_rows] = subset["time_stop_raw"][chosen_rows, chosen_cols]
        chosen_clean_prob[chosen_rows] = clean_prob[chosen_rows, chosen_cols]
        chosen_stopout_prob[chosen_rows] = stopout_prob[chosen_rows, chosen_cols]
        chosen_clean_label[chosen_rows] = subset["clean"][chosen_rows, chosen_cols]
        chosen_stopout_label[chosen_rows] = subset["stopout"][chosen_rows, chosen_cols]
        chosen_mae_10[chosen_rows] = subset["mae_10"][chosen_rows, chosen_cols]

    rows["flat_score"] = utility[:, 0]
    rows["best_nonflat_score"] = best_token_score
    rows["decision_margin"] = rows["best_nonflat_score"] - rows["flat_score"]
    rows["chosen_action_id"] = chosen_action_id
    rows["chosen_side"] = chosen_side
    rows["chosen_strike"] = chosen_strike
    rows["chosen_time_stop_pnl"] = chosen_time_stop
    rows["pred_clean_entry_prob"] = chosen_clean_prob
    rows["pred_stopout_risk"] = chosen_stopout_prob
    rows["chosen_clean_entry_label"] = chosen_clean_label
    rows["chosen_stopout_label"] = chosen_stopout_label
    rows["chosen_mae_10"] = chosen_mae_10
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


def _select_daily_trades(pred_df: pd.DataFrame, decision_margin: float) -> pd.DataFrame:
    eligible = pred_df[
        (pred_df["chosen_action_id"] > 0)
        & np.isfinite(pred_df["best_nonflat_score"])
        & (pred_df["decision_margin"] >= decision_margin)
    ].copy()
    if eligible.empty:
        return eligible
    idx = eligible.groupby("day")["best_nonflat_score"].idxmax()
    return eligible.loc[idx].sort_values(["day", "bar_index"]).reset_index(drop=True)


def _stress_metrics(trades: pd.DataFrame, equity: float, slippage: float) -> dict[str, float]:
    stressed = (trades["chosen_time_stop_pnl"] - float(slippage)).tolist() if not trades.empty else []
    metrics = replay_metrics_from_pnls(stressed, equity)
    return {
        "slippage_round_trip": float(slippage),
        "pf": float(metrics["pf"]) if stressed else 0.0,
        "max_dd_pct": float(metrics["max_dd_pct"]),
        "mean_pnl": float(metrics["mean_pnl"]),
        "trades": int(len(stressed)),
    }


def _trade_metrics(trades: pd.DataFrame, equity: float, total_days: int) -> dict[str, Any]:
    metrics = replay_metrics_from_pnls(trades["chosen_time_stop_pnl"].tolist() if not trades.empty else [], equity)
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


def _calibrate_decision_margin(val_pred: pd.DataFrame, equity: float) -> dict[str, Any]:
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
        metrics = _trade_metrics(trades, equity, total_days)
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


def _window_report(window_idx: int, trades: pd.DataFrame, equity: float, total_days: int, decision_margin: float) -> dict[str, Any]:
    metrics = _trade_metrics(trades, equity, total_days)
    return {
        "window_idx": int(window_idx),
        "decision_margin": float(decision_margin),
        **metrics,
    }


def _aggregate_report(
    chosen_trades: pd.DataFrame,
    per_window: list[dict[str, Any]],
    total_days: int,
    equity: float,
    seed: int,
    tier: str,
) -> dict[str, Any]:
    metrics = _trade_metrics(chosen_trades, equity, total_days)
    window_pfs = np.asarray([w["pf"] for w in per_window], dtype=np.float64)
    slippage = {
        str(int(slip)): _stress_metrics(chosen_trades, equity, float(slip))
        for slip in DEFAULT_SIZZLE_SLIPPAGE
    }
    return {
        "seed": int(seed),
        "tier": tier,
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
            "w1_gate_pf_vs_baseline": float(metrics["pf"]) > DEFAULT_BASELINE_PF,
            "w1_gate_dd_vs_layer25": (
                float(metrics["max_dd_pct"]) <= DEFAULT_BASELINE_LAYER25_DD
                or float(metrics["pf"]) >= DEFAULT_BASELINE_PF + 0.15
            ),
            "w1_gate_trade_share": 0.25 <= float(metrics["trade_share"]) <= 0.70,
            "patience_gate_fast_loser_improvement_vs_old_baseline": float(metrics["fast_loser_rate"]) <= (113.0 / 432.0) * 0.80,
            "exit_gate_placeholder": False,
        },
    }


def _run_single_seed(args: argparse.Namespace, seed: int, device: str) -> dict[str, Any]:
    ensure_dir(args.run_dir)
    seed_dir = os.path.join(args.run_dir, f"seed_{seed}")
    ensure_dir(seed_dir)

    bundle = load_export_bundle(args.dataset)
    rows: pd.DataFrame = bundle["rows"]
    meta = bundle["meta"]
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

        fit = _slice_inputs(fit_mask, bundle, utility_blend=args.utility_blend)
        val = _slice_inputs(val_mask, bundle, utility_blend=args.utility_blend)
        oos = _slice_inputs(oos_mask, bundle, utility_blend=args.utility_blend)

        predictor, train_info = train_unified_action_model(
            scalar_train=fit["scalar"],
            seq_train=fit["seq"],
            seq_mask_train=fit["seq_mask"],
            contracts_train=fit["contracts"],
            contract_mask_train=fit["contract_mask"],
            utility_train=fit["utility"],
            utility_raw_train=fit["utility_raw"],
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
            w_clean=args.w_clean,
            w_stopout=args.w_stopout,
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
        )
        calibration = _calibrate_decision_margin(val_pred, args.equity)

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
        )
        oos_pred["window_idx"] = wi
        oos_pred["seed"] = seed
        chosen_trades = _select_daily_trades(oos_pred, calibration["decision_margin"]).copy()
        chosen_trades["window_idx"] = wi
        chosen_trades["seed"] = seed

        window_dir = os.path.join(seed_dir, f"window_{wi:02d}")
        ensure_dir(window_dir)
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
                "w_clean": float(args.w_clean),
                "w_stopout": float(args.w_stopout),
                "utility_blend": float(args.utility_blend),
            },
            "aggregate_by_seed": reports,
        },
    )
    print(f"Saved unified-policy manifest: {os.path.join(args.run_dir, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
