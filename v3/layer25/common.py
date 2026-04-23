from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from v3.config import GuardrailConfig
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.common import (
    build_labeled_day,
    effective_direction,
    load_json,
    load_pickle,
    per_day_choice,
    replay_metrics_from_pnls,
)


DEFAULT_ROLLING_DIR = os.path.join("v3", "artifacts", "rolling_l2")
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "layer25_entry_patience_surface")
DEFAULT_HORIZON_BARS = 10
DEFAULT_MAE_FLOOR_PCT = -20.0
DEFAULT_THRESHOLDS = (0.40, 0.50, 0.60)
DEFAULT_MIN_TRAIN_ROWS = 5000
DEFAULT_EQUITY = 25_000.0


def feature_names(manifest: dict[str, Any]) -> list[str]:
    if "augmented_feature_names" in manifest:
        return list(manifest["augmented_feature_names"])
    if "feature_names" in manifest:
        return list(manifest["feature_names"])
    raise KeyError("rolling manifest missing augmented_feature_names")


def selected_excursion(bar: Any, direction: str, horizon_bars: int, kind: str) -> float | None:
    attr = f"{kind}_{horizon_bars}min_call" if direction == "call" else f"{kind}_{horizon_bars}min_put"
    return getattr(bar.labels, attr)


def timing_bucket(time_stop_pnl: float | None, mae_pct: float | None, mae_floor_pct: float) -> str:
    if time_stop_pnl is None or mae_pct is None:
        return "missing"
    if time_stop_pnl > 0 and mae_pct > mae_floor_pct:
        return "clean_winner"
    if time_stop_pnl > 0:
        return "shakeout_winner"
    if mae_pct <= mae_floor_pct:
        return "fast_loser"
    return "drift_loser"


def surface_candidate_mask(df: pd.DataFrame) -> pd.Series:
    direction_ok = df["effective_direction"].isin(["call", "put"])
    pnl_ok = df["selected_time_stop_value"].notna()
    score_ok = df["entry_score"].notna() & df["side_conf"].notna()
    contract_ok = df["predicted_direction"] != ""
    return direction_ok & pnl_ok & score_ok & contract_ok


def build_surface_frame(
    rolling_dir: str,
    equity: float,
    horizon_bars: int,
    mae_floor_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = load_json(os.path.join(rolling_dir, "manifest.json"))
    safe_feature_names = feature_names(manifest)
    model_feature_names = safe_feature_names + ["entry_score", "side_conf", "direction_is_call"]
    thresholds_by_window = {
        int(w["window_idx"]): dict(w["thresholds"]) for w in manifest["windows"]
    }

    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    day_cache: dict[str, tuple[Any, Any]] = {}
    rows: list[dict[str, Any]] = []

    for window in manifest["windows"]:
        wi = int(window["window_idx"])
        oos_pred = load_pickle(os.path.join(rolling_dir, f"window_{wi:02d}", "oos_predictions.pkl"))
        oos_pred = oos_pred.copy()
        if "effective_direction" not in oos_pred.columns:
            oos_pred["effective_direction"] = oos_pred.apply(
                effective_direction,
                axis=1,
                direction_mode=manifest["direction_mode"],
                policy_mode="scalar_side",
            )
        if "selected_time_stop_value" not in oos_pred.columns:
            raise RuntimeError(
                f"oos_predictions for window {wi} missing selected_time_stop_value; "
                "rerun rolling_l2_retrain before running layer25"
            )

        oos_pred = oos_pred[surface_candidate_mask(oos_pred)].copy()
        for day, day_rows in oos_pred.groupby("day"):
            day = str(day)
            if day not in day_cache:
                day_cache[day] = build_labeled_day(ds, day, cfg, equity=equity)
            log, sidecar = day_cache[day]
            if log is None or sidecar is None:
                continue

            bar_lookup = {int(bar.bar_index): bar for bar in log.bars}
            for row in day_rows.itertuples(index=False):
                bar = bar_lookup.get(int(row.bar_index))
                if bar is None:
                    continue
                direction = str(row.effective_direction)
                mae_selected = selected_excursion(bar, direction, horizon_bars, "mae")
                mfe_selected = selected_excursion(bar, direction, horizon_bars, "mfe")
                time_stop_value = float(row.selected_time_stop_value)

                payload = {
                    name: getattr(row, name)
                    for name in model_feature_names
                    if hasattr(row, name)
                }
                payload["window_idx"] = wi
                payload["day"] = day
                payload["bar_index"] = int(row.bar_index)
                payload["predicted_direction"] = str(row.predicted_direction)
                payload["effective_direction"] = direction
                payload["direction_is_call"] = 1.0 if direction == "call" else 0.0
                payload["selected_time_stop_value"] = time_stop_value
                payload["mae_selected"] = mae_selected
                payload["mfe_selected"] = mfe_selected
                payload["timing_bucket"] = timing_bucket(time_stop_value, mae_selected, mae_floor_pct)
                payload["clean_entry_label"] = int(
                    time_stop_value > 0
                    and mae_selected is not None
                    and mae_selected > mae_floor_pct
                )
                rows.append(payload)

    df = pd.DataFrame(rows).sort_values(["window_idx", "day", "bar_index"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("layer25 surface builder found zero rows")

    return df, {
        "manifest": manifest,
        "model_feature_names": model_feature_names,
        "thresholds_by_window": thresholds_by_window,
    }


def walkforward_probs(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_rows: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pred_prob = np.full(len(df), np.nan, dtype=np.float64)
    reports: list[dict[str, Any]] = []

    for wi in sorted(df["window_idx"].unique()):
        train_df = df[df["window_idx"] < wi]
        test_df = df[df["window_idx"] == wi]
        prior_mean = float(train_df["clean_entry_label"].mean()) if len(train_df) > 0 else 0.5

        if len(train_df) < min_train_rows or train_df["clean_entry_label"].nunique() < 2:
            probs = np.full(len(test_df), prior_mean, dtype=np.float64)
            mode = "prior_mean"
            auc = None
        else:
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_depth=4,
                min_samples_leaf=100,
                random_state=42,
            )
            X_train = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
            y_train = train_df["clean_entry_label"].to_numpy(dtype=np.int64)
            X_test = test_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            auc = None
            if test_df["clean_entry_label"].nunique() > 1:
                auc = float(roc_auc_score(test_df["clean_entry_label"].to_numpy(dtype=np.int64), probs))
            mode = "model"

        pred_prob[test_df.index.to_numpy()] = probs
        reports.append({
            "window_idx": int(wi),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_clean_rate": prior_mean,
            "test_clean_rate": float(test_df["clean_entry_label"].mean()),
            "mode": mode,
            "auc": auc,
        })

    return pred_prob, reports


def policy_trades(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    thresholds_by_window: dict[int, dict[str, float]],
    patience_threshold: float | None,
) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []

    for wi, win_df in df.groupby("window_idx"):
        th = thresholds_by_window[int(wi)]
        for day, day_rows in win_df.groupby("day"):
            candidates = day_rows
            if patience_threshold is not None:
                candidates = candidates[candidates["clean_entry_prob"] >= patience_threshold]
                if candidates.empty:
                    continue
            chosen = per_day_choice(
                candidates,
                th["entry_threshold"],
                th["side_threshold"],
                score_mode=manifest["score_mode"],
                side_score_weight=float(manifest["side_score_weight"]),
                policy_mode="scalar_side",
                direction_mode=manifest["direction_mode"],
            )
            if chosen is None:
                continue
            trades.append({
                "window_idx": int(wi),
                "day": str(day),
                "bar_index": int(chosen["bar_index"]),
                "effective_direction": str(chosen["effective_direction"]),
                "pnl": float(chosen["selected_time_stop_value"]),
                "clean_entry_prob": float(chosen.get("clean_entry_prob", np.nan)),
            })

    if not trades:
        return pd.DataFrame(columns=["window_idx", "day", "bar_index", "effective_direction", "pnl", "clean_entry_prob"])
    return pd.DataFrame(trades).sort_values(["window_idx", "day", "bar_index"]).reset_index(drop=True)


def trade_metrics(trades: pd.DataFrame, equity: float, total_days: int) -> dict[str, Any]:
    metrics = replay_metrics_from_pnls(trades["pnl"].tolist() if not trades.empty else [], equity)
    return {
        "trades": int(len(trades)),
        "trade_share": float(len(trades) / max(total_days, 1)),
        "pf": float(metrics["pf"]),
        "max_dd_pct": float(metrics["max_dd_pct"]),
        "mean_pnl": float(metrics["mean_pnl"]),
    }


def threshold_results(
    surface_df: pd.DataFrame,
    manifest: dict[str, Any],
    thresholds_by_window: dict[int, dict[str, float]],
    thresholds: list[float],
    equity: float,
) -> list[dict[str, Any]]:
    total_days = int(surface_df["day"].nunique())
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        gated_trades = policy_trades(
            surface_df,
            manifest=manifest,
            thresholds_by_window=thresholds_by_window,
            patience_threshold=float(threshold),
        )
        results.append({
            "threshold": float(threshold),
            **trade_metrics(gated_trades, equity, total_days),
        })
    return results


def build_report_payload(
    surface_df: pd.DataFrame,
    meta: dict[str, Any],
    walkforward_reports: list[dict[str, Any]],
    threshold_eval: list[dict[str, Any]],
    equity: float,
) -> dict[str, Any]:
    manifest = meta["manifest"]
    thresholds_by_window = meta["thresholds_by_window"]
    total_days = int(surface_df["day"].nunique())
    aucs = [r["auc"] for r in walkforward_reports if r["auc"] is not None]
    baseline_trades = policy_trades(
        surface_df,
        manifest=manifest,
        thresholds_by_window=thresholds_by_window,
        patience_threshold=None,
    )
    recommended = max(threshold_eval, key=lambda r: (r["pf"], r["trades"]))

    return {
        "surface": {
            "rows": int(len(surface_df)),
            "days": total_days,
            "clean_entry_rate": float(surface_df["clean_entry_label"].mean()),
            "timing_bucket_counts": {
                k: int(v) for k, v in surface_df["timing_bucket"].value_counts().sort_index().items()
            },
        },
        "walkforward_model": {
            "feature_count": int(len(meta["model_feature_names"])),
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "per_window": walkforward_reports,
        },
        "policy_meta": {
            "rolling_dir": meta.get("rolling_dir"),
            "direction_mode": manifest["direction_mode"],
            "score_mode": manifest["score_mode"],
            "side_score_weight": float(manifest["side_score_weight"]),
            "thresholds_by_window": {
                str(k): v for k, v in thresholds_by_window.items()
            },
        },
        "baseline_policy": trade_metrics(baseline_trades, equity, total_days),
        "threshold_results": threshold_eval,
        "recommended_threshold_by_pf": recommended,
    }

