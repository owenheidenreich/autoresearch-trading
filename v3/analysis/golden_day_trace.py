"""Single-day microscope for the SPX 0DTE unified action stack.

This diagnostic intentionally stays on the production data path: it rebuilds
one day through the same action-surface builder used by the full artifact,
then compares that one-day bundle against the saved artifact row-for-row.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from v3.config import GuardrailConfig
from v3.harness.rolling_windows import generate_rolling_windows
from v3.harness.v2_adapter import V2Dataset
from v3.layer2.action_surface_dataset import (
    DEFAULT_ACTION_SURFACE_DATASET_PATH,
    build_action_surface_bundle,
    validate_action_surface_bundle,
)
from v3.layer2.common import ensure_dir, load_export_bundle, load_pickle, save_json
from v3.layer2.train_unified_policy import (
    _golden_epoch_trace_rows,
    _prediction_frame,
    _select_daily_trades,
    _slice_inputs,
)
from v3.layer2.unified_policy import train_unified_action_model


DEFAULT_DAY = "2024-04-01"
DEFAULT_DATASET = "v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl"
DEFAULT_MODEL_DIR = "v3/artifacts/layer2_unified_policy_spx_live_hybrid_001_seed42/seed_42"
DEFAULT_ORACLE = "v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed42.npz"
DEFAULT_OUT_DIR = os.path.join("v3", "artifacts", "golden_day_trace", DEFAULT_DAY)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day", default=DEFAULT_DAY)
    p.add_argument("--dataset", default=DEFAULT_DATASET if os.path.exists(DEFAULT_DATASET) else DEFAULT_ACTION_SURFACE_DATASET_PATH)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--oracle", default=DEFAULT_ORACLE)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--equity", type=float, default=25_000.0)
    p.add_argument("--utility-target", default="hybrid_live", choices=("time_stop", "horizon", "simulated_l3", "hybrid_live"))
    p.add_argument("--utility-blend", type=float, default=0.0)
    p.add_argument("--run-overfit-test", action="store_true")
    p.add_argument("--overfit-epochs", type=int, default=120)
    p.add_argument("--overfit-lr", type=float, default=1e-3)
    p.add_argument("--overfit-min-action-match", type=float, default=0.70)
    p.add_argument("--overfit-min-strong-put-score-rate", type=float, default=0.70)
    p.add_argument("--overfit-min-loss-drop", type=float, default=0.25)
    p.add_argument("--strong-put-edge", type=float, default=250.0)
    return p.parse_args()


def _load_oracle(path: str, n_rows: int, n_actions: int) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if not path:
        return None, None, {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"oracle not found: {path}")
    z = np.load(path, allow_pickle=True)
    pnl = z["l3_exit_pnl"]
    exit_bar = z["l3_exit_bar"] if "l3_exit_bar" in z.files else None
    if pnl.shape != (n_rows, n_actions):
        raise RuntimeError(f"oracle shape {pnl.shape} does not match dataset {(n_rows, n_actions)}")
    meta: dict[str, Any] = {}
    if "meta_json" in z.files:
        meta = json.loads(str(z["meta_json"].item()))
    return pnl, exit_bar, meta


def _day_window(rows: pd.DataFrame, day: str) -> int:
    unique_days = sorted(rows["day"].astype(str).unique().tolist())
    for window in generate_rolling_windows(unique_days):
        if day in set(window.oos_days):
            return int(window.window_idx)
    return -1


def _rebuild_one_day(day: str, full_meta: dict[str, Any], equity: float) -> dict[str, Any]:
    ds = V2Dataset.load()
    cfg = GuardrailConfig()
    execution = full_meta.get("execution_window", {})
    return build_action_surface_bundle(
        ds,
        cfg,
        equity,
        days=[day],
        history_bars=int(full_meta["history_bars"]),
        top_k_contracts=int(full_meta["top_k_contracts_per_side"]),
        execution_start_bar=int(execution.get("start_bar", 15)),
        execution_end_bar=int(execution.get("end_bar", 120)),
        utility_horizon_bars=int(full_meta.get("utility_horizon_bars", 60)),
        contract_selection_mode=str(full_meta.get("contract_selection_mode", "risk_band")),
    )


def _compare_day_bundle(full_bundle: dict[str, Any], one_day_bundle: dict[str, Any], full_idx: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    full_rows = full_bundle["rows"].iloc[full_idx].reset_index(drop=True)
    one_rows = one_day_bundle["rows"].reset_index(drop=True)
    out["row_count_match"] = bool(len(full_rows) == len(one_rows))
    try:
        pd.testing.assert_frame_equal(
            full_rows.reset_index(drop=True),
            one_rows.loc[:, full_rows.columns].reset_index(drop=True),
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
        )
        out["rows_match"] = True
        out["rows_mismatch"] = ""
    except Exception as exc:
        out["rows_match"] = False
        out["rows_mismatch"] = str(exc)[:500]

    for key in ("sequence_features", "sequence_mask", "contract_features", "contract_mask", "contract_strike"):
        left = full_bundle[key][full_idx]
        right = one_day_bundle[key]
        out[f"{key}_match"] = bool(left.shape == right.shape and np.allclose(left, right, equal_nan=True))
        out[f"{key}_shape_full"] = list(left.shape)
        out[f"{key}_shape_one_day"] = list(right.shape)
    label_matches = {}
    for name, full_arr in full_bundle["action_labels"].items():
        left = full_arr[full_idx]
        right = one_day_bundle["action_labels"].get(name)
        label_matches[name] = bool(right is not None and left.shape == right.shape and np.allclose(left, right, equal_nan=True))
    out["label_matches"] = label_matches
    out["all_labels_match"] = bool(all(label_matches.values()))
    out["all_match"] = bool(
        out["row_count_match"]
        and out["rows_match"]
        and out["sequence_features_match"]
        and out["sequence_mask_match"]
        and out["contract_features_match"]
        and out["contract_mask_match"]
        and out["contract_strike_match"]
        and out["all_labels_match"]
    )
    return out


def _best_side_fields(
    values: np.ndarray,
    tradeable: np.ndarray,
    contract_strike: np.ndarray,
    *,
    top_k: int,
) -> dict[str, np.ndarray]:
    token_values = values[:, 1:].copy()
    token_values[~tradeable[:, 1:]] = -np.inf
    call_values = token_values[:, :top_k]
    put_values = token_values[:, top_k:]
    best_call_slot = np.argmax(call_values, axis=1)
    best_put_slot = np.argmax(put_values, axis=1)
    row_ids = np.arange(values.shape[0])
    best_call = call_values[row_ids, best_call_slot]
    best_put = put_values[row_ids, best_put_slot]
    best_call = np.where(np.isfinite(best_call), best_call, np.nan)
    best_put = np.where(np.isfinite(best_put), best_put, np.nan)
    best_call_action = best_call_slot + 1
    best_put_action = best_put_slot + 1 + top_k
    best_call_strike = contract_strike[row_ids, best_call_slot]
    best_put_strike = contract_strike[row_ids, best_put_slot + top_k]
    return {
        "best_call": best_call,
        "best_put": best_put,
        "best_call_action_id": best_call_action.astype(int),
        "best_put_action_id": best_put_action.astype(int),
        "best_call_strike": best_call_strike,
        "best_put_strike": best_put_strike,
    }


def _prediction_for_day(
    model_dir: str,
    window_idx: int,
    subset: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[dict[str, np.ndarray] | None, pd.DataFrame | None, dict[str, Any] | None]:
    if not model_dir or window_idx < 0:
        return None, None, None
    model_path = os.path.join(model_dir, f"window_{window_idx:02d}", "model.pkl")
    calibration_path = os.path.join(model_dir, f"window_{window_idx:02d}", "calibration.json")
    if not os.path.exists(model_path):
        return None, None, None
    predictor = load_pickle(model_path)
    pred = predictor.predict(
        subset["scalar"],
        subset["seq"],
        subset["seq_mask"],
        subset["contracts"],
        subset["contract_mask"],
    )
    pred_frame = _prediction_frame(
        subset,
        pred,
        top_k_contracts=int(meta["top_k_contracts_per_side"]),
        contract_feature_names=list(meta.get("contract_feature_names", ())),
    )
    calibration = {}
    if os.path.exists(calibration_path):
        with open(calibration_path) as f:
            calibration = json.load(f)
    return pred, pred_frame, calibration


def _make_summary_frame(
    subset: dict[str, Any],
    label_values: np.ndarray,
    pred: dict[str, np.ndarray] | None,
    pred_frame: pd.DataFrame | None,
    calibration: dict[str, Any] | None,
    *,
    top_k: int,
) -> pd.DataFrame:
    rows = subset["rows"].reset_index(drop=True).copy()
    tradeable = np.nan_to_num(subset["tradeable_mask"], nan=0.0) > 0.5
    label_best = _best_side_fields(label_values, tradeable, subset["contract_strike"], top_k=top_k)
    out = pd.DataFrame(
        {
            "day": rows["day"].astype(str),
            "bar_index": rows["bar_index"].astype(int),
            "underlying_close": rows["underlying_close"].astype(float),
            "vwap": rows["vwap"].astype(float),
            "first15_high": rows["first15_high"].astype(float),
            "first15_low": rows["first15_low"].astype(float),
            "sigma_pos": rows.get("sigma_pos", pd.Series(np.nan, index=rows.index)).astype(float),
            "omar_mid_pos_units": rows.get("omar_mid_pos_units", pd.Series(np.nan, index=rows.index)).astype(float),
            "label_best_call": label_best["best_call"],
            "label_best_put": label_best["best_put"],
            "label_put_minus_call": label_best["best_put"] - label_best["best_call"],
            "label_best_side": np.where(
                np.nan_to_num(label_best["best_put"], nan=-np.inf) > np.nan_to_num(label_best["best_call"], nan=-np.inf),
                "put",
                "call",
            ),
            "label_best_call_action_id": label_best["best_call_action_id"],
            "label_best_put_action_id": label_best["best_put_action_id"],
            "label_best_call_strike": label_best["best_call_strike"],
            "label_best_put_strike": label_best["best_put_strike"],
        }
    )
    if pred is not None:
        score_best = _best_side_fields(pred["utility"], tradeable, subset["contract_strike"], top_k=top_k)
        out["score_flat"] = pred["utility"][:, 0]
        out["score_best_call"] = score_best["best_call"]
        out["score_best_put"] = score_best["best_put"]
        out["score_put_minus_call"] = score_best["best_put"] - score_best["best_call"]
        out["score_best_side"] = np.where(
            np.nan_to_num(score_best["best_put"], nan=-np.inf) > np.nan_to_num(score_best["best_call"], nan=-np.inf),
            "put",
            "call",
        )
        out["score_label_side_match"] = out["score_best_side"] == out["label_best_side"]
    if pred_frame is not None:
        out["model_chosen_action_id"] = pred_frame["chosen_action_id"].astype(int)
        out["model_chosen_side"] = pred_frame["chosen_side"].astype(str)
        out["model_chosen_strike"] = pred_frame["chosen_strike"].astype(float)
        out["decision_margin"] = pred_frame["decision_margin"].astype(float)
        out["pred_win_prob"] = pred_frame["pred_win_prob"].astype(float)
        out["pred_stopout_risk"] = pred_frame["pred_stopout_risk"].astype(float)
        out["model_chosen_label"] = [
            float(label_values[i, int(action)]) if int(action) >= 0 and int(action) < label_values.shape[1] else np.nan
            for i, action in enumerate(pred_frame["chosen_action_id"].astype(int))
        ]
        policy = calibration or {}
        out["abstention_pass"] = (
            (out["model_chosen_action_id"] > 0)
            & np.isfinite(out["decision_margin"])
            & (out["decision_margin"] >= float(policy.get("decision_margin", 0.0)))
            & (out["pred_win_prob"].fillna(0.0) >= float(policy.get("min_win_prob", 0.0)))
            & (out["pred_stopout_risk"].fillna(1.0) <= float(policy.get("max_stopout_prob", 1.0)))
        )
        selected = _select_daily_trades(pred_frame, abstention_policy=policy).copy()
        out["selected_by_daily_policy"] = False
        if not selected.empty:
            selected_bar = int(selected.iloc[0]["bar_index"])
            out.loc[out["bar_index"].astype(int) == selected_bar, "selected_by_daily_policy"] = True
    return out


def _make_token_frame(
    subset: dict[str, Any],
    full_day_indices: np.ndarray,
    label_values: np.ndarray,
    pred: dict[str, np.ndarray] | None,
    l3_pnl: np.ndarray | None,
    *,
    top_k: int,
    contract_feature_names: list[str],
) -> pd.DataFrame:
    rows = subset["rows"].reset_index(drop=True)
    labels = subset
    records: list[dict[str, Any]] = []
    feature_idx = {name: i for i, name in enumerate(contract_feature_names)}
    for row_i, row in rows.iterrows():
        for token_i in range(top_k * 2):
            action_id = token_i + 1
            side = "call" if token_i < top_k else "put"
            rec: dict[str, Any] = {
                "day": str(row["day"]),
                "bar_index": int(row["bar_index"]),
                "source_row_index": int(full_day_indices[row_i]),
                "action_id": int(action_id),
                "side": side,
                "slot": int(token_i % top_k),
                "present": bool(subset["contract_mask"][row_i, token_i] > 0.5),
                "tradeable": bool(subset["tradeable_mask"][row_i, action_id] > 0.5),
                "strike": float(subset["contract_strike"][row_i, token_i]),
                "target_utility": float(label_values[row_i, action_id]),
                "time_stop_pnl": float(subset["time_stop_raw"][row_i, action_id]),
                "return_on_premium": float(subset["return_on_premium"][row_i, action_id]),
                "stopout_risk": float(subset["stopout"][row_i, action_id]),
            }
            for feature in ("premium", "abs_delta", "delta", "spread_fraction", "slot_role", "moneyness_bucket", "risk_band"):
                if feature in feature_idx:
                    rec[feature] = float(subset["contracts"][row_i, token_i, feature_idx[feature]])
            if l3_pnl is not None:
                rec["simulated_l3_pnl"] = float(l3_pnl[full_day_indices[row_i], action_id])
            if pred is not None:
                rec["model_score"] = float(pred["utility"][row_i, action_id])
                rec["pred_win_prob"] = float(pred["win_prob"][row_i, action_id])
                rec["pred_stopout_prob"] = float(pred["stopout_prob"][row_i, action_id])
            records.append(rec)
    return pd.DataFrame(records)


def _write_dashboard(
    out_path: str,
    *,
    day: str,
    summary: pd.DataFrame,
    token_detail: pd.DataFrame,
    report_md: str,
) -> None:
    selected = (
        summary.loc[summary["selected_by_daily_policy"].astype(bool)]
        if "selected_by_daily_policy" in summary
        else summary.iloc[0:0]
    )
    token_preview = token_detail.sort_values(["bar_index", "target_utility"], ascending=[True, False]).groupby("bar_index").head(4)
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #17202a; }
    h1, h2 { margin-bottom: 8px; }
    table { border-collapse: collapse; font-size: 12px; width: 100%; margin: 12px 0 24px; }
    th, td { border: 1px solid #d6dde6; padding: 4px 6px; text-align: right; }
    th { background: #eef3f8; position: sticky; top: 0; }
    td:first-child, th:first-child { text-align: left; }
    .call { color: #166534; font-weight: 600; }
    .put { color: #9f1239; font-weight: 600; }
    .note { background: #fff8db; border: 1px solid #f1d06a; padding: 10px 12px; margin: 12px 0; }
    pre { white-space: pre-wrap; background: #f7f8fa; padding: 12px; border: 1px solid #d6dde6; }
    """
    compact_cols = [
        "bar_index",
        "underlying_close",
        "vwap",
        "label_best_side",
        "label_best_call",
        "label_best_put",
        "label_put_minus_call",
        "score_best_side",
        "score_best_call",
        "score_best_put",
        "score_put_minus_call",
        "model_chosen_side",
        "model_chosen_strike",
        "abstention_pass",
        "selected_by_daily_policy",
    ]
    compact = summary[[c for c in compact_cols if c in summary.columns]].copy()
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Golden Day Trace {html.escape(day)}</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>Golden Day Trace: {html.escape(day)}</h1>",
        "<div class='note'>Observable completed-bar context and model scores are separated from future labels. "
        "Labels and PnL columns are future outcomes and must not be model inputs.</div>",
        "<h2>Selected Model Entry</h2>",
        selected.to_html(index=False, escape=False) if not selected.empty else "<p>No daily policy selection.</p>",
        "<h2>Per-Minute Call vs Put Trace</h2>",
        compact.to_html(index=False, escape=False),
        "<h2>Top Contract Tokens Per Minute By Future Utility</h2>",
        token_preview.head(360).to_html(index=False, escape=False),
        "<h2>Report Markdown</h2>",
        f"<pre>{html.escape(report_md)}</pre>",
        "</body></html>",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(html_parts))


def _side_counts(summary: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in summary:
        return {}
    return {str(k): int(v) for k, v in summary[column].value_counts(dropna=False).items()}


def _run_overfit(
    subset: dict[str, Any],
    meta: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    trace_rows: list[dict[str, Any]] = []

    def trace_callback(epoch: int, parts: dict[str, float], pred: dict[str, np.ndarray]) -> None:
        trace_rows.extend(
            _golden_epoch_trace_rows(
                subset,
                pred,
                epoch=epoch,
                epoch_parts=parts,
                top_k_contracts=int(meta["top_k_contracts_per_side"]),
            )
        )

    predictor, info = train_unified_action_model(
        scalar_train=subset["scalar"],
        seq_train=subset["seq"],
        seq_mask_train=subset["seq_mask"],
        contracts_train=subset["contracts"],
        contract_mask_train=subset["contract_mask"],
        utility_train=subset["utility"],
        utility_raw_train=subset["utility_raw"],
        dollar_train=subset["dollar"],
        return_train=subset["return_multiple"],
        win_train=subset["win"],
        risk_band_train=subset["risk_band"],
        clean_train=subset["clean"],
        stopout_train=subset["stopout"],
        available_mask_train=subset["available_mask"],
        tradeable_mask_train=subset["tradeable_mask"],
        scalar_val=subset["scalar"],
        seq_val=subset["seq"],
        seq_mask_val=subset["seq_mask"],
        contracts_val=subset["contracts"],
        contract_mask_val=subset["contract_mask"],
        utility_val=subset["utility"],
        utility_raw_val=subset["utility_raw"],
        dollar_val=subset["dollar"],
        return_val=subset["return_multiple"],
        win_val=subset["win"],
        risk_band_val=subset["risk_band"],
        clean_val=subset["clean"],
        stopout_val=subset["stopout"],
        available_mask_val=subset["available_mask"],
        tradeable_mask_val=subset["tradeable_mask"],
        device="cpu",
        seed=20240401,
        hidden_dim=96,
        seq_hidden_dim=48,
        contract_hidden_dim=48,
        depth=2,
        dropout=0.0,
        lr=float(args.overfit_lr),
        weight_decay=0.0,
        batch_size=256,
        max_epochs=int(args.overfit_epochs),
        patience=int(args.overfit_epochs),
        w_regression=0.5,
        w_ranking=1.0,
        w_side_contrastive=0.0,
        w_dollar=0.25,
        w_return=0.25,
        w_win=0.25,
        w_clean=0.35,
        w_stopout=0.35,
        trace_eval=subset,
        trace_callback=trace_callback,
    )
    pred = predictor.predict(
        subset["scalar"],
        subset["seq"],
        subset["seq_mask"],
        subset["contracts"],
        subset["contract_mask"],
    )
    tradeable = np.nan_to_num(subset["tradeable_mask"], nan=0.0) > 0.5
    pred_action = np.argmax(np.where(tradeable, pred["utility"], -np.inf), axis=1)
    true_action = np.argmax(np.where(tradeable, subset["utility_raw"], -np.inf), axis=1)
    top_k = int(meta["top_k_contracts_per_side"])
    label_best = _best_side_fields(subset["utility_raw"], tradeable, subset["contract_strike"], top_k=top_k)
    score_best = _best_side_fields(pred["utility"], tradeable, subset["contract_strike"], top_k=top_k)
    strong_put = (label_best["best_put"] - label_best["best_call"]) >= float(args.strong_put_edge)
    all_bad = np.nanmax(np.where(tradeable[:, 1:], subset["utility_raw"][:, 1:], -np.inf), axis=1) <= 0.0
    history = info.get("loss_history", [])
    first_loss = float(history[0]["val_loss"]) if history else float("nan")
    last_loss = float(history[-1]["val_loss"]) if history else float("nan")
    summary = {
        "passed": bool(
            float((pred_action == true_action).mean()) >= float(args.overfit_min_action_match)
            and (
                np.isfinite(first_loss)
                and np.isfinite(last_loss)
                and float(first_loss - last_loss) >= float(args.overfit_min_loss_drop)
            )
            and (
                not strong_put.any()
                or float((score_best["best_put"][strong_put] > score_best["best_call"][strong_put]).mean())
                >= float(args.overfit_min_strong_put_score_rate)
            )
        ),
        "action_match_rate": float((pred_action == true_action).mean()),
        "strong_put_bars": int(strong_put.sum()),
        "strong_put_score_put_above_call_rate": float(
            (score_best["best_put"][strong_put] > score_best["best_call"][strong_put]).mean()
        )
        if strong_put.any()
        else None,
        "all_bad_bars": int(all_bad.sum()),
        "flat_on_all_bad_rate": float((pred_action[all_bad] == 0).mean()) if all_bad.any() else None,
        "first_val_loss": first_loss,
        "last_val_loss": last_loss,
        "loss_drop": float(first_loss - last_loss) if np.isfinite(first_loss) and np.isfinite(last_loss) else None,
        "best_val_loss": float(info["best_val_loss"]),
        "best_epoch": int(info["best_epoch"]),
    }
    return summary, pd.DataFrame(trace_rows)


def _report_text(
    *,
    args: argparse.Namespace,
    compare: dict[str, Any],
    summary: pd.DataFrame,
    token_detail: pd.DataFrame,
    window_idx: int,
    calibration: dict[str, Any] | None,
    overfit_summary: dict[str, Any] | None,
) -> str:
    selected = (
        summary.loc[summary["selected_by_daily_policy"].astype(bool)]
        if "selected_by_daily_policy" in summary
        else summary.iloc[0:0]
    )
    lines = [
        f"# Golden Day Trace — {args.day}",
        "",
        "## Verdict",
        "",
        f"- One-day production rebuild matches full artifact: `{compare['all_match']}`.",
        f"- Rolling OOS window: `{window_idx}`.",
        f"- Future label best-side counts: `{_side_counts(summary, 'label_best_side')}`.",
        f"- Model score best-side counts: `{_side_counts(summary, 'score_best_side')}`.",
        f"- Daily selected trade rows: `{len(selected)}`.",
    ]
    if not selected.empty:
        row = selected.iloc[0]
        lines.extend(
            [
                (
                    f"- Selected bar `{int(row['bar_index'])}`: model chose "
                    f"`{row.get('model_chosen_side', '')}` strike `{row.get('model_chosen_strike', np.nan)}` "
                    f"with label `{float(row.get('model_chosen_label', np.nan)):.2f}`."
                ),
                (
                    f"- At that bar, best call label `{float(row['label_best_call']):.2f}`, "
                    f"best put label `{float(row['label_best_put']):.2f}`, "
                    f"put-call edge `{float(row['label_put_minus_call']):.2f}`."
                ),
            ]
        )
    if calibration:
        lines.append(f"- Calibration: `{json.dumps(calibration, sort_keys=True)[:800]}`.")
    if overfit_summary is not None:
        lines.extend(
            [
                "",
                "## One-Day Overfit Canary",
                "",
                f"- Passed: `{overfit_summary['passed']}`.",
                f"- Action match rate: `{overfit_summary['action_match_rate']:.3f}`.",
                f"- Strong-put bars: `{overfit_summary['strong_put_bars']}`.",
                f"- Strong-put score put-above-call rate: `{overfit_summary['strong_put_score_put_above_call_rate']}`.",
                f"- Loss drop: `{overfit_summary['loss_drop']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `day_summary.csv`: per-minute observable context, future labels, model scores, and abstention result.",
            "- `contract_tokens.csv`: all 24 action tokens per minute with metadata, labels, and scores.",
            "- `dashboard.html`: visual inspection dashboard.",
            "- `one_day_rebuild_compare.json`: production rebuild vs full artifact comparison.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)
    bundle = load_export_bundle(args.dataset)
    validate_action_surface_bundle(bundle)
    rows: pd.DataFrame = bundle["rows"].reset_index(drop=True)
    day_mask = rows["day"].astype(str).eq(str(args.day)).to_numpy()
    full_idx = np.flatnonzero(day_mask)
    if full_idx.size == 0:
        raise RuntimeError(f"day {args.day} not found in {args.dataset}")

    one_day_bundle = _rebuild_one_day(str(args.day), bundle["meta"], float(args.equity))
    compare = _compare_day_bundle(bundle, one_day_bundle, full_idx)
    save_json(os.path.join(args.out_dir, "one_day_rebuild_compare.json"), compare)
    with open(os.path.join(args.out_dir, "one_day_bundle.pkl"), "wb") as f:
        pickle.dump(one_day_bundle, f)

    n_actions = int(bundle["action_labels"]["tradeable_mask"].shape[1])
    l3_pnl, l3_exit_bar, oracle_meta = _load_oracle(args.oracle, len(rows), n_actions) if args.oracle else (None, None, {})
    subset = _slice_inputs(
        day_mask,
        bundle,
        utility_blend=float(args.utility_blend),
        utility_target=str(args.utility_target),
        simulated_l3_pnl=l3_pnl,
        simulated_l3_exit_bar=l3_exit_bar,
    )
    window_idx = _day_window(rows, str(args.day))
    pred, pred_frame, calibration = _prediction_for_day(args.model_dir, window_idx, subset, bundle["meta"])
    top_k = int(bundle["meta"]["top_k_contracts_per_side"])
    label_values = subset["utility_raw"]
    summary = _make_summary_frame(
        subset,
        label_values,
        pred,
        pred_frame,
        calibration,
        top_k=top_k,
    )
    token_detail = _make_token_frame(
        subset,
        full_idx,
        label_values,
        pred,
        l3_pnl,
        top_k=top_k,
        contract_feature_names=list(bundle["meta"].get("contract_feature_names", ())),
    )
    summary.to_csv(os.path.join(args.out_dir, "day_summary.csv"), index=False)
    token_detail.to_csv(os.path.join(args.out_dir, "contract_tokens.csv"), index=False)
    if pred_frame is not None:
        pred_frame.to_csv(os.path.join(args.out_dir, "model_prediction_frame.csv"), index=False)

    overfit_summary = None
    if args.run_overfit_test:
        overfit_summary, overfit_trace = _run_overfit(subset, bundle["meta"], args=args)
        save_json(os.path.join(args.out_dir, "overfit_summary.json"), overfit_summary)
        overfit_trace.to_csv(os.path.join(args.out_dir, "overfit_epoch_trace.csv"), index=False)

    payload = {
        "day": str(args.day),
        "dataset": str(args.dataset),
        "model_dir": str(args.model_dir),
        "oracle": str(args.oracle) if args.oracle else None,
        "oracle_meta": oracle_meta,
        "window_idx": int(window_idx),
        "rows": int(len(summary)),
        "label_best_side_counts": _side_counts(summary, "label_best_side"),
        "model_score_best_side_counts": _side_counts(summary, "score_best_side"),
        "selected_rows": int(summary["selected_by_daily_policy"].sum()) if "selected_by_daily_policy" in summary else 0,
        "one_day_rebuild_match": compare,
        "overfit_summary": overfit_summary,
    }
    save_json(os.path.join(args.out_dir, "summary.json"), payload)
    report_md = _report_text(
        args=args,
        compare=compare,
        summary=summary,
        token_detail=token_detail,
        window_idx=window_idx,
        calibration=calibration,
        overfit_summary=overfit_summary,
    )
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write(report_md)
    _write_dashboard(
        os.path.join(args.out_dir, "dashboard.html"),
        day=str(args.day),
        summary=summary,
        token_detail=token_detail,
        report_md=report_md,
    )
    print(f"Saved golden-day trace to {args.out_dir}")
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
