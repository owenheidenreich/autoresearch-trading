"""Mine v2/v3 historical data for reusable priors.

The output is intentionally framed as prior discovery, not proof of executable
edge. v2/v3 labels are built from Polygon minute bars and proxy bid/ask, while
v4 Databento CBBO labels are the execution-grade target.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


V2_DATA = Path("v2/data.pt")
V3_ACTION_SURFACE_LIVE = Path("v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v2-data", type=Path, default=V2_DATA)
    p.add_argument("--v3-action-surface", type=Path, default=V3_ACTION_SURFACE_LIVE)
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/historical_priors"))
    p.add_argument("--min-group-n", type=int, default=500)
    return p.parse_args()


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _time_bucket_from_bar(bar: int) -> str:
    if bar < 30:
        return "first_30"
    if bar < 120:
        return "post_open_morning"
    if bar < 240:
        return "midday"
    return "late_afternoon"


def _sign_bucket(values: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    out = np.full(values.shape, "flat", dtype=object)
    out[values > eps] = "positive"
    out[values < -eps] = "negative"
    out[~np.isfinite(values)] = "unknown"
    return out


def _metrics(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "total": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_loss = abs(float(losses.sum()))
    return {
        "n": int(arr.size),
        "total": float(arr.sum()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "win_rate": float((arr > 0).mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
    }


def _summarize_groups(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    min_group_n: int,
) -> list[dict]:
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(_metrics(group[value_col].astype(float)))
        yearly = []
        for year, year_group in group.groupby("year"):
            if len(year_group) >= max(25, min_group_n // 20):
                m = _metrics(year_group[value_col].astype(float))
                yearly.append({"year": str(year), **m})
        row["years_with_sample"] = len(yearly)
        row["years_positive_mean"] = sum(1 for item in yearly if item["mean"] > 0)
        row["years_pf_gt_1"] = sum(1 for item in yearly if item["profit_factor"] > 1.0)
        row["yearly"] = yearly
        if row["n"] >= min_group_n:
            rows.append(row)
    rows.sort(
        key=lambda x: (
            x["years_pf_gt_1"],
            x["years_positive_mean"],
            x["profit_factor"],
            x["mean"],
            x["n"],
        ),
        reverse=True,
    )
    return rows


def mine_v2_bar_priors(path: Path, *, min_group_n: int) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    dates = np.asarray([str(x) for x in data["dates"]], dtype=object)
    bar = _to_numpy(data["bar_of_day"]).astype(int)
    x = _to_numpy(data["X_sim"]).astype(float)
    feature_names = list(data["feature_names"])
    idx = {name: feature_names.index(name) for name in feature_names}
    valid = _to_numpy(data["slice_label_trade_valid"]).astype(bool)
    pnl = _to_numpy(data["slice_best_contract_pnl"]).astype(float)

    frame = pd.DataFrame(
        {
            "day": dates,
            "year": [str(d)[:4] for d in dates],
            "bar": bar,
            "time_bucket": [_time_bucket_from_bar(int(b)) for b in bar],
            "slice_best_contract_pnl": pnl,
            "valid": valid,
        }
    )
    frame["above_vwap"] = x[:, idx["vwap_dist"]] > 0
    frame["omar_sign"] = _sign_bucket(x[:, idx["omar_mid_pos_units"]], eps=0.0)
    frame["last10_range_bucket"] = pd.qcut(
        pd.Series(x[:, idx["last10_range_over_omar"]]).replace([np.inf, -np.inf], np.nan),
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    ).astype(str)
    frame["iv_percentile_bucket"] = pd.qcut(
        pd.Series(x[:, idx["iv_percentile"]]).replace([np.inf, -np.inf], np.nan),
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    ).astype(str)
    frame = frame[frame["valid"] & np.isfinite(frame["slice_best_contract_pnl"])]
    return {
        "source": str(path),
        "framing": (
            "v2 bar labels are broad opportunity labels from the exact-chain sidecars. "
            "They are useful for time/regime priors, not final executable PnL."
        ),
        "rows": int(len(frame)),
        "days": int(frame["day"].nunique()),
        "first_day": str(frame["day"].min()) if len(frame) else None,
        "last_day": str(frame["day"].max()) if len(frame) else None,
        "time_bucket": _summarize_groups(
            frame,
            group_cols=["time_bucket"],
            value_col="slice_best_contract_pnl",
            min_group_n=min_group_n,
        ),
        "time_above_vwap": _summarize_groups(
            frame,
            group_cols=["time_bucket", "above_vwap"],
            value_col="slice_best_contract_pnl",
            min_group_n=min_group_n,
        )[:20],
        "time_omar": _summarize_groups(
            frame,
            group_cols=["time_bucket", "omar_sign"],
            value_col="slice_best_contract_pnl",
            min_group_n=min_group_n,
        )[:20],
        "time_iv_range": _summarize_groups(
            frame,
            group_cols=["time_bucket", "iv_percentile_bucket", "last10_range_bucket"],
            value_col="slice_best_contract_pnl",
            min_group_n=min_group_n,
        )[:20],
    }


def _action_side_indices(bundle: dict) -> dict[str, list[int]]:
    out = {"C": [], "P": []}
    for item in bundle["meta"]["token_schema"]:
        action_id = int(item["action_id"])
        side = "C" if item["side"] == "call" else "P"
        out[side].append(action_id)
    return out


def _best_side_frame(bundle: dict) -> pd.DataFrame:
    rows = bundle["rows"].copy()
    side_indices = _action_side_indices(bundle)
    labels = np.asarray(bundle["action_labels"]["hybrid_live_utility"], dtype=float)
    horizon = np.asarray(bundle["action_labels"]["horizon_pnl"], dtype=float)
    tradeable = np.asarray(bundle["action_labels"]["tradeable_mask"], dtype=bool)

    records = []
    for side, action_ids in side_indices.items():
        idx = np.asarray(action_ids, dtype=int)
        side_labels = labels[:, idx]
        side_horizon = horizon[:, idx]
        side_mask = tradeable[:, idx] & np.isfinite(side_labels)
        best = np.full(len(rows), np.nan, dtype=float)
        best_horizon = np.full(len(rows), np.nan, dtype=float)
        any_side = side_mask.any(axis=1)
        if any_side.any():
            masked = np.where(side_mask, side_labels, -np.inf)
            best_idx = np.argmax(masked, axis=1)
            row_ix = np.arange(len(rows))
            best[any_side] = side_labels[row_ix[any_side], best_idx[any_side]]
            best_horizon[any_side] = side_horizon[row_ix[any_side], best_idx[any_side]]
        records.append(
            pd.DataFrame(
                {
                    "day": rows["day"].astype(str),
                    "year": rows["day"].astype(str).str[:4],
                    "bar": rows["bar_index"].astype(int),
                    "side": side,
                    "time_bucket": [_time_bucket_from_bar(int(b)) for b in rows["bar_index"]],
                    "above_vwap": rows["underlying_close"].astype(float) > rows["vwap"].astype(float),
                    "omar_sign": _sign_bucket(rows["omar_mid_pos_units"].astype(float).to_numpy(), eps=0.0),
                    "best_hybrid_live_utility": best,
                    "best_horizon_pnl": best_horizon,
                    "has_tradeable_side": any_side,
                }
            )
        )
    frame = pd.concat(records, ignore_index=True)
    return frame[frame["has_tradeable_side"] & np.isfinite(frame["best_hybrid_live_utility"])]


def mine_v3_action_surface_priors(path: Path, *, min_group_n: int) -> dict:
    with path.open("rb") as f:
        bundle = pickle.load(f)
    frame = _best_side_frame(bundle)
    days = sorted(frame["day"].unique().tolist())
    return {
        "source": str(path),
        "framing": (
            "v3 action-surface priors use the best tradeable candidate per side "
            "inside the morning execution window. Labels use proxy bid/ask and "
            "should be validated against v4 CBBO before becoming policy rules."
        ),
        "rows": int(len(frame)),
        "days": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "time_side": _summarize_groups(
            frame,
            group_cols=["time_bucket", "side"],
            value_col="best_hybrid_live_utility",
            min_group_n=min_group_n,
        ),
        "time_side_above_vwap": _summarize_groups(
            frame,
            group_cols=["time_bucket", "side", "above_vwap"],
            value_col="best_hybrid_live_utility",
            min_group_n=min_group_n,
        )[:20],
        "time_side_omar": _summarize_groups(
            frame,
            group_cols=["time_bucket", "side", "omar_sign"],
            value_col="best_hybrid_live_utility",
            min_group_n=min_group_n,
        )[:20],
    }


def _compact_rows(rows: list[dict], *, limit: int = 12) -> list[dict]:
    keep = []
    for row in rows[:limit]:
        item = {k: v for k, v in row.items() if k != "yearly"}
        keep.append(item)
    return keep


def _fmt(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return f"{value:.3f}"


def _row_label(row: dict, cols: list[str]) -> str:
    return " | ".join(str(row[col]) for col in cols)


def _write_group_table(lines: list[str], title: str, rows: list[dict], cols: list[str]) -> None:
    lines += [
        "",
        f"## {title}",
        "",
        "| Group | N | Mean | Median | Win Rate | PF | Years PF>1 / Years |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:12]:
        years = row.get("years_with_sample", 0)
        lines.append(
            f"| {_row_label(row, cols)} | {row['n']} | {_fmt(row['mean'])} | "
            f"{_fmt(row['median'])} | {_fmt(row['win_rate'])} | {_fmt(row['profit_factor'])} | "
            f"{row.get('years_pf_gt_1', 0)} / {years} |"
        )


def write_report(path: Path, payload: dict) -> None:
    v2 = payload["v2_bar_priors"]
    v3 = payload["v3_action_surface_priors"]
    lines = [
        "# Historical Priors From v2/v3",
        "",
        "These are prior-discovery diagnostics, not executable edge claims.",
        "",
        "## Read This First",
        "",
        "- v2/v3 covers many more environments than v4: 2022-04-11 through 2026-04-24.",
        "- v2/v3 bid/ask is proxy-derived from Polygon minute OHLC; v4 Databento CBBO remains the execution-grade validator.",
        "- v3 action-surface coverage is morning-focused, so it cannot by itself validate late-afternoon rules.",
        "",
        "## v2 Bar Opportunity Coverage",
        "",
        f"- Rows with valid slice labels: {v2['rows']:,}.",
        f"- Days: {v2['days']} ({v2['first_day']} to {v2['last_day']}).",
        f"- Framing: {v2['framing']}",
    ]
    _write_group_table(lines, "v2 Time Buckets", v2["time_bucket"], ["time_bucket"])
    _write_group_table(lines, "v2 Time x VWAP", v2["time_above_vwap"], ["time_bucket", "above_vwap"])
    _write_group_table(lines, "v2 Time x OMAR", v2["time_omar"], ["time_bucket", "omar_sign"])

    lines += [
        "",
        "## v3 Action-Surface Coverage",
        "",
        f"- Rows after side expansion: {v3['rows']:,}.",
        f"- Days: {v3['days']} ({v3['first_day']} to {v3['last_day']}).",
        f"- Framing: {v3['framing']}",
    ]
    _write_group_table(lines, "v3 Time x Side", v3["time_side"], ["time_bucket", "side"])
    _write_group_table(
        lines,
        "v3 Time x Side x VWAP",
        v3["time_side_above_vwap"],
        ["time_bucket", "side", "above_vwap"],
    )
    _write_group_table(
        lines,
        "v3 Time x Side x OMAR",
        v3["time_side_omar"],
        ["time_bucket", "side", "omar_sign"],
    )

    lines += [
        "",
        "## Translation To v4",
        "",
        "The recurring priors to carry forward are not hard rules yet. They become candidate causal features and pre-registered evaluation slices in v4:",
        "",
        "- time bucket, especially post-open morning; late afternoon must be validated primarily in v4 because v3 action-surface rows are morning-focused.",
        "- above/below VWAP and OMAR sign as state variables, not as hand-forced directions.",
        "- side-aware opportunity: calls and puts should remain separate actions rather than one generic long-option class.",
        "- volatility/range buckets as diagnostics for whether a policy is regime brittle.",
        "",
        "Next validation step: train a v4 action model with these priors represented explicitly, then require the March CBBO holdout to improve across seeds before buying more data.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    payload = {
        "v2_bar_priors": mine_v2_bar_priors(args.v2_data, min_group_n=args.min_group_n),
        "v3_action_surface_priors": mine_v3_action_surface_priors(
            args.v3_action_surface,
            min_group_n=args.min_group_n,
        ),
    }
    payload["compact"] = {
        "v2_time_bucket": _compact_rows(payload["v2_bar_priors"]["time_bucket"]),
        "v2_time_above_vwap": _compact_rows(payload["v2_bar_priors"]["time_above_vwap"]),
        "v3_time_side": _compact_rows(payload["v3_action_surface_priors"]["time_side"]),
        "v3_time_side_above_vwap": _compact_rows(
            payload["v3_action_surface_priors"]["time_side_above_vwap"]
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")
    md_path = args.out_dir / "report.md"
    write_report(md_path, payload)
    print(json_path)
    print(md_path)
    print(json.dumps(payload["compact"], indent=2, allow_nan=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
