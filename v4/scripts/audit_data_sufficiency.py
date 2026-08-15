"""Audit whether v3/v4 data is sufficient for the next modeling step.

The audit separates high-quality executable quote data from broader historical
context. v3/v2 artifacts are valuable, but their option bid/ask fields are
derived from Polygon minute OHLC, not observed NBBO quotes. v4 Databento OPRA
CBBO remains the executable-label source of truth.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from v4.model.environment_diagnostics import time_bucket
from v4.model.supervised_pilot import split_name


V4_RAW_DIRS = {
    "definition": Path("data/raw/databento/opra_spxw_definition"),
    "cbbo_1m": Path("data/raw/databento/opra_spxw_cbbo_1m"),
    "ohlcv_1m": Path("data/raw/databento/opra_spxw_ohlcv_1m"),
    "statistics": Path("data/raw/databento/opra_spxw_statistics"),
    "cbbo_1s_audit": Path("data/raw/audit/opra_spxw_cbbo_1s"),
    "spx_1m": Path("data/raw/index/spx_1m"),
    "vix_1m": Path("data/raw/index/vix_1m"),
}

V3_ACTION_SURFACE = Path("v3/artifacts/layer2_action_surface_dataset.pkl")
V3_ACTION_SURFACE_LIVE = Path("v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl")
V2_DATA = Path("v2/data.pt")
V2_SIDECARS = Path("v2/data_sidecars")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v4-normalized-dir", type=Path, default=Path("v4/normalized"))
    p.add_argument("--v4-neural-dir", type=Path, default=Path("data/processed/spxw_0dte_neural_derived"))
    p.add_argument("--out-dir", type=Path, default=Path("v4/audit/data_sufficiency"))
    p.add_argument(
        "--max-sidecars",
        type=int,
        default=0,
        help="Limit v2 sidecar audit for debugging. 0 means all sidecars.",
    )
    return p.parse_args()


def _pct(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _year_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(str(v)[:4] for v in values).items()))


def _month_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(str(v)[:7] for v in values).items()))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _parquet_inventory(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.parquet"))
    rows = 0
    sessions = []
    for file in files:
        try:
            rows += pq.ParquetFile(file).metadata.num_rows
        except Exception:
            pass
        if len(file.name) >= 10 and file.name[:10].count("-") == 2:
            sessions.append(file.name[:10])
    return {
        "path": str(path),
        "files": len(files),
        "rows": int(rows),
        "sessions": len(set(sessions)),
        "first_session": min(sessions) if sessions else None,
        "last_session": max(sessions) if sessions else None,
    }


def audit_v4_raw() -> dict[str, Any]:
    return {name: _parquet_inventory(path) for name, path in V4_RAW_DIRS.items()}


def audit_v4_context_sources() -> dict[str, Any]:
    """Inspect whether index bars are official bars, proxies, or derived context."""
    out: dict[str, Any] = {}
    for name, path in (("spx_1m", V4_RAW_DIRS["spx_1m"]), ("vix_1m", V4_RAW_DIRS["vix_1m"])):
        source_counts = Counter()
        rows = 0
        files = sorted(path.glob("*.parquet"))
        for file in files:
            columns = _selected_columns(
                file,
                ["context_source", "proxy_source", "is_derived", "is_proxy", "symbol"],
            )
            table = pq.read_table(file, columns=columns)
            frame = table.to_pandas()
            rows += len(frame)
            if "context_source" in frame:
                source_counts.update(frame["context_source"].dropna().astype(str).tolist())
            if "proxy_source" in frame:
                source_counts.update(frame["proxy_source"].dropna().astype(str).tolist())
            if not columns:
                source_counts["unknown_or_official_shape"] += len(frame)
        out[name] = {
            "files": len(files),
            "rows": rows,
            "sources": dict(source_counts),
            "is_official_index_data": False if source_counts else None,
        }
    return out


def _selected_columns(path: Path, wanted: list[str]) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    available = set(schema.names)
    return [col for col in wanted if col in available]


def audit_v4_normalized(normalized_dir: Path) -> dict[str, Any]:
    files = sorted(normalized_dir.glob("databento_spxw_0dte_*_derived_context.parquet"))
    wanted = [
        "event_time",
        "contract_id",
        "root",
        "expiry",
        "strike",
        "right",
        "settlement_style",
        "bid",
        "ask",
        "mid",
        "bid_size",
        "ask_size",
        "quote_gap_seconds",
        "stat_open_interest",
        "option_ohlcv_volume",
        "underlying_price",
        "iv",
        "delta",
        "gamma",
        "theta",
    ]
    totals = Counter()
    roots = Counter()
    rights = Counter()
    settlements = Counter()
    sessions: list[str] = []
    unique_contracts: set[str] = set()
    quote_gaps: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    rows_by_session: dict[str, int] = {}

    for path in files:
        session = path.name.removeprefix("databento_spxw_0dte_").removesuffix(
            "_derived_context.parquet"
        )
        sessions.append(session)
        table = pq.read_table(path, columns=_selected_columns(path, wanted))
        df = table.to_pandas()
        n = len(df)
        rows_by_session[session] = n
        totals["rows"] += n

        roots.update(df["root"].astype(str).tolist())
        rights.update(df["right"].astype(str).tolist())
        settlements.update(df["settlement_style"].astype(str).tolist())
        unique_contracts.update(df["contract_id"].astype(str).unique().tolist())

        strike = df["strike"].map(float).to_numpy(dtype=float)
        mod = np.mod(strike, 5.0)
        aligned = np.isclose(mod, 0.0, atol=1e-6) | np.isclose(mod, 5.0, atol=1e-6)
        totals["bad_strike_alignment"] += int((~aligned).sum())

        event_dates = pd.to_datetime(df["event_time"], utc=True).dt.date.astype(str)
        expiry_dates = pd.to_datetime(df["expiry"]).dt.date.astype(str)
        totals["expiry_not_event_date"] += int((event_dates != expiry_dates).sum())

        totals["bad_root"] += int((df["root"].astype(str) != "SPXW").sum())
        totals["bad_settlement"] += int((df["settlement_style"].astype(str) != "PM").sum())

        bid = df["bid"].to_numpy(dtype=float)
        ask = df["ask"].to_numpy(dtype=float)
        mid = df["mid"].to_numpy(dtype=float)
        has_bidask = np.isfinite(bid) & np.isfinite(ask)
        valid_bidask = has_bidask & (bid >= 0) & (ask > 0) & (ask >= bid)
        totals["rows_with_bidask"] += int(has_bidask.sum())
        totals["invalid_bidask"] += int((has_bidask & ~valid_bidask).sum())
        totals["crossed_bidask"] += int((has_bidask & (ask < bid)).sum())
        totals["rows_with_mid"] += int(np.isfinite(mid).sum())
        spread = ask[valid_bidask] - bid[valid_bidask]
        if spread.size:
            spreads.append(spread.astype(float))

        for col in [
            "bid_size",
            "ask_size",
            "stat_open_interest",
            "option_ohlcv_volume",
            "underlying_price",
            "iv",
            "delta",
            "gamma",
            "theta",
        ]:
            totals[f"nonnull_{col}"] += int(pd.notna(df[col]).sum())

        gap = df["quote_gap_seconds"].to_numpy(dtype=float)
        finite_gap = gap[np.isfinite(gap)]
        totals["finite_quote_gap"] += int(finite_gap.size)
        totals["quote_gap_lte_90s"] += int((finite_gap <= 90.0).sum())
        if finite_gap.size:
            quote_gaps.append(finite_gap)

    n_rows = int(totals["rows"])
    quote_gap_all = np.concatenate(quote_gaps) if quote_gaps else np.asarray([], dtype=float)
    spread_all = np.concatenate(spreads) if spreads else np.asarray([], dtype=float)
    return {
        "files": len(files),
        "sessions": len(set(sessions)),
        "first_session": min(sessions) if sessions else None,
        "last_session": max(sessions) if sessions else None,
        "rows": n_rows,
        "rows_by_month": _month_counts(sessions),
        "rows_by_session_min": int(min(rows_by_session.values())) if rows_by_session else 0,
        "rows_by_session_median": float(np.median(list(rows_by_session.values()))) if rows_by_session else 0.0,
        "rows_by_session_max": int(max(rows_by_session.values())) if rows_by_session else 0,
        "unique_contracts": len(unique_contracts),
        "roots": dict(roots),
        "rights": dict(rights),
        "settlement_styles": dict(settlements),
        "bad_root_rows": int(totals["bad_root"]),
        "bad_settlement_rows": int(totals["bad_settlement"]),
        "bad_strike_alignment_rows": int(totals["bad_strike_alignment"]),
        "expiry_not_event_date_rows": int(totals["expiry_not_event_date"]),
        "bidask_coverage": _pct(totals["rows_with_bidask"], n_rows),
        "mid_coverage": _pct(totals["rows_with_mid"], n_rows),
        "invalid_bidask_rows": int(totals["invalid_bidask"]),
        "crossed_bidask_rows": int(totals["crossed_bidask"]),
        "bid_size_coverage": _pct(totals["nonnull_bid_size"], n_rows),
        "ask_size_coverage": _pct(totals["nonnull_ask_size"], n_rows),
        "ohlcv_volume_coverage": _pct(totals["nonnull_option_ohlcv_volume"], n_rows),
        "open_interest_coverage": _pct(totals["nonnull_stat_open_interest"], n_rows),
        "underlying_coverage": _pct(totals["nonnull_underlying_price"], n_rows),
        "iv_coverage": _pct(totals["nonnull_iv"], n_rows),
        "delta_coverage": _pct(totals["nonnull_delta"], n_rows),
        "gamma_coverage": _pct(totals["nonnull_gamma"], n_rows),
        "theta_coverage": _pct(totals["nonnull_theta"], n_rows),
        "quote_gap_populated_fraction": _pct(totals["finite_quote_gap"], n_rows),
        "quote_gap_lte_90s_fraction": (
            _pct(totals["quote_gap_lte_90s"], totals["finite_quote_gap"])
            if totals["finite_quote_gap"]
            else None
        ),
        "quote_gap_seconds_p50": float(np.nanmedian(quote_gap_all)) if quote_gap_all.size else None,
        "quote_gap_seconds_p95": float(np.nanpercentile(quote_gap_all, 95)) if quote_gap_all.size else None,
        "spread_p50": float(np.nanmedian(spread_all)) if spread_all.size else None,
        "spread_p95": float(np.nanpercentile(spread_all, 95)) if spread_all.size else None,
    }


def audit_v4_neural(neural_dir: Path) -> dict[str, Any]:
    files = sorted(neural_dir.glob("*.pkl"))
    sessions = [path.stem for path in files]
    rows_by_session: dict[str, int] = {}
    rows_by_split = Counter()
    bucket_counts = Counter()
    totals = Counter()
    policy_names: tuple[str, ...] | None = None
    option_feature_names: tuple[str, ...] | None = None
    option_feature_finite = Counter()
    option_feature_total = Counter()
    finite_net_by_policy: Counter[int] = Counter()
    finite_mid_by_policy: Counter[int] = Counter()
    rows_with_label_by_policy: Counter[int] = Counter()
    candidate_counts = []
    net_mid_diffs = []

    for path in files:
        session = path.stem
        with path.open("rb") as f:
            rows = pickle.load(f)
        rows_by_session[session] = len(rows)
        rows_by_split[split_name(session)] += len(rows)
        for row in rows:
            if policy_names is None:
                policy_names = tuple(row["label_names"])
            if option_feature_names is None:
                option_feature_names = tuple(row["feature_names"])
            totals["decision_rows"] += 1
            bucket_counts[time_bucket(row["decision_time"])] += 1
            mask = np.asarray(row["candidate_mask"], dtype=bool)
            option_ladder = np.asarray(row["option_ladder"], dtype=float)
            candidate_count = int(mask.sum())
            candidate_counts.append(candidate_count)
            totals["candidate_slots"] += int(mask.size)
            totals["candidate_mask_true"] += candidate_count
            market = np.asarray(row["market_window"], dtype=float)
            totals["latest_market_complete"] += int(np.isfinite(market[-1]).all())
            totals["full_market_window_complete"] += int(np.isfinite(market).all())
            if option_feature_names is not None:
                for feature_i, name in enumerate(option_feature_names):
                    values = option_ladder[:, :, feature_i]
                    option_feature_total[name] += int(mask.sum())
                    option_feature_finite[name] += int((np.isfinite(values) & mask).sum())

            net = np.asarray(row["labels_net_pnl"], dtype=float)
            mid = np.asarray(row["labels_mid_pnl"], dtype=float)
            for policy_i in range(net.shape[-1]):
                finite_net = np.isfinite(net[:, :, policy_i]) & mask
                finite_mid = np.isfinite(mid[:, :, policy_i]) & mask
                finite_net_by_policy[policy_i] += int(finite_net.sum())
                finite_mid_by_policy[policy_i] += int(finite_mid.sum())
                rows_with_label_by_policy[policy_i] += int(finite_net.any())
                both = finite_net & finite_mid
                if both.any():
                    net_mid_diffs.append(np.abs(net[:, :, policy_i][both] - mid[:, :, policy_i][both]))

    decision_rows = int(totals["decision_rows"])
    diffs = np.concatenate(net_mid_diffs) if net_mid_diffs else np.asarray([], dtype=float)
    return {
        "files": len(files),
        "sessions": len(set(sessions)),
        "first_session": min(sessions) if sessions else None,
        "last_session": max(sessions) if sessions else None,
        "sessions_by_month": _month_counts(sessions),
        "decision_rows": decision_rows,
        "decision_rows_by_split": dict(rows_by_split),
        "decision_rows_by_time_bucket": dict(bucket_counts),
        "decision_rows_per_session_min": int(min(rows_by_session.values())) if rows_by_session else 0,
        "decision_rows_per_session_median": float(np.median(list(rows_by_session.values()))) if rows_by_session else 0.0,
        "decision_rows_per_session_max": int(max(rows_by_session.values())) if rows_by_session else 0,
        "candidate_slots": int(totals["candidate_slots"]),
        "candidate_mask_true": int(totals["candidate_mask_true"]),
        "candidate_mask_true_fraction": _pct(totals["candidate_mask_true"], totals["candidate_slots"]),
        "candidate_count_per_row_p10": float(np.percentile(candidate_counts, 10)) if candidate_counts else 0.0,
        "candidate_count_per_row_p50": float(np.percentile(candidate_counts, 50)) if candidate_counts else 0.0,
        "candidate_count_per_row_p90": float(np.percentile(candidate_counts, 90)) if candidate_counts else 0.0,
        "latest_market_complete_fraction": _pct(totals["latest_market_complete"], decision_rows),
        "full_market_window_complete_fraction": _pct(totals["full_market_window_complete"], decision_rows),
        "label_names": list(policy_names or ()),
        "option_feature_coverage": {
            name: _pct(option_feature_finite[name], option_feature_total[name])
            for name in sorted(option_feature_total)
        },
        "finite_net_labels_by_policy": {
            (policy_names[i] if policy_names else str(i)): int(finite_net_by_policy[i])
            for i in sorted(finite_net_by_policy)
        },
        "finite_mid_labels_by_policy": {
            (policy_names[i] if policy_names else str(i)): int(finite_mid_by_policy[i])
            for i in sorted(finite_mid_by_policy)
        },
        "decision_rows_with_net_label_by_policy": {
            (policy_names[i] if policy_names else str(i)): int(rows_with_label_by_policy[i])
            for i in sorted(rows_with_label_by_policy)
        },
        "net_vs_mid_abs_diff_p50": float(np.nanmedian(diffs)) if diffs.size else None,
        "net_vs_mid_abs_diff_p95": float(np.nanpercentile(diffs, 95)) if diffs.size else None,
    }


def audit_v2_data(data_path: Path) -> dict[str, Any]:
    if not data_path.exists():
        return {"exists": False}
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    dates = [str(x) for x in data["dates"]]
    unique_days = sorted(set(dates))
    out = {
        "exists": True,
        "bars": len(dates),
        "days": len(unique_days),
        "first_day": unique_days[0] if unique_days else None,
        "last_day": unique_days[-1] if unique_days else None,
        "days_by_year": _year_counts(unique_days),
        "feature_count": len(data.get("feature_names", [])),
        "metadata_version": data.get("metadata", {}).get("version"),
        "label_scheme": data.get("metadata", {}).get("label_scheme"),
        "trade_window": data.get("metadata", {}).get("trade_window"),
        "chain_schema_version": data.get("metadata", {}).get("chain_schema_version"),
        "execution_filters": data.get("metadata", {}).get("execution_filters"),
        "risk_policy": data.get("metadata", {}).get("risk_policy"),
    }
    for name in [
        "label_trade_valid",
        "label_trade",
        "slice_label_trade_valid",
        "slice_label_trade",
    ]:
        arr = _to_numpy(data[name])
        out[f"{name}_mean"] = float(np.nanmean(arr))
    for name in ["best_contract_pnl", "slice_best_contract_pnl"]:
        arr = _to_numpy(data[name]).astype(float)
        out[f"{name}_mean"] = float(np.nanmean(arr))
        out[f"{name}_p50"] = float(np.nanmedian(arr))
    return out


def audit_v2_sidecars(sidecar_dir: Path, max_sidecars: int = 0) -> dict[str, Any]:
    files = sorted(sidecar_dir.glob("*.pt"))
    if max_sidecars > 0:
        files = files[:max_sidecars]
    dates = [p.stem.replace(" 2", "") for p in files]
    canonical_counts = Counter(dates)
    duplicate_dates = sorted(day for day, count in canonical_counts.items() if count > 1)
    totals = Counter()
    contracts_per_day = []
    bars_per_day = []
    first_day = min(dates) if dates else None
    last_day = max(dates) if dates else None

    for path in files:
        sidecar = torch.load(path, map_location="cpu", weights_only=False)
        contracts = np.asarray(sidecar.get("contract_strike", []), dtype=float)
        contracts_per_day.append(len(contracts))
        bars_per_day.append(int(sidecar.get("n_bars", 0)))
        totals["days_loaded"] += 1
        totals["bad_expiry_date"] += int(str(sidecar.get("expiry", "")) != str(sidecar.get("date", "")).replace("-", ""))
        if contracts.size:
            mod = np.mod(contracts, 5.0)
            aligned = np.isclose(mod, 0.0, atol=1e-6) | np.isclose(mod, 5.0, atol=1e-6)
            totals["bad_strike_alignment_contracts"] += int((~aligned).sum())

        mid = np.asarray(sidecar.get("contract_mid", []), dtype=float)
        bid = np.asarray(sidecar.get("contract_bid", []), dtype=float)
        ask = np.asarray(sidecar.get("contract_ask", []), dtype=float)
        quality = np.asarray(sidecar.get("contract_quality", []), dtype=int)
        if mid.size:
            totals["contract_bar_cells"] += int(mid.size)
            has_mid = np.isfinite(mid) & (mid > 0)
            has_bidask = np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask > 0)
            totals["cells_with_mid"] += int(has_mid.sum())
            totals["cells_with_bidask"] += int(has_bidask.sum())
            totals["crossed_bidask"] += int((has_bidask & (ask < bid)).sum())
        if quality.size:
            totals["quality_corrupt"] += int((quality == 0).sum())
            totals["quality_partial"] += int((quality == 1).sum())
            totals["quality_valid"] += int((quality == 2).sum())

    cells = int(totals["contract_bar_cells"])
    return {
        "path": str(sidecar_dir),
        "files_loaded": len(files),
        "canonical_dates": len(canonical_counts),
        "duplicate_canonical_dates": duplicate_dates[:20],
        "duplicate_canonical_date_count": len(duplicate_dates),
        "first_day": first_day,
        "last_day": last_day,
        "days_by_year": _year_counts(sorted(canonical_counts)),
        "contracts_per_day_min": int(min(contracts_per_day)) if contracts_per_day else 0,
        "contracts_per_day_median": float(np.median(contracts_per_day)) if contracts_per_day else 0.0,
        "contracts_per_day_max": int(max(contracts_per_day)) if contracts_per_day else 0,
        "bars_per_day_median": float(np.median(bars_per_day)) if bars_per_day else 0.0,
        "bad_expiry_date_days": int(totals["bad_expiry_date"]),
        "bad_strike_alignment_contracts": int(totals["bad_strike_alignment_contracts"]),
        "contract_bar_cells": cells,
        "mid_coverage": _pct(totals["cells_with_mid"], cells),
        "bidask_coverage": _pct(totals["cells_with_bidask"], cells),
        "crossed_bidask_cells": int(totals["crossed_bidask"]),
        "quality_valid_fraction": _pct(totals["quality_valid"], cells),
        "quality_partial_fraction": _pct(totals["quality_partial"], cells),
        "quality_corrupt_fraction": _pct(totals["quality_corrupt"], cells),
        "bidask_provenance": (
            "proxy: v2/pipeline/build_v2_dataset.py computes bid/ask from Polygon "
            "minute close +/- spread_fraction_proxy(close, high, low) / 2"
        ),
    }


def audit_action_surface(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    with path.open("rb") as f:
        bundle = pickle.load(f)
    rows = bundle["rows"]
    days = sorted(rows["day"].astype(str).unique().tolist())
    meta = bundle["meta"]
    contract_mask = np.asarray(bundle["contract_mask"], dtype=float)
    tradeable = np.asarray(bundle["action_labels"]["tradeable_mask"], dtype=float)[:, 1:]
    available = np.asarray(bundle["action_labels"]["available_mask"], dtype=float)[:, 1:]
    out = {
        "exists": True,
        "path": str(path),
        "rows": int(len(rows)),
        "days": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "days_by_year": _year_counts(days),
        "execution_window": meta.get("execution_window"),
        "top_k_contracts_per_side": meta.get("top_k_contracts_per_side"),
        "n_action_contract_tokens": meta.get("n_action_contract_tokens"),
        "contract_selection_mode": meta.get("contract_selection_mode"),
        "label_timing": meta.get("label_timing"),
        "utility_horizon_bars": meta.get("utility_horizon_bars"),
        "contract_feature_names": meta.get("contract_feature_names"),
        "action_label_names": meta.get("action_label_names"),
        "tokens_per_row_mean": float(np.nansum(contract_mask, axis=1).mean()),
        "tradeable_actions_per_row_mean": float(np.nansum(tradeable, axis=1).mean()),
        "available_actions_per_row_mean": float(np.nansum(available, axis=1).mean()),
        "rows_with_any_tradeable_action": float((np.nansum(tradeable, axis=1) > 0).mean()),
    }
    for label in [
        "entry_fill_mid",
        "entry_spread_fraction",
        "spread_cost",
        "horizon_pnl",
        "hybrid_live_utility",
        "best_exit_pnl",
    ]:
        arr = np.asarray(bundle["action_labels"].get(label), dtype=float)[:, 1:]
        finite = arr[np.isfinite(arr)]
        out[f"{label}_finite_fraction"] = _pct(finite.size, arr.size)
        out[f"{label}_p50"] = float(np.nanmedian(finite)) if finite.size else None
    return out


def cross_dataset_summary(v4: dict[str, Any], v2: dict[str, Any], v3_live: dict[str, Any]) -> dict[str, Any]:
    v4_sessions = set()
    if v4.get("normalized", {}).get("first_session"):
        first = v4["normalized"]["first_session"]
        last = v4["normalized"]["last_session"]
        # Use neural sessions for exact set when available.
        neural_months = v4.get("neural", {}).get("sessions_by_month", {})
        v4_sessions = set()
        for month, count in neural_months.items():
            v4_sessions.add(month)
    return {
        "v4_high_quality_executable_days": v4.get("normalized", {}).get("sessions", 0),
        "v4_decision_rows": v4.get("neural", {}).get("decision_rows", 0),
        "v4_candidate_slots": v4.get("neural", {}).get("candidate_slots", 0),
        "v3_historical_days": v2.get("days", 0),
        "v3_action_surface_rows": v3_live.get("rows", 0),
        "v3_safe_use": [
            "regime priors",
            "time-of-day priors",
            "architecture pretraining or ablations",
            "feature importance triage",
        ],
        "v3_unsafe_use_without_relabeling": [
            "final executable PnL",
            "bid/ask microstructure edge claims",
            "broad purchase gate for Databento-quality data",
        ],
        "v4_safe_use": [
            "executable ask-entry / bid-exit labels",
            "SPXW PM-settled 0DTE neural prototype",
            "small-sample holdout gates",
            "calibration experiments before spending more",
        ],
        "enough_data_verdict": (
            "Enough data for a serious prototype and data-quality falsification; "
            "not enough Databento-quality months for a robust broad-market neural edge claim."
        ),
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    v4_raw = payload["v4"]["raw"]
    v4_context = payload["v4"]["context_sources"]
    v4_norm = payload["v4"]["normalized"]
    v4_neural = payload["v4"]["neural"]
    v2_data = payload["v3_v2"]["v2_data"]
    sidecars = payload["v3_v2"]["sidecars"]
    live = payload["v3_v2"]["action_surface_live_0945_1130"]
    lines = [
        "# Data Sufficiency Audit",
        "",
        "This audit separates broad historical context from executable quote truth.",
        "",
        "## Verdict",
        "",
        payload["cross_dataset"]["enough_data_verdict"],
        "",
        "- v4 Databento CBBO is the source of truth for executable labels.",
        "- v4 SPX/VIX context is currently derived/proxy context, not official Cboe index data.",
        "- v3/v2 is broad and valuable, but its bid/ask is proxy-derived from Polygon minute OHLC.",
        "- Use v3/v2 to learn durable market structure; use v4 to validate whether that structure survives real bid/ask execution.",
        "",
        "## v4 Raw Inventory",
        "",
        "| Dataset | Files | Sessions | Rows | First | Last |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, row in v4_raw.items():
        lines.append(
            f"| {name} | {row['files']} | {row['sessions']} | {row['rows']} | "
            f"{row['first_session']} | {row['last_session']} |"
        )

    lines += [
        "",
        "## v4 Context Source Caveat",
        "",
        "| Context | Files | Rows | Sources | Official Index Data |",
        "|---|---:|---:|---|---|",
    ]
    for name, row in v4_context.items():
        lines.append(
            f"| {name} | {row['files']} | {row['rows']} | {row['sources']} | "
            f"{row['is_official_index_data']} |"
        )
    lines += [
        "",
        "The current neural market window is usable for prototype wiring, but it is not a substitute for official SPX and VIX bars when we make promotion-grade claims.",
        "",
        "## v4 Normalized Quality",
        "",
        f"- Sessions: {v4_norm['sessions']} ({v4_norm['first_session']} to {v4_norm['last_session']})",
        f"- Rows: {v4_norm['rows']:,}; unique contracts: {v4_norm['unique_contracts']:,}",
        f"- Bad roots: {v4_norm['bad_root_rows']}; bad settlements: {v4_norm['bad_settlement_rows']}; bad strike alignment: {v4_norm['bad_strike_alignment_rows']}",
        f"- Bid/ask coverage: {_fmt_pct(v4_norm['bidask_coverage'])}; invalid bid/ask rows: {v4_norm['invalid_bidask_rows']}",
        f"- OHLCV volume coverage: {_fmt_pct(v4_norm['ohlcv_volume_coverage'])}; OI coverage: {_fmt_pct(v4_norm['open_interest_coverage'])}",
        f"- Underlying coverage: {_fmt_pct(v4_norm['underlying_coverage'])}; IV/delta/gamma coverage: {_fmt_pct(v4_norm['iv_coverage'])} / {_fmt_pct(v4_norm['delta_coverage'])} / {_fmt_pct(v4_norm['gamma_coverage'])}",
        f"- Quote gap populated: {_fmt_pct(v4_norm['quote_gap_populated_fraction'])}; quote gap <= 90s: {_fmt_pct(v4_norm['quote_gap_lte_90s_fraction'])}; spread p50/p95: {v4_norm['spread_p50']:.3f} / {v4_norm['spread_p95']:.3f}",
        "",
        "## v4 Neural Dataset",
        "",
        f"- Decision rows: {v4_neural['decision_rows']:,} across {v4_neural['sessions']} sessions.",
        f"- Candidate slots: {v4_neural['candidate_slots']:,}; valid candidate fraction: {_fmt_pct(v4_neural['candidate_mask_true_fraction'])}.",
        f"- Candidate count per row p10/p50/p90: {v4_neural['candidate_count_per_row_p10']:.0f} / {v4_neural['candidate_count_per_row_p50']:.0f} / {v4_neural['candidate_count_per_row_p90']:.0f}.",
        f"- Latest market complete: {_fmt_pct(v4_neural['latest_market_complete_fraction'])}; full 30m market window complete: {_fmt_pct(v4_neural['full_market_window_complete_fraction'])}.",
        f"- Decision rows by split: {v4_neural['decision_rows_by_split']}.",
        f"- Decision rows by time bucket: {v4_neural['decision_rows_by_time_bucket']}.",
        f"- Neural IV/delta/gamma/theta coverage on valid candidates: {_fmt_pct(v4_neural['option_feature_coverage'].get('iv'))} / {_fmt_pct(v4_neural['option_feature_coverage'].get('delta'))} / {_fmt_pct(v4_neural['option_feature_coverage'].get('gamma'))} / {_fmt_pct(v4_neural['option_feature_coverage'].get('theta'))}.",
        "",
        "| Label Policy | Finite Net Labels | Decision Rows With Label |",
        "|---|---:|---:|",
    ]
    for name, count in v4_neural["finite_net_labels_by_policy"].items():
        rows = v4_neural["decision_rows_with_net_label_by_policy"].get(name, 0)
        lines.append(f"| {name} | {count:,} | {rows:,} |")

    lines += [
        "",
        "## v3/v2 Historical Coverage",
        "",
        f"- v2 bars: {v2_data['bars']:,}; days: {v2_data['days']} ({v2_data['first_day']} to {v2_data['last_day']}).",
        f"- v2 days by year: {v2_data['days_by_year']}.",
        f"- v2 label scheme: `{v2_data['label_scheme']}`; trade window: `{v2_data['trade_window']}`.",
        f"- Sidecars loaded: {sidecars['files_loaded']}; canonical dates: {sidecars['canonical_dates']}; duplicates: {sidecars['duplicate_canonical_date_count']}.",
        f"- Sidecar contracts/day median: {sidecars['contracts_per_day_median']:.0f}; mid coverage: {_fmt_pct(sidecars['mid_coverage'])}; bid/ask coverage: {_fmt_pct(sidecars['bidask_coverage'])}.",
        f"- Sidecar quality valid/partial/corrupt: {_fmt_pct(sidecars['quality_valid_fraction'])} / {_fmt_pct(sidecars['quality_partial_fraction'])} / {_fmt_pct(sidecars['quality_corrupt_fraction'])}.",
        f"- Bid/ask provenance: {sidecars['bidask_provenance']}.",
        "",
        "## v3 Action Surface",
        "",
        f"- Rows: {live['rows']:,}; days: {live['days']} ({live['first_day']} to {live['last_day']}).",
        f"- Days by year: {live['days_by_year']}.",
        f"- Execution window: {live['execution_window']}; tokens: {live['n_action_contract_tokens']}.",
        f"- Tradeable actions per row mean: {live['tradeable_actions_per_row_mean']:.2f}; rows with any tradeable action: {_fmt_pct(live['rows_with_any_tradeable_action'])}.",
        f"- Entry mid finite: {_fmt_pct(live['entry_fill_mid_finite_fraction'])}; entry spread finite: {_fmt_pct(live['entry_spread_fraction_finite_fraction'])}.",
        "",
        "## Practical Use",
        "",
        "Use v3/v2 now for:",
    ]
    lines.extend(f"- {item}" for item in payload["cross_dataset"]["v3_safe_use"])
    lines += ["", "Do not use v3/v2 directly for:"]
    lines.extend(f"- {item}" for item in payload["cross_dataset"]["v3_unsafe_use_without_relabeling"])
    lines += [
        "",
        "Use v4 now for:",
    ]
    lines.extend(f"- {item}" for item in payload["cross_dataset"]["v4_safe_use"])
    lines += [
        "",
        "## Next Modeling Direction",
        "",
        "1. Mine v3/v2 for stable environment priors across 2022-2026.",
        "2. Re-express those priors as causal v4 features, not as copied v3 labels.",
        "3. Train/calibrate on v4 ask-entry/bid-exit labels only.",
        "4. Require any v3-discovered prior to improve the v4 multi-seed March holdout before buying more data.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    v4 = {
        "raw": audit_v4_raw(),
        "context_sources": audit_v4_context_sources(),
        "normalized": audit_v4_normalized(args.v4_normalized_dir),
        "neural": audit_v4_neural(args.v4_neural_dir),
    }
    v2 = audit_v2_data(V2_DATA)
    sidecars = audit_v2_sidecars(V2_SIDECARS, max_sidecars=args.max_sidecars)
    v3_full = audit_action_surface(V3_ACTION_SURFACE)
    v3_live = audit_action_surface(V3_ACTION_SURFACE_LIVE)
    payload = {
        "v4": v4,
        "v3_v2": {
            "v2_data": v2,
            "sidecars": sidecars,
            "action_surface_full": v3_full,
            "action_surface_live_0945_1130": v3_live,
        },
        "cross_dataset": cross_dataset_summary(v4, v2, v3_live),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=True) + "\n")
    md_path = args.out_dir / "report.md"
    write_report(md_path, payload)
    print(json_path)
    print(md_path)
    print(json.dumps(_jsonable(payload["cross_dataset"]), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
