"""Outcome-blind characterization of the frozen owned CMBP signed-flow slice.

This module deliberately stops at the tape itself.  It reads only market-event
fields from the 64-session manifest pinned on 2026-08-23: no label, P&L, entry
or exit value, forward return, forward path, model, or threshold fit.

Inference is session-clustered throughout.  A trade count is a census count of
the frozen bytes, never an inferential sample size.  Every estimated population
quantity carries a deterministic whole-session bootstrap interval; exact byte
or row counts are labelled as exact censuses instead of being given a fake
standard error.

The primary signing population reproduces the pinned semantic gate exactly: a
trade at the immediately prior row's ask is a buy and one at its bid is a sell,
after stable sorting within instrument by receive time then event time.  The
Parquet/DBN schema has no sequence field, however, and some prior rows share the
trade's receive and event timestamps.  Those tied-clock observations remain in
the pinned-law result for comparability and are separately removed in a
strictly-earlier-receive-time sensitivity.  Neither version is evidence of an
edge or profitability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.ops.verify_cmbp_touch_semantics import classify_session


SCHEMA_VERSION = "v5.cmbp-signed-flow-characterization.v1"
FAILURE_SCHEMA_VERSION = "v5.cmbp-signed-flow-characterization.failure.v1"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260823
CI_LEVEL = 0.95
ET = "America/New_York"
RTH_OPEN_MINUTE = 9 * 60 + 30
RTH_CLOSE_MINUTE = 16 * 60
ACF_LAGS = (1, 5, 15)
INTRADAY_WIDTH_MINUTES = 30
CONFIRMATION_START = date(2026, 8, 6)
EXPECTED_SESSIONS = 64
EXPECTED_ROWS = 173_470_783
EXPECTED_MANIFEST_SHA256 = "55dd5a92f969c6218ffaada8eddceeff355d4d12407fdffed9a5735592149b69"
EXPECTED_DATA_ROOT = Path("data/raw/audit/protocol101_highres_opra/cmbp-1")

# This allowlist is the outcome firewall.  PyArrow is asked for exactly these
# columns, so an outcome-bearing table cannot be opened accidentally.
READ_COLUMNS = (
    "ts_event",
    "action",
    "price",
    "size",
    "bid_px_00",
    "ask_px_00",
    "bid_sz_00",
    "ask_sz_00",
    "instrument_id",
    "symbol",
    "publisher_id",
)
REQUIRED_AFTER_RESET = ("ts_recv",) + READ_COLUMNS
FORBIDDEN_WORDS = ("label", "pnl", "profit", "entry_value", "exit_value", "forward_return")


class FlowCharacterizationError(RuntimeError):
    """The outcome-blind analysis cannot prove its input or statistical law."""


@dataclass(frozen=True)
class FrozenInput:
    session: str
    parquet: Path
    rows: int
    symbols: int
    bytes: int


def _metric(
    estimate: float,
    draws: np.ndarray,
    *,
    n_sessions: int,
    unit: str,
    estimand: str,
) -> dict[str, Any]:
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not math.isfinite(float(estimate)) or finite.size == 0:
        return {
            "estimate": None,
            "ci_95": [None, None],
            "n_sessions": int(n_sessions),
            "unit": unit,
            "estimand": estimand,
            "uncertainty": "not estimable: statistic is constant, absent, or non-finite",
        }
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo, hi = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "estimate": float(estimate),
        "ci_95": [float(lo), float(hi)],
        "n_sessions": int(n_sessions),
        "unit": unit,
        "estimand": estimand,
        "uncertainty": (
            f"{CI_LEVEL:.0%} percentile interval from {BOOTSTRAP_REPS:,} deterministic "
            "whole-session bootstrap resamples"
        ),
    }


def _exact(value: Any, *, unit: str, reason: str) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "uncertainty": f"no sampling interval: exact census of the frozen manifest; {reason}",
    }


class SessionBootstrap:
    """One shared deterministic whole-session resample for every statistic."""

    def __init__(self, n_sessions: int, *, seed: int = BOOTSTRAP_SEED,
                 reps: int = BOOTSTRAP_REPS):
        if n_sessions < 2:
            raise FlowCharacterizationError("session bootstrap needs at least two sessions")
        self.n_sessions = int(n_sessions)
        self.reps = int(reps)
        rng = np.random.default_rng(seed)
        self.indices = rng.integers(0, n_sessions, size=(reps, n_sessions))

    def summarize(self, values: Sequence[float], *, unit: str, estimand: str,
                  statistic: str = "mean") -> dict[str, Any]:
        a = np.asarray(values, dtype=float)
        if a.shape != (self.n_sessions,):
            raise FlowCharacterizationError(
                f"bootstrap expected {self.n_sessions} session values, got {a.shape}"
            )
        valid = np.isfinite(a)
        n = int(valid.sum())
        if n == 0:
            return _metric(float("nan"), np.array([]), n_sessions=0, unit=unit,
                           estimand=estimand)
        sampled = a[self.indices]
        if statistic == "mean":
            estimate = float(np.nanmean(a))
            draws = np.nanmean(sampled, axis=1)
        elif statistic.startswith("q"):
            q = float(statistic[1:])
            estimate = float(np.nanquantile(a, q))
            draws = np.nanquantile(sampled, q, axis=1)
        else:
            raise FlowCharacterizationError(f"unknown bootstrap statistic {statistic}")
        return _metric(estimate, draws, n_sessions=n, unit=unit, estimand=estimand)

    def distribution(self, values: Sequence[float], *, unit: str,
                     name: str) -> dict[str, Any]:
        return {
            "session_equal_mean": self.summarize(
                values, unit=unit, estimand=f"mean session {name}", statistic="mean"
            ),
            "session_p10": self.summarize(
                values, unit=unit, estimand=f"10th percentile across sessions of {name}",
                statistic="q0.10",
            ),
            "session_median": self.summarize(
                values, unit=unit, estimand=f"median across sessions of {name}",
                statistic="q0.50",
            ),
            "session_p90": self.summarize(
                values, unit=unit, estimand=f"90th percentile across sessions of {name}",
                statistic="q0.90",
            ),
        }


def _self_hash(payload: dict[str, Any], field: str) -> str:
    unsigned = {k: v for k, v in payload.items() if k != field}
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_manifest(path: Path, *, repo_root: Path) -> tuple[dict[str, Any], list[FrozenInput]]:
    path = Path(path)
    if not path.is_file():
        raise FlowCharacterizationError(f"no frozen manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    claimed = manifest.get("manifest_sha256")
    actual = _self_hash(manifest, "manifest_sha256")
    if claimed != actual or claimed != EXPECTED_MANIFEST_SHA256:
        raise FlowCharacterizationError(
            f"manifest self-hash mismatch: claimed {claimed}, computed {actual}"
        )
    sessions = manifest.get("sessions") or []
    files = manifest.get("files") or []
    if len(sessions) != EXPECTED_SESSIONS or len(files) != EXPECTED_SESSIONS:
        raise FlowCharacterizationError(
            f"frozen population drifted: {len(sessions)} sessions and {len(files)} files"
        )
    if sessions != sorted(set(sessions)):
        raise FlowCharacterizationError("manifest sessions are not unique and chronological")
    if int(manifest.get("total_rows", -1)) != EXPECTED_ROWS:
        raise FlowCharacterizationError("manifest row total drifted")
    if any(date.fromisoformat(s) >= CONFIRMATION_START for s in sessions):
        raise FlowCharacterizationError("manifest includes a confirmation-reserved session")
    forbidden = " ".join(str(x).lower() for x in manifest.get("forbidden", []))
    if not all(word in forbidden for word in ("label", "p&l", "forward return")):
        raise FlowCharacterizationError("manifest no longer states the outcome firewall")

    root = Path(repo_root).resolve()
    expected_data_root = (root / EXPECTED_DATA_ROOT).resolve()
    resolved: list[FrozenInput] = []
    metadata_rows = 0
    for record, session in zip(files, sessions, strict=True):
        if record.get("session") != session:
            raise FlowCharacterizationError(f"manifest file/session order drift at {session}")
        relative = Path(record.get("parquet", ""))
        target = (root / relative).resolve()
        if target.parent != expected_data_root or target.name != f"{session}.cmbp-1.parquet":
            raise FlowCharacterizationError(f"out-of-scope input path for {session}: {relative}")
        if not target.is_file():
            raise FlowCharacterizationError(f"missing frozen input for {session}: {relative}")
        schema = pq.ParquetFile(target)
        rows = int(schema.metadata.num_rows)
        expected = int(record.get("rows", -1))
        if rows != expected:
            raise FlowCharacterizationError(
                f"{session}: parquet metadata rows {rows:,} != manifest {expected:,}"
            )
        names = set(schema.schema_arrow.names)
        missing = sorted(set(READ_COLUMNS) - names)
        if missing:
            raise FlowCharacterizationError(f"{session}: required market fields missing: {missing}")
        bad = sorted(c for c in names if any(word in c.lower() for word in FORBIDDEN_WORDS))
        if bad:
            raise FlowCharacterizationError(
                f"{session}: outcome-like columns present in raw market file: {bad}"
            )
        # The absence is a causal-order caveat, not permission to invent order.
        if "sequence" in names:
            raise FlowCharacterizationError(
                f"{session}: schema unexpectedly gained sequence; review the pinned ordering law"
            )
        metadata_rows += rows
        resolved.append(
            FrozenInput(
                session=session,
                parquet=target,
                rows=rows,
                symbols=int(record.get("symbols", -1)),
                bytes=int(target.stat().st_size),
            )
        )
    if metadata_rows != EXPECTED_ROWS:
        raise FlowCharacterizationError("sum of Parquet metadata rows does not match frozen total")
    return manifest, resolved


def _right_multiplier(symbols: pd.Series, session: str) -> np.ndarray:
    right = symbols.astype(str).str.extract(r"\d{6}([CP])", expand=False)
    if right.isna().any():
        examples = sorted(symbols[right.isna()].astype(str).unique())[:5]
        raise FlowCharacterizationError(f"{session}: cannot parse option right: {examples}")
    return np.where(right.to_numpy() == "C", 1, -1).astype(np.int8)


def signed_events(frame: pd.DataFrame, session: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the exact pinned prior-row signs plus clock-order diagnostics."""
    missing = sorted(set(REQUIRED_AFTER_RESET) - set(frame.columns))
    if missing:
        raise FlowCharacterizationError(f"{session}: market frame missing {missing}")
    d = frame.sort_values(["instrument_id", "ts_recv", "ts_event"], kind="stable").copy()
    g = d.groupby("instrument_id", sort=False)
    for source in (
        "ts_recv", "ts_event", "action", "publisher_id", "bid_px_00", "ask_px_00",
        "bid_sz_00", "ask_sz_00",
    ):
        d[f"prior_{source}"] = g[source].shift(1)

    trade = d["action"].astype(str).eq("T")
    prior = d["prior_bid_px_00"].notna() & d["prior_ask_px_00"].notna()
    clean = prior & (d["prior_bid_px_00"] < d["prior_ask_px_00"])
    at_ask = trade & clean & (d["price"] == d["prior_ask_px_00"])
    at_bid = trade & clean & (d["price"] == d["prior_bid_px_00"])
    keep = at_ask | at_bid

    out = d.loc[keep, [
        "ts_recv", "ts_event", "instrument_id", "symbol", "publisher_id", "price", "size",
        "prior_ts_recv", "prior_ts_event", "prior_action", "prior_publisher_id",
        "prior_bid_px_00", "prior_ask_px_00", "prior_bid_sz_00", "prior_ask_sz_00",
    ]].copy()
    out["execution_sign"] = np.where(at_ask[keep], 1, -1).astype(np.int8)
    out["right_multiplier"] = _right_multiplier(out["symbol"], session)
    out["directional_sign"] = (
        out["execution_sign"].to_numpy(np.int8)
        * out["right_multiplier"].to_numpy(np.int8)
    )
    out["strict_earlier_receive"] = out["prior_ts_recv"] < out["ts_recv"]
    out["tied_receive"] = out["prior_ts_recv"] == out["ts_recv"]
    out["tied_event"] = out["prior_ts_event"] == out["ts_event"]
    if (~(out["strict_earlier_receive"] | out["tied_receive"])).any():
        raise FlowCharacterizationError(f"{session}: prior receive time is later than trade")

    mid = (out["prior_bid_px_00"] + out["prior_ask_px_00"]) / 2.0
    out["prior_relative_spread_bps"] = np.where(
        mid > 0.0,
        (out["prior_ask_px_00"] - out["prior_bid_px_00"]) / mid * 10_000.0,
        np.nan,
    )
    out["prior_total_depth"] = out["prior_bid_sz_00"] + out["prior_ask_sz_00"]
    out["same_side_depth"] = np.where(
        out["execution_sign"] > 0, out["prior_ask_sz_00"], out["prior_bid_sz_00"]
    )
    out["size_to_touch"] = np.where(
        out["same_side_depth"] > 0, out["size"] / out["same_side_depth"], np.nan
    )
    depth_sum = out["prior_total_depth"].to_numpy(float)
    raw_depth_imbalance = np.divide(
        out["prior_bid_sz_00"].to_numpy(float) - out["prior_ask_sz_00"].to_numpy(float),
        depth_sum,
        out=np.full(len(out), np.nan),
        where=depth_sum > 0,
    )
    out["directional_prior_depth_imbalance"] = (
        raw_depth_imbalance * out["right_multiplier"].to_numpy(float)
    )
    out["premium_notional"] = out["price"] * out["size"] * 100.0

    # This identity is a guard against drifting away from the pinned semantic law.
    pinned = classify_session(frame, session)
    if pinned.signed != len(out):
        raise FlowCharacterizationError(
            f"{session}: extracted {len(out):,} signs != pinned law {pinned.signed:,}"
        )
    out = out.sort_values(
        ["ts_recv", "ts_event", "instrument_id", "publisher_id"], kind="stable"
    ).reset_index(drop=True)
    diagnostic = {
        "all_trades": int(pinned.trades),
        "pinned_signed": int(len(out)),
        "strict_earlier_receive_signed": int(out["strict_earlier_receive"].sum()),
        "tied_receive_signed": int(out["tied_receive"].sum()),
        "tied_receive_and_event_signed": int(
            (out["tied_receive"] & out["tied_event"]).sum()
        ),
        "tied_receive_prior_quote_action": int(
            (out["tied_receive"] & out["prior_action"].astype(str).ne("T")).sum()
        ),
        "tied_receive_prior_trade_action": int(
            (out["tied_receive"] & out["prior_action"].astype(str).eq("T")).sum()
        ),
        "tied_receive_cross_publisher": int(
            (out["tied_receive"] & (out["prior_publisher_id"] != out["publisher_id"])).sum()
        ),
    }
    return out, diagnostic


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _safe_corr(x: Iterable[float], y: Iterable[float], *, spearman: bool = False) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    if spearman:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def _acf(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) <= lag:
        return float("nan")
    return _safe_corr(x[:-lag], x[lag:])


def _minute_number(ts: pd.Series) -> np.ndarray:
    local = ts.dt.tz_convert(ET)
    return (local.dt.hour.to_numpy() * 60 + local.dt.minute.to_numpy()).astype(int)


def _clock_exposure(frame: pd.DataFrame, session: str) -> tuple[int, int, int]:
    local_minute = _minute_number(frame["ts_recv"])
    live_book = (
        np.isfinite(frame["bid_px_00"].to_numpy(float))
        & np.isfinite(frame["ask_px_00"].to_numpy(float))
        & (frame["bid_px_00"].to_numpy(float) < frame["ask_px_00"].to_numpy(float))
        & (local_minute >= RTH_OPEN_MINUTE)
        & (local_minute < RTH_CLOSE_MINUTE)
    )
    minutes = local_minute[live_book]
    if minutes.size == 0:
        raise FlowCharacterizationError(f"{session}: no valid RTH book exposure")
    start, end = int(minutes.min()), int(minutes.max())
    if start != RTH_OPEN_MINUTE or not RTH_OPEN_MINUTE <= end < RTH_CLOSE_MINUTE:
        raise FlowCharacterizationError(
            f"{session}: unexpected delivered clock {start // 60:02d}:{start % 60:02d}-"
            f"{end // 60:02d}:{end % 60:02d} ET"
        )
    return start, end, end - start + 1


def _run_lengths(signs: np.ndarray) -> np.ndarray:
    if len(signs) == 0:
        return np.array([], dtype=int)
    starts = np.flatnonzero(np.r_[True, signs[1:] != signs[:-1]])
    return np.diff(np.r_[starts, len(signs)])


def _fano(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    mean = counts.mean() if counts.size else 0.0
    return float(counts.var(ddof=1) / mean) if counts.size > 1 and mean > 0 else float("nan")


def _subset_metrics(events: pd.DataFrame, clock_start: int, clock_end: int) -> dict[str, float]:
    signs = events["directional_sign"].to_numpy(float)
    sizes = events["size"].to_numpy(float)
    notional = events["premium_notional"].to_numpy(float)
    execution = events["execution_sign"].to_numpy(float)
    minute = _minute_number(events["ts_recv"]) if len(events) else np.array([], dtype=int)
    grid = np.arange(clock_start, clock_end + 1)
    index = minute - clock_start
    live = (index >= 0) & (index < len(grid))
    index = index[live]
    net_contracts = np.bincount(
        index, weights=signs[live] * sizes[live], minlength=len(grid)
    ).astype(float)
    total_contracts = np.bincount(index, weights=sizes[live], minlength=len(grid)).astype(float)
    event_counts = np.bincount(index, minlength=len(grid)).astype(float)

    if len(events) > 1:
        recv = events["ts_recv"].astype("int64").to_numpy()
        gap = np.diff(recv) / 1e9
        same = signs[1:] == signs[:-1]
        same_instrument = (
            events["instrument_id"].to_numpy()[1:]
            == events["instrument_id"].to_numpy()[:-1]
        )
        positive_gap = gap > 0
        p = float((signs > 0).mean())
        independence = p * p + (1.0 - p) ** 2
        mean_gap, std_gap = float(gap.mean()), float(gap.std(ddof=1))
        burstiness = _safe_ratio(std_gap - mean_gap, std_gap + mean_gap)
    else:
        gap = same = same_instrument = positive_gap = np.array([])
        independence = burstiness = float("nan")
    runs = _run_lengths(signs)

    # Removing broad 30-minute time-of-day shape separates event bursts from
    # the mechanical open/close intensity curve.
    within_fano = []
    for start in range(clock_start, clock_end + 1, INTRADAY_WIDTH_MINUTES):
        chunk = event_counts[start - clock_start : start - clock_start + INTRADAY_WIDTH_MINUTES]
        value = _fano(chunk)
        if math.isfinite(value):
            within_fano.append(value)

    minute_table = pd.DataFrame({
        "minute": grid,
        "event_count": event_counts,
        "total_contracts": total_contracts,
        "net_contracts": net_contracts,
    })
    if len(events):
        event_minute = pd.Series(minute, index=events.index, name="minute")
        e = events.assign(minute=event_minute)
        market = e.groupby("minute", sort=True).agg(
            prior_relative_spread_bps=("prior_relative_spread_bps", "median"),
            prior_total_depth=("prior_total_depth", "median"),
            trade_size=("size", "median"),
        ).reset_index()
        minute_table = minute_table.merge(market, on="minute", how="left")
    else:
        minute_table[["prior_relative_spread_bps", "prior_total_depth", "trade_size"]] = np.nan
    minute_table["abs_contract_imbalance"] = np.divide(
        np.abs(minute_table["net_contracts"]),
        minute_table["total_contracts"],
        out=np.full(len(minute_table), np.nan),
        where=minute_table["total_contracts"] > 0,
    )

    return {
        "execution_count_imbalance": _safe_ratio(float(execution.sum()), float(len(execution))),
        "directional_count_imbalance": _safe_ratio(float(signs.sum()), float(len(signs))),
        "directional_contract_imbalance": _safe_ratio(
            float(np.sum(signs * sizes)), float(np.sum(sizes))
        ),
        "directional_notional_imbalance": _safe_ratio(
            float(np.sum(signs * notional)), float(np.sum(notional))
        ),
        "same_direction_probability": float(same.mean()) if same.size else float("nan"),
        "same_direction_independence_baseline": independence,
        "same_direction_excess": (
            float(same.mean()) - independence if same.size else float("nan")
        ),
        "same_direction_positive_gap_probability": (
            float(same[positive_gap].mean()) if positive_gap.any() else float("nan")
        ),
        "same_direction_positive_gap_excess": (
            float(same[positive_gap].mean()) - independence
            if positive_gap.any() else float("nan")
        ),
        "same_direction_same_instrument_probability": (
            float(same[same_instrument].mean()) if same_instrument.any() else float("nan")
        ),
        "same_direction_cross_instrument_probability": (
            float(same[~same_instrument].mean()) if (~same_instrument).any() else float("nan")
        ),
        "zero_interarrival_share": float((gap == 0).mean()) if gap.size else float("nan"),
        "interarrival_median_seconds": float(np.median(gap)) if gap.size else float("nan"),
        "interarrival_p90_seconds": float(np.quantile(gap, 0.90)) if gap.size else float("nan"),
        "interarrival_burstiness": burstiness,
        "mean_same_direction_run_events": float(runs.mean()) if runs.size else float("nan"),
        "p90_same_direction_run_events": float(np.quantile(runs, 0.90)) if runs.size else float("nan"),
        "event_count_fano_1m": _fano(event_counts),
        "event_count_fano_within_30m": (
            float(np.mean(within_fano)) if within_fano else float("nan")
        ),
        **{f"net_contract_acf_{lag}m": _acf(net_contracts, lag) for lag in ACF_LAGS},
        "rho_event_intensity_vs_prior_spread": _safe_corr(
            minute_table["event_count"], minute_table["prior_relative_spread_bps"], spearman=True
        ),
        "rho_event_intensity_vs_prior_depth": _safe_corr(
            minute_table["event_count"], minute_table["prior_total_depth"], spearman=True
        ),
        "rho_abs_imbalance_vs_prior_spread": _safe_corr(
            minute_table["abs_contract_imbalance"],
            minute_table["prior_relative_spread_bps"], spearman=True,
        ),
        "rho_abs_imbalance_vs_prior_depth": _safe_corr(
            minute_table["abs_contract_imbalance"], minute_table["prior_total_depth"],
            spearman=True,
        ),
        "rho_trade_size_vs_same_side_depth": _safe_corr(
            events["size"], events["same_side_depth"], spearman=True
        ) if len(events) else float("nan"),
        "rho_direction_vs_directional_prior_depth_imbalance": _safe_corr(
            events["directional_sign"], events["directional_prior_depth_imbalance"],
            spearman=True,
        ) if len(events) else float("nan"),
    }


def characterize_session(frame: pd.DataFrame, session: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events, diagnostics = signed_events(frame, session)
    clock_start, clock_end, exposure = _clock_exposure(frame, session)
    symbols = int(frame["symbol"].nunique())
    metrics = _subset_metrics(events, clock_start, clock_end)
    strict_events = events[events["strict_earlier_receive"]].reset_index(drop=True)
    strict = _subset_metrics(strict_events, clock_start, clock_end)
    strict.update({
        "signed_events_per_minute": _safe_ratio(len(strict_events), exposure),
        "signed_events_per_symbol_minute": _safe_ratio(
            len(strict_events), exposure * symbols
        ),
        "signed_contracts_per_symbol_minute": _safe_ratio(
            float(strict_events["size"].sum()), exposure * symbols
        ),
        "signed_share_of_all_trades": _safe_ratio(
            len(strict_events), diagnostics["all_trades"]
        ),
        "median_prior_relative_spread_bps": float(
            strict_events["prior_relative_spread_bps"].median()
        ),
        "median_prior_total_depth_contracts": float(
            strict_events["prior_total_depth"].median()
        ),
        "median_trade_size_contracts": float(strict_events["size"].median()),
        "median_trade_to_same_side_touch": float(strict_events["size_to_touch"].median()),
        "share_trade_size_at_least_same_side_touch": float(
            (strict_events["size_to_touch"] >= 1.0).mean()
        ),
    })
    tied = events["tied_receive"].to_numpy(bool)
    metrics.update({
        "signed_events_per_minute": _safe_ratio(len(events), exposure),
        "signed_events_per_symbol_minute": _safe_ratio(len(events), exposure * symbols),
        "signed_contracts_per_symbol_minute": _safe_ratio(float(events["size"].sum()), exposure * symbols),
        "signed_share_of_all_trades": _safe_ratio(len(events), diagnostics["all_trades"]),
        "strict_earlier_receive_share_of_pinned_signed": _safe_ratio(
            diagnostics["strict_earlier_receive_signed"], len(events)
        ),
        "tied_receive_share_of_pinned_signed": _safe_ratio(
            diagnostics["tied_receive_signed"], len(events)
        ),
        "tied_receive_event_time_also_tied_share": _safe_ratio(
            diagnostics["tied_receive_and_event_signed"], diagnostics["tied_receive_signed"]
        ),
        "tied_receive_cross_publisher_share": _safe_ratio(
            diagnostics["tied_receive_cross_publisher"], diagnostics["tied_receive_signed"]
        ),
        "median_prior_relative_spread_bps": float(events["prior_relative_spread_bps"].median()),
        "median_prior_total_depth_contracts": float(events["prior_total_depth"].median()),
        "median_trade_size_contracts": float(events["size"].median()),
        "median_trade_to_same_side_touch": float(events["size_to_touch"].median()),
        "share_trade_size_at_least_same_side_touch": float(
            (events["size_to_touch"] >= 1.0).mean()
        ),
    })

    intraday: list[dict[str, Any]] = []
    minute = _minute_number(events["ts_recv"])
    for start in range(RTH_OPEN_MINUTE, RTH_CLOSE_MINUTE, INTRADAY_WIDTH_MINUTES):
        stop = min(start + INTRADAY_WIDTH_MINUTES, RTH_CLOSE_MINUTE)
        effective_stop = min(stop, clock_end + 1)
        minutes_exposed = max(0, effective_stop - max(start, clock_start))
        if minutes_exposed <= 0:
            continue
        mask = (minute >= start) & (minute < stop)
        part = events.loc[mask]
        sign = part["directional_sign"].to_numpy(float)
        size = part["size"].to_numpy(float)
        intraday.append({
            "bin": f"{start // 60:02d}:{start % 60:02d}-{stop // 60:02d}:{stop % 60:02d}",
            "minutes_exposed": int(minutes_exposed),
            "signed_events": int(len(part)),
            "signed_events_per_symbol_minute": _safe_ratio(
                len(part), minutes_exposed * symbols
            ),
            "directional_contract_imbalance": _safe_ratio(
                float(np.sum(sign * size)), float(np.sum(size))
            ),
            "median_prior_relative_spread_bps": (
                float(part["prior_relative_spread_bps"].median()) if len(part) else float("nan")
            ),
            "tied_receive_share": float(part["tied_receive"].mean()) if len(part) else float("nan"),
        })

    record: dict[str, Any] = {
        "session": session,
        "symbols": symbols,
        "clock_start_et": f"{clock_start // 60:02d}:{clock_start % 60:02d}",
        "clock_end_et": f"{clock_end // 60:02d}:{clock_end % 60:02d}",
        "exposure_minutes": exposure,
        **diagnostics,
        **metrics,
        "strict_earlier_receive_sensitivity": strict,
        "exact_session_census_uncertainty": (
            "these are exact summaries of one observed frozen session; no within-session "
            "trade-level inferential interval is claimed"
        ),
    }
    return record, intraday


METRIC_UNITS = {
    "execution_count_imbalance": "fraction [-1,1]",
    "directional_count_imbalance": "fraction [-1,1]",
    "directional_contract_imbalance": "fraction [-1,1]",
    "directional_notional_imbalance": "fraction [-1,1]",
    "same_direction_probability": "probability",
    "same_direction_independence_baseline": "probability",
    "same_direction_excess": "probability points",
    "same_direction_positive_gap_probability": "probability",
    "same_direction_positive_gap_excess": "probability points",
    "same_direction_same_instrument_probability": "probability",
    "same_direction_cross_instrument_probability": "probability",
    "zero_interarrival_share": "probability",
    "interarrival_median_seconds": "seconds",
    "interarrival_p90_seconds": "seconds",
    "interarrival_burstiness": "dimensionless [-1,1]",
    "mean_same_direction_run_events": "events",
    "p90_same_direction_run_events": "events",
    "event_count_fano_1m": "variance/mean",
    "event_count_fano_within_30m": "variance/mean",
    "net_contract_acf_1m": "correlation",
    "net_contract_acf_5m": "correlation",
    "net_contract_acf_15m": "correlation",
    "rho_event_intensity_vs_prior_spread": "Spearman rho",
    "rho_event_intensity_vs_prior_depth": "Spearman rho",
    "rho_abs_imbalance_vs_prior_spread": "Spearman rho",
    "rho_abs_imbalance_vs_prior_depth": "Spearman rho",
    "rho_trade_size_vs_same_side_depth": "Spearman rho",
    "rho_direction_vs_directional_prior_depth_imbalance": "Spearman rho",
    "signed_events_per_minute": "events/minute",
    "signed_events_per_symbol_minute": "events/symbol-minute",
    "signed_contracts_per_symbol_minute": "contracts/symbol-minute",
    "signed_share_of_all_trades": "fraction",
    "strict_earlier_receive_share_of_pinned_signed": "fraction",
    "tied_receive_share_of_pinned_signed": "fraction",
    "tied_receive_event_time_also_tied_share": "fraction",
    "tied_receive_cross_publisher_share": "fraction",
    "median_prior_relative_spread_bps": "basis points",
    "median_prior_total_depth_contracts": "contracts",
    "median_trade_size_contracts": "contracts",
    "median_trade_to_same_side_touch": "ratio",
    "share_trade_size_at_least_same_side_touch": "fraction",
}


def _summaries(records: list[dict[str, Any]], bootstrap: SessionBootstrap,
               *, nested: str | None = None,
               metrics: Sequence[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    selected = list(metrics) if metrics is not None else list(METRIC_UNITS)
    for name in selected:
        unit = METRIC_UNITS[name]
        values = [
            (record[nested].get(name, float("nan")) if nested else record.get(name, float("nan")))
            for record in records
        ]
        out[name] = bootstrap.distribution(values, unit=unit, name=name)
    return out


def _half_stability(records: list[dict[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    n = len(records)
    cut = n // 2
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    first_index = rng.integers(0, cut, size=(BOOTSTRAP_REPS, cut))
    second_n = n - cut
    second_index = rng.integers(0, second_n, size=(BOOTSTRAP_REPS, second_n))
    result: dict[str, Any] = {
        "definition": (
            f"first {cut} chronological sessions versus last {second_n}; each half resampled "
            "by whole session independently"
        )
    }
    for name in metrics:
        a = np.asarray([record[name] for record in records[:cut]], dtype=float)
        b = np.asarray([record[name] for record in records[cut:]], dtype=float)
        draw_a = np.nanmean(a[first_index], axis=1)
        draw_b = np.nanmean(b[second_index], axis=1)
        result[name] = {
            "first_half": _metric(float(np.nanmean(a)), draw_a, n_sessions=int(np.isfinite(a).sum()),
                                  unit=METRIC_UNITS[name], estimand="mean first-half session"),
            "second_half": _metric(float(np.nanmean(b)), draw_b, n_sessions=int(np.isfinite(b).sum()),
                                   unit=METRIC_UNITS[name], estimand="mean second-half session"),
            "second_minus_first": _metric(
                float(np.nanmean(b) - np.nanmean(a)), draw_b - draw_a,
                n_sessions=int(np.isfinite(a).sum() + np.isfinite(b).sum()),
                unit=METRIC_UNITS[name], estimand="second-half mean minus first-half mean",
            ),
        }
    return result


def _intraday_summary(session_bins: list[list[dict[str, Any]]],
                      bootstrap: SessionBootstrap) -> list[dict[str, Any]]:
    labels = [
        f"{m // 60:02d}:{m % 60:02d}-{(m + INTRADAY_WIDTH_MINUTES) // 60:02d}:"
        f"{(m + INTRADAY_WIDTH_MINUTES) % 60:02d}"
        for m in range(RTH_OPEN_MINUTE, RTH_CLOSE_MINUTE, INTRADAY_WIDTH_MINUTES)
    ]
    maps = [{row["bin"]: row for row in rows} for rows in session_bins]
    result = []
    for label in labels:
        row: dict[str, Any] = {"bin_et": label}
        for name, unit in (
            ("signed_events_per_symbol_minute", "events/symbol-minute"),
            ("directional_contract_imbalance", "fraction [-1,1]"),
            ("median_prior_relative_spread_bps", "basis points"),
            ("tied_receive_share", "fraction"),
        ):
            values = [mapping.get(label, {}).get(name, float("nan")) for mapping in maps]
            row[name] = bootstrap.summarize(
                values, unit=unit, estimand=f"mean session {name} in {label} ET"
            )
        row["sessions_observed"] = _exact(
            int(sum(label in mapping for mapping in maps)), unit="sessions",
            reason="early-close sessions have no exposure in later bins",
        )
        result.append(row)
    return result


def analyze(manifest_path: Path, *, repo_root: Path) -> dict[str, Any]:
    manifest, inputs = verify_manifest(manifest_path, repo_root=repo_root)
    records: list[dict[str, Any]] = []
    bins: list[list[dict[str, Any]]] = []
    for i, frozen in enumerate(inputs, start=1):
        frame = pd.read_parquet(frozen.parquet, columns=list(READ_COLUMNS)).reset_index()
        if len(frame) != frozen.rows:
            raise FlowCharacterizationError(
                f"{frozen.session}: decoded rows {len(frame):,} != frozen {frozen.rows:,}"
            )
        record, intraday = characterize_session(frame, frozen.session)
        if record["symbols"] != frozen.symbols:
            raise FlowCharacterizationError(
                f"{frozen.session}: decoded symbols {record['symbols']} != manifest {frozen.symbols}"
            )
        records.append(record)
        bins.append(intraday)
        print(
            f"{i:02d}/{len(inputs)} {frozen.session}: {record['pinned_signed']:,} signed; "
            f"tied prior {record['tied_receive_share_of_pinned_signed']:.1%}", flush=True,
        )

    bootstrap = SessionBootstrap(len(records))
    total_signed = sum(int(record["pinned_signed"]) for record in records)
    total_tied = sum(int(record["tied_receive_signed"]) for record in records)
    total_strict = sum(int(record["strict_earlier_receive_signed"]) for record in records)
    total_trades = sum(int(record["all_trades"]) for record in records)
    total_tied_event = sum(int(record["tied_receive_and_event_signed"]) for record in records)
    total_cross_publisher = sum(int(record["tied_receive_cross_publisher"]) for record in records)
    total_prior_quote = sum(int(record["tied_receive_prior_quote_action"]) for record in records)
    total_prior_trade = sum(int(record["tied_receive_prior_trade_action"]) for record in records)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "verdict": "OUTCOME_BLIND_CHARACTERIZATION_ONLY_CAUSAL_ORDER_UNVERIFIED",
        "manifest_sha256": manifest["manifest_sha256"],
        "reads_no_outcome": True,
        "alpha_charged": False,
        "fits_run": 0,
        "inference_unit": "session",
        "population": {
            "sessions": _exact(len(records), unit="sessions", reason="all frozen sessions read"),
            "rows": _exact(sum(x.rows for x in inputs), unit="market rows",
                           reason="Parquet metadata and decoded row counts agree"),
            "raw_trades": _exact(total_trades, unit="trade prints",
                                 reason="all frozen trade rows classified"),
            "pinned_prior_row_signed": _exact(
                total_signed, unit="signed trade prints", reason="exact pinned prior-row law"
            ),
            "strictly_earlier_receive_signed": _exact(
                total_strict, unit="signed trade prints",
                reason="subset whose prior row has an earlier receive timestamp",
            ),
            "tied_receive_signed": _exact(
                total_tied, unit="signed trade prints",
                reason="prior row and trade share receive timestamp",
            ),
            "selected_slice_parquet_bytes": _exact(
                sum(x.bytes for x in inputs), unit="bytes", reason="filesystem sizes"
            ),
        },
        "selection_caveat": (
            "HEAVILY SELECTED: the requested symbols were chosen around roughly 31 prior-route "
            "trades per session (up to seven symbols), not as a broad unbiased option band. Raw "
            "CMBP contains many tape trades around those selected symbols, but prevalence, "
            "imbalance, intensity, spread, size, and intraday results do not estimate the market-wide "
            "population. This analysis can describe only the frozen selected slice."
        ),
        "signing_law": {
            "pinned_primary": (
                "within instrument, stable sort by ts_recv then ts_event; use the immediately "
                "prior row's bid/ask; ask print=buy, bid print=sell; call buy/put sell=bullish; "
                "call sell/put buy=bearish; inside/outside/locked/crossed/no-prior are unsigned"
            ),
            "strict_clock_sensitivity": (
                "same calculations after requiring prior_ts_recv < trade ts_recv"
            ),
            "causal_order_status": "UNKNOWN_NO_SEQUENCE_FIELD",
            "causal_order_evidence": {
                "schema_sequence_field_present": _exact(
                    False, unit="boolean", reason="all 64 local Parquet schemas inspected"
                ),
                "tied_receive_and_event": _exact(
                    total_tied_event, unit="signed trade prints",
                    reason="prior row shares both event and receive timestamps",
                ),
                "tied_receive_cross_publisher": _exact(
                    total_cross_publisher, unit="signed trade prints",
                    reason="publisher differs between tied prior row and trade",
                ),
                "tied_receive_prior_action_quote": _exact(
                    total_prior_quote, unit="signed trade prints", reason="prior action is not trade"
                ),
                "tied_receive_prior_action_trade": _exact(
                    total_prior_trade, unit="signed trade prints", reason="prior action is trade"
                ),
                "interpretation": (
                    "stable file row order exists, but with no sequence field it cannot independently "
                    "prove causal order among identical-clock records; no causal lead claim is made"
                ),
            },
        },
        "uncertainty_law": {
            "method": "whole-session nonparametric percentile bootstrap",
            "reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "trade_level_standard_errors": False,
            "exact_counts": (
                "manifest/row/event counts are censuses of frozen bytes and carry an explicit "
                "no-interval reason; this does not cure selection bias"
            ),
        },
        "metric_definitions": {
            "directional_sign": "call buy or put sell = +1; call sell or put buy = -1",
            "imbalance": "sum(sign*weight)/sum(weight), weights=count/contracts/premium notional",
            "autocorrelation": "Pearson autocorrelation of one-minute net directional contracts",
            "fano": "variance/mean of signed-event counts per minute; also within fixed 30m bins",
            "same_sign_excess": "observed adjacent same-direction probability minus p^2+(1-p)^2",
            "arrival_denominator": (
                "delivered RTH span from first 09:30 valid book minute through last valid book "
                "minute; per-symbol metrics divide again by selected-symbol count"
            ),
            "spread": "prior touch width divided by prior mid, in basis points",
            "size": "trade contracts and consolidated prior bid/ask displayed contracts",
            "intraday": "fixed 30-minute ET bins; early-close sessions contribute only while exposed",
        },
        "pinned_law_session_distributions": _summaries(records, bootstrap),
        "strict_earlier_receive_sensitivity": _summaries(
            records,
            bootstrap,
            nested="strict_earlier_receive_sensitivity",
            metrics=(
                "execution_count_imbalance",
                "directional_count_imbalance",
                "directional_contract_imbalance",
                "directional_notional_imbalance",
                "same_direction_probability",
                "same_direction_independence_baseline",
                "same_direction_excess",
                "same_direction_positive_gap_probability",
                "same_direction_positive_gap_excess",
                "same_direction_same_instrument_probability",
                "same_direction_cross_instrument_probability",
                "zero_interarrival_share",
                "interarrival_median_seconds",
                "interarrival_p90_seconds",
                "interarrival_burstiness",
                "mean_same_direction_run_events",
                "p90_same_direction_run_events",
                "event_count_fano_1m",
                "event_count_fano_within_30m",
                "net_contract_acf_1m",
                "net_contract_acf_5m",
                "net_contract_acf_15m",
                "rho_event_intensity_vs_prior_spread",
                "rho_event_intensity_vs_prior_depth",
                "rho_abs_imbalance_vs_prior_spread",
                "rho_abs_imbalance_vs_prior_depth",
                "rho_trade_size_vs_same_side_depth",
                "rho_direction_vs_directional_prior_depth_imbalance",
                "signed_events_per_minute",
                "signed_events_per_symbol_minute",
                "signed_contracts_per_symbol_minute",
                "signed_share_of_all_trades",
                "median_prior_relative_spread_bps",
                "median_prior_total_depth_contracts",
                "median_trade_size_contracts",
                "median_trade_to_same_side_touch",
                "share_trade_size_at_least_same_side_touch",
            ),
        ),
        "chronological_session_stability": _half_stability(
            records,
            (
                "directional_contract_imbalance",
                "signed_events_per_symbol_minute",
                "net_contract_acf_1m",
                "same_direction_positive_gap_excess",
                "tied_receive_share_of_pinned_signed",
                "median_prior_relative_spread_bps",
            ),
        ),
        "intraday_30m": _intraday_summary(bins, bootstrap),
        "bugs_and_surprises": {
            "tied_clock_prior_rows": (
                "A material share of pinned signs uses a prior row with identical receive and event "
                "timestamps. The strict-earlier sensitivity proves whether headline structure "
                "survives without those rows; the missing sequence field keeps their causal order UNKNOWN."
            ),
            "clustering_mechanism_checks": (
                "zero-gap share, positive-gap same-sign excess, same-versus-cross-instrument "
                "persistence, raw Fano, and within-30m Fano separate exact-clock bundles, contract "
                "stickiness, and broad time-of-day intensity from residual clustering"
            ),
            "selected_population": (
                "varying selected-symbol count and prior-route construction can mechanically change "
                "arrival intensity and imbalance; per-symbol normalization cannot remove selection"
            ),
        },
        "failure_path": {
            "input_or_invariant_failure": (
                "raise FlowCharacterizationError; wrapper writes one immutable sibling failure "
                "receipt with FAIL_WRAPPER_OR_INPUT_INVARIANT and exits 5"
            ),
            "output_exists": (
                "refuse before reading market files; neither success nor failure receipt is overwritten"
            ),
            "failure_receipt_exists": (
                "refuse overwrite and exit 6; a new numbered attempt is required"
            ),
            "silent_skip": False,
            "partial_success": False,
        },
        "not_determined": [
            "any relation to labels, returns, P&L, future paths, entry/exit economics, or profitability",
            "market-wide prevalence or stability on an unbiased broad option band",
            "causal order among records sharing both receive and event timestamps",
            "train/live event parity or executable latency",
        ],
        "per_session_exact": records,
    }
    return receipt


def _failure_payload(exc: BaseException, *, manifest_path: Path, out_path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "verdict": "FAIL_WRAPPER_OR_INPUT_INVARIANT",
        "reads_no_outcome": True,
        "alpha_charged": False,
        "fits_run": 0,
        "manifest_path": str(manifest_path),
        "intended_output": str(out_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "failure_law": (
            "no session is silently dropped, no partial characterization is called success, and "
            "this failure receipt is immutable; rerun requires a new numbered attempt"
        ),
    }
    value["receipt_sha256"] = _self_hash(value, "receipt_sha256")
    return value


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--failure-out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--wrapper", type=Path)
    args = parser.parse_args(argv)

    # Refuse overwrite before any data read.  Existing artifacts belong to an
    # earlier immutable attempt, whether they record success or failure.
    if args.out.exists():
        print(f"REFUSE_OVERWRITE: success receipt exists at {args.out}", file=sys.stderr)
        return 6
    if args.failure_out.exists():
        print(f"REFUSE_OVERWRITE: failure receipt exists at {args.failure_out}", file=sys.stderr)
        return 6
    try:
        receipt = analyze(args.manifest, repo_root=args.repo_root)
        implementation = {
            "analysis_module": file_sha256(Path(__file__)),
            "manifest_file": file_sha256(args.manifest),
        }
        if args.wrapper is not None:
            implementation["archived_wrapper"] = file_sha256(args.wrapper)
        receipt["implementation_sha256"] = implementation
        receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
        _exclusive_json(args.out, receipt)
        print(json.dumps({
            "verdict": receipt["verdict"],
            "receipt_sha256": receipt["receipt_sha256"],
            "sessions": receipt["population"]["sessions"]["value"],
        }, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - every failure is preserved, never skipped
        failure = _failure_payload(exc, manifest_path=args.manifest, out_path=args.out)
        try:
            _exclusive_json(args.failure_out, failure)
        except FileExistsError:
            print(f"REFUSE_OVERWRITE: failure receipt exists at {args.failure_out}", file=sys.stderr)
            return 6
        print(f"FAIL_WRAPPER_OR_INPUT_INVARIANT: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(cli())
